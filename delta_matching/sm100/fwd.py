"""Host entry of the sm100-mx forward: reuses FA4's _flash_attn_fwd plumbing with DMForwardSm100 swapped in.

Compile cache: persistent (FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 + FLASH_ATTENTION_CUTE_DSL_CACHE_DIR) under a
fingerprint of FA4's sources AND this package's kernel sources, so AOT compiles (aot.py, fake tensors, no GPU) are
reused by GPU runs.  The compile key does not depend on B / N, only on (causal, GQA ratio, config).
"""
import dataclasses
import hashlib
import os
import threading
from pathlib import Path

import torch
import flash_attn.cute.interface as fi
from flash_attn.cute import cache_utils, config as fa_config

from .dm_fwd_sm100 import DMForwardSm100

_HERE = Path(__file__).resolve().parent
_LOCK = threading.Lock()
# ex2_emu_freq=0: every forward exp on MUFU ex2.approx (no FA4 deg-3 FMA polynomial). Accuracy rule (user: no accuracy
# trade-offs): vs FA4's causal fp8 emulation (deg-3 on 25% of exps) all-MUFU has 100-1000x fewer P codes differing from
# RN_e4m3(exact 2^x), ~9x (up to 46x) lower LSE error and no worse O (kda/runs/fwdemu_*; candidate FWD-emu-off).
# DM_FWD_OPTS=ex2_emu_freq=N re-enables the emulation for experiments only.
FORCE = dict(use_2cta_instrs=False, pack_gqa=False, use_s_ping_pong=False, num_splits=1, ex2_emu_freq=0)


def _fingerprint(opts=""):
    h = hashlib.sha256(cache_utils._compute_source_fingerprint().encode())
    h.update(opts.encode())   # kernel variant knobs (DM_FWD_OPTS, read in DMForwardSm100.__init__)
    for f in sorted(_HERE.glob("*.py")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:24]


_CACHES = {}


def _cache():
    """One compile cache per DM_FWD_OPTS string (so several variants can run in one process)."""
    opts = os.environ.get("DM_FWD_OPTS", "")
    if opts not in _CACHES:
        if cache_utils.CUTE_DSL_CACHE_ENABLED:
            _CACHES[opts] = cache_utils.JITPersistentCache(cache_utils.get_cache_path() / "dm_sm100" / _fingerprint(opts) / "fwd")
        else:
            _CACHES[opts] = cache_utils.JITCache()
    return _CACHES[opts]


_SEL_MEMO = {}   # host-overhead cut: memoized config selection (pure function of FA4's inputs + our overrides)


def _select(orig_select, over):
    ok = tuple(sorted((k, repr(v)) for k, v in over.items()))

    def select(inputs):
        try:
            mk = (inputs, ok)
            hit = _SEL_MEMO.get(mk)
        except TypeError:
            mk, hit = None, None
        if hit is not None:
            return hit
        cfg = _select_impl(inputs)
        if mk is not None:
            _SEL_MEMO[mk] = cfg
        return cfg

    def _select_impl(inputs):
        cfg = orig_select(inputs)
        cfg = dataclasses.replace(cfg, **{k: v for k, v in over.items() if hasattr(cfg, k) and k != "kv_stage"})
        overlap, kv_stage = fa_config.select_sm100_smem_layout(
            inputs, tile_m=cfg.tile_m, tile_n=cfg.tile_n, q_stage=cfg.q_stage, num_splits=cfg.num_splits,
            use_2cta_instrs=cfg.use_2cta_instrs, is_static_persistent=cfg.is_static_persistent,
            use_clc_scheduler=cfg.use_clc_scheduler)
        if "kv_stage" in over:
            kv_stage = min(kv_stage, over["kv_stage"])
        cfg = dataclasses.replace(cfg, overlap_sO_sQ=overlap, kv_stage=kv_stage)
        if "reg_sm" in over:     # setmaxnreg split: 2*softmax + correction + other = 512
            sm_, co_ = over["reg_sm"], over["reg_corr"]
            cfg = dataclasses.replace(cfg, registers=type(cfg.registers)(softmax=sm_, correction=co_, other=512 - 2 * sm_ - co_))
        return cfg
    return select


def _ue8m0_words(s):
    """(B, H, NB) fp32 powers of 2 -> (B, H, NB/4) int32 of 4 packed UE8M0 bytes (block 4j+i in byte i)."""
    b = ((s.contiguous().view(torch.int32) >> 23) & 0xFF).to(torch.uint8)
    return b.view(*s.shape[:-1], s.shape[-1] // 4, 4).contiguous().view(torch.int32).squeeze(-1)


def pack_kv_scales(ks, vs):
    """(B, Hk, NB) K and V scales -> (B, Hk, 2 * NB/4) int32: [2t] = K bytes of n_block t, [2t+1] = V bytes (SFB word)."""
    return torch.stack([_ue8m0_words(ks), _ue8m0_words(vs)], dim=-1).flatten(-2).contiguous()


def dm_fwd(q8, qs, k8, ks, v8, vs, causal: bool, config_override: dict | None = None):
    """q8 (B,N,H,D) e4m3 (sm baked), k8/v8 (B,N,Hk,D) e4m3; qs (B,H,N/32), ks/vs (B,Hk,N/32) fp32 powers of 2.
    Returns O (B,N,H,D) bf16, LSE (B,H,N) fp32 natural log."""
    over = dict(FORCE)
    if "2cta=1" in os.environ.get("DM_FWD_OPTS", ""):
        over.pop("use_2cta_instrs")          # let FA4's config pick 2-CTA (non-causal hd128)
    for kv in os.environ.get("DM_FWD_OPTS", "").split(","):     # config-level knobs, e.g. ex2_emu_freq=0,kv_stage=6
        if "=" in kv and kv.split("=")[0] in ("ex2_emu_freq", "ex2_emu_start_frg", "kv_stage", "q_stage", "reg_sm", "reg_corr"):
            over[kv.split("=")[0]] = int(kv.split("=")[1])
        if kv == "s_pp":        # FA4 S ping-pong (both softmax stages share one S slot, exp of tile t overlaps S MMA of t+1)
            over["use_s_ping_pong"] = True
    if config_override:
        over.update(config_override)
    if "kvpack=0" not in os.environ.get("DM_FWD_OPTS", ""):
        # ks may be passed pre-packed (int32, from pack_kv_scales; producer side) with vs=None
        qsc, ksc, vsc = qs.contiguous(), (ks if ks.dtype == torch.int32 else pack_kv_scales(ks, vs)), None
    else:
        qsc, ksc, vsc = qs.contiguous(), ks.contiguous(), vs.contiguous()
    ids = {id(qsc), id(ksc), id(vsc)} - {id(None)}
    orig = (fi.FlashAttentionForwardSm100, fi._validate_tensor, fi._flash_attn_fwd.compile_cache, fi.select_fwd_config,
            fi.to_cute_tensor)

    def validate(t, name, *a, **k):
        if name.endswith("_descale"):
            return
        return orig[1](t, name, *a, **k)

    def tct(t, *a, **k):
        if id(t) in ids:          # scales are (B, H, NB) with NB unit-stride
            k["leading_dim"] = 2
        return orig[4](t, *a, **k)

    from flash_attn.cute import utils as _fu
    _emu0 = _fu.ex2_emulation_2
    with _LOCK:
        if "poly6" in os.environ.get("DM_FWD_OPTS", "").split(","):
            # poly6: the FA4 ex2-emulation slots (ex2_emu_freq / ex2_emu_start_frg) use the degree-6 polynomial
            # (dm_utils.ex2_poly6_2: at least as accurate as ex2.approx.ftz on every element metric) - trace-time only
            from . import dm_utils as _du
            _fu.ex2_emulation_2 = lambda x, y, **kw: _du.ex2_poly6_2(x, y)
        fi.FlashAttentionForwardSm100 = DMForwardSm100
        fi._validate_tensor = validate
        fi._flash_attn_fwd.compile_cache = _cache()
        fi.select_fwd_config = _select(orig[3], over)
        fi.to_cute_tensor = tct
        try:
            out, lse, *_ = fi._flash_attn_fwd(
                q8, k8, v8, softmax_scale=1.0, causal=causal, return_lse=True,
                q_descale=qsc, k_descale=ksc, v_descale=vsc, pack_gqa=False)
        finally:
            _fu.ex2_emulation_2 = _emu0
            (fi.FlashAttentionForwardSm100, fi._validate_tensor, fi._flash_attn_fwd.compile_cache,
             fi.select_fwd_config, fi.to_cute_tensor) = orig
    return out, lse
