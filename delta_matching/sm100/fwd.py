"""Host entry of the sm100-mx forward: reuses FA4's _flash_attn_fwd plumbing with DMForwardSm100 swapped in.

Compile cache: persistent (FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 + FLASH_ATTENTION_CUTE_DSL_CACHE_DIR) under a
fingerprint of FA4's sources AND this package's kernel sources, so AOT compiles (aot.py, fake tensors, no GPU) are
reused by GPU runs.  The compile key does not depend on B / N, only on (causal, GQA ratio, config).
"""
import dataclasses
import threading

import torch
import flash_attn.cute.interface as fi
from flash_attn.cute import cache_utils, config as fa_config

from .dm_fwd_sm100 import DMForwardSm100
from . import launch_cache_sm100
from .compile_cache_sm100 import compile_env, fingerprint

_LOCK = threading.Lock()
# ex2_emu_freq=0: every forward exp on MUFU ex2.approx (no FA4 deg-3 FMA polynomial): vs FA4's causal fp8 emulation
# (deg-3 on 25% of exps) all-MUFU has 100-1000x fewer P codes differing from RN_e4m3(exact 2^x), ~9x (up to 46x) lower
# LSE error and no worse O.
FORCE = dict(use_2cta_instrs=False, pack_gqa=False, use_s_ping_pong=False, num_splits=1, ex2_emu_freq=0)


_CACHES = {}


def _cache():
    """One compile cache per compile-time environment."""
    mk = compile_env()   # memo keyed on every compile-time input
    if mk not in _CACHES:
        if cache_utils.CUTE_DSL_CACHE_ENABLED:
            _CACHES[mk] = cache_utils.JITPersistentCache(cache_utils.get_cache_path() / "dm_sm100" / fingerprint("") / "fwd")
        else:
            _CACHES[mk] = cache_utils.JITCache()
    return _CACHES[mk]


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
        cfg = dataclasses.replace(cfg, **{k: v for k, v in over.items() if hasattr(cfg, k)})
        overlap, kv_stage = fa_config.select_sm100_smem_layout(
            inputs, tile_m=cfg.tile_m, tile_n=cfg.tile_n, q_stage=cfg.q_stage, num_splits=cfg.num_splits,
            use_2cta_instrs=cfg.use_2cta_instrs, is_static_persistent=cfg.is_static_persistent,
            use_clc_scheduler=cfg.use_clc_scheduler)
        return dataclasses.replace(cfg, overlap_sO_sQ=overlap, kv_stage=kv_stage)
    return select


def _ue8m0_words(s):
    """(B, H, NB) fp32 powers of 2 -> (B, H, NB/4) int32 of 4 packed UE8M0 bytes (block 4j+i in byte i)."""
    b = ((s.contiguous().view(torch.int32) >> 23) & 0xFF).to(torch.uint8)
    return b.view(*s.shape[:-1], s.shape[-1] // 4, 4).contiguous().view(torch.int32).squeeze(-1)


def pack_kv_scales(ks, vs):
    """(B, Hk, NB) K and V scales -> (B, Hk, 2 * NB/4) int32: [2t] = K bytes of n_block t, [2t+1] = V bytes (SFB word)."""
    return torch.stack([_ue8m0_words(ks), _ue8m0_words(vs)], dim=-1).flatten(-2).contiguous()


def dm_fwd(q8, qs, k8, ks, v8, vs, causal: bool):
    """q8 (B,N,H,D) e4m3 (sm baked), k8/v8 (B,N,Hk,D) e4m3; qs (B,H,N/32), ks/vs (B,Hk,N/32) fp32 powers of 2.
    ks may be passed pre-packed (int32, from pack_kv_scales; producer side) with vs=None.
    Returns O (B,N,H,D) bf16, LSE (B,H,N) fp32 natural log."""
    over = dict(FORCE)
    qsc, ksc, vsc = qs.contiguous(), (ks if ks.dtype == torch.int32 else pack_kv_scales(ks, vs)), None
    return launch_cache_sm100.call("fwd", dict(q8=q8, k8=k8, v8=v8, qsc=qsc, ksc=ksc, vsc=vsc),
                         (bool(causal), repr(sorted(over.items())), compile_env()),
                         lambda: _dm_fwd_slow(q8, k8, v8, qsc, ksc, vsc, causal, over))


def _dm_fwd_slow(q8, k8, v8, qsc, ksc, vsc, causal, over):
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

    with _LOCK:
        fi.FlashAttentionForwardSm100 = DMForwardSm100
        fi._validate_tensor = validate
        fi._flash_attn_fwd.compile_cache = launch_cache_sm100.RecCache(_cache())
        fi.select_fwd_config = _select(orig[3], over)
        fi.to_cute_tensor = tct
        try:
            out, lse, *_ = fi._flash_attn_fwd(
                q8, k8, v8, softmax_scale=1.0, causal=causal, return_lse=True,
                q_descale=qsc, k_descale=ksc, v_descale=vsc, pack_gqa=False)
            ret = (out, lse)
        finally:
            (fi.FlashAttentionForwardSm100, fi._validate_tensor, fi._flash_attn_fwd.compile_cache,
             fi.select_fwd_config, fi.to_cute_tensor) = orig
    return ret
