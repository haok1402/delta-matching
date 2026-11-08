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

import flash_attn.cute.interface as fi
from flash_attn.cute import cache_utils, config as fa_config

from .dm_fwd_sm100 import DMForwardSm100

_HERE = Path(__file__).resolve().parent
_LOCK = threading.Lock()
FORCE = dict(use_2cta_instrs=False, pack_gqa=False, use_s_ping_pong=False, num_splits=1)


def _fingerprint():
    h = hashlib.sha256(cache_utils._compute_source_fingerprint().encode())
    h.update(os.environ.get("DM_FWD_OPTS", "").encode())   # kernel variant knobs (read in DMForwardSm100.__init__)
    for f in sorted(_HERE.glob("*.py")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:24]


def _make_cache():
    if cache_utils.CUTE_DSL_CACHE_ENABLED:
        return cache_utils.JITPersistentCache(cache_utils.get_cache_path() / "dm_sm100" / _fingerprint() / "fwd")
    return cache_utils.JITCache()


_CACHE = _make_cache()


def _select(orig_select, over):
    def select(inputs):
        cfg = orig_select(inputs)
        cfg = dataclasses.replace(cfg, **{k: v for k, v in over.items() if hasattr(cfg, k) and k != "kv_stage"})
        overlap, kv_stage = fa_config.select_sm100_smem_layout(
            inputs, tile_m=cfg.tile_m, tile_n=cfg.tile_n, q_stage=cfg.q_stage, num_splits=cfg.num_splits,
            use_2cta_instrs=cfg.use_2cta_instrs, is_static_persistent=cfg.is_static_persistent,
            use_clc_scheduler=cfg.use_clc_scheduler)
        if "kv_stage" in over:
            kv_stage = min(kv_stage, over["kv_stage"])
        return dataclasses.replace(cfg, overlap_sO_sQ=overlap, kv_stage=kv_stage)
    return select


def dm_fwd(q8, qs, k8, ks, v8, vs, causal: bool, config_override: dict | None = None):
    """q8 (B,N,H,D) e4m3 (sm baked), k8/v8 (B,N,Hk,D) e4m3; qs (B,H,N/32), ks/vs (B,Hk,N/32) fp32 powers of 2.
    Returns O (B,N,H,D) bf16, LSE (B,H,N) fp32 natural log."""
    over = dict(FORCE)
    if config_override:
        over.update(config_override)
    qsc, ksc, vsc = qs.contiguous(), ks.contiguous(), vs.contiguous()
    ids = {id(qsc), id(ksc), id(vsc)}
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
        fi._flash_attn_fwd.compile_cache = _CACHE
        fi.select_fwd_config = _select(orig[3], over)
        fi.to_cute_tensor = tct
        try:
            out, lse, *_ = fi._flash_attn_fwd(
                q8, k8, v8, softmax_scale=1.0, causal=causal, return_lse=True,
                q_descale=qsc, k_descale=ksc, v_descale=vsc, pack_gqa=False)
        finally:
            (fi.FlashAttentionForwardSm100, fi._validate_tensor, fi._flash_attn_fwd.compile_cache,
             fi.select_fwd_config, fi.to_cute_tensor) = orig
    return out, lse
