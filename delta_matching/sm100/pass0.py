"""Host entry of the matched-delta pass 0 (DMPass0Sm100): FA4's _flash_attn_fwd plumbing with the kernel class swapped.

dm_delta(q8, qs, k8, ks, v8, vs, do8, dos, lse, causal) -> delta (B, H, N) fp32 =
    sum_j P dP / sum_j P with P = exp2(fma(S, (ks*qs)*log2e, -lse*log2e)), dP = (dO8 V8^T) * (vs*dos)   (dm_common contract)
Scales, dO codes and lse travel as aux tensors; delta comes back in the LSE output slot.
"""
import dataclasses
import os
import threading

import torch
import flash_attn.cute.interface as fi
from flash_attn.cute import cache_utils

from .dm_pass0_sm100 import DMPass0Sm100
from .fwd import _fingerprint

_LOCK = threading.Lock()
FORCE = dict(use_2cta_instrs=False, pack_gqa=False, use_s_ping_pong=False, num_splits=1, use_tma_o=False,
             overlap_sO_sQ=False, kv_stage=4)
REGS = dict(softmax=208, correction=32, other=64)  # P0: correction warps idle -> registers to softmax
_c3r = os.environ.get("DM_P0_C3REGS", "168,96,64").split(",")   # +c3: correction WG computes chunk 3
REGS_C3 = dict(softmax=int(_c3r[0]), correction=int(_c3r[1]), other=int(_c3r[2]))


def _make_cache():
    if cache_utils.CUTE_DSL_CACHE_ENABLED:
        return cache_utils.JITPersistentCache(cache_utils.get_cache_path() / "dm_sm100" / _fingerprint() / "pass0")
    return cache_utils.JITCache()


_CACHES = {}
_CLASSES = {}


def _variant(exp_variant: str):
    if exp_variant not in _CLASSES:
        _CLASSES[exp_variant] = type(f"DMPass0Sm100_{exp_variant.replace('+', '_F_')}", (DMPass0Sm100,), {"dm_exp_variant": exp_variant})
        c = _make_cache()
        if cache_utils.CUTE_DSL_CACHE_ENABLED:
            c = cache_utils.JITPersistentCache(cache_utils.get_cache_path() / "dm_sm100" / _fingerprint() / f"pass0_{exp_variant}")
        _CACHES[exp_variant] = c
    return _CLASSES[exp_variant], _CACHES[exp_variant]


DPDUMP = None
PDUMP = None   # DEBUG (kda-sm100-fix0): set to a (B, H, N, N) fp32 tensor before dm_delta with exp variant "+pd"


def _aux(t: torch.Tensor, leading_dim: int):
    t.__leading_dim__ = leading_dim
    t.__assumed_align__ = 16 if t.dtype == torch.uint8 else 4
    return t


def dm_delta(q8, qs, k8, ks, v8, vs, do8, dos, lse, causal: bool, config_override: dict | None = None,
             exp_variant: str | None = None):
    if exp_variant is None:
        exp_variant = os.environ.get("DM_P0_EXP", "p3_16_0+ma+h2+f64+pf+sq")   # P0-v8 (p0floor): +pf scale prefetch, +sq per-stage specialized code (both bitwise). P0-v7b: fp64 flush of the fp32 partials EVERY tile (most accurate of the measured variants; f64k4 = every 4 tiles), delta in fp64. P0-v6 (fix0): ALL exps via dm_bwd_p_pair (MUFU), bitwise the backward's P -> rowsum(dS)=0 at fp32 round-off; keeps h2 halves + 4 accumulators; /sum P kept
    over = dict(FORCE)
    if config_override:
        over.update(config_override)
    aux = [_aux(qs.contiguous(), 2), _aux(ks.contiguous(), 2), _aux(vs.contiguous(), 2), _aux(dos.contiguous(), 2),
           _aux(do8.contiguous().view(torch.uint8), 3), _aux(lse.contiguous(), 2)]
    if "tr" in exp_variant.split("+")[1:]:
        global TRACE
        TRACE = torch.zeros(64, dtype=torch.int64, device=lse.device)
        aux.append(_aux(TRACE, 0))
    if "pd" in exp_variant.split("+")[1:]:       # DEBUG: (B, H, N, N) fp32 P dump (last aux)
        global PDUMP
        aux.append(_aux(PDUMP, 3))
        aux.append(_aux(DPDUMP, 3))
    orig = (fi.FlashAttentionForwardSm100, fi._flash_attn_fwd.compile_cache, fi.select_fwd_config)

    def select(inputs):
        cfg = orig[2](inputs)
        cfg = dataclasses.replace(cfg, **{k: v for k, v in over.items() if hasattr(cfg, k) and k != "registers"})
        regs = over.get("registers", REGS_C3 if "c3" in exp_variant.split("+")[1:] else REGS)
        for fl in exp_variant.split("+")[1:]:      # +r<softmax>_<correction>: register split (other = 64)
            if fl.startswith("r") and "_" in fl and fl[1:].split("_")[0].isdigit():
                r_ = [int(x) for x in fl[1:].split("_")]
                regs = dict(softmax=r_[0], correction=r_[1], other=r_[2] if len(r_) > 2 else 64)
        if regs is not None:
            cfg = dataclasses.replace(cfg, registers=dataclasses.replace(cfg.registers, **regs))
        return cfg

    with _LOCK:
        cls, cache = _variant(exp_variant)
        fi.FlashAttentionForwardSm100 = cls
        fi._flash_attn_fwd.compile_cache = cache
        fi.select_fwd_config = select
        try:
            _, delta, *_ = fi._flash_attn_fwd(q8, k8, v8, softmax_scale=1.0, causal=causal, return_lse=True,
                                              pack_gqa=False, aux_tensors=aux)
        finally:
            fi.FlashAttentionForwardSm100, fi._flash_attn_fwd.compile_cache, fi.select_fwd_config = orig
    return delta
