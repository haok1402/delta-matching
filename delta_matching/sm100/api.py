"""sm100 (GB200) hd128 Delta-Matching attention, sm100-mx numerics: dm_attn_func(q, k, v, causal, mode).

forward : quant_qkv2 (fused producers) -> dm_fwd (C1: fp8 QK, MXFP8 PV)
backward: dO quant -> delta (matched: pass-0 kernel dm_delta, sum P dP / sum P; stale: rowsum(dO * O))
          -> dm_bwd_main (B1: 2-CTA, MXFP8 dV / dK / dQ, deterministic)
q (B, N, H, 128) bf16, k / v (B, N, Hk, 128) bf16 -> o (B, N, H, 128) bf16.  N % 128 == 0, softmax scale 128^-0.5.
"""
import os

import torch

from .quant import quant_qkv2, quant_do_delta
from .fwd import dm_fwd
from .bwd import dm_bwd_main
from .pass0 import dm_delta

D = 128


class _DMAttnSm100(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, mode):
        sm = D ** -0.5
        q8, qs, k8, ks, v8, vs, kvw, _ = quant_qkv2(q, k, v, sm)
        if mode == "matched_cp":
            o, lse, obar, kbt = _cp_fwd(q8, qs, k8, kvw, v8, causal)
            ctx.save_for_backward(q8, qs, k8, ks, v8, vs, obar, lse, kbt)
        else:
            o, lse = dm_fwd(q8, qs, k8, kvw, v8, None, causal)
            ctx.save_for_backward(q8, qs, k8, ks, v8, vs, o, lse)
        ctx.causal, ctx.mode = causal, mode
        return o

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, do):
        do = do.contiguous()
        if ctx.mode == "matched_cp":
            q8, qs, k8, ks, v8, vs, obar, lse, kbt = ctx.saved_tensors
            do8, dos, delta = quant_do_delta(do, None, obar=obar)   # delta = dequant(dO8) . Obar (exact code-P sums)
            dq, dk, dv = _cp_bwd(q8, qs, k8, ks, v8, vs, do8, dos, lse, delta, ctx.causal, kbt)
            return dq, dk, dv, None, None
        q8, qs, k8, ks, v8, vs, o, lse = ctx.saved_tensors
        do8, dos, delta = quant_do_delta(do, o if ctx.mode == "stale" else None)   # stale delta fused
        if ctx.mode == "matched":
            delta = dm_delta(q8, qs, k8, ks, v8, vs, do8, dos, lse, ctx.causal)
        dq, dk, dv = dm_bwd_main(q8, qs, k8, ks, v8, vs, do8, dos, lse, delta, ctx.causal, D ** -0.5)
        return dq, dk, dv, None, None


def _with_opts(var, extra):
    """run with `extra` added to the DM_*_OPTS env string var"""
    def wrap(fn):
        def run(*a, **k):
            old = os.environ.get(var, "")
            os.environ[var] = ",".join(x for x in (old, extra) if x)
            try:
                return fn(*a, **k)
            finally:
                os.environ[var] = old
        return run
    return wrap


@_with_opts("DM_FWD_OPTS", "cp=1")
def _cp_fwd(q8, qs, k8, kvw, v8, causal):
    """sm100-mx-cp forward: (O bf16, LSE natural from the code sum, Obar fp32, kbT s16 (B,H,N/32,N))."""
    return dm_fwd(q8, qs, k8, kvw, v8, None, causal)


@_with_opts("DM_BWD_OPTS", "cp")
def _cp_bwd(q8, qs, k8, ks, v8, vs, do8, dos, lse, delta, causal, kbt):
    return dm_bwd_main(q8, qs, k8, ks, v8, vs, do8, dos, lse, delta, causal, D ** -0.5, kbt=kbt)


def dm_attn_func(q, k, v, causal=True, mode="matched"):
    assert q.shape[-1] == D and q.dtype == torch.bfloat16 and q.shape[1] % 128 == 0
    assert mode in ("matched", "stale", "matched_cp")
    q, k, v = (x if x.stride(-1) == 1 else x.contiguous() for x in (q, k, v))
    return _DMAttnSm100.apply(q, k, v, causal, mode)


@torch.no_grad()
def dm_attn_debug(q, k, v, do, causal=True, mode="matched"):
    """Validator hook: the exact delta the backward consumes and the LSE it recomputes P from, for these inputs
    (same producers, forward kernel and delta path as dm_attn_func's autograd backward).  (B,H,N) fp32 each, LSE natural."""
    sm = D ** -0.5
    q, k, v = (x if x.stride(-1) == 1 else x.contiguous() for x in (q, k, v))
    q8, qs, k8, ks, v8, vs, kvw, _ = quant_qkv2(q, k, v, sm)
    if mode == "matched_cp":
        o, lse, obar, kbt = _cp_fwd(q8, qs, k8, kvw, v8, causal)
        do8, dos, delta = quant_do_delta(do.contiguous(), None, obar=obar)
        return {"delta": delta.float().contiguous(), "lse": lse.float().contiguous(), "obar": obar, "kbt": kbt}
    o, lse = dm_fwd(q8, qs, k8, kvw, v8, None, causal)
    do8, dos, delta = quant_do_delta(do.contiguous(), o if mode == "stale" else None)
    if mode == "matched":
        delta = dm_delta(q8, qs, k8, ks, v8, vs, do8, dos, lse, causal)
    return {"delta": delta.float().contiguous(), "lse": lse.float().contiguous()}
