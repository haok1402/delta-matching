"""sm100 (GB200) hd128 Delta-Matching attention, sm100-mx numerics: dm_attn_func(q, k, v, causal, mode).

forward : quant_qkv2 (fused producers) -> dm_fwd (C1: fp8 QK, MXFP8 PV)
backward: dO quant -> delta (matched: pass-0 kernel dm_delta, sum P dP / sum P; stale: rowsum(dO * O))
          -> dm_bwd_main (B1: 2-CTA, MXFP8 dV / dK / dQ, deterministic)
q (B, N, H, 128) bf16, k / v (B, N, Hk, 128) bf16 -> o (B, N, H, 128) bf16.  N % 128 == 0, softmax scale 128^-0.5.
"""
import torch

from .quant import quant_qkv2, quant_rows
from .fwd import dm_fwd
from .bwd import dm_bwd_main
from .pass0 import dm_delta

D = 128


class _DMAttnSm100(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, mode):
        sm = D ** -0.5
        q8, qs, k8, ks, v8, vs, kvw, _ = quant_qkv2(q, k, v, sm)
        o, lse = dm_fwd(q8, qs, k8, kvw, v8, None, causal)
        ctx.save_for_backward(q8, qs, k8, ks, v8, vs, o, lse)
        ctx.causal, ctx.mode = causal, mode
        return o

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, do):
        q8, qs, k8, ks, v8, vs, o, lse = ctx.saved_tensors
        do = do.contiguous()
        do8, dos = quant_rows(do)
        if ctx.mode == "matched":
            delta = dm_delta(q8, qs, k8, ks, v8, vs, do8, dos, lse, ctx.causal)
        else:
            delta = (do.float() * o.float()).sum(-1).permute(0, 2, 1).contiguous()
        dq, dk, dv = dm_bwd_main(q8, qs, k8, ks, v8, vs, do8, dos, lse, delta, ctx.causal, D ** -0.5)
        return dq, dk, dv, None, None


def dm_attn_func(q, k, v, causal=True, mode="matched"):
    assert q.shape[-1] == D and q.dtype == torch.bfloat16 and q.shape[1] % 128 == 0
    assert mode in ("matched", "stale")
    q, k, v = (x if x.stride(-1) == 1 else x.contiguous() for x in (q, k, v))
    return _DMAttnSm100.apply(q, k, v, causal, mode)
