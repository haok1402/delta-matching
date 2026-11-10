"""
Delta-Matching attention at head_dim 256 on SM100, MXFP8 (spec sm100-mx-d256).
Forward: quant_qkv2 (softmax scale 2^-4 baked into Q), then the forward kernel.
Backward: quant_do_delta, the delta (matched: the num / den pass 0, bitwise on
the backward's P and dP; stale: rowsum(dO * O)), then attention_bwd_hd256.
"""
import torch

from .attention_fwd_hd256_sm100 import attention_fwd_hd256
from .quant import quant_qkv2, quant_do_delta
from .pass0_hd256 import dm_delta_hd256
from .attention_bwd_hd256_sm100 import attention_bwd_hd256

__all__ = ["dm_attn_func_hd256_sm100", "dm_attn_debug_hd256_sm100", "forward_from_bf16", "delta_for_backward"]

D = 256
SOFTMAX_SCALE = D ** -0.5     # baked into Q by quant_qkv2 (= 2^-4, exact)


def forward_from_bf16(q, k, v, causal):
    """
    (o, lse, (q8, qs, k8, ks, v8, vs, k_mean)) for bf16 inputs: the forward dm_attn_func runs.
    """
    q8, qs, k8, ks, v8, vs, kvw, mean = quant_qkv2(q, k, v, SOFTMAX_SCALE)   # kvw == pack_kv_scales(ks, vs)
    o, lse = attention_fwd_hd256(q8, qs, k8, kvw, v8, causal)
    return o, lse, (q8, qs, k8, ks, v8, vs, mean)


def delta_for_backward(q8, qs, k8, ks, v8, vs, lse, o, do, mode, causal):
    """
    (do8, dos, delta) exactly as the backward consumes them; matched: the num / den pass 0.
    """
    if mode == "stale":
        return quant_do_delta(do, o)
    do8, dos, _ = quant_do_delta(do, None)
    return do8, dos, dm_delta_hd256(q8, qs, k8, ks, v8, vs, do8, dos, lse, causal)


class AttentionHd256Sm100(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, mode, det):
        o, lse, (q8, qs, k8, ks, v8, vs, _) = forward_from_bf16(q, k, v, causal)
        ctx.save_for_backward(q8, qs, k8, ks, v8, vs, o, lse)
        ctx.causal, ctx.mode, ctx.det = causal, mode, det
        return o

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, do):
        q8, qs, k8, ks, v8, vs, o, lse = ctx.saved_tensors
        do8, dos, delta = delta_for_backward(q8, qs, k8, ks, v8, vs, lse, o, do.contiguous(), ctx.mode, ctx.causal)
        dq, dk, dv = attention_bwd_hd256(q8, qs, k8, ks, v8, vs, do8, dos, lse, delta, ctx.causal)
        return dq, dk, dv, None, None, None


def dm_attn_func_hd256_sm100(q, k, v, causal=True, mode="matched", deterministic=None):
    """
    dm_attn_func at head_dim 256; the backward is always deterministic.
    """
    assert q.shape[-1] == D and q.dtype == torch.bfloat16 and q.shape[1] % 128 == 0
    assert mode in ("matched", "stale")
    q, k, v = (x if x.stride(-1) == 1 else x.contiguous() for x in (q, k, v))
    return AttentionHd256Sm100.apply(q, k, v, causal, mode, True)


@torch.no_grad()
def dm_attn_debug_hd256_sm100(q, k, v, do, causal=True, mode="matched", identity=None, deterministic=None):
    """
    dm_attn_debug at head_dim 256: the delta and LSE the backward consumes. With identity (matched
    mode) also the backward's in-kernel rowsum(dS) and sum |P dP| over the fp32 dS that KV2
    quantizes, from the same chunk plan as the production call.
    """
    import os
    q, k, v = (x if x.stride(-1) == 1 else x.contiguous() for x in (q, k, v))
    o, lse, (q8, qs, k8, ks, v8, vs, _) = forward_from_bf16(q, k, v, causal)
    do8, dos, delta = delta_for_backward(q8, qs, k8, ks, v8, vs, lse, o, do.contiguous(), mode, causal)
    out = {"delta": delta.float().contiguous(), "lse": lse.float().contiguous()}
    if identity is None:
        identity = os.environ.get("DM_DEBUG_IDENTITY", "0") == "1"
    if identity and mode == "matched":
        B, N, H, _ = q8.shape
        sums = attention_bwd_hd256(q8, qs, k8, ks, v8, vs, do8, dos, lse, delta, causal, identity=True)[3].view(torch.float64).reshape(B, H, N, 4)
        out["rowsum_dS"] = sums[..., 0].float().contiguous()
        out["sum_abs_PdP"] = sums[..., 1].float().contiguous()
    return out
