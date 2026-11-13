"""
Autograd units on the q/k/v projection output y (head_dim 256; columns q_h|gate_h per head, then k, then v):

    o, gate = qkv_dm_attention(y, gq, gk, cos, sin, H, G, eps, mode="matched", bn=None, dy_post=None)
        DM arm: fused producers (qkv_fwd_sm100.fwd_producers) -> the PR #3 hd256 forward; backward = the PR #3
        backward (dO quant, matched pass 0 / stale delta, KV2 + dQ) -> fused producer backward (qkv_bwd_sm100).
        Bitwise the unfused chain (TE rmsnorm -> RoPE (native bf16 table) -> attention.dm_attn_func) at CP 1.
    q, k, v, gate = qkv_bf16(y, gq, gk, cos, sin, H, G, eps, bn=None, dy_post=None)
        BF16 arm: the same fused norm + RoPE in bf16 (fwd_bf16) and the same fused backward; v and gate are views of y.

y is (B, N, C) or (B*N, C) with bn = (B, N); gq / gk the q / k norm weights (D,); cos / sin (B, N, R) bf16 (this
rank's rows of the native table under CP). Outputs are (B, N, h, D). gate (and v) are views of y: read them only.
dy_post(dY, amax, scale) -> grad of y (e.g. TE's e5m2 Float8Tensor built from the kernel's amax / scale); None = the
bf16 dY. Ring CP for the DM arm goes through the ring orchestrator with fwd_producers_ksum / k_quant_mean instead.
"""
import torch

from . import qkv_bwd_sm100 as _B
from . import qkv_fwd_sm100 as _F

__all__ = ["qkv_dm_attention", "qkv_bf16"]


def _y3(y, bn):
    if y.dim() == 3:
        return y
    B, N = bn
    return y.view(B, N, y.shape[-1])


def _grads(y, gq, gk, cos, sin, H, G, eps, rq, rk, dq, dk, dv, dgate, dy_post):
    """the fused producer backward -> (grad y, grad gq, grad gk)."""
    B, N = rq.shape[0], rq.shape[2]
    D = 256
    dev = y.device
    if dq is None:
        dq = torch.zeros(B, N, H, D, device=dev, dtype=torch.bfloat16)
    if dk is None:
        dk = torch.zeros(B, N, G, D, device=dev, dtype=torch.bfloat16)
    if dv is None:
        dv = torch.zeros(B, N, G, D, device=dev, dtype=torch.bfloat16)
    if dgate is None:
        dgate = torch.zeros(B, N, H, D, device=dev, dtype=torch.bfloat16)
    r = _B.bwd_producers(dq, dk, dv, dgate.transpose(0, 1), y, gq, gk, cos, sin, H, G, eps, rstd_q=rq, rstd_k=rk,
                         want_amax=dy_post is not None)
    dY, dgq, dgk = r[:3]
    if dy_post is not None:
        dY = dy_post(dY, r[3], r[4])
    return dY, dgq.to(gq.dtype), dgk.to(gk.dtype)


class _QKVDMAttn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, y, gq, gk, cos, sin, H, G, eps, mode, bn, dy_post):
        from ..fwd_hd256 import dm_fwd_hd256_mx
        y3 = _y3(y, bn)
        r = _F.fwd_producers(y3, gq, gk, cos, sin, H, G, cos.shape[-1], eps)
        o, lse = dm_fwd_hd256_mx(r["q8"], r["qs"], r["k8"], r["kvw"], r["v8"], True)
        ctx.save_for_backward(y, gq, gk, cos, sin, r["q8"], r["qs"], r["k8"], r["ks"], r["v8"], r["vs"], o, lse,
                              r["rstd_q"], r["rstd_k"])
        ctx.meta = (H, G, float(eps), mode, dy_post)
        return o, r["gate"]

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, do, dgate):
        from ..api_hd256 import _delta_hd256
        from ..attention_bwd_hd256_sm100 import attention_bwd_hd256
        y, gq, gk, cos, sin, q8, qs, k8, ks, v8, vs, o, lse, rq, rk = ctx.saved_tensors
        H, G, eps, mode, dy_post = ctx.meta
        if do is None:
            do = torch.zeros_like(o)
        do8, dos, delta = _delta_hd256(q8, qs, k8, ks, v8, vs, lse, o, do.contiguous(), mode, True)
        dq, dk, dv = attention_bwd_hd256(q8, qs, k8, ks, v8, vs, do8, dos, lse, delta, True)
        dY, dgq, dgk = _grads(y, gq, gk, cos, sin, H, G, eps, rq, rk, dq, dk, dv, dgate, dy_post)
        return dY, dgq, dgk, None, None, None, None, None, None, None, None


class _QKVBF16(torch.autograd.Function):

    @staticmethod
    def forward(ctx, y, gq, gk, cos, sin, H, G, eps, bn, dy_post):
        y3 = _y3(y, bn)
        qb, kb, rq, rk = _F.fwd_bf16(y3, gq, gk, cos, sin, H, G, cos.shape[-1], eps)
        ctx.save_for_backward(y, gq, gk, cos, sin, rq, rk)
        ctx.meta = (H, G, float(eps), dy_post)
        D = 256
        yv = y3.unflatten(-1, (-1, D))
        v = yv[:, :, 2 * H + G:]
        gate = yv[:, :, : 2 * H].unflatten(2, (H, 2))[:, :, :, 1]
        return qb, kb, v, gate

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dq, dk, dv, dgate):
        y, gq, gk, cos, sin, rq, rk = ctx.saved_tensors
        H, G, eps, dy_post = ctx.meta
        dY, dgq, dgk = _grads(y, gq, gk, cos, sin, H, G, eps, rq, rk, dq, dk, dv, dgate, dy_post)
        return dY, dgq, dgk, None, None, None, None, None, None, None


def _args(y, gq, gk, cos, sin, H, G, bn):
    if y.dim() == 2 and bn is None:
        raise ValueError("qkv producers: a 2D y needs bn=(B, N)")
    if gq.dtype != torch.bfloat16 or gk.dtype != torch.bfloat16:
        raise ValueError("qkv producers: bf16 norm weights expected (TE rmsnorm math)")
    return y, gq, gk, cos.contiguous(), sin.contiguous()


def qkv_dm_attention(y, gq, gk, cos, sin, H, G, eps=1e-6, mode="matched", bn=None, dy_post=None):
    """DM arm (CP 1): (o (B, N, H, 256) bf16, gate view (B, N, H, 256))."""
    if mode not in ("matched", "stale"):
        raise ValueError(f"mode must be matched | stale, got {mode!r}")
    y, gq, gk, cos, sin = _args(y, gq, gk, cos, sin, H, G, bn)
    return _QKVDMAttn.apply(y, gq, gk, cos, sin, int(H), int(G), float(eps), mode, bn, dy_post)


def qkv_bf16(y, gq, gk, cos, sin, H, G, eps=1e-6, bn=None, dy_post=None):
    """BF16 arm: (q_rot (B, N, H, 256), k_rot (B, N, G, 256), v view, gate view), all bf16."""
    y, gq, gk, cos, sin = _args(y, gq, gk, cos, sin, H, G, bn)
    return _QKVBF16.apply(y, gq, gk, cos, sin, int(H), int(G), float(eps), bn, dy_post)
