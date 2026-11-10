"""hd256 route of dm_attn_func (api.py branches here on q.shape[-1] == 256 before any hd128 code).
Numerics: sm100-mx at D=256 (USER DECISION 2026-11-10, validator preset sm100-mx-d256).
forward : quant_hd256.quant_qkv_mx256 (hd128 quant.py producers at D=256) -> fwd_hd256.dm_fwd_hd256_mx
backward: pending (kda-hd256 track H2); raises NotImplementedError until the hd256 backward lands.
q (B, N, H, 256) bf16, k / v (B, N, Hk, 256) bf16 -> o (B, N, H, 256) bf16. N % 128 == 0, softmax scale 256^-0.5 (= 2^-4)."""
import torch

from .quant_hd256 import quant_qkv_mx256
from .fwd import pack_kv_scales
from .fwd_hd256 import dm_fwd_hd256_mx

D = 256


def dm_fwd_hd256_from_bf16(q, k, v, causal):
    """(o, lse, saved) for bf16 inputs: the exact forward dm_attn_func runs at hd256."""
    q8, qs, k8, ks, v8, vs, mean = quant_qkv_mx256(q, k, v)
    o, lse = dm_fwd_hd256_mx(q8, qs, k8, pack_kv_scales(ks, vs), v8, causal)
    return o, lse, (q8, qs, k8, ks, v8, vs, mean)


class _DMAttnHd256Sm100(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, mode, det):
        o, lse, (q8, qs, k8, ks, v8, vs, _) = dm_fwd_hd256_from_bf16(q, k, v, causal)
        ctx.save_for_backward(q8, qs, k8, ks, v8, vs, o, lse)
        ctx.causal, ctx.mode, ctx.det = causal, mode, det
        return o

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, do):
        raise NotImplementedError("Delta-Matching hd256 backward is not landed yet (kda-hd256 track H2)")


def dm_attn_func_hd256(q, k, v, causal=True, mode="matched", deterministic=None):
    assert q.shape[-1] == D and q.dtype == torch.bfloat16 and q.shape[1] % 128 == 0
    assert mode in ("matched", "stale")
    q, k, v = (x if x.stride(-1) == 1 else x.contiguous() for x in (q, k, v))
    return _DMAttnHd256Sm100.apply(q, k, v, causal, mode, True)
