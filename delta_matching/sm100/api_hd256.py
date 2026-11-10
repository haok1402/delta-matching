"""hd256 route of dm_attn_func (api.py branches here on q.shape[-1] == 256 before any hd128 code).
Numerics: sm100-mx at D=256 (USER DECISION 2026-11-10, validator preset sm100-mx-d256).
forward : quant_hd256.quant_qkv_mx256 (hd128 quant.py producers at D=256) -> fwd_hd256.dm_fwd_hd256_mx
backward: dO -> quant_do_mx256 -> delta (matched: pass-0 kernel pass0_hd256.dm_delta_hd256, num/den; stale: quant.quant_do_delta(dO, O))
          -> bwd_hd256mx.dm_bwd_hd256mx (dv / dk / dq kernels, deterministic, MXFP8 out-MMAs).  Same calls as dm-validator's
          validated glue (validate/cands/vm8a2_det.py, 06:2x PASS), so values are bitwise those of that run.
q (B, N, H, 256) bf16, k / v (B, N, Hk, 256) bf16 -> o (B, N, H, 256) bf16. N % 128 == 0, softmax scale 256^-0.5 (= 2^-4)."""
import torch

from .quant_hd256 import quant_qkv_mx256
from .fwd import pack_kv_scales
from .fwd_hd256 import dm_fwd_hd256_mx
from .quant_hd256 import quant_do_mx256
from .quant import quant_do_delta
from .pass0_hd256 import dm_delta_hd256
from .bwd_hd256mx import dm_bwd_hd256mx

D = 256


def dm_fwd_hd256_from_bf16(q, k, v, causal):
    """(o, lse, saved) for bf16 inputs: the exact forward dm_attn_func runs at hd256."""
    q8, qs, k8, ks, v8, vs, mean = quant_qkv_mx256(q, k, v)
    o, lse = dm_fwd_hd256_mx(q8, qs, k8, pack_kv_scales(ks, vs), v8, causal)
    return o, lse, (q8, qs, k8, ks, v8, vs, mean)


def _delta_hd256(q8, qs, k8, ks, v8, vs, lse, o, do, mode, causal):
    """(do8, dos, delta) exactly as the backward consumes them. matched: num/den pass 0 (bitwise the bwd's P/dP)."""
    if mode == "stale":
        return quant_do_delta(do, o)
    do8, dos = quant_do_mx256(do)
    return do8, dos, dm_delta_hd256(q8, qs, k8, ks, v8, vs, do8, dos, lse, causal)


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
        q8, qs, k8, ks, v8, vs, o, lse = ctx.saved_tensors
        do8, dos, delta = _delta_hd256(q8, qs, k8, ks, v8, vs, lse, o, do.contiguous(), ctx.mode, ctx.causal)
        dq, dk, dv = dm_bwd_hd256mx(q8, qs, k8, ks, v8, vs, do8, dos, lse, delta, ctx.causal)
        return dq, dk, dv, None, None, None


def dm_attn_func_hd256(q, k, v, causal=True, mode="matched", deterministic=None):
    assert q.shape[-1] == D and q.dtype == torch.bfloat16 and q.shape[1] % 128 == 0
    assert mode in ("matched", "stale")
    q, k, v = (x if x.stride(-1) == 1 else x.contiguous() for x in (q, k, v))
    return _DMAttnHd256Sm100.apply(q, k, v, causal, mode, True)


@torch.no_grad()
def dm_attn_debug_hd256(q, k, v, do, causal=True, mode="matched", identity=None, deterministic=None):
    """Validator hook at hd256 (same contract as api.dm_attn_debug): delta + LSE the backward consumes; with identity
    (matched mode) also the backward's IN-KERNEL pre-quant rowsum(dS) and sum|P dP| (dbg=1 build, same arithmetic)."""
    import os
    q, k, v = (x if x.stride(-1) == 1 else x.contiguous() for x in (q, k, v))
    o, lse, (q8, qs, k8, ks, v8, vs, _) = dm_fwd_hd256_from_bf16(q, k, v, causal)
    do8, dos, delta = _delta_hd256(q8, qs, k8, ks, v8, vs, lse, o, do.contiguous(), mode, causal)
    out = {"delta": delta.float().contiguous(), "lse": lse.float().contiguous()}
    if identity is None:
        identity = os.environ.get("DM_DEBUG_IDENTITY", "0") == "1"
    if identity and mode == "matched":
        _, _, _, aux = dm_bwd_hd256mx(q8, qs, k8, ks, v8, vs, do8, dos, lse, delta, causal, dbg=1)
        B, N, H, _ = q8.shape
        idt = aux["ident"].reshape(B, H, N, 4)
        out["rowsum_dS"] = idt[..., 0].float().contiguous()
        out["sum_abs_PdP"] = idt[..., 1].float().contiguous()
    return out
