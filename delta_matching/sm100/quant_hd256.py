"""hd256 producers for the sm100 kernels: the [64,32] fp32 block-scale contract (spec sm90-numerics-spec-hd256 §1),
realized by the sm90 Triton producers of this tree UNCHANGED (delta_matching/producers.py == delta-matching 51ea382:
quant_q_m17 / quant_kv_m17 / doq_m17, blocks (64, 32)); only the code layout is converted to the sm100 kernels'
(B, N, H, 256).  Spec §8.2.1: reuse the producers so codes, non-dyadic fp32 scales, centered K and sm-baked Q are bitwise.
hd128's quant.py is untouched.

  quant_qkv_hd256(q, k, v)  q (B, N, H, 256), k / v (B, N, Hk, 256) bf16 ->
      q8 (B, N, H, 256) e4m3, qs (B, H, N/64) fp32, k8, ks (B, Hk, N/32), v8, vs (B, Hk, N/32)
      (Q is pre-multiplied by sm = 256^-0.5 = 2^-4 before quantization; K is centered by its per-head mean)
  quant_do_hd256(do)        do (B, N, H, 256) bf16 -> do8 (B, N, H, 256) e4m3, dos (B, H, N/64) fp32
"""
import torch

from .. import producers as PR

D = 256
BLK_Q, BLK_K = 64, 32


def _bnhd(x8, B, H, N):
    """[B*H, N, D] -> (B, N, H, D) contiguous (layout only)."""
    return x8.view(B, H, N, D).transpose(1, 2).contiguous()


def quant_qkv_hd256(q, k, v):
    B, N, H, Dq = q.shape
    Hk = k.shape[2]
    assert Dq == D and q.dtype == torch.bfloat16 and N % 128 == 0
    q8, qs, _ = PR.quant_q_m17(q.transpose(1, 2), BLK_Q, want_twin=False)
    k8, ks, _kt, v8, vs = PR.quant_kv_m17(k.transpose(1, 2), v.transpose(1, 2), BLK_K)
    return (_bnhd(q8, B, H, N), qs.view(B, H, N // BLK_Q), _bnhd(k8, B, Hk, N), ks.view(B, Hk, N // BLK_K),
            _bnhd(v8, B, Hk, N), vs.view(B, Hk, N // BLK_K))


def quant_do_hd256(do):
    B, N, H, Dq = do.shape
    assert Dq == D and do.dtype == torch.bfloat16
    do = do if do.stride(-1) == 1 else do.contiguous()
    do8, dos, _, _ = PR.doq_m17(do, None, H, BLK_Q, want_delta=False, want_twin=False)
    return _bnhd(do8, B, H, N), dos.view(B, H, N // BLK_Q)
