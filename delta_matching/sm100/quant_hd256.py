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


# ----------------------------------------------------------------------------------------------------------------------
# MXFP8 (sm100-mx at D=256, dm-validator preset "sm100-mx-d256") = THE hd256 spec since the USER DECISION 2026-11-10.
# The hd128 sm100-mx producers of quant.py run unchanged at D=256 (they are D-generic): one pow2 (UE8M0-representable)
# scale per 32 tokens over all of D, code = rn_e4m3(y * 2^-e), Q pre-multiplied by sm = 256^-0.5, K centered by its
# fixed-order fp32 mean (quant_qkv2), dO plain.  Bitwise = quant.quant_rows_ref with the returned mean (t_p0mx checks).
# hd256-fwd owns the production hd256 producer; reconcile against these if it differs.
# ----------------------------------------------------------------------------------------------------------------------
def quant_qkv_mx256(q, k, v):
    """q (B, N, H, 256), k / v (B, N, Hk, 256) bf16 -> (q8, qs, k8, ks, v8, vs, mean): codes (B, N, H|Hk, 256) e4m3,
    scales (B, H|Hk, N/32) fp32 powers of two, mean (B, Hk, 256) fp32 (the K-centering mean)."""
    from . import quant as Q
    assert q.shape[-1] == D and q.shape[1] % 128 == 0
    q8, qs, k8, ks, v8, vs, _kvw, mean = Q.quant_qkv2(q, k, v, D ** -0.5)
    return q8, qs, k8, ks, v8, vs, mean


def quant_do_mx256(do):
    """do (B, N, H, 256) bf16 -> do8 (B, N, H, 256) e4m3, dos (B, H, N/32) fp32 powers of two."""
    from . import quant as Q
    assert do.shape[-1] == D
    do8, dos, _ = Q.quant_do_delta(do.contiguous(), None)
    return do8, dos


def quant_qkv_mx256_kvw(q, k, v):
    """quant_qkv_mx256 + the packed UE8M0 K/V words quant_qkv2 writes in the same launches (hd256-8k G1): (q8, qs, k8, ks, v8,
    vs, mean, kvw), kvw (B, Hk, N/64) int32 == fwd.pack_kv_scales(ks, vs) byte for byte ([2t] K bytes, [2t+1] V bytes of
    128-key tile t, byte i = biased exponent of 32-key block 4t+i), without the 7 extra elementwise/cat launches."""
    from . import quant as Q
    assert q.shape[-1] == D and q.shape[1] % 128 == 0
    q8, qs, k8, ks, v8, vs, kvw, mean = Q.quant_qkv2(q, k, v, D ** -0.5)
    return q8, qs, k8, ks, v8, vs, mean, kvw
