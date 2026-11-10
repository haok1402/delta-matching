"""Validator candidate: the sm100 head_dim 128 kernels of this repository (delta_matching/sm100), deterministic backward.
python validator/validate.py validator/candidates/sm100_hd128.py --shapes quick
debug() also returns the backward's own in-kernel identity (debug build, same arithmetic: per row sum_j dS_ij and
sum_j |P dP|), the debug build's gradients (which must equal production bitwise) and the producer's K-centering mean."""
import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
from delta_matching.sm100.api import dm_attn_func, dm_attn_debug  # noqa: E402
from delta_matching.sm100.quant import quant_qkv2, quant_do_delta  # noqa: E402
from delta_matching.sm100.fwd import dm_fwd  # noqa: E402
from delta_matching.sm100.bwd import dm_bwd_main  # noqa: E402
from delta_matching.sm100.pass0 import dm_delta  # noqa: E402
import delta_matching.sm100.api as _api  # noqa: E402
# the kernels must come from this tree, never from another delta_matching on sys.path
assert os.path.realpath(_api.__file__).startswith(os.path.realpath(_ROOT) + os.sep), _api.__file__

NUMERICS = "sm100-mx"
SUPPORTS = {"seqlen_multiple": 128, "noncausal": True, "matched_noncausal": True}
DET = True


def attn(q, k, v, softmax_scale=None, causal=True, mode="matched"):
    return dm_attn_func(q, k, v, causal=causal, mode=mode, deterministic=DET)


def debug(q, k, v, do, causal, mode):
    d = dict(dm_attn_debug(q, k, v, do, causal=causal, mode=mode, deterministic=DET))
    D = q.shape[-1]
    q8, qs, k8, ks, v8, vs, kvw, kmean_ = quant_qkv2(q.contiguous(), k.contiguous(), v.contiguous(), D ** -0.5)
    o, lse = dm_fwd(q8, qs, k8, kvw, v8, None, causal)
    do8, dos, delta = quant_do_delta(do.contiguous(), o if mode == "stale" else None)
    if mode == "matched":
        delta = dm_delta(q8, qs, k8, ks, v8, vs, do8, dos, lse, causal)
    B, N, H, _ = q.shape
    trace = torch.zeros(B, H, N, 2, dtype=torch.float64, device=q.device)
    old = os.environ.get("DM_BWD_OPTS", "")
    os.environ["DM_BWD_OPTS"] = ",".join(x for x in (old, "rsdbg") if x)
    try:
        dq, dk, dv = dm_bwd_main(q8, qs, k8, ks, v8, vs, do8, dos, lse, delta, causal, D ** -0.5, trace=trace,
                                 deterministic=DET)
    finally:
        os.environ["DM_BWD_OPTS"] = old
    torch.cuda.synchronize()
    d["rowsum_dS"] = trace[..., 0].float()
    d["sum_abs_PdP"] = trace[..., 1].float()
    d["dbg_dq"], d["dbg_dk"], d["dbg_dv"] = dq, dk, dv
    d["kmean"] = kmean_
    return d
