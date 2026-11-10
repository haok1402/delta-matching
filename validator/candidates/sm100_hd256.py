"""Validator candidate: the sm100 head_dim 256 kernels of this repository (delta_matching/sm100), deterministic backward.
python validator/validate.py validator/candidates/sm100_hd256.py --hd 256 --shapes quick
debug() = dm_attn_debug(identity=True) (the dK/dV kernel's own in-kernel rowsum(dS) on the production chunk plan) plus the
debug build's gradients (which must equal production bitwise) and the producer's K-centering mean."""
import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
from delta_matching.sm100.api import dm_attn_func, dm_attn_debug  # noqa: E402
import delta_matching.sm100.api_hd256 as _H  # noqa: E402
import delta_matching.sm100.api as _api  # noqa: E402
assert os.path.realpath(_api.__file__).startswith(os.path.realpath(_ROOT) + os.sep), _api.__file__

NUMERICS = "sm100-mx-d256"
SUPPORTS = {"seqlen_multiple": 128, "noncausal": True, "matched_noncausal": True}
DETERMINISTIC = True


def attn(q, k, v, softmax_scale=None, causal=True, mode="matched"):
    return dm_attn_func(q, k, v, causal=causal, mode=mode, deterministic=True)


def debug(q, k, v, do, causal, mode):
    d = dict(dm_attn_debug(q, k, v, do, causal=causal, mode=mode, identity=True, deterministic=True))
    o, lse, (q8, qs, k8, ks, v8, vs, mean) = _H.dm_fwd_hd256_from_bf16(q.contiguous(), k.contiguous(), v.contiguous(), causal)
    do8, dos, delta = _H._delta_hd256(q8, qs, k8, ks, v8, vs, lse, o, do.contiguous(), mode, causal)
    r = _H._bwd(q8, qs, k8, ks, v8, vs, do8, dos, lse, delta, causal, dbg=1)
    torch.cuda.synchronize()
    d["dbg_dq"], d["dbg_dk"], d["dbg_dv"], d["kmean"] = r[0], r[1], r[2], mean
    return d
