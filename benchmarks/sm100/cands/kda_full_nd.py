"""fa4-bench wrapper: kda-sm100 full candidate in its NON-DETERMINISTIC backward mode (dm_attn_func(..., deterministic=False),
available from 43db3c7), from a pinned snapshot (env KDA_SNAP).  Refuses snapshots without the switch (no silent det run)."""
import inspect, os, subprocess, sys
SNAP = os.environ["KDA_SNAP"]
sys.path.insert(0, SNAP)
from delta_matching.sm100.api import dm_attn_func  # noqa: E402
if "deterministic" not in inspect.signature(dm_attn_func).parameters:
    raise RuntimeError(f"{SNAP}: dm_attn_func has no deterministic switch (needs 43db3c7 or later)")
_r = lambda *a: subprocess.run(["git", "-C", SNAP, *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
CAND_INFO = {"snap": SNAP, "sha": _r("rev-parse", "--short", "HEAD"), "deterministic": False}
NUMERICS = "sm100-mx"
SUPPORTS = {"seqlen_multiple": 128, "noncausal": True, "matched_noncausal": True}


def attn(q, k, v, causal, mode):
    return dm_attn_func(q, k, v, causal=causal, mode=mode, deterministic=False)
