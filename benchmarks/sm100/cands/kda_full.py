"""fa4-bench wrapper: kda-sm100 full candidate (dm_attn_func: fwd + P0 matched delta + deterministic bwd) from a PINNED
snapshot worktree (env KDA_SNAP, default bench/snap/6a31419), never the live worktree."""
import os, subprocess, sys
SNAP = os.environ.get("KDA_SNAP", "/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/sm100/bench/snap/6a31419")
sys.path.insert(0, SNAP)
from delta_matching.sm100.api import dm_attn_func  # noqa: E402
_r = lambda *a: subprocess.run(["git", "-C", SNAP, *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
CAND_INFO = {"snap": SNAP, "sha": _r("rev-parse", "--short", "HEAD"), "dirty": bool(_r("status", "--short", "--untracked-files=no"))}
NUMERICS = "sm100-mx"
SUPPORTS = {"seqlen_multiple": 128, "noncausal": True, "matched_noncausal": True}


def attn(q, k, v, causal, mode):
    return dm_attn_func(q, k, v, causal=causal, mode=mode)
