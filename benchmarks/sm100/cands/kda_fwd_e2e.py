"""fa4-bench wrapper: kda-sm100 forward candidate, END TO END from bf16 (producers quant_qkv2 + dm_fwd kernel), forward only.
Source: worktree $KDA_FWD_WT (default sm100/wt/kda-sm100-fwd); the git sha is recorded in CAND_INFO."""
import os, subprocess, sys
WT = os.environ.get("KDA_FWD_WT", "/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/sm100/wt/kda-sm100-fwd")
sys.path.insert(0, WT)
from delta_matching.sm100.fwd import dm_fwd           # noqa: E402
from delta_matching.sm100.quant import quant_qkv2     # noqa: E402
CAND_INFO = {"wt": WT, "sha": subprocess.run(["git", "-C", WT, "rev-parse", "--short", "HEAD"], capture_output=True,
                                             text=True).stdout.strip(), "opts": os.environ.get("DM_FWD_OPTS", "")}
FWD_ONLY = True
SUPPORTS = {"seqlen_multiple": 128, "noncausal": True}


def attn(q, k, v, causal=True, mode="matched"):
    q8, qs, k8, ks, v8, vs, kvw, _ = quant_qkv2(q, k, v, q.shape[-1] ** -0.5)
    return dm_fwd(q8, qs, k8, kvw, v8, None, causal)[0]
