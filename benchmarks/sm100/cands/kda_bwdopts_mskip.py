"""fa4-bench wrapper: kda full candidate (KDA_SNAP) with DM_BWD_OPTS='nomskipu,mskip' set around each call (deterministic mode).
kda's bwd keeps one compile cache per DM_BWD_OPTS string, so several variants can be timed interleaved in one process."""
import os, subprocess, sys
SNAP = os.environ["KDA_SNAP"]
sys.path.insert(0, SNAP)
from delta_matching.sm100.api import dm_attn_func  # noqa: E402
OPTS = "nomskipu,mskip"
CAND_INFO = {"snap": SNAP, "sha": subprocess.run(["git", "-C", SNAP, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(), "DM_BWD_OPTS": OPTS}
NUMERICS = "sm100-mx"
SUPPORTS = {"seqlen_multiple": 128, "noncausal": True}


class _Opts:
    def __enter__(self):
        self.old = os.environ.get("DM_BWD_OPTS")
        os.environ["DM_BWD_OPTS"] = OPTS
    def __exit__(self, *a):
        if self.old is None: os.environ.pop("DM_BWD_OPTS", None)
        else: os.environ["DM_BWD_OPTS"] = self.old


class _F:
    pass


import torch  # noqa: E402


class _Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, mode):
        with _Opts():
            with torch.enable_grad():
                qq, kk, vv = (t.detach().requires_grad_() for t in (q, k, v))
                o = dm_attn_func(qq, kk, vv, causal=causal, mode=mode)
        ctx.saved = (qq, kk, vv, o)
        return o.detach()

    @staticmethod
    def backward(ctx, do):
        qq, kk, vv, o = ctx.saved
        with _Opts():
            dq, dk, dv = torch.autograd.grad(o, (qq, kk, vv), do)
        return dq, dk, dv, None, None


def attn(q, k, v, causal, mode):
    return _Wrap.apply(q, k, v, causal, mode)
