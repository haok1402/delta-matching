"""Self-test candidate (honest): the torch emulation itself wrapped as an autograd function.  Must PASS.
Imports the reference, so it only runs with DMVAL_SELFTEST=1 (which lifts the reference-import check)."""
import importlib.util, os, torch
_p = os.path.join(os.path.dirname(__file__), "..", "..", "delta_matching", "reference.py")
_s = importlib.util.spec_from_file_location("dm_ref_for_selftest", _p); R = importlib.util.module_from_spec(_s)
_s.loader.exec_module(R)
NUMERICS = os.environ.get("SELFTEST_PRESET", "sm90")
SUPPORTS = {"seqlen_multiple": 128, "noncausal": True}
VARIANT = os.environ.get("SELFTEST_VARIANT", "honest")


class _F(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, mode):
        ctx.save_for_backward(q, k, v)
        ctx.causal, ctx.mode = causal, mode
        out = R.dm_reference(q, k, v, torch.zeros_like(q), causal, "stale", NUMERICS)
        ctx.o = out["O"]
        return out["O"].clone()

    @staticmethod
    def backward(ctx, do):
        q, k, v = ctx.saved_tensors
        mode = ctx.mode
        if VARIANT == "stale_as_matched":
            mode = "stale"
        out = R.dm_reference(q, k, v, do.contiguous(), ctx.causal, mode, NUMERICS, o_for_stale=ctx.o)
        return out["dQ"].to(q.dtype), out["dK"].to(k.dtype), out["dV"].to(v.dtype), None, None


def attn(q, k, v, softmax_scale=None, causal=True, mode="matched"):
    return _F.apply(q, k, v, causal, mode)
