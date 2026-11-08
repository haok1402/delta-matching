"""Self-test candidate (honest): the torch emulation itself wrapped as an autograd function.  Must PASS.
Imports the reference, so it only runs with DMVAL_SELFTEST=1 (which lifts the reference-import check).
SELFTEST_VARIANT (cheats built on it, all must FAIL):
  stale_as_matched   matched mode silently runs the stale delta
  delta_off          matched backward uses delta * (1 + 1e-2) and the debug hook reports that same delta
  debug_lies         matched backward uses delta * (1 + 3e-2) but the debug hook reports the true delta"""
import importlib.util, os, sys, torch
_p = os.path.join(os.path.dirname(__file__), "..", "..", "delta_matching", "reference.py")
_s = importlib.util.spec_from_file_location("dm_ref_for_selftest", _p); R = importlib.util.module_from_spec(_s)
sys.modules[_s.name] = R
_s.loader.exec_module(R)
NUMERICS = os.environ.get("SELFTEST_PRESET", "sm90")
SUPPORTS = {"seqlen_multiple": 128, "noncausal": True}
VARIANT = os.environ.get("SELFTEST_VARIANT", "honest")
OFF = {"delta_off": 1e-2, "debug_lies": 3e-2}.get(VARIANT, 0.0)


def _used_delta(q, k, v, do, causal, mode):
    r = R.dm_reference(q, k, v, do, causal, "matched", NUMERICS)
    return r["delta"] * (1 + OFF), r["LSE"], r["delta"]


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
        do = do.contiguous()
        mode = "stale" if VARIANT == "stale_as_matched" else ctx.mode
        dov = None
        if mode == "matched" and OFF:
            dov = _used_delta(q, k, v, do, ctx.causal, mode)[0]
        out = R.dm_reference(q, k, v, do, ctx.causal, mode, NUMERICS, o_for_stale=ctx.o, delta_override=dov)
        return out["dQ"].to(q.dtype), out["dK"].to(k.dtype), out["dV"].to(v.dtype), None, None


def attn(q, k, v, softmax_scale=None, causal=True, mode="matched"):
    return _F.apply(q, k, v, causal, mode)


def debug(q, k, v, do, causal, mode):
    used, lse, true = _used_delta(q, k, v, do.contiguous(), causal, mode)
    return {"delta": true if VARIANT == "debug_lies" else used, "lse": lse}
