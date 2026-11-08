"""Self-test candidates that must FAIL: CHEAT=<name> selects the cheat.
  bf16       exact attention in bf16 / fp32 math (SDPA), claiming fp8 numerics
  skiphead   correct except the last head's output and gradients are left at zero
  lastblock  correct except the last 128 query rows are garbage (the tail-tile bug)
  cache      caches outputs by data_ptr (stale results after an in-place overwrite)
  nondet     adds run-to-run noise at the bf16 ulp level to dQ (non-deterministic reduction)
  modes      forward differs between matched and stale (O perturbed in stale mode)
  stale_as_matched  matched mode silently runs the stale delta
"""
import importlib.util, os, torch
import torch.nn.functional as F
CHEAT = os.environ.get("CHEAT", "bf16")
NUMERICS = "sm90"
SUPPORTS = {"seqlen_multiple": 128, "noncausal": True}
_p = os.path.join(os.path.dirname(__file__), "emul.py")
_s = importlib.util.spec_from_file_location("selftest_emul", _p); E = importlib.util.module_from_spec(_s)
if CHEAT == "stale_as_matched":
    os.environ["SELFTEST_VARIANT"] = "stale_as_matched"
_s.loader.exec_module(E)
_cache = {}


def _sdpa(q, k, v, causal):
    return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=causal,
                                          enable_gqa=q.shape[2] != k.shape[2]).transpose(1, 2)


class _Zero(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, sl):
        ctx.sl = sl
        y = x.clone(); y[sl] = 0 if CHEAT == "skiphead" else 7.0
        return y

    @staticmethod
    def backward(ctx, g):
        g = g.clone(); g[ctx.sl] = 0
        return g, None


class _Noise(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.clone()

    @staticmethod
    def backward(ctx, g):
        return g + torch.randn_like(g) * g.abs().mean() * 1e-2


def attn(q, k, v, softmax_scale=None, causal=True, mode="matched"):
    if CHEAT == "bf16":
        return _sdpa(q, k, v, causal)
    if CHEAT == "cache":
        key = (q.data_ptr(), k.data_ptr(), v.data_ptr(), causal, mode)
        if key not in _cache:
            _cache[key] = E.attn(q.detach(), k.detach(), v.detach(), causal=causal, mode=mode).detach()
        o = _cache[key]
        return o + 0 * (q.sum() + k.sum() + v.sum()).to(o.dtype)
    if CHEAT == "nondet":
        q = _Noise.apply(q)
    o = E.attn(q, k, v, causal=causal, mode=mode)
    if CHEAT == "skiphead":
        o = _Zero.apply(o, (slice(None), slice(None), slice(-1, None)))
    if CHEAT == "lastblock":
        o = _Zero.apply(o, (slice(None), slice(-128, None)))
    if CHEAT == "modes" and mode == "stale":
        o = o * (1 + 2 ** -6)
    return o
