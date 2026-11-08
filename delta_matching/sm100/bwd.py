"""Host entry of the sm100-mx backward main pass: FA4's _flash_attn_bwd host code, source-patched (no edits to the
shared FA4 checkout) to run DMBackwardSm100 on fp8 codes, with our delta / LSE instead of FA4's preprocess, scales
passed as aux tensors [qs, ks, vs, dos], bf16 outputs, and no softmax-scale factor on dK (Q codes carry sm)."""
import inspect
import math
import os

import torch
import flash_attn.cute.interface as fi
from flash_attn.cute import cache_utils

from .dm_bwd_sm100 import DMBackwardSm100
from .fwd import _fingerprint

LOG2E = math.log2(math.e)


_CACHES = {}


def _dm_bwd_cache():
    """One compile cache per DM_BWD_OPTS string (kernel variant knobs read in DMBackwardSm100.__init__), so A/B
    variants can run in one process."""
    opts = os.environ.get("DM_BWD_OPTS", "")
    if opts not in _CACHES:
        if cache_utils.CUTE_DSL_CACHE_ENABLED:
            _CACHES[opts] = cache_utils.JITPersistentCache(
                cache_utils.get_cache_path() / "dm_sm100" / _fingerprint(opts) / "bwd")
        else:
            _CACHES[opts] = cache_utils.JITCache()
    return _CACHES[opts]


_CTX = {}


def _dm_bwd_preprocess(out, dout, dpsum, lse, lse_log2, dq_accum, *args, fake_mode=False, **kw):
    """Replaces FA4's preprocess: dpsum <- our delta (_CTX), lse_log2 <- lse * log2e, dq_accum <- 0."""
    if fake_mode:
        return
    delta = _CTX["delta"]
    N = lse.shape[-1]
    dpsum[..., :N].copy_(delta)
    dpsum[..., N:].zero_()
    lse_log2[..., :N].copy_(lse * LOG2E)
    lse_log2[..., N:].zero_()
    if dq_accum is not None:
        dq_accum.zero_()


def _patched_source():
    src = inspect.getsource(fi._flash_attn_bwd)
    subs = [
        ("def _flash_attn_bwd(", "def _dm_flash_attn_bwd("),
        ('assert q.dtype in [torch.float16, torch.bfloat16], "inputs must be float16 or bfloat16"',
         'assert q.dtype == torch.float8_e4m3fn, "dm: fp8 codes expected"'),
        ("assert q.dtype == k.dtype == v.dtype == out.dtype == dout.dtype, (", "assert True, ("),
        ("out_torch_dtype = q.dtype", "out_torch_dtype = torch.bfloat16"),
        ("dq = torch.empty_like(q)", "dq = torch.empty_like(q, dtype=torch.bfloat16)"),
        ("dk = torch.empty_like(k)", "dk = torch.empty_like(k, dtype=torch.bfloat16)"),
        ("dv = torch.empty_like(v)", "dv = torch.empty_like(v, dtype=torch.bfloat16)"),
        ("    _bwd_preprocess(\n        out, dout,", "    _dm_bwd_preprocess(\n        out, dout,"),
        ("fa_bwd_obj = FlashAttentionBackwardSm100(", "fa_bwd_obj = _DMBackwardSm100("),
        ("_flash_attn_bwd.compile_cache", "_dm_bwd_cache()"),
        ("dk_accum, dk, softmax_scale,", "dk_accum, dk, 1.0,"),
        ("            q.detach(),\n            k.detach(),\n            v.detach(),\n            dout,\n",
         "            q.detach().view(torch.uint8),\n            k.detach().view(torch.uint8),\n"
         "            v.detach().view(torch.uint8),\n            dout.view(torch.uint8),\n"),
    ]
    for a, b in subs:
        if a not in src:
            raise RuntimeError(f"bwd host patch anchor not found: {a!r}")
        src = src.replace(a, b)
    a = "        config = select_bwd_config(inputs)\n"
    if a not in src:
        raise RuntimeError("bwd host patch anchor not found: select_bwd_config")
    src = src.replace(a, a + "        config = _dm_cfg(config)\n")
    # bf16 for every host-side dtype decision (config, postprocess output); the kernel reads fp8 from the tensors
    a = "dtype = torch2cute_dtype_map[q.dtype]"
    if a in src:
        src = src.replace(a, "dtype = torch2cute_dtype_map[torch.bfloat16]")
    return src


def _dm_cfg(config):
    """bwd config overrides: no pack_gqa; DM_BWD_OPTS spt0 / spt1 force the deterministic dQ lock order."""
    import dataclasses
    over = dict(pack_gqa=False)
    opts = os.environ.get("DM_BWD_OPTS", "").split(",")
    if "spt0" in opts:
        over["spt"] = False
    if "spt1" in opts:
        over["spt"] = True
    for o in opts:
        if o.startswith("regs="):           # regs=reduce:compute:load_mma (setmaxnreg split, 2-CTA)
            r, c, l = (int(x) for x in o[5:].split(":"))
            from flash_attn.cute.config import Sm100BwdRegisters
            over["registers"] = Sm100BwdRegisters(reduce=r, compute=c, load=l, mma=l, empty=24)
    return dataclasses.replace(config, **over)


_ns = dict(vars(fi))
_ns.update(_DMBackwardSm100=DMBackwardSm100, _dm_bwd_preprocess=_dm_bwd_preprocess, _dm_bwd_cache=_dm_bwd_cache, _dm_cfg=_dm_cfg)
exec(compile(_patched_source(), fi.__file__ + ":dm_patched_bwd", "exec"), _ns)
_dm_flash_attn_bwd = _ns["_dm_flash_attn_bwd"]

FORCE = dict(pack_gqa=False)


def dm_bwd_main(q8, qs, k8, ks, v8, vs, do8, dos, lse, delta, causal: bool, sm: float, deterministic=True, trace=None):
    """Main backward pass. q8 (B,N,H,D) e4m3 (sm baked), k8/v8 (B,N,Hk,D), do8 (B,N,H,D) e4m3; scales (B,H|Hk,N/32) fp32
    pow2; lse (B,H,N) natural; delta (B,H,N) fp32 in dP units. Returns dq, dk, dv bf16 (dk/dv per KV head)."""
    aux = [qs.contiguous(), ks.contiguous(), vs.contiguous(), dos.contiguous()]
    if trace is not None:          # int64 (16,) accumulators of MMA-warp wait cycles (DM_BWD_OPTS must contain 'trace')
        aux.append(trace)
    _CTX["delta"] = delta
    dq, dk, dv = _dm_flash_attn_bwd(
        q8, k8, v8, do8, do8, lse.contiguous(), softmax_scale=sm, causal=causal,
        deterministic=deterministic, aux_tensors=aux)[:3]
    return dq, dk, dv
