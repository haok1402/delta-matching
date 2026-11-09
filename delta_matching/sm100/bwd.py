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


# ---- scratch prestaging (E1): FA4's host allocates + zero-fills scratch (dq_accum, dpsum, lse_log2, dK/dV accum,
# semaphores) right before the main kernel. With prestaging, the allocations recorded on the first call of a shape are
# re-created and zeroed on a side stream while pass 0 runs, so only the delta copy stays on the critical path.
_REC = {}          # key -> [(kind, shape, dtype)] of torch.empty / torch.zeros calls in the host (recorded once)
_SIDE = {}


def _dm_alloc(kind, *shape, dtype=None, device=None, **kw):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)):
        shape = tuple(shape[0])
    pool = _CTX.get("pool")
    if pool:
        t = pool.pop(0)
        assert tuple(t.shape) == tuple(shape) and t.dtype == dtype, (t.shape, shape, t.dtype, dtype)
        return t
    rec = _CTX.get("rec")
    if kind == "z" and dtype == torch.float32 and _fws(_CTX.get("deterministic", True)):
        kind = "e"     # dK/dV accum (GQA): the first group head stores (DM fws), no zero-fill
    if rec is not None:
        rec.append((kind, tuple(shape), dtype))
    return (torch.zeros if kind == "z" else torch.empty)(*shape, dtype=dtype, device=device, **kw)


def _dm_zeros(*a, **k):
    return _dm_alloc("z", *a, **k)


def _dm_empty(*a, **k):
    return _dm_alloc("e", *a, **k)


def _fws(deterministic):
    """dQ first-writer-stores in the kernel (DMBackwardSm100.dm_fws): dq_accum needs no zero-fill."""
    return bool(deterministic) and "nofws" not in os.environ.get("DM_BWD_OPTS", "").split(",")


def _auto_opts(q8, k8, causal=False):
    """Effective DM_BWD_OPTS for this shape. DM gloop (GQA, auto): short sequences spend most of a KV tile on the
    fp32 dK/dV accumulation across the qh query-head CTAs + postprocess; loop the group inside one CTA instead (dK/dV
    in TMEM, stored once). Long causal sequences lose (4x longer tiles -> tail), so only N <= gloopN (default 2048).
    'nogloop' disables, 'gloop' forces."""
    ol = [o for o in os.environ.get("DM_BWD_OPTS", "").split(",") if o]
    gN = int(next((o[7:] for o in ol if o.startswith("gloopN=")), "2048"))
    if q8.shape[2] > k8.shape[2] and "gloop" not in ol and "nogloop" not in ol and q8.shape[1] <= gN:
        ol = ol + ["gloop"]
    # DM pers (MHA, auto): persistent grid of 148 CTAs (74 clusters) for short sequences, where the per-CTA
    # launch/prologue/epilogue is a large share of a KV tile; long sequences keep the hardware's dynamic CTA order
    # (static striding loses 4-18% there). 'nopers' disables, explicit 'pers=N' overrides, persN= sets the cut.
    pN = int(next((o[5:] for o in ol if o.startswith("persN=")), "0"))   # off by default: no gain at 8k-64k
    # DM pcN (MHA causal, auto, pers2): dynamic persistent grid (pdyn) over all SMs for causal N <= pcN (default 8192):
    # +6-8% bwd main at causal 8k (2 GPUs, same-process A/B); -1.5% at 32k/64k and -4% non-causal, so off there.
    pcN = int(next((o[4:] for o in ol if o.startswith("pcN=")), "8192"))
    if (q8.shape[2] == k8.shape[2] and "nopers" not in ol and not any(o.startswith("pers=") for o in ol)):
        if q8.shape[1] <= pN:
            ol = ol + ["pers=148"]
        elif causal and q8.shape[1] <= pcN:
            ol = ol + [f"pers={_num_sms(q8)}", "pdyn"]
    return ",".join(ol)


def _num_sms(q8):
    n = int(os.environ.get("DM_NSM", "0"))
    if n:
        return n
    try:
        return int(torch.cuda.get_device_properties(q8.device).multi_processor_count)
    except Exception:
        return 148


def _key(q8, k8, causal, deterministic):
    return (tuple(q8.shape), tuple(k8.shape), bool(causal), bool(deterministic), _auto_opts(q8, k8, causal),
            q8.device.index)


def dm_bwd_prestage(q8, k8, lse, causal: bool, deterministic=True):
    """Call BEFORE pass 0 (current stream = the backward's stream): if this shape was seen, allocate + zero the main
    pass's scratch and compute lse_log2 on a side stream; dm_bwd_main then waits on it. Bitwise identical."""
    key = _key(q8, k8, causal, deterministic)
    rec = _REC.get(key)
    if rec is None or os.environ.get("DM_E2E", "1") == "0":
        return False
    main = torch.cuda.current_stream()
    side = _SIDE.get(q8.device.index)
    if side is None:
        side = _SIDE[q8.device.index] = torch.cuda.Stream(device=q8.device)
    side.wait_stream(main)          # lse ready; memory freed on main is reusable
    N = lse.shape[-1]
    with torch.cuda.stream(side):
        pool = [(torch.zeros if kind == "z" else torch.empty)(*shape, dtype=dt, device=q8.device)
                for kind, shape, dt in rec]
        dq_accum, dpsum, lse_log2 = pool[0], pool[1], pool[2]
        if not _fws(deterministic):
            dq_accum.zero_()
        dpsum[..., N:].zero_()
        torch.mul(lse, LOG2E, out=lse_log2[..., :N])
        lse_log2[..., N:].zero_()
        ev = torch.cuda.Event()
        ev.record(side)
    for t in pool:
        t.record_stream(main)
    _CTX["staged"] = (key, pool, ev)
    return True


def _dm_bwd_preprocess(out, dout, dpsum, lse, lse_log2, dq_accum, *args, fake_mode=False, **kw):
    """Replaces FA4's preprocess: dpsum <- our delta (_CTX), lse_log2 <- lse * log2e, dq_accum <- 0."""
    if fake_mode:
        return
    delta = _CTX["delta"]
    N = lse.shape[-1]
    if _CTX.get("prestaged"):
        torch.cuda.current_stream().wait_event(_CTX["prestaged"])
        dpsum[..., :N].copy_(delta)
        return
    dpsum[..., :N].copy_(delta)
    dpsum[..., N:].zero_()
    lse_log2[..., :N].copy_(lse * LOG2E)
    lse_log2[..., N:].zero_()
    if dq_accum is not None and not _fws(_CTX.get("deterministic", True)):
        dq_accum.zero_()


def _patched_source():
    src = inspect.getsource(fi._flash_attn_bwd)
    subs = [
        ("def _flash_attn_bwd(", "def _dm_flash_attn_bwd("),
        ('assert q.dtype in [torch.float16, torch.bfloat16], "inputs must be float16 or bfloat16"',
         'assert q.dtype == torch.float8_e4m3fn, "dm: fp8 codes expected"'),
        ("assert q.dtype == k.dtype == v.dtype == out.dtype == dout.dtype, (", "assert True, ("),
        ("out_torch_dtype = q.dtype", "out_torch_dtype = torch.bfloat16"),
        ("dq = torch.empty_like(q)", "dq = _CTX['dq'] if _CTX.get('dq') is not None else torch.empty_like(q, dtype=torch.bfloat16)"),
        ("dk = torch.empty_like(k)", "dk = torch.empty_like(k, dtype=torch.bfloat16)"),
        ("dv = torch.empty_like(v)", "dv = torch.empty_like(v, dtype=torch.bfloat16)"),
        ("    _bwd_preprocess(\n        out, dout,", "    _dm_bwd_preprocess(\n        out, dout,"),
        ("fa_bwd_obj = FlashAttentionBackwardSm100(", "fa_bwd_obj = _DMBackwardSm100("),
        ("_flash_attn_bwd.compile_cache", "_dm_bwd_cache()"),
        ("dk_accum, dk, softmax_scale,", "dk_accum, dk, 1.0,"),
        ("dKV_postprocess = qhead_per_kvhead > 1 and not use_dedicated_hd256_kernel and not bwd_pack_gqa",
         "dKV_postprocess = qhead_per_kvhead > 1 and not use_dedicated_hd256_kernel and not bwd_pack_gqa and not _dm_gloop()"),
        ("            q.detach(),\n            k.detach(),\n            v.detach(),\n            dout,\n",
         "            q.detach().view(torch.uint8),\n            k.detach().view(torch.uint8),\n"
         "            v.detach().view(torch.uint8),\n            dout.view(torch.uint8),\n"),
    ]
    for a, b in subs:
        if a not in src:
            raise RuntimeError(f"bwd host patch anchor not found: {a!r}")
        src = src.replace(a, b)
    src = src.replace("torch.zeros(", "_dm_zeros(").replace("torch.empty(", "_dm_empty(")
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
def _dm_post_convert(acc, out, *a, **k):
    """FA4 postprocess wrapper: skipped for dQ when the kernel's last writer already wrote bf16 dQ (DM lwc)."""
    if _CTX.get("dq") is not None and out is _CTX["dq"]:
        return
    return _ORIG_POST(acc, out, *a, **k)


_ORIG_POST = fi._bwd_postprocess_convert
_ns.update(_bwd_postprocess_convert=_dm_post_convert, _CTX=_CTX)
def _dm_gloop():
    """DM_BWD_OPTS gloop (GQA): dK/dV kept in TMEM across the query-head group, stored directly (no dKV postprocess)."""
    return "gloop" in os.environ.get("DM_BWD_OPTS", "").split(",")


_ns.update(_dm_gloop=_dm_gloop)
_ns.update(_dm_zeros=_dm_zeros, _dm_empty=_dm_empty, _DMBackwardSm100=DMBackwardSm100, _dm_bwd_preprocess=_dm_bwd_preprocess, _dm_bwd_cache=_dm_bwd_cache, _dm_cfg=_dm_cfg)
exec(compile(_patched_source(), fi.__file__ + ":dm_patched_bwd", "exec"), _ns)
_dm_flash_attn_bwd = _ns["_dm_flash_attn_bwd"]

FORCE = dict(pack_gqa=False)


def dm_bwd_main(q8, qs, k8, ks, v8, vs, do8, dos, lse, delta, causal: bool, sm: float, deterministic=True, trace=None):
    """Main backward pass. q8 (B,N,H,D) e4m3 (sm baked), k8/v8 (B,N,Hk,D), do8 (B,N,H,D) e4m3; scales (B,H|Hk,N/32) fp32
    pow2; lse (B,H,N) natural; delta (B,H,N) fp32 in dP units. Returns dq, dk, dv bf16 (dk/dv per KV head)."""
    aux = [qs.contiguous(), ks.contiguous(), vs.contiguous(), dos.contiguous()]
    if "lwc" in os.environ.get("DM_BWD_OPTS", "").split(","):
        assert trace is None and deterministic and abs(sm - q8.shape[-1] ** -0.5) < 1e-12, "lwc: det, sm = D^-0.5, no trace"
        _CTX["dq"] = torch.empty(q8.shape, dtype=torch.bfloat16, device=q8.device)
        aux.append(_CTX["dq"])
    if "pdyn" in _auto_opts(q8, k8, causal).split(","):
        # DM pdyn: claim counter for the dynamic persistent scheduler (last aux tensor), zeroed per launch
        _CTX["pdyn_ctr"] = torch.zeros(1, dtype=torch.int32, device=q8.device)
    if trace is not None:          # int64 (16,) accumulators of MMA-warp wait cycles (DM_BWD_OPTS must contain 'trace')
        aux.append(trace)
    if "pdyn" in _auto_opts(q8, k8, causal).split(","):
        aux.append(_CTX.pop("pdyn_ctr"))
    _CTX["delta"] = delta
    _CTX["deterministic"] = deterministic
    opts0 = os.environ.get("DM_BWD_OPTS", "")
    os.environ["DM_BWD_OPTS"] = _auto_opts(q8, k8, causal)     # kernel knobs + compile cache follow the effective opts
    key = _key(q8, k8, causal, deterministic)
    st = _CTX.pop("staged", None)
    if st is not None and st[0] == key and trace is None:
        _CTX["pool"], _CTX["prestaged"] = list(st[1]), st[2]
    elif key not in _REC and not isinstance(q8, torch._subclasses.fake_tensor.FakeTensor):
        _CTX["rec"] = []
    try:
        dq, dk, dv = _dm_flash_attn_bwd(
            q8, k8, v8, do8, do8, lse.contiguous(), softmax_scale=sm, causal=causal,
            deterministic=deterministic, aux_tensors=aux)[:3]
        if _CTX.get("rec") is not None:
            r = _CTX["rec"]
            if len(r) >= 3 and all(x[0] == "e" and x[2] == torch.float32 for x in r[:3]):
                _REC[key] = r
        assert not _CTX.get("pool"), "prestaged scratch not fully consumed"
    finally:
        os.environ["DM_BWD_OPTS"] = opts0
        _CTX.pop("dq", None)
        _CTX.pop("rec", None)
        _CTX.pop("pool", None)
        _CTX.pop("prestaged", None)
    return dq, dk, dv
