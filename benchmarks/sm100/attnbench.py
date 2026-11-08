#!/usr/bin/env python
"""sm100 attention benchmark: FA4 (bf16 / fp8-fwd), cuDNN SDPA, and any candidate with dm_attn_func's signature.

All arms are timed by the SAME code path on the SAME inputs:
  * inputs  q (B, N, H, D), k / v (B, N, Hk, D), bf16, contiguous, requires_grad; dO ~ randn.
  * fwd     : flush L2 -> ev0 -> fn(q,k,v) -> ev1
  * bwd     : o = fn(q,k,v) (untimed) -> flush L2 -> ev0 -> torch.autograd.grad(o, (q,k,v), dO) -> ev1
  * fwd_bwd : flush L2 -> ev0 -> o = fn(q,k,v); grad(o, (q,k,v), dO) -> ev1
  Per shape: every (arm, pass) is warmed up, then timed in --rounds interleaved rounds (rotating order), each round
  1/R of the budget (total >= --min-iters calls and >= --min-time s per (arm, pass)), so drift from clocks or the
  neighbors' power hits all arms alike (gpupool README: ~4% neighbor/GPU coupling).  Speedups = ratios of medians
  taken in the same process on the same GPU; per-round medians are kept (round_ms) to show the spread of the ratio.  Reported: median ms, p10/p90, and TFLOPS from the median.
  Two independent clocks per pass:
    ev_ms   : CUDA events on the current stream around each call, the GPU pre-loaded with a spin so the host is ahead
              (host launch latency is excluded unless the call syncs; host_ms is reported separately)
    cupti_ms: sum of device kernel durations inside the call measured by torch.profiler/CUPTI (all streams; excludes
              the L2 flush).  ev >> cupti => launch-bound or idle gaps; cupti >> ev => work escaped the timed stream.
  FLOPs: FlashAttention convention: fwd = 4*B*H*N^2*D (/2 causal); bwd = 2.5x fwd; fwd_bwd = 3.5x fwd.  TFLOPS of a
  candidate is "FA-equivalent" (its extra work, e.g. a recompute pass, is not credited).
  Sanity gate: per shape, each arm's O, dQ, dK, dV are compared (rel. L2 error) to the fp32 math reference on a slice
  (or to FA4 bf16 when the shape is too big); an arm whose error exceeds --max-rel-err is marked SUSPECT.  This is a
  cheap guard against fast-but-wrong numbers, NOT the validator (see sm100/STATUS-dm-validator.md).

Candidates: --cand NAME=/abs/path/file.py:func  or  NAME=pkg.module:func  (optional --pythonpath DIR, repeatable);
  ':func' defaults to ':attn' (the validator's candidate contract, sm100/validate).  The function is called as
  fn(q, k, v, causal=c[, softmax_scale=None][, mode=m]) (keywords only if it accepts them);
  per shape: causal -> mode "matched", non-causal -> mode "stale" (override with --cand-mode).
Output: one JSON line per (arm, shape, pass) appended to --out; render with report.py.
"""
from __future__ import annotations

import argparse, contextlib, datetime, importlib, importlib.util, inspect, json, math, os, socket, statistics, sys, time

os.environ.setdefault("KINETO_LOG_LEVEL", "5")     # silence the profiler's per-call USDT lines
import torch

HD = 128

# ----------------------------------------------------------------------------------------------------------- shapes
# FlashAttention's own benchmark convention (benchmarks/benchmark_attn.py): total 32k tokens per batch, 16 heads at
# hd128 (hidden 2048), seqlen 1k..32k.  Plus GQA at Llama-3-8B (32q/8kv) and Llama-3-70B (64q/8kv, per-TP-shard
# variants are the same per-head work) ratios.  B*H*ceil(N/128) >= 4096 CTA tiles everywhere: all 148 SMs are filled.
SEQLENS = [1024, 2048, 4096, 8192, 16384, 32768]
SUITES = {
    # name: list of (H, Hk, total_tokens)
    "main": [(16, 16, 32768), (32, 8, 32768)],
    "full": [(16, 16, 32768), (32, 8, 32768), (64, 8, 32768), (32, 32, 32768)],
    "quick": [(16, 16, 32768)],
}


def shapes_for(suite, seqlens, causals):
    out = []
    for (H, Hk, tot) in SUITES[suite]:
        for N in seqlens:
            for c in causals:
                out.append(dict(B=max(1, tot // N), N=N, H=H, Hk=Hk, D=HD, causal=c))
    return out


def fa_flops(s, kind):
    f = 4 * s["B"] * s["H"] * s["N"] ** 2 * s["D"] / (2 if s["causal"] else 1)
    return f * {"fwd": 1.0, "bwd": 2.5, "fwd_bwd": 3.5}[kind]


# ------------------------------------------------------------------------------------------------------------- arms
def _first(r):
    return r[0] if isinstance(r, (tuple, list)) else r


def make_fa4(deterministic=False):
    from flash_attn.cute import flash_attn_func as fa4

    def fn(q, k, v, causal):
        return _first(fa4(q, k, v, causal=causal, deterministic=deterministic))
    return fn


def make_sdpa(backend_name):
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    be = getattr(SDPBackend, backend_name)

    def fn(q, k, v, causal):
        gqa = q.shape[2] != k.shape[2]
        with sdpa_kernel([be]):
            o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                               is_causal=causal, enable_gqa=gqa)
        return o.transpose(1, 2)
    return fn


class FP8Fwd:
    """FA4 fp8 (e4m3) forward, per-(batch, head) descales.  Kernel-only: inputs are quantized once, outside timing
    (so this is a lower bound for an fp8 forward that starts from bf16 activations).  Forward only."""

    def __init__(self):
        from flash_attn.cute.interface import _flash_attn_fwd
        self.f = _flash_attn_fwd
        self.cache = None

    def quant(self, x, hk):                   # x (B, N, H, D) bf16 -> e4m3 codes, descale (B, hk): FA4 takes one descale
        B, N, H, D = x.shape                  # per (batch, kv head), so q's scale is shared by its GQA group
        amax = x.detach().float().abs().amax(dim=(1, 3)).view(B, hk, H // hk).amax(-1).clamp_min(1e-12)
        scale = 448.0 / amax
        codes = (x.detach().float() * scale.repeat_interleave(H // hk, dim=1)[:, None, :, None]).to(torch.float8_e4m3fn)
        return codes, (1.0 / scale).contiguous()

    def __call__(self, q, k, v, causal):
        key = (q.data_ptr(), k.data_ptr(), v.data_ptr())
        if self.cache is None or self.cache[0] != key:
            self.cache = (key, [self.quant(t, k.shape[2]) for t in (q, k, v)])
        (q8, qd), (k8, kd), (v8, vd) = self.cache[1]
        return _first(self.f(q8, k8, v8, softmax_scale=HD ** -0.5, causal=causal, q_descale=qd, k_descale=kd,
                             v_descale=vd))


def load_candidate(spec, pythonpaths):
    for p in pythonpaths:
        if p not in sys.path:
            sys.path.insert(0, p)
    target, func = spec.rsplit(":", 1) if ":" in spec else (spec, "attn")
    if target.endswith(".py"):
        mspec = importlib.util.spec_from_file_location(f"cand_{abs(hash(target))}", target)
        mod = importlib.util.module_from_spec(mspec); sys.modules[mspec.name] = mod; mspec.loader.exec_module(mod)
    else:
        mod = importlib.import_module(target)
    return getattr(mod, func)


def wrap_candidate(f, mode_override):
    """Accepts dm_attn_func(q, k, v, softmax_scale=None, causal=False, mode=...) and the validator's candidate contract
    attn(q, k, v, causal, mode).  softmax_scale / mode are passed only if the function takes them."""
    try:
        params = inspect.signature(f).parameters
        has_mode, has_scale = "mode" in params, "softmax_scale" in params
    except (TypeError, ValueError):
        has_mode, has_scale = True, False

    def fn(q, k, v, causal):
        kw = dict(causal=causal)
        if has_scale:
            kw["softmax_scale"] = None
        if has_mode:
            kw["mode"] = mode_override or ("matched" if causal else "stale")
        return _first(f(q, k, v, **kw))
    fn.supports = getattr(sys.modules.get(getattr(f, "__module__", ""), None), "SUPPORTS", None) or {}
    return fn


# ----------------------------------------------------------------------------------------------------------- timing
SPIN_HZ = 2.1e9   # cycles/s for torch.cuda._sleep (upper bound of the SM clock: lower clocks only lengthen the spin)


class Flusher:
    def __init__(self, dev):
        l2 = getattr(torch.cuda.get_device_properties(dev), "L2_cache_size", 128 << 20)
        self.buf = torch.empty(max(4 * l2, 512 << 20) // 4, dtype=torch.int32, device=dev)

    def __call__(self):
        self.buf.zero_()


def time_iters(make_call, flusher, min_iters, min_time, max_iters, host_ms):
    """make_call() -> (pre, run): pre() runs untimed before the flush (e.g. the forward for a bwd pass), run() timed.
    Before the start event the GPU gets a spin (torch.cuda._sleep) long enough for the CPU to enqueue the whole call, so
    the events measure GPU execution with the host ahead (as in a training step), not Python/launch latency.  Host time
    of run() is recorded separately (host_ms list); a call whose host time exceeds its GPU time is launch-bound.
    A host<->device sync inside run() defeats the spin and shows up as ev_ms ~ host_ms (reported, not hidden).
    Returns the list of per-call CUDA-event times (ms)."""
    ts, spent = [], 0.0
    while (len(ts) < min_iters or spent < min_time) and len(ts) < max_iters:
        pre, run = make_call()
        pre()
        flusher()
        lead = max(1.0, 1.5 * (statistics.median(host_ms[-5:]) if host_ms else 2.0))   # ms of GPU spin
        torch.cuda._sleep(int(lead * 1e-3 * SPIN_HZ))
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        h0 = time.perf_counter(); run(); host_ms.append((time.perf_counter() - h0) * 1e3)
        e1.record()
        e1.synchronize()
        t = e0.elapsed_time(e1); ts.append(t); spent += t / 1e3
    return ts


def summarize(ts):
    ts_sorted = sorted(ts)
    q = lambda p: ts_sorted[min(len(ts_sorted) - 1, int(p * len(ts_sorted)))]
    return dict(ev_ms=statistics.median(ts), p10_ms=q(0.10), p90_ms=q(0.90), min_ms=ts_sorted[0], iters=len(ts))


def cupti_ms(make_call, flusher, n):
    """Median over n calls of the summed device-kernel time of run() (all streams; pre() and the flush excluded)."""
    from torch.profiler import profile, ProfilerActivity
    kts = []
    for _ in range(n):
        pre, run = make_call(); pre(); flusher(); torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            run(); torch.cuda.synchronize()
        kts.append(sum(e.time_range.elapsed_us() for e in prof.events()
                       if e.device_type == torch.autograd.DeviceType.CUDA) / 1e3)
    return statistics.median(kts)


# ------------------------------------------------------------------------------------------------------- reference
def rel_err(a, b):
    a = a.float(); b = b.float()
    return (a - b).norm().item() / max(b.norm().item(), 1e-30)


def ref_attn_fp32(q, k, v, causal, do):
    """fp32 math reference (B, N, H, D) on the first batch element and first 2 kv groups; returns o, dq, dk, dv."""
    Hk = k.shape[2]; g = q.shape[2] // Hk
    nk = min(Hk, 2)
    qs = q[:1, :, : nk * g].detach().float().requires_grad_()
    ks = k[:1, :, :nk].detach().float().requires_grad_()
    vs = v[:1, :, :nk].detach().float().requires_grad_()
    kk = ks.repeat_interleave(g, dim=2); vv = vs.repeat_interleave(g, dim=2)
    s = torch.einsum("bnhd,bmhd->bhnm", qs, kk) * HD ** -0.5
    if causal:
        N = q.shape[1]
        s = s.masked_fill(torch.ones(N, N, dtype=torch.bool, device=q.device).triu(1), float("-inf"))
    o = torch.einsum("bhnm,bmhd->bnhd", s.softmax(-1), vv)
    dq, dk, dv = torch.autograd.grad(o, (qs, ks, vs), do[:1, :, : nk * g].float())
    return dict(o=o, dq=dq, dk=dk, dv=dv), nk, g


def sanity(fn, s, q, k, v, do, ref, nk, g, fwd_only):
    o = fn(q, k, v, s["causal"])
    errs = dict(o=rel_err(o[:1, :, : nk * g], ref["o"]))
    if not fwd_only:
        dq, dk, dv = torch.autograd.grad(o, (q, k, v), do)
        errs.update(dq=rel_err(dq[:1, :, : nk * g], ref["dq"]), dk=rel_err(dk[:1, :, :nk], ref["dk"]),
                    dv=rel_err(dv[:1, :, :nk], ref["dv"]))
    # anti-caching: perturb q, the output must change
    with torch.no_grad():
        q2 = (q + 0.5 * torch.randn_like(q)).requires_grad_()
    o2 = fn(q2, k, v, s["causal"])
    errs["changes_with_input"] = bool(rel_err(o2, o) > 1e-3)
    return errs


# ------------------------------------------------------------------------------------------------------------ clocks
class Clocks:
    def __init__(self, dev):
        try:
            import pynvml
            pynvml.nvmlInit()
            idx = torch.cuda.get_device_properties(dev).uuid
            self.h = None
            for i in range(pynvml.nvmlDeviceGetCount()):
                hh = pynvml.nvmlDeviceGetHandleByIndex(i)
                u = pynvml.nvmlDeviceGetUUID(hh)
                u = u.decode() if isinstance(u, bytes) else u
                if str(idx) in u or u.endswith(str(idx)):
                    self.h = hh
            if self.h is None:
                self.h = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
            self.nv = pynvml
        except Exception:
            self.nv = None

    def read(self):
        if self.nv is None:
            return {}
        nv, h = self.nv, self.h
        try:
            return dict(sm_mhz=nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM),
                        mem_mhz=nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM),
                        power_w=nv.nvmlDeviceGetPowerUsage(h) / 1e3,
                        temp_c=nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU),
                        throttle=int(nv.nvmlDeviceGetCurrentClocksEventReasons(h)))
        except Exception as ex:
            return {"nvml_err": str(ex)}


def make_call(fn, fwd_only, p, q, k, v, do, c):
    if p == "fwd":
        def mk():
            def run():
                with torch.no_grad() if fwd_only else contextlib.nullcontext():
                    fn(q, k, v, c)
            return (lambda: None), run
    elif p == "bwd":
        def mk():
            st = {}
            def pre(): st["o"] = fn(q, k, v, c)
            def run(): torch.autograd.grad(st["o"], (q, k, v), do)
            return pre, run
    else:
        def mk():
            def run():
                o = fn(q, k, v, c); torch.autograd.grad(o, (q, k, v), do)
            return (lambda: None), run
    return mk


# -------------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="fa4,cudnn,fa4_fp8", help="comma list of fa4,fa4_det,cudnn,flash_sdpa,fa4_fp8")
    ap.add_argument("--cand", action="append", default=[], help="NAME=path.py:func or NAME=module:func")
    ap.add_argument("--cand-mode", default=None, help="force mode for candidates (default matched if causal else stale)")
    ap.add_argument("--pythonpath", action="append", default=[])
    ap.add_argument("--suite", default="main", choices=list(SUITES))
    ap.add_argument("--seqlens", default=",".join(map(str, SEQLENS)))
    ap.add_argument("--causal", default="both", choices=["true", "false", "both"])
    ap.add_argument("--passes", default="fwd,bwd,fwd_bwd")
    ap.add_argument("--shard", default="0/1", help="i/n: run every n-th shape starting at i")
    ap.add_argument("--min-iters", type=int, default=30)
    ap.add_argument("--max-iters", type=int, default=1000)
    ap.add_argument("--min-time", type=float, default=1.0, help="seconds of timed work per pass")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=6, help="interleaved timing rounds across arms (same GPU/process)")
    ap.add_argument("--cupti-iters", type=int, default=5, help="0 disables the CUPTI kernel-sum clock")
    ap.add_argument("--max-rel-err", type=float, default=0.08)
    ap.add_argument("--tag", default="", help="free text, e.g. 'filler-on' / 'exclusive'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true", help="skip (arm, shape, pass) already timed in --out (relaunch "
                    "after a time limit: same command + --resume continues where it stopped)")
    a = ap.parse_args()

    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    seqlens = [int(x) for x in a.seqlens.split(",")]
    causals = {"true": [True], "false": [False], "both": [True, False]}[a.causal]
    shards = shapes_for(a.suite, seqlens, causals)
    si, sn = map(int, a.shard.split("/"))
    shards = shards[si::sn]
    passes = a.passes.split(",")

    flusher = Flusher(dev)
    clocks = Clocks(dev)
    meta = dict(host=socket.gethostname(), gpu=torch.cuda.get_device_name(), cvd=os.environ.get("CUDA_VISIBLE_DEVICES"),
                torch=torch.__version__, cuda=torch.version.cuda, tag=a.tag,
                time=datetime.datetime.now().isoformat(timespec="seconds"))
    meta["run_id"] = f"{meta['host']}-{os.getpid()}-{meta['time']}"
    try:
        import flash_attn.cute as _fc
        meta["fa4"] = getattr(_fc, "__version__", "?")
    except Exception:
        pass
    arms = {}
    for name in [x for x in a.arms.split(",") if x]:
        try:
            if name == "fa4":
                arms["fa4_bf16"] = (make_fa4(), False)
            elif name == "fa4_det":                      # FA4 with deterministic bwd (like-for-like vs a deterministic DM bwd)
                arms["fa4_bf16_det"] = (make_fa4(True), False)
            elif name == "cudnn":
                arms["sdpa_cudnn_bf16"] = (make_sdpa("CUDNN_ATTENTION"), False)
            elif name == "flash_sdpa":
                arms["sdpa_flash_bf16"] = (make_sdpa("FLASH_ATTENTION"), False)
            elif name == "fa4_fp8":
                arms["fa4_fp8_fwd"] = (FP8Fwd(), True)
            else:
                raise ValueError(name)
        except Exception as ex:
            print(f"[arm {name}] unavailable: {type(ex).__name__}: {ex}", file=sys.stderr)
    for c in a.cand:
        name, spec = c.split("=", 1)
        f = load_candidate(spec, a.pythonpath)
        mod = sys.modules.get(getattr(f, "__module__", ""), None)
        arms[name] = (wrap_candidate(f, a.cand_mode), bool(getattr(mod, "FWD_ONLY", False)))   # FWD_ONLY: fwd pass only
        if getattr(mod, "CAND_INFO", None):
            meta.setdefault("cand_info", {})[name] = mod.CAND_INFO

    print(json.dumps(meta))
    done = set()
    if a.resume and os.path.exists(a.out):
        for line in open(a.out):
            try:
                r = json.loads(line)
            except ValueError:
                continue                       # a line cut by the kill
            if "pass_" in r and "error" not in r:
                done.add((r["arm"], r["B"], r["N"], r["H"], r["Hk"], r["causal"], r["pass_"]))
    fout = open(a.out, "a")
    try:                                         # gpupool late stop: filler leaves before timing, returns after
        from gpupool_hook import gpupool_ready, gpupool_done
    except ImportError:
        gpupool_ready = gpupool_done = lambda: None
    gpupool_ready()
    for s in shards:
        g = torch.Generator(device=dev).manual_seed(1234)
        B, N, H, Hk, D = s["B"], s["N"], s["H"], s["Hk"], s["D"]
        q = torch.randn(B, N, H, D, device=dev, dtype=torch.bfloat16, generator=g).requires_grad_()
        k = torch.randn(B, N, Hk, D, device=dev, dtype=torch.bfloat16, generator=g).requires_grad_()
        v = torch.randn(B, N, Hk, D, device=dev, dtype=torch.bfloat16, generator=g).requires_grad_()
        do = torch.randn(B, N, H, D, device=dev, dtype=torch.bfloat16, generator=g)
        ref = None
        if N <= 8192:
            ref, nk, gg = ref_attn_fp32(q, k, v, s["causal"], do)
        items = []                                 # (arm, pass, make_call, rec0)
        for arm, (fn, fwd_only) in arms.items():
            sup = getattr(fn, "supports", {})
            # validator schema (sm100/wt/dm-validator/validator/README.md): {"seqlen_multiple": m, "noncausal": bool}
            mult = sup.get("seqlen_multiple", sup.get("min_seqlen_multiple", 1))
            noncausal_ok = sup.get("noncausal", False in sup.get("causal", [True, False]))
            if (not s["causal"] and not noncausal_ok) or N % mult:
                continue                           # outside what the candidate declares (and was validated for)
            rec0 = dict(meta, arm=arm, **s)
            todo = [p for p in passes if not (fwd_only and p != "fwd")
                    and (arm, B, N, H, Hk, s["causal"], p) not in done]
            if not todo:
                continue
            if ref is not None:
                try:
                    errs = sanity(fn, s, q, k, v, do, ref, nk, gg, fwd_only)
                except Exception as ex:
                    print(f"[{arm} {s}] failed: {type(ex).__name__}: {str(ex)[:300]}", file=sys.stderr)
                    fout.write(json.dumps(dict(rec0, error=f"{type(ex).__name__}: {str(ex)[:300]}")) + "\n"); fout.flush()
                    continue
                worst = max(v_ for k_, v_ in errs.items() if k_ != "changes_with_input")
                errs["suspect"] = bool(worst > a.max_rel_err or not errs["changes_with_input"])
                rec0["sanity"] = errs
            for p in todo:
                items.append((arm, p, make_call(fn, fwd_only, p, q, k, v, do, s["causal"]), rec0))
        # warm up every item (JIT compile, autotune), then time in R interleaved rounds: each round runs every
        # (arm, pass) for 1/R of the budget, the order rotating per round, so slow drifts (clocks, neighbors' power)
        # hit all arms alike and the arm-to-arm ratio within this process/GPU is robust.
        ok = []
        for it in items:
            arm, p, mk, rec0 = it
            try:
                for _ in range(a.warmup):
                    pre, run = mk(); pre(); run()
                torch.cuda.synchronize()
                ok.append(it)
            except Exception as ex:
                print(f"[{arm} {p} {s}] failed: {type(ex).__name__}: {str(ex)[:300]}", file=sys.stderr)
                fout.write(json.dumps(dict(rec0, pass_=p, error=f"{type(ex).__name__}: {str(ex)[:300]}")) + "\n")
                fout.flush()
        R = max(1, a.rounds)
        ts = {i: [] for i in range(len(ok))}
        rmed = {i: [] for i in range(len(ok))}
        hms = {i: [] for i in range(len(ok))}
        c0 = clocks.read()
        for r in range(R):
            order = list(range(len(ok)))
            order = order[r % max(1, len(order)):] + order[:r % max(1, len(order))]
            for i in order:
                t = time_iters(ok[i][2], flusher, max(1, a.min_iters // R), a.min_time / R, max(1, a.max_iters // R),
                               hms[i])
                ts[i] += t; rmed[i].append(statistics.median(t))
        c1 = clocks.read()
        for i, (arm, p, mk, rec0) in enumerate(ok):
            res = summarize(ts[i]); res["round_ms"] = rmed[i]
            res["host_ms"] = statistics.median(hms[i])
            res["launch_bound"] = bool(res["host_ms"] > 0.9 * res["ev_ms"])
            if a.cupti_iters:
                res["cupti_ms"] = cupti_ms(mk, flusher, a.cupti_iters)
            res["tflops"] = fa_flops(s, p) / (res["ev_ms"] * 1e-3) / 1e12
            rec = dict(rec0, pass_=p, **res, rounds=R, clk_before=c0, clk_after=c1)
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            print(f"{arm:18s} {p:7s} B{B:<3d} N{N:<6d} H{H}/{Hk} c={int(s['causal'])}  {res['ev_ms']:8.3f} ms "
                  f"[{res['p10_ms']:.3f},{res['p90_ms']:.3f}] cupti {res.get('cupti_ms', float('nan')):8.3f}  "
                  f"host {res['host_ms']:7.3f}  {res['tflops']:7.1f} TF  sm {c1.get('sm_mhz', '?')}MHz", flush=True)
        del q, k, v, do, ref
        torch.cuda.empty_cache()
    gpupool_done()


if __name__ == "__main__":
    main()
