"""The paper's own timing protocol (Appendix C.2) for the 1.67B GDN/GQA hybrid's attention shape (Hao, round 16).

    python benchmarks/paper_c2.py run --arms dm_final=.local/frozen/r14_h3,dm_base=.local/frozen/base,fa3_bf16,cudnn_bf16 \
        --out results/paper_c2/raw.json
    python benchmarks/paper_c2.py report --raw results/paper_c2/raw.json --md results/paper_c2.md

One H100; B2 N8192 Hq16 Hk4 D128 causal (and Hq32 Hk4, Qwen3-30B-A3B attention; no paper number for it) (matched mode for delta-matching).  Five interleaved rounds; in each round every
arm runs in its own fresh process (two delta_matching builds cannot share one process), after 3 s of matmul clock
warm-up, as the median of 20 CUDA-event-timed calls after 5 warm-up calls, for the forward and for forward+backward;
backward = (fwd+bwd) - fwd.  TFLOP/s from nominal work: forward 2*B*Hq*N^2*d, backward 5*B*Hq*N^2*d (fwd+bwd the sum);
mean +- standard error over the rounds.
delta-matching arms: prequantized inputs (the producers / quantizers of q, k, v and dO run once, outside the timing), no
autograd wrapper: forward = the fwd_fp8 op; backward = the matched-delta correction pass (pass 0) + the bwd_fp8_t op
(pass 1 + pass 2) on the saved codes / twins.  The GQA dK / dV head fold of the autograd wrapper is excluded with the
wrapper.  FA3 BF16: flash_attn_interface.flash_attn_func (native GQA); cuDNN BF16: torch SDPA restricted to the cuDNN
backend (enable_gqa); cuDNN / TE FP8: needs transformer_engine (reported as not run when absent).
"""
import argparse, json, math, os, statistics, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SHAPE = dict(B=2, N=8192, H=16, Hk=4, D=128)    # set per run from --H: 16 (the paper's 1.67B) or 32 (Qwen3-30B-A3B)
HEADS = (16, 32)
ROUNDS, WARM, ITERS, CLOCK_WARM_S = 5, 5, 20, 3.0
PAPER = {"fwd": 661.0, "bwd": 421.5, "fwdbwd": 470.2}   # paper Table (matched, this shape): TFLOP/s


def work(B, N, H, D, **_):
    f = 2.0 * B * H * N * N * D
    return {"fwd": f, "bwd": 2.5 * f, "fwdbwd": 3.5 * f}


def _median_events(fn, warm=WARM, iters=ITERS):
    import torch
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); b.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts), ts


def _clock_warm(seconds=CLOCK_WARM_S):
    import torch
    x = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    t0 = time.time()
    while time.time() - t0 < seconds:
        for _ in range(10):
            x @ x
        torch.cuda.synchronize()


def _inputs():
    import torch
    s = SHAPE
    g = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(s["B"], s["N"], s["H"], s["D"], device="cuda", dtype=torch.bfloat16, generator=g)
    k = torch.randn(s["B"], s["N"], s["Hk"], s["D"], device="cuda", dtype=torch.bfloat16, generator=g)
    v = torch.randn(s["B"], s["N"], s["Hk"], s["D"], device="cuda", dtype=torch.bfloat16, generator=g)
    do = torch.randn(s["B"], s["N"], s["H"], s["D"], device="cuda", dtype=torch.bfloat16, generator=g)
    return q, k, v, do


def arm_dm(root):
    """delta-matching kernels on prequantized inputs (quantizers outside the timing)."""
    import torch
    sys.path.insert(0, os.path.abspath(root))
    import delta_matching as dm
    from delta_matching import attn as A
    from delta_matching.core import blocks, _matched_delta_inkernel
    from delta_matching.producers import quant_qkv_m17, doq_m17
    pkg = os.path.realpath(os.path.dirname(dm.__file__))
    if not pkg.startswith(os.path.realpath(root)):
        raise SystemExit(f"imported {pkg}, not the arm's root {root}")
    s = SHAPE; B, N, H, Hk, D = s["B"], s["N"], s["H"], s["Hk"], s["D"]
    q, k, v, do = _inputs()
    qb, kb, vb = (x.permute(0, 2, 1, 3) for x in (q, k, v))       # the API's (B, H, N, D) view
    ops = A._ops_gt(); sm = D ** -0.5
    Q8, Q8s, K8, K8s, V8, V8s, kTt, Qt = quant_qkv_m17(qb, kb, vb, want_qt=True, blocks=blocks(D))

    def bshd(x, h=H):
        return x.view(B, h, N, D).permute(0, 2, 1, 3)

    def fwd():
        return ops.fwd_fp8(bshd(Q8), bshd(K8, Hk), bshd(V8, Hk), Q8s.view(B, H, -1), K8s.view(B, Hk, -1),
                           V8s.view(B, Hk, -1), sm, True)
    O, Lse = fwd()
    dO8, dO8s, _, dOt = doq_m17(do, None, H, blocks(D)[0], want_delta=False, want_twin=True)
    if A._bwd_heads_per_call(B, H, Hk, H // Hk, N) < H:
        raise SystemExit("the C.2 shape must run unsplit")

    def fwdbwd():
        O_, Lse_ = fwd()
        Dlt = _matched_delta_inkernel(Q8, Q8s, K8, K8s, V8, V8s, dO8, dO8s, kTt, Lse_.view(B * H, N), H, N, D,
                                      causal=True, Hk=Hk)
        return ops.bwd_fp8_t(bshd(dO8), bshd(Q8), bshd(K8, Hk), bshd(V8, Hk), Lse_, Dlt.view(B, H, N),
                             kTt.view(B, Hk, D, N), Qt.view(B, H, D, N), dOt.view(B, H, D, N), Q8s.view(B, H, -1),
                             K8s.view(B, Hk, -1), V8s.view(B, Hk, -1), dO8s.view(B, H, -1), sm, True)
    import hashlib
    lib = dm.core.loaded_library()
    ident = {"package": pkg, "library_sha256": hashlib.sha256(open(lib, "rb").read()).hexdigest()}
    return fwd, fwdbwd, ident


def arm_fa3():
    import torch
    from flash_attn_interface import flash_attn_func as fa3
    q, k, v, do = _inputs()
    q, k, v = (x.requires_grad_() for x in (q, k, v))

    def first(r):
        return r[0] if isinstance(r, tuple) else r

    def fwd():   # training forward (paper C.2): gradients enabled, softmax statistics saved for the backward
        with torch.enable_grad():
            o = first(fa3(q, k, v, causal=True))
        assert o.requires_grad and o.grad_fn is not None
        return o

    def fwdbwd():
        q.grad = k.grad = v.grad = None
        first(fa3(q, k, v, causal=True)).backward(do)
    import flash_attn_interface
    return fwd, fwdbwd, {"module": os.path.realpath(flash_attn_interface.__file__)}


def arm_cudnn():
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    import torch.nn.functional as F
    q, k, v, do = _inputs()
    qt, kt, vt = (x.permute(0, 2, 1, 3).detach().requires_grad_() for x in (q, k, v))
    dot = do.permute(0, 2, 1, 3)

    def call():
        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
            return F.scaled_dot_product_attention(qt, kt, vt, is_causal=True, enable_gqa=True)

    def fwd():   # training forward (paper C.2): gradients enabled, softmax statistics saved for the backward
        with torch.enable_grad():
            o = call()
        assert o.requires_grad and o.grad_fn is not None
        return o

    def fwdbwd():
        qt.grad = kt.grad = vt.grad = None
        call().backward(dot)
    return fwd, fwdbwd, {"torch": torch.__version__, "cudnn": torch.backends.cudnn.version()}


def diag(root):
    """Hao's forward reconciliation (round 17): the DM forward at the C.2 shape timed several ways in one process --
    per-call events (the protocol), one event pair around 20 back-to-back calls, the profiler's kernel time -- with SM
    clocks sampled, and the forward inside fwd+bwd (fwd+bwd minus the backward ops alone)."""
    import torch, subprocess
    fwd, fwdbwd, ident = arm_dm(root)
    _clock_warm()
    per_call, _ = _median_events(fwd)
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(20):
        fwd()
    b.record(); b.synchronize()
    batched = a.elapsed_time(b) / 20
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            fwd()
        torch.cuda.synchronize()
    ks = {}
    for e in prof.events():
        if e.device_type.name == "CUDA":
            ks.setdefault(e.name, []).append(e.device_time_total / 1e3 if hasattr(e, "device_time_total") else e.cuda_time_total / 1e3)
    kern = {n[:90]: statistics.median(v) for n, v in ks.items()}
    clk = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm,power.draw", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    _clock_warm()
    fb, _ = _median_events(fwdbwd)
    return {"per_call_ms": per_call, "batched_ms": batched, "kernels_ms": kern, "fwdbwd_ms": fb, "clock": clk,
            "identity": ident, "shape": dict(SHAPE)}


def arm_te_fp8():
    import transformer_engine  # noqa: F401  (absent on this cluster: the arm reports "not run")
    raise SystemExit("TE FP8 arm not implemented without transformer_engine")


def one(arm, root, H=16):
    import torch
    SHAPE["H"] = H
    torch.backends.cuda.matmul.allow_tf32 = False
    if arm.startswith("dm_"):
        fwd, fwdbwd, ident = arm_dm(root)
    elif arm == "fa3_bf16":
        fwd, fwdbwd, ident = arm_fa3()
    elif arm == "cudnn_bf16":
        fwd, fwdbwd, ident = arm_cudnn()
    elif arm == "te_fp8":
        fwd, fwdbwd, ident = arm_te_fp8()
    else:
        raise SystemExit(f"unknown arm {arm}")
    _clock_warm()
    f, fts = _median_events(fwd)
    _clock_warm()
    fb, fbts = _median_events(fwdbwd)
    return {"arm": arm, "fwd_ms": f, "fwdbwd_ms": fb, "bwd_ms": fb - f, "fwd_events": fts, "fwdbwd_events": fbts,
            "identity": ident, "device": torch.cuda.get_device_name(0)}


def run(a):
    arms = []
    for x in a.arms.split(","):
        name, _, root = x.partition("=")
        arms.append((name, root or None))
    heads = [int(x) for x in a.heads.split(",")]
    out = {"protocol": {"shape": dict(SHAPE, H=None), "heads": heads, "rounds": a.rounds, "warmup_calls": WARM, "events": ITERS,
                        "clock_warmup_s": CLOCK_WARM_S}, "job": os.environ.get("SLURM_JOB_ID"),
           "host": os.uname().nodename, "rounds": []}
    for r in range(a.rounds):
        row = {}
        for H in heads:
            for name, root in arms:
                cmd = [sys.executable, os.path.abspath(__file__), "one", "--arm", name, "--H", str(H)] + \
                      (["--root", root] if root else [])
                p = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
                if p.returncode != 0:
                    x = {"error": (p.stderr.strip().splitlines() or ["?"])[-1][:300]}
                else:
                    x = json.loads(p.stdout.strip().splitlines()[-1])
                row[f"H{H}/{name}"] = x
                print(f"round {r} H{H} {name}: " + (f"fwd {x['fwd_ms']:.3f} fwd+bwd {x['fwdbwd_ms']:.3f} ms"
                                                    if "error" not in x else x["error"]), flush=True)
        out["rounds"].append(row)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)


def summarize(raw):
    arms = list(raw["rounds"][0])
    nr = raw["protocol"]["rounds"]
    if len(raw["rounds"]) != nr or any(sorted(r) != sorted(arms) for r in raw["rounds"]):
        raise SystemExit(f"raw coverage: {len(raw['rounds'])} rounds of {nr}, or arms differ between rounds")
    for name in arms:
        ok = [r[name] for r in raw["rounds"] if "error" not in r[name]]
        if ok and len(ok) != nr:
            raise SystemExit(f"{name}: failed in some rounds only")
        if len({json.dumps(x.get("identity"), sort_keys=True) for x in ok}) > 1:
            raise SystemExit(f"{name}: build / module identity differs between rounds")
        for x in ok:
            if not (x["fwd_ms"] > 0 and x["fwdbwd_ms"] > x["fwd_ms"] and len(x["fwd_events"]) == ITERS
                    and abs(statistics.median(x["fwd_events"]) - x["fwd_ms"]) < 1e-9):
                raise SystemExit(f"{name}: malformed timings")
    res = {}
    for name in arms:
        W = work(**dict(raw["protocol"]["shape"], H=int(name.split("/")[0][1:])))
        rows = [r[name] for r in raw["rounds"] if name in r]
        if any("error" in x for x in rows):
            res[name] = {"error": next(x["error"] for x in rows if "error" in x)}
            continue
        res[name] = {"identity": rows[0].get("identity")}
        for ph in ("fwd", "bwd", "fwdbwd"):
            ms = [x[f"{ph}_ms"] for x in rows]
            tf = [W[ph] / (m * 1e-3) / 1e12 for m in ms]
            n = len(tf)
            res[name][ph] = {"ms_mean": statistics.mean(ms), "tflops_mean": statistics.mean(tf),
                             "tflops_se": statistics.stdev(tf) / math.sqrt(n) if n > 1 else float("nan"), "rounds": n}
    return res


def report(a):
    raw = json.load(open(a.raw)); res = summarize(raw)
    s = raw["protocol"]["shape"]
    L = [f"# Paper protocol (Appendix C.2): B{s['B']} N{s['N']} Hq 16 / 32, Hk{s['Hk']} D{s['D']} causal", "",
         f"Raw: `{a.raw}` (job {raw.get('job')}, {raw.get('host')}); {raw['protocol']['rounds']} interleaved rounds, "
         f"{CLOCK_WARM_S:.0f} s matmul clock warm-up per benchmark, median of {ITERS} CUDA events after {WARM} warm-ups; "
         "TFLOP/s from nominal work (fwd 2*B*Hq*N^2*d, bwd 5*B*Hq*N^2*d); mean +- SE over rounds; bwd = fwd+bwd - fwd. "
         "delta-matching: prequantized inputs, no autograd wrapper (GQA dK/dV fold excluded), matched-delta correction "
         "pass included.", "",
         "| heads / arm | fwd TFLOP/s | bwd TFLOP/s | fwd+bwd TFLOP/s | fwd ms | fwd+bwd ms |", "|---|---:|---:|---:|---:|---:|"]
    for name, r in res.items():
        if "error" in r:
            L.append(f"| {name} | not run: {r['error'][:80]} | | | | |")
            continue
        L.append(f"| {name} | {r['fwd']['tflops_mean']:.1f} +- {r['fwd']['tflops_se']:.1f} | "
                 f"{r['bwd']['tflops_mean']:.1f} +- {r['bwd']['tflops_se']:.1f} | "
                 f"{r['fwdbwd']['tflops_mean']:.1f} +- {r['fwdbwd']['tflops_se']:.1f} | {r['fwd']['ms_mean']:.3f} | "
                 f"{r['fwdbwd']['ms_mean']:.3f} |")
    L.append(f"| H16 / paper (matched) | {PAPER['fwd']} | {PAPER['bwd']} | {PAPER['fwdbwd']} | | |")
    for H in raw["protocol"]["heads"]:
        ref = res.get(f"H{H}/fa3_bf16")
        if ref and "error" not in ref:
            L += ["", f"H{H}: fwd+bwd TFLOP/s relative to FA3 BF16: " + ", ".join(
                f"{n.split('/')[1]} {r['fwdbwd']['tflops_mean'] / ref['fwdbwd']['tflops_mean']:.2f}x"
                for n, r in res.items() if n.startswith(f"H{H}/") and "error" not in r)]
    out = "\n".join(L) + "\n"
    if a.md:
        open(a.md, "w").write(out)
    print(out)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run"); r.add_argument("--arms", required=True); r.add_argument("--out", required=True)
    r.add_argument("--rounds", type=int, default=ROUNDS); r.add_argument("--heads", default="16,32")
    o = sp.add_parser("one"); o.add_argument("--arm", required=True); o.add_argument("--root", default=None)
    o.add_argument("--H", type=int, choices=HEADS, default=16)
    p = sp.add_parser("report"); p.add_argument("--raw", required=True); p.add_argument("--md", default=None)
    g = sp.add_parser("diag"); g.add_argument("--root", required=True); g.add_argument("--H", type=int, default=16)
    a = ap.parse_args()
    if a.cmd == "run":
        sys.path.insert(0, HERE)
        from _paths import results_path
        a.out = results_path(a.out)
        run(a)
    elif a.cmd == "one":
        print(json.dumps(one(a.arm, a.root, a.H)))
    elif a.cmd == "diag":
        SHAPE["H"] = a.H
        print(json.dumps(diag(a.root), indent=1))
    else:
        report(a)


if __name__ == "__main__":
    main()
