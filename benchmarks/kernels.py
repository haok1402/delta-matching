"""Per-kernel timing and speed-of-light accounting for the delta_matching kernels on Hopper.

Each configuration runs the full API route (dm_attn_func forward + backward) and splits the device time per kernel with
the torch profiler (CUPTI kernel records): the forward (fwd_fp8), the forward producers (Triton quantizers + twins), the
backward producer (dO quantizer, stale delta), pass 0 (the matched delta), pass 1 (dK / dV + the dS' tiles), pass 2 (dQ),
the twins, and the torch glue (GQA dK / dV fold, copies).  The wall-clock forward and forward + backward come from CUDA
events without the profiler.  The SM clock is sampled with nvidia-smi during the timed region.

Speed of light per kernel: max(FLOPs / peak, bytes / HBM bandwidth), with the dense fp8 peak taken at the measured
sustained clock (132 SMs x 8192 fp8 FLOP / clk / SM: 1979 TFLOPS at 1830 MHz) and at its nominal value, and the HBM3
bandwidth at 3.35 TB/s.  FLOPs count the matmuls only (causal: the N (N + 1) / 2 visible pairs); bytes are the compulsory
traffic of each kernel (operands read once, outputs written once; the dS' tiles that pass 1 writes and pass 2 reads are
counted because the split design needs them).

    python benchmarks/kernels.py --grid main --out results/baseline.json
    python benchmarks/kernels.py --grid holdout --out results/holdout.json
    python benchmarks/kernels.py --D 128 --N 8192 --B 2 --H 16 --Hk 4      # one shape
    python benchmarks/kernels.py ... --fa3                                   # add FlashAttention-3 bf16 / fp8 arms
"""
import argparse, json, os, statistics, subprocess, sys, threading, time

import torch

import delta_matching as dm

SMS = 132
FP8_FLOP_PER_CLK_SM = 8192
BF16_FLOP_PER_CLK_SM = 4096
NOMINAL_FP8 = 1979e12
NOMINAL_BF16 = 989e12
HBM = 3.35e12

# grid of (D, B, H, Hk, N); tokens B * N = 32k per call, H = 16 at D 128 and 8 at D 256 (the same q bytes)
MAIN = [(D, 32768 // N, H, Hk, N) for D, H in ((128, 16), (256, 8)) for N in (2048, 4096, 8192, 16384, 32768)
        for Hk in (H, H // 4)]
# held-out shapes: never tuned on (odd sequence lengths, other head counts / GQA ratios, batch 3)
HOLDOUT = [(128, 3, 12, 4, 6144), (128, 1, 24, 8, 12288), (256, 3, 6, 2, 6144), (256, 1, 12, 4, 12288),
           (128, 2, 32, 32, 3072), (256, 2, 16, 1, 5120)]
MODES = (("causal", "matched"), ("causal", "stale"), ("full", "stale"))

CATS = ("fwd", "prod_fwd", "prod_bwd", "pass0", "pass1", "pass2", "twin", "glue")


def classify(name):
    n = name
    if "FlashAttnFwdSm90Fq8F" in n:
        return "fwd"
    if "FlashAttnBwdSm90Fp8" in n:
        return "pass1"
    if "fq8p2" in n or "dq_pass2" in n or "Pass2" in n:
        return "pass2"
    if "delta" in n.lower() or "fp11d" in n or "Fp11Delta" in n or "fp11_reduce" in n:
        return "pass0"
    if "twin" in n.lower():
        return "twin"
    if "_doq" in n:
        return "prod_bwd"
    if "quant" in n or "kmean" in n:
        return "prod_fwd"
    return "glue"


class ClockSampler:
    """nvidia-smi SM clock / power samples every 20 ms, for the GPU this process uses."""

    def __init__(self):
        self.samples, self._stop, self._p = [], False, None

    def __enter__(self):
        idx = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[torch.cuda.current_device()]
        try:
            self._p = subprocess.Popen(["nvidia-smi", "-i", idx, "--query-gpu=clocks.sm,power.draw,utilization.gpu",
                                        "--format=csv,noheader,nounits", "-lms", "20"], stdout=subprocess.PIPE, text=True)
        except FileNotFoundError:
            return self
        self._t = threading.Thread(target=self._read, daemon=True)
        self._t.start()
        return self

    def _read(self):
        for line in self._p.stdout:
            try:
                c, p, u = (float(x) for x in line.split(","))
                self.samples.append((time.time(), c, p, u))
            except ValueError:
                pass

    def window(self, t0, t1):
        s = [c for (t, c, p, u) in self.samples if t0 <= t <= t1 and u > 50]
        return statistics.median(s) if s else float("nan")

    def __exit__(self, *a):
        if self._p:
            self._p.terminate()


def pairs(N, causal):
    return N * (N + 1) / 2 if causal else float(N) * N


def accounting(D, B, H, Hk, N, causal, mode):
    """FLOPs and compulsory bytes per kernel category for one call."""
    P = pairs(N, causal) * B * H          # visible (query, key) pairs over all heads
    BHq, BHk, ND = B * H, B * Hk, N * D
    f, b = {}, {}
    f["fwd"] = 4 * P * D
    b["fwd"] = BHq * ND + 2 * BHk * ND + 2 * BHq * ND + 4 * BHq * N            # Q K V codes, O bf16, LSE
    f["pass0"] = 4 * P * D if mode == "matched" else 0
    b["pass0"] = (2 * BHq * ND + 2 * BHk * ND + 8 * BHq * N) if mode == "matched" else 0
    f["pass1"] = 8 * P * D
    # Q, dO, their twins, K, V, K twin codes; LSE, delta; dK and dV per query head (bf16); dS' codes (one byte per pair)
    b["pass1"] = 4 * BHq * ND + 3 * BHk * ND + 8 * BHq * N + 4 * BHq * ND + P
    f["pass2"] = 2 * P * D
    b["pass2"] = P + BHk * ND + 2 * BHq * ND                                    # dS' codes, K twin, dQ bf16
    f["prod_fwd"] = 0
    b["prod_fwd"] = 2 * (BHq + 2 * BHk) * ND + (2 * BHq + 3 * BHk) * ND         # bf16 in; Q, Qt, K, Kt, V codes out
    f["prod_bwd"] = 0
    b["prod_bwd"] = 2 * BHq * ND + BHq * ND + (2 * BHq * ND + 4 * BHq * N if mode == "stale" else 0)
    f["twin"] = 0
    b["twin"] = 2 * BHq * ND                                                    # dO twin (read + write)
    f["glue"] = 0
    b["glue"] = (2 * 2 * BHq * ND + 2 * 2 * BHk * ND) if Hk < H else 0          # GQA fold of dK / dV
    return f, b


def time_events(fn, iters, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    st = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    en = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        st[i].record(); fn(); en[i].record()
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) for s, e in zip(st, en))


def kernel_times(fn, iters, warm=3):
    """Mean device ms per call for every kernel name (summed over launches within a call)."""
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA], acc_events=True) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    tot = {}
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            dt = getattr(e, "device_time", None)
            if dt is None:
                dt = e.cuda_time
            tot[e.name] = tot.get(e.name, 0.0) + dt / 1e3
    return {k: v / iters for k, v in tot.items()}


def sustained_clock(clock, seconds=4.0):
    """Median SM clock under a sustained fp8 attention load (D 128, N 8192, causal fwd + bwd, back to back)."""
    B, H, Hk, N, D = 2, 16, 16, 8192, 128
    q, k, v = (torch.randn(B, N, h, D, device="cuda", dtype=torch.bfloat16, requires_grad=True) for h in (H, Hk, Hk))
    do = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
    t0 = time.time()
    while time.time() - t0 < seconds:
        for _ in range(20):
            dm.dm_attn_func(q, k, v, causal=True, mode="matched").backward(do)
        torch.cuda.synchronize()
    t1 = time.time()
    return clock.window(t0 + 1.0, t1)


def run_config(D, B, H, Hk, N, attn, mode, iters, clock, fa3, clk_sustained):
    causal = attn == "causal"
    g = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16, generator=g, requires_grad=True)
    k = torch.randn(B, N, Hk, D, device="cuda", dtype=torch.bfloat16, generator=g, requires_grad=True)
    v = torch.randn(B, N, Hk, D, device="cuda", dtype=torch.bfloat16, generator=g, requires_grad=True)
    do = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16, generator=g)

    def fwd():
        return dm.dm_attn_func(q, k, v, causal=causal, mode=mode)

    def step():
        q.grad = k.grad = v.grad = None
        fwd().backward(do)

    t0 = time.time()
    rec = {"D": D, "B": B, "H": H, "Hk": Hk, "N": N, "attn": attn, "mode": mode}
    rec["e2e_fwd_ms"] = time_events(fwd, iters)
    rec["e2e_fwdbwd_ms"] = time_events(step, iters)
    names = kernel_times(step, iters)
    rec["clock_mhz"] = clock.window(t0, time.time())
    per = {c: 0.0 for c in CATS}
    for n, ms in names.items():
        per[classify(n)] += ms
    rec["kernels"] = names
    rec["ms"] = per
    f, b = accounting(D, B, H, Hk, N, causal, mode)
    clk = clk_sustained
    peak = SMS * FP8_FLOP_PER_CLK_SM * clk * 1e6
    sol = {}
    for c in CATS:
        t_f, t_b = f[c] / peak * 1e3, b[c] / HBM * 1e3
        t_fn = f[c] / NOMINAL_FP8 * 1e3
        sol[c] = {"flops": f[c], "bytes": b[c], "sol_ms": max(t_f, t_b), "sol_nominal_ms": max(t_fn, t_b),
                  "bound": "compute" if t_f >= t_b else "memory",
                  "tflops": f[c] / (per[c] * 1e-3) / 1e12 if per[c] > 0 else 0.0,
                  "pct_sol": 100 * max(t_f, t_b) / per[c] if per[c] > 0 else float("nan")}
    rec["sol"] = sol
    if fa3:
        rec["fa3"] = fa3_arms(q, k, v, do, causal, iters)
    del q, k, v, do
    torch.cuda.empty_cache()
    return rec


def fa3_arms(q, k, v, do, causal, iters):
    try:
        from flash_attn_interface import flash_attn_func as fa3
    except Exception:  # noqa: BLE001
        return None

    def first(r):
        return r[0] if isinstance(r, tuple) else r
    out = {}
    out["bf16_fwd_ms"] = time_events(lambda: first(fa3(q, k, v, causal=causal)), iters)

    def bstep():
        q.grad = k.grad = v.grad = None
        first(fa3(q, k, v, causal=causal)).backward(do)
    out["bf16_fwdbwd_ms"] = time_events(bstep, iters)
    kt = kernel_times(bstep, iters)
    out["bf16_fwd_kernel_ms"] = sum(ms for n, ms in kt.items() if "FlashAttnFwd" in n)
    out["bf16_bwd_kernel_ms"] = sum(kt.values()) - out["bf16_fwd_kernel_ms"]   # preprocess + main + dQ convert
    q8, k8, v8 = (x.detach().to(torch.float8_e4m3fn) for x in (q, k, v))
    try:
        kt = kernel_times(lambda: first(fa3(q8, k8, v8, causal=causal)), iters)
        out["fp8_fwd_kernel_ms"] = sum(ms for n, ms in kt.items() if "FlashAttnFwd" in n)
    except Exception as e:  # noqa: BLE001
        out["fp8_fwd_error"] = str(e)[:200]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=("main", "holdout", "one"), default="one")
    ap.add_argument("--D", type=int, default=128); ap.add_argument("--N", type=int, default=8192)
    ap.add_argument("--B", type=int, default=2); ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--Hk", type=int, default=4)
    ap.add_argument("--modes", default="all", help="all or a comma list of causal-matched,causal-stale,full-stale")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--fa3", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--names", action="store_true", help="print every kernel name and its category")
    a = ap.parse_args()
    shapes = MAIN if a.grid == "main" else HOLDOUT if a.grid == "holdout" else [(a.D, a.B, a.H, a.Hk, a.N)]
    modes = MODES if a.modes == "all" else tuple(tuple(m.split("-")) for m in a.modes.split(","))
    recs = []
    with ClockSampler() as clock:
        clk = sustained_clock(clock)
        if clk != clk:
            clk = 1830.0
        print(f"sustained SM clock {clk:.0f} MHz -> dense fp8 peak {SMS * FP8_FLOP_PER_CLK_SM * clk / 1e6:.0f} TFLOPS", flush=True)
        for (D, B, H, Hk, N) in shapes:
            for attn, mode in modes:
                r = run_config(D, B, H, Hk, N, attn, mode, a.iters, clock, a.fa3, clk)
                recs.append(r)
                ms = r["ms"]
                print(f"D{D} B{B} H{H} Hk{Hk} N{N} {attn:6s} {mode:7s} clk {r['clock_mhz']:.0f}  e2e fwd {r['e2e_fwd_ms']:.3f} "
                      f"fwd+bwd {r['e2e_fwdbwd_ms']:.3f} | " + " ".join(f"{c} {ms[c]:.3f}" for c in CATS if ms[c] > 0), flush=True)
                if a.names:
                    for n, t in sorted(r["kernels"].items(), key=lambda x: -x[1]):
                        print(f"    {t:8.4f} ms  {classify(n):8s} {n[:150]}")
                if r.get("fa3"):
                    print("    fa3 " + " ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in r["fa3"].items()))
    meta = {"sustained_clock_mhz": clk, "torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0),
            "kernel_version": {k: v for k, v in dm.kernel_version().items() if k != "numerics"},
            "env": {k: v for k, v in os.environ.items() if k.startswith("DM_")}}
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump({"meta": meta, "records": recs}, fh, indent=1, default=str)


if __name__ == "__main__":
    sys.exit(main())
