"""Whole-backward timeline of the matched path (kda-fuse helper): per shape, the backward wall time (CUDA events around
torch.autograd.grad on a retained graph, so only the backward runs), and from a CUDA-only profiler trace of single
backward calls: per-category kernel time, the busy union, idle gaps between kernels (classified by the categories on
both sides), launch counts.  The build is the delta_matching package found on PYTHONPATH (set by the caller).

    python bwd_timeline.py --shapes 16:2:8192,32:2:8192,16:1:65536,32:1:65536 --iters 20 --out t.json
"""
import argparse, json, os, statistics, sys, tempfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))


def classify(n):
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


def build(B, H, Hk, N, D=128, seed=0):
    import delta_matching as dm
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16, generator=g, requires_grad=True)
    k = torch.randn(B, N, Hk, D, device="cuda", dtype=torch.bfloat16, generator=g, requires_grad=True)
    v = torch.randn(B, N, Hk, D, device="cuda", dtype=torch.bfloat16, generator=g, requires_grad=True)
    do = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16, generator=g)
    o = dm.dm_attn_func(q, k, v, causal=True, mode="matched")
    return (q, k, v), o, do


def bwd_fn(ins, o, do):
    def f():
        return torch.autograd.grad(o, ins, do, retain_graph=True)
    return f


def wall_ms(f, iters, warm=3):
    for _ in range(warm):
        f()
    torch.cuda.synchronize()
    st = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    en = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        st[i].record(); f(); en[i].record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in zip(st, en)]


def trace_one(f, reps):
    """Per backward call: the kernel list (name, start us, dur us) from a chrome trace, calls separated by syncs."""
    from torch.profiler import profile, ProfilerActivity
    f(); torch.cuda.synchronize()
    calls = []
    for _ in range(reps):
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            f(); torch.cuda.synchronize()
        fd, path = tempfile.mkstemp(suffix=".json"); os.close(fd)
        prof.export_chrome_trace(path)
        ev = json.load(open(path))["traceEvents"]
        os.remove(path)
        ks = sorted(((e["name"], float(e["ts"]), float(e["dur"]), e.get("tid")) for e in ev
                     if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and e.get("ph") == "X"), key=lambda x: x[1])
        calls.append(ks)
    return calls


def analyze(ks):
    t0 = ks[0][1]; t1 = max(s + d for _, s, d, _ in ks)
    cats, n = {}, {}
    for name, s, d, _ in ks:
        c = classify(name); cats[c] = cats.get(c, 0.0) + d; n[c] = n.get(c, 0) + 1
    # busy union and gaps (in order of start; overlapping kernels on several streams merge)
    busy, gaps, cur_end, cur_cat = 0.0, {}, None, None
    for name, s, d, _ in ks:
        c = classify(name)
        if cur_end is None:
            busy += d; cur_end = s + d; cur_cat = c; continue
        if s > cur_end:
            key = f"{cur_cat}->{c}"; gaps[key] = gaps.get(key, 0.0) + (s - cur_end)
            busy += d; cur_end = s + d
        else:
            busy += max(0.0, s + d - cur_end); cur_end = max(cur_end, s + d)
        cur_cat = c
    return {"wall_us": t1 - t0, "busy_us": busy, "idle_us": (t1 - t0) - busy, "cat_us": cats, "launches": n,
            "gaps_us": gaps, "n_kernels": len(ks)}


def median_dict(ds):
    keys = set().union(*ds)
    return {k: statistics.median(d.get(k, 0.0) for d in ds) for k in sorted(keys)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="16:2:8192,32:2:8192,16:1:65536,32:1:65536")
    ap.add_argument("--Hk", type=int, default=4)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--traces", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kernels", action="store_true", help="dump the kernel list of the first trace")
    a = ap.parse_args()
    import delta_matching as dm
    res = {"package": os.path.dirname(dm.__file__), "env": {k: v for k, v in os.environ.items() if k.startswith("DM_")},
           "shapes": []}
    for s in a.shapes.split(","):
        H, B, N = (int(x) for x in s.split(":"))
        ins, o, do = build(B, H, a.Hk, N)
        f = bwd_fn(ins, o, do)
        iters = a.iters if N <= 16384 else max(5, a.iters // 4)
        w = wall_ms(f, iters)
        calls = trace_one(f, a.traces)
        an = [analyze(c) for c in calls]
        rec = {"H": H, "B": B, "Hk": a.Hk, "N": N, "wall_ms_med": statistics.median(w), "wall_ms": w,
               "trace_wall_us": statistics.median(x["wall_us"] for x in an),
               "trace_busy_us": statistics.median(x["busy_us"] for x in an),
               "trace_idle_us": statistics.median(x["idle_us"] for x in an),
               "cat_us": median_dict([x["cat_us"] for x in an]), "gaps_us": median_dict([x["gaps_us"] for x in an]),
               "launches": an[0]["launches"], "n_kernels": an[0]["n_kernels"]}
        if a.kernels:
            rec["kernel_list"] = [(n_, round(s_ - calls[0][0][1], 1), round(d_, 1)) for n_, s_, d_, _ in calls[0]][:400]
        res["shapes"].append(rec)
        print(f"H{H} B{B} N{N}: bwd wall {rec['wall_ms_med']:.3f} ms | trace wall {rec['trace_wall_us']/1e3:.3f} busy "
              f"{rec['trace_busy_us']/1e3:.3f} idle {rec['trace_idle_us']/1e3:.3f} ms | kernels {rec['n_kernels']} | "
              + " ".join(f"{c} {v/1e3:.3f}" for c, v in rec["cat_us"].items()), flush=True)
        print("   gaps: " + " ".join(f"{k} {v:.0f}us" for k, v in sorted(rec["gaps_us"].items(), key=lambda x: -x[1])[:8]),
              flush=True)
        del ins, o, do, f
        torch.cuda.empty_cache()
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
