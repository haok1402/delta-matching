#!/usr/bin/env python
"""Headline for the user's focus lengths: python headline.py RESULTS_PREFIX [ARM]   (reads results/<prefix>.[sg]*.jsonl)
With several runs per shape (run_headline.sh: every shape on every GPU), each cell is the MEDIAN over runs/GPUs of the
same-run ratio, and a spread table (min..max over GPUs) follows.
Per-length table of fwd+bwd (and fwd, bwd) speedup = FA4_ms / arm_ms, same-run pairs, for N in 8k/32k/64k and both head
configs, causal and non-causal, plus the geomean over the 6 shapes (2 head configs x 3 N) per causal flag."""
import glob, json, math, statistics, sys


import os as _os
_HS = json.load(open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "headline_shapes.json")))["shapes"]
TARGET = {(x["H"], x["Hk"], x["N"], x["causal"]) for x in _HS}   # only the user's target shapes count

args = [x for x in sys.argv[1:] if not x.startswith("--calib=")]
calib = next((x.split("=", 1)[1] for x in sys.argv[1:] if x.startswith("--calib=")), None)
pre, arm = args[0], (args[1] if len(args) > 1 else "dm_matched")
d = {}
for f in glob.glob(f"results/{pre}.s*.jsonl") + glob.glob(f"results/{pre}.g*.jsonl"):
    for l in open(f):
        r = json.loads(l)
        if "pass_" in r and "error" not in r and (r["H"], r["Hk"], r["N"], r["causal"]) in TARGET:
            d[(r["run_id"], r["arm"], r["H"], r["Hk"], r["N"], r["causal"], r["pass_"])] = r
gm = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")  # noqa: E731
Ns = sorted({x["N"] for x in json.load(open(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "headline_shapes.json")))["shapes"]})
spread = {}
hosts = sorted({r["host"] for r in d.values()})
print(f"arm={arm}  files=results/{pre}.[sg]*.jsonl  host={','.join(hosts)}")
for base in ("fa4_bf16", "fa4_bf16_det"):
    print(f"\nvs {base}  (speedup = {base}_ms / {arm}_ms)\n")
    print("| causal | pass | H/Hk | " + " | ".join(f"{n//1024}k" for n in Ns) + f" | geomean ({2*len(Ns)} shapes) |")
    print("|" + "---|" * (4 + len(Ns)))
    for c in sorted({k[5] for k in d}, reverse=True):
        for p in ("fwd_bwd", "fwd", "bwd"):
            allx = []
            rows = []
            for (H, Hk) in ((16, 16), (32, 8)):
                cells = []
                for N in Ns:
                    xs = [d[k][ "ev_ms"] for k in d if k[1:] == (base, H, Hk, N, c, p)]
                    pair = [(d[k]["ev_ms"], d.get((k[0], arm) + k[2:])) for k in d if k[1:] == (base, H, Hk, N, c, p)]
                    pair = [b / r["ev_ms"] for b, r in pair if r is not None]
                    if pair:
                        med = statistics.median(pair)
                        spread[(c, p, H, N, base)] = (min(pair), max(pair), len(pair))
                        cells.append(f"{med:.3f}"); allx.append(med)
                    else:
                        cells.append("-")
                rows.append((H, Hk, cells))
            for i, (H, Hk, cells) in enumerate(rows):
                g = f"**{gm(allx):.3f}**" if (i == 0 and c and p == "fwd_bwd") else (f"{gm(allx):.3f}" if i == 0 else "")
                print(f"| {int(c)} | {p} | {H}/{Hk} | " + " | ".join(cells) + f" | {g} |")

nmax = max((v[2] for v in spread.values()), default=0)
if nmax > 1:
    print(f"\nspread across runs/GPUs (min..max of the per-GPU ratio, n = {nmax}), causal fwd+bwd vs fa4_bf16:\n")
    print("| H/Hk | " + " | ".join(f"{n//1024}k" for n in Ns) + " |")
    print("|" + "---|" * (1 + len(Ns)))
    for (H, Hk) in ((16, 16), (32, 8)):
        print(f"| {H}/{Hk} | " + " | ".join("%.3f..%.3f" % spread[(True, "fwd_bwd", H, N, "fa4_bf16")][:2]
                                         if (True, "fwd_bwd", H, N, "fa4_bf16") in spread else "-" for N in Ns) + " |")


# --calib=hl_<sha>: single-GPU headline -> placement-corrected estimate.  For each shape, factor = (median over the 4 GPUs)
# / (ratio on the GPU this run used), both from the 4-GPU run hl_<sha>; the estimate = this run's ratio x factor.  Valid
# while the node and the FA4 build are the same (FA4's per-GPU bias is what is corrected; the DM kernel is GPU-insensitive).
if calib:
    ref = {}
    for f in glob.glob(f"results/{calib}.g*.jsonl"):
        for l in open(f):
            r = json.loads(l)
            if "pass_" in r and "error" not in r:
                ref[(r["run_id"], r["cvd"], r["arm"], r["H"], r["Hk"], r["N"], r["causal"], r["pass_"])] = r
    cvds = sorted({str(r["cvd"]) for r in d.values()})
    print(f"\ncalibrated with {calib} (this run on GPU {','.join(cvds)}; host {','.join(sorted({r['host'] for r in ref.values()}))}):\n")
    print("| H/Hk | " + " | ".join(f"{n//1024}k" for n in Ns) + " | (causal fwd+bwd vs fa4_bf16: raw -> calibrated) |")
    print("|" + "---|" * (2 + len(Ns)))
    est_all = []
    for (H, Hk) in ((16, 16), (32, 8)):
        cells = []
        for N in Ns:
            per = {}
            for k, r in ref.items():
                if k[2:] == ("dm_matched", H, Hk, N, True, "fwd_bwd"):
                    b = ref.get((k[0], k[1], "fa4_bf16") + k[3:])
                    if b:
                        per[k[1]] = b["ev_ms"] / r["ev_ms"]
            pair = [(d[k_]["ev_ms"] / d[(k_[0], arm) + k_[2:]]["ev_ms"], d[k_]["cvd"]) for k_ in d
                    if k_[1:] == ("fa4_bf16", H, Hk, N, True, "fwd_bwd") and (k_[0], arm) + k_[2:] in d]
            if not pair or not per:
                cells.append("-"); continue
            x, g = pair[-1]
            fac = statistics.median(per.values()) / per.get(g, statistics.median(per.values()))
            cells.append(f"{x:.3f} -> {x * fac:.3f}"); est_all.append(x * fac)
        print(f"| {H}/{Hk} | " + " | ".join(cells) + " | |")
    print(f"\ncalibrated geomean ({len(est_all)} causal shapes with a calibration): **{gm(est_all):.3f}**")
