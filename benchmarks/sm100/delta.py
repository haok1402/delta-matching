#!/usr/bin/env python
"""Compare two FULL runs: python delta.py OLD_SHA NEW_SHA  (reads results/full_<sha>.s*.jsonl).
Geomean speedups vs FA4 (same-run pairs) for each dm arm/causal/pass, change, vs FA4-det, per-shape candidate-time change.
Refuses incomplete data (shape sets must match)."""
import collections, glob, json, math, sys


def load(tag):
    d = {}
    for f in glob.glob(f"results/{tag}.s*.jsonl" if tag.startswith(("ab_", "focus_")) else f"results/full_{tag}.s*.jsonl"):
        for l in open(f):
            r = json.loads(l)
            if "pass_" in r and "error" not in r:
                d[(r["run_id"], r["arm"], r["H"], r["Hk"], r["N"], r["causal"], r["pass_"])] = r
    return d


def ratios(d, base="fa4_bf16"):
    out = collections.defaultdict(dict)
    for k, r in d.items():
        if k[1].startswith("dm_"):
            b = d.get((k[0], base) + k[2:])
            if b:
                out[(k[1], k[5], k[6])][k[2:5]] = (b["ev_ms"] / r["ev_ms"], r["ev_ms"])
    return out


gm = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs))  # noqa: E731
old, new = sys.argv[1], sys.argv[2]
A, B, Bd = ratios(load(old)), ratios(load(new)), ratios(load(new), "fa4_bf16_det")
print(f"| arm | causal | pass | {old} | {new} | change | {new} vs FA4-det | per-shape speedup of cand (min..max) | n |")
print("|---|---|---|---|---|---|---|---|---|")
for key in sorted(B):
    a, b = A[key], B[key]
    if set(a) != set(b):
        sys.exit(f"incomplete data for {key}: {len(a)} vs {len(b)} shapes")
    ga, gb = gm([x[0] for x in a.values()]), gm([x[0] for x in b.values()])
    ch = [a[s][1] / b[s][1] for s in b]
    print(f"| {key[0]} | {int(key[1])} | {key[2]} | {ga:.3f} | {gb:.3f} | {100 * (gb / ga - 1):+.1f}% | "
          f"{gm([x[0] for x in Bd[key].values()]):.3f} | {min(ch):.3f}..{max(ch):.3f} | {len(b)} |")
