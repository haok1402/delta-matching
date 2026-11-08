#!/usr/bin/env python
"""Render attnbench.py JSONL into markdown tables.

  python report.py results/x.jsonl [more.jsonl ...] [--base fa4_bf16] [--clock ev_ms|cupti_ms] [--tag TAG]

Table 1: per (shape, pass): median ms and TFLOPS for every arm.  Table 2: speedup of each arm over --base
(base_ms / arm_ms; > 1 = faster than the base), with geometric means per pass and causal flag.  Speedups pair an arm only with the
base timed in the same process (same run_id, same GPU, interleaved rounds).  Rows whose sanity gate
failed are marked '!' and excluded from the geomeans.  When several records exist for the same key (re-runs), the
latest wins."""
import argparse, collections, json, math


def key(r):
    return (r["H"], r["Hk"], r["N"], r["B"], r["causal"], r["pass_"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--base", default="fa4_bf16")
    ap.add_argument("--clock", default="ev_ms")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    recs, byrun = {}, {}
    arms = []
    for f in a.files:
        for line in open(f):
            line = line.strip()
            if not line.startswith("{"):
                continue
            r = json.loads(line)
            if "pass_" not in r or "error" in r or (a.tag is not None and r.get("tag") != a.tag):
                continue
            recs[(r["arm"],) + key(r)] = r
            byrun[(r.get("run_id"), r["arm"]) + key(r)] = r
            if r["arm"] not in arms:
                arms.append(r["arm"])
    keys = sorted({k[1:] for k in recs}, key=lambda k: (k[5] != "fwd", k[5] != "bwd", not k[4], k[0], k[1], k[2]))
    print(f"clock = {a.clock}; TFLOPS = FA-convention FLOPs / median time\n")
    print("| H/Hk | N | B | causal | pass | " + " | ".join(f"{x} ms (TF)" for x in arms) + " |")
    print("|" + "---|" * (5 + len(arms)))
    sp = collections.defaultdict(list)
    for k in keys:
        cells = []
        for arm in arms:
            r = recs.get((arm,) + k)
            if r is None:
                cells.append("-"); continue
            t = r.get(a.clock, float("nan"))
            bad = r.get("sanity", {}).get("suspect", False)
            cells.append(f"{t:.3f} ({r['tflops'] * r['ev_ms'] / t if t == t else float('nan'):.0f}){' !' if bad else ''}")
            b = byrun.get((r.get("run_id"), a.base) + k)   # ratios only from the same process / GPU
            if b is not None and arm != a.base and not bad and t == t:
                sp[(arm, k[5], k[4])].append(b.get(a.clock) / t)
        print(f"| {k[0]}/{k[1]} | {k[2]} | {k[3]} | {int(k[4])} | {k[5]} | " + " | ".join(cells) + " |")
    print(f"\nSpeedup over {a.base} (geomean of {a.base}_ms / arm_ms over shapes):\n")
    print("| arm | pass | causal | geomean | min | max | n |")
    print("|---|---|---|---|---|---|---|")
    for (arm, p, c), xs in sorted(sp.items()):
        gm = math.exp(sum(map(math.log, xs)) / len(xs))
        print(f"| {arm} | {p} | {int(c)} | {gm:.3f} | {min(xs):.3f} | {max(xs):.3f} | {len(xs)} |")


if __name__ == "__main__":
    main()
