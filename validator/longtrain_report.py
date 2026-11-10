#!/usr/bin/env python
"""Summarize the long training runs: per arm, mean train loss per 1000-step window, held-out eval losses, divergence
flags, and the matched / stale gaps to bf16.  Writes <dir>/REPORT.md and prints it.
  python validator/longtrain_report.py [runs dir]"""
import glob
import json
import math
import os
import sys

R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ.get("DMVAL_HOME", os.path.expanduser("~/.cache/dm_validator")), "longtrain", "runs")
W = 1000
arms = {}
for d in sorted(glob.glob(os.path.join(R, "*_s*"))):
    recs = [json.loads(l) for l in open(os.path.join(d, "log.jsonl")) if l.strip()]
    if not recs:
        continue
    by = {}
    for r in recs:                       # restarts may repeat steps: keep the last record per step
        by[r["step"]] = r
    steps = sorted(by)
    arms[os.path.basename(d)] = (steps, by)
lines = [f"# Long training: bf16 vs fp8 matched vs fp8 stale (seq 8192, 16L d1536, same seed + data)", ""]
names = sorted(arms)
maxstep = max((s[-1] for s, _ in arms.values()), default=0)
lines.append("| step window | " + " | ".join(names) + " |")
lines.append("|---|" + "---|" * len(names))
for w0 in range(0, maxstep, W):
    row = []
    for n in names:
        steps, by = arms[n]
        v = [by[s]["loss"] for s in steps if w0 < s <= w0 + W and math.isfinite(by[s]["loss"])]
        row.append(f"{sum(v) / len(v):.4f}" if v else "")
    lines.append(f"| {w0 + 1}-{w0 + W} | " + " | ".join(row) + " |")
lines += ["", "Held-out eval loss (latest):"]
for n in names:
    steps, by = arms[n]
    ev = [(s, by[s]["eval"]) for s in steps if "eval" in by[s]]
    nonfin = sum(1 for s in steps if not math.isfinite(by[s]["loss"]))
    lines.append(f"- {n}: step {steps[-1]}, eval {ev[-1][1]:.4f} @ {ev[-1][0]}" if ev else f"- {n}: step {steps[-1]}"
                 + (f", NON-FINITE steps: {nonfin}" if nonfin else ""))
base = next((n for n in names if n.startswith("bf16")), None)
if base:
    lines += ["", f"Gap to {base} (mean train loss over the last common {W} steps):"]
    common = min(arms[n][0][-1] for n in names)
    def tail(n):
        steps, by = arms[n]
        v = [by[s]["loss"] for s in steps if common - W < s <= common and math.isfinite(by[s]["loss"])]
        return sum(v) / len(v) if v else float("nan")
    b = tail(base)
    for n in names:
        if n != base:
            lines.append(f"- {n}: {tail(n) - b:+.4f} ({(tail(n) / b - 1) * 100:+.2f}%) at steps {common - W + 1}-{common}")
open(os.path.join(os.path.dirname(R.rstrip('/')), "REPORT.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
