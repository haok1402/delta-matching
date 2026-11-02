"""Median nvidia-smi SM clock / power / throttle reasons per phase (realclock.jsonl, sustain.jsonl, calibrate window).

    python solcheck/smi_phases.py <rundir>
"""
import collections, datetime, json, os, statistics, sys

d = sys.argv[1]
rows = []
for i, line in enumerate(open(os.path.join(d, "smi.csv"))):
    if i == 0:
        continue
    p = [x.strip() for x in line.split(",")]
    try:
        t = datetime.datetime.strptime(p[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
        rows.append((t, float(p[1]), float(p[3]), int(p[5]), p[6]))
    except (ValueError, IndexError):
        pass


def window(t0, t1, skip=0.5):
    sel = [r for r in rows if t0 + skip <= r[0] <= t1]
    if not sel:
        return None
    reasons = collections.Counter(r[4] for r in sel)
    return {"samples": len(sel), "mhz_median": statistics.median(r[1] for r in sel), "mhz_min": min(r[1] for r in sel),
            "watts_median": statistics.median(r[2] for r in sel), "temp_max": max(r[3] for r in sel),
            "reasons": dict(reasons.most_common(3))}


out = {}
for fn in ("realclock.jsonl",):
    p = os.path.join(d, fn)
    if os.path.exists(p):
        for line in open(p):
            r = json.loads(line)
            out["real_" + r["phase"]] = dict(window(r["t0"], r["t1"]) or {}, ms_per_call=r["ms_per_call"], shape=r["shape"])
for fn in ("sustain.jsonl", "runs.jsonl"):
    p = os.path.join(d, fn)
    if not os.path.exists(p):
        continue
    marks = [json.loads(l) for l in open(p) if l.startswith('{"mark"')]
    for a, b in zip(marks, marks[1:]):
        w = window(a["t"], b["t"], skip=0.3)
        if w:
            out[("sus_" if fn.startswith("sus") else "") + a["mark"]] = w
json.dump(out, open(os.path.join(d, "smi_phases.json"), "w"), indent=1)
for k, v in out.items():
    if k.startswith(("real", "sus", "syn", "t_SS128_w2", "t_SS64_w4", "x_")):
        print(k, {kk: vv for kk, vv in v.items() if kk != "shape"})
