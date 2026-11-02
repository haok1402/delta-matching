import json, os, statistics, sys, math
O = sys.argv[1]; roots = [os.path.basename(r.rstrip('/')) for r in sys.argv[2:]]
def load(tag):
    runs = []
    for f in sorted(os.listdir(O)):
        if f.startswith(tag + "_") and f.endswith(".json") and f[len(tag)+1:-5].isdigit():
            runs.append(json.load(open(os.path.join(O, f))))
    return runs
data = {t: load(t) for t in roots}
A = roots[0]
shapes = [(s["H"], s["B"], s["N"]) for s in data[A][0]["shapes"]]
def get(t, key, cat=None):
    out = {}
    for sh in shapes:
        vals = []
        for run in data[t]:
            for s in run["shapes"]:
                if (s["H"], s["B"], s["N"]) == sh:
                    vals.append(s[key] if cat is None else s["cat_us"].get(cat, 0.0) / 1e3)
        out[sh] = vals
    return out
lines = []
for t in roots[1:]:
    lines.append(f"== {t} vs {A} ({len(data[t])} / {len(data[A])} reps; medians; range of B reps)")
    for what, cat in (("bwd wall", None), ("pass0", "pass0"), ("pass1", "pass1"), ("pass2", "pass2"), ("glue", "glue")):
        a = get(A, "wall_ms_med" if cat is None else None, cat); b = get(t, "wall_ms_med" if cat is None else None, cat)
        rat = []; cells = []
        for sh in shapes:
            ma, mb = statistics.median(a[sh]), statistics.median(b[sh])
            if ma <= 0: continue
            rat.append(mb / ma); cells.append(f"H{sh[0]}B{sh[1]}N{sh[2]//1024}k {100*(mb/ma-1):+.2f}% [{min(b[sh]):.3f}-{max(b[sh]):.3f} vs {min(a[sh]):.3f}-{max(a[sh]):.3f}]")
        if rat:
            g = math.exp(sum(math.log(x) for x in rat) / len(rat))
            lines.append(f"  {what:8s} geomean {100*(g-1):+.2f}% | " + " | ".join(cells))
print("\n".join(lines)); open(os.path.join(O, "summary.txt"), "w").write("\n".join(lines) + "\n")
