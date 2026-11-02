"""Same-job comparison: real kernels (sol4-v2 on kernels.py --grid paper, frozen build) vs the ideal synthetic step
kernels of the same instruction mix, both against the same calibration.

    python solcheck/summary.py <rundir> [<rundir> ...] --build <builddir> [--md out.md]
"""
import argparse, json, os, statistics

CATS = {"fwd": "fwd", "pass1": "pass1", "pass0": "pass0"}
VARIANTS = ("syn_{c}_dep", "syn_{c}_free", "v_{c}_ilv", "v_{c}_spec", "v_{c}_pp")


def one(run):
    an = json.load(open(os.path.join(run, "analysis.json")))
    real = json.load(open(os.path.join(run, "real_sol4.json")))
    t = real["totals"]["acceptance"]["kernels"]
    calib = an["peak_tflops"]
    out = {"calib_tflops": calib, "job": an["calib"]["job"], "kernels": {}}
    for c in CATS:
        r = an["results"]
        var = {v.format(c=c): r[v.format(c=c)]["pct_of_floor"] for v in VARIANTS if v.format(c=c) in r}
        best = max(var, key=var.get)
        out["kernels"][c] = {"real_pct": t[c]["pct_of_floor"], "real_ms": t[c]["measured_ms"],
                             "sol4_ms": t[c]["sol_ms"], "controlling": t[c]["controlling"],
                             "variants": var, "best": best, "best_pct": var[best],
                             "best_mhz": r[best]["mhz"], "dep_cyc_per_step": r[f"syn_{c}_dep"]["cyc_per_step"],
                             "real_over_best": t[c]["pct_of_floor"] / var[best]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+"); ap.add_argument("--md")
    a = ap.parse_args()
    res = [one(r) for r in a.runs]
    L = []
    for s in res:
        L += [f"job {s['job']}, calibrated fp8 GEMM {s['calib_tflops']:.1f} TFLOP/s", "",
              "| kernel | real % of sol4 | dep | free | ilv | spec | pp | best synthetic | real / best |",
              "|---|---:|---:|---:|---:|---:|---:|---|---:|"]
        for c, k in s["kernels"].items():
            v = k["variants"]
            f = lambda n: f"{v[n.format(c=c)]:.1f}%" if n.format(c=c) in v else "-"
            L.append(f"| {c} | {k['real_pct']:.1f}% | {f('syn_{c}_dep')} | {f('syn_{c}_free')} | {f('v_{c}_ilv')} | "
                     f"{f('v_{c}_spec')} | {f('v_{c}_pp')} | {k['best']} {k['best_pct']:.1f}% | {k['real_over_best']:.3f} |")
        L.append("")
    if len(res) > 1:
        L += ["Mean over jobs:", "", "| kernel | real % of sol4 | best synthetic % of sol4 | real / best |", "|---|---:|---:|---:|"]
        for c in CATS:
            L.append(f"| {c} | {statistics.mean(s['kernels'][c]['real_pct'] for s in res):.1f}% | "
                     f"{statistics.mean(s['kernels'][c]['best_pct'] for s in res):.1f}% | "
                     f"{statistics.mean(s['kernels'][c]['real_over_best'] for s in res):.3f} |")
    print("\n".join(L))
    if a.md:
        open(a.md, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
