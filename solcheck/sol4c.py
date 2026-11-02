"""Ideal-mix corrected floors ("sol4c", kda-solcheck proposal): the sol4-v2 floor of each compute kernel divided by the
fraction of that floor an ideal kernel with the same step-loop instruction mix reaches on this H100 (the best
synthetic step kernel, measured against the same-job calibration; solcheck/summary.py).  The ratio real / ideal does
not depend on the calibration of the job, since both sides are measured against the same one.

    python solcheck/sol4c.py --sol4 sol4v2.json [--factors results/solcheck/factors.json] [--md out.md]
"""
import argparse, json, os

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sol4", required=True)
    ap.add_argument("--factors", default=os.path.join(HERE, "..", "results", "solcheck", "factors.json"))
    ap.add_argument("--md")
    a = ap.parse_args()
    s = json.load(open(a.sol4)); f = json.load(open(a.factors))
    L = [f"sol4c (ideal-mix floors, factors {os.path.basename(a.factors)}: "
         + ", ".join(f"{k} {v['ideal_fraction_of_sol4']:.3f}" for k, v in f["kernels"].items()) + ")", "",
         "| part | kernel | measured ms | sol4 floor ms | % of sol4 | ideal-mix floor ms | % of ideal-mix floor |",
         "|---|---|---:|---:|---:|---:|---:|"]
    for part, T in s["totals"].items():
        for c, k in T["kernels"].items():
            if c not in f["kernels"]:
                continue
            fl = k["sol_ms"] / f["kernels"][c]["ideal_fraction_of_sol4"]
            L.append(f"| {part} | {c} | {k['measured_ms']:.3f} | {k['sol_ms']:.3f} | {k['pct_of_floor']:.1f}% | {fl:.3f} | "
                     f"{100 * fl / k['measured_ms']:.1f}% |")
    print("\n".join(L))
    if a.md:
        open(a.md, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
