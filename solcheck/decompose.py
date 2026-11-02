"""Per-kernel decomposition of the ideal synthetic's fraction of sol4 (one run directory with analysis.json):
    fraction = [8192 x f_eff / (calibrated peak per SM)] x [C_tensor / C_best]
C_tensor = tensor cycles per step at 8192 FLOP / clk / SM; C_alu = cycles per step of the ALU mix alone (same warpgroups,
x_*_dep, and the role-split v_*_spec_noT); C_best = cycles per step of the best variant (slowest role);
kappa = (C_best - C_alu) / C_tensor: ALU-mix cycles lost per tensor cycle.

    python solcheck/decompose.py <rundir> [...]
"""
import json, os, statistics, sys

rows = {}
for run in sys.argv[1:]:
    a = json.load(open(os.path.join(run, "analysis.json")))
    r, peak = a["results"], a["peak_tflops"] * 1e12 / 132
    for c in ("fwd", "pass1", "pass0"):
        vs = [n for n in (f"syn_{c}_dep", f"syn_{c}_free", f"v_{c}_ilv", f"v_{c}_spec", f"v_{c}_pp") if n in r]
        best = max(vs, key=lambda n: r[n]["pct_of_floor"])
        b = r[best]
        flop = b["flop_per_clk_sm"] * b["cyc_per_step"]
        ct = flop / 8192
        cb = max(b.get("wg_cyc_per_step") or [b["cyc_per_step"]])
        xa = r[f"x_{c}_dep"]["cyc_per_step"]
        sn = r.get(f"v_{c}_spec_noT", {}).get("wg_cyc_per_step")
        ca = max(sn[1:]) if sn else xa
        f = b["mhz"] * 1e6
        rows.setdefault(c, []).append({"best": best, "pct": b["pct_of_floor"], "C_tensor": ct, "C_alu_x": xa, "C_alu_spec": ca,
                                       "C_best": cb, "f_eff_mhz": b["mhz"], "clock_factor": 8192 * f / (peak),
                                       "tensor_util": ct / cb, "kappa": (cb - ca) / ct,
                                       "x_ipc": r[f"x_{c}_dep"]["ipc_sm"], "x_mufu": r[f"x_{c}_dep"]["mufu_lanes_per_clk_sm"],
                                       "x_ctl": r[f"x_{c}_dep"]["controlling"], "x_pct": r[f"x_{c}_dep"]["pct_of_floor"]})
print("| kernel | best variant | % of sol4 | C_tensor | C_alu alone (x / spec_noT) | C_best | kappa | f_eff MHz | "
      "8192 f_eff / calib | C_tensor / C_best | ALU mix alone: IPC / MUFU lanes / bound |")
print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
for c, rs in rows.items():
    m = lambda k: statistics.mean(x[k] for x in rs)
    print(f"| {c} | {rs[0]['best']} | {m('pct'):.1f}% | {m('C_tensor'):.0f} | {m('C_alu_x'):.0f} / {m('C_alu_spec'):.0f} | "
          f"{m('C_best'):.0f} | {m('kappa'):.2f} | {m('f_eff_mhz'):.0f} | {m('clock_factor'):.3f} | {m('tensor_util'):.3f} | "
          f"{m('x_ipc'):.2f} / {m('x_mufu'):.1f} / {rs[0]['x_ctl']} {m('x_pct'):.0f}% |")
