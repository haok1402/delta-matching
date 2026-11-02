"""Analysis of a solcheck run: rates per microbenchmark and the sol4 fraction of each synthetic step kernel.

    python solcheck/analyze.py --build <builddir> --run <rundir> [--md out.md] [--json out.json]

Per kernel (median over reps): cycles per step per warpgroup (clock64, CTA average), effective SM clock (clock64 /
globaltimer), wall TFLOP/s, tensor FLOP per clock per SM, warp instructions issued per clock per SM (loop count from
the SASS x warps / cycles), MUFU lane-ops per clock per SM.  For the synthetic step kernels, the four sol4-v2 floors
of the synthetic itself (tensor at the same-job calibrated fp8 GEMM peak; MUFU 16 / clk / SM and issue 4 / clk / SM
at the calibration's nvidia-smi clock) and the fraction max(floor) / measured.
"""
import argparse, collections, json, os, statistics, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmarks"))
import sol4  # noqa: E402


def loop_stats(sass_text):
    funcs = sol4.parse_sass(sass_text)
    (name, body), = funcs.items()
    loops = []
    for hi_i, t in sol4.back_edges(body):
        lo_i = next(i for i, b in enumerate(body) if b[0] == t)
        loops.append(body[lo_i:hi_i + 1])
    # the step loop: the last loop in the function (the operand fill loop precedes it); with role-split kernels
    # (warpgroups running different loops) every loop after the fill is a step loop: report the per-role loops
    seg = max(loops, key=lambda l: l[0][0])
    h = collections.Counter(o if o.startswith(("QGMMA", "MUFU")) else o.split(".")[0] for _, _, o, _ in seg)
    return {"instr": len(seg), "mufu": sum(v for k, v in h.items() if k.startswith("MUFU")),
            "qgmma_flops": sum(sol4.qgmma_flops(o) for _, _, o, _ in seg), "hist": dict(h)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True); ap.add_argument("--run", required=True)
    ap.add_argument("--md"); ap.add_argument("--json")
    a = ap.parse_args()
    cfgs = {c["name"]: c for c in json.load(open(os.path.join(a.build, "all.json")))}
    calib = json.load(open(os.path.join(a.run, "calib.json")))
    peak = calib["gemm_fp8"]["peak_tflops"] * 1e12
    clk = calib["sm_clock_mhz"] * 1e6
    runs = collections.defaultdict(list)
    for line in open(os.path.join(a.run, "runs.jsonl")):
        r = json.loads(line)
        if "kernel" in r and "ms" in r:
            runs[r["kernel"]].append(r)
    res = {}
    for name, rs in runs.items():
        c = cfgs[name]
        st = loop_stats(open(os.path.join(a.build, "sass", name + ".sass")).read())
        wg_mult = 1
        if c.get("equiv"):   # role-split / rescheduled variant: charge the per-step work of the synthetic it mirrors
            ref, wg_mult = c["equiv"]
            st = dict(loop_stats(open(os.path.join(a.build, "sass", ref + ".sass")).read()), own=st)
            if wg_mult != 1:
                st = dict(st, wgs_equiv=wg_mult)
        ms = statistics.median(r["ms"] for r in rs) / rs[0]["launches"]          # per launch
        cyc = statistics.median(r["cyc_per_launch"] for r in rs)
        mhz = statistics.median(r["mhz_last"] for r in rs)
        steps, sms, wgs = rs[0]["steps"], rs[0]["sms"], (wg_mult if c.get("equiv") and wg_mult != 1 else c["wgs"])
        cps = cyc / steps                                                         # cycles per step (each WG one step)
        warps = wgs * 4
        flop_step_sm = wgs * st["qgmma_flops"]                                    # per step, all WGs of the SM
        d = {"ms_launch": ms, "steps": steps, "cyc_per_step": cps, "mhz": mhz, "wgs": wgs,
             "instr_per_warp_step": st["instr"], "mufu_per_warp_step": st["mufu"],
             "flop_per_clk_sm": flop_step_sm / cps, "tflops": flop_step_sm * steps * sms / (ms * 1e-3) / 1e12,
             "ipc_sm": warps * st["instr"] / cps, "mufu_lanes_per_clk_sm": warps * st["mufu"] * 32 / cps,
             "hist": st["hist"], "kind": c.get("kind", "syn"), "spread_ms": [r["ms"] / r["launches"] for r in rs]}
        if c.get("target_instr") is not None or c.get("kind") in ("tensor_of", "alu_of", "mixed", "tensor", "variant"):
            fl = {"tensor": flop_step_sm * steps * sms / peak * 1e3,
                  "mufu": warps * steps * st["mufu"] * 32 / (16 * clk) * 1e3,
                  "issue": warps * steps * st["instr"] / (4 * clk) * 1e3}
            ctl = max(fl, key=fl.get)
            d.update(floors_ms=fl, controlling=ctl, pct_of_floor=100 * fl[ctl] / ms)
        if rs[0].get("wg_cyc"):
            d["wg_cyc_per_step"] = [statistics.median(r["wg_cyc"][w] for r in rs) / steps for w in range(len(rs[0]["wg_cyc"]))]
        res[name] = d
    lines = [f"calibration: fp8 GEMM {calib['gemm_fp8']['peak_tflops']:.1f} TFLOP/s, nvidia-smi clock "
             f"{calib['sm_clock_mhz']:.0f} MHz ({calib['gemm_fp8']['peak_tflops'] * 1e12 / (132 * clk):.0f} FLOP/clk/SM "
             f"at that clock; dense fp8 peak 8192), copy {calib['copy']['peak_gbs']:.0f} GB/s, job {calib.get('job')}", ""]
    hdr = ("| kernel | WGs | cyc/step | MHz | TFLOP/s | FLOP/clk/SM | IPC/SM | MUFU lanes/clk/SM | instr/warp/step | "
           "floors t / m / i (ms) | ctl | % of sol4 |")
    lines += [hdr, "|" + "---|" * 12]
    for name in sorted(res, key=lambda n: (res[n]["kind"], n)):
        d = res[name]
        fl = d.get("floors_ms")
        fls = f"{fl['tensor']:.3f} / {fl['mufu']:.3f} / {fl['issue']:.3f}" if fl else ""
        lines.append(f"| {name} | {d['wgs']} | {d['cyc_per_step']:.1f} | {d['mhz']:.0f} | {d['tflops']:.0f} | "
                     f"{d['flop_per_clk_sm']:.0f} | {d['ipc_sm']:.2f} | {d['mufu_lanes_per_clk_sm']:.2f} | "
                     f"{d['instr_per_warp_step']} | {fls} | {d.get('controlling', '')} | "
                     f"{d['pct_of_floor']:.1f}% |" if fl else
                     f"| {name} | {d['wgs']} | {d['cyc_per_step']:.1f} | {d['mhz']:.0f} | {d['tflops']:.0f} | "
                     f"{d['flop_per_clk_sm']:.0f} | {d['ipc_sm']:.2f} | {d['mufu_lanes_per_clk_sm']:.2f} | "
                     f"{d['instr_per_warp_step']} | | | |")
    lines += ["", "Per-warpgroup cycles per step (role-split kernels; warpgroup 0 = tensor role):", "",
              "| kernel | MHz | cycles per step per warpgroup |", "|---|---|---|"]
    for name in sorted(res):
        d = res[name]
        if name.startswith(("v_", "i_")) and d.get("wg_cyc_per_step"):
            lines.append(f"| {name} | {d['mhz']:.0f} | " + " / ".join(f"{x:.0f}" for x in d["wg_cyc_per_step"]) + " |")
    out = "\n".join(lines)
    print(out)
    if a.md:
        open(a.md, "w").write(out + "\n")
    if a.json:
        json.dump({"calib": {k: calib[k] for k in ("job", "host", "sm_clock_mhz", "sha256")},
                   "peak_tflops": calib["gemm_fp8"]["peak_tflops"], "results": res}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
