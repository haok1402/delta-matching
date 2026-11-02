"""Four-floor speed-of-light model, version "sol4-v1" (Hao's audit direction 5).  The historical two-floor model
(kernels.model_sol) is unchanged; every JSON written before stays reproducible.

    python benchmarks/sol4.py counts --lib <_C.so> --out counts.json           # emitted-code loop counts (CPU only)
    python benchmarks/sol4.py model --runs A_median.json --calib calib.json --counts counts.json --out sol4.json \\
        [--producers strided|copy] [--md table.md]

Per kernel category and configuration it emits four floors separately, their maximum and the controlling floor:
  tensor = tile-granular matmul FLOPs (whole diagonal tiles, as the kernels compute them) / calibrated fp8 GEMM peak
  MUFU   = executed MUFU lane-ops / (16 per clock per SM x SMs x measured SM clock)
  issue  = executed warp instructions / (4 per clock per SM x SMs x measured SM clock)
  bytes  = modeled compulsory HBM bytes (kernels.accounting, with the glue ledger selected by the producer mode) /
           calibrated device-copy bandwidth
Executed counts (CUDA kernels): the steady-state main loop of the selected emitted function (the smallest backward-branch
loop whose MUFU.EX2 count is a whole multiple of the declared per-thread exps of one step and that holds the step's QGMMA
batch), times declared multiplicities: units (visited tiles of the kernel's step granularity, per head, per warpgroup)
x 4 warps per warpgroup / steps per loop iteration.  The static loop body includes both arms of any in-loop branch, so the
issue floor is an approximation (high where an in-loop branch is rarely taken); the counts file records every candidate
loop so the choice can be audited.  Triton producers, twins and glue (memory kernels) get the bytes floor only
(tensor / MUFU 0, issue not modeled: stated per record).  These are MODELED floors, not hardware utilization.
"""
import argparse, hashlib, json, math, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

MODEL_VERSION = "sol4-v1"
MUFU_PER_CLK_SM = 16
ISSUE_PER_CLK_SM = 4
CATS = ("fwd", "prod_fwd", "prod_bwd", "pass0", "pass1", "pass2", "twin", "glue")

# D128 causal CUDA kernels of the target grid: symbol pattern, the step (rows x keys per warpgroup per loop step), the
# unit tiling for visited-tile counting (rows, keys per unit, warpgroups that each run one step per unit), matmul FLOPs
# per score element (2 per GEMM), exps per thread per step.
KERNELS_D128_CAUSAL = {
    "fwd": {"symbol": r"FlashAttnFwdSm90Fq8F.*ILi128EEENS7_ILi256EEESA_.*DynamicPersistent", "unit": (128, 256), "wgs": 2,
            "step_elems_per_wg": 64 * 256, "gemm_flop_per_elem": 2 * 2 * 128, "exps_per_thread_step": 128},
    "pass1": {"symbol": r"INS1_20FlashAttnBwdSm90Fp8R.*SingleTileBwdLPTScheduler", "unit": (64, 128), "wgs": 2,
              "step_elems_per_wg": 64 * 64, "gemm_flop_per_elem": 4 * 2 * 128, "exps_per_thread_step": 32},
    "pass0": {"symbol": r"DeltaV6ILb1E", "unit": (128, 64), "wgs": 2,
              "step_elems_per_wg": 64 * 64, "gemm_flop_per_elem": 2 * 2 * 128, "exps_per_thread_step": 32},
    "pass2": {"symbol": r"fq8_dq_pass2_kernel_hd128ILb1E", "unit": (64, 128), "wgs": 1,
              "step_elems_per_wg": 64 * 128, "gemm_flop_per_elem": 1 * 2 * 128, "exps_per_thread_step": 0},
}


class ModelError(Exception):
    pass


# ----------------------------------------------------------------------------------------------- emitted-code counts
def sass_functions(lib):
    out = subprocess.run(["cuobjdump", "-sass", lib], capture_output=True, text=True, check=True).stdout
    funcs, cur = {}, None
    for line in out.splitlines():
        m = re.search(r"Function : (\S+)", line)
        if m:
            cur = m.group(1); funcs[cur] = []
            continue
        m = re.search(r"/\*([0-9a-f]{4,6})\*/\s+(@!?U?P\w+\s+)?([A-Z0-9_.]+)([^;]*)", line)
        if cur and m:
            funcs[cur].append((int(m.group(1), 16), m.group(3), m.group(4)))
    return funcs


def loops(body):
    """Every backward-branch loop [target, branch] of a function body with its instruction / MUFU.EX2 / QGMMA counts."""
    out = []
    for addr, op, rest in body:
        if op.startswith("BRA"):
            m = re.search(r"0x([0-9a-f]+)", rest)
            if m and int(m.group(1), 16) < addr:
                lo = int(m.group(1), 16)
                ins = [o for a, o, _ in body if lo <= a <= addr]
                out.append({"start": lo, "end": addr, "instr": len(ins), "mufu_ex2": sum(o.startswith("MUFU.EX2") for o in ins),
                            "qgmma": sum(o.startswith("QGMMA") for o in ins)})
    return out


def choose_loop(cands, exps):
    """The steady-state step loop: holds QGMMA; MUFU.EX2 a whole positive multiple of the per-step exps (any count when
    the kernel has no exp); the smallest such loop (the unmasked variant of a masked / unmasked pair)."""
    ok = []
    for c in cands:
        if c["qgmma"] == 0:
            continue
        if exps:
            if c["mufu_ex2"] < exps:
                continue
            steps = c["mufu_ex2"] // exps   # per-row exps (basis / rescale / fold) add less than one step's worth
            if steps < 1:
                continue
        else:
            steps = 1
        ok.append(dict(c, steps=steps))
    if not ok:
        raise ModelError("no main loop consistent with the declared step")
    return min(ok, key=lambda c: c["instr"])


def counts(a):
    funcs = sass_functions(a.lib)
    res = {"version": MODEL_VERSION, "library": os.path.realpath(a.lib), "library_sha256": _sha(a.lib), "kernels": {}}
    for cat, k in KERNELS_D128_CAUSAL.items():
        names = [n for n in funcs if re.search(k["symbol"], n)]
        if len(names) != 1:
            raise ModelError(f"{cat}: {len(names)} functions match {k['symbol']!r}")
        cands = loops(funcs[names[0]])
        ch = choose_loop(cands, k["exps_per_thread_step"])
        res["kernels"][cat] = {"function": names[0], "chosen": ch, "candidates": cands, "declared": k}
        print(f"{cat:6s} loop {ch['start']:#x}-{ch['end']:#x}: {ch['instr']} instr, {ch['mufu_ex2']} MUFU.EX2, "
              f"{ch['qgmma']} QGMMA, {ch['steps']} step(s)")
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------- the model
def visited_units(N, bm, bn, causal):
    nbm, nbn = N // bm, N // bn
    if not causal:
        return nbm * nbn
    return sum(min(nbn, -(-((mb + 1) * bm) // bn)) for mb in range(nbm))


def glue_bytes(D, B, H, Hk, N, producers):
    """The glue ledger by producer mode: 'copy' = the historical (B, N, H, D) -> (B*H, N, D) q / k / v copies plus the GQA
    fold; 'strided' (H3, round 14) = the GQA dK / dV fold only."""
    ND, BHq, BHk = N * D, B * H, B * Hk
    fold = (2 * 2 * BHq * ND + 2 * 2 * BHk * ND) if Hk < H else 0
    if producers == "copy":
        return 2 * 2 * (BHq + 2 * BHk) * ND + fold
    if producers == "strided":
        return fold
    raise ModelError(f"unknown producer mode {producers!r}")


def floors(rec, calib, cnt, producers):
    """sol4 floors (ms) for one D128 causal record."""
    import kernels as K
    if not calib or not calib.get("sha256") or not calib.get("sm_clock_mhz") or not calib.get("sms"):
        raise ModelError("calibration with sha256, SM clock and SM count is required")
    peak_f = calib["gemm_fp8"]["peak_tflops"] * 1e12
    peak_b = calib["copy"]["peak_gbs"] * 1e9
    clk = calib["sm_clock_mhz"] * 1e6
    sms = calib["sms"]
    if not (peak_f > 0 and peak_b > 0 and clk > 0):
        raise ModelError("non-positive calibrated peak / clock")
    D, B, H, Hk, N = rec["D"], rec["B"], rec["H"], rec["Hk"], rec["N"]
    causal = rec["attn"] == "causal"
    if D != 128 or not causal or rec["mode"] != "matched":
        raise ModelError("sol4-v1 covers the D128 causal matched configurations")
    _, b = K.accounting(D, B, H, Hk, N, causal, rec["mode"], rec.get("layout", "bnhd"))
    b = dict(b, glue=glue_bytes(D, B, H, Hk, N, producers))
    out = {}
    for c in CATS:
        fl = {"bytes": b[c] / peak_b * 1e3, "tensor": 0.0, "mufu": 0.0, "issue": None}
        if c in KERNELS_D128_CAUSAL:
            k = KERNELS_D128_CAUSAL[c]
            kc = (cnt or {}).get("kernels", {}).get(c)
            if not kc:
                raise ModelError(f"emitted-code counts for {c} are required")
            units = visited_units(N, k["unit"][0], k["unit"][1], causal) * B * H
            elems = units * k["unit"][0] * k["unit"][1]
            fl["tensor"] = elems * k["gemm_flop_per_elem"] / peak_f * 1e3
            ch = kc["chosen"]
            steps_per_unit = k["unit"][0] * k["unit"][1] / k["wgs"] / k["step_elems_per_wg"]   # loop steps per warpgroup
            warp_iters = units * k["wgs"] * 4 * steps_per_unit / ch["steps"]
            fl["issue"] = warp_iters * ch["instr"] / (ISSUE_PER_CLK_SM * sms * clk) * 1e3
            fl["mufu"] = warp_iters * ch["mufu_ex2"] * 32 / (MUFU_PER_CLK_SM * sms * clk) * 1e3
        vals = {n: v for n, v in fl.items() if v is not None}
        ctl = max(vals, key=vals.get)
        ms = rec["ms"][c]
        out[c] = {"floors_ms": fl, "sol_ms": vals[ctl], "controlling": ctl, "measured_ms": ms,
                  "pct_of_floor": 100 * vals[ctl] / ms if ms > 0 else None,
                  "issue_modeled": fl["issue"] is not None}
    return out


def model(a):
    runs = json.load(open(a.runs)); calib = json.load(open(a.calib)); cnt = json.load(open(a.counts))
    recs = [r for r in runs["records"] if r["D"] == 128 and r["attn"] == "causal" and r["mode"] == "matched"]
    res = {"version": MODEL_VERSION, "runs": a.runs, "calibration_sha256": calib.get("sha256"),
           "counts_library_sha256": cnt.get("library_sha256"), "producers": a.producers, "configs": []}
    for r in recs:
        res["configs"].append({"key": [r["D"], r["B"], r["H"], r["Hk"], r["N"]], "target": r["N"] in (4096, 8192),
                               "floors": floors(r, calib, cnt, a.producers)})
    tot = {}
    for part in ("target", "guard"):
        sel = [c for c in res["configs"] if c["target"] == (part == "target")]
        t = {}
        for c in CATS:
            m = sum(x["floors"][c]["measured_ms"] for x in sel); s = sum(x["floors"][c]["sol_ms"] for x in sel)
            fl = {n: sum((x["floors"][c]["floors_ms"][n] or 0) for x in sel) for n in ("tensor", "mufu", "issue", "bytes")}
            t[c] = {"measured_ms": m, "sol_ms": s, "waste_ms": m - s, "pct_of_floor": 100 * s / m if m else None,
                    "floors_ms": fl, "controlling": max(fl, key=fl.get)}
        tot[part] = {"configs": len(sel), "kernels": t}
    res["totals"] = tot
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    lines = []
    for part in ("target", "guard"):
        t = tot[part]["kernels"]
        lines += [f"### {part} ({tot[part]['configs']} configurations, model {MODEL_VERSION}, calibration "
                  f"{res['calibration_sha256'][:12]})", "",
                  "| kernel | measured ms | tensor | MUFU | issue | bytes | floor (max) | controlling | % of floor | waste ms |",
                  "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|"]
        for c in sorted(CATS, key=lambda c: -t[c]["waste_ms"]):
            v = t[c]; fl = v["floors_ms"]
            lines.append(f"| {c} | {v['measured_ms']:.3f} | {fl['tensor']:.3f} | {fl['mufu']:.3f} | {fl['issue']:.3f} | "
                         f"{fl['bytes']:.3f} | {v['sol_ms']:.3f} | {v['controlling']} | {v['pct_of_floor']:.1f}% | {v['waste_ms']:.3f} |")
        lines.append("")
    if a.md:
        open(a.md, "w").write("\n".join(lines))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("counts"); c.add_argument("--lib", required=True); c.add_argument("--out", required=True)
    m = sp.add_parser("model")
    for n in ("runs", "calib", "counts", "out"):
        m.add_argument(f"--{n}", required=True)
    m.add_argument("--producers", choices=("strided", "copy"), default="strided"); m.add_argument("--md", default=None)
    a = ap.parse_args()
    try:
        counts(a) if a.cmd == "counts" else model(a)
    except ModelError as e:
        raise SystemExit(f"sol4: {e}")


if __name__ == "__main__":
    main()
