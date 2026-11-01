"""Phase profile of pass 1 (dK / dV / dS') from a DM_PROF=1 build (clock64 probes; see csrc/kda_prof.h).

Nsight Compute counters are not available on the cluster this was written for (RmProfilingAdminOnly=1), so pass 1 is
profiled by its own probes: per consumer warpgroup, the cycles spent in each phase of the per-m-block tile loop
(pipeline waits, wgmma exposure, exp, dS, the psi barrier, conversions, the dS' transpose / store, the fold), per
tile (prologue, exit) and per CTA (lifetime, epilogue), summed over every CTA of every pass-1 launch.

    PYTHONPATH=<DM_PROF build root> python benchmarks/profile_pass1.py --out prof.json        # phase cycles
    PYTHONPATH=<any build root>     python benchmarks/profile_pass1.py --time-only --out t.json  # pass-1 kernel time

The tensor-core ideal per CTA m-block step is 1024 cycles at both head dims (4 wgmma of 2^21 FLOP per CTA step at
D 128 (64 x 128 tile) and at D 256 (64 x 64 tile), 8192 dense fp8 FLOP / clk / SM).
"""
import argparse, json, os, statistics, sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kernels as K  # noqa: E402

CONFIGS = [  # (D, B, H, Hk, N, attn)
    (128, 4, 16, 16, 8192, "causal"), (128, 4, 16, 16, 8192, "full"), (128, 1, 16, 16, 32768, "causal"),
    (128, 4, 16, 4, 8192, "causal"),
    (256, 4, 8, 8, 8192, "causal"), (256, 4, 8, 8, 8192, "full"), (256, 1, 8, 8, 32768, "causal"),
]
PHASES = {   # slot -> label, per variant (0 = hd128 P1-R, 1 = d256 V10)
    0: {2: "prologue (to first m-block)", 3: "wait Q stage", 4: "issue S + wait dO + issue dP", 5: "wait S (wgmma)",
        6: "mask + exp2 + release Q", 7: "wait dP (wgmma)", 8: "dS line + release dO", 9: "psi amax + shuffles",
        10: "psi barrier (cross-WG)", 11: "psi math + P / dS' fp8 conversion", 12: "dS' transpose shfl / prmt",
        13: "store drain + WG barriers + stmatrix + TMA", 14: "dV rescale + wait dOt + issue dV",
        15: "wait Qt + issue dK", 16: "wait dV (wgmma) + release", 17: "wait dK (wgmma) + release",
        18: "dK fold + dV promote", 19: "exit (drain + unfold + dV readback)"},
    1: {2: "prologue (to first m-block)", 3: "wait Q stage", 4: "issue S + wait dO + issue dP", 5: "wait S (wgmma)",
        6: "release Q + exp2", 7: "wait dP (wgmma) + release dO", 8: "dS line", 9: "psi amax + slot publish",
        10: "psi barrier (cross-WG)", 11: "psi math + P conversion + store drain + PdS barrier",
        12: "P~ / dS' exchange stores (smem)", 13: "PdS barrier + TMA dS' store", 14: "dV rescale + wait dOt + issue dV",
        15: "wait Qt + issue dK", 16: "wait dV (wgmma) + release", 17: "wait dK (wgmma) + releases",
        18: "dK fold + dV promote", 19: "exit (drain + dV readback)"},
}
IDEAL_STEP = 1024


def run(cfg, steps, prof):
    import delta_matching as dm
    D, B, H, Hk, N, attn = cfg
    causal = attn == "causal"
    mode = "matched" if causal else "stale"
    g = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16, generator=g, requires_grad=True)
    k = torch.randn(B, N, Hk, D, device="cuda", dtype=torch.bfloat16, generator=g, requires_grad=True)
    v = torch.randn(B, N, Hk, D, device="cuda", dtype=torch.bfloat16, generator=g, requires_grad=True)
    do = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16, generator=g)

    def step():
        q.grad = k.grad = v.grad = None
        dm.dm_attn_func(q, k, v, causal=causal, mode=mode).backward(do)
    names = K.kernel_times(step, steps)
    p1 = sum(ms for n, ms in names.items() if K.classify(n) == "pass1")
    rec = {"cfg": list(cfg), "mode": mode, "pass1_ms": p1, "steps": steps}
    if prof:
        buf = torch.ops.delta_matching.prof_buffer()
        buf.zero_()
        torch.cuda.synchronize()
        for _ in range(steps):
            step()
        torch.cuda.synchronize()
        rec["buf"] = buf.cpu().tolist()
    return rec


def analyze(rec, clk):
    D = rec["cfg"][0]
    var = 0 if D == 128 else 1
    out = {}
    for wg in (0, 1):
        sl = rec["buf"][var][wg]
        nstep = sl[0]
        if nstep == 0:
            continue
        mma = sl[1]
        phases = {PHASES[var][i]: sl[i] / nstep for i in range(3, 19)}
        loop = sum(sl[i] for i in range(3, 19)) / nstep
        out[f"wg{wg}"] = {"m_block_steps": nstep, "cycles_per_step": loop, "phases_per_step": phases,
                          "prologue_per_tile": sl[2], "exit_cycles_total": sl[19], "mma_cycles_total": mma,
                          "tensor_busy_upper_bound": IDEAL_STEP / loop}
        if var == 0 and sl[22]:
            out[f"wg{wg}"].update(tiles=sl[22], cta_cycles_per_tile=sl[20] / sl[22], epilogue_cycles_per_tile=sl[21] / sl[22],
                                  prologue_cycles_per_tile=sl[2] / sl[22], exit_cycles_per_tile=sl[19] / sl[22])
    # SM-cycle view: the kernel's SM-cycles (132 SMs x time x clock) vs the cycles the consumer loops account for
    sm_cycles = 132 * rec["pass1_ms"] * 1e-3 * clk * 1e6 * rec["steps"]
    if "wg0" in out:
        out["sm_cycles_total"] = sm_cycles
        out["loop_share_of_sm_cycles"] = out["wg0"]["cycles_per_step"] * out["wg0"]["m_block_steps"] / sm_cycles
        out["ideal_tensor_share"] = IDEAL_STEP * out["wg0"]["m_block_steps"] / sm_cycles
    return out


def fmt(rec, an):
    D, B, H, Hk, N, attn = rec["cfg"]
    lines = [f"### D{D} B{B} H{H}/{Hk} N{N} {attn} ({rec['mode']}): pass 1 {rec['pass1_ms']:.3f} ms per call", ""]
    if "wg0" not in an:
        return "\n".join(lines + ["(no probe data)"])
    w0, w1 = an["wg0"], an.get("wg1", {})
    lines += [f"m-block steps per call (WG0): {w0['m_block_steps'] / rec['steps']:.0f}; cycles per step: WG0 {w0['cycles_per_step']:.0f}, "
              f"WG1 {w1.get('cycles_per_step', float('nan')):.0f}; tensor-core ideal {IDEAL_STEP} -> tensor busy <= "
              f"{100 * w0['tensor_busy_upper_bound']:.0f}%; consumer loops cover {100 * an['loop_share_of_sm_cycles']:.0f}% of the "
              f"kernel's SM-cycles (ideal tensor share {100 * an['ideal_tensor_share']:.0f}%)", ""]
    if "cta_cycles_per_tile" in w0:
        lines += [f"per tile (CTA): lifetime {w0['cta_cycles_per_tile']:.0f} cycles, prologue {w0['prologue_cycles_per_tile']:.0f}, "
                  f"exit {w0['exit_cycles_per_tile']:.0f}, epilogue (dK/dV store) {w0['epilogue_cycles_per_tile']:.0f}", ""]
    lines += ["| phase | WG0 cycles / step | share | WG1 cycles / step |", "|---|---:|---:|---:|"]
    for ph, c in w0["phases_per_step"].items():
        c1 = w1.get("phases_per_step", {}).get(ph, float("nan"))
        lines.append(f"| {ph} | {c:.0f} | {100 * c / w0['cycles_per_step']:.1f}% | {c1:.0f} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--time-only", action="store_true")
    ap.add_argument("--clock", type=float, default=1950.0, help="SM clock (MHz) for the SM-cycle view")
    a = ap.parse_args()
    import delta_matching as dm
    prof = not a.time_only
    if prof and not hasattr(torch.ops.delta_matching, "prof_buffer"):
        dm.core._ops()
        if not hasattr(torch.ops.delta_matching, "prof_buffer"):
            raise SystemExit("not a DM_PROF=1 build (no prof_buffer op)")
    dm.core._ops()
    recs = []
    for cfg in CONFIGS:
        r = run(cfg, a.steps, prof)
        if prof:
            r["analysis"] = analyze(r, a.clock)
            print(fmt(r, r["analysis"]), "\n", flush=True)
        else:
            print(f"{cfg}: pass 1 {r['pass1_ms']:.3f} ms", flush=True)
        recs.append(r)
    so = dm.core.loaded_library()
    with open(a.out, "w") as f:
        json.dump({"meta": {"library": os.path.realpath(so), "library_sha256": K.sha256(so), "clock_mhz": a.clock,
                            "prof": prof}, "records": recs}, f, indent=1)


if __name__ == "__main__":
    main()
