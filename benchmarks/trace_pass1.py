"""Sparse timeline of pass 1 from a DM_TRACE=1 build (csrc/kda_prof.h KDA_TRACE): absolute clock64 event times for a
fixed sample of CTAs (linear block id lb % 17 == 0, lb / 17 < 8), the first 12 m-block steps of each sampled CTA's first
tile, for the two consumer warpgroups (actors 0, 1) and the producer warp (actor 2).

    python benchmarks/trace_pass1.py capture --root <DM_TRACE build root> --out trace.json   # one backward per config
    python benchmarks/trace_pass1.py summarize trace.json > summary.md

Timestamps are compared only within one CTA (one SM).  Every derived quantity is a BRACKETED REGION between two
probe events (it includes whatever work and probe overhead lies between them, not only the named wait); wgmma
"outstanding" windows run from issue to the observed warpgroup_wait / release, a software bound, not tensor-pipe
activity; q_issue_lead is how long before the consumers asked for Q(m+1) the producer had ISSUED it (not when it landed).
Producer acquire regions for dOt / Q / Qt start at the previous event (no separate entry marker).  Event ids (consumer; both variants): 0 step start (before the Q
wait), 3 Q wait done, 21 S issued (before the dO wait), 22 dO wait done, 4 dP issued, 5 S retired, 6 exp done, 7 dP
retired, 8 dS done, 9 psi amax published, 10 psi barrier passed, 11 conversions (hd128) / first PdS barrier (d256) done,
12 transpose (hd128) / exchange stores (d256) done, 13 dS' store issued, 23 / 24 dOt wait entry / exit, 14 dV issued,
25 / 26 Qt wait entry / exit, 15 dK issued, 16 dV retired, 17 dK retired, 18 step end; 30 / 31 Q stage index / phase.
Producer: 30 / 31 stage index / phase, 1 / 2 dO acquire entry / exit, 3 dO issued, 4 dOt acquired, 5 dOt issued,
7 Q(m+1) acquired, 8 Q(m+1) issued, 9 Qt(m+1) acquired, 10 Qt(m+1) issued; header 8 = K / V issued.
"""
import argparse, json, os, statistics, sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import profile_pass1 as PP  # noqa: E402

KT_S, KT_M, KT_E, KT_HDR = 8, 12, 32, 16
KT_EVENTS = 2 * KT_S * KT_M * 3 * KT_E
CONFIGS = [(128, 4, 16, 16, 8192, "causal"), (128, 4, 16, 16, 8192, "full"),
           (256, 4, 8, 8, 8192, "causal"), (256, 4, 8, 8, 8192, "full")]


def capture(a):
    meta = PP.select_build(a.root)
    import delta_matching as dm
    ops = torch.ops.delta_matching
    if not hasattr(ops, "prof_buffer") or ops.prof_buffer().numel() != KT_EVENTS + 2 * KT_S * KT_HDR:
        raise SystemExit("not a DM_TRACE=1 build (prof_buffer of the trace size)")
    out = []
    for cfg in CONFIGS:
        D, B, H, Hk, N, attn = cfg
        causal = attn == "causal"
        mode = "matched" if causal else "stale"
        g = torch.Generator(device="cuda").manual_seed(0)
        q = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16, generator=g, requires_grad=True)
        k = torch.randn(B, N, Hk, D, device="cuda", dtype=torch.bfloat16, generator=g, requires_grad=True)
        v = torch.randn(B, N, Hk, D, device="cuda", dtype=torch.bfloat16, generator=g, requires_grad=True)
        do = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16, generator=g)
        dm.dm_attn_func(q, k, v, causal=causal, mode=mode).backward(do)      # warm-up (first-call checks, tables)
        torch.cuda.synchronize()
        buf = ops.prof_buffer()
        buf.zero_()
        torch.cuda.synchronize()
        o = dm.dm_attn_func(q, k, v, causal=causal, mode=mode)               # exactly one pass-1 launch (unsplit)
        o.backward(do)
        torch.cuda.synchronize()
        out.append({"cfg": list(cfg), "mode": mode, "buf": buf.cpu().tolist()})
        print(f"captured {cfg}", flush=True)
    with open(a.out, "w") as f:
        json.dump({"meta": dict(meta, sample="lb % 17 == 0, lb / 17 < 8; first 12 m-block steps of the first tile",
                                layout=[KT_S, KT_M, KT_E, KT_HDR]), "records": out}, f)


def decode(rec):
    buf = rec["buf"]
    var = 0 if rec["cfg"][0] == 128 else 1
    samples = []
    for s in range(KT_S):
        h = buf[KT_EVENTS + (var * KT_S + s) * KT_HDR: KT_EVENTS + (var * KT_S + s + 1) * KT_HDR]
        if h[0] != 1:
            continue
        steps = []
        for j in range(KT_M):
            acts = []
            for actor in range(3):
                base = (((var * KT_S + s) * KT_M + j) * 3 + actor) * KT_E
                acts.append(buf[base: base + KT_E])
            if acts[0][0] == 0:
                break
            steps.append(acts)
        samples.append({"n_block": h[1], "bidh": h[2], "bidb": h[3], "smid": h[4], "m_block_min": h[5], "lb": h[6],
                        "t_start": h[7], "t_kv_issued": h[8], "steps": steps})
    return samples


def overlap(a, b):
    """Total length of the intersection of two interval lists."""
    t = 0
    for (x0, x1) in a:
        for (y0, y1) in b:
            t += max(0, min(x1, y1) - max(x0, y0))
    return t


def windows(ev):
    """(vector intervals, wgmma-in-flight intervals) of one warpgroup step from its events."""
    vec = [(ev[5], ev[6]), (ev[7], ev[9]), (ev[10], ev[23]), (ev[17], ev[18])]
    mma = [(ev[21], ev[7]), (ev[14], ev[17])]
    return [(x, y) for x, y in vec if y > x], [(x, y) for x, y in mma if y > x]


def summarize(path):
    d = json.load(open(path))
    lines = [f"# Pass-1 sparse timeline ({d['meta']['sample']})", "",
             f"Build: {d['meta']['library']} (sha256 {d['meta']['library_sha256'][:16]}), route {d['meta']['route']}.", ""]
    agg = {}
    for rec in d["records"]:
        samples = decode(rec)
        cfg = rec["cfg"]
        lines.append(f"## D{cfg[0]} B{cfg[1]} H{cfg[2]}/{cfg[3]} N{cfg[4]} {cfg[5]} ({rec['mode']}): {len(samples)} sampled CTAs")
        lines.append("")
        st = {k: [] for k in ("step_wg0", "step_wg1", "offset", "region_Q_wait", "region_dO_wait", "region_dOt_wait", "region_Qt_wait", "region_S_retire", "region_dP_retire",
                              "region_dV_retire_release", "region_dK_retire_release_state", "region_psi_barrier", "vec_regions_wg0", "vec_regions_wg0_in_wg1_outstanding", "vec_regions_wg1",
                              "vec_regions_wg1_in_wg0_outstanding", "p_region_dO_acquire", "p_region_dOt_acquire_bracket", "p_region_Q_acquire_bracket", "p_region_Qt_acquire_bracket", "q_issue_lead")}
        for smp in samples:
            steps = smp["steps"]
            for j in range(1, len(steps) - 1):          # skip the first (prologue) and last (tail) sampled step
                w0, w1, pr = steps[j]
                n0, n1 = steps[j + 1][0], steps[j + 1][1]
                if not (w0[0] and w1[0] and n0[0] and n1[0]):
                    continue
                st["step_wg0"].append(n0[0] - w0[0]); st["step_wg1"].append(n1[0] - w1[0])
                st["offset"].append(w1[0] - w0[0])
                for ev in (w0, w1):
                    st["region_Q_wait"].append(ev[3] - ev[0]); st["region_dO_wait"].append(ev[22] - ev[21])
                    st["region_dOt_wait"].append(ev[24] - ev[23]); st["region_Qt_wait"].append(ev[26] - ev[25])
                    st["region_S_retire"].append(ev[5] - ev[4]); st["region_dP_retire"].append(ev[7] - ev[6])
                    st["region_dV_retire_release"].append(ev[16] - ev[15]); st["region_dK_retire_release_state"].append(ev[17] - ev[16])
                    st["region_psi_barrier"].append(ev[10] - ev[9])
                v0, m0 = windows(w0); v1, m1 = windows(w1)
                st["vec_regions_wg0"].append(sum(y - x for x, y in v0)); st["vec_regions_wg0_in_wg1_outstanding"].append(overlap(v0, m1))
                st["vec_regions_wg1"].append(sum(y - x for x, y in v1)); st["vec_regions_wg1_in_wg0_outstanding"].append(overlap(v1, m0))
                if pr[1]:
                    st["p_region_dO_acquire"].append(pr[2] - pr[1]); st["p_region_dOt_acquire_bracket"].append(pr[4] - pr[3])
                    st["p_region_Q_acquire_bracket"].append(pr[7] - pr[5]); st["p_region_Qt_acquire_bracket"].append(pr[9] - pr[8])
                    # how long before the consumers asked for Q(m+1) the producer had issued its load
                    st["q_issue_lead"].append(steps[j + 1][0][0] - pr[8])
        def med(k):
            return statistics.median(st[k]) if st[k] else float("nan")
        n = len(st["step_wg0"])
        vec_share0 = sum(st["vec_regions_wg0_in_wg1_outstanding"]) / max(1, sum(st["vec_regions_wg0"]))
        vec_share1 = sum(st["vec_regions_wg1_in_wg0_outstanding"]) / max(1, sum(st["vec_regions_wg1"]))
        lines += [f"{n} interior steps; medians in cycles (SM clock):", "",
                  "| quantity | median |", "|---|---:|"]
        for k in st:
            lines.append(f"| {k} | {med(k):.0f} |")
        lines += ["", f"Share of WG0's instrumented vector regions inside WG1's software-outstanding wgmma windows (issue to "
                      f"observed retire / release): {100 * vec_share0:.0f}%; WG1 in WG0's: {100 * vec_share1:.0f}%.", ""]
        agg[f"D{cfg[0]}_{cfg[5]}"] = {k: med(k) for k in st} | {"vec_under_other_mma": [vec_share0, vec_share1], "steps": n}
    return "\n".join(lines), agg


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("capture"); c.add_argument("--root", required=True); c.add_argument("--out", required=True)
    s = sp.add_parser("summarize"); s.add_argument("trace"); s.add_argument("--json-out", default=None)
    a = ap.parse_args()
    if a.cmd == "capture":
        capture(a)
    else:
        text, agg = summarize(a.trace)
        print(text)
        if a.json_out:
            json.dump(agg, open(a.json_out, "w"), indent=1)


if __name__ == "__main__":
    main()
