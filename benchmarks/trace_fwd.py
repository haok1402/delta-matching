"""Sparse timeline of the FORWARD kernel from a DM_TRACE=1 build (csrc/kda_prof.h, forward section; schema v1).

    python benchmarks/trace_fwd.py capture --root <DM_TRACE build root> --out fwd_trace.json   # one forward per config
    python benchmarks/trace_fwd.py time --root <any build root> --out t.json                   # forward kernel ms
    python benchmarks/trace_fwd.py summarize fwd_trace.json [--json-out s.json]                # validate + decode

Sample: the persistent grid's CTAs 0, 17, ..., 119 (lb % 17 == 0, lb / 17 < 8), the first tile of each, its first 12
n-block steps; actors: consumer warpgroups 0 / 1 (first thread), producer warpgroup thread 0.  Timestamps are clock64
of one SM: compared only within a CTA.  Consumer events (PV-first step): 0 step start (before the V wait), 1 V ready,
2 warp-scheduler barrier passed, 3 PV slab chain issued (with its in-chain promotes), 4 PV retired, 5 last promote +
V released, 6 K ready, 7 QK issued + scheduler arrive, 8 QK retired + K released, 9 mask + descale, 10 grouped max /
scale, 11 O rescale, 12 grouped online softmax, 13 P conversion, 14 v-scale load + step end; 30 / 31 V stage index /
phase.  Producer (main load loop): 1 before load_V (Vt acquire + TMA), 3 before load_K (K acquire + TMA), 6 after
load_K, 11 after copy_Vt_to_V (Vt wait, V acquire, in-smem transpose, commit, barrier, Vt release); 30 / 31 K/V stage
index / phase.  Every quantity is a bracketed region between two probe events (it includes all work and probe overhead
between them); "outstanding" wgmma windows run from issue to the observed wait, a software bound, not tensor-pipe
activity.
"""
import argparse, json, os, statistics, sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kernels as K  # noqa: E402
import profile_pass1 as PP  # noqa: E402

SCHEMA = 1
KF_S, KF_M, KF_E, KF_HDR = 8, 12, 32, 16
KF_EVENTS = KF_S * KF_M * 3 * KF_E
CONS_EVENTS = list(range(15))
PROD_EVENTS = [1, 3, 6, 11]
CONFIGS = [(128, 4, 16, 16, 8192, "causal"), (128, 4, 16, 16, 8192, "full"), (128, 4, 16, 4, 8192, "causal"),
           (256, 4, 8, 8, 8192, "causal"), (256, 4, 8, 8, 8192, "full"), (256, 4, 8, 2, 8192, "causal")]


def inputs(cfg):
    D, B, H, Hk, N, attn = cfg
    g = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16, generator=g)
    k = torch.randn(B, N, Hk, D, device="cuda", dtype=torch.bfloat16, generator=g)
    v = torch.randn(B, N, Hk, D, device="cuda", dtype=torch.bfloat16, generator=g)
    return q, k, v, attn == "causal"


def capture(a):
    meta = PP.select_build(a.root)
    import delta_matching as dm
    ops = torch.ops.delta_matching
    if not hasattr(ops, "prof_fwd_buffer"):
        raise SystemExit("not a DM_TRACE=1 build with the forward trace (no prof_fwd_buffer op)")
    out = []
    for cfg in CONFIGS:
        q, k, v, causal = inputs(cfg)
        buf = ops.prof_fwd_buffer(cfg[0])
        if buf.numel() != KF_EVENTS + KF_S * KF_HDR:
            raise SystemExit(f"forward trace buffer has {buf.numel()} words, schema v{SCHEMA} expects {KF_EVENTS + KF_S * KF_HDR}")
        with torch.no_grad():
            mode = "matched" if causal else "stale"
            dm.dm_attn_func(q, k, v, causal=causal, mode=mode)          # warm-up
            torch.cuda.synchronize()
            buf.zero_(); torch.cuda.synchronize()
            dm.dm_attn_func(q, k, v, causal=causal, mode=mode)          # exactly one forward launch
            torch.cuda.synchronize()
        out.append({"cfg": list(cfg), "buf": buf.cpu().tolist()})
        print(f"captured {cfg}", flush=True)
    with open(a.out, "w") as f:
        json.dump({"meta": dict(meta, schema=SCHEMA, layout=[KF_S, KF_M, KF_E, KF_HDR],
                                sample="lb % 17 == 0, lb / 17 < 8; first 12 n-block steps of the first tile"),
                   "records": out}, f)


def time_fwd(a):
    meta = PP.select_build(a.root)
    import delta_matching as dm
    recs = []
    for cfg in CONFIGS:
        q, k, v, causal = inputs(cfg)
        with torch.no_grad():
            mode = "matched" if causal else "stale"
            names = K.kernel_times(lambda: dm.dm_attn_func(q, k, v, causal=causal, mode=mode), 10)
        recs.append({"cfg": list(cfg), "fwd_ms": sum(ms for n, ms in names.items() if K.classify(n) == "fwd")})
        print(f"{cfg}: forward kernel {recs[-1]['fwd_ms']:.3f} ms", flush=True)
    with open(a.out, "w") as f:
        json.dump({"meta": meta, "records": recs}, f, indent=1)


def decode(rec):
    """Validated samples: header present, every required event present and in order for every recorded step."""
    buf = rec["buf"]
    if len(buf) != KF_EVENTS + KF_S * KF_HDR:
        raise ValueError(f"buffer length {len(buf)} != schema v{SCHEMA}")
    samples = []
    for s in range(KF_S):
        h = buf[KF_EVENTS + s * KF_HDR: KF_EVENTS + (s + 1) * KF_HDR]
        if h[0] != 1:
            raise ValueError(f"sample {s}: missing header")
        steps = []
        for j in range(KF_M):
            acts = [buf[((s * KF_M + j) * 3 + a) * KF_E: ((s * KF_M + j) * 3 + a + 1) * KF_E] for a in range(3)]
            if all(x == 0 for a in acts for x in a):
                break
            for a in (0, 1):
                ev = [acts[a][e] for e in CONS_EVENTS]
                if any(x == 0 for x in ev) or any(ev[i] > ev[i + 1] for i in range(len(ev) - 1)):
                    raise ValueError(f"sample {s} step {j} warpgroup {a}: missing or out-of-order consumer events {ev}")
                if not (0 <= acts[a][30] < 8 and acts[a][31] in (0, 1)):
                    raise ValueError(f"sample {s} step {j} warpgroup {a}: invalid stage / phase")
            steps.append(acts)
        if len(steps) < 3:
            raise ValueError(f"sample {s}: only {len(steps)} steps recorded")
        samples.append({"m_block": h[1], "bidh": h[2], "bidb": h[3], "smid": h[4], "n_block_max": h[5],
                        "n_block_min": h[6], "steps": steps})
    if not samples:
        raise ValueError("no samples")
    return samples


def overlap(a, b):
    return sum(max(0, min(x1, y1) - max(x0, y0)) for x0, x1 in a for y0, y1 in b)


REGIONS = [("V wait", 0, 1), ("scheduler barrier", 1, 2), ("PV chain issue + in-chain promotes", 2, 3),
           ("PV retire", 3, 4), ("last promote + V release", 4, 5), ("K wait", 5, 6), ("QK issue + arrive", 6, 7),
           ("QK retire + K release", 7, 8), ("mask + descale", 8, 9), ("grouped max / scale", 9, 10),
           ("O rescale", 10, 11), ("grouped online softmax", 11, 12), ("P conversion", 12, 13),
           ("v-scale load + step end", 13, 14)]


def summarize(path):
    d = json.load(open(path))
    if d["meta"].get("schema") != SCHEMA:
        raise SystemExit(f"trace schema {d['meta'].get('schema')} != {SCHEMA}")
    lines = [f"# Forward sparse timeline ({d['meta']['sample']})", "",
             f"Build {d['meta']['library']} (sha256 {d['meta']['library_sha256'][:16]}). Bracketed regions of the "
             "instrumented kernel; medians over interior sampled steps (the first and last recorded step of each sample "
             "excluded); cycles.", ""]
    agg = {}
    for rec in d["records"]:
        cfg = rec["cfg"]
        samples = decode(rec)
        st = {name: [] for name, _, _ in REGIONS}
        st.update(step=[], wg_offset=[], vec=[], vec_in_other_outstanding=[], p_load_V=[], p_load_K=[], p_transpose=[],
                  p_step=[])
        for smp in samples:
            steps = smp["steps"]
            for j in range(1, len(steps) - 1):
                for w in (0, 1):
                    ev, nx, ot = steps[j][w], steps[j + 1][w], steps[j][1 - w]
                    st["step"].append(nx[0] - ev[0])
                    for name, a_, b_ in REGIONS:
                        st[name].append(ev[b_] - ev[a_])
                    vec = [(ev[8], ev[14])]
                    mma = [(ot[2], ot[4]), (ot[6], ot[8])]
                    st["vec"].append(ev[14] - ev[8]); st["vec_in_other_outstanding"].append(overlap(vec, mma))
                st["wg_offset"].append(steps[j][1][0] - steps[j][0][0])
                pr, pn = steps[j][2], steps[j + 1][2]
                if pr[1] and pr[11] and pn[1]:
                    st["p_load_V"].append(pr[3] - pr[1]); st["p_load_K"].append(pr[6] - pr[3])
                    st["p_transpose"].append(pr[11] - pr[6]); st["p_step"].append(pn[1] - pr[1])
        med = {k: (statistics.median(v) if v else float("nan")) for k, v in st.items()}
        share = sum(st["vec_in_other_outstanding"]) / max(1, sum(st["vec"]))
        lines += [f"## D{cfg[0]} B{cfg[1]} H{cfg[2]}/{cfg[3]} N{cfg[4]} {cfg[5]}: {len(samples)} samples, "
                  f"{len(st['step'])} warpgroup-steps", "", "| region | median cycles | share of step |", "|---|---:|---:|"]
        for name, _, _ in REGIONS:
            lines.append(f"| {name} | {med[name]:.0f} | {100 * med[name] / med['step']:.1f}% |")
        lines += [f"| step (warpgroup) | {med['step']:.0f} | |", "",
                  f"WG1 - WG0 step-start offset: {med['wg_offset']:.0f} cycles; share of each warpgroup's vector regions "
                  f"(mask .. P conversion) inside the other warpgroup's outstanding PV / QK windows: {100 * share:.0f}%.",
                  f"Producer (thread 0): load_V bracket {med['p_load_V']:.0f}, load_K bracket {med['p_load_K']:.0f}, V "
                  f"transpose bracket {med['p_transpose']:.0f}, producer step {med['p_step']:.0f} cycles.", ""]
        agg[f"D{cfg[0]}_{cfg[5]}_Hk{cfg[3]}"] = dict(med, vec_share_in_other_outstanding=share)
    return "\n".join(lines), agg


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("capture"); c.add_argument("--root", required=True); c.add_argument("--out", required=True)
    t = sp.add_parser("time"); t.add_argument("--root", required=True); t.add_argument("--out", required=True)
    s = sp.add_parser("summarize"); s.add_argument("trace"); s.add_argument("--json-out", default=None)
    a = ap.parse_args()
    if a.cmd == "capture":
        capture(a)
    elif a.cmd == "time":
        time_fwd(a)
    else:
        try:
            text, agg = summarize(a.trace)
        except ValueError as e:
            raise SystemExit(f"invalid trace: {e}")
        print(text)
        if a.json_out:
            json.dump(agg, open(a.json_out, "w"), indent=1)


if __name__ == "__main__":
    main()
