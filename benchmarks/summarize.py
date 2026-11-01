"""Summarize benchmarks/kernels.py JSON output as markdown: the per-kernel waste ranking against speed of light, the
per-shape tables (ms, TFLOPS, % of SOL), FlashAttention-3 comparisons, and (with --base) a before / after comparison.

    python benchmarks/summarize.py results/baseline_main.json [--holdout results/baseline_holdout.json] > results/BASELINE.md
    python benchmarks/summarize.py new_main.json --base results/baseline_main.json --holdout new_hold.json --base-holdout base_hold.json
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kernels as K   # noqa: E402  (the accounting model; does not import delta_matching)

CATS = ("fwd", "pass1", "pass0", "pass2", "prod_fwd", "prod_bwd", "twin", "glue")
LABEL = {"fwd": "forward (fwd_fp8)", "pass0": "pass 0 (matched delta)", "pass1": "pass 1 (dK, dV, dS')",
         "pass2": "pass 2 (dQ)", "prod_fwd": "forward producers (Triton Q/K/V quant)", "prod_bwd": "dO producer (Triton)",
         "twin": "twins (Q, dO transposes)", "glue": "torch glue (layout copies, GQA fold)"}


REMODEL = False


def load(p):
    with open(p) as f:
        d = json.load(f)
    if REMODEL:   # recompute the modeled SOL with the current accounting (kernels.accounting) at the run's clock
        clk = d["meta"].get("sustained_clock_mhz", 1830.0)
        for r in d["records"]:
            r["sol"] = K.model_sol(r, clk)
    return d


def key(r):
    return (r["D"], r["B"], r["H"], r["Hk"], r["N"], r["attn"], r["mode"])


def shape(r):
    g = "MHA" if r["H"] == r["Hk"] else f"GQA{r['H'] // r['Hk']}"
    return f"D{r['D']} B{r['B']} H{r['H']}/{r['Hk']} ({g}) N{r['N']}"


def ranking(recs):
    rows = []
    for c in CATS:
        t = sum(r["ms"][c] for r in recs)
        s = sum(r["sol"][c]["sol_ms"] for r in recs)
        if t <= 0:
            continue
        rows.append((t - s, c, t, s))
    rows.sort(reverse=True)
    out = ["| rank | kernel | measured ms (sum over grid) | modeled SOL ms | wasted vs model ms | % of modeled SOL | share of fwd+bwd |",
           "|---:|---|---:|---:|---:|---:|---:|"]
    tot = sum(r["e2e_fwdbwd_ms"] for r in recs)
    for i, (w, c, t, s) in enumerate(rows, 1):
        out.append(f"| {i} | {LABEL[c]} | {t:.2f} | {s:.2f} | {w:.2f} | {100 * s / t:.0f}% | {100 * t / tot:.0f}% |")
    return "\n".join(out)


def per_shape(recs, attn, mode, fa3=True):
    sel = [r for r in recs if r["attn"] == attn and r["mode"] == mode]
    cats = [c for c in CATS if any(r["ms"][c] > 0 for r in sel)]
    head = "| shape | e2e fwd | e2e fwd+bwd | " + " | ".join(c for c in cats) + " |"
    if fa3:
        head += " FA3 fp8 fwd | FA3 bf16 fwd | FA3 bf16 bwd | FA3 bf16 fwd+bwd |"
    out = [head, "|" + "---|" * (head.count("|") - 1)]
    for r in sel:
        cells = []
        for c in cats:
            m, so = r["ms"][c], r["sol"][c]
            if so["flops"] > 0:
                cells.append(f"{m:.3f} ms / {so['tflops']:.0f} TF / {so['pct_sol']:.0f}%")
            else:
                cells.append(f"{m:.3f} ms / {so['pct_sol']:.0f}%")
        line = f"| {shape(r)} | {r['e2e_fwd_ms']:.3f} | {r['e2e_fwdbwd_ms']:.3f} | " + " | ".join(cells) + " |"
        if fa3:
            f = r.get("fa3") or {}
            line += " " + " | ".join(f"{f[k]:.3f}" if isinstance(f.get(k), (int, float)) else "-" for k in
                                     ("fp8_fwd_kernel_ms", "bf16_fwd_kernel_ms", "bf16_bwd_kernel_ms", "bf16_fwdbwd_ms")) + " |"
        out.append(line)
    return "\n".join(out)


def compare(new, base, title):
    b = {key(r): r for r in base}
    out = [f"### {title}", "", "| shape | attn | mode | fwd+bwd before | after | change | " +
           " | ".join(f"{c} before / after" for c in CATS) + " |", "|" + "---|" * (6 + len(CATS))]
    tn = tb = 0.0
    worst = 0.0
    missing = sorted(set(b) ^ {key(r) for r in new})
    if missing:
        raise SystemExit(f"summarize: {len(missing)} shape(s) present in only one of the two files, e.g. {missing[0]}")
    for r in new:
        o = b[key(r)]
        ch = 100 * (r["e2e_fwdbwd_ms"] / o["e2e_fwdbwd_ms"] - 1)
        worst = max(worst, ch)
        tn += r["e2e_fwdbwd_ms"]; tb += o["e2e_fwdbwd_ms"]
        cells = [f"{o['ms'][c]:.3f} / {r['ms'][c]:.3f}" if (o["ms"][c] or r["ms"][c]) else "-" for c in CATS]
        out.append(f"| {shape(r)} | {r['attn']} | {r['mode']} | {o['e2e_fwdbwd_ms']:.3f} | {r['e2e_fwdbwd_ms']:.3f} | "
                   f"{ch:+.1f}% | " + " | ".join(cells) + " |")
    out.append("")
    out.append(f"Sum of fwd+bwd: {tb:.2f} -> {tn:.2f} ms ({100 * (tn / tb - 1):+.2f}%); worst single-shape change {worst:+.2f}%.")
    per = []
    for c in CATS:
        sb = sum(b[key(r)]["ms"][c] for r in new if key(r) in b)
        sn = sum(r["ms"][c] for r in new if key(r) in b)
        if sb > 0:
            per.append(f"{c} {sb:.2f} -> {sn:.2f} ms ({100 * (sn / sb - 1):+.1f}%)")
    out.append("Per kernel (sum over shapes): " + "; ".join(per) + ".")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("main"); ap.add_argument("--holdout"); ap.add_argument("--base"); ap.add_argument("--base-holdout")
    ap.add_argument("--remodel", action="store_true", help="recompute the modeled SOL with the current accounting")
    a = ap.parse_args()
    global REMODEL
    REMODEL = a.remodel
    m = load(a.main)
    recs = m["records"]
    meta = m["meta"]
    clk = meta.get("sustained_clock_mhz", 1830)
    print(f"Device {meta['device']}, torch {meta['torch']}, CUDA {meta['cuda']}, kernel {meta['kernel_version'].get('code_sha')}. "
          f"Sustained SM clock {clk:.0f} MHz -> dense fp8 peak {132 * 8192 * clk / 1e6:.0f} TFLOPS (nominal 1979 at 1830 MHz); "
          "HBM3 3.35 TB/s. Cells: measured ms / achieved TFLOPS / % of modeled speed of light (an optimistic reference: unique bytes, perfect overlap).\n")
    if a.base:
        print(compare(recs, load(a.base)["records"], "Main grid: after vs before"))
        if a.holdout and a.base_holdout:
            print()
            print(compare(load(a.holdout)["records"], load(a.base_holdout)["records"], "Held-out shapes: after vs before"))
        return
    print("## Waste ranking over the main grid (all shapes and modes)\n")
    print(ranking(recs))
    for attn, mode in (("causal", "matched"), ("causal", "stale"), ("full", "stale")):
        print(f"\n## {attn} / {mode}\n")
        print(per_shape(recs, attn, mode))
    if a.holdout:
        h = load(a.holdout)["records"]
        print("\n## Held-out shapes (never tuned on)\n")
        print(ranking(h))
        for attn, mode in (("causal", "matched"), ("causal", "stale"), ("full", "stale")):
            print(f"\n### {attn} / {mode}\n")
            print(per_shape(h, attn, mode))


if __name__ == "__main__":
    main()
