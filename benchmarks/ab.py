"""Paired A/B timing of two delta_matching builds in one job, and freezing a build for later comparisons.

    # a self-contained copy of a revision's Python package plus its compiled library
    python benchmarks/ab.py freeze --rev <git rev> --so <built _C.so> --out <dir>      # -> <dir>/delta_matching

    # alternate the two builds (A B, B A, A B, ...) over --reps repetitions of benchmarks/kernels.py, each a fresh process
    # with PYTHONPATH set to the build's root; aggregate per-kernel medians and the spread across repetitions
    python benchmarks/ab.py run --a <root A> --b <root B> --grid main --reps 3 --out-dir <dir>

`run` writes <dir>/{A,B}_<rep>.json (raw kernels.py records), <dir>/{A,B}_median.json (per-config medians, the
kernels.py format, so summarize.py --base works on them) and <dir>/ab.md (per-kernel sums: medians, change, spread;
per-shape end-to-end changes).  The same benchmarks/kernels.py (this file's directory) measures both builds.
`--slow-b CYCLES` adds a spin kernel to every B step (KDA_BENCH_SLOWDOWN_CYCLES): a self-test that a slower B shows up.
"""
import argparse, json, os, shutil, statistics, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kernels as K   # noqa: E402


def freeze(a):
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    repo = os.path.dirname(HERE)
    subprocess.run(f"git -C {repo} archive {a.rev} delta_matching | tar -x -C {out}", shell=True, check=True)
    shutil.copyfile(a.so, os.path.join(out, "delta_matching", "_C.so"))
    info = os.path.join(os.path.dirname(os.path.realpath(a.so)), "_build_info.py")
    if os.path.isfile(info):
        shutil.copyfile(info, os.path.join(out, "delta_matching", "_build_info.py"))
    rev = subprocess.run(["git", "-C", repo, "rev-parse", a.rev], capture_output=True, text=True).stdout.strip()
    with open(os.path.join(out, "FROZEN"), "w") as f:
        f.write(f"rev {rev}\nso {os.path.realpath(a.so)}\n")
    print(f"froze {rev} + {a.so} -> {out}")


def key(r):
    return (r["D"], r["B"], r["H"], r["Hk"], r["N"], r["attn"], r["mode"])


def median_records(runs):
    """Per-config median over repetitions of every timing field (kernels.py record format)."""
    by = {}
    for recs in runs:
        for r in recs:
            by.setdefault(key(r), []).append(r)
    out = []
    for k_, rs in by.items():
        m = dict(rs[0])
        m["e2e_fwd_ms"] = statistics.median(r["e2e_fwd_ms"] for r in rs)
        m["e2e_fwdbwd_ms"] = statistics.median(r["e2e_fwdbwd_ms"] for r in rs)
        m["ms"] = {c: statistics.median(r["ms"][c] for r in rs) for c in K.CATS}
        m["reps"] = len(rs)
        out.append(m)
    return out


def run(a):
    os.makedirs(a.out_dir, exist_ok=True)
    roots = {"A": os.path.abspath(a.a), "B": os.path.abspath(a.b)}
    raw = {"A": [], "B": []}
    meta = {}
    for rep in range(a.reps):
        order = ("A", "B") if rep % 2 == 0 else ("B", "A")
        for x in order:
            out = os.path.join(a.out_dir, f"{x}_{rep}.json")
            env = dict(os.environ, PYTHONPATH=roots[x])
            env.pop("KDA_BENCH_SLOWDOWN_CYCLES", None)
            if x == "B" and a.slow_b:
                env["KDA_BENCH_SLOWDOWN_CYCLES"] = str(a.slow_b)
            cmd = [sys.executable, os.path.join(HERE, "kernels.py"), "--grid", a.grid, "--out", out,
                   "--iters", str(a.iters), "--sustain", str(a.sustain)] + (["--modes", a.modes] if a.modes else [])
            if a.grid == "one":
                cmd += ["--D", str(a.D), "--N", str(a.N), "--B", str(a.B), "--H", str(a.H), "--Hk", str(a.Hk)]
            print(f"[ab] rep {rep} {x}: PYTHONPATH={roots[x]}", flush=True)
            p = subprocess.run(cmd, env=env, capture_output=True, text=True)
            if p.returncode != 0:
                sys.stderr.write(p.stdout[-4000:] + p.stderr[-4000:])
                raise SystemExit(f"kernels.py failed for {x} rep {rep}")
            with open(out) as f:
                d = json.load(f)
            raw[x].append(d["records"])
            meta.setdefault(x, []).append(d["meta"])
    med = {}
    for x in ("A", "B"):
        clk = statistics.median(m["sustained_clock_mhz"] for m in meta[x])
        recs = median_records(raw[x])
        for r in recs:
            r["sol"] = K.model_sol(r, clk)
        med[x] = recs
        with open(os.path.join(a.out_dir, f"{x}_median.json"), "w") as f:
            json.dump({"meta": dict(meta[x][0], sustained_clock_mhz=clk, ab_root=roots[x], reps=a.reps),
                       "records": recs}, f, indent=1, default=str)
    keys_a, keys_b = {key(r) for r in med["A"]}, {key(r) for r in med["B"]}
    if keys_a != keys_b:
        raise SystemExit("A and B measured different shape sets")
    lines = [f"# A/B: A = {roots['A']}, B = {roots['B']}, grid {a.grid}, {a.reps} reps alternating, median per config", "",
             "| kernel | A ms (sum, median) | B ms | change | A rep spread | B rep spread |", "|---|---:|---:|---:|---:|---:|"]
    summary = {}
    for c in K.CATS + ("e2e_fwd", "e2e_fwdbwd"):
        def tot(recs, x=c):
            return sum(r[f"{x}_ms"] if x.startswith("e2e") else r["ms"][x] for r in recs)
        sa, sb = tot(med["A"]), tot(med["B"])
        if sa <= 0 and sb <= 0:
            continue
        ra = [tot(rs) for rs in raw["A"]]; rb = [tot(rs) for rs in raw["B"]]
        spa = 100 * (max(ra) - min(ra)) / statistics.median(ra) if sa > 0 else 0.0
        spb = 100 * (max(rb) - min(rb)) / statistics.median(rb) if sb > 0 else 0.0
        ch = 100 * (sb / sa - 1) if sa > 0 else float("inf")
        summary[c] = {"A": sa, "B": sb, "change_pct": ch, "spread_A_pct": spa, "spread_B_pct": spb}
        lines.append(f"| {c} | {sa:.3f} | {sb:.3f} | {ch:+.2f}% | {spa:.2f}% | {spb:.2f}% |")
    bm = {key(r): r for r in med["B"]}
    lines += ["", "| shape | attn | mode | A fwd+bwd | B fwd+bwd | change | A fwd | B fwd | change |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    worst = -1e9
    for r in sorted(med["A"], key=key):
        o = bm[key(r)]
        c1 = 100 * (o["e2e_fwdbwd_ms"] / r["e2e_fwdbwd_ms"] - 1)
        c2 = 100 * (o["e2e_fwd_ms"] / r["e2e_fwd_ms"] - 1)
        worst = max(worst, c1)
        lines.append(f"| D{r['D']} B{r['B']} H{r['H']}/{r['Hk']} N{r['N']} | {r['attn']} | {r['mode']} | {r['e2e_fwdbwd_ms']:.3f} | "
                     f"{o['e2e_fwdbwd_ms']:.3f} | {c1:+.2f}% | {r['e2e_fwd_ms']:.3f} | {o['e2e_fwd_ms']:.3f} | {c2:+.2f}% |")
    lines += ["", f"Worst single-shape fwd+bwd change (B vs A): {worst:+.2f}%"]
    with open(os.path.join(a.out_dir, "ab.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(a.out_dir, "ab.json"), "w") as f:
        json.dump({"summary": summary, "worst_shape_fwdbwd_change_pct": worst, "roots": roots, "grid": a.grid,
                   "reps": a.reps}, f, indent=1)
    print("\n".join(lines[:4 + len(summary)]))
    print(f"Worst single-shape fwd+bwd change (B vs A): {worst:+.2f}%")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("freeze")
    f.add_argument("--rev", required=True); f.add_argument("--so", required=True); f.add_argument("--out", required=True)
    r = sp.add_parser("run")
    r.add_argument("--a", required=True); r.add_argument("--b", required=True)
    r.add_argument("--grid", choices=("main", "holdout", "one"), default="main")
    r.add_argument("--reps", type=int, default=3); r.add_argument("--iters", type=int, default=10)
    r.add_argument("--sustain", type=float, default=2.0)
    r.add_argument("--modes", default=None)
    r.add_argument("--D", type=int, default=128); r.add_argument("--N", type=int, default=8192)
    r.add_argument("--B", type=int, default=2); r.add_argument("--H", type=int, default=16); r.add_argument("--Hk", type=int, default=4)
    r.add_argument("--slow-b", type=int, default=0)
    r.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    freeze(a) if a.cmd == "freeze" else run(a)


if __name__ == "__main__":
    main()
