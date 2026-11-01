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


def resolve_build(root):
    """The package directory and the one compiled library inside a build root (no DM_SO: the root decides)."""
    import glob
    pkg = os.path.realpath(os.path.join(root, "delta_matching"))
    libs = sorted(glob.glob(os.path.join(pkg, "_C*.so")))
    if len(libs) != 1:
        raise SystemExit(f"build root {root}: expected exactly one delta_matching/_C*.so, found {libs}")
    return pkg, os.path.realpath(libs[0]), K.sha256(libs[0])


def child_env(root, lib):
    """The environment of a measurement child: PYTHONPATH = the root, DM_SO = the root's own library (an inherited
    DM_SO or slowdown hook never leaks in)."""
    env = dict(os.environ, PYTHONPATH=os.path.abspath(root), DM_SO=lib)
    env.pop("KDA_BENCH_SLOWDOWN_CYCLES", None)
    return env


def validate_run(d, build, expected, label):
    """A child's JSON must come from the selected build and contain exactly the expected configurations, once each,
    with finite positive timings."""
    import math
    pkg, lib, sha = build
    m = d["meta"]
    if m.get("package") != pkg or m.get("library") != lib or m.get("library_sha256") != sha:
        raise SystemExit(f"{label}: loaded package / library {m.get('package')} {m.get('library')} "
                         f"{str(m.get('library_sha256'))[:12]} is not the selected build {pkg} {lib} {sha[:12]}")
    keys = [key(r) for r in d["records"]]
    if len(keys) != len(set(keys)):
        raise SystemExit(f"{label}: duplicate configurations")
    if set(keys) != set(expected):
        miss, extra = set(expected) - set(keys), set(keys) - set(expected)
        raise SystemExit(f"{label}: configurations differ from the expected grid (missing {sorted(miss)[:3]}, "
                         f"extra {sorted(extra)[:3]})")
    for r in d["records"]:
        vals = [r["e2e_fwd_ms"], r["e2e_fwdbwd_ms"]] + [r["ms"][c] for c in K.CATS]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in vals) or \
                r["e2e_fwd_ms"] <= 0 or r["e2e_fwdbwd_ms"] <= 0:
            raise SystemExit(f"{label}: non-finite or non-positive timing in {key(r)}")


def median_records(runs):
    """Per-config median over repetitions of every timing field (kernels.py record format); every repetition must
    hold the same configurations (validate_run)."""
    by = {}
    for recs in runs:
        for r in recs:
            by.setdefault(key(r), []).append(r)
    n = len(runs)
    out = []
    for k_, rs in by.items():
        if len(rs) != n:
            raise SystemExit(f"configuration {k_} measured in {len(rs)} of {n} repetitions")
        m = dict(rs[0])
        m["e2e_fwd_ms"] = statistics.median(r["e2e_fwd_ms"] for r in rs)
        m["e2e_fwdbwd_ms"] = statistics.median(r["e2e_fwdbwd_ms"] for r in rs)
        m["ms"] = {c: statistics.median(r["ms"][c] for r in rs) for c in K.CATS}
        m["reps"] = len(rs)
        out.append(m)
    return out


def spread(v):
    med = statistics.median(v)
    return 100 * (max(v) - min(v)) / med if med > 0 else 0.0


def analyze(raw_a, raw_b, subset=None, gate_pct=1.0):
    """Per-kernel sums (all configs and the affected subset), per-shape fwd+bwd repetition values / medians / spreads,
    and the shapes whose median fwd+bwd change exceeds gate_pct (flagged beyond_spread when every B repetition is
    slower than every A repetition by more than gate_pct of the A median)."""
    med_a, med_b = median_records(raw_a), median_records(raw_b)
    if {key(r) for r in med_a} != {key(r) for r in med_b}:
        raise SystemExit("A and B measured different shape sets")

    def sel(recs):
        return [r for r in recs if subset is None or subset(r)]

    def kernel_table(ra_runs, rb_runs, ma, mb):
        out = {}
        for c in K.CATS + ("e2e_fwd", "e2e_fwdbwd"):
            def tot(recs, x=c):
                return sum(r[f"{x}_ms"] if x.startswith("e2e") else r["ms"][x] for r in recs)
            sa, sb = tot(ma), tot(mb)
            if sa <= 0 and sb <= 0:
                continue
            ra = [tot(rs) for rs in ra_runs]; rb = [tot(rs) for rs in rb_runs]
            out[c] = {"A": sa, "B": sb, "change_pct": 100 * (sb / sa - 1) if sa > 0 else float("inf"),
                      "spread_A_pct": spread(ra) if sa > 0 else 0.0, "spread_B_pct": spread(rb) if sb > 0 else 0.0,
                      "reps_A": ra, "reps_B": rb}
        return out
    res = {"all": kernel_table(raw_a, raw_b, med_a, med_b)}
    if subset is not None:
        res["subset"] = kernel_table([sel(x) for x in raw_a], [sel(x) for x in raw_b], sel(med_a), sel(med_b))
    shapes = []
    by_a = {}
    for recs in raw_a:
        for r in recs:
            by_a.setdefault(key(r), []).append(r["e2e_fwdbwd_ms"])
    by_b = {}
    for recs in raw_b:
        for r in recs:
            by_b.setdefault(key(r), []).append(r["e2e_fwdbwd_ms"])
    for k_ in sorted(by_a):
        va, vb = by_a[k_], by_b[k_]
        ma, mb = statistics.median(va), statistics.median(vb)
        ch = 100 * (mb / ma - 1)
        shapes.append({"key": list(k_), "A": va, "B": vb, "median_A": ma, "median_B": mb, "change_pct": ch,
                       "spread_A_pct": spread(va), "spread_B_pct": spread(vb),
                       "over_gate": ch > gate_pct, "beyond_spread": ch > gate_pct and min(vb) > max(va) * (1 + gate_pct / 100)})
    res["shapes"] = shapes
    res["over_gate"] = [s for s in shapes if s["over_gate"]]
    res["worst_shape_change_pct"] = max(s["change_pct"] for s in shapes)
    return res, med_a, med_b


def fmt_kernels(tab, title):
    lines = [f"### {title}", "", "| kernel | A ms (sum of medians) | B ms | change | A rep spread | B rep spread |",
             "|---|---:|---:|---:|---:|---:|"]
    for c, v in tab.items():
        lines.append(f"| {c} | {v['A']:.3f} | {v['B']:.3f} | {v['change_pct']:+.2f}% | {v['spread_A_pct']:.2f}% | "
                     f"{v['spread_B_pct']:.2f}% |")
    return lines


SUBSETS = {"D128": lambda r: r["D"] == 128, "D256": lambda r: r["D"] == 256}


def run(a):
    if a.reps < 3 and not a.smoke:
        raise SystemExit("an acceptance A/B needs --reps >= 3 (use --smoke for a labelled, non-acceptance run)")
    os.makedirs(a.out_dir, exist_ok=True)
    roots = {"A": os.path.abspath(a.a), "B": os.path.abspath(a.b)}
    builds = {x: resolve_build(roots[x]) for x in roots}
    one = (a.D, a.B, a.H, a.Hk, a.N) if a.grid == "one" else None
    expected = K.expected_configs(a.grid, a.modes, one)
    raw = {"A": [], "B": []}
    meta = {}
    for rep in range(a.reps):
        order = ("A", "B") if rep % 2 == 0 else ("B", "A")
        for x in order:
            out = os.path.join(a.out_dir, f"{x}_{rep}.json")
            env = child_env(roots[x], builds[x][1])
            if x == "B" and a.slow_b:
                env["KDA_BENCH_SLOWDOWN_CYCLES"] = str(a.slow_b)
            cmd = [sys.executable, os.path.join(HERE, "kernels.py"), "--grid", a.grid, "--out", out,
                   "--iters", str(a.iters), "--sustain", str(a.sustain)] + (["--modes", a.modes] if a.modes else [])
            if a.grid == "one":
                cmd += ["--D", str(a.D), "--N", str(a.N), "--B", str(a.B), "--H", str(a.H), "--Hk", str(a.Hk)]
            print(f"[ab] rep {rep} {x}: {roots[x]} ({builds[x][2][:12]})", flush=True)
            p = subprocess.run(cmd, env=env, capture_output=True, text=True)
            if p.returncode != 0:
                sys.stderr.write(p.stdout[-4000:] + p.stderr[-4000:])
                raise SystemExit(f"kernels.py failed for {x} rep {rep}")
            with open(out) as f:
                d = json.load(f)
            validate_run(d, builds[x], expected, f"{x} rep {rep}")
            raw[x].append(d["records"])
            meta.setdefault(x, []).append(d["meta"])
    subset = SUBSETS.get(a.subset) if a.subset else None
    res, med_a, med_b = analyze(raw["A"], raw["B"], subset, a.gate)
    for x, recs in (("A", med_a), ("B", med_b)):
        clk = statistics.median(m["sustained_clock_mhz"] for m in meta[x])
        for r in recs:
            r["sol"] = K.model_sol(r, clk)
        with open(os.path.join(a.out_dir, f"{x}_median.json"), "w") as f:
            json.dump({"meta": dict(meta[x][0], sustained_clock_mhz=clk, ab_root=roots[x], reps=a.reps),
                       "records": recs}, f, indent=1, default=str)
    tag = "SMOKE (not eligible for acceptance)" if a.smoke else "acceptance run"
    lines = [f"# A/B ({tag}): grid {a.grid}, {a.reps} reps alternating A/B order, median per config", "",
             f"- A = {roots['A']} (library sha256 {builds['A'][2][:16]})",
             f"- B = {roots['B']} (library sha256 {builds['B'][2][:16]})", ""]
    lines += fmt_kernels(res["all"], "Per-kernel sums, all configurations")
    if subset is not None:
        lines += [""] + fmt_kernels(res["subset"], f"Per-kernel sums, affected subset {a.subset}")
    lines += ["", f"### Shapes over the {a.gate:.1f}% fwd+bwd gate", ""]
    if res["over_gate"]:
        for s_ in res["over_gate"]:
            lines.append(f"- {tuple(s_['key'])}: {s_['change_pct']:+.2f}% (A reps {[round(v, 3) for v in s_['A']]}, "
                         f"B reps {[round(v, 3) for v in s_['B']]}; spreads {s_['spread_A_pct']:.2f}% / "
                         f"{s_['spread_B_pct']:.2f}%){' BEYOND SPREAD' if s_['beyond_spread'] else ' (within spread)'}")
    else:
        lines.append("- none")
    lines += ["", "### Per shape (fwd+bwd, ms)", "",
              "| shape | attn | mode | A median | B median | change | A spread | B spread |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for s_ in res["shapes"]:
        D, B, H, Hk, N, attn, mode = s_["key"]
        lines.append(f"| D{D} B{B} H{H}/{Hk} N{N} | {attn} | {mode} | {s_['median_A']:.3f} | {s_['median_B']:.3f} | "
                     f"{s_['change_pct']:+.2f}% | {s_['spread_A_pct']:.2f}% | {s_['spread_B_pct']:.2f}% |")
    lines += ["", f"Worst single-shape fwd+bwd change (B vs A): {res['worst_shape_change_pct']:+.2f}%"]
    with open(os.path.join(a.out_dir, "ab.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(a.out_dir, "ab.json"), "w") as f:
        json.dump({"tag": tag, "roots": roots, "builds": builds, "grid": a.grid, "reps": a.reps, "subset": a.subset,
                   "gate_pct": a.gate, **res}, f, indent=1)
    print("\n".join(lines[:lines.index("### Per shape (fwd+bwd, ms)")]))


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
    r.add_argument("--smoke", action="store_true", help="allow --reps < 3; the report is labelled non-acceptance")
    r.add_argument("--subset", choices=sorted(SUBSETS), default=None, help="also report per-kernel sums on this subset")
    r.add_argument("--gate", type=float, default=1.0, help="per-shape fwd+bwd regression gate, percent")
    r.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    freeze(a) if a.cmd == "freeze" else run(a)


if __name__ == "__main__":
    main()
