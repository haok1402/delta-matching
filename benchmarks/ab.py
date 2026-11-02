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
import argparse, json, math, os, shutil, statistics, subprocess, sys

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
    else:   # never label a binary with an unrelated stamp: no sidecar, no stamp
        stale = os.path.join(out, "delta_matching", "_build_info.py")
        if os.path.exists(stale):
            os.remove(stale)
        print(f"freeze: no _build_info.py beside {a.so}; the frozen package carries no build stamp")
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
        zero = [c for c in K.required_cats(r["mode"]) if not r["ms"][c] > 0]
        if zero:
            raise SystemExit(f"{label}: required kernel categories {zero} have no measured time in {key(r)}")
        kern = r.get("kernels")
        if not isinstance(kern, dict) or not kern:
            raise SystemExit(f"{label}: empty or missing kernel-event collection in {key(r)}")
        for n, ms in kern.items():
            if not isinstance(n, str) or not n or isinstance(ms, bool) or not isinstance(ms, (int, float)) \
                    or not math.isfinite(ms) or ms <= 0:
                raise SystemExit(f"{label}: invalid kernel event {n!r}: {ms!r} in {key(r)} (finite positive ms required)")
        recomputed = {c: 0.0 for c in K.CATS}
        for n, ms in kern.items():
            recomputed[K.classify(n)] += ms
        for c in K.CATS:
            if not math.isfinite(recomputed[c]) or abs(recomputed[c] - r["ms"][c]) > 1e-9 + 1e-6 * abs(r["ms"][c]):
                raise SystemExit(f"{label}: category {c} total {r['ms'][c]} disagrees with its kernel events "
                                 f"({recomputed[c]}) in {key(r)}")


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
        if any(r.get("fa3") for r in rs):   # B20: the FA3 reference over the same repetitions, never repetition 0 alone
            m["fa3"], m["fa3_spread_pct"] = median_fa3([r.get("fa3") for r in rs], k_)
        out.append(m)
    return out


def median_fa3(arms, k_):
    """Per-field median (and spread %) of the FA3 reference timings; every repetition must carry the same finite fields."""
    if any(not a for a in arms):
        raise SystemExit(f"configuration {k_}: FA3 timings missing in some repetitions")
    fields = set(arms[0])
    if any(set(a) != fields for a in arms):
        raise SystemExit(f"configuration {k_}: FA3 field sets differ between repetitions")
    med, spr = {}, {}
    for f in sorted(fields):
        v = [a[f] for a in arms]
        if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
            if not all(math.isfinite(x) and x > 0 for x in v):
                raise SystemExit(f"configuration {k_}: non-finite / non-positive FA3 {f}")
            med[f], spr[f] = statistics.median(v), spread(v)
        else:
            if any(x != v[0] for x in v):
                raise SystemExit(f"configuration {k_}: FA3 {f} differs between repetitions")
            med[f] = v[0]
    return med, spr


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
    for s_ in shapes:
        # rule (Hao, round 10): a shape is SLOWER BEYOND ITS RUN-TO-RUN SPREAD when its median fwd+bwd slowdown exceeds
        # the larger of its two repetition ranges (max - min, ms).  (All-B-above-all-A would flag ~1 in 20 unchanged
        # shapes at 3 x 3 repetitions.)
        rng = max(max(s_["A"]) - min(s_["A"]), max(s_["B"]) - min(s_["B"]))
        s_["slower_beyond_spread"] = s_["median_B"] - s_["median_A"] > rng
    res["shapes"] = shapes
    res["over_gate"] = [s for s in shapes if s["over_gate"]]
    res["worst_shape_change_pct"] = max(s["change_pct"] for s in shapes)
    return res, med_a, med_b


TUNE_DIM = 128           # rule (Hao, round 10): the evolved head dim
# scope (Hao, round 12): the tuned / measured configurations are D128 causal matched only; everything else (stale, full,
# D256) must stay correct (tests, equivalence) but is reported only, never a predicate


def is_matched(r, tune_dim=TUNE_DIM):
    return r["D"] == tune_dim and r["attn"] == "causal" and r["mode"] == "matched"


def is_tuned(r, tune_dim=TUNE_DIM):
    """The acceptance set (Hao, round 16): kernels.PAPER_SET, D128 causal matched Hq 16 / 32, Hk 4, at B2 N8192 and
    B1 N65536 (the paper's training shapes)."""
    return is_matched(r, tune_dim) and (r["D"], r["B"], r["H"], r["Hk"], r["N"]) in K.PAPER_SET


def is_guarded(r, tune_dim=TUNE_DIM):
    """No separate guard since round 16 (kernels.PAPER_GUARD is empty): every acceptance shape is itself gated."""
    return is_matched(r, tune_dim) and (r["D"], r["B"], r["H"], r["Hk"], r["N"]) in K.PAPER_GUARD
TARGET_PCT = 3.0


def decision(raw_a, raw_b, res, target, tune_dim=TUNE_DIM, threshold=TARGET_PCT):
    """Acceptance predicates for target kernel category `target` (Hao, round 16): (1) over the four acceptance shapes
    (kernels.PAPER_SET), the geometric mean of the per-shape target-kernel median ratios improves by >= threshold %
    (equal weight per shape); (2) no acceptance shape is slower beyond its spread; (3) all four shapes measured.
    Other configurations are reported, not gated."""
    def per_rep(runs, sel):
        return [sum(r["ms"][target] for r in recs if sel(r)) for recs in runs]
    med_a, med_b = median_records(raw_a), median_records(raw_b)
    tune = lambda r: is_tuned(r, tune_dim)
    other = lambda r: not is_tuned(r, tune_dim)
    sa = sum(r["ms"][target] for r in med_a if tune(r)); sb = sum(r["ms"][target] for r in med_b if tune(r))
    if sa <= 0:
        raise SystemExit(f"target {target!r} has no time on the D{tune_dim} causal matched configurations")
    oa = sum(r["ms"][target] for r in med_a if other(r)); ob = sum(r["ms"][target] for r in med_b if other(r))
    ra, rb = per_rep(raw_a, other), per_rep(raw_b, other)
    other_rng = max(max(ra) - min(ra), max(rb) - min(rb)) if ra else 0.0
    kd = lambda k: dict(D=k[0], B=k[1], H=k[2], Hk=k[3], N=k[4], attn=k[5], mode=k[6])
    kt = lambda k: is_tuned(kd(k), tune_dim) or is_guarded(kd(k), tune_dim)
    slow_tune = [s for s in res["shapes"] if kt(s["key"]) and s["slower_beyond_spread"]]
    slow_other = [s for s in res["shapes"] if not kt(s["key"]) and s["slower_beyond_spread"]]
    # equal weight per shape (Hao, round 16): the geometric mean of the per-shape target-kernel median ratios
    by_b = {key(r): r for r in med_b}
    ratios = [by_b[key(r)]["ms"][target] / r["ms"][target] for r in med_a if tune(r) and r["ms"][target] > 0]
    gm = math.exp(sum(math.log(x) for x in ratios) / len(ratios))
    d = {"target": target, "tune_dim": tune_dim, "threshold_pct": threshold, "weighting": "geometric mean per shape",
         "target_tune_A": sa, "target_tune_B": sb, "target_tune_change_pct": 100 * (gm - 1),
         "target_tune_sum_change_pct": 100 * (sb / sa - 1),
         "target_shape_ratios": {str(k): v for k, v in zip([key(r) for r in med_a if tune(r) and r["ms"][target] > 0], ratios)},
         "target_other_A": oa, "target_other_B": ob, "target_other_change_pct": 100 * (ob / oa - 1) if oa > 0 else 0.0,
         "target_other_slower_beyond_spread": ob - oa > other_rng,
         "tune_shapes_slower_beyond_spread": [s["key"] for s in slow_tune],
         "other_shapes_slower_beyond_spread": [s["key"] for s in slow_other]}
    d["tuned_configs"] = sum(1 for r in med_a if tune(r))
    d["checks"] = {"target_gain": d["target_tune_change_pct"] <= -threshold, "tune_shapes": not slow_tune,
                   "complete_set": d["tuned_configs"] == len(K.PAPER_SET) == len(ratios)}
    d["accept"] = all(d["checks"].values())
    return d


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
    if a.target:
        res["decision"] = decision(raw["A"], raw["B"], res, a.target)
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
    if "decision" in res:
        d = res["decision"]
        lines += ["", f"### Acceptance predicates (target {d['target']}, D{d['tune_dim']} tuned)", "",
                  f"- target over all D{d['tune_dim']} shapes: {d['target_tune_A']:.3f} -> {d['target_tune_B']:.3f} ms "
                  f"({d['target_tune_change_pct']:+.2f}%; needs <= -{d['threshold_pct']:.1f}%)",
                  f"- D{d['tune_dim']} shapes slower beyond spread: {d['tune_shapes_slower_beyond_spread'] or 'none'}",
                  f"- other head dims: target {d['target_other_A']:.3f} -> {d['target_other_B']:.3f} ms "
                  f"({d['target_other_change_pct']:+.2f}%, slower beyond spread: {d['target_other_slower_beyond_spread']}); "
                  f"shapes slower beyond spread: {d['other_shapes_slower_beyond_spread'] or 'none'}",
                  f"- checks {d['checks']} -> **{'ACCEPT (this job)' if d['accept'] else 'REJECT'}**"]
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
                     f"{s_['change_pct']:+.2f}% | {s_['spread_A_pct']:.2f}% | {s_['spread_B_pct']:.2f}% |"
                     f"{' slower beyond spread' if s_['slower_beyond_spread'] else ''}")
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
    r.add_argument("--grid", choices=("paper", "d128m", "all", "main", "holdout", "one"), default="paper")
    r.add_argument("--target", choices=K.CATS, default=None, help="evaluate the acceptance predicates for this kernel")
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
