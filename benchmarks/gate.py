"""Acceptance-gate runner for one kernel candidate: every step is a checked subprocess with its full output in the run
directory; the first failing step stops the gate with a non-zero exit naming the step.

    python benchmarks/gate.py --tag psipf --worktree .local/wt/psipf --prev-root .local/frozen/r9_p2st3 \\
        --prev-lib .local/jit/p2st3/_C.so --target pass2 --kernel fq8p2 --mode full \\
        --base-dumps .local/equiv/base3_default.pt,.local/equiv/base4_kv.pt,.local/equiv/base4_head.pt,.local/equiv/base4_streams2.pt \\
        --prev-dumps .local/equiv/p2st3_default.pt,... --run-dir .local/rounds/r10/gate_psipf_2

Modes: full (tooling tests, build, ELF-complete resources / SASS with per-symbol stack / local growth fatal, candidate
pytest with in-process identity, four 72-case equivalence suites vs the baseline and the previous accepted revision,
ONE merged-grid A/B (every shape, no held-out split) vs the previous accepted revision, bench.py both D), confirm
(A/B + bench only, against an already built candidate root), timing (build, resources, A/B, bench: feasibility builds;
never an acceptance).  Acceptance rule (Hao, round 10): the --target kernel >= 3% faster summed over all D128 shapes,
no D128 shape slower beyond its run-to-run spread, D256 correct and not slower (ab.py decision()); a full job and a
fresh confirm job must both pass.  The run directory must not exist (fresh).
GATE_PYTHON / GATE_CUOBJDUMP override the executables (CPU failure-injection tests).
"""
import argparse, hashlib, json, os, re, shutil, subprocess, sys
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ROUTES = ("default", "kv", "head", "streams2")


class GateFailure(Exception):
    pass


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Gate:
    def __init__(self, run_dir):
        self.run_dir, self.n, self.record = run_dir, 0, {"steps": []}
        self.py = os.environ.get("GATE_PYTHON", sys.executable)
        self.cuobjdump = os.environ.get("GATE_CUOBJDUMP", "cuobjdump")

    def run(self, name, cmd, cwd=None, env=None):
        self.n += 1
        log = os.path.join(self.run_dir, f"{self.n:02d}_{name}.log")
        with open(log, "w") as f:
            p = subprocess.run(cmd, cwd=cwd or REPO, env=env, stdout=f, stderr=subprocess.STDOUT, text=True)
        self.record["steps"].append({"step": name, "cmd": cmd, "rc": p.returncode, "log": log})
        with open(log) as f:
            tail = f.read()[-1500:]
        print(f"[gate] {name}: rc {p.returncode} ({log})", flush=True)
        if p.returncode != 0:
            print(tail, flush=True)
            raise GateFailure(f"step {name} failed with exit code {p.returncode} (log {log})")
        return log

    def save(self, **kw):
        self.record.update(kw)
        with open(os.path.join(self.run_dir, "gate.json"), "w") as f:
            json.dump(self.record, f, indent=1)


DIAG_FATAL_RE = re.compile(r"C7510|(warning|error)\b.*serializ", re.IGNORECASE)
SPILL_RE = re.compile(r"(\d+) bytes spill stores,\s*(\d+) bytes spill loads")


def build_diagnostics(path):
    """Fatal compiler diagnostics in a build log (B10): C7510 / serialization warnings or errors and NONZERO ptxas
    spill counts; zero-count resource reports and incidental path / source text are not diagnostics."""
    out = []
    for line in open(path, errors="replace"):
        m = SPILL_RE.search(line)
        if m and (int(m.group(1)) or int(m.group(2))):
            out.append(line.strip())
        elif DIAG_FATAL_RE.search(line):
            out.append(line.strip())
    return out
RES_RE = re.compile(r"REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)")
ELF_TEXT_RE = re.compile(r"^\s*[0-9a-f]+\s+[0-9a-f]+\s+([0-9a-f]+)\s.*\sPROGBITS\s.*\s\.text\.(\S+)\s*$")
SASS_OFF_RE = re.compile(r"/\*([0-9a-f]{4,6})\*/")


def elf_text_sizes(path, kernel, tag):
    """{symbol: .text section size in bytes} for every function matching `kernel` in a `cuobjdump --dump-elf` log;
    a symbol with two .text sections (duplicate records) fails."""
    sizes = {}
    for line in open(path).read().splitlines():
        m = ELF_TEXT_RE.match(line)
        if m and re.search(kernel, m.group(2)):
            if m.group(2) in sizes:
                raise GateFailure(f"{tag}: duplicate ELF .text section for {m.group(2)[:80]}")
            sizes[m.group(2)] = int(m.group(1), 16)
    return sizes


def resources(gate, lib, kernel, tag):
    """Resource rows and SASS bodies of every function matching `kernel`; fails unless the selection is non-empty, the
    resource and SASS outputs name the same functions, every function has a well-formed resource row and a body."""
    out = os.path.join(gate.run_dir, f"resource_{tag}.txt")
    log = gate.run(f"resources_{tag}", [gate.cuobjdump, "--dump-resource-usage", lib])
    shutil.copyfile(log, out)
    sass_log = gate.run(f"sass_{tag}", [gate.cuobjdump, "-sass", lib])
    elf_log = gate.run(f"elf_{tag}", [gate.cuobjdump, "--dump-elf", lib])
    sizes = elf_text_sizes(elf_log, kernel, tag)
    res_lines = open(out).read().splitlines()
    rows = {}
    for i, line in enumerate(res_lines):
        m = re.match(r"\s*Function (\S+):\s*$", line)
        if m and re.search(kernel, m.group(1)):
            nxt = res_lines[i + 1] if i + 1 < len(res_lines) else ""
            r = RES_RE.search(nxt)
            if m.group(1) in rows:
                raise GateFailure(f"{tag}: duplicate resource row for {m.group(1)[:80]}")
            if not r:
                raise GateFailure(f"{tag}: resource row missing or malformed for {m.group(1)[:80]}")
            rows[m.group(1)] = dict(zip(("REG", "STACK", "SHARED", "LOCAL"), map(int, r.groups())))
    bodies, cur = {}, None
    for line in open(sass_log).read().splitlines():
        m = re.match(r"\s*Function : (\S+)", line)
        if m:
            cur = m.group(1) if re.search(kernel, m.group(1)) else None
            if cur:
                if cur in bodies:
                    raise GateFailure(f"{tag}: duplicate SASS function record for {cur[:80]}")
                bodies[cur] = []
            continue
        if cur and SASS_OFF_RE.search(line):
            bodies[cur].append(line)
    if not rows or not bodies:
        raise GateFailure(f"{tag}: no function matches {kernel!r} (resources {len(rows)}, sass {len(bodies)})")
    if set(rows) != set(bodies):
        raise GateFailure(f"{tag}: resource and SASS outputs select different functions for {kernel!r}")
    if set(sizes) != set(bodies):
        raise GateFailure(f"{tag}: ELF .text sections and SASS select different functions for {kernel!r}")
    for n, b in bodies.items():
        # complete coverage: unique, contiguous 16-byte instruction offsets from 0 to the exact .text section end
        offs = [int(SASS_OFF_RE.search(l).group(1), 16) for l in b]
        if sizes[n] <= 0 or sizes[n] % 16 or offs != list(range(0, sizes[n], 16)):
            raise GateFailure(f"{tag}: incomplete SASS for {n[:80]}: {len(offs)} instructions, offsets "
                              f"{'not contiguous/unique from 0' if offs[:1] != [0] or len(set(offs)) != len(offs) else 'end at ' + hex(offs[-1] if offs else 0)}"
                              f", ELF .text size {hex(sizes[n])}")
    keep = [l for n in sorted(bodies) for l in [f"Function : {n}"] + bodies[n]]
    with open(os.path.join(gate.run_dir, f"target_{tag}.sass"), "w") as f:
        f.write("\n".join(keep))
    counts = {k: sum(1 for l in keep if k in l) for k in ("QGMMA", "STL", "LDL", "WARPGROUP.DEPBAR", "WARPGROUP.ARRIVE", "BAR.SYNC", "BAR.ARV", "CALL")}
    return {"functions": len(rows), "rows": rows, "text_bytes": sizes, "lines": len(keep), "counts": counts}


def resource_regressions(prev, cand):
    """Per-symbol STACK / LOCAL growth (candidate vs previous build, same symbol); also symbols present on one side."""
    out = [f"{n[:100]}: only in {'previous' if n in prev['rows'] else 'candidate'}" for n in set(prev["rows"]) ^ set(cand["rows"])]
    for n in sorted(set(prev["rows"]) & set(cand["rows"])):
        a, b = prev["rows"][n], cand["rows"][n]
        for k in ("STACK", "LOCAL"):
            if b[k] > a[k]:
                out.append(f"{n[:100]}: {k} {a[k]} -> {b[k]}")
    return out


BENCH_PCT = 1.0
BENCH_ROWS = ("delta_matching (matched)", "delta_matching (stale)")


def bench_decision(files):
    """bench.py medians per (D, row) for A and B over the repetitions: every row complete, finite and positive; the
    fwd+bwd median of B within BENCH_PCT % of A's for matched and stale at both D."""
    import math, statistics
    vals = {}
    for x, D, path in files:
        if not os.path.isfile(path):
            raise GateFailure(f"bench.py wrote no JSON ({path})")
        with open(path) as f:
            rows = json.load(f).get("rows", {})
        for r in BENCH_ROWS:
            v = rows.get(r, {}).get("fwdbwd_ms")
            if not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
                raise GateFailure(f"bench.py {path}: missing or invalid {r!r} fwd+bwd timing")
            vals.setdefault((D, r, x), []).append(v)
    ch, ok = {}, True
    for D in sorted({D for _, D, _ in files}):
        for r in BENCH_ROWS:
            a_, b_ = vals.get((D, r, "A")), vals.get((D, r, "B"))
            if not a_ or not b_ or len(a_) != len(b_):
                raise GateFailure(f"bench.py: incomplete repetitions for D{D} {r}")
            c = 100 * (statistics.median(b_) / statistics.median(a_) - 1)
            ch[f"D{D} {r}"] = round(c, 3)
            ok = ok and c <= BENCH_PCT
    return {"changes_pct": ch, "ok": ok, "limit_pct": BENCH_PCT}


ALLOWED_SKIPS = ("needs two GPUs", "could not import 'flash_attn'")
MIN_PASSED = 49


def check_junit(path):
    """The candidate's test report: no failures / errors, at least MIN_PASSED passed, only the permitted skip reasons."""
    if not os.path.isfile(path):
        raise GateFailure("pytest wrote no junit report")
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    failed = [c for c in cases if c.find("failure") is not None or c.find("error") is not None]
    skipped = [c for c in cases if c.find("skipped") is not None]
    passed = len(cases) - len(failed) - len(skipped)
    bad_skips = [c.find("skipped").get("message", "") for c in skipped
                 if not any(r in (c.find("skipped").get("message", "") or "") for r in ALLOWED_SKIPS)]
    if failed or passed < MIN_PASSED or bad_skips:
        raise GateFailure(f"candidate tests: {passed} passed, {len(failed)} failed, skips outside the permitted set "
                          f"{bad_skips[:3]} (need >= {MIN_PASSED} passed)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True); ap.add_argument("--worktree", required=True)
    ap.add_argument("--prev-root", required=True); ap.add_argument("--prev-lib", required=True)
    ap.add_argument("--subset", default=None, help="supplementary per-kernel report on D128 / D256 (not a predicate)")
    ap.add_argument("--target", required=True, help="kernel category the candidate targets (acceptance predicates)")
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--mode", choices=("full", "confirm", "timing"), required=True)
    ap.add_argument("--base-dumps", default=""); ap.add_argument("--prev-dumps", default="")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--lib", default=None, help="prebuilt candidate library (no rebuild in the gate); needs --build-log")
    ap.add_argument("--build-log", default=None, help="the prebuilt library's build log (spill / serialization scan)")
    ap.add_argument("--tooling-tests", action="store_true", help="also run the CPU tooling tests (default: run them "
                    "outside GPU jobs)")
    a = ap.parse_args()
    if os.path.exists(a.run_dir):
        raise SystemExit(f"run directory {a.run_dir} exists: every gate run gets a fresh directory")
    os.makedirs(a.run_dir)
    run_dir = os.path.abspath(a.run_dir)
    gate = Gate(run_dir)
    wt = os.path.abspath(a.worktree)
    env = dict(os.environ, DM_FA_SRC=os.path.join(REPO, "third_party", "flash-attention"))
    env.pop("DM_SO", None)
    try:
        if a.mode in ("full", "timing"):
            if a.tooling_tests:
                gate.run("tooling_tests", [gate.py, "-m", "pytest", os.path.join(HERE, "test_tooling.py"), "-q"])
            if a.lib:
                # Hao (round 11): never rebuild in the gate when the library exists; link the worktree to it
                # resolve and validate BEFORE touching the package (B9): the package entry may itself be the input
                lib = os.path.realpath(a.lib)
                if not os.path.isfile(lib) or not a.build_log or not os.path.isfile(a.build_log):
                    raise GateFailure("--lib needs an existing library and its --build-log")
                build_log = os.path.abspath(a.build_log)
                link = os.path.join(wt, "delta_matching", "_C.so")
                if not (os.path.exists(link) and os.path.realpath(link) == lib):
                    if os.path.lexists(link):
                        os.remove(link)          # a different library (or a dangling link): never the input itself
                    os.symlink(lib, link)
            else:
                jit = os.path.join(run_dir, "jit")
                build_log = gate.run("build", [gate.py, "-m", "delta_matching.jit"], cwd=wt,
                                     env=dict(env, MAX_JOBS="8", DM_JIT_DIR=jit))
                lib = os.path.join(jit, "_C.so")
            if not os.path.isfile(lib):
                raise GateFailure(f"build produced no library at {lib}")
            diag = build_diagnostics(build_log)
            if diag:   # ptxas spill / wgmma serialization diagnostics stay hard failures
                raise GateFailure(f"build diagnostics: {diag[0][:160]} ({len(diag)} lines)")
            linked = os.path.realpath(os.path.join(wt, "delta_matching", "_C.so"))
            if linked != os.path.realpath(lib):
                raise GateFailure(f"the worktree package links {linked}, not the fresh build {lib}")
            gate.save(candidate_library=lib, candidate_sha256=sha256(lib))
            r_prev = resources(gate, os.path.abspath(a.prev_lib), a.kernel, "prev")
            r_cand = resources(gate, lib, a.kernel, "cand")
            if r_prev["functions"] != r_cand["functions"]:
                raise GateFailure(f"target {a.kernel!r}: {r_prev['functions']} functions in the previous build, "
                                  f"{r_cand['functions']} in the candidate")
            regress = resource_regressions(r_prev, r_cand)
            gate.save(resources={"prev": r_prev, "cand": r_cand}, resource_regressions=regress)
            for line in regress:
                print(f"[gate] resource regression: {line}", flush=True)
            if regress:   # Hao (round 11): a flagged warning; the A/B still runs and the timing decides
                print(f"[gate] WARNING: per-symbol stack / local growth ({len(regress)}); recorded, not fatal", flush=True)
            for n in sorted(set(r_prev["rows"]) | set(r_cand["rows"])):
                print(f"[gate] resources {n[-90:]}: prev {r_prev['rows'].get(n)} {r_prev['text_bytes'].get(n)} B -> "
                      f"cand {r_cand['rows'].get(n)} {r_cand['text_bytes'].get(n)} B", flush=True)
            print(f"[gate] counts prev {r_prev['counts']}\n[gate] counts cand {r_cand['counts']}  (inspect target_*.sass)", flush=True)
        if a.mode == "full":
            # the candidate's own tests, in a process pinned to the fresh build and verified from inside (gate_identity)
            junit = os.path.join(run_dir, "pytest.xml")
            tenv = dict(env, PYTHONPATH=os.pathsep.join([wt, HERE]), DM_SO=lib, KDA_EXPECT_PKG=os.path.join(wt, "delta_matching"),
                        KDA_EXPECT_LIB=lib, KDA_EXPECT_SHA256=gate.record["candidate_sha256"])
            for k_ in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS"):
                tenv.pop(k_, None)
            gate.run("pytest", [gate.py, "-m", "pytest", "tests/", "-q", "-rs", "-p", "gate_identity", f"--junitxml={junit}"],
                     cwd=wt, env=tenv)
            check_junit(junit)
            base = a.base_dumps.split(","); prev = a.prev_dumps.split(",")
            if len(base) != 4 or len(prev) != 4:
                raise GateFailure("--base-dumps / --prev-dumps need one dump per route: default,kv,head,streams2")
            for i, r in enumerate(ROUTES):
                out = os.path.join(run_dir, f"cand_{r}.pt")
                gate.run(f"dump_{r}", [gate.py, os.path.join(HERE, "equivalence.py"), "dump", "--route", r,
                                       "--expect-root", wt, "--out", out], env=dict(env, PYTHONPATH=wt))
                if not os.path.isfile(out):
                    raise GateFailure(f"dump {r} wrote nothing")
                gate.run(f"compare_base_{r}", [gate.py, os.path.join(HERE, "equivalence.py"), "compare", base[i], out])
                gate.run(f"compare_prev_{r}", [gate.py, os.path.join(HERE, "equivalence.py"), "compare", prev[i], out])
        bench_files = []
        ab_env = dict(env, DM_SO=os.path.join(REPO, "delta_matching", "_C.so"))   # conflicting on purpose: ignored
        # rule (Hao, round 10): ONE merged grid (no held-out split); acceptance = target >= 3% over all D128 shapes, no
        # D128 shape slower beyond its spread, D256 not slower (ab.py decision()); the first job and the confirmation
        # job must each pass.
        d = os.path.join(run_dir, "ab_all")
        gate.run("ab_all", [gate.py, os.path.join(HERE, "ab.py"), "run", "--a", os.path.abspath(a.prev_root), "--b", wt,
                            "--grid", "all", "--reps", str(a.reps), "--target", a.target, "--out-dir", d]
                 + (["--subset", a.subset] if a.subset else []), env=ab_env)
        if not os.path.isfile(os.path.join(d, "ab.json")):
            raise GateFailure("A/B wrote no ab.json")
        with open(os.path.join(d, "ab.json")) as f:
            dec = json.load(f).get("decision")
        if not dec or "accept" not in dec:
            raise GateFailure("A/B report has no acceptance decision")
        gate.save(decision=dec)
        print(f"[gate] decision: {dec.get('checks')} target D{dec.get('tune_dim')} "
              f"{dec.get('target_tune_change_pct', float('nan')):+.2f}% -> {'ACCEPT' if dec['accept'] else 'REJECT'}", flush=True)
        for rep in range(3):
            order = ("A", "B") if rep != 1 else ("B", "A")
            for x in order:
                root = os.path.abspath(a.prev_root) if x == "A" else wt
                for D in (128, 256):
                    bj = os.path.join(run_dir, f"bench_{x}_rep{rep}_D{D}.json")
                    bench_files.append((x, D, bj))
                    gate.run(f"bench_{x}_rep{rep}_D{D}", [gate.py, os.path.join(HERE, "bench.py"), "--D", str(D), "--N", "8192",
                                                          "--json-out", bj],
                             env=dict(env, PYTHONPATH=root))
        bdec = bench_decision(bench_files)
        gate.save(bench_decision=bdec)
        print(f"[gate] bench: {bdec['changes_pct']} -> {'ok' if bdec['ok'] else 'REGRESSION'}", flush=True)
        if not dec["accept"]:
            raise GateFailure(f"acceptance predicates not met: {dec['checks']}")
        if not bdec["ok"]:
            raise GateFailure(f"bench.py fwd+bwd regression above {BENCH_PCT}%: {bdec['changes_pct']}")
        gate.save(result="PASSED (all steps exited 0 and the merged-grid acceptance predicates hold for this job)")
        print("[gate] all steps passed", flush=True)
        return 0
    except GateFailure as e:
        gate.save(result=f"FAILED: {e}")
        print(f"[gate] FAILED: {e}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
