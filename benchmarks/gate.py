"""Acceptance-gate runner for one kernel candidate: every step is a checked subprocess with its full output in the run
directory; the first failing step stops the gate with a non-zero exit naming the step.

    python benchmarks/gate.py --tag h1b --worktree .local/wt/h1b --prev-root .local/frozen/r3_do2 \\
        --prev-lib .local/jit/do2/_C.so --subset D128 --kernel FlashAttnBwdSm90Fp8R --mode full \\
        --base-dumps .local/equiv/base3_default.pt,.local/equiv/base4_kv.pt,.local/equiv/base4_head.pt,.local/equiv/base4_streams2.pt \\
        --prev-dumps .local/equiv/do2_4_default.pt,... --run-dir .local/rounds/h1b_run1

Modes: full (tooling tests, build, resources / SASS, candidate pytest, four 72-case equivalence suites vs the baseline
and the previous accepted revision, A/B main + held-out vs the previous accepted revision, bench.py both D), confirm
(A/B + bench only, against an already built candidate root), timing (build, resources, A/B, bench: for feasibility
builds whose numerics are intentionally incomplete; never an acceptance).  The run directory must not exist (fresh).
GATE_PYTHON / GATE_CUOBJDUMP override the executables (CPU failure-injection tests).
"""
import argparse, hashlib, json, os, re, shutil, subprocess, sys

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


def resources(gate, lib, kernel, tag):
    out = os.path.join(gate.run_dir, f"resource_{tag}.txt")
    log = gate.run(f"resources_{tag}", [gate.cuobjdump, "--dump-resource-usage", lib])
    shutil.copyfile(log, out)
    sass_log = gate.run(f"sass_{tag}", [gate.cuobjdump, "-sass", lib])
    text = open(sass_log).read().splitlines()
    keep, sel = [], False
    for line in text:
        if "Function : " in line:
            sel = re.search(kernel, line) is not None
        if sel:
            keep.append(line)
    with open(os.path.join(gate.run_dir, f"target_{tag}.sass"), "w") as f:
        f.write("\n".join(keep))
    res = open(out).read()
    counts = {k: sum(1 for l in keep if k in l) for k in ("QGMMA", "STL", "LDL", "WARPGROUP.DEPBAR", "WARPGROUP.ARRIVE", "BAR.SYNC", "BAR.ARV", "CALL")}
    blocks = re.findall(r"Function [^\n]*" + kernel + r"[^\n]*:\n[^\n]*", res)
    return {"lines": len(keep), "counts": counts, "resource_lines": [b.split("\n")[1].strip() for b in blocks]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True); ap.add_argument("--worktree", required=True)
    ap.add_argument("--prev-root", required=True); ap.add_argument("--prev-lib", required=True)
    ap.add_argument("--subset", required=True); ap.add_argument("--kernel", required=True)
    ap.add_argument("--mode", choices=("full", "confirm", "timing"), required=True)
    ap.add_argument("--base-dumps", default=""); ap.add_argument("--prev-dumps", default="")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--reps", type=int, default=3)
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
            if a.mode == "full":
                gate.run("tooling_tests", [gate.py, "-m", "pytest", os.path.join(HERE, "test_tooling.py"), "-q"])
            jit = os.path.join(run_dir, "jit")
            gate.run("build", [gate.py, "-m", "delta_matching.jit"], cwd=wt, env=dict(env, MAX_JOBS="8", DM_JIT_DIR=jit))
            lib = os.path.join(jit, "_C.so")
            if not os.path.isfile(lib):
                raise GateFailure(f"build produced no library at {lib}")
            linked = os.path.realpath(os.path.join(wt, "delta_matching", "_C.so"))
            if linked != os.path.realpath(lib):
                raise GateFailure(f"the worktree package links {linked}, not the fresh build {lib}")
            gate.save(candidate_library=lib, candidate_sha256=sha256(lib))
            r_prev = resources(gate, os.path.abspath(a.prev_lib), a.kernel, "prev")
            r_cand = resources(gate, lib, a.kernel, "cand")
            gate.save(resources={"prev": r_prev, "cand": r_cand})
            print(f"[gate] resources prev {r_prev['resource_lines']} {r_prev['counts']}\n[gate] resources cand "
                  f"{r_cand['resource_lines']} {r_cand['counts']}  (inspect target_*.sass)", flush=True)
        if a.mode == "full":
            gate.run("pytest", [gate.py, "-m", "pytest", "tests/", "-q"], cwd=wt)
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
        ab_env = dict(env, DM_SO=os.path.join(REPO, "delta_matching", "_C.so"))   # conflicting on purpose: ignored
        for grid in ("main", "holdout"):
            d = os.path.join(run_dir, f"ab_{grid}")
            gate.run(f"ab_{grid}", [gate.py, os.path.join(HERE, "ab.py"), "run", "--a", os.path.abspath(a.prev_root),
                                    "--b", wt, "--grid", grid, "--reps", str(a.reps), "--subset", a.subset,
                                    "--out-dir", d], env=ab_env)
            if not os.path.isfile(os.path.join(d, "ab.json")):
                raise GateFailure(f"A/B {grid} wrote no ab.json")
        for rep in range(3):
            order = ("A", "B") if rep != 1 else ("B", "A")
            for x in order:
                root = os.path.abspath(a.prev_root) if x == "A" else wt
                for D in (128, 256):
                    gate.run(f"bench_{x}_rep{rep}_D{D}", [gate.py, os.path.join(HERE, "bench.py"), "--D", str(D), "--N", "8192"],
                             env=dict(env, PYTHONPATH=root))
        gate.save(result="PASSED (all steps exited 0; the acceptance decision reads ab_*/ab.json and the logs)")
        print("[gate] all steps passed", flush=True)
        return 0
    except GateFailure as e:
        gate.save(result=f"FAILED: {e}")
        print(f"[gate] FAILED: {e}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
