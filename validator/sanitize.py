#!/usr/bin/env python
"""Sanitizer tier: run a candidate under compute-sanitizer memcheck on small tail / partial-tile shapes.

  python validator/sanitize.py /abs/path/cand.py[:attn] [--tool memcheck|initcheck|racecheck]

PASS iff the sanitizer reports 0 errors and the child finished.  The PyTorch caching allocator is disabled
(PYTORCH_NO_CUDA_MEMORY_CACHING=1): with it, a read just past a tensor lands inside the cached segment and memcheck sees
nothing (this hid the forward's K-scale over-read of kda-sm100 21b6b15 / a911afe).  Required for promotion."""
import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RACE_BASE = os.path.expanduser("~/workspace/haok1402/sm100/validate/racecheck_baseline.json")
CS = os.environ.get("DMVAL_SANITIZER",
                    os.path.expanduser("~/workspace/haok1402/sm100/env/cuda-13.0/bin/compute-sanitizer"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate")
    ap.add_argument("--tool", default="memcheck")
    ap.add_argument("--log", default=None)
    ap.add_argument("--timeout", type=float, default=600.0, help="seconds; a hang under the sanitizer is a FAIL")
    a = ap.parse_args()
    tgt, sep, fn = a.candidate.partition(":")
    cand = os.path.abspath(tgt) + sep + fn
    env = dict(os.environ, PYTORCH_NO_CUDA_MEMORY_CACHING="1")
    # API-error reports are off: the CuTe DSL runtime probes optional driver entry points (cuGetProcAddress ->
    # CUDA_ERROR_INVALID_VALUE) during arch detection; those are not memory errors.  Memory errors are counted below.
    cmd = [CS, "--tool", a.tool, "--report-api-errors", "no", "--print-limit", "50", "--error-exitcode", "9", sys.executable,
           os.path.join(HERE, "sanitize_child.py"), cand]
    import tempfile
    log = a.log or tempfile.mktemp(prefix="dmval_san_", suffix=".txt")
    with open(log, "w") as lf:                      # streamed: a hang still leaves the evidence on disk
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            p.wait(timeout=a.timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            import signal
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            timed_out = True
    out = open(log, errors="replace").read()
    m = re.search(r"ERROR SUMMARY: (\d+) error", out)
    nerr = int(m.group(1)) if m else -1
    done = "SANITIZE_CHILD_DONE" in out
    ninv = len(re.findall(r"Invalid __(?:global|shared|local)__ (?:read|write)", out))
    races = {}
    for k in re.findall(r"Race reported between \w+ access at (kernel_cutlass_kernel_[A-Za-z0-9_]+?)_object_at", out):
        races[k] = races.get(k, 0) + 1
    if a.tool == "racecheck":
        # compute-sanitizer racecheck does not model sm100 mbarrier / TMA / tcgen05 pipeline synchronization: the validated
        # 0c3f69d (bitwise-deterministic over every stress repeat) reports 50 hazard pairs in its backward and FA4's own
        # bf16 backward 24. So racecheck gates as NO INCREASE vs the baseline counts per kernel (racecheck_baseline.json);
        # races are primarily caught by the run-to-run spread and 20-rep stress tiers.
        base = json.load(open(RACE_BASE)) if os.path.exists(RACE_BASE) else {}
        worse = {k: (n, base.get(k)) for k, n in races.items() if base.get(k) is None or n > base[k]}
        ok = (not timed_out) and done and not worse
        print("SUMMARY_JSON " + json.dumps({"pass": ok, "tool": "racecheck", "races": races, "baseline": base,
                                            "worse_than_baseline": worse, "child_done": done, "timed_out": timed_out,
                                            "log": log}))
        print(("PASS" if ok else "FAIL") + f": racecheck {sum(races.values())} hazard pairs, "
              f"{len(worse)} kernel(s) above the baseline count")
        sys.exit(0 if ok else 1)
    ok = (not timed_out) and p.returncode == 0 and nerr == 0 and ninv == 0 and done
    first = next((ln for ln in out.splitlines() if "Invalid" in ln or "Error" in ln or "Traceback" in ln), None)
    frames = sorted(set(re.findall(r"Host Frame: (\w+) in ([\w./]+:\d+)", out)))[:8]
    print("SUMMARY_JSON " + json.dumps({"pass": ok, "tool": a.tool, "errors": nerr, "invalid_accesses": ninv, "child_done": done, "timed_out": timed_out, "log": log,
                                        "rc": p.returncode, "first": first, "frames": frames}))
    print(("PASS" if ok else "FAIL") + f": sanitizer {a.tool} errors={nerr} invalid_accesses={ninv} child_done={done}"
          + (f" TIMEOUT after {a.timeout:.0f} s (hang under the sanitizer)" if timed_out else ""))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
