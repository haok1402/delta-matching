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
CS = os.environ.get("DMVAL_SANITIZER",
                    os.path.expanduser("~/workspace/haok1402/sm100/env/cuda-13.0/bin/compute-sanitizer"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate")
    ap.add_argument("--tool", default="memcheck")
    ap.add_argument("--log", default=None)
    a = ap.parse_args()
    tgt, sep, fn = a.candidate.partition(":")
    cand = os.path.abspath(tgt) + sep + fn
    env = dict(os.environ, PYTORCH_NO_CUDA_MEMORY_CACHING="1")
    cmd = [CS, "--tool", a.tool, "--print-limit", "50", "--error-exitcode", "9", sys.executable,
           os.path.join(HERE, "sanitize_child.py"), cand]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    out = p.stdout + p.stderr
    if a.log:
        open(a.log, "w").write(out)
    m = re.search(r"ERROR SUMMARY: (\d+) error", out)
    nerr = int(m.group(1)) if m else -1
    done = "SANITIZE_CHILD_DONE" in out
    ok = p.returncode == 0 and nerr == 0 and done
    first = next((ln for ln in out.splitlines() if "Invalid" in ln or "Error" in ln or "Traceback" in ln), None)
    frames = sorted(set(re.findall(r"Host Frame: (\w+) in ([\w./]+:\d+)", out)))[:8]
    print("SUMMARY_JSON " + json.dumps({"pass": ok, "tool": a.tool, "errors": nerr, "child_done": done,
                                        "rc": p.returncode, "first": first, "frames": frames}))
    print(("PASS" if ok else "FAIL") + f": sanitizer {a.tool} errors={nerr} child_done={done}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
