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
RACE_SITES = os.path.expanduser("~/workspace/haok1402/sm100/validate/racecheck_sites.json")
_KER = r"kernel_cutlass_kernel_([A-Za-z0-9_]+?)_object_at\S*\+(0x[0-9a-f]+)(?: in (\S+?):(\d+))?"


def race_src_sites(out):
    """SOURCE-keyed hazard sites (accuracy-audit + lead 17:05): kernel|Kind@file:line|Kind@file:line from a lineinfo build.
    Returns (mapped set, unmapped PC-keyed set): a pair with either side lacking a source line is unmapped."""
    src, unm, cur = set(), set(), None
    for ln in out.splitlines():
        m = re.search(r"Race reported between (\w+) access at " + _KER, ln)
        if m:
            cur = (m.group(2), m.group(1), m.group(3), m.group(4), m.group(5))
            continue
        m = re.search(r"and (\w+) access at " + _KER, ln)
        if m and cur:
            if cur[3] and m.group(4):
                src.add(f"{cur[0]}|{cur[1]}@{os.path.basename(cur[3])}:{cur[4]}|{m.group(1)}@{os.path.basename(m.group(4))}:{m.group(5)}")
            else:
                unm.add(f"{cur[0]}|{cur[1]}@{cur[2]}|{m.group(1)}@{m.group(3)}")
    return src, unm


def race_sites(out):
    """Hazard sites {kernel|Access@pc|Access@pc}: one per (first access, conflicting access) pair of every race record."""
    s, cur = set(), None
    for ln in out.splitlines():
        m = re.search(r"Race reported between (\w+) access at " + _KER, ln)
        if m:
            cur = (m.group(2), m.group(1), m.group(3))
            continue
        m = re.search(r"and (\w+) access at " + _KER, ln)
        if m and cur:
            s.add(f"{cur[0]}|{cur[1]}@{cur[2]}|{m.group(1)}@{m.group(3)}")
    return s


def cand_mode(path):
    """'nd' if the candidate declares DETERMINISTIC = False (a different backward binary, its own site set), else 'det'."""
    src = open(path.partition(":")[0]).read()
    return "nd" if re.search(r"^DETERMINISTIC\s*=\s*False", src, re.M) else "det"
CS = os.environ.get("DMVAL_SANITIZER",
                    os.path.expanduser("~/workspace/haok1402/sm100/env/cuda-13.0/bin/compute-sanitizer"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate")
    ap.add_argument("--tool", default="memcheck")
    ap.add_argument("--log", default=None)
    ap.add_argument("--timeout", type=float, default=600.0, help="seconds; a hang under the sanitizer is a FAIL "
                    "(racecheck with lineinfo: 1500 by default, the small q_stage-1 shapes compile under the sanitizer)")
    ap.add_argument("--accept-sites", default=None, metavar="NOTE",
                    help="racecheck: add the measured sites to the baseline site set of this mode with a written "
                         "explanation (which mbarrier / TMA / tcgen05 sync orders them); only after review")
    a = ap.parse_args()
    if a.tool == "racecheck" and a.timeout == 600.0 and os.environ.get("DMVAL_RACE_LINEINFO", "1") != "0":
        a.timeout = 1500.0
    tgt, sep, fn = a.candidate.partition(":")
    cand = os.path.abspath(tgt) + sep + fn
    env = dict(os.environ, PYTORCH_NO_CUDA_MEMORY_CACHING="1")
    lineinfo_dir = None
    if a.tool == "racecheck" and os.environ.get("DMVAL_RACE_LINEINFO", "1") != "0":
        # hazard sites keyed by SOURCE (file:line), not PC (lead 2026-11-09): kernels built with line info, in a fresh
        # compile cache so a cached non-lineinfo cubin is never reused
        # persistent per-candidate lineinfo cache: fill it ahead of time in a --gpus 0 run (CUTE_DSL_LINEINFO=1 aot_snap.py with
        # FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=<this dir>/cute) so the lease does not compile; never shared with normal builds
        lineinfo_dir = os.environ.get("DMVAL_LINEINFO_CACHE") or os.path.expanduser(
            "~/workspace/haok1402/sm100/validate/lineinfo_cache/" + os.path.basename(tgt).replace(".py", ""))
        os.makedirs(lineinfo_dir, exist_ok=True)
        env.update(CUTE_DSL_LINEINFO="1", FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=os.path.join(lineinfo_dir, "cute"),
                   TRITON_CACHE_DIR=os.path.join(lineinfo_dir, "triton"))
    # API-error reports are off: the CuTe DSL runtime probes optional driver entry points (cuGetProcAddress ->
    # CUDA_ERROR_INVALID_VALUE) during arch detection; those are not memory errors.  Memory errors are counted below.
    cmd = [CS, "--tool", a.tool, "--report-api-errors", "no", "--print-limit", "0" if a.tool == "racecheck" else "50", "--error-exitcode", "9", sys.executable,
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
        # backward (bitwise-deterministic over every stress repeat) and FA4's own bf16 backward report hazards.  Gate (per
        # accuracy-audit 2026-11-09): the SET of hazard sites (kernel, access@pc, access@pc) must be a subset of the
        # reviewed baseline site set of the same mode (racecheck_sites.json).  Count-only gating let a new real race through
        # when an old artefact vanished, and was capped by --print-limit 50 (every "50 pairs" so far was that cap; the
        # limit is now off for racecheck).  PCs are stable for an unchanged kernel binary; a changed kernel shows new
        # sites and needs a written explanation before they are accepted (--accept-sites "NOTE").
        mode = cand_mode(tgt)
        sites = race_sites(out)
        src, unm = race_src_sites(out)
        allb = json.load(open(RACE_SITES)) if os.path.exists(RACE_SITES) else {}
        base = allb.get(mode, {"sites": [], "notes": []})
        src_keyed = "src_sites" in base and lineinfo_dir is not None
        if src_keyed:
            # SOURCE-keyed gate (accuracy-audit + lead 17:05): every (kernel, W/R file:line pair) must be in the reviewed set;
            # sites without a source line fall back to a count that must not grow.  (racecheck does not name the smem buffer;
            # the write / read source lines identify the code that touches it.)
            new = sorted(src - set(base["src_sites"]))
            unm_ok = len(unm) <= base.get("unmapped_max", 0)
        else:
            new = sorted(sites - set(base["sites"]))
            unm_ok = True
        if a.accept_sites and done and not timed_out:
            if lineinfo_dir is not None:
                base["src_sites"] = sorted(set(base.get("src_sites", [])) | src)
                base["unmapped_max"] = max(base.get("unmapped_max", 0), len(unm))
            base["sites"] = sorted(set(base["sites"]) | sites)
            base["notes"].append({"note": a.accept_sites, "added": len(new), "candidate": cand,
                                  "keyed": "source" if lineinfo_dir is not None else "pc"})
            allb[mode] = base
            tmp = RACE_SITES + ".tmp"
            json.dump(allb, open(tmp, "w"), indent=1)
            os.replace(tmp, RACE_SITES)
            print(f"accepted {len(new)} new site(s) into the {mode} baseline")
            new, unm_ok = [], True
        by_k = {}
        for x in new:
            by_k[x.split("|")[0]] = by_k.get(x.split("|")[0], 0) + 1
        ok = (not timed_out) and done and not new and unm_ok
        print("SUMMARY_JSON " + json.dumps({"pass": ok, "tool": "racecheck", "mode": mode, "keyed": "source" if src_keyed else "pc",
                                            "race_records": races, "pc_sites": len(sites), "src_sites": len(src),
                                            "unmapped": len(unm), "unmapped_max": base.get("unmapped_max"),
                                            "new_sites_by_kernel": by_k, "new_sites": new[:200], "child_done": done,
                                            "timed_out": timed_out, "log": log}))
        print(("PASS" if ok else "FAIL") + f": racecheck ({mode}, {'source' if src_keyed else 'pc'}-keyed) "
              f"{sum(races.values())} race records, {len(src)} source sites + {len(unm)} unmapped, "
              f"{len(new)} not in the reviewed baseline" + (f" {by_k}" if by_k else "") + ("" if unm_ok else " UNMAPPED COUNT GREW"))
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
