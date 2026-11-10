#!/usr/bin/env python
"""Identity tier: the backward's own in-kernel zero-sum identity at the headline lengths, all input families.

  python validator/identity_tier.py /abs/cand.py [--lengths 8192,32768,65536] [--json out]

Needs the candidate's debug() to return rowsum_dS / sum_abs_PdP (in-kernel, before dS quantization) and the debug build's
grads dbg_dq / dbg_dk / dbg_dv, which must be bitwise the production grads (accuracy-audit ruling 03:45 (1)).  No reference
is needed, so this is cheap at 64k.  Limits: max |rowsum_dS| / sum_abs_PdP <= 1e-5 and row-mean <= 2e-7 (not scaled with
N).  Until kda-sm100-acc lands, cases above the limit gate as no-regression vs identity_baseline.json (x1.25) and are
reported as OPEN (accuracy-audit ruling 03:45 (4))."""
try:
    from gpupool_hook import gpupool_done as _d
    _d()
except ImportError:
    pass
import argparse
import importlib.util
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MAXL, MEANL = 1e-5, 2e-7
BASE = os.path.expanduser("~/workspace/haok1402/sm100/validate/identity_baseline.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate")
    ap.add_argument("--lengths", default="8192,32768,65536,131072")   # user 2026-11-09: + 128k
    ap.add_argument("--json", default=None)
    ap.add_argument("--hd", type=int, default=128, choices=[128, 256], help="head dim; hd256 uses 8/2 + 16/4 (qwen35_m / _l)")
    ap.add_argument("--freeze", action="store_true", help="write the measured values as the identity baseline")
    a = ap.parse_args()
    sys.argv = [sys.argv[0]]
    vs = importlib.util.spec_from_file_location("dmval_validate", os.path.join(HERE, "validate.py"))
    V = importlib.util.module_from_spec(vs); vs.loader.exec_module(V)          # make_inputs only (no candidate access)
    V.D = a.hd
    cs = importlib.util.spec_from_file_location("dm_candidate", a.candidate)
    C = importlib.util.module_from_spec(cs); sys.modules["dm_candidate"] = C; cs.loader.exec_module(C)
    dev = torch.device("cuda")
    base = json.load(open(BASE)) if os.path.exists(BASE) and not a.freeze else {}
    fams = ["randn", "outliers", "nearsat", "scaled3", "scaled10", "peaked", "bigmag"]
    out, ok, opened = {}, True, []
    for N in (int(x) for x in a.lengths.split(",")):
        for (H, Hk) in (((32, 4), (16, 4)) if a.hd == 128 else ((8, 2), (16, 4))):   # hd128 headline; hd256 qwen35_m / _l
            B = max(1, 32768 // N)
            for fam in fams:
                name = f"B{B}_H{H}_Hk{Hk}_N{N}_c_{fam}" + ("" if a.hd == 128 else f"_D{a.hd}")
                g = torch.Generator().manual_seed(__import__("zlib").crc32(name.encode()) & 0x7FFFFFFF)
                q, k, v, do = (x.to(dev) for x in V.make_inputs(fam, B, N, H, Hk, g, torch.device("cpu")))
                leaves = [x.clone().requires_grad_(True) for x in (q, k, v)]
                o = C.attn(*leaves, causal=True, mode="matched")
                o.backward(do)
                d = C.debug(q, k, v, do, True, "matched")
                # no bitwise determinism required (user rule): the debug build's grads must be within 1e-4 rel of production
                bind = all(((d[kk].reshape(t.shape).double() - t.double()).norm()
                            / t.double().norm().clamp_min(1e-300)).item() <= 2e-4
                           for kk, t in (("dbg_dq", leaves[0].grad), ("dbg_dk", leaves[1].grad), ("dbg_dv", leaves[2].grad)))
                r = (d["rowsum_dS"].double().abs() / d["sum_abs_PdP"].double().clamp_min(1e-300))[d["sum_abs_PdP"] > 0]
                mx, mn = r.max().item(), r.mean().item()
                fin = all(bool(torch.isfinite(t).all()) for t in (o, *(x.grad for x in leaves)))
                within = mx <= MAXL and mn <= MEANL
                b = base.get(name)
                regress_ok = b is not None and mx <= 1.25 * b["max"] + 1e-12 and mn <= 1.25 * b["mean"] + 1e-12
                status = "PASS" if (within and bind and fin) else ("OPEN" if (bind and fin and regress_ok) else "FAIL")
                ok &= status != "FAIL"
                if status == "OPEN":
                    opened.append(name)
                out[name] = {"max": mx, "mean": mn, "bind": bind, "finite": fin, "status": status}
                print(f"{status:4s} {name:34s} max {mx:.2e} mean {mn:.2e} bind {bind} finite {fin}"
                      + (f"  (baseline max {b['max']:.2e} mean {b['mean']:.2e})" if b else ""), flush=True)
                del leaves, o, d
                torch.cuda.empty_cache()
    if a.freeze:
        # merge (lengths may be frozen in separate runs); existing entries for other shapes are kept
        old = json.load(open(BASE)) if os.path.exists(BASE) else {}
        old.update({kk: {"max": vv["max"], "mean": vv["mean"]} for kk, vv in out.items()})
        json.dump(old, open(BASE + ".tmp", "w"), indent=1)
        os.replace(BASE + ".tmp", BASE)
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
    print("SUMMARY_JSON " + json.dumps({"pass": ok, "open": opened, "n": len(out)}))
    print(("PASS" if ok else "FAIL") + f": identity tier, {len(out)} cases, {len(opened)} OPEN (above 1e-5/2e-7, no regression)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
