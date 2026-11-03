"""residual-v5 error clause (TASK.md): per case at most 1e-5 relative above the reference, and the mean over cases does not
rise; no stale / full allowance.  Inputs: numerics.py production captures data/<tag>/err_<route>.json.

    python err_audit.py <cand_tag> <ref_tag> [--out report.json]

Reading A: err_cand <= err_ref + 1e-5 (absolute in relative-error units, the v4 reading).
Reading B: err_cand <= err_ref * (1 + 1e-5).
Mean: per route and output, mean over all 72 cases (finite errors; non-finite pairs listed), cand vs ref, any rise.
"""
import argparse, json, math, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROUTES = ("default", "kv", "head", "streams2")
OUTS = ("O", "dQ", "dK", "dV")
TOL = 1e-5
# round 24 (Hao): relative clause: matched <= 1.01 x the original's error, stale / full <= 1.10 x; mean <= 1.01 x
REL_MATCHED, REL_OTHER, REL_MEAN = 0.01, 0.10, 0.01


def audit(cand, ref):
    rep = {"cand": cand, "ref": ref, "routes": {}}
    for r in ROUTES:
        C = json.load(open(os.path.join(HERE, "data", cand, f"err_{r}.json")))
        R = json.load(open(os.path.join(HERE, "data", ref, f"err_{r}.json")))
        ent = {"cand_library": C["identity"]["library_sha256"], "ref_library": R["identity"]["library_sha256"],
               "problems": [], "fail_A": [], "fail_B": [], "fail_rel": [], "fail_mean_rel": [], "nonfinite": [], "rows": [], "mean": {}}
        if set(C["cases"]) != set(R["cases"]):
            ent["problems"].append("case sets differ")
        sums = {o: [0.0, 0.0, 0] for o in OUTS}
        for k in sorted(set(C["cases"]) & set(R["cases"])):
            c, g = C["cases"][k], R["cases"][k]
            if c["input_id"] != g["input_id"] or c["ref_id"] != g["ref_id"]:
                ent["problems"].append(f"{k}: input / reference identity differs")
            for o in OUTS:
                ec, eo = c["errors"][o], g["errors"][o]
                if not (isinstance(ec, (int, float)) and isinstance(eo, (int, float)) and math.isfinite(ec) and math.isfinite(eo)):
                    ent["nonfinite"].append({"case": k, "out": o, "cand": ec, "ref": eo})
                    continue
                sums[o][0] += ec; sums[o][1] += eo; sums[o][2] += 1
                d = ec - eo
                row = {"case": k, "out": o, "cand": ec, "ref": eo, "diff": d, "rel": d / eo if eo > 0 else (0.0 if d == 0 else math.inf)}
                ent["rows"].append(row)
                if ec > eo + TOL:
                    ent["fail_A"].append(row)
                if ec > eo * (1 + TOL):
                    ent["fail_B"].append(row)
                if ec > eo * (1 + (REL_MATCHED if k.endswith("/matched") else REL_OTHER)):
                    ent["fail_rel"].append(row)
        for o, (sc, so, n) in sums.items():
            ent["mean"][o] = {"cand": sc / n, "ref": so / n, "rise": sc / n - so / n, "n": n}
            if sc / n > (so / n) * (1 + REL_MEAN):
                ent["fail_mean_rel"].append(o)
        ent["n_cases"] = len(C["cases"])
        ent["n_changed"] = sum(1 for x in ent["rows"] if x["diff"] != 0)
        ent["worst_abs"] = sorted(ent["rows"], key=lambda x: -x["diff"])[:5]
        ent["worst_rel"] = sorted(ent["rows"], key=lambda x: -x["rel"])[:5]
        del ent["rows"]
        rep["routes"][r] = ent
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cand"); ap.add_argument("ref"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rep = audit(a.cand, a.ref)
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1)
    for r, e in rep["routes"].items():
        rise = {o: m["rise"] for o, m in e["mean"].items()}
        print(f"{a.cand} vs {a.ref} {r}: changed {e['n_changed']}, fail A {len(e['fail_A'])}, fail B {len(e['fail_B'])}, "
              f"REL-CLAUSE fail {len(e['fail_rel'])} mean-fail {e['fail_mean_rel']}, worst rel {e['worst_rel'][0]['rel'] if e['worst_rel'] else 0:+.3e}, "
              f"non-finite {len(e['nonfinite'])}, problems {len(e['problems'])}, mean rise " +
              " ".join(f"{o} {v:+.2e}" for o, v in rise.items()))


if __name__ == "__main__":
    main()
