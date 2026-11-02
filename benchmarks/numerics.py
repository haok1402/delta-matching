"""Numerical certification of an output-changing candidate (Q2 / Q3; Hao's audit direction 2).

    python benchmarks/numerics.py capture --root <build root> --route default --out c.json            # production build
    python benchmarks/numerics.py capture --root <DM_DBG=1 build root> --dbg --route default --out d.json
    python benchmarks/numerics.py check --route default --base b.json --prev p.json --cand c.json --cand-repeat c2.json \\
        [--cand-dbg d.json --cand-dbg-repeat d2.json] [--prev-dbg pd.json --ws same|differs] [--rel-tol 0]

capture: every fixed case of the route (equivalence.SHAPES x MODES: 72; stress and former held-out included, no length or
input-kind limit) through the public API (equivalence.run_case: bnhd, bhnd and packed layouts), with
  - relative-L2 errors of O / dQ / dK / dV against an fp32 reference (TF32 disabled) in the common (B, N, H, D) layout,
  - byte hashes of those four outputs and an identity hash of the reference outputs (the same inputs and reference
    definition in every capture: the check requires equal reference identities across files),
  - with --dbg (a DM_DBG=1 build): every backward group call (unsplit or split, on its producing stream) of the API is
    routed from bwd_fp8_t to the twin-taking bwd_fp8_t_dbg with the SAME arguments, so the gradients the API returns
    come from the very launch whose workspaces are captured: per group the realized dS' codes / psi on the VISITED tiles
    only (kbn 128 at D128, 64 at D256; causal: nb*kbn <= 64*mb+63, full: all; unwritten storage excluded), their
    validation (shape, dtype, finite non-negative psi, zero-psi tiles carry zero codes), a workspace hash, and for
    matched cases the row residual |sum(dS)| / sum(|dS|) from the kernel's row / absolute-row partials, reduced per row
    in fp64 over (n-tile ascending, warpgroup ascending), against the declared bound sk * 2^-23 (zero numerator required
    when the denominator is zero).
check: exits nonzero unless every predicate holds (see check()).
"""
import argparse, hashlib, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import equivalence as EQ  # noqa: E402

SCHEMA = 1
OUTPUTS = ("O", "dQ", "dK", "dV")
RESIDUAL_ULP_PER_KEY = 2.0 ** -23


def _hash_tensor(t):
    import torch
    t = t.detach().contiguous().cpu()
    if t.dtype in (torch.bfloat16, torch.float16):
        t = t.view(torch.int16)
    elif t.dtype.is_floating_point and t.element_size() == 1:
        t = t.view(torch.uint8)
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()


def residual_bound(sk):
    return sk * RESIDUAL_ULP_PER_KEY


def row_residual(rows, rowsabs, delta=None):
    """max over (bh, row) of |sum_j t_j| / (sum_j |t_j| + 2^8 |delta|) with t_j = P^_j (dP_j - delta) the kernel's
    row / absolute-row partials (rows / rowsabs [BH, nbn, 2, sq] fp32, reduced in fp64 over n-tile ascending, then
    warpgroup ascending).  The 2^8 |delta| term (= |delta| * sum_j P^_j, P^ = 2^8 P) is the magnitude of the centering
    term: the plain |sum t| / sum |t| is exactly 1 for any row with a single nonzero term (a causal row 0 has one key),
    whatever the rounding residue.  delta [BH, sq] (None: the plain ratio, for tests).  Returns (max ratio, number of
    zero-denominator rows with a nonzero numerator)."""
    import torch
    r = rows.double().permute(0, 3, 1, 2).reshape(rows.shape[0], rows.shape[3], -1)
    a = rowsabs.double().permute(0, 3, 1, 2).reshape(rows.shape[0], rows.shape[3], -1)
    num, den = r.sum(-1).abs(), a.sum(-1)
    if delta is not None:
        den = den + 256.0 * delta.double().abs().reshape(den.shape)
    zero = den == 0
    bad_zero = int((zero & (num != 0)).sum())
    ratio = torch.where(zero, torch.zeros_like(num), num / torch.where(zero, torch.ones_like(den), den))
    return float(ratio.max()) if ratio.numel() else 0.0, bad_zero


def visited_mask(nbm, nbn, kbn, causal):
    import torch
    mb = torch.arange(nbm).view(-1, 1)
    nb = torch.arange(nbn).view(1, -1)
    return (nb * kbn <= 64 * mb + 63) if causal else torch.ones(nbm, nbn, dtype=torch.bool)


def check_workspace(ds8, psi, D, causal):
    """Validation of one group's realized dS' / psi on the visited tiles; returns (problems, hash, visited tiles)."""
    import torch
    probs = []
    kbn = 128 if D == 128 else 64
    if ds8.dim() != 5 or ds8.shape[3] != 64 or ds8.shape[4] != kbn:
        return [f"dS' shape {tuple(ds8.shape)}, expected [BH, nbm, nbn, 64, {kbn}]"], None, 0
    if psi.dim() != 4 or tuple(psi.shape[:3]) != tuple(ds8.shape[:3]) or psi.shape[3] < 1:
        return [f"psi shape {tuple(psi.shape)} does not match dS' {tuple(ds8.shape)}"], None, 0
    if ds8.element_size() != 1 or psi.dtype != torch.float32:
        probs.append(f"dtypes dS' {ds8.dtype} psi {psi.dtype}")
    m = visited_mask(ds8.shape[1], ds8.shape[2], kbn, causal)
    codes = ds8.contiguous().view(torch.uint8)[:, m]          # [BH, visited, 64, kbn]
    p = psi[:, m]                                             # [BH, visited, halves]
    if not torch.isfinite(p).all() or (p < 0).any():
        probs.append("non-finite or negative psi on a visited tile")
    zt = (p == 0).all(-1)                                     # zero-tile sentinel
    if ((codes[zt] & 0x7F) != 0).any():
        probs.append("a zero-psi tile carries nonzero dS' codes")
    h = hashlib.sha256(p.contiguous().numpy().tobytes() + codes.contiguous().numpy().tobytes()).hexdigest()
    return probs, h, int(m.sum())


class _DbgOps:
    """Forwards every attribute to the real op namespace; bwd_fp8_t -> bwd_fp8_t_dbg (same arguments), recording the
    launch's workspaces per group; returns that launch's (dq, dk, dv) to the API."""

    def __init__(self, real, rec):
        self._real, self._rec = real, rec

    def __getattr__(self, name):
        fn = getattr(self._real, name)
        if name != "bwd_fp8_t":
            return fn
        dbg, rec = getattr(self._real, "bwd_fp8_t_dbg"), self._rec

        def wrapped(*args):
            dq, dk, dv, rows, rowsabs, cks, ds8, psi = dbg(*args)
            rec["groups"].append({"h": args[0].shape[2], "causal": bool(args[-1]), "sk": args[2].shape[1],
                                  "rows": rows, "rowsabs": rowsabs, "ds8": ds8, "psi": psi,
                                  "delta": args[5].detach().clone()})
            return dq, dk, dv
        return wrapped


class dbg_routing:
    """Context manager: route the API's backward group calls through bwd_fp8_t_dbg; restore on every exit path."""

    def __init__(self, rec):
        self.rec = rec

    def __enter__(self):
        from delta_matching import attn as A, core as C
        self.saved = (A._ops_gt, C._ops)
        real_gt, real_core = self.saved
        A._ops_gt = lambda: _DbgOps(real_gt(), self.rec)
        C._ops = lambda: _DbgOps(real_core(), self.rec)
        return self.rec

    def __exit__(self, *exc):
        from delta_matching import attn as A, core as C
        A._ops_gt, C._ops = self.saved
        return False


def capture(a):
    os.environ.update(EQ.ROUTES[a.route])
    os.environ.pop("DM_SO", None)
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    import delta_matching as dm
    dm.core._ops()
    so = os.path.realpath(dm.core.loaded_library())
    pkg = os.path.realpath(os.path.join(a.root, "delta_matching"))
    if os.path.realpath(os.path.dirname(dm.__file__)) != pkg:
        raise SystemExit(f"imported {dm.__file__}, expected the package of {a.root}")
    if a.dbg and not hasattr(torch.ops.delta_matching, "bwd_fp8_t_dbg"):
        raise SystemExit("--dbg needs a DM_DBG=1 build with bwd_fp8_t_dbg")
    out = {"schema": SCHEMA, "route": a.route, "dbg": a.dbg, "library": so, "library_sha256": EQ.sha256(so),
           "cases": {}}
    for key, desc in EQ.selected_cases(a.route):
        rec = {"groups": []}
        if a.dbg:
            with dbg_routing(rec):
                t, _, _ = EQ.run_case(dm, desc)
        else:
            t, _, _ = EQ.run_case(dm, desc)
        torch.cuda.synchronize()
        q, k, v, do = EQ.make_inputs(desc["D"], desc["B"], desc["H"], desc["Hk"], desc["N"], desc["kind"], desc["seed"])
        ref = EQ.reference(q, k, v, do, desc["attn"] == "causal")
        got = [t["api.O"], t["api.dQ"], t["api.dK"], t["api.dV"]]
        c = {"errors": {n_: EQ.rel(g.cuda(), r) for n_, g, r in zip(OUTPUTS, got, ref)},
             "nonfinite_output": {n_: bool(not torch.isfinite(g.float()).all()) for n_, g in zip(OUTPUTS, got)},
             "nonfinite_reference": {n_: bool(not torch.isfinite(r).all()) for n_, r in zip(OUTPUTS, ref)},
             "out_hash": {n_: _hash_tensor(g) for n_, g in zip(OUTPUTS, got)},
             "ref_id": hashlib.sha256("".join(_hash_tensor(r) for r in ref).encode()).hexdigest()}
        if a.dbg:
            groups = []
            for g in rec["groups"]:
                probs, h, nvis = check_workspace(g["ds8"].cpu(), g["psi"].cpu(), desc["D"], g["causal"])
                ent = {"heads": g["h"], "problems": probs, "ws_hash": h, "visited_tiles": nvis}
                if desc["mode"] == "matched":
                    ratio, bad_zero = row_residual(g["rows"].cpu(), g["rowsabs"].cpu(), g["delta"].cpu())
                    ent.update(residual=ratio, residual_bound=residual_bound(g["sk"]), zero_den_nonzero_num=bad_zero)
                groups.append(ent)
            if sum(g["heads"] for g in groups) != desc["H"]:   # each group call covers all batches
                groups.append({"problems": [f"groups cover {sum(g['heads'] for g in groups)} query heads, "
                                            f"expected {desc['H']}"]})
            c["groups"] = groups
        out["cases"][key] = c
        print(f"{key:40s} " + " ".join(f"{n_} {e:.3e}" for n_, e in c["errors"].items())
              + (f" groups {len(c.get('groups', []))}" if a.dbg else ""), flush=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {a.out}: {len(out['cases'])} cases, route {a.route}, dbg {a.dbg}, library {so}")


def _load(path, route, label, dbg, probs, defect_cases=()):
    if not path:
        return None
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception as e:   # unreadable / malformed
        probs.append(f"{label}: cannot read {path}: {e}")
        return None
    if d.get("schema") != SCHEMA or d.get("route") != route or bool(d.get("dbg")) != dbg:
        probs.append(f"{label}: schema / route / dbg mismatch ({d.get('schema')}, {d.get('route')}, {d.get('dbg')})")
    want = [k for k, _ in EQ.selected_cases(route)]
    have = d.get("cases", {})
    miss = [k for k in want if k not in have]
    if miss or len(have) != len(want):
        probs.append(f"{label}: {len(miss)} missing cases (e.g. {miss[:2]}), {len(have)} present, {len(want)} required")
    for k in want:
        c = have.get(k, {})
        errs = c.get("errors", {})
        if set(errs) != set(OUTPUTS) or not all(isinstance(errs[o], (int, float)) for o in OUTPUTS if o in errs):
            probs.append(f"{label} {k}: missing errors {errs}")
        if set(c.get("out_hash", {})) != set(OUTPUTS) or not c.get("ref_id"):
            probs.append(f"{label} {k}: missing output hashes or reference identity")
        if dbg:
            gs = c.get("groups")
            if not gs:
                probs.append(f"{label} {k}: no captured backward groups")
            for g in gs or []:
                if g.get("problems") and k not in defect_cases:
                    probs.append(f"{label} {k}: workspace {g['problems'][0]}")
                if "residual" in g:
                    if not (isinstance(g["residual"], (int, float)) and math.isfinite(g["residual"])):
                        probs.append(f"{label} {k}: non-finite matched residual")
                    elif g["residual"] > g["residual_bound"]:
                        probs.append(f"{label} {k}: matched residual {g['residual']:.3e} > bound {g['residual_bound']:.3e}")
                    if g.get("zero_den_nonzero_num"):
                        probs.append(f"{label} {k}: {g['zero_den_nonzero_num']} zero-denominator rows with a nonzero sum")
                elif k.endswith("/matched"):
                    probs.append(f"{label} {k}: matched case without a residual")
    return d


def check(a):
    """Predicates (each failure is listed; any failure -> exit 1):
    1. every file: schema, route, the complete case set, finite errors, output hashes, reference identity;
    2. the same reference identity for every case in every file (identical inputs / reference definition);
    3. no per-case / per-output error increase of the candidate over the base and over the previous accepted revision
       (cand <= ref * (1 + rel_tol); rel_tol defaults to 0, the plan's strict reading);
    4. deterministic candidate: an independent-process repeat with byte-identical outputs;
    5. with DBG captures: DBG outputs byte-identical to the production candidate's; valid workspaces, matched
       residuals within bound; DBG repeat with identical workspace hashes; --ws same: workspace hashes equal to the
       previous revision's DBG capture."""
    probs = []
    base = _load(a.base, a.route, "base", False, probs)
    # BASELINE DEFECTS: cases where the frozen original baseline itself produces non-finite errors (e.g. NaN outputs);
    # listed in the report.  There the candidate must reproduce the baseline's non-finite pattern exactly (no new and no
    # silently "fixed" non-finite output goes unnoticed) and its finite errors must not increase; workspace findings in
    # those cases are reported, not fatal.  Every other case must be entirely finite.
    defect = {k: {o: not math.isfinite(e) for o, e in c["errors"].items()}
              for k, c in (base or {}).get("cases", {}).items()
              if any(not math.isfinite(e) for e in c.get("errors", {}).values())}
    prev = _load(a.prev, a.route, "prev", False, probs, defect)
    cand = _load(a.cand, a.route, "cand", False, probs, defect)
    rep = _load(a.cand_repeat, a.route, "cand-repeat", False, probs, defect)
    cdbg = _load(a.cand_dbg, a.route, "cand-dbg", True, probs, defect)
    cdbg2 = _load(a.cand_dbg_repeat, a.route, "cand-dbg-repeat", True, probs, defect)
    pdbg = _load(a.prev_dbg, a.route, "prev-dbg", True, probs, defect)
    if not (base and prev and cand and rep):
        probs.append("base, prev, cand and cand-repeat captures are all required")
    if a.cand_dbg and not a.cand_dbg_repeat:
        probs.append("a DBG capture needs its independent repeat")
    if a.ws == "same" and not (a.prev_dbg and a.cand_dbg):
        probs.append("--ws same needs --prev-dbg and --cand-dbg")
    report = {"cases": {}}
    if not probs:
        files = [x for x in (base, prev, cand, rep, cdbg, cdbg2, pdbg) if x]
        for k, _ in EQ.selected_cases(a.route):
            ids = {x["cases"][k]["ref_id"] for x in files}
            if len(ids) != 1:
                probs.append(f"{k}: reference identities differ across captures")
            ce = cand["cases"][k]["errors"]
            row = {}
            for nm, x in (("base", base), ("prev", prev), ("cand", cand), ("cand-repeat", rep)):
                pat = {o: not math.isfinite(e) for o, e in x["cases"][k]["errors"].items()}
                if k in defect:
                    if pat != defect[k]:
                        probs.append(f"{k}: {nm} non-finite pattern {pat} differs from the baseline defect {defect[k]}")
                elif any(pat.values()):
                    probs.append(f"{k}: {nm} has non-finite errors {x['cases'][k]['errors']}")
            for ref_name, ref in (("base", base), ("prev", prev)):
                for o in OUTPUTS:
                    r = ref["cases"][k]["errors"][o]
                    if not (math.isfinite(r) and math.isfinite(ce[o])):
                        continue   # covered by the non-finite predicates above
                    if ce[o] > r * (1 + a.rel_tol):
                        probs.append(f"{k} {o}: error {ce[o]:.6e} > {ref_name} {r:.6e}")
                    row[f"{o}_vs_{ref_name}"] = ce[o] / r if r > 0 else (0.0 if ce[o] == 0 else float("inf"))
            if cand["cases"][k]["out_hash"] != rep["cases"][k]["out_hash"]:
                probs.append(f"{k}: candidate outputs differ between independent runs (nondeterministic)")
            if cdbg:
                if cdbg["cases"][k]["out_hash"] != cand["cases"][k]["out_hash"]:
                    probs.append(f"{k}: DBG-build outputs differ from the production candidate")
                h1 = [g.get("ws_hash") for g in cdbg["cases"][k]["groups"]]
                h2 = [g.get("ws_hash") for g in cdbg2["cases"][k]["groups"]]
                if h1 != h2:
                    probs.append(f"{k}: candidate workspaces differ between independent runs")
                if a.ws == "same" and h1 != [g.get("ws_hash") for g in pdbg["cases"][k]["groups"]]:
                    probs.append(f"{k}: workspaces declared unchanged differ from the previous revision")
                res = [g["residual"] for g in cdbg["cases"][k]["groups"] if "residual" in g]
                if res:
                    row["residual_max"] = max(res)
            report["cases"][k] = row
    report["baseline_defect_cases"] = defect
    report["problems"] = probs
    report["result"] = "PASS" if not probs else "FAIL"
    if a.report:
        with open(a.report, "w") as f:
            json.dump(report, f, indent=1)
    for p in probs[:40]:
        print("FAIL:", p)
    print(f"numerics check route {a.route}: {report['result']} ({len(probs)} problems)")
    return 0 if not probs else 1


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("capture")
    c.add_argument("--root", required=True); c.add_argument("--route", choices=sorted(EQ.ROUTES), default="default")
    c.add_argument("--dbg", action="store_true"); c.add_argument("--out", required=True)
    k = sp.add_parser("check")
    k.add_argument("--route", choices=sorted(EQ.ROUTES), default="default")
    for n in ("base", "prev", "cand", "cand-repeat", "cand-dbg", "cand-dbg-repeat", "prev-dbg", "report"):
        k.add_argument(f"--{n}", default=None)
    k.add_argument("--ws", choices=("same", "differs"), default="differs")
    k.add_argument("--rel-tol", type=float, default=0.0)
    a = ap.parse_args()
    if a.cmd == "capture":
        capture(a)
        return 0
    return check(a)


if __name__ == "__main__":
    sys.exit(main())
