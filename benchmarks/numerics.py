"""Numerical certification of an output-changing candidate (Q2 / Q3; Hao's audit direction 2).  Schema v2.

    python benchmarks/numerics.py capture --root <build root> [--dbg] --route default --out c.json [--expect-sha <sha256>]
    python benchmarks/numerics.py check --route default --base b.json --prev p.json --prev-dbg pd.json \\
        --cand c.json --cand-repeat c2.json --cand-dbg d.json --cand-dbg-repeat d2.json \\
        --defects m.json [--ws same|differs] [--rel-tol 0] [--report r.json]
    python benchmarks/numerics.py manifest --base b.json --base-dbg bd.json --route default --out m.json   # freeze defects

capture: every fixed case of the route (equivalence.SHAPES x MODES: 72) through the public API, with the loaded build's
identities (library sha256, Python-source hash, build stamp / flags), the case descriptors, input and reference
identities, relative-L2 errors of O / dQ / dK / dV against an fp32 reference (TF32 off), and per output the hashes of
its bytes, NaN mask, Inf mask and finite values.  With --dbg (a DM_DBG=1 build) every backward group call of the API is
routed to the twin-taking bwd_fp8_t_dbg with the same arguments (the certified gradients come from the launch whose
workspaces are captured) and per group: the exact dS' / psi schema derived from the descriptor and route, visited-tile
coverage, finite decoded codes, the zero-psi sentinel per logical key half, region hashes (psi and codes per key half;
one psi is broadcast to both halves), the row-partial schema, and the matched residuals:
  - residual_plan (the plan's AC-2.3 measurement, "plan-v1"): max over rows of |sum_j t_j| / sum_j |t_j|,
    t_j = P^_j (dP_j - delta) from the kernel's partials (fp64, n-tile then warpgroup ascending), with the worst row, its
    numerator / denominator and its key count.  It is ill-conditioned (a row with one nonzero term, e.g. a causal row 0,
    gives exactly 1 for any rounding residue), so the accepted build cannot meet a fixed round-off bound on it; the
    check applies NON-REGRESSION vs the previous accepted revision per (case, group) and reports the values.
  - residual_centered (a labelled DIAGNOSTIC, "centered-v1", NOT the plan's AC-2.3 predicate): |sum t| /
    (sum |t| + 2^8 |delta|), bound sk * 2^-23 derived by the checker.  Making it an acceptance predicate needs the
    user's authorization (open question).
check: exits nonzero unless every predicate holds; known defects only through the frozen, identity-bound manifest.
"""
import argparse, hashlib, json, math, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import equivalence as EQ  # noqa: E402

SCHEMA = 2
ROUTES4 = ("default", "kv", "head", "streams2")
OUTPUTS = ("O", "dQ", "dK", "dV")
RESIDUAL_ULP_PER_KEY = 2.0 ** -23
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _hash_tensor(t):
    import torch
    t = t.detach().contiguous().cpu()
    if t.dtype in (torch.bfloat16, torch.float16):
        t = t.view(torch.int16)
    elif t.dtype == torch.bool:
        t = t.to(torch.uint8)
    elif t.dtype.is_floating_point and t.element_size() == 1:
        t = t.view(torch.uint8)
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()


def output_signature(x):
    """Bytes, NaN-mask, Inf-mask and finite-value hashes of one output (element-level non-finite pattern)."""
    import torch
    f = x.float()
    nan, inf = torch.isnan(f), torch.isinf(f)
    fin = torch.where(torch.isfinite(f), f, torch.zeros_like(f))
    return {"bytes": _hash_tensor(x), "nan_mask": _hash_tensor(nan), "inf_mask": _hash_tensor(inf),
            "finite": _hash_tensor(fin), "n_nan": int(nan.sum()), "n_inf": int(inf.sum())}


def residual_bound(sk):
    return sk * RESIDUAL_ULP_PER_KEY


def row_residual(rows, rowsabs, delta=None):
    """(max ratio, zero-denominator rows with a nonzero numerator, worst-row detail).  rows / rowsabs [BH, nbn, 2, sq]
    fp32 partials, reduced per row in fp64 (n-tile ascending, then warpgroup).  delta None: the plan's ratio
    |sum t| / sum |t|; delta [BH, sq]: the centered diagnostic |sum t| / (sum |t| + 2^8 |delta|)."""
    import torch
    r = rows.double().permute(0, 3, 1, 2).reshape(rows.shape[0], rows.shape[3], -1)
    a = rowsabs.double().permute(0, 3, 1, 2).reshape(rows.shape[0], rows.shape[3], -1)
    num, den = r.sum(-1).abs(), a.sum(-1)
    if delta is not None:
        den = den + 256.0 * delta.double().abs().reshape(den.shape)
    zero = den == 0
    bad_zero = int((zero & (num != 0)).sum())
    ratio = torch.where(zero, torch.zeros_like(num), num / torch.where(zero, torch.ones_like(den), den))
    if ratio.numel() == 0:
        return 0.0, bad_zero, None
    i = int(ratio.argmax())
    bh, row = divmod(i, ratio.shape[1])
    return float(ratio.max()), bad_zero, {"bh": bh, "row": row, "num": float(num[bh, row]), "den": float(den[bh, row])}


def visited_mask(nbm, nbn, kbn, causal):
    import torch
    mb = torch.arange(nbm).view(-1, 1)
    nb = torch.arange(nbn).view(1, -1)
    return (nb * kbn <= 64 * mb + 63) if causal else torch.ones(nbm, nbn, dtype=torch.bool)


def expected_group_schema(desc, route, psi_halves=1):
    """Derived from the case descriptor and the route alone: heads per backward call, group count, dS' / psi /
    row-partial shapes and the visited-tile count."""
    D, B, H, Hk, N = desc["D"], desc["B"], desc["H"], desc["Hk"], desc["N"]
    hp = EQ.heads_per_call(route, H, Hk)
    kbn = 128 if D == 128 else 64
    nbm, nbn = N // 64, N // kbn
    nvis = int(visited_mask(nbm, nbn, kbn, desc["attn"] == "causal").sum())
    return {"heads": hp, "groups": H // hp, "kbn": kbn, "ds8": [B * hp, nbm, nbn, 64, kbn],
            "psi": [B * hp, nbm, nbn, psi_halves], "rows": [B * hp, nbn, 2, N], "visited_tiles": nvis}


def check_workspace(ds8, psi, rows, rowsabs, sch, causal):
    """One group's realized dS' / psi / row partials against the derived schema: (problems, region hashes per logical
    key half, visited tiles)."""
    import torch
    probs = []
    if list(ds8.shape) != sch["ds8"] or ds8.dtype != torch.float8_e4m3fn:
        return [f"dS' {tuple(ds8.shape)} {ds8.dtype}, expected {sch['ds8']} float8_e4m3fn"], None, 0
    if list(psi.shape) != sch["psi"] or psi.dtype != torch.float32:
        return [f"psi {tuple(psi.shape)} {psi.dtype}, expected {sch['psi']} float32"], None, 0
    for nm, t in (("rows", rows), ("rowsabs", rowsabs)):
        if list(t.shape) != sch["rows"] or t.dtype != torch.float32 or not torch.isfinite(t).all():
            probs.append(f"{nm} {tuple(t.shape)} {t.dtype} is not the finite {sch['rows']} fp32 schema")
    if not probs and ((rowsabs < 0).any() or (rows.abs() > rowsabs * (1 + 1e-5) + 1e-30).any()):
        probs.append("row partials inconsistent: |rows| > rowsabs or a negative absolute partial")
    kbn = sch["kbn"]
    m = visited_mask(sch["ds8"][1], sch["ds8"][2], kbn, causal)
    nvis = int(m.sum())
    if nvis != sch["visited_tiles"] or sch["ds8"][0] == 0:
        probs.append(f"visited tiles {nvis} x BH {sch['ds8'][0]}, expected {sch['visited_tiles']}")
    codes = ds8.contiguous().view(torch.uint8)[:, m]          # [BH, visited, 64, kbn]
    p = psi[:, m]                                             # [BH, visited, halves]
    if not torch.isfinite(p).all() or (p < 0).any():
        probs.append("non-finite or negative psi on a visited tile")
    if ((codes & 0x7F) == 0x7F).any():
        probs.append("NaN e4m3 code on a visited tile")
    halves = p.shape[-1]
    regions = []
    for hf in range(2):
        ph = p[..., hf if halves == 2 else 0]
        ch = codes[..., hf * kbn // 2:(hf + 1) * kbn // 2]
        if ((ch[ph == 0] & 0x7F) != 0).any():
            probs.append(f"a zero-psi key half {hf} carries nonzero dS' codes")
        regions.append(hashlib.sha256(ph.contiguous().numpy().tobytes() + ch.contiguous().numpy().tobytes()).hexdigest())
    return probs, regions, nvis


class _DbgOps:
    """bwd_fp8_t -> bwd_fp8_t_dbg (same arguments), recording the launch's workspaces per group."""

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


def build_identity(pkg_dir, so):
    """Loaded library sha256, the package's Python-source hash, the library's build stamp (if any) and flags."""
    import glob, importlib.util
    py = hashlib.sha256()
    for f in sorted(glob.glob(os.path.join(pkg_dir, "*.py"))):
        if os.path.basename(f) == "_build_info.py":
            continue
        py.update(os.path.basename(f).encode()); py.update(open(f, "rb").read())
    stamp = None
    for cand in (os.path.join(os.path.dirname(so), "_build_info.py"), os.path.join(pkg_dir, "_build_info.py")):
        if os.path.isfile(cand):
            spec = importlib.util.spec_from_file_location("_bi", cand); m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m); stamp = m.BUILD_INFO
            break
    return {"library_sha256": EQ.sha256(so), "library": so, "package": pkg_dir, "python_sha256": py.hexdigest(),
            "code_sha": (stamp or {}).get("code_sha"), "flags": (stamp or {}).get("flags"),
            "dbg_build": (stamp or {}).get("dbg"), "prof": (stamp or {}).get("prof"), "trace": (stamp or {}).get("trace")}


def capture(a):
    os.environ.update(EQ.ROUTES[a.route])
    os.environ.pop("DM_SO", None)
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    import delta_matching as dm
    from delta_matching import split as dm_split
    dm.core._ops()
    so = os.path.realpath(dm.core.loaded_library())
    pkg = os.path.realpath(os.path.join(a.root, "delta_matching"))
    if os.path.realpath(os.path.dirname(dm.__file__)) != pkg:
        raise SystemExit(f"imported {dm.__file__}, expected the package of {a.root}")
    ident = build_identity(pkg, so)
    if a.expect_sha and ident["library_sha256"] != a.expect_sha:
        raise SystemExit(f"loaded library {ident['library_sha256']}, expected {a.expect_sha}")
    if a.dbg and not hasattr(torch.ops.delta_matching, "bwd_fp8_t_dbg"):
        raise SystemExit("--dbg needs a DM_DBG=1 build with bwd_fp8_t_dbg")
    cases = EQ.selected_cases(a.route)
    out = {"schema": SCHEMA, "route": a.route, "dbg": a.dbg, "identity": ident,
           "resolved_route": {"groups": dm_split._BWD_GROUPS, "streams": dm_split._BWD_STREAMS},
           "descriptors_sha256": hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest(),
           "residual_versions": {"plan": "plan-v1", "centered": "centered-v1"}, "cases": {}}
    for key, desc in cases:
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
             "outputs": {n_: output_signature(g) for n_, g in zip(OUTPUTS, got)},
             "reference_finite": bool(all(torch.isfinite(r).all() for r in ref)),
             "input_id": hashlib.sha256("".join(_hash_tensor(x) for x in (q, k, v, do)).encode()).hexdigest(),
             "ref_id": hashlib.sha256("".join(_hash_tensor(r) for r in ref).encode()).hexdigest()}
        if a.dbg:
            sch = expected_group_schema(desc, a.route)
            groups = []
            for g in rec["groups"]:
                probs, regions, nvis = check_workspace(g["ds8"].cpu(), g["psi"].cpu(), g["rows"].cpu(), g["rowsabs"].cpu(),
                                                       sch, g["causal"])
                ent = {"heads": g["h"], "problems": probs, "region_hashes": regions, "visited_tiles": nvis}
                if desc["mode"] == "matched":
                    rp, zp, wp = row_residual(g["rows"].cpu(), g["rowsabs"].cpu())
                    rc, zc, _ = row_residual(g["rows"].cpu(), g["rowsabs"].cpu(), g["delta"].cpu())
                    if wp is not None:
                        wp["keys"] = min(g["sk"], wp["row"] + 1) if g["causal"] else g["sk"]
                    ent.update(residual_plan=rp, residual_plan_worst=wp, zero_den_nonzero_num=zp + zc,
                               residual_centered=rc)
                groups.append(ent)
            c["groups"] = groups
        out["cases"][key] = c
        print(f"{key:40s} " + " ".join(f"{n_} {e:.3e}" for n_, e in c["errors"].items())
              + (f" groups {len(c.get('groups', []))}" if a.dbg else ""), flush=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {a.out}: {len(out['cases'])} cases, route {a.route}, dbg {a.dbg}, library {so}")


def _fin(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def _load(path, route, label, dbg, probs, manifest):
    """Schema / identity / coverage / finiteness / workspace validation of one capture."""
    if not path:
        probs.append(f"{label}: capture required")
        return None
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception as e:
        probs.append(f"{label}: cannot read {path}: {e}")
        return None
    if d.get("schema") != SCHEMA or d.get("route") != route or bool(d.get("dbg")) != dbg:
        probs.append(f"{label}: schema / route / dbg mismatch ({d.get('schema')}, {d.get('route')}, {d.get('dbg')})")
        return None
    ident = d.get("identity") or {}
    if not HEX64.match(str(ident.get("library_sha256", ""))) or not HEX64.match(str(ident.get("python_sha256", ""))):
        probs.append(f"{label}: missing library / Python-source identity")
    if dbg and ident.get("dbg_build") is not True:
        probs.append(f"{label}: a DBG capture from a library whose stamp is not a DBG build")
    cases = EQ.selected_cases(route)
    if d.get("descriptors_sha256") != hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest():
        probs.append(f"{label}: case descriptors differ from the independently selected set")
    have = d.get("cases", {})
    want = [k for k, _ in cases]
    miss = [k for k in want if k not in have]
    if miss or len(have) != len(want):
        probs.append(f"{label}: {len(miss)} missing cases (e.g. {miss[:2]}), {len(have)} present, {len(want)} required")
    known = manifest.get("cases", {})
    for k, desc in cases:
        c = have.get(k)
        if c is None:
            continue
        if not c.get("reference_finite"):
            probs.append(f"{label} {k}: non-finite fp32 reference")
        errs, outs = c.get("errors", {}), c.get("outputs", {})
        if set(errs) != set(OUTPUTS) or set(outs) != set(OUTPUTS):
            probs.append(f"{label} {k}: missing errors / output signatures")
            continue
        for o in OUTPUTS:
            kd = known.get(k, {}).get("outputs", {}).get(o)
            sig = {x: outs[o].get(x) for x in ("nan_mask", "inf_mask", "finite")}
            clean = _fin(errs[o]) and errs[o] >= 0 and not outs[o].get("n_nan") and not outs[o].get("n_inf")
            if kd is not None:   # a frozen known defect: the exact element-level signature, or repaired if allowed
                if sig != {x: kd[x] for x in ("nan_mask", "inf_mask", "finite")} and not (manifest.get("allow_repair") and clean):
                    probs.append(f"{label} {k} {o}: differs from the frozen defect signature")
            elif not clean:
                probs.append(f"{label} {k} {o}: non-finite error / output ({errs[o]}, nan {outs[o].get('n_nan')}, "
                             f"inf {outs[o].get('n_inf')})")
        if dbg:
            sch = expected_group_schema(desc, route)
            gs = c.get("groups") or []
            if len(gs) != sch["groups"] or any(g.get("heads") != sch["heads"] for g in gs):
                probs.append(f"{label} {k}: groups {[g.get('heads') for g in gs]}, expected {sch['groups']} x {sch['heads']} heads")
            allowed_ws = set(known.get(k, {}).get("workspace_problems", []))
            for g in gs:
                for p in g.get("problems", []):
                    if p not in allowed_ws:
                        probs.append(f"{label} {k}: workspace {p}")
                if g.get("visited_tiles") != sch["visited_tiles"] or len(g.get("region_hashes") or []) != 2 or \
                        not all(HEX64.match(str(h)) for h in g.get("region_hashes") or []):
                    probs.append(f"{label} {k}: coverage / region hashes invalid")
                if desc["mode"] == "matched":
                    rp, rc = g.get("residual_plan"), g.get("residual_centered")
                    if not (_fin(rp) and 0 <= rp <= 1 and _fin(rc) and rc >= 0) or g.get("zero_den_nonzero_num") is None:
                        probs.append(f"{label} {k}: residual record missing / out of range ({rp}, {rc})")
                    elif g["zero_den_nonzero_num"]:
                        probs.append(f"{label} {k}: {g['zero_den_nonzero_num']} zero-denominator rows with a nonzero sum")
                    elif rc > residual_bound(desc["N"]) and k not in known:
                        probs.append(f"{label} {k}: centered residual diagnostic {rc:.3e} > derived bound "
                                     f"{residual_bound(desc['N']):.3e}")
    return d


def check(a):
    """1. every capture valid (_load: schema, identities, descriptors, 72 cases, finite references / errors / outputs
    except frozen defects, DBG group schema / coverage / residual records); 2. the same input and reference identities
    everywhere; 3. no per-case / output error increase vs base and prev (rel_tol 0 by default); 4. candidate production
    and its repeat: identical identity and outputs; both DBG runs: one library, outputs == production, identical region
    hashes; the DBG build from the candidate's Python source, code_sha and flags; 5. plan residual of every matched group
    not above the previous accepted revision's; 6. --ws same: every region hash equal to the previous DBG capture."""
    probs = []
    manifest = {}
    if a.defects:
        try:
            manifest = json.load(open(a.defects))
        except Exception as e:
            probs.append(f"defect manifest unreadable: {e}")
    caps = {}
    for tag, path, dbg in (("base", a.base, False), ("prev", a.prev, False), ("cand", a.cand, False),
                           ("cand-repeat", a.cand_repeat, False), ("cand-dbg", a.cand_dbg, True),
                           ("cand-dbg-repeat", a.cand_dbg_repeat, True), ("prev-dbg", a.prev_dbg, True)):
        caps[tag] = _load(path, a.route, tag, dbg, probs, manifest)
    if manifest and caps.get("base") and manifest.get("base_library_sha256") != caps["base"]["identity"]["library_sha256"]:
        probs.append("the defect manifest is not bound to this baseline library")
    if manifest and manifest.get("route") not in (None, a.route):
        probs.append("the defect manifest is for another route")
    report = {"cases": {}, "known_defects": sorted(manifest.get("cases", {})),
              "residual_versions": {"plan": "plan-v1", "centered": "centered-v1 (diagnostic)"}}
    if all(caps.values()):
        base, prev, cand, rep, cd, cd2, pd = (caps[t] for t in ("base", "prev", "cand", "cand-repeat", "cand-dbg",
                                                                 "cand-dbg-repeat", "prev-dbg"))
        ci, di = cand["identity"], cd["identity"]
        if ci != rep["identity"]:
            probs.append("candidate production repeat ran a different build")
        if di != cd2["identity"]:
            probs.append("candidate DBG repeat ran a different build")
        if ci.get("python_sha256") != di.get("python_sha256"):
            probs.append("candidate DBG capture used a different Python source")
        if not ci.get("code_sha") or ci.get("code_sha") != di.get("code_sha"):
            probs.append("candidate DBG build is not from the candidate's source (code_sha missing or different)")
        if ci.get("flags") != di.get("flags"):
            probs.append("candidate DBG build has different numerical flags")
        for k, desc in EQ.selected_cases(a.route):
            if any(k not in x.get("cases", {}) for x in caps.values()):
                continue   # already reported as a missing case by _load
            rows = [x["cases"][k] for x in caps.values()]
            if len({r.get("input_id") for r in rows}) != 1 or len({r.get("ref_id") for r in rows}) != 1:
                probs.append(f"{k}: input / reference identities differ across captures")
            row = {}
            ce = cand["cases"][k]["errors"]
            for rn, ref in (("base", base), ("prev", prev)):
                for o in OUTPUTS:
                    r = ref["cases"][k]["errors"][o]
                    if _fin(r) and _fin(ce[o]):
                        if ce[o] > r * (1 + a.rel_tol):
                            probs.append(f"{k} {o}: error {ce[o]:.6e} > {rn} {r:.6e}")
                        row[f"{o}_vs_{rn}"] = ce[o] / r if r > 0 else (0.0 if ce[o] == 0 else float("inf"))
            out = lambda x: {o: x["cases"][k]["outputs"][o]["bytes"] for o in OUTPUTS}
            if out(cand) != out(rep):
                probs.append(f"{k}: candidate outputs differ between independent runs (nondeterministic)")
            if out(cd) != out(cand) or out(cd2) != out(cand):
                probs.append(f"{k}: a DBG-build run's outputs differ from the production candidate")
            rh = lambda x: [g.get("region_hashes") for g in x["cases"][k].get("groups", [])]
            if rh(cd) != rh(cd2):
                probs.append(f"{k}: candidate workspaces differ between independent runs")
            if a.ws == "same" and rh(cd) != rh(pd):
                probs.append(f"{k}: workspaces declared unchanged differ from the previous revision")
            if desc["mode"] == "matched":
                cg, pg = cd["cases"][k].get("groups", []), pd["cases"][k].get("groups", [])
                if len(cg) == len(pg):
                    for i, (x, y) in enumerate(zip(cg, pg)):
                        if _fin(x.get("residual_plan")) and _fin(y.get("residual_plan")) and \
                                x["residual_plan"] > y["residual_plan"] * (1 + a.rel_tol):
                            probs.append(f"{k} group {i}: plan residual {x['residual_plan']:.3e} above the previous "
                                         f"revision's {y['residual_plan']:.3e}")
                    row["residual_plan_max"] = max((x.get("residual_plan", 0) for x in cg), default=0.0)
                    row["residual_centered_max"] = max((x.get("residual_centered", 0) for x in cg), default=0.0)
            report["cases"][k] = row
    report["ws_declaration"] = a.ws
    report["problems"] = probs
    report["result"] = ("FAIL" if probs else ("PASS (qualified by known defects: %d cases)" % len(report["known_defects"])
                                               if report["known_defects"] else "PASS"))
    if a.report:
        with open(a.report, "w") as f:
            json.dump(report, f, indent=1)
    for p in probs[:40]:
        print("FAIL:", p)
    print(f"numerics check route {a.route}: {report['result']} ({len(probs)} problems)")
    return 0 if not probs else 1


def make_manifest(a):
    """Freeze the baseline's known defects (non-finite outputs, workspace problems) with exact element-level
    signatures, bound to the baseline library sha256 and the route."""
    base = json.load(open(a.base)); bdbg = json.load(open(a.base_dbg)) if a.base_dbg else None
    m = {"route": a.route, "base_library_sha256": base["identity"]["library_sha256"], "allow_repair": False, "cases": {}}
    for k, c in base["cases"].items():
        bad = {o: {x: c["outputs"][o][x] for x in ("nan_mask", "inf_mask", "finite", "n_nan", "n_inf")}
               for o in OUTPUTS if c["outputs"][o]["n_nan"] or c["outputs"][o]["n_inf"] or not _fin(c["errors"][o])}
        ws = sorted({p for g in (bdbg or {}).get("cases", {}).get(k, {}).get("groups", []) for p in g.get("problems", [])})
        if bad or ws:
            m["cases"][k] = {"outputs": bad, "workspace_problems": ws}
    json.dump(m, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}: {len(m['cases'])} known-defect cases on route {a.route}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("capture")
    c.add_argument("--root", required=True); c.add_argument("--route", choices=sorted(EQ.ROUTES), default="default")
    c.add_argument("--dbg", action="store_true"); c.add_argument("--out", required=True)
    c.add_argument("--expect-sha", default=None)
    k = sp.add_parser("check")
    k.add_argument("--route", choices=sorted(EQ.ROUTES), default="default")
    for n in ("base", "prev", "cand", "cand-repeat", "cand-dbg", "cand-dbg-repeat", "prev-dbg", "report", "defects"):
        k.add_argument(f"--{n}", default=None)
    k.add_argument("--ws", choices=("same", "differs"), default="differs")
    k.add_argument("--rel-tol", type=float, default=0.0)
    m = sp.add_parser("manifest")
    m.add_argument("--base", required=True); m.add_argument("--base-dbg", default=None)
    m.add_argument("--route", choices=sorted(EQ.ROUTES), default="default"); m.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "capture":
        capture(a)
        return 0
    if a.cmd == "manifest":
        make_manifest(a)
        return 0
    return check(a)


if __name__ == "__main__":
    sys.exit(main())
