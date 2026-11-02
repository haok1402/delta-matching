"""residual-v5 (Hao, round 21) measurement and check.  STAGING tool (.local/rounds/r21-v5); not yet in benchmarks/.

    capture: python numerics_v5.py capture --root <tree with a DM_DBG=1 + v5 build> --suite cases|accept|stress \
                 [--route default|kv|head|streams2] [--seeds 0,1,2] --out <file.pt>
    check:   python numerics_v5.py check --orig <orig.pt> --cand <cand.pt> [--report r.json] [--worst 10]

Measured tensors (bwd_fp8_t_dbg5, DBG builds only; see apply_v5_dbg.py), per backward group call, per matched row r:
  - rows / rowsabs [BH, nbn, 2, s_q] fp32: per (n-tile, warpgroup) partials of t_rj = P^_rj (dP_rj - delta_r) and |t_rj|,
    with P^ = 2^8 exp(S - lse) pass 1's fp32 exp output and dP = fl(dP_acc * do8s * v8s) pass 1's fp32 dP: the dS terms
    BEFORE the k8s fold and the e4m3 quantization of dS' (the identity sum_j dS_rj = 0 is defined on these).
    num_r = |sum t|, den_r = sum |t| reduced in fp64 on the GPU.
  - rowsx [3, BH, nbn, 2, s_q] fp64: sum_j P^_rj dP_rj, sum_j P^_rj, sum_j |P^_rj dP_rj| from the SAME fp32 values (exact
    fp64 products, fp64 sums).  delta_recomputed_r = 2^-8 sum_j P^ dP (pass 0's matched definition: fp32 fma chains of
    P^ * fl(dP_acc * do8s * v8s) per 8 keys, x 2^-8, fp64 fold).  delta_used_r = the delta argument of the launch.
Predicate (no exemptions): every matched row: ratio_cand <= ratio_orig + k_r 2^-23 (ratio = num / den, 0 when den = 0;
k_r visible keys from the geometry); den = 0 requires num = 0; row layouts must match.
"""
import argparse, hashlib, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_BENCH = os.path.realpath(os.path.join(HERE, "..", "..", "..", "benchmarks"))
VERSION = "residual-v5"
ULP = 2.0 ** -23
TWO_M8 = 2.0 ** -8

# acceptance shapes: (name, B, H, Hk, N), D128 causal matched
ACCEPT = [("b2n8k_16_4", 2, 16, 4, 8192), ("b2n8k_32_4", 2, 32, 4, 8192),
          ("b1n64k_16_4", 1, 16, 4, 65536), ("b1n64k_32_4", 1, 32, 4, 65536)]
# stress: (name, kind, B, H, Hk, N, seeds)
STRESS = [("scale3", "scale3", 1, 16, 4, 8192, (0, 1)), ("scale10", "scale10", 1, 16, 4, 8192, (0, 1)),
          ("outliers", "outliers", 1, 16, 4, 8192, (0, 1)), ("onehot2", "onehot2", 1, 16, 4, 8192, (0, 1)),
          ("onehot1", "onehot1", 1, 16, 4, 8192, (0,)), ("long_scale3", "scale3", 1, 4, 1, 65536, (0,)),
          ("long_onehot2", "onehot2", 1, 4, 1, 65536, (0,))]


# ------------------------------------------------------------------------------------------------ reduction (GPU)
def reduce_group(rows, rowsabs, rowsx, delta):
    """Per row of one backward group: num, den, sum P^dP, sum P^, sum |P^dP| (fp64) and delta_used (fp32), all on the
    tensors' device; rows / rowsabs [BH, nbn, 2, sq] fp32, rowsx [3, BH, nbn, 2, sq] fp64, delta (B, h, sq) fp32."""
    bh, sq = rows.shape[0], rows.shape[3]
    num = rows.double().sum(dim=(1, 2)).abs()
    den = rowsabs.double().sum(dim=(1, 2))
    a, b, c = (rowsx[i].sum(dim=(1, 2)) for i in range(3))
    return {"num": num, "den": den, "pdp": a, "psum": b, "pdpabs": c, "du": delta.reshape(bh, sq).float()}


class _V5Ops:
    """bwd_fp8_t -> bwd_fp8_t_dbg5 with the same arguments; each group is reduced on the GPU at once (no ds8 / psi /
    partials kept), its per-row records moved to the host."""

    def __init__(self, real, rec):
        self._real, self._rec = real, rec

    def __getattr__(self, name):
        fn = getattr(self._real, name)
        if name != "bwd_fp8_t":
            return fn
        dbg, rec = getattr(self._real, "bwd_fp8_t_dbg5"), self._rec

        def wrapped(*args):
            dq, dk, dv, rows, rowsabs, cks, ds8, psi, rowsx = dbg(*args)
            g = reduce_group(rows, rowsabs, rowsx, args[5])
            del rows, rowsabs, cks, ds8, psi, rowsx
            rec["groups"].append({k: v.cpu() for k, v in g.items()})
            rec["causal"].append(bool(args[-1]))
            rec["sk"].append(int(args[2].shape[1]))
            return dq, dk, dv
        return wrapped


class v5_routing:
    def __init__(self, rec):
        self.rec = rec

    def __enter__(self):
        from delta_matching import attn as A, core as C
        self.saved = (A._ops_gt, C._ops)
        real_gt, real_core = self.saved
        A._ops_gt = lambda: _V5Ops(real_gt(), self.rec)
        C._ops = lambda: _V5Ops(real_core(), self.rec)
        return self.rec

    def __exit__(self, *exc):
        from delta_matching import attn as A, core as C
        A._ops_gt, C._ops = self.saved
        return False


def _cat(rec):
    import torch
    if not rec["groups"]:
        return None
    if len(set(rec["causal"])) != 1 or len(set(rec["sk"])) != 1:
        raise RuntimeError("groups of one case disagree on causal / sk")
    out = {k: torch.cat([g[k] for g in rec["groups"]], 0) for k in rec["groups"][0]}
    out["groups"] = len(rec["groups"])
    return out


def _hash(t):
    import torch
    t = t.detach().contiguous().cpu()
    if t.dtype in (torch.bfloat16, torch.float16):
        t = t.view(torch.int16)
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()


def make_inputs(kind, B, H, Hk, N, D, seed):
    """Stress / acceptance inputs (bf16, (B, N, h, D)); the equivalence kinds plus near-one-hot rows."""
    import torch
    import equivalence as EQ
    if not kind.startswith("onehot"):
        return EQ.make_inputs(D, B, H, Hk, N, kind, seed)
    alpha = float(kind[len("onehot"):])
    g = torch.Generator(device="cuda").manual_seed(seed)
    r = lambda *s: torch.randn(*s, device="cuda", dtype=torch.float32, generator=g)
    k, v, do = r(B, N, Hk, D), r(B, N, Hk, D), r(B, N, H, D)
    # each query row copies one visible key of its KV head (scaled by alpha): softmax mass concentrates on that key
    i = torch.arange(N, device="cuda")
    tgt = (torch.rand(B, N, H, device="cuda", generator=g) * (i + 1).view(1, N, 1)).long().clamp(max=N - 1)
    kv_of = torch.arange(H, device="cuda") // (H // Hk)
    q = alpha * k[torch.arange(B, device="cuda").view(B, 1, 1), tgt, kv_of.view(1, 1, H)]
    q = q + 0.05 * r(B, N, H, D)
    return q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16), do.to(torch.bfloat16)


_ROOT = None


def capture(a):
    global _ROOT
    import torch
    _ROOT = os.path.realpath(a.root)
    sys.path.insert(0, _ROOT)
    sys.path.insert(1, REPO_BENCH)   # ONE equivalence.py (the repo's) for every build: identical cases and inputs
    import equivalence as EQ
    if a.suite == "cases":
        os.environ.update(EQ.ROUTES[a.route])
    torch.backends.cuda.matmul.allow_tf32 = False
    import delta_matching as dm
    dm.core._ops()
    pkg = os.path.realpath(os.path.dirname(dm.__file__))
    if pkg != os.path.join(_ROOT, "delta_matching"):
        raise SystemExit(f"imported {pkg}, expected the package of {_ROOT}")
    if not hasattr(torch.ops.delta_matching, "bwd_fp8_t_dbg5"):
        raise SystemExit("needs a DM_DBG=1 build with the v5 extension (bwd_fp8_t_dbg5)")
    so = os.path.realpath(dm.core.loaded_library())
    out = {"version": VERSION, "suite": a.suite, "route": a.route, "library": so,
           "library_sha256": EQ.sha256(so), "package": pkg, "cases": {}}
    jobs = []
    if a.suite == "cases":
        for key, desc in EQ.selected_cases(a.route):
            jobs.append((key, desc))
    elif a.suite == "accept":
        for name, B, H, Hk, N in ACCEPT:
            for s in a.seeds:
                jobs.append((f"{name}/s{s}", {"kind": "gauss", "B": B, "H": H, "Hk": Hk, "N": N, "D": 128, "seed": s,
                                              "attn": "causal", "mode": "matched"}))
    else:
        for name, kind, B, H, Hk, N, seeds in STRESS:
            for s in seeds:
                jobs.append((f"{name}/s{s}", {"kind": kind, "B": B, "H": H, "Hk": Hk, "N": N, "D": 128, "seed": s,
                                              "attn": "causal", "mode": "matched"}))
    for key, desc in jobs:
        rec = {"groups": [], "causal": [], "sk": []}
        if a.suite == "cases":
            with v5_routing(rec):
                t, _, _ = EQ.run_case(dm, desc)
            hashes = {n: _hash(t[f"api.{n}"]) for n in ("O", "dQ", "dK", "dV")}
            del t
        else:
            # long rows: the backward splits per head (the DBG partials are per group: [h, nbn, 2, N])
            os.environ["DM_BWD_HEAD_GROUPS"] = "head" if desc["N"] > 16384 else "auto"
            from delta_matching import split as dm_split
            dm_split._BWD_GROUPS = os.environ["DM_BWD_HEAD_GROUPS"]
            q, k, v, do = make_inputs(desc["kind"], desc["B"], desc["H"], desc["Hk"], desc["N"], desc["D"], desc["seed"])
            leaves = [x.detach().requires_grad_(True) for x in (q, k, v)]
            with v5_routing(rec):
                o = dm.dm_attn_func(*leaves, causal=True, mode="matched")
                o.backward(do)
            torch.cuda.synchronize()
            hashes = {"O": _hash(o), "dQ": _hash(leaves[0].grad), "dK": _hash(leaves[1].grad), "dV": _hash(leaves[2].grad)}
            del q, k, v, do, leaves, o
            torch.cuda.empty_cache()
        ent = {"desc": desc, "outputs": hashes}
        if desc["mode"] == "matched":
            r = _cat(rec)
            if r is None:
                raise SystemExit(f"{key}: no backward group recorded")
            ent.update(r)
            ent["causal"], ent["sk"] = rec["causal"][0], rec["sk"][0]
        out["cases"][key] = ent
        print(f"{key:36s} groups {len(rec['groups'])} rows {ent['num'].numel() if 'num' in ent else 0}", flush=True)
    torch.save(out, a.out)
    print(f"wrote {a.out}: {len(out['cases'])} cases, suite {a.suite}, route {a.route}, library {so}")


# ------------------------------------------------------------------------------------------------ check (CPU)
def visible_keys(sq, sk, causal):
    import torch
    k = (torch.arange(sq) + 1).clamp(max=sk) if causal else torch.full((sq,), sk)
    return k.to(torch.float64)


def ratio_of(num, den):
    import torch
    pos = den > 0
    return torch.where(pos, num / torch.where(pos, den, torch.ones_like(den)), torch.zeros_like(num))


def abs_scale(x):
    """v5-abs per-row magnitude of the computation: S_r = sum_j |P^ dP| + |delta_used| sum_j P^ (the two products of
    t = P^ (dP - delta); P^ carries the 2^8 factor in t, rows and rowsx alike, so no rescaling)."""
    return x["pdpabs"] + x["du"].double().abs() * x["psum"]


def case_check(key, o, c, worst_n=5, env=None, form="v5"):
    """residual-v5 on one case: o / c = the original's / candidate's records (num, den, pdp, psum, pdpabs, du [R, sq])."""
    import torch
    res = {"problems": []}
    if o["num"].shape != c["num"].shape:
        res["problems"].append(f"{key}: row layouts differ {tuple(o['num'].shape)} vs {tuple(c['num'].shape)}")
        return res
    sq = c["num"].shape[1]
    keys = visible_keys(sq, c["sk"], c["causal"]).view(1, -1).expand_as(c["num"])
    ro, rc = ratio_of(o["num"], o["den"]), ratio_of(c["num"], c["den"])
    tol = keys * ULP
    worse = rc > ro + tol
    zbad = (c["den"] == 0) & (c["num"] != 0)
    res.update(rows=int(rc.numel()), worse=int(worse.sum()), zero_den_nonzero_num=int(zbad.sum()),
               cand_max=float(rc.max()), cand_median=float(rc.median()), orig_max=float(ro.max()),
               orig_median=float(ro.median()), better=int((rc < ro - tol).sum()), equal_rows=int((rc == ro).sum()),
               max_excess=float((rc - ro - tol).max()))
    # v5-abs: |sum t|_cand <= |sum t|_orig + c k_r 2^-23 S_r (S_r of the ORIGINAL's row; c = 1); minimum c needed
    S = abs_scale(o)
    unit = keys * ULP * S
    dn = c["num"] - o["num"]
    worse_abs = dn > unit
    need = torch.where(unit > 0, dn / torch.where(unit > 0, unit, torch.ones_like(unit)),
                       torch.where(dn > 0, torch.full_like(dn, float("inf")), torch.zeros_like(dn)))
    res["abs_worse_c1"] = int(worse_abs.sum())
    res["abs_c_needed"] = max(float(need.max()), 0.0)
    if bool(worse_abs.any()):
        i = int(need.flatten().argmax()); r_, q_ = divmod(i, sq)
        res["abs_worst"] = {"bh": r_, "row": q_, "keys": int(keys[r_, q_]), "num_orig": float(o["num"][r_, q_]),
                            "num_cand": float(c["num"][r_, q_]), "S_orig": float(S[r_, q_]), "c_needed": float(need[r_, q_])}
    if form == "v5-abs" and bool(worse_abs.any()):
        res["problems"].append(f"{key}: {int(worse_abs.sum())} rows worse than the original under v5-abs (c = 1)")
    # degenerate / ill-conditioned rows that v4 exempted, reported separately (v5 has no exemptions)
    v4ex = (keys <= 1) | (o["den"] < 1e-30) | (ro > 1e-3)
    if env is not None:   # DIAGNOSTIC only (v5 has no envelope): the original's own LSE +-1 ulp row movement
        e = torch.zeros_like(ro)
        for x in env:
            e = torch.maximum(e, (ratio_of(x["num"], x["den"]) - ro).abs())
        res["worse_beyond_orig_lse_envelope"] = int((rc > ro + tol + e).sum())
        res["orig_lse_envelope_rows_moved_up"] = int(sum(int((ratio_of(x["num"], x["den"]) > ro + tol).sum()) for x in env))
    res["v4_exempt_rows"] = int(v4ex.sum())
    res["worse_among_v4_exempt"] = int((worse & v4ex).sum())
    if bool(worse.any()):
        ex = torch.where(worse, rc - ro - tol, torch.full_like(rc, -1.0)).flatten()
        top = torch.topk(ex, min(worst_n, int(worse.sum()))).indices
        res["worst"] = []
        for i in top.tolist():
            r_, q_ = divmod(i, sq)
            res["worst"].append({"bh": r_, "row": q_, "keys": int(keys[r_, q_]), "ratio_orig": float(ro[r_, q_]),
                                 "ratio_cand": float(rc[r_, q_]), "num_cand": float(c["num"][r_, q_]),
                                 "den_cand": float(c["den"][r_, q_]), "num_orig": float(o["num"][r_, q_]),
                                 "den_orig": float(o["den"][r_, q_]), "delta_used_cand": float(c["du"][r_, q_]),
                                 "delta_used_orig": float(o["du"][r_, q_])})
        if form == "v5":
            res["problems"].append(f"{key}: {int(worse.sum())} rows worse than the original by more than k 2^-23")
    if bool(zbad.any()):
        res["problems"].append(f"{key}: {int(zbad.sum())} zero-denominator rows with a nonzero sum")
    for tag, x in (("cand", c), ("orig", o)):
        dr = x["pdp"] * TWO_M8
        du = x["du"].double()
        d = (du - dr).abs()
        scale = (x["pdpabs"] * TWO_M8)
        rel_abs = torch.where(scale > 0, d / torch.where(scale > 0, scale, torch.ones_like(scale)), torch.zeros_like(d))
        rel_d = torch.where(dr.abs() > 0, d / torch.where(dr.abs() > 0, dr.abs(), torch.ones_like(dr)),
                            torch.where(d > 0, torch.full_like(d, float("inf")), torch.zeros_like(d)))
        res[f"delta_{tag}"] = {"max_abs": float(d.max()), "max_rel_to_sumabs": float(rel_abs.max()),
                               "max_rel_to_delta": float(rel_d.max()), "median_rel_to_sumabs": float(rel_abs.median()),
                               "max_norm_err": float((x["psum"] * TWO_M8 - 1.0).abs().max())}
    return res


def check(a):
    import torch
    O, C = torch.load(a.orig, weights_only=False), torch.load(a.cand, weights_only=False)
    ENV = [torch.load(p, weights_only=False) for p in (a.env or [])]
    probs, report = [], {"version": VERSION, "orig": {k: O[k] for k in ("library", "library_sha256", "suite", "route")},
                         "cand": {k: C[k] for k in ("library", "library_sha256", "suite", "route")}, "cases": {}}
    if O["suite"] != C["suite"] or O["route"] != C["route"]:
        probs.append("orig / cand captures are of different suites / routes")
    if set(O["cases"]) != set(C["cases"]):
        probs.append(f"case sets differ: {sorted(set(O['cases']) ^ set(C['cases']))[:4]}")
    tot = {"rows": 0, "worse": 0, "better": 0, "worse_beyond_orig_lse_envelope": 0, "abs_worse_c1": 0}
    for key in sorted(set(O["cases"]) & set(C["cases"])):
        o, c = O["cases"][key], C["cases"][key]
        if o["desc"] != c["desc"]:
            probs.append(f"{key}: descriptors differ")
            continue
        ent = {"outputs_equal": o["outputs"] == c["outputs"]}
        if "num" in c and "num" in o:
            env = [E["cases"][key] for E in ENV] if ENV else None
            r = case_check(key, o, c, a.worst, env, a.form)
            probs += r.pop("problems")
            ent.update(r)
            for x in ("rows", "worse", "better", "worse_beyond_orig_lse_envelope", "abs_worse_c1"):
                tot[x] += r.get(x, 0)
            tot["abs_c_needed"] = max(tot.get("abs_c_needed", 0.0), r.get("abs_c_needed", 0.0))
        report["cases"][key] = ent
    report["totals"] = tot
    report["problems"] = probs
    report["form"] = a.form
    report["result"] = "PASS" if not probs else "FAIL"
    if a.report:
        with open(a.report, "w") as f:
            json.dump(report, f, indent=1)
    for p in probs[:30]:
        print("FAIL:", p)
    print(f"{VERSION} [{a.form}] check {C['suite']}/{C['route']}: {report['result']} ({tot['worse']} of {tot['rows']} rows "
          f"ratio-worse, {tot['abs_worse_c1']} abs-worse at c=1, max c needed {tot.get('abs_c_needed', 0):.4g}, {len(probs)} problems)")
    return 0 if not probs else 1


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("capture")
    c.add_argument("--root", required=True)
    c.add_argument("--suite", choices=("cases", "accept", "stress"), required=True)
    c.add_argument("--route", choices=("default", "kv", "head", "streams2"), default="default")
    c.add_argument("--seeds", default="0,1,2")
    c.add_argument("--out", required=True)
    k = sp.add_parser("check")
    k.add_argument("--orig", required=True); k.add_argument("--cand", required=True)
    k.add_argument("--report", default=None); k.add_argument("--worst", type=int, default=5)
    k.add_argument("--form", choices=("v5", "v5-abs"), default="v5")
    k.add_argument("--env", nargs=2, default=None, help="DIAGNOSTIC: captures of the ORIGINAL with LSE +1 / -1 ulp")
    a = ap.parse_args()
    if a.cmd == "capture":
        a.seeds = [int(s) for s in a.seeds.split(",")]
        capture(a)
        return 0
    return check(a)


if __name__ == "__main__":
    sys.exit(main())
