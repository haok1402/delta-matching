"""Byte-level equivalence of two delta_matching builds (library + Python package), for changes that must leave every
output and intermediate bit unchanged.

    # one dump per build, each in its own process (the package on PYTHONPATH decides which build is imported)
    PYTHONPATH=<frozen baseline root> python benchmarks/equivalence.py dump --expect-root <root> --out base.pt
    PYTHONPATH=<checkout>             python benchmarks/equivalence.py dump --expect-root <checkout> --out cand.pt
    python benchmarks/equivalence.py compare base.pt cand.pt          # exit 1 on a mismatching byte, 2 on an invalid dump

Capture: every case runs ONE forward + backward through the public API (dm_attn_func / dm_attn_func_bhnd) inside a
capture context that wraps the producers the API route calls (attn.quant_qkv_m17, attn.doq_m17) and both op accessors
(attn._ops_gt, core._ops) with a forwarding proxy: each real operation runs exactly once, and the proxy snapshots (on
the producing stream) the forward's O / LSE, and for every delta_fp8 / bwd_fp8_t call the LSE and delta it consumed and
what it returned, keyed by the call's query-head range (one group unsplit; one per KV group or query head on the split
routes). Nothing is replayed.

Schema: the expected tensor set, shapes and dtypes of a case are derived independently of the dump from its descriptor
(B, H, Hk, N, D, mode), the block-scale contract (128 / 64 rows at D 128, 64 / 32 at D 256) and the route's group size;
`compare` validates each dump on its own (exact case set for a full suite, exact tensor set per case, non-empty tensors
of the expected shape / dtype, recorded manifest = expected schema, route environment and resolved route, heads per
call) before comparing bytes.  Filtered dumps are partial and need `--partial`.

Cases: D 128 and 256; MHA, GQA 4, GQA 3, MQA; batch 1 and > 1; the shortest legal N, N across the 512-row dV promotion
window, a held-out shape; causal matched, causal stale, full stale; (B, N, H, D) inputs, packed QKV views, (B, H, N, D)
contiguous inputs; stress inputs (zeros, constant K, q / k scaled by 3 and 10, sparse large outliers).  Routes set the
complete routing environment before import: default (no split), kv (per KV group), head (per query head), streams2 (per
KV group on 2 streams).  `dump --perturb <tensor>` flips one bit of a captured tensor (comparator self-test).
"""
import argparse, os, sys
from contextlib import contextmanager

MANIFEST_VERSION = 3
ROUTES = {"default": {"DM_BWD_HEAD_GROUPS": "off", "DM_BWD_STREAMS": "1", "DM_BWD_WS_LIMIT_GB": "8"},
          "kv": {"DM_BWD_HEAD_GROUPS": "kv", "DM_BWD_STREAMS": "1", "DM_BWD_WS_LIMIT_GB": "8"},
          "head": {"DM_BWD_HEAD_GROUPS": "head", "DM_BWD_STREAMS": "1", "DM_BWD_WS_LIMIT_GB": "8"},
          "streams2": {"DM_BWD_HEAD_GROUPS": "kv", "DM_BWD_STREAMS": "2", "DM_BWD_WS_LIMIT_GB": "8"}}

# (name, D, B, H, Hk, N, input kind, layout)
SHAPES = [
    ("d128_min", 128, 1, 4, 4, 256, "gauss", "bnhd"),
    ("d128_gqa4", 128, 2, 8, 2, 2048, "gauss", "bnhd"),
    ("d128_gqa3_promote", 128, 1, 6, 2, 1280, "gauss", "bnhd"),
    ("d128_mqa", 128, 2, 4, 1, 1024, "gauss", "bnhd"),
    ("d128_packed", 128, 2, 4, 4, 1024, "gauss", "packed"),
    ("d128_bhnd", 128, 1, 8, 2, 1024, "gauss", "bhnd"),
    ("d128_heldout", 128, 3, 12, 4, 3072, "gauss", "bnhd"),
    ("d256_min", 256, 1, 4, 4, 128, "gauss", "bnhd"),
    ("d256_gqa4", 256, 2, 8, 2, 2048, "gauss", "bnhd"),
    ("d256_gqa3_promote", 256, 1, 6, 2, 640, "gauss", "bnhd"),
    ("d256_mqa", 256, 2, 4, 1, 1024, "gauss", "bnhd"),
    ("d256_packed", 256, 2, 4, 4, 1024, "gauss", "packed"),
    ("d256_bhnd", 256, 1, 8, 2, 1024, "gauss", "bhnd"),
    ("d256_heldout", 256, 3, 6, 2, 3072, "gauss", "bnhd"),
] + [(f"d{D}_{kind}", D, 1, 4, 2, 1024, kind, "bnhd") for D in (128, 256)
     for kind in ("zeros", "constk", "scale3", "scale10", "outliers")]
MODES = (("causal", "matched"), ("causal", "stale"), ("full", "stale"))
REF_MAX_N = 2048   # fp32 reference (informational) only up to this length


# ----------------------------------------------------------------------------------------------- the expected schema
def selected_cases(route, only=None):
    """(key, descriptor) of the cases a dump of this route runs: the split routes cover the GQA / MQA / held-out cases."""
    out = []
    for i, (name, D, B, H, Hk, N, kind, layout) in enumerate(SHAPES):
        if only is not None and only not in name:
            continue
        if route != "default" and not (H != Hk or name.endswith("heldout")):
            continue
        for attn, mode in MODES:
            out.append((f"{name}/{attn}/{mode}", {"name": name, "D": D, "B": B, "H": H, "Hk": Hk, "N": N, "kind": kind,
                                                  "layout": layout, "attn": attn, "mode": mode, "seed": 1000 + i}))
    return out


def heads_per_call(route, H, Hk):
    """Query heads per backward call on this route, from the route's definition (README: kv = one KV group of
    H / Hk query heads, head = one query head, default = no split); never more than H."""
    if route == "default":
        return H
    if route in ("kv", "streams2"):
        return min(H // Hk, H)
    if route == "head":
        return 1
    raise ValueError(route)


def expected_schema(desc, route):
    """{tensor name: (shape, dtype name)} of a case, derived from its descriptor and the route only."""
    B, H, Hk, N, D = desc["B"], desc["H"], desc["Hk"], desc["N"], desc["D"]
    bq, bk = (128, 64) if D == 128 else (64, 32)
    BH, BHk = B * H, B * Hk
    f8, f32, bf = "torch.float8_e4m3fn", "torch.float32", "torch.bfloat16"
    s = {"api.O": ((B, N, H, D), bf), "api.dQ": ((B, N, H, D), bf), "api.dK": ((B, N, Hk, D), bf),
         "api.dV": ((B, N, Hk, D), bf),
         "quant.Q8": ((BH, N, D), f8), "quant.Q8s": ((BH, N // bq), f32), "quant.Qt": ((BH, D, N), f8),
         "quant.K8": ((BHk, N, D), f8), "quant.K8s": ((BHk, N // bk), f32), "quant.kTt": ((BHk, D, N), f8),
         "quant.V8": ((BHk, N, D), f8), "quant.V8s": ((BHk, N // bk), f32),
         "fwd.O": ((B, N, H, D), bf), "fwd.LSE": ((B, H, N), f32),
         "doq.dO8": ((BH, N, D), f8), "doq.dO8s": ((BH, N // bq), f32), "doq.dOt": ((BH, D, N), f8)}
    if desc["mode"] == "stale":
        s["doq.delta"] = ((BH, N), f32)
    hp = heads_per_call(route, H, Hk)
    for h0 in range(0, H, hp):
        g = f"g{h0:02d}-{h0 + hp:02d}"
        if desc["mode"] == "matched":
            s[f"{g}.delta_fp8.lse"] = ((B, hp, N), f32)
            s[f"{g}.delta_fp8.out"] = ((B, hp, N), f32)
        s[f"{g}.bwd.lse"] = ((B, hp, N), f32)
        s[f"{g}.bwd.delta"] = ((B, hp, N), f32)
        for t in ("dq", "dk", "dv"):
            s[f"{g}.bwd.{t}"] = ((B, N, hp, D), bf)
    return s


def validate_dump(X, label, partial_ok=False):
    """Every check a single dump must pass on its own.  Returns a list of problems (empty = valid)."""
    import torch
    bad = []
    m = X.get("manifest") if isinstance(X, dict) else None
    if not isinstance(m, dict) or m.get("version") != MANIFEST_VERSION:
        return [f"{label}: missing manifest or version != {MANIFEST_VERSION}"]
    route = m.get("route")
    if route not in ROUTES:
        return [f"{label}: unknown route {route!r}"]
    if m.get("route_env") != ROUTES[route]:
        bad.append(f"{label}: route_env {m.get('route_env')} != {ROUTES[route]}")
    want_res = {"groups": ROUTES[route]["DM_BWD_HEAD_GROUPS"], "streams": int(ROUTES[route]["DM_BWD_STREAMS"]),
                "ws_limit": float(ROUTES[route]["DM_BWD_WS_LIMIT_GB"]) * 2 ** 30}
    if m.get("resolved_route") != want_res:
        bad.append(f"{label}: resolved route {m.get('resolved_route')} != {want_res}")
    cases = m.get("cases")
    if not isinstance(cases, dict) or not cases:
        return bad + [f"{label}: empty or missing case list"]
    expected_full = dict(selected_cases(route))
    if m.get("suite") == "full":
        if cases != expected_full:
            bad.append(f"{label}: a full-suite dump must hold exactly the {len(expected_full)} expected cases")
    elif m.get("suite") == "partial":
        if not partial_ok:
            bad.append(f"{label}: partial dump (pass --partial for a non-certifying comparison)")
        if any(expected_full.get(k) != v for k, v in cases.items()):
            bad.append(f"{label}: partial dump with cases / descriptors outside the expected set")
    else:
        bad.append(f"{label}: suite {m.get('suite')!r}")
    tensors = X.get("cases") if isinstance(X.get("cases"), dict) else {}
    if set(tensors) != set(cases):
        bad.append(f"{label}: tensor cases differ from the manifest's case list")
    mt = m.get("tensors") if isinstance(m.get("tensors"), dict) else None
    if mt is None:
        bad.append(f"{label}: manifest tensor table missing")
        mt = {}
    hpc = X.get("heads_per_call") if isinstance(X.get("heads_per_call"), dict) else {}
    for key, desc in cases.items():
        hp = heads_per_call(route, desc["H"], desc["Hk"])
        if hpc.get(key) != hp:
            bad.append(f"{label} {key}: heads per call {hpc.get(key)} != the route's {hp}")
        sch = expected_schema(desc, route)
        have = tensors.get(key) if isinstance(tensors.get(key), dict) else {}
        if set(have) != set(sch):
            miss, extra = sorted(set(sch) - set(have)), sorted(set(have) - set(sch))
            bad.append(f"{label} {key}: tensor set differs from the schema (missing {miss[:4]}, unexpected {extra[:4]})")
        rec_case = mt.get(key) if isinstance(mt.get(key), dict) else {}
        if set(rec_case) != set(sch):
            bad.append(f"{label} {key}: manifest tensor list differs from the schema")
        for name, (shape, dt) in sch.items():
            t = have.get(name)
            if t is not None and (not isinstance(t, torch.Tensor) or tuple(t.shape) != shape or str(t.dtype) != dt
                                  or t.numel() == 0):
                got = (tuple(t.shape), str(t.dtype)) if isinstance(t, torch.Tensor) else type(t).__name__
                bad.append(f"{label} {key} {name}: {got} != expected {(shape, dt)}")
            if name in rec_case and rec_case[name] != [list(shape), dt]:
                bad.append(f"{label} {key} {name}: manifest entry {rec_case[name]} != {[list(shape), dt]}")
    return bad


# ----------------------------------------------------------------------------------------------- inputs / reference
def make_inputs(D, B, H, Hk, N, kind, seed):
    import torch
    g = torch.Generator(device="cuda").manual_seed(seed)

    def r(*s):
        return torch.randn(*s, device="cuda", dtype=torch.float32, generator=g)
    q, k, v = r(B, N, H, D), r(B, N, Hk, D), r(B, N, Hk, D)
    do = r(B, N, H, D)
    if kind == "zeros":
        q, k, v, do = q * 0, k * 0, v * 0, do * 0
    elif kind == "constk":
        k = k[:, :1].expand_as(k).clone()
    elif kind == "scale3":
        q, k = q * 3, k * 3
    elif kind == "scale10":
        q, k = q * 10, k * 10
    elif kind == "outliers":
        m = (torch.rand(q.shape, device="cuda", generator=g) < 1e-4)
        q = torch.where(m, q * 100, q)
        mk = (torch.rand(k.shape, device="cuda", generator=g) < 1e-4)
        k = torch.where(mk, k * 100, k)
        v = torch.where(mk, v * 100, v)
    return q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16), do.to(torch.bfloat16)


def reference(q, k, v, do, causal):
    import torch
    B, N, H, D = q.shape
    Hk = k.shape[2]
    qf, kf, vf = (x.float().transpose(1, 2).requires_grad_(True) for x in (q, k, v))
    kr, vr = kf.repeat_interleave(H // Hk, 1), vf.repeat_interleave(H // Hk, 1)
    s = (qf @ kr.transpose(-1, -2)) * D ** -0.5
    if causal:
        s = s.masked_fill(~torch.ones(N, N, device=q.device, dtype=torch.bool).tril(), float("-inf"))
    o = torch.softmax(s, -1) @ vr
    o.backward(do.float().transpose(1, 2))
    return o.transpose(1, 2), qf.grad.transpose(1, 2), kf.grad.transpose(1, 2), vf.grad.transpose(1, 2)


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()


# ----------------------------------------------------------------------------------------------- capture
class _OpsProxy:
    """Forwards every attribute to the real op namespace; records fwd_fp8 / delta_fp8 / bwd_fp8_t calls."""

    def __init__(self, real, rec):
        self._real, self._rec = real, rec

    def __getattr__(self, name):
        fn = getattr(self._real, name)
        if name not in ("fwd_fp8", "delta_fp8", "bwd_fp8_t"):
            return fn
        rec = self._rec

        def wrapped(*args):
            out = fn(*args)
            snap, t = rec["snap"], rec["t"]
            if name == "fwd_fp8":
                t["fwd.O"], t["fwd.LSE"] = snap(out[0]), snap(out[1])
                return out
            hg = args[0].shape[2]                       # query heads of this call (dout8 is (b, s, h, d))
            ctr = "delta_h" if name == "delta_fp8" else "bwd_h"
            h0 = rec[ctr]; rec[ctr] += hg
            g = f"g{h0:02d}-{h0 + hg:02d}"
            if name == "delta_fp8":
                t[f"{g}.delta_fp8.lse"], t[f"{g}.delta_fp8.out"] = snap(args[4]), snap(out)
            else:
                t[f"{g}.bwd.lse"], t[f"{g}.bwd.delta"] = snap(args[4]), snap(args[5])
                for n_, x in zip(("dq", "dk", "dv"), out):
                    t[f"{g}.bwd.{n_}"] = snap(x)
            return out
        return wrapped


@contextmanager
def capture():
    """Patch the API route's producer and op bindings for one forward + backward; restore them in every case."""
    import torch
    from delta_matching import attn as A, core as C

    def snap(x):   # copy on the current (= producing) stream; the device is synchronized before serialization
        return x.detach().clone()
    rec = {"t": {}, "delta_h": 0, "bwd_h": 0, "snap": snap}
    saved = (A.quant_qkv_m17, A.doq_m17, A._ops_gt, C._ops)
    real_quant, real_doq, real_gt, real_core = saved

    def quant(*a, **k):
        out = real_quant(*a, **k)
        for n_, x in zip(("Q8", "Q8s", "K8", "K8s", "V8", "V8s", "kTt", "Qt"), out):
            rec["t"][f"quant.{n_}"] = snap(x)
        return out

    def doq(*a, **k):
        out = real_doq(*a, **k)
        dO8, dO8s, dlt, dOt = out
        rec["t"]["doq.dO8"], rec["t"]["doq.dO8s"], rec["t"]["doq.dOt"] = snap(dO8), snap(dO8s), snap(dOt)
        if dlt is not None:
            rec["t"]["doq.delta"] = snap(dlt)
        return out
    try:
        A.quant_qkv_m17, A.doq_m17 = quant, doq
        A._ops_gt = lambda: _OpsProxy(real_gt(), rec)
        C._ops = lambda: _OpsProxy(real_core(), rec)
        yield rec
        torch.cuda.synchronize()
    finally:
        A.quant_qkv_m17, A.doq_m17, A._ops_gt, C._ops = saved


def run_case(dm, desc):
    import torch
    causal, mode = desc["attn"] == "causal", desc["mode"]
    q, k, v, do = make_inputs(desc["D"], desc["B"], desc["H"], desc["Hk"], desc["N"], desc["kind"], desc["seed"])
    with capture() as rec:
        if desc["layout"] == "bhnd":       # (B, H, N, D) contiguous leaves through the bhnd entry point
            leaves = [x.detach().transpose(1, 2).contiguous().requires_grad_(True) for x in (q, k, v)]
            strides = {n_: list(t.stride()) for n_, t in zip("qkv", leaves)}
            o = dm.dm_attn_func_bhnd(*leaves, causal=causal, delta_mode=mode)
            o.backward(do)
            dq, dk, dv = (x.grad.transpose(1, 2) for x in leaves)
        elif desc["layout"] == "packed":   # one (B, N, 3, H, D) leaf; q / k / v are strided views into it
            qkv = torch.stack([q, k, v], 2).detach().requires_grad_(True)
            views = qkv.unbind(2)
            strides = {n_: list(t.stride()) for n_, t in zip("qkv", views)}
            o = dm.dm_attn_func(*views, causal=causal, mode=mode)
            o.backward(do)
            dq, dk, dv = qkv.grad.unbind(2)
        else:
            leaves = [x.detach().requires_grad_(True) for x in (q, k, v)]
            strides = {n_: list(t.stride()) for n_, t in zip("qkv", leaves)}
            o = dm.dm_attn_func(*leaves, causal=causal, mode=mode)
            o.backward(do)
            dq, dk, dv = (x.grad for x in leaves)
        for n_, t in (("api.O", o), ("api.dQ", dq), ("api.dK", dk), ("api.dV", dv)):
            rec["t"][n_] = t.detach().contiguous().clone()
    errs = {}
    if desc["N"] <= REF_MAX_N and desc["kind"] == "gauss":   # informational: not part of the byte comparison
        ro, rdq, rdk, rdv = reference(q, k, v, do, causal)
        errs = {"O": rel(o, ro), "dQ": rel(dq, rdq), "dK": rel(dk, rdk), "dV": rel(dv, rdv)}
    return {n_: t.contiguous().cpu() for n_, t in rec["t"].items()}, errs, strides


def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(a):
    os.environ.update(ROUTES[a.route])
    os.environ.pop("DM_SO", None)          # the package on PYTHONPATH decides the library ...
    if a.so:
        os.environ["DM_SO"] = a.so         # ... unless one is named explicitly (recorded below)
    import torch
    import delta_matching as dm
    from delta_matching import split as dm_split
    torch.backends.cuda.matmul.allow_tf32 = False
    dm.core._ops()   # load the library before any capture (the loader records its path on the unpatched accessor)
    cases = selected_cases(a.route, a.only)
    if not cases:
        raise SystemExit(f"no case matches --only {a.only!r}")
    resolved = {"groups": dm_split._BWD_GROUPS, "streams": dm_split._BWD_STREAMS, "ws_limit": dm_split._BWD_WS_LIMIT}
    want = ROUTES[a.route]
    if resolved["groups"] != want["DM_BWD_HEAD_GROUPS"] or resolved["streams"] != int(want["DM_BWD_STREAMS"]):
        raise SystemExit(f"route {a.route}: the imported package resolved {resolved}, expected {want}")
    res = {"manifest": {"version": MANIFEST_VERSION, "route": a.route, "route_env": want, "resolved_route": resolved,
                        "suite": "full" if a.only is None else "partial", "only": a.only, "cases": dict(cases),
                        "tensors": {}},
           "meta": {"package": os.path.realpath(os.path.dirname(dm.__file__)), "dm_so_arg": a.so}, "cases": {},
           "errors": {}, "heads_per_call": {}, "input_strides": {}}
    for key, desc in cases:
        t, e, st = run_case(dm, desc)
        res["cases"][key] = t
        res["errors"][key] = e
        res["input_strides"][key] = st
        res["heads_per_call"][key] = min(dm_split._bwd_heads_per_call(desc["B"], desc["H"], desc["Hk"],
                                                                      desc["H"] // desc["Hk"], desc["N"]), desc["H"])
        res["manifest"]["tensors"][key] = {n_: [list(x.shape), str(x.dtype)] for n_, x in t.items()}
        print(f"{key:40s} heads/call {res['heads_per_call'][key]:3d} tensors {len(t):3d} "
              + " ".join(f"{k_} {v_:.4f}" for k_, v_ in e.items()), flush=True)
    so = dm.core.loaded_library()
    res["meta"].update(library=os.path.realpath(so), library_sha256=sha256(so))
    if a.expect_root:
        import glob
        pkg = os.path.realpath(os.path.join(a.expect_root, "delta_matching"))
        libs = [os.path.realpath(x) for x in glob.glob(os.path.join(pkg, "_C*.so"))]
        if res["meta"]["package"] != pkg or libs != [os.path.realpath(so)]:
            raise SystemExit(f"imported {res['meta']['package']} / {os.path.realpath(so)}, expected {pkg} / {libs}")
    problems = validate_dump(res, "this dump", partial_ok=True)
    if problems:
        raise SystemExit("the captured dump does not match its schema:\n  " + "\n  ".join(problems[:20]))
    if a.perturb:   # self-test: flip one bit of a captured tensor (after validation, as a corrupted dump would be)
        key = next(iter(res["cases"]))
        x = res["cases"][key][a.perturb]
        flat = x.view(-1).view(torch.uint8)
        flat[flat.numel() // 2] ^= 1
        print(f"perturbed {key} {a.perturb} at byte {flat.numel() // 2}")
    torch.save(res, a.out)
    print(f"wrote {a.out}: {len(res['cases'])} cases ({res['manifest']['suite']}), route {a.route} {resolved}, "
          f"library {res['meta']['library']} sha256 {res['meta']['library_sha256'][:16]}")


def compare(a):
    import torch
    A, B = torch.load(a.a, weights_only=False), torch.load(a.b, weights_only=False)
    problems = validate_dump(A, "A", a.partial) + validate_dump(B, "B", a.partial)
    if not problems:
        ma, mb = A["manifest"], B["manifest"]
        if ma["route"] != mb["route"]:
            problems.append(f"route mismatch: {ma['route']} vs {mb['route']}")
        if ma["cases"] != mb["cases"]:
            problems.append("the two dumps ran different case sets / descriptors")
    if problems:
        print("INVALID DUMP(S):")
        for p in problems[:40]:
            print("  " + p)
        return 2
    bad = 0
    for key in sorted(A["cases"]):
        ta, tb = A["cases"][key], B["cases"][key]
        for name in sorted(ta):
            x, y = ta[name], tb[name]
            xb, yb = x.reshape(-1).view(torch.uint8), y.reshape(-1).view(torch.uint8)
            if not torch.equal(xb, yb):
                es = x.element_size()
                diff = (xb.view(-1, es) != yb.view(-1, es)).any(1)
                first = int(diff.nonzero()[0])
                idx = tuple(int(i) for i in torch.unravel_index(torch.tensor(first), x.shape))
                print(f"DIFF {key} {name}: {int(diff.sum())} of {x.numel()} elements differ, first at {idx}")
                bad += 1
    ea, eb = A.get("errors", {}), B.get("errors", {})
    worse = [f"{k_} {n_} {ea[k_][n_]:.5f} -> {v_:.5f}" for k_ in sorted(set(ea) & set(eb))
             for n_, v_ in eb[k_].items() if n_ in ea[k_] and v_ > ea[k_][n_] * (1 + 1e-6)]
    partial = A["manifest"]["suite"] != "full" or B["manifest"]["suite"] != "full"
    print(f"route {A['manifest']['route']}, {'PARTIAL (not a certification)' if partial else 'FULL SUITE'}, "
          f"A {A['meta'].get('library_sha256', '?')[:12]} vs B {B['meta'].get('library_sha256', '?')[:12]}")
    print(f"compared {len(A['cases'])} cases: {'BITWISE EQUAL' if bad == 0 else f'{bad} mismatching tensors'}; "
          f"error vs fp32 larger in {len(worse)} (case, tensor) pairs")
    for w in worse[:20]:
        print("  worse:", w)
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    d = sp.add_parser("dump")
    d.add_argument("--out", required=True)
    d.add_argument("--route", choices=sorted(ROUTES), default="default")
    d.add_argument("--only", default=None, help="substring filter on case names (a partial, non-certifying dump)")
    d.add_argument("--perturb", default=None, help="flip one bit of this captured tensor in the first case (self-test)")
    d.add_argument("--so", default=None, help="load this library explicitly (recorded); default: the package's own")
    d.add_argument("--expect-root", default=None, help="fail unless the package and library come from this build root")
    c = sp.add_parser("compare")
    c.add_argument("a"); c.add_argument("b")
    c.add_argument("--partial", action="store_true", help="accept filtered dumps (reported as a partial comparison)")
    a = ap.parse_args()
    if a.cmd == "dump":
        dump(a)
        return 0
    return compare(a)


if __name__ == "__main__":
    sys.exit(main())
