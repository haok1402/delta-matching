"""Byte-level equivalence of two delta_matching builds (library + Python package), for schedule / layout changes that
must leave every output bit unchanged.

    # one dump per build, each in its own process (the package on PYTHONPATH decides which build is imported)
    PYTHONPATH=<frozen baseline root> python benchmarks/equivalence.py dump --out base.pt
    PYTHONPATH=<checkout>             python benchmarks/equivalence.py dump --out cand.pt
    python benchmarks/equivalence.py compare base.pt cand.pt          # exit 1 on any mismatching byte

A dump holds, for every case of the fixed set below: the API-route outputs O, dQ, dK, dV (dm_attn_func, forward +
backward) and the forward LSE; the producer codes, scales and twins (Q8 / Q8s / Qt, K8 / K8s / kTt, V8 / V8s, dO8 / dO8s /
dOt); the stale delta and the matched delta (pass 0).  Tensors are compared as raw bytes (signed zeros and NaN payloads
count).  The relative error of O / dQ / dK / dV against fp32 attention is recorded for the cases small enough for the
reference (informational: the gate for a declared-bitwise change is byte equality).

Cases: D 128 and 256; MHA, GQA 4, GQA 3, MQA; batch 1 and > 1; the shortest legal N, N across the 512-row dV promotion
window, a held-out shape; causal matched, causal stale, full stale; input layouts (B, N, H, D) contiguous, packed QKV
views, (B, H, N, D) contiguous through dm_attn_func_bhnd; stress inputs (zeros, constant K, q / k scaled by 3 and 10,
sparse large outliers).  The routing knobs (DM_BWD_HEAD_GROUPS, DM_BWD_STREAMS) are read at import, so a dump is taken
per route: `dump --route kv` sets the environment before importing the package.

`dump --perturb dV` flips the lowest mantissa bit of one dV element after computing it: a self-test of the comparator.
"""
import argparse, os, sys

ROUTES = {"default": {}, "kv": {"DM_BWD_HEAD_GROUPS": "kv"}, "head": {"DM_BWD_HEAD_GROUPS": "head"},
          "streams2": {"DM_BWD_HEAD_GROUPS": "kv", "DM_BWD_STREAMS": "2"}}

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
REF_MAX_N = 2048   # fp32 reference only up to this length (memory / time)


def make_inputs(D, B, H, Hk, N, kind, layout, seed):
    import torch
    g = torch.Generator(device="cuda").manual_seed(seed)

    def r(*s):
        return torch.randn(*s, device="cuda", dtype=torch.float32, generator=g)
    if layout == "packed":
        assert H == Hk
        qkv = r(B, N, 3, H, D)
        q, k, v = qkv.unbind(2)
    else:
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
    if layout == "packed":
        qkv = torch.stack([q, k, v], 2).to(torch.bfloat16)
        q, k, v = qkv.unbind(2)          # strided views into one (B, N, 3, H, D) buffer
    else:
        q, k, v = (x.to(torch.bfloat16) for x in (q, k, v))
    return q, k, v, do.to(torch.bfloat16)


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


def run_case(dm, name, D, B, H, Hk, N, kind, layout, attn, mode, seed):
    import torch
    from delta_matching.core import blocks, _matched_delta_inkernel
    from delta_matching.producers import quant_qkv_m17, doq_m17
    causal = attn == "causal"
    q, k, v, do = make_inputs(D, B, H, Hk, N, kind, layout, seed)
    out = {}
    if layout == "bhnd":       # (B, H, N, D) contiguous leaves through the bhnd entry point
        qb, kb, vb = (x.detach().transpose(1, 2).contiguous().requires_grad_(True) for x in (q, k, v))
        o = dm.dm_attn_func_bhnd(qb, kb, vb, causal=causal, delta_mode=mode)
        o.backward(do)
        dq, dk, dv = (x.grad.transpose(1, 2) for x in (qb, kb, vb))
    elif layout == "packed":   # one (B, N, 3, H, D) leaf; q / k / v are strided views into it
        qkv = torch.stack([q, k, v], 2).detach().requires_grad_(True)
        o = dm.dm_attn_func(*qkv.unbind(2), causal=causal, mode=mode)
        o.backward(do)
        dq, dk, dv = qkv.grad.unbind(2)
    else:
        q_, k_, v_ = (x.detach().requires_grad_(True) for x in (q, k, v))
        o = dm.dm_attn_func(q_, k_, v_, causal=causal, mode=mode)
        o.backward(do)
        dq, dk, dv = q_.grad, k_.grad, v_.grad
    out.update(O=o, dQ=dq, dK=dk, dV=dv)
    # intermediates through the same producers / ops the API route uses
    qh, kh, vh = (x.detach().transpose(1, 2) for x in (q, k, v))
    Q8, Q8s, K8, K8s, V8, V8s, kTt, Qt = quant_qkv_m17(qh, kh, vh, want_qt=True, blocks=blocks(D))
    ops = torch.ops.delta_matching

    def bshd(x, h):
        return x.view(B, h, N, D).permute(0, 2, 1, 3)
    O2, Lse = ops.fwd_fp8(bshd(Q8, H), bshd(K8, Hk), bshd(V8, Hk), Q8s.view(B, H, -1), K8s.view(B, Hk, -1),
                          V8s.view(B, Hk, -1), D ** -0.5, causal)
    dO8, dO8s, Dlt, dOt = doq_m17(do, O2, H, blocks(D)[0], want_delta=True, want_twin=True)
    out.update(Q8=Q8, Q8s=Q8s, Qt=Qt, K8=K8, K8s=K8s, kTt=kTt, V8=V8, V8s=V8s, LSE=Lse, O_op=O2,
               dO8=dO8, dO8s=dO8s, dOt=dOt, delta_stale=Dlt)
    if causal:
        out["delta_matched"] = _matched_delta_inkernel(Q8, Q8s, K8, K8s, V8, V8s, dO8, dO8s, kTt,
                                                       Lse.view(B * H, N), H, N, D, causal=True, Hk=Hk)
    errs = {}
    if N <= REF_MAX_N and kind == "gauss":
        ro, rdq, rdk, rdv = reference(q, k, v, do, causal)
        errs = {"O": rel(o, ro), "dQ": rel(dq, rdq), "dK": rel(dk, rdk), "dV": rel(dv, rdv)}
    return {k_: v_.detach().contiguous().cpu() for k_, v_ in out.items()}, errs


def dump(a):
    os.environ.update(ROUTES[a.route])
    import torch
    import delta_matching as dm
    torch.backends.cuda.matmul.allow_tf32 = False
    res = {"meta": {"route": a.route, "package": os.path.dirname(dm.__file__)}, "cases": {}, "errors": {}}
    shapes = [s for s in SHAPES if (a.only is None or a.only in s[0])]
    if a.route != "default":   # the split routes only matter for the backward: GQA / MQA cases
        shapes = [s for s in shapes if s[3] != s[4] or s[0].endswith("heldout")]
    for i, (name, D, B, H, Hk, N, kind, layout) in enumerate(shapes):
        for attn, mode in MODES:
            key = f"{name}/{attn}/{mode}"
            t, e = run_case(dm, name, D, B, H, Hk, N, kind, layout, attn, mode, seed=1000 + i)
            res["cases"][key] = t
            res["errors"][key] = e
            print(f"{key:40s} " + " ".join(f"{k} {v:.4f}" for k, v in e.items()), flush=True)
    res["meta"]["library"] = dm.core.loaded_library()
    if a.perturb:
        key = next(iter(res["cases"]))
        x = res["cases"][key][a.perturb]
        flat = x.view(-1).view(torch.int16) if x.element_size() == 2 else x.view(-1).view(torch.int32)
        flat[flat.numel() // 2] ^= 1
        print(f"perturbed {key} {a.perturb} at flat index {flat.numel() // 2}")
    torch.save(res, a.out)
    print(f"wrote {a.out}: {len(res['cases'])} cases, library {res['meta']['library']}")


def compare(a):
    import torch
    A, B = torch.load(a.a, weights_only=False), torch.load(a.b, weights_only=False)
    bad = 0
    keys = sorted(set(A["cases"]) | set(B["cases"]))
    for key in keys:
        if key not in A["cases"] or key not in B["cases"]:
            print(f"MISSING {key} in {'A' if key not in A['cases'] else 'B'}")
            bad += 1
            continue
        ta, tb = A["cases"][key], B["cases"][key]
        for name in sorted(set(ta) | set(tb)):
            if name not in ta or name not in tb:
                print(f"MISSING {key} {name}")
                bad += 1
                continue
            x, y = ta[name], tb[name]
            if x.shape != y.shape or x.dtype != y.dtype:
                print(f"DIFF {key} {name}: shape/dtype {tuple(x.shape)} {x.dtype} vs {tuple(y.shape)} {y.dtype}")
                bad += 1
                continue
            xb, yb = x.reshape(-1).view(torch.uint8), y.reshape(-1).view(torch.uint8)
            if not torch.equal(xb, yb):
                ne = (xb != yb).nonzero()
                first = ne[0].item() // x.element_size()
                n = int((x.reshape(-1).view(torch.uint8).view(-1, x.element_size()) !=
                         y.reshape(-1).view(torch.uint8).view(-1, x.element_size())).any(1).sum())
                md = (x.float() - y.float()).abs().max().item() if x.is_floating_point() or x.dtype.is_floating_point else float("nan")
                print(f"DIFF {key} {name}: {n} of {x.numel()} elements differ, first at flat index {first} "
                      f"(unravel {tuple(int(i) for i in torch.unravel_index(torch.tensor(first), x.shape))}), max |diff| {md:.3g}")
                bad += 1
    ea, eb = A.get("errors", {}), B.get("errors", {})
    worse = []
    for key in sorted(set(ea) & set(eb)):
        for n, v in eb[key].items():
            if n in ea[key] and v > ea[key][n] * (1 + 1e-6):
                worse.append(f"{key} {n} {ea[key][n]:.5f} -> {v:.5f}")
    print(f"compared {len(keys)} cases: {'BITWISE EQUAL' if bad == 0 else f'{bad} mismatching tensors'}; "
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
    d.add_argument("--only", default=None, help="substring filter on case names")
    d.add_argument("--perturb", default=None, help="flip one bit of this tensor in the first case (comparator self-test)")
    c = sp.add_parser("compare")
    c.add_argument("a"); c.add_argument("b")
    a = ap.parse_args()
    if a.cmd == "dump":
        dump(a)
        return 0
    return compare(a)


if __name__ == "__main__":
    sys.exit(main())
