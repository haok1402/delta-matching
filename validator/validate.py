#!/usr/bin/env python
"""Validate a candidate Delta-Matching attention kernel: PASS / FAIL with numbers.

  python validator/validate.py <candidate spec> [--tier quick|full|bench] [--preset NAME] [--device cuda:0] [--json out]
  candidate spec: /path/file.py[:func] (func defaults to `attn`) or pkg.module[:func]

The candidate function has dm_attn_func's contract: fn(q, k, v, softmax_scale=None, causal=..., mode=...) with q (B, N,
H, D) bf16, k / v (B, N, Hk, D), D = 128, returns o (B, N, H, D) bf16 and backpropagates into q, k, v.  The module may
set NUMERICS = "<preset of delta_matching/reference.py>" (default "sm90") and SUPPORTS = {"seqlen_multiple": m,
"noncausal": bool, "matched_noncausal": bool}.

The candidate runs in a separate process (runner.py) that never imports the reference; inputs come from a seed drawn
here at run time.  Every check is relative to what an fp8 kernel of the declared numerics achieves on the same inputs
(the torch emulation, delta_matching/reference.py), so adversarial inputs get the tolerance their own fp8 floor implies
and no more.  See validator/README.md for the checks and why each tolerance is what it is.
"""
from __future__ import annotations
try:                    # gpupool: hand the GPU to the filler before this process even creates a CUDA context
    from gpupool_hook import gpupool_done as _gp_done
    _gp_done()
except ImportError:
    pass


import argparse
import importlib.util
import json
import math
import os
import secrets
import subprocess
import sys
import tempfile
import zlib
import time
import zlib

import time as _time
import torch


def _trace(msg):
    path = os.environ.get("DMVAL_TRACE")       # a file path: append "HH:MM:SS.mmm <file> <event>" lines
    if path:
        with open(path, "a") as f:
            f.write(f"{_time.strftime('%H:%M:%S')}.{int(_time.time() * 1000) % 1000:03d} {os.path.basename(__file__)} {msg}\n")

HERE = os.path.dirname(os.path.abspath(__file__))
REF_PATH = os.path.join(os.path.dirname(HERE), "delta_matching", "reference.py")
BF = torch.bfloat16
D = 128
NAMES = ("O", "dQ", "dK", "dV")


def load_reference():
    ms = importlib.util.spec_from_file_location("dm_reference_standalone", REF_PATH)
    mod = importlib.util.module_from_spec(ms)
    sys.modules[ms.name] = mod
    ms.loader.exec_module(mod)
    return mod


R = load_reference()

# ------------------------------------------------------------------------------------------------------- tolerances
# A candidate may be at most this much worse than the emulation of its own numerics, per tensor, against exact
# attention (global relative L2), and per (batch, head, 128-row block) at most BLK_RATIO worse than the emulation's own
# worst block.  Below FLOOR_RATIO of the emulation's error it is "too accurate": it is not doing the fp8 it declares.
TOL_RATIO, TOL_ABS = 1.25, 2e-3
BLK_RATIO, BLK_ABS = 1.5, 5e-3
FLOOR_RATIO = 0.5
# faithfulness: the candidate must sit at most this fraction of the fp8 error away from its declared emulation (a
# correct kernel differs from the emulation only by accumulation order / exp2 approximations flipping a few codes)
FAITH_RATIO = 0.5
# matched-mode identity: sum over keys of dK (= Q^T rowsum(dS)) relative to sqrt(N) * |dK| rows, at most this much above
# the emulation's own level
COLSUM_RATIO, COLSUM_ABS = 2.0, 1e-4
# direct identity (debug hook): max over rows of |delta sum_j P - sum_j P dP| / sum_j |P dP| (pass-0 style kernels ~1e-6,
# a delta from the forward's O with a perturbed backward P ~1e-3..1e-2, stale 1e-2..1); binding: candidate dQ / dK vs the
# emulated backward driven by the candidate's own reported delta
ID_TOL = 1e-3
BIND_ABS, BIND_REL = 3e-3, 0.05


def rel(a, b):
    a = a.double(); b = b.double()
    return ((a - b).norm() / b.norm().clamp_min(1e-300)).item()


def block_rel(a, b, blk=128):
    """Relative L2 error per (batch, head, block of `blk` rows), (B, N, H, D) layout; blocks normalized by the whole
    head's reference norm / sqrt(#blocks) so empty-ish blocks do not explode.  Returns the max."""
    B, N, H, Dd = b.shape
    nb = (N + blk - 1) // blk
    pad = nb * blk - N
    a = a.double(); b = b.double()
    if pad:
        a = torch.nn.functional.pad(a, (0, 0, 0, 0, 0, pad)); b = torch.nn.functional.pad(b, (0, 0, 0, 0, 0, pad))
    e = (a - b).reshape(B, nb, blk, H, Dd).pow(2).sum((2, 4)).sqrt()
    head = b.reshape(B, nb, blk, H, Dd).pow(2).sum((1, 2, 4)).sqrt() / math.sqrt(nb)
    return (e / head[:, None, :].clamp_min(1e-300)).max().item()


def colsum_metric(dk, causal):
    """|| sum_j dK_j || / (sqrt(N) * rms row norm of dK), per head, max over (batch, kv head)."""
    dk = dk.double()
    N = dk.shape[1]
    cs = dk.sum(1).norm(dim=-1)                       # (B, Hk)
    rms = dk.pow(2).sum(-1).mean(1).sqrt()            # (B, Hk)
    return (cs / (math.sqrt(N) * rms.clamp_min(1e-300))).max().item()


# ------------------------------------------------------------------------------------------------------------ inputs
def _hook(name):
    """gpupool late-stop hooks (no-ops outside gpupool)."""
    _trace(name)
    try:
        import gpupool_hook
        getattr(gpupool_hook, name)()
    except ImportError:
        pass


def make_inputs(kind, B, N, H, Hk, gen, dev):
    def rn(*s):
        return torch.randn(*s, generator=gen, device=dev, dtype=torch.float32)
    q, k, v, do = rn(B, N, H, D), rn(B, N, Hk, D), rn(B, N, Hk, D), rn(B, N, H, D)
    if kind in ("randn", "stress"):
        pass
    elif kind.startswith("scaled"):                   # larger logit range: q, k * s
        s = float(kind[6:])
        q, k = q * s, k * s
    elif kind == "outliers":                          # activation outliers: a few hot channels, a few huge tokens
        ch = torch.randint(0, D, (4,), generator=gen, device=dev)
        q[..., ch] *= 30; k[..., ch] *= 30
        tok = torch.randint(0, N, (max(1, N // 512),), generator=gen, device=dev)
        v[:, tok] *= 100; do[:, tok] *= 50
    elif kind == "bigmag":                            # huge and tiny magnitudes (scale handling, no overflow/underflow)
        q, k, v, do = q * 3e3, k * 1e-3, v * 2e4, do * 1e-4
    elif kind == "nearsat":                           # values on and next to e4m3 code boundaries after scaling
        def grid(x):
            g = torch.tensor([448., 416., 432., 440., 464., 240., 1., 1.0625, 0.0078125, 2 ** -9], device=dev)
            idx = torch.randint(0, len(g), x.shape, generator=gen, device=dev)
            sign = torch.where(torch.rand(x.shape, generator=gen, device=dev) < 0.5, -1.0, 1.0)
            return g[idx] * sign / 448.0 * 4.0
        q, k, v, do = grid(q), grid(k), grid(v), grid(do)
    elif kind == "peaked":                            # attention concentrated on a few keys (P near one-hot)
        k = k.clone()
        hot = torch.randint(0, N, (max(1, N // 256),), generator=gen, device=dev)
        k[:, hot] = q[:, hot][:, :, :Hk] * 4
        q = q * 2
    elif kind == "uniform":                           # zero logits: P uniform, LSE = log(#keys)
        q = torch.zeros_like(q)
    elif kind == "zero_do":
        do = torch.zeros_like(do)
    elif kind == "lowrank_v":                         # V constant across keys: O = v0, dQ = dK = 0 exactly
        v = v[:, :1].expand_as(v).contiguous()
    else:
        raise ValueError(kind)
    return [x.to(BF) for x in (q, k, v, do)]


FWD_ONLY = False
CACHE = os.environ.get("DMVAL_CACHE", "1") == "1"
CACHED_TIERS = ("full", "bench")
CACHE_ROOT = os.environ.get("DMVAL_CACHE_DIR", os.path.expanduser("~/workspace/haok1402/sm100/validate/refcache"))


def _ref_hash():
    """Identity of everything that determines the cached reference outputs: reference.py, the input generator, this
    file's case builder (shapes, kinds, seeds).  Any edit invalidates the cache."""
    import hashlib
    import inspect
    h = hashlib.sha256(open(REF_PATH, "rb").read())
    h.update(inspect.getsource(make_inputs).encode())
    h.update(inspect.getsource(build_cases).encode())
    return h.hexdigest()[:16]


def _cache_path(case, preset):
    return os.path.join(CACHE_ROOT, _ref_hash(), preset, f"{case['name']}_s{case['seed']}.pt")


def cached_refs(case, preset, q, k, v, do, exact, qbk, dev):
    """Candidate-independent reference outputs of one case: exact attention, the matched-mode emulation (with aux) and the
    bf16-emulation fingerprints of both modes (their stale arm reads the emulation's own O).  Computed once per
    (reference hash, preset, case, seed) and reused by every later candidate; the candidate-dependent ones (stale
    emulation on the candidate's O, the debug-hook emulation) are always computed live."""
    causal = case["causal"]
    path = _cache_path(case, preset) if (CACHE and case.get("cached")) else None
    if path and os.path.exists(path):
        d = torch.load(path, weights_only=False)
        return {kk: {n: (t.to(dev) if torch.is_tensor(t) else t) for n, t in vv.items()} for kk, vv in d.items()}
    keep = ("O", "LSE", "dQ", "dK", "dV", "delta", "rowsum_dS", "colsum_dK")
    out = {"ex": exact(q, k, v, do, causal)}
    if causal or case.get("matched_ok"):
        em = R.dm_reference(q, k, v, do, causal, "matched", preset, aux=True, q_block=qbk)
        out["em_matched"] = {n: em[n] for n in keep if n in em}
        out["eb_matched"] = R.dm_reference(q, k, v, do, causal, "matched", "bf16", q_block=qbk)
    out["eb_stale"] = R.dm_reference(q, k, v, do, causal, "stale", "bf16", q_block=qbk)
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmpf = path + f".tmp{os.getpid()}"
        torch.save({kk: {n: (t.cpu() if torch.is_tensor(t) else t) for n, t in vv.items()} for kk, vv in out.items()}, tmpf)
        os.replace(tmpf, path)
    return out


def build_cases(tier, supports, seed):
    m = int(supports.get("seqlen_multiple", 128))
    nc = bool(supports.get("noncausal", True))

    def up(n):
        return max(m, (n + m - 1) // m * m)
    shapes = []  # (B, H, Hk, N, causal, kinds)
    base_kinds = ["randn"]
    adv = ["scaled3", "scaled10", "outliers", "bigmag", "nearsat", "peaked", "uniform", "zero_do", "lowrank_v"]
    if tier in ("quick", "full"):
        shapes += [(1, 4, 4, up(256), True, base_kinds + ["outliers", "scaled10"]),
                   (1, 8, 2, up(1024), True, base_kinds + (adv if tier == "full" else ["nearsat", "peaked"])),
                   (2, 4, 1, up(768), True, base_kinds),                      # MQA, non-power-of-two seqlen
                   (1, 16, 16, up(4096), True, base_kinds),
                   # odd numbers of 128-row blocks (partial 2-CTA clusters, tail tiles): causal and full, MHA and GQA
                   (1, 4, 4, up(1152), True, base_kinds), (1, 8, 2, up(1408), True, base_kinds),
                   (1, 4, 1, up(384), True, base_kinds)]
        if nc:
            shapes += [(1, 4, 4, up(1152), False, base_kinds), (1, 8, 2, up(1408), False, base_kinds)]
        if nc:
            shapes += [(1, 8, 2, up(1024), False, base_kinds + ["outliers"]), (2, 4, 4, up(640), False, base_kinds)]
        if m < 128:
            shapes += [(1, 4, 2, up(1000), True, base_kinds), (1, 2, 1, up(1), True, base_kinds), (1, 4, 2, up(77), True, base_kinds)]
    if tier == "full":
        shapes += [(2, 16, 16, up(2944), True, base_kinds), (1, 32, 8, up(4992), True, base_kinds),
                   (3, 8, 4, up(1664), True, ["randn", "scaled3"]), (1, 2, 2, up(128), True, base_kinds)]
        if nc:
            shapes += [(2, 16, 16, up(2944), False, base_kinds), (1, 32, 8, up(4992), False, base_kinds)]
        shapes += [(1, 8, 8, up(1280), True, adv), (1, 16, 4, up(8192), True, base_kinds + ["outliers"]),
                   (1, 4, 4, up(16384), True, base_kinds), (3, 6, 3, up(2304), True, ["randn", "scaled3"])]
        if nc:
            shapes += [(1, 8, 2, up(2048), False, adv[:5]), (1, 16, 4, up(8192), False, base_kinds)]
    # stress: repeated calls on large MHA shapes (many CTAs in flight) catch races that a single call misses
    stress_n = 8
    shapes += [(1, 16, 16, up(4096), True, ["stress"]),
               # odd numbers of 128-row blocks, with the allocator state cycled between calls (out-of-bounds reads)
               (1, 4, 4, up(1152), True, ["stress"]), (1, 4, 1, up(384), True, ["stress"]), (1, 8, 2, up(1408), True, ["stress"])]
    if nc:
        shapes += [(1, 4, 4, up(1152), False, ["stress"])]
    if tier != "quick":
        shapes += [(1, 16, 16, up(8192), True, ["stress"]), (1, 4, 4, up(4096), True, ["stress"])]
        if nc:
            shapes += [(1, 16, 16, up(8192), False, ["stress"])]
    if tier == "bench":   # exactly the shapes fa4-bench times (sm100/bench/attnbench.py SUITES "main")
        for (H, Hk) in ((16, 16), (32, 8)):
            for N in (1024, 2048, 4096, 8192, 16384, 32768):
                for c in ((True, False) if nc else (True,)):
                    shapes.append((max(1, 32768 // N), H, Hk, N, c, ["randn"]))
    cases = []
    for i, (B, H, Hk, N, causal, kinds) in enumerate(shapes):
        for kind in kinds:
            name = f"B{B}_H{H}_Hk{Hk}_N{N}_{'c' if causal else 'nc'}_{kind}"
            modes = ["matched", "stale"] if causal or supports.get("matched_noncausal") else ["stale"]
            if FWD_ONLY:
                modes = modes[:1]
            runs = [(md, "plain") for md in modes]
            big = B * H * N * N > 2 ** 33
            if kind == "randn" and not big:
                runs += [(modes[0], "repeat"), (modes[0], "views")]
            if kind == "randn" and i == 1:
                runs += [(modes[0], "mutate")]
            if kind == "stress":
                runs = [(md, f"stress:{stress_n}") for md in modes]
            # full / bench: a fixed seed per case name, so the reference outputs can be cached and reused across candidates;
            # quick: fresh random inputs on every run (no candidate can be tuned to fixed inputs without failing quick)
            cseed = (zlib.crc32(name.encode()) & 0x7FFFFFFF) if (tier in CACHED_TIERS and CACHE) else seed * 1000003 + len(cases)
            cases.append(dict(name=name, B=B, H=H, Hk=Hk, N=N, causal=causal, kind=kind, runs=runs, seed=cseed))
    return cases


# ---------------------------------------------------------------------------------------------------------- checking
def check_case(case, cr, preset, dev, verbose):
    """Compare one case's candidate outputs with exact attention and the emulation.  Returns (ok, record)."""
    rec = {"name": case["name"], "checks": {}, "metrics": {}}
    fails = []

    def chk(name, ok, detail=""):
        rec["checks"][name] = {"ok": bool(ok), "detail": detail}
        if not ok:
            fails.append(f"{name}: {detail}")

    if "error" in cr:
        chk("runs", False, cr["error"][-1500:])
        rec["fails"] = fails
        return False, rec
    if case["kind"] == "stress":
        for key, st in cr.get("stress", {}).items():
            rec["metrics"][f"{key}"] = st
            chk(f"{key}:finite_every_call", st["nonfinite_calls"] == 0,
                f"{st['nonfinite_calls']}/{st['calls']} calls had NaN/Inf outputs")
            chk(f"{key}:bitwise_every_call", st["mismatch_calls"] == 0,
                f"{st['mismatch_calls']}/{st['calls'] - 1} repeat calls differed from the first")
        rec["fails"] = fails
        return not fails, rec
    q, k, v, do = (case[n].to(dev) for n in ("q", "k", "v", "do"))
    B, N, H, Dd = q.shape
    Hk = k.shape[2]
    causal = case["causal"]
    exact_dtype = torch.float64 if B * H * N * N <= 2 ** 28 else torch.float32
    exact = (lambda *a: R.attention_fp32(*a, dtype=exact_dtype)) if B * H * N * N <= 2 ** 31 else \
        (lambda *a: R.attention_exact_blocked(*a, q_block=max(128, min(1024, 2 ** 31 // (B * H * N) // 128 * 128))))
    qbk0 = max(128, min(1024, 2 ** 31 // (B * H * N) // 128 * 128))
    refs = cached_refs(case, preset, q, k, v, do, exact, qbk0, dev)
    ex = refs["ex"]
    runs = {k_: [t.to(dev) if t is not None else None for t in v_] for k_, v_ in cr["runs"].items()}
    modes = sorted({k_.split("/")[0] for k_ in runs})
    shapes = (q.shape, q.shape, k.shape, k.shape)
    names = ("O",) if FWD_ONLY else NAMES
    for key, out in runs.items():
        for nm, t, shp in zip(names, out, shapes):
            ok = t is not None and tuple(t.shape) == tuple(shp) and t.dtype == BF
            if not ok:
                chk(f"{key}:{nm}:shape_dtype", False, f"got {None if t is None else (tuple(t.shape), t.dtype)}, "
                                                       f"want {tuple(shp)} bf16")
            elif not torch.isfinite(t).all():
                chk(f"{key}:{nm}:finite", False, "non-finite values")
    if fails:
        rec["fails"] = fails
        return False, rec
    emul_bf16 = {}
    for mode in modes:
        cand = runs[f"{mode}/plain"]
        qbk = max(128, min(1024, 2 ** 31 // (B * H * N) // 128 * 128))
        if mode == "matched" and "em_matched" in refs:
            em = refs["em_matched"]          # matched emulation does not depend on the candidate: cached
        else:                                # stale: the emulated delta reads the candidate's own O (live)
            em = R.dm_reference(q, k, v, do, causal, mode, preset, aux=True, o_for_stale=cand[0], q_block=qbk)
        eb = refs.get("eb_" + mode) or R.dm_reference(q, k, v, do, causal, mode, "bf16", q_block=qbk)
        emul_bf16[mode] = eb
        degenerate = False
        for nm, c in zip(names, cand):
            e_c, e_e = rel(c, ex[nm]), rel(em[nm], ex[nm])
            d_ce, d_cb = rel(c, em[nm]), rel(c, eb[nm])
            b_c, b_e = block_rel(c, ex[nm]), block_rel(em[nm], ex[nm])
            rec["metrics"][f"{mode}:{nm}"] = dict(err=e_c, emul_err=e_e, to_emul=d_ce, to_bf16_emul=d_cb,
                                                  blk=b_c, emul_blk=b_e)
            ref_scale = ex["dV"].double().norm().item() if nm != "O" else ex["O"].double().norm().item()
            if nm in ("dQ", "dK") and ex[nm].double().norm().item() <= 1e-4 * ref_scale:
                # degenerate reference (e.g. constant V: exact dQ = dK = 0 up to round-off): relative errors are
                # meaningless.  The fp8 kernel's own value is then pure quantization residue (0 in matched mode, the
                # stale-delta bias in stale mode), so compare with the emulation in absolute terms, at the scale of a
                # non-degenerate gradient (dV's norm)
                d = (c.double() - em[nm].double()).norm().item()
                rec["metrics"][f"{mode}:{nm}"]["degenerate_dist_over_dV"] = d / max(ref_scale, 1e-300)
                chk(f"{mode}:{nm}:degenerate_ref", d <= 0.25 * em[nm].double().norm().item() + 1e-3 * ref_scale,
                    f"exact ~0: |cand - emulation| {d:.3e} vs |emulation| {em[nm].double().norm().item():.3e}, "
                    f"|dV| {ref_scale:.3e}")
                degenerate = True
                continue
            chk(f"{mode}:{nm}:accuracy", e_c <= TOL_RATIO * e_e + TOL_ABS,
                f"rel err {e_c:.4f} vs fp8 emulation {e_e:.4f} (allowed {TOL_RATIO}x + {TOL_ABS})")
            chk(f"{mode}:{nm}:blocks", b_c <= BLK_RATIO * b_e + BLK_ABS,
                f"worst 128-row block err {b_c:.4f} vs emulation {b_e:.4f} (allowed {BLK_RATIO}x + {BLK_ABS})")
            if e_e > 4 * TOL_ABS:                     # fingerprint only where fp8 noise dominates round-off
                chk(f"{mode}:{nm}:not_too_accurate", e_c >= FLOOR_RATIO * e_e,
                    f"rel err {e_c:.4f} < {FLOOR_RATIO} x the fp8 emulation's {e_e:.4f}: not the declared fp8 numerics")
                chk(f"{mode}:{nm}:faithful", d_ce <= FAITH_RATIO * e_e,
                    f"distance to the {preset} emulation {d_ce:.4f} > {FAITH_RATIO} x its fp8 error {e_e:.4f}: the "
                    f"kernel does not implement the declared numerics (declare the right preset)")
                chk(f"{mode}:{nm}:fingerprint", d_ce < d_cb,
                    f"closer to the bf16 emulation ({d_cb:.4f}) than to the declared {preset} ({d_ce:.4f})")
        if mode == "matched" and not FWD_ONLY and not degenerate:
            cs_c, cs_e = colsum_metric(cand[2], causal), colsum_metric(em["dK"], causal)
            cs_s = colsum_metric(runs["stale/plain"][2], causal) if "stale/plain" in runs else float("nan")
            rec["metrics"]["matched:colsum_dK"] = dict(cand=cs_c, emul=cs_e, cand_stale=cs_s,
                                                       emul_rowsum_dS=em["rowsum_dS"].abs().max().item())
            dbg = {kk: vv.to(dev) for kk, vv in (cr.get("debug") or {}).items()}
            if "delta" in dbg and "lse" in dbg:
                # the identity, directly: the backward's P comes from the candidate's own LSE over the bitwise-reproducible
                # quantized operands; rowsum(dS) = delta * sum_j P - sum_j P dP must vanish relative to sum_j |P dP|
                lse_c = dbg["lse"].reshape(B, H, N).float()
                dl_c = dbg["delta"].reshape(B, H, N).float()
                lse_err = (lse_c.double() - em["LSE"].double()).abs()
                lse_tol = 1e-3 + 1e-5 * em["LSE"].double().abs()
                emc = R.dm_reference(q, k, v, do, causal, mode, preset, aux=True, lse_override=lse_c,
                                     delta_override=dl_c, q_block=qbk)
                den = emc["sum_absPdP"]
                live = den > 0
                r = ((dl_c.double() * emc["sum_P"] - emc["sum_PdP"]).abs() / den.clamp_min(1e-300))[live]
                bind = max(rel(cand[1], emc["dQ"]), rel(cand[2], emc["dK"]))
                bind_tol = BIND_ABS + BIND_REL * max(rel(em["dQ"], ex["dQ"]), rel(em["dK"], ex["dK"]))
                m = {"lse_maxabs": lse_err.max().item(), "rowsum_rel_max": r.max().item() if r.numel() else 0.0,
                     "rowsum_rel_mean": r.mean().item() if r.numel() else 0.0, "bind_dist": bind}
                rec["metrics"]["matched:debug_delta"] = m
                chk("matched:debug_lse", bool((lse_err <= lse_tol).all()),
                    f"candidate LSE vs the {preset} emulation: max |diff| {m['lse_maxabs']:.3e}")
                chk("matched:rowsum_dS_direct", m["rowsum_rel_max"] <= ID_TOL,
                    f"|delta sum P - sum P dP| / sum |P dP| = {m['rowsum_rel_max']:.2e} (mean {m['rowsum_rel_mean']:.1e}), "
                    f"allowed {ID_TOL:.0e}; stale-mode deltas sit at 1e-2..1")
                chk("matched:debug_binds_grads", bind <= bind_tol,
                    f"candidate dQ/dK vs the emulated backward driven by the candidate's reported delta: {bind:.4f} "
                    f"(allowed {bind_tol:.4f}): the reported delta is not the one the gradients use")
            elif not FWD_ONLY and not os.environ.get("DMVAL_NO_DEBUG_OK"):
                chk("matched:debug_hook_required", False,
                    "matched mode needs the candidate's debug(q, k, v, do, causal, mode) -> {'delta', 'lse'} for the direct "
                    "rowsum(dS) identity check (README); without it a matched candidate cannot PASS")
            chk("matched:rowsum_dS_identity", cs_c <= COLSUM_RATIO * cs_e + COLSUM_ABS,
                f"|sum_j dK_j| metric {cs_c:.2e} vs emulation {cs_e:.2e} (stale run {cs_s:.2e})")
    if "matched/plain" in runs and "stale/plain" in runs:
        a, b = runs["matched/plain"], runs["stale/plain"]
        chk("modes_share_O", torch.equal(a[0], b[0]), "O differs between matched and stale")
        chk("modes_share_dV", torch.equal(a[3], b[3]), "dV differs between matched and stale")
        if do.float().abs().max() > 0 and ex["dQ"].double().norm() > 1e-4 * ex["dV"].double().norm():
            chk("modes_differ_dQ", not torch.equal(a[1], b[1]), "matched and stale dQ are bitwise equal")
    for key, out in runs.items():
        mode, kind = key.split("/")
        base = runs[f"{mode}/plain"]
        if kind in ("repeat", "views"):
            for nm, x, y in zip(names, out, base):
                chk(f"{key}:{nm}:bitwise", torch.equal(x, y),
                    f"max |diff| {(x.float() - y.float()).abs().max().item():.3e} vs the plain run")
        if kind == "mutate":
            q2, k2, v2 = (case[n].to(dev) for n in ("q2", "k2", "v2"))
            em2 = R.dm_reference(q2, k2, v2, do, causal, mode, preset, o_for_stale=out[0])
            ex2 = exact(q2, k2, v2, do, causal)
            for nm, x in zip(names, out):
                e_c, e_e = rel(x, ex2[nm]), rel(em2[nm], ex2[nm])
                chk(f"{key}:{nm}:no_stale_cache", e_c <= TOL_RATIO * e_e + TOL_ABS,
                    f"after overwriting the inputs in place: err {e_c:.4f} vs emulation {e_e:.4f}")
    rec["fails"] = fails
    return not fails, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate")
    ap.add_argument("--tier", "--shapes", dest="tier", default="quick", choices=["quick", "full", "bench"])
    ap.add_argument("--preset", default=None, help="override the candidate's NUMERICS")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=None, help="default: random (printed)")
    ap.add_argument("--json", default=None)
    ap.add_argument("--only", default=None, help="substring filter on case names")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--shard", default=None, help="i/n: run every n-th case starting at i (split a tier across GPUs)")
    ap.add_argument("--fwd-only", action="store_true", help="validate the forward (O) only; never a promotion PASS")
    a = ap.parse_args()
    global FWD_ONLY
    FWD_ONLY = a.fwd_only
    tgt, sep, fn_ = a.candidate.partition(":")
    if tgt.endswith(".py"):
        a.candidate = os.path.abspath(tgt) + sep + fn_
    dev = torch.device(a.device)
    torch.cuda.set_device(dev)
    torch.zeros(1, device=dev)
    _hook("gpupool_done")                  # after context creation: validation is untimed and shares the GPU with the filler
    seed = a.seed if a.seed is not None else secrets.randbits(31)
    t0 = time.time()
    # one probe run to learn NUMERICS / SUPPORTS (no inputs)
    # case files go through node-local RAM (/dev/shm): Lustre reads / writes of GB-sized outputs were the main idle gaps
    tmpdir = os.environ.get("DMVAL_TMP") or ("/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK)
                                             else None)
    tmp = tempfile.mkdtemp(prefix="dmval_", dir=tmpdir)
    probe_job, probe_out = os.path.join(tmp, "probe.pt"), os.path.join(tmp, "probe_out.pt")
    meta = _static_meta(a.candidate)          # NUMERICS / SUPPORTS as literals: no extra process on the reserved GPU
    if meta is None:
        torch.save({"device": a.device, "cases": []}, probe_job)
        meta = _run_runner(a.candidate, probe_job, probe_out)
    if "fatal" in meta:
        print("FAIL: candidate import\n" + meta["fatal"]); sys.exit(1)
    preset = a.preset or meta["meta"].get("NUMERICS") or "sm90"
    supports = meta["meta"].get("SUPPORTS") or {}
    if preset not in R.PRESETS:
        print(f"FAIL: NUMERICS={preset!r} is not a preset of reference.py ({sorted(R.PRESETS)})"); sys.exit(1)
    cases = build_cases(a.tier, supports, seed)
    if a.only:
        cases = [c for c in cases if a.only in c["name"]]
    if a.shard:
        si, sn = (int(x) for x in a.shard.split("/"))
        cases = cases[si::sn]
    for c in cases:
        c["cached"] = a.tier in CACHED_TIERS and CACHE
        c["matched_ok"] = bool(supports.get("matched_noncausal"))
    print(f"validator: candidate {a.candidate}  preset {preset}  tier {a.tier}  seed {seed}  {len(cases)} cases  "
          f"supports {supports}", flush=True)
    results, all_ok = [], True
    leaked = set(meta["meta"].get("leaked_modules", []))
    # 1) inputs for every case (GPU generator, then to disk), 2) ONE candidate process runs all cases back to back (no
    # per-case process start / CUDA init while the GPU is reserved), 3) the checks (GPU reference math) case by case.
    files = []
    _hook("gpupool_done")                  # input generation and file writes are CPU work: let the filler run
    cpu = torch.device("cpu")
    for i, case in enumerate(cases):
        gen = torch.Generator(device=cpu).manual_seed(case["seed"])
        case["q"], case["k"], case["v"], case["do"] = make_inputs(case["kind"], case["B"], case["N"], case["H"],
                                                                  case["Hk"], gen, cpu)
        if any(r[1] == "mutate" for r in case["runs"]):
            case["q2"], case["k2"], case["v2"], _ = make_inputs("randn", case["B"], case["N"], case["H"], case["Hk"],
                                                                gen, cpu)
        cf = os.path.join(tmp, f"case{i:03d}.pt")
        torch.save({kk: (vv.cpu() if torch.is_tensor(vv) else vv) for kk, vv in case.items()}, cf)
        files.append(cf)
        for kk in ("q", "k", "v", "do", "q2", "k2", "v2"):
            case.pop(kk, None)
    jp, op = os.path.join(tmp, "job.pt"), os.path.join(tmp, "out.pt")
    torch.save({"device": a.device, "case_files": files, "fwd_only": FWD_ONLY}, jp)
    out = _run_runner(a.candidate, jp, op)
    if "fatal" in out:
        print("FAIL: " + out["fatal"]); sys.exit(1)
    leaked |= set(out["meta"].get("leaked_modules", []))
    # one GPU phase for all checks: the next case's outputs and inputs are prepared (file read, CPU input generation)
    # in a thread while the current case is checked on the GPU
    from concurrent.futures import ThreadPoolExecutor

    def prep(i):
        case = cases[i]
        full = torch.load(files[i] + ".out", weights_only=False) if os.path.exists(files[i] + ".out") else \
            {"name": case["name"], "error": "the runner produced no output for this case"}
        gen = torch.Generator(device=cpu).manual_seed(case["seed"])
        ins = make_inputs(case["kind"], case["B"], case["N"], case["H"], case["Hk"], gen, cpu)
        ins2 = make_inputs("randn", case["B"], case["N"], case["H"], case["Hk"], gen, cpu)[:3] \
            if any(r[1] == "mutate" for r in case["runs"]) else None
        return full, ins, ins2
    pool = ThreadPoolExecutor(max_workers=1)
    fut = pool.submit(prep, 0) if cases else None
    # no gpupool_ready(): the checks are not timed and share the GPU with the gpupool filler (see runner.py)
    for i, case in enumerate(cases):
        full, ins, ins2 = fut.result()
        if i + 1 < len(cases):
            fut = pool.submit(prep, i + 1)
        case["q"], case["k"], case["v"], case["do"] = (x.to(dev) for x in ins)
        if ins2 is not None:
            case["q2"], case["k2"], case["v2"] = (x.to(dev) for x in ins2)
        if "runs" in full:
            full["runs"] = {kk: [None if t is None else t.to(dev) for t in vv] for kk, vv in full["runs"].items()}
        if full.get("debug"):
            full["debug"] = {kk: vv.to(dev) for kk, vv in full["debug"].items()}
        _trace("check start " + case["name"])
        ok, rec = check_case(case, full, preset, dev, a.verbose)
        _trace("check end " + case["name"])
        all_ok &= ok
        results.append(rec)
        _print_case(rec, ok)
        for kk in ("q", "k", "v", "do", "q2", "k2", "v2"):
            case.pop(kk, None)
        if os.path.exists(files[i] + ".out"):
            os.remove(files[i] + ".out")
        del full
        torch.cuda.empty_cache()
    pool.shutdown()
    try:                                    # GPU part done: let the gpupool filler resume while the report is written
        from gpupool_hook import gpupool_done
        gpupool_done()
    except ImportError:
        pass
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    if leaked and not os.environ.get("DMVAL_SELFTEST"):
        all_ok = False
        print(f"FAIL: the candidate imported the reference / validator: {sorted(leaked)}")
    summary = {"candidate": a.candidate, "preset": preset, "tier": a.tier + ("-fwd" if FWD_ONLY else ""), "seed": seed, "pass": all_ok,
               "n_cases": len(results), "n_failed": sum(bool(r["fails"]) for r in results),
               "seconds": round(time.time() - t0, 1), "device": torch.cuda.get_device_name(dev),
               "tolerances": dict(TOL_RATIO=TOL_RATIO, TOL_ABS=TOL_ABS, BLK_RATIO=BLK_RATIO, BLK_ABS=BLK_ABS,
                                  FLOOR_RATIO=FLOOR_RATIO, FAITH_RATIO=FAITH_RATIO, COLSUM_RATIO=COLSUM_RATIO, COLSUM_ABS=COLSUM_ABS,
                                  ID_TOL=ID_TOL, BIND_ABS=BIND_ABS, BIND_REL=BIND_REL),
               "cases": results}
    if a.json:
        with open(a.json, "w") as f:
            json.dump(summary, f, indent=1)
    print("SUMMARY_JSON " + json.dumps(_one_line(summary)))
    print(f"\n{'PASS' if all_ok else 'FAIL'}: {summary['n_cases'] - summary['n_failed']}/{summary['n_cases']} cases "
          f"(tier {a.tier}, preset {preset}, seed {seed}, {summary['seconds']} s)")
    sys.exit(0 if all_ok else 1)


def _static_meta(spec):
    import ast
    tgt = spec.partition(":")[0]
    if not tgt.endswith(".py"):
        return None
    try:
        tree = ast.parse(open(tgt).read())
        vals = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                    and node.targets[0].id in ("NUMERICS", "SUPPORTS"):
                vals[node.targets[0].id] = ast.literal_eval(node.value)
        return {"meta": {"NUMERICS": vals.get("NUMERICS"), "SUPPORTS": vals.get("SUPPORTS"), "leaked_modules": []}}
    except Exception:  # noqa: BLE001  (non-literal values: ask the candidate process)
        return None


def _run_runner(spec, job, out):
    p = subprocess.run([sys.executable, "-u", os.path.join(HERE, "runner.py"), spec, job, out],
                       capture_output=True, text=True, cwd=os.path.dirname(job))
    if p.returncode != 0 or not os.path.exists(out):
        return {"fatal": f"runner exited {p.returncode}\n{p.stdout[-3000:]}\n{p.stderr[-6000:]}"}
    r = torch.load(out, weights_only=False)
    os.remove(out)
    return r


def _one_line(summary):
    """The one-line JSON for candidates.jsonl: pass / fail, worst per-output ratios to the emulation, worst block,
    identity and fingerprint numbers over all cases."""
    worst = {}
    for rec in summary["cases"]:
        for key, m in rec["metrics"].items():
            if "err" in m and m["emul_err"] > 0:
                w = worst.setdefault(key, {"err": 0, "err_over_emul": 0, "blk_over_emul": 0, "to_emul": 0,
                                           "fingerprint_margin": 1e9})
                w["err"] = max(w["err"], m["err"])
                w["err_over_emul"] = max(w["err_over_emul"], m["err"] / m["emul_err"])
                w["blk_over_emul"] = max(w["blk_over_emul"], m["blk"] / max(m["emul_blk"], 1e-30))
                w["to_emul"] = max(w["to_emul"], m["to_emul"])
                w["fingerprint_margin"] = min(w["fingerprint_margin"], m["to_bf16_emul"] / max(m["to_emul"], 1e-30))
            elif key == "matched:colsum_dK":
                w = worst.setdefault(key, {"cand": 0, "emul": 0})
                w["cand"] = max(w["cand"], m["cand"]); w["emul"] = max(w["emul"], m["emul"])
            elif key == "matched:debug_delta":
                w = worst.setdefault(key, {"lse_maxabs": 0, "rowsum_rel_max": 0, "bind_dist": 0})
                for kk in w:
                    w[kk] = max(w[kk], m.get(kk, 0))
    for w in worst.values():
        for kk, vv in w.items():
            w[kk] = float(f"{vv:.4g}")
    failed = [r["name"] for r in summary["cases"] if r["fails"]]
    return {"pass": summary["pass"], "candidate": summary["candidate"], "preset": summary["preset"],
            "tier": summary["tier"], "seed": summary["seed"], "n_cases": summary["n_cases"], "failed": failed,
            "first_fail": next((r["fails"][0] for r in summary["cases"] if r["fails"]), None), "worst": worst,
            "seconds": summary["seconds"]}


def _print_case(rec, ok):
    ms = rec["metrics"]
    parts = []
    for key in ("matched", "stale"):
        if f"{key}:O" in ms:
            parts.append(key[0] + " " + " ".join(f"{nm}={ms[f'{key}:{nm}']['err']:.3f}/{ms[f'{key}:{nm}']['emul_err']:.3f}"
                                                 for nm in NAMES if f"{key}:{nm}" in ms))
    if "matched:colsum_dK" in ms:
        c = ms["matched:colsum_dK"]
        parts.append(f"colsum {c['cand']:.1e}/{c['emul']:.1e} (stale {c['cand_stale']:.1e})")
    print(f"{'ok  ' if ok else 'FAIL'} {rec['name']:<36} " + "  ".join(parts), flush=True)
    for f_ in rec["fails"]:
        print("       - " + f_[:400], flush=True)


if __name__ == "__main__":
    main()

