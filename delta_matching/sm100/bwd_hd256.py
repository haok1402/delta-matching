"""Host entry of the hd256 backward (fork hd256-bwd; kernels in dm_bwd_hd256_sm100.py; design kda/hd256/bwd-design.md).
hd128's bwd.py / api.py are untouched.

dm_bwd_hd256(q8, qs, k8, ks, v8, vs, do8, dos, lse, delta, causal) -> dq (B,N,H,256), dk, dv (B,N,Hk,256) bf16
  q8 / do8 (B, N, H, 256) e4m3, qs / dos (B, H, N/64) fp32; k8 / v8 (B, N, Hk, 256) e4m3, ks / vs (B, Hk, N/32) fp32;
  lse (B, H, N) fp32 natural log; delta (B, H, N) fp32 from pass 0 (pass0_hd256.dm_delta_hd256, default num/den form).
Per head chunk [h0, h1) (workspace budget env DM_H256_WS_GB, default 24): kernel A (dS kernel) -> dV fold -> dK fold -> dQ fold.
Deterministic: every output element has one writer, fixed loop orders, no atomics; head chunks run in ascending order.
"""
import threading

import torch
from flash_attn.cute import cache_utils
from flash_attn.cute.cute_dsl_utils import to_cute_tensor
import cutlass
import cutlass.cute as cute
from cutlass import Int32

from .dm_bwd_hd256_sm100 import DmHd256DS, DmHd256Fold
from .fwd import _fingerprint
from .fastpath import env as _env

D = 256
_LOCK = threading.Lock()
_CACHE = {}
LAST = {}


def _cache():
    if "c" not in _CACHE:
        if cache_utils.CUTE_DSL_CACHE_ENABLED:
            _CACHE["c"] = cache_utils.JITPersistentCache(cache_utils.get_cache_path() / "dm_sm100" / _fingerprint() / "bwdh")
        else:
            _CACHE["c"] = cache_utils.JITCache()
    return _CACHE["c"]


def _fake():
    from flash_attn.cute.testing import is_fake_mode
    return is_fake_mode()


def _ct(t, ld, align=16):
    return to_cute_tensor(t, assumed_align=align, leading_dim=ld)


def _u8(t):
    return t.view(torch.uint8) if t.dtype == torch.float8_e4m3fn else t


def _ds_args(q8, k8, v8, do8, qs, ks, vs, dos, lse, delta, wPt, wdS, wdSt, wPsi, wG, ident, dump):
    p = lambda x: x.permute(1, 3, 2, 0)   # (B, N, H, D) -> (N, D, H, B)  # noqa: E731
    ten = [p(q8), p(k8), p(v8), p(do8)]
    lds = [1, 1, 1, 1]
    for t in (qs, ks, vs, dos, lse, delta):
        ten.append(t); lds.append(2)
    for t, ld in ((wPt, 2), (wdS, 2), (wdSt, 2), (wPsi, 2), (wG, 2), (ident, 3), (dump, 3)):
        ten.append(t); lds.append(ld)
    return ten, lds


def _run(key, make_obj, ten, lds, ints):
    c = _cache()
    if key not in c:
        if _env("DM_H256_NO_COMPILE", "0") == "1":
            raise RuntimeError(f"hd256 bwd: {key} is not AOT-compiled (DM_H256_NO_COMPILE=1: no compiling in a GPU lease)")
        cts = [_ct(t, ld, 16 if t.element_size() == 1 or t.dtype == torch.bfloat16 else 4) for t, ld in zip(ten, lds)]
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        c[key] = cute.compile(make_obj(), *cts, *[Int32(0) for _ in ints], stream, options="--enable-tvm-ffi")
    if not _fake():
        c[key](*[_u8(t) for t in ten], *ints)


def plan_chunks(B, N, H, Hk, budget_gb=None):
    """Heads per workspace chunk: 3 code planes (P~^T, dS', dS'^T) of N x N bytes + psi/g per head (dense layout)."""
    gb = float(budget_gb if budget_gb is not None else _env("DM_H256_WS_GB", "24"))
    per = B * (3 * N * N + 2 * 4 * (N // 64) ** 2)
    return max(1, min(H, int(gb * 2 ** 30) // per))


def dm_bwd_hd256(q8, qs, k8, ks, v8, vs, do8, dos, lse, delta, causal, *, heads_per_chunk=None, dbg=0, ws=None):
    B, N, H, Dq = q8.shape
    Hk = k8.shape[2]
    assert Dq == D and N % 128 == 0 and H % Hk == 0
    assert qs.shape == (B, H, N // 64) and ks.shape == (B, Hk, N // 32)
    rep = H // Hk
    dev = q8.device
    causal = bool(causal)
    Hc = heads_per_chunk or plan_chunks(B, N, H, Hk)
    nmb = N // 64
    nkt = N // 128
    f8 = torch.float8_e4m3fn
    # transposed operand twins (layout only): (B*heads, 256, N), K-major for the token contractions
    tw = lambda x: x.permute(0, 2, 3, 1).reshape(B * x.shape[2], D, N).contiguous()  # noqa: E731
    qt, dot, kt = tw(q8), tw(do8), tw(k8)
    if ws is None:
        ws = {}
    need = (B * Hc, N)
    if ws.get("shape") != need:
        ws.update(shape=need, Pt=torch.empty(B * Hc, N, N, device=dev, dtype=torch.uint8),
                  dS=torch.empty(B * Hc, N, N, device=dev, dtype=torch.uint8),
                  dSt=torch.empty(B * Hc, N, N, device=dev, dtype=torch.uint8),
                  psi=torch.empty(B * Hc, nmb, nmb, device=dev, dtype=torch.float32),
                  g=torch.empty(B * Hc, nmb, nmb, device=dev, dtype=torch.float32))
    wPt, wdS, wdSt, wPsi, wG = ws["Pt"], ws["dS"], ws["dSt"], ws["psi"], ws["g"]
    ident = torch.zeros(B * Hc, N, N // 32, 4, device=dev) if dbg & 1 else torch.zeros(1, 1, 1, 4, device=dev)
    dump = torch.zeros(B * Hc, N, N, 4, device=dev) if dbg & 2 else torch.zeros(1, 1, 1, 4, device=dev)
    dq = torch.empty(B, N, H, D, device=dev, dtype=torch.bfloat16)
    dk = torch.empty(B, N, Hk, D, device=dev, dtype=torch.bfloat16)
    dv = torch.empty(B, N, Hk, D, device=dev, dtype=torch.bfloat16)
    multi = Hc < H
    dk_acc = torch.empty(B, N, Hk, D, device=dev) if multi else torch.empty(1, 1, 1, D, device=dev)
    dv_acc = torch.empty(B, N, Hk, D, device=dev) if multi else torch.empty(1, 1, 1, D, device=dev)
    fdv_all = dos * (1.0 / 256.0)            # f = (1/256) * do8s (exact: power-of-2 scaling)
    dbgout = []
    for h0 in range(0, H, Hc):
        h1 = min(H, h0 + Hc)
        hcn = h1 - h0
        ten, lds = _ds_args(q8.view(f8), k8.view(f8), v8.view(f8), do8.view(f8), qs, ks, vs, dos, lse, delta,
                            wPt, wdS, wdSt, wPsi, wG, ident, dump)
        _run(("ds", causal, dbg), lambda: DmHd256DS(causal, dbg), ten, lds, [h0, rep, nkt, Hc, B])
        A = lambda w: w.view(f8).permute(1, 2, 0)          # (B*Hc, M, K) -> (M, K, L)  # noqa: E731
        Bt = lambda t: t.view(f8).permute(1, 2, 0)         # (L, 256, K) -> (256, K, L)  # noqa: E731
        g0, ng = h0 // rep, (h1 - 1) // rep - h0 // rep + 1
        fdv = torch.zeros(B * Hc, nmb, device=dev)
        fdv.view(B, Hc, nmb)[:, :hcn] = fdv_all[:, h0:h1]
        # dV: f[la][c] (fsm = 0)
        _run(("fold", "dv", causal), lambda: DmHd256Fold("dv", causal),
             [A(wPt), Bt(dot), fdv.view(-1), ks, dv, dv_acc], [1, 1, 0, 2, 3, 3],
             [h0, h1, rep, g0, Hc, H, 1, nmb, 0, 1, nkt, ng, B, Hk])
        # dK: F[la][m64 = nb][c = mb] = g[la, mb, nb]
        _run(("fold", "dk", causal), lambda: DmHd256Fold("dk", causal),
             [A(wdSt), Bt(qt), wG.view(-1), ks, dk, dk_acc], [1, 1, 0, 2, 3, 3],
             [h0, h1, rep, g0, Hc, H, 1, nmb * nmb, 1, nmb, nkt, ng, B, Hk])
        # dQ: F[la][m64 = mb][c = nb] = psi[la, mb, nb]; B = Kt of the kv head
        _run(("fold", "dq", causal), lambda: DmHd256Fold("dq", causal),
             [A(wdS), Bt(kt), wPsi.view(-1), ks, dq, dq], [1, 1, 0, 2, 3, 3],
             [h0, h1, 1, h0, Hc, Hk, rep, nmb * nmb, nmb, 1, nkt, hcn, B, H])
        if dbg:
            dbgout.append(dict(h0=h0, h1=h1, ident=ident.clone() if dbg & 1 else None,
                               dump=dump.clone() if dbg & 2 else None, psi=wPsi.clone(), g=wG.clone(),
                               dS=wdS.clone(), Pt=wPt.clone(), dSt=wdSt.clone()))
    LAST["chunks"] = Hc
    if dbg:
        return dq, dk, dv, dbgout
    return dq, dk, dv
