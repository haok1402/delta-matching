"""Host entry of the hd256 MXFP8 backward (spec dm-validator preset sm100-mx-d256; fork hd256-bwdmx; kernels in
dm_bwd_hd256mx_sm100.py; design kda/hd256/bwd-design.md Part B).  hd128's bwd.py / api.py are untouched.

dm_bwd_hd256mx(q8, qs, k8, ks, v8, vs, do8, dos, lse, delta, causal) -> dq (B,N,H,256), dk, dv (B,N,Hk,256) bf16
  q8 / do8 (B, N, H, 256) e4m3, qs / dos (B, H, N/32) fp32 pow2; k8 / v8 (B, N, Hk, 256) e4m3, ks / vs (B, Hk, N/32)
  (quant_hd256.quant_qkv_mx256 / quant_do_mx256); lse (B, H, N) fp32 natural log; delta (B, H, N) fp32 from pass 0
  (pass0_hd256.dm_delta_hd256, num/den).
Three launches, no workspace, no atomics, every output element written once: dV kernel, dK kernel (keys on rows, GQA q-head
loop inside the CTA), dQ kernel (queries on rows).  Deterministic by construction.
"""
import threading

import torch
from flash_attn.cute import cache_utils
from flash_attn.cute.cute_dsl_utils import to_cute_tensor
import cutlass.cute as cute
from cutlass import Int32

from .dm_bwd_hd256mx_sm100 import DmHd256Mx
from .fwd import _fingerprint
from .fastpath import env as _env

D = 256
NUMERICS = "sm100-mx-d256"
DETERMINISTIC = True
_LOCK = threading.Lock()
_CACHE = {}
PROF = []


def _cache():
    if "c" not in _CACHE:
        if cache_utils.CUTE_DSL_CACHE_ENABLED:
            _CACHE["c"] = cache_utils.JITPersistentCache(cache_utils.get_cache_path() / "dm_sm100" / _fingerprint() / "bwdhmx")
        else:
            _CACHE["c"] = cache_utils.JITCache()
    return _CACHE["c"]


def _fake():
    from flash_attn.cute.testing import is_fake_mode
    return is_fake_mode()


def _u8(t):
    return t.view(torch.uint8) if t.dtype == torch.float8_e4m3fn else t


def _run(key, make_obj, ten, lds, ints):
    c = _cache()
    ev = None
    if _env("DM_H256_PROF", "0") == "1" and not _fake():
        ev = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
    with _LOCK:
        if key not in c:
            if _env("DM_H256_NO_COMPILE", "0") == "1":
                raise RuntimeError(f"hd256 mx bwd: {key} is not AOT-compiled (DM_H256_NO_COMPILE=1: no compiling in a GPU lease)")
            print(f"[bwd_hd256mx] compiling {key}", flush=True)
            cts = [to_cute_tensor(t, assumed_align=16 if t.element_size() <= 2 else 4, leading_dim=ld) for t, ld in zip(ten, lds)]
            stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
            c[key] = cute.compile(make_obj(), *cts, *[Int32(0) for _ in ints], stream, options="--enable-tvm-ffi")
    if not _fake():
        if ev is not None:
            ev[0].record()
        c[key](*[_u8(t) for t in ten], *ints)
        if ev is not None:
            ev[1].record()
            PROF.append((key[0], ev))


def twins(q8, do8, k8):
    """K-major token-contraction operands (layout only): (B*heads, 256, N) contiguous."""
    tw = lambda x: x.permute(0, 2, 3, 1).reshape(x.shape[0] * x.shape[2], D, x.shape[1]).contiguous()  # noqa: E731
    return tw(q8), tw(do8), tw(k8)


def dm_bwd_hd256mx(q8, qs, k8, ks, v8, vs, do8, dos, lse, delta, causal, *, dbg=0, modes=("dv", "dk", "dq"), tw=None,
                   knob=""):
    B, N, H, Dq = q8.shape
    Hk = k8.shape[2]
    assert Dq == D and N % 128 == 0 and H % Hk == 0
    assert qs.shape == (B, H, N // 32) and ks.shape == (B, Hk, N // 32) and dos.shape == qs.shape and vs.shape == ks.shape
    dev = q8.device
    causal = bool(causal)
    f8 = torch.float8_e4m3fn
    p = lambda x: x.view(f8).permute(1, 3, 2, 0)        # (B, N, h, D) -> (N, D, h, B)  # noqa: E731
    tt = lambda x: x.view(f8).permute(3, 1, 2, 0)       # (B, N, h, 256) -> (256, N, h, B): MN-major B of the out MMAs  # noqa: E731
    dq = torch.empty(B, N, H, D, device=dev, dtype=torch.bfloat16)
    dk = torch.empty(B, N, Hk, D, device=dev, dtype=torch.bfloat16)
    dv = torch.empty(B, N, Hk, D, device=dev, dtype=torch.bfloat16)
    big = dbg & 2
    L = B * H
    ident = torch.zeros(L, N, 4, device=dev, dtype=torch.float64) if dbg & 1 else torch.zeros(1, 1, 4, device=dev, dtype=torch.float64)
    dumps = {}
    lds_tail = [2, 2, 2, 2, 2, 2, 3, 2, 3, 2, 2]
    n_t = N // 128
    for mode in modes:
        if mode == "dv":
            X, Xb, Y, Yb, Tw, out = k8, k8, q8, q8, do8, dv
        elif mode == "dk":
            X, Xb, Y, Yb, Tw, out = k8, v8, q8, do8, q8, dk
        else:
            X, Xb, Y, Yb, Tw, out = q8, do8, k8, v8, k8, dq
        dump = torch.zeros(L, N, N, 4, device=dev) if big else torch.zeros(1, 1, 1, 4, device=dev)
        codes = torch.zeros(L, N, N, device=dev, dtype=torch.uint8) if big else torch.zeros(1, 1, 16, device=dev, dtype=torch.uint8)
        sb = torch.zeros(L, N, N // 32, device=dev, dtype=torch.int32) if big else torch.zeros(1, 1, 4, device=dev, dtype=torch.int32)
        ten = [p(X), p(Xb), p(Y), p(Yb), tt(Tw), qs, ks, vs, dos, lse, delta, out, ident, dump, codes, sb]
        lds = [1, 1, 1, 1, 0] + lds_tail
        kd = dbg & 3 if mode == "dq" else dbg & 2
        key = (mode, causal, kd) if not knob else (mode, causal, kd, knob)
        _run(key, lambda: DmHd256Mx(mode, causal, kd, knob), ten, lds, [n_t, H, Hk])
        if big:
            dumps[mode] = dict(dump=dump, codes=codes, sb=sb)
    if dbg:
        return dq, dk, dv, dict(ident=ident, **dumps)
    return dq, dk, dv


def delta_test(q8, qs, k8, ks, v8, vs, do8, dos, lse, causal):
    """TEST-ONLY matched delta over bitwise this backward's own P / dP (dq kernel, dbg bit 4): fp64 num / den per row,
    ascending keys.  Production delta = pass0_hd256.dm_delta_hd256 (checked against this in t_bwdmx)."""
    B, N, H, _ = q8.shape
    Hk = k8.shape[2]
    dev = q8.device
    f8 = torch.float8_e4m3fn
    p = lambda x: x.view(f8).permute(1, 3, 2, 0)  # noqa: E731
    nd = torch.zeros(B * H, N, 2, device=dev, dtype=torch.float64)
    z = torch.zeros(1, 1, 1, 4, device=dev)
    ten = [p(q8), p(do8), p(k8), p(v8), k8.view(f8).permute(3, 1, 2, 0), qs, ks, vs, dos, lse, lse,
           torch.zeros(1, 1, 1, D, device=dev, dtype=torch.bfloat16), nd,
           z, torch.zeros(1, 1, 16, device=dev, dtype=torch.uint8), torch.zeros(1, 1, 4, device=dev, dtype=torch.int32)]
    lds = [1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 3, 2, 3, 2, 2]
    _run(("delta", bool(causal)), lambda: DmHd256Mx("dq", bool(causal), 4), ten, lds, [N // 128, H, Hk])
    num, den = nd[..., 0], nd[..., 1]
    return torch.where(den > 0, num / den, torch.zeros_like(num)).float().view(B, H, N), nd
