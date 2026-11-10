"""sm100-mx producers: e4m3 codes with one power-of-2 (UE8M0-representable) scale per 32 tokens over all of head_dim.

quant_rows(x, prescale, mean)  x (B, N, H, D) bf16/fp32 (any strides, unit stride on D)
    y = (x - mean[b, h]) * prescale            (fp32; mean optional, (B, H, D) fp32)
    per (b, h, 32-token block):  amax = max |y| = m * 2^E (m in [1, 2), fp32 bits);
        scale = the smallest power of 2 with amax <= 448 * scale  = 2^(E-8) if m <= 1.75 else 2^(E-7)
        (amax == 0 -> scale = 1);  code = rn_e4m3(y / scale)  (exact multiply by 2^-e; never exceeds 448)
    returns codes (B, N, H, D) float8_e4m3fn contiguous, scales (B, H, N // 32) fp32 (exact powers of 2)

The scale exponent biased by 127 is the UE8M0 byte used by the tcgen05 block-scaled MMAs.

kda-sm100-cutequant: every producer is a CuTe DSL kernel (quant_cute.py, compiled with --enable-tvm-ffi), bitwise equal to
the former Triton kernels incl. their fp32 reduction trees (kda/docs/cutequant-design.md).  D (128 or 256) is taken from
the tensors and is a compile-time parameter.  No Triton.
"""
import threading

import torch

from . import quant_cute as qc

BLK = 32
_U8 = torch.uint8
_LOCK = threading.Lock()
_FNS = {}
_CACHE = []


def _cache():
    """Persistent compile cache under the package fingerprint (FA4 + delta_matching/sm100/*.py sources + compile_env)."""
    if not _CACHE:
        from flash_attn.cute import cache_utils
        from .fwd import _fingerprint
        if cache_utils.CUTE_DSL_CACHE_ENABLED:
            _CACHE.append(cache_utils.JITPersistentCache(cache_utils.get_cache_path() / "dm_sm100" / _fingerprint("quant") / "quant"))
        else:
            _CACHE.append(cache_utils.JITCache())
    return _CACHE[0]


def _fake(dtype, rank):
    import cutlass.cute as cute
    return cute.runtime.make_fake_tensor(dtype, tuple(cute.sym_int() for _ in range(rank)),
                                         stride=tuple(cute.sym_int64() for _ in range(rank)), assumed_align=1)


def _compile(key, obj, sig):
    """sig: list of ('t', cutlass dtype, rank) / ('i',) / ('f',) in call order (stream appended)."""
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32, Float32
    args = []
    for a in sig:
        if a[0] == "t":
            args.append(_fake(a[1], a[2]))
        elif a[0] == "i":
            args.append(Int32(0))
        else:
            args.append(Float32(0.0))
    args.append(cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True))
    from . import fastpath
    fastpath.EPOCH[0] += 1
    return cute.compile(obj, *args, options="--enable-tvm-ffi")


def _get(key, make, sig):
    f = _FNS.get(key)
    if f is None:
        with _LOCK:
            f = _FNS.get(key)
            if f is None:
                c = _cache()
                f = c.get(key) if hasattr(c, "get") else None
                if f is None:
                    f = _compile(key, make(), sig)
                    c[key] = f
                _FNS[key] = f
    return f


def _vec(t, ndim=3):
    """16-byte vector loads are legal: base and every outer stride (bytes) multiples of 16."""
    es = t.element_size()
    return t.data_ptr() % 16 == 0 and all((s * es) % 16 == 0 for s in t.stride()[:ndim])


def _tri_aligned(t, nheads):
    """Triton's specialization picks the 16-byte K layout (cutequant-design R1): pointer % 16, the batch / token strides
    % 16 (elements), the head stride % 16 unless nheads == 1 (specialized away)."""
    sb, sn, sh = t.stride()[:3]
    return t.data_ptr() % 16 == 0 and sb % 16 == 0 and sn % 16 == 0 and (nheads == 1 or sh % 16 == 0)


def _u8(t):
    return t.view(_U8)


def _dt(t):
    import cutlass
    return {torch.bfloat16: cutlass.BFloat16, torch.float32: cutlass.Float32, torch.uint8: cutlass.Uint8,
            torch.int32: cutlass.Int32}[t.dtype]



def _tile_fn(D, x, o, has_mean, has_pre, w_off, delta, mean_t, wb_t):
    import cutlass
    vx = _vec(x)
    vo = _vec(o) if delta is not None else False
    xf32 = x.dtype == torch.float32
    key = ("tile", D, vx, has_mean, has_pre, w_off, delta, vo, xf32, str(x.dtype), str(o.dtype) if o is not None else None)
    oo = o if o is not None else x
    sig = [("t", _dt(x), 4), ("t", _dt(oo), 4), ("t", cutlass.Float32, 3), ("t", cutlass.Uint8, 4),
           ("t", cutlass.Float32, 3), ("t", cutlass.Float32, 3), ("t", cutlass.Uint8, 1)] + [("i",)] * 9 + [("f",)]
    return _get(key, lambda: qc.Tile(D, vx, has_mean, has_pre, w_off, delta, vo, xf32), sig)


def _tile(x, prescale=None, mean=None, o=None, delta=None, y=None, s=None, dl=None, wb=None, w_off=None):
    """Generic 32-row tile launch: codes y (uint8 view), scales s, optional delta dl, optional UE8M0 bytes wb."""
    B, N, H, D = x.shape
    f = _tile_fn(D, x, o, mean is not None, prescale is not None, w_off, delta, mean, wb)
    oo = o if o is not None else x
    mm = mean if mean is not None else s
    dd = dl if dl is not None else s
    ww = wb if wb is not None else _u8(y).view(-1)
    f(x, oo, mm, _u8(y), s, dd, ww, N, H, B, x.stride(0), x.stride(1), x.stride(2), oo.stride(0), oo.stride(1),
      oo.stride(2), float(prescale) if prescale is not None else 1.0)


def quant_rows(x: torch.Tensor, prescale: float = 1.0, mean: torch.Tensor | None = None):
    B, N, H, D = x.shape
    assert N % BLK == 0 and x.stride(-1) == 1
    y = torch.empty(B, N, H, D, device=x.device, dtype=torch.float8_e4m3fn)
    s = torch.empty(B, H, N // BLK, device=x.device, dtype=torch.float32)
    _tile(x, prescale=prescale, mean=mean.contiguous() if mean is not None else None, y=y, s=s)
    return y, s


def kmean(k: torch.Tensor) -> torch.Tensor:
    """(B, N, Hk, D) -> (B, Hk, D) fp32 mean over tokens (K centering)."""
    return k.float().mean(dim=1)


def quant_rows_ref(x: torch.Tensor, prescale: float = 1.0, mean: torch.Tensor | None = None):
    """Pure-torch reference of quant_rows (bitwise target)."""
    B, N, H, D = x.shape
    y = x.float()
    if mean is not None:
        y = y - mean[:, None, :, :]
    y = y * prescale
    yb = y.view(B, N // BLK, BLK, H, D)
    amax = yb.abs().amax(dim=(2, 4))                              # (B, N/32, H)
    ab = amax.contiguous().view(torch.int32)
    eb = ((ab >> 23) & 0xFF) - 8 + ((ab & 0x7FFFFF) > 0x600000).to(torch.int32)
    eb = eb.clamp(min=1, max=254)
    eb = torch.where(amax > 0, eb, torch.full_like(eb, 127))
    scale = (eb << 23).view(torch.float32)
    inv = ((254 - eb) << 23).view(torch.float32)
    codes = (yb * inv[:, :, None, :, None]).to(torch.float8_e4m3fn).view(B, N, H, D)
    return codes, scale.permute(0, 2, 1).contiguous()


# ----------------------------------------------------------------------------------------------------------------------
# Fused producers (kda-sm100-fwd): one launch for Q, K, V (+ packed UE8M0 K/V words for the forward kernel), and a
# deterministic two-stage K mean.  Bitwise equal to quant_rows(...) for the same mean.
# ----------------------------------------------------------------------------------------------------------------------
def kmean_fast(k: torch.Tensor, rows: int = 512) -> torch.Tensor:
    """(B, N, Hk, D) -> (B, Hk, D) fp32 mean over tokens; deterministic (fixed-order partials, no atomics)."""
    import cutlass
    B, N, Hk, D = k.shape
    rows = min(rows, N)
    while N % rows:
        rows //= 2
    assert rows % 32 == 0
    S = N // rows
    part = torch.empty(B * Hk, S, D, device=k.device, dtype=torch.float32)
    vk, kal = _vec(k), _tri_aligned(k, Hk)
    f = _get(("kmeanp", D, vk, kal), lambda: qc.KMeanPartial(D, vk, kal),
             [("t", cutlass.BFloat16, 4), ("t", cutlass.Float32, 3)] + [("i",)] * 7)
    f(k, part, N, Hk, B, k.stride(0), k.stride(1), k.stride(2), rows)
    return (part.sum(dim=1) / N).view(B, Hk, D)


def quant_qkv(q, k, v, sm: float, mean: torch.Tensor | None = None):
    """Fused producer: returns (q8, qs, k8, ks, v8, vs, kvw, mean); kvw (B, Hk, N/64) int32 = forward kernel's
    packed K/V UE8M0 words ([2t] K bytes of 128-key tile t, [2t+1] V bytes).  N % 128 == 0."""
    B, N, H, D = q.shape
    Hk = k.shape[2]
    assert N % 128 == 0 and q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1
    if mean is None:
        mean = kmean_fast(k)
    mean = mean.contiguous()
    dev = q.device
    e4 = torch.float8_e4m3fn
    q8 = torch.empty(B, N, H, D, device=dev, dtype=e4)
    k8 = torch.empty(B, N, Hk, D, device=dev, dtype=e4)
    v8 = torch.empty(B, N, Hk, D, device=dev, dtype=e4)
    qs = torch.empty(B, H, N // 32, device=dev, dtype=torch.float32)
    ks = torch.empty(B, Hk, N // 32, device=dev, dtype=torch.float32)
    vs = torch.empty(B, Hk, N // 32, device=dev, dtype=torch.float32)
    kvw = torch.empty(B, Hk, N // 64, device=dev, dtype=torch.int32)
    wb = kvw.view(_U8).view(-1)
    _tile(q, prescale=sm, y=q8, s=qs)
    _tile(k, mean=mean, y=k8, s=ks, wb=wb, w_off=0)
    _tile(v, y=v8, s=vs, wb=wb, w_off=1)
    return q8, qs, k8, ks, v8, vs, kvw, mean


def _prod_fns(q, k, v):
    import cutlass
    D = q.shape[-1]
    Hk = k.shape[2]
    vq, vk, vv, kal = _vec(q), _vec(k), _vec(v), _tri_aligned(k, Hk)
    fa = _get(("prod_a", D, vq, vk, vv, kal), lambda: qc.ProdA(D, vq, vk, vv, kal),
              [("t", cutlass.BFloat16, 4)] * 3 + [("t", cutlass.Uint8, 4)] * 2 + [("t", cutlass.Float32, 3)] * 2
              + [("t", cutlass.Uint8, 1), ("t", cutlass.Float32, 3)] + [("i",)] * 13 + [("f",), ("i",)])
    fb = _get(("prod_b", D, vk), lambda: qc.ProdB(D, vk),
              [("t", cutlass.BFloat16, 4), ("t", cutlass.Uint8, 4), ("t", cutlass.Float32, 3), ("t", cutlass.Uint8, 1),
               ("t", cutlass.Float32, 3), ("t", cutlass.Float32, 3)] + [("i",)] * 7)
    return fa, fb


def quant_qkv2(q, k, v, sm: float, rows: int = 1024, warps_a: int = 2, warps_b: int = 2, tr: int = 32):
    """Two-launch fused producer.  Returns (q8, qs, k8, ks, v8, vs, kvw, mean); kvw (B, Hk, N/64) int32 packed UE8M0
    words ([2t] K bytes of 128-key tile t, [2t+1] V bytes).  The K mean is (sequential sum of ROWS-row fp32 partial
    sums, each a sum of 64-row chunk sums) / N.  Bitwise = quant_rows(...) with that mean."""
    B, N, H, D = q.shape
    Hk = k.shape[2]
    assert N % 128 == 0 and q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1
    assert (warps_a, warps_b, tr) == (2, 2, 32), "cutequant: only the shipped producer layout (2, 2, 32) is ported"
    ROWS = rows
    while N % ROWS:
        ROWS //= 2
    assert ROWS % 64 == 0
    S = N // ROWS
    dev = q.device
    e4 = torch.float8_e4m3fn
    q8 = torch.empty(B, N, H, D, device=dev, dtype=e4)
    k8 = torch.empty(B, N, Hk, D, device=dev, dtype=e4)
    v8 = torch.empty(B, N, Hk, D, device=dev, dtype=e4)
    qs = torch.empty(B, H, N // 32, device=dev, dtype=torch.float32)
    ks = torch.empty(B, Hk, N // 32, device=dev, dtype=torch.float32)
    vs = torch.empty(B, Hk, N // 32, device=dev, dtype=torch.float32)
    kvw = torch.empty(B, Hk, N // 64, device=dev, dtype=torch.int32)
    part = torch.empty(B * Hk, S, D, device=dev, dtype=torch.float32)
    mean = torch.empty(B, Hk, D, device=dev, dtype=torch.float32)
    kvwb = kvw.view(_U8).view(-1)
    fa, fb = _prod_fns(q, k, v)
    fa(q, k, v, q8.view(_U8), v8.view(_U8), qs, vs, kvwb, part, N, H, Hk, B,
       q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2),
       v.stride(0), v.stride(1), v.stride(2), float(sm), ROWS)
    fb(k, k8.view(_U8), ks, kvwb, part, mean, N, Hk, B, k.stride(0), k.stride(1), k.stride(2), ROWS)
    return q8, qs, k8, ks, v8, vs, kvw, mean


def quant_do_delta(do: torch.Tensor, o: torch.Tensor | None, obar: torch.Tensor | None = None):
    """obar (B, N, H, D) fp32 given -> delta = dequant(dO8) . obar (pass-0-free matched delta); else as below."""
    if obar is not None:
        B, N, H, D = do.shape
        assert do.stride(-1) == 1 and obar.stride(-1) == 1 and obar.dtype == torch.float32
        y = torch.empty(B, N, H, D, device=do.device, dtype=torch.float8_e4m3fn)
        s = torch.empty(B, H, N // BLK, device=do.device, dtype=torch.float32)
        dl = torch.empty(B, H, N, device=do.device, dtype=torch.float32)
        _tile(do, o=obar, delta="deq", y=y, s=s, dl=dl)
        return y, s, dl
    return _quant_do_delta_stale(do, o)


def _quant_do_delta_stale(do: torch.Tensor, o: torch.Tensor | None):
    """(dO codes, scales, delta or None) in one pass; delta = rowsum(dO * O) fp32 (B, H, N) when o is given."""
    B, N, H, D = do.shape
    assert do.stride(-1) == 1 and (o is None or o.stride(-1) == 1)
    y = torch.empty(B, N, H, D, device=do.device, dtype=torch.float8_e4m3fn)
    s = torch.empty(B, H, N // BLK, device=do.device, dtype=torch.float32)
    dl = torch.empty(B, H, N, device=do.device, dtype=torch.float32) if o is not None else None
    _tile(do, o=o, delta="stale" if o is not None else None, y=y, s=s, dl=dl)
    return y, s, dl
