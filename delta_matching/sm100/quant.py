"""sm100-mx producers: e4m3 codes with one power-of-2 (UE8M0-representable) scale per 32 tokens over all of head_dim.

quant_rows(x, prescale, mean)  x (B, N, H, D) bf16/fp32 (any strides, unit stride on D)
    y = (x - mean[b, h]) * prescale            (fp32; mean optional, (B, H, D) fp32)
    per (b, h, 32-token block):  amax = max |y| = m * 2^E (m in [1, 2), fp32 bits);
        scale = the smallest power of 2 with amax <= 448 * scale  = 2^(E-8) if m <= 1.75 else 2^(E-7)
        (amax == 0 -> scale = 1);  code = rn_e4m3(y / scale)  (exact multiply by 2^-e; never exceeds 448)
    returns codes (B, N, H, D) float8_e4m3fn contiguous, scales (B, H, N // 32) fp32 (exact powers of 2)

The scale exponent biased by 127 is the UE8M0 byte used by the tcgen05 block-scaled MMAs.
"""
import torch
import triton
import triton.language as tl

BLK = 32


@triton.jit
def _quant_rows_kernel(X, MEAN, Y, S, N, H, sxb, sxn, sxh, prescale,
                       HAS_MEAN: tl.constexpr, D: tl.constexpr, BLK: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    rows = pid_n * BLK + tl.arange(0, BLK)
    cols = tl.arange(0, D)
    x = tl.load(X + b * sxb + h * sxh + rows[:, None] * sxn + cols[None, :]).to(tl.float32)
    if HAS_MEAN:
        mu = tl.load(MEAN + pid_bh * D + cols)
        x = x - mu[None, :]
    x = x * prescale
    amax = tl.max(tl.max(tl.abs(x), axis=1), axis=0)
    ab = amax.to(tl.int32, bitcast=True)
    eb = ((ab >> 23) & 0xFF) - 8 + ((ab & 0x7FFFFF) > 0x600000).to(tl.int32)   # smallest pow2 s: amax <= 448 s
    eb = tl.minimum(tl.maximum(eb, 1), 254)
    eb = tl.where(amax > 0, eb, 127)
    scale = (eb << 23).to(tl.float32, bitcast=True)
    inv = ((254 - eb) << 23).to(tl.float32, bitcast=True)
    y = (x * inv).to(tl.float8e4nv)
    tl.store(Y + (b * N * H + rows[:, None] * H + h) * D + cols[None, :], y)
    tl.store(S + pid_bh * (N // BLK) + pid_n, scale)


def quant_rows(x: torch.Tensor, prescale: float = 1.0, mean: torch.Tensor | None = None):
    B, N, H, D = x.shape
    assert N % BLK == 0 and x.stride(-1) == 1
    y = torch.empty(B, N, H, D, device=x.device, dtype=torch.float8_e4m3fn)
    s = torch.empty(B, H, N // BLK, device=x.device, dtype=torch.float32)
    m = mean.contiguous() if mean is not None else s
    _quant_rows_kernel[(N // BLK, B * H)](x, m, y, s, N, H, x.stride(0), x.stride(1), x.stride(2), float(prescale),
                                          HAS_MEAN=mean is not None, D=D, BLK=BLK, num_warps=4)
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
@triton.jit
def _kmean_partial_kernel(K, P, N, Hk, skb, skn, skh, ROWS: tl.constexpr, D: tl.constexpr):
    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)
    b = pid_bh // Hk
    h = pid_bh % Hk
    cols = tl.arange(0, D)
    acc = tl.zeros([D], dtype=tl.float32)
    for r0 in range(0, ROWS, 32):
        rows = pid_s * ROWS + r0 + tl.arange(0, 32)
        x = tl.load(K + b * skb + h * skh + rows[:, None] * skn + cols[None, :]).to(tl.float32)
        acc += tl.sum(x, axis=0)
    tl.store(P + (pid_bh * tl.num_programs(1) + pid_s) * D + cols, acc)


def kmean_fast(k: torch.Tensor, rows: int = 512) -> torch.Tensor:
    """(B, N, Hk, D) -> (B, Hk, D) fp32 mean over tokens; deterministic (fixed-order partials, no atomics)."""
    B, N, Hk, D = k.shape
    rows = min(rows, N)
    while N % rows:
        rows //= 2
    S = N // rows
    part = torch.empty(B * Hk, S, D, device=k.device, dtype=torch.float32)
    _kmean_partial_kernel[(B * Hk, S)](k, part, N, Hk, k.stride(0), k.stride(1), k.stride(2), ROWS=rows, D=D, num_warps=4)
    return (part.sum(dim=1) / N).view(B, Hk, D)


@triton.jit
def _quant_tile(X, MEAN, Y, S, W, N, H, h, b, pid_n, sxb, sxn, sxh, prescale, HAS_MEAN: tl.constexpr,
                WANT_W: tl.constexpr, W_OFF: tl.constexpr, D: tl.constexpr):
    rows = pid_n * 128 + tl.arange(0, 128)
    cols = tl.arange(0, D)
    x = tl.load(X + b * sxb + h * sxh + rows[:, None] * sxn + cols[None, :]).to(tl.float32)
    if HAS_MEAN:
        mu = tl.load(MEAN + (b * H + h) * D + cols)
        x = x - mu[None, :]
    x = x * prescale
    x3 = tl.reshape(x, (4, 32, D))
    amax = tl.max(tl.max(tl.abs(x3), axis=2), axis=1)                    # [4]
    ab = amax.to(tl.int32, bitcast=True)
    eb = ((ab >> 23) & 0xFF) - 8 + ((ab & 0x7FFFFF) > 0x600000).to(tl.int32)
    eb = tl.minimum(tl.maximum(eb, 1), 254)
    eb = tl.where(amax > 0, eb, 127)
    scale = (eb << 23).to(tl.float32, bitcast=True)
    inv = ((254 - eb) << 23).to(tl.float32, bitcast=True)
    y = tl.reshape(x3 * inv[:, None, None], (128, D)).to(tl.float8e4nv)
    tl.store(Y + (b * N * H + rows[:, None] * H + h) * D + cols[None, :], y)
    tl.store(S + (b * H + h) * (N // 32) + pid_n * 4 + tl.arange(0, 4), scale)
    if WANT_W:
        word = tl.sum(eb << (tl.arange(0, 4) * 8), axis=0)
        tl.store(W + (b * H + h) * (N // 64) + pid_n * 2 + W_OFF, word)


@triton.jit
def _quant_qkv_kernel(Q, K, V, MEAN, Q8, K8, V8, QS, KS, VS, KVW, N, H, Hk, B,
                      sqb, sqn, sqh, skb, skn, skh, svb, svn, svh, sm, D: tl.constexpr):
    pid_n = tl.program_id(0)
    pid = tl.program_id(1)
    if pid < B * H:
        _quant_tile(Q, MEAN, Q8, QS, KVW, N, H, pid % H, pid // H, pid_n, sqb, sqn, sqh, sm,
                    HAS_MEAN=False, WANT_W=False, W_OFF=0, D=D)
    elif pid < B * H + B * Hk:
        p = pid - B * H
        _quant_tile(K, MEAN, K8, KS, KVW, N, Hk, p % Hk, p // Hk, pid_n, skb, skn, skh, 1.0,
                    HAS_MEAN=True, WANT_W=True, W_OFF=0, D=D)
    else:
        p = pid - B * H - B * Hk
        _quant_tile(V, MEAN, V8, VS, KVW, N, Hk, p % Hk, p // Hk, pid_n, svb, svn, svh, 1.0,
                    HAS_MEAN=False, WANT_W=True, W_OFF=1, D=D)


def quant_qkv(q, k, v, sm: float, mean: torch.Tensor | None = None):
    """Fused producer: returns (q8, qs, k8, ks, v8, vs, kvw, mean); kvw (B, Hk, N/64) int32 = forward kernel's
    packed K/V UE8M0 words ([2t] K bytes of 128-key tile t, [2t+1] V bytes).  N % 128 == 0."""
    B, N, H, D = q.shape
    Hk = k.shape[2]
    assert N % 128 == 0 and q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1
    if mean is None:
        mean = kmean_fast(k)
    dev = q.device
    e4 = torch.float8_e4m3fn
    q8 = torch.empty(B, N, H, D, device=dev, dtype=e4)
    k8 = torch.empty(B, N, Hk, D, device=dev, dtype=e4)
    v8 = torch.empty(B, N, Hk, D, device=dev, dtype=e4)
    qs = torch.empty(B, H, N // 32, device=dev, dtype=torch.float32)
    ks = torch.empty(B, Hk, N // 32, device=dev, dtype=torch.float32)
    vs = torch.empty(B, Hk, N // 32, device=dev, dtype=torch.float32)
    kvw = torch.empty(B, Hk, N // 64, device=dev, dtype=torch.int32)
    _quant_qkv_kernel[(N // 128, B * (H + 2 * Hk))](
        q, k, v, mean.contiguous(), q8, k8, v8, qs, ks, vs, kvw, N, H, Hk, B,
        q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2), float(sm), D=D, num_warps=8)
    return q8, qs, k8, ks, v8, vs, kvw, mean


@triton.jit
def _qtile(X, MEAN, Y, S, WB, N, H, h, b, pid_n, sxb, sxn, sxh, prescale, HAS_MEAN: tl.constexpr,
           WANT_W: tl.constexpr, W_OFF: tl.constexpr, TR: tl.constexpr, D: tl.constexpr):
    """TR-row tile (TR % 32 == 0): codes, per-32 scales, and (WANT_W) the UE8M0 bytes into the packed word buffer
    WB viewed as bytes: byte (2*(row//128) + W_OFF)*4 + (row//32)%4 of head (b, h)."""
    NB: tl.constexpr = TR // 32
    rows = pid_n * TR + tl.arange(0, TR)
    cols = tl.arange(0, D)
    x = tl.load(X + b * sxb + h * sxh + rows[:, None] * sxn + cols[None, :]).to(tl.float32)
    if HAS_MEAN:
        mu = tl.load(MEAN + (b * H + h) * D + cols)
        x = x - mu[None, :]
    x = x * prescale
    x3 = tl.reshape(x, (NB, 32, D))
    amax = tl.max(tl.max(tl.abs(x3), axis=2), axis=1)
    ab = amax.to(tl.int32, bitcast=True)
    eb = ((ab >> 23) & 0xFF) - 8 + ((ab & 0x7FFFFF) > 0x600000).to(tl.int32)
    eb = tl.minimum(tl.maximum(eb, 1), 254)
    eb = tl.where(amax > 0, eb, 127)
    scale = (eb << 23).to(tl.float32, bitcast=True)
    inv = ((254 - eb) << 23).to(tl.float32, bitcast=True)
    y = tl.reshape(x3 * inv[:, None, None], (TR, D)).to(tl.float8e4nv)
    tl.store(Y + (b * N * H + rows[:, None] * H + h) * D + cols[None, :], y)
    blk = pid_n * NB + tl.arange(0, NB)
    tl.store(S + (b * H + h) * (N // 32) + blk, scale)
    if WANT_W:
        tl.store(WB + ((b * H + h) * (N // 64) + 2 * (blk // 4) + W_OFF) * 4 + blk % 4, eb.to(tl.uint8))


@triton.jit
def _prod_a_kernel(Q, K, V, Q8, V8, QS, VS, KVWB, PART, N, H, Hk, B,
                   sqb, sqn, sqh, skb, skn, skh, svb, svn, svh, sm, ROWS: tl.constexpr, TR: tl.constexpr,
                   D: tl.constexpr):
    """1-D grid: [K-mean partials (ROWS rows each) | Q tiles | V tiles]  (partials first: longest programs start first)."""
    NT = N // TR
    nq = B * H * NT
    nv = B * Hk * NT
    npart = B * Hk * (N // ROWS)
    pid = tl.program_id(0) - npart
    if pid < 0:
        pid = pid + nq + nv + npart
    if pid < nq:
        bh = pid // NT
        _qtile(Q, Q, Q8, QS, KVWB, N, H, bh % H, bh // H, pid % NT, sqb, sqn, sqh, sm,
               HAS_MEAN=False, WANT_W=False, W_OFF=0, TR=TR, D=D)
    elif pid < nq + nv:
        p = pid - nq
        bh = p // NT
        _qtile(V, V, V8, VS, KVWB, N, Hk, bh % Hk, bh // Hk, p % NT, svb, svn, svh, 1.0,
               HAS_MEAN=False, WANT_W=True, W_OFF=1, TR=TR, D=D)
    else:
        p = pid - nq - nv
        S = N // ROWS
        bh = p // S
        s = p % S
        b = bh // Hk
        h = bh % Hk
        cols = tl.arange(0, D)
        acc = tl.zeros([D], dtype=tl.float32)
        for r0 in range(0, ROWS, 64):
            rows = s * ROWS + r0 + tl.arange(0, 64)
            x = tl.load(K + b * skb + h * skh + rows[:, None] * skn + cols[None, :]).to(tl.float32)
            acc += tl.sum(x, axis=0)
        tl.store(PART + (bh * S + s) * D + cols, acc)


@triton.jit
def _prod_b_kernel(K, K8, KS, KVWB, PART, MEAN, N, Hk, skb, skn, skh, ROWS: tl.constexpr, TR: tl.constexpr,
                   D: tl.constexpr):
    """grid (N/TR, B*Hk): mean = (fixed-order sum of the partials) / N, then quantize the K tile."""
    pid_n = tl.program_id(0)
    bh = tl.program_id(1)
    S = N // ROWS
    cols = tl.arange(0, D)
    acc = tl.zeros([D], dtype=tl.float32)
    for s in range(0, S):
        acc += tl.load(PART + (bh * S + s) * D + cols)
    mu = acc / N
    if pid_n == 0:
        tl.store(MEAN + bh * D + cols, mu)
    rows = pid_n * TR + tl.arange(0, TR)
    b = bh // Hk
    h = bh % Hk
    NB: tl.constexpr = TR // 32
    x = tl.load(K + b * skb + h * skh + rows[:, None] * skn + cols[None, :]).to(tl.float32) - mu[None, :]
    x3 = tl.reshape(x, (NB, 32, D))
    amax = tl.max(tl.max(tl.abs(x3), axis=2), axis=1)
    ab = amax.to(tl.int32, bitcast=True)
    eb = ((ab >> 23) & 0xFF) - 8 + ((ab & 0x7FFFFF) > 0x600000).to(tl.int32)
    eb = tl.minimum(tl.maximum(eb, 1), 254)
    eb = tl.where(amax > 0, eb, 127)
    scale = (eb << 23).to(tl.float32, bitcast=True)
    inv = ((254 - eb) << 23).to(tl.float32, bitcast=True)
    y = tl.reshape(x3 * inv[:, None, None], (TR, D)).to(tl.float8e4nv)
    tl.store(K8 + (b * N * Hk + rows[:, None] * Hk + h) * D + cols[None, :], y)
    blk = pid_n * NB + tl.arange(0, NB)
    tl.store(KS + bh * (N // 32) + blk, scale)
    tl.store(KVWB + (bh * (N // 64) + 2 * (blk // 4)) * 4 + blk % 4, eb.to(tl.uint8))


def _tl(jf):
    from .fastpath import TLaunch
    return TLaunch(jf)


def quant_qkv2(q, k, v, sm: float, rows: int = 1024, warps_a: int = 2, warps_b: int = 2, tr: int = 32):
    """Two-launch fused producer.  Returns (q8, qs, k8, ks, v8, vs, kvw, mean); kvw (B, Hk, N/64) int32 packed UE8M0
    words ([2t] K bytes of 128-key tile t, [2t+1] V bytes).  The K mean is (sequential sum of ROWS-row fp32 partial
    sums, each a sum of 64-row chunk sums) / N.  Bitwise = quant_rows(...) with that mean."""
    B, N, H, D = q.shape
    Hk = k.shape[2]
    assert N % 128 == 0 and q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1
    ROWS = rows
    while N % ROWS:
        ROWS //= 2
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
    NT = N // tr
    kvwb = kvw.view(torch.uint8)
    _PROD_A((B * H * NT + B * Hk * NT + B * Hk * S,),
        q, k, v, q8, v8, qs, vs, kvwb, part, N, H, Hk, B,
        q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2), float(sm), ROWS=ROWS, TR=tr, D=D, num_warps=warps_a)
    _PROD_B((NT, B * Hk), k, k8, ks, kvwb, part, mean, N, Hk, k.stride(0), k.stride(1), k.stride(2),
                                 ROWS=ROWS, TR=tr, D=D, num_warps=warps_b)
    return q8, qs, k8, ks, v8, vs, kvw, mean


@triton.jit
def _quant_do_delta_kernel(X, O, Y, S, DL, N, H, sxb, sxn, sxh, sob, son, soh,
                           WANT_DELTA: tl.constexpr, D: tl.constexpr, BLK: tl.constexpr, DEQ_DELTA: tl.constexpr = False):
    """dO codes + pow2 scale per 32 tokens (as quant_rows) and, optionally, the stale delta rowsum(dO_bf16 * O_bf16)."""
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    rows = pid_n * BLK + tl.arange(0, BLK)
    cols = tl.arange(0, D)
    x = tl.load(X + b * sxb + h * sxh + rows[:, None] * sxn + cols[None, :]).to(tl.float32)
    if WANT_DELTA:
        o = tl.load(O + b * sob + h * soh + rows[:, None] * son + cols[None, :]).to(tl.float32)
        if not DEQ_DELTA:
            tl.store(DL + pid_bh * N + rows, tl.sum(x * o, axis=1))
    amax = tl.max(tl.max(tl.abs(x), axis=1), axis=0)
    ab = amax.to(tl.int32, bitcast=True)
    eb = ((ab >> 23) & 0xFF) - 8 + ((ab & 0x7FFFFF) > 0x600000).to(tl.int32)
    eb = tl.minimum(tl.maximum(eb, 1), 254)
    eb = tl.where(amax > 0, eb, 127)
    scale = (eb << 23).to(tl.float32, bitcast=True)
    inv = ((254 - eb) << 23).to(tl.float32, bitcast=True)
    y = (x * inv).to(tl.float8e4nv)
    tl.store(Y + (b * N * H + rows[:, None] * H + h) * D + cols[None, :], y)
    tl.store(S + pid_bh * (N // BLK) + pid_n, scale)
    if WANT_DELTA:
        if DEQ_DELTA:
            # matched delta from the forward's exact-P output: dequant(dO8) . Obar (fp32), as the fp8 dP MMA sees dO
            tl.store(DL + pid_bh * N + rows, tl.sum((y.to(tl.float32) * scale) * o, axis=1))


_PROD_A, _PROD_B, _QDD = _tl(_prod_a_kernel), _tl(_prod_b_kernel), _tl(_quant_do_delta_kernel)


def quant_do_delta(do: torch.Tensor, o: torch.Tensor | None, obar: torch.Tensor | None = None):
    """obar (B, N, H, D) fp32 given -> delta = dequant(dO8) . obar (pass-0-free matched delta); else as below."""
    if obar is not None:
        B, N, H, D = do.shape
        y = torch.empty(B, N, H, D, device=do.device, dtype=torch.float8_e4m3fn)
        s = torch.empty(B, H, N // BLK, device=do.device, dtype=torch.float32)
        dl = torch.empty(B, H, N, device=do.device, dtype=torch.float32)
        _QDD((N // BLK, B * H), do, obar, y, s, dl, N, H, do.stride(0), do.stride(1), do.stride(2),
                                                  obar.stride(0), obar.stride(1), obar.stride(2),
                                                  WANT_DELTA=True, DEQ_DELTA=True, D=D, BLK=BLK, num_warps=2)
        return y, s, dl
    return _quant_do_delta_stale(do, o)


def _quant_do_delta_stale(do: torch.Tensor, o: torch.Tensor | None):
    """(dO codes, scales, delta or None) in one pass; delta = rowsum(dO * O) fp32 (B, H, N) when o is given."""
    B, N, H, D = do.shape
    assert do.stride(-1) == 1 and (o is None or o.stride(-1) == 1)
    y = torch.empty(B, N, H, D, device=do.device, dtype=torch.float8_e4m3fn)
    s = torch.empty(B, H, N // BLK, device=do.device, dtype=torch.float32)
    dl = torch.empty(B, H, N, device=do.device, dtype=torch.float32) if o is not None else s
    oo = o if o is not None else do
    _QDD((N // BLK, B * H), do, oo, y, s, dl, N, H, do.stride(0), do.stride(1), do.stride(2),
                                              oo.stride(0), oo.stride(1), oo.stride(2),
                                              WANT_DELTA=o is not None, D=D, BLK=BLK, num_warps=2)
    return y, s, (dl if o is not None else None)
