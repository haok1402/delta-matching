"""
Fused BACKWARD of the sm100 q/k/v producers (both arms: the DM fp8 producers of qkv_fwd_sm100.fwd_producers and the
BF16 arm's fwd_bf16 have the same differentiable math, norm + RoPE): RoPE backward + te.RMSNorm backward (TE 2.19
rmsnorm_bwd_general_kernel math AND row partition, so dx and dgamma are bitwise TE's) + the q|gate / k / v interleave
into the q/k/v GEMM's dY layout (no cat) + optionally amax and TE's e5m2 current-scaling scale of dY.
Port of the Hopper rebuttal kernel (dmlm/dm_qkv_bwd_cute.py), device code unchanged; sm100 contract:

    dY, dgq, dgk[, amax, scale] = bwd_producers(dq, dk, dv, dgate, y, gq, gk, cos, sin, H, G, eps=1e-6,
                                                rstd_q=None, rstd_k=None, want_amax=False, out=None)

  dq     (B, N, H, D) bf16  grad wrt the rotated, UNSCALED bf16 q (the sm100 attention backward's dq: the 2^-4
                            softmax scale is already applied inside it; the BF16 arm's core grad wrt q_rot)
  dk, dv (B, N, G, D) bf16  folded per kv head, token-major (sm100 KV2 / cuDNN GQA output), any strides with a unit
                            last stride
  dgate  (N, B, H, D) bf16  grad of the gate view [s, b, H, D] of y, any strides
  y      (B, N, C) or (B*N, C) bf16, C = (2H+2G)D: the saved GEMM output (q_h|gate_h per head, then k, then v)
  gq, gk (D,) norm weights; cos / sin (B, N, R) bf16; rstd_q (B, H, N) / rstd_k (B, G, N) fp32 (both or neither):
  the forward's rsigma (TE row order [b, h, s]); None = recomputed with TE's forward math.
  Returns dY (y's dims) bf16, dgq / dgk (D,) bf16, amax / scale fp32 (1,) device tensors.

Kernels: _bwd_kernel (grid = TE's ctas_per_col virtual CTAs, one per CTA at D = 256, one-row-ahead loads) + _fin_kernel
(dgamma finalize in TE's order, amax / scale).  TVM-FFI entry: per call only the addresses and the stream are passed.
"""
from __future__ import annotations

import os

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_ptr

F32 = cutlass.Float32
BF16 = cutlass.BFloat16
I32 = cutlass.Int32
_CACHE = {}
# TE 2.19 rmsnorm_bwd_general_kernel occupancy (CTAs/SM, 128 threads), bf16 in/out/weight, head_dim 256: 4 on H200
# (528 = 132*4 CTAs in the profiler grid); GB200 value = the one whose dgamma is bitwise TE's (gate t_bwd.py; override
# BWDPROD_TE_OCC).  ctas_per_col = min(ceil(rows / 4), SMs * occ).
TE_BWD_OCC = {256: int(os.environ.get("BWDPROD_TE_OCC", "4"))}


# ---------------------------------------------------------------- exact scalar fp32 ops (never contracted)
def _mul(a, b):
    return cute.arch.inline_ptx("mul.rn.f32 {$w0}, {$r0}, {$r1};", write_only_types=[F32], read_only_args=[a, b])


def _add(a, b):
    return cute.arch.inline_ptx("add.rn.f32 {$w0}, {$r0}, {$r1};", write_only_types=[F32], read_only_args=[a, b])


def _sub(a, b):
    return cute.arch.inline_ptx("sub.rn.f32 {$w0}, {$r0}, {$r1};", write_only_types=[F32], read_only_args=[a, b])


def _fma(a, b, c):
    return cute.arch.inline_ptx("fma.rn.f32 {$w0}, {$r0}, {$r1}, {$r2};", write_only_types=[F32],
                                read_only_args=[a, b, c])


def _hex(c):
    import struct
    return "0f%08X" % struct.unpack("<I", struct.pack("<f", float(c)))[0]


def _addc(a, c):
    """a + c, c a python float (fp32 immediate; inline-PTX operands must be registers)."""
    return cute.arch.inline_ptx("add.rn.f32 {$w0}, {$r0}, " + _hex(c) + ";", write_only_types=[F32], read_only_args=[a])


def _mulc(a, c):
    return cute.arch.inline_ptx("mul.rn.f32 {$w0}, {$r0}, " + _hex(c) + ";", write_only_types=[F32], read_only_args=[a])


def _fma0(a, b):
    """fma(a, b, +0.0): the first step of a `acc = 0; acc += a * b` chain."""
    return cute.arch.inline_ptx("fma.rn.f32 {$w0}, {$r0}, {$r1}, 0f00000000;", write_only_types=[F32],
                                read_only_args=[a, b])


def _neg(a):
    return cute.arch.inline_ptx("neg.f32 {$w0}, {$r0};", write_only_types=[F32], read_only_args=[a])


def _rsqrt(a):
    return cute.arch.inline_ptx("rsqrt.approx.f32 {$w0}, {$r0};", write_only_types=[F32], read_only_args=[a])


def _bits(a):
    return cute.arch.inline_ptx("mov.b32 {$w0}, {$r0};", write_only_types=[I32], read_only_args=[a])


def _e5m2_scale(a):
    """TE compute_scale_from_amax(amax, 57344, pow2 off, eps 0, FLT_MAX): amax in {0, inf, nan} -> 1;
    __fdiv_rn(57344, amax); inf -> FLT_MAX (branch-free: selp on the IEEE compares)."""
    return cute.arch.inline_ptx(
        "{ .reg .pred ok, big; .reg .f32 q;\n"
        " setp.gt.f32 ok, {$r0}, 0f00000000;\n"
        " setp.lt.and.f32 ok, {$r0}, 0f7F800000, ok;\n"
        " div.rn.f32 q, 0f47600000, {$r0};\n"
        " setp.eq.f32 big, q, 0f7F800000;\n"
        " selp.f32 q, 0f7F7FFFFF, q, big;\n"
        " selp.f32 {$w0}, q, 0f3F800000, ok; }",
        write_only_types=[F32], read_only_args=[a])


def _bfly_sum(v):
    """TE Reducer<.., WARPS_N=1>::allreduce_: data = data + shfl_xor(data, it), it = 1, 2, 4, 8, 16."""
    for o in (1, 2, 4, 8, 16):
        v = _add(v, cute.arch.shuffle_sync_bfly(v, offset=o))
    return v


@cute.jit
def _rc(r, H: cutlass.Constexpr, N: cutlass.Constexpr):
    """TE row r = (b*H + h)*N + s -> (b, h, s)."""
    b = r // (H * N)
    rem = r - b * (H * N)
    h = rem // N
    return b, h, rem - h * N


@cute.jit
def _load(r, lane, bx, bg, bt, bc, bs, br, DQ, YQ, DT, CS, SN, RS, H: cutlass.Constexpr, N: cutlass.Constexpr,
          RL: cutlass.Constexpr, HAS_RSTD: cutlass.Constexpr, GATED: cutlass.Constexpr, FOLD: cutlass.Constexpr):
    """Issue every global load of row r into the register buffers (x, grad, pass-through T = gate / v, cos, sin, rs)."""
    b, h, s = _rc(r, H, N)
    if cutlass.const_expr(GATED):
        bx.store(YQ[b, s, h, 0, lane, None].load())
    else:
        bx.store(YQ[b, s, h, lane, None].load())
    if cutlass.const_expr(FOLD > 1):    # the FOLD query-head rows of kv head h (unfolded dk / dv)
        for i in cutlass.range_constexpr(FOLD):
            bg[i, None].store(DQ[b, s, h, i, lane, None].load())
            bt[i, None].store(DT[b, s, h, i, lane, None].load())
    else:
        bg.store(DQ[b, s, h, lane, None].load())
        bt.store(DT[b, s, h, lane, None].load())
    if lane < RL:
        bc.store(CS[b, s, lane, None].load())
        bs.store(SN[b, s, lane, None].load())
    if cutlass.const_expr(HAS_RSTD):
        br[0] = RS[b, h, s]


@cute.jit
def _fold(bf, NE: cutlass.Constexpr, FOLD: cutlass.Constexpr):
    """bf16 (FOLD, NE) -> fp32 (NE,) TensorSSA: per lane element ((0 + x0) + x1) + ... in fp32 RN (torch's sum over the
    n_rep axis: vt0 = 4 accumulators each 0 + x_i, combined in order -> the same value)."""
    xf = cute.make_rmem_tensor((FOLD, NE), F32)
    xf.store(bf.load().to(F32))
    o = cute.make_rmem_tensor((NE,), F32)
    for j in cutlass.range_constexpr(NE):
        a = _addc(xf[0, j], 0.0)
        for i in cutlass.range_constexpr(1, FOLD):
            a = _add(a, xf[i, j])
        o[j] = a
    return o.load()


@cute.jit
def _compute(r, lane, bx, bg, bt, bc, bs, br, acc, am, gam, OQ, OT, H: cutlass.Constexpr, N: cutlass.Constexpr,
             D: cutlass.Constexpr, NE: cutlass.Constexpr, RL: cutlass.Constexpr, HL: cutlass.Constexpr,
             eps: cutlass.Constexpr, HAS_RSTD: cutlass.Constexpr, GATED: cutlass.Constexpr,
             WANT_AMAX: cutlass.Constexpr, FOLD: cutlass.Constexpr):
    """RoPE bwd + RMSNorm bwd of row r from the buffers -> OQ (dx) and the pass-through copy -> OT.  FOLD > 1: the
    grad and the pass-through are the GQA fold of FOLD query-head rows (torch's bf16 sum: fp32 (((0+x0)+x1)+..),
    bf16_rn)."""
    b, h, s = _rc(r, H, N)
    rn = 1.0 / D
    g = cute.make_rmem_tensor((NE,), F32)
    x = cute.make_rmem_tensor((NE,), F32)
    gc = cute.make_rmem_tensor((NE,), F32)
    gs = cute.make_rmem_tensor((NE,), F32)
    x.store(bx.load().to(F32))
    tv = bt.load() if cutlass.const_expr(FOLD == 1) else _fold(bt, NE, FOLD).to(BF16)
    if cutlass.const_expr(FOLD > 1):
        g.store(_fold(bg, NE, FOLD).to(BF16).to(F32))
    else:
        g.store(bg.load().to(F32))
    for j in cutlass.range_constexpr(NE):
        gc[j] = F32(0.0)
        gs[j] = F32(0.0)
    if lane < RL:
        c = bc.load().to(F32)
        sn = bs.load().to(F32)
        for j in cutlass.range_constexpr(NE):
            gc[j] = _mul(g[j], c[j])
            gs[j] = _mul(g[j], sn[j])
    gp = cute.make_rmem_tensor((NE,), F32)
    for j in cutlass.range_constexpr(NE):
        gp[j] = cute.arch.shuffle_sync_bfly(gs[j], offset=HL)
    d = cute.make_rmem_tensor((NE,), F32)
    if lane < HL:
        for j in cutlass.range_constexpr(NE):
            d[j] = _add(_addc(gp[j], 0.0), gc[j])          # (0 + gs) == (gs + 0) in IEEE
        g.store(d.load().to(BF16).to(F32))
    elif lane < RL:
        for j in cutlass.range_constexpr(NE):
            d[j] = _add(_addc(_neg(gp[j]), 0.0), gc[j])
        g.store(d.load().to(BF16).to(F32))
    # g now holds dz (bf16 values in fp32)
    rs = F32(0.0)
    if cutlass.const_expr(HAS_RSTD):
        rs = br[0]
    else:
        ss = _fma0(x[0], x[0])
        for j in cutlass.range_constexpr(1, NE):
            ss = _fma(x[j], x[j], ss)
        ss = _bfly_sum(ss)
        rs = _rsqrt(_addc(_mulc(ss, rn), eps))
    yv = cute.make_rmem_tensor((NE,), F32)
    dy = cute.make_rmem_tensor((NE,), F32)
    for j in cutlass.range_constexpr(NE):
        yv[j] = _mul(rs, x[j])
        dy[j] = _mul(gam[j], g[j])
    m = _fma0(dy[0], yv[0])
    for j in cutlass.range_constexpr(1, NE):
        m = _fma(dy[j], yv[j], m)
    m = _mulc(_bfly_sum(m), rn)
    nm = _neg(m)
    for j in cutlass.range_constexpr(NE):
        d[j] = _mul(rs, _fma(nm, yv[j], dy[j]))
        acc[j] = _fma(g[j], yv[j], acc[j])
    dxb = d.load().to(BF16)
    if cutlass.const_expr(GATED):
        OQ[b, s, h, 0, lane, None].store(dxb)
        OQ[b, s, h, 1, lane, None].store(tv)
    else:
        OQ[b, s, h, lane, None].store(dxb)
        OT[b, s, h, lane, None].store(tv)
    if cutlass.const_expr(WANT_AMAX):
        a = cute.arch.fmax(cute.math.absf(dxb.to(F32)).reduce(cute.ReductionOp.MAX, F32(0.0), 0),
                           cute.math.absf(tv.to(F32)).reduce(cute.ReductionOp.MAX, F32(0.0), 0))
        am[0] = cute.arch.fmax(am[0], a)


@cute.jit
def _chain(c, w, lane, acc, am, gam, Gw, DQ, YQ, DT, OQ, OT, CS, SN, RS, P, smem, H: cutlass.Constexpr,
           B: cutlass.Constexpr, N: cutlass.Constexpr, D: cutlass.Constexpr, NE: cutlass.Constexpr,
           RL: cutlass.Constexpr, HL: cutlass.Constexpr, CC: cutlass.Constexpr, eps: cutlass.Constexpr,
           HAS_RSTD: cutlass.Constexpr, GATED: cutlass.Constexpr, WANT_AMAX: cutlass.Constexpr,
           FOLD: cutlass.Constexpr):
    """TE virtual CTA c of CC, warp w: rows c*4 + w + k*4*CC in order (the dgamma fma chain), loads one row ahead;
    then TE's in-CTA dgamma reduction ((w0 + w1) + w2) + w3 -> P[c]."""
    for j in cutlass.range_constexpr(NE):
        acc[j] = F32(0.0)
    gv = Gw[lane, None].load().to(F32)
    for j in cutlass.range_constexpr(NE):
        gam[j] = _addc(gv[j], 0.0)                         # TE: g = gamma + (zero_centered ? 1 : 0)
    GD = 4 * CC
    r0 = c * 4 + w
    n = (B * H * N - r0 + GD - 1) // GD
    ax = cute.make_rmem_tensor((NE,), BF16)
    ac = cute.make_rmem_tensor((NE,), BF16)
    asn = cute.make_rmem_tensor((NE,), BF16)
    ar = cute.make_rmem_tensor((1,), F32)
    if cutlass.const_expr(FOLD > 1):   # FOLD x 2 row loads per row already in flight: no prefetch (registers)
        ag = cute.make_rmem_tensor((FOLD, NE), BF16)
        at = cute.make_rmem_tensor((FOLD, NE), BF16)
        for k in cutlass.range(n):
            r = r0 + k * GD
            _load(r, lane, ax, ag, at, ac, asn, ar, DQ, YQ, DT, CS, SN, RS, H, N, RL, HAS_RSTD, GATED, FOLD)
            _compute(r, lane, ax, ag, at, ac, asn, ar, acc, am, gam, OQ, OT, H, N, D, NE, RL, HL, eps, HAS_RSTD,
                     GATED, WANT_AMAX, FOLD)
    else:
        ag = cute.make_rmem_tensor((NE,), BF16)
        at = cute.make_rmem_tensor((NE,), BF16)
        nx = cute.make_rmem_tensor((NE,), BF16)
        ng = cute.make_rmem_tensor((NE,), BF16)
        nt = cute.make_rmem_tensor((NE,), BF16)
        nc = cute.make_rmem_tensor((NE,), BF16)
        nsn = cute.make_rmem_tensor((NE,), BF16)
        nr = cute.make_rmem_tensor((1,), F32)
        if n > 0:
            _load(r0, lane, ax, ag, at, ac, asn, ar, DQ, YQ, DT, CS, SN, RS, H, N, RL, HAS_RSTD, GATED, 1)
        for k in cutlass.range(n):
            r = r0 + k * GD
            if k + 1 < n:
                _load(r + GD, lane, nx, ng, nt, nc, nsn, nr, DQ, YQ, DT, CS, SN, RS, H, N, RL, HAS_RSTD, GATED, 1)
            _compute(r, lane, ax, ag, at, ac, asn, ar, acc, am, gam, OQ, OT, H, N, D, NE, RL, HL, eps, HAS_RSTD,
                     GATED, WANT_AMAX, 1)
            ax.store(nx.load())
            ag.store(ng.load())
            at.store(nt.load())
            ac.store(nc.load())
            asn.store(nsn.load())
            ar[0] = nr[0]
    _partial(acc, w, lane, P, c, smem, NE)


@cute.jit
def _partial(acc, w, lane, P, c, smem, NE: cutlass.Constexpr):
    """TE's in-CTA dgamma reduction (WARPS_M = 4; warp 0 owns it = 0): ((w0 + w1) + w2) + w3 -> P[c]."""
    if w != 0:
        for j in cutlass.range_constexpr(NE):
            smem[w, lane, j] = acc[j]
    cute.arch.barrier()
    if w == 0:
        for k in cutlass.range_constexpr(1, 4):
            for j in cutlass.range_constexpr(NE):
                acc[j] = _add(acc[j], smem[k, lane, j])
        P[c, lane, None].store(acc.load())


@cute.jit
def _chain2(cA, cB, w, lane, accA, accB, am, gam, Gw, DQ, YQ, DT, OQ, OT, CS, SN, RS, P, smA, smB,
            H: cutlass.Constexpr, B: cutlass.Constexpr, N: cutlass.Constexpr, D: cutlass.Constexpr,
            NE: cutlass.Constexpr, RL: cutlass.Constexpr, HL: cutlass.Constexpr, CC: cutlass.Constexpr,
            eps: cutlass.Constexpr, HAS_RSTD: cutlass.Constexpr, GATED: cutlass.Constexpr,
            WANT_AMAX: cutlass.Constexpr, FOLD: cutlass.Constexpr):
    """Two TE virtual CTAs (cA < cB) per physical CTA: warp w runs chain (cA, w) and chain (cB, w) interleaved row by
    row (each chain keeps its own order and accumulator), both rows' loads issued before either compute: 2 rows in
    flight per warp with half the CTAs (one wave at D = 128, where 1584 CTAs of the 1-chain layout need 1.3 waves)."""
    for j in cutlass.range_constexpr(NE):
        accA[j] = F32(0.0)
        accB[j] = F32(0.0)
    gv = Gw[lane, None].load().to(F32)
    for j in cutlass.range_constexpr(NE):
        gam[j] = _addc(gv[j], 0.0)
    GD = 4 * CC
    rA = cA * 4 + w
    rB = cB * 4 + w
    nA = 0
    nB = 0
    if cA < CC:
        nA = (B * H * N - rA + GD - 1) // GD
    if cB < CC:
        nB = (B * H * N - rB + GD - 1) // GD      # rB > rA: nB <= nA
    xA = cute.make_rmem_tensor((NE,), BF16)
    cA_ = cute.make_rmem_tensor((NE,), BF16)
    sA = cute.make_rmem_tensor((NE,), BF16)
    rsA = cute.make_rmem_tensor((1,), F32)
    xB = cute.make_rmem_tensor((NE,), BF16)
    cB_ = cute.make_rmem_tensor((NE,), BF16)
    sB = cute.make_rmem_tensor((NE,), BF16)
    rsB = cute.make_rmem_tensor((1,), F32)
    shp = (FOLD, NE) if cutlass.const_expr(FOLD > 1) else (NE,)
    gA = cute.make_rmem_tensor(shp, BF16)
    tA = cute.make_rmem_tensor(shp, BF16)
    gB = cute.make_rmem_tensor(shp, BF16)
    tB = cute.make_rmem_tensor(shp, BF16)
    for k in cutlass.range(nA):
        r = rA + k * GD
        _load(r, lane, xA, gA, tA, cA_, sA, rsA, DQ, YQ, DT, CS, SN, RS, H, N, RL, HAS_RSTD, GATED, FOLD)
        if k < nB:
            _load(rB + k * GD, lane, xB, gB, tB, cB_, sB, rsB, DQ, YQ, DT, CS, SN, RS, H, N, RL, HAS_RSTD, GATED,
                  FOLD)
        _compute(r, lane, xA, gA, tA, cA_, sA, rsA, accA, am, gam, OQ, OT, H, N, D, NE, RL, HL, eps, HAS_RSTD,
                 GATED, WANT_AMAX, FOLD)
        if k < nB:
            _compute(rB + k * GD, lane, xB, gB, tB, cB_, sB, rsB, accB, am, gam, OQ, OT, H, N, D, NE, RL, HL, eps,
                     HAS_RSTD, GATED, WANT_AMAX, FOLD)
    if cA < CC:
        _partial(accA, w, lane, P, cA, smA, NE)
    if cB < CC:
        _partial(accB, w, lane, P, cB, smB, NE)


@cute.kernel
def _bwd_kernel(DQ, DG, YQ, OQ, DK, DV, YK, OK, OV, CS, SN, RSQ, RSK, GQ, GK, PQ, PK, AMW,
                B: cutlass.Constexpr, N: cutlass.Constexpr, H: cutlass.Constexpr, G: cutlass.Constexpr,
                D: cutlass.Constexpr, NE: cutlass.Constexpr, RL: cutlass.Constexpr, HL: cutlass.Constexpr,
                CQ: cutlass.Constexpr, CK: cutlass.Constexpr, eps: cutlass.Constexpr, HAS_RSTD: cutlass.Constexpr,
                WANT_AMAX: cutlass.Constexpr, FOLD: cutlass.Constexpr, CPW: cutlass.Constexpr, PG: cutlass.Constexpr):
    tid, _, _ = cute.arch.thread_idx()
    c, _, _ = cute.arch.block_idx()
    lane = tid % 32
    w = tid // 32
    am = cute.make_rmem_tensor((1,), F32)
    am[0] = F32(0.0)
    acc = cute.make_rmem_tensor((NE,), F32)
    gam = cute.make_rmem_tensor((NE,), F32)
    smq = cute.make_tensor(cute.arch.alloc_smem(F32, 4 * 32 * NE),
                           cute.make_layout((4, 32, NE), stride=(32 * NE, NE, 1)))
    smk = cute.make_tensor(cute.arch.alloc_smem(F32, 4 * 32 * NE),
                           cute.make_layout((4, 32, NE), stride=(32 * NE, NE, 1)))
    if cutlass.const_expr(CPW == 1):
        if c < CQ:   # q rows (TE row order [b, h, s]) + the gate copy of the same (token, head)
            _chain(c, w, lane, acc, am, gam, GQ, DQ, YQ, DG, OQ, OQ, CS, SN, RSQ, PQ, smq, H, B, N, D, NE, RL, HL,
                   CQ, eps, HAS_RSTD, True, WANT_AMAX, 1)
        if c < CK:   # k rows (TE row order [b, g, s]) + the v copy of the same (token, kv head)
            _chain(c, w, lane, acc, am, gam, GK, DK, YK, DV, OK, OV, CS, SN, RSK, PK, smk, G, B, N, D, NE, RL, HL,
                   CK, eps, HAS_RSTD, False, WANT_AMAX, FOLD)
    else:            # two virtual CTAs (c, c + PG) per physical CTA
        acc2 = cute.make_rmem_tensor((NE,), F32)
        smq2 = cute.make_tensor(cute.arch.alloc_smem(F32, 4 * 32 * NE),
                                cute.make_layout((4, 32, NE), stride=(32 * NE, NE, 1)))
        smk2 = cute.make_tensor(cute.arch.alloc_smem(F32, 4 * 32 * NE),
                                cute.make_layout((4, 32, NE), stride=(32 * NE, NE, 1)))
        _chain2(c, c + PG, w, lane, acc, acc2, am, gam, GQ, DQ, YQ, DG, OQ, OQ, CS, SN, RSQ, PQ, smq, smq2, H, B, N,
                D, NE, RL, HL, CQ, eps, HAS_RSTD, True, WANT_AMAX, 1)
        if cutlass.const_expr(FOLD > 1):   # fold rows already carry 2*FOLD loads: the two k chains one after other
            if c < CK:
                _chain(c, w, lane, acc, am, gam, GK, DK, YK, DV, OK, OV, CS, SN, RSK, PK, smk, G, B, N, D, NE, RL,
                       HL, CK, eps, HAS_RSTD, False, WANT_AMAX, FOLD)
            if c + PG < CK:
                _chain(c + PG, w, lane, acc, am, gam, GK, DK, YK, DV, OK, OV, CS, SN, RSK, PK, smk2, G, B, N, D, NE,
                       RL, HL, CK, eps, HAS_RSTD, False, WANT_AMAX, FOLD)
        else:
            _chain2(c, c + PG, w, lane, acc, acc2, am, gam, GK, DK, YK, DV, OK, OV, CS, SN, RSK, PK, smk, smk2, G, B,
                    N, D, NE, RL, HL, CK, eps, HAS_RSTD, False, WANT_AMAX, FOLD)
    if cutlass.const_expr(WANT_AMAX):
        a = am[0]
        for o in (1, 2, 4, 8, 16):
            a = cute.arch.fmax(a, cute.arch.shuffle_sync_bfly(a, offset=o))
        if lane == 0:
            AMW[c * 4 + w] = a


@cute.jit
def _fin_group(P, OG, row, cg, tid, st, sr, C: cutlass.Constexpr, CW: cutlass.Constexpr):
    """dgamma columns [cg*CW, cg*CW + CW) of partials P (C, D): stage into smem, then TE finalize order: chain m sums
    rows m, m+4, ... from 0 (add.rn); (s0 + s2) + (s1 + s3); bf16_rn -> OG[row, col]."""
    NT = 128
    for it in cutlass.range_constexpr((C * CW + NT - 1) // NT):
        i = it * NT + tid
        if i < C * CW:
            st[i // CW, i % CW] = P[i // CW, cg * CW + i % CW]
    cute.arch.barrier()
    if tid < 4 * CW:
        cc = tid % CW
        m = tid // CW
        acc = cute.make_rmem_tensor((1,), F32)
        acc[0] = F32(0.0)
        for i in cutlass.range((C - m + 3) // 4, unroll=8):
            acc[0] = _add(acc[0], st[m + 4 * i, cc])
        sr[m, cc] = acc[0]          # every chain m in 0..3 is written: an empty chain (C <= m) is +0, as TE's
    cute.arch.barrier()
    if tid < CW:
        acc2 = cute.make_rmem_tensor((1,), F32)
        acc2[0] = _add(_add(sr[0, tid], sr[2, tid]), _add(sr[1, tid], sr[3, tid]))
        OG[row, cg * CW + tid] = acc2.load().to(BF16)[0]


@cute.kernel
def _fin_kernel(PQ2, PK2, OG, AMW, AM, D: cutlass.Constexpr, CQ: cutlass.Constexpr, CK: cutlass.Constexpr,
                CW: cutlass.Constexpr, NAM: cutlass.Constexpr, WANT_AMAX: cutlass.Constexpr):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    CM = CQ if CQ > CK else CK
    st = cute.make_tensor(cute.arch.alloc_smem(F32, CM * CW), cute.make_layout((CM, CW), stride=(CW, 1)))
    sr = cute.make_tensor(cute.arch.alloc_smem(F32, 4 * CW), cute.make_layout((4, CW), stride=(CW, 1)))
    NG = D // CW
    if bid < NG:
        _fin_group(PQ2, OG, 0, bid, tid, st, sr, CQ, CW)
    elif bid < 2 * NG:
        _fin_group(PK2, OG, 1, bid - NG, tid, st, sr, CK, CW)
    else:
        if cutlass.const_expr(WANT_AMAX):
            a = F32(0.0)
            for it in cutlass.range_constexpr((NAM + 127) // 128):
                if it * 128 + tid < NAM:
                    a = cute.arch.fmax(a, AMW[it * 128 + tid])
            for o in (1, 2, 4, 8, 16):
                a = cute.arch.fmax(a, cute.arch.shuffle_sync_bfly(a, offset=o))
            if tid % 32 == 0:
                sr[0, tid // 32] = a
            cute.arch.barrier()
            if tid == 0:
                a = cute.arch.fmax(cute.arch.fmax(sr[0, 0], sr[0, 1]), cute.arch.fmax(sr[0, 2], sr[0, 3]))
                AM[0] = a
                AM[1] = _e5m2_scale(a)


def _mk(p, spec):
    return cute.make_tensor(p, cute.make_layout(spec[0], stride=spec[1]))


@cute.jit
def _launch(p_dq, p_dg, p_y, p_o, p_dk, p_dv, p_cs, p_sn, p_rq, p_rk, p_gq, p_gk, p_ws, p_og, p_am,
            S: cutlass.Constexpr, B: cutlass.Constexpr, N: cutlass.Constexpr,
            H: cutlass.Constexpr, G: cutlass.Constexpr, D: cutlass.Constexpr, NE: cutlass.Constexpr,
            RL: cutlass.Constexpr, HL: cutlass.Constexpr, CQ: cutlass.Constexpr, CK: cutlass.Constexpr,
            eps: cutlass.Constexpr, HAS_RSTD: cutlass.Constexpr, WANT_AMAX: cutlass.Constexpr,
            FOLD: cutlass.Constexpr, CPW: cutlass.Constexpr, stream):
    S = dict(S)
    HD2 = 2 * H * D
    DQ, DG, YQ, YK = _mk(p_dq, S["dq"]), _mk(p_dg, S["dg"]), _mk(p_y, S["yq"]), _mk(p_y + HD2, S["yk"])
    OQ, OK, OV = _mk(p_o, S["oq"]), _mk(p_o + HD2, S["ok"]), _mk(p_o + (HD2 + G * D), S["ok"])
    DK, DV, CS, SN = _mk(p_dk, S["dk"]), _mk(p_dv, S["dv"]), _mk(p_cs, S["cs"]), _mk(p_sn, S["sn"])
    RSQ, RSK = _mk(p_rq, S["rq"]), _mk(p_rk, S["rk"])
    GQ, GK = _mk(p_gq, ((32, NE), (NE, 1))), _mk(p_gk, ((32, NE), (NE, 1)))
    PQ, PK = _mk(p_ws, ((CQ, 32, NE), (D, NE, 1))), _mk(p_ws + CQ * D, ((CK, 32, NE), (D, NE, 1)))
    PQ2, PK2 = _mk(p_ws, ((CQ, D), (D, 1))), _mk(p_ws + CQ * D, ((CK, D), (D, 1)))
    CM = CQ if CQ > CK else CK
    PG = (CM + CPW - 1) // CPW               # physical CTAs
    AMW = _mk(p_ws + (CQ + CK) * D, ((4 * CM,), (1,)))
    AM, OG = _mk(p_am, ((2,), (1,))), _mk(p_og, ((2, D), (D, 1)))
    _bwd_kernel(DQ, DG, YQ, OQ, DK, DV, YK, OK, OV, CS, SN, RSQ, RSK, GQ, GK, PQ, PK, AMW,
                B, N, H, G, D, NE, RL, HL, CQ, CK, eps, HAS_RSTD, WANT_AMAX, FOLD, CPW, PG).launch(
        grid=[PG, 1, 1], block=[128, 1, 1], stream=stream)
    CW = 4
    _fin_kernel(PQ2, PK2, OG, AMW, AM, D, CQ, CK, CW, 4 * PG, WANT_AMAX).launch(
        grid=[2 * (D // CW) + (1 if WANT_AMAX else 0), 1, 1], block=[128, 1, 1], stream=stream)


_SMS = int(os.environ.get("BWDPROD_SMS", "0"))   # 0 = the device's SM count (GB200: 152)
_DEVSM = {}


def _sm_count(dev):
    if _SMS:
        return _SMS
    if dev not in _DEVSM:
        _DEVSM[dev] = torch.cuda.get_device_properties(dev).multi_processor_count
    return _DEVSM[dev]


def te_ctas_per_col(rows, D, sms):
    return min((rows + 3) // 4, sms * TE_BWD_OCC[D])


_PTYPES = (BF16, BF16, BF16, BF16, BF16, BF16, BF16, BF16, F32, F32, BF16, BF16, F32, BF16, F32)
_PALIGN = (16, 16, 16, 16, 16, 16, 16, 16, 4, 4, 16, 16, 16, 4, 4)
_FNS = {}
_PCACHE = []


def _pcache():
    if not _PCACHE:
        import hashlib
        from pathlib import Path
        from flash_attn.cute import cache_utils
        from ..compile_cache_sm100 import fingerprint
        h = hashlib.sha256(fingerprint("qkvprod").encode())
        h.update(Path(__file__).read_bytes())
        if cache_utils.CUTE_DSL_CACHE_ENABLED:
            _PCACHE.append(cache_utils.JITPersistentCache(cache_utils.get_cache_path() / "dm_sm100" / h.hexdigest()[:24]
                                                           / "qkvbwd"))
        else:
            _PCACHE.append(cache_utils.JITCache())
    return _PCACHE[0]


def _compile(key, S, const):
    f = _FNS.get(key)
    if f is None:
        c = _pcache()
        f = c.get(key) if hasattr(c, "get") else None
        if f is None:
            from cutlass.cute.runtime import make_fake_stream
            ptrs = [make_ptr(t, 16, cute.AddressSpace.gmem, assumed_align=al) for t, al in zip(_PTYPES, _PALIGN)]
            f = cute.compile(_launch, *ptrs, S, *const, make_fake_stream(use_tvm_ffi_env_stream=False),
                             options="--enable-tvm-ffi")
            c[key] = f
        _FNS[key] = f
    return f


def _u(shape, stride, perm, NE):
    """logical (shape, stride) permuted by perm, last dim D split as (32, NE)."""
    sh = tuple(shape[i] for i in perm)
    sd = tuple(stride[i] for i in perm)
    return (sh[:-1] + (32, NE), sd[:-1] + (NE, 1))


def _spec(B, N, H, G, D, R, C, ydim, strides, has_rstd, eps, want_amax, sms):
    """(cache key, S, const) of one launch configuration: every shape / stride the kernel bakes in."""
    NE = D // 32
    sdq, sdk, sdv, sdg, sy, scs, ssn, srq, srk = strides
    ys = (N * sy[0], sy[0], 1) if ydim == 2 else tuple(sy)
    CQ, CK = te_ctas_per_col(B * H * N, D, sms), te_ctas_per_col(B * G * N, D, sms)
    S = dict(dq=_u((B, N, H, D), sdq, (0, 1, 2, 3), NE), dg=_u((N, B, H, D), sdg, (1, 0, 2, 3), NE),
             dk=_u((B, N, G, D), sdk, (0, 1, 2, 3), NE), dv=_u((B, N, G, D), sdv, (0, 1, 2, 3), NE),
             yq=((B, N, H, 2, 32, NE), (ys[0], ys[1], 2 * D, D, NE, 1)),
             yk=((B, N, G, 32, NE), (ys[0], ys[1], D, NE, 1)),
             oq=((B, N, H, 2, 32, NE), (N * C, C, 2 * D, D, NE, 1)), ok=((B, N, G, 32, NE), (N * C, C, D, NE, 1)),
             cs=((B, N, R // NE, NE), (scs[0], scs[1], NE, 1)), sn=((B, N, R // NE, NE), (ssn[0], ssn[1], NE, 1)),
             rq=((B, H, N), tuple(srq) if has_rstd else (0, 0, 0)),
             rk=((B, G, N), tuple(srk) if has_rstd else (0, 0, 0)))
    S = tuple(sorted(S.items()))
    const = (B, N, H, G, D, NE, R // NE, (R // 2) // NE, CQ, CK, float(eps), bool(has_rstd), bool(want_amax), 1, 1)
    return ("bwd",) + const + (S,), S, const, CQ, CK


def aot(B, N, H, G, R=64, D=256, eps=1e-6, sms=152, want_amax=(False, True)):
    """compile the default dense layouts (dq / dk / dv / y contiguous, dgate the [s, b] view of a dense [b, s] tensor,
    rstd given) for this shape, CPU only."""
    C = (2 * H + 2 * G) * D
    dense = lambda *s: tuple(int(x) for x in torch.empty(s, device="meta").stride())  # noqa: E731
    sdg = (H * D, N * H * D, D, 1)                                   # (N, B, H, D) view of a (B, N, H, D) tensor
    strides = (dense(B, N, H, D), dense(B, N, G, D), dense(B, N, G, D), sdg, dense(B, N, C), dense(B, N, R),
               dense(B, N, R), dense(B, H, N), dense(B, G, N))
    for wa in want_amax:
        for ydim in (2, 3):
            st = strides if ydim == 3 else strides[:4] + (dense(B * N, C),) + strides[5:]
            key, S, const, _, _ = _spec(B, N, H, G, D, R, C, ydim, st, True, eps, wa, sms)
            _compile(key, S, const)


def _check(dq, dk, dv, dgate, y, gq, gk, cos, sin, H, G, rstd_q, rstd_k):
    B, N, Hd, D = dq.shape
    C = y.shape[-1]
    R = cos.shape[-1]
    if D not in TE_BWD_OCC:
        raise ValueError(f"bwd_producers: head_dim {D} unsupported")
    if Hd != H or H % G or C != (2 * H + 2 * G) * D:
        raise ValueError("bwd_producers: dq (B, N, H, D) / y (.., (2H+2G)D) / H % G mismatch")
    if y.shape not in ((B, N, C), (B * N, C)) or not y.is_contiguous():
        raise ValueError(f"bwd_producers: y {tuple(y.shape)} must be a contiguous (B, N, C) or (B*N, C)")
    if not (dk.shape == dv.shape == (B, N, G, D)):
        raise ValueError(f"bwd_producers: dk/dv {tuple(dk.shape)} must be (B, N, G, D) (folded, token-major)")
    if dgate.shape != (N, B, H, D) or cos.shape != (B, N, R) or sin.shape != (B, N, R):
        raise ValueError("bwd_producers: dgate (N, B, H, D) / cos, sin (B, N, R) shape mismatch")
    NE = D // 32
    if not (R % (2 * NE) == 0 and (R // 2) // NE in (1, 2, 4, 8, 16) and R <= D):
        raise ValueError(f"bwd_producers: rotary dim {R} unsupported at D {D}")
    for nm, t in (("dq", dq), ("dk", dk), ("dv", dv), ("dgate", dgate), ("y", y), ("cos", cos), ("sin", sin)):
        if t.dtype != torch.bfloat16 or t.stride(-1) != 1 or t.device != y.device:
            raise ValueError(f"bwd_producers: {nm} must be bf16 on y's device with a unit last stride")
    if (rstd_q is None) != (rstd_k is None):
        raise ValueError("bwd_producers: pass both rstd_q and rstd_k or neither")
    if rstd_q is not None and (rstd_q.shape != (B, H, N) or rstd_k.shape != (B, G, N) or
                               rstd_q.dtype != torch.float32 or rstd_k.dtype != torch.float32):
        raise ValueError("bwd_producers: rstd_q (B, H, N) / rstd_k (B, G, N) fp32")
    return B, N, D, C, R


def _vec_ok(t, NE):
    return t.data_ptr() % 16 == 0 and not any(st % NE for st in t.stride()[:-1])


_WS = {}


def bwd_producers(dq, dk, dv, dgate, y, gq, gk, cos, sin, H, G, eps=1e-6, rstd_q=None, rstd_k=None,
                  want_amax=False, out=None):
    """see the module docstring.  One CUDA stream at a time per (device, shape): the dgamma workspace is reused
    stream-ordered."""
    B, N, D, C, R = _check(dq, dk, dv, dgate, y, gq, gk, cos, sin, H, G, rstd_q, rstd_k)
    NE = D // 32
    dq, dk, dv, dgate, cos, sin = (t if _vec_ok(t, NE) else t.contiguous()
                                   for t in (dq, dk, dv, dgate, cos, sin))
    has_rstd = rstd_q is not None
    dev = y.device
    sms = _sm_count(dev.index)
    strides = (dq.stride(), dk.stride(), dv.stride(), dgate.stride(), y.stride(), cos.stride(), sin.stride(),
               rstd_q.stride() if has_rstd else None, rstd_k.stride() if has_rstd else None)
    key, S, const, CQ, CK = _spec(B, N, H, G, D, R, C, y.dim(), strides, has_rstd, eps, want_amax, sms)
    f = _compile(key, S, const)
    wk = (dev.index, CQ, CK, D)
    ws = _WS.get(wk)
    if ws is None:
        ws = _WS[wk] = torch.empty((CQ + CK) * D + 4 * max(CQ, CK), device=dev, dtype=torch.float32)
    if out is None:
        dY = torch.empty(y.shape, device=dev, dtype=torch.bfloat16)
    else:
        dY = out
        if not (dY.is_contiguous() and dY.shape == y.shape and dY.dtype == torch.bfloat16 and dY.data_ptr() % 16 == 0):
            raise ValueError("bwd_producers: out must be a contiguous, 16 B aligned bf16 tensor like y")
    og = torch.empty(2, D, device=dev, dtype=torch.bfloat16)              # dgq | dgk
    am = torch.empty(2, device=dev, dtype=torch.float32) if want_amax else ws
    gq = gq.to(torch.bfloat16).contiguous()
    gk = gk.to(torch.bfloat16).contiguous()
    if y.data_ptr() % 16 or gq.data_ptr() % 16 or gk.data_ptr() % 16:
        raise ValueError("bwd_producers: y / gq / gk must be 16 B aligned")
    wsa = ws.data_ptr()
    f(dq.data_ptr(), dgate.data_ptr(), y.data_ptr(), dY.data_ptr(), dk.data_ptr(), dv.data_ptr(), cos.data_ptr(),
      sin.data_ptr(), rstd_q.data_ptr() if has_rstd else wsa, rstd_k.data_ptr() if has_rstd else wsa,
      gq.data_ptr(), gk.data_ptr(), wsa, og.data_ptr(), am.data_ptr(), torch.cuda.current_stream(dev).cuda_stream)
    dgq, dgk = og.unbind(0)
    if not want_amax:
        return dY, dgq, dgk
    amax, scale = am.split(1)
    return dY, dgq, dgk, amax, scale
