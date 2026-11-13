"""
Fused q/k/v producers for the sm100 Delta-Matching attention (head_dim 256): CuTe DSL kernels that read the q/k/v
projection's bf16 output y [B, N, (2H+2G)D] once (columns: per head [q_h | gate_h], then k, then v) and write the
attention-ready sm100-mx operands, bitwise equal to the unfused chain

    te.RMSNorm (TE 2.19 rmsnorm fwd math) -> partial RoPE (native bf16 cos/sin table; mul.rn.ftz, mul.rn.ftz,
    add.rn.ftz, bf16) -> delta_matching.sm100.quant.quant_qkv2 (Q * D^-0.5, smooth-K mean, V),

    fwd_producers(y, gq, gk, cos, sin, H, G, R, eps)
        -> q8 (B,N,H,D) e4m3, qs (B,H,N/32) fp32, k8 / v8 (B,N,G,D), ks / vs (B,G,N/32), kvw (B,G,N/64) int32,
           rstd_q (B,H,N), rstd_k (B,G,N) fp32 (TE rsigma, row order b, head, token), kmean (B,G,D) fp32, gate view
    split K path (ring CP: all-reduce the column sum between the two calls):
    fwd_producers_ksum(...) -> same without k8 / ks / K bytes of kvw; ksum (B,G,D) = this rank's column sum
    k_quant_mean(y, gk, cos, sin, km, H, G, R, eps, out) -> k8, ks (+ K bytes of out["kvw"])

Kernels: launch A = K1 CTAs [k norm + RoPE of 64 rows stashed in smem as bf16, V quant (two 32-row MX blocks), the
quant_qkv2 column tree of the 64-row chunk; the last K1 CTA of a (b, kv head) folds the chunk sums in quant_qkv2's
order into the mean (or the sum)] then Q CTAs [64 rows of one head: norm + RoPE + D^-0.5, MX quant per 32 rows].
Launch B = K2 CTAs [k_rot re-derived from y, minus the mean, MX quant]. A CTA covers whole heads (256 columns), so
every 32 x 256 amax is CTA-local. No transposed twins (tcgen05 reads MN-major operands through TMA).

Layout per row: R/8 lanes, lane c owns the D/R chunks m = c + (R/8) k of 8 elements (TE's lane l = chunk m, so the
TE butterfly is the same tree); chunk 0 is its rotated chunk, the RoPE partner is lane c ^ (R/16).
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_ptr
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

F32 = cutlass.Float32
I32 = cutlass.Int32
TR = 32                      # rows per MX block
RB = 64                      # rows per CTA
_FMA = True                  # TE's fma chain for the sum of squares


# ------------------------------------------------------------------------------------------------ exact PTX helpers
def _asm(ret, args, s, cons, loc, ip, side=False):
    return llvm.inline_asm(ret, [a.ir_value(loc=loc, ip=ip) for a in args], s, cons, has_side_effects=side,
                           is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def f_op2(a, b, op: str, *, loc=None, ip=None):
    return F32(_asm(T.f32(), [F32(a), F32(b)], f"{op} $0, $1, $2;", "=f,f,f", loc, ip))


@dsl_user_op
def f_fma(a, b, c, *, loc=None, ip=None):
    return F32(_asm(T.f32(), [F32(a), F32(b), F32(c)], "fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f", loc, ip))


@dsl_user_op
def f_rsqrt(a, *, loc=None, ip=None):
    return F32(_asm(T.f32(), [F32(a)], "rsqrt.approx.f32 $0, $1;", "=f,f", loc, ip))


@dsl_user_op
def f_bf16r(a, *, loc=None, ip=None):
    """RTNE to bf16, back to fp32 (exact)."""
    return F32(_asm(T.f32(), [F32(a)], "{ .reg .b16 h; cvt.rn.bf16.f32 h, $1; cvt.f32.bf16 $0, h; }", "=f,f", loc, ip))


@dsl_user_op
def f_sel_lt(a, b, c, k: int, *, loc=None, ip=None):
    """c < k ? a : b  (c Int32, k immediate)."""
    return F32(_asm(T.f32(), [F32(a), F32(b), I32(c)],
                    f"{{ .reg .pred p; setp.lt.s32 p, $3, {k}; selp.f32 $0, $1, $2, p; }}", "=f,f,f,r", loc, ip))


@dsl_user_op
def f_abs(a, *, loc=None, ip=None):
    return F32(_asm(T.f32(), [F32(a)], "abs.f32 $0, $1;", "=f,f", loc, ip))


@dsl_user_op
def f_max(a, b, *, loc=None, ip=None):
    return F32(_asm(T.f32(), [F32(a), F32(b)], "max.f32 $0, $1, $2;", "=f,f,f", loc, ip))


@dsl_user_op
def i2f(n, *, loc=None, ip=None):
    return F32(_asm(T.f32(), [I32(n)], "cvt.rn.f32.s32 $0, $1;", "=f,r", loc, ip))


@dsl_user_op
def e8m0(amax, *, loc=None, ip=None):
    """quant.quant_rows' exponent byte: ((bits >> 23) & 0xFF) - 8 + (mantissa > 0x600000), clamp [1, 254]; 127 when
    amax is not > 0."""
    return I32(_asm(T.i32(), [F32(amax)],
                    "{\n\t.reg .b32 a, m, e;\n\t.reg .pred p, q;\n\t"
                    "mov.b32 a, $1;\n\tshr.u32 e, a, 23;\n\tand.b32 e, e, 255;\n\tsub.s32 e, e, 8;\n\t"
                    "and.b32 m, a, 8388607;\n\tsetp.gt.u32 p, m, 6291456;\n\tselp.s32 m, 1, 0, p;\n\tadd.s32 e, e, m;\n\t"
                    "max.s32 e, e, 1;\n\tmin.s32 e, e, 254;\n\t"
                    "setp.gt.f32 q, $1, 0f00000000;\n\tselp.s32 $0, e, 127, q;\n\t}", "=r,f", loc, ip))


@dsl_user_op
def shl23f(e, *, loc=None, ip=None):
    return F32(_asm(T.f32(), [I32(e)], "shl.b32 $0, $1, 23;", "=f,r", loc, ip))


@dsl_user_op
def e4m3x4(a, b, c, d, *, loc=None, ip=None):
    """4 fp32 -> 4 e4m3 bytes (byte 0 = a), cvt.rn.satfinite (as quant_cute.e4m3x4)."""
    return I32(_asm(T.i32(), [F32(a), F32(b), F32(c), F32(d)],
                    "{ .reg .b16 l, h; cvt.rn.satfinite.e4m3x2.f32 l, $2, $1; cvt.rn.satfinite.e4m3x2.f32 h, $4, $3; "
                    "mov.b32 $0, {l, h}; }", "=r,f,f,f,f", loc, ip))


@dsl_user_op
def bf_lo(u, *, loc=None, ip=None):
    return F32(_asm(T.f32(), [I32(u)], "{ .reg .b32 t; shl.b32 t, $1, 16; mov.b32 $0, t; }", "=f,r", loc, ip))


@dsl_user_op
def bf_hi(u, *, loc=None, ip=None):
    return F32(_asm(T.f32(), [I32(u)], "{ .reg .b32 t; and.b32 t, $1, 0xffff0000; mov.b32 $0, t; }", "=f,r", loc, ip))


@dsl_user_op
def pack_bf16x2(lo, hi, *, loc=None, ip=None):
    return I32(_asm(T.i32(), [F32(lo), F32(hi)], "cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,f,f", loc, ip))


@dsl_user_op
def st_u8(addr, v, *, loc=None, ip=None):
    _asm(None, [cutlass.Int64(addr), I32(v)], "st.global.u8 [$0], $1;", "l,r", loc, ip, side=True)


@dsl_user_op
def ld_cg(addr, *, loc=None, ip=None):
    return F32(_asm(T.f32(), [cutlass.Int64(addr)], "ld.global.cg.f32 $0, [$1];", "=f,l", loc, ip, side=True))


def _mul(a, b):
    return f_op2(a, b, "mul.rn.f32")


def _add(a, b):
    return f_op2(a, b, "add.rn.f32")


def _lay(shape):
    st, acc = [], 1
    for s in reversed(shape):
        st.append(acc)
        acc *= s
    return cute.make_layout(tuple(shape), stride=tuple(reversed(st)))


# ------------------------------------------------------------------------------------------------ shared row math
@cute.jit
def _ld8(T8W, row, col):
    """8 bf16 at a [rows, W/8, 4] int32 view -> rmem f32[8]."""
    w = T8W[row, col, None].load()
    o = cute.make_rmem_tensor((8,), F32)
    for t in cutlass.range_constexpr(4):
        o[2 * t] = bf_lo(w[t])
        o[2 * t + 1] = bf_hi(w[t])
    return o


@cute.jit
def _load_g(G4, c, RC: cutlass.Constexpr, NCH: cutlass.Constexpr):
    g = cute.make_rmem_tensor((NCH * 8,), F32)
    for k in cutlass.range_constexpr(NCH):
        gk = _ld8(G4, 0, c + RC * k)
        for j in cutlass.range_constexpr(8):
            g[k * 8 + j] = gk[j]
    return g


@cute.jit
def _norm_rope_row(Y4, row, col0, g, C4, S4, c, eps, D: cutlass.Constexpr, RC: cutlass.Constexpr,
                   NCH: cutlass.Constexpr):
    """The lane's NCH chunks (m = c + RC*k) of one row: te.RMSNorm (TE's tree: per-lane fma chain over the lane's 8
    elements, butterfly over the row's lanes = TE's xor 1..RC/2, then the chunk tree = TE's xor RC..16) and the
    partial RoPE on chunk 0.  Returns (out rmem f32[NCH*8] bf16-valued, rs)."""
    HC = RC // 2
    x = cute.make_rmem_tensor((NCH * 8,), F32)
    for k in cutlass.range_constexpr(NCH):
        xk = _ld8(Y4, row, col0 + c + RC * k)
        for j in cutlass.range_constexpr(8):
            x[k * 8 + j] = xk[j]
    p = cute.make_rmem_tensor((NCH,), F32)
    for k in cutlass.range_constexpr(NCH):
        s = F32(0.0)
        for j in cutlass.range_constexpr(8):
            if cutlass.const_expr(_FMA):
                s = f_fma(x[k * 8 + j], x[k * 8 + j], s)
            else:
                s = _add(s, _mul(x[k * 8 + j], x[k * 8 + j]))
        p[k] = s
    off = 1
    for _ in cutlass.range_constexpr(RC.bit_length() - 1):
        for k in cutlass.range_constexpr(NCH):
            p[k] = _add(p[k], cute.arch.shuffle_sync_bfly(p[k], offset=off))
        off *= 2
    step = 1
    for _ in cutlass.range_constexpr(NCH.bit_length() - 1):
        for k in cutlass.range_constexpr(0, NCH, 2 * step):
            p[k] = _add(p[k], p[k + step])
        step *= 2
    rs = f_rsqrt(_add(_mul(p[0], F32(1.0 / D)), eps))
    o = cute.make_rmem_tensor((NCH * 8,), F32)
    for k in cutlass.range_constexpr(NCH):
        for j in cutlass.range_constexpr(8):
            o[k * 8 + j] = f_bf16r(_mul(g[k * 8 + j], _mul(rs, x[k * 8 + j])))
    cs = _ld8(C4, row, c)
    sn = _ld8(S4, row, c)
    sgn = f_sel_lt(F32(-1.0), F32(1.0), c, HC)
    for j in cutlass.range_constexpr(8):
        pr = cute.arch.shuffle_sync_bfly(o[j], offset=HC)
        o[j] = f_bf16r(f_op2(f_op2(o[j], cs[j], "mul.rn.ftz.f32"), f_op2(pr * sgn, sn[j], "mul.rn.ftz.f32"),
                             "add.rn.ftz.f32"))
    return o, rs


@cute.jit
def _cta_amax(v, red):
    """max over the CTA's 256 threads (8 warps) of v; red: smem f32[8] private to this call site."""
    lane = cute.arch.lane_idx()
    warp = cute.arch.warp_idx()
    w = cute.arch.warp_reduction_max(v)
    if lane == 0:
        red[warp] = w
    cute.arch.barrier()
    m = red[0]
    for wi in cutlass.range_constexpr(1, 8):
        m = f_max(m, red[wi])
    return m


@cute.jit
def _codes8(v, k: cutlass.Constexpr, inv):
    """two u32 words of e4m3(v[8k : 8k+8] * inv)."""
    w = cute.make_rmem_tensor((2,), I32)
    t = cute.make_rmem_tensor((8,), F32)
    for j in cutlass.range_constexpr(8):
        t[j] = _mul(v[k * 8 + j], inv)
    w[0] = e4m3x4(t[0], t[1], t[2], t[3])
    w[1] = e4m3x4(t[4], t[5], t[6], t[7])
    return w


# ------------------------------------------------------------------------------------------------ Q
@cute.jit
def _q_body(tid, pm, pbh, Y4, G4, C4, S4, Q8W, QS, RSQ, red, eps, N: cutlass.Constexpr, H: cutlass.Constexpr,
            D: cutlass.Constexpr, R: cutlass.Constexpr):
    b = pbh // H
    h = pbh - b * H
    RC = R // 8
    NCH = D // R
    RPW = 32 // RC
    NSLOT = 8 * RPW
    SM = D ** -0.5
    lane = tid % 32
    warp = tid // 32
    c = lane % RC
    slot = warp * RPW + lane // RC
    g = _load_g(G4, c, RC, NCH)
    for i in cutlass.range_constexpr(RB // NSLOT):          # pass i = one 32-row MX block
        r = i * NSLOT + slot
        n_ = pm * RB + r
        row = b * N + n_
        o, rs = _norm_rope_row(Y4, row, h * (2 * D // 8), g, C4, S4, c, eps, D, RC, NCH)
        if c == 0:
            RSQ[pbh * N + n_] = rs
        t = cute.make_rmem_tensor((NCH * 8,), F32)
        am = F32(0.0)
        for j in cutlass.range_constexpr(NCH * 8):
            t[j] = _mul(o[j], F32(SM))
            am = f_max(am, f_abs(t[j]))
        m = _cta_amax(am, red[i, None])
        eb = e8m0(m)
        inv = shl23f(I32(254) - eb)
        for k in cutlass.range_constexpr(NCH):
            Q8W[row, h, c + RC * k, None].store(_codes8(t, k, inv).load())
        if tid == 0:
            QS[pbh, (pm * RB + i * NSLOT) // TR] = shl23f(eb)


# ------------------------------------------------------------------------------------------------ K1 (+ V, + fold)
@cute.jit
def _k1_body(tid, pm, pbg, buf, red, CTR, KM, Y4, G4, C4, S4, KP, RSK, V8W, VS, pKVW, eps, SUMONLY: cutlass.Constexpr,
             N: cutlass.Constexpr, H: cutlass.Constexpr, G: cutlass.Constexpr, D: cutlass.Constexpr,
             R: cutlass.Constexpr, ROWS: cutlass.Constexpr, WITH_V: cutlass.Constexpr):
    b = pbg // G
    gh = pbg - b * G
    RC = R // 8
    NCH = D // R
    RPW = 32 // RC
    NSLOT = 8 * RPW
    lane = tid % 32
    warp = tid // 32
    c = lane % RC
    slot = warp * RPW + lane // RC
    g = _load_g(G4, c, RC, NCH)
    kcol = (2 * H * D + gh * D) // 8
    vcol = (2 * H * D + G * D + gh * D) // 8
    for i in cutlass.range_constexpr(RB // NSLOT):
        r = i * NSLOT + slot
        n_ = pm * RB + r
        row = b * N + n_
        o, rs = _norm_rope_row(Y4, row, kcol, g, C4, S4, c, eps, D, RC, NCH)
        if c == 0:
            RSK[pbg * N + n_] = rs
        for k in cutlass.range_constexpr(NCH):
            w = cute.make_rmem_tensor((4,), I32)
            for t in cutlass.range_constexpr(4):
                w[t] = pack_bf16x2(o[k * 8 + 2 * t], o[k * 8 + 2 * t + 1])
            buf[r, c + RC * k, None].store(w.load())
    if cutlass.const_expr(WITH_V):
        for i in cutlass.range_constexpr(RB // NSLOT):
            r = i * NSLOT + slot
            n_ = pm * RB + r
            row = b * N + n_
            v = cute.make_rmem_tensor((NCH * 8,), F32)
            am = F32(0.0)
            for k in cutlass.range_constexpr(NCH):
                vk = _ld8(Y4, row, vcol + c + RC * k)
                for j in cutlass.range_constexpr(8):
                    v[k * 8 + j] = vk[j]
                    am = f_max(am, f_abs(vk[j]))
            m = _cta_amax(am, red[i, None])
            eb = e8m0(m)
            inv = shl23f(I32(254) - eb)
            for k in cutlass.range_constexpr(NCH):
                V8W[row, gh, c + RC * k, None].store(_codes8(v, k, inv).load())
            if tid == 0:
                blk = (pm * RB + i * NSLOT) // TR
                VS[pbg, blk] = shl23f(eb)
                st_u8(pKVW + (cutlass.Int64(pbg) * (N // 64) + 2 * (blk // 4) + 1) * 4 + blk % 4, eb)
    cute.arch.barrier()
    # quant_qkv2's chunk tree (aligned K): column d = T32(rows 0,2,..,62) + T32(rows 1,3,..,63), T32 = balanced pairwise
    bh16 = cute.make_tensor(cute.recast_ptr(buf.iterator, dtype=cutlass.BFloat16), _lay((RB, D)))
    if tid < D:
        tot = F32(0.0)
        for w2 in cutlass.range_constexpr(2):
            lv = cute.make_rmem_tensor((32,), F32)
            for rep in cutlass.range_constexpr(32):
                lv[rep] = bh16[2 * rep + w2, tid].to(F32)
            for lvl in cutlass.range_constexpr(5):
                for q in cutlass.range_constexpr(16 >> lvl):
                    lv[q] = _add(lv[2 * q], lv[2 * q + 1])
            if cutlass.const_expr(w2 == 0):
                tot = lv[0]
            else:
                tot = _add(tot, lv[0])
        KP[pbg, pm, tid] = tot
    # the last K1 CTA of this (b, kv head) folds the chunk sums: partial s = ((0 + c_0) + ..) over ROWS/64 chunks,
    # acc = ((0 + P_0) + P_1) + ..; mean = div.full(acc, N)  (quant_qkv2: fold_partial, fold_head)
    cute.arch.fence_acq_rel_gpu()
    cute.arch.barrier()
    if tid == 0:
        red[2, 0] = i2f(cute.arch.atomic_add(CTR.iterator + pbg, I32(1), sem="acq_rel", scope="gpu"))
    cute.arch.barrier()
    if red[2, 0] == F32(N // RB - 1):
        cute.arch.fence_acq_rel_gpu()
        if tid < D:
            NCHP = ROWS // RB
            acc = F32(0.0)
            for s in cutlass.range(N // ROWS, unroll=1):
                part = F32(0.0)
                for cc in cutlass.range_constexpr(NCHP):
                    part = _add(part, KP[pbg, s * NCHP + cc, tid])
                acc = _add(acc, part)
            if cutlass.const_expr(SUMONLY):
                KM[pbg, tid] = acc
            else:
                KM[pbg, tid] = f_op2(acc, F32(float(N)), "div.full.f32")
        if tid == 0:
            CTR[pbg] = I32(0)


@cute.kernel
def _a_kernel(CTR, KM, Y4, GQ4, GK4, C4, S4, Q8W, QS, RSQ, KP, RSK, V8W, VS, pKVW, eps: F32,
              N: cutlass.Constexpr, H: cutlass.Constexpr, G: cutlass.Constexpr, BG: cutlass.Constexpr,
              D: cutlass.Constexpr, R: cutlass.Constexpr, ROWS: cutlass.Constexpr, SUMONLY: cutlass.Constexpr,
              ROLE: cutlass.Constexpr):
    """K1 (+ V, + last-CTA fold) CTAs first, then the Q CTAs: one launch (ROLE 0); ROLE 1 = K1 (+ V) only, ROLE 2 =
    Q only (ring CP: the K-sum all-reduce overlaps the Q launch)."""
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    NK1 = (N // RB) * BG
    red = cute.make_tensor(cute.arch.alloc_smem(F32, 32, 16), _lay((4, 8)))
    buf = red
    if cutlass.const_expr(ROLE != 2):     # the K1 bf16 stash (Q-only launches need no big smem)
        buf = cute.make_tensor(cute.arch.alloc_smem(I32, RB * D // 2, 16), _lay((RB, D // 8, 4)))
    if cutlass.const_expr(ROLE == 2):
        _q_body(tid, bid % (N // RB), bid // (N // RB), Y4, GQ4, C4, S4, Q8W, QS, RSQ, red, eps, N, H, D, R)
    elif cutlass.const_expr(ROLE == 1):
        _k1_body(tid, bid % (N // RB), bid // (N // RB), buf, red, CTR, KM, Y4, GK4, C4, S4, KP, RSK, V8W, VS, pKVW,
                 eps, SUMONLY, N, H, G, D, R, ROWS, True)
    else:
        if bid < NK1:
            _k1_body(tid, bid % (N // RB), bid // (N // RB), buf, red, CTR, KM, Y4, GK4, C4, S4, KP, RSK, V8W, VS,
                     pKVW, eps, SUMONLY, N, H, G, D, R, ROWS, True)
        else:
            q = bid - NK1
            _q_body(tid, q % (N // RB), q // (N // RB), Y4, GQ4, C4, S4, Q8W, QS, RSQ, red, eps, N, H, D, R)


@cute.jit
def _launch_a(pCTR, pKM, pY, pGQ, pGK, pC, pS, pQ8, pQS, pRSQ, pKP, pRSK, pV8, pVS, pKVW: cutlass.Int64, eps: F32,
              B: cutlass.Constexpr, N: cutlass.Constexpr, H: cutlass.Constexpr, G: cutlass.Constexpr,
              D: cutlass.Constexpr, R: cutlass.Constexpr, ROWS: cutlass.Constexpr, SUMONLY: cutlass.Constexpr,
              ROLE: cutlass.Constexpr, stream):
    W = (2 * H + 2 * G) * D
    BH = B * H
    BG = B * G
    CTR = cute.make_tensor(pCTR, _lay((BG,)))
    KM = cute.make_tensor(pKM, _lay((BG, D)))
    Y4 = cute.make_tensor(pY, _lay((B * N, W // 8, 4)))
    GQ4 = cute.make_tensor(pGQ, _lay((1, D // 8, 4)))
    GK4 = cute.make_tensor(pGK, _lay((1, D // 8, 4)))
    C4 = cute.make_tensor(pC, _lay((B * N, R // 8, 4)))
    S4 = cute.make_tensor(pS, _lay((B * N, R // 8, 4)))
    Q8W = cute.make_tensor(pQ8, _lay((B * N, H, D // 8, 2)))
    QS = cute.make_tensor(pQS, _lay((BH, N // TR)))
    RSQ = cute.make_tensor(pRSQ, _lay((BH * N,)))
    KP = cute.make_tensor(pKP, _lay((BG, N // RB, D)))
    RSK = cute.make_tensor(pRSK, _lay((BG * N,)))
    V8W = cute.make_tensor(pV8, _lay((B * N, G, D // 8, 2)))
    VS = cute.make_tensor(pVS, _lay((BG, N // TR)))
    ncta = (N // RB) * ((BG if ROLE != 2 else 0) + (BH if ROLE != 1 else 0))
    _a_kernel(CTR, KM, Y4, GQ4, GK4, C4, S4, Q8W, QS, RSQ, KP, RSK, V8W, VS, pKVW, eps, N, H, G, BG, D, R, ROWS,
              SUMONLY, ROLE).launch(grid=[ncta, 1, 1], block=[256, 1, 1], stream=stream)


# ------------------------------------------------------------------------------------------------ K2
@cute.kernel
def _k2_kernel(Y4, G4, C4, S4, KM, K8W, KS, pKVW: cutlass.Int64, eps: F32, N: cutlass.Constexpr, H: cutlass.Constexpr,
               G: cutlass.Constexpr, D: cutlass.Constexpr, R: cutlass.Constexpr):
    tid, _, _ = cute.arch.thread_idx()
    pm, pbg, _ = cute.arch.block_idx()
    b = pbg // G
    gh = pbg - b * G
    RC = R // 8
    NCH = D // R
    RPW = 32 // RC
    NSLOT = 8 * RPW
    lane = tid % 32
    warp = tid // 32
    c = lane % RC
    slot = warp * RPW + lane // RC
    red = cute.make_tensor(cute.arch.alloc_smem(F32, 32, 16), _lay((4, 8)))
    g = _load_g(G4, c, RC, NCH)
    km = cute.make_rmem_tensor((NCH * 8,), F32)
    for k in cutlass.range_constexpr(NCH):
        for j in cutlass.range_constexpr(8):
            km[k * 8 + j] = KM[pbg, (c + RC * k) * 8 + j]
    kcol = (2 * H * D + gh * D) // 8
    for i in cutlass.range_constexpr(RB // NSLOT):
        r = i * NSLOT + slot
        n_ = pm * RB + r
        row = b * N + n_
        o, rs = _norm_rope_row(Y4, row, kcol, g, C4, S4, c, eps, D, RC, NCH)
        t = cute.make_rmem_tensor((NCH * 8,), F32)
        am = F32(0.0)
        for j in cutlass.range_constexpr(NCH * 8):
            t[j] = f_op2(o[j], km[j], "sub.rn.f32")
            am = f_max(am, f_abs(t[j]))
        m = _cta_amax(am, red[i, None])
        eb = e8m0(m)
        inv = shl23f(I32(254) - eb)
        for k in cutlass.range_constexpr(NCH):
            K8W[row, gh, c + RC * k, None].store(_codes8(t, k, inv).load())
        if tid == 0:
            blk = (pm * RB + i * NSLOT) // TR
            KS[pbg, blk] = shl23f(eb)
            st_u8(pKVW + (cutlass.Int64(pbg) * (N // 64) + 2 * (blk // 4)) * 4 + blk % 4, eb)


@cute.jit
def _launch_k2(pY, pG, pC, pS, pKM, pK8, pKS, pKVW: cutlass.Int64, eps: F32, B: cutlass.Constexpr,
               N: cutlass.Constexpr, H: cutlass.Constexpr, G: cutlass.Constexpr, D: cutlass.Constexpr,
               R: cutlass.Constexpr, stream):
    W = (2 * H + 2 * G) * D
    BG = B * G
    Y4 = cute.make_tensor(pY, _lay((B * N, W // 8, 4)))
    G4 = cute.make_tensor(pG, _lay((1, D // 8, 4)))
    C4 = cute.make_tensor(pC, _lay((B * N, R // 8, 4)))
    S4 = cute.make_tensor(pS, _lay((B * N, R // 8, 4)))
    KM = cute.make_tensor(pKM, _lay((BG, D)))
    K8W = cute.make_tensor(pK8, _lay((B * N, G, D // 8, 2)))
    KS = cute.make_tensor(pKS, _lay((BG, N // TR)))
    _k2_kernel(Y4, G4, C4, S4, KM, K8W, KS, pKVW, eps, N, H, G, D, R).launch(
        grid=[N // RB, BG, 1], block=[256, 1, 1], stream=stream)


# ------------------------------------------------------------------------------------------------ bf16 mode
@cute.kernel
def _bf_kernel(Y4, GQ4, GK4, C4, S4, QB, KB, RSQ, RSK, eps: F32, N: cutlass.Constexpr, H: cutlass.Constexpr,
               G: cutlass.Constexpr, D: cutlass.Constexpr, R: cutlass.Constexpr):
    """norm + RoPE only, bf16 out (the BF16-fair arm's q / k, and the producers' intermediate for the gates):
    grid (N/64, B*(H+G)); heads 0..H-1 of a batch row are q, H..H+G-1 are k."""
    tid, _, _ = cute.arch.thread_idx()
    pm, pz, _ = cute.arch.block_idx()
    b = pz // (H + G)
    hz = pz - b * (H + G)
    RC = R // 8
    NCH = D // R
    RPW = 32 // RC
    NSLOT = 8 * RPW
    lane = tid % 32
    warp = tid // 32
    c = lane % RC
    slot = warp * RPW + lane // RC
    isq = hz < H
    gq = _load_g(GQ4, c, RC, NCH)
    gk = _load_g(GK4, c, RC, NCH)
    g = cute.make_rmem_tensor((NCH * 8,), F32)
    for j in cutlass.range_constexpr(NCH * 8):
        g[j] = gq[j] if isq else gk[j]
    col0 = hz * (2 * D // 8) if isq else (2 * H * D + (hz - H) * D) // 8
    for i in cutlass.range_constexpr(RB // NSLOT):
        r = i * NSLOT + slot
        n_ = pm * RB + r
        row = b * N + n_
        o, rs = _norm_rope_row(Y4, row, col0, g, C4, S4, c, eps, D, RC, NCH)
        for k in cutlass.range_constexpr(NCH):
            w = cute.make_rmem_tensor((4,), I32)
            for t in cutlass.range_constexpr(4):
                w[t] = pack_bf16x2(o[k * 8 + 2 * t], o[k * 8 + 2 * t + 1])
            if isq:
                QB[row, hz, c + RC * k, None].store(w.load())
            else:
                KB[row, hz - H, c + RC * k, None].store(w.load())
        if c == 0:
            if isq:
                RSQ[(b * H + hz) * N + n_] = rs
            else:
                RSK[(b * G + hz - H) * N + n_] = rs


@cute.jit
def _launch_bf(pY, pGQ, pGK, pC, pS, pQB, pKB, pRSQ, pRSK, eps: F32, B: cutlass.Constexpr, N: cutlass.Constexpr,
               H: cutlass.Constexpr, G: cutlass.Constexpr, D: cutlass.Constexpr, R: cutlass.Constexpr, stream):
    W = (2 * H + 2 * G) * D
    Y4 = cute.make_tensor(pY, _lay((B * N, W // 8, 4)))
    GQ4 = cute.make_tensor(pGQ, _lay((1, D // 8, 4)))
    GK4 = cute.make_tensor(pGK, _lay((1, D // 8, 4)))
    C4 = cute.make_tensor(pC, _lay((B * N, R // 8, 4)))
    S4 = cute.make_tensor(pS, _lay((B * N, R // 8, 4)))
    QB = cute.make_tensor(pQB, _lay((B * N, H, D // 8, 4)))
    KB = cute.make_tensor(pKB, _lay((B * N, G, D // 8, 4)))
    RSQ = cute.make_tensor(pRSQ, _lay((B * H * N,)))
    RSK = cute.make_tensor(pRSK, _lay((B * G * N,)))
    _bf_kernel(Y4, GQ4, GK4, C4, S4, QB, KB, RSQ, RSK, eps, N, H, G, D, R).launch(
        grid=[N // RB, B * (H + G), 1], block=[256, 1, 1], stream=stream)


# ------------------------------------------------------------------------------------------------ host
_GM = cute.AddressSpace.gmem
_DT = {torch.bfloat16: cutlass.BFloat16, torch.float32: cutlass.Float32, torch.float8_e4m3fn: cutlass.Int32,
       torch.int32: cutlass.Int32}
_FNS = {}
_PCACHE = []
_CTR = {}


def _fingerprint():
    from ..compile_cache_sm100 import fingerprint
    h = hashlib.sha256(fingerprint("qkvprod").encode())
    h.update(Path(__file__).read_bytes())
    return h.hexdigest()[:24]


def _pcache():
    if not _PCACHE:
        from flash_attn.cute import cache_utils
        if cache_utils.CUTE_DSL_CACHE_ENABLED:
            _PCACHE.append(cache_utils.JITPersistentCache(cache_utils.get_cache_path() / "dm_sm100" / _fingerprint()
                                                           / "qkvprod"))
        else:
            _PCACHE.append(cache_utils.JITCache())
    return _PCACHE[0]


def _fake_ptr(dt):
    return make_ptr(dt, 16, _GM, assumed_align=16)


def _compile(key, fn, dts, consts):
    """compile with placeholder pointers (no CUDA context needed: AOT on a CPU node), persistent cache."""
    f = _FNS.get(key)
    if f is None:
        c = _pcache()
        f = c.get(key) if hasattr(c, "get") else None
        if f is None:
            from cutlass.cute.runtime import make_fake_stream
            args = [_fake_ptr(d) if d is not None else cutlass.Int64(0) for d in dts]
            f = cute.compile(fn, *args, F32(1e-6), *consts, make_fake_stream(use_tvm_ffi_env_stream=False),
                             options="--enable-tvm-ffi")
            c[key] = f
        _FNS[key] = f
    return f


def _rows(N):
    rows = 1024
    while N % rows:
        rows //= 2
    assert rows % RB == 0, "N must be a multiple of 64 rows per chunk"
    return rows


_A_DTS = [cutlass.Int32, cutlass.Float32, cutlass.Int32, cutlass.Int32, cutlass.Int32, cutlass.Int32, cutlass.Int32,
          cutlass.Int32, cutlass.Float32, cutlass.Float32, cutlass.Float32, cutlass.Float32, cutlass.Int32,
          cutlass.Float32, None]
_K2_DTS = [cutlass.Int32, cutlass.Int32, cutlass.Int32, cutlass.Int32, cutlass.Float32, cutlass.Int32,
           cutlass.Float32, None]


def _key_a(B, N, H, G, D, R, sumonly, role=0):
    return ("a", B, N, H, G, D, R, _rows(N), bool(sumonly), int(role))


def _key_k2(B, N, H, G, D, R):
    return ("k2", B, N, H, G, D, R)


_BF_DTS = [cutlass.Int32] * 7 + [cutlass.Float32] * 2


def aot(B, N, H, G, R=64, D=256):
    """compile every launch for this shape (CPU-only; no CUDA context)."""
    _compile(("bf", B, N, H, G, D, R), _launch_bf, _BF_DTS, [B, N, H, G, D, R])
    _compile(_key_a(B, N, H, G, D, R, False), _launch_a, _A_DTS, [B, N, H, G, D, R, _rows(N), False, 0])
    for role in (0, 1, 2):
        _compile(_key_a(B, N, H, G, D, R, True, role), _launch_a, _A_DTS, [B, N, H, G, D, R, _rows(N), True, role])
    _compile(_key_k2(B, N, H, G, D, R), _launch_k2, _K2_DTS, [B, N, H, G, D, R])


def _ctr(n, dev):
    s = torch.cuda.current_stream(dev)
    k = (dev.index, s.cuda_stream)
    c = _CTR.get(k)
    if c is None or c.numel() < n:
        c = _CTR[k] = torch.zeros(max(n, 64), device=dev, dtype=torch.int32)
    return c


def _dense16(t):
    if not t.is_contiguous() or t.data_ptr() % 16 != 0:
        t = t.contiguous().clone()
    return t


def _check(y, gq, gk, cos, sin, H, G, D, R):
    if y.dim() != 3:
        raise ValueError(f"qkv producers: y must be [B, N, (2H+2G)D], got {tuple(y.shape)}")
    B, N, W = y.shape
    if W != (2 * H + 2 * G) * D or y.dtype != torch.bfloat16 or not y.is_contiguous() or y.data_ptr() % 16:
        raise ValueError("qkv producers: y must be bf16, contiguous, 16 B aligned, last dim (2H+2G)D")
    if not (D == 256 and R == 64 and N % 128 == 0):
        raise ValueError(f"qkv producers: D=256, R=64, N % 128 == 0 only (got D={D} R={R} N={N})")
    for name, t, shp in (("cos", cos, (B, N, R)), ("sin", sin, (B, N, R)), ("gq", gq, (D,)), ("gk", gk, (D,))):
        if tuple(t.shape) != shp or t.dtype != torch.bfloat16 or t.device != y.device:
            raise ValueError(f"qkv producers: {name} must be bf16 {shp}")
    return B, N


def _out(out, name, shape, dtype, dev):
    t = None if out is None else out.get(name)
    if t is None:
        return torch.empty(shape, device=dev, dtype=dtype)
    if not (tuple(t.shape) == tuple(shape) and t.dtype == dtype and t.is_contiguous() and t.data_ptr() % 16 == 0):
        raise ValueError(f"qkv producers: out[{name!r}] must be a contiguous 16 B aligned {dtype} {tuple(shape)}")
    return t


def _stage_a(y, gq, gk, cos, sin, H, G, R, eps, out, sumonly, role=0):
    D = 256
    B, N = _check(y, gq, gk, cos, sin, H, G, D, R)
    dev = y.device
    e4, f32 = torch.float8_e4m3fn, torch.float32
    q8 = _out(out, "q8", (B, N, H, D), e4, dev)
    qs = _out(out, "qs", (B, H, N // TR), f32, dev)
    v8 = _out(out, "v8", (B, N, G, D), e4, dev)
    vs = _out(out, "vs", (B, G, N // TR), f32, dev)
    kvw = _out(out, "kvw", (B, G, N // 64), torch.int32, dev)
    rq = _out(out, "rstd_q", (B, H, N), f32, dev)
    rk = _out(out, "rstd_k", (B, G, N), f32, dev)
    km = _out(out, "kmean", (B, G, D), f32, dev)
    kp = torch.empty(B * G * (N // RB) * D, device=dev, dtype=f32)
    ctr = _ctr(B * G, dev)
    f = _compile(_key_a(B, N, H, G, D, R, sumonly, role), _launch_a, _A_DTS,
                 [B, N, H, G, D, R, _rows(N), bool(sumonly), int(role)])
    i32 = torch.int32
    tens = [ctr, km, y.view(i32), _dense16(gq).view(i32), _dense16(gk).view(i32), _dense16(cos).view(i32),
            _dense16(sin).view(i32), q8.view(i32), qs, rq, kp, rk, v8.view(i32), vs]
    f(*[t.data_ptr() for t in tens], kvw.data_ptr(), float(eps), torch.cuda.current_stream(dev).cuda_stream)
    gate = y.unflatten(-1, (-1, D))[:, :, : 2 * H].unflatten(2, (H, 2))[:, :, :, 1]
    return dict(q8=q8, qs=qs, v8=v8, vs=vs, kvw=kvw, rstd_q=rq, rstd_k=rk, kmean=km, gate=gate,
                _keep=(gq, gk, cos, sin))


def k_quant_mean(y, gk, cos, sin, km, H, G, R=64, eps=1e-6, out=None):
    """K2: k_rot re-derived from y's k columns, minus km (fp32 (B,G,D)), MX codes / scales; K bytes into out['kvw'].
    -> (k8 (B,N,G,D), ks (B,G,N/32), kvw)."""
    D = 256
    B, N = _check(y, gk, gk, cos, sin, H, G, D, R)
    dev = y.device
    if tuple(km.shape) != (B, G, D) or km.dtype != torch.float32:
        raise ValueError("k_quant_mean: km must be fp32 (B, G, D)")
    k8 = _out(out, "k8", (B, N, G, D), torch.float8_e4m3fn, dev)
    ks = _out(out, "ks", (B, G, N // TR), torch.float32, dev)
    kvw = _out(out, "kvw", (B, G, N // 64), torch.int32, dev)
    f = _compile(_key_k2(B, N, H, G, D, R), _launch_k2, _K2_DTS, [B, N, H, G, D, R])
    i32 = torch.int32
    tens = [y.view(i32), _dense16(gk).view(i32), _dense16(cos).view(i32), _dense16(sin).view(i32), _dense16(km),
            k8.view(i32), ks]
    f(*[t.data_ptr() for t in tens], kvw.data_ptr(), float(eps), torch.cuda.current_stream(dev).cuda_stream)
    return k8, ks, kvw


def fwd_producers(y, gq, gk, cos, sin, H, G, R=64, eps=1e-6, out=None):
    """y bf16 [B, N, (2H+2G)256] contiguous (TE GEMM output: q_h|gate_h per head, k, v); gq, gk bf16 [256];
    cos, sin bf16 [B, N, R] contiguous.  Returns a dict: q8, qs, k8, ks, v8, vs, kvw, rstd_q, rstd_k, kmean, gate.
    Bitwise the unfused chain (TE rmsnorm -> partial RoPE -> quant.quant_qkv2) on the same y."""
    r = _stage_a(y, gq, gk, cos, sin, H, G, R, eps, out, False)
    k8, ks, _ = k_quant_mean(y, gk, cos, sin, r["kmean"], H, G, R, eps,
                             out=dict(k8=None if out is None else out.get("k8"),
                                      ks=None if out is None else out.get("ks"), kvw=r["kvw"]))
    r.update(k8=k8, ks=ks)
    return r


def fwd_producers_ksum(y, gq, gk, cos, sin, H, G, R=64, eps=1e-6, out=None, role=0):
    """Launch A with the K column SUM (B,G,D) in r['kmean'] instead of the mean (ring CP: all-reduce it, divide by the
    global N, then k_quant_mean).  The sum is quant_qkv2's tree without the final / N: per 64-row chunk T32(even rows)
    + T32(odd rows), ROWS-row partials ((0 + c0) + c1) + .., then ((0 + P0) + P1) + ...
    role 0 = K stats + V + Q (one launch); role 1 = K stats + V only (rstd_q / q8 / qs untouched); role 2 = Q only
    (kmean / v8 / vs / rstd_k / V bytes of kvw untouched): issue role 1, start the all-reduce, then role 2 overlaps it.
    Callers pass the same `out` dict to both role calls."""
    return _stage_a(y, gq, gk, cos, sin, H, G, R, eps, out, True, role)


def fwd_bf16(y, gq, gk, cos, sin, H, G, R=64, eps=1e-6):
    """q / k norm + partial RoPE, bf16 out: q_rot (B,N,H,D), k_rot (B,N,G,D), rstd_q (B,H,N), rstd_k (B,G,N).
    The same row math as fwd_producers (its pre-quant values)."""
    D = 256
    B, N = _check(y, gq, gk, cos, sin, H, G, D, R)
    dev = y.device
    qb = torch.empty(B, N, H, D, device=dev, dtype=torch.bfloat16)
    kb = torch.empty(B, N, G, D, device=dev, dtype=torch.bfloat16)
    rq = torch.empty(B, H, N, device=dev, dtype=torch.float32)
    rk = torch.empty(B, G, N, device=dev, dtype=torch.float32)
    f = _compile(("bf", B, N, H, G, D, R), _launch_bf, _BF_DTS, [B, N, H, G, D, R])
    i32 = torch.int32
    tens = [y.view(i32), _dense16(gq).view(i32), _dense16(gk).view(i32), _dense16(cos).view(i32),
            _dense16(sin).view(i32), qb.view(i32), kb.view(i32), rq, rk]
    f(*[t.data_ptr() for t in tens], float(eps), torch.cuda.current_stream(dev).cuda_stream)
    return qb, kb, rq, rk
