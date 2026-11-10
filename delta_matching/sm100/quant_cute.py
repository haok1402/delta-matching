"""CuTe DSL kernels of the sm100-mx producers (kda-sm100-cutequant; design + Triton inventory: kda/docs/cutequant-design.md).

Bitwise equal to the former Triton kernels of quant.py (_prod_a / _prod_b / _quant_do_delta / _quant_rows / _quant_qkv /
_kmean_partial).  Quantization (amax, E8M0 scale, e4m3 codes) is elementwise or an exact max, so it does not depend on the
thread mapping.  The fp32 reductions (K mean partials, K mean fold, stale delta) reproduce Triton's trees literally (same
grouping, same fma fusion, same shuffle order); every fp32 op of a reduction is inline PTX with an explicit .rn, so neither
LLVM nor ptxas can contract or reassociate it.  SIMT kernels: no TMA / tcgen05 / pipelines (no dm_utils fence wrappers).
D (128 or 256) is a compile-time parameter.
"""
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Float32, const_expr
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm
import cuda.bindings.driver as cuda

NT = 64            # threads per CTA (= Triton num_warps 2) for every kernel except KMeanPartial (128 = num_warps 4)
TR = 32            # rows per quant tile (one E8M0 block)


def _asm(ret, args, s, cons, side=False, loc=None, ip=None):
    return llvm.inline_asm(ret, args, s, cons, has_side_effects=side, is_align_stack=False,
                           asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


def _v(x, ty, loc, ip):
    return ty(x).ir_value(loc=loc, ip=ip)


# ---------------------------------------------------------------------------------------------------------- PTX helpers
@dsl_user_op
def ld_v4(addr: Int64, *, loc=None, ip=None):
    """16 B global load -> 4 x b32."""
    st = _asm(llvm.StructType.get_literal([T.i32()] * 4), [_v(addr, Int64, loc, ip)],
              "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,l", loc=loc, ip=ip)
    return [Int32(llvm.extractvalue(T.i32(), st, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def ld_v4f(addr: Int64, *, loc=None, ip=None):
    st = _asm(llvm.StructType.get_literal([T.f32()] * 4), [_v(addr, Int64, loc, ip)],
              "ld.global.v4.f32 {$0, $1, $2, $3}, [$4];", "=f,=f,=f,=f,l", loc=loc, ip=ip)
    return [Float32(llvm.extractvalue(T.f32(), st, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def ld_f32(addr: Int64, *, loc=None, ip=None) -> Float32:
    return Float32(_asm(T.f32(), [_v(addr, Int64, loc, ip)], "ld.global.f32 $0, [$1];", "=f,l", loc=loc, ip=ip))


@dsl_user_op
def ld_bf16f(addr: Int64, *, loc=None, ip=None) -> Float32:
    """One bf16 (2 B aligned) -> f32 (cvt.f32.bf16, as Triton)."""
    return Float32(_asm(T.f32(), [_v(addr, Int64, loc, ip)],
                        "{\n\t.reg .b16 h;\n\tld.global.b16 h, [$1];\n\tcvt.f32.bf16 $0, h;\n\t}", "=f,l", loc=loc, ip=ip))


@dsl_user_op
def bf2(w: Int32, *, loc=None, ip=None):
    """b32 holding 2 bf16 (lo = lower address) -> (f32 lo, f32 hi) via cvt.f32.bf16 (as Triton)."""
    st = _asm(llvm.StructType.get_literal([T.f32()] * 2), [_v(w, Int32, loc, ip)],
              "{\n\t.reg .b16 l, h;\n\tmov.b32 {l, h}, $2;\n\tcvt.f32.bf16 $0, l;\n\tcvt.f32.bf16 $1, h;\n\t}",
              "=f,=f,r", loc=loc, ip=ip)
    return [Float32(llvm.extractvalue(T.f32(), st, [i], loc=loc, ip=ip)) for i in range(2)]


def _f2(op, cons="=f,f,f"):
    @dsl_user_op
    def f(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
        return Float32(_asm(T.f32(), [_v(a, Float32, loc, ip), _v(b, Float32, loc, ip)], op + " $0, $1, $2;", cons,
                            loc=loc, ip=ip))
    return f


fadd = _f2("add.rn.f32")
fsub = _f2("sub.rn.f32")
fmul = _f2("mul.rn.f32")
fmax = _f2("max.f32")
fdivfull = _f2("div.full.f32")


@dsl_user_op
def ffma(a: Float32, b: Float32, c: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(_asm(T.f32(), [_v(x, Float32, loc, ip) for x in (a, b, c)], "fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f",
                        loc=loc, ip=ip))


@dsl_user_op
def fabs(a: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(_asm(T.f32(), [_v(a, Float32, loc, ip)], "abs.f32 $0, $1;", "=f,f", loc=loc, ip=ip))


@dsl_user_op
def i2f(n: Int32, *, loc=None, ip=None) -> Float32:
    return Float32(_asm(T.f32(), [_v(n, Int32, loc, ip)], "cvt.rn.f32.s32 $0, $1;", "=f,r", loc=loc, ip=ip))


@dsl_user_op
def shfl_bfly(x: Float32, off: int, *, loc=None, ip=None) -> Float32:
    return Float32(_asm(T.f32(), [_v(x, Float32, loc, ip)], f"shfl.sync.bfly.b32 $0, $1, {off}, 31, -1;", "=f,f",
                        side=True, loc=loc, ip=ip))


@dsl_user_op
def e8m0(amax: Float32, *, loc=None, ip=None) -> Int32:
    """eb = ((bits >> 23) & 0xFF) - 8 + ((bits & 0x7FFFFF) > 0x600000), clamped to [1, 254]; 127 unless amax > 0."""
    return Int32(_asm(T.i32(), [_v(amax, Float32, loc, ip)],
                      "{\n\t.reg .b32 a, m, e;\n\t.reg .pred p, q;\n\t"
                      "mov.b32 a, $1;\n\tshr.u32 e, a, 23;\n\tand.b32 e, e, 255;\n\tsub.s32 e, e, 8;\n\t"
                      "and.b32 m, a, 8388607;\n\tsetp.gt.u32 p, m, 6291456;\n\tselp.s32 m, 1, 0, p;\n\tadd.s32 e, e, m;\n\t"
                      "max.s32 e, e, 1;\n\tmin.s32 e, e, 254;\n\t"
                      "setp.gt.f32 q, $1, 0f00000000;\n\tselp.s32 $0, e, 127, q;\n\t}",
                      "=r,f", loc=loc, ip=ip))


@dsl_user_op
def shl23f(e: Int32, *, loc=None, ip=None) -> Float32:
    return Float32(_asm(T.f32(), [_v(e, Int32, loc, ip)], "shl.b32 $0, $1, 23;", "=f,r", loc=loc, ip=ip))


@dsl_user_op
def e4m3x4(a: Float32, b: Float32, c: Float32, d: Float32, *, loc=None, ip=None) -> Int32:
    """4 fp32 -> 4 e4m3 bytes (byte 0 = a), cvt.rn.satfinite.e4m3x2.f32 (as Triton)."""
    return Int32(_asm(T.i32(), [_v(x, Float32, loc, ip) for x in (a, b, c, d)],
                      "{\n\t.reg .b16 l, h;\n\tcvt.rn.satfinite.e4m3x2.f32 l, $2, $1;\n\t"
                      "cvt.rn.satfinite.e4m3x2.f32 h, $4, $3;\n\tmov.b32 $0, {l, h};\n\t}", "=r,f,f,f,f", loc=loc, ip=ip))


@dsl_user_op
def e4m3x4_deq(w: Int32, *, loc=None, ip=None):
    """4 e4m3 bytes -> 4 f32 through f16 (cvt.rn.f16x2.e4m3x2, cvt.f32.f16; as Triton's y.to(float32))."""
    st = _asm(llvm.StructType.get_literal([T.f32()] * 4), [_v(w, Int32, loc, ip)],
              "{\n\t.reg .b16 l, h, a, b, c, d;\n\t.reg .b32 x, y;\n\tmov.b32 {l, h}, $4;\n\t"
              "cvt.rn.f16x2.e4m3x2 x, l;\n\tcvt.rn.f16x2.e4m3x2 y, h;\n\tmov.b32 {a, b}, x;\n\tmov.b32 {c, d}, y;\n\t"
              "cvt.f32.f16 $0, a;\n\tcvt.f32.f16 $1, b;\n\tcvt.f32.f16 $2, c;\n\tcvt.f32.f16 $3, d;\n\t}",
              "=f,=f,=f,=f,r", loc=loc, ip=ip)
    return [Float32(llvm.extractvalue(T.f32(), st, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def st_v4(addr: Int64, w0: Int32, w1: Int32, w2: Int32, w3: Int32, *, loc=None, ip=None):
    _asm(None, [_v(addr, Int64, loc, ip)] + [_v(x, Int32, loc, ip) for x in (w0, w1, w2, w3)],
         "st.global.v4.b32 [$0], {$1, $2, $3, $4};", "l,r,r,r,r", side=True, loc=loc, ip=ip)


@dsl_user_op
def st_b32(addr: Int64, w: Int32, *, loc=None, ip=None):
    _asm(None, [_v(addr, Int64, loc, ip), _v(w, Int32, loc, ip)], "st.global.b32 [$0], $1;", "l,r", side=True,
         loc=loc, ip=ip)


@dsl_user_op
def st_f32_if(addr: Int64, v: Float32, pred: Int32, *, loc=None, ip=None):
    _asm(None, [_v(addr, Int64, loc, ip), _v(v, Float32, loc, ip), _v(pred, Int32, loc, ip)],
         "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $2, 0;\n\t@p st.global.f32 [$0], $1;\n\t}", "l,f,r", side=True,
         loc=loc, ip=ip)


@dsl_user_op
def st_v4f_if(addr: Int64, a: Float32, b: Float32, c: Float32, d: Float32, pred: Int32, *, loc=None, ip=None):
    _asm(None, [_v(addr, Int64, loc, ip)] + [_v(x, Float32, loc, ip) for x in (a, b, c, d)] + [_v(pred, Int32, loc, ip)],
         "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $5, 0;\n\t@p st.global.v4.f32 [$0], {$1, $2, $3, $4};\n\t}",
         "l,f,f,f,f,r", side=True, loc=loc, ip=ip)


@dsl_user_op
def st_u8_if(addr: Int64, v: Int32, pred: Int32, *, loc=None, ip=None):
    _asm(None, [_v(addr, Int64, loc, ip), _v(v, Int32, loc, ip), _v(pred, Int32, loc, ip)],
         "{\n\t.reg .pred p;\n\t.reg .b16 h;\n\tsetp.ne.s32 p, $2, 0;\n\tcvt.u16.u32 h, $1;\n\t@p st.global.b8 [$0], h;\n\t}",
         "l,r,r", side=True, loc=loc, ip=ip)


@dsl_user_op
def sts_f32_if(addr: Int32, v: Float32, pred: Int32, *, loc=None, ip=None):
    _asm(None, [_v(addr, Int32, loc, ip), _v(v, Float32, loc, ip), _v(pred, Int32, loc, ip)],
         "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $2, 0;\n\t@p st.shared.f32 [$0], $1;\n\t}", "r,f,r", side=True,
         loc=loc, ip=ip)


@dsl_user_op
def lds_f32(addr: Int32, *, loc=None, ip=None) -> Float32:
    return Float32(_asm(T.f32(), [_v(addr, Int32, loc, ip)], "ld.shared.f32 $0, [$1];", "=f,r", side=True,
                        loc=loc, ip=ip))


@dsl_user_op
def bar_sync(*, loc=None, ip=None):
    _asm(None, [], "bar.sync 0;", "", side=True, loc=loc, ip=ip)


@dsl_user_op
def smem_base(nbytes: int, *, loc=None, ip=None) -> Int32:
    """32-bit shared address of a static smem buffer (CTA-private)."""
    from cutlass.utils import SmemAllocator
    sm = SmemAllocator()
    t = sm.allocate_tensor(Float32, cute.make_layout(nbytes // 4), byte_alignment=16)
    return Int32(t.iterator.toint())


# ------------------------------------------------------------------------------------------------------- trace helpers
def tree(v):
    """Balanced pairwise tree, level by level: ((v0+v1)+(v2+v3))+... (len a power of 2)."""
    while len(v) > 1:
        v = [fadd(v[2 * i], v[2 * i + 1]) for i in range(len(v) // 2)]
    return v[0]


def load_row16(addr, vec):
    """16 consecutive bf16 at byte address addr -> 16 f32."""
    if vec:
        w = ld_v4(addr) + ld_v4(addr + Int64(16))
        out = []
        for x in w:
            out += bf2(x)
        return out
    return [ld_bf16f(addr + Int64(2 * j)) for j in range(16)]


def load_row16_f32(addr, vec):
    if vec:
        out = []
        for k in range(4):
            out += ld_v4f(addr + Int64(16 * k))
        return out
    return [ld_f32(addr + Int64(4 * j)) for j in range(16)]


def load_row8(addr, vec):
    if vec:
        out = []
        for x in ld_v4(addr):
            out += bf2(x)
        return out
    return [ld_bf16f(addr + Int64(2 * j)) for j in range(8)]


def row_delta(x, o):
    """Triton's in-thread tree of sum_j x_j * o_j over 16 columns (cutequant-design R3)."""
    p = {j: fmul(x[j], o[j]) for j in (2, 3, 6, 7, 10, 11, 14, 15)}
    ae = [ffma(x[4 * k], o[4 * k], p[4 * k + 2]) for k in range(4)]
    ao = [ffma(x[4 * k + 1], o[4 * k + 1], p[4 * k + 3]) for k in range(4)]
    e = fadd(fadd(ae[0], ae[1]), fadd(ae[2], ae[3]))
    od = fadd(fadd(ao[0], ao[1]), fadd(ao[2], ao[3]))
    return fadd(e, od)


class TileGeo:
    """32-row x D quant tile on 64 threads, Triton layout spt[1,16] tpw[32/(D/16), D/16] wpc[2,1]."""

    def __init__(self, D, tid):
        self.D = D
        self.tpw1 = D // 16
        self.tpw0 = 32 // self.tpw1
        self.rpt = self.tpw0 * 2
        self.reps = TR // self.rpt
        self.w = tid // 32
        self.lane = tid % 32
        self.l0 = self.lane // self.tpw1
        self.l1 = self.lane % self.tpw1
        self.c0 = self.l1 * 16

    def row(self, rep):
        return self.w * self.tpw0 + self.l0 + rep * self.rpt


def quant_tile(g: TileGeo, xs, red, n_tok, h_out, nh_out, bh, row0, mY, mS, mWB, w_off, b):
    """xs[rep] = 16 f32 (already centred / prescaled). Stores codes, the scale and (w_off not None) the UE8M0 byte.
    Returns (scale, codes words per rep). red: smem address of 2 floats (cross-warp amax)."""
    m = fabs(xs[0][0])
    for r in xs:
        for v in r:
            m = fmax(m, fabs(v))
    for off in (16, 8, 4, 2, 1):
        m = fmax(m, shfl_bfly(m, off))
    sts_f32_if(red + g.w * 4, m, Int32(g.lane == 0))
    bar_sync()
    m = fmax(lds_f32(red), lds_f32(red + 4))
    eb = e8m0(m)
    scale = shl23f(eb)
    inv = shl23f(Int32(254) - eb)
    blk = row0 // TR
    words = []
    for rep in range(g.reps):
        y = [fmul(v, inv) for v in xs[rep]]
        wds = [e4m3x4(y[4 * k], y[4 * k + 1], y[4 * k + 2], y[4 * k + 3]) for k in range(4)]
        words.append(wds)
        tok = Int64(row0 + g.row(rep))
        addr = mY + ((Int64(b) * Int64(n_tok) + tok) * Int64(nh_out) + Int64(h_out)) * Int64(g.D) + Int64(g.c0)
        st_v4(addr, *wds)
    first = Int32(g.w == 0) * Int32(g.lane == 0)
    st_f32_if(mS + (Int64(bh) * Int64(n_tok // TR) + Int64(blk)) * 4, scale, first)
    if w_off is not None:
        baddr = mWB + (Int64(bh) * Int64(n_tok // 64) + Int64(2 * (blk // 4) + w_off)) * 4 + Int64(blk % 4)
        st_u8_if(baddr, eb, first)
    return scale, words


def load_tile(g: TileGeo, mX, b, h, row0, sb, sn, sh, vec, f32=False):
    xs = []
    for rep in range(g.reps):
        off = Int64(b) * Int64(sb) + Int64(h) * Int64(sh) + Int64(row0 + g.row(rep)) * Int64(sn) + Int64(g.c0)
        xs.append(load_row16_f32(mX + off * 4, vec) if f32 else load_row16(mX + off * 2, vec))
    return xs


class ChunkGeo:
    """K-mean partial chunk (cutequant-design R1/R4): `rows` rows x D columns on nw warps.
    Lanes l1 own 8 columns each (D/8 lanes); G = groups of rows = 32*nw / (D/8); a group g = w * tpw0 + l0 holds
    reps = rows / G rows: Triton-ALIGNED row r = rep * G + g, UNALIGNED (one thread per column, contiguous) r = g * reps + rep."""

    def __init__(self, D, nw, rows, tid, aligned):
        self.D, self.nw, self.rows, self.aligned = D, nw, rows, aligned
        self.tpw1 = D // 8
        self.tpw0 = 32 // self.tpw1
        self.G = self.tpw0 * nw
        self.reps = rows // self.G
        self.w = tid // 32
        self.lane = tid % 32
        self.l0 = self.lane // self.tpw1
        self.l1 = self.lane % self.tpw1
        self.c0 = self.l1 * 8
        self.g = self.w * self.tpw0 + self.l0

    def row(self, rep):
        return rep * self.G + self.g if self.aligned else self.g * self.reps + rep


def chunk_colsum(cg: ChunkGeo, mK, b, h, row0, sb, sn, sh, vec, red, warp_order):
    """Column sums of one chunk in Triton's tree; returns 8 f32 (valid in warp 0, lanes with l0 == 0).
    red: smem address of nw * D floats."""
    base = Int64(b) * Int64(sb) + Int64(h) * Int64(sh) + Int64(cg.c0)
    vals = [load_row8(mK + (base + Int64(row0 + cg.row(rep)) * Int64(sn)) * 2, vec) for rep in range(cg.reps)]
    s = [tree([vals[rep][j] for rep in range(cg.reps)]) for j in range(8)]
    if cg.tpw0 == 2:
        s = [fadd(v, shfl_bfly(v, 16)) for v in s]
    if cg.nw == 1:
        return s
    # cross-warp: every warp's lanes with l0 == 0 publish their 8 column sums, warp 0 combines in warp_order
    pub = Int32(cg.l0 == 0)
    for j in range(8):
        sts_f32_if(red + (cg.w * cg.D + cg.c0 + j) * 4, s[j], pub)
    bar_sync()
    out = []
    for j in range(8):
        ws = [lds_f32(red + (wi * cg.D + cg.c0 + j) * 4) for wi in range(cg.nw)]
        out.append(warp_order(ws))
    bar_sync()
    return out


def warps2(ws):
    return fadd(ws[0], ws[1])


# ---------------------------------------------------------------------------------------------------------------- kernels
class ProdA:
    """= _prod_a_kernel: 1-D grid [K-mean partials (ROWS rows) | Q tiles | V tiles], 64 threads."""

    def __init__(self, D, vq, vk, vv, kal):
        assert D in (128, 256)
        self.D, self.vq, self.vk, self.vv, self.kal = D, vq, vk, vv, kal

    @cute.jit
    def __call__(self, mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor, mQ8: cute.Tensor, mV8: cute.Tensor,
                 mQS: cute.Tensor, mVS: cute.Tensor, mWB: cute.Tensor, mP: cute.Tensor,
                 N: Int32, H: Int32, Hk: Int32, B: Int32, sqb: Int32, sqn: Int32, sqh: Int32, skb: Int32, skn: Int32,
                 skh: Int32, svb: Int32, svn: Int32, svh: Int32, sm: Float32, ROWS: Int32, stream: cuda.CUstream):
        ntl = N // TR
        total = B * H * ntl + B * Hk * ntl + B * Hk * (N // ROWS)
        self.kernel(mQ, mK, mV, mQ8, mV8, mQS, mVS, mWB, mP, N, H, Hk, B, sqb, sqn, sqh, skb, skn, skh, svb, svn, svh,
                    sm, ROWS).launch(grid=[total, 1, 1], block=[NT, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor, mQ8: cute.Tensor, mV8: cute.Tensor,
               mQS: cute.Tensor, mVS: cute.Tensor, mWB: cute.Tensor, mP: cute.Tensor,
               N: Int32, H: Int32, Hk: Int32, B: Int32, sqb: Int32, sqn: Int32, sqh: Int32, skb: Int32, skn: Int32,
               skh: Int32, svb: Int32, svn: Int32, svh: Int32, sm: Float32, ROWS: Int32):
        D = self.D
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        red = smem_base(2 * D * 4)
        ntl = N // TR
        nq = B * H * ntl
        nv = B * Hk * ntl
        S = N // ROWS
        npart = B * Hk * S
        pid = bid - npart
        if bid < npart:
            pid = pid + nq + nv + npart
        if pid < nq:
            g = TileGeo(D, tid)
            bh = pid // ntl
            b = bh // H
            h = bh % H
            row0 = (pid % ntl) * TR
            xs = load_tile(g, mQ.iterator.toint(), b, h, row0, sqb, sqn, sqh, self.vq)
            xs = [[fmul(v, sm) for v in r] for r in xs]
            quant_tile(g, xs, red, N, h, H, bh, row0, mQ8.iterator.toint(), mQS.iterator.toint(), None, None, b)
        elif pid < nq + nv:
            g = TileGeo(D, tid)
            p = pid - nq
            bh = p // ntl
            b = bh // Hk
            h = bh % Hk
            row0 = (p % ntl) * TR
            xs = load_tile(g, mV.iterator.toint(), b, h, row0, svb, svn, svh, self.vv)
            quant_tile(g, xs, red, N, h, Hk, bh, row0, mV8.iterator.toint(), mVS.iterator.toint(),
                       mWB.iterator.toint(), 1, b)
        else:
            p = pid - nq - nv
            bh = p // S
            s = p % S
            b = bh // Hk
            h = bh % Hk
            cg = ChunkGeo(D, 2, 64, tid, self.kal)
            acc = cute.make_rmem_tensor(8, Float32)
            for j in cutlass.range_constexpr(8):
                acc[j] = Float32(0.0)
            for ci in cutlass.range(ROWS // 64, unroll=1):
                tot = chunk_colsum(cg, mK.iterator.toint(), b, h, s * ROWS + ci * 64, skb, skn, skh, self.vk, red,
                                   warps2)
                for j in cutlass.range_constexpr(8):
                    acc[j] = fadd(acc[j], tot[j])
            pa = mP.iterator.toint() + ((Int64(bh) * Int64(S) + Int64(s)) * Int64(D) + Int64(cg.c0)) * 4
            ok = Int32(cg.w == 0) * Int32(cg.l0 == 0)
            st_v4f_if(pa, acc[0], acc[1], acc[2], acc[3], ok)
            st_v4f_if(pa + Int64(16), acc[4], acc[5], acc[6], acc[7], ok)


class ProdB:
    """= _prod_b_kernel: grid (N/32, B*Hk): mu = div.full(sequential sum of the S partials, N); K tile codes."""

    def __init__(self, D, vk):
        assert D in (128, 256)
        self.D, self.vk = D, vk

    @cute.jit
    def __call__(self, mK: cute.Tensor, mK8: cute.Tensor, mKS: cute.Tensor, mWB: cute.Tensor, mP: cute.Tensor,
                 mM: cute.Tensor, N: Int32, Hk: Int32, B: Int32, skb: Int32, skn: Int32, skh: Int32, ROWS: Int32,
                 stream: cuda.CUstream):
        self.kernel(mK, mK8, mKS, mWB, mP, mM, N, Hk, skb, skn, skh, ROWS).launch(
            grid=[N // TR, B * Hk, 1], block=[NT, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mK: cute.Tensor, mK8: cute.Tensor, mKS: cute.Tensor, mWB: cute.Tensor, mP: cute.Tensor,
               mM: cute.Tensor, N: Int32, Hk: Int32, skb: Int32, skn: Int32, skh: Int32, ROWS: Int32):
        D = self.D
        tid, _, _ = cute.arch.thread_idx()
        pid_n, bh, _ = cute.arch.block_idx()
        red = smem_base(2 * 4)
        mus = smem_base(D * 4)
        S = N // ROWS
        nf = i2f(N)
        pbase = mP.iterator.toint()
        for jj in cutlass.range_constexpr(D // NT):
            c = tid + jj * NT
            acc = Float32(0.0)
            for s in cutlass.range(S, unroll=1):
                acc = fadd(acc, ld_f32(pbase + ((Int64(bh) * Int64(S) + Int64(s)) * Int64(D) + Int64(c)) * 4))
            mu = fdivfull(acc, nf)
            sts_f32_if(mus + c * 4, mu, Int32(1))
            st_f32_if(mM.iterator.toint() + (Int64(bh) * Int64(D) + Int64(c)) * 4, mu, Int32(pid_n == 0))
        bar_sync()
        g = TileGeo(D, tid)
        b = bh // Hk
        h = bh % Hk
        row0 = pid_n * TR
        xs = load_tile(g, mK.iterator.toint(), b, h, row0, skb, skn, skh, self.vk)
        mu16 = [lds_f32(mus + (g.c0 + j) * 4) for j in range(16)]
        xs = [[fsub(r[j], mu16[j]) for j in range(16)] for r in xs]
        quant_tile(g, xs, red, N, h, Hk, bh, row0, mK8.iterator.toint(), mKS.iterator.toint(), mWB.iterator.toint(),
                   0, b)


class Tile:
    """Generic 32-row quant tile, grid (N/32, B*H): = _quant_do_delta (delta None / "stale" / "deq") and, with a mean /
    prescale / UE8M0 byte, = _quant_rows and the per-tensor tiles of _quant_qkv."""

    def __init__(self, D, vx, has_mean=False, has_pre=False, w_off=None, delta=None, vo=False, xf32=False):
        assert D in (128, 256) and delta in (None, "stale", "deq")
        self.D, self.vx, self.has_mean, self.has_pre, self.w_off, self.delta, self.vo, self.xf32 = (
            D, vx, has_mean, has_pre, w_off, delta, vo, xf32)

    @cute.jit
    def __call__(self, mX: cute.Tensor, mO: cute.Tensor, mMean: cute.Tensor, mY: cute.Tensor, mS: cute.Tensor,
                 mDL: cute.Tensor, mWB: cute.Tensor, N: Int32, H: Int32, B: Int32, sxb: Int32, sxn: Int32, sxh: Int32,
                 sob: Int32, son: Int32, soh: Int32, pre: Float32, stream: cuda.CUstream):
        self.kernel(mX, mO, mMean, mY, mS, mDL, mWB, N, H, sxb, sxn, sxh, sob, son, soh, pre).launch(
            grid=[N // TR, B * H, 1], block=[NT, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mX: cute.Tensor, mO: cute.Tensor, mMean: cute.Tensor, mY: cute.Tensor, mS: cute.Tensor,
               mDL: cute.Tensor, mWB: cute.Tensor, N: Int32, H: Int32, sxb: Int32, sxn: Int32, sxh: Int32,
               sob: Int32, son: Int32, soh: Int32, pre: Float32):
        D = self.D
        tid, _, _ = cute.arch.thread_idx()
        pid_n, bh, _ = cute.arch.block_idx()
        red = smem_base(2 * 4)
        g = TileGeo(D, tid)
        b = bh // H
        h = bh % H
        row0 = pid_n * TR
        xs = load_tile(g, mX.iterator.toint(), b, h, row0, sxb, sxn, sxh, self.vx, self.xf32)
        xs = center_scale(xs, mMean.iterator.toint(), bh, D, g, self.has_mean, pre if const_expr(self.has_pre) else None)
        mWBa = mWB.iterator.toint() if const_expr(self.w_off is not None) else None
        os_ = load_o(g, mO.iterator.toint(), b, h, row0, sob, son, soh, self.vo, self.delta)
        scale, words = quant_tile(g, xs, red, N, h, H, bh, row0, mY.iterator.toint(), mS.iterator.toint(), mWBa,
                                  self.w_off, b)
        if const_expr(self.delta is not None):
            tile_delta(g, xs, os_, scale, words, self.delta, mDL.iterator.toint(), bh, N, row0)


def load_o(g, ob, b, h, row0, sob, son, soh, vo, delta):
    if delta is None:
        return None
    os_ = []
    for rep in range(g.reps):
        off = Int64(b) * Int64(sob) + Int64(h) * Int64(soh) + Int64(row0 + g.row(rep)) * Int64(son) + Int64(g.c0)
        os_.append(load_row16(ob + off * 2, vo) if delta == "stale" else load_row16_f32(ob + off * 4, vo))
    return os_


def tile_delta(g, xs, os_, scale, words, delta, dl, bh, N, row0):
    """Per row: R3 in-thread tree, then shfl.bfly over the D/16 column lanes (largest offset first); lane l1 == 0 stores."""
    for rep in range(g.reps):
        if delta == "stale":
            xv = xs[rep]
        else:
            dq = []
            for wd in words[rep]:
                dq += e4m3x4_deq(wd)
            xv = [fmul(scale, v) for v in dq]
        t = row_delta(xv, os_[rep])
        off = g.tpw1 // 2
        while off >= 1:
            t = fadd(t, shfl_bfly(t, off))
            off //= 2
        st_f32_if(dl + (Int64(bh) * Int64(N) + Int64(row0 + g.row(rep))) * 4, t, Int32(g.l1 == 0))


def center_scale(xs, ma, bh, D, g, has_mean, pre):
    if has_mean:
        ma = ma + (Int64(bh) * Int64(D) + Int64(g.c0)) * 4
        mu = [ld_f32(ma + Int64(4 * j)) for j in range(16)]
        xs = [[fsub(r[j], mu[j]) for j in range(16)] for r in xs]
    if pre is not None:
        xs = [[fmul(v, pre) for v in r] for r in xs]
    return xs


def warps4_02_13(ws):
    """Triton's 4-warp smem + shfl.bfly 2, 1 combine (cutequant-design R4): (w0 + w2) + (w1 + w3)."""
    return fadd(fadd(ws[0], ws[2]), fadd(ws[1], ws[3]))


class KMeanPartial:
    """= _kmean_partial_kernel (kmean_fast; not on the shipped path): grid (B*Hk, N/ROWS), 128 threads, 32-row chunks."""

    def __init__(self, D, vk, kal):
        self.D, self.vk, self.kal = D, vk, kal

    @cute.jit
    def __call__(self, mK: cute.Tensor, mP: cute.Tensor, N: Int32, Hk: Int32, B: Int32, skb: Int32, skn: Int32,
                 skh: Int32, ROWS: Int32, stream: cuda.CUstream):
        self.kernel(mK, mP, N, Hk, skb, skn, skh, ROWS).launch(grid=[B * Hk, N // ROWS, 1], block=[128, 1, 1],
                                                               stream=stream)

    @cute.kernel
    def kernel(self, mK: cute.Tensor, mP: cute.Tensor, N: Int32, Hk: Int32, skb: Int32, skn: Int32, skh: Int32,
               ROWS: Int32):
        D = self.D
        tid, _, _ = cute.arch.thread_idx()
        bh, s, _ = cute.arch.block_idx()
        red = smem_base(4 * D * 4)
        S = N // ROWS
        b = bh // Hk
        h = bh % Hk
        if const_expr(self.kal):
            cg = ChunkGeo(D, 4, 32, tid, True)
            acc = cute.make_rmem_tensor(8, Float32)
            for j in cutlass.range_constexpr(8):
                acc[j] = Float32(0.0)
            for ci in cutlass.range(ROWS // 32, unroll=1):
                tot = chunk_colsum(cg, mK.iterator.toint(), b, h, s * ROWS + ci * 32, skb, skn, skh, self.vk, red,
                                   warps4_02_13)
                for j in cutlass.range_constexpr(8):
                    acc[j] = fadd(acc[j], tot[j])
            pa = mP.iterator.toint() + ((Int64(bh) * Int64(S) + Int64(s)) * Int64(D) + Int64(cg.c0)) * 4
            ok = Int32(cg.w == 0) * Int32(cg.l0 == 0)
            st_v4f_if(pa, acc[0], acc[1], acc[2], acc[3], ok)
            st_v4f_if(pa + Int64(16), acc[4], acc[5], acc[6], acc[7], ok)
        else:
            # Triton UNALIGNED layout spt[1,1] tpw[1,32] wpc[1,4]: whole columns per thread (tid + 128 j)
            acc = cute.make_rmem_tensor(D // 128, Float32)
            for j in cutlass.range_constexpr(D // 128):
                acc[j] = Float32(0.0)
            for ci in cutlass.range(ROWS // 32, unroll=1):
                tot = kmp_u_colsum(mK.iterator.toint(), b, h, s * ROWS + ci * 32, skb, skn, skh, tid, D)
                for j in cutlass.range_constexpr(D // 128):
                    acc[j] = fadd(acc[j], tot[j])
            pa = mP.iterator.toint() + (Int64(bh) * Int64(S) + Int64(s)) * Int64(D) * 4
            for j in cutlass.range_constexpr(D // 128):
                st_f32_if(pa + Int64(tid + 128 * j) * 4, acc[j], Int32(1))


def kmp_u_colsum(mK, b, h, row0, sb, sn, sh, tid, D):
    """UNALIGNED _kmean_partial chunk (32 rows): D=128 (1 column / thread) Triton packs ROW pairs into f32x2 ->
    T16(even rows) + T16(odd rows) (kmp_D128_Hk4_pad8.ptx:423-524); D=256 (2 columns / thread) packs column pairs ->
    plain T32 per column."""
    out = []
    for j in range(D // 128):
        base = Int64(b) * Int64(sb) + Int64(h) * Int64(sh) + Int64(tid + 128 * j)
        v = [ld_bf16f(mK + (base + Int64(row0 + r) * Int64(sn)) * 2) for r in range(32)]
        if D == 128:
            out.append(fadd(tree(v[0::2]), tree(v[1::2])))
        else:
            out.append(tree(v))
    return out


def warps4_01_23(ws):
    """UNALIGNED (contiguous row groups): T32 = (T8(w0) + T8(w1)) + (T8(w2) + T8(w3))."""
    return fadd(fadd(ws[0], ws[1]), fadd(ws[2], ws[3]))
