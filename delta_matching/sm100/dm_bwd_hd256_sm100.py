"""hd256 backward ([64,32] fp32 contract, "Delta-Matching-256 v1" floor; fork hd256-bwd, bring-up B3+D2 "workspace" design).
hd128 does not import this file.  Design + exactness argument: kda/hd256/bwd-design.md.

Two kernel classes (all 1-CTA, tcgen05 kind::f8f6f4, no atomics, fixed order -> bitwise deterministic):

DmHd256DS  ("kernel A", K-outer): CTA = (128-key tile kt, q head h, batch b); m ascending.
  S_acc = Q8 K8^T, dP_acc = dO8 V8^T: M=128 rows, N=128 keys, 8 K=32 steps ascending d, zero-init (the same instruction
  and k order as pass 0 P1).  Per element, with the dm_common_hd256 helpers (the pass-0 <-> bwd bitwise contract):
     P_hat = dm_p0h_phat_pair(S_acc (causal -inf on raw acc), dsc, l2)
     dS'   = P_hat * fma(dP_acc, (do8s*v8s)*k8s, -(delta*k8s))          (spec §4.1)
     P~8   = e4m3_satfinite(P_hat)                                        (constant 2^8 basis)
  psi per 64x64 tile with F2 / F2b and the per-n-block g_max chain over ascending m-blocks (spec §4.2, sm90 hpp:1951-2009,
  2112-2128), dS'8 = e4m3_satfinite(dS' * psi_inv).  Writes P~8^T, dS'8, dS'8^T, psi, g to a dense workspace.
DmHd256Fold ("kernel B", generic token contraction): out[128 rows, 256 d] over 64-token chunks c (ascending), per chunk
  temp = A8[128 x 64] B8[256 x 64]^T (zero-init TMEM temp), then per thread / element the floor's site-exact fold:
     mode "dk": acc = fma(temp, g_c, acc)                                   (plan-A, hpp:2216-2219)
     mode "dv": win = win * clamp(f_prev / f_c) ; win = win + temp ; f_prev = f_c ; 512-row promote out = fma(win, f_prev, out)
                (hpp:2101-2105, 2230-2237; f_c = (1/256) * do8s[c])
     mode "dq": dominance skip, win = (psi_w ? win * clamp(psi_w / psi_c) : 0) + temp, 8-chunk promote out = fma(win, psi_w, out)
                (fq8_dq_pass2.h:262-345)
  post: dk * (1/k8s) (div.approx), dq * 2^-12, dv * 1; bf16 RN.  GQA: the q heads of a kv group in ascending order inside
  one CTA, fp32 across heads (one bf16 rounding); head chunks carry an fp32 accumulator in global memory.
"""
import math

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Float32, Int32, Int64, const_expr
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

from delta_matching.sm100 import dm_common_hd256 as dmh

F8 = cutlass.Float8E4M3FN
INV448 = float.fromhex("0x1.24924ap-9")      # fp32 RN(1/448) (= C++ 1.0f/448.0f)
F2B_C = float.fromhex("0x1.5798eep-19")      # fp32 RN(256.0f * 1e-8f): 1e-8f = 0x1.5798eep-27 exact x 2^8
assert abs(INV448 * 448 - 1) < 1e-7 and abs(F2B_C / 256e-8 - 1) < 1e-7


# ---------------------------------------------------------------- PTX helpers (local; hd128's dm_utils untouched)
def _asm(ret, args, asm, cons, side=False, loc=None, ip=None):
    return llvm.inline_asm(ret, args, asm, cons, has_side_effects=side, is_align_stack=False,
                           asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def tmem_ld32(addr: Int32, *, loc=None, ip=None):
    """tcgen05.ld.32x32b.x32: thread t of warp w gets TMEM lane (addr.lane + t), columns addr.col .. +31."""
    regs = ",".join(f"${i}" for i in range(32))
    res = _asm(llvm.StructType.get_literal([T.f32()] * 32), [Int32(addr).ir_value(loc=loc, ip=ip)],
               "tcgen05.ld.sync.aligned.32x32b.x32.b32 {" + regs + "}, [$32];", ",".join(["=f"] * 32) + ",r", True,
               loc, ip)
    return [Float32(llvm.extractvalue(T.f32(), res, [i], loc=loc, ip=ip)) for i in range(32)]


@dsl_user_op
def fma_f32(a: Float32, b: Float32, c: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(_asm(T.f32(), [Float32(x).ir_value(loc=loc, ip=ip) for x in (a, b, c)],
                        "fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f", False, loc, ip))


@dsl_user_op
def mul_f32(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """FMUL RN, never contracted with a following add."""
    return Float32(_asm(T.f32(), [Float32(x).ir_value(loc=loc, ip=ip) for x in (a, b)],
                        "mul.rn.f32 $0, $1, $2;", "=f,f,f", False, loc, ip))


@dsl_user_op
def add_f32(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(_asm(T.f32(), [Float32(x).ir_value(loc=loc, ip=ip) for x in (a, b)],
                        "add.rn.f32 $0, $1, $2;", "=f,f,f", False, loc, ip))


@dsl_user_op
def div_approx(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """nvcc --use_fast_math 'a / b' = div.approx.ftz.f32."""
    return Float32(_asm(T.f32(), [Float32(x).ir_value(loc=loc, ip=ip) for x in (a, b)],
                        "div.approx.ftz.f32 $0, $1, $2;", "=f,f,f", False, loc, ip))


@dsl_user_op
def rcp_approx(a: Float32, *, loc=None, ip=None) -> Float32:
    """nvcc --use_fast_math '1.0f / x' = rcp.approx.ftz.f32."""
    return Float32(_asm(T.f32(), [Float32(a).ir_value(loc=loc, ip=ip)], "rcp.approx.ftz.f32 $0, $1;", "=f,f", False,
                        loc, ip))


@dsl_user_op
def abs_f32(a: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(_asm(T.f32(), [Float32(a).ir_value(loc=loc, ip=ip)], "abs.f32 $0, $1;", "=f,f", False, loc, ip))


@dsl_user_op
def max_f32(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """fmaxf (IEEE maxNum: NaN-ignoring)."""
    return Float32(_asm(T.f32(), [Float32(x).ir_value(loc=loc, ip=ip) for x in (a, b)],
                        "max.f32 $0, $1, $2;", "=f,f,f", False, loc, ip))


@dsl_user_op
def min_f32(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(_asm(T.f32(), [Float32(x).ir_value(loc=loc, ip=ip) for x in (a, b)],
                        "min.f32 $0, $1, $2;", "=f,f,f", False, loc, ip))


@dsl_user_op
def warp_max_f32(a: Float32, *, loc=None, ip=None) -> Float32:
    """max over the warp of non-negative a (fmaxf butterfly, as sm90's __shfl_xor max)."""
    return Float32(_asm(T.f32(), [Float32(a).ir_value(loc=loc, ip=ip)],
                        "{\n\t.reg .f32 t;\n\t"
                        + "".join(f"shfl.sync.bfly.b32 t, $0, {o}, 0x1f, 0xffffffff;\n\tmax.f32 $0, $0, t;\n\t"
                                  for o in (16, 8, 4, 2, 1)) + "}",
                        "=f,0", True, loc, ip))


@dsl_user_op
def e4m3x4(x0: Float32, x1: Float32, x2: Float32, x3: Float32, *, loc=None, ip=None) -> Int32:
    """cvt.rn.satfinite.e4m3x2.f32 x2: bytes [x0, x1, x2, x3] little-endian."""
    return Int32(_asm(T.i32(), [Float32(x).ir_value(loc=loc, ip=ip) for x in (x0, x1, x2, x3)],
                      "{\n\t.reg .b16 lo, hi;\n\tcvt.rn.satfinite.e4m3x2.f32 lo, $2, $1;\n\t"
                      "cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3;\n\tmov.b32 $0, {lo, hi};\n\t}",
                      "=r,f,f,f,f", False, loc, ip))


@dsl_user_op
def bf16x2(x0: Float32, x1: Float32, *, loc=None, ip=None) -> Int32:
    """cvt.rn.bf16x2.f32: low half x0, high half x1."""
    return Int32(_asm(T.i32(), [Float32(x).ir_value(loc=loc, ip=ip) for x in (x0, x1)],
                      "cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,f,f", False, loc, ip))


@dsl_user_op
def st_shared_u8(addr: Int32, v: Int32, *, loc=None, ip=None):
    _asm(None, [Int32(addr).ir_value(loc=loc, ip=ip), Int32(v).ir_value(loc=loc, ip=ip)],
         "st.shared.u8 [$0], $1;", "r,r", True, loc, ip)


@dsl_user_op
def st_shared_f32(addr: Int32, v: Float32, *, loc=None, ip=None):
    _asm(None, [Int32(addr).ir_value(loc=loc, ip=ip), Float32(v).ir_value(loc=loc, ip=ip)],
         "st.shared.f32 [$0], $1;", "r,f", True, loc, ip)


@dsl_user_op
def ld_shared_f32(addr: Int32, *, loc=None, ip=None) -> Float32:
    return Float32(_asm(T.f32(), [Int32(addr).ir_value(loc=loc, ip=ip)], "ld.shared.f32 $0, [$1];", "=f,r", True,
                        loc, ip))


@dsl_user_op
def ld_shared_v4(addr: Int32, *, loc=None, ip=None):
    res = _asm(llvm.StructType.get_literal([T.i32()] * 4), [Int32(addr).ir_value(loc=loc, ip=ip)],
               "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,r", True, loc, ip)
    return [Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def st_global_v4(addr: Int64, a: Int32, b: Int32, c: Int32, d: Int32, *, loc=None, ip=None):
    _asm(None, [Int64(addr).ir_value(loc=loc, ip=ip)] + [Int32(x).ir_value(loc=loc, ip=ip) for x in (a, b, c, d)],
         "st.global.v4.b32 [$0], {$1, $2, $3, $4};", "l,r,r,r,r", True, loc, ip)


@dsl_user_op
def st_global_v4f(addr: Int64, a: Float32, b: Float32, c: Float32, d: Float32, *, loc=None, ip=None):
    _asm(None, [Int64(addr).ir_value(loc=loc, ip=ip)] + [Float32(x).ir_value(loc=loc, ip=ip) for x in (a, b, c, d)],
         "st.global.v4.f32 [$0], {$1, $2, $3, $4};", "l,f,f,f,f", True, loc, ip)


@dsl_user_op
def ld_global_v4f(addr: Int64, *, loc=None, ip=None):
    res = _asm(llvm.StructType.get_literal([T.f32()] * 4), [Int64(addr).ir_value(loc=loc, ip=ip)],
               "ld.global.v4.f32 {$0, $1, $2, $3}, [$4];", "=f,=f,=f,=f,l", True, loc, ip)
    return [Float32(llvm.extractvalue(T.f32(), res, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def ld_global_f32(addr: Int64, *, loc=None, ip=None) -> Float32:
    return Float32(_asm(T.f32(), [Int64(addr).ir_value(loc=loc, ip=ip)], "ld.global.f32 $0, [$1];", "=f,l", True,
                        loc, ip))


@dsl_user_op
def st_global_f32(addr: Int64, v: Float32, *, loc=None, ip=None):
    _asm(None, [Int64(addr).ir_value(loc=loc, ip=ip), Float32(v).ir_value(loc=loc, ip=ip)],
         "st.global.f32 [$0], $1;", "l,f", True, loc, ip)


def _gaddr(t: cute.Tensor) -> Int64:
    return Int64(t.iterator.toint())


# ---------------------------------------------------------------- kernel A
class DmHd256DS:
    """dS kernel. dbg bit 1: per (row, 32-key group) identity sums [sum dS', sum |P_hat * dP_acc*dp|, sum P_hat];
    dbg bit 2: raw dump [S_acc, dP_acc, P_hat] per (row, key) (small N only)."""
    TM, TN, D = 128, 128, 256

    def __init__(self, causal: bool, dbg: int = 0):
        self.causal = bool(causal)
        self.dbg = int(dbg)
        self.threads = 192   # warps 0-3 compute, 4 MMA, 5 load

    @cute.jit
    def __call__(self, mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor, mdO: cute.Tensor,
                 mQs: cute.Tensor, mKs: cute.Tensor, mVs: cute.Tensor, mdOs: cute.Tensor,
                 mLse: cute.Tensor, mDelta: cute.Tensor,
                 wPt: cute.Tensor, wdS: cute.Tensor, wdSt: cute.Tensor, wPsi: cute.Tensor, wG: cute.Tensor,
                 mIdent: cute.Tensor, mDump: cute.Tensor,
                 h0: Int32, rep: Int32, n_kt: Int32, Hc: Int32, nbatch: Int32, stream):
        tiler = (self.TM, self.TN, self.D)
        tiled_mma = sm100_utils.make_trivial_tiled_mma(F8, F8, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
                                                       Float32, tcgen05.CtaGroup.ONE, tiler[:2])
        a_l = sm100_utils.make_smem_layout_a(tiled_mma, tiler, F8, 1)
        b_l = sm100_utils.make_smem_layout_b(tiled_mma, tiler, F8, 1)
        a1 = cute.select(a_l, mode=[0, 1, 2])
        b1 = cute.select(b_l, mode=[0, 1, 2])
        cta_vmnk = cute.tiled_divide(cute.make_layout((1, 1, 1)), (tiled_mma.thr_id,))
        op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_Q, tQ = cute.nvgpu.make_tiled_tma_atom_A(op, mQ, a1, tiler, tiled_mma, cta_vmnk.shape)
        tma_dO, tdO = cute.nvgpu.make_tiled_tma_atom_A(op, mdO, a1, tiler, tiled_mma, cta_vmnk.shape)
        tma_K, tK = cute.nvgpu.make_tiled_tma_atom_B(op, mK, b1, tiler, tiled_mma, cta_vmnk.shape)
        tma_V, tV = cute.nvgpu.make_tiled_tma_atom_B(op, mV, b1, tiler, tiled_mma, cta_vmnk.shape)
        self.bytes_a = cute.size_in_bytes(F8, a1)
        self.bytes_b = cute.size_in_bytes(F8, b1)

        @cute.struct
        class SS:
            bars: cute.struct.MemRange[cutlass.Int64, 16]
            tmem_hold: Int32
            amax: cute.struct.MemRange[Float32, 16]
            sQ: cute.struct.Align[cute.struct.MemRange[F8, cute.cosize(a_l)], 1024]
            sdO: cute.struct.Align[cute.struct.MemRange[F8, cute.cosize(a_l)], 1024]
            sK: cute.struct.Align[cute.struct.MemRange[F8, cute.cosize(b_l)], 1024]
            sV: cute.struct.Align[cute.struct.MemRange[F8, cute.cosize(b_l)], 1024]
            sPt: cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, 128 * 128], 128]
            sdSt: cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, 128 * 128], 128]
        self.SS = SS
        self.kernel(tiled_mma, tma_Q, tQ, tma_dO, tdO, tma_K, tK, tma_V, tV, a_l, b_l,
                    mQs, mKs, mVs, mdOs, mLse, mDelta, wPt, wdS, wdSt, wPsi, wG, mIdent, mDump,
                    h0, rep, n_kt, Hc).launch(grid=[n_kt, Hc, nbatch], block=[self.threads, 1, 1],
                                              smem=SS.size_in_bytes(), stream=stream)

    @cute.kernel
    def kernel(self, tiled_mma: cute.TiledMma, tma_Q: cute.CopyAtom, mQ: cute.Tensor, tma_dO: cute.CopyAtom,
               mdO: cute.Tensor, tma_K: cute.CopyAtom, mK: cute.Tensor, tma_V: cute.CopyAtom, mV: cute.Tensor,
               a_l: cute.ComposedLayout, b_l: cute.ComposedLayout,
               mQs: cute.Tensor, mKs: cute.Tensor, mVs: cute.Tensor, mdOs: cute.Tensor, mLse: cute.Tensor,
               mDelta: cute.Tensor, wPt: cute.Tensor, wdS: cute.Tensor, wdSt: cute.Tensor, wPsi: cute.Tensor,
               wG: cute.Tensor, mIdent: cute.Tensor, mDump: cute.Tensor,
               h0: Int32, rep: Int32, n_kt: Int32, Hc: Int32):
        tidx = cute.arch.thread_idx()[0]
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        kt, hc, b = cute.arch.block_idx()
        h = h0 + hc
        hk = h // rep
        l = b * Hc + hc
        N = n_kt * 128
        nmb = n_kt * 2
        mt0 = Int32(0)
        if const_expr(self.causal):
            mt0 = kt
        n_it = n_kt - mt0

        smem = cutlass.utils.SmemAllocator()
        st = smem.allocate(self.SS)
        sQ = st.sQ.get_tensor(a_l.outer, swizzle=a_l.inner)
        sdO = st.sdO.get_tensor(a_l.outer, swizzle=a_l.inner)
        sK = st.sK.get_tensor(b_l.outer, swizzle=b_l.inner)
        sV = st.sV.get_tensor(b_l.outer, swizzle=b_l.inner)
        sPt_addr = Int32(st.sPt.data_ptr().toint())
        sdSt_addr = Int32(st.sdSt.data_ptr().toint())
        amax_addr = Int32(st.amax.data_ptr().toint())
        bars = st.bars.data_ptr()

        if warp == 5:
            cpasync.prefetch_descriptor(tma_Q)
            cpasync.prefetch_descriptor(tma_dO)
            cpasync.prefetch_descriptor(tma_K)
            cpasync.prefetch_descriptor(tma_V)
        kv_p, kv_c = pipeline.PipelineTmaUmma.create(
            num_stages=1, producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread), tx_count=2 * self.bytes_b,
            barrier_storage=bars).make_participants()
        qd_p, qd_c = pipeline.PipelineTmaUmma.create(
            num_stages=1, producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread), tx_count=2 * self.bytes_a,
            barrier_storage=bars + 2).make_participants()
        acc_p, acc_c = pipeline.PipelineUmmaAsync.create(
            num_stages=2, producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 128),
            barrier_storage=bars + 4).make_participants()

        tmem = utils.TmemAllocator(st.tmem_hold.ptr,
                                   barrier_for_retrieve=pipeline.NamedBarrier(barrier_id=1, num_threads=self.threads))
        tmem.allocate(512)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(Float32)
        tmem_base = Int32(tmem_ptr.toint())

        thr_mma = tiled_mma.get_slice(0)
        acc_shape = tiled_mma.partition_shape_C(self.tiler_mn())
        acc_l = tiled_mma.make_fragment_C(acc_shape).layout
        tCrQ = tiled_mma.make_fragment_A(sQ)
        tCrdO = tiled_mma.make_fragment_A(sdO)
        tCrK = tiled_mma.make_fragment_B(sK)
        tCrV = tiled_mma.make_fragment_B(sV)

        if warp == 5:
            gQ = cute.local_tile(mQ[None, None, h, b], (self.TM, self.D), (None, 0))
            gdO = cute.local_tile(mdO[None, None, h, b], (self.TM, self.D), (None, 0))
            gK = cute.local_tile(mK[None, None, hk, b], (self.TN, self.D), (None, 0))
            gV = cute.local_tile(mV[None, None, hk, b], (self.TN, self.D), (None, 0))
            tQs, tQg = cpasync.tma_partition(tma_Q, 0, cute.make_layout(1), cute.group_modes(sQ, 0, 3),
                                             cute.group_modes(thr_mma.partition_A(gQ), 0, 3))
            tdOs, tdOg = cpasync.tma_partition(tma_dO, 0, cute.make_layout(1), cute.group_modes(sdO, 0, 3),
                                               cute.group_modes(thr_mma.partition_A(gdO), 0, 3))
            tKs, tKg = cpasync.tma_partition(tma_K, 0, cute.make_layout(1), cute.group_modes(sK, 0, 3),
                                             cute.group_modes(thr_mma.partition_B(gK), 0, 3))
            tVs, tVg = cpasync.tma_partition(tma_V, 0, cute.make_layout(1), cute.group_modes(sV, 0, 3),
                                             cute.group_modes(thr_mma.partition_B(gV), 0, 3))
            ke = kv_p.acquire_and_advance()
            cute.copy(tma_K, tKg[(None, kt)], tKs[(None, 0)], tma_bar_ptr=ke.barrier)
            cute.copy(tma_V, tVg[(None, kt)], tVs[(None, 0)], tma_bar_ptr=ke.barrier)
            for it in cutlass.range(n_it, unroll=1):
                qe = qd_p.acquire_and_advance()
                cute.copy(tma_Q, tQg[(None, mt0 + it)], tQs[(None, 0)], tma_bar_ptr=qe.barrier)
                cute.copy(tma_dO, tdOg[(None, mt0 + it)], tdOs[(None, 0)], tma_bar_ptr=qe.barrier)
            qd_p.tail()

        if warp == 4:
            kv_c.wait_and_advance()
            for it in cutlass.range(n_it, unroll=1):
                qf = qd_c.wait_and_advance()
                ae = acc_p.acquire_and_advance()
                accS = cute.make_tensor(tmem_ptr + ae.index * 256, acc_l)
                accP = cute.make_tensor(tmem_ptr + ae.index * 256 + 128, acc_l)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for kb in cutlass.range_constexpr(cute.size(tCrQ, mode=[2])):
                    cute.gemm(tiled_mma, accS, tCrQ[(None, None, kb, 0)], tCrK[(None, None, kb, 0)], accS)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for kb in cutlass.range_constexpr(cute.size(tCrdO, mode=[2])):
                    cute.gemm(tiled_mma, accP, tCrdO[(None, None, kb, 0)], tCrV[(None, None, kb, 0)], accP)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                qf.release()
                ae.commit()
            acc_p.tail()

        if warp < 4:
            r = tidx                     # row within the 128-row tile = TMEM lane
            sub = r // 64                # 64-row m-block within the tile
            lane_off = Int32((warp * 32) << 16)
            ks = [Float32(mKs[b, hk, kt * 4 + j]) for j in range(4)]
            vs = [Float32(mVs[b, hk, kt * 4 + j]) for j in range(4)]
            gmax0 = Float32(0.0)
            gmax1 = Float32(0.0)
            ws_ds = _gaddr(wdS)
            ws_pt = _gaddr(wPt)
            ws_dst = _gaddr(wdSt)
            for it in cutlass.range(n_it, unroll=1):
                mt = mt0 + it
                row = mt * 128 + r
                mb = row // 64
                qsv = Float32(mQs[b, h, mb])
                dosv = Float32(mdOs[b, h, mb])
                qsm = [Float32(mQs[b, h, 2 * mt]), Float32(mQs[b, h, 2 * mt + 1])]
                l2 = dmh.dm_p0h_l2(Float32(mLse[b, h, row]))
                dl = Float32(mDelta[b, h, row])
                dsc = [dmh.dm_p0h_dsc(qsv, ks[j]) for j in range(4)]
                dpj = [dmh.dm_p0h_dp_scale(dosv, vs[j]) for j in range(4)]
                dpf = [dpj[j] * ks[j] for j in range(4)]
                ndk = [-(dl * ks[j]) for j in range(4)]
                af = acc_c.wait_and_advance()
                sbase = tmem_base + af.index * 256 + lane_off
                par = it & 1
                row_ds = ws_ds + ((Int64(l) * N + row) * N + kt * 128)
                for hf in cutlass.range_constexpr(2):
                    ds = cute.make_rmem_tensor(cute.make_layout(64), Float32)
                    for jj in cutlass.range_constexpr(2):
                        j = 2 * hf + jj
                        c0 = 32 * j
                        sv = tmem_ld32(sbase + c0)
                        dv = tmem_ld32(sbase + 128 + c0)
                        cute.arch.fence_view_async_tmem_load()
                        if const_expr(hf == 1 and jj == 1):
                            af.release()
                        id1 = Float32(0.0)
                        id2 = Float32(0.0)
                        id3 = Float32(0.0)
                        for i in cutlass.range_constexpr(0, 32, 4):
                            pv = [None] * 4
                            for q2 in cutlass.range_constexpr(0, 4, 2):
                                s0 = sv[i + q2]
                                s1 = sv[i + q2 + 1]
                                if const_expr(self.causal):
                                    key0 = kt * 128 + c0 + i + q2
                                    s0 = -Float32.inf if key0 > row else s0
                                    s1 = -Float32.inf if key0 + 1 > row else s1
                                p0, p1 = dmh.dm_p0h_phat_pair(s0, s1, dsc[j], l2)
                                pv[q2] = p0
                                pv[q2 + 1] = p1
                            for q in cutlass.range_constexpr(4):
                                x = mul_f32(pv[q], fma_f32(dv[i + q], dpf[j], ndk[j]))
                                ds[32 * jj + i + q] = x
                                if const_expr(self.dbg & 1):
                                    id1 = add_f32(id1, x)
                                    id2 = add_f32(id2, abs_f32(mul_f32(pv[q], mul_f32(dv[i + q], dpj[j]))))
                                    id3 = add_f32(id3, pv[q])
                                if const_expr(self.dbg & 2):
                                    da = _gaddr(mDump) + ((Int64(l) * N + row) * N + kt * 128 + c0 + i + q) * 16
                                    st_global_v4f(da, sv[i + q], dv[i + q], pv[q], x)
                            w = e4m3x4(pv[0], pv[1], pv[2], pv[3])
                            for q in cutlass.range_constexpr(4):
                                st_shared_u8(sPt_addr + (c0 + i + q) * 128 + r, w >> (8 * q))
                        if const_expr(self.dbg & 1):
                            ia = _gaddr(mIdent) + ((Int64(l) * N + row) * (N // 32) + kt * 4 + j) * 16
                            st_global_v4f(ia, id1, id2, id3, Float32(0.0))
                    # psi of the 64x64 tiles (2 m-blocks x this n-block)
                    am = Float32(0.0)
                    for i in cutlass.range_constexpr(64):
                        am = max_f32(am, abs_f32(ds[i]))
                    am = warp_max_f32(am)
                    slot = amax_addr + (par * 8 + hf * 4) * 4
                    if lane == 0:
                        st_shared_f32(slot + warp * 4, am)
                    cute.arch.barrier(barrier_id=2, number_of_threads=128)
                    nb = 2 * kt + hf
                    gm = Float32(0.0)
                    if const_expr(hf == 0):
                        gm = gmax0
                    else:
                        gm = gmax1
                    pinv_mine = Float32(0.0)
                    for sb in cutlass.range_constexpr(2):
                        a = max_f32(ld_shared_f32(slot + (2 * sb) * 4), ld_shared_f32(slot + (2 * sb + 1) * 4))
                        a448 = a * Float32(INV448)
                        tz = a448 < Float32(1e-30)
                        praw = max_f32(a448, Float32(1e-30))
                        psi = max_f32(praw, div_approx(gm * Float32(F2B_C), qsm[sb]))
                        pinv = Float32(0.0) if tz else rcp_approx(psi)
                        g = (psi * Float32(dmh.INV_PHAT)) * qsm[sb]
                        gm = gm if tz else max_f32(gm, g)
                        pinv_mine = pinv if sub == sb else pinv_mine
                        if tidx == 0:
                            wPsi[l, 2 * mt + sb, nb] = Float32(0.0) if tz else psi
                            wG[l, 2 * mt + sb, nb] = Float32(0.0) if tz else g
                    if const_expr(hf == 0):
                        gmax0 = gm
                    else:
                        gmax1 = gm
                    # dS'8 codes: natural row (direct 16 B stores) + transposed staging
                    for v in cutlass.range_constexpr(4):
                        wv = [None] * 4
                        for u in cutlass.range_constexpr(4):
                            i0 = 16 * v + 4 * u
                            wv[u] = e4m3x4(ds[i0] * pinv_mine, ds[i0 + 1] * pinv_mine, ds[i0 + 2] * pinv_mine,
                                           ds[i0 + 3] * pinv_mine)
                            for q in cutlass.range_constexpr(4):
                                st_shared_u8(sdSt_addr + (64 * hf + i0 + q) * 128 + r, wv[u] >> (8 * q))
                        st_global_v4(row_ds + 64 * hf + 16 * v, wv[0], wv[1], wv[2], wv[3])
                cute.arch.barrier(barrier_id=2, number_of_threads=128)
                # copy-out of the transposed tiles: thread r owns key kt*128 + r, rows mt*128 .. +127
                keyrow = (Int64(l) * N + kt * 128 + r) * N + mt * 128
                for v in cutlass.range_constexpr(8):
                    x = ld_shared_v4(sPt_addr + r * 128 + 16 * v)
                    st_global_v4(ws_pt + keyrow + 16 * v, x[0], x[1], x[2], x[3])
                    y = ld_shared_v4(sdSt_addr + r * 128 + 16 * v)
                    st_global_v4(ws_dst + keyrow + 16 * v, y[0], y[1], y[2], y[3])
                cute.arch.barrier(barrier_id=2, number_of_threads=128)

        tmem.relinquish_alloc_permit()
        cute.arch.barrier(barrier_id=3, number_of_threads=self.threads)
        tmem.free(tmem_ptr)

    def tiler_mn(self):
        return (self.TM, self.TN)


# ---------------------------------------------------------------- kernel B
class DmHd256Fold:
    """Token-contraction kernel: out[b, M-tile rows, oh, 256] for one output head per CTA (y = group index).
    mode: "dk" | "dv" | "dq".  A (M, K, LA) and B (256, K, LB) are K-major fp8; F is a flat fp32 tensor indexed
    F[la * fsl + m64 * fsm + c * fsk].  Head loop hh in [max(h0, g*gs), min(h1, (g+1)*gs)), A index la = b*Hc + hh - h0,
    B index lb = b*HB + hh // bdiv.  Chunks c: dk/dv (keys on M): [causal ? 2*mt : 0, N/64); dq (queries on M):
    [0, causal ? 2*mt + 2 : N/64)."""
    TM, D, KC = 128, 256, 64
    STAGES = 3

    def __init__(self, mode: str, causal: bool):
        assert mode in ("dk", "dv", "dq")
        self.mode = mode
        self.causal = bool(causal)
        self.threads = 320   # warps 0-7 fold (lane quadrant w%4, column half w//4), 8 MMA, 9 load
        self.win = mode in ("dv", "dq")

    @cute.jit
    def __call__(self, mA: cute.Tensor, mB: cute.Tensor, mF: cute.Tensor, mKs: cute.Tensor, mOut: cute.Tensor,
                 mAcc: cute.Tensor, h0: Int32, h1: Int32, gs: Int32, g0: Int32, Hc: Int32, HB: Int32, bdiv: Int32,
                 fsl: Int32, fsm: Int32, fsk: Int32, n_mt: Int32, n_groups: Int32, nbatch: Int32, Hout: Int32, stream):
        tiler = (self.TM, self.D, self.KC)
        tiled_mma = sm100_utils.make_trivial_tiled_mma(F8, F8, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
                                                       Float32, tcgen05.CtaGroup.ONE, tiler[:2])
        a_l = sm100_utils.make_smem_layout_a(tiled_mma, tiler, F8, self.STAGES)
        b_l = sm100_utils.make_smem_layout_b(tiled_mma, tiler, F8, self.STAGES)
        a1 = cute.select(a_l, mode=[0, 1, 2])
        b1 = cute.select(b_l, mode=[0, 1, 2])
        cta_vmnk = cute.tiled_divide(cute.make_layout((1, 1, 1)), (tiled_mma.thr_id,))
        op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_A, tA = cute.nvgpu.make_tiled_tma_atom_A(op, mA, a1, tiler, tiled_mma, cta_vmnk.shape)
        tma_B, tB = cute.nvgpu.make_tiled_tma_atom_B(op, mB, b1, tiler, tiled_mma, cta_vmnk.shape)
        self.bytes_ab = cute.size_in_bytes(F8, a1) + cute.size_in_bytes(F8, b1)
        nout = 128 * 256 if const_expr(self.win) else 4

        @cute.struct
        class SS:
            bars: cute.struct.MemRange[cutlass.Int64, 2 * self.STAGES + 4]
            tmem_hold: Int32
            sOut: cute.struct.Align[cute.struct.MemRange[Float32, nout], 128]
            sA: cute.struct.Align[cute.struct.MemRange[F8, cute.cosize(a_l)], 1024]
            sB: cute.struct.Align[cute.struct.MemRange[F8, cute.cosize(b_l)], 1024]
        self.SS = SS
        self.kernel(tiled_mma, tma_A, tA, tma_B, tB, a_l, b_l, mF, mKs, mOut, mAcc, h0, h1, gs, g0, Hc, HB, bdiv,
                    fsl, fsm, fsk, n_mt, Hout).launch(grid=[n_mt, n_groups, nbatch], block=[self.threads, 1, 1],
                                                      smem=SS.size_in_bytes(), stream=stream)

    @cute.kernel
    def kernel(self, tiled_mma: cute.TiledMma, tma_A: cute.CopyAtom, mA: cute.Tensor, tma_B: cute.CopyAtom,
               mB: cute.Tensor, a_l: cute.ComposedLayout, b_l: cute.ComposedLayout, mF: cute.Tensor,
               mKs: cute.Tensor, mOut: cute.Tensor, mAcc: cute.Tensor, h0: Int32, h1: Int32, gs: Int32, g0: Int32,
               Hc: Int32, HB: Int32, bdiv: Int32, fsl: Int32, fsm: Int32, fsk: Int32, n_mt: Int32, Hout: Int32):
        tidx = cute.arch.thread_idx()[0]
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        mt, gy, b = cute.arch.block_idx()
        g = g0 + gy
        hlo = cutlass.max(h0, g * gs)
        hhi = cutlass.min(h1, (g + 1) * gs)
        nh = hhi - hlo
        N = n_mt * 128
        nck = n_mt * 2
        c_lo = Int32(0)
        c_hi = nck
        if const_expr(self.causal):
            if const_expr(self.mode == "dq"):
                c_hi = 2 * mt + 2
            else:
                c_lo = 2 * mt
        ncs = c_hi - c_lo
        total = nh * ncs

        smem = cutlass.utils.SmemAllocator()
        st = smem.allocate(self.SS)
        sA = st.sA.get_tensor(a_l.outer, swizzle=a_l.inner)
        sB = st.sB.get_tensor(b_l.outer, swizzle=b_l.inner)
        sOut = Int32(st.sOut.data_ptr().toint())
        bars = st.bars.data_ptr()
        if warp == 9:
            cpasync.prefetch_descriptor(tma_A)
            cpasync.prefetch_descriptor(tma_B)
        ab_p, ab_c = pipeline.PipelineTmaUmma.create(
            num_stages=self.STAGES, producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread), tx_count=self.bytes_ab,
            barrier_storage=bars).make_participants()
        acc_p, acc_c = pipeline.PipelineUmmaAsync.create(
            num_stages=2, producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 256),
            barrier_storage=bars + 2 * self.STAGES).make_participants()
        tmem = utils.TmemAllocator(st.tmem_hold.ptr,
                                   barrier_for_retrieve=pipeline.NamedBarrier(barrier_id=1, num_threads=self.threads))
        tmem.allocate(512)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(Float32)
        tmem_base = Int32(tmem_ptr.toint())
        thr_mma = tiled_mma.get_slice(0)
        acc_l = tiled_mma.make_fragment_C(tiled_mma.partition_shape_C((self.TM, self.D))).layout
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)

        if warp == 9:
            for hi in cutlass.range(nh, unroll=1):
                hh = hlo + hi
                la = b * Hc + hh - h0
                lb = b * HB + hh // bdiv
                gA = cute.local_tile(mA[None, None, la], (self.TM, self.KC), (mt, None))
                gB = cute.local_tile(mB[None, None, lb], (self.D, self.KC), (0, None))
                tAs, tAg = cpasync.tma_partition(tma_A, 0, cute.make_layout(1), cute.group_modes(sA, 0, 3),
                                                 cute.group_modes(thr_mma.partition_A(gA), 0, 3))
                tBs, tBg = cpasync.tma_partition(tma_B, 0, cute.make_layout(1), cute.group_modes(sB, 0, 3),
                                                 cute.group_modes(thr_mma.partition_B(gB), 0, 3))
                for ci in cutlass.range(ncs, unroll=1):
                    e = ab_p.acquire_and_advance()
                    cute.copy(tma_A, tAg[(None, c_lo + ci)], tAs[(None, e.index)], tma_bar_ptr=e.barrier)
                    cute.copy(tma_B, tBg[(None, c_lo + ci)], tBs[(None, e.index)], tma_bar_ptr=e.barrier)
            ab_p.tail()

        if warp == 8:
            for t in cutlass.range(total, unroll=1):
                f = ab_c.wait_and_advance()
                ae = acc_p.acquire_and_advance()
                acc = cute.make_tensor(tmem_ptr + ae.index * 256, acc_l)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for kb in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                    cute.gemm(tiled_mma, acc, tCrA[(None, None, kb, f.index)], tCrB[(None, None, kb, f.index)], acc)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                f.release()
                ae.commit()
            acc_p.tail()

        if warp < 8:
            quad = warp % 4
            ch = warp // 4
            r = quad * 32 + lane
            row = mt * 128 + r
            m64 = 2 * mt + r // 64
            lane_off = Int32((quad * 32) << 16)
            acc = cute.make_rmem_tensor(cute.make_layout(128), Float32)   # dk: acc; dv/dq: window
            for i in cutlass.range_constexpr(128):
                acc[i] = Float32(0.0)
            # out row base (bf16 / fp32 carry) and smem out (dv/dq) with an XOR swizzle per lane
            obase = ((Int64(b) * N + row) * Hout + g) * 256 + ch * 128
            first_head = hlo == g * gs
            last_head = hhi == (g + 1) * gs
            sbase = sOut + (r * 256 + ch * 128) * 4
            if const_expr(self.win):
                if first_head:
                    for i in cutlass.range_constexpr(128):
                        st_shared_f32(sbase + ((i ^ lane) * 4), Float32(0.0))
                else:
                    for i in cutlass.range_constexpr(0, 128, 4):
                        v4 = ld_global_v4f(_gaddr(mAcc) + (obase + i) * 4)
                        for q in cutlass.range_constexpr(4):
                            st_shared_f32(sbase + (((i + q) ^ lane) * 4), v4[q])
            else:
                if first_head == False:  # noqa: E712
                    for i in cutlass.range_constexpr(0, 128, 4):
                        v4 = ld_global_v4f(_gaddr(mAcc) + (obase + i) * 4)
                        for q in cutlass.range_constexpr(4):
                            acc[i + q] = v4[q]
            fpos = Int32(0)   # flat F index base
            for hi in cutlass.range(nh, unroll=1):
                hh = hlo + hi
                la = b * Hc + hh - h0
                fprev = Float32(1.0)      # dv basis (f_prev starts at 1.0 per head, hpp:1715)
                psi_w = Float32(0.0)      # dq window basis
                psi_max = Float32(0.0)
                for ci in cutlass.range(ncs, unroll=1):
                    c = c_lo + ci
                    fc = Float32(mF[la * fsl + m64 * fsm + c * fsk])
                    af = acc_c.wait_and_advance()
                    tb = tmem_base + af.index * 256 + ch * 128 + lane_off
                    tv = cute.make_rmem_tensor(cute.make_layout(128), Float32)
                    for p in cutlass.range_constexpr(4):
                        vv = tmem_ld32(tb + 32 * p)
                        for i in cutlass.range_constexpr(32):
                            tv[32 * p + i] = vv[i]
                    cute.arch.fence_view_async_tmem_load()
                    af.release()
                    if const_expr(self.mode == "dk"):
                        for i in cutlass.range_constexpr(128):
                            acc[i] = fma_f32(tv[i], fc, acc[i])
                    elif const_expr(self.mode == "dv"):
                        rf = min_f32(max_f32(div_approx(fprev, fc), Float32(1e-8)), Float32(1e8))
                        for i in cutlass.range_constexpr(128):
                            acc[i] = add_f32(mul_f32(acc[i], rf), tv[i])
                        fprev = fc
                        if (c & 7) == 7 or c == c_hi - 1:
                            for i in cutlass.range_constexpr(128):
                                a = sbase + ((i ^ lane) * 4)
                                st_shared_f32(a, fma_f32(acc[i], fprev, ld_shared_f32(a)))
                                acc[i] = Float32(0.0)
                    else:
                        if fc != Float32(0.0) and not (fc < psi_max * Float32(1e-8)):
                            if psi_w != Float32(0.0):
                                ratio = min_f32(max_f32(div_approx(psi_w, fc), Float32(1e-8)), Float32(1e8))
                                for i in cutlass.range_constexpr(128):
                                    acc[i] = add_f32(mul_f32(acc[i], ratio), tv[i])
                            else:
                                for i in cutlass.range_constexpr(128):
                                    acc[i] = tv[i]
                            psi_w = fc
                            psi_max = max_f32(psi_max, fc)
                        if (c & 7) == 7 or c == c_hi - 1:
                            if psi_w != Float32(0.0):
                                for i in cutlass.range_constexpr(128):
                                    a = sbase + ((i ^ lane) * 4)
                                    st_shared_f32(a, fma_f32(acc[i], psi_w, ld_shared_f32(a)))
                                psi_w = Float32(0.0)
            # epilogue
            if const_expr(self.win):
                for i in cutlass.range_constexpr(128):
                    acc[i] = ld_shared_f32(sbase + ((i ^ lane) * 4))
            if last_head:
                post = Float32(1.0)
                if const_expr(self.mode == "dk"):
                    post = div_approx(Float32(1.0), Float32(mKs[b, g, row // 32]))
                if const_expr(self.mode == "dq"):
                    post = Float32(2.0 ** -12)
                ob = _gaddr(mOut) + obase * 2
                for i in cutlass.range_constexpr(0, 128, 8):
                    w = [None] * 4
                    for q in cutlass.range_constexpr(4):
                        x0 = acc[i + 2 * q]
                        x1 = acc[i + 2 * q + 1]
                        if const_expr(self.mode != "dv"):
                            x0 = mul_f32(x0, post)
                            x1 = mul_f32(x1, post)
                        w[q] = bf16x2(x0, x1)
                    st_global_v4(ob + i * 2, w[0], w[1], w[2], w[3])
            else:
                for i in cutlass.range_constexpr(0, 128, 4):
                    st_global_v4f(_gaddr(mAcc) + (obase + i) * 4, acc[i], acc[i + 1], acc[i + 2], acc[i + 3])

        tmem.relinquish_alloc_permit()
        cute.arch.barrier(barrier_id=3, number_of_threads=self.threads)
        tmem.free(tmem_ptr)
