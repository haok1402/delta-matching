"""
hd256 dQ kernels (spec sm100-mx-d256), bitwise the same dQ:
- DmHd256Mx2 recomputes S and dP per (q tile, key tile), forms P and dS with the
  pass-0 arithmetic, quantizes dS per (128 q x 32 keys) and accumulates dQ = dS K
  in TMEM (the kv2 path, used when no workspace plan fits);
- DmHd256Dqg forms dQ = dS K from the dS codes and block exponents KV2 stored to
  the workspace (no recompute).
Both run kind::mxf8f6f4 M=128 N=256 K=32 MMAs, 4 ascending k-steps per 128-key
tile, key tiles ascending, and the same x 2^-4 bf16 epilogue.
"""
import threading

import torch
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Float32, Int32, Int64, const_expr
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass._mlir.dialects import llvm
from flash_attn.cute import cache_utils
from flash_attn.cute.cute_dsl_utils import to_cute_tensor
from flash_attn.cute.testing import is_fake_mode

from delta_matching.sm100.cute_ptx_sm100 import tmem_ld32, e4m3x4, bf16x2, st_global_v4, st_shared_f32, abs_f32, max_f32, mul_f32, global_address, inline_asm
from delta_matching.sm100.cute_ptx_sm100 import mbar_init, tc_commit, redux_max_abs_f32, tmem_st_x1, tmem_st_x4, st_shared_v4_b32
from delta_matching.sm100.attention_math_sm100 import dm_mx256_c, dm_mx256_dp_scale, dm_mx256_l2, dm_mx256_p_pair
from delta_matching.sm100.fp8_e8m0_scale_sm100 import add_sb_clamp, inv_pow2_from_sb, pow2_sb
from delta_matching.sm100.mma_sm100 import gemm_ptx_bs
from delta_matching.sm100.tcgen05_fence_sm100 import mbar_wait_parity
from delta_matching.sm100.compile_cache_sm100 import fingerprint

__all__ = ["DmHd256Mx2", "DmHd256Dqg", "dq_recompute_launch", "dq_workspace_launch", "ld_shared_v4f", "sb_of", "pack4", "sel_eq_i32", "tc_fence_before", "tc_fence_after", "run_kernel"]

F8 = cutlass.Float8E4M3FN
SM = 0.0625          # 256 ** -0.5, exact



@dsl_user_op
def ld_shared_v4f(addr: Int32, *, loc=None, ip=None):
    res = inline_asm(llvm.StructType.get_literal([T.f32()] * 4), [Int32(addr).ir_value(loc=loc, ip=ip)],
               "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];", "=f,=f,=f,=f,r", True, loc, ip)
    return [Float32(llvm.extractvalue(T.f32(), res, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def sb_of(x: Float32, *, loc=None, ip=None) -> Int32:
    """Biased exponent byte of an fp32 power of two (its UE8M0 code)."""
    return Int32(inline_asm(T.i32(), [Float32(x).ir_value(loc=loc, ip=ip)],
                      "{\n\t.reg .b32 t;\n\tmov.b32 t, $1;\n\tbfe.u32 $0, t, 23, 8;\n\t}", "=r,f", False, loc, ip))


@dsl_user_op
def pack4(b0: Int32, b1: Int32, b2: Int32, b3: Int32, *, loc=None, ip=None) -> Int32:
    """bytes [b0, b1, b2, b3] little-endian (each < 256)."""
    return Int32(inline_asm(T.i32(), [Int32(x).ir_value(loc=loc, ip=ip) for x in (b0, b1, b2, b3)],
                      "{\n\t.reg .b32 t;\n\tshl.b32 t, $4, 8;\n\tor.b32 t, t, $3;\n\tshl.b32 t, t, 8;\n\tor.b32 t, t, $2;\n\t"
                      "shl.b32 t, t, 8;\n\tor.b32 $0, t, $1;\n\t}", "=r,r,r,r,r", False, loc, ip))


@dsl_user_op
def tc_fence_before(*, loc=None, ip=None):
    inline_asm(None, [], "tcgen05.fence::before_thread_sync;", "", True, loc, ip)


@dsl_user_op
def tc_fence_after(*, loc=None, ip=None):
    inline_asm(None, [], "tcgen05.fence::after_thread_sync;", "", True, loc, ip)


@dsl_user_op
def sel_eq_i32(x: Int32, y: Int32, a: Int32, b: Int32, *, loc=None, ip=None) -> Int32:
    """a if x == y else b"""
    return Int32(inline_asm(T.i32(), [Int32(v).ir_value(loc=loc, ip=ip) for v in (x, y, a, b)],
                      "{\n\t.reg .pred p;\n\tsetp.eq.s32 p, $1, $2;\n\tselp.b32 $0, $3, $4, p;\n\t}", "=r,r,r,r,r",
                      False, loc, ip))


@dsl_user_op
def ld_global_u32(addr: Int64, *, loc=None, ip=None) -> Int32:
    return Int32(inline_asm(T.i32(), [Int64(addr).ir_value(loc=loc, ip=ip)], "ld.global.u32 $0, [$1];", "=r,l", True, loc, ip))


@dsl_user_op
def byte_of(w: Int32, k: int, *, loc=None, ip=None) -> Int32:
    return Int32(inline_asm(T.i32(), [Int32(w).ir_value(loc=loc, ip=ip)], f"bfe.u32 $0, $1, {8 * k}, 8;", "=r,r", False, loc, ip))


@dsl_user_op
def bulk_g2s(saddr: Int32, gaddr: Int64, nbytes: int, mbar: Int32, *, loc=None, ip=None):
    """1-D bulk copy gmem -> smem completing nbytes of tx on mbar (wsw: the W tile is already the swizzled smem image)."""
    inline_asm(None, [Int32(saddr).ir_value(loc=loc, ip=ip), Int64(gaddr).ir_value(loc=loc, ip=ip), Int32(mbar).ir_value(loc=loc, ip=ip)],
         f"cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1], {int(nbytes)}, [$2];",
         "r,l,r", True, loc, ip)


@dsl_user_op
def warp_arrive(mbar: Int32, *, loc=None, ip=None):
    """bar.warp.sync, then lane 0 mbarrier.arrive.release.cta (every lane did its own wait::st + fence::before first)."""
    inline_asm(None, [Int32(mbar).ir_value(loc=loc, ip=ip)],
         "{\n\t.reg .pred p;\n\t.reg .b32 l;\n\tbar.warp.sync 0xffffffff;\n\tmov.u32 l, %laneid;\n\t"
         "setp.eq.u32 p, l, 0;\n\t@p mbarrier.arrive.release.cta.shared::cta.b64 _, [$0];\n\t}", "r", True, loc, ip)


class DmHd256Mx2:
    TM, TN, D = 128, 128, 256
    COL_SF, SF_STRIDE, COL_OUT = 128, 16, 256
    SFB_OFF = 8

    def __init__(self, causal: bool):
        self.causal = bool(causal)
        self.threads = 320

    @cute.jit
    def __call__(self, mXa: cute.Tensor, mXb: cute.Tensor, mYa: cute.Tensor, mYb: cute.Tensor, mT: cute.Tensor,
                 mQs: cute.Tensor, mKs: cute.Tensor, mVs: cute.Tensor, mdOs: cute.Tensor,
                 mLse: cute.Tensor, mDelta: cute.Tensor, mOut: cute.Tensor,
                 mIdent: cute.Tensor, mDump: cute.Tensor, mCodes: cute.Tensor, mSB: cute.Tensor, mHash: cute.Tensor,
                 n_t: Int32, H: Int32, Hk: Int32, stream):
        D = self.D
        s_tiler = (self.TM, self.TN, D)
        mma_s = sm100_utils.make_trivial_tiled_mma(F8, F8, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
                                                   Float32, tcgen05.CtaGroup.ONE, s_tiler[:2])
        a_l = sm100_utils.make_smem_layout_a(mma_s, s_tiler, F8, 1)
        b_l = sm100_utils.make_smem_layout_b(mma_s, s_tiler, F8, 1)
        o_tiler = (self.TM, D, self.TN)
        mma_o = sm100_utils.make_trivial_tiled_mma(F8, F8, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.MN,
                                                   Float32, tcgen05.CtaGroup.ONE, o_tiler[:2])
        c_l = sm100_utils.make_smem_layout_a(mma_o, o_tiler, F8, 2)       # 2 code stages
        t_l = sm100_utils.make_smem_layout_b(mma_o, o_tiler, F8, 1)
        a1 = cute.select(a_l, mode=[0, 1, 2])
        b1 = cute.select(b_l, mode=[0, 1, 2])
        t1 = cute.select(t_l, mode=[0, 1, 2])
        c1 = cute.select(c_l, mode=[0, 1, 2])
        vmnk = cute.tiled_divide(cute.make_layout((1, 1, 1)), (mma_s.thr_id,))
        op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_Xa, tXa = cute.nvgpu.make_tiled_tma_atom_A(op, mXa, a1, s_tiler, mma_s, vmnk.shape)
        tma_Xb, tXb = cute.nvgpu.make_tiled_tma_atom_A(op, mXb, a1, s_tiler, mma_s, vmnk.shape)
        tma_Ya, tYa = cute.nvgpu.make_tiled_tma_atom_B(op, mYa, b1, s_tiler, mma_s, vmnk.shape)
        tma_Yb, tYb = cute.nvgpu.make_tiled_tma_atom_B(op, mYb, b1, s_tiler, mma_s, vmnk.shape)
        tma_T, tT = cute.nvgpu.make_tiled_tma_atom_B(op, mT, t1, o_tiler, mma_o, vmnk.shape)
        self.bytes_a = cute.size_in_bytes(F8, a1)
        self.bytes_b = cute.size_in_bytes(F8, b1)
        self.bytes_t = cute.size_in_bytes(F8, t1)
        self.bytes_c = cute.size_in_bytes(F8, c1)
        nx = 2

        @cute.struct
        class SS:
            bars: cute.struct.MemRange[cutlass.Int64, 24]
            tmem_hold: Int32
            xch: cute.struct.Align[cute.struct.MemRange[Float32, 2 * 2 * 128], 16]
            sLD: cute.struct.Align[cute.struct.MemRange[Float32, 2 * 2 * 128], 16]
            sId: cute.struct.Align[cute.struct.MemRange[Float32, 2 * 128 * 8], 16]
            sXa: cute.struct.Align[cute.struct.MemRange[F8, cute.cosize(a_l)], 1024]
            sXb: cute.struct.Align[cute.struct.MemRange[F8, cute.cosize(a_l) if nx == 2 else 1024], 1024]
            sYa: cute.struct.Align[cute.struct.MemRange[F8, cute.cosize(b_l) * (2)], 1024]
            sYb: cute.struct.Align[cute.struct.MemRange[F8, cute.cosize(b_l) if nx == 2 else 1024], 1024]
            sT: cute.struct.Align[cute.struct.MemRange[F8, 1024], 1024]
            sC: cute.struct.Align[cute.struct.MemRange[F8, cute.cosize(c_l)], 1024]
        self.SS = SS
        self.kernel(mma_s, mma_o, tma_Xa, tXa, tma_Xb, tXb, tma_Ya, tYa, tma_Yb, tYb, tma_T, tT, a_l, b_l, c_l, t_l,
                    mQs, mKs, mVs, mdOs, mLse, mDelta, mOut, mIdent, mDump, mCodes, mSB, mHash, n_t, H, Hk).launch(
            grid=[n_t, H, cute.size(mXa, mode=[3])], block=[self.threads, 1, 1], smem=SS.size_in_bytes(), stream=stream)

    @cute.kernel
    def kernel(self, mma_s: cute.TiledMma, mma_o: cute.TiledMma, tma_Xa: cute.CopyAtom, mXa: cute.Tensor,
               tma_Xb: cute.CopyAtom, mXb: cute.Tensor, tma_Ya: cute.CopyAtom, mYa: cute.Tensor, tma_Yb: cute.CopyAtom,
               mYb: cute.Tensor, tma_T: cute.CopyAtom, mT: cute.Tensor, a_l: cute.ComposedLayout,
               b_l: cute.ComposedLayout, c_l: cute.ComposedLayout, t_l: cute.ComposedLayout,
               mQs: cute.Tensor, mKs: cute.Tensor, mVs: cute.Tensor, mdOs: cute.Tensor, mLse: cute.Tensor,
               mDelta: cute.Tensor, mOut: cute.Tensor, mIdent: cute.Tensor, mDump: cute.Tensor, mCodes: cute.Tensor,
               mSB: cute.Tensor, mHash: cute.Tensor, n_t: Int32, H: Int32, Hk: Int32):
        D = self.D
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        rt, hx, b = cute.arch.block_idx()
        if const_expr(self.causal):
            nbz = cute.arch.grid_dim()[2]
            lin = rt + n_t * (hx + H * b)
            rt = n_t - 1 - lin // (H * nbz)
            rr_ = lin - (lin // (H * nbz)) * (H * nbz)
            hx = rr_ % H
            b = rr_ // H
        rep = H // Hk
        N = n_t * 128
        nhl = Int32(1)
        c_lo = Int32(0)
        c_hi = n_t
        if const_expr(self.causal):
            c_hi = rt + 1
        ncs = c_hi - c_lo
        total = nhl * ncs

        smem = cutlass.utils.SmemAllocator()
        st = smem.allocate(self.SS)
        sXa = st.sXa.get_tensor(a_l.outer, swizzle=a_l.inner)
        sXb = st.sXb.get_tensor(a_l.outer, swizzle=a_l.inner)
        sYb = st.sYb.get_tensor(b_l.outer, swizzle=b_l.inner)
        ya2_l = sm100_utils.make_smem_layout_b(mma_s, (self.TM, self.TN, self.D), F8, 2)
        ty2_l = sm100_utils.make_smem_layout_b(mma_o, (self.TM, self.D, self.TN), F8, 2)
        assert cute.cosize(ya2_l) == 2 * cute.cosize(b_l) and cute.cosize(ty2_l) == 2 * cute.cosize(t_l)
        sYa = st.sYa.get_tensor(ya2_l.outer, swizzle=ya2_l.inner)
        sT = st.sYa.get_tensor(ty2_l.outer, swizzle=ty2_l.inner)      # out-B view of the same 2 Ya stages
        sC = st.sC.get_tensor(c_l.outer, swizzle=c_l.inner)
        sC_addr = Int32(st.sC.data_ptr().toint())
        xch_addr = Int32(st.xch.data_ptr().toint())
        bars = st.bars.data_ptr()

        if warp == 9:
            cpasync.prefetch_descriptor(tma_Xa)
            cpasync.prefetch_descriptor(tma_Ya)
            cpasync.prefetch_descriptor(tma_T)
            cpasync.prefetch_descriptor(tma_Xb)
            cpasync.prefetch_descriptor(tma_Yb)
        nx = 2
        thr1 = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        thr256 = pipeline.CooperativeGroup(pipeline.Agent.Thread, 256)
        x_p, x_c = pipeline.PipelineTmaUmma.create(num_stages=1, producer_group=thr1, consumer_group=thr1,
                                                   tx_count=nx * self.bytes_a, barrier_storage=bars).make_participants()
        ya_p, ya_c = pipeline.PipelineTmaUmma.create(num_stages=2, producer_group=thr1, consumer_group=thr1,
                                                     tx_count=self.bytes_b,
                                                     barrier_storage=bars + (16)).make_participants()
        yb_p, yb_c = pipeline.PipelineTmaUmma.create(num_stages=1, producer_group=thr1, consumer_group=thr1,
                                                     tx_count=self.bytes_b, barrier_storage=bars + 4).make_participants()
        t_p, t_c = pipeline.PipelineTmaUmma.create(num_stages=1, producer_group=thr1, consumer_group=thr1,
                                                   tx_count=self.bytes_t, barrier_storage=bars + 6).make_participants()
        sc_p, sc_c = pipeline.PipelineUmmaAsync.create(num_stages=1, producer_group=thr1, consumer_group=thr256,
                                                       barrier_storage=bars + 8).make_participants()
        cd_p, cd_c = pipeline.PipelineAsyncUmma.create(num_stages=2, producer_group=thr256, consumer_group=thr1,
                                                       barrier_storage=bars + 10).make_participants()
        fin_p, fin_c = pipeline.PipelineUmmaAsync.create(num_stages=1, producer_group=thr1, consumer_group=thr256,
                                                         barrier_storage=bars + 14).make_participants()

        tmem = utils.TmemAllocator(st.tmem_hold.ptr,
                                   barrier_for_retrieve=pipeline.NamedBarrier(barrier_id=1, num_threads=self.threads))
        tmem.allocate(512)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(Float32)
        tmem_base = Int32(tmem_ptr.toint())

        thr_s = mma_s.get_slice(0)
        acc_l = mma_s.make_fragment_C(mma_s.partition_shape_C((self.TM, self.TN))).layout
        tCrXa = mma_s.make_fragment_A(sXa)
        tCrXb = mma_s.make_fragment_A(sXb)
        tCrYa = mma_s.make_fragment_B(sYa)
        tCrYb = mma_s.make_fragment_B(sYb)
        tCrC = mma_o.make_fragment_A(sC)
        tCrT = mma_o.make_fragment_B(sT)

        # ------------------------------------------------------------------ TMA warp
        if warp == 9:
            hy0 = hx // rep
            gXa = cute.local_tile(mXa[None, None, hx, b], (self.TM, D), (None, 0))
            tXas, tXag = cpasync.tma_partition(tma_Xa, 0, cute.make_layout(1), cute.group_modes(sXa, 0, 3),
                                               cute.group_modes(thr_s.partition_A(gXa), 0, 3))
            xe = x_p.acquire_and_advance()
            cute.copy(tma_Xa, tXag[(None, rt)], tXas[(None, 0)], tma_bar_ptr=xe.barrier)
            gXb = cute.local_tile(mXb[None, None, hx, b], (self.TM, D), (None, 0))
            tXbs, tXbg = cpasync.tma_partition(tma_Xb, 0, cute.make_layout(1), cute.group_modes(sXb, 0, 3),
                                               cute.group_modes(thr_s.partition_A(gXb), 0, 3))
            cute.copy(tma_Xb, tXbg[(None, rt)], tXbs[(None, 0)], tma_bar_ptr=xe.barrier)
            thr_o = mma_o.get_slice(0)
            for hi in cutlass.range(nhl, unroll=1):
                hy = hy0
                gYa = cute.local_tile(mYa[None, None, hy, b], (self.TN, D), (None, 0))
                tYas, tYag = cpasync.tma_partition(tma_Ya, 0, cute.make_layout(1), cute.group_modes(sYa, 0, 3),
                                                   cute.group_modes(thr_s.partition_B(gYa), 0, 3))
                gYb = cute.local_tile(mYb[None, None, hy, b], (self.TN, D), (None, 0))
                tYbs, tYbg = cpasync.tma_partition(tma_Yb, 0, cute.make_layout(1), cute.group_modes(sYb, 0, 3),
                                                   cute.group_modes(thr_s.partition_B(gYb), 0, 3))
                gT = cute.local_tile(mT[None, None, hy, b], (D, self.TN), (0, None))
                tTs, tTg = cpasync.tma_partition(tma_T, 0, cute.make_layout(1), cute.group_modes(sT, 0, 3),
                                                 cute.group_modes(thr_o.partition_B(gT), 0, 3))
                for ci in cutlass.range(ncs, unroll=1):
                    ct = c_lo + ci
                    ye = ya_p.acquire_and_advance()
                    cute.copy(tma_Ya, tYag[(None, ct)], tYas[(None, ye.index)], tma_bar_ptr=ye.barrier)
                    yb = yb_p.acquire_and_advance()
                    cute.copy(tma_Yb, tYbg[(None, ct)], tYbs[(None, 0)], tma_bar_ptr=yb.barrier)
            ya_p.tail()
            yb_p.tail()

        # ------------------------------------------------------------------ MMA warp
        if warp == 8:
            x_c.wait_and_advance()
            mma_op = cute.make_mma_atom(mma_o.op).op
            accS = cute.make_tensor(tmem_ptr, acc_l)
            ya_o = ya_c.clone()                            # yr: second cursor over the Ya stages, released by out(t-1)
            for t in cutlass.range(total + 1, unroll=1):
                if t < total:
                    yf = ya_c.wait_and_advance()
                    se = sc_p.acquire_and_advance()            # score region free (compute read dP(t-1) / S(t-1))
                    tc_fence_after()                       # BW2
                    mma_s.set(tcgen05.Field.ACCUMULATE, False)
                    for kb in cutlass.range_constexpr(cute.size(tCrXa, mode=[2])):
                        cute.gemm(mma_s, accS, tCrXa[(None, None, kb, 0)], tCrYa[(None, None, kb, yf.index)], accS)
                        mma_s.set(tcgen05.Field.ACCUMULATE, True)
                    se.commit()
                if t > 0:
                    # out(t-1): codes stage / SF set (t-1) & 1; B = K(t-1) from its Ya stage (ya_o cursor)
                    tp = t - 1
                    s = tp & 1
                    yo = ya_o.current_handle()
                    cf = cd_c.wait_and_advance()
                    tc_fence_after()
                    gemm_ptx_bs(
                        mma_op, tmem_base + self.COL_OUT, tCrC[(None, None, None, 0)], tCrT[(None, None, None, 0)],
                        sC[(None, None, None, s)], sT[(None, None, None, yo.index)],
                        tmem_base + self.COL_SF + self.SF_STRIDE * s, tmem_base + self.COL_SF + self.SF_STRIDE * s + self.SFB_OFF,
                        ((0, 0), (0, 1), (4, 2), (4, 3)),
                        tuple((0, 0) for k in range(4)),
                        zero_init=cutlass.Boolean(tp == 0))
                    yo.release()                       # Ya stage free once out(t-1) (and S(t-1) before it) completed
                    ya_o.advance()
                    cf.release()
                if t < total:
                    yg = yb_c.wait_and_advance()
                    de = sc_p.acquire_and_advance()        # compute read S(t)
                    tc_fence_after()                   # BW2
                    mma_s.set(tcgen05.Field.ACCUMULATE, False)
                    for kb in cutlass.range_constexpr(cute.size(tCrXb, mode=[2])):
                        cute.gemm(mma_s, accS, tCrXb[(None, None, kb, 0)], tCrYb[(None, None, kb, 0)], accS)
                        mma_s.set(tcgen05.Field.ACCUMULATE, True)
                    de.commit()
                    yg.release()
            fe = fin_p.acquire_and_advance()
            fe.commit()
            sc_p.tail()
            fin_p.tail()

        # ------------------------------------------------------------------ compute warps
        if warp < 8:
            qd = warp % 4
            hf = warp // 4
            r = qd * 32 + lane
            rtok = rt * 128 + r
            tb = tmem_base + Int32((qd * 32) << 16)
            col0 = hf * 64
            hk_row = hx // rep
            row_s = Float32(mQs[b, hx, rtok // 32])
            row_a = Float32(mdOs[b, hx, rtok // 32])
            l2r = dm_mx256_l2(Float32(mLse[b, hx, rtok]))
            dlr = Float32(mDelta[b, hx, rtok])
            crow = sC_addr + (r // 8) * 1024 + (r % 8) * 128
            rsw = r % 8
            # permanent SFB (1.0) of both SF sets: one writer per lane quadrant (warps 0-3), ordered before the first
            # out MMA of each set by the cd pipeline (fence + arrive below)
            if hf == 0:
                one = Int32(0x7F7F7F7F)
                for s in cutlass.range_constexpr(2):
                    tmem_st_x4(tb + self.COL_SF + self.SF_STRIDE * s + self.SFB_OFF, one, one, one, one)
                    tmem_st_x4(tb + self.COL_SF + self.SF_STRIDE * s + self.SFB_OFF + 4, one, one, one, one)
            nk0 = Float32(mKs[b, hk_row, 2 * hf])
            nk1 = Float32(mKs[b, hk_row, 2 * hf + 1])
            nv0 = Float32(mVs[b, hk_row, 2 * hf])
            nv1 = Float32(mVs[b, hk_row, 2 * hf + 1])
            for t in cutlass.range(total, unroll=1):
                hi = t // ncs
                ct = c_lo + t - hi * ncs
                par = t & 1
                stg = t & 1
                ctok0 = ct * 128
                # per-chunk scales of this thread's two chunks j = 2 hf + jj, and the other operand's sb of all 4 chunks
                cs = [Float32(0.0)] * 2
                ca = [Float32(0.0)] * 2
                osbo = [Int32(0)] * 2         # dqopt: the other operand's sb of this thread's two chunks
                kc = [nk0, nk1]
                vc = [nv0, nv1]
                ctq = cutlass.min(ct + 1, c_hi - 1)          # Q1: issue tile t+1's loads now (clamped; last tile reloads itself)
                nk0 = Float32(mKs[b, hk_row, ctq * 4 + 2 * hf])
                nk1 = Float32(mKs[b, hk_row, ctq * 4 + 2 * hf + 1])
                nv0 = Float32(mVs[b, hk_row, ctq * 4 + 2 * hf])
                nv1 = Float32(mVs[b, hk_row, ctq * 4 + 2 * hf + 1])
                for jj in cutlass.range_constexpr(2):
                    cs[jj] = dm_mx256_c(kc[jj], row_s)
                    ca[jj] = dm_mx256_dp_scale(vc[jj], row_a)
                    osbo[jj] = sb_of(kc[jj])
                xb = xch_addr + par * 1024
                # ---- S(t) -> registers, release the score region ----
                sf = sc_c.wait_and_advance()
                tc_fence_after()                       # BW1
                sv = [tmem_ld32(tb + col0), tmem_ld32(tb + col0 + 32)]
                cute.arch.fence_view_async_tmem_load()
                tc_fence_before()                      # BW2
                sf.release()
                vals = [[Float32(0.0)] * 32, [Float32(0.0)] * 32]       # P (then dS) of chunk jj
                for jj in cutlass.range_constexpr(2):
                    cb = col0 + 32 * jj                                  # first score column of this chunk
                    for i in cutlass.range_constexpr(0, 32, 4):
                        for q2 in cutlass.range_constexpr(0, 4, 2):
                            s0 = sv[jj][i + q2]
                            s1 = sv[jj][i + q2 + 1]
                            if const_expr(self.causal):
                                c0 = ctok0 + cb + i + q2
                                s0 = -Float32.inf if c0 > rtok else s0
                                s1 = -Float32.inf if c0 + 1 > rtok else s1
                            p0, p1 = dm_mx256_p_pair(s0, s1, cs[jj], l2r)
                            vals[jj][i + q2] = p0
                            vals[jj][i + q2 + 1] = p1
                sbs = [Int32(0)] * 2         # this thread's two chunks
                # ---- dP(t) -> registers, release; dS = P * fma(dP, a, -delta) ----
                df = sc_c.wait_and_advance()
                tc_fence_after()                   # BW1
                dvv = [tmem_ld32(tb + col0), tmem_ld32(tb + col0 + 32)]
                cute.arch.fence_view_async_tmem_load()
                tc_fence_before()                  # BW2
                df.release()
                am2 = [Float32(0.0)] * 2
                for jj in cutlass.range_constexpr(2):
                    cb = col0 + 32 * jj
                    for i in cutlass.range_constexpr(0, 32, 4):
                        dlv = [dlr] * 4
                        pv = [vals[jj][i + q] for q in range(4)]
                        for q2 in cutlass.range_constexpr(0, 4, 2):
                            x0, x1 = cute.arch.fma_packed_f32x2((dvv[jj][i + q2], dvv[jj][i + q2 + 1]),
                                                                (ca[jj], ca[jj]), (-dlv[q2], -dlv[q2 + 1]))
                            x0, x1 = cute.arch.mul_packed_f32x2((pv[q2], pv[q2 + 1]), (x0, x1))
                            vals[jj][i + q2] = x0
                            vals[jj][i + q2 + 1] = x1
                    am = Float32(0.0)
                    for i in cutlass.range_constexpr(32):
                        am = cute.arch.fmax(am, abs_f32(vals[jj][i]))
                    am2[jj] = am
                # ---- block max exchange ----
                for jj in cutlass.range_constexpr(2):
                    a_ = redux_max_abs_f32(am2[jj])
                    if lane == 0:
                        st_shared_f32(xb + ((2 * hf + jj) * 4 + qd) * 4, a_)
                # Q2: chunks 2 hf, 2 hf + 1 are reduced over the 4 lane quadrants of THIS half only
                if hf == 0:
                    cute.arch.barrier(barrier_id=5, number_of_threads=128)
                else:
                    cute.arch.barrier(barrier_id=6, number_of_threads=128)
                for jj in cutlass.range_constexpr(2):
                    m4 = ld_shared_v4f(xb + (2 * hf + jj) * 16)
                    sbs[jj] = pow2_sb(max_f32(max_f32(m4[0], m4[1]), max_f32(m4[2], m4[3])))
                ce = cd_p.acquire_and_advance()
                tc_fence_after()                   # BW3
                # ---- codes -> smem stage stg (own two chunks) ----
                cbase = crow + stg * self.bytes_c
                for jj in cutlass.range_constexpr(2):
                    j2 = 2 * hf + jj
                    inv = inv_pow2_from_sb(sbs[jj])
                    for hv in cutlass.range_constexpr(2):
                        w = [None] * 4
                        for u in cutlass.range_constexpr(4):
                            i0 = 16 * hv + 4 * u
                            y0, y1 = cute.arch.mul_packed_f32x2((vals[jj][i0], vals[jj][i0 + 1]), (inv, inv))
                            y2, y3 = cute.arch.mul_packed_f32x2((vals[jj][i0 + 2], vals[jj][i0 + 3]), (inv, inv))
                            w[u] = e4m3x4(y0, y1, y2, y3)
                        st_shared_v4_b32(cbase + (((2 * j2 + hv) ^ rsw) * 16), w[0], w[1], w[2], w[3])
                v0_ = add_sb_clamp(sbs[0], osbo[0])
                v1_ = add_sb_clamp(sbs[1], osbo[1])
                z_ = Int32(0)
                sfw = sel_eq_i32(hf, Int32(0), pack4(v0_, v1_, z_, z_), pack4(z_, z_, v0_, v1_))
                tmem_st_x1(tb + self.COL_SF + self.SF_STRIDE * stg + 4 * hf + qd, sfw)
                cute.arch.fence_view_async_tmem_store()
                cute.arch.fence_view_async_shared()
                tc_fence_before()
                ce.commit()
            # ---- epilogue: out row (256 fp32 in TMEM) -> bf16; this half's 128 columns ----
            fw = fin_c.wait_and_advance()
            tc_fence_after()
            ob = global_address(mOut) + (((Int64(b) * N + rtok) * H + hx) * D) * 2
            for cq in cutlass.range_constexpr(4):
                cblk = 4 * hf + cq
                ov = tmem_ld32(tb + self.COL_OUT + 32 * cblk)
                cute.arch.fence_view_async_tmem_load()
                for i in cutlass.range_constexpr(0, 32, 8):
                    w = [None] * 4
                    for q in cutlass.range_constexpr(4):
                        x0 = ov[i + 2 * q]
                        x1 = ov[i + 2 * q + 1]
                        x0 = mul_f32(x0, Float32(SM))
                        x1 = mul_f32(x1, Float32(SM))
                        w[q] = bf16x2(x0, x1)
                    st_global_v4(ob + (32 * cblk + i) * 2, w[0], w[1], w[2], w[3])
            tc_fence_before()
            fw.release()

        tmem.relinquish_alloc_permit()
        tc_fence_before()                                  # BW9
        cute.arch.barrier(barrier_id=3, number_of_threads=self.threads)
        tc_fence_after()
        tmem.free(tmem_ptr)


class DmHd256Dqg:
    TM, TN, D = 128, 128, 256
    COL_ACC, COL_SF = 0, 256
    MMA_WARP, LOAD_WARP = 4, 5
    S = 3

    def __init__(self, causal: bool, band: bool = False):
        # band (KV2 opt dqc chunks; causal only): this launch forms dQ of q tiles [q0, q0 + n_t) only (n_t argument =
        # the band's tile count), from a band-local W (T = l * tiles(q0, q1) + (ct (ct + 1) - q0 (q0 + 1)) / 2 + kt, kt = 0..ct: every
        # key tile of these q tiles is in the band's W -> complete dQ rows, no carry); dQ rows at their global position (mdQ full N).
        self.band = bool(band)
        assert not self.band or causal
        self.causal = bool(causal)
        self.threads = 192
        self.COL_SFB = self.COL_SF + 4 * self.S

    @cute.jit
    def __call__(self, mA: cute.Tensor, mKt: cute.Tensor, mWsb: cute.Tensor, mKs: cute.Tensor, mdQ: cute.Tensor,
                 n_t: Int32, H: Int32, Hk: Int32, q0: Int32, stream):
        D = self.D
        # static 128 x 128 tile modes (TMA needs a static first mode): W viewed as (q, key, T), q contiguous
        mW = mA
        mA = cute.make_tensor(mA.iterator, cute.make_layout((self.TM, self.TN, cute.size(mA, mode=[2])),
                                                            stride=(1, self.TM, self.TM * self.TN)))
        o_tiler = (self.TM, D, self.TN)               # M = 128 queries, N = 256 d, K = 128 keys
        mma_o = sm100_utils.make_trivial_tiled_mma(F8, F8, tcgen05.OperandMajorMode.MN, tcgen05.OperandMajorMode.MN,
                                                   Float32, tcgen05.CtaGroup.ONE, o_tiler[:2])
        a_l = sm100_utils.make_smem_layout_a(mma_o, o_tiler, F8, self.S)
        t_l = sm100_utils.make_smem_layout_b(mma_o, o_tiler, F8, self.S)
        a1 = cute.select(a_l, mode=[0, 1, 2])
        t1 = cute.select(t_l, mode=[0, 1, 2])
        vmnk = cute.tiled_divide(cute.make_layout((1, 1, 1)), (mma_o.thr_id,))
        op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_A, tA = cute.nvgpu.make_tiled_tma_atom_A(op, mA, a1, o_tiler, mma_o, vmnk.shape)
        tma_T, tT = cute.nvgpu.make_tiled_tma_atom_B(op, mKt, t1, o_tiler, mma_o, vmnk.shape)
        self.bytes_a = cute.size_in_bytes(F8, a1)
        self.bytes_t = cute.size_in_bytes(F8, t1)

        @cute.struct
        class SS:
            bars: cute.struct.MemRange[cutlass.Int64, 16]
            kb: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, 8], 8]   # sff[0..S-1], fin
            tmem_hold: Int32
            sA: cute.struct.Align[cute.struct.MemRange[F8, cute.cosize(a_l)], 1024]
            sT: cute.struct.Align[cute.struct.MemRange[F8, cute.cosize(t_l)], 1024]
        self.SS = SS
        nbz = cute.size(mdQ, mode=[0])
        self.kernel(mma_o, tma_A, tA, tma_T, tT, a_l, t_l, mW, mWsb, mKs, mdQ, n_t, H, Hk, q0).launch(
            grid=[n_t * H * nbz, 1, 1], block=[self.threads, 1, 1], smem=SS.size_in_bytes(), stream=stream)

    @cute.kernel
    def kernel(self, mma_o: cute.TiledMma, tma_A: cute.CopyAtom, mA: cute.Tensor, tma_T: cute.CopyAtom, mKt: cute.Tensor,
               a_l: cute.ComposedLayout, t_l: cute.ComposedLayout, mW: cute.Tensor, mWsb: cute.Tensor, mKs: cute.Tensor, mdQ: cute.Tensor,
               n_t: Int32, H: Int32, Hk: Int32, q0: Int32):
        D = self.D
        S = self.S
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        lin, _, _ = cute.arch.block_idx()
        nbz = cute.arch.grid_dim()[0] // (n_t * H)
        # causal global LPT: longest q tiles first over all (head, batch)
        ct = n_t - 1 - lin // (H * nbz)
        rr_ = lin - (lin // (H * nbz)) * (H * nbz)
        h = rr_ % H
        b = rr_ // H
        hk = h // (H // Hk)
        l = b * H + h
        total = n_t
        tbase = (Int64(l) * n_t + ct) * n_t
        nrow = n_t * 128                 # dQ rows of the sequence (band: the full mdQ length)
        if const_expr(self.band):
            ct = ct + q0
            q1 = q0 + n_t
            total = ct + 1
            tbase = Int64(l) * ((q1 * (q1 + 1) - q0 * (q0 + 1)) // 2) + (ct * (ct + 1) - q0 * (q0 + 1)) // 2
            nrow = cute.size(mdQ, mode=[1])
        elif const_expr(self.causal):
            total = ct + 1
            tbase = Int64(l) * (n_t * (n_t + 1) // 2) + ct * (ct + 1) // 2

        smem = cutlass.utils.SmemAllocator()
        st = smem.allocate(self.SS)
        sA = st.sA.get_tensor(a_l.outer, swizzle=a_l.inner)
        sT = st.sT.get_tensor(t_l.outer, swizzle=t_l.inner)
        bars = st.bars.data_ptr()
        kbar = Int32(st.kb.data_ptr().toint())
        SFF = kbar
        FIN = kbar + 8 * S

        if warp == self.LOAD_WARP:
            cpasync.prefetch_descriptor(tma_A)
            cpasync.prefetch_descriptor(tma_T)
        thr1 = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        ab_p, ab_c = pipeline.PipelineTmaUmma.create(num_stages=S, producer_group=thr1, consumer_group=thr1,
                                                     tx_count=self.bytes_a + self.bytes_t,
                                                     barrier_storage=bars).make_participants()
        if warp == self.MMA_WARP:
            with cute.arch.elect_one():
                for s_ in cutlass.range_constexpr(S):
                    mbar_init(SFF + 8 * s_, 4)
                mbar_init(FIN, 1)
        cute.arch.mbarrier_init_fence()

        tmem = utils.TmemAllocator(st.tmem_hold.ptr,
                                   barrier_for_retrieve=pipeline.NamedBarrier(barrier_id=1, num_threads=self.threads))
        tmem.allocate(512)
        tc_fence_before()            # canonical: alloc -> fence::before -> bar.sync -> fence::after -> ld.shared
        tmem.wait_for_alloc()
        tc_fence_after()
        tmem_ptr = tmem.retrieve_ptr(Float32)
        tmem_base = Int32(tmem_ptr.toint())
        abe = bars + S               # PipelineTmaUmma: full[0..S-1], empty[S..2S-1]
        abe_addr = Int32(abe.toint())

        # ------------------------------------------------------------------ TMA warp
        if warp == self.LOAD_WARP:
            thr_o = mma_o.get_slice(0)
            sA_addr = Int32(st.sA.data_ptr().toint())
            a_stage = cute.cosize(a_l) // S
            wbase = global_address(mW)
            tAs, tAg = cpasync.tma_partition(tma_A, 0, cute.make_layout(1), cute.group_modes(sA, 0, 3),
                                             cute.group_modes(thr_o.partition_A(mA), 0, 3))
            gT = cute.local_tile(mKt[None, None, hk, b], (D, self.TN), (0, None))
            tTs, tTg = cpasync.tma_partition(tma_T, 0, cute.make_layout(1), cute.group_modes(sT, 0, 3),
                                             cute.group_modes(thr_o.partition_B(gT), 0, 3))
            for kt in cutlass.range(total, unroll=1):
                e = ab_p.acquire_and_advance()
                if lane == 0:
                    bulk_g2s(sA_addr + e.index * a_stage, wbase + (tbase + kt) * 16384, 16384, Int32(e.barrier.toint()))
                cute.copy(tma_T, tTg[(None, kt)], tTs[(None, e.index)], tma_bar_ptr=e.barrier)
            ab_p.tail()

        # ------------------------------------------------------------------ MMA warp
        if warp == self.MMA_WARP:
            mma_op = cute.make_mma_atom(mma_o.op).op
            tCrA = mma_o.make_fragment_A(sA)
            tCrT = mma_o.make_fragment_B(sT)
            for kt in cutlass.range(total, unroll=1):
                s = kt % S
                f = ab_c.wait_and_advance()
                mbar_wait_parity(SFF + 8 * s, (kt // S) & 1)
                tc_fence_after()
                gemm_ptx_bs(
                    mma_op, tmem_base + self.COL_ACC, tCrA[(None, None, None, f.index)], tCrT[(None, None, None, f.index)],
                    sA[(None, None, None, f.index)], sT[(None, None, None, f.index)],
                    tmem_base + self.COL_SF + 4 * s, tmem_base + self.COL_SFB,
                    tuple((0, k) for k in range(4)), tuple((0, 0) for k in range(4)),
                    zero_init=cutlass.Boolean(kt == 0))
                f.release()      # tcgen05.commit -> ab-empty[s]: covers this stage's 4 MMAs
            with cute.arch.elect_one():
                tc_commit(FIN)

        # ------------------------------------------------------------------ SF writers + epilogue (warps 0-3)
        if warp < 4:
            tb = tmem_base + Int32((warp * 32) << 16)
            # SFB = 1.0 (0x7F), once; published by the first sff arrive below (same threads, program order + fences)
            onew = Int32(0x7F7F7F7F)
            tmem_st_x4(tb + self.COL_SFB, onew, onew, onew, onew)       # N = 256: 8 SFB columns (as Mx2)
            tmem_st_x4(tb + self.COL_SFB + 4, onew, onew, onew, onew)
            wsb_base = global_address(mWsb)
            wnext = ld_global_u32(wsb_base + tbase * 4)
            for kt in cutlass.range(total, unroll=1):
                s = kt % S
                w = wnext
                kv = [Float32(0.0)] * 4
                for k in cutlass.range_constexpr(4):
                    kv[k] = Float32(mKs[b, hk, kt * 4 + k])
                if kt + 1 < total:
                    wnext = ld_global_u32(wsb_base + (tbase + kt + 1) * 4)
                sfa = pack4(add_sb_clamp(byte_of(w, 0), sb_of(kv[0])),
                            add_sb_clamp(byte_of(w, 1), sb_of(kv[1])),
                            add_sb_clamp(byte_of(w, 2), sb_of(kv[2])),
                            add_sb_clamp(byte_of(w, 3), sb_of(kv[3])))
                if kt >= S:
                    mbar_wait_parity(abe_addr + 8 * s, ((kt // S) - 1) & 1)   # MMA(kt - S) complete
                    tc_fence_after()
                tmem_st_x1(tb + self.COL_SF + 4 * s + warp, sfa)       # SFA of rows 32 w.. at column + w (as Mx2)
                cute.arch.fence_view_async_tmem_store()
                tc_fence_before()
                warp_arrive(SFF + 8 * s)
            # ---- epilogue: row 32 w + lane, 256 fp32 -> x 2^-4 -> bf16 ----
            mbar_wait_parity(FIN, Int32(0))
            tc_fence_after()
            q = ct * 128 + warp * 32 + lane
            ob = global_address(mdQ) + ((Int64(b) * nrow + q) * H + h) * (D * 2)
            for cb in cutlass.range_constexpr(8):
                ov = tmem_ld32(tb + self.COL_ACC + 32 * cb)
                cute.arch.fence_view_async_tmem_load()
                for i in cutlass.range_constexpr(0, 32, 8):
                    wv = [None] * 4
                    for qq in cutlass.range_constexpr(4):
                        wv[qq] = bf16x2(mul_f32(ov[i + 2 * qq], Float32(SM)), mul_f32(ov[i + 2 * qq + 1], Float32(SM)))
                    st_global_v4(ob + (32 * cb + i) * 2, wv[0], wv[1], wv[2], wv[3])
            tc_fence_before()
        # teardown: every TMEM reader / MMA issuer fences before the barrier, the deallocator after
        if warp == self.MMA_WARP:
            tc_fence_before()
        tmem.relinquish_alloc_permit()
        cute.arch.barrier(barrier_id=2, number_of_threads=self.threads)
        tc_fence_after()
        tmem.free(tmem_ptr)


LOCK = threading.Lock()
CACHE = {}


def compile_cache():
    if "c" not in CACHE:
        if cache_utils.CUTE_DSL_CACHE_ENABLED:
            CACHE["c"] = cache_utils.JITPersistentCache(cache_utils.get_cache_path() / "dm_sm100" / fingerprint() / "bwdhmx")
        else:
            CACHE["c"] = cache_utils.JITCache()
    return CACHE["c"]


def as_bytes(t):
    return t.view(torch.uint8) if t.dtype == torch.float8_e4m3fn else t


def run_kernel(key, make_kernel, tensors, leading_dims, ints):
    """
    Compile (once per key) and launch; under FakeTensorMode only compile.
    """
    cache = compile_cache()
    with LOCK:
        if key not in cache:
            args = [to_cute_tensor(t, assumed_align=16 if t.element_size() <= 2 else 4, leading_dim=ld) for t, ld in zip(tensors, leading_dims)]
            stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
            cache[key] = cute.compile(make_kernel(), *args, *[Int32(0) for _ in ints], stream, options="--enable-tvm-ffi")
    if not is_fake_mode():
        cache[key](*[as_bytes(t) for t in tensors], *ints)


PLACEHOLDERS = {}


def placeholders(device):
    """
    Per-device stand-ins for the kernel's unused debug tensor arguments (never read or written).
    """
    key = str(device)
    if key not in PLACEHOLDERS:
        PLACEHOLDERS[key] = (torch.zeros(1, 1, 4, device=device, dtype=torch.float64), torch.zeros(1, 1, 1, 4, device=device, dtype=torch.float32),
                             torch.zeros(1, 1, 16, device=device, dtype=torch.uint8), torch.zeros(1, 1, 4, device=device, dtype=torch.int32),
                             torch.zeros(1, 1, 4, device=device, dtype=torch.int64))
    return PLACEHOLDERS[key]


def dq_recompute_launch(q8, qs, k8, ks, v8, vs, do8, dos, lse, delta, causal):
    """
    dQ (B, N, H, 256) bf16, recomputing S and dP (DmHd256Mx2). Inputs as attention_bwd_hd256.
    """
    B, N, H, _ = q8.shape
    Hk = k8.shape[2]
    assert N % 128 == 0 and H % Hk == 0
    assert qs.shape == (B, H, N // 32) and ks.shape == (B, Hk, N // 32) and dos.shape == qs.shape and vs.shape == ks.shape
    causal = bool(causal)
    f8 = torch.float8_e4m3fn

    def k_major(x):
        return x.view(f8).permute(1, 3, 2, 0)          # (B, N, h, D) -> (N, D, h, B)

    def mn_major(x):
        return x.view(f8).permute(3, 1, 2, 0)          # (B, N, h, 256) -> (256, N, h, B): MN-major B of the out MMA
    dq = torch.empty(B, N, H, 256, device=q8.device, dtype=torch.bfloat16)
    ident, dump, codes, sb, hashes = placeholders(q8.device)
    tensors = [k_major(q8), k_major(do8), k_major(k8), k_major(v8), mn_major(k8), qs, ks, vs, dos, lse, delta, dq, ident, dump, codes, sb, hashes]
    leading_dims = [1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 3, 2, 3, 2, 2, 2]
    run_kernel(("i13", "dq", causal, 0), lambda: DmHd256Mx2(causal), tensors, leading_dims, [N // 128, H, Hk])
    return dq


def dq_workspace_launch(W, Wsb, k8, ks, B, N, H, causal, *, out=None, band=None):
    """
    dQ (B, N, H, 256) bf16 from the workspace (W codes, Wsb block exponents) and K (DmHd256Dqg). With
    band = (q_start, q_end), only those q tiles, from a band-local workspace, written at their global
    rows.
    """
    Hk = k8.shape[2]
    f8 = torch.float8_e4m3fn
    dq = torch.empty(B, N, H, 256, device=k8.device, dtype=torch.bfloat16) if out is None else out
    assert dq.shape == (B, N, H, 256) and dq.is_contiguous()
    mA = W.view(f8).permute(2, 1, 0)                       # (T, key, q) -> (q, key, T): MN-major A (queries contiguous)
    mKt = k8.view(f8).permute(3, 1, 2, 0)                  # (B, N, Hk, 256) -> (256, N, Hk, B): MN-major B
    has_band = band is not None
    num_q_tiles, q_start = (band[1] - band[0], band[0]) if has_band else (N // 128, 0)
    key = ("dqg", bool(causal), 3, "", True) + (("band",) if has_band else ())
    run_kernel(key, lambda: DmHd256Dqg(bool(causal), has_band), [mA, mKt, Wsb, ks, dq], [0, 0, 2, 2, 3], [num_q_tiles, H, Hk, q_start])
    return dq
