"""hd256 MXFP8 backward, schedule O1 ("A1", fork hd256-bwdopt; kda/hd256/bwd-plan.md candidate queue,
codex/hd256-bwdopt-01-design.md).  Same numerics as DmHd256Mx (dm_bwd_hd256mx_sm100.py, 14df86c), bitwise: every S / dP
is the same M=128 N=128 K=256 kind::f8f6f4 chain (8 ascending K=32 steps, zero init), P / dS via the same dm_mx256_* helpers
in the same fp32 op order, the same quantization blocks (dv: P^T per (key, 32 q); dk / dq: one pow2 per (128 q x 32 keys),
an order-free max), the same out MMA (kind::mxf8f6f4 M=128 N=256, 4 ascending K=32 steps, zero-init on t == 0), the same
bf16 epilogue.  Only the schedule / ownership changes:

  TMEM: cols 0-127 ONE score region, used S(t) then dP(t) (dk / dq) or S(t) only (dv); cols 128+16s .. +11 SF set s
        (SFA 4 cols, SFB 8 cols of constant 0x7F written once), s = t & 1; cols 256-511 out accumulator.
  Warps: 0-7 compute (warp w: TMEM lane quadrant w % 4, row r = 32 (w % 4) + lane, score columns 64 (w // 4) .. +63 =
         quantization chunks j = 2 (w // 4) + {0, 1}); 8 MMA; 9 TMA.  Each thread holds its 64 P / dS in registers (no stash).
  MMA order per tile: S(t) -> out(t-1) -> dP(t) (dk / dq), S(t) -> out(t-1) (dv).  No od barrier: SFs never alias scores.
  Hand-offs (each with a positive-control knob, see KNOBS):
    sc  score region: MMA S/dP -> 256 compute threads tcgen05.ld -> release (sc pipeline, 1 stage, used twice per tile)
    cd  codes SMEM stage + SFA set (t & 1): compute -> out(t) MMA (cd pipeline, 2 stages, released by UMMA commit)
    ld  l2 / delta staging (keys-on-rows): stage t+1 written before the per-tile barrier of t (parity buffers)
    xc  per-tile exchange slots (dk |dS| max per row half, dq per-chunk warp max, dv sbs of the upper half) -> barrier
    fin out accumulator -> epilogue
dbg bit 1 (dq): identity sums [sum dS, sum |P dP a|, sum P] (fp32 per 32-key chunk, fp64 over chunks; the two column halves
  combined at the end) -> mIdent.  dbg bit 2: dumps exactly as DmHd256Mx (S_acc, dP_acc, P, dS per (q, k); codes; sb).
"""
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Float32, Float64, Int32, Int64, const_expr
from cutlass.cute.nvgpu import cpasync, tcgen05

from delta_matching.sm100 import dm_common_hd256 as dmh
from delta_matching.sm100 import dm_utils
from delta_matching.sm100.dm_bwd_hd256_sm100 import (tmem_ld32, e4m3x4, bf16x2, st_global_v4, st_global_f32,
                                                     ld_shared_f32, st_shared_f32, abs_f32, max_f32, add_f32, mul_f32,
                                                     _gaddr)
from delta_matching.sm100.dm_bwd_hd256_sm100 import ld_shared_v4 as ld_shared_v4i
from delta_matching.sm100.dm_bwd_hd256mx_sm100 import (st_shared_v4, ld_shared_v4f, spin, sb_of, pack4, tc_fence_before,
                                                       tc_fence_after, st_global_u8, st_global_f64x2, f2i, hmix, hidx,
                                                       zext64, red_add_u64, red_add_f64, lk_cols_u32, lk_cols_f64,
                                                       sel_eq_i32)

F8 = cutlass.Float8E4M3FN
SM = 0.0625


class DmHd256Mx2:
    TM, TN, D = 128, 128, 256
    COL_SF, SF_STRIDE, COL_OUT = 128, 16, 256
    KNOBS = ("sc", "cd", "ld", "xc", "fin")
    DLY = 1 << 15

    def __init__(self, mode: str, causal: bool, dbg: int = 0, knob: str = "", ordf: bool = False):
        """ordf: tcgen05 thread-sync fence pairs on every cross-thread TMEM hand-off (ordinv sites BW1-BW3, BW9 mapped to this
        schedule: sc full -> ld (after), ld -> sc release (before), sc acquire -> S/dP MMA (after), cd acquire -> SF st (after),
        epilogue / teardown); host default True (DM_HD256_ORDF), as DmHd256Mx."""
        self.ordf = bool(ordf)
        assert mode in ("dv", "dk", "dq")
        assert not (dbg & 4)
        self.kn = knob.replace("_safe", "")
        self.unsafe = bool(knob) and not knob.endswith("_safe")
        assert self.kn in ("",) + self.KNOBS, knob
        self.mode = mode
        self.causal = bool(causal)
        self.dbg = int(dbg)
        self.has_dp = mode != "dv"
        self.rows_are_keys = mode != "dq"
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
        nx = 2 if const_expr(self.has_dp) else 1

        @cute.struct
        class SS:
            bars: cute.struct.MemRange[cutlass.Int64, 24]
            tmem_hold: Int32
            xch: cute.struct.Align[cute.struct.MemRange[Float32, 2 * 2 * 128], 16]
            sLD: cute.struct.Align[cute.struct.MemRange[Float32, 2 * 2 * 128], 16]
            sId: cute.struct.Align[cute.struct.MemRange[Float32, 2 * 128 * 8], 16]
            sXa: cute.struct.Align[cute.struct.MemRange[F8, cute.cosize(a_l)], 1024]
            sXb: cute.struct.Align[cute.struct.MemRange[F8, cute.cosize(a_l) if nx == 2 else 1024], 1024]
            sYa: cute.struct.Align[cute.struct.MemRange[F8, cute.cosize(b_l)], 1024]
            sYb: cute.struct.Align[cute.struct.MemRange[F8, cute.cosize(b_l) if nx == 2 else 1024], 1024]
            sT: cute.struct.Align[cute.struct.MemRange[F8, cute.cosize(t_l)], 1024]
            sC: cute.struct.Align[cute.struct.MemRange[F8, cute.cosize(c_l)], 1024]
        self.SS = SS
        nh = Hk if const_expr(self.rows_are_keys) else H
        self.kernel(mma_s, mma_o, tma_Xa, tXa, tma_Xb, tXb, tma_Ya, tYa, tma_Yb, tYb, tma_T, tT, a_l, b_l, c_l, t_l,
                    mQs, mKs, mVs, mdOs, mLse, mDelta, mOut, mIdent, mDump, mCodes, mSB, mHash, n_t, H, Hk).launch(
            grid=[n_t, nh, cute.size(mXa, mode=[3])], block=[self.threads, 1, 1], smem=SS.size_in_bytes(), stream=stream)

    @cute.kernel
    def kernel(self, mma_s: cute.TiledMma, mma_o: cute.TiledMma, tma_Xa: cute.CopyAtom, mXa: cute.Tensor,
               tma_Xb: cute.CopyAtom, mXb: cute.Tensor, tma_Ya: cute.CopyAtom, mYa: cute.Tensor, tma_Yb: cute.CopyAtom,
               mYb: cute.Tensor, tma_T: cute.CopyAtom, mT: cute.Tensor, a_l: cute.ComposedLayout,
               b_l: cute.ComposedLayout, c_l: cute.ComposedLayout, t_l: cute.ComposedLayout,
               mQs: cute.Tensor, mKs: cute.Tensor, mVs: cute.Tensor, mdOs: cute.Tensor, mLse: cute.Tensor,
               mDelta: cute.Tensor, mOut: cute.Tensor, mIdent: cute.Tensor, mDump: cute.Tensor, mCodes: cute.Tensor,
               mSB: cute.Tensor, mHash: cute.Tensor, n_t: Int32, H: Int32, Hk: Int32):
        D = self.D
        tidx = cute.arch.thread_idx()[0]
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        rt, hx, b = cute.arch.block_idx()
        rep = H // Hk
        N = n_t * 128
        nhl = Int32(1)
        if const_expr(self.rows_are_keys):
            nhl = rep
        c_lo = Int32(0)
        c_hi = n_t
        if const_expr(self.causal):
            if const_expr(self.rows_are_keys):
                c_lo = rt
            else:
                c_hi = rt + 1
        ncs = c_hi - c_lo
        total = nhl * ncs

        smem = cutlass.utils.SmemAllocator()
        st = smem.allocate(self.SS)
        sXa = st.sXa.get_tensor(a_l.outer, swizzle=a_l.inner)
        sXb = st.sXb.get_tensor(a_l.outer, swizzle=a_l.inner)
        sYa = st.sYa.get_tensor(b_l.outer, swizzle=b_l.inner)
        sYb = st.sYb.get_tensor(b_l.outer, swizzle=b_l.inner)
        sT = st.sT.get_tensor(t_l.outer, swizzle=t_l.inner)
        sC = st.sC.get_tensor(c_l.outer, swizzle=c_l.inner)
        sC_addr = Int32(st.sC.data_ptr().toint())
        sLD_addr = Int32(st.sLD.data_ptr().toint())
        xch_addr = Int32(st.xch.data_ptr().toint())
        sId_addr = Int32(st.sId.data_ptr().toint())
        bars = st.bars.data_ptr()

        if warp == 9:
            cpasync.prefetch_descriptor(tma_Xa)
            cpasync.prefetch_descriptor(tma_Ya)
            cpasync.prefetch_descriptor(tma_T)
            if const_expr(self.has_dp):
                cpasync.prefetch_descriptor(tma_Xb)
                cpasync.prefetch_descriptor(tma_Yb)
        nx = 2 if const_expr(self.has_dp) else 1
        thr1 = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        thr256 = pipeline.CooperativeGroup(pipeline.Agent.Thread, 256)
        x_p, x_c = pipeline.PipelineTmaUmma.create(num_stages=1, producer_group=thr1, consumer_group=thr1,
                                                   tx_count=nx * self.bytes_a, barrier_storage=bars).make_participants()
        ya_p, ya_c = pipeline.PipelineTmaUmma.create(num_stages=1, producer_group=thr1, consumer_group=thr1,
                                                     tx_count=self.bytes_b, barrier_storage=bars + 2).make_participants()
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
            hy0 = hx
            if const_expr(not self.rows_are_keys):
                hy0 = hx // rep
            gXa = cute.local_tile(mXa[None, None, hx, b], (self.TM, D), (None, 0))
            tXas, tXag = cpasync.tma_partition(tma_Xa, 0, cute.make_layout(1), cute.group_modes(sXa, 0, 3),
                                               cute.group_modes(thr_s.partition_A(gXa), 0, 3))
            xe = x_p.acquire_and_advance()
            cute.copy(tma_Xa, tXag[(None, rt)], tXas[(None, 0)], tma_bar_ptr=xe.barrier)
            if const_expr(self.has_dp):
                gXb = cute.local_tile(mXb[None, None, hx, b], (self.TM, D), (None, 0))
                tXbs, tXbg = cpasync.tma_partition(tma_Xb, 0, cute.make_layout(1), cute.group_modes(sXb, 0, 3),
                                                   cute.group_modes(thr_s.partition_A(gXb), 0, 3))
                cute.copy(tma_Xb, tXbg[(None, rt)], tXbs[(None, 0)], tma_bar_ptr=xe.barrier)
            thr_o = mma_o.get_slice(0)
            for hi in cutlass.range(nhl, unroll=1):
                hy = hy0
                if const_expr(self.rows_are_keys):
                    hy = hx * rep + hi
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
                    cute.copy(tma_Ya, tYag[(None, ct)], tYas[(None, 0)], tma_bar_ptr=ye.barrier)
                    if const_expr(self.has_dp):
                        yb = yb_p.acquire_and_advance()
                        cute.copy(tma_Yb, tYbg[(None, ct)], tYbs[(None, 0)], tma_bar_ptr=yb.barrier)
                    te = t_p.acquire_and_advance()
                    cute.copy(tma_T, tTg[(None, ct)], tTs[(None, 0)], tma_bar_ptr=te.barrier)
            ya_p.tail()
            if const_expr(self.has_dp):
                yb_p.tail()
            t_p.tail()

        # ------------------------------------------------------------------ MMA warp
        if warp == 8:
            x_c.wait_and_advance()
            mma_op = cute.make_mma_atom(mma_o.op).op
            accS = cute.make_tensor(tmem_ptr, acc_l)
            for t in cutlass.range(total + 1, unroll=1):
                if t < total:
                    yf = ya_c.wait_and_advance()
                    se = sc_p.acquire_and_advance()            # score region free (compute read dP(t-1) / S(t-1))
                    if const_expr(self.ordf):
                        tc_fence_after()                       # BW2
                    mma_s.set(tcgen05.Field.ACCUMULATE, False)
                    for kb in cutlass.range_constexpr(cute.size(tCrXa, mode=[2])):
                        cute.gemm(mma_s, accS, tCrXa[(None, None, kb, 0)], tCrYa[(None, None, kb, 0)], accS)
                        mma_s.set(tcgen05.Field.ACCUMULATE, True)
                    se.commit()
                    yf.release()
                if t > 0:
                    # out(t-1): codes stage / SF set (t-1) & 1
                    if const_expr(self.kn in ("cd", "fin")):
                        spin(self.DLY)
                    tp = t - 1
                    s = tp & 1
                    if const_expr(self.kn == "cd" and self.unsafe):
                        s = tp * 0
                    tf = t_c.wait_and_advance()
                    cf = cd_c.wait_and_advance()
                    tc_fence_after()
                    dm_utils.gemm_ptx_bs(
                        mma_op, tmem_base + self.COL_OUT, tCrC[(None, None, None, 0)], tCrT[(None, None, None, 0)],
                        sC[(None, None, None, s)], sT[(None, None, None, 0)],
                        tmem_base + self.COL_SF + self.SF_STRIDE * s, tmem_base + self.COL_SF + self.SF_STRIDE * s + 4,
                        tuple((0, k) for k in range(4)), tuple((0, 0) for k in range(4)),
                        zero_init=cutlass.Boolean(tp == 0))
                    tf.release()
                    cf.release()
                if const_expr(self.has_dp):
                    if t < total:
                        yg = yb_c.wait_and_advance()
                        de = sc_p.acquire_and_advance()        # compute read S(t)
                        if const_expr(self.ordf):
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
            ctid = tidx                 # 0..255
            tb = tmem_base + Int32((qd * 32) << 16)
            col0 = hf * 64
            hk_row = hx
            if const_expr(not self.rows_are_keys):
                hk_row = hx // rep
            row_s = Float32(0.0)
            row_a = Float32(0.0)
            l2r = Float32(0.0)
            dlr = Float32(0.0)
            if const_expr(self.rows_are_keys):
                row_s = Float32(mKs[b, hx, rtok // 32])
                row_a = Float32(mVs[b, hx, rtok // 32])
            else:
                row_s = Float32(mQs[b, hx, rtok // 32])
                row_a = Float32(mdOs[b, hx, rtok // 32])
                l2r = dmh.dm_mx256_l2(Float32(mLse[b, hx, rtok]))
                dlr = Float32(mDelta[b, hx, rtok])
            idn = Float64(0.0)
            idd = Float64(0.0)
            idp = Float64(0.0)
            lk_p = Int64(0)
            lk_s = Int64(0)
            lk_c = Int64(0)
            lk_r = Float64(0.0)
            crow = sC_addr + (r // 8) * 1024 + (r % 8) * 128
            rsw = r % 8
            # permanent SFB (1.0) of both SF sets: one writer per lane quadrant (warps 0-3), ordered before the first
            # out MMA of each set by the cd pipeline (fence + arrive below)
            if hf == 0:
                one = Int32(0x7F7F7F7F)
                for s in cutlass.range_constexpr(2):
                    dm_utils.tmem_st_x4(tb + self.COL_SF + self.SF_STRIDE * s + 4, one, one, one, one)
                    dm_utils.tmem_st_x4(tb + self.COL_SF + self.SF_STRIDE * s + 8, one, one, one, one)
            # keys-on-rows: stage l2 / delta of tile 0 (threads 0-127 l2, 128-255 delta)
            if const_expr(self.rows_are_keys):
                h0 = hx * rep
                if ctid < 128:
                    st_shared_f32(sLD_addr + ctid * 4, dmh.dm_mx256_l2(Float32(mLse[b, h0, c_lo * 128 + ctid])))
                else:
                    st_shared_f32(sLD_addr + ctid * 4, Float32(mDelta[b, h0, c_lo * 128 + ctid - 128]))
                cute.arch.barrier(barrier_id=2, number_of_threads=256)
            for t in cutlass.range(total, unroll=1):
                hi = t // ncs
                ct = c_lo + t - hi * ncs
                h = hx
                if const_expr(self.rows_are_keys):
                    h = hx * rep + hi
                l = b * H + h
                par = t & 1
                stg = t & 1
                if const_expr(self.kn == "cd" and self.unsafe):
                    stg = t * 0
                ctok0 = ct * 128
                # per-chunk scales of this thread's two chunks j = 2 hf + jj, and the other operand's sb of all 4 chunks
                cs = [Float32(0.0)] * 2
                ca = [Float32(0.0)] * 2
                osb = [Int32(0)] * 4
                for jj in cutlass.range_constexpr(2):
                    cj = ct * 4 + 2 * hf + jj
                    if const_expr(self.rows_are_keys):
                        qsj = Float32(mQs[b, h, cj])
                        dosj = Float32(mdOs[b, h, cj])
                        cs[jj] = dmh.dm_mx256_c(row_s, qsj)
                        ca[jj] = dmh.dm_mx256_dp_scale(row_a, dosj)
                    else:
                        ksj = Float32(mKs[b, hk_row, cj])
                        vsj = Float32(mVs[b, hk_row, cj])
                        cs[jj] = dmh.dm_mx256_c(ksj, row_s)
                        ca[jj] = dmh.dm_mx256_dp_scale(vsj, row_a)
                for j in cutlass.range_constexpr(4):
                    if const_expr(self.mode == "dv"):
                        osb[j] = sb_of(Float32(mdOs[b, h, ct * 4 + j]))
                    elif const_expr(self.mode == "dk"):
                        osb[j] = sb_of(Float32(mQs[b, h, ct * 4 + j]))
                    else:
                        osb[j] = sb_of(Float32(mKs[b, hk_row, ct * 4 + j]))
                ldb = sLD_addr + par * 1024
                ldn = sLD_addr + (1 - par) * 1024
                if const_expr(self.kn == "ld" and self.unsafe):
                    ldb = sLD_addr
                    ldn = sLD_addr
                xb = xch_addr + par * 1024
                # ---- S(t) -> registers, release the score region ----
                sf = sc_c.wait_and_advance()
                if const_expr(self.ordf):
                    tc_fence_after()                           # BW1
                if const_expr(self.kn == "sc"):
                    if const_expr(self.unsafe):
                        sf.release()
                    spin(self.DLY)
                sv = [tmem_ld32(tb + col0), tmem_ld32(tb + col0 + 32)]
                cute.arch.fence_view_async_tmem_load()
                if const_expr(not (self.kn == "sc" and self.unsafe)):
                    if const_expr(self.ordf):
                        tc_fence_before()                      # BW2
                    sf.release()
                if const_expr(self.kn == "ld"):
                    if warp == 3:
                        spin(self.DLY)
                vals = [[Float32(0.0)] * 32, [Float32(0.0)] * 32]       # P (then dS) of chunk jj
                for jj in cutlass.range_constexpr(2):
                    cb = col0 + 32 * jj                                  # first score column of this chunk
                    for i in cutlass.range_constexpr(0, 32, 4):
                        l2v = [l2r] * 4
                        if const_expr(self.rows_are_keys):
                            l2v = ld_shared_v4f(ldb + (cb + i) * 4)
                        for q2 in cutlass.range_constexpr(0, 4, 2):
                            s0 = sv[jj][i + q2]
                            s1 = sv[jj][i + q2 + 1]
                            if const_expr(self.causal):
                                c0 = ctok0 + cb + i + q2
                                if const_expr(self.rows_are_keys):
                                    s0 = -Float32.inf if rtok > c0 else s0
                                    s1 = -Float32.inf if rtok > c0 + 1 else s1
                                else:
                                    s0 = -Float32.inf if c0 > rtok else s0
                                    s1 = -Float32.inf if c0 + 1 > rtok else s1
                            if const_expr(self.rows_are_keys):
                                p0, p1 = dmh.dm_mx256_p_pair2(s0, s1, cs[jj], l2v[q2], l2v[q2 + 1])
                            else:
                                p0, p1 = dmh.dm_mx256_p_pair(s0, s1, cs[jj], l2r)
                            vals[jj][i + q2] = p0
                            vals[jj][i + q2 + 1] = p1
                if const_expr((self.dbg & 8) and self.mode != "dk"):
                    for jj in cutlass.range_constexpr(2):
                        cb = col0 + 32 * jj
                        hpl = [Int32(0)] * 32
                        for i in cutlass.range_constexpr(32):
                            if const_expr(self.rows_are_keys):
                                hpl[i] = hmix(f2i(vals[jj][i]), hidx(ctok0 + cb + i, rtok))
                            else:
                                hpl[i] = hmix(f2i(vals[jj][i]), hidx(rtok, ctok0 + cb + i))
                        if const_expr(self.rows_are_keys):
                            lk_cols_u32(hpl, lane, _gaddr(mHash) + (Int64(l) * N + ctok0 + cb) * 32)
                        else:
                            hs_ = Int32(0)
                            for i in cutlass.range_constexpr(32):
                                hs_ = hs_ + hpl[i]
                            lk_p = lk_p + zext64(hs_)
                if const_expr(self.dbg & 2):
                    for jj in cutlass.range_constexpr(2):
                        cb = col0 + 32 * jj
                        dstp = Int64(16)
                        dpa = _gaddr(mDump) + ((Int64(l) * N + rtok) * N + ctok0 + cb) * 16
                        if const_expr(self.rows_are_keys):
                            dstp = Int64(N) * 16
                            dpa = _gaddr(mDump) + ((Int64(l) * N + ctok0 + cb) * N + rtok) * 16
                        for i in cutlass.range_constexpr(32):
                            st_global_f32(dpa, sv[jj][i])
                            if const_expr(self.mode != "dk"):
                                st_global_f32(dpa + 8, vals[jj][i])
                            dpa = dpa + dstp
                sbs = [Int32(0)] * 2         # this thread's two chunks
                ce = None
                if const_expr(self.has_dp):
                    # ---- dP(t) -> registers, release; dS = P * fma(dP, a, -delta) ----
                    df = sc_c.wait_and_advance()
                    if const_expr(self.ordf):
                        tc_fence_after()                       # BW1
                    dvv = [tmem_ld32(tb + col0), tmem_ld32(tb + col0 + 32)]
                    cute.arch.fence_view_async_tmem_load()
                    if const_expr(self.ordf):
                        tc_fence_before()                      # BW2
                    df.release()
                    am2 = [Float32(0.0)] * 2
                    idc = [[Float32(0.0)] * 3, [Float32(0.0)] * 3]
                    for jj in cutlass.range_constexpr(2):
                        cb = col0 + 32 * jj
                        id1 = Float32(0.0)
                        id2 = Float32(0.0)
                        id3 = Float32(0.0)
                        for i in cutlass.range_constexpr(0, 32, 4):
                            dlv = [dlr] * 4
                            if const_expr(self.rows_are_keys):
                                dlv = ld_shared_v4f(ldb + 512 + (cb + i) * 4)
                            pv = [vals[jj][i + q] for q in range(4)]
                            for q2 in cutlass.range_constexpr(0, 4, 2):
                                x0, x1 = cute.arch.fma_packed_f32x2((dvv[jj][i + q2], dvv[jj][i + q2 + 1]),
                                                                    (ca[jj], ca[jj]), (-dlv[q2], -dlv[q2 + 1]))
                                x0, x1 = cute.arch.mul_packed_f32x2((pv[q2], pv[q2 + 1]), (x0, x1))
                                vals[jj][i + q2] = x0
                                vals[jj][i + q2 + 1] = x1
                            if const_expr((self.dbg & 1) and self.mode == "dq"):
                                for q in cutlass.range_constexpr(4):
                                    id1 = add_f32(id1, vals[jj][i + q])
                                    id2 = add_f32(id2, abs_f32(mul_f32(pv[q], mul_f32(dvv[jj][i + q], ca[jj]))))
                                    id3 = add_f32(id3, pv[q])
                        idc[jj] = [id1, id2, id3]
                        am = Float32(0.0)
                        for i in cutlass.range_constexpr(32):
                            am = cute.arch.fmax(am, abs_f32(vals[jj][i]))
                        am2[jj] = am
                    if const_expr(self.dbg & 2):
                        for jj in cutlass.range_constexpr(2):
                            cb = col0 + 32 * jj
                            dstp = Int64(16)
                            dpa = _gaddr(mDump) + ((Int64(l) * N + rtok) * N + ctok0 + cb) * 16
                            if const_expr(self.rows_are_keys):
                                dstp = Int64(N) * 16
                                dpa = _gaddr(mDump) + ((Int64(l) * N + ctok0 + cb) * N + rtok) * 16
                            for i in cutlass.range_constexpr(32):
                                st_global_f32(dpa + 4, dvv[jj][i])
                                st_global_f32(dpa + 12, vals[jj][i])
                                dpa = dpa + dstp
                    # ---- stage l2 / delta of tile t+1 (before this tile's barrier) ----
                    if const_expr(self.rows_are_keys):
                        if t + 1 < total:
                            hn = hx * rep + (t + 1) // ncs
                            ctn = c_lo + (t + 1) - ((t + 1) // ncs) * ncs
                            if ctid < 128:
                                st_shared_f32(ldn + ctid * 4, dmh.dm_mx256_l2(Float32(mLse[b, hn, ctn * 128 + ctid])))
                            else:
                                st_shared_f32(ldn + ctid * 4, Float32(mDelta[b, hn, ctn * 128 + ctid - 128]))
                    # ---- block max exchange ----
                    if const_expr(self.mode == "dk"):
                        # row max over this half -> slot [hf][r]; partner half's max after the barrier
                        st_shared_f32(xb + (hf * 128 + r) * 4, cute.arch.fmax(am2[0], am2[1]))
                    else:
                        for jj in cutlass.range_constexpr(2):
                            a_ = dm_utils.redux_max_abs_f32(am2[jj])
                            if lane == 0:
                                st_shared_f32(xb + ((2 * hf + jj) * 4 + qd) * 4, a_)
                    if const_expr((self.dbg & 1) and self.mode == "dq"):
                        if hf == 1:
                            for jj in cutlass.range_constexpr(2):
                                for u in cutlass.range_constexpr(3):
                                    st_shared_f32(sId_addr + ((par * 128 + r) * 8 + 4 * jj + u) * 4, idc[jj][u])
                    if const_expr(self.kn == "xc"):
                        if warp == 0:
                            spin(self.DLY)
                    if const_expr(not (self.kn == "xc" and self.unsafe)):
                        cute.arch.barrier(barrier_id=2, number_of_threads=256)
                    if const_expr((self.dbg & 1) and self.mode == "dq"):
                        if hf == 0:
                            # original order: chunks 0, 1 (own) then 2, 3 (upper half), fp64 over ascending tiles
                            for jj in cutlass.range_constexpr(2):
                                idn = idn + Float64(idc[jj][0])
                                idd = idd + Float64(idc[jj][1])
                                idp = idp + Float64(idc[jj][2])
                            for jj in cutlass.range_constexpr(2):
                                idn = idn + Float64(ld_shared_f32(sId_addr + ((par * 128 + r) * 8 + 4 * jj) * 4))
                                idd = idd + Float64(ld_shared_f32(sId_addr + ((par * 128 + r) * 8 + 4 * jj + 1) * 4))
                                idp = idp + Float64(ld_shared_f32(sId_addr + ((par * 128 + r) * 8 + 4 * jj + 2) * 4))
                    if const_expr(self.mode == "dk"):
                        mo = ld_shared_f32(xb + ((1 - hf) * 128 + r) * 4)
                        sb = dm_utils.pow2_sb(dm_utils.redux_max_abs_f32(cute.arch.fmax(cute.arch.fmax(am2[0], am2[1]), mo)))
                        sbs = [sb, sb]
                        sball = [sb] * 4
                    else:
                        sball = [Int32(0)] * 4
                        for j in cutlass.range_constexpr(4):
                            m4 = ld_shared_v4f(xb + j * 16)
                            sball[j] = dm_utils.pow2_sb(max_f32(max_f32(m4[0], m4[1]), max_f32(m4[2], m4[3])))
                        sbs = [sel_eq_i32(hf, Int32(0), sball[0], sball[2]), sel_eq_i32(hf, Int32(0), sball[1], sball[3])]
                    if const_expr(self.kn == "cd" and self.unsafe):
                        pass
                    else:
                        ce = cd_p.acquire_and_advance()
                        if const_expr(self.ordf):
                            tc_fence_after()                   # BW3
                else:
                    # dv: P^T codes per (key, 32 queries), own chunks
                    for jj in cutlass.range_constexpr(2):
                        am = Float32(0.0)
                        for i in cutlass.range_constexpr(32):
                            am = cute.arch.fmax(am, vals[jj][i])
                        sbs[jj] = dm_utils.pow2_sb(am)
                    if const_expr(self.rows_are_keys):
                        if t + 1 < total:
                            hn = hx * rep + (t + 1) // ncs
                            ctn = c_lo + (t + 1) - ((t + 1) // ncs) * ncs
                            if ctid < 128:
                                st_shared_f32(ldn + ctid * 4, dmh.dm_mx256_l2(Float32(mLse[b, hn, ctn * 128 + ctid])))
                            else:
                                st_shared_f32(ldn + ctid * 4, Float32(mDelta[b, hn, ctn * 128 + ctid - 128]))
                    if hf == 1:
                        st_shared_f32(xb + r * 4, Float32(0.0) + Float32(sbs[0] + 256 * sbs[1]))
                    if const_expr(self.kn == "xc"):
                        if warp == 0:
                            spin(self.DLY)
                    if const_expr(not (self.kn == "xc" and self.unsafe)):
                        cute.arch.barrier(barrier_id=2, number_of_threads=256)
                    hv_ = Int32(ld_shared_f32(xb + r * 4))      # upper half's packed sbs (only the hf == 0 owner uses it)
                    sball = [sbs[0], sbs[1], hv_ % 256, hv_ // 256]
                    if const_expr(not (self.kn == "cd" and self.unsafe)):
                        ce = cd_p.acquire_and_advance()
                        if const_expr(self.ordf):
                            tc_fence_after()                   # BW3
                # ---- codes -> smem stage stg (own two chunks) ----
                cbase = crow + stg * self.bytes_c
                for jj in cutlass.range_constexpr(2):
                    j2 = 2 * hf + jj
                    hsl = [Int32(0)] * 32
                    hcl = [Int32(0)] * 32
                    inv = dm_utils.inv_pow2_from_sb(sbs[jj])
                    for hv in cutlass.range_constexpr(2):
                        w = [None] * 4
                        for u in cutlass.range_constexpr(4):
                            i0 = 16 * hv + 4 * u
                            y0, y1 = cute.arch.mul_packed_f32x2((vals[jj][i0], vals[jj][i0 + 1]), (inv, inv))
                            y2, y3 = cute.arch.mul_packed_f32x2((vals[jj][i0 + 2], vals[jj][i0 + 3]), (inv, inv))
                            w[u] = e4m3x4(y0, y1, y2, y3)
                        st_shared_v4(cbase + (((2 * j2 + hv) ^ rsw) * 16), w[0], w[1], w[2], w[3])
                        if const_expr((self.dbg & 8) and self.has_dp):
                            for u in cutlass.range_constexpr(4):
                                for q in cutlass.range_constexpr(4):
                                    e_ = 16 * hv + 4 * u + q
                                    ce_ = ctok0 + 32 * j2 + e_
                                    ix = hidx(rtok, ce_)
                                    if const_expr(self.rows_are_keys):
                                        ix = hidx(ce_, rtok)
                                    hsl[e_] = hmix(f2i(vals[jj][e_]), ix)
                                    hcl[e_] = hmix(((w[u] >> (8 * q)) & 255) | (sbs[jj] << 8), ix)
                    if const_expr((self.dbg & 8) and self.has_dp):
                        if const_expr(self.rows_are_keys):
                            a0_ = _gaddr(mHash) + (Int64(l) * N + ctok0 + 32 * j2) * 32
                            lk_cols_u32(hsl, lane, a0_ + 8)
                            lk_cols_u32(hcl, lane, a0_ + 16)
                            lk_cols_f64(vals[jj], lane, a0_ + 24)
                        else:
                            hs_ = Int32(0)
                            hc_ = Int32(0)
                            rs_ = Float64(0.0)
                            for i in cutlass.range_constexpr(32):
                                hs_ = hs_ + hsl[i]
                                hc_ = hc_ + hcl[i]
                                rs_ = rs_ + Float64(vals[jj][i])
                            lk_s = lk_s + zext64(hs_)
                            lk_c = lk_c + zext64(hc_)
                            lk_r = lk_r + rs_
                    if const_expr(self.dbg & 2):
                        mSB[l, rtok, ct * 4 + j2] = sbs[jj]
                if const_expr(self.dbg & 2):
                    stc = Int64(1)
                    ca0 = _gaddr(mCodes) + (Int64(l) * N + rtok) * N + ctok0
                    if const_expr(self.rows_are_keys):
                        stc = Int64(N)
                        ca0 = _gaddr(mCodes) + (Int64(l) * N + ctok0) * N + rtok
                    for vv in cutlass.range(4, unroll=1):
                        v = 4 * hf + vv
                        x4 = ld_shared_v4i(cbase + ((v ^ rsw) * 16))
                        cp = ca0 + Int64(16 * v) * stc
                        for u in cutlass.range_constexpr(4):
                            for q in cutlass.range_constexpr(4):
                                st_global_u8(cp, x4[u] >> (8 * q))
                                cp = cp + stc
                if hf == 0:
                    sfa = pack4(dm_utils.add_sb_clamp(sball[0], osb[0]), dm_utils.add_sb_clamp(sball[1], osb[1]),
                                dm_utils.add_sb_clamp(sball[2], osb[2]), dm_utils.add_sb_clamp(sball[3], osb[3]))
                    dm_utils.tmem_st_x1(tb + self.COL_SF + self.SF_STRIDE * stg + qd, sfa)
                cute.arch.fence_view_async_tmem_store()
                cute.arch.fence_view_async_shared()
                tc_fence_before()
                if const_expr(self.kn == "cd" and self.unsafe):
                    ce = cd_p.acquire_and_advance()
                ce.commit()
            if const_expr((self.dbg & 1) and self.mode == "dq"):
                if hf == 0:
                    ia = _gaddr(mIdent) + (Int64(b * H + hx) * N + rtok) * 32
                    st_global_f64x2(ia, idn, idd)
                    st_global_f64x2(ia + 16, idp, Float64(0.0))
            if const_expr((self.dbg & 8) and self.mode == "dq"):
                ha = _gaddr(mHash) + (Int64(b * H + hx) * N + rtok) * 32
                red_add_u64(ha, lk_p)
                red_add_u64(ha + 8, lk_s)
                red_add_u64(ha + 16, lk_c)
                red_add_f64(ha + 24, lk_r)
            # ---- epilogue: out row (256 fp32 in TMEM) -> bf16; this half's 128 columns ----
            fw = None
            if const_expr(not (self.kn == "fin" and self.unsafe)):
                fw = fin_c.wait_and_advance()
            tc_fence_after()
            nho = H
            if const_expr(self.rows_are_keys):
                nho = Hk
            ob = _gaddr(mOut) + (((Int64(b) * N + rtok) * nho + hx) * D) * 2
            for cq in cutlass.range_constexpr(4):
                cblk = 4 * hf + cq
                ov = tmem_ld32(tb + self.COL_OUT + 32 * cblk)
                cute.arch.fence_view_async_tmem_load()
                for i in cutlass.range_constexpr(0, 32, 8):
                    w = [None] * 4
                    for q in cutlass.range_constexpr(4):
                        x0 = ov[i + 2 * q]
                        x1 = ov[i + 2 * q + 1]
                        if const_expr(self.mode == "dq"):
                            x0 = mul_f32(x0, Float32(SM))
                            x1 = mul_f32(x1, Float32(SM))
                        w[q] = bf16x2(x0, x1)
                    st_global_v4(ob + (32 * cblk + i) * 2, w[0], w[1], w[2], w[3])
            if const_expr(self.kn == "fin" and self.unsafe):
                fw = fin_c.wait_and_advance()
            if const_expr(self.ordf):
                tc_fence_before()
            fw.release()

        tmem.relinquish_alloc_permit()
        if const_expr(self.ordf):
            tc_fence_before()                                  # BW9
        cute.arch.barrier(barrier_id=3, number_of_threads=self.threads)
        if const_expr(self.ordf):
            tc_fence_after()
        tmem.free(tmem_ptr)
