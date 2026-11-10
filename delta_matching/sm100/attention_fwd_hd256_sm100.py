"""
Delta-Matching attention forward at head dim 256 for sm100, MXFP8 numerics.

Inputs are the FP8 codes and power-of-2 scales of quant_qkv2 (one scale per
32 tokens over all 256 dims, softmax scale 2^-4 baked into Q). Per 128-query
row block and 128-key tile, with b the 32-key block of a key:
  S       = Q8 K8^T, tcgen05 kind::f8f6f4, 8 K=32 steps in ascending d.
  c_b     = (qs * log2(e)) * ks_b; bmax_b = max(S over block b) * c_b.
  m       integer running max in log2 units: m_cand = ceil(max_b bmax_b); m
          grows only when m_cand > m + 32, then O and l are rescaled by
          acc_scale = 2^(m - m_cand).
  P       kb_b = max(floor(bmax_b), -1e30); p = ex2(fma(S, c_b, 7 - kb_b));
          the P codes are e4m3(p) with block scale SFA_b = UE8M0(kb_b - 7 - m)
          and weight w_b = 2^(kb_b - 7 - m).
  l       l = l * acc_scale + sum_b (pairwise sum of p over block b) * w_b.
  O       += P V, one block-scaled kind::mxf8f6f4 MMA chain per tile
          (SFA = P block scales, SFB = V scales).
  output  O = bf16(O * rcp(l)); LSE = (m + lg2(l)) * ln(2), natural log.
"""
import math

import torch
import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils_basic
from cutlass import Float32, Int32, Int64, const_expr, pipeline
from cutlass.cute.nvgpu import cpasync
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op
from flash_attn.cute import cache_utils, utils
from flash_attn.cute.cute_dsl_utils import to_cute_tensor
from flash_attn.cute.named_barrier import NamedBarrierFwdSm100
from flash_attn.cute.testing import is_fake_mode
import flash_attn.cute.pipeline as pipeline_custom

from delta_matching.sm100.cute_ptx_sm100 import bf16x2, ceil_f32, e4m3x4, f32_unary, floor_f32, ld_shared_f32, ld_shared_v4, mbar_init, mul_f32, st_global_v4, st_shared_f32, st_shared_v4_b32, tc_commit, tmem_ld32, tmem_st_x1
from delta_matching.sm100.fp8_e8m0_scale_sm100 import byte_pow2_f32, byte_shl23_f32, exp2i_f32, pack_bytes4, ue8m0_byte
from delta_matching.sm100.mma_sm100 import gemm_ptx_bs
from delta_matching.sm100.tcgen05_fence_sm100 import mbar_arrive, mbar_wait_parity, tcgen05_fence_after, tcgen05_fence_before

__all__ = ["attention_fwd_hd256"]


@dsl_user_op
def tmem_wait_ld(*, loc=None, ip=None):
    llvm.inline_asm(None, [], "tcgen05.wait::ld.sync.aligned;", "", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def tmem_wait_st(*, loc=None, ip=None):
    llvm.inline_asm(None, [], "tcgen05.wait::st.sync.aligned;", "", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


class AttentionForwardHd256:
    """
    One CTA per (128-query row block, q head, batch), 16 warps: two softmax warpgroups, a correction
    warpgroup (lazy O rescale and the epilogue), an MMA warp and a load warp.

    TMEM: S slot s at columns 128 s holds S of the tiles of parity s; once the softmax has read S,
    the slot holds that tile's PV scale factors (SFA at +32, SFB at +64); O at [256, 512).
    """

    def __init__(self, is_causal: bool, qhead_per_kvhead: int):
        self.is_causal = is_causal
        self.qhead_per_kvhead = qhead_per_kvhead
        self.m_block_size = 128
        self.n_block_size = 128
        self.head_dim = 256
        self.kv_stage = 4
        self.rescale_threshold = 32.0
        self.mma_tiler_qk = (128, 128, 256)
        self.mma_tiler_pv = (128, 256, 128)
        # softmax warpgroup 0 (warps 0-3) owns S columns 0-63 and the running (m, l); warpgroup 1 owns columns 64-127
        self.softmax_warps = (0, 1, 2, 3, 4, 5, 6, 7)
        self.correction_warps = (8, 9, 10, 11)
        self.mma_warp = 12
        self.load_warp = 14
        self.idle_warps = (13, 15)
        self.threads_per_cta = 512
        self.num_regs_softmax = 192
        self.num_regs_correction = 80
        self.num_regs_other = 48
        self.buffer_align_bytes = 1024
        self.tmem_o_col = 256
        self.sfa_col_offset = 32
        self.sfb_col_offset = 64
        # mbarrier byte offsets in the mbar_dm block ([2] = one per S slot / tile parity)
        self.s_full = 0          # [2] QK(j) complete
        self.p_full = 16         # [2] P codes and scale factors of tile j written (both softmax warpgroups)
        self.o_full = 32         # PV(j) complete
        self.o_ready = 40        # O rescaled for tile j
        self.stats_full = 48     # [2] acc_scale of tile j published
        self.final_full = 64     # 1 / l published
        self.partial_full = 80   # [2] warpgroup 1's block maxima and sums of tile j published
        # sStats float offsets: acc_scale [2][128], 1 / l [128], unused [640], partials [2][4][128]
        self.stats_final = 2 * 128
        self.stats_partial = 1024

    @cute.jit
    def __call__(self, mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor, mO: cute.Tensor, mLSE: cute.Tensor, mQS: cute.Tensor, mKVW: cute.Tensor, stream):
        self.q_dtype = mQ.element_type
        mQt = cute.make_tensor(mQ.iterator, cute.select(mQ.layout, mode=[1, 3, 2, 0]))
        mKt = cute.make_tensor(mK.iterator, cute.select(mK.layout, mode=[1, 3, 2, 0]))
        mVt = cute.make_tensor(mV.iterator, cute.select(mV.layout, mode=[3, 1, 2, 0]))
        cta_group = tcgen05.CtaGroup.ONE
        k_major, mn_major = cute.nvgpu.OperandMajorMode.K, cute.nvgpu.OperandMajorMode.MN
        tiled_mma_qk = sm100_utils_basic.make_trivial_tiled_mma(self.q_dtype, self.q_dtype, k_major, k_major, Float32, cta_group, self.mma_tiler_qk[:2])
        tiled_mma_pv = sm100_utils_basic.make_trivial_tiled_mma(self.q_dtype, self.q_dtype, k_major, mn_major, Float32, cta_group, self.mma_tiler_pv[:2])
        cluster_layout_vmnk = cute.tiled_divide(cute.make_layout((1, 1, 1)), (tiled_mma_qk.thr_id.shape,))
        sQ_layout = sm100_utils_basic.make_smem_layout_a(tiled_mma_qk, self.mma_tiler_qk, self.q_dtype, 1)
        sK_layout = sm100_utils_basic.make_smem_layout_b(tiled_mma_qk, self.mma_tiler_qk, self.q_dtype, self.kv_stage)
        sV_layout = sm100_utils_basic.make_smem_layout_b(tiled_mma_pv, self.mma_tiler_pv, self.q_dtype, self.kv_stage)
        sP_layout = sm100_utils_basic.make_smem_layout_a(tiled_mma_pv, self.mma_tiler_pv, self.q_dtype, 2)
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)
        tma_atom_Q, mQt = cute.nvgpu.make_tiled_tma_atom_A(tma_load_op, mQt, cute.select(sQ_layout, mode=[0, 1, 2]), self.mma_tiler_qk, tiled_mma_qk, cluster_layout_vmnk.shape)
        tma_atom_K, mKt = cute.nvgpu.make_tiled_tma_atom_B(tma_load_op, mKt, cute.select(sK_layout, mode=[0, 1, 2]), self.mma_tiler_qk, tiled_mma_qk, cluster_layout_vmnk.shape)
        tma_atom_V, mVt = cute.nvgpu.make_tiled_tma_atom_B(tma_load_op, mVt, cute.select(sV_layout, mode=[0, 1, 2]), self.mma_tiler_pv, tiled_mma_pv, cluster_layout_vmnk.shape)
        self.tma_bytes_q = cute.size_in_bytes(self.q_dtype, cute.select(sQ_layout, mode=[0, 1, 2]))
        self.tma_bytes_kv = cute.size_in_bytes(self.q_dtype, cute.select(sK_layout, mode=[0, 1, 2]))
        assert self.tma_bytes_kv == cute.size_in_bytes(self.q_dtype, cute.select(sV_layout, mode=[0, 1, 2]))
        kv_ring_size = cutlass.max(cute.cosize(sK_layout), cute.cosize(sV_layout))
        assert kv_ring_size * (self.q_dtype.width // 8) >= 128 * 528, "the O epilogue stages 128 rows of 528 B in the K/V ring"

        @cute.struct
        class SharedStorage:
            mbar_q: cute.struct.MemRange[Int64, 2]
            mbar_kv: cute.struct.MemRange[Int64, self.kv_stage * 2]
            mbar_dm: cute.struct.MemRange[Int64, 14]
            tmem_holding_buf: Int32
            sStats: cute.struct.MemRange[Float32, 2 * 128 + 2 * 128 + 4 * 128 + 8 * 128]
            sQ: cute.struct.Align[cute.struct.MemRange[self.q_dtype, cute.cosize(sQ_layout)], self.buffer_align_bytes]
            sK: cute.struct.Align[cute.struct.MemRange[self.q_dtype, kv_ring_size], self.buffer_align_bytes]
            sP: cute.struct.Align[cute.struct.MemRange[self.q_dtype, cute.cosize(sP_layout)], self.buffer_align_bytes]

        self.shared_storage = SharedStorage
        assert SharedStorage.size_in_bytes() <= 232448, f"SMEM {SharedStorage.size_in_bytes()}"
        n_m = cute.ceil_div(cute.size(mQt.shape[0]), self.m_block_size)
        grid = (n_m, cute.size(mQt.shape[2]), cute.size(mQt.shape[3]))
        self.kernel(mQt, mKt, mVt, mO, mLSE, mQS, mKVW, tma_atom_Q, tma_atom_K, tma_atom_V, sQ_layout, sK_layout, sV_layout, sP_layout, tiled_mma_qk, tiled_mma_pv).launch(grid=grid, block=[self.threads_per_cta, 1, 1], stream=stream, min_blocks_per_mp=1)

    @cute.kernel
    def kernel(self, mQ, mK, mV, mO, mLSE, mQS, mKVW, tma_atom_Q, tma_atom_K, tma_atom_V, sQ_layout: cute.ComposedLayout, sK_layout: cute.ComposedLayout, sV_layout: cute.ComposedLayout, sP_layout: cute.ComposedLayout, tiled_mma_qk: cute.TiledMma, tiled_mma_pv: cute.TiledMma):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            for atom in (tma_atom_Q, tma_atom_K, tma_atom_V):
                cpasync.prefetch_descriptor(atom)
        bx, by, bz = cute.arch.block_idx()
        n_m = cute.arch.grid_dim()[0]
        # causal: the longest row blocks start first
        m_block = n_m - 1 - bx if const_expr(self.is_causal) else bx
        head_idx, batch_idx = by, bz
        kv_head_idx = head_idx // self.qhead_per_kvhead
        num_tiles = m_block + 1 if const_expr(self.is_causal) else n_m

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        tmem_alloc_barrier = pipeline_custom.NamedBarrier(barrier_id=int(NamedBarrierFwdSm100.TmemPtr), num_threads=cute.arch.WARP_SIZE * (1 + len(self.softmax_warps) + len(self.correction_warps)))
        tmem = cutlass.utils.TmemAllocator(storage.tmem_holding_buf.ptr, barrier_for_retrieve=tmem_alloc_barrier, allocator_warp_id=self.mma_warp, is_two_cta=False)
        mma_group, tma_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1), pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        cluster_layout_vmnk = cute.tiled_divide(cute.make_layout((1, 1, 1)), (tiled_mma_qk.thr_id.shape,))
        pipeline_q = pipeline_custom.PipelineTmaUmma.create(barrier_storage=storage.mbar_q.data_ptr(), num_stages=1, producer_group=tma_group, consumer_group=mma_group, tx_count=self.tma_bytes_q, cta_layout_vmnk=cluster_layout_vmnk, defer_sync=True)
        pipeline_kv = pipeline_custom.PipelineTmaUmma.create(barrier_storage=storage.mbar_kv.data_ptr(), num_stages=self.kv_stage, producer_group=tma_group, consumer_group=mma_group, tx_count=self.tma_bytes_kv, cta_layout_vmnk=cluster_layout_vmnk, defer_sync=True)
        mbar = Int32(storage.mbar_dm.data_ptr().toint())
        if cute.arch.thread_idx()[0] == 0:
            self.init_barriers(mbar)
        pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = cute.make_tensor(cute.recast_ptr(sK.iterator, sV_layout.inner), sV_layout.outer)
        sP = storage.sP.get_tensor(sP_layout.outer, swizzle=sP_layout.inner)
        s_stats = Int32(storage.sStats.data_ptr().toint())
        s_p_base = Int32(sP.iterator.toint())
        thr_mma_qk, thr_mma_pv = tiled_mma_qk.get_slice(0), tiled_mma_pv.get_slice(0)
        acc_qk = thr_mma_qk.make_fragment_C(thr_mma_qk.partition_shape_C(self.mma_tiler_qk[:2]))
        acc_pv = thr_mma_pv.make_fragment_C(thr_mma_pv.partition_shape_C(self.mma_tiler_pv[:2]))
        pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        for i in cutlass.range_constexpr(len(self.idle_warps)):
            if warp_idx == self.idle_warps[i]:
                cute.arch.setmaxregister_decrease(self.num_regs_other)
        if warp_idx == self.load_warp:
            self.load_loop(mQ, mK, mV, tma_atom_Q, tma_atom_K, tma_atom_V, sQ, sK, sV, thr_mma_qk, thr_mma_pv, pipeline_q, pipeline_kv, m_block, head_idx, kv_head_idx, batch_idx, num_tiles)
        if warp_idx == self.mma_warp:
            self.mma_loop(tmem, tmem_alloc_barrier, tiled_mma_qk, tiled_mma_pv, sQ, sK, sV, sP, acc_qk, acc_pv, pipeline_q, pipeline_kv, mbar, num_tiles)
        if warp_idx <= self.softmax_warps[-1]:
            self.softmax_loop(tmem_alloc_barrier, mbar, s_stats, s_p_base, mLSE, mQS, mKVW, m_block, head_idx, kv_head_idx, batch_idx, num_tiles)
        if warp_idx >= self.correction_warps[0] and warp_idx <= self.correction_warps[-1]:
            self.correction_loop(tmem_alloc_barrier, mbar, s_stats, mO, m_block, head_idx, batch_idx, num_tiles, sK)
        return

    def init_barriers(self, mbar):
        """
        Initialize the raw mbarriers; barriers 9, 12 and 13 are spare (initialized, never used).
        """
        mbar_init(mbar + self.s_full, 1)
        mbar_init(mbar + self.s_full + 8, 1)
        mbar_init(mbar + self.p_full, 256)
        mbar_init(mbar + self.p_full + 8, 256)
        mbar_init(mbar + self.o_full, 1)
        mbar_init(mbar + self.o_ready, 128)
        mbar_init(mbar + self.stats_full, 128)
        mbar_init(mbar + self.stats_full + 8, 128)
        mbar_init(mbar + self.final_full, 128)
        mbar_init(mbar + 72, 128)
        mbar_init(mbar + self.partial_full, 128)
        mbar_init(mbar + self.partial_full + 8, 128)
        mbar_init(mbar + 96, 128)
        mbar_init(mbar + 104, 128)
        cute.arch.mbarrier_init_fence()

    @cute.jit
    def load_loop(self, mQ, mK, mV, tma_atom_Q, tma_atom_K, tma_atom_V, sQ, sK, sV, thr_mma_qk, thr_mma_pv, pipeline_q, pipeline_kv, m_block, head_idx, kv_head_idx, batch_idx, num_tiles):
        """
        TMA loads in the MMA warp's order: K(0), Q, then K(j + 1), V(j) per tile, V(last). Tile j is
        key block num_tiles - 1 - j.
        """
        cute.arch.setmaxregister_decrease(self.num_regs_other)
        gQ = cute.local_tile(mQ[None, None, head_idx, batch_idx], (self.m_block_size, self.head_dim), (m_block, 0))
        gK = cute.local_tile(mK[None, None, kv_head_idx, batch_idx], (self.n_block_size, self.head_dim), (None, 0))
        gV = cute.local_tile(mV[None, None, kv_head_idx, batch_idx], (self.head_dim, self.n_block_size), (0, None))
        tSgQ = thr_mma_qk.partition_A(gQ)
        tSgK = thr_mma_qk.partition_B(gK)
        tOgV = thr_mma_pv.partition_B(gV)
        tQsQ, tQgQ = cpasync.tma_partition(tma_atom_Q, 0, cute.make_layout(1), cute.group_modes(sQ, 0, 3), cute.group_modes(tSgQ, 0, 3))
        tKsK, tKgK = cpasync.tma_partition(tma_atom_K, 0, cute.make_layout(1), cute.group_modes(sK, 0, 3), cute.group_modes(tSgK, 0, 3))
        tVsV, tVgV = cpasync.tma_partition(tma_atom_V, 0, cute.make_layout(1), cute.group_modes(sV, 0, 3), cute.group_modes(tOgV, 0, 3))
        kv_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.kv_stage)
        q_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
        pipeline_kv.producer_acquire(kv_state)
        cute.copy(tma_atom_K, tKgK[None, num_tiles - 1], tKsK[None, kv_state.index], tma_bar_ptr=pipeline_kv.producer_get_barrier(kv_state))
        kv_state.advance()
        pipeline_q.producer_acquire(q_state)
        cute.copy(tma_atom_Q, tQgQ, tQsQ[None, 0], tma_bar_ptr=pipeline_q.producer_get_barrier(q_state))
        for i in cutlass.range(num_tiles - 1, unroll=1):
            pipeline_kv.producer_acquire(kv_state)
            cute.copy(tma_atom_K, tKgK[None, num_tiles - 2 - i], tKsK[None, kv_state.index], tma_bar_ptr=pipeline_kv.producer_get_barrier(kv_state))
            kv_state.advance()
            pipeline_kv.producer_acquire(kv_state)
            cute.copy(tma_atom_V, tVgV[None, num_tiles - 1 - i], tVsV[None, kv_state.index], tma_bar_ptr=pipeline_kv.producer_get_barrier(kv_state))
            kv_state.advance()
        # the last write of this CTA to the K/V ring: the epilogue reuses the ring after the last PV completed
        pipeline_kv.producer_acquire(kv_state)
        cute.copy(tma_atom_V, tVgV[None, 0], tVsV[None, kv_state.index], tma_bar_ptr=pipeline_kv.producer_get_barrier(kv_state))
        kv_state.advance()

    @cute.jit
    def mma_loop(self, tmem, tmem_alloc_barrier, tiled_mma_qk, tiled_mma_pv, sQ, sK, sV, sP, acc_qk, acc_pv, pipeline_q, pipeline_kv, mbar, num_tiles):
        """
        Allocate TMEM, issue QK(0), then QK(j) and PV(j - 1) for each later tile j, then
        PV(last); free TMEM.
        """
        cute.arch.setmaxregister_decrease(self.num_regs_other)
        tmem.allocate(512)
        tmem_alloc_barrier.arrive_and_wait_unaligned()
        tmem_ptr = tmem.retrieve_ptr(Float32)
        tSrQ = tiled_mma_qk.make_fragment_A(sQ)  # (MMA, 1, 8, 1)
        tSrK = tiled_mma_qk.make_fragment_B(sK)  # (MMA, 1, 8, kv_stage)
        tOrP = tiled_mma_pv.make_fragment_A(sP)  # (MMA, 1, 4, 2)
        tOrV = tiled_mma_pv.make_fragment_B(sV)  # (MMA, 1, 4, kv_stage)
        atom_qk = cute.make_mma_atom(tiled_mma_qk.op)
        pv_op = cute.make_mma_atom(tiled_mma_pv.op).op
        kv_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.kv_stage)
        q_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
        pipeline_q.consumer_wait(q_state)
        pipeline_kv.consumer_wait(kv_state)
        self.issue_qk(atom_qk, acc_qk, tSrQ, tSrK, kv_state.index, Int32(0))
        with cute.arch.elect_one():
            tc_commit(mbar + self.s_full)
        pipeline_kv.consumer_release(kv_state)
        kv_state.advance()
        for i in cutlass.range(num_tiles - 1, unroll=1):
            tile_idx = i + 1
            slot = tile_idx & 1
            # with j = tile_idx: slot j & 1 holds the scale factors of tile j - 2, read by PV(j - 2); PTX does not order
            # PV(j - 2) before QK(j) by issue order, so wait for its completion. PV(j - 3) is complete (PV(j - 2) waited
            # O_ready(j - 2), which follows O_full(PV(j - 3))) and PV(j - 1) is not issued yet: the parity is unambiguous.
            if i >= 1:
                mbar_wait_parity(mbar + self.o_full, (tile_idx - 2) & 1)
                tcgen05_fence_after()
            pipeline_kv.consumer_wait(kv_state)
            self.issue_qk(atom_qk, acc_qk, tSrQ, tSrK, kv_state.index, slot)
            with cute.arch.elect_one():
                tc_commit(mbar + self.s_full + 8 * slot)
            pipeline_kv.consumer_release(kv_state)
            kv_state.advance()
            self.issue_pv(pv_op, tOrP, tOrV, sP, sV, pipeline_kv, kv_state, mbar, i)
            kv_state.advance()
        self.issue_pv(pv_op, tOrP, tOrV, sP, sV, pipeline_kv, kv_state, mbar, num_tiles - 1)
        kv_state.advance()
        pipeline_q.consumer_release(q_state)
        tmem.relinquish_alloc_permit()
        tcgen05_fence_before()
        tmem_alloc_barrier.arrive_and_wait_unaligned()
        tcgen05_fence_after()
        tmem.free(tmem_ptr)

    @cute.jit
    def issue_qk(self, atom_qk, acc_qk, tSrQ, tSrK, stage, slot):
        """
        S slot `slot` = Q K^T from K/V stage `stage`, K=32 steps in ascending d, the
        first one zero-initializing.
        """
        tS = cute.make_tensor(acc_qk.iterator + slot * 128, acc_qk.layout)
        for k in cutlass.range_constexpr(cute.size(tSrQ.shape[2])):
            atom_qk.set(tcgen05.Field.ACCUMULATE, k != 0)
            cute.gemm(atom_qk, tS, tSrQ[None, None, k, 0], tSrK[None, None, k, stage], tS)

    @cute.jit
    def issue_pv(self, pv_op, tOrP, tOrV, sP, sV, pipeline_kv, kv_state, mbar, tile_idx):
        """
        O += P V of the tile once its V tile, P codes, scale factors and the rescaled O are ready.
        """
        slot = tile_idx & 1
        pipeline_kv.consumer_wait(kv_state)
        stage = kv_state.index
        mbar_wait_parity(mbar + self.p_full + 8 * slot, (tile_idx >> 1) & 1)
        if tile_idx >= 1:
            mbar_wait_parity(mbar + self.o_ready, (tile_idx - 1) & 1)
        tcgen05_fence_after()
        sf_col = slot * 128
        gemm_ptx_bs(pv_op, self.tmem_o_col, tOrP[None, None, None, slot], tOrV[None, None, None, stage], sP[None, None, None, slot], sV[None, None, None, stage], sf_col + self.sfa_col_offset, sf_col + self.sfb_col_offset, tuple((0, k) for k in range(4)), tuple((0, k) for k in range(4)), zero_init=(tile_idx == 0))
        with cute.arch.elect_one():
            tc_commit(mbar + self.o_full)
        pipeline_kv.consumer_release(kv_state)

    @cute.jit
    def softmax_loop(self, tmem_alloc_barrier, mbar, s_stats, s_p_base, mLSE, mQS, mKVW, m_block, head_idx, kv_head_idx, batch_idx, num_tiles):
        """
        Every tile through softmax_step; warpgroup 0 then publishes LSE and 1 / l.
        """
        cute.arch.setmaxregister_increase(self.num_regs_softmax)
        tmem_alloc_barrier.arrive_and_wait_unaligned()
        tidx = cute.arch.thread_idx()[0] % 128
        half = cute.arch.make_warp_uniform(cute.arch.warp_idx()) // 4
        row = m_block * self.m_block_size + tidx
        q_scale_log2 = Float32(mQS[batch_idx, head_idx, row // 32]) * Float32(math.log2(math.e))
        lane_base = (cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4) * 32
        row_max = -Float32.inf
        row_sum = Float32(0.0)
        if half == 0:
            row_max, row_sum = self.softmax_step(Int32(0), num_tiles - 1, True, 0, mbar, s_stats, s_p_base, tidx, lane_base, q_scale_log2, mKVW, kv_head_idx, batch_idx, row_max, row_sum)
            for i in cutlass.range(num_tiles - 1, unroll=1):
                row_max, row_sum = self.softmax_step(i + 1, num_tiles - 2 - i, False, 0, mbar, s_stats, s_p_base, tidx, lane_base, q_scale_log2, mKVW, kv_head_idx, batch_idx, row_max, row_sum)
            zero_or_nan = row_sum == Float32(0.0) or row_sum != row_sum
            rcp_sum = cute.arch.rcp_approx(row_sum if not zero_or_nan else Float32(1.0))
            # the hd128 forward's LSE expression, kept term for term so both head dims round alike
            lse = (row_max * Float32(1.0) + (cute.math.log2(row_sum, fastmath=True) - Float32(0.0))) * math.log(2.0) if not zero_or_nan else -Float32.inf
            mLSE[batch_idx, head_idx, row] = lse
            st_shared_f32(s_stats + 4 * (self.stats_final + tidx), rcp_sum)
            mbar_arrive(mbar + self.final_full)
        else:
            self.softmax_step(Int32(0), num_tiles - 1, True, 1, mbar, s_stats, s_p_base, tidx, lane_base, q_scale_log2, mKVW, kv_head_idx, batch_idx, row_max, row_sum)
            for i in cutlass.range(num_tiles - 1, unroll=1):
                self.softmax_step(i + 1, num_tiles - 2 - i, False, 1, mbar, s_stats, s_p_base, tidx, lane_base, q_scale_log2, mKVW, kv_head_idx, batch_idx, row_max, row_sum)
        tcgen05_fence_before()
        tmem_alloc_barrier.arrive_unaligned()

    @cute.jit
    def softmax_step(self, tile_idx, key_block, is_first: cutlass.Constexpr, half: cutlass.Constexpr, mbar, s_stats, s_p_base, tidx, lane_base, q_scale_log2, mKVW, kv_head_idx, batch_idx, row_max, row_sum):
        """
        One tile (key block key_block) for one column half: the m-independent part of blocks 2 half
        and 2 half + 1, then half 1 publishes its partials and half 0 runs the ordered tail. Returns
        the updated (m, l).
        """
        slot = tile_idx & 1
        s_partial = s_stats + 4 * (self.stats_partial + slot * 512)  # [4][128]: block max 2, 3, block sum 2, 3
        k_scale_word = Int32(mKVW[batch_idx, kv_head_idx, 2 * key_block])
        v_scale_word = Int32(mKVW[batch_idx, kv_head_idx, 2 * key_block + 1])
        k_scales = [byte_pow2_f32(k_scale_word, block) for block in range(4)]
        mbar_wait_parity(mbar + self.s_full + 8 * slot, (tile_idx >> 1) & 1)
        tcgen05_fence_after()
        tmem_row = (lane_base << 16) + slot * 128
        col0 = 64 * half
        s_lo, s_hi = tmem_ld32(tmem_row + col0), tmem_ld32(tmem_row + col0 + 32)
        tmem_wait_ld()
        scores = s_lo + s_hi  # scores[col] = S(row, 64 half + col)
        if const_expr(self.is_causal and is_first):
            for col in cutlass.range_constexpr(64):
                scores[col] = -Float32.inf if Int32(col + col0) > tidx else scores[col]
        exp_scale = [q_scale_log2 * k_scales[2 * half + block] for block in range(2)]
        block_max = self.block_maxima(scores, exp_scale)
        block_exp, exp_bias = [None] * 2, [None] * 2
        for block in cutlass.range_constexpr(2):
            block_exp[block] = cute.arch.fmax(floor_f32(block_max[block]), Float32(-1e30))
            exp_bias[block] = Float32(7.0) - block_exp[block]
        probs = self.exponentiate_and_store_codes(scores, exp_scale, exp_bias, s_p_base, slot, half, tidx)
        block_sum = self.block_sums(probs)
        new_max, new_sum = row_max, row_sum
        if const_expr(half == 1):
            # the partials of tile j are overwritten at j + 2 only after half 0 read them: read(j) -> P_full(j) -> PV(j) -> QK(j + 2) -> S_full(j + 2)
            st_shared_f32(s_partial + 4 * (0 * 128 + tidx), block_max[0])
            st_shared_f32(s_partial + 4 * (1 * 128 + tidx), block_max[1])
            st_shared_f32(s_partial + 4 * (2 * 128 + tidx), block_sum[0])
            st_shared_f32(s_partial + 4 * (3 * 128 + tidx), block_sum[1])
            # orders this half's S loads (wait::ld) before half 0's SFB stores into the same slot
            tcgen05_fence_before()
            mbar_arrive(mbar + self.partial_full + 8 * slot)
            cute.arch.fence_view_async_shared()
            tcgen05_fence_before()
            mbar_arrive(mbar + self.p_full + 8 * slot)
        else:
            new_max, new_sum = self.softmax_tail(tile_idx, is_first, slot, block_max, block_exp, block_sum, s_partial, v_scale_word, tmem_row, mbar, s_stats, tidx, row_max, row_sum)
        return new_max, new_sum

    def block_maxima(self, scores, exp_scale):
        """
        bmax_b = max(S over 32-key block b) * c_b for the two blocks of scores, in the max-tree
        order of the hd128 forward.
        """
        block_max = [None] * 2
        for block in range(2):
            first = 32 * block
            part_max = [utils.fmax(scores[first + 3 * group], scores[first + 3 * group + 1], scores[first + 3 * group + 2]) for group in range(4)]
            for i in range(12, 32, 2):
                group = ((i - 12) // 2) % 4
                part_max[group] = utils.fmax(part_max[group], scores[first + i], scores[first + i + 1])
            block_max[block] = utils.fmax(utils.fmax(part_max[0], part_max[1], part_max[2]), part_max[3]) * exp_scale[block]
        return block_max

    def exponentiate_and_store_codes(self, scores, exp_scale, exp_bias, s_p_base, slot, half, tidx):
        """
        Return p = ex2(fma(S, c_b, bias_b)), computed in place in scores, after storing its e4m3
        codes into this row of P tile slot. The tile is K-major SW128: 16 B chunk q of row r sits at
        (q ^ (r % 8)) * 16.
        """
        for block in range(2):
            for i in range(0, 32, 2):
                scores[32 * block + i], scores[32 * block + i + 1] = cute.arch.fma_packed_f32x2((scores[32 * block + i], scores[32 * block + i + 1]), (exp_scale[block], exp_scale[block]), (exp_bias[block], exp_bias[block]))
        p_row = s_p_base + slot * 16384 + tidx * 128
        for block in range(2):
            for i in range(32):
                scores[32 * block + i] = f32_unary("ex2.approx.ftz.f32 $0, $1;", scores[32 * block + i])
        for block in range(2):
            for half16 in range(2):
                first = 32 * block + 16 * half16
                words = [e4m3x4(scores[first + 4 * i], scores[first + 4 * i + 1], scores[first + 4 * i + 2], scores[first + 4 * i + 3]) for i in range(4)]
                chunk = Int32(2 * (2 * half + block) + half16)
                st_shared_v4_b32(p_row + ((chunk ^ (tidx & 7)) << 4), words[0], words[1], words[2], words[3])
        return scores

    def block_sums(self, probs):
        """
        Pairwise sum of p over each 32-key block.
        """
        block_sum = [None] * 2
        for block in range(2):
            pair = (probs[32 * block], probs[32 * block + 1])
            for i in range(2, 32, 2):
                pair = cute.arch.add_packed_f32x2(pair, (probs[32 * block + i], probs[32 * block + i + 1]))
            block_sum[block] = pair[0] + pair[1]
        return block_sum

    @cute.jit
    def softmax_tail(self, tile_idx, is_first: cutlass.Constexpr, slot, block_max, block_exp, block_sum, s_partial, v_scale_word, tmem_row, mbar, s_stats, tidx, row_max, row_sum):
        """
        Column half 0, in tile order: the lazy running-max rule over all 4 blocks, acc_scale to the
        correction warps, the scale factors of the tile, P_full, and the new l.
        """
        mbar_wait_parity(mbar + self.partial_full + 8 * slot, (tile_idx >> 1) & 1)
        tcgen05_fence_after()
        block_max2 = ld_shared_f32(s_partial + 4 * (0 * 128 + tidx))
        block_max3 = ld_shared_f32(s_partial + 4 * (1 * 128 + tidx))
        block_sum2 = ld_shared_f32(s_partial + 4 * (2 * 128 + tidx))
        block_sum3 = ld_shared_f32(s_partial + 4 * (3 * 128 + tidx))
        all_max = [block_max[0], block_max[1], block_max2, block_max3]
        all_sum = [block_sum[0], block_sum[1], block_sum2, block_sum3]
        all_exp = [block_exp[0], block_exp[1], cute.arch.fmax(floor_f32(block_max2), Float32(-1e30)), cute.arch.fmax(floor_f32(block_max3), Float32(-1e30))]
        tile_max = utils.fmax(utils.fmax(all_max[0], all_max[1], all_max[2]), all_max[3])
        m_cand = ceil_f32(tile_max)
        acc_scale = Float32(1.0)
        new_max = m_cand
        if const_expr(is_first):
            acc_scale = Float32(0.0)
        else:
            new_max = row_max
            if m_cand > row_max + Float32(self.rescale_threshold):
                new_max = m_cand
                acc_scale = exp2i_f32(row_max - m_cand)
        st_shared_f32(s_stats + 4 * (slot * 128 + tidx), acc_scale)
        mbar_arrive(mbar + self.stats_full + 8 * slot)
        m_safe = new_max if new_max != -Float32.inf else Float32(0.0)
        block_weight, sfa_bytes = [None] * 4, [None] * 4
        for block in cutlass.range_constexpr(4):
            sfa_bytes[block] = ue8m0_byte(all_exp[block] - Float32(7.0) - m_safe)
            block_weight[block] = byte_shl23_f32(sfa_bytes[block])
        sfa_word = pack_bytes4(sfa_bytes[0], sfa_bytes[1], sfa_bytes[2], sfa_bytes[3])
        for col in cutlass.range_constexpr(4):
            tmem_st_x1(tmem_row + self.sfa_col_offset + col, sfa_word)
        for col in cutlass.range_constexpr(8):
            tmem_st_x1(tmem_row + self.sfb_col_offset + col, v_scale_word)
        cute.arch.fence_view_async_shared()
        tmem_wait_st()
        tcgen05_fence_before()
        mbar_arrive(mbar + self.p_full + 8 * slot)
        new_sum = Float32(0.0) if const_expr(is_first) else row_sum * acc_scale
        for block in cutlass.range_constexpr(4):
            new_sum = new_sum + all_sum[block] * block_weight[block]
        return new_max, new_sum

    @cute.jit
    def correction_loop(self, tmem_alloc_barrier, mbar, s_stats, mO, m_block, head_idx, batch_idx, num_tiles, sK):
        """
        Rescale O before each PV that needs it, then store O * (1 / l) as bf16.
        """
        cute.arch.setmaxregister_decrease(self.num_regs_correction)
        tmem_alloc_barrier.arrive_and_wait_unaligned()
        s_ring_base = Int32(sK.iterator.toint())
        tidx = cute.arch.thread_idx()[0] % 128
        lane_base = (cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4) * 32
        tmem_o_row = (lane_base << 16) + self.tmem_o_col
        for i in cutlass.range(num_tiles - 1, unroll=1):
            tile_idx = i + 1
            slot = tile_idx & 1
            mbar_wait_parity(mbar + self.stats_full + 8 * slot, (tile_idx >> 1) & 1)
            acc_scale = ld_shared_f32(s_stats + 4 * (slot * 128 + tidx))
            mbar_wait_parity(mbar + self.o_full, (tile_idx - 1) & 1)
            tcgen05_fence_after()
            self.rescale_o(tmem_o_row, acc_scale)
            tcgen05_fence_before()
            mbar_arrive(mbar + self.o_ready)
        mbar_wait_parity(mbar + self.final_full, Int32(0))
        rcp_sum = ld_shared_f32(s_stats + 4 * (self.stats_final + tidx))
        mbar_wait_parity(mbar + self.o_full, (num_tiles - 1) & 1)
        tcgen05_fence_after()
        self.store_o(mO, tmem_o_row, rcp_sum, s_ring_base, m_block, head_idx, batch_idx, tidx)
        tcgen05_fence_before()
        tmem_alloc_barrier.arrive_unaligned()

    @cute.jit
    def rescale_o(self, tmem_o_row, acc_scale):
        """
        O *= acc_scale in TMEM when any row of the warp needs it.
        """
        if cute.arch.vote_ballot_sync(acc_scale < Float32(1.0)) != 0:
            for chunk in cutlass.range_constexpr(8):
                o_chunk = list(tmem_ld32(tmem_o_row + 32 * chunk))
                tmem_wait_ld()
                for k in cutlass.range_constexpr(0, 32, 2):
                    o_chunk[k], o_chunk[k + 1] = cute.arch.mul_packed_f32x2((o_chunk[k], o_chunk[k + 1]), (acc_scale, acc_scale))
                llvm.inline_asm(None, [Int32(tmem_o_row + 32 * chunk).ir_value()] + [Float32(x).ir_value() for x in o_chunk], "tcgen05.st.sync.aligned.32x32b.x32.b32 [$0], {" + ",".join(f"${k + 1}" for k in range(32)) + "};", "r" + ",f" * 32, has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)
            tmem_wait_st()

    @cute.jit
    def store_o(self, mO, tmem_o_row, rcp_sum, s_ring_base, m_block, head_idx, batch_idx, tidx):
        """
        Each thread converts its O row to bf16 into a 528 B staging row in the idle K/V ring; each
        warp then copies its own 32 rows out, one 512 B row per store instruction.
        """
        N, H = mO.shape[1], mO.shape[2]
        # the ring was last written by TMA (async proxy) and last read by the final PV, which O_full has completed
        llvm.inline_asm(None, [], "fence.proxy.async.shared::cta;", "", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)
        ROW_BYTES = 528  # 512 B row + 16 B pad: bank-conflict free for the row writes and the column reads
        s_row = s_ring_base + Int32(ROW_BYTES) * tidx
        lane = tidx % 32
        warp_row0 = tidx - lane
        # chunk c + 1 is loaded while chunk c is converted; wait::ld covers every prior load, so each load is
        # issued only after the previous wait
        o_chunk = tmem_ld32(tmem_o_row)
        tmem_wait_ld()
        for chunk in cutlass.range_constexpr(8):
            if const_expr(chunk < 7):
                o_next_chunk = tmem_ld32(tmem_o_row + 32 * (chunk + 1))
            for quarter in cutlass.range_constexpr(4):
                words = [bf16x2(mul_f32(o_chunk[8 * quarter + 2 * i], rcp_sum), mul_f32(o_chunk[8 * quarter + 2 * i + 1], rcp_sum)) for i in range(4)]
                st_shared_v4_b32(s_row + (64 * chunk + 16 * quarter), words[0], words[1], words[2], words[3])
            if const_expr(chunk < 7):
                tmem_wait_ld()
                o_chunk = o_next_chunk
        # a warp copies out only its own rows, so a warp-level barrier is enough
        llvm.inline_asm(None, [], "bar.warp.sync -1;", "", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)
        for i in cutlass.range(32, unroll=4):
            row = warp_row0 + i
            words = ld_shared_v4(s_ring_base + Int32(ROW_BYTES) * row + 16 * lane)
            row_addr = Int64(mO.iterator.toint()) + (Int64((batch_idx * N + m_block * self.m_block_size + row) * H + head_idx) * Int64(self.head_dim * 2)) + Int64(16 * lane)
            st_global_v4(row_addr, words[0], words[1], words[2], words[3])


CACHE = []


def compile_cache():
    """
    The persistent compile cache of this kernel (in-memory when the CuTe DSL cache is off).
    """
    from delta_matching.sm100.compile_cache_sm100 import fingerprint
    if not CACHE:
        if cache_utils.CUTE_DSL_CACHE_ENABLED:
            CACHE.append(cache_utils.JITPersistentCache(cache_utils.get_cache_path() / "dm_sm100" / fingerprint() / "attention_fwd_hd256"))
        else:
            CACHE.append(cache_utils.JITCache())
    return CACHE[0]


def attention_fwd_hd256(q8, qs, k8, kvw, v8, causal: bool):
    """
    Forward at head dim 256 from quant_qkv2's outputs.

    q8 (B,N,H,256) e4m3 with the softmax scale baked in; k8, v8 (B,N,Hk,256) e4m3; qs (B,H,N/32)
    fp32 powers of 2; kvw (B,Hk,N/64) int32, the packed UE8M0 K and V scale words. Needs N % 128 = 0
    and H % Hk = 0. Returns O (B,N,H,256) bf16 and LSE (B,H,N) fp32 in natural log. Under
    FakeTensorMode the call only compiles and returns empty outputs.
    """
    B, N, H, D = q8.shape
    Hk = k8.shape[2]
    assert D == 256 and N % 128 == 0 and H % Hk == 0
    assert q8.dtype == torch.float8_e4m3fn and k8.dtype == q8.dtype and v8.dtype == q8.dtype
    assert qs.shape == (B, H, N // 32) and kvw.shape == (B, Hk, N // 64) and kvw.dtype == torch.int32
    q8, k8, v8, qs, kvw = (t.contiguous() for t in (q8, k8, v8, qs, kvw))
    o = torch.empty(B, N, H, D, device=q8.device, dtype=torch.bfloat16)
    lse = torch.empty(B, H, N, device=q8.device, dtype=torch.float32)
    key = ("attention_fwd_hd256", bool(causal), H // Hk)
    cache = compile_cache()
    if key not in cache:
        args = [to_cute_tensor(t) for t in (q8, k8, v8, o)]
        args += [to_cute_tensor(lse, assumed_align=4), to_cute_tensor(qs, assumed_align=4), to_cute_tensor(kvw, assumed_align=4)]
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        cache[key] = cute.compile(AttentionForwardHd256(bool(causal), H // Hk), *args, stream, options="--enable-tvm-ffi")
    if is_fake_mode():
        return o, lse
    cache[key](q8.view(torch.uint8), k8.view(torch.uint8), v8.view(torch.uint8), o, lse, qs, kvw)
    return o, lse
