"""hd256 Delta-Matching forward for sm100 (GB200), [64,32] fp32 block-scale contract ("Delta-Matching-256 v1" floor).

Numerics scheme "a" (DM_HD256_NUMERICS=a, the only one implemented): the sm90 floor's algebra at every site
(kda/docs/sm90-numerics-spec-hd256.md §2), see kda/hd256/fwd-design.md:
  S      = S_acc * fmax(q8s[row//64] * k8s[key//32], FLT_MIN)        (S_acc = f8f6f4 over D=256, -inf mask on S_acc)
  gm_g   = max_{j in 32-key group g} S ;  m = max(m_old, max_g gm_g) ;  alpha = ex2((m_old - m) * l2)
  ms_g   = fmax(gm_g * l2, m*l2 - 12) - log2(448)
  p~     = ex2(fma(S, l2, -ms_g)) ;  C = e4m3_rn_satfinite(p~)                 (PV A-operand codes)
  l      = l * alpha + sum_g e_g * sum_{j in g} p~ ,   e_g = ex2(ms_g - m*l2)  (pre-cast fp32 p~)
  O      = O * alpha + sum_g w_g * (C_g @ V8_g) ,       w_g = v8s_g * e_g      (one K=32 kind::f8f6f4 MMA per group
                                                                              into a zero-init fp32 TMEM temp, fp32 FMA fold)
  O_out  = bf16_rn(O * (1/l)) ;  LSE = m*(l2*ln2) + ln(l)  (natural log)
All exps are MUFU ex2.approx.ftz.f32; no pow2 / MX approximation of any factor.

Layout (bring-up "F1-T", 1-CTA): 16 warps / 512 threads. softmax WG 0-3 (one thread per query row), correction WG 8-11
(O fold + epilogue), MMA warp 12, load warp 14, the rest idle. Tile 128 q x 128 keys, q_stage 1, S ping-pong.
TMEM (512 cols): slot0 [0,128), slot1 [128,256), O [256,512) (O is only touched by the correction WG; no MMA writes O).
Slot j&1 holds S(j) (QK of key tile j); once the softmax has read S(j) the same columns hold the per-(group, d-slab)
PV temp T of tile j (M=128, N=128 d, K=32 keys). P codes live in SMEM (SS MMA), 2 x 16 KiB.
SMEM: sQ 32 KiB, K/V ring kv_stage x 32 KiB, sP 32 KiB, stats 5 KiB.
"""
import math

import numpy as np
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, const_expr
from cutlass.cute.nvgpu import cpasync
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils_basic
from cutlass import pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

import flash_attn.cute.pipeline as pipeline_custom
from flash_attn.cute.named_barrier import NamedBarrierFwdSm100

from delta_matching.sm100 import dm_utils

LOG2E = float(np.float32(math.log2(math.e)))           # float(M_LOG2E) in fp32
LN2 = float(np.float32(math.log(2.0)))                 # __logf: lg2.approx * 0x3F317218
L2LN2 = float(np.float32(np.float32(LOG2E) * np.float32(LN2)))   # l2 * float(M_LN2), fp32 product (spec §2.5)
LOG2_448 = float(np.float32(8.8073549220576042))
FLT_MIN = 1.1754943508222875e-38
FLOOR_B = 12.0


# ---------------------------------------------------------------------------------------------------------------------
# explicit-rounding scalar ops (PTX .rn forbids FMA contraction, so every op is exactly the one written)
# ---------------------------------------------------------------------------------------------------------------------
def _f1(asm, *xs, loc=None, ip=None):
    return Float32(llvm.inline_asm(T.f32(), [Float32(x).ir_value(loc=loc, ip=ip) for x in xs], asm,
                                   "=f" + ",f" * len(xs), has_side_effects=False, is_align_stack=False,
                                   asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def mul_rn(a, b, *, loc=None, ip=None):
    return _f1("mul.rn.f32 $0, $1, $2;", a, b, loc=loc, ip=ip)


@dsl_user_op
def add_rn(a, b, *, loc=None, ip=None):
    return _f1("add.rn.f32 $0, $1, $2;", a, b, loc=loc, ip=ip)


@dsl_user_op
def sub_rn(a, b, *, loc=None, ip=None):
    return _f1("sub.rn.f32 $0, $1, $2;", a, b, loc=loc, ip=ip)


@dsl_user_op
def fma_rn(a, b, c, *, loc=None, ip=None):
    return _f1("fma.rn.f32 $0, $1, $2, $3;", a, b, c, loc=loc, ip=ip)


@dsl_user_op
def fmax_f(a, b, *, loc=None, ip=None):
    return _f1("max.f32 $0, $1, $2;", a, b, loc=loc, ip=ip)


@dsl_user_op
def ex2_ftz(a, *, loc=None, ip=None):
    return _f1("ex2.approx.ftz.f32 $0, $1;", a, loc=loc, ip=ip)


@dsl_user_op
def lg2_ftz(a, *, loc=None, ip=None):
    return _f1("lg2.approx.ftz.f32 $0, $1;", a, loc=loc, ip=ip)


@dsl_user_op
def rcp_ftz(a, *, loc=None, ip=None):
    return _f1("rcp.approx.ftz.f32 $0, $1;", a, loc=loc, ip=ip)


@dsl_user_op
def e4m3x4(x0, x1, x2, x3, *, loc=None, ip=None) -> Int32:
    """4 fp32 -> 4 e4m3 bytes (x0 in the low byte), cvt.rn.satfinite (spec: convert_type_out)."""
    return Int32(llvm.inline_asm(
        T.i32(), [Float32(x).ir_value(loc=loc, ip=ip) for x in (x0, x1, x2, x3)],
        "{\n\t.reg .b16 lo, hi;\n\tcvt.rn.satfinite.e4m3x2.f32 lo, $2, $1;\n\tcvt.rn.satfinite.e4m3x2.f32 hi, $4, $3;\n\t"
        "mov.b32 $0, {lo, hi};\n\t}", "=r,f,f,f,f", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def bf16x2(lo, hi, *, loc=None, ip=None) -> Int32:
    return Int32(llvm.inline_asm(
        T.i32(), [Float32(lo).ir_value(loc=loc, ip=ip), Float32(hi).ir_value(loc=loc, ip=ip)],
        "cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,f,f", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def st_global_v4_b32(addr: Int64, a, b, c, d, *, loc=None, ip=None):
    llvm.inline_asm(None, [Int64(addr).ir_value(loc=loc, ip=ip)] + [Int32(x).ir_value(loc=loc, ip=ip) for x in (a, b, c, d)],
                    "st.global.v4.b32 [$0], {$1, $2, $3, $4};", "l,r,r,r,r", has_side_effects=True,
                    is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def st_shared_f32(addr: Int32, v, *, loc=None, ip=None):
    llvm.inline_asm(None, [Int32(addr).ir_value(loc=loc, ip=ip), Float32(v).ir_value(loc=loc, ip=ip)],
                    "st.shared.f32 [$0], $1;", "r,f", has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def ld_shared_f32(addr: Int32, *, loc=None, ip=None):
    return Float32(llvm.inline_asm(T.f32(), [Int32(addr).ir_value(loc=loc, ip=ip)], "ld.shared.f32 $0, [$1];",
                                   "=f,r", has_side_effects=True, is_align_stack=False,
                                   asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def tmem_wait_st(*, loc=None, ip=None):
    llvm.inline_asm(None, [], "tcgen05.wait::st.sync.aligned;", "", has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


class DMFwdHd256Sm100:
    """See module docstring. Inputs (all contiguous):
    q8 (B,N,H,256) e4m3 (sm baked), k8/v8 (B,N,Hk,256) e4m3; qs (B,H,N/64), ks/vs (B,Hk,N/32) fp32 (non-dyadic);
    O (B,N,H,256) bf16, LSE (B,H,N) fp32 natural log."""

    def __init__(self, is_causal: bool, qhead_per_kvhead: int, kv_stage: int = 4, numerics: str = "a", t_n: int = 64):
        assert numerics == "a", f"DM_HD256_NUMERICS={numerics}: only scheme 'a' (exact sm90 [64,32] floor) exists"
        self.is_causal = is_causal
        self.qhead_per_kvhead = qhead_per_kvhead
        self.m_block_size = 128
        self.n_block_size = 128
        self.head_dim = 256
        self.kv_stage = kv_stage
        # PV temp T: N = t_n d-columns per group MMA. t_n=128: one T buffer per slot (8 group/slab MMAs per tile,
        # MMA and fold strictly alternate). t_n=64: two T buffers per slot (16 MMAs per tile), the MMA of fold f+1 runs
        # while the correction folds f. Same arithmetic per element (the d-slab split does not change any value).
        assert t_n in (64, 128)
        self.t_n = t_n
        self.t_nbuf = 128 // t_n
        self.t_nslab = 256 // t_n
        self.mma_tiler_qk = (128, 128, 256)
        self.mma_tiler_pv = (128, 256, 128)      # smem / TMA tile of V; the MMAs use N=128 d-slabs
        self.softmax_warp_ids = (0, 1, 2, 3)
        self.correction_warp_ids = (8, 9, 10, 11)
        self.mma_warp_id = 12
        self.load_warp_id = 14
        self.idle_warp_ids = (4, 5, 6, 7, 13, 15)
        self.threads_per_cta = 512
        self.num_regs_softmax = 256
        self.num_regs_correction = 160
        self.num_regs_other = 48
        self.buffer_align_bytes = 1024
        # TMEM columns
        self.tmem_slot = (0, 128)
        self.tmem_o = 256

    @cute.jit
    def __call__(self, mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor, mO: cute.Tensor, mLSE: cute.Tensor,
                 mQS: cute.Tensor, mKS: cute.Tensor, mVS: cute.Tensor, stream):
        self.q_dtype = mQ.element_type
        mO_raw = mO
        # (b, s, h, d) -> (s, d, h, b); V -> (d, s, h, b)
        mQt = cute.make_tensor(mQ.iterator, cute.select(mQ.layout, mode=[1, 3, 2, 0]))
        mKt = cute.make_tensor(mK.iterator, cute.select(mK.layout, mode=[1, 3, 2, 0]))
        mVt = cute.make_tensor(mV.iterator, cute.select(mV.layout, mode=[3, 1, 2, 0]))

        cta_group = tcgen05.CtaGroup.ONE
        K = cute.nvgpu.OperandMajorMode.K
        MN = cute.nvgpu.OperandMajorMode.MN
        tiled_mma_qk = sm100_utils_basic.make_trivial_tiled_mma(
            self.q_dtype, self.q_dtype, K, K, Float32, cta_group, self.mma_tiler_qk[:2])
        tiled_mma_pv = sm100_utils_basic.make_trivial_tiled_mma(
            self.q_dtype, self.q_dtype, K, MN, Float32, cta_group, (128, self.t_n))
        cluster_layout_vmnk = cute.tiled_divide(cute.make_layout((1, 1, 1)), (tiled_mma_qk.thr_id.shape,))

        sQ_layout = sm100_utils_basic.make_smem_layout_a(tiled_mma_qk, self.mma_tiler_qk, self.q_dtype, 1)
        sK_layout = sm100_utils_basic.make_smem_layout_b(tiled_mma_qk, self.mma_tiler_qk, self.q_dtype, self.kv_stage)
        sV_layout = sm100_utils_basic.make_smem_layout_b(tiled_mma_pv, self.mma_tiler_pv, self.q_dtype, self.kv_stage)
        sP_layout = sm100_utils_basic.make_smem_layout_a(tiled_mma_pv, self.mma_tiler_pv, self.q_dtype, 2)

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)
        tma_atom_Q, mQt = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op, mQt, cute.select(sQ_layout, mode=[0, 1, 2]), self.mma_tiler_qk, tiled_mma_qk,
            cluster_layout_vmnk.shape)
        tma_atom_K, mKt = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op, mKt, cute.select(sK_layout, mode=[0, 1, 2]), self.mma_tiler_qk, tiled_mma_qk,
            cluster_layout_vmnk.shape)
        tma_atom_V, mVt = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op, mVt, cute.select(sV_layout, mode=[0, 1, 2]), self.mma_tiler_pv, tiled_mma_pv,
            cluster_layout_vmnk.shape)
        self.tma_bytes_q = cute.size_in_bytes(self.q_dtype, cute.select(sQ_layout, mode=[0, 1, 2]))
        self.tma_bytes_kv = cute.size_in_bytes(self.q_dtype, cute.select(sK_layout, mode=[0, 1, 2]))
        tma_bytes_v = cute.size_in_bytes(self.q_dtype, cute.select(sV_layout, mode=[0, 1, 2]))
        assert self.tma_bytes_kv == tma_bytes_v

        sKV_size = cutlass.max(cute.cosize(sK_layout), cute.cosize(sV_layout))

        @cute.struct
        class SharedStorage:
            mbar_q: cute.struct.MemRange[Int64, 2]
            mbar_kv: cute.struct.MemRange[Int64, self.kv_stage * 2]
            # raw: S_full[2], P_full[2], T_full, T_empty, stats_full[2], final_full, pad
            mbar_dm: cute.struct.MemRange[Int64, 12]
            tmem_holding_buf: Int32
            sStats: cute.struct.MemRange[Float32, 2 * 5 * 128 + 128]
            sQ: cute.struct.Align[cute.struct.MemRange[self.q_dtype, cute.cosize(sQ_layout)], self.buffer_align_bytes]
            sK: cute.struct.Align[cute.struct.MemRange[self.q_dtype, sKV_size], self.buffer_align_bytes]
            sP: cute.struct.Align[cute.struct.MemRange[self.q_dtype, cute.cosize(sP_layout)], self.buffer_align_bytes]

        self.shared_storage = SharedStorage
        assert SharedStorage.size_in_bytes() <= 232448, f"SMEM {SharedStorage.size_in_bytes()}"

        n_m = cute.ceil_div(cute.size(mQt.shape[0]), self.m_block_size)
        grid = (n_m, cute.size(mQt.shape[2]), cute.size(mQt.shape[3]))
        self.kernel(mQt, mKt, mVt, mO_raw, mLSE, mQS, mKS, mVS, tma_atom_Q, tma_atom_K, tma_atom_V,
                    sQ_layout, sK_layout, sV_layout, sP_layout, tiled_mma_qk, tiled_mma_pv).launch(
            grid=grid, block=[self.threads_per_cta, 1, 1], stream=stream, min_blocks_per_mp=1)

    @cute.kernel
    def kernel(self, mQ, mK, mV, mO, mLSE, mQS, mKS, mVS, tma_atom_Q, tma_atom_K, tma_atom_V,
               sQ_layout: cute.ComposedLayout, sK_layout: cute.ComposedLayout, sV_layout: cute.ComposedLayout,
               sP_layout: cute.ComposedLayout, tiled_mma_qk: cute.TiledMma, tiled_mma_pv: cute.TiledMma):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            for a in (tma_atom_Q, tma_atom_K, tma_atom_V):
                cpasync.prefetch_descriptor(a)
        bidx, bidy, bidz = cute.arch.block_idx()
        n_m = cute.arch.grid_dim()[0]
        # LPT for causal: heaviest q-blocks first
        m_block = n_m - 1 - bidx if const_expr(self.is_causal) else bidx
        head_idx = bidy
        batch_idx = bidz
        kv_head_idx = head_idx // self.qhead_per_kvhead
        n_block_max = m_block + 1 if const_expr(self.is_causal) else n_m
        cnt = n_block_max      # key tiles n = n_block_max-1 ... 0, visited descending (spec §2.1)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        tmem_alloc_barrier = pipeline_custom.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100.TmemPtr),
            num_threads=cute.arch.WARP_SIZE * (1 + len(self.softmax_warp_ids) + len(self.correction_warp_ids)))
        tmem = cutlass.utils.TmemAllocator(storage.tmem_holding_buf.ptr, barrier_for_retrieve=tmem_alloc_barrier,
                                           allocator_warp_id=self.mma_warp_id, is_two_cta=False)

        TCG = pipeline.CooperativeGroup
        mma_warp = TCG(pipeline.Agent.Thread, 1)
        tma_warp = TCG(pipeline.Agent.Thread, 1)
        cluster_layout_vmnk = cute.tiled_divide(cute.make_layout((1, 1, 1)), (tiled_mma_qk.thr_id.shape,))
        pipeline_q = pipeline_custom.PipelineTmaUmma.create(
            barrier_storage=storage.mbar_q.data_ptr(), num_stages=1, producer_group=tma_warp, consumer_group=mma_warp,
            tx_count=self.tma_bytes_q, cta_layout_vmnk=cluster_layout_vmnk, defer_sync=True)
        pipeline_kv = pipeline_custom.PipelineTmaUmma.create(
            barrier_storage=storage.mbar_kv.data_ptr(), num_stages=self.kv_stage, producer_group=tma_warp,
            consumer_group=mma_warp, tx_count=self.tma_bytes_kv, cta_layout_vmnk=cluster_layout_vmnk, defer_sync=True)
        mbar = Int32(storage.mbar_dm.data_ptr().toint())
        # mbar + 8*i: 0,1 S_full[slot] | 2,3 P_full[slot] | 4,5 T_full[buf] | 6,7 stats_full[slot] | 8 final_full |
        #             9,10 T_empty[buf]
        if cute.arch.thread_idx()[0] == 0:
            dm_utils.mbar_init(mbar + 0, 1)
            dm_utils.mbar_init(mbar + 8, 1)
            dm_utils.mbar_init(mbar + 16, 128)
            dm_utils.mbar_init(mbar + 24, 128)
            dm_utils.mbar_init(mbar + 32, 1)
            dm_utils.mbar_init(mbar + 40, 1)
            dm_utils.mbar_init(mbar + 72, 128)
            dm_utils.mbar_init(mbar + 80, 128)
            dm_utils.mbar_init(mbar + 48, 128)
            dm_utils.mbar_init(mbar + 56, 128)
            dm_utils.mbar_init(mbar + 64, 128)
            cute.arch.mbarrier_init_fence()
        pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = cute.make_tensor(cute.recast_ptr(sK.iterator, sV_layout.inner), sV_layout.outer)
        sP = storage.sP.get_tensor(sP_layout.outer, swizzle=sP_layout.inner)
        stats = Int32(storage.sStats.data_ptr().toint())
        sp_base = Int32(sP.iterator.toint())

        thr_mma_qk = tiled_mma_qk.get_slice(0)
        thr_mma_pv = tiled_mma_pv.get_slice(0)
        acc_qk = thr_mma_qk.make_fragment_C(thr_mma_qk.partition_shape_C(self.mma_tiler_qk[:2]))
        acc_pv = thr_mma_pv.make_fragment_C(thr_mma_pv.partition_shape_C((128, self.t_n)))

        pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        for w in cutlass.range_constexpr(len(self.idle_warp_ids)):
            if warp_idx == self.idle_warp_ids[w]:
                cute.arch.setmaxregister_decrease(self.num_regs_other)

        # ---------------------------------------------------------------- LOAD
        if warp_idx == self.load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            mQ_cur = mQ[None, None, head_idx, batch_idx]
            mK_cur = mK[None, None, kv_head_idx, batch_idx]
            mV_cur = mV[None, None, kv_head_idx, batch_idx]
            gQ = cute.local_tile(mQ_cur, (self.m_block_size, self.head_dim), (m_block, 0))
            gK = cute.local_tile(mK_cur, (self.n_block_size, self.head_dim), (None, 0))
            gV = cute.local_tile(mV_cur, (self.head_dim, self.n_block_size), (0, None))
            tSgQ = thr_mma_qk.partition_A(gQ)
            tSgK = thr_mma_qk.partition_B(gK)
            tOgV = thr_mma_pv.partition_B(gV)
            tQsQ, tQgQ = cpasync.tma_partition(tma_atom_Q, 0, cute.make_layout(1), cute.group_modes(sQ, 0, 3),
                                               cute.group_modes(tSgQ, 0, 3))
            tKsK, tKgK = cpasync.tma_partition(tma_atom_K, 0, cute.make_layout(1), cute.group_modes(sK, 0, 3),
                                               cute.group_modes(tSgK, 0, 3))
            tVsV, tVgV = cpasync.tma_partition(tma_atom_V, 0, cute.make_layout(1), cute.group_modes(sV, 0, 3),
                                               cute.group_modes(tOgV, 0, 3))
            kv_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.kv_stage)
            q_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            # K(n_max-1), Q, then [K(n-1), V(n)] ..., V(0): the MMA consumption order QK0, [QK(i+1), PV(i)], PV(last)
            pipeline_kv.producer_acquire(kv_state)
            cute.copy(tma_atom_K, tKgK[None, n_block_max - 1], tKsK[None, kv_state.index],
                      tma_bar_ptr=pipeline_kv.producer_get_barrier(kv_state))
            kv_state.advance()
            pipeline_q.producer_acquire(q_state)
            cute.copy(tma_atom_Q, tQgQ, tQsQ[None, 0], tma_bar_ptr=pipeline_q.producer_get_barrier(q_state))
            for i in cutlass.range(cnt - 1, unroll=1):
                pipeline_kv.producer_acquire(kv_state)
                cute.copy(tma_atom_K, tKgK[None, n_block_max - 2 - i], tKsK[None, kv_state.index],
                          tma_bar_ptr=pipeline_kv.producer_get_barrier(kv_state))
                kv_state.advance()
                pipeline_kv.producer_acquire(kv_state)
                cute.copy(tma_atom_V, tVgV[None, n_block_max - 1 - i], tVsV[None, kv_state.index],
                          tma_bar_ptr=pipeline_kv.producer_get_barrier(kv_state))
                kv_state.advance()
            pipeline_kv.producer_acquire(kv_state)
            cute.copy(tma_atom_V, tVgV[None, 0], tVsV[None, kv_state.index],
                      tma_bar_ptr=pipeline_kv.producer_get_barrier(kv_state))
            kv_state.advance()

        # ---------------------------------------------------------------- MMA
        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            tmem.allocate(512)
            tmem_alloc_barrier.arrive_and_wait_unaligned()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            self.mma(tiled_mma_qk, tiled_mma_pv, sQ, sK, sV, sP, acc_qk, acc_pv, pipeline_q, pipeline_kv, mbar, cnt)
            tmem.relinquish_alloc_permit()
            tmem_alloc_barrier.arrive_and_wait_unaligned()
            tmem.free(tmem_ptr)

        # ---------------------------------------------------------------- SOFTMAX
        if warp_idx <= self.softmax_warp_ids[-1]:
            cute.arch.setmaxregister_increase(self.num_regs_softmax)
            tmem_alloc_barrier.arrive_and_wait_unaligned()
            self.softmax(acc_qk, mbar, stats, sp_base, mLSE, mQS, mKS, mVS, m_block, head_idx, kv_head_idx,
                         batch_idx, n_block_max, cnt)
            tmem_alloc_barrier.arrive_unaligned()

        # ---------------------------------------------------------------- CORRECTION (O fold + epilogue)
        if warp_idx >= self.correction_warp_ids[0] and warp_idx <= self.correction_warp_ids[-1]:
            cute.arch.setmaxregister_increase(self.num_regs_correction)
            tmem_alloc_barrier.arrive_and_wait_unaligned()
            self.correction(acc_pv, mbar, stats, mO, m_block, head_idx, batch_idx, cnt)
            tmem_alloc_barrier.arrive_unaligned()
        return

    # -----------------------------------------------------------------------------------------------------------------
    @cute.jit
    def mma(self, tiled_mma_qk, tiled_mma_pv, sQ, sK, sV, sP, acc_qk, acc_pv, pipeline_q, pipeline_kv, mbar, cnt):
        tSrQ = tiled_mma_qk.make_fragment_A(sQ)          # (MMA, 1, 8, 1)
        tSrK = tiled_mma_qk.make_fragment_B(sK)          # (MMA, 1, 8, kv_stage)
        tOrP = tiled_mma_pv.make_fragment_A(sP)          # (MMA, 1, 4, 2)
        tOrV = tiled_mma_pv.make_fragment_B(sV)          # (MMA, 2, 4, kv_stage)
        atom_qk = cute.make_mma_atom(tiled_mma_qk.op)
        atom_pv = cute.make_mma_atom(tiled_mma_pv.op)
        atom_pv.set(tcgen05.Field.ACCUMULATE, False)     # every PV group MMA is a zero-init K=32 temp
        kv_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.kv_stage)
        q_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
        pipeline_q.consumer_wait(q_state)

        # QK(0) -> slot 0
        pipeline_kv.consumer_wait(kv_state)
        self.mma_qk(atom_qk, acc_qk, tSrQ, tSrK, kv_state.index, Int32(0))
        with cute.arch.elect_one():
            dm_utils.tc_commit(mbar + 0)
        pipeline_kv.consumer_release(kv_state)
        kv_state.advance()
        for i in cutlass.range(cnt - 1, unroll=1):
            j = i + 1
            slot = j & 1
            if i >= 1:
                # slot j&1 held the PV temps of tile j-2: wait for the last fold of every T buffer (8 uses per buffer
                # per tile -> the last use has parity 1)
                for b in cutlass.range_constexpr(self.t_nbuf):
                    dm_utils.mbar_wait_parity(mbar + 72 + 8 * b, Int32(1))
                dm_utils.tc_fence_after()
            pipeline_kv.consumer_wait(kv_state)
            self.mma_qk(atom_qk, acc_qk, tSrQ, tSrK, kv_state.index, slot)
            with cute.arch.elect_one():
                dm_utils.tc_commit(mbar + 8 * slot)
            pipeline_kv.consumer_release(kv_state)
            kv_state.advance()
            self.mma_pv(atom_pv, acc_pv, tOrP, tOrV, pipeline_kv, kv_state, mbar, i)
            kv_state.advance()
        self.mma_pv(atom_pv, acc_pv, tOrP, tOrV, pipeline_kv, kv_state, mbar, cnt - 1)
        kv_state.advance()
        pipeline_q.consumer_release(q_state)

    @cute.jit
    def mma_qk(self, atom_qk, acc_qk, tSrQ, tSrK, stage, slot):
        tS = cute.make_tensor(acc_qk.iterator + slot * 128, acc_qk.layout)
        for k in cutlass.range_constexpr(cute.size(tSrQ.shape[2])):     # 8 K=32 steps, increasing d
            atom_qk.set(tcgen05.Field.ACCUMULATE, k != 0)
            cute.gemm(atom_qk, tS, tSrQ[None, None, k, 0], tSrK[None, None, k, stage], tS)

    @cute.jit
    def mma_pv(self, atom_pv, acc_pv, tOrP, tOrV, pipeline_kv, kv_state, mbar, i):
        slot = i & 1
        pipeline_kv.consumer_wait(kv_state)
        # P(i) in sP[slot] written and S(i) read by the softmax (P_full parity = use count of the slot)
        dm_utils.mbar_wait_parity(mbar + 16 + 8 * slot, (i >> 1) & 1)
        dm_utils.tc_fence_after()
        nb = self.t_nbuf
        for f in cutlass.range_constexpr(4 * self.t_nslab):      # fold f: d-slab s = f // 4, key group g = f % 4
            s, g, b = f // 4, f % 4, f % nb
            tT = cute.make_tensor(acc_pv.iterator + slot * 128 + self.t_n * b, acc_pv.layout)
            if const_expr(f >= nb):
                # buffer b free: the correction read fold f - nb of this tile (use (f - nb) // nb of the buffer)
                dm_utils.mbar_wait_parity(mbar + 72 + 8 * b, Int32(((f - nb) // nb) & 1))
                dm_utils.tc_fence_after()
            else:
                # first use of buffer b in tile i: the last use in tile i-1 (other slot) must be read, so that
                # T_full[b] never runs two phases ahead of the correction (8 uses per tile -> parity 1)
                if i >= 1:
                    dm_utils.mbar_wait_parity(mbar + 72 + 8 * b, Int32(1))
                    dm_utils.tc_fence_after()
            cute.gemm(atom_pv, tT[None, 0, 0], tOrP[None, 0, g, slot], tOrV[None, s, g, kv_state.index], tT[None, 0, 0])
            with cute.arch.elect_one():
                dm_utils.tc_commit(mbar + 32 + 8 * b)
        pipeline_kv.consumer_release(kv_state)

    # -----------------------------------------------------------------------------------------------------------------
    @cute.jit
    def softmax(self, acc_qk, mbar, stats, sp_base, mLSE, mQS, mKS, mVS, m_block, head_idx, kv_head_idx, batch_idx,
                n_block_max, cnt):
        tidx = cute.arch.thread_idx()[0] % 128
        tSAcc = acc_qk[(None, None), 0, 0]
        ld_atom = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), Float32)
        thr_ld = tcgen05.make_tmem_copy(ld_atom, tSAcc).get_slice(tidx)
        tStS = thr_ld.partition_S(tSAcc)
        tScS = thr_ld.partition_D(cute.make_identity_tensor((128, 128)))
        row = m_block * self.m_block_size + tidx
        qsc = Float32(mQS[batch_idx, head_idx, row // 64])
        m_old = -Float32.inf
        row_sum = Float32(0.0)
        m_old, row_sum = self.softmax_step(Int32(0), n_block_max - 1, True, thr_ld, tStS, tScS, mbar, stats, sp_base,
                                           tidx, qsc, mKS, mVS, kv_head_idx, batch_idx, m_old, row_sum)
        for i in cutlass.range(cnt - 1, unroll=1):
            m_old, row_sum = self.softmax_step(i + 1, n_block_max - 2 - i, False, thr_ld, tStS, tScS, mbar, stats,
                                               sp_base, tidx, qsc, mKS, mVS, kv_head_idx, batch_idx, m_old, row_sum)
        # finalize (spec §2.5)
        bad = row_sum == Float32(0.0) or row_sum != row_sum
        inv = Float32(0.0)
        lse = -Float32.inf
        if not bad:
            inv = rcp_ftz(row_sum)
            lse = fma_rn(m_old, Float32(L2LN2), mul_rn(lg2_ftz(row_sum), Float32(LN2)))
        mLSE[batch_idx, head_idx, row] = lse
        st_shared_f32(stats + 4 * (2 * 5 * 128 + tidx), inv)
        dm_utils.mbar_arrive(mbar + 64)

    @cute.jit
    def softmax_step(self, j, n_block, is_first: cutlass.Constexpr, thr_ld, tStS, tScS, mbar, stats, sp_base, tidx,
                     qsc, mKS, mVS, kv_head_idx, batch_idx, m_old, row_sum):
        slot = j & 1
        dm_utils.mbar_wait_parity(mbar + 8 * slot, (j >> 1) & 1)
        dm_utils.tc_fence_after()
        tStS_cur = cute.make_tensor(tStS.iterator + slot * 128, tStS.layout)
        s = cute.make_rmem_tensor(tScS.shape, Float32)
        cute.copy(thr_ld, tStS_cur, s)
        # causal: only the first (diagonal) tile n_block == m_block is masked: key col > query row -> -inf on S_acc
        if const_expr(self.is_causal and is_first):
            for c in cutlass.range_constexpr(128):
                if Int32(c) > tidx:
                    s[c] = -Float32.inf
        c_g = [None] * 4
        v_g = [None] * 4
        for g in cutlass.range_constexpr(4):
            ks = Float32(mKS[batch_idx, kv_head_idx, n_block * 4 + g])
            v_g[g] = Float32(mVS[batch_idx, kv_head_idx, n_block * 4 + g])
            c_g[g] = fmax_f(mul_rn(qsc, ks), Float32(FLT_MIN))     # (q8sc*k8s) product first; FLT_MIN floor
        gm = [None] * 4
        for g in cutlass.range_constexpr(4):
            for c in cutlass.range_constexpr(0, 32, 2):
                k = 32 * g + c
                s[k], s[k + 1] = cute.arch.mul_packed_f32x2((s[k], s[k + 1]), (c_g[g], c_g[g]))
            mx = fmax_f(s[32 * g], s[32 * g + 1])
            for c in cutlass.range_constexpr(2, 32):
                mx = fmax_f(mx, s[32 * g + c])
            gm[g] = mx
        tile_max = fmax_f(fmax_f(gm[0], gm[1]), fmax_f(gm[2], gm[3]))
        m_new = tile_max
        if const_expr(not is_first):
            m_new = fmax_f(m_old, tile_max)
        m_cur = m_new
        if m_new == -Float32.inf:
            m_cur = Float32(0.0)
        l2 = Float32(LOG2E)
        alpha = Float32(1.0)
        if const_expr(not is_first):
            alpha = ex2_ftz(mul_rn(sub_rn(m_old, m_cur), l2))
            row_sum = mul_rn(row_sum, alpha)
        ml2 = mul_rn(m_cur, l2)
        ms = [None] * 4
        e = [None] * 4
        for g in cutlass.range_constexpr(4):
            ms[g] = sub_rn(fmax_f(mul_rn(gm[g], l2), sub_rn(ml2, Float32(FLOOR_B))), Float32(LOG2_448))
            e[g] = ex2_ftz(sub_rn(ms[g], ml2))
        # p~ = ex2(fma(S, l2, -ms_g)); fp32 group sums; e4m3 codes -> sP[slot] (K-major SW128: row r at r*128 B,
        # 16-B chunk q at (q ^ (r & 7)) * 16)
        bsum = [None] * 4
        prow = sp_base + slot * 16384 + tidx * 128
        for g in cutlass.range_constexpr(4):
            nms = -ms[g]
            for c in cutlass.range_constexpr(0, 32, 2):
                k = 32 * g + c
                x0, x1 = cute.arch.fma_packed_f32x2((s[k], s[k + 1]), (l2, l2), (nms, nms))
                s[k] = ex2_ftz(x0)
                s[k + 1] = ex2_ftz(x1)
            ps = (s[32 * g], s[32 * g + 1])
            for c in cutlass.range_constexpr(2, 32, 2):
                ps = cute.arch.add_packed_f32x2(ps, (s[32 * g + c], s[32 * g + c + 1]))
            bsum[g] = add_rn(ps[0], ps[1])
            for q in cutlass.range_constexpr(2):
                k = 32 * g + 16 * q
                w = [e4m3x4(s[k + 4 * t], s[k + 4 * t + 1], s[k + 4 * t + 2], s[k + 4 * t + 3]) for t in range(4)]
                chunk = Int32(2 * g + q)
                dm_utils.st_shared_v4_b32(prow + ((chunk ^ (tidx & 7)) << 4), w[0], w[1], w[2], w[3])
        cute.arch.fence_view_async_shared()
        dm_utils.tc_fence_before()
        dm_utils.mbar_arrive(mbar + 16 + 8 * slot)          # P_full[slot]: S(j) read, P(j) in sP
        for g in cutlass.range_constexpr(4):
            row_sum = fma_rn(e[g], bsum[g], row_sum)
        sb = stats + 4 * (slot * 640 + tidx)
        st_shared_f32(sb, alpha)
        for g in cutlass.range_constexpr(4):
            st_shared_f32(sb + 4 * 128 * (g + 1), mul_rn(v_g[g], e[g]))      # w_g = v8s_g * e_g
        dm_utils.mbar_arrive(mbar + 48 + 8 * slot)
        return m_new, row_sum

    # -----------------------------------------------------------------------------------------------------------------
    @cute.jit
    def correction(self, acc_pv, mbar, stats, mO, m_block, head_idx, batch_idx, cnt):
        tidx = cute.arch.thread_idx()[0] % 128
        tAcc = acc_pv[(None, None), 0, 0]      # (128, t_n)
        tAcc32 = cute.composition(tAcc, cute.make_layout((128, 32)))
        ld_atom = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), Float32)
        st_atom = cute.make_copy_atom(tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(32)), Float32)
        thr_ld = tcgen05.make_tmem_copy(ld_atom, tAcc32).get_slice(tidx)
        thr_st = tcgen05.make_tmem_copy(st_atom, tAcc32).get_slice(tidx)
        tX_ld = thr_ld.partition_S(tAcc32)
        tX_st = thr_st.partition_D(tAcc32)
        frag_shape = thr_ld.partition_D(cute.make_identity_tensor((128, 32))).shape
        self.correction_tile(Int32(0), True, thr_ld, thr_st, tX_ld, tX_st, frag_shape, mbar, stats, tidx)
        for i in cutlass.range(cnt - 1, unroll=1):
            self.correction_tile(i + 1, False, thr_ld, thr_st, tX_ld, tX_st, frag_shape, mbar, stats, tidx)
        # epilogue: O * inv -> bf16, row-contiguous 16-B global stores
        dm_utils.mbar_wait_parity(mbar + 64, Int32(0))
        inv = ld_shared_f32(stats + 4 * (2 * 5 * 128 + tidx))
        N = mO.shape[1]
        H = mO.shape[2]
        row = m_block * self.m_block_size + tidx
        base = Int64(mO.iterator.toint()) + (Int64((batch_idx * N + row) * H + head_idx) * Int64(self.head_dim * 2))
        for c in cutlass.range_constexpr(8):
            o = cute.make_rmem_tensor(frag_shape, Float32)
            cute.copy(thr_ld, cute.make_tensor(tX_ld.iterator + self.tmem_o + 32 * c, tX_ld.layout), o)
            for q in cutlass.range_constexpr(4):
                wv = [bf16x2(mul_rn(o[8 * q + 2 * t], inv), mul_rn(o[8 * q + 2 * t + 1], inv)) for t in range(4)]
                st_global_v4_b32(base + Int64(64 * c + 16 * q), wv[0], wv[1], wv[2], wv[3])

    @cute.jit
    def correction_tile(self, j, is_first: cutlass.Constexpr, thr_ld, thr_st, tX_ld, tX_st, frag_shape, mbar, stats,
                        tidx):
        slot = j & 1
        dm_utils.mbar_wait_parity(mbar + 48 + 8 * slot, (j >> 1) & 1)
        sb = stats + 4 * (slot * 640 + tidx)
        alpha = ld_shared_f32(sb)
        w = [ld_shared_f32(sb + 4 * 128 * (g + 1)) for g in range(4)]
        nb = self.t_nbuf
        for f in cutlass.range_constexpr(4 * self.t_nslab):
            s, g, b = f // 4, f % 4, f % nb
            dm_utils.mbar_wait_parity(mbar + 32 + 8 * b, Int32((f // nb) & 1))     # T_full[b] (8 uses per tile)
            dm_utils.tc_fence_after()
            for c in cutlass.range_constexpr(self.t_n // 32):
                t = cute.make_rmem_tensor(frag_shape, Float32)
                cute.copy(thr_ld, cute.make_tensor(tX_ld.iterator + slot * 128 + self.t_n * b + 32 * c, tX_ld.layout), t)
                o = cute.make_rmem_tensor(frag_shape, Float32)
                o_off = self.tmem_o + self.t_n * s + 32 * c
                if const_expr(is_first and g == 0):
                    for k in cutlass.range_constexpr(0, 32, 2):
                        o[k], o[k + 1] = cute.arch.mul_packed_f32x2((t[k], t[k + 1]), (w[g], w[g]))
                else:
                    cute.copy(thr_ld, cute.make_tensor(tX_ld.iterator + o_off, tX_ld.layout), o)
                    if const_expr(g == 0):
                        for k in cutlass.range_constexpr(0, 32, 2):
                            o[k], o[k + 1] = cute.arch.mul_packed_f32x2((o[k], o[k + 1]), (alpha, alpha))
                    for k in cutlass.range_constexpr(0, 32, 2):
                        o[k], o[k + 1] = cute.arch.fma_packed_f32x2((t[k], t[k + 1]), (w[g], w[g]),
                                                                   (o[k], o[k + 1]))
                cute.copy(thr_st, o, cute.make_tensor(tX_st.iterator + o_off, tX_st.layout))
            tmem_wait_st()
            dm_utils.tc_fence_before()
            dm_utils.mbar_arrive(mbar + 72 + 8 * b)                                # T_empty[b]
