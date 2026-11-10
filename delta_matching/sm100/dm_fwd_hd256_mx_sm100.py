"""hd256 Delta-Matching forward for sm100 (GB200), sm100-mx numerics (MXFP8 is the sm100 spec at
both head dims; validator preset sm100-mx-d256).  The hd128 forward's mx path (dm_fwd_sm100.py softmax_step, "DM sm100-mx
softmax step") at D=256, op for op:
  inputs   one power-of-2 scale per 32 tokens over ALL 256 d (quant.quant_qkv2 at D=256)
  S        S_acc = tcgen05 kind::f8f6f4 over D=256 (8 K=32 steps, ascending d); -inf mask on S_acc;
           c_b = (qs[row/32] * log2e) * ks_b (pow2), per 32-key block b: bmax_b = max(S_acc) * c_b
  m        integer running max in log2 units: m_cand = ceil(max_b bmax_b); first tile m = m_cand; later m grows only when
           m_cand > m + T (T = 32, lazy rescale), acc_scale = 2^(m_old - m_cand) (O and l rescaled)
  P        kb = max(floor(bmax_b), -1e30); x = fma(S_acc, c_b, 7 - kb); p = ex2.approx.ftz(x) in [0, 256); code = e4m3_rn(p)
           SFA_b = UE8M0(kb - 7 - m) (P block scale per (row, 32 keys)), weight w_b = 2^(kb - 7 - m)
  l        l = l * acc_scale + sum_b (pairwise sum of the fp32 p of block b) * w_b
  PV       ONE block-scaled tcgen05 kind::mxf8f6f4 (scale_vec::1X) MMA chain per tile: M128 N256 K=4x32, A = P codes (SMEM),
           B = V codes, SFA = P block scales, SFB = V scale bytes; accumulates straight into O (no folds)
  final    O_out = bf16(O * rcp.approx(l)), LSE = (m + lg2.approx(l)) * ln2 (natural log)
Layout (bring-up, 1-CTA): 16 warps / 512 thr: softmax WG 0-3, correction WG 8-11 (lazy O rescale + epilogue), MMA 12, load 14.
TMEM 512: slot s (S ping-pong) at 128 s: S(j) of key tile j; after the softmax read S(j) the slot holds the PV scale factors
of tile j: SFA at slot + 32 (4 cols, the row's packed 4-byte word replicated), SFB at slot + 64 (8 cols: the V-scale word
replicated on every lane, covers the N=256 SFB layout); O fp32 at [256, 512).  P codes in SMEM (2 x 16 KiB, SS MMA).
"""
import math

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
from flash_attn.cute import utils

from delta_matching.sm100.cute_ptx_sm100 import ceil_f32, floor_f32, mbar_init, tc_commit
from delta_matching.sm100.fp8_e8m0_scale_sm100 import byte_pow2_f32, byte_shl23_f32, exp2i_f32, pack_bytes4, ue8m0_byte
from delta_matching.sm100.mma_sm100 import gemm_ptx_bs
from delta_matching.sm100.tcgen05_fence_sm100 import mbar_arrive, mbar_wait_parity, tcgen05_fence_after, tcgen05_fence_before


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
def ex2_ftz(a, *, loc=None, ip=None):
    return _f1("ex2.approx.ftz.f32 $0, $1;", a, loc=loc, ip=ip)


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
def st_shared_v4_b32(addr: Int32, a, b, c, d, *, loc=None, ip=None):
    llvm.inline_asm(None, [Int32(addr).ir_value(loc=loc, ip=ip)] + [Int32(x).ir_value(loc=loc, ip=ip) for x in (a, b, c, d)],
                    "st.shared.v4.b32 [$0], {$1, $2, $3, $4};", "r,r,r,r,r", has_side_effects=True,
                    is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def ld_shared_v4_b32(addr: Int32, *, loc=None, ip=None):
    res = llvm.inline_asm(llvm.StructType.get_literal([T.i32()] * 4), [Int32(addr).ir_value(loc=loc, ip=ip)],
                          "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,r", has_side_effects=True,
                          is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return [Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def fence_proxy_async_shared(*, loc=None, ip=None):
    llvm.inline_asm(None, [], "fence.proxy.async.shared::cta;", "", has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def bar_warp_sync(*, loc=None, ip=None):
    llvm.inline_asm(None, [], "bar.warp.sync -1;", "", has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


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


@dsl_user_op
def tmem_ld32(addr: Int32, *, loc=None, ip=None):
    """tcgen05.ld.32x32b.x32 (thread t of the warp -> lane base + t, 32 consecutive columns); NO wait::ld."""
    regs = ",".join(f"${i}" for i in range(32))
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 32), [Int32(addr).ir_value(loc=loc, ip=ip)],
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 {" + regs + "}, [$32];",
        ",".join(["=f"] * 32) + ",r", has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return [Float32(llvm.extractvalue(T.f32(), res, [i], loc=loc, ip=ip)) for i in range(32)]


@dsl_user_op
def tmem_st32(addr: Int32, vals, *, loc=None, ip=None):
    regs = ",".join(f"${i + 1}" for i in range(32))
    llvm.inline_asm(None, [Int32(addr).ir_value(loc=loc, ip=ip)] + [Float32(v).ir_value(loc=loc, ip=ip) for v in vals],
                    "tcgen05.st.sync.aligned.32x32b.x32.b32 [$0], {" + regs + "};", "r" + ",f" * 32,
                    has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def tmem_wait_ld(*, loc=None, ip=None):
    llvm.inline_asm(None, [], "tcgen05.wait::ld.sync.aligned;", "", has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def tmem_st1(addr: Int32, w: Int32, *, loc=None, ip=None):
    llvm.inline_asm(None, [Int32(addr).ir_value(loc=loc, ip=ip), Int32(w).ir_value(loc=loc, ip=ip)],
                    "tcgen05.st.sync.aligned.32x32b.x1.b32 [$0], {$1};", "r,r",
                    has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


class DMFwdHd256MxSm100:
    """sm100-mx hd256 forward (module docstring).  Inputs (contiguous): q8 (B,N,H,256) e4m3 (sm baked), k8/v8 (B,N,Hk,256);
    qs (B,H,N/32) fp32 pow2; kvw (B,Hk,N/64) int32 packed UE8M0 words ([2t] K bytes of tile t, [2t+1] V bytes).
    Out: O (B,N,H,256) bf16, LSE (B,H,N) fp32 natural log."""

    def __init__(self, is_causal: bool, qhead_per_kvhead: int):
        self.is_causal = is_causal
        self.qhead_per_kvhead = qhead_per_kvhead
        self.m_block_size = 128
        self.n_block_size = 128
        self.head_dim = 256
        self.kv_stage = 4
        self.rescale_T = 32.0
        self.mma_tiler_qk = (128, 128, 256)
        self.mma_tiler_pv = (128, 256, 128)
        self.softmax_warp_ids = (0, 1, 2, 3, 4, 5, 6, 7)
        self.correction_warp_ids = (8, 9, 10, 11)
        self.mma_warp_id = 12
        self.load_warp_id = 14
        self.idle_warp_ids = (13, 15)
        self.threads_per_cta = 512
        self.num_regs_softmax = 192
        self.num_regs_correction = 80
        self.num_regs_other = 48
        self.buffer_align_bytes = 1024
        self.tmem_o = 256
        self.sfa_off = 32
        self.sfb_off = 64

    @cute.jit
    def __call__(self, mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor, mO: cute.Tensor, mLSE: cute.Tensor,
                 mQS: cute.Tensor, mKVW: cute.Tensor, stream):
        self.q_dtype = mQ.element_type
        mO_raw = mO
        mQt = cute.make_tensor(mQ.iterator, cute.select(mQ.layout, mode=[1, 3, 2, 0]))
        mKt = cute.make_tensor(mK.iterator, cute.select(mK.layout, mode=[1, 3, 2, 0]))
        mVt = cute.make_tensor(mV.iterator, cute.select(mV.layout, mode=[3, 1, 2, 0]))
        cta_group = tcgen05.CtaGroup.ONE
        K = cute.nvgpu.OperandMajorMode.K
        MN = cute.nvgpu.OperandMajorMode.MN
        tiled_mma_qk = sm100_utils_basic.make_trivial_tiled_mma(
            self.q_dtype, self.q_dtype, K, K, Float32, cta_group, self.mma_tiler_qk[:2])
        tiled_mma_pv = sm100_utils_basic.make_trivial_tiled_mma(
            self.q_dtype, self.q_dtype, K, MN, Float32, cta_group, self.mma_tiler_pv[:2])
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
        assert self.tma_bytes_kv == cute.size_in_bytes(self.q_dtype, cute.select(sV_layout, mode=[0, 1, 2]))
        sKV_size = cutlass.max(cute.cosize(sK_layout), cute.cosize(sV_layout))
        assert sKV_size * (self.q_dtype.width // 8) >= 128 * 528, "F1a O staging needs 66 KiB of the K/V ring"

        @cute.struct
        class SharedStorage:
            mbar_q: cute.struct.MemRange[Int64, 2]
            mbar_kv: cute.struct.MemRange[Int64, self.kv_stage * 2]
            # raw: 0,1 S_full[slot] | 2,3 P_full[slot] | 4 O_full | 5 O_ready | 6,7 stats_full[slot] | 8 final_full
            mbar_dm: cute.struct.MemRange[Int64, 14]   # + 9 S_empty (sf_bank), 12/13 S_empty[slot] (C7b)
            tmem_holding_buf: Int32
            sStats: cute.struct.MemRange[Float32, 2 * 128 + 2 * 128 + 4 * 128 + 8 * 128]   # + mailbox state_m[2][128], state_l[2][128] (+ C8 part[2][4][128])
            sQ: cute.struct.Align[cute.struct.MemRange[self.q_dtype, cute.cosize(sQ_layout)], self.buffer_align_bytes]
            sK: cute.struct.Align[cute.struct.MemRange[self.q_dtype, sKV_size], self.buffer_align_bytes]
            sP: cute.struct.Align[cute.struct.MemRange[self.q_dtype, cute.cosize(sP_layout)], self.buffer_align_bytes]

        self.shared_storage = SharedStorage
        assert SharedStorage.size_in_bytes() <= 232448, f"SMEM {SharedStorage.size_in_bytes()}"
        n_m = cute.ceil_div(cute.size(mQt.shape[0]), self.m_block_size)
        grid = (n_m, cute.size(mQt.shape[2]), cute.size(mQt.shape[3]))
        kargs = [mQt, mKt, mVt, mO_raw, mLSE, mQS, mKVW, tma_atom_Q, tma_atom_K, tma_atom_V,
                 sQ_layout, sK_layout, sV_layout, sP_layout, tiled_mma_qk, tiled_mma_pv]
        self.kernel(*kargs).launch(
            grid=grid, block=[self.threads_per_cta, 1, 1], stream=stream, min_blocks_per_mp=1)

    @cute.kernel
    def kernel(self, mQ, mK, mV, mO, mLSE, mQS, mKVW, tma_atom_Q, tma_atom_K, tma_atom_V,
               sQ_layout: cute.ComposedLayout, sK_layout: cute.ComposedLayout, sV_layout: cute.ComposedLayout,
               sP_layout: cute.ComposedLayout, tiled_mma_qk: cute.TiledMma, tiled_mma_pv: cute.TiledMma):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            for a in (tma_atom_Q, tma_atom_K, tma_atom_V):
                cpasync.prefetch_descriptor(a)
        bidx, bidy, bidz = cute.arch.block_idx()
        n_m = cute.arch.grid_dim()[0]
        m_block = n_m - 1 - bidx if const_expr(self.is_causal) else bidx
        head_idx = bidy
        batch_idx = bidz
        kv_head_idx = head_idx // self.qhead_per_kvhead
        n_block_max = m_block + 1 if const_expr(self.is_causal) else n_m
        cnt = n_block_max

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
        if cute.arch.thread_idx()[0] == 0:
            mbar_init(mbar + 0, 1)
            mbar_init(mbar + 8, 1)
            mbar_init(mbar + 16, 256)     # P_full[0] (C8: both WGs)
            mbar_init(mbar + 24, 256)     # P_full[1]
            mbar_init(mbar + 32, 1)
            mbar_init(mbar + 40, 128)
            mbar_init(mbar + 48, 128)
            mbar_init(mbar + 56, 128)
            mbar_init(mbar + 64, 128)
            mbar_init(mbar + 72, 128)
            mbar_init(mbar + 80, 128)     # state_full[0] (two_wg)
            mbar_init(mbar + 88, 128)     # state_full[1]
            mbar_init(mbar + 96, 128)     # S_empty[0] (C7b)
            mbar_init(mbar + 104, 128)    # S_empty[1]
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
        acc_pv = thr_mma_pv.make_fragment_C(thr_mma_pv.partition_shape_C(self.mma_tiler_pv[:2]))
        pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        for w in cutlass.range_constexpr(len(self.idle_warp_ids)):
            if warp_idx == self.idle_warp_ids[w]:
                cute.arch.setmaxregister_decrease(self.num_regs_other)

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
            # Ring WAR edge of the F1a/F1c epilogue: the loads above
            # are this CTA's LAST writes to the K/V ring (non-persistent CTA, 2*cnt TMA loads, all consumed before O_full(PV(last)),
            # which the correction WG waits before staging O in the ring).

        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            tmem.allocate(512)
            tmem_alloc_barrier.arrive_and_wait_unaligned()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            self.mma(tiled_mma_qk, tiled_mma_pv, sQ, sK, sV, sP, acc_qk, acc_pv, pipeline_q, pipeline_kv, mbar, cnt)
            tmem.relinquish_alloc_permit()
            tcgen05_fence_before()
            tmem_alloc_barrier.arrive_and_wait_unaligned()
            tcgen05_fence_after()
            tmem.free(tmem_ptr)

        if warp_idx <= self.softmax_warp_ids[-1]:
            cute.arch.setmaxregister_increase(self.num_regs_softmax)
            tmem_alloc_barrier.arrive_and_wait_unaligned()
            self.softmax3(mbar, stats, sp_base, mLSE, mQS, mKVW, m_block, head_idx, kv_head_idx, batch_idx,
                          n_block_max, cnt)
            tcgen05_fence_before()
            tmem_alloc_barrier.arrive_unaligned()

        if warp_idx >= self.correction_warp_ids[0] and warp_idx <= self.correction_warp_ids[-1]:
            if const_expr(self.num_regs_correction > 128):
                cute.arch.setmaxregister_increase(self.num_regs_correction)
            else:
                cute.arch.setmaxregister_decrease(self.num_regs_correction)
            tmem_alloc_barrier.arrive_and_wait_unaligned()
            self.correction(mbar, stats, mO, m_block, head_idx, batch_idx, cnt, Int32(sK.iterator.toint()))
            tcgen05_fence_before()
            tmem_alloc_barrier.arrive_unaligned()
        return

    @cute.jit
    def mma(self, tiled_mma_qk, tiled_mma_pv, sQ, sK, sV, sP, acc_qk, acc_pv, pipeline_q, pipeline_kv, mbar, cnt):
        tSrQ = tiled_mma_qk.make_fragment_A(sQ)          # (MMA, 1, 8, 1)
        tSrK = tiled_mma_qk.make_fragment_B(sK)          # (MMA, 1, 8, kv_stage)
        tOrP = tiled_mma_pv.make_fragment_A(sP)          # (MMA, 1, 4, 2)
        tOrV = tiled_mma_pv.make_fragment_B(sV)          # (MMA, 1, 4, kv_stage)
        atom_qk = cute.make_mma_atom(tiled_mma_qk.op)
        pv_op = cute.make_mma_atom(tiled_mma_pv.op).op
        kv_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.kv_stage)
        q_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
        self.mma_c6(atom_qk, tSrQ, tSrK, tOrP, tOrV, sP, sV, acc_qk, acc_pv, pv_op, pipeline_q, pipeline_kv, mbar, cnt,
                    kv_state, q_state, sQ, sK)

    @cute.jit
    def mma_c6(self, atom_qk, tSrQ, tSrK, tOrP, tOrV, sP, sV, acc_qk, acc_pv, pv_op, pipeline_q, pipeline_kv, mbar, cnt,
               kv_state, q_state, sQ, sK):
        pipeline_q.consumer_wait(q_state)
        pipeline_kv.consumer_wait(kv_state)
        self.mma_qk(atom_qk, acc_qk, tSrQ, tSrK, kv_state.index, Int32(0))
        with cute.arch.elect_one():
            tc_commit(mbar + 0)
        pipeline_kv.consumer_release(kv_state)
        kv_state.advance()
        for i in cutlass.range(cnt - 1, unroll=1):
            j = i + 1
            slot = j & 1
            # slot j&1 holds SF(j-2), read by PV(j-2): PTX does not order PV(j-2) (mxf8f6f4, acc O) before QK(j) (f8f6f4,
            # acc S) by issue order -> wait PV(j-2)'s completion. Phase-safe: PV(j-3) is complete
            # (PV(j-2) waited O_ready(j-2), arrived after O_full(PV(j-3))) and PV(j-1) is not issued yet.
            if i >= 1:
                mbar_wait_parity(mbar + 32, (j - 2) & 1)
                tcgen05_fence_after()
            pipeline_kv.consumer_wait(kv_state)
            self.mma_qk(atom_qk, acc_qk, tSrQ, tSrK, kv_state.index, slot)
            with cute.arch.elect_one():
                tc_commit(mbar + 8 * slot)      # S_full
            pipeline_kv.consumer_release(kv_state)
            kv_state.advance()
            self.mma_pv(pv_op, acc_pv, tOrP, tOrV, sP, sV, pipeline_kv, kv_state, mbar, i)
            kv_state.advance()
        self.mma_pv(pv_op, acc_pv, tOrP, tOrV, sP, sV, pipeline_kv, kv_state, mbar, cnt - 1)
        kv_state.advance()
        pipeline_q.consumer_release(q_state)

    @cute.jit
    def mma_qk(self, atom_qk, acc_qk, tSrQ, tSrK, stage, slot):
        tS = cute.make_tensor(acc_qk.iterator + slot * 128, acc_qk.layout)
        for k in cutlass.range_constexpr(cute.size(tSrQ.shape[2])):
            atom_qk.set(tcgen05.Field.ACCUMULATE, k != 0)
            cute.gemm(atom_qk, tS, tSrQ[None, None, k, 0], tSrK[None, None, k, stage], tS)

    @cute.jit
    def mma_pv(self, pv_op, acc_pv, tOrP, tOrV, sP, sV, pipeline_kv, kv_state, mbar, i):
        slot = i & 1
        pipeline_kv.consumer_wait(kv_state)
        stage = kv_state.index
        mbar_wait_parity(mbar + 16 + 8 * slot, (i >> 1) & 1)     # P_full[slot]: P(i) codes + SF(i) written
        if i >= 1:
            mbar_wait_parity(mbar + 40, (i - 1) & 1)            # O_ready(i): O rescaled (or not) after PV(i-1)
        tcgen05_fence_after()
        sf = slot * 128
        gemm_ptx_bs(
            pv_op, self.tmem_o, tOrP[None, None, None, slot], tOrV[None, None, None, stage],
            sP[None, None, None, slot], sV[None, None, None, stage], sf + self.sfa_off, sf + self.sfb_off,
            tuple((0, k) for k in range(4)), tuple((0, k) for k in range(4)), zero_init=(i == 0))
        with cute.arch.elect_one():
            tc_commit(mbar + 32)                                   # O_full(i)
        pipeline_kv.consumer_release(kv_state)

    # -----------------------------------------------------------------------------------------------------------------
    @cute.jit
    def softmax3(self, mbar, stats, sp_base, mLSE, mQS, mKVW, m_block, head_idx, kv_head_idx, batch_idx, n_block_max, cnt):
        """C8: column split (see __init__ c8).  wg = warp // 4 owns columns 64 wg .. 64 wg + 63 of every tile."""
        tidx = cute.arch.thread_idx()[0] % 128
        wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) // 4
        row = m_block * self.m_block_size + tidx
        qsc = Float32(mQS[batch_idx, head_idx, row // 32]) * Float32(math.log2(math.e))
        lane = (cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4) * 32
        m_last = -Float32.inf
        l_last = Float32(0.0)
        if wg == 0:
            m_last, l_last = self.softmax_step3(Int32(0), n_block_max - 1, True, 0, mbar, stats, sp_base, tidx, lane, qsc, mKVW,
                                                kv_head_idx, batch_idx, m_last, l_last)
            for i in cutlass.range(cnt - 1, unroll=1):
                m_last, l_last = self.softmax_step3(i + 1, n_block_max - 2 - i, False, 0, mbar, stats, sp_base, tidx, lane, qsc,
                                                    mKVW, kv_head_idx, batch_idx, m_last, l_last)
            row_sum = l_last
            zero_or_nan = row_sum == Float32(0.0) or row_sum != row_sum
            scale = cute.arch.rcp_approx(row_sum if not zero_or_nan else Float32(1.0))
            LN2_ = math.log(2.0)
            lse = (m_last * Float32(1.0) + (cute.math.log2(row_sum, fastmath=True) - Float32(0.0))) * LN2_ \
                if not zero_or_nan else -Float32.inf
            mLSE[batch_idx, head_idx, row] = lse
            st_shared_f32(stats + 4 * (2 * 128 + tidx), scale)
            mbar_arrive(mbar + 64)
        else:
            self.softmax_step3(Int32(0), n_block_max - 1, True, 1, mbar, stats, sp_base, tidx, lane, qsc, mKVW, kv_head_idx,
                               batch_idx, m_last, l_last)
            for i in cutlass.range(cnt - 1, unroll=1):
                self.softmax_step3(i + 1, n_block_max - 2 - i, False, 1, mbar, stats, sp_base, tidx, lane, qsc, mKVW,
                                   kv_head_idx, batch_idx, m_last, l_last)

    @cute.jit
    def softmax_step3(self, j, n_block, is_first: cutlass.Constexpr, h: cutlass.Constexpr, mbar, stats, sp_base, tidx, lane,
                      qsc, mKVW, kv_head_idx, batch_idx, m_old, row_sum):
        """C8 tile step of WG h (h = 0: blocks 0, 1 + the ordered tail; h = 1: blocks 2, 3 + publish)."""
        slot = j & 1
        part = stats + 4 * (1024 + slot * 512)                      # part[slot][k][128]: bmax2, bmax3, bs2, bs3
        dm_kw = Int32(mKVW[batch_idx, kv_head_idx, 2 * n_block])
        dm_vw = Int32(mKVW[batch_idx, kv_head_idx, 2 * n_block + 1])
        dm_ks = [byte_pow2_f32(dm_kw, b) for b in range(4)]
        mbar_wait_parity(mbar + 8 * slot, (j >> 1) & 1)          # S_full[slot]
        tcgen05_fence_after()
        sfbase = (lane << 16) + slot * 128
        c0 = 64 * h
        s0 = tmem_ld32(sfbase + c0)
        s1 = tmem_ld32(sfbase + c0 + 32)
        tmem_wait_ld()
        s = s0 + s1                                                          # s[c] = S(row, 64 h + c), c = 0..63
        if const_expr(self.is_causal and is_first):
            for c in cutlass.range_constexpr(64):
                s[c] = -Float32.inf if Int32(c + c0) > tidx else s[c]
        # ---- m-independent prefix of blocks 2h, 2h+1 (hd128 sm100-mx line, same expressions) ----
        dm_c = [qsc * dm_ks[2 * h + bb] for bb in range(2)]
        dm_bmax = [None] * 2
        for bb in cutlass.range_constexpr(2):
            b0 = 32 * bb
            lm = [utils.fmax(s[b0 + 3 * t], s[b0 + 3 * t + 1], s[b0 + 3 * t + 2]) for t in range(4)]
            for i in cutlass.range_constexpr(12, 32, 2):
                t = ((i - 12) // 2) % 4
                lm[t] = utils.fmax(lm[t], s[b0 + i], s[b0 + i + 1])
            v = utils.fmax(utils.fmax(lm[0], lm[1], lm[2]), lm[3])
            dm_bmax[bb] = v * dm_c[bb]
        dm_kb = [None] * 2
        dm_bias = [None] * 2
        for bb in cutlass.range_constexpr(2):
            dm_kb[bb] = cute.arch.fmax(floor_f32(dm_bmax[bb]), Float32(-1e30))
            dm_bias[bb] = Float32(7.0) - dm_kb[bb]
        for bb in cutlass.range_constexpr(2):
            for i in cutlass.range_constexpr(0, 32, 2):
                s[32 * bb + i], s[32 * bb + i + 1] = cute.arch.fma_packed_f32x2(
                    (s[32 * bb + i], s[32 * bb + i + 1]), (dm_c[bb], dm_c[bb]), (dm_bias[bb], dm_bias[bb]))
        prow = sp_base + slot * 16384 + tidx * 128
        for bb in cutlass.range_constexpr(2):
            for i in cutlass.range_constexpr(32):
                s[32 * bb + i] = ex2_ftz(s[32 * bb + i])
        for bb in cutlass.range_constexpr(2):
            for q in cutlass.range_constexpr(2):
                k = 32 * bb + 16 * q
                w = [e4m3x4(s[k + 4 * t], s[k + 4 * t + 1], s[k + 4 * t + 2], s[k + 4 * t + 3]) for t in range(4)]
                chunk = Int32(2 * (2 * h + bb) + q)
                st_shared_v4_b32(prow + ((chunk ^ (tidx & 7)) << 4), w[0], w[1], w[2], w[3])
        dm_bs = [None] * 2
        for bb in cutlass.range_constexpr(2):
            ps = (s[32 * bb], s[32 * bb + 1])
            for i in cutlass.range_constexpr(2, 32, 2):
                ps = cute.arch.add_packed_f32x2(ps, (s[32 * bb + i], s[32 * bb + i + 1]))
            dm_bs[bb] = ps[0] + ps[1]
        row_max = m_old
        dm_l = row_sum
        if const_expr(h == 1):
            # ---- publish the high half: part(j) reuse at j+2 is ordered by WG0's read(j) -> P_full(j) -> PV(j) -> (C0)
            # QK(j+2) -> S_full(j+2), which this WG waits before writing part(j+2) ----
            st_shared_f32(part + 4 * (0 * 128 + tidx), dm_bmax[0])
            st_shared_f32(part + 4 * (1 * 128 + tidx), dm_bmax[1])
            st_shared_f32(part + 4 * (2 * 128 + tidx), dm_bs[0])
            st_shared_f32(part + 4 * (3 * 128 + tidx), dm_bs[1])
            tcgen05_fence_before()                                   # our S loads (wait::ld) before WG0's SFB stores
            mbar_arrive(mbar + 80 + 8 * slot)                   # partial_full[slot]
            cute.arch.fence_view_async_shared()
            tcgen05_fence_before()
            mbar_arrive(mbar + 16 + 8 * slot)                   # P_full[slot] (our P codes)
        else:
            # ---- ordered tail (WG0): exactly the C6 tail over the 4 blocks ----
            mbar_wait_parity(mbar + 80 + 8 * slot, (j >> 1) & 1)   # partial_full[slot]
            tcgen05_fence_after()
            bmax2 = ld_shared_f32(part + 4 * (0 * 128 + tidx))
            bmax3 = ld_shared_f32(part + 4 * (1 * 128 + tidx))
            bs2 = ld_shared_f32(part + 4 * (2 * 128 + tidx))
            bs3 = ld_shared_f32(part + 4 * (3 * 128 + tidx))
            bmax = [dm_bmax[0], dm_bmax[1], bmax2, bmax3]
            bsum = [dm_bs[0], dm_bs[1], bs2, bs3]
            kb = [dm_kb[0], dm_kb[1], cute.arch.fmax(floor_f32(bmax2), Float32(-1e30)),
                  cute.arch.fmax(floor_f32(bmax3), Float32(-1e30))]
            dm_xmax = utils.fmax(utils.fmax(bmax[0], bmax[1], bmax[2]), bmax[3])
            dm_mcand = ceil_f32(dm_xmax)
            acc_scale = Float32(1.0)
            row_max = dm_mcand
            if const_expr(is_first):
                acc_scale = Float32(0.0)
            else:
                row_max = m_old
                if dm_mcand > m_old + Float32(self.rescale_T):
                    row_max = dm_mcand
                    acc_scale = exp2i_f32(m_old - dm_mcand)
            st_shared_f32(stats + 4 * (slot * 128 + tidx), acc_scale)
            mbar_arrive(mbar + 48 + 8 * slot)                   # stats_full[slot] (G1a)
            m_safe = row_max if row_max != -Float32.inf else Float32(0.0)
            dm_w = [None] * 4
            dm_sfa = [None] * 4
            for b in cutlass.range_constexpr(4):
                e = kb[b] - Float32(7.0) - m_safe
                dm_sfa[b] = ue8m0_byte(e)
                dm_w[b] = byte_shl23_f32(dm_sfa[b])
            sfa_word = pack_bytes4(dm_sfa[0], dm_sfa[1], dm_sfa[2], dm_sfa[3])
            for r in cutlass.range_constexpr(4):
                tmem_st1(sfbase + self.sfa_off + r, sfa_word)
            for r in cutlass.range_constexpr(8):
                tmem_st1(sfbase + self.sfb_off + r, dm_vw)
            cute.arch.fence_view_async_shared()
            tmem_wait_st()
            tcgen05_fence_before()
            mbar_arrive(mbar + 16 + 8 * slot)                       # P_full[slot]
            dm_l = Float32(0.0) if const_expr(is_first) else row_sum * acc_scale
            for b in cutlass.range_constexpr(4):
                dm_l = dm_l + bsum[b] * dm_w[b]
        return row_max, dm_l

    # -----------------------------------------------------------------------------------------------------------------
    @cute.jit
    def correction(self, mbar, stats, mO, m_block, head_idx, batch_idx, cnt, sk_base):
        tidx = cute.arch.thread_idx()[0] % 128
        lane = (cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4) * 32
        obase = (lane << 16) + self.tmem_o
        for i in cutlass.range(cnt - 1, unroll=1):
            j = i + 1
            slot = j & 1
            mbar_wait_parity(mbar + 48 + 8 * slot, (j >> 1) & 1)          # stats(j)
            acc_scale = ld_shared_f32(stats + 4 * (slot * 128 + tidx))
            mbar_wait_parity(mbar + 32, (j - 1) & 1)                       # O_full(PV(j-1))
            tcgen05_fence_after()
            if cute.arch.vote_ballot_sync(acc_scale < Float32(1.0)) != 0:      # hd128 predicate
                for c in cutlass.range_constexpr(8):
                    o = list(tmem_ld32(obase + 32 * c))
                    tmem_wait_ld()
                    for k in cutlass.range_constexpr(0, 32, 2):
                        o[k], o[k + 1] = cute.arch.mul_packed_f32x2((o[k], o[k + 1]), (acc_scale, acc_scale))
                    tmem_st32(obase + 32 * c, o)
                tmem_wait_st()
            tcgen05_fence_before()
            mbar_arrive(mbar + 40)                                      # O_ready(j)
        mbar_wait_parity(mbar + 64, Int32(0))
        scale = ld_shared_f32(stats + 4 * (2 * 128 + tidx))
        mbar_wait_parity(mbar + 32, (cnt - 1) & 1)                         # O_full(PV(last))
        tcgen05_fence_after()
        N = mO.shape[1]
        H = mO.shape[2]
        # F1a: O_full(last) = the last PV (last reader of the sK/sV ring; every TMA load into it was consumed) completed.
        # The ring was last written by TMA (async proxy): order our generic writes after it.
        fence_proxy_async_shared()
        RS = 528                                                     # 512 B row + 16 B pad: conflict-free both ways
        srow = sk_base + Int32(RS) * tidx
        ln = tidx % 32
        w0 = tidx - ln                                               # first row of this warp (TMEM lane quadrant)
        # F1c: TMEM chunk c+1 is loaded while chunk c is converted (wait::ld covers all prior loads, so the next
        # load is issued only after the previous wait)
        o = tmem_ld32(obase)
        tmem_wait_ld()
        for c in cutlass.range_constexpr(8):
            if const_expr(c < 7):
                on = tmem_ld32(obase + 32 * (c + 1))
            for q in cutlass.range_constexpr(4):
                wv = [bf16x2(mul_rn(o[8 * q + 2 * t], scale), mul_rn(o[8 * q + 2 * t + 1], scale)) for t in range(4)]
                st_shared_v4_b32(srow + (64 * c + 16 * q), wv[0], wv[1], wv[2], wv[3])
            if const_expr(c < 7):
                tmem_wait_ld()
                o = on
        bar_warp_sync()                                              # warp-local hand-off: a warp copies only its own rows
        for r in cutlass.range(32, unroll=4):
            rr = w0 + r
            v = ld_shared_v4_b32(sk_base + Int32(RS) * rr + 16 * ln)
            gaddr = Int64(mO.iterator.toint()) + (Int64((batch_idx * N + m_block * self.m_block_size + rr) * H + head_idx)
                                                  * Int64(self.head_dim * 2)) + Int64(16 * ln)
            st_global_v4_b32(gaddr, v[0], v[1], v[2], v[3])
