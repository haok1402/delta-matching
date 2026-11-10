# Supported features:
# - BF16 & FP16 dtype
# - noncausal & causal attention
# - MHA, GQA, MQA
# - hdim 64, 96, 128, (192, 128).
# - varlen
# - sliding window
# - split-kv
# Unsupported features that will be added later:
# - page size != 128
# - more hdim (192, 256)
# Based on the cutlass example and cute-dsl example:
# https://github.com/NVIDIA/cutlass/tree/main/examples/77_blackwell_fmha
# https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell/fmha.py

import math
from typing import Tuple, Callable, Optional, Literal
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Boolean, const_expr
from cutlass.cute.nvgpu import cpasync
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils_basic
from cutlass import pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils import ClcDynamicPersistentTileScheduler
from cutlass.base_dsl.arch import Arch
from cutlass.cutlass_dsl import BaseDSL, T, dsl_user_op
from cutlass._mlir.dialects import llvm

from quack import copy_utils, layout_utils

from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned
from flash_attn.cute import utils
from delta_matching.sm100.cute_ptx_sm100 import bitcast_f32_i32, bitcast_i32_f32, ceil_f32, floor_f32, ld_shared_b32, mbar_init, st_shared_b32, tc_commit
from delta_matching.sm100.fp8_e8m0_scale_sm100 import byte_pow2_f32, byte_shl23_f32, exp2i_f32, pack_bytes4, ue8m0_byte
from delta_matching.sm100.tcgen05_fence_sm100 import FENCE_MODE, begin_kernel_fences, end_kernel_fences, fence_warp_relay, mark_fenced, mbar_arrive, mbar_wait_parity, tcgen05_fence_after, tcgen05_fence_before
import flash_attn.cute.pipeline as pipeline_custom
import cutlass.pipeline as cutlass_pipeline
from flash_attn.cute.mask import AttentionMask, mask_r2p_lambda, r2p_bitmask_below
from flash_attn.cute.softmax import (
    SoftmaxSm100,
)
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.block_info import BlockInfo
from flash_attn.cute.block_sparsity import BlockSparseTensors
from flash_attn.cute import mma_sm100_desc as sm100_desc
from flash_attn.cute import blackwell_helpers as sm100_utils
from flash_attn.cute.named_barrier import NamedBarrierFwdSm100
from quack.cute_dsl_utils import ParamsBase
from flash_attn.cute.tile_scheduler import (
    SchedulerState,
    SchedulingMode,
    TileSchedulerArguments,
    TileSchedulerProtocol,
    SingleTileScheduler,
    StaticPersistentTileScheduler,
    SingleTileLPTScheduler,
)
from flash_attn.cute.utils import AuxData
from flash_attn.cute.config import (
    SM100_LAUNCH_REGISTERS,
    SM100_MAX_KV_STAGE,
    SM100_THREADS_PER_CTA,
    FwdConfig,
    sm100_smem_capacity,
)

# Note [Low Precision Scaling]
# P is in (0, 1] and is cast to the input dtype before P @ V, so scaling it by 2^max_offset
# spends the dtype's unused upper code points on the probability tail. A positive
# rescale_threshold lets the row max lag by that many log2 units, so P can reach
# 2^(max_offset + rescale_threshold); above the dtype max the top probabilities saturate
# while the FP32 denominator still counts them in full, shrinking the output (#2716).

def block_scaled_idesc(idesc_plain: int, k: int) -> int:
    """Block-scaled (kind::mxf8f6f4, E8M0 scales) instruction descriptor from the plain f8f6f4 one:
    bits [4,6) c_format -> b_sf_id, bit 23 -> scale_format E8M0, bits [29,31) -> a_sf_id (= k-step byte)."""
    d = idesc_plain & ~(0x3 << 4) & ~(0x7 << 29)
    return d | (1 << 23) | ((k & 3) << 29) | ((k & 3) << 4)


def gemm_ptx_bs_ts(
    op, acc_tmem_addr, tCrA, tCrB, sB, sfa_tmem_addr, sfb_tmem_addr,
    mbar_ptr=None, mbar_phase=None, split_arrive=None, zero_init=False, tA_addr=None, cta_group: int = 1, offs_a=None,
):
    """TS block-scaled MMA: A (e4m3) from TMEM, B (e4m3) from smem, SFA/SFB (UE8M0) in TMEM.
    One 32-bit SF word per row holds the 4 k-step scales; k-step k selects byte k via sf_id (idesc).
    Mirrors flash_attn.cute.blackwell_helpers.gemm_ptx_partial (TS branch)."""
    from flash_attn.cute import mma_sm100_desc as sm100_desc
    from flash_attn.cute.blackwell_helpers import i64_to_i32x2
    idesc_plain: int = sm100_desc.mma_op_to_idesc(op)
    nk = cute.size(tCrA.shape[2])
    assert nk <= 4, "one SF word covers 4 k-steps"
    sB_layout = sB.layout
    sB_swizzle = sB.iterator.type.swizzle_type
    smem_desc_base_b: int = sm100_desc.make_smem_desc_base(
        cute.recast_layout(128, op.b_dtype.width, sB_layout[0]), sB_swizzle,
        sm100_desc.Major.K if op.b_major_mode == cute.nvgpu.tcgen05.mma.OperandMajorMode.K else sm100_desc.Major.MN)
    smem_desc_base_b_lo, smem_desc_b_hi = i64_to_i32x2(smem_desc_base_b)
    tCrA_layout = cute.recast_layout(32, tCrA.element_type.width, tCrA.layout)
    offset_a = [cute.crd2idx((0, 0, k), tCrA_layout) for k in range(nk)] if offs_a is None else list(offs_a)
    offset_b = [cute.crd2idx((0, 0, k), tCrB.layout) for k in range(nk)]
    smem_desc_start_b_lo = Int32(smem_desc_base_b_lo | sm100_desc.make_smem_desc_start_addr(sB[None, None, 0].iterator))
    pred_str = "p" if isinstance(zero_init, cutlass.Boolean) else "0" if zero_init else "1"
    tA_addr = tCrA[None, None, 0].iterator.toint() if tA_addr is None else tA_addr
    input_args = [
        Int32(cute.arch.make_warp_uniform(tA_addr)).ir_value(),
        Int32(cute.arch.make_warp_uniform(smem_desc_start_b_lo)).ir_value(),
        (Int32(1) - Int32(zero_init)).ir_value() if isinstance(zero_init, cutlass.Boolean) else Int32(not zero_init).ir_value(),
        Int32(cute.arch.make_warp_uniform(acc_tmem_addr)).ir_value(),
        Int32(cute.arch.make_warp_uniform(sfa_tmem_addr)).ir_value(),
        Int32(cute.arch.make_warp_uniform(sfb_tmem_addr)).ir_value(),
    ]
    split_idx = nk
    wait_str = ""
    if mbar_ptr is not None:
        split_idx = split_arrive // op.shape_mnk[2]
        input_args.append(mbar_ptr.toint().ir_value())
        input_args.append(Int32(mbar_phase).ir_value())
        wait_str = (".reg .pred P1; \n\tLAB_WAIT: \n\t"
                    "mbarrier.try_wait.parity.shared::cta.b64 P1, [$6], $7, 10000000; \n\t"
                    "@P1 bra DONE; \n\tbra     LAB_WAIT; \n\tDONE: \n\t"
                    + ("tcgen05.fence::after_thread_sync; \n\t" if FENCE_MODE[0] >= 1 else ""))
    instr = f"tcgen05.mma.cta_group::{cta_group}.kind::mxf8f6f4.block_scale.scale_vec::1X"

    def one(k, pred):
        return (f"add.u32 smem_desc_b_lo, smem_desc_b_lo_start, {hex(offset_b[k])};\n\t"
                f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                f"@leader_thread {instr} [tmem_acc], [tmem_a + {hex(offset_a[k])}], smem_desc_b, "
                f"{hex(block_scaled_idesc(idesc_plain, k))}, [tmem_sfa], [tmem_sfb], {pred};\n\t")

    body = ("{\n\t.reg .pred leader_thread;\n\t.reg .pred p;\n\t.reg .b32 tmem_acc, tmem_a, tmem_sfa, tmem_sfb;\n\t"
            ".reg .b32 smem_desc_b_lo_start, smem_desc_b_lo, smem_desc_b_hi;\n\t.reg .b64 smem_desc_b;\n\t"
            "elect.sync _|leader_thread, -1;\n\t"
            "mov.b32 tmem_acc, $3;\n\tmov.b32 tmem_a, $0;\n\tmov.b32 tmem_sfa, $4;\n\tmov.b32 tmem_sfb, $5;\n\t"
            "mov.b32 smem_desc_b_lo_start, $1;\n\t"
            f"mov.b32 smem_desc_b_hi, {hex(smem_desc_b_hi)};\n\t"
            "setp.ne.b32 p, $2, 0;\n\t"
            + one(0, pred_str)
            + "".join(one(k, "1") for k in range(1, split_idx))
            + wait_str
            + "".join(one(k, "1") for k in range(split_idx, nk))
            + "}\n")
    llvm.inline_asm(None, input_args, body, ",".join(["r"] * len(input_args)),
                    has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def byte_pow2_f32_rt(word: Int32, pos: Int32, *, loc=None, ip=None) -> Float32:
    """byte_pow2_f32 with a runtime bit position pos = 8 j (same bfe/shl -> same value)."""
    return Float32(llvm.inline_asm(
        T.f32(), [Int32(word).ir_value(loc=loc, ip=ip), Int32(pos).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b32 t;\n\tbfe.u32 t, $1, $2, 8;\n\tshl.b32 $0, t, 23;\n\t}",
        "=f,r,r", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


# NOTE [SM100 forward coordinate spaces]
# R = m_block_size, C = cta_group_size, S = q_stage; a work tile spans R*C*S effective rows.
#
#   physical CTA / CLC response
#        |  scheduler (NOTE [SM100 forward work coordinates], tile_scheduler.py)
#   (work_m, head, batch, split)        same on every CTA of a cluster
#        |  work_q_block(work_m, stage, cta_rank)
#   q_block                             R-row block index
#        |  * R + row_in_cta
#   q_row                               batch-local effective row, packed under pack_gqa
#        |  layouts, seqlen_info offsets
#   memory
#
# Row strides for (row_in_cta, cta_rank, stage, work_m) are (1, R, R*C, R*C*S).
# R=128, C=S=2, work_m=0:
#              CTA 0       CTA 1
#   stage 0    0..127      128..255
#   stage 1  256..383      384..511
#
# Users: LSE and non-TMA Q/O index by the CTA's q_block. TMA Q/O tile work_m and partition by
# stage and CTA. Mask and score_mod pass cta_rank=0 because their thr_mma partitions already
# include the CTA's rows; the block-sparse stage end is (work_q_block(work_m, stage, 0) + C) * R.
# q_row drives the Q/O/LSE tail predicates and the packed-GQA sink head.


def work_q_block(work_m, stage, cta_rank, *, q_stage: int, cta_group_size: int):
    """R-row block index for one CTA and Q stage of a work tile.

    See NOTE [SM100 forward coordinate spaces].
    """
    return (work_m * q_stage + stage) * cta_group_size + cta_rank


from flash_attn.cute.flash_fwd_sm100 import DescaleTensors  # same class as interface.py builds


class DMForwardSm100:
    """FA4 sm100 forward forked for the Delta-Matching sm100-mx numerics.
    descale_tensors carries (qs, ks, vs): power-of-2 fp32 scales per 32 tokens, (B,H,N/32), (B,Hk,N/32), (B,Hk,N/32).
    P is MXFP8: e4m3 codes per (row, 32-key block) in [128,256) at the block max, UE8M0 SFA = 2^(kb-7-m),
    SFB = V block scale; PV runs as tcgen05.mma kind::mxf8f6f4.block_scale with P and both SFs in TMEM."""


    def __init__(
        self,
        head_dim: int,
        head_dim_v: Optional[int] = None,
        qhead_per_kvhead: cutlass.Constexpr[int] = 1,
        is_causal: bool = False,
        is_local: bool = False,
        q_subtile_factor: int = 1,
        kv_subtile_factor: int = 1,
        score_mod: cutlass.Constexpr | None = None,
        mask_mod: cutlass.Constexpr | None = None,
        has_aux_tensors: cutlass.Constexpr = False,
        paged_kv_non_tma: bool = False,
        is_varlen_q: bool = False,
        has_tile_count_semaphore: bool = False,
        seqlen_k_per_split: Optional[int] = None,
        *,
        config: FwdConfig,
    ):
        """config is the FwdConfig fwd.py selects (1-CTA, no split, no pack-GQA, no S ping-pong, all-MUFU exp)."""
        # The launcher (fwd.py) only builds dense causal / non-causal hd128 attention: every FA4 feature switch below is
        # fixed, and the kernel keeps only that path.
        assert head_dim == 128 and head_dim_v in (None, 128), "head_dim 128 only"
        assert not (is_local or score_mod is not None or mask_mod is not None or has_aux_tensors or paged_kv_non_tma
                    or is_varlen_q or has_tile_count_semaphore or seqlen_k_per_split is not None), "dense attention only"
        assert q_subtile_factor == 1 and kv_subtile_factor == 1
        assert not (config.use_2cta_instrs or config.pack_gqa or config.use_s_ping_pong or config.num_splits > 1
                    or config.ex2_emu_freq != 0 or config.overlap_sO_sQ or not config.use_tma_o), "unsupported config"
        self.head_dim_padded = 128
        self.head_dim_v_padded = 128
        self.m_block_size = config.tile_m
        self.n_block_size = config.tile_n
        assert self.m_block_size == 128 and self.n_block_size == 128
        self.q_stage = config.q_stage
        assert self.q_stage in [1, 2]
        # If split_P_arrive, the softmax warps write some columns of P first, signal to the MMA warp
        # to being the P @ V MMA, then write the rest of P and signal again. This allows some overlap
        # between compute the last couple columns of P and the P @ V MMA.
        self.split_P_arrive = self.n_block_size // 4 * 3
        self.split_P_arrive = int(self.split_P_arrive / 32) * 32  # multiple of 32
        self.arch = BaseDSL._get_dsl().get_arch_enum()
        assert self.arch.is_family_of(Arch.sm_100f), "Only SM 10.x is supported"
        self.cta_tiler = (self.q_stage * self.m_block_size, self.n_block_size, self.head_dim_padded)
        self.mma_tiler_qk = (self.m_block_size, self.n_block_size, self.head_dim_padded)
        self.mma_tiler_pv = (self.m_block_size, self.head_dim_v_padded, self.n_block_size)
        self.qk_acc_dtype = Float32
        self.pv_acc_dtype = Float32
        self.cluster_shape_mn = (1, 1)
        self.is_static_persistent = config.is_static_persistent
        self.is_causal = is_causal
        self.qhead_per_kvhead = qhead_per_kvhead
        self.use_clc_scheduler = config.use_clc_scheduler
        self.dynamic_persistent = self.use_clc_scheduler
        self.is_persistent = self.dynamic_persistent or self.is_static_persistent
        self.kv_stage = config.kv_stage
        assert 2 <= self.kv_stage <= SM100_MAX_KV_STAGE, f"kv_stage={self.kv_stage}"
        self.s_stage = 2
        self.sched_stages = 1
        self.scheduling_mode = SchedulingMode.CLC if self.use_clc_scheduler else SchedulingMode.STATIC
        if self.is_causal or self.use_clc_scheduler:
            self.TileScheduler = SingleTileLPTScheduler
        elif self.is_static_persistent:
            self.TileScheduler = StaticPersistentTileScheduler
        else:
            self.TileScheduler = SingleTileScheduler

        self.softmax0_warp_ids = (0, 1, 2, 3)
        self.softmax1_warp_ids = (4, 5, 6, 7)
        self.correction_warp_ids = (8, 9, 10, 11)
        self.mma_warp_id = 12
        self.epilogue_warp_ids = (13,)
        self.load_warp_ids = (14,)
        self.empty_warp_ids = (15,)
        self.tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")
        self.threads_per_cta = SM100_THREADS_PER_CTA
        # S/P/O pipeline depth: one TMEM slot per Q stage
        self.s_p_o_stage = self.q_stage
        if self.q_stage == 1:
            self.empty_warp_ids = self.empty_warp_ids + self.softmax1_warp_ids
            self.softmax1_warp_ids = ()
        self.scheduler_warp_id = self.empty_warp_ids[0] if self.dynamic_persistent else None
        # NOTE [global LPT] (config.py); validate_fwd_config admits it only on this scheduler path.
        self.lpt_global = config.lpt_global
        assert not self.lpt_global or (self.use_clc_scheduler and self.TileScheduler is SingleTileLPTScheduler)
        # Claim after the current tile's loads to avoid reserving work during long causal tiles.
        self.clc_late_claim = self.use_clc_scheduler and self.TileScheduler is SingleTileLPTScheduler

        self.tmem_s_offset = [0, self.n_block_size]  # e.g., 0, 128
        self.tmem_o_offset = [
            self.tmem_s_offset[-1] + self.n_block_size + i * self.head_dim_v_padded
            for i in range(self.q_stage)
        ]  # e.g., 256, 384
        self.tmem_total = self.tmem_o_offset[-1] + self.head_dim_v_padded
        assert self.tmem_total <= self.tmem_alloc_cols
        self.tmem_s_to_p_offset = self.n_block_size // 2
        self.tmem_p_offset = [
            self.tmem_s_offset[i] + self.tmem_s_to_p_offset for i in range(2)
        ]  # 0, 128

        # DM sm100-mx: SFA (4 cols) + SFB (4 cols) right after the fp8 P columns of each S stage
        self.dm_sf_col = self.tmem_s_to_p_offset + self.n_block_size // 4
        # vec buffer for row_max & row_sum
        self.tmem_vec_offset = self.tmem_s_offset

        # Slice softmax (q_stage 2): MUFU-first softmax. 4 slice warpgroups (warps 0-15); slice WG q owns
        # key block q (32 keys = one MX block) of the CURRENT stage; all slices process stage 0 then stage 1 of each
        # n_block, so every exp warp feeds one stage while the other stage's PV -> QK turnaround runs. Row-level work
        # (m, acc_scale, SFA bytes, weights, l, O rescale, epilogue) moves to the correction WG (warps 16-19): slices
        # publish x_b = max(S_b) c_b and the fp32 block sum through SMEM. P codes of block q go to S_s + 32 q (the
        # block's own S columns), SF to S_s + 16 (block 0's columns, written after every slice loaded S).
        # q_stage 1 (one 128-row tile) keeps FA4's row softmax.
        self.dm_slice = self.q_stage == 2
        if self.dm_slice:
            # K/V smem ring depth cap (budget of the slice layout's smem buffers)
            self.kv_stage = min(self.kv_stage, 6)
            self.softmax0_warp_ids = tuple(range(16))
            self.softmax1_warp_ids = ()
            self.correction_warp_ids = (16, 17, 18, 19)
            self.mma_warp_id = 20
            self.epilogue_warp_ids = (21,)
            self.empty_warp_ids = (23,)
            self.load_warp_ids = (22,)
            self.scheduler_warp_id = self.empty_warp_ids[0] if self.dynamic_persistent else None
            self.threads_per_cta = 24 * cute.arch.WARP_SIZE
            self.split_P_arrive = 0
            self.dm_sl_sfcol = 16
            # setmaxnreg split (per thread). Launch = 65536 / 768 -> 80 per thread; the CTA register pool only holds what
            # setmaxnreg.dec released, so 4 (rs - 80) <= (80 - rc) + (80 - ro) or the slice WGs' alloc spins forever.
            self.dm_sl_regs = (80, 112, 48)

        # PV_s reads its SFA/SFB scale factors (block-scaled tcgen05.mma) from S_s's TMEM columns, and the next QK_s
        # (a different accumulator/shape/kind) overwrites S_s. PTX ISA 9.7.18.6.2 pipelines MMA A/metadata reads ->
        # later MMA D writes, but not scale-factor reads, so issue order alone is not a guarantee: tcgen05.commit
        # after every PV_s -> mbarrier ord[s]; the MMA warp waits it (+ fence::after_thread_sync) before the next QK_s.
        self.num_regs_softmax = config.registers.softmax
        self.num_regs_correction = config.registers.correction
        self.num_regs_other = config.registers.other
        if self.dm_slice:
            self.num_regs_softmax, self.num_regs_correction, self.num_regs_other = self.dm_sl_regs

        self.buffer_align_bytes = 1024

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
        mK: cute.Tensor,  # (b_k, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k or (num_pages, page_size, h_k, d) if there is page_table
        mV: cute.Tensor,  # (b_k, s_k, h_k, dv) or (total_k, h_k, dv) if there is cu_seqlens_k or (num_pages, page_size, h_k, dv) if there is page_table
        mO: cute.Tensor,  # (b, s_q, h, dv) or (total_q, h, dv) if there is cu_seqlens_q
        mLSE: Optional[cute.Tensor],
        softmax_scale: Float32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        mPageTable: Optional[cute.Tensor] = None,  # (b_k, max_num_pages_per_seq)
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        learnable_sink: Optional[cute.Tensor] = None,
        descale_tensors: Optional[DescaleTensors] = None,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        aux_data: AuxData = AuxData(),
        num_splits_dynamic_ptr: Optional[cute.Tensor] = None,
        tile_count_semaphore: Optional[cute.Tensor] = None,
        virtual_batch_idx_ptr: Optional[cute.Tensor] = None,
        num_nheads_in_l2_ptr: Optional[cute.Tensor] = None,
        mCuTotalMBlocks: Optional[cute.Tensor] = None,
        mCuTotalSplitsMBlocks: Optional[cute.Tensor] = None,
        mBlocksToBatchIdx: Optional[cute.Tensor] = None,
        max_seqlen_q: Int32 | int | None = None,
        tensormap_workspace: Optional[cute.Tensor] = None,  # (num_ctas, 16) Int64, one TMA descriptor per CTA
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        """Execute the Fused Multi-Head Attention operation on the provided tensors.

        This method prepares the input tensors for processing, validates their shapes and types,
        configures the computation parameters, and launches the CUDA kernel.

        The method handles:
        1. Tensor layout transformations for specific memory access patterns
        2. Validation of tensor shapes and data types
        3. Initialization of hardware-specific parameters and memory layouts
        4. Configuration of TMA (Tensor Memory Access) operations
        5. Grid and work scheduling computation
        6. Kernel launch with appropriate parameters
        """
        # setup static attributes before smem/grid/tma computation
        self.q_dtype = mQ.element_type
        self.k_dtype = mK.element_type
        self.v_dtype = mV.element_type
        self.o_dtype = mO.element_type
        assert mCuSeqlensQ is None and mCuSeqlensK is None and mSeqUsedQ is None and mSeqUsedK is None and mPageTable is None
        assert learnable_sink is None and blocksparse_tensors is None and mLSE is not None
        mQ, mK, mV, mO = [assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)]
        mQ = cute.make_tensor(mQ.iterator, cute.select(mQ.layout, mode=[1, 3, 2, 0]))
        # (s_k, d, h_k, b_k)
        mK, mV = [cute.make_tensor(t.iterator, cute.select(t.layout, mode=[1, 3, 2, 0])) for t in (mK, mV)]
        num_splits = Int32(1)
        mO = cute.make_tensor(mO.iterator, cute.select(mO.layout, mode=[1, 3, 2, 0]))
        mLSE = cute.make_tensor(mLSE.iterator, cute.select(mLSE.layout, mode=[2, 1, 0]))
        # (s, d, h, b) -> (d, s, h, b)
        mV = cute.make_tensor(mV.iterator, cute.select(mV.layout, mode=[1, 0, 2, 3]))

        # check type consistency
        if const_expr(self.q_dtype != self.k_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.k_dtype}")
        if const_expr(self.q_dtype != self.v_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.v_dtype}")

        cta_group = tcgen05.CtaGroup.ONE
        q_major_mode = cute.nvgpu.OperandMajorMode.K
        k_major_mode = cute.nvgpu.OperandMajorMode.K
        v_major_mode = cute.nvgpu.OperandMajorMode.MN
        self.o_layout = cutlass.utils.LayoutEnum.from_tensor(mO)
        # the intermediate tensor p is from tmem & mK-major
        p_source = tcgen05.OperandSource.TMEM
        p_major_mode = cute.nvgpu.OperandMajorMode.K
        tiled_mma_qk = sm100_utils_basic.make_trivial_tiled_mma(
            self.q_dtype,
            self.q_dtype,
            q_major_mode,
            k_major_mode,
            self.qk_acc_dtype,
            cta_group,
            self.mma_tiler_qk[:2],
        )
        tiled_mma_pv = sm100_utils_basic.make_trivial_tiled_mma(
            self.v_dtype,
            self.v_dtype,
            p_major_mode,
            v_major_mode,
            self.pv_acc_dtype,
            cta_group,
            self.mma_tiler_pv[:2],
            p_source,
        )

        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (tiled_mma_qk.thr_id.shape,)
        )

        # epi_tile is per-CTA (not full 2CTA) since each CTA writes its own O portion
        self.epi_tile = (self.m_block_size, self.head_dim_v_padded)

        sQ_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_qk, self.mma_tiler_qk, self.q_dtype, self.q_stage
        )
        sK_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_qk, self.mma_tiler_qk, self.k_dtype, self.kv_stage
        )
        tP_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_pv, self.mma_tiler_pv, self.q_dtype, self.s_stage
        )
        sV_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_pv, self.mma_tiler_pv, self.v_dtype, self.kv_stage
        )
        sO_layout = sm100_utils_basic.make_smem_layout_epi(
            self.o_dtype, self.o_layout, self.epi_tile, self.q_stage
        )

        self.tma_copy_bytes = {
            name: cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1, 2]))
            for name, mX, layout in [
                ("Q", mQ, sQ_layout),
                ("K", mK, sK_layout),
                ("V", mV, sV_layout),
            ]
        }

        # TMA load for Q
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)
        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()

        tma_atom_Q, mQ = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mQ,
            cute.select(sQ_layout, mode=[0, 1, 2]),
            self.mma_tiler_qk,
            tiled_mma_qk,
            cta_layout_vmnk.shape,
        )
        # TMA load for K
        tma_atom_K, mK = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mK,
            cute.select(sK_layout, mode=[0, 1, 2]),
            self.mma_tiler_qk,
            tiled_mma_qk,
            cta_layout_vmnk.shape,
        )
        # TMA load for V
        tma_atom_V, mV = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mV,
            cute.select(sV_layout, mode=[0, 1, 2]),
            self.mma_tiler_pv,
            tiled_mma_pv,
            cta_layout_vmnk.shape,
        )

        self.num_epilogue_threads = cute.arch.WARP_SIZE * len(self.epilogue_warp_ids)
        tma_atom_O, mO = cpasync.make_tiled_tma_atom(
            tma_store_op, mO, cute.select(sO_layout, mode=[0, 1]), self.epi_tile
        )

        if const_expr(max_seqlen_q is None):
            eff_seqlen_q = cute.size(mQ.shape[0])
        else:
            eff_seqlen_q = max_seqlen_q
        TileScheduler = self.TileScheduler
        num_work_m = cute.ceil_div(eff_seqlen_q, self.cta_tiler[0])
        tile_sched_args = TileSchedulerArguments(
            num_work_m,
            cute.size(mQ.shape[2]),
            cute.size(mQ.shape[3]),
            num_splits,
            cute.size(mK.shape[0]),
            mQ.shape[1],
            mV.shape[0],  # Note that this is different from Sm90 since we transpose mV in Sm100
            total_q=cute.size(mQ.shape[0]) * cute.size(mQ.shape[3]),
            tile_shape_mn=self.cta_tiler[:2],
            mCuSeqlensQ=None,
            mSeqUsedQ=None,
            qhead_per_kvhead_packgqa=1,
            element_size=self.k_dtype.width // 8,
            is_persistent=self.is_persistent,
            lpt=self.is_causal,
            is_split_kv=False,
            cluster_shape_mn=self.cluster_shape_mn,
            use_cluster_idx=False,
            lpt_global=self.lpt_global,
            num_splits_dynamic_ptr=None,
            virtual_batch_idx_ptr=None,
            num_nheads_in_l2_ptr=None,
            cu_total_m_blocks_ptr=None,
            cu_total_splits_m_blocks_ptr=None,
            blocks_to_batch_idx_ptr=None,
            tile_count_semaphore=None,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(
            tile_sched_args, scheduling_mode=self.scheduling_mode
        )
        self.tile_scheduler_cls = TileScheduler
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        sO_size = cute.cosize(sO_layout)
        sQ_size = cute.cosize(sQ_layout)
        # K and V alias the same physical buffer and may have different extents.
        sKV_size = cutlass.max(cute.cosize(sK_layout), cute.cosize(sV_layout))

        sched_response_size = self.sched_stages * 4 if self.dynamic_persistent else 0
        sched_mbar_size = self.sched_stages * 2 if self.dynamic_persistent else 0
        load_epi_mbar_size = 0

        dm_nb1, dm_nb2 = (0, 0)
        dm_nb3 = 0
        dm_sc = 1
        dm_sP_size = 0
        dm_slb = 6 if self.dm_slice else 0      # slice: kbfull[2], bsfull[2], bsempty[2]
        dm_ordn = 2      # PV_s completion barrier per stage (ord)
        dm_slf = 2 * 2 * 4 * self.m_block_size if self.dm_slice else 0   # slice: sX[2][4][128], sB[2][4][128] fp32

        @cute.struct
        class SharedStorage:
            # m_barriers for pipelines
            mbar_load_Q: cute.struct.MemRange[Int64, self.q_stage * 2]
            mbar_load_KV: cute.struct.MemRange[Int64, self.kv_stage * 2]
            mbar_S_full_P_full_O_rescaled: cute.struct.MemRange[Int64, self.s_p_o_stage * 2]
            mbar_P_full_lastsplit: cute.struct.MemRange[Int64, self.s_p_o_stage * 2]
            mbar_O_full: cute.struct.MemRange[Int64, self.s_p_o_stage * 2]
            mbar_softmax_stats: cute.struct.MemRange[Int64, self.s_p_o_stage * 2]
            # mbar_softmax_stats: cute.struct.MemRange[Int64, self.q_stage * 4 * 2]
            mbar_O_epi: cute.struct.MemRange[Int64, self.q_stage * 2]
            mbar_load_epi: cute.struct.MemRange[Int64, load_epi_mbar_size]
            mbar_s0_s1_sequence: cute.struct.MemRange[Int64, 2 * 2]
            mbar_dm_sfree: cute.struct.MemRange[Int64, dm_nb1]
            mbar_dm_pvdone: cute.struct.MemRange[Int64, dm_nb2]
            mbar_dm_cread: cute.struct.MemRange[Int64, dm_nb3]
            mbar_dm_slice: cute.struct.MemRange[Int64, dm_slb]
            mbar_dm_ord: cute.struct.MemRange[Int64, dm_ordn]
            # Tmem dealloc cluster barrier
            tmem_dealloc_mbar: Int64
            # Tmem holding buffer
            tmem_holding_buf: Int32
            # Smem tensors
            # store row max and row sum
            sScale: cute.struct.MemRange[Float32, self.s_p_o_stage * self.m_block_size * 2 * dm_sc]
            sDmSlice: cute.struct.MemRange[Float32, dm_slf]
            # Scheduler buffers placed here to utilize padding before sO's 1024-byte
            # alignment. This avoids adding bytes at the end when we're at the smem limit.
            # PipelineClcFetchAsync / PipelineAsync both expect
            # 2 * sched_stages mbarriers (full + empty). Response is 4 Int32 per stage
            # (CLC HW response, or work_info written by dynamic persistent producer).
            sched_mbar_ptr: cute.struct.MemRange[Int64, sched_mbar_size]
            sched_response: cute.struct.MemRange[Int32, sched_response_size]
            # Large TMA buffers with 1024-byte alignment
            sO: cute.struct.Align[
                cute.struct.MemRange[self.o_dtype, sO_size], self.buffer_align_bytes
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, sQ_size], self.buffer_align_bytes
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, sKV_size],
                self.buffer_align_bytes,
            ]
            sP: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, dm_sP_size],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage
        smem_capacity = sm100_smem_capacity(self.arch.major * 10 + self.arch.minor)
        assert SharedStorage.size_in_bytes() <= smem_capacity, (
            f"SMEM {SharedStorage.size_in_bytes()} > {smem_capacity}: kv_stage={self.kv_stage}"
        )

        softmax_scale_log2, softmax_scale = utils.compute_softmax_scale_log2(softmax_scale, None)
        fastdiv_mods = utils.compute_fastdiv_mods(mQ, mK, self.qhead_per_kvhead, False, aux_data.tensors, None)
        # Launch the kernel synchronously
        self.kernel(
            mQ,
            mK,
            mV,
            mO,
            mLSE,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_O,
            softmax_scale_log2,
            softmax_scale,
            descale_tensors,
            sQ_layout,
            sK_layout,
            tP_layout,
            sV_layout,
            sO_layout,
            tiled_mma_qk,
            tiled_mma_pv,
            tile_sched_params,
            num_splits,
            aux_data,
            fastdiv_mods,
        ).launch(
            grid=grid_dim,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk if cute.size(self.cluster_shape_mnk) > 1 else None,
            stream=stream,
            min_blocks_per_mp=1,
        )

    #  GPU device kernel
    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,  # (s_q, d, h, b)
        mK: cute.Tensor,  # (s_k, d, h_k, b_k)
        mV: cute.Tensor,  # (d, s_k, h_k, b_k)
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_O: cute.CopyAtom,
        softmax_scale_log2: Float32,
        softmax_scale: Float32 | None,
        descale_tensors: DescaleTensors,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        tP_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tile_sched_params: ParamsBase,
        num_splits: Int32,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=(None, None),
    ):
        """The device kernel implementation of the Fused Multi-Head Attention.

        This kernel coordinates multiple specialized warps to perform different phases of the FMHA computation:
        1. Load warp: Loads Q, K, V data from global memory to shared memory using TMA
        2. MMA warp: Performs matrix multiplications (Q*K^T and P*V)
        3. Softmax warps: Compute softmax normalization on attention scores
        4. Correction warps: Apply adjustments to intermediate results
        5. Epilogue warp: Handles final output transformation and storage

        The kernel implements a complex pipeline with overlapping computation and memory operations,
        using tensor memory access (TMA) for efficient data loading, warp specialization for different
        computation phases, and optional attention masking.
        """
        begin_kernel_fences(1)   # tcgen05 thread-sync fences (FENCE_MODE)

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # Prefetch tma descriptor
        if warp_idx == 0:
            for tma_atom in (tma_atom_Q, tma_atom_K, tma_atom_V, tma_atom_O):
                cpasync.prefetch_descriptor(tma_atom)

        cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (tiled_mma_qk.thr_id.shape,)
        )
        # Setup cta/thread coordinates
        mma_tile_coord_v = 0
        is_leader_cta = True

        # Alloc
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # TMEM handoffs span warp-role branches, so all waits and arrivals must be unaligned.
        tmem_alloc_barrier = pipeline_custom.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100.TmemPtr),
            num_threads=cute.arch.WARP_SIZE * len(
                (self.mma_warp_id,
                 *self.softmax0_warp_ids,
                 *self.softmax1_warp_ids,
                 *self.correction_warp_ids)
            ),
        )
        # Tensor memory dealloc barrier init
        tmem = cutlass.utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.mma_warp_id,
            is_two_cta=False,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

        ThreadCooperativeGroup = partial(pipeline.CooperativeGroup, pipeline.Agent.Thread)
        mma_warp = ThreadCooperativeGroup(len([self.mma_warp_id]))
        tma_warp = ThreadCooperativeGroup(1)
        softmax_threads = ThreadCooperativeGroup(cute.arch.WARP_SIZE * len(self.softmax0_warp_ids))
        correction_threads = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.correction_warp_ids)
        )
        epilogue_threads = ThreadCooperativeGroup(cute.arch.WARP_SIZE * len(self.epilogue_warp_ids))
        softmax_warps_cluster = ThreadCooperativeGroup(len(self.softmax0_warp_ids))
        correction_threads_cluster = ThreadCooperativeGroup(cute.arch.WARP_SIZE * len(self.correction_warp_ids))
        softmax_correction_threads_cluster = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.softmax0_warp_ids + self.correction_warp_ids)
        )
        pipeline_q = pipeline_custom.PipelineTmaUmma.create(
            barrier_storage=storage.mbar_load_Q.data_ptr(),
            num_stages=self.q_stage,
            producer_group=tma_warp,
            consumer_group=mma_warp,
            tx_count=self.tma_copy_bytes["Q"],
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_kv = pipeline_custom.PipelineTmaUmma.create(
            barrier_storage=storage.mbar_load_KV.data_ptr(),
            num_stages=self.kv_stage,
            producer_group=tma_warp,
            consumer_group=mma_warp,
            tx_count=self.tma_copy_bytes["K"],
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        # This pipeline is not the typical producer-consumer pipeline. The "producer" mma warp
        # uses it to signal that S is ready, and the softmax threads wait for S to be ready.
        # When softmax threads write P to tmem and the correction threads have rescaled O, they
        # signal as "consumer". The mma warp then waits for that signal to do the P @ V gemm.
        pipeline_s_p_o = pipeline_custom.PipelineUmmaAsync.create(
            barrier_storage=storage.mbar_S_full_P_full_O_rescaled.data_ptr(),
            num_stages=self.s_p_o_stage,
            producer_group=mma_warp,
            consumer_group=softmax_correction_threads_cluster,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_p_lastsplit = pipeline_custom.PipelineAsyncUmma.create(
            barrier_storage=storage.mbar_P_full_lastsplit.data_ptr(),
            num_stages=self.s_p_o_stage,
            producer_group=softmax_warps_cluster,
            consumer_group=mma_warp,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        # MMA warp uses this to signal to the correction warps that O is ready.
        # Ping-pong commits it after every PV (slot pv_count & 1); otherwise once per tile per stage.
        pipeline_o_acc = pipeline_custom.PipelineUmmaAsync.create(
            barrier_storage=storage.mbar_O_full.data_ptr(),
            num_stages=self.s_p_o_stage,
            producer_group=mma_warp,
            consumer_group=correction_threads_cluster,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_sm_stats = pipeline_custom.PipelineAsync.create(
            barrier_storage=storage.mbar_softmax_stats.data_ptr(),
            num_stages=self.s_p_o_stage,
            producer_group=softmax_threads,
            consumer_group=correction_threads,
            defer_sync=True,
        )
        # Should put the NamedBarrier inside the pipeline class so we'll just have pipeline_sm_stats
        sm_stats_barrier = pipeline_custom.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100.SoftmaxStatsW0), num_threads=cute.arch.WARP_SIZE * 2
        )
        pipeline_o_epi = pipeline_custom.PipelineAsync.create(
            barrier_storage=storage.mbar_O_epi.data_ptr(),
            num_stages=self.q_stage,
            producer_group=correction_threads,
            consumer_group=epilogue_threads,
            defer_sync=True,
        )


        # mark the tcgen05 hand-off pipelines / barriers (mark_fenced)
        mark_fenced("ua", pipeline_s_p_o, pipeline_o_acc)
        mark_fenced("au", pipeline_p_lastsplit)
        mark_fenced("bar", sm_stats_barrier, tmem_alloc_barrier)
        # Cluster arrive after barrier init
        if const_expr(self.dm_slice):
            self.dm_slbar = Int32(storage.mbar_dm_slice.data_ptr().toint())
            self.dm_slbuf = Int32(storage.sDmSlice.data_ptr().toint())
            if cute.arch.thread_idx()[0] == 0:
                for _s in range(2):
                    mbar_init(self.dm_slbar + 8 * _s, 16)        # kbfull[s]: one arrival per slice warp
                    mbar_init(self.dm_slbar + 16 + 8 * _s, 16)   # bsfull[s]
                    mbar_init(self.dm_slbar + 32 + 8 * _s, 4)    # bsempty[s]: one per correction warp
                cute.arch.mbarrier_init_fence()
        self.dm_ord = Int32(storage.mbar_dm_ord.data_ptr().toint())
        if cute.arch.thread_idx()[0] == 0:
            mbar_init(self.dm_ord, 1)
            mbar_init(self.dm_ord + 8, 1)
            cute.arch.mbarrier_init_fence()
        pipeline_init_arrive(cluster_shape_mn=cta_layout_vmnk, is_relaxed=True)

        #  Generate smem tensor Q/K/V/O
        # (MMA, MMA_Q, MMA_D, PIPE)
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        # (MMA, MMA_K, MMA_D, PIPE)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        # (MMA, MMA_K, MMA_D, PIPE)
        # Strip swizzle info to reuse smem
        sV = cute.make_tensor(cute.recast_ptr(sK.iterator, sV_layout.inner), sV_layout.outer)
        sO = storage.sO.get_tensor(sO_layout.outer, swizzle=sO_layout.inner)

        sScale = storage.sScale.get_tensor(cute.make_layout(self.s_p_o_stage * self.m_block_size * 2 * 1))

        thr_mma_qk = tiled_mma_qk.get_slice(mma_tile_coord_v)
        thr_mma_pv = tiled_mma_pv.get_slice(mma_tile_coord_v)

        qk_acc_shape = thr_mma_qk.partition_shape_C(self.mma_tiler_qk[:2])
        # This is a fake tensor, by right we need to retrieve tmem_ptr. But we know that we always
        # request 512 columns of tmem, so we know that it starts at 0.
        tStS = thr_mma_qk.make_fragment_C(cute.append(qk_acc_shape, self.s_stage))
        pv_acc_shape = thr_mma_pv.partition_shape_C(self.mma_tiler_pv[:2])
        tOtO = thr_mma_pv.make_fragment_C(cute.append(pv_acc_shape, self.q_stage))
        tOtO = cute.make_tensor(tOtO.iterator + self.tmem_o_offset[0], tOtO.layout)
        tP = cute.make_tensor(tStS.iterator, tP_layout.outer)
        tOrP = thr_mma_pv.make_fragment_A(tP)[None, None, None, 0]
        # Need to multiply by width ratio bc tP is in v_dtype but tmem offsets are in FP32
        tP_width_ratio = Float32.width // self.v_dtype.width
        # Need to adjust the stage stride manually since the two stages aren't contiguous in tmem
        tP_stage_stride = (self.tmem_p_offset[1] - self.tmem_p_offset[0]) * tP_width_ratio
        tOrP = cute.make_tensor(
            tOrP.iterator + self.tmem_p_offset[0] * tP_width_ratio,
            cute.append(tOrP.layout, cute.make_layout((self.s_stage,), stride=(tP_stage_stride,)))
        )

        block_info = BlockInfo(
            self.cta_tiler[0],
            self.cta_tiler[1],
            self.is_causal,
            False,
            False,
            None,
            None,
            qhead_per_kvhead_packgqa=1,
            num_splits=num_splits,
            pack_split_idx=False,
            num_n_blocks_per_split=None,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0],
            seqlen_k_static=mK.shape[0],
            mCuSeqlensQ=None,
            mCuSeqlensK=None,
            mSeqUsedQ=None,
            mSeqUsedK=None,
            mCuTotalMBlocks=None,
            mCuBlockIdxOffsets=None,
        )
        AttentionMaskCls = partial(
            AttentionMask,
            self.m_block_size,
            self.n_block_size,
            window_size_left=None,
            window_size_right=None,
            qhead_per_kvhead_packgqa=1,
        )
        # Cluster wait before tensor memory alloc
        pipeline_init_wait(cluster_shape_mn=cta_layout_vmnk)

        sched_ctx = None
        if const_expr(self.use_clc_scheduler):
            sched_response_ptr = storage.sched_response.data_ptr()
            sched_mbar_ptr = storage.sched_mbar_ptr.data_ptr()
            sched_producer_group = cutlass_pipeline.CooperativeGroup(
                cutlass_pipeline.Agent.Thread
            )
            num_sched_consumer_warps_per_cta = self.threads_per_cta // cute.arch.WARP_SIZE
            # NB on CTA0 warp15 == scheduler on CTA1 == empty but still both consume
            num_sched_consumer_warps = num_sched_consumer_warps_per_cta
            sched_consumer_group = cutlass_pipeline.CooperativeGroup(
                cutlass_pipeline.Agent.Thread,
                cute.arch.WARP_SIZE * num_sched_consumer_warps,
            )
            block_idx = cute.arch.block_idx()
            sched_ctx = SchedulerState.create_clc(
                hw_scheduler=ClcDynamicPersistentTileScheduler.create(
                    self.tile_scheduler_cls.clc_problem_shape(tile_sched_params),
                    block_idx,
                    cute.arch.grid_dim(),
                    sched_response_ptr,
                ),
                pipeline=cutlass_pipeline.PipelineClcFetchAsync.create(
                    barrier_storage=sched_mbar_ptr,
                    num_stages=self.sched_stages,
                    producer_group=sched_producer_group,
                    consumer_group=sched_consumer_group,
                    tx_count=16,
                    cta_layout_vmnk=cta_layout_vmnk,
                ),
                consumer_state=cutlass_pipeline.make_pipeline_state(
                    cutlass_pipeline.PipelineUserType.Consumer, self.sched_stages
                ),
                producer_state=cutlass_pipeline.make_pipeline_state(
                    cutlass_pipeline.PipelineUserType.Producer, self.sched_stages
                ),
            )
        if const_expr(self.use_clc_scheduler):
            tile_scheduler = self.tile_scheduler_cls.create(tile_sched_params, ctx=sched_ctx)
        else:
            tile_scheduler = self.tile_scheduler_cls.create(tile_sched_params)
        assert isinstance(tile_scheduler, TileSchedulerProtocol), f"tile_scheduler is not a TileSchedulerProtocol: {type(tile_scheduler)}"

        # ///////////////////////////////////////////////////////////////////////////////
        #  EMPTY / SCHEDULER WARP
        # ///////////////////////////////////////////////////////////////////////////////
        if const_expr(self.dynamic_persistent):
            # CLC with late claims: the load warp claims; the scheduler warp only follows the schedule.
            if warp_idx == self.scheduler_warp_id:
                cute.arch.setmaxregister_decrease(self.num_regs_other)
                self.empty_warp(tile_scheduler)
            for i in cutlass.range_constexpr(len(self.empty_warp_ids)):
                if warp_idx == self.empty_warp_ids[i] and warp_idx != self.scheduler_warp_id:
                    cute.arch.setmaxregister_decrease(self.num_regs_other)
                    self.empty_warp(tile_scheduler)
        else:
            for i in cutlass.range_constexpr(len(self.empty_warp_ids)):
                if warp_idx == self.empty_warp_ids[i]:
                    cute.arch.setmaxregister_decrease(self.num_regs_other)

        # ///////////////////////////////////////////////////////////////////////////////
        #  LOAD
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.load_warp_ids[0] and warp_idx <= self.load_warp_ids[-1]:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            self.load(
                thr_mma_qk,
                thr_mma_pv,
                mQ,
                mK,
                mV,
                sQ,
                sK,
                sV,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                pipeline_q,
                pipeline_kv,
                block_info,
                num_splits,
                SeqlenInfoCls,
                tile_scheduler=tile_scheduler,
            )

        # ///////////////////////////////////////////////////////////////////////////////
        #  MMA
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            # Alloc tensor memory buffer
            tmem.allocate(cute.arch.get_max_tmem_alloc_cols("sm_100"))
            tmem_alloc_barrier.arrive_and_wait_unaligned()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            self.mma(
                tiled_mma_qk,
                tiled_mma_pv,
                sQ,
                sK,
                sV,
                tStS,
                tOtO,
                tOrP,
                pipeline_q,
                pipeline_kv,
                pipeline_s_p_o,
                pipeline_p_lastsplit,
                pipeline_o_acc,
                is_leader_cta,
                block_info,
                num_splits,
                SeqlenInfoCls,
                tile_scheduler=tile_scheduler,
            )
            # Dealloc the tensor memory buffer
            tmem.relinquish_alloc_permit()
            tmem_alloc_barrier.arrive_and_wait_unaligned()
            tmem.free(tmem_ptr)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Epilogue
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.epilogue_warp_ids[0] and warp_idx <= self.epilogue_warp_ids[-1]:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            self.epilogue_s2g(
                mO,
                sO,
                tma_atom_O,
                pipeline_o_epi,
                block_info,
                num_splits,
                SeqlenInfoCls,
                mma_tile_coord_v,
                tile_scheduler=tile_scheduler,
            )

        # ///////////////////////////////////////////////////////////////////////////////
        #  Softmax
        # ///////////////////////////////////////////////////////////////////////////////
        if const_expr(self.dm_slice):
            if warp_idx < 16:
                if const_expr(self.num_regs_softmax > 80):
                    cute.arch.setmaxregister_increase(self.num_regs_softmax)
                elif const_expr(self.num_regs_softmax < 80):
                    cute.arch.setmaxregister_decrease(self.num_regs_softmax)
                tmem_alloc_barrier.arrive_and_wait_unaligned()
                tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
                self.slice_softmax_loop(
                    thr_mma_qk, tStS, descale_tensors, pipeline_s_p_o, block_info, num_splits, SeqlenInfoCls,
                    tile_scheduler=tile_scheduler,
                )
                tmem_alloc_barrier.arrive_unaligned()
        elif warp_idx <= self.softmax0_warp_ids[-1]:
            # increase register after decreasing
            cute.arch.setmaxregister_increase(self.num_regs_softmax)
            # sync with mma warp before retrieving tmem ptr
            tmem_alloc_barrier.arrive_and_wait_unaligned()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            softmax_loop = partial(
                self.softmax_loop,
                softmax_scale_log2=softmax_scale_log2,
                softmax_scale=softmax_scale,
                descale_tensors=descale_tensors,
                thr_mma_qk=thr_mma_qk,
                sScale=sScale,
                mLSE=mLSE,
                pipeline_s_p_o=pipeline_s_p_o,
                pipeline_p_lastsplit=pipeline_p_lastsplit,
                pipeline_sm_stats=pipeline_sm_stats,
                sm_stats_barrier=sm_stats_barrier,
                block_info=block_info,
                num_splits=num_splits,
                SeqlenInfoCls=SeqlenInfoCls,
                AttentionMaskCls=AttentionMaskCls,
                aux_data=aux_data,
                fastdiv_mods=fastdiv_mods,
                tile_scheduler=tile_scheduler,
            )
            stage = Int32(0)
            softmax_loop(stage=stage, tStS=tStS)
            tmem_alloc_barrier.arrive_unaligned()

        # ///////////////////////////////////////////////////////////////////////////////
        #  Correction
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.correction_warp_ids[0] and warp_idx < self.mma_warp_id:
            # setmaxnreg.dec cannot exceed the launch register count.
            if const_expr(self.dm_slice):
                if const_expr(self.num_regs_correction > 65536 // self.threads_per_cta // 8 * 8):
                    cute.arch.setmaxregister_increase(self.num_regs_correction)
                elif const_expr(self.num_regs_correction < 65536 // self.threads_per_cta // 8 * 8):
                    cute.arch.setmaxregister_decrease(self.num_regs_correction)
            elif const_expr(self.num_regs_correction > SM100_LAUNCH_REGISTERS):
                cute.arch.setmaxregister_increase(self.num_regs_correction)
            else:
                cute.arch.setmaxregister_decrease(self.num_regs_correction)
            # sync with mma warp before retrieving tmem ptr
            tmem_alloc_barrier.arrive_and_wait_unaligned()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            if const_expr(self.dm_slice):
                self.slice_correction_loop(
                    thr_mma_qk, thr_mma_pv, tStS, tOtO, mO, mLSE, sO, pipeline_s_p_o, pipeline_o_acc, pipeline_o_epi,
                    descale_tensors, block_info, num_splits, SeqlenInfoCls,
                    tile_scheduler=tile_scheduler,
                )
                tmem_alloc_barrier.arrive_unaligned()
            else:
                self.correction_loop(
                    thr_mma_qk,
                    thr_mma_pv,
                    tStS,
                    tOtO,
                    sScale,
                    mO,
                    mLSE,
                    sO,
                    pipeline_s_p_o,
                    pipeline_o_acc,
                    pipeline_sm_stats,
                    sm_stats_barrier,
                    pipeline_o_epi,
                    block_info,
                    num_splits,
                    SeqlenInfoCls,
                    tile_scheduler=tile_scheduler,
                )
                tmem_alloc_barrier.arrive_unaligned()

        end_kernel_fences()
        return

    @cute.jit
    def load(
        self,
        thr_mma_qk: cute.ThrMma,
        thr_mma_pv: cute.ThrMma,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        pipeline_q: pipeline.PipelineAsync,
        pipeline_kv: pipeline.PipelineAsync,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        tile_scheduler: TileSchedulerProtocol,
    ):
        q_producer_phase = Int32(1)
        kv_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.kv_stage
        )
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            work_m, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
            head_idx_kv = head_idx // self.qhead_per_kvhead
            mK_cur = mK[None, None, head_idx_kv, batch_idx]
            mV_cur = mV[None, None, head_idx_kv, batch_idx]
            gK = cute.local_tile(mK_cur, cute.select(self.mma_tiler_qk, mode=[1, 2]), (None, 0))
            gV = cute.local_tile(mV_cur, cute.select(self.mma_tiler_pv, mode=[1, 2]), (0, None))
            tSgK = thr_mma_qk.partition_B(gK)
            tOgV = thr_mma_pv.partition_B(gV)
            tiler_gQ = ((self.mma_tiler_qk[0] * self.q_stage), self.head_dim_padded)
            gQ = cute.local_tile(mQ_cur, tiler_gQ, (work_m, 0))  # (128 * 2, 128)
            gQ = layout_utils.select(
                cute.flat_divide(gQ, (self.mma_tiler_qk[0],)), mode=[0, 2, 1]
            )  # (128, 128, 2)
            tSgQ = thr_mma_qk.partition_A(gQ)
            load_Q_fn, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_Q, 0, cute.make_layout(1), tSgQ, sQ
            )
            load_Q = partial(self.load_Q, load_Q_fn, pipeline_q=pipeline_q, phase=q_producer_phase)
            tKsK, tKgK = cpasync.tma_partition(
                tma_atom_K,
                0,  # no multicast
                cute.make_layout(1),
                cute.group_modes(sK, 0, 3),
                cute.group_modes(tSgK, 0, 3),
            )
            tVsV, tVgV = cpasync.tma_partition(
                tma_atom_V,
                0,  # no multicast
                cute.make_layout(1),
                cute.group_modes(sV, 0, 3),
                cute.group_modes(tOgV, 0, 3),
            )
            load_K = partial(self.load_KV, tma_atom_K, tKgK, tKsK, sK, pipeline_kv=pipeline_kv, K_or_V="K")
            load_V = partial(self.load_KV, tma_atom_V, tVgV, tVsV, sV, pipeline_kv=pipeline_kv, K_or_V="V")

            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen,
                work_m,
                split_idx=split_idx,
                num_splits=num_splits,
            )
            if self.process_work_tile(block_info, seqlen, n_block_min, n_block_max):
                load_K(block=n_block_max - 1, producer_state=kv_producer_state)  # K0
                load_Q(block=0, stage=0)
                kv_producer_state.advance()
                if const_expr(self.q_stage == 2):
                    load_Q(block=1, stage=1)
                q_producer_phase ^= 1
                load_V(block=n_block_max - 1, producer_state=kv_producer_state)  # V0
                kv_producer_state.advance()
                for i in cutlass.range(n_block_max - 1 - n_block_min, unroll=1):
                    n_block = n_block_max - 2 - i
                    load_K(block=n_block, producer_state=kv_producer_state)  # Ki
                    kv_producer_state.advance()
                    load_V(block=n_block, producer_state=kv_producer_state)  # Vi
                    kv_producer_state.advance()

            if const_expr(self.clc_late_claim):
                tile_scheduler.prefetch_next_work()
            work_tile = tile_scheduler.advance_to_next_work()

        if const_expr(self.clc_late_claim):
            tile_scheduler.producer_tail()
        pipeline_kv.producer_tail(kv_producer_state)
        # This is equivalent to pipeline_q.producer_tail for the TMA-Q producer warp.
        pipeline_q.producer_acquire_w_index_phase(self.q_stage - 1, q_producer_phase)

    @cute.jit
    def mma(
        self,
        tiled_mma_qk: cute.ThrMma,
        tiled_mma_pv: cute.ThrMma,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tStS: cute.Tensor,
        tOtO: cute.Tensor,
        tOrP: cute.Tensor,
        pipeline_q: pipeline.PipelineAsync,
        pipeline_kv: pipeline.PipelineAsync,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_p_lastsplit: pipeline.PipelineAsync,
        pipeline_o_acc: pipeline.PipelineAsync,
        is_leader_cta: Boolean,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        tile_scheduler=None,
    ):
        tSrQ = tiled_mma_qk.make_fragment_A(sQ)
        tSrK = tiled_mma_qk.make_fragment_B(sK)
        tOrV = tiled_mma_pv.make_fragment_B(sV)

        qk_mma_op, pv_mma_op = tiled_mma_qk.op, tiled_mma_pv.op
        qk_mma_kind = sm100_utils._tcgen05_mma_kind(qk_mma_op)
        q_smem_base = sm100_desc.smem_desc_base_from_tensor(sQ, sm100_desc.Major.K)
        k_smem_base = sm100_desc.smem_desc_base_from_tensor(sK, sm100_desc.Major.K)
        q_smem_start = [sm100_desc.make_smem_desc_start_addr(sQ[None, None, None, stage].iterator) for stage in range(self.q_stage)]

        sm100_utils.declare_ptx_smem_desc(q_smem_start[self.q_stage - 1], q_smem_base, tSrQ[None, None, None, 0].layout, var_name_prefix="fa_fwd_q_smem_desc")
        sm100_utils.declare_ptx_idesc(qk_mma_op, var_name="fa_fwd_qk_mma_idesc")
        sm100_utils.declare_ptx_idesc(pv_mma_op, var_name="fa_fwd_pv_mma_idesc")

        sQ_stage_stride = (sQ.layout.stride[-1] * sQ.element_type.width // 8) >> 4
        if const_expr(self.q_stage == 1):
            sQ_stage_stride = 0
        gemm_Si = [
            partial(
                sm100_utils.gemm_ptx_precomputed_varname,
                self.tmem_s_offset[stage],
                smem_desc_base_b=k_smem_base,
                tCrB_layout=tSrK[None, None, None, 0].layout,
                smem_var_name_prefix="fa_fwd_q_smem_desc",
                idesc_var_name="fa_fwd_qk_mma_idesc",
                kind=qk_mma_kind,
                smem_offset=-sQ_stage_stride if stage == 0 else sQ_stage_stride,
                zero_init=True,
                cta_group=1,
            )
            for stage in range(self.s_p_o_stage)
        ]
        gemm_Pi = [
            partial(
                gemm_ptx_bs_ts,
                pv_mma_op,
                self.tmem_o_offset[stage % self.q_stage],
                tOrP[None, None, None, stage],
                sfa_tmem_addr=self.tmem_s_offset[stage] + (self.dm_sl_sfcol if self.dm_slice else self.dm_sf_col),
                sfb_tmem_addr=self.tmem_s_offset[stage] + (self.dm_sl_sfcol if self.dm_slice else self.dm_sf_col) + 4,
                split_arrive=self.split_P_arrive if self.split_P_arrive > 0 else None,
                # slice: P codes of k-step (key block) k at S_s + 32 k, SF at S_s + 16
                tA_addr=(Int32(self.tmem_s_offset[stage]) if const_expr(self.dm_slice) else None),
                offs_a=((0, 32, 64, 96) if const_expr(self.dm_slice) else None),
                cta_group=1,
            )
            for stage in range(self.s_p_o_stage)
        ]

        mma_q_consumer_phase = Int32(0)
        mma_kv_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.kv_stage
        )
        P_full_O_rescaled_phase = Int32(0)
        dm_pvn = Int32(0)   # PV rounds issued (per stage) by this CTA so far, across work tiles

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            work_m, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            block_iter_count = Int32(0)
            process_tile = False
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen,
                work_m,
                split_idx=split_idx,
                num_splits=num_splits,
            )
            block_iter_count = n_block_max - n_block_min
            process_tile = self.process_work_tile(block_info, seqlen, n_block_min, n_block_max)

            if process_tile and is_leader_cta:
                for stage in cutlass.range_constexpr(self.q_stage):
                    # GEMM_QK00 (Q0 * K0 -> S0) or GEMM_QK01 (Q1 * K0 -> S1)
                    # 1. wait for Q0 / Q1
                    pipeline_q.consumer_wait_w_index_phase(stage, mma_q_consumer_phase)
                    # 2. wait for K0
                    if const_expr(stage == 0):
                        pipeline_kv.consumer_wait(mma_kv_consumer_state)
                    Ki_index = mma_kv_consumer_state.index
                    # We don't need to acquire empty S0 / S1.
                    # For the first iteration, we don't need to wait as we're guaranteed S0 / S1
                    # are empty. For subsequent iterations, the wait happened at the end
                    # of the while loop.
                    # 3. gemm
                    sK_cur = sK[None, None, None, Ki_index]
                    # the previous work tile's last PV_s (reads SF in S_s) must be complete before QK_s
                    if dm_pvn > 0:
                        mbar_wait_parity(self.dm_ord + 8 * stage, (dm_pvn - 1) & 1)
                        tcgen05_fence_after()
                    gemm_Si[stage](
                        smem_desc_start_b=sm100_desc.make_smem_desc_start_addr(sK_cur.iterator)
                    )
                    # 4. release S0 / S1
                    pipeline_s_p_o.producer_commit_w_index(stage)
                mma_q_consumer_phase ^= 1
                # 5. release K0
                pipeline_kv.consumer_release(mma_kv_consumer_state)
                mma_kv_consumer_state.advance()
                # End of GEMM (Q1 * K0 -> S1)
                # Note: Q0 & Q1 are still needed in the seqlen_kv loop
                # so we need to release them after the seqlen_kv loop

                # O hasn't been accumulated yet, its first MMA calculation doesn't need to accumulate
                block_loop_count = block_iter_count - 1
                O_should_accumulate = False
                for i in cutlass.range(block_loop_count, unroll=1):
                    # GEMM_PV00 (P0 * V0 -> O0_partial), O0 needs to be accumulated in the seqlen_kv loop
                    # 1. wait for V0
                    pipeline_kv.consumer_wait(mma_kv_consumer_state)
                    mma_kv_release_state = mma_kv_consumer_state.clone()
                    Vi_index = mma_kv_consumer_state.index
                    tOrVi = tOrV[None, None, None, Vi_index]
                    for stage in cutlass.range_constexpr(self.q_stage):
                        # 2. acquire corrected O0/O1_partial and P0 / P1
                        # For the first iteration in this work tile, waiting for O0/O1_partial
                        # means that the correction warps has finished reading tO during
                        # the last iteration of the previous work tile.
                        pipeline_s_p_o.producer_acquire_w_index_phase(stage, P_full_O_rescaled_phase)
                        if const_expr(self.dm_slice):
                            tcgen05_fence_after()   # slice: P / SF / O tcgen05.st of other warps before this PV
                        # 3. gemm
                        sV_cur = sV[None, None, None, Vi_index]
                        gemm_Pi[stage](
                            tCrB=tOrVi,
                            sB=sV_cur,
                            zero_init=not O_should_accumulate,
                            mbar_ptr=pipeline_p_lastsplit.sync_object_full.get_barrier(stage) if self.split_P_arrive > 0 else None,
                            mbar_phase=P_full_O_rescaled_phase,
                        )
                        with cute.arch.elect_one():
                            tc_commit(self.dm_ord + 8 * stage)   # PV_s(i) completion #dm_pvn
                        # Don't need to signal O_full to the correction warps since the
                        # correction warps wait for the softmax warps anyway. By the time the softmax
                        # warps finished, S_i for the next iteration must have been done, so O_i-1
                        # must have been done as well.
                        # 4. release V(i-1)
                        if const_expr(stage == self.q_stage - 1):
                            pipeline_kv.consumer_release(mma_kv_release_state)
                            mma_kv_release_state.advance()
                        # End of GEMM_PV00 (P0 * V0 -> O0_partial)

                        # GEMM_QK0i (Q0 * Ki -> S0)
                        # 1. wait for Ki
                        if const_expr(stage == 0):
                            mma_kv_consumer_state.advance()
                            pipeline_kv.consumer_wait(mma_kv_consumer_state)
                        Ki_index = mma_kv_consumer_state.index
                        # 2. gemm
                        # Don't need to wait for the softmax warp to have finished reading the previous
                        # Si, since this gemm is scheduled after the PV gemm, which guaranteed that Si
                        # has been read and Pi has been written.
                        sK_cur = sK[None, None, None, Ki_index]
                        # PV_s(i) read SFA/SFB from S_s; wait its completion before QK_s(i+1) overwrites S_s.
                        # Phase-safe: PV_s(i-1) is complete (P_full(i) needs S_s(i) loaded, whose commit tracks it).
                        mbar_wait_parity(self.dm_ord + 8 * stage, dm_pvn & 1)
                        tcgen05_fence_after()
                        gemm_Si[stage](
                            smem_desc_start_b=sm100_desc.make_smem_desc_start_addr(sK_cur.iterator)
                        )
                        # 3. release S0 / S1
                        pipeline_s_p_o.producer_commit_w_index(stage)
                        # End of GEMM_QK0i (Q0 * Ki -> S0)
                    # 4. release Ki
                    pipeline_kv.consumer_release(mma_kv_consumer_state)
                    mma_kv_consumer_state.advance()
                    P_full_O_rescaled_phase ^= 1
                    O_should_accumulate = True
                    dm_pvn += 1
                # End of seqlen_kv loop

                # release Q0 & Q1
                for stage in cutlass.range(self.q_stage):
                    pipeline_q.consumer_release_w_index(stage)

                # GEMM_PV00 (P0 * V0 -> O0_partial), O0 needs to be accumulated in the seqlen_kv loop
                # 1. wait for V0
                pipeline_kv.consumer_wait(mma_kv_consumer_state)
                Vi_index = mma_kv_consumer_state.index
                tOrVi = tOrV[None, None, None, Vi_index]
                for stage in cutlass.range_constexpr(self.q_stage):
                    # 2. acquire corrected Oi_partial and Pi
                    pipeline_s_p_o.producer_acquire_w_index_phase(stage, P_full_O_rescaled_phase)
                    if const_expr(self.dm_slice):
                        tcgen05_fence_after()
                    # 3. gemm
                    sV_cur = sV[None, None, None, Vi_index]
                    gemm_Pi[stage](
                        tCrB=tOrVi,
                        sB=sV_cur,
                        zero_init=not O_should_accumulate,
                        mbar_ptr=pipeline_p_lastsplit.sync_object_full.get_barrier(stage) if self.split_P_arrive > 0 else None,
                        mbar_phase=P_full_O_rescaled_phase,
                    )
                    with cute.arch.elect_one():
                        tc_commit(self.dm_ord + 8 * stage)
                    # 4. release accumulated O0_partial
                    # We do need O_full here since for the last tile, by the time the softmax warp
                    # has signaled to the correction warps, the softmax warp has just finished
                    # computing the row sum of the current tile. It does not guarantee that the 1st
                    # tile of the next work tile has been computed yet.
                    pipeline_o_acc.producer_commit_w_index(stage)
                    # End of GEMM_PV00 (P0 * V0 -> O0_partial)
                P_full_O_rescaled_phase ^= 1
                dm_pvn += 1
                # 5. release Vi_end
                pipeline_kv.consumer_release(mma_kv_consumer_state)
                mma_kv_consumer_state.advance()
                # End of GEMM_PV1(i_end) (P1 * Vi_end -> O1)

            # Advance to next tile
            work_tile = tile_scheduler.advance_to_next_work()
        # End of persistent scheduler loop

        # We don't need pipeline_s_p_o.producer_tail() since there's no dangling mbarrier at the end
        # We don't need pipeline_o_acc.producer_tail() since we don't call
        # pipeline_o_acc.producer_acquire() inside the loop.

    def _q_block(self, work_m, stage, cta_rank=0):
        return work_q_block(work_m, stage, cta_rank, q_stage=self.q_stage, cta_group_size=1)

    @cute.jit
    def softmax_loop(
        self,
        stage: int | Int32,
        softmax_scale_log2: Float32,
        softmax_scale: Float32 | None,
        descale_tensors: DescaleTensors,
        thr_mma_qk: cute.ThrMma,
        tStS: cute.Tensor,  # ((TILE_M, TILE_N), 1, 1, q_stage)
        sScale: cute.Tensor,
        mLSE: cute.Tensor,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_p_lastsplit: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        sm_stats_barrier: pipeline.NamedBarrier,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=(None, None),
        tile_scheduler=None,
    ):
        """Compute softmax on attention scores from QK matrix multiplication.

        This method handles the softmax computation for either the first or second half of the
        attention matrix, depending on the 'stage' parameter. It calculates row-wise maximum
        and sum values needed for stable softmax computation, applies optional masking, and
        transforms raw attention scores into probability distributions.

        The implementation uses specialized memory access patterns and efficient math operations
        for computing exp(x) using exp2 functions. It also coordinates pipeline
        synchronization between MMA, correction, and sequence processing stages.
        """
        tidx = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * (len(self.softmax0_warp_ids)))
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4

        tSAcc = tStS[(None, None), 0, 0, stage]  # (128, 128)
        tStScale = cute.composition(tSAcc, cute.make_layout((self.m_block_size, 1)))

        tilePlikeFP32 = self.mma_tiler_qk[1] // Float32.width * self.v_dtype.width
        tStP_layout = cute.composition(
            tSAcc.layout, cute.make_layout((self.m_block_size, tilePlikeFP32))
        )
        tStP = cute.make_tensor(tSAcc.iterator + self.tmem_s_to_p_offset, tStP_layout)

        tmem_load_op = tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32))
        tmem_load_atom = cute.make_copy_atom(tmem_load_op, self.qk_acc_dtype)
        thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tSAcc).get_slice(tidx)
        tStS_t2r = thr_tmem_load.partition_S(tSAcc)  # (((32,32),1),1,4)

        tmem_store_scale_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(1)), Float32
        )
        thr_tmem_store_scale = tcgen05.make_tmem_copy(tmem_store_scale_atom, tStScale).get_slice(
            tidx
        )
        tStScale_r2t = thr_tmem_store_scale.partition_D(tStScale)
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(
                tcgen05.copy.Repetition(8 if const_expr(self.q_dtype.width == 8) else 16)
            ),
            Float32,
        )
        thr_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tStP).get_slice(tidx)
        tStP_r2t = thr_tmem_store.partition_D(tStP)  # (((16,32),1),1,4)
        # DM sm100-mx: scale-factor words (4 x SFA replicated, 4 x SFB) in the free S columns after fp8 P
        tStSF = cute.make_tensor(
            tSAcc.iterator + self.dm_sf_col,
            cute.composition(tSAcc.layout, cute.make_layout((self.m_block_size, 8)))
        )
        tmem_store_sf_atom = cute.make_copy_atom(tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(8)), Float32)
        thr_tmem_store_sf = tcgen05.make_tmem_copy(tmem_store_sf_atom, tStSF).get_slice(tidx)
        tStSF_r2t = thr_tmem_store_sf.partition_D(tStSF)

        s_p_o_consumer_count = Int32(0)
        sm_stats_producer_phase = Int32(1)
        s0_s1_sequence_phase = Int32(1 if stage == 0 else 0)

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            work_m, head_idx, batch_idx, split_idx = work_tile.tile_idx
            kv_head_idx = head_idx // self.qhead_per_kvhead
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, work_m, split_idx=split_idx, num_splits=num_splits,
            )

            mask = AttentionMaskCls(seqlen)
            shared_mask_kwargs = dict(
                # cta_rank=0: see NOTE [SM100 forward coordinate spaces].
                m_block=self._q_block(work_m, stage),
                thr_mma=thr_mma_qk,
                thr_tmem_load=thr_tmem_load,
                mask_causal=self.is_causal,
                mask_local=False,
                batch_idx=batch_idx,
                head_idx=head_idx,
                aux_data=aux_data,
                vec_size=1,
            )
            mask_fn = partial(
                mask.apply_mask_sm100,
                mask_mod=None,
                fastdiv_mods=fastdiv_mods,
                head_divmod=None,
                **shared_mask_kwargs,
            )

            # DM sm100-mx: row max kept in log2 units (integer valued); scales applied per 32-key block in softmax_step.
            dm_row = self._q_block(work_m, stage, thr_mma_qk.thr_idx) * self.m_block_size + tidx
            # rows past seqlen_q (second q stage of the last tile) are never stored: clamp the scale index (no OOB read)
            dm_qblk = cutlass.min(dm_row // 32, (seqlen.seqlen_q + 31) // 32 - 1)
            dm_qsc = Float32(descale_tensors.q_descale[batch_idx, head_idx, dm_qblk]) * Float32(math.log2(math.e))
            softmax = SoftmaxSm100.create(
                Float32(1.0),
                rescale_threshold=0.0,
                softmax_scale=None,
                max_offset=0,
            )
            softmax.reset()

            has_work = self.process_work_tile(block_info, seqlen, n_block_min, n_block_max)

            softmax_step = partial(
                self.softmax_step,
                softmax=softmax,
                thr_mma_qk=thr_mma_qk,
                pipeline_s_p_o=pipeline_s_p_o,
                pipeline_p_lastsplit=pipeline_p_lastsplit,
                pipeline_sm_stats=pipeline_sm_stats,
                sm_stats_barrier=sm_stats_barrier,
                thr_tmem_load=thr_tmem_load,
                thr_tmem_store=thr_tmem_store,
                thr_tmem_store_scale=thr_tmem_store_scale,
                tStS_t2r=tStS_t2r,
                tStScale_r2t=tStScale_r2t,
                tStP_r2t=tStP_r2t,
                sScale=sScale,
                thr_tmem_store_sf=thr_tmem_store_sf,
                tStSF_r2t=tStSF_r2t,
                dm_qsc=dm_qsc,
                dm_scales=descale_tensors,
                kv_head_idx=kv_head_idx,
                stage=stage,
                batch_idx=batch_idx,
                head_idx=head_idx,
                m_block=self._q_block(work_m, stage),
                seqlen=seqlen,
            )

            if has_work:
                pipeline_sm_stats.producer_acquire_w_index_phase(stage, sm_stats_producer_phase)
                sm_stats_producer_phase ^= 1

            if has_work:
                s_p_o_consumer_count, sm_stats_producer_phase, s0_s1_sequence_phase = softmax_step(
                    s_p_o_consumer_count,
                    sm_stats_producer_phase,
                    s0_s1_sequence_phase,
                    n_block_max - 1,
                    is_first=True,
                    mask_fn=partial(mask_fn, mask_seqlen=True),
                )
                n_block_max -= 1
                # Next couple of iterations with causal masking
                if const_expr(self.is_causal):
                    n_block_min_causal_local_mask = block_info.get_n_block_min_causal_local_mask(
                        seqlen, work_m, n_block_min
                    )
                    for n_tile in cutlass.range(n_block_max - n_block_min_causal_local_mask, unroll=1):
                        n_block = n_block_max - 1 - n_tile
                        s_p_o_consumer_count, sm_stats_producer_phase, s0_s1_sequence_phase = (
                            softmax_step(
                                s_p_o_consumer_count,
                                sm_stats_producer_phase,
                                s0_s1_sequence_phase,
                                n_block,
                                mask_fn=partial(mask_fn, mask_seqlen=False),
                            )
                        )
                    n_block_max = cutlass.min(n_block_max, n_block_min_causal_local_mask)
                # The remaining iterations have no masking
                n_block_min_before_local_mask = block_info.get_n_block_min_before_local_mask(
                    seqlen, work_m, n_block_min
                )
                for n_tile in cutlass.range(n_block_max - n_block_min_before_local_mask, unroll=1):
                    n_block = n_block_max - n_tile - 1
                    s_p_o_consumer_count, sm_stats_producer_phase, s0_s1_sequence_phase = softmax_step(
                        s_p_o_consumer_count, sm_stats_producer_phase, s0_s1_sequence_phase, n_block,
                    )

                # Dense path always writes scale / signals
                sm_stats_index = stage
                sScale[tidx + sm_stats_index * self.m_block_size] = softmax.row_sum[0]
                sScale[
                    tidx + sm_stats_index * self.m_block_size + self.s_p_o_stage * self.m_block_size
                ] = softmax.row_max[0]
                sm_stats_barrier.arrive_w_index(index=sm_stats_index * 4 + warp_idx)

            # Advance to next tile
            work_tile = tile_scheduler.advance_to_next_work()
        # End of persistent scheduler loop

        # This is equivalent to pipeline_sm_stats.producer_tail
        pipeline_sm_stats.producer_acquire_w_index_phase(stage, sm_stats_producer_phase)

    # ------------------------------------------------------------------------------------------------------------------
    # fwdmufu slice=1: slice softmax (warps 0-15) + row-level correction (warps 16-19). See __init__ and
    # Every value is computed by today's op sequence (bitwise O / LSE / P codes).
    # ------------------------------------------------------------------------------------------------------------------
    @cute.jit
    def slice_softmax_loop(
        self,
        thr_mma_qk: cute.ThrMma,
        tStS: cute.Tensor,
        descale_tensors: DescaleTensors,
        pipeline_s_p_o: pipeline.PipelineAsync,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        tile_scheduler=None,
    ):
        tidx = cute.arch.thread_idx()[0] % (4 * cute.arch.WARP_SIZE)   # row; warp % 4 = TMEM lane quadrant
        q = cute.arch.make_warp_uniform(cute.arch.warp_idx()) // 4     # key block owned by this slice warpgroup
        tSAcc = tStS[(None, None), 0, 0, 0]
        tSblk = cute.make_tensor(tSAcc.iterator, cute.composition(tSAcc.layout, cute.make_layout((self.m_block_size, 32))))
        thr_ld = tcgen05.make_tmem_copy(
            cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.qk_acc_dtype), tSblk
        ).get_slice(tidx)
        tSt_ld = thr_ld.partition_S(tSblk)
        rS = cute.make_rmem_tensor(thr_ld.partition_D(cute.make_identity_tensor((self.m_block_size, 32))).shape, Float32)
        tSP = cute.make_tensor(tSAcc.iterator, cute.composition(tSAcc.layout, cute.make_layout((self.m_block_size, 8))))
        thr_st = tcgen05.make_tmem_copy(
            cute.make_copy_atom(tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(8)), Float32), tSP
        ).get_slice(tidx)
        tSt_st = thr_st.partition_D(tSP)
        rP = cute.make_rmem_tensor(thr_st.partition_S(cute.make_identity_tensor((self.m_block_size, 8))).shape, Float32)
        rP8 = cute.make_tensor(cute.recast_ptr(rP.iterator, dtype=self.q_dtype), rS.layout)
        st = cute.make_rmem_tensor((3,), Int32)   # [0] s_p_o full phase, [1] steps done per stage, [2] prefetched K scale word
        st[0] = Int32(0)
        st[1] = Int32(0)
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            work_m, head_idx, batch_idx, split_idx = work_tile.tile_idx
            kv_head_idx = head_idx // self.qhead_per_kvhead
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, work_m, split_idx=split_idx, num_splits=num_splits,
            )
            has_work = self.process_work_tile(block_info, seqlen, n_block_min, n_block_max)
            qsc = [None] * 2
            for s in cutlass.range_constexpr(2):
                dm_row = self._q_block(work_m, s, thr_mma_qk.thr_idx) * self.m_block_size + tidx
                dm_qblk = cutlass.min(dm_row // 32, (seqlen.seqlen_q + 31) // 32 - 1)
                qsc[s] = Float32(descale_tensors.q_descale[batch_idx, head_idx, dm_qblk]) * Float32(math.log2(math.e))
            step = partial(self._slice_step, q=q, tidx=tidx, st=st, qsc=qsc, rS=rS, rP=rP, rP8=rP8, tSt_ld=tSt_ld,
                           tSt_st=tSt_st, thr_ld=thr_ld, thr_st=thr_st, pipeline_s_p_o=pipeline_s_p_o,
                           descale_tensors=descale_tensors, batch_idx=batch_idx, kv_head_idx=kv_head_idx,
                           work_m=work_m, seqlen=seqlen)
            if has_work:
                st[2] = Int32(descale_tensors.k_descale[batch_idx, kv_head_idx, 2 * cutlass.max(n_block_max - 1, 0)])
                step(n_block_max - 1, mask_seqlen=True, mask_causal=self.is_causal)
                n_block_max -= 1
                if const_expr(self.is_causal):
                    n_block_min_causal_local_mask = block_info.get_n_block_min_causal_local_mask(
                        seqlen, work_m, n_block_min
                    )
                    for n_tile in cutlass.range(n_block_max - n_block_min_causal_local_mask, unroll=1):
                        step(n_block_max - 1 - n_tile, mask_seqlen=False, mask_causal=True)
                    n_block_max = cutlass.min(n_block_max, n_block_min_causal_local_mask)
                for n_tile in cutlass.range(n_block_max - n_block_min, unroll=1):
                    step(n_block_max - 1 - n_tile, mask_seqlen=False, mask_causal=False)
            work_tile = tile_scheduler.advance_to_next_work()

    @cute.jit
    def _slice_step(
        self, n_block: Int32, mask_seqlen: cutlass.Constexpr, mask_causal: cutlass.Constexpr, q: Int32, tidx: Int32,
        st: cute.Tensor, qsc, rS: cute.Tensor, rP: cute.Tensor, rP8: cute.Tensor, tSt_ld: cute.Tensor,
        tSt_st: cute.Tensor, thr_ld, thr_st, pipeline_s_p_o, descale_tensors,
        batch_idx: Int32, kv_head_idx: Int32, work_m: Int32, seqlen,
    ):
        """One n_block for both Q stages (stage 0 then stage 1) of this slice warp's key block q."""
        dm_kw = st[2]
        # next n_block's K scale word (loaded now, consumed next step; clamped index on the last step)
        st[2] = Int32(descale_tensors.k_descale[batch_idx, kv_head_idx, 2 * cutlass.max(n_block - 1, 0)])
        dm_ks = byte_pow2_f32_rt(dm_kw, q * 8)
        ph = st[0]
        nstep = st[1]
        for s in cutlass.range_constexpr(2):
            c = qsc[s] * dm_ks
            pipeline_s_p_o.consumer_wait_w_index_phase(s, ph)
            col = Int32(self.tmem_s_offset[s]) + q * 32
            cute.copy(thr_ld, cute.make_tensor(tSt_ld.iterator + col, tSt_ld.layout), rS)
            if const_expr(mask_seqlen or mask_causal):
                # = AttentionMask.apply_mask_sm100 (non-pack-GQA, 1-CTA) restricted to columns [32 q, 32 q + 32)
                n_blk = cutlass.max(n_block, 0)   # apply_mask_sm100 treats negative n_blocks as block 0 (empty KV range)
                seqlenk_col_limit = seqlen.seqlen_k - n_blk * self.n_block_size
                if const_expr(mask_causal):
                    causal_row_offset = seqlen.seqlen_k - n_blk * self.n_block_size - seqlen.seqlen_q
                    row_idx = tidx + self._q_block(work_m, s) * self.m_block_size
                    col_limit = row_idx + causal_row_offset + 1
                    if const_expr(mask_seqlen):
                        col_limit = cutlass.min(col_limit, seqlenk_col_limit)
                else:
                    col_limit = seqlenk_col_limit
                lim = col_limit - q * 32
                mask_r2p_lambda(rS, lambda ch: r2p_bitmask_below(lim, ch), rank1=True)
            # block max: today's FMNMX order, then x_b = max * c_b
            lm = [utils.fmax(rS[3 * t], rS[3 * t + 1], rS[3 * t + 2]) for t in range(4)]
            for i in cutlass.range_constexpr(12, 32, 2):
                t = ((i - 12) // 2) % 4
                lm[t] = utils.fmax(lm[t], rS[i], rS[i + 1])
            v = utils.fmax(utils.fmax(lm[0], lm[1], lm[2]), lm[3])
            xb = v * c
            st_shared_b32(self.dm_slbuf + 4 * ((s * 4 + q) * self.m_block_size + tidx), bitcast_f32_i32(xb))
            # The correction WG's SF tcgen05.st overwrites S_s cols 16..23 that slice WG 0 loaded; PTX
            # 9.7.18.6.4.4/.5: a cross-thread ld -> st hand-off needs tcgen05.wait::ld before the fence + arrive
            # (register use of the loaded values is not a memory ordering)
            cute.arch.fence_view_async_tmem_load()
            tcgen05_fence_before()     # this warp's S loads ordered before the kb-published arrival (SF overwrites S)
            fence_warp_relay()   # fwdfence v3: every lane fences + warp sync before the elected leader's release
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                mbar_arrive(self.dm_slbar + 8 * s)
            kb = cute.arch.fmax(floor_f32(xb), Float32(-1e30))
            bias = Float32(7.0) - kb
            for i in cutlass.range_constexpr(0, 32, 2):
                rS[i], rS[i + 1] = cute.arch.fma_packed_f32x2((rS[i], rS[i + 1]), (c, c), (bias, bias))
            for i in cutlass.range_constexpr(0, 32, 2):
                rS[i] = cute.math.exp2(rS[i], fastmath=True)
                rS[i + 1] = cute.math.exp2(rS[i + 1], fastmath=True)
            rP8.store(rS.load().to(self.q_dtype))
            cute.copy(thr_st, rP, cute.make_tensor(tSt_st.iterator + col, tSt_st.layout))
            cute.arch.fence_view_async_tmem_store()
            tcgen05_fence_before()     # P tcgen05.st ordered before the P-full arrival (PV reads it)
            pipeline_s_p_o.consumer_release_w_index(s)
            ps = (rS[0], rS[1])
            for i in cutlass.range_constexpr(2, 32, 2):
                ps = cute.arch.add_packed_f32x2(ps, (rS[i], rS[i + 1]))
            bsum = ps[0] + ps[1]
            if nstep > 0:
                mbar_wait_parity(self.dm_slbar + 32 + 8 * s, (nstep - 1) & 1)
            st_shared_b32(self.dm_slbuf + 4 * (1024 + (s * 4 + q) * self.m_block_size + tidx),
                                   bitcast_f32_i32(bsum))
            fence_warp_relay()   # fwdfence v3: every lane fences + warp sync before the elected leader's release
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                mbar_arrive(self.dm_slbar + 16 + 8 * s)
        st[0] = ph ^ 1
        st[1] = nstep + 1

    @cute.jit
    def slice_correction_loop(
        self,
        thr_mma_qk: cute.ThrMma,
        thr_mma_pv: cute.ThrMma,
        tStS: cute.Tensor,
        tOtO: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        sO: cute.Tensor,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_o_acc: pipeline.PipelineAsync,
        pipeline_o_epi: pipeline.PipelineAsync,
        descale_tensors: DescaleTensors,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        tile_scheduler=None,
    ):
        tidx = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.correction_warp_ids))
        mma_tile_coord_v = thr_mma_qk.thr_idx
        tSAcc = tStS[(None, None), 0, 0, 0]
        tSF = cute.make_tensor(tSAcc.iterator, cute.composition(tSAcc.layout, cute.make_layout((self.m_block_size, 8))))
        thr_sf = tcgen05.make_tmem_copy(
            cute.make_copy_atom(tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(8)), Float32), tSF
        ).get_slice(tidx)
        tSt_sf = thr_sf.partition_D(tSF)
        rSF = cute.make_rmem_tensor(thr_sf.partition_S(cute.make_identity_tensor((self.m_block_size, 8))).shape, Float32)
        rst = cute.make_rmem_tensor((5,), Float32)    # m0, m1, l0, l1, acc (scratch)
        cnt = cute.make_rmem_tensor((2,), Int32)      # [0] steps done per stage (kb / bs barrier parity), [1] V scale word
        cnt[0] = Int32(0)
        o_corr_consumer_phase = Int32(0)
        corr_epi_producer_phase = Int32(1)
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            work_m, head_idx, batch_idx, split_idx = work_tile.tile_idx
            kv_head_idx = head_idx // self.qhead_per_kvhead
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, work_m, split_idx=split_idx, num_splits=num_splits,
            )
            mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[None, None, head_idx]
            tiler_gO = ((self.mma_tiler_pv[0] * self.q_stage), self.head_dim_v_padded)
            gO = cute.local_tile(mO_cur, tiler_gO, (work_m, 0))
            gO = layout_utils.select(cute.flat_divide(gO, (self.mma_tiler_pv[0],)), mode=[0, 2, 1])
            gO = cute.flat_divide(gO, (self.mma_tiler_pv[0],))[None, mma_tile_coord_v, None, None]
            stats = [(0.0, -Float32.inf, True)] * self.q_stage
            total_block_count = n_block_max - n_block_min
            has_work = self.process_work_tile(block_info, seqlen, n_block_min, n_block_max)
            if has_work:
                cnt[1] = Int32(descale_tensors.k_descale[batch_idx, kv_head_idx, 2 * cutlass.max(n_block_max - 1, 0) + 1])
                # one step even when the range is empty (seqlen_k == 0): slices and MMA always run the first step
                for k in cutlass.range(cutlass.max(total_block_count, 1), unroll=1):
                    n_block = n_block_max - 1 - k
                    sfb_word = bitcast_i32_f32(cnt[1])   # V scale word of n_block (prefetched one step ahead)
                    cnt[1] = Int32(descale_tensors.k_descale[batch_idx, kv_head_idx, 2 * cutlass.max(n_block - 1, 0) + 1])
                    nstep = cnt[0]
                    for s in cutlass.range_constexpr(2):
                        # ---- row max / rescale / SF bytes (today's softmax_step prefix, verbatim ops) ----
                        mbar_wait_parity(self.dm_slbar + 8 * s, nstep & 1)
                        tcgen05_fence_after()
                        x = [bitcast_i32_f32(ld_shared_b32(
                            self.dm_slbuf + 4 * ((s * 4 + j) * self.m_block_size + tidx))) for j in range(4)]
                        dm_xmax = utils.fmax(utils.fmax(x[0], x[1], x[2]), x[3])
                        dm_mcand = ceil_f32(dm_xmax)
                        m_old = rst[s]
                        row_max = m_old
                        acc_scale = Float32(1.0)
                        if k == 0:
                            row_max = dm_mcand
                            acc_scale = Float32(0.0)
                        else:
                            if dm_mcand > m_old + Float32(32.0):   # rescale threshold (log2 units)
                                row_max = dm_mcand
                                acc_scale = exp2i_f32(m_old - dm_mcand)
                        rst[s] = row_max
                        m_safe = row_max if row_max != -Float32.inf else Float32(0.0)
                        dm_w = [None] * 4
                        dm_sfa = [None] * 4
                        for j in cutlass.range_constexpr(4):
                            kb = cute.arch.fmax(floor_f32(x[j]), Float32(-1e30))
                            e = kb - Float32(7.0) - m_safe
                            dm_sfa[j] = ue8m0_byte(e)
                            dm_w[j] = byte_shl23_f32(dm_sfa[j])
                        sfa_word = bitcast_i32_f32(pack_bytes4(dm_sfa[0], dm_sfa[1], dm_sfa[2], dm_sfa[3]))
                        for j in cutlass.range_constexpr(4):
                            rSF[j] = sfa_word
                            rSF[4 + j] = sfb_word
                        cute.copy(thr_sf, rSF, cute.make_tensor(
                            tSt_sf.iterator + (self.tmem_s_offset[s] + self.dm_sl_sfcol), tSt_sf.layout))
                        if k > 0:
                            # S_s(k) complete (seen through the slices' kb arrivals) => PV_s(k-1) complete
                            should_rescale = cute.arch.vote_ballot_sync(acc_scale < 1.0) != 0
                            if should_rescale:
                                self.correction_rescale(thr_mma_pv, tOtO[None, None, None, s], tidx, acc_scale)
                        cute.arch.fence_view_async_tmem_store()
                        cute.arch.fence_view_async_tmem_load()   # (k == 0) previous tile's epilogue O loads done
                        tcgen05_fence_before()
                        pipeline_s_p_o.consumer_release_w_index(s)    # SF written + O rescaled (+ O free at k == 0)
                        # ---- l = l * acc + sum_b s_b w_b (today's order) ----
                        mbar_wait_parity(self.dm_slbar + 16 + 8 * s, nstep & 1)
                        bs = [bitcast_i32_f32(ld_shared_b32(
                            self.dm_slbuf + 4 * (1024 + (s * 4 + j) * self.m_block_size + tidx))) for j in range(4)]
                        fence_warp_relay()   # fwdfence v3: every lane fences + warp sync before the elected leader's release
                        cute.arch.sync_warp()
                        with cute.arch.elect_one():
                            mbar_arrive(self.dm_slbar + 32 + 8 * s)
                        dm_l = rst[2 + s] * acc_scale
                        if k == 0:
                            dm_l = Float32(0.0)
                        for j in cutlass.range_constexpr(4):
                            dm_l = dm_l + bs[j] * dm_w[j]
                        rst[2 + s] = dm_l
                    cnt[0] = nstep + 1
                # ---- epilogue (= correction_loop, dense non-split path) ----
                for stage in cutlass.range_constexpr(self.q_stage):
                    row_sum = rst[2 + stage]
                    row_max = rst[stage]
                    acc_O_mn_row_is_zero_or_nan = row_sum == 0.0 or row_sum != row_sum
                    stats[stage] = (row_sum, row_max, acc_O_mn_row_is_zero_or_nan)
                    scale = cute.arch.rcp_approx(row_sum if not acc_O_mn_row_is_zero_or_nan else 1.0)
                    scale = scale * Float32(1.0)
                    pipeline_o_acc.consumer_wait_w_index_phase(stage, o_corr_consumer_phase)
                    tcgen05_fence_after()
                    pipeline_o_epi.producer_acquire_w_index_phase(stage, corr_epi_producer_phase)
                    self.correction_epilogue(
                        thr_mma_pv,
                        tOtO[None, None, None, stage],
                        tidx,
                        stage,
                        work_m,
                        seqlen.seqlen_q,
                        scale,
                        sO[None, None, stage],
                        mO_cur,
                        gO[None, None, stage],
                        zero_fill=seqlen.seqlen_k == 0,
                    )
                    pipeline_o_epi.producer_commit_w_index(stage)
                o_corr_consumer_phase ^= 1
                corr_epi_producer_phase ^= 1
            mLSE_cur = mLSE[None, head_idx, batch_idx]
            for stage in cutlass.range_constexpr(self.q_stage):
                q_block = self._q_block(work_m, stage, mma_tile_coord_v)
                row_sum, row_max, acc_O_mn_row_is_zero_or_nan = stats[stage]
                LN2 = math.log(2.0)
                lse = (
                    (row_max * Float32(1.0) + (cute.math.log2(row_sum, fastmath=True) - Float32(0.0))) * LN2
                    if not acc_O_mn_row_is_zero_or_nan
                    else -Float32.inf
                )
                gLSE = cute.local_tile(mLSE_cur, (self.m_block_size,), (q_block,))
                if tidx < seqlen.seqlen_q - q_block * self.m_block_size:
                    gLSE[tidx] = lse
            work_tile = tile_scheduler.advance_to_next_work()
        pipeline_o_epi.producer_acquire_w_index_phase(self.q_stage - 1, corr_epi_producer_phase)

    @cute.jit
    def softmax_step(
        self,
        s_p_o_consumer_count: Int32,
        sm_stats_producer_phase: Int32,
        s0_s1_sequence_phase: Int32,
        n_block: Int32,
        softmax: SoftmaxSm100,
        thr_mma_qk: cute.ThrMma,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_p_lastsplit: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        sm_stats_barrier: pipeline.NamedBarrier,
        thr_tmem_load: cute.CopyAtom,
        thr_tmem_store: cute.CopyAtom,
        thr_tmem_store_scale: cute.CopyAtom,
        tStS_t2r: cute.Tensor,
        tStScale_r2t: cute.Tensor,
        tStP_r2t: cute.Tensor,
        sScale: cute.Tensor,
        thr_tmem_store_sf: cute.CopyAtom,
        tStSF_r2t: cute.Tensor,
        dm_qsc: Float32,
        dm_scales,
        kv_head_idx: Int32,
        stage: int | Int32,
        batch_idx: Int32,
        head_idx: Int32,
        m_block: Int32,
        seqlen,
        mask_fn: Optional[Callable] = None,
        is_first: bool = False,
    ) -> Tuple[cute.Int32, cute.Int32, cute.Int32]:
        """Perform a single step of the softmax computation on a block of attention scores.

        This method processes one block of the attention matrix, computing numerically stable
        softmax by first finding the row maximum, subtracting it from all elements, applying
        exponential function, and then normalizing by the sum of exponentials. It also handles
        optional masking of attention scores.

        The method involves several key operations:
        1. Loading attention scores from tensor memory
        2. Applying optional masking based on position
        3. Computing row-wise maximum values for numerical stability
        4. Transforming scores using exp2(x*scale - max*scale)
        5. Computing row sums for normalization
        6. Coordinating pipeline synchronization between different processing stages
        """
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        tilePlikeFP32 = self.mma_tiler_qk[1] // Float32.width * self.v_dtype.width
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tScS = tScS[(None, None), 0, 0]  # (128, 128)
        cta_qk_tiler = (self.mma_tiler_qk[0] // thr_mma_qk.thr_id.shape, self.mma_tiler_qk[1])
        tScS_shape = cta_qk_tiler  # (128, 128)
        tScP_shape = (tScS_shape[0], tilePlikeFP32)  # (128, 64)

        # DM: issue this tile's K/V block-scale loads before waiting for S (hides the L2 latency behind the QK MMA)
        # packed UE8M0 bytes: k_descale[b, hk, 2*n_block] = K scale bytes, [.., 2*n_block+1] = V scale bytes (= SFB word)
        dm_kw = Int32(dm_scales.k_descale[batch_idx, kv_head_idx, 2 * n_block])
        dm_vw = Int32(dm_scales.k_descale[batch_idx, kv_head_idx, 2 * n_block + 1])
        dm_ks = [byte_pow2_f32(dm_kw, j) for j in range(4)]
        # Wait for Si
        s_p_o_index = stage
        pipeline_s_p_o.consumer_wait_w_index_phase(stage, s_p_o_consumer_count)
        tSrS_t2r = cute.make_rmem_tensor(thr_tmem_load.partition_D(tScS).shape, self.qk_acc_dtype)
        cute.copy(thr_tmem_load, tStS_t2r, tSrS_t2r)
        # formal tcgen05.wait::ld of S before the P release lets the next QK overwrite S (P aliases S)
        cute.arch.fence_view_async_tmem_load()

        if const_expr(mask_fn is not None):
            mask_fn(tSrS_t2r, n_block=n_block)
        # ---------------- DM sm100-mx softmax step ----------------
        # x = S * c_b (log2 units), c_b = qs_row * log2e * ks_b; per 32-key block b: kb = floor(max_b x),
        # p~ = exp2(x - kb + 7) in [0, 256) -> e4m3 code; SFA_b = 2^(kb - 7 - m), SFB_b = vs_b; m = integer running max.
        dm_c = [dm_qsc * dm_ks[j] for j in range(4)]
        dm_bmax = [None] * 4
        for j in cutlass.range_constexpr(4):
            b0 = 32 * j
            lm = [utils.fmax(tSrS_t2r[b0], tSrS_t2r[b0 + 1]), utils.fmax(tSrS_t2r[b0 + 2], tSrS_t2r[b0 + 3]),
                  utils.fmax(tSrS_t2r[b0 + 4], tSrS_t2r[b0 + 5]), utils.fmax(tSrS_t2r[b0 + 6], tSrS_t2r[b0 + 7])]
            for i in cutlass.range_constexpr(8, 32, 8):
                for t in cutlass.range_constexpr(4):
                    lm[t] = utils.fmax(lm[t], tSrS_t2r[b0 + i + 2 * t], tSrS_t2r[b0 + i + 2 * t + 1])
            v = utils.fmax(utils.fmax(lm[0], lm[1]), lm[2], lm[3])
            dm_bmax[j] = v * dm_c[j]
        dm_xmax = cute.arch.fmax(cute.arch.fmax(dm_bmax[0], dm_bmax[1]), cute.arch.fmax(dm_bmax[2], dm_bmax[3]))
        dm_mcand = ceil_f32(dm_xmax)
        acc_scale = Float32(1.0)
        row_max = dm_mcand
        if const_expr(is_first):
            row_max = dm_mcand
            acc_scale = Float32(0.0)
        else:
            m_old = softmax.row_max[0]
            row_max = m_old
            if dm_mcand > m_old + Float32(32.0):   # rescale threshold (log2 units)
                row_max = dm_mcand
                acc_scale = exp2i_f32(m_old - dm_mcand)
        softmax.row_max[0] = row_max
        m_safe = row_max if row_max != -Float32.inf else Float32(0.0)
        # fully masked blocks: kb clamps to -1e30 -> weight 0, SF byte 0, finite bias (x stays -inf -> p 0)
        dm_bias = [None] * 4
        dm_w = [None] * 4
        dm_sfa = [None] * 4
        for j in cutlass.range_constexpr(4):
            kb = cute.arch.fmax(floor_f32(dm_bmax[j]), Float32(-1e30))
            e = kb - Float32(7.0) - m_safe
            dm_bias[j] = Float32(7.0) - kb
            dm_w[j] = exp2i_f32(e)
            dm_sfa[j] = ue8m0_byte(e)
        sfa_word = bitcast_i32_f32(pack_bytes4(dm_sfa[0], dm_sfa[1], dm_sfa[2], dm_sfa[3]))
        sfb_word = bitcast_i32_f32(dm_vw)
        tSrSF = cute.make_rmem_tensor(thr_tmem_store_sf.partition_S(cute.make_identity_tensor((self.m_block_size, 8))).shape, Float32)
        for j in cutlass.range_constexpr(4):
            tSrSF[j] = sfa_word
            tSrSF[4 + j] = sfb_word
        cute.copy(thr_tmem_store_sf, tSrSF, tStSF_r2t)

        sm_stats_index = s_p_o_index
        if const_expr(not is_first):
            thread_idx = thr_tmem_load.thr_idx
            sScale[thread_idx + sm_stats_index * self.m_block_size] = acc_scale
        sm_stats_barrier.arrive_w_index(index=sm_stats_index * 4 + warp_idx)

        # x = S * c_b + (7 - kb)
        for j in cutlass.range_constexpr(4):
            for i in cutlass.range_constexpr(0, 32, 2):
                tSrS_t2r[32 * j + i], tSrS_t2r[32 * j + i + 1] = cute.arch.fma_packed_f32x2(
                    (tSrS_t2r[32 * j + i], tSrS_t2r[32 * j + i + 1]),
                    (dm_c[j], dm_c[j]),
                    (dm_bias[j], dm_bias[j]),
                )
        tSrP_r2t_f32 = cute.make_rmem_tensor(
            thr_tmem_store.partition_S(cute.make_identity_tensor(tScP_shape)).shape, Float32
        )
        tSrP_r2t = cute.make_tensor(
            cute.recast_ptr(tSrP_r2t_f32.iterator, dtype=self.q_dtype), tSrS_t2r.layout
        )
        softmax.apply_exp2_convert(
            tSrS_t2r,
            tSrP_r2t,
            ex2_emu_freq=0,
            ex2_emu_start_frg=1,
        )
        for i in cutlass.range_constexpr(cute.size(tStP_r2t.shape[2])):
            cute.copy(thr_tmem_store, tSrP_r2t_f32[None, None, i], tStP_r2t[None, None, i])
            if const_expr(self.split_P_arrive > 0):
                split_P_arrive_idx = cute.size(tStP_r2t.shape[2]) * self.split_P_arrive // self.n_block_size
                if const_expr(i + 1 == split_P_arrive_idx):
                    cute.arch.fence_view_async_tmem_store()
                    pipeline_s_p_o.consumer_release_w_index(s_p_o_index)
        cute.arch.fence_view_async_tmem_store()
        tcgen05_fence_before()   # SF tcgen05.st ordered before the P-full release
        if const_expr(self.split_P_arrive > 0):
            cute.arch.sync_warp()
            fence_warp_relay()   # all lanes fence + warp sync before the leader's arrive
            with cute.arch.elect_one():
                pipeline_p_lastsplit.producer_commit_w_index(s_p_o_index)
        else:
            pipeline_s_p_o.consumer_release_w_index(s_p_o_index)
        pipeline_sm_stats.producer_acquire_w_index_phase(stage, sm_stats_producer_phase)
        # l = l * acc_scale + sum_b w_b * sum_{j in b} p~_j   (fp32 pre-rounding values)
        dm_l = Float32(0.0) if const_expr(is_first) else softmax.row_sum[0] * acc_scale
        for j in cutlass.range_constexpr(4):
            ps = (tSrS_t2r[32 * j], tSrS_t2r[32 * j + 1])
            for i in cutlass.range_constexpr(2, 32, 2):
                ps = cute.arch.add_packed_f32x2(ps, (tSrS_t2r[32 * j + i], tSrS_t2r[32 * j + i + 1]))
            dm_l = dm_l + (ps[0] + ps[1]) * dm_w[j]
        softmax.row_sum[0] = dm_l
        return s_p_o_consumer_count ^ 1, sm_stats_producer_phase ^ 1, s0_s1_sequence_phase ^ 1

    @cute.jit
    def correction_loop(
        self,
        thr_mma_qk: cute.ThrMma,
        thr_mma_pv: cute.ThrMma,
        tStS: cute.Tensor,
        tOtO: cute.Tensor,
        sScale: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        sO: cute.Tensor,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_o_acc: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        sm_stats_barrier: pipeline.NamedBarrier,
        pipeline_o_epi: pipeline.PipelineAsync,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        tile_scheduler=None,
    ):
        assert self.q_stage == 1, "the q_stage-2 forward uses slice_correction_loop"
        tidx = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.correction_warp_ids))
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        mma_tile_coord_v = thr_mma_qk.thr_idx

        # First iter: no correction is required
        # Notify mma warp that O has been rescaled
        for stage in cutlass.range(self.q_stage):
            pipeline_s_p_o.consumer_release_w_index(stage)

        sm_stats_consumer_phase = Int32(0)
        o_corr_consumer_phase = Int32(0)
        corr_epi_producer_phase = Int32(1)

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            work_m, head_idx, batch_idx, split_idx = work_tile.tile_idx
            # DM sm100-mx: row_max is in log2 units, row_sum carries the SFA weights, V scales live in SFB.
            v_descale = Float32(1.0)
            softmax_scale_log2_eff = Float32(1.0)
            max_offset = Float32(0.0)
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, work_m, split_idx=split_idx, num_splits=num_splits,
            )
            mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[None, None, head_idx]
            tiler_gO = ((self.mma_tiler_pv[0] * self.q_stage), self.head_dim_v_padded)
            gO = cute.local_tile(mO_cur, tiler_gO, (work_m, 0))  # (128 * 2, 128)
            gO = layout_utils.select(
                cute.flat_divide(gO, (self.mma_tiler_pv[0],)), mode=[0, 2, 1]
            )  # (128, 128, 2)
            gO = cute.flat_divide(gO, (self.mma_tiler_pv[0],))[None, mma_tile_coord_v, None, None]

            # Default LSE to -inf for tiles without work
            stats = [(0.0, -Float32.inf, True)] * self.q_stage

            total_block_count = n_block_max - n_block_min
            has_work = self.process_work_tile(block_info, seqlen, n_block_min, n_block_max)

            if has_work:
                # Ignore first signal from softmax as no correction is required
                sm_stats_barrier.arrive_and_wait_w_index(index=0 * 4 + warp_idx)
                pipeline_sm_stats.consumer_release_w_index(0)
                sm_stats_consumer_phase ^= 1

                for i in cutlass.range(total_block_count - 1, unroll=1):
                    for stage in cutlass.range_constexpr(self.q_stage):
                        sm_stats_index = stage
                        # wait for S0 / S1
                        sm_stats_barrier.arrive_and_wait_w_index(index=sm_stats_index * 4 + warp_idx)
                        scale = sScale[tidx + sm_stats_index * self.m_block_size]
                        should_rescale = cute.arch.vote_ballot_sync(scale < 1.0) != 0
                        # Don't need O_full anymore, since by the time softmax has signaled the correction
                        # warps, S_i must have been done, so O_i-1 must have been done as well.
                        if should_rescale:
                            self.correction_rescale(thr_mma_pv, tOtO[None, None, None, stage], tidx, scale)
                        # Notify mma warp that O has been rescaled
                        cute.arch.fence_view_async_tmem_load()   # O rescale loads complete (formal)
                        tcgen05_fence_before()   # O rescale tcgen05.st ordered before the release
                        pipeline_s_p_o.consumer_release_w_index(stage)
                        pipeline_sm_stats.consumer_release_w_index(self.q_stage - 1 - stage)
                    sm_stats_consumer_phase ^= 1
                # End of seqlen_corr_loop_steps

                for stage in cutlass.range_constexpr(self.q_stage):
                    sm_stats_index = stage
                    sm_stats_barrier.arrive_and_wait_w_index(index=sm_stats_index * 4 + warp_idx)
                    row_sum = sScale[tidx + sm_stats_index * self.m_block_size]
                    row_max = sScale[tidx + sm_stats_index * self.m_block_size + self.s_p_o_stage * self.m_block_size]
                    pipeline_sm_stats.consumer_release_w_index(sm_stats_index)
                    acc_O_mn_row_is_zero_or_nan = row_sum == 0.0 or row_sum != row_sum
                    stats[stage] = (row_sum, row_max, acc_O_mn_row_is_zero_or_nan)
                    scale = cute.arch.rcp_approx(row_sum if not acc_O_mn_row_is_zero_or_nan else 1.0)
                    scale = scale * v_descale
                    # Wait for the last O to be ready from the MMA warp
                    pipeline_o_acc.consumer_wait_w_index_phase(stage, o_corr_consumer_phase)
                    pipeline_o_epi.producer_acquire_w_index_phase(stage, corr_epi_producer_phase)
                    self.correction_epilogue(
                        thr_mma_pv,
                        tOtO[None, None, None, stage],
                        tidx,
                        stage,
                        work_m,
                        seqlen.seqlen_q,
                        scale,
                        sO[None, None, stage],
                        mO_cur,
                        gO[None, None, stage],
                        # TMA KV loads a dummy tile when seqlen_k == 0 (see correction_epilogue)
                        zero_fill=seqlen.seqlen_k == 0,
                    )
                    # Signal for the next work tile that O buffers in tmem are already read, so
                    # mma warp can write to them: epilogue O tcgen05.ld complete + ordered before the release
                    # that lets the next work tile's PV_s (zero-init) overwrite O_s
                    cute.arch.fence_view_async_tmem_load()
                    tcgen05_fence_before()
                    pipeline_s_p_o.consumer_release_w_index(stage)
                    pipeline_o_epi.producer_commit_w_index(stage)

                o_corr_consumer_phase ^= 1
                sm_stats_consumer_phase ^= 1
                corr_epi_producer_phase ^= 1

            mLSE_cur = mLSE[None, head_idx, batch_idx]
            for stage in cutlass.range_constexpr(self.q_stage):
                q_block = self._q_block(work_m, stage, mma_tile_coord_v)
                row_sum, row_max, acc_O_mn_row_is_zero_or_nan = stats[stage]
                LN2 = math.log(2.0)
                lse = (
                    (row_max * softmax_scale_log2_eff + (cute.math.log2(row_sum, fastmath=True) - max_offset)) * LN2
                    if not acc_O_mn_row_is_zero_or_nan
                    else -Float32.inf
                )
                seqlen_q = seqlen.seqlen_q
                gLSE = cute.local_tile(mLSE_cur, (self.m_block_size,), (q_block,))
                if tidx < seqlen_q - q_block * self.m_block_size:
                    gLSE[tidx] = lse

            # Advance to next tile
            work_tile = tile_scheduler.advance_to_next_work()
        # End of persistent scheduler loop

        # This is equivalent to pipeline_o_epi.consumer_tail() for the correction warps
        pipeline_o_epi.producer_acquire_w_index_phase(self.q_stage - 1, corr_epi_producer_phase)

    @cute.jit
    def correction_rescale(
        self,
        thr_mma: cute.ThrMma,
        tOtO: cute.Tensor,
        tidx: Int32,
        scale: Float32,
    ):
        """Rescale intermediate attention results based on softmax normalization factor.

        This method performs a crucial correction step in the attention computation pipeline.
        When processing attention in blocks, the softmax normalization factors may change
        as new blocks are processed. This method rescales previously computed partial
        output values to account for updated normalization factors.

        The implementation uses efficient tensor memory operations to:
        1. Load existing partial attention output from tensor memory
        2. Apply the scaling factor to all elements
        3. Store the rescaled results back to tensor memory
        """
        tOcO = thr_mma.partition_C(cute.make_identity_tensor(self.mma_tiler_pv[:2]))
        corr_tile_size = 16  # tuneable parameter
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(corr_tile_size)), self.pv_acc_dtype
        )
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.pv_acc_dtype,
        )
        tOtO_i = cute.composition(tOtO, cute.make_layout((self.m_block_size, corr_tile_size)))
        tOcO_i = cute.composition(tOcO, cute.make_layout((self.m_block_size, corr_tile_size)))
        thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tOtO_i).get_slice(tidx)
        thr_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tOtO_i).get_slice(tidx)
        tOtO_t2r = thr_tmem_load.partition_S(tOtO_i)
        tOrO_t2r_shape = thr_tmem_load.partition_D(tOcO_i).shape
        tOtO_r2t = thr_tmem_store.partition_D(tOtO_i)

        frg_count = self.head_dim_v_padded // corr_tile_size
        tOrO_frg = cute.make_rmem_tensor((tOrO_t2r_shape, frg_count), self.pv_acc_dtype)
        for i in cutlass.range_constexpr(frg_count):
            tOrO_frg = cute.make_rmem_tensor(tOrO_t2r_shape, self.pv_acc_dtype)
            tOtO_t2r_i = cute.make_tensor(tOtO_t2r.iterator + i * corr_tile_size, tOtO_t2r.layout)
            cute.copy(thr_tmem_load, tOtO_t2r_i, tOrO_frg)
            for j in cutlass.range(0, cute.size(tOrO_frg), 2, unroll_full=True):
                tOrO_frg[j], tOrO_frg[j + 1] = cute.arch.mul_packed_f32x2(
                    (tOrO_frg[j], tOrO_frg[j + 1]), (scale, scale)
                )
            tOtO_r2t_i = cute.make_tensor(tOtO_r2t.iterator + i * corr_tile_size, tOtO_r2t.layout)
            cute.copy(thr_tmem_store, tOrO_frg, tOtO_r2t_i)
        cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def correction_epilogue(
        self,
        thr_mma: cute.ThrMma,
        tOtO: cute.Tensor,
        tidx: Int32,
        stage: Int32,
        work_m: Int32,
        seqlen_q: Int32,
        scale: Float32,
        sO: cute.Tensor,
        mO_cur: Optional[cute.Tensor] = None,
        gO: Optional[cute.Tensor] = None,
        zero_fill: bool | Boolean = False,
    ):
        """Apply final scaling and transformation to attention output before writing to global memory.

        This correction_epilogue function handles the final processing step for attention output values.
        It applies a scaling factor to the accumulated attention results and prepares the
        data for efficient transfer back to global memory.

        The method performs:
        1. Loading of accumulated attention results from tensor memory
        2. Application of the final output scaling factor
        3. Type conversion if necessary (typically from higher precision accumulator to output precision)
        4. Reorganization of data for optimal memory access patterns
        5. Preparation for efficient TMA store operations

        :param thr_mma: Thread MMA operation for the computation
        :type thr_mma: cute.ThrMma
        :param tOtO: Tensor containing accumulated attention output
        :type tOtO: cute.Tensor
        :param scale: Final scaling factor to apply to the output
        :type scale: Float32
        :param sO: Shared memory tensor for the final output
        :type sO: cute.Tensor
        """

        corr_tile_size = 8 * 32 // self.o_dtype.width
        # Use CTA 0 mapping for smem partitioning since sO is per-CTA sized
        tOsO = thr_mma.get_slice(0).partition_C(sO)
        tOcO = thr_mma.partition_C(cute.make_identity_tensor(self.mma_tiler_pv[:2]))

        tOtO_i = cute.logical_divide(tOtO, cute.make_layout((self.m_block_size, corr_tile_size)))
        tOcO_i = cute.logical_divide(tOcO, cute.make_layout((self.m_block_size, corr_tile_size)))
        tOsO_i = cute.logical_divide(tOsO, cute.make_layout((self.m_block_size, corr_tile_size)))

        epi_subtile = (self.epi_tile[0], corr_tile_size)
        tmem_copy_atom = sm100_utils_basic.get_tmem_load_op(
            self.mma_tiler_pv,
            self.o_layout,
            self.o_dtype,
            self.pv_acc_dtype,
            epi_subtile,
            use_2cta_instrs=False,
        )
        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_copy_atom, tOtO_i[(None, None), 0])
        thr_tmem_load = tiled_tmem_load.get_slice(tidx)
        smem_copy_atom = sm100_utils_basic.get_smem_store_op(
            self.o_layout, self.o_dtype, self.pv_acc_dtype, tiled_tmem_load
        )
        tiled_smem_store = cute.make_tiled_copy_D(smem_copy_atom, tiled_tmem_load)

        tOtO_t2r = thr_tmem_load.partition_S(tOtO_i[(None, None), None])
        tOsO_s2r = copy_utils.partition_D_position_independent(thr_tmem_load, tOsO_i[(None, None), None])
        tOcO_t2r = thr_tmem_load.partition_D(tOcO_i[(None, None), None])
        num_frg_per_chunk = self.head_dim_v_padded // corr_tile_size
        for i in cutlass.range(num_frg_per_chunk, unroll_full=True):
            tOtO_t2r_i = tOtO_t2r[None, 0, 0, i]
            tOsO_r2s_i = tOsO_s2r[None, 0, 0, i]
            tOrO_frg = cute.make_rmem_tensor(tOcO_t2r[None, 0, 0, i].shape, self.pv_acc_dtype)
            if zero_fill:
                # Run time (seqlen_k == 0): the fully masked dummy block still loaded
                # a KV tile, and P = 0 times a NaN V row is NaN; scaling by 0 would keep it.
                tOrO_frg.fill(0.0)
            else:
                cute.copy(tiled_tmem_load, tOtO_t2r_i, tOrO_frg)
                for j in cutlass.range(0, cute.size(tOrO_frg), 2, unroll_full=True):
                    tOrO_frg[j], tOrO_frg[j + 1] = cute.arch.mul_packed_f32x2(
                        (tOrO_frg[j], tOrO_frg[j + 1]), (scale, scale)
                    )
            copy_utils.cvt_copy(tiled_smem_store, tOrO_frg, tOsO_r2s_i)
        cute.arch.fence_view_async_shared()

    @cute.jit
    def epilogue_s2g(
        self,
        mO: cute.Tensor,
        sO: cute.Tensor,
        tma_atom_O: cute.CopyAtom,
        pipeline_o_epi: pipeline.PipelineAsync,
        block_info: BlockInfo,
        num_splits: int,
        SeqlenInfoCls: Callable,
        mma_tile_coord_v: Int32 = 0,
        tile_scheduler=None,
    ):
        epi_consumer_phase = Int32(0)
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            work_m, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, work_m, split_idx=split_idx, num_splits=num_splits,
            )
            if self.process_work_tile(block_info, seqlen, n_block_min, n_block_max):
                mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[None, None, head_idx]
                tiler_gO = ((self.mma_tiler_pv[0] * self.q_stage), self.head_dim_v_padded)
                gO = cute.local_tile(mO_cur, tiler_gO, (work_m, 0))  # (128 * 2, 128)
                gO = layout_utils.select(
                    cute.flat_divide(gO, (self.mma_tiler_pv[0],)), mode=[0, 2, 1]
                )  # (128, 128, 2)
                gO = cute.flat_divide(gO, (self.mma_tiler_pv[0],))[None, mma_tile_coord_v, None, None]
                store_O, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_O, 0, cute.make_layout(1), sO, gO
                )
                for stage in cutlass.range(self.q_stage, unroll_full=True):
                    # wait from corr, issue tma store on smem
                    # 1. wait for O0 / O1 final
                    pipeline_o_epi.consumer_wait_w_index_phase(stage, epi_consumer_phase)
                    # 2. copy O0 / O1 to gmem
                    store_O(src_idx=stage, dst_idx=stage)
                    cute.arch.cp_async_bulk_commit_group()
                for stage in cutlass.range_constexpr(self.q_stage):
                    # Ensure O0 / O1 buffer is ready to be released
                    cute.arch.cp_async_bulk_wait_group(self.q_stage - 1 - stage, read=True)
                    pipeline_o_epi.consumer_release_w_index(stage)

                epi_consumer_phase ^= 1

            # Advance to next tile
            work_tile = tile_scheduler.advance_to_next_work()

    @cute.jit
    def empty_warp(
        self,
        tile_scheduler: TileSchedulerProtocol,
    ):
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            work_tile = tile_scheduler.advance_to_next_work()

    def load_Q(
        self,
        load_Q_fn: Callable,
        pipeline_q: pipeline.PipelineAsync,
        block: Int32,
        stage: int,
        phase: Int32,
    ):
        pipeline_q.producer_acquire_w_index_phase(stage, phase)
        load_Q_fn(src_idx=block, dst_idx=stage, tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(stage))

    @cute.jit
    def load_KV(
        self,
        tma_atom: cute.CopyAtom,
        tXgX: cute.Tensor,
        tXsX: cute.Tensor,
        sX: cute.Tensor,
        block: Int32,
        pipeline_kv: pipeline.PipelineAsync,
        producer_state: pipeline.PipelineState,
        K_or_V: Literal["K", "V"],
    ):
        assert K_or_V in ("K", "V")
        stage = producer_state.index
        extra_tx_count = self.tma_copy_bytes[K_or_V] - self.tma_copy_bytes["K"]
        pipeline_kv.producer_acquire(producer_state, extra_tx_count=extra_tx_count)
        tXsX_cur = tXsX[None, stage]
        tXgX_cur = tXgX[None, block]
        cute.copy(tma_atom, tXgX_cur, tXsX_cur, tma_bar_ptr=pipeline_kv.producer_get_barrier(producer_state))

    @cute.jit
    def process_work_tile(
        self,
        block_info: BlockInfo,
        seqlen_info: SeqlenInfoQK,
        n_block_min: Int32,
        n_block_max: Int32,
    ):
        process_work_tile_k = block_info.has_kv_work(n_block_min, n_block_max)
        process_work_tile_q = True
        return process_work_tile_k and process_work_tile_q
