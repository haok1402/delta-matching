"""
Matched delta at head_dim 256 on SM100, MXFP8 (spec sm100-mx-d256): pass 0 of
the backward. delta = sum_j P dP / sum_j P per query row, with P and dP formed
by the attention_math_sm100 helpers the backward uses, so the backward's
rowsum(dS) is zero at fp32 round-off. The kernel runs through FA4's forward
entry with its kernel class, compile cache and config swapped in.
"""
import dataclasses
import threading
from functools import partial
from typing import Callable, Optional, Literal

import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils_basic
import flash_attn.cute.interface as fi
import flash_attn.cute.pipeline as pipeline_custom
from cutlass import Float32, Float64, Int32, const_expr, pipeline
from cutlass.base_dsl.arch import Arch
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import BaseDSL
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from flash_attn.cute import blackwell_helpers as sm100_utils
from flash_attn.cute import cache_utils, utils
from flash_attn.cute import mma_sm100_desc as sm100_desc
from flash_attn.cute.block_info import BlockInfo
from flash_attn.cute.block_sparsity import BlockSparseTensors
from flash_attn.cute.config import SM100_MAX_KV_STAGE, FwdConfig, sm100_smem_capacity, sm100_uneven_kv_smem
from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned
from flash_attn.cute.flash_fwd_sm100 import DescaleTensors
from flash_attn.cute.named_barrier import NamedBarrierFwdSm100
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.tile_scheduler import SchedulingMode, SingleTileLPTScheduler, SingleTileScheduler, StaticPersistentTileScheduler, TileSchedulerArguments, TileSchedulerProtocol
from flash_attn.cute.utils import AuxData
from quack import copy_utils, layout_utils
from quack.cute_dsl_utils import ParamsBase

from delta_matching.sm100 import launch_cache_sm100
from delta_matching.sm100.attention_math_sm100 import dm_mx256_c, dm_mx256_dp_scale, dm_mx256_l2, dm_mx256_p_pair
from delta_matching.sm100.compile_cache_sm100 import fingerprint
from delta_matching.sm100.cute_ptx_sm100 import tmem_ld_16x64b_x16, tmem_ld_16x64b_x32
from delta_matching.sm100.tcgen05_fence_sm100 import FenceSkipSite, begin_kernel_fences, end_kernel_fences, mark_fenced

__all__ = ["attention_delta_hd256"]

D = 256
# FA4 config fields the pass-0 kernel needs (the kernel asserts them); FA4 still picks the scheduler.
FORCED_CONFIG = {"use_2cta_instrs": False, "pack_gqa": False, "use_s_ping_pong": False, "num_splits": 1, "use_tma_o": False, "overlap_sO_sQ": False, "kv_stage": 2, "q_stage": 2, "tile_m": 128, "tile_n": 128}
# 640 threads: per SMSP 4 compute warps + 1 other warp (4 x 112 + 32 = 480 registers per thread slot)
REGISTERS = {"softmax": 112, "correction": 112, "other": 32}
LAUNCH_LOCK = threading.Lock()
COMPILE_CACHES = {}
CONFIG_MEMO = {}


def attention_delta_hd256(q8, qs, k8, ks, v8, vs, do8, dos, lse, causal: bool):
    """
    delta (B, H, N) fp32 = sum_j P dP / sum_j P for every query row, MXFP8 (sm100-mx-d256). q8 / do8
    (B, N, H, 256) e4m3 with qs / dos (B, H, N/32) fp32; k8 / v8 (B, N, Hk, 256) e4m3 with ks / vs
    (B, Hk, N/32) fp32; every scale an exact power of two per 32 tokens over all of D (quant_qkv2 /
    quant_do_delta). lse (B, H, N) fp32, natural log. N % 128 == 0, H % Hk == 0.
    """
    B, N, H, head_dim = q8.shape
    assert head_dim == D and k8.shape[-1] == D and v8.shape[-1] == D and do8.shape == q8.shape and N % 128 == 0
    assert qs.shape == (B, H, N // 32) and dos.shape == (B, H, N // 32), "Q / dO: one pow2 scale per 32 rows (MXFP8)"
    assert ks.shape == (B, k8.shape[2], N // 32) and vs.shape == ks.shape, "K / V: one pow2 scale per 32 keys (MXFP8)"
    aux = [as_aux(qs.contiguous(), 2), as_aux(ks.contiguous(), 2), as_aux(vs.contiguous(), 2), as_aux(dos.contiguous(), 2), as_aux(do8.contiguous().view(torch.uint8), 3), as_aux(lse.contiguous(), 2)]
    names = ["qs", "ks", "vs", "dos", "do8", "lse"]
    inputs = dict(q8=q8, k8=k8, v8=v8, **{name: t for name, t in zip(names, aux, strict=True)})
    return launch_cache_sm100.call("attention_delta_hd256", inputs, (bool(causal), repr(sorted(FORCED_CONFIG.items()))), lambda: launch_through_fa4(q8, k8, v8, aux, causal), cmp=[1])[1]


def as_aux(t: torch.Tensor, leading_dim: int):
    """
    Tag an aux tensor with the leading dim and alignment FA4's aux plumbing reads.
    """
    t.__leading_dim__ = leading_dim
    t.__assumed_align__ = 16 if t.dtype == torch.uint8 else 4
    return t


def compile_cache():
    """
    The pass-0 compile cache, persistent when the CuTe DSL cache is enabled.
    """
    key = ("attention_delta_hd256",)
    if key not in COMPILE_CACHES:
        if cache_utils.CUTE_DSL_CACHE_ENABLED:
            COMPILE_CACHES[key] = cache_utils.JITPersistentCache(cache_utils.get_cache_path() / "dm_sm100" / fingerprint() / "attention_delta_hd256")
        else:
            COMPILE_CACHES[key] = cache_utils.JITCache()
    return COMPILE_CACHES[key]


def launch_through_fa4(q8, k8, v8, aux, causal):
    """
    FA4's forward entry with the pass-0 class, compile cache and forced config swapped in; returns
    (unused O, delta).
    """
    fa4_class, fa4_cache, fa4_select = fi.FlashAttentionForwardSm100, fi._flash_attn_fwd.compile_cache, fi.select_fwd_config

    def select_config(inputs):
        try:
            memo_key = (inputs, tuple(sorted((k, repr(v)) for k, v in FORCED_CONFIG.items())))
            hit = CONFIG_MEMO.get(memo_key)
        except TypeError:
            memo_key, hit = None, None
        if hit is not None:
            return hit
        config = fa4_select(inputs)
        config = dataclasses.replace(config, **{k: v for k, v in FORCED_CONFIG.items() if hasattr(config, k)})
        config = dataclasses.replace(config, registers=dataclasses.replace(config.registers, **REGISTERS))
        if memo_key is not None:
            CONFIG_MEMO[memo_key] = config
        return config

    with LAUNCH_LOCK:
        cache = compile_cache()   # before any FA4 global is swapped: a failure here leaves FA4 untouched
        fi.FlashAttentionForwardSm100 = MatchedDeltaHd256
        fi._flash_attn_fwd.compile_cache = launch_cache_sm100.RecCache(cache)
        fi.select_fwd_config = select_config
        try:
            unused_out, delta, *_ = fi._flash_attn_fwd(q8, k8, v8, softmax_scale=1.0, causal=causal, return_lse=True, pack_gqa=False, aux_tensors=aux)
        finally:
            fi.FlashAttentionForwardSm100, fi._flash_attn_fwd.compile_cache, fi.select_fwd_config = fa4_class, fa4_cache, fa4_select
    return (unused_out, delta)


class MatchedDeltaHd256:
    """
    Pass-0 kernel, a CuTe DSL fork of FA4's FlashAttentionForwardSm100: 1-CTA, q_stage 2, kv_stage
    2, two lanes per query row. Per KV tile the MMA warp issues S = Q8 K8^T and dP = dO8 V8^T (both
    kind::f8f6f4, V loaded K-major like K; S slot i at TMEM column 128 i, dP slot i at 256 + 128 i).
    Q / K / V / dO carry one pow2 scale per 32 tokens over all of D, so the scales sit on the MMAs'
    non-contraction dims and multiply the fp32 accumulator. The compute warps form P, num += P dP a
    and den += P with per-lane TwoSum cross-tile sums, combine in fp64 and write delta into mLSE.
    aux_data.tensors = [qs, ks, vs, dos, dO8 (uint8 view, (B, N, H, D)), lse (B, H, N)].
    """

    def __init__(self, head_dim: int, head_dim_v: Optional[int] = None, qhead_per_kvhead: cutlass.Constexpr[int] = 1, is_causal: bool = False, is_local: bool = False, q_subtile_factor: int = 1, kv_subtile_factor: int = 1, score_mod: cutlass.Constexpr | None = None,
                 mask_mod: cutlass.Constexpr | None = None, has_aux_tensors: cutlass.Constexpr = False, paged_kv_non_tma: bool = False, is_varlen_q: bool = False, has_tile_count_semaphore: bool = False, seqlen_k_per_split: Optional[int] = None, *, config: FwdConfig):
        """
        FA4's FlashAttentionForwardSm100 arguments; accepts only the forced pass-0 configuration.
        """
        assert head_dim == 256 and head_dim_v == 256, "MatchedDeltaHd256 is the head_dim 256 pass 0"
        assert not (is_local or score_mod is not None or mask_mod is not None or paged_kv_non_tma or is_varlen_q or has_tile_count_semaphore or seqlen_k_per_split is not None) and has_aux_tensors
        assert config.tile_m == 128 and config.tile_n == 128 and config.q_stage == 2 and config.num_splits == 1
        assert not (config.use_2cta_instrs or config.pack_gqa or config.use_tma_o or config.use_s_ping_pong or config.use_clc_scheduler or config.overlap_sO_sQ or config.lpt_global)
        self.head_dim, self.head_dim_v = head_dim, head_dim_v
        self.m_block_size, self.n_block_size = config.tile_m, config.tile_n
        self.q_stage = config.q_stage
        self.arch = BaseDSL._get_dsl().get_arch_enum()
        assert self.arch.is_family_of(Arch.sm_100f) or self.arch.is_family_of(Arch.sm_110f), "Only SM 10.x and 11.x are supported"
        self.cta_group_size = 1
        self.cta_tiler = (self.q_stage * self.m_block_size, self.n_block_size, self.head_dim)
        self.mma_tiler_qk = (self.m_block_size, self.n_block_size, self.head_dim)
        self.mma_tiler_pv = (self.m_block_size, self.head_dim_v, self.n_block_size)
        self.acc_dtype = Float32
        self.cluster_shape_mn = (1, 1)
        self.is_persistent = config.is_static_persistent
        self.is_causal = is_causal
        self.qhead_per_kvhead = qhead_per_kvhead
        self.kv_stage = config.kv_stage
        assert 2 <= self.kv_stage <= SM100_MAX_KV_STAGE, f"kv_stage={self.kv_stage}"
        self.s_stage = 2
        assert not sm100_uneven_kv_smem(self.head_dim, self.head_dim_v, self.kv_stage, False)
        if self.is_causal:
            self.TileScheduler = SingleTileLPTScheduler
        elif self.is_persistent:
            self.TileScheduler = StaticPersistentTileScheduler
        else:
            self.TileScheduler = SingleTileScheduler
        # Two lanes per query row (tcgen05.ld.16x64b), 4 compute warps per SMSP: warps 0-3 / 4-7 = stage 0 / 1 rows 0-15 of
        # each TMEM quadrant, warps 8-11 / 12-15 = stage 0 / 1 rows 16-31. Warps 17 and 19 only lower their registers.
        self.lower_compute_warps = (0, 1, 2, 3, 4, 5, 6, 7)
        self.upper_compute_warps = (8, 9, 10, 11, 12, 13, 14, 15)
        self.mma_warp = 16
        self.load_warps = (18,)
        self.idle_warps = (19, 17)
        self.threads_per_cta = 640
        self.tmem_s_offset = [0, self.n_block_size]
        # dP slot i at 256 + 128 i: S0 0, S1 128, dP0 256, dP1 384
        self.tmem_dp_offset = [self.tmem_s_offset[-1] + self.n_block_size + i * self.n_block_size for i in range(self.q_stage)]
        self.num_regs_lower_compute = config.registers.softmax
        self.num_regs_upper_compute = config.registers.correction
        self.num_regs_other = config.registers.other
        self.buffer_align_bytes = 1024

    @cute.jit
    def __call__(self, mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor, mO: cute.Tensor, mLSE: Optional[cute.Tensor], softmax_scale: Float32, mCuSeqlensQ: Optional[cute.Tensor] = None, mCuSeqlensK: Optional[cute.Tensor] = None, mSeqUsedQ: Optional[cute.Tensor] = None,
                 mSeqUsedK: Optional[cute.Tensor] = None, mPageTable: Optional[cute.Tensor] = None, window_size_left: Int32 | int | None = None, window_size_right: Int32 | int | None = None, learnable_sink: Optional[cute.Tensor] = None, descale_tensors: Optional[DescaleTensors] = None,
                 blocksparse_tensors: Optional[BlockSparseTensors] = None, aux_data: AuxData = None, num_splits_dynamic_ptr: Optional[cute.Tensor] = None, tile_count_semaphore: Optional[cute.Tensor] = None, virtual_batch_idx_ptr: Optional[cute.Tensor] = None,
                 num_nheads_in_l2_ptr: Optional[cute.Tensor] = None, mCuTotalMBlocks: Optional[cute.Tensor] = None, mCuTotalSplitsMBlocks: Optional[cute.Tensor] = None, mBlocksToBatchIdx: Optional[cute.Tensor] = None, max_seqlen_q: Int32 | int | None = None,
                 tensormap_workspace: Optional[cute.Tensor] = None, stream: cuda.CUstream = None):
        """
        FA4's forward call signature (positional, stream last). mQ / mK / mV / mO (b, s, h, d), mLSE
        receives delta; the varlen / paged / sparse / sink arguments must be None.
        """
        assert mCuSeqlensQ is None and mCuSeqlensK is None and mSeqUsedQ is None and mSeqUsedK is None
        assert mPageTable is None and learnable_sink is None and blocksparse_tensors is None and max_seqlen_q is not None
        self.q_dtype, self.k_dtype, self.v_dtype, self.o_dtype = mQ.element_type, mK.element_type, mV.element_type, mO.element_type
        mQ, mK, mV, mO = [assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)]
        mQ = cute.make_tensor(mQ.iterator, cute.select(mQ.layout, mode=[1, 3, 2, 0]))  # (s_q, d, h, b)
        # dO codes (uint8 aux tensor) viewed as fp8, same layout as Q
        mdO_bytes = aux_data.tensors[4]
        mdO = cute.make_tensor(cute.recast_ptr(mdO_bytes.iterator, dtype=self.q_dtype), cute.select(mdO_bytes.layout, mode=[1, 3, 2, 0]))
        mdO = assume_tensor_aligned(mdO)
        mK, mV = [cute.make_tensor(t.iterator, cute.select(t.layout, mode=[1, 3, 2, 0])) for t in (mK, mV)]  # (s_k, d, h_k, b_k)
        num_splits = Int32(1)
        mO = cute.make_tensor(mO.iterator, cute.select(mO.layout, mode=[1, 3, 2, 0]))
        mLSE = cute.make_tensor(mLSE.iterator, cute.select(mLSE.layout, mode=[2, 1, 0]))  # (s_q, h, b)
        if const_expr(self.q_dtype != self.k_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.k_dtype}")
        if const_expr(self.q_dtype != self.v_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.v_dtype}")

        cta_group = tcgen05.CtaGroup.ONE
        tiled_mma_qk = sm100_utils_basic.make_trivial_tiled_mma(self.q_dtype, self.q_dtype, cute.nvgpu.OperandMajorMode.K, cute.nvgpu.OperandMajorMode.K, self.acc_dtype, cta_group, self.mma_tiler_qk[:2])
        # FA4's PV MMA: never issued by pass 0, but its kernel parameter is part of the binary
        tiled_mma_pv = sm100_utils_basic.make_trivial_tiled_mma(self.v_dtype, self.v_dtype, cute.nvgpu.OperandMajorMode.K, cute.nvgpu.OperandMajorMode.MN, self.acc_dtype, cta_group, self.mma_tiler_pv[:2], tcgen05.OperandSource.TMEM)
        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        cta_layout_vmnk = cute.tiled_divide(cute.make_layout(self.cluster_shape_mnk), (tiled_mma_qk.thr_id.shape,))
        sQ_layout = sm100_utils_basic.make_smem_layout_a(tiled_mma_qk, self.mma_tiler_qk, self.q_dtype, self.q_stage)
        sK_layout = sm100_utils_basic.make_smem_layout_b(tiled_mma_qk, self.mma_tiler_qk, self.k_dtype, self.kv_stage)
        sV_layout = sK_layout  # V is a K-major B operand of dP = dO V^T

        self.tma_copy_bytes = {name: cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1, 2])) for name, mX, layout in [("Q", mQ, sQ_layout), ("K", mK, sK_layout), ("V", mV, sV_layout)]}
        self.tma_copy_bytes["Q"] *= 2  # Q and dO tiles land on the same pipeline_q stage
        for name in ("Q", "K", "V"):
            self.tma_copy_bytes[name] *= self.cta_group_size
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)
        tma_atom_Q, mQ = cute.nvgpu.make_tiled_tma_atom_A(tma_load_op, mQ, cute.select(sQ_layout, mode=[0, 1, 2]), self.mma_tiler_qk, tiled_mma_qk, cta_layout_vmnk.shape)
        tma_atom_dO, mdO = cute.nvgpu.make_tiled_tma_atom_A(tma_load_op, mdO, cute.select(sQ_layout, mode=[0, 1, 2]), self.mma_tiler_qk, tiled_mma_qk, cta_layout_vmnk.shape)
        tma_atom_K, mK = cute.nvgpu.make_tiled_tma_atom_B(tma_load_op, mK, cute.select(sK_layout, mode=[0, 1, 2]), self.mma_tiler_qk, tiled_mma_qk, cta_layout_vmnk.shape)
        tma_atom_V, mV = cute.nvgpu.make_tiled_tma_atom_B(tma_load_op, mV, cute.select(sK_layout, mode=[0, 1, 2]), self.mma_tiler_qk, tiled_mma_qk, cta_layout_vmnk.shape)

        TileScheduler = self.TileScheduler
        num_work_m = cute.ceil_div(max_seqlen_q, self.cta_tiler[0] * self.cta_group_size)
        tile_sched_args = TileSchedulerArguments(num_work_m, cute.size(mQ.shape[2]), cute.size(mQ.shape[3]), num_splits, cute.size(mK.shape[0]), mQ.shape[1], mQ.shape[1], total_q=cute.size(mQ.shape[0]) * cute.size(mQ.shape[3]), tile_shape_mn=self.cta_tiler[:2], mCuSeqlensQ=mCuSeqlensQ,
                                                 mSeqUsedQ=mSeqUsedQ, qhead_per_kvhead_packgqa=1, element_size=self.k_dtype.width // 8, is_persistent=self.is_persistent, lpt=self.is_causal, is_split_kv=False, cluster_shape_mn=self.cluster_shape_mn, use_cluster_idx=False, lpt_global=False,
                                                 num_splits_dynamic_ptr=num_splits_dynamic_ptr, virtual_batch_idx_ptr=virtual_batch_idx_ptr, num_nheads_in_l2_ptr=num_nheads_in_l2_ptr, cu_total_m_blocks_ptr=mCuTotalMBlocks, cu_total_splits_m_blocks_ptr=mCuTotalSplitsMBlocks,
                                                 blocks_to_batch_idx_ptr=mBlocksToBatchIdx, tile_count_semaphore=tile_count_semaphore.iterator if tile_count_semaphore is not None else None)
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args, scheduling_mode=SchedulingMode.STATIC)
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        # K and V alias the same physical buffer. The zero-sized and unused members keep FA4's shared-memory layout.
        sQ_size = cute.cosize(sQ_layout)
        sKV_size = cutlass.max(cute.cosize(sK_layout), cute.cosize(sV_layout))

        @cute.struct
        class SharedStorage:
            mbar_load_Q: cute.struct.MemRange[cutlass.Int64, self.q_stage * 2]
            mbar_load_KV: cute.struct.MemRange[cutlass.Int64, self.kv_stage * 2]
            mbar_S_full: cute.struct.MemRange[cutlass.Int64, self.q_stage * 2]
            mbar_P_full: cute.struct.MemRange[cutlass.Int64, self.q_stage * 2]
            mbar_dP_full: cute.struct.MemRange[cutlass.Int64, self.q_stage * 2]
            mbar_softmax_stats: cute.struct.MemRange[cutlass.Int64, self.q_stage * 2]
            mbar_O_epi: cute.struct.MemRange[cutlass.Int64, self.q_stage * 2]
            mbar_load_epi: cute.struct.MemRange[cutlass.Int64, 0]
            mbar_s0_s1_sequence: cute.struct.MemRange[cutlass.Int64, 2 * 2]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: Int32
            sScale: cute.struct.MemRange[Float32, self.q_stage * self.m_block_size * 2]
            sched_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 0]
            sched_response: cute.struct.MemRange[Int32, 0]
            sO: cute.struct.Align[cute.struct.MemRange[self.o_dtype, 0], self.buffer_align_bytes]
            sQ: cute.struct.Align[cute.struct.MemRange[self.q_dtype, sQ_size], self.buffer_align_bytes]
            sdO: cute.struct.Align[cute.struct.MemRange[self.q_dtype, cute.cosize(sQ_layout)], self.buffer_align_bytes]
            sK: cute.struct.Align[cute.struct.MemRange[self.k_dtype, sKV_size], self.buffer_align_bytes]

        self.shared_storage = SharedStorage
        smem_capacity = sm100_smem_capacity(self.arch.major * 10 + self.arch.minor)
        assert SharedStorage.size_in_bytes() <= smem_capacity, f"SMEM {SharedStorage.size_in_bytes()} > {smem_capacity}: kv_stage={self.kv_stage}"

        # FA4's softmax scale, window and fastdiv arguments: unused by pass 0, kept as kernel parameters of the binary
        softmax_scale_log2, softmax_scale = utils.compute_softmax_scale_log2(softmax_scale, None)
        window_size_left = Int32(window_size_left) if window_size_left is not None else None
        window_size_right = Int32(window_size_right) if window_size_right is not None else None
        fastdiv_mods = utils.compute_fastdiv_mods(mQ, mK, self.qhead_per_kvhead, False, aux_data.tensors, None)
        self.kernel(mQ, mK, mV, mO, mLSE, tma_atom_Q, tma_atom_K, tma_atom_V, softmax_scale_log2, softmax_scale, window_size_left, window_size_right, descale_tensors, sQ_layout, sK_layout, sV_layout, tiled_mma_qk, tiled_mma_pv, tile_sched_params, num_splits, num_splits_dynamic_ptr,
                    tile_count_semaphore, virtual_batch_idx_ptr, num_nheads_in_l2_ptr, aux_data, fastdiv_mods, tma_atom_dO, mdO).launch(grid=grid_dim, block=[self.threads_per_cta, 1, 1], cluster=None, stream=stream, min_blocks_per_mp=1)

    @cute.kernel
    def kernel(self, mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor, mO: cute.Tensor, mLSE: cute.Tensor, tma_atom_Q: cute.CopyAtom, tma_atom_K: cute.CopyAtom, tma_atom_V: cute.CopyAtom, softmax_scale_log2: Float32, softmax_scale: Float32 | None, window_size_left: Optional[Int32],
               window_size_right: Optional[Int32], descale_tensors: Optional[DescaleTensors], sQ_layout: cute.ComposedLayout, sK_layout: cute.ComposedLayout, sV_layout: cute.ComposedLayout, tiled_mma_qk: cute.TiledMma, tiled_mma_pv: cute.TiledMma, tile_sched_params: ParamsBase,
               num_splits: Int32, num_splits_dynamic_ptr: Optional[cute.Tensor] = None, tile_count_semaphore: Optional[cute.Tensor] = None, virtual_batch_idx_ptr: Optional[cute.Tensor] = None, num_nheads_in_l2_ptr: Optional[cute.Tensor] = None, aux_data: AuxData = None,
               fastdiv_mods=(None, None), tma_atom_dO: Optional[cute.CopyAtom] = None, mdO: Optional[cute.Tensor] = None):
        """
        Warp-specialized pass 0: the load warp brings Q, dO, K and V in by TMA, the MMA warp issues
        S and dP per KV tile, and 16 compute warps (two lanes per query row) form P, num and den and
        write delta. mO, softmax_scale*, window_size_*, descale_tensors, tiled_mma_pv and
        fastdiv_mods are unused.
        """
        begin_kernel_fences(1)
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            for tma_atom in (tma_atom_Q, tma_atom_K, tma_atom_V, tma_atom_dO):
                cpasync.prefetch_descriptor(tma_atom)
        cta_layout_vmnk = cute.tiled_divide(cute.make_layout(self.cluster_shape_mnk), (tiled_mma_qk.thr_id.shape,))
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        # TMEM handoffs span warp-role branches, so all waits and arrivals must be unaligned.
        tmem_alloc_barrier = pipeline_custom.NamedBarrier(barrier_id=int(NamedBarrierFwdSm100.TmemPtr), num_threads=cute.arch.WARP_SIZE * len((self.mma_warp, *self.lower_compute_warps, *self.upper_compute_warps)))
        tmem = cutlass.utils.TmemAllocator(storage.tmem_holding_buf.ptr, barrier_for_retrieve=tmem_alloc_barrier, allocator_warp_id=self.mma_warp, is_two_cta=False, two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr)

        thread_group = partial(pipeline.CooperativeGroup, pipeline.Agent.Thread)
        mma_warp_group = thread_group(1)
        tma_warp_group = thread_group(1)
        warpgroup_threads = thread_group(cute.arch.WARP_SIZE * 4)
        upper_compute_threads = thread_group(cute.arch.WARP_SIZE * len(self.upper_compute_warps))
        compute_threads = thread_group(cute.arch.WARP_SIZE * 4 * self.cta_group_size * 2)
        pipeline_q = pipeline_custom.PipelineTmaUmma.create(barrier_storage=storage.mbar_load_Q.data_ptr(), num_stages=self.q_stage, producer_group=tma_warp_group, consumer_group=mma_warp_group, tx_count=self.tma_copy_bytes["Q"], cta_layout_vmnk=cta_layout_vmnk, defer_sync=True)
        pipeline_kv = pipeline_custom.PipelineTmaUmma.create(barrier_storage=storage.mbar_load_KV.data_ptr(), num_stages=self.kv_stage, producer_group=tma_warp_group, consumer_group=mma_warp_group, tx_count=self.tma_copy_bytes["K"], cta_layout_vmnk=cta_layout_vmnk, defer_sync=True)
        # The MMA warp signals S ready (producer); all 16 compute warps release the S slot (consumer).
        pipeline_s = pipeline_custom.PipelineUmmaAsync.create(barrier_storage=storage.mbar_S_full.data_ptr(), num_stages=self.q_stage, producer_group=mma_warp_group, consumer_group=compute_threads, cta_layout_vmnk=cta_layout_vmnk, defer_sync=True)
        # FA4's P hand-off, s0 / s1 sequence and softmax-stats barriers: initialised but never waited on by pass 0
        # (kept: their initialisation is part of the binary).
        pipeline_p = pipeline_custom.PipelineAsyncUmma.create(barrier_storage=storage.mbar_P_full.data_ptr(), num_stages=self.q_stage, producer_group=thread_group(4 * self.cta_group_size), consumer_group=mma_warp_group, cta_layout_vmnk=cta_layout_vmnk, defer_sync=True)
        pipeline_dp = pipeline_custom.PipelineUmmaAsync.create(barrier_storage=storage.mbar_dP_full.data_ptr(), num_stages=self.q_stage, producer_group=mma_warp_group, consumer_group=compute_threads, cta_layout_vmnk=cta_layout_vmnk, defer_sync=True)
        pipeline_custom.PipelineAsync.create(barrier_storage=storage.mbar_s0_s1_sequence.data_ptr(), num_stages=2, producer_group=warpgroup_threads, consumer_group=warpgroup_threads, defer_sync=True)
        pipeline_custom.PipelineAsync.create(barrier_storage=storage.mbar_softmax_stats.data_ptr(), num_stages=self.q_stage, producer_group=warpgroup_threads, consumer_group=upper_compute_threads, defer_sync=True)
        stats_barrier = pipeline_custom.NamedBarrier(barrier_id=int(NamedBarrierFwdSm100.SoftmaxStatsW0), num_threads=cute.arch.WARP_SIZE * 2)
        # tcgen05 hand-off pipelines and barriers get their thread-sync fences (tcgen05_fence_sm100)
        mark_fenced("ua", pipeline_s, pipeline_dp)
        mark_fenced("au", pipeline_p)
        mark_fenced("bar", stats_barrier, tmem_alloc_barrier)
        pipeline_init_arrive(cluster_shape_mn=cta_layout_vmnk, is_relaxed=True)

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)  # (MMA, MMA_Q, MMA_D, PIPE)
        sdO = storage.sdO.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)  # (MMA, MMA_Q, MMA_D, PIPE)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)  # (MMA, MMA_K, MMA_D, PIPE)
        sV = cute.make_tensor(cute.recast_ptr(sK.iterator, sV_layout.inner), sV_layout.outer)  # same smem as sK, swizzle stripped
        thr_mma_qk = tiled_mma_qk.get_slice(0)
        qk_acc_shape = thr_mma_qk.partition_shape_C(self.mma_tiler_qk[:2])
        # TMEM is always allocated in full (512 columns), so the accumulator view starts at column 0
        tStS = thr_mma_qk.make_fragment_C(cute.append(qk_acc_shape, self.s_stage))  # (MMA, MMA_M, MMA_N, STAGE)
        block_info = BlockInfo(self.cta_tiler[0] * self.cta_group_size, self.cta_tiler[1], self.is_causal, False, False, window_size_left, window_size_right, qhead_per_kvhead_packgqa=1, num_splits=num_splits, pack_split_idx=num_splits_dynamic_ptr is not None, num_n_blocks_per_split=None)
        make_seqlen_info = partial(SeqlenInfoQK.create, seqlen_q_static=mQ.shape[0], seqlen_k_static=mK.shape[0], mCuSeqlensQ=None, mCuSeqlensK=None, mSeqUsedQ=None, mSeqUsedK=None, mCuTotalMBlocks=None, mCuBlockIdxOffsets=None)
        pipeline_init_wait(cluster_shape_mn=cta_layout_vmnk)
        tile_scheduler = self.TileScheduler.create(tile_sched_params)
        assert isinstance(tile_scheduler, TileSchedulerProtocol), f"tile_scheduler is not a TileSchedulerProtocol: {type(tile_scheduler)}"

        for i in cutlass.range_constexpr(len(self.idle_warps)):
            if warp_idx == self.idle_warps[i]:
                cute.arch.setmaxregister_decrease(self.num_regs_other)
        if warp_idx >= self.load_warps[0] and warp_idx <= self.load_warps[-1]:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            self.load_loop(thr_mma_qk, mQ, mK, mV, sQ, sK, sV, tma_atom_Q, tma_atom_K, tma_atom_V, pipeline_q, pipeline_kv, block_info, num_splits, make_seqlen_info, tile_scheduler, sdO, tma_atom_dO, mdO)
        if warp_idx == self.mma_warp:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            tmem.allocate(cute.arch.get_max_tmem_alloc_cols("sm_100"))
            tmem_alloc_barrier.arrive_and_wait_unaligned()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            self.mma_loop(tiled_mma_qk, sQ, sK, pipeline_q, pipeline_kv, pipeline_s, pipeline_dp, block_info, num_splits, make_seqlen_info, tile_scheduler, sdO)
            tmem.relinquish_alloc_permit()
            tmem_alloc_barrier.arrive_and_wait_unaligned()
            tmem.free(tmem_ptr)
        # The constant 'or' term is kept on purpose: without it the compiler lays the role blocks out in another order.
        if warp_idx <= self.lower_compute_warps[-1] or const_expr(False):
            cute.arch.setmaxregister_increase(self.num_regs_lower_compute)
            tmem_alloc_barrier.arrive_and_wait_unaligned()
            tmem.retrieve_ptr(self.acc_dtype)
            if warp_idx < self.lower_compute_warps[4]:
                self.compute_loop(0, 0, tStS, mLSE, pipeline_s, pipeline_dp, aux_data, block_info, num_splits, make_seqlen_info, tile_scheduler)
            else:
                self.compute_loop(1, 0, tStS, mLSE, pipeline_s, pipeline_dp, aux_data, block_info, num_splits, make_seqlen_info, tile_scheduler)
            tmem_alloc_barrier.arrive_unaligned()
        if warp_idx >= self.upper_compute_warps[0] and warp_idx < self.mma_warp:
            cute.arch.setmaxregister_increase(self.num_regs_upper_compute)
            tmem_alloc_barrier.arrive_and_wait_unaligned()
            tmem.retrieve_ptr(self.acc_dtype)
            if warp_idx < self.upper_compute_warps[4]:
                self.compute_loop(0, 1, tStS, mLSE, pipeline_s, pipeline_dp, aux_data, block_info, num_splits, make_seqlen_info, tile_scheduler)
            else:
                self.compute_loop(1, 1, tStS, mLSE, pipeline_s, pipeline_dp, aux_data, block_info, num_splits, make_seqlen_info, tile_scheduler)
            tmem_alloc_barrier.arrive_unaligned()
        end_kernel_fences()

    @cute.jit
    def load_loop(self, thr_mma_qk: cute.ThrMma, mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor, sQ: cute.Tensor, sK: cute.Tensor, sV: cute.Tensor, tma_atom_Q: cute.CopyAtom, tma_atom_K: cute.CopyAtom, tma_atom_V: cute.CopyAtom, pipeline_q: pipeline.PipelineAsync,
                  pipeline_kv: pipeline.PipelineAsync, block_info: BlockInfo, num_splits: Int32, make_seqlen_info: Callable, tile_scheduler: TileSchedulerProtocol, sdO: cute.Tensor, tma_atom_dO: cute.CopyAtom, mdO: cute.Tensor):
        """
        Load warp: per work tile, both Q and dO stages, then every K / V tile by descending n_block.
        """
        q_producer_phase = Int32(1)
        kv_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.kv_stage)
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            work_m, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = make_seqlen_info(batch_idx)
            mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
            kv_head_idx = head_idx // self.qhead_per_kvhead
            mK_cur = mK[None, None, kv_head_idx, batch_idx]
            mV_cur = mV[None, None, kv_head_idx, batch_idx]
            gK = cute.local_tile(mK_cur, cute.select(self.mma_tiler_qk, mode=[1, 2]), (None, 0))
            gV = cute.local_tile(mV_cur, cute.select(self.mma_tiler_qk, mode=[1, 2]), (None, 0))  # V tiled like K
            tSgK = thr_mma_qk.partition_B(gK)
            tSgV = thr_mma_qk.partition_B(gV)
            q_tiler = ((self.mma_tiler_qk[0] * self.q_stage), self.head_dim)
            gQ = cute.local_tile(mQ_cur, q_tiler, (work_m, 0))  # (128 * 2, 256)
            gQ = layout_utils.select(cute.flat_divide(gQ, (self.mma_tiler_qk[0],)), mode=[0, 2, 1])  # (128, 256, 2)
            tSgQ = thr_mma_qk.partition_A(gQ)
            copy_q, _, _ = copy_utils.tma_get_copy_fn(tma_atom_Q, 0, cute.make_layout(1), tSgQ, sQ)
            mdO_cur = seqlen.offset_batch_Q(mdO, batch_idx, dim=3)[None, None, head_idx]
            gdO = cute.local_tile(mdO_cur, q_tiler, (work_m, 0))
            gdO = layout_utils.select(cute.flat_divide(gdO, (self.mma_tiler_qk[0],)), mode=[0, 2, 1])
            tSgdO = thr_mma_qk.partition_A(gdO)
            copy_do, _, _ = copy_utils.tma_get_copy_fn(tma_atom_dO, 0, cute.make_layout(1), tSgdO, sdO)
            load_q_and_do = partial(self.load_q_and_do, copy_q, pipeline_q=pipeline_q, phase=q_producer_phase, copy_do=copy_do)
            tKsK, tKgK = cpasync.tma_partition(tma_atom_K, 0, cute.make_layout(1), cute.group_modes(sK, 0, 3), cute.group_modes(tSgK, 0, 3))  # no multicast
            tVsV, tVgV = cpasync.tma_partition(tma_atom_V, 0, cute.make_layout(1), cute.group_modes(sV, 0, 3), cute.group_modes(tSgV, 0, 3))  # no multicast
            load_k = partial(self.load_kv_tile, tma_atom_K, tKgK, tKsK, pipeline_kv=pipeline_kv, operand="K")
            load_v = partial(self.load_kv_tile, tma_atom_V, tVgV, tVsV, pipeline_kv=pipeline_kv, operand="V")
            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, work_m, split_idx=split_idx, num_splits=num_splits)
            if self.has_kv_work(block_info, n_block_min, n_block_max):
                load_k(block=n_block_max - 1, producer_state=kv_producer_state)
                load_q_and_do(block=0, stage=0)
                kv_producer_state.advance()
                load_q_and_do(block=1, stage=1)
                q_producer_phase ^= 1
                load_v(block=n_block_max - 1, producer_state=kv_producer_state)
                kv_producer_state.advance()
                for i in cutlass.range(n_block_max - 1 - n_block_min, unroll=1):
                    n_block = n_block_max - 2 - i
                    load_k(block=n_block, producer_state=kv_producer_state)
                    kv_producer_state.advance()
                    load_v(block=n_block, producer_state=kv_producer_state)
                    kv_producer_state.advance()
            work_tile = tile_scheduler.advance_to_next_work()
        pipeline_kv.producer_tail(kv_producer_state)
        # the producer tail of pipeline_q for the TMA-Q producer warp
        pipeline_q.producer_acquire_w_index_phase(self.q_stage - 1, q_producer_phase)

    @cute.jit
    def mma_loop(self, tiled_mma_qk: cute.ThrMma, sQ: cute.Tensor, sK: cute.Tensor, pipeline_q: pipeline.PipelineAsync, pipeline_kv: pipeline.PipelineAsync, pipeline_s: pipeline.PipelineAsync, pipeline_dp: pipeline.PipelineAsync, block_info: BlockInfo, num_splits: Int32,
                 make_seqlen_info: Callable, tile_scheduler: TileSchedulerProtocol, sdO: cute.Tensor):
        """
        MMA warp: per KV tile, S_s = Q_s K^T for both stages, then dP_s = dO_s V^T for both stages.
        """
        tSrQ = tiled_mma_qk.make_fragment_A(sQ)
        tSrK = tiled_mma_qk.make_fragment_B(sK)
        qk_mma_op = tiled_mma_qk.op
        qk_mma_kind = sm100_utils._tcgen05_mma_kind(qk_mma_op)
        q_smem_base = sm100_desc.smem_desc_base_from_tensor(sQ, sm100_desc.Major.K)
        k_smem_base = sm100_desc.smem_desc_base_from_tensor(sK, sm100_desc.Major.K)  # V is K-major too (sK view)
        q_smem_start = [sm100_desc.make_smem_desc_start_addr(sQ[None, None, None, stage].iterator) for stage in range(self.q_stage)]
        sm100_utils.declare_ptx_smem_desc(q_smem_start[self.q_stage - 1], q_smem_base, tSrQ[None, None, None, 0].layout, var_name_prefix="fa_fwd_q_smem_desc")
        sm100_utils.declare_ptx_idesc(qk_mma_op, var_name="fa_fwd_qk_mma_idesc")
        do_smem_start = [sm100_desc.make_smem_desc_start_addr(sdO[None, None, None, stage].iterator) for stage in range(self.q_stage)]
        sm100_utils.declare_ptx_smem_desc(do_smem_start[self.q_stage - 1], q_smem_base, tSrQ[None, None, None, 0].layout, var_name_prefix="dm_p0_do_smem_desc")
        sQ_stage_stride = (sQ.layout.stride[-1] * sQ.element_type.width // 8) >> 4
        gemm_s = [partial(sm100_utils.gemm_ptx_precomputed_varname, self.tmem_s_offset[stage], smem_desc_base_b=k_smem_base, tCrB_layout=tSrK[None, None, None, 0].layout, smem_var_name_prefix="fa_fwd_q_smem_desc", idesc_var_name="fa_fwd_qk_mma_idesc", kind=qk_mma_kind,
                          smem_offset=-sQ_stage_stride if stage == 0 else sQ_stage_stride, zero_init=True, cta_group=self.cta_group_size) for stage in range(self.q_stage)]
        # dP_s = dO_s V^T into the dP slots (same instruction shape as S = Q K^T)
        gemm_dp = [partial(sm100_utils.gemm_ptx_precomputed_varname, self.tmem_dp_offset[stage], smem_desc_base_b=k_smem_base, tCrB_layout=tSrK[None, None, None, 0].layout, smem_var_name_prefix="dm_p0_do_smem_desc", idesc_var_name="fa_fwd_qk_mma_idesc", kind=qk_mma_kind,
                           smem_offset=-sQ_stage_stride if stage == 0 else sQ_stage_stride, zero_init=True, cta_group=self.cta_group_size) for stage in range(self.q_stage)]
        q_consumer_phase = Int32(0)
        kv_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.kv_stage)
        slot_empty_phase = Int32(1)  # S / dP slots start empty
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            work_m, _, batch_idx, split_idx = work_tile.tile_idx
            seqlen = make_seqlen_info(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, work_m, split_idx=split_idx, num_splits=num_splits)
            num_kv_tiles = n_block_max - n_block_min
            if self.has_kv_work(block_info, n_block_min, n_block_max):
                for stage in cutlass.range_constexpr(self.q_stage):
                    pipeline_q.consumer_wait_w_index_phase(stage, q_consumer_phase)
                q_consumer_phase ^= 1
                for _ in cutlass.range(num_kv_tiles, unroll=1):
                    pipeline_kv.consumer_wait(kv_consumer_state)
                    sK_cur = sK[None, None, None, kv_consumer_state.index]
                    for stage in cutlass.range_constexpr(self.q_stage):
                        pipeline_s.producer_acquire_w_index_phase(stage, slot_empty_phase)
                        gemm_s[stage](smem_desc_start_b=sm100_desc.make_smem_desc_start_addr(sK_cur.iterator))
                        pipeline_s.producer_commit_w_index(stage)
                    pipeline_kv.consumer_release(kv_consumer_state)
                    kv_consumer_state.advance()
                    pipeline_kv.consumer_wait(kv_consumer_state)
                    sV_cur = sK[None, None, None, kv_consumer_state.index]
                    for stage in cutlass.range_constexpr(self.q_stage):
                        pipeline_dp.producer_acquire_w_index_phase(stage, slot_empty_phase)
                        gemm_dp[stage](smem_desc_start_b=sm100_desc.make_smem_desc_start_addr(sV_cur.iterator))
                        pipeline_dp.producer_commit_w_index(stage)
                    pipeline_kv.consumer_release(kv_consumer_state)
                    kv_consumer_state.advance()
                    slot_empty_phase ^= 1
                for stage in cutlass.range(self.q_stage):
                    pipeline_q.consumer_release_w_index(stage)
            work_tile = tile_scheduler.advance_to_next_work()

    @cute.jit
    def compute_loop(self, stage: int, row_half: int, tStS, mLSE, pipeline_s, pipeline_dp, aux_data, block_info, num_splits, make_seqlen_info, tile_scheduler):
        """
        Compute warps of one Q stage and row half: two lanes per query row. tcgen05.ld.16x64b gives
        thread t of the warp TMEM lane quad 32 + 16 row_half + t / 4 + 8 (t & 1) and the columns of
        parity (t >> 1) & 1, so the row's two lanes hold the x / y halves of every packed pair.
        Every den / num / TwoSum chain lives on one lane; the halves meet only in the final fp64
        combine (shuffle xor 2).
        """
        aux = aux_data.tensors
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        quad = warp_idx % 4
        lane = cute.arch.lane_idx()
        col_parity = (lane >> 1) & 1
        row_in_tile = quad * 32 + 16 * row_half + (lane >> 2) + 8 * (lane & 1)
        lane_base = Int32((quad * 32 + 16 * row_half) << 16)
        tSAcc = tStS[(None, None), 0, 0, stage]
        s_addr = Int32(tSAcc.iterator.toint()) + lane_base
        dp_addr = s_addr + Int32(self.tmem_dp_offset[0] - self.tmem_s_offset[0])
        s_full_phase = Int32(0)
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            work_m, head_idx, batch_idx, split_idx = work_tile.tile_idx
            kv_head_idx = self.kv_head_index(head_idx)
            seqlen = make_seqlen_info(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, work_m, split_idx=split_idx, num_splits=num_splits)
            has_work = self.has_kv_work(block_info, n_block_min, n_block_max)
            m_block = self.q_block_index(work_m, stage)
            row = m_block * self.m_block_size + row_in_tile
            row_clamped = cutlass.min(row, seqlen.seqlen_q - 1)
            # one pow2 Q / dO scale per 32 rows, one K / V scale per 32 keys (4 per n_block)
            q_scale = Float32(aux[0][batch_idx, head_idx, row_clamped // 32])
            do_scale = Float32(aux[3][batch_idx, head_idx, row_clamped // 32])
            lse_log2 = dm_mx256_l2(Float32(aux[5][batch_idx, head_idx, row_clamped]))
            # this lane's component of the TwoSum pairs, packed as (num, den)
            sum_pair = (Float32(0.0), Float32(0.0))
            err_pair = (Float32(0.0), Float32(0.0))
            # the raw K / V scales of a tile are loaded one iteration ahead
            scale_idx = cutlass.max(n_block_max - 1, 0) * (self.n_block_size // 32)
            k_scales = tuple(Float32(aux[1][batch_idx, kv_head_idx, scale_idx + j]) for j in range(4))
            v_scales = tuple(Float32(aux[2][batch_idx, kv_head_idx, scale_idx + j]) for j in range(4))
            if has_work:
                for n_tile in cutlass.range(n_block_max - n_block_min, unroll=1):
                    n_block = n_block_max - 1 - n_tile
                    exp_scales = [None] * 4
                    dp_scales = [None] * 4
                    for j in cutlass.range_constexpr(4):
                        exp_scales[j] = dm_mx256_c(k_scales[j], q_scale)
                        dp_scales[j] = dm_mx256_dp_scale(v_scales[j], do_scale)
                    scale_idx = cutlass.max(n_block - 1, n_block_min) * (self.n_block_size // 32)
                    k_scales = tuple(Float32(aux[1][batch_idx, kv_head_idx, scale_idx + j]) for j in range(4))
                    v_scales = tuple(Float32(aux[2][batch_idx, kv_head_idx, scale_idx + j]) for j in range(4))
                    # den chains of this lane's component: den01 = pairs v % 4 == 0, den23 = pairs v % 4 == 2
                    den01 = (Float32(0.0), Float32(0.0))
                    den23 = (Float32(0.0), Float32(0.0))
                    num = (Float32(0.0), Float32(0.0))   # (num of this component, unused)
                    pipeline_s.consumer_wait_w_index_phase(stage, s_full_phase)
                    is_diag_tile = False
                    if const_expr(self.is_causal):
                        is_diag_tile = (n_block + 1) * self.n_block_size > m_block * self.m_block_size
                    key_start = n_block * self.n_block_size
                    s_row = cute.make_rmem_tensor(cute.make_layout(64), self.acc_dtype)
                    # both S halves to registers first; the S slot is released before any exp, so the MMA warp's
                    # next S does not wait on this tile's dP
                    for h in cutlass.range_constexpr(2):
                        vals = tmem_ld_16x64b_x32(s_addr + Int32(64 * h))
                        for i in cutlass.range_constexpr(32):
                            s_row[32 * h + i] = vals[i]
                    cute.arch.fence_view_async_tmem_load()
                    with FenceSkipSite("r0"):
                        pipeline_s.consumer_release_w_index(stage)
                    for h in cutlass.range_constexpr(2):
                        if const_expr(self.is_causal):
                            if is_diag_tile:
                                for i in cutlass.range_constexpr(32 * h, 32 * h + 32):
                                    s_row[i] = -Float32.inf if key_start + 2 * i + col_parity > row else s_row[i]
                        for j in cutlass.range_constexpr(2 * h, 2 * h + 2):
                            for v in cutlass.range_constexpr(0, 16, 2):
                                i0 = 16 * j + v
                                p = dm_mx256_p_pair(s_row[i0], s_row[i0 + 1], exp_scales[j], lse_log2)
                                s_row[i0], s_row[i0 + 1] = p
                                if const_expr(v % 4 == 0):
                                    den01 = cute.arch.add_packed_f32x2(den01, p)
                                else:
                                    den23 = cute.arch.add_packed_f32x2(den23, p)
                        if const_expr(h == 0):
                            pipeline_dp.consumer_wait_w_index_phase(stage, s_full_phase)
                        for j in cutlass.range_constexpr(2 * h, 2 * h + 2):
                            dp_vals = tmem_ld_16x64b_x16(dp_addr + Int32(32 * j))
                            cute.arch.fence_view_async_tmem_load()
                            if const_expr(j == 3):
                                with FenceSkipSite("r1"):
                                    pipeline_dp.consumer_release_w_index(stage)
                            acc01 = (Float32(0.0), Float32(0.0))
                            acc23 = (Float32(0.0), Float32(0.0))
                            for v in cutlass.range_constexpr(0, 16, 2):
                                i0 = 16 * j + v
                                if const_expr(v % 4 == 0):
                                    acc01 = cute.arch.fma_packed_f32x2((s_row[i0], s_row[i0 + 1]), (dp_vals[v], dp_vals[v + 1]), acc01)
                                else:
                                    acc23 = cute.arch.fma_packed_f32x2((s_row[i0], s_row[i0 + 1]), (dp_vals[v], dp_vals[v + 1]), acc23)
                            # num = fma((acc0 + acc1) + (acc2 + acc3), a_j, num) per component
                            acc_sum = cute.arch.add_packed_f32x2((acc01[0], acc23[0]), (acc01[1], acc23[1]))
                            acc_total = cute.arch.add_packed_f32x2((acc_sum[0], Float32(0.0)), (acc_sum[1], Float32(0.0)))
                            num = cute.arch.fma_packed_f32x2((acc_total[0], Float32(0.0)), (dp_scales[j], Float32(0.0)), num)
                    # den = (den0 + den1) + (den2 + den3) per component
                    den_sum = cute.arch.add_packed_f32x2((den01[0], den23[0]), (den01[1], den23[1]))
                    den = cute.arch.add_packed_f32x2((den_sum[0], Float32(0.0)), (den_sum[1], Float32(0.0)))
                    # packed TwoSum of (num, den) of this component into (sum_pair, err_pair)
                    tile_pair = (num[0], den[0])
                    new_sum = cute.arch.add_packed_f32x2(sum_pair, tile_pair)
                    tile_part = cute.arch.add_packed_f32x2(new_sum, (-sum_pair[0], -sum_pair[1]))
                    sum_err = cute.arch.add_packed_f32x2(sum_pair, (-(new_sum[0] - tile_part[0]), -(new_sum[1] - tile_part[1])))
                    tile_err = cute.arch.add_packed_f32x2(tile_pair, (-tile_part[0], -tile_part[1]))
                    err_pair = cute.arch.add_packed_f32x2(err_pair, cute.arch.add_packed_f32x2(sum_err, tile_err))
                    sum_pair = new_sum
                    s_full_phase ^= 1
            # partner lane (other column parity of the same row) = lane ^ 2
            partner_sum_num = cute.arch.shuffle_sync_bfly(sum_pair[0], offset=2)
            partner_sum_den = cute.arch.shuffle_sync_bfly(sum_pair[1], offset=2)
            partner_err_num = cute.arch.shuffle_sync_bfly(err_pair[0], offset=2)
            partner_err_den = cute.arch.shuffle_sync_bfly(err_pair[1], offset=2)
            if col_parity == 0:
                num64 = (Float64(sum_pair[0]) + Float64(partner_sum_num)) + (Float64(err_pair[0]) + Float64(partner_err_num))
                den64 = (Float64(sum_pair[1]) + Float64(partner_sum_den)) + (Float64(err_pair[1]) + Float64(partner_err_den))
                delta = Float32(num64 / den64) if den64 != Float64(0.0) else Float32(0.0)
                if row < seqlen.seqlen_q:
                    mLSE[row, head_idx, batch_idx] = delta
            work_tile = tile_scheduler.advance_to_next_work()

    def q_block_index(self, work_m, stage):
        """
        128-row Q block of one stage of a work tile (1 CTA, q_stage stages per tile).
        """
        return work_m * self.q_stage + stage

    @cute.jit
    def kv_head_index(self, head_idx: Int32) -> Int32:
        """
        KV head of a query head.
        """
        return head_idx // self.qhead_per_kvhead

    def load_q_and_do(self, copy_q: Callable, pipeline_q: pipeline.PipelineAsync, block: Int32, stage: int, phase: Int32, copy_do: Callable):
        """
        Q and dO tiles of one stage, on the same pipeline_q barrier.
        """
        pipeline_q.producer_acquire_w_index_phase(stage, phase)
        copy_q(src_idx=block, dst_idx=stage, tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(stage))
        copy_do(src_idx=block, dst_idx=stage, tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(stage))

    @cute.jit
    def load_kv_tile(self, tma_atom: cute.CopyAtom, tXgX: cute.Tensor, tXsX: cute.Tensor, block: Int32, pipeline_kv: pipeline.PipelineAsync, producer_state: pipeline.PipelineState, operand: Literal["K", "V"]):
        """
        One K or V tile into the next pipeline_kv stage.
        """
        assert operand in ("K", "V")
        stage = producer_state.index
        extra_tx_count = self.tma_copy_bytes[operand] - self.tma_copy_bytes["K"]
        pipeline_kv.producer_acquire(producer_state, extra_tx_count=extra_tx_count)
        cute.copy(tma_atom, tXgX[None, block], tXsX[None, stage], tma_bar_ptr=pipeline_kv.producer_get_barrier(producer_state))

    @cute.jit
    def has_kv_work(self, block_info: BlockInfo, n_block_min: Int32, n_block_max: Int32):
        """
        Whether the work tile has at least one KV tile.
        """
        return block_info.has_kv_work(n_block_min, n_block_max)
