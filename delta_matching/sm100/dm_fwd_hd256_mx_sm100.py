"""hd256 Delta-Matching forward for sm100 (GB200), sm100-mx numerics (USER DECISION 2026-11-10: MXFP8 is the sm100 spec at
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

import numpy as np
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, const_expr
from cutlass.cute import FastDivmodDivisorV2
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

from delta_matching.sm100 import dm_utils
from delta_matching.sm100.dm_utils import tmem_st_x4

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


# ---- DEBUG event trace (fork hd256-fwd3; DM_HD256_FWD_TRACE=1; compiled out by default) ----
@dsl_user_op
def tr_st(addr: Int64, val: Int64, en: Int32, *, loc=None, ip=None):
    """lane 0 of the warp stores val (u64) to addr when en != 0 (predicated: no branch, no divergence)."""
    llvm.inline_asm(None, [Int64(addr).ir_value(loc=loc, ip=ip), Int64(val).ir_value(loc=loc, ip=ip),
                           Int32(en).ir_value(loc=loc, ip=ip)],
                    "{\n\t.reg .pred p, q;\n\t.reg .u32 l;\n\tmov.u32 l, %laneid;\n\tsetp.eq.u32 p, l, 0;\n\t"
                    "setp.ne.and.u32 q, $2, 0, p;\n\t@q st.global.u64 [$0], $1;\n\t}",
                    "l,l,r", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def tr_wait_clock(mbar: Int32, phase: Int32, *, loc=None, ip=None) -> Int64:
    """Observer: spin on mbarrier phase parity; return %clock64 when it completed, 0 after ~4M clk (never hangs)."""
    return Int64(llvm.inline_asm(
        T.i64(), [Int32(mbar).ir_value(loc=loc, ip=ip), Int32(phase).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p, q;\n\t.reg .u64 t0, t1, d;\n\tmov.u64 t0, %clock64;\n\tTRW_L:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2;\n\t@p bra TRW_D;\n\t"
        "mov.u64 t1, %clock64;\n\tsub.u64 d, t1, t0;\n\tsetp.lt.u64 q, d, 4000000;\n\t@q bra TRW_L;\n\t"
        "mov.u64 $0, 0;\n\tbra TRW_E;\n\tTRW_D:\n\tmov.u64 $0, %clock64;\n\tTRW_E:\n\t}",
        "=l,r,r", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


TR_E = 32   # events per row; row 0 = CTA header, row j + 1 = tile j (see DMFwdHd256MxSm100.tr)


# ---- C14 persistent CTAs (fork hd256-fwd6; DM_HD256_FWD_PERS=1): cluster launch control (1-CTA clusters) ----
@dsl_user_op
def clc_try_cancel(resp: Int32, mbar: Int32, *, loc=None, ip=None):
    """clusterlaunchcontrol.try_cancel: asynchronously write the 16-byte response for the next not-yet-launched CTA into
    resp (16-B aligned SMEM), completing 16 tx bytes on mbar (the caller arrives with expect_tx(16) first)."""
    llvm.inline_asm(None, [Int32(resp).ir_value(loc=loc, ip=ip), Int32(mbar).ir_value(loc=loc, ip=ip)],
                    "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 [$0], [$1];",
                    "r,r", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def clc_decode(resp: Int32, *, loc=None, ip=None):
    """(valid, x, y, z) of a try_cancel response in SMEM; the ctaid is extracted ONLY for a canceled (valid) response
    (PTX: get_first_ctaid of a failed response is undefined), else x = y = z = 0."""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4), [Int32(resp).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b128 r;\n\t.reg .pred p;\n\t.reg .b64 lo, hi;\n\t.reg .b32 w;\n\t"
        "ld.shared.v2.b64 {lo, hi}, [$4];\n\tmov.b128 r, {lo, hi};\n\t"
        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p, r;\n\t"
        "selp.u32 $0, 1, 0, p;\n\tmov.u32 $1, 0;\n\tmov.u32 $2, 0;\n\tmov.u32 $3, 0;\n\t"
        "@p clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {$1, $2, $3, w}, r;\n\t}",
        "=r,=r,=r,=r,r", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return [Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4)]


def gemm_qk_ptx(op, acc_tmem_addr, tCrA, tCrB, sA, sB, b_off: int = 0):
    """Plain (non-scaled) tcgen05 kind::f8f6f4 SS MMA chain over all k-steps (k ascending, k-step 0 zero-init), raw PTX.
    op: the MMA op whose instruction descriptor (M, N) is used; tCrA / tCrB / sA / sB: the N=128 QK fragments and smem tensors of
    one stage; b_off: B start offset in descriptor units (16 B), e.g. 896 = key row 112 of the K-major SW128 tile (1 KiB-atom
    aligned, so the N=128 layout's LBO/SBO/swizzle stay valid for the sub-tile).  fork hd256-fwd2 C7."""
    from flash_attn.cute import mma_sm100_desc as sm100_desc
    from flash_attn.cute.blackwell_helpers import i64_to_i32x2
    idesc: int = sm100_desc.mma_op_to_idesc(op)
    nk = cute.size(tCrA.shape[2])

    def desc_base(t, dtype_width, major_mode):
        b = sm100_desc.make_smem_desc_base(
            cute.recast_layout(128, dtype_width, t.layout[0]), t.iterator.type.swizzle_type,
            sm100_desc.Major.K if major_mode == cute.nvgpu.tcgen05.mma.OperandMajorMode.K else sm100_desc.Major.MN)
        return i64_to_i32x2(b)

    a_lo, a_hi = desc_base(sA, op.a_dtype.width, op.a_major_mode)
    b_lo, b_hi = desc_base(sB, op.b_dtype.width, op.b_major_mode)
    offset_a = [cute.crd2idx((0, 0, k), tCrA.layout) for k in range(nk)]
    offset_b = [cute.crd2idx((0, 0, k), tCrB.layout) + b_off for k in range(nk)]
    start_a = Int32(a_lo | sm100_desc.make_smem_desc_start_addr(sA[None, None, 0].iterator))
    start_b = Int32(b_lo | sm100_desc.make_smem_desc_start_addr(sB[None, None, 0].iterator))
    input_args = [Int32(cute.arch.make_warp_uniform(start_a)).ir_value(),
                  Int32(cute.arch.make_warp_uniform(start_b)).ir_value(),
                  Int32(cute.arch.make_warp_uniform(acc_tmem_addr)).ir_value()]
    instr = "tcgen05.mma.cta_group::1.kind::f8f6f4"
    body = ("{\n\t.reg .pred leader_thread;\n\t.reg .b32 tmem_acc, a0, b0, a_lo, b_lo, a_hi, b_hi;\n\t.reg .b64 adesc, bdesc;\n\t"
            "elect.sync _|leader_thread, -1;\n\t"
            "mov.b32 a0, $0;\n\tmov.b32 b0, $1;\n\tmov.b32 tmem_acc, $2;\n\t"
            f"mov.b32 a_hi, {hex(a_hi)};\n\tmov.b32 b_hi, {hex(b_hi)};\n\t")
    for k in range(nk):
        body += (f"add.u32 a_lo, a0, {hex(offset_a[k])};\n\tmov.b64 adesc, {{a_lo, a_hi}};\n\t"
                 f"add.u32 b_lo, b0, {hex(offset_b[k])};\n\tmov.b64 bdesc, {{b_lo, b_hi}};\n\t"
                 f"@leader_thread {instr} [tmem_acc], adesc, bdesc, {hex(idesc)}, {0 if k == 0 else 1};\n\t")
    body += "}\n"
    llvm.inline_asm(None, input_args, body, ",".join(["r"] * len(input_args)),
                    has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


class DMFwdHd256MxSm100:
    """sm100-mx hd256 forward (module docstring).  Inputs (contiguous): q8 (B,N,H,256) e4m3 (sm baked), k8/v8 (B,N,Hk,256);
    qs (B,H,N/32) fp32 pow2; kvw (B,Hk,N/64) int32 packed UE8M0 words ([2t] K bytes of tile t, [2t+1] V bytes).
    Out: O (B,N,H,256) bf16, LSE (B,H,N) fp32 natural log."""

    def __init__(self, is_causal: bool, qhead_per_kvhead: int, kv_stage: int = 4, rescale_T: float = 32.0,
                 sf_wait: bool = True, pc_inj: int = 0, sf_bank: bool = False, two_wg: bool = True, ord_td: bool = False,
                 lpt: bool = False, epi: int = 0, split: int = 0, trace: bool = False, g1: int = 0, c8: int = 0,
                 c10: int = 0, c11: int = 0, c12: int = 0, c13: int = 0, tsplit: int = 0, pers: int = 0):
        # C14 pers (fork hd256-fwd6, DM_HD256_FWD_PERS=1; requires C13, epi 2, causal T2 order; Codex hd256-fwd6-01): persistent CTAs.
        # The grid is unchanged (one CTA id per item, T2 map); a resident CTA finishes its item and claims the next not-yet-launched
        # CTA id with clusterlaunchcontrol.try_cancel (load warp 14 issues one claim per item after its loads; 2-slot response ring
        # clc_full[s] (tx 16 B) / clc_empty[s] (15 warp acks: 8 prefix, 4 correction, QK 12, load 14, PV 15)); every role warp loops
        # over the same item sequence and stops at the first failed claim.  Item k has a global tile base J0 (sum of earlier cnt):
        # every slot / ring / barrier phase uses the global tile index J = J0 + j (S slot J&1, rings J%3, O_full / O_done2 / S_empty /
        # W1L / P_full3 / partial_full3 phases), K/V sequence numbers are 2*J0 + the item-local order, Q one phase per item, O_ready
        # round J - k - 1; item-local j keeps the semantics (diagonal mask and first tail at j = 0, PV zero-init, last-tile SF at its
        # own slot +80, key block n_block_max-1-j).  Softmax WG h owns the tiles with J&1 == h.  Values per item are op-for-op C13.
        # Cross-item edges (pers only): QK(J0) waits PV(J0-2) (not J0-3: PV(J0-1) of the previous item may already have completed on
        # the same O_done2 parity) and QK(J0+1) waits PV(J0-1) (the previous last tile's SF at own slot +80 is in the slot it
        # overwrites).  O epilogue staging (two column halves, 128 rows x 256 B XOR-swizzled, in ONE K/V ring stage = the stage of the
        # next item's gated load g = 2*J0' + (3 if cnt' >= 2 else 1); the next item's first loads are K0, K1, V0, K2 so g is K2 /
        # V1 / V0): the load warp waits EPI_done (count 128, after the copy-out + fence.proxy.async) before loading sequence g.
        # The old O TMEM reads are ordered before PV0(next) by the correction's P_full(J0') publication (epilogue -> tail0 in
        # program order, tcgen05.fence::before_thread_sync before the arrive), so no O_free barrier (Codex fwd6-01 A).
        # DEBUG PCs (pers): 22 load warp skips the EPI_done wait + correction sleeps 20 us between staging and copy-out of half 0
        # (the gated TMA overwrites the staged rows) -> must corrupt; 23 QK warp skips the previous-last SF wait (j = 1) + PV warp
        # sleeps 20 us before PV(last) of every item -> must corrupt; 24 QK warp releases Q before the item's last QK and sleeps
        # 20 us before it, load warp issues Q before K0 (else Q(k+1) waits behind K0's stage = the last QK's) -> must corrupt.
        # DEBUG 25 (probe, not a control): row 0 LSE of each item = k + 1000*smid (items per persistent CTA).
        self.pers = bool(pers)
        if pers:
            c13 = 1
            assert is_causal and tsplit > 0 and epi == 2 and not trace and kv_stage == 4, "pers: causal T2, epi 2, no trace"
        # c8 (fork hd256-fwd3 C8, DM_HD256_FWD_C8=1; Codex hd256-fwd2-02 C, hd256-fwd3-02): column split of every tile over BOTH
        # softmax WGs.  WG h loads S columns 64h..64h+63 (blocks 2h, 2h+1) and runs the m-independent prefix op for op (bmax, kb,
        # bias, FFMA2, ex2, e4m3 codes -> sP, pairwise block sums).  WG1 publishes bmax2/3, bs2/3 (SMEM part[slot][k][128],
        # partial_full[slot] = mbar+80/+88, 128 arrivals, after wait::ld + fence::before) and arrives P_full (count 256).  WG0 keeps
        # (m, l) in registers over all tiles, waits partial_full (+fence::after), and runs the ordered tail exactly as C6 (same
        # xmax expression, kb2/3 recomputed from the published bmax, SFA for 4 blocks, SF stores, l update in block order 0..3)
        # with G1a (stats_full right after the m rule).  MMA warp / correction unchanged.  Values bitwise == C6.
        # DEBUG PCs (c8): 4 WG0 skips the partial wait; 5 WG1 arrives partial before its payload (+20 us); 11 WG1 arrives P_full
        # before its P stores (+20 us); 12 WG0 stores SFB right after S_full while WG1 delays its S load 20 us  -> all must corrupt.
        # c8 = 2 (C8b): WG1 stores SFB(j) (the m-independent V scale word, columns slot+64..71 = its own S columns) right after its
        # S load (wait::ld) and completes it (wait::st + fence::before) before P_full; WG0's tail stores only SFA.  Removes the
        # WG1 S-load -> WG0 SFB-store TMEM edge and 8 stores from WG0's serial tail.  PC 12 (C8b): WG1 stores SFB BEFORE its
        # S load -> must corrupt.
        assert c8 in (0, 1, 2)
        self.c8 = c8
        # C10 (fork hd256-fwd4, DM_HD256_FWD_C10=1, requires C8 = 1, split 0; Codex hd256-fwd4-02): SF(j) lives in the NEXT tile's
        # S slot so QK(j+2) no longer waits PV(j)'s completion (only PV(j-1)'s, whose SF(j-1) sits in slot j&1) and S(j+1) is ready
        # early (WG1 runs ahead and fills WG0's serial tail with MUFU work).  TMEM: j < cnt-1: SFA(j) at slot (j+1)&1 + 64..67,
        # SFB(j) + 72..79 (inside S(j+1)'s WG1 columns); last tile: own slot + 80..83 / + 88..95 (SF(cnt-2) occupies +64..79 there).
        # Ordering: both WGs arrive S_empty[slot](t) (256) after their own S(t) load; WG1 also arrives W1L[slot](t) (128); WG0's tail(j)
        # waits W1L(j+1) (+fence::after) before storing SF(j) into slot (j+1)&1 (WAR vs WG1's S(j+1) load; the old SF(j-2) there was
        # read by PV(j-2), complete before QK(j+1) was issued -> S_full(j+1) -> WG1 -> W1L).  MMA: QK(j+2) after S_empty(j) +
        # O_full(PV(j-1)) (SF(j-1) WAR), issued BEFORE PV(j).  Because S_full(j+2) no longer follows PV(j)'s completion, every
        # per-tile softmax resource is a 3-deep ring b = j % 3 (phase (j // 3) & 1): sP[b] (3 x 16 KiB), part[b], stats[b], and the
        # barriers P_full[b] (256), partial_full[b], stats_full[b]; reuse at j+3 follows S_full(j+3) <- QK(j+3) issued after the
        # MMA waited O_full(PV(j)), i.e. after PV(j) (and P_full(j), O_ready(j)) consumed everything tile j handed off.  Values bitwise == C8.
        # DEBUG PCs (C10): 6 MMA skips S_empty + WG1 sleeps 20 us before each S load; 7 both WGs arrive S_empty before their S load
        # (+WG1 sleep); 8 MMA skips O_full(PV(j-1)) before QK(j+2) (informational); 14 WG0 skips the W1L wait (+WG1 sleep) -> must
        # corrupt; 1 / 2 poison SFA(i) at its C10 address after PV(i) issue / completion.
        # C11 (fork hd256-fwd4, DM_HD256_FWD_C11=1, implies C10): the ordered tail moves from WG0 to the correction WG (which
        # already consumes acc_scale).  Both softmax WGs only run the m-independent prefix of their 2 blocks and publish
        # (bmax, bs) x 2 into part[j%3] (k = 4h..4h+3; partial_full3 count 256, after fence::before) and their P codes
        # (P_full3 count 384 = 256 WG + 128 correction).  The correction WG holds (m, l) in registers and per tile j: waits partial
        # (+fence::after), runs exactly the C8 WG0 tail (same xmax / ceil / lazy rule / kb / SFA / w expressions, l in block order),
        # stores SFA/SFB at the C10 address (after W1L(j+1) for j < cnt-1), wait::st + fence::before -> P_full; then (j >= 1)
        # O_full(PV(j-1)) -> conditional rescale -> O_ready(j); finally LSE + 1/l and the epilogue (no stats / final_full hop).
        # DEBUG PCs (C11): 4 correction skips the partial wait; 15 correction arrives P_full before its SF stores (+20 us).
        # C12 (fork hd256-fwd4, DM_HD256_FWD_C12=1, implies C11 / C10): the two softmax WGs own alternate tiles again (WG j&1 owns
        # tile j and S slot j&1, as C6) but with NO running state: each runs the full 4-block m-independent prefix of its tile
        # (C6 / C8 expressions, op for op) and publishes (bmax_b, bs_b) into part[j%3] in C11's layout + its P codes.  The
        # correction WG runs the ordered tail (C11).  Owner-only barriers: S_empty[slot] and W1L[slot] count 128 (the owner arrives
        # both after its S(j) load), partial_full3 count 128, P_full3 count 256 (owner 128 + correction 128).  The WGs are now
        # independent and naturally staggered by half a cycle, so one WG's latency-bound phases overlap the other's MUFU work.
        # DEBUG PCs (C12, Codex hd256-fwd4-02 reuse controls): 16 sP ring depth 2 + PV(i) issued 20 us late (P(i+2) overwrites
        # sP before PV(i) reads it) and 17 part ring depth 2 + correction reads part(j) 20 us late -> both must corrupt.
        # C13 (fork hd256-fwd4, DM_HD256_FWD_C13=1, implies C12): a second MMA issuer.  Warp 12 issues only QK (mma_c13_qk: the
        # mma_c10 QK part), warp 15 only PV (mma_c13_pv).  No cross-warp MMA order is relied on: QK(j+2) waits PV(j-1)'s completion
        # through O_done2[(j-1)&1] (new per-parity barrier, tcgen05.commit by warp 15 after PV(j-1); a parity barrier because PV(j)
        # can complete before warp 12 reaches its wait - PV(j+1) cannot: it needs P_full(j+1) <- W1L(j+2) <- QK(j+2)); PV(j)
        # depends on QK only through S(j+1) -> W1L(j+1) -> SF(j) (correction).  K stages are released by warp 12, V stages by warp 15
        # (one release per stage, each a commit tracking its reader).  Trace: warp 13 observes S_full and O_full in pipe order.
        assert c13 in (0, 1)
        self.c13 = bool(c13)
        if c13:
            c12 = 1
        assert c12 in (0, 1)
        self.c12 = bool(c12)
        if c12:
            c11 = 1
        assert c11 in (0, 1)
        self.c11 = bool(c11)
        if c11:
            c10 = 1
        assert c10 in (0, 1)
        self.c10 = bool(c10)
        assert not c10 or (c8 == 1 and split == 0)
        if c8:
            g1 = g1 | 1
            assert two_wg and not sf_bank and (split == 0 or (split in (2, 3) and c8 == 1))
        # C9 (fork hd256-fwd4, c8 == 1 + split == 2 = DM_HD256_FWD_SPLIT=2 with the default C8): the C8 softmax (column split) on the
        # C7c MMA program.  SF(j) at slot+112..127 (SFA +112..115, SFB +120..127; both inside WG1's S columns 64..127, so WG0's SF
        # stores stay ordered by C8's WG1 wait::ld -> fence::before -> partial_full -> WG0 wait + fence::after edge); QK(j) = QK_a
        # (N112, cols 0..111) + QK_b (N16, cols 112..127).  BOTH WGs arrive S_empty[slot] (count 256) right after their own
        # tcgen05.ld + wait::ld (+fence::before); the MMA issues QK_a(j+2) at S_empty(j) (overlaps softmax(j)), PV(j) at P_full(j),
        # and QK_b(j+2) only after O_full(PV(j)) (+fence::after: PV(j) reads SF(j) there).  Values bitwise == C8 (same per-element
        # 8-step K=32 S accumulation, same softmax ops).  DEBUG PCs (C9): 6 = MMA skips S_empty AND WG1 sleeps 20 us before its
        # S load (must corrupt); 7 = both WGs arrive S_empty before their S load, WG1 sleeps 20 us before it (must corrupt);
        # 13 = WG0 stores (zero) SFA words at +112 right after S_full while WG1 delays its S load 20 us (must corrupt; Codex fwd4-01).
        # split == 3 (C9r, c8 == 1 only): same layout / S_empty / softmax as C9, MMA program mma_c9r (QK_b one iteration earlier:
        # iteration j = [O_full(PV(j-1))] QK_b(j+1) -> S_full(j+1); [S_empty(j)] QK_a(j+2); [P_full(j), O_ready(j)] PV(j)), so the
        # issuer never blocks on the completion of the PV it just issued (fwd4 trace: C9 issuer saw P_full 674 clk late).
        self.c9 = bool(c8 == 1 and split in (2, 3))
        self.sempty = self.c9 or bool(c10)      # both WGs arrive S_empty (count 256)
        self.n_sp = 3 if c10 else 2             # sP buffers
        if c11:
            self.num_regs_softmax_c11 = 184
            self.num_regs_correction_c11 = 96
        self.n_mbar_dm = 28 if c13 else 26 if c10 else 14
        if pers:
            self.n_mbar_dm = 32       # + 216 EPI_done, 224/232 clc_full[s], 240/248 clc_empty[s]
        self.n_stats = 2 * 128 + 2 * 128 + 4 * 128 + ((24 if c11 else 12 if c10 else 8) * 128 if c8 else 0)
        self.part_stride = 1024 if c11 else 512      # floats per part[] buffer (C11: 8 x 128)
        # g1 (fork hd256-fwd3, DM_HD256_FWD_G1 bitmask; profile/hd256-fwd3-tl/REPORT.md): critical-path trims of C6, values bitwise
        # (same ops, same tensor-pipe MMAs; only program order / issue form).  1 = G1a: the softmax publishes acc_scale
        # (stats_full) right after the ordered m rule, before the SF stores / P_full (O_ready no longer trails P_full).
        # 2 = G1b: QK(j) issued as raw PTX (gemm_qk_ptx, the N128 QK op's idesc and descriptors) instead of 8 cute.gemm.
        # 4 = G1c: the MMA warp waits K(j) (TMA full) before the C0 O_full(PV(j-2)) wait (off the PV -> QK path).
        # 8 = G1d: SFB (V scale word, m-independent) stored before the mailbox wait; SFA / SFB as tcgen05.st x4 (same columns).
        self.g1 = g1
        assert g1 == 0 or (two_wg and not sf_bank and (split == 0 or (c8 == 1 and g1 == 1))), "G1 trims apply to C6/C8 (C9: G1a only)"
        # trace (DEBUG, fork hd256-fwd3; DM_HD256_FWD_TRACE=1; False = compiled out, default PTX unchanged): %clock64 per event
        # into mDbg (int64 [n_cta, n_m + 1, TR_E]); events listed in tr().  Warps 13 / 15 (idle) observe S_full / O_full
        # completions.  Numerics untouched (stores to a separate buffer; lane-0 predicated, no divergence).
        self.trace = trace
        # split (fork hd256-fwd2 C7; DM_HD256_FWD_SPLIT): SF of tile j moved to slot+112..127 (SFA +112..115, SFB +120..127) and
        # QK(j) split into QK_a (M128 N112, S cols 0..111) + QK_b (M128 N16, cols 112..127; K rows 112..127).  Each S element is
        # the same 8-step K=32 accumulation (k ascending, zero-init at k 0): bitwise.  2 = C7b: the softmax arrives S_empty[slot]
        # right after loading S(j) (tcgen05.ld + wait::ld + fence::before); the MMA warp issues QK_a(j+2) at S_empty(j) (no
        # TMEM column of it is read by PV(j)), QK_b(j+2) only after O_full(PV(j)) (+fence::after: PV(j) reads SF(j) there).
        # 1 = C7a: split only (QK_a right after PV(j) issue, no S_empty).  0 = C6 (SF at +32/+64, N=128 QK).
        # DEBUG PC (C7b): pc_inj 6 = MMA skips the S_empty wait (QK_a(j+2) may overwrite S(j) before the softmax loaded it ->
        # must corrupt); 7 = the softmax arrives S_empty BEFORE its tcgen05.ld of S(j) (-> must corrupt).
        # 8 = MMA skips the O_full(PV(i)) wait before QK_b(i+2) (the SF WAR edge; like SFWAIT=0 it may stay clean in practice
        # because the tensor pipe executes in order -> informational, not a must-corrupt control).
        # C7c (Codex hd256-fwd2-02/03): the schedule is QK_a(i+2), PV(i), [O_full(PV(i))] QK_b(i+2) -> S_full (see mma_split).
        self.split = split
        assert split in (0, 2, 3), "split: 0 (C6), 2 (C7c / C9), 3 (C9r)"
        assert split != 3 or c8 == 1, "C9r needs C8"
        assert split == 0 or (two_wg and not sf_bank), "C7 split requires C6 (two_wg) and the C0 layout"
        assert split == 0 or kv_stage >= 4, "C7c ring order needs >= 4 K/V stages (Codex hd256-fwd2-03 P1: deadlock below)"
        # epi (fork hd256-8k F1a/F1c; host default DM_HD256_FWD_EPI=2): O epilogue through SMEM. 0 = row-per-thread st.global.v4
        # (each warp store hits 32 rows H*512 B apart). 1 = each correction thread writes its bf16 row (same mul.rn + cvt bits)
        # into a 528 B-strided staging row in the (by then idle) sK ring, bar.warp.sync, then each warp copies ITS OWN 32 rows
        # out row by row (lane l -> 16 B chunk l: one 512 B contiguous store per instruction). Values bitwise; store path only.
        # 2 = 1 + TMEM chunk prefetch (F1c). 8 = DEBUG positive control (deterministic): copy out before staging (stale ring
        # bytes -> O must be wrong, LSE unchanged). 9 = DEBUG PC (racy): copy the NEIGHBOUR warp's rows without a CTA barrier.
        self.epi = epi
        # lpt (fork hd256-8k L1; host default DM_HD256_FWD_LPT=1): causal CTA order = GLOBAL longest-first. Linear CTA id u
        # (x fastest) -> m_block = n_m-1 - u // (H*B), (head, batch) from u % (H*B). Each CTA still computes exactly one
        # (m_block, head, batch) with the same code (bitwise); only which CTA id gets which row block changes.
        # False = the per-(head, batch) longest-first order (m_block = n_m-1-bidx, head = bidy, batch = bidz).
        self.lpt = lpt
        # tsplit (fork hd256-fwd5 candidate T, host DM_HD256_FWD_TSPLIT; causal only, lpt off): "split tail" CTA order. With
        # Tm = min(T, n_m) and n_hb = H*B: linear CTA ids u < n_hb*(n_m-Tm) keep the per-(head, batch) longest-first order
        # (pair = u // (n_m-Tm), m_block = n_m-1 - u % (n_m-Tm), head = pair % H, batch = pair // H); the last n_hb*Tm ids
        # are the short items m_block < Tm of ALL pairs, globally longest-first (v = u - n_hb*(n_m-Tm): m_block = Tm-1 - v // n_hb,
        # pair = v % n_hb).  A bijection onto (m_block, head, batch); each CTA computes its item with the same code (bitwise).
        # Removes the end-of-grid tail (long CTAs of the last pairs launched last) while the bulk keeps its L2 locality
        # (global LPT, L1, lost it).  DEBUG PC 18 (non-bijective map: tail m_block shifted by one -> m_block 0 rows never written,
        # m_block Tm duplicated) -> the hole must be caught (pc_fwdmx poisons the output allocation first).
        assert tsplit == 0 or (is_causal and not lpt), "tsplit: causal, lpt off"
        self.tsplit = tsplit
        # ord_td (fork hd256-ordinv, site FW15 of kda/hd256/ordinv-hd256.md; opt-in DM_HD256_ORDT=1): tcgen05 fence pair around
        # the TMEM teardown barrier (users: fence::before_thread_sync before their arrive; allocator: before + after around its
        # arrive_and_wait, then dealloc).  Host default True (DM_HD256_ORDT); False = the pre-fix code (32e0ecd PTX), debug only.
        self.ord_td = ord_td
        # C6 (two_wg, default; Codex hd256-fwd-03, ncu run hd256-fwd-r3): TWO softmax warpgroups by tile parity - WG0 (warps 0-3)
        # owns tiles j = 0, 2, ... (S slot 0), WG1 (warps 4-7) tiles 1, 3, ... (slot 1). Per tile the m-independent prefix (S load,
        # block maxima, kb, bias, FFMA2, ex2, e4m3 codes -> sP, block sums) runs without the running state; the ordered tail
        # (m lazy rule, acc_scale, SFA/w, SF store, P_full, stats, l update) waits the previous tile's (m, l) from the other WG
        # through an SMEM mailbox (state_full[s], 128 arrivals). Every value is the 1-WG op sequence (bitwise).
        # PC (DEBUG): 4 = consumer skips the mailbox wait (stale state) -> must corrupt; 5 = producer arrives before writing.
        assert not (two_wg and sf_bank), "two_wg requires the C0 layout"
        self.two_wg = two_wg
        # C5 (sf_bank, opt-in: REJECTED, slower than C0 - loses the S ping-pong overlap; Codex hd256-fwd-02 option 1): ONE S buffer [0,128) released by the softmax right after its
        # tcgen05.ld (S_empty), SF in two dedicated banks [128 + 32 b, +32) (b = tile parity: SFA +0..3, SFB +8..15), O [256,512).
        # QK never touches SF columns, so there is no PV -> QK WAR; QK(j+1) waits S_empty(j) only. sf_bank=False = C0 layout.
        self.sf_bank = sf_bank
        # DEBUG positive control for the SF hand-off (default 0 = compiled out, bitwise-neutral):
        #  1: right after issuing PV(i), the MMA warp overwrites SFA(i) of lanes 0-31 (tcgen05.st, byte 0x80 = 2^1) WITHOUT
        #     waiting for PV(i) -> the unsafe class "tcgen05 write of PV's SF after PV issue, before PV completion" (what
        #     QK(i+2) does with SFWAIT=0); must corrupt rows 0-31 of some tiles if the window is real and the stress sees it.
        #  2: the same write AFTER waiting O_full(PV(i)) + fence::after (the C0 protocol) -> must stay bitwise clean.
        #  3 (sf_bank only): the MMA warp skips the S_empty wait -> QK(j+1) may overwrite S(j) during the softmax load (unsafe).
        self.pc_inj = pc_inj
        self.sf_wait = sf_wait   # False = DEBUG/A-B only: the pre-C0 schedule without the PV(j-2) -> QK(j) completion wait
        self.is_causal = is_causal
        self.qhead_per_kvhead = qhead_per_kvhead
        self.m_block_size = 128
        self.n_block_size = 128
        self.head_dim = 256
        self.kv_stage = kv_stage
        self.rescale_T = rescale_T
        self.mma_tiler_qk = (128, 128, 256)
        self.mma_tiler_pv = (128, 256, 128)
        self.softmax_warp_ids = (0, 1, 2, 3)
        self.correction_warp_ids = (8, 9, 10, 11)
        self.mma_warp_id = 12
        self.load_warp_id = 14
        self.idle_warp_ids = (4, 5, 6, 7, 13, 15)
        self.threads_per_cta = 512
        self.num_regs_softmax = 256
        self.num_regs_correction = 160
        self.num_regs_other = 48
        if two_wg:
            self.softmax_warp_ids = (0, 1, 2, 3, 4, 5, 6, 7)
            self.idle_warp_ids = (13, 15)
            self.num_regs_softmax = 192
            self.num_regs_correction = 80
        if self.c11:
            self.num_regs_softmax = 184          # prefix only (no running state)
            self.num_regs_correction = 96        # + ordered tail state; 184*256 + 96*128 + 48*128 = 65536
        if self.c13:
            self.idle_warp_ids = (13,)           # warp 15 = PV issuer (decreases its own registers)
        self.buffer_align_bytes = 1024
        self.tmem_o = 256
        self.sfa_col = 32
        self.sfb_col = 64
        self.sfa_off = 0 if sf_bank else self.sfa_col
        self.sfb_off = 8 if sf_bank else self.sfb_col
        if split:
            self.sfa_off, self.sfb_off = 112, 120

    @cute.jit
    def __call__(self, mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor, mO: cute.Tensor, mLSE: cute.Tensor,
                 mQS: cute.Tensor, mKVW: cute.Tensor, stream, mDbg=None):
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
        self.op_qa = cute.make_mma_atom(sm100_utils_basic.make_trivial_tiled_mma(
            self.q_dtype, self.q_dtype, K, K, Float32, cta_group, (128, 112)).op).op
        self.op_qb = cute.make_mma_atom(sm100_utils_basic.make_trivial_tiled_mma(
            self.q_dtype, self.q_dtype, K, K, Float32, cta_group, (128, 16)).op).op
        self.op_q128 = cute.make_mma_atom(tiled_mma_qk.op).op
        cluster_layout_vmnk = cute.tiled_divide(cute.make_layout((1, 1, 1)), (tiled_mma_qk.thr_id.shape,))
        sQ_layout = sm100_utils_basic.make_smem_layout_a(tiled_mma_qk, self.mma_tiler_qk, self.q_dtype, 1)
        sK_layout = sm100_utils_basic.make_smem_layout_b(tiled_mma_qk, self.mma_tiler_qk, self.q_dtype, self.kv_stage)
        sV_layout = sm100_utils_basic.make_smem_layout_b(tiled_mma_pv, self.mma_tiler_pv, self.q_dtype, self.kv_stage)
        sP_layout = sm100_utils_basic.make_smem_layout_a(tiled_mma_pv, self.mma_tiler_pv, self.q_dtype, self.n_sp)
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
        assert self.epi == 0 or sKV_size * (self.q_dtype.width // 8) >= 128 * 528, "F1a O staging needs 66 KiB of the K/V ring"

        @cute.struct
        class SharedStorage:
            mbar_q: cute.struct.MemRange[Int64, 2]
            mbar_kv: cute.struct.MemRange[Int64, self.kv_stage * 2]
            # raw: 0,1 S_full[slot] | 2,3 P_full[slot] | 4 O_full | 5 O_ready | 6,7 stats_full[slot] | 8 final_full
            mbar_dm: cute.struct.MemRange[Int64, self.n_mbar_dm]   # + 9 S_empty (sf_bank), 12/13 S_empty[slot] (C7b); C10 14..24
            tmem_holding_buf: Int32
            sStats: cute.struct.MemRange[Float32, self.n_stats]   # + mailbox state_m[2][128], state_l[2][128] (+ C8 part[2][4][128])
            sQ: cute.struct.Align[cute.struct.MemRange[self.q_dtype, cute.cosize(sQ_layout)], self.buffer_align_bytes]
            sK: cute.struct.Align[cute.struct.MemRange[self.q_dtype, sKV_size], self.buffer_align_bytes]
            sP: cute.struct.Align[cute.struct.MemRange[self.q_dtype, cute.cosize(sP_layout)], self.buffer_align_bytes]

        @cute.struct
        class SharedStoragePers:
            mbar_q: cute.struct.MemRange[Int64, 2]
            mbar_kv: cute.struct.MemRange[Int64, self.kv_stage * 2]
            mbar_dm: cute.struct.MemRange[Int64, self.n_mbar_dm]
            clc_resp: cute.struct.Align[cute.struct.MemRange[Int64, 4], 16]      # C14: 2 x 16-B try_cancel responses
            tmem_holding_buf: Int32
            sStats: cute.struct.MemRange[Float32, self.n_stats]
            sQ: cute.struct.Align[cute.struct.MemRange[self.q_dtype, cute.cosize(sQ_layout)], self.buffer_align_bytes]
            sK: cute.struct.Align[cute.struct.MemRange[self.q_dtype, sKV_size], self.buffer_align_bytes]
            sP: cute.struct.Align[cute.struct.MemRange[self.q_dtype, cute.cosize(sP_layout)], self.buffer_align_bytes]

        self.shared_storage = SharedStoragePers if const_expr(self.pers) else SharedStorage
        assert SharedStorage.size_in_bytes() <= 232448, f"SMEM {SharedStorage.size_in_bytes()}"
        n_m = cute.ceil_div(cute.size(mQt.shape[0]), self.m_block_size)
        grid = (n_m, cute.size(mQt.shape[2]), cute.size(mQt.shape[3]))
        kargs = [mQt, mKt, mVt, mO_raw, mLSE, mQS, mKVW, tma_atom_Q, tma_atom_K, tma_atom_V,
                 sQ_layout, sK_layout, sV_layout, sP_layout, tiled_mma_qk, tiled_mma_pv]
        if const_expr(self.trace):
            kargs.append(mDbg)
        tsd = None
        if const_expr(self.tsplit > 0):
            # hd256-fwd5 T2: host-side FastDivmod divisors for the split-tail map (the in-kernel 32-bit divisions cost ~0.35 us per CTA
            # before the CTA barrier; trace analysis/gap_4014d81.txt).  tm = min(T, n_m), nb = n_m - tm (divisor max(nb, 1): only
            # used when nb > 0), n_hb = H * B, H.
            n_h = cute.size(mQt.shape[2])
            n_hb = n_h * cute.size(mQt.shape[3])
            tm = cutlass.min(Int32(self.tsplit), n_m)
            nb = n_m - tm
            tsd = (FastDivmodDivisorV2(cutlass.max(nb, Int32(1))), FastDivmodDivisorV2(n_hb), FastDivmodDivisorV2(n_h), tm, nb)
        if const_expr(not self.trace):
            kargs.append(None)
        kargs.append(tsd)
        self.kernel(*kargs).launch(
            grid=grid, block=[self.threads_per_cta, 1, 1], stream=stream, min_blocks_per_mp=1)

    @cute.kernel
    def kernel(self, mQ, mK, mV, mO, mLSE, mQS, mKVW, tma_atom_Q, tma_atom_K, tma_atom_V,
               sQ_layout: cute.ComposedLayout, sK_layout: cute.ComposedLayout, sV_layout: cute.ComposedLayout,
               sP_layout: cute.ComposedLayout, tiled_mma_qk: cute.TiledMma, tiled_mma_pv: cute.TiledMma, mDbg=None, tsd=None):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            for a in (tma_atom_Q, tma_atom_K, tma_atom_V):
                cpasync.prefetch_descriptor(a)
        bidx, bidy, bidz = cute.arch.block_idx()
        n_m = cute.arch.grid_dim()[0]
        m_block = n_m - 1 - bidx if const_expr(self.is_causal) else bidx
        head_idx = bidy
        batch_idx = bidz
        if const_expr(self.is_causal and self.lpt):
            n_h = cute.arch.grid_dim()[1]
            n_hb = n_h * cute.arch.grid_dim()[2]
            u = bidx + n_m * (bidy + n_h * bidz)
            m_block = n_m - 1 - u // n_hb
            head_idx = (u % n_hb) % n_h
            batch_idx = (u % n_hb) // n_h
        if const_expr(self.tsplit > 0):
            nb_div, hb_div, h_div, tm, nb = tsd             # host FastDivmod (T2): nb = n_m - tm, n_hb = H*B, H
            n_h = cute.arch.grid_dim()[1]
            n_hb = n_h * cute.arch.grid_dim()[2]
            u = bidx + n_m * (bidy + n_h * bidz)
            pair = Int32(0)
            if u < n_hb * nb:
                pair, r = divmod(u, nb_div)                 # bulk: per-pair longest-first over m_block >= tm
                m_block = n_m - 1 - r
            else:
                q, r = divmod(u - n_hb * nb, hb_div)        # tail: m_block < tm, globally longest-first, pair fastest
                pair = r
                m_block = tm - 1 - q
                if const_expr(self.pc_inj == 18):
                    m_block = m_block + 1                   # DEBUG PC 18: non-bijective map (hole at m_block 0; in bounds)
                    if m_block > n_m - 1:
                        m_block = n_m - 1
            batch_idx, head_idx = divmod(pair, h_div)
        kv_head_idx = head_idx // self.qhead_per_kvhead
        n_block_max = m_block + 1 if const_expr(self.is_causal) else n_m
        cnt = n_block_max
        if const_expr(self.trace):
            cta_lin = bidx + n_m * (bidy + cute.arch.grid_dim()[1] * bidz)
            self._tb = Int64(mDbg.iterator.toint()) + Int64(cta_lin) * Int64(mDbg.shape[1] * TR_E * 8)

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
            dm_utils.mbar_init(mbar + 0, 1)
            dm_utils.mbar_init(mbar + 8, 1)
            dm_utils.mbar_init(mbar + 16, 256 if self.c8 else 128)     # P_full[0] (C8: both WGs)
            dm_utils.mbar_init(mbar + 24, 256 if self.c8 else 128)     # P_full[1]
            dm_utils.mbar_init(mbar + 32, 1)
            dm_utils.mbar_init(mbar + 40, 128)
            dm_utils.mbar_init(mbar + 48, 128)
            dm_utils.mbar_init(mbar + 56, 128)
            dm_utils.mbar_init(mbar + 64, 128)
            dm_utils.mbar_init(mbar + 72, 128)
            dm_utils.mbar_init(mbar + 80, 128)     # state_full[0] (two_wg)
            dm_utils.mbar_init(mbar + 88, 128)     # state_full[1]
            dm_utils.mbar_init(mbar + 96, 256 if (self.sempty and not self.c12) else 128)   # S_empty[0] (C9/C10: both WGs; C12 owner)
            dm_utils.mbar_init(mbar + 104, 256 if (self.sempty and not self.c12) else 128)  # S_empty[1]
            if const_expr(self.c13):
                for k in cutlass.range_constexpr(2):
                    dm_utils.mbar_init(mbar + 200 + 8 * k, 1)              # O_done2[parity]: PV(j) complete (warp 15 commit)
            if const_expr(self.pers):
                dm_utils.mbar_init(mbar + 216, 128)                        # EPI_done: staged O copied out (correction WG)
                for k in cutlass.range_constexpr(2):
                    dm_utils.mbar_init(mbar + 224 + 8 * k, 1)              # clc_full[s]: expect_tx(16) + try_cancel response
                    dm_utils.mbar_init(mbar + 240 + 8 * k, 15)             # clc_empty[s]: one ack per consuming warp
            if const_expr(self.c10):
                for k in cutlass.range_constexpr(2):
                    dm_utils.mbar_init(mbar + 112 + 8 * k, 128)            # W1L[slot]: WG1 loaded S(t)
                for k in cutlass.range_constexpr(3):
                    dm_utils.mbar_init(mbar + 128 + 8 * k, 256 if self.c12 else 384 if self.c11 else 256)   # P_full3[b]
                    dm_utils.mbar_init(mbar + 152 + 8 * k, 128)            # stats_full3[b]
                    dm_utils.mbar_init(mbar + 176 + 8 * k, 256 if (self.c11 and not self.c12) else 128)   # partial_full3[b]
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
        if const_expr(self.trace and self.c13):
            if warp_idx == 13:                     # C13: one observer, pipe order QK(0), QK(1), [QK(j+2)], PV(j)
                for j0 in cutlass.range_constexpr(2):
                    if j0 < cnt:
                        self.tr(j0, 28, tr_wait_clock(mbar + 8 * j0, 0))
                for j in cutlass.range(cnt, unroll=1):
                    if j + 2 < cnt:
                        self.tr(j + 2, 28, tr_wait_clock(mbar + 8 * (j & 1), ((j + 2) >> 1) & 1))
                    self.tr(j, 29, tr_wait_clock(mbar + 32, j & 1))
        elif const_expr(self.trace and self.two_wg):
            # DEBUG observers: warp 13 records each S_full(j) completion (QK(j) done), warp 15 each O_full (PV(j) done)
            if warp_idx == 13:
                for j in cutlass.range(cnt, unroll=1):
                    self.tr(j, 28, tr_wait_clock(mbar + 8 * (j & 1), (j >> 1) & 1))
            if warp_idx == 15:
                for j in cutlass.range(cnt, unroll=1):
                    self.tr(j, 29, tr_wait_clock(mbar + 32, j & 1))

        if const_expr(self.pers):
            self.roles_pers(warp_idx, mQ, mK, mV, mO, mLSE, mQS, mKVW, tma_atom_Q, tma_atom_K, tma_atom_V, sQ, sK, sV, sP,
                            thr_mma_qk, thr_mma_pv, tiled_mma_qk, tiled_mma_pv, acc_pv, pipeline_q, pipeline_kv, mbar, stats,
                            sp_base, tmem, tmem_alloc_barrier, Int32(storage.clc_resp.data_ptr().toint()), tsd)
        else:
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
                if const_expr(self.split != 0 or self.c10):
                    self.load_split(tma_atom_Q, tma_atom_K, tma_atom_V, tQgQ, tQsQ, tKgK, tKsK, tVgV, tVsV, pipeline_q, pipeline_kv,
                                    q_state, n_block_max, cnt)
                else:
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
                # Ring WAR edge of the F1a/F1c epilogue (accuracy-audit 12:2x Q1; cherry-picked from hd256-8k2 252ece6): the loads above
                # are this CTA's LAST writes to the K/V ring (non-persistent CTA, 2*cnt TMA loads, all consumed before O_full(PV(last)),
                # which the correction WG waits before staging O in the ring).
                if const_expr(self.epi == 10):
                    # DEBUG POSITIVE CONTROL ONLY: an unordered ring writer ("ring released early"): wait final_full, let the last PV
                    # finish, poison the staging rows while the correction WG (epi 10) holds its staged rows -> O must be corrupted.
                    dm_utils.mbar_wait_parity(mbar + 64, Int32(0))
                    for _ in cutlass.range(8, unroll=1):
                        dm_utils.nanosleep_1us()
                    skb = Int32(sK.iterator.toint())
                    ln = cute.arch.lane_idx()
                    pz = Int32(-1)
                    for r in cutlass.range((128 * 528) // (16 * 32), unroll=1):
                        st_shared_v4_b32(skb + Int32(16) * (r * 32 + ln), pz, pz, pz, pz)

            if const_expr(self.c13):
                if warp_idx == 15:
                    cute.arch.setmaxregister_decrease(self.num_regs_other)
                    self.mma_c13_pv(tiled_mma_pv, sV, sP, acc_pv, pipeline_kv, mbar, cnt)
            if warp_idx == self.mma_warp_id:
                cute.arch.setmaxregister_decrease(self.num_regs_other)
                tmem.allocate(512)
                tmem_alloc_barrier.arrive_and_wait_unaligned()
                tmem_ptr = tmem.retrieve_ptr(Float32)
                if const_expr(self.trace):
                    self.tr(-1, 0, Int64(m_block))
                    self.tr(-1, 1, Int64(cnt))
                    self.tr(-1, 2, Int64(dm_utils.smid()))
                    self.tr(-1, 3, dm_utils.globaltimer())
                    self.tr(-1, 4)
                    self.tr(-1, 7, Int64(head_idx))
                    self.tr(-1, 8, Int64(batch_idx))
                self.mma(tiled_mma_qk, tiled_mma_pv, sQ, sK, sV, sP, acc_qk, acc_pv, pipeline_q, pipeline_kv, mbar, cnt)
                tmem.relinquish_alloc_permit()
                if const_expr(self.ord_td):
                    dm_utils.tc_fence_before()
                tmem_alloc_barrier.arrive_and_wait_unaligned()
                if const_expr(self.ord_td):
                    dm_utils.tc_fence_after()
                tmem.free(tmem_ptr)

            if warp_idx <= self.softmax_warp_ids[-1]:
                cute.arch.setmaxregister_increase(self.num_regs_softmax)
                tmem_alloc_barrier.arrive_and_wait_unaligned()
                if const_expr(self.c12):
                    self.softmax4(mbar, sp_base, stats, mQS, mKVW, m_block, head_idx, kv_head_idx, batch_idx, n_block_max, cnt)
                elif const_expr(self.two_wg):
                    if const_expr(self.c8):
                        self.softmax3(mbar, stats, sp_base, mLSE, mQS, mKVW, m_block, head_idx, kv_head_idx, batch_idx,
                                      n_block_max, cnt)
                    else:
                        self.softmax2(acc_qk, mbar, stats, sp_base, mLSE, mQS, mKVW, m_block, head_idx, kv_head_idx,
                                  batch_idx, n_block_max, cnt)
                else:
                    self.softmax(acc_qk, mbar, stats, sp_base, mLSE, mQS, mKVW, m_block, head_idx, kv_head_idx,
                                 batch_idx, n_block_max, cnt)
                if const_expr(self.ord_td):
                    dm_utils.tc_fence_before()
                tmem_alloc_barrier.arrive_unaligned()

            if warp_idx >= self.correction_warp_ids[0] and warp_idx <= self.correction_warp_ids[-1]:
                if const_expr(self.num_regs_correction > 128):
                    cute.arch.setmaxregister_increase(self.num_regs_correction)
                else:
                    cute.arch.setmaxregister_decrease(self.num_regs_correction)
                tmem_alloc_barrier.arrive_and_wait_unaligned()
                if const_expr(self.c11):
                    self.correction_c11(mbar, stats, mO, mLSE, mKVW, m_block, head_idx, kv_head_idx, batch_idx, n_block_max, cnt,
                                        Int32(sK.iterator.toint()))
                else:
                    self.correction(mbar, stats, mO, m_block, head_idx, batch_idx, cnt, Int32(sK.iterator.toint()))
                if const_expr(self.trace):
                    self.tr(-1, 5, None, warp_idx == 8)
                    self.tr(-1, 6, dm_utils.globaltimer(), warp_idx == 8)
                if const_expr(self.ord_td):
                    dm_utils.tc_fence_before()
                tmem_alloc_barrier.arrive_unaligned()
        return

    # -----------------------------------------------------------------------------------------------------------------
    def tr(self, j, ev: int, val=None, en=True):
        """DEBUG trace (self.trace only): lane 0 of the calling warp stores val (default %clock64) at row j + 1, event ev.
        Events (row j + 1 = tile j): MMA warp 0 C0 wait start (O_full(PV(j-2)) before QK(j)) | 1 C0 wait end | 2 K(j) ready |
        3 QK(j) issued+committed | 4 PV(j) start | 5 V(j) ready | 6 P_full(j) seen | 7 O_ready(j) seen | 8 PV(j) issued |
        9 QK_a(j) issued (C7c) | softmax WG leader (warp 4*wg) 10 S_full wait start | 11 S_full seen | 12 prefix done |
        13 mailbox seen | 14 P_full arrive | 15 state published | per softmax warp q = warp % 4: 16+q S loaded, 20+q P_full
        arrive | correction warp 8: 24 stats(j) seen | 25 O_full(PV(j-1)) seen | 26 O_ready(j) arrive | observers: 28 S_full(j)
        completion (warp 13) | 29 O_full(PV(j)) completion (warp 15).  Header row 0: 0 m_block 1 cnt 2 smid 3 globaltimer
        start 4 clock start (MMA warp) 5 clock end 6 globaltimer end (correction warp 8) 7 head 8 batch."""
        if const_expr(self.trace):
            v = dm_utils.clock64() if val is None else val
            tr_st(self._tb + Int64(Int32(j) + 1) * Int64(TR_E * 8) + Int64(ev * 8), v, Int32(en))

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
        if const_expr(self.c13):
            self.mma_c13_qk(tSrQ, tSrK, sQ, sK, pipeline_q, pipeline_kv, mbar, cnt, q_state)
        elif const_expr(self.c10):
            self.mma_c10(atom_qk, tSrQ, tSrK, tOrP, tOrV, sQ, sK, sP, sV, acc_qk, acc_pv, pv_op, pipeline_q, pipeline_kv, mbar, cnt,
                         q_state)
        elif const_expr(self.split == 3):
            self.mma_c9r(tSrQ, tSrK, tOrP, tOrV, sQ, sK, sP, sV, acc_pv, pv_op, pipeline_q, pipeline_kv, mbar, cnt, q_state)
        elif const_expr(self.split != 0):
            self.mma_split(tSrQ, tSrK, tOrP, tOrV, sQ, sK, sP, sV, acc_qk, acc_pv, pv_op, pipeline_q, pipeline_kv, mbar, cnt,
                           q_state)
        else:
            self.mma_c6(atom_qk, tSrQ, tSrK, tOrP, tOrV, sP, sV, acc_qk, acc_pv, pv_op, pipeline_q, pipeline_kv, mbar, cnt,
                        kv_state, q_state, sQ, sK)

    @cute.jit
    def mma_c6(self, atom_qk, tSrQ, tSrK, tOrP, tOrV, sP, sV, acc_qk, acc_pv, pv_op, pipeline_q, pipeline_kv, mbar, cnt,
               kv_state, q_state, sQ, sK):
        pipeline_q.consumer_wait(q_state)
        pipeline_kv.consumer_wait(kv_state)
        self.tr(0, 2)
        self.qk128(atom_qk, acc_qk, tSrQ, tSrK, sQ, sK, kv_state.index, Int32(0))
        with cute.arch.elect_one():
            dm_utils.tc_commit(mbar + 0)
        self.tr(0, 3)
        pipeline_kv.consumer_release(kv_state)
        kv_state.advance()
        for i in cutlass.range(cnt - 1, unroll=1):
            j = i + 1
            slot = j & 1
            # slot j&1 holds SF(j-2), read by PV(j-2): PTX does not order PV(j-2) (mxf8f6f4, acc O) before QK(j) (f8f6f4,
            # acc S) by issue order (Codex fwd-01 / PTX 9.7.16.6) -> wait PV(j-2)'s completion. Phase-safe: PV(j-3) is complete
            # (PV(j-2) waited O_ready(j-2), arrived after O_full(PV(j-3))) and PV(j-1) is not issued yet.
            if const_expr(self.sf_bank):
                # single S buffer: the softmax has loaded S(j-1) (S_empty(j-1), parity (j-1)&1)
                if const_expr(self.pc_inj != 3):
                    dm_utils.mbar_wait_parity(mbar + 72, (j - 1) & 1)
                    dm_utils.tc_fence_after()
                slot = Int32(0)
            elif const_expr(self.sf_wait) and i >= 1:
                if const_expr(self.g1 & 4):
                    pipeline_kv.consumer_wait(kv_state)                 # G1c: K(j) first (no dependency on PV(j-2))
                self.tr(j, 0)
                dm_utils.mbar_wait_parity(mbar + 32, (j - 2) & 1)
                dm_utils.tc_fence_after()
                self.tr(j, 1)
            if const_expr((self.g1 & 4) and self.sf_wait and not self.sf_bank):
                if i < 1:
                    pipeline_kv.consumer_wait(kv_state)
            else:
                pipeline_kv.consumer_wait(kv_state)
            self.tr(j, 2)
            self.qk128(atom_qk, acc_qk, tSrQ, tSrK, sQ, sK, kv_state.index, slot)
            with cute.arch.elect_one():
                dm_utils.tc_commit(mbar + 8 * slot)      # S_full (sf_bank: always mbar+0, slot 0)
            self.tr(j, 3)
            pipeline_kv.consumer_release(kv_state)
            kv_state.advance()
            self.mma_pv(pv_op, acc_pv, tOrP, tOrV, sP, sV, pipeline_kv, kv_state, mbar, i)
            kv_state.advance()
        self.mma_pv(pv_op, acc_pv, tOrP, tOrV, sP, sV, pipeline_kv, kv_state, mbar, cnt - 1)
        kv_state.advance()
        pipeline_q.consumer_release(q_state)

    @cute.jit
    def load_split(self, tma_atom_Q, tma_atom_K, tma_atom_V, tQgQ, tQsQ, tKgK, tKsK, tVgV, tVsV, pipeline_q, pipeline_kv, q_state,
                   n_block_max, cnt):
        """C7c load order = the MMA consumption order: K0, Q, K1, then for i = 0..cnt-1: [K(i+2) if i+2 < cnt], V(i).
        Tile j = key block n_block_max-1-j.  Sequence number n -> stage n % S, producer phase ((n // S) & 1) ^ 1."""
        self.load_one(tma_atom_K, tKgK, tKsK, pipeline_kv, n_block_max - 1, Int32(0))
        pipeline_q.producer_acquire(q_state)
        cute.copy(tma_atom_Q, tQgQ, tQsQ[None, 0], tma_bar_ptr=pipeline_q.producer_get_barrier(q_state))
        if cnt > 1:
            self.load_one(tma_atom_K, tKgK, tKsK, pipeline_kv, n_block_max - 2, Int32(1))
        for i in cutlass.range(cnt, unroll=1):
            if i + 2 < cnt:
                self.load_one(tma_atom_K, tKgK, tKsK, pipeline_kv, n_block_max - 3 - i, self.kseq(i + 2))
            self.load_one(tma_atom_V, tVgV, tVsV, pipeline_kv, n_block_max - 1 - i, self.vseq(i, cnt))

    @cute.jit
    def load_one(self, atom, tg, ts, pipeline_kv, blk, seq):
        from flash_attn.cute.pipeline import _make_state
        st = _make_state(seq % self.kv_stage, ((seq // self.kv_stage) & 1) ^ 1)
        pipeline_kv.producer_acquire(st)
        cute.copy(atom, tg[None, blk], ts[None, st.index], tma_bar_ptr=pipeline_kv.producer_get_barrier(st))

    @cute.jit
    def qk_part(self, part: cutlass.Constexpr, tSrQ, tSrK, sQ, sK, acc_qk, stage, slot):
        """C7: QK_a (part 0: N112, S cols 0..111) or QK_b (part 1: N16, cols 112..127, K rows 112..127 = +896 desc units)."""
        base = Int32(slot) * 128           # TMEM S slot column (allocation base 0, as mma_qk / self.tmem_o)
        if const_expr(part == 0):
            gemm_qk_ptx(self.op_qa, base, tSrQ[None, None, None, 0], tSrK[None, None, None, stage], sQ[None, None, None, 0],
                        sK[None, None, None, stage], 0)
        else:
            gemm_qk_ptx(self.op_qb, base + 112, tSrQ[None, None, None, 0], tSrK[None, None, None, stage],
                        sQ[None, None, None, 0], sK[None, None, None, stage], (112 * 128) // 16)

    @cute.jit
    def mma_split(self, tSrQ, tSrK, tOrP, tOrV, sQ, sK, sP, sV, acc_qk, acc_pv, pv_op, pipeline_q, pipeline_kv, mbar, cnt,
                  q_state):
        """C7c MMA program (Codex hd256-fwd2-02 repaired schedule; see __init__ split).
            prologue: QK(0) = QK_a(0)+QK_b(0) -> S_full[0]; QK(1) -> S_full[1]
            for i = 0..cnt-1:  [i+2 < cnt: S_empty(i) -> QK_a(i+2)]   (slot i&1: S(i) already in the softmax's registers)
                               P_full(i), O_ready(i) -> PV(i) -> O_full
                               [i+2 < cnt: O_full(PV(i)) -> QK_b(i+2) -> S_full[i&1], release K(i+2)]
        S_full(i+2)'s commit tracks only QK_a(i+2), PV(i) (complete) and QK_b(i+2): no unrelated MMA on its path (C7b's
        order made S_full(t+1) track QK_a(t+2): Codex hd256-fwd2-03).  Ring order K0, K1, then per i: [K(i+2)], V(i)."""
        pipeline_q.consumer_wait(q_state)
        S = self.kv_stage
        for j0 in cutlass.range_constexpr(2):
            if j0 < cnt:
                ks0 = self.kseq(Int32(j0))
                self.kv_wait(pipeline_kv, ks0)
                self.qk_part(0, tSrQ, tSrK, sQ, sK, acc_qk, ks0 % S, Int32(j0))
                self.qk_part(1, tSrQ, tSrK, sQ, sK, acc_qk, ks0 % S, Int32(j0))
                with cute.arch.elect_one():
                    dm_utils.tc_commit(mbar + 8 * j0)                                     # S_full[j0]
                pipeline_kv.consumer_release_w_index(ks0 % S)
        for i in cutlass.range(cnt, unroll=1):
            slot = i & 1
            if i + 2 < cnt:
                self.tr(i + 2, 0)
                if const_expr(self.pc_inj != 6):
                    dm_utils.mbar_wait_parity(mbar + 96 + 8 * slot, (i >> 1) & 1)        # S_empty(i): S(i) in registers
                    dm_utils.tc_fence_after()
                self.tr(i + 2, 1)
                ks = self.kseq(i + 2)
                self.kv_wait(pipeline_kv, ks)
                self.tr(i + 2, 2)
                self.qk_part(0, tSrQ, tSrK, sQ, sK, acc_qk, ks % S, slot)                # QK_a(i+2): cols 0..111
                self.tr(i + 2, 9)
            vs = self.vseq(i, cnt)
            self.mma_pv(pv_op, acc_pv, tOrP, tOrV, sP, sV, pipeline_kv, None, mbar, i, vs % S, (vs // S) & 1)
            if i + 2 < cnt:
                if const_expr(self.pc_inj != 8):
                    dm_utils.mbar_wait_parity(mbar + 32, i & 1)                          # O_full(PV(i)): SF(i) read
                    dm_utils.tc_fence_after()
                ks2 = self.kseq(i + 2)
                self.qk_part(1, tSrQ, tSrK, sQ, sK, acc_qk, ks2 % S, slot)               # QK_b(i+2): cols 112..127
                with cute.arch.elect_one():
                    dm_utils.tc_commit(mbar + 8 * slot)                                   # S_full[slot] (tile i+2)
                self.tr(i + 2, 3)
                pipeline_kv.consumer_release_w_index(ks2 % S)
        pipeline_q.consumer_release(q_state)

    @cute.jit
    def mma_c9r(self, tSrQ, tSrK, tOrP, tOrV, sQ, sK, sP, sV, acc_pv, pv_op, pipeline_q, pipeline_kv, mbar, cnt, q_state):
        """C9r MMA program (fork hd256-fwd4; ring order and K/V sequence numbers as C7c: K0, K1, then per i [K(i+2)], V(i)).
            prologue: QK(0) = QK_a(0) + QK_b(0) -> S_full[0]; [cnt > 1] QK_a(1)
            for j = 0..cnt-1:
              [j+1 < cnt]  (j >= 1: O_full(PV(j-1)) + fence::after: PV(j-1) read SF(j-1) in cols 112..127 of slot (j+1)&1)
                           QK_b(j+1) -> S_full[(j+1)&1] (tracks QK_a(j+1) too), release K(j+1)
              [j+2 < cnt]  S_empty(j) + fence::after (both WGs hold S(j)) -> QK_a(j+2) into slot j&1 cols 0..111
              P_full(j), O_ready(j) -> PV(j) -> O_full, release V(j)
        QK_b(j+1) also overwrites S(j-1) cols 112..127: loaded before S_empty(j-1), waited in iteration j-1 before QK_a(j+1).
        Ring: iteration j needs K(j+2) (seq 2j+2, stage freed by K(j)'s release in iteration j-1) and V(j) (seq 2j+3, freed by V(j-2))."""
        pipeline_q.consumer_wait(q_state)
        S = self.kv_stage
        self.kv_wait(pipeline_kv, Int32(0))
        self.qk_part(0, tSrQ, tSrK, sQ, sK, None, Int32(0), Int32(0))
        self.qk_part(1, tSrQ, tSrK, sQ, sK, None, Int32(0), Int32(0))
        with cute.arch.elect_one():
            dm_utils.tc_commit(mbar + 0)                                              # S_full[0] (tile 0)
        pipeline_kv.consumer_release_w_index(Int32(0))
        if cnt > 1:
            self.kv_wait(pipeline_kv, Int32(1))
            self.qk_part(0, tSrQ, tSrK, sQ, sK, None, Int32(1) % S, Int32(1))           # QK_a(1)
        for j in cutlass.range(cnt, unroll=1):
            if j + 1 < cnt:
                if j >= 1:
                    self.tr(j + 1, 0)
                    if const_expr(self.pc_inj != 8):
                        dm_utils.mbar_wait_parity(mbar + 32, (j - 1) & 1)            # O_full(PV(j-1)): SF(j-1) read
                        dm_utils.tc_fence_after()
                    self.tr(j + 1, 1)
                ks1 = self.kseq(j + 1)
                self.qk_part(1, tSrQ, tSrK, sQ, sK, None, ks1 % S, (j + 1) & 1)         # QK_b(j+1): cols 112..127
                with cute.arch.elect_one():
                    dm_utils.tc_commit(mbar + 8 * ((j + 1) & 1))                       # S_full[(j+1)&1] (tile j+1)
                self.tr(j + 1, 3)
                pipeline_kv.consumer_release_w_index(ks1 % S)
            if j + 2 < cnt:
                if const_expr(self.pc_inj != 6):
                    dm_utils.mbar_wait_parity(mbar + 96 + 8 * (j & 1), (j >> 1) & 1)   # S_empty(j): S(j) in registers
                    dm_utils.tc_fence_after()
                ks2 = self.kseq(j + 2)
                self.kv_wait(pipeline_kv, ks2)
                self.tr(j + 2, 2)
                self.qk_part(0, tSrQ, tSrK, sQ, sK, None, ks2 % S, j & 1)              # QK_a(j+2): cols 0..111
                self.tr(j + 2, 9)
            vs = self.vseq(j, cnt)
            self.mma_pv(pv_op, acc_pv, tOrP, tOrV, sP, sV, pipeline_kv, None, mbar, j, vs % S, (vs // S) & 1)
        pipeline_q.consumer_release(q_state)

    @cute.jit
    def mma_c10(self, atom_qk, tSrQ, tSrK, tOrP, tOrV, sQ, sK, sP, sV, acc_qk, acc_pv, pv_op, pipeline_q, pipeline_kv, mbar, cnt,
                q_state):
        """C10 MMA program (see __init__ c10).  Ring order = C7c (K0, K1, then per j [K(j+2)], V(j); kseq / vseq).
            prologue: QK(0) -> S_full[0]; [cnt > 1] QK(1) -> S_full[1]
            for j: [j+2 < cnt: S_empty(j) (both WGs hold S(j)), j >= 1: O_full(PV(j-1)) (SF(j-1) in slot j&1 read), fence::after
                    -> K(j+2) -> QK(j+2) into slot j&1 -> S_full[j&1], release K(j+2)]
                   P_full3[j%3], O_ready(j) -> PV(j) (SF at the C10 address) -> O_full, release V(j)
        Phase safety: the MMA waits O_full(PV(j-1)) before issuing PV(j) (<= j completions, parity (j-1)&1 exact); S_empty(j+2)
        arrivals follow S_full(j+2), committed after this wait of S_empty(j)."""
        pipeline_q.consumer_wait(q_state)
        S = self.kv_stage
        for j0 in cutlass.range_constexpr(2):
            if j0 < cnt:
                self.kv_wait(pipeline_kv, Int32(j0))
                self.qk_ptx(tSrQ, tSrK, sQ, sK, Int32(j0), Int32(j0))
                with cute.arch.elect_one():
                    dm_utils.tc_commit(mbar + 8 * j0)                                   # S_full[j0]
                pipeline_kv.consumer_release_w_index(Int32(j0))
        for j in cutlass.range(cnt, unroll=1):
            if j + 2 < cnt:
                self.tr(j + 2, 0)
                if const_expr(self.pc_inj != 6):
                    dm_utils.mbar_wait_parity(mbar + 96 + 8 * (j & 1), (j >> 1) & 1)   # S_empty(j): S(j) in registers
                else:
                    if j == 0:
                        # DEBUG PC 6 keeps the j == 0 wait: nothing else gates QK(2), so skipping it can complete QK(0) and QK(2)
                        # before the owner's first S_full[0] wait (parity overrun -> deadlock, not corruption); for j >= 1 QK(j+2)
                        # is gated by O_full(PV(j-1)) <- W1L(j) <- the owner's S(j) load, so the skip is a pure data race.
                        dm_utils.mbar_wait_parity(mbar + 96, Int32(0))
                if j >= 1:
                    if const_expr(self.pc_inj != 8):
                        dm_utils.mbar_wait_parity(mbar + 32, (j - 1) & 1)             # O_full(PV(j-1)): SF(j-1) read
                dm_utils.tc_fence_after()
                self.tr(j + 2, 1)
                ks = self.kseq(j + 2)
                self.kv_wait(pipeline_kv, ks)
                self.tr(j + 2, 2)
                self.qk_ptx(tSrQ, tSrK, sQ, sK, ks % S, j & 1)
                with cute.arch.elect_one():
                    dm_utils.tc_commit(mbar + 8 * (j & 1))                              # S_full (tile j+2)
                self.tr(j + 2, 3)
                pipeline_kv.consumer_release_w_index(ks % S)
            vs = self.vseq(j, cnt)
            self.mma_pv(pv_op, acc_pv, tOrP, tOrV, sP, sV, pipeline_kv, None, mbar, j, vs % S, (vs // S) & 1, cnt)
        pipeline_q.consumer_release(q_state)

    @cute.jit
    def qk_ptx(self, tSrQ, tSrK, sQ, sK, stage, slot):
        """C10: QK(j) N128 as raw PTX (= G1b: the N128 QK op's idesc / descriptors, k ascending, zero-init at k 0; bitwise == cute.gemm,
        fwd3 G1 arms) - no stateful MMA atom inside the dynamic prologue branch."""
        gemm_qk_ptx(self.op_q128, Int32(slot) * 128, tSrQ[None, None, None, 0], tSrK[None, None, None, stage],
                    sQ[None, None, None, 0], sK[None, None, None, stage], 0)

    @cute.jit
    def mma_c13_qk(self, tSrQ, tSrK, sQ, sK, pipeline_q, pipeline_kv, mbar, cnt, q_state):
        """C13 warp 12: the QK half of mma_c10; PV(j-1)'s completion via O_done2[(j-1)&1] (parity ((j-1)>>1)&1)."""
        pipeline_q.consumer_wait(q_state)
        S = self.kv_stage
        for j0 in cutlass.range_constexpr(2):
            if j0 < cnt:
                self.kv_wait(pipeline_kv, Int32(j0))
                self.qk_ptx(tSrQ, tSrK, sQ, sK, Int32(j0), Int32(j0))
                with cute.arch.elect_one():
                    dm_utils.tc_commit(mbar + 8 * j0)                                   # S_full[j0]
                pipeline_kv.consumer_release_w_index(Int32(j0))
        for j in cutlass.range(cnt, unroll=1):
            if j + 2 < cnt:
                self.tr(j + 2, 0)
                if const_expr(self.pc_inj != 6):
                    dm_utils.mbar_wait_parity(mbar + 96 + 8 * (j & 1), (j >> 1) & 1)   # S_empty(j)
                else:
                    if j == 0:
                        dm_utils.mbar_wait_parity(mbar + 96, Int32(0))   # DEBUG PC 6: keep the j == 0 wait (as mma_c10; Codex fwd5-01)
                if j >= 1:
                    if const_expr(self.pc_inj != 8):
                        dm_utils.mbar_wait_parity(mbar + 200 + 8 * ((j - 1) & 1), ((j - 1) >> 1) & 1)   # O_done2: PV(j-1) done
                dm_utils.tc_fence_after()
                self.tr(j + 2, 1)
                ks = self.kseq(j + 2)
                self.kv_wait(pipeline_kv, ks)
                self.tr(j + 2, 2)
                self.qk_ptx(tSrQ, tSrK, sQ, sK, ks % S, j & 1)
                with cute.arch.elect_one():
                    dm_utils.tc_commit(mbar + 8 * (j & 1))                              # S_full (tile j+2)
                self.tr(j + 2, 3)
                pipeline_kv.consumer_release_w_index(ks % S)
        pipeline_q.consumer_release(q_state)

    @cute.jit
    def mma_c13_pv(self, tiled_mma_pv, sV, sP, acc_pv, pipeline_kv, mbar, cnt):
        """C13 warp 15: PV(j) for j = 0..cnt-1 (mma_pv: V(j), P_full3, O_ready -> PV -> O_full + O_done2[j&1], release V(j))."""
        tOrP = tiled_mma_pv.make_fragment_A(sP)
        tOrV = tiled_mma_pv.make_fragment_B(sV)
        pv_op = cute.make_mma_atom(tiled_mma_pv.op).op
        S = self.kv_stage
        for j in cutlass.range(cnt, unroll=1):
            vs = self.vseq(j, cnt)
            self.mma_pv(pv_op, acc_pv, tOrP, tOrV, sP, sV, pipeline_kv, None, mbar, j, vs % S, (vs // S) & 1, cnt)

    def sf_col10(self, j, cnt):
        """C10 SFA column of tile j (SFB = +8): slot (j+1)&1 + 64 for j < cnt-1, own slot + 80 for the last tile."""
        c = Int32(Int32(j) + 1 < cnt)
        return c * (((Int32(j) + 1) & 1) * 128 + 64) + (1 - c) * ((Int32(j) & 1) * 128 + 80)

    def sf_col14(self, J, j, cnt):
        """C14 SFA column of tile j (global J): slot (J+1)&1 + 64 unless j is the item's last tile (own slot J&1 + 80)."""
        c = Int32(Int32(j) + 1 < cnt)
        return c * (((Int32(J) + 1) & 1) * 128 + 64) + (1 - c) * ((Int32(J) & 1) * 128 + 80)

    def kseq(self, m):
        """C7c ring sequence number of K(m): K0, K1, then K(m) at 2m-2 (each followed by V(m-2))."""
        lt = Int32(Int32(m) < 2)
        return Int32(m) * lt + (2 * Int32(m) - 2) * (1 - lt)

    def vseq(self, m, cnt):
        """C7c ring sequence number of V(m): 2m+3 while K(m+2) precedes it, else m + cnt (tail of V loads)."""
        c = Int32(Int32(m) + 2 < cnt)
        return (2 * Int32(m) + 3) * c + (Int32(m) + cnt) * (1 - c)

    @cute.jit
    def kv_wait(self, pipeline_kv, seq):
        pipeline_kv.consumer_wait_w_index_phase(seq % self.kv_stage, (seq // self.kv_stage) & 1)
        dm_utils.tc_fence_after()

    @cute.jit
    def qk128(self, atom_qk, acc_qk, tSrQ, tSrK, sQ, sK, stage, slot):
        """QK(j) into S slot `slot`: 8 cute.gemm (C6) or, G1b, the same N128 chain as raw PTX (gemm_qk_ptx, b_off 0)."""
        if const_expr(self.g1 & 2):
            gemm_qk_ptx(self.op_q128, Int32(slot) * 128, tSrQ[None, None, None, 0], tSrK[None, None, None, stage],
                        sQ[None, None, None, 0], sK[None, None, None, stage], 0)
        else:
            self.mma_qk(atom_qk, acc_qk, tSrQ, tSrK, stage, slot)

    @cute.jit
    def mma_qk(self, atom_qk, acc_qk, tSrQ, tSrK, stage, slot):
        tS = cute.make_tensor(acc_qk.iterator + slot * 128, acc_qk.layout)
        for k in cutlass.range_constexpr(cute.size(tSrQ.shape[2])):
            atom_qk.set(tcgen05.Field.ACCUMULATE, k != 0)
            cute.gemm(atom_qk, tS, tSrQ[None, None, k, 0], tSrK[None, None, k, stage], tS)

    @cute.jit
    def mma_pv(self, pv_op, acc_pv, tOrP, tOrV, sP, sV, pipeline_kv, kv_state, mbar, i, kv_idx=None, kv_ph=None, cnt=None):
        slot = i & 1
        pb = slot                     # sP buffer (C10: i % 3)
        stage = kv_idx
        self.tr(i, 4)
        if const_expr(kv_idx is None):
            pipeline_kv.consumer_wait(kv_state)
            stage = kv_state.index
        else:
            pipeline_kv.consumer_wait_w_index_phase(kv_idx, kv_ph)
        self.tr(i, 5)
        if const_expr(self.c10):
            pb = i % 3
            dm_utils.mbar_wait_parity(mbar + 128 + 8 * pb, (i // 3) & 1)  # P_full3[i%3]: P(i) codes + SF(i) written
            if const_expr(self.pc_inj == 16):
                pb = i & 1                                                  # DEBUG PC 16: sP ring depth 2, PV(i) issued late
                for _d in cutlass.range_constexpr(20):
                    dm_utils.nanosleep_1us()
        else:
            dm_utils.mbar_wait_parity(mbar + 16 + 8 * slot, (i >> 1) & 1)     # P_full[slot]: P(i) codes + SF(i) written
        self.tr(i, 6)
        if i >= 1:
            dm_utils.mbar_wait_parity(mbar + 40, (i - 1) & 1)            # O_ready(i): O rescaled (or not) after PV(i-1)
        self.tr(i, 7)
        dm_utils.tc_fence_after()
        sf = (128 + 32 * slot) if const_expr(self.sf_bank) else slot * 128
        sfa_c = sf + self.sfa_off
        sfb_c = sf + self.sfb_off
        if const_expr(self.c10):
            sfa_c = self.sf_col10(i, cnt)
            sfb_c = sfa_c + 8
        dm_utils.gemm_ptx_bs(
            pv_op, self.tmem_o, tOrP[None, None, None, pb], tOrV[None, None, None, stage],
            sP[None, None, None, pb], sV[None, None, None, stage], sfa_c, sfb_c,
            tuple((0, k) for k in range(4)), tuple((0, k) for k in range(4)), zero_init=(i == 0))
        with cute.arch.elect_one():
            dm_utils.tc_commit(mbar + 32)                                   # O_full(i)
            if const_expr(self.c13):
                dm_utils.tc_commit(mbar + 200 + 8 * (i & 1))                # O_done2[i&1] (C13: QK warp's PV(i) completion)
        self.tr(i, 8)
        if const_expr(self.pc_inj in (1, 2)):
            if const_expr(self.pc_inj == 2):
                dm_utils.mbar_wait_parity(mbar + 32, i & 1)
                dm_utils.tc_fence_after()
            lane0 = (cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4) * 32
            for r in cutlass.range_constexpr(4):
                tmem_st1((lane0 << 16) + sfa_c + r, Int32(-2139062144))     # 0x80808080
            tmem_wait_st()
        if const_expr(kv_idx is None):
            pipeline_kv.consumer_release(kv_state)
        else:
            pipeline_kv.consumer_release_w_index(kv_idx)

    # -----------------------------------------------------------------------------------------------------------------
    @cute.jit
    def softmax(self, acc_qk, mbar, stats, sp_base, mLSE, mQS, mKVW, m_block, head_idx, kv_head_idx, batch_idx,
                n_block_max, cnt):
        tidx = cute.arch.thread_idx()[0] % 128
        tSAcc = acc_qk[(None, None), 0, 0]
        ld_atom = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), Float32)
        thr_ld = tcgen05.make_tmem_copy(ld_atom, tSAcc).get_slice(tidx)
        tStS = thr_ld.partition_S(tSAcc)
        tScS = thr_ld.partition_D(cute.make_identity_tensor((128, 128)))
        row = m_block * self.m_block_size + tidx
        qsc = Float32(mQS[batch_idx, head_idx, row // 32]) * Float32(math.log2(math.e))
        lane = (cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4) * 32
        m_old = -Float32.inf
        row_sum = Float32(0.0)
        m_old, row_sum = self.softmax_step(Int32(0), n_block_max - 1, True, thr_ld, tStS, tScS, mbar, stats, sp_base,
                                           tidx, lane, qsc, mKVW, kv_head_idx, batch_idx, m_old, row_sum)
        for i in cutlass.range(cnt - 1, unroll=1):
            m_old, row_sum = self.softmax_step(i + 1, n_block_max - 2 - i, False, thr_ld, tStS, tScS, mbar, stats,
                                               sp_base, tidx, lane, qsc, mKVW, kv_head_idx, batch_idx, m_old, row_sum)
        # finalize (hd128 correction epilogue formulas): scale = rcp.approx(l or 1), LSE = (m*1 + (log2(l) - 0)) * ln2
        zero_or_nan = row_sum == Float32(0.0) or row_sum != row_sum
        scale = cute.arch.rcp_approx(row_sum if not zero_or_nan else Float32(1.0))
        LN2_ = math.log(2.0)
        lse = (m_old * Float32(1.0) + (cute.math.log2(row_sum, fastmath=True) - Float32(0.0))) * LN2_ \
            if not zero_or_nan else -Float32.inf
        mLSE[batch_idx, head_idx, row] = lse
        st_shared_f32(stats + 4 * (2 * 128 + tidx), scale)
        dm_utils.mbar_arrive(mbar + 64)

    @cute.jit
    def softmax_step(self, j, n_block, is_first: cutlass.Constexpr, thr_ld, tStS, tScS, mbar, stats, sp_base, tidx,
                     lane, qsc, mKVW, kv_head_idx, batch_idx, m_old, row_sum):
        slot = j & 1
        dm_kw = Int32(mKVW[batch_idx, kv_head_idx, 2 * n_block])
        dm_vw = Int32(mKVW[batch_idx, kv_head_idx, 2 * n_block + 1])
        dm_ks = [dm_utils.byte_pow2_f32(dm_kw, b) for b in range(4)]
        if const_expr(self.sf_bank):
            dm_utils.mbar_wait_parity(mbar + 0, j & 1)                 # S_full (single buffer, one completion per tile)
            dm_utils.tc_fence_after()
            tStS_cur = tStS
        else:
            dm_utils.mbar_wait_parity(mbar + 8 * slot, (j >> 1) & 1)
            dm_utils.tc_fence_after()
            tStS_cur = cute.make_tensor(tStS.iterator + slot * 128, tStS.layout)
        s = cute.make_rmem_tensor(tScS.shape, Float32)
        cute.copy(thr_ld, tStS_cur, s)
        cute.arch.fence_view_async_tmem_load()
        if const_expr(self.sf_bank):
            dm_utils.tc_fence_before()
            dm_utils.mbar_arrive(mbar + 72)                            # S_empty(j): QK(j+1) may overwrite S
        if const_expr(self.is_causal and is_first):
            for c in cutlass.range_constexpr(128):
                if Int32(c) > tidx:
                    s[c] = -Float32.inf
        # ---- hd128 "DM sm100-mx softmax step", op for op ----
        dm_c = [qsc * dm_ks[b] for b in range(4)]
        dm_bmax = [None] * 4
        for b in cutlass.range_constexpr(4):
            b0 = 32 * b
            lm = [utils.fmax(s[b0 + 3 * t], s[b0 + 3 * t + 1], s[b0 + 3 * t + 2]) for t in range(4)]
            for i in cutlass.range_constexpr(12, 32, 2):
                t = ((i - 12) // 2) % 4
                lm[t] = utils.fmax(lm[t], s[b0 + i], s[b0 + i + 1])
            v = utils.fmax(utils.fmax(lm[0], lm[1], lm[2]), lm[3])
            dm_bmax[b] = v * dm_c[b]
        dm_xmax = utils.fmax(utils.fmax(dm_bmax[0], dm_bmax[1], dm_bmax[2]), dm_bmax[3])
        dm_mcand = dm_utils.ceil_f32(dm_xmax)
        acc_scale = Float32(1.0)
        row_max = dm_mcand
        if const_expr(is_first):
            acc_scale = Float32(0.0)          # hd128 literal (unused: the first tile has no correction)
        else:
            row_max = m_old
            if dm_mcand > m_old + Float32(self.rescale_T):
                row_max = dm_mcand
                acc_scale = dm_utils.exp2i_f32(m_old - dm_mcand)
        m_safe = row_max if row_max != -Float32.inf else Float32(0.0)
        dm_bias = [None] * 4
        dm_w = [None] * 4
        dm_sfa = [None] * 4
        for b in cutlass.range_constexpr(4):
            kb = cute.arch.fmax(dm_utils.floor_f32(dm_bmax[b]), Float32(-1e30))
            e = kb - Float32(7.0) - m_safe
            dm_bias[b] = Float32(7.0) - kb
            dm_sfa[b] = dm_utils.ue8m0_byte(e)
            dm_w[b] = dm_utils.byte_shl23_f32(dm_sfa[b])
        sfa_word = dm_utils.pack_bytes4(dm_sfa[0], dm_sfa[1], dm_sfa[2], dm_sfa[3])
        sfbase = (lane << 16) + ((128 + 32 * slot) if const_expr(self.sf_bank) else slot * 128)
        for r in cutlass.range_constexpr(4):
            tmem_st1(sfbase + self.sfa_off + r, sfa_word)
        for r in cutlass.range_constexpr(8):
            tmem_st1(sfbase + self.sfb_off + r, dm_vw)
        # x = S * c_b + (7 - kb); p = ex2(x); codes -> sP[slot] (K-major SW128: row r at r*128 B, chunk q at (q ^ (r&7))*16)
        for b in cutlass.range_constexpr(4):
            for i in cutlass.range_constexpr(0, 32, 2):
                s[32 * b + i], s[32 * b + i + 1] = cute.arch.fma_packed_f32x2(
                    (s[32 * b + i], s[32 * b + i + 1]), (dm_c[b], dm_c[b]), (dm_bias[b], dm_bias[b]))
        prow = sp_base + slot * 16384 + tidx * 128
        for b in cutlass.range_constexpr(4):
            for i in cutlass.range_constexpr(32):
                s[32 * b + i] = ex2_ftz(s[32 * b + i])
            for q in cutlass.range_constexpr(2):
                k = 32 * b + 16 * q
                w = [e4m3x4(s[k + 4 * t], s[k + 4 * t + 1], s[k + 4 * t + 2], s[k + 4 * t + 3]) for t in range(4)]
                chunk = Int32(2 * b + q)
                dm_utils.st_shared_v4_b32(prow + ((chunk ^ (tidx & 7)) << 4), w[0], w[1], w[2], w[3])
        cute.arch.fence_view_async_shared()
        tmem_wait_st()
        dm_utils.tc_fence_before()
        dm_utils.mbar_arrive(mbar + 16 + 8 * slot)          # P_full[slot]
        st_shared_f32(stats + 4 * (slot * 128 + tidx), acc_scale)
        dm_utils.mbar_arrive(mbar + 48 + 8 * slot)          # stats_full[slot]
        dm_bs = [None] * 4
        for b in cutlass.range_constexpr(4):
            ps = (s[32 * b], s[32 * b + 1])
            for i in cutlass.range_constexpr(2, 32, 2):
                ps = cute.arch.add_packed_f32x2(ps, (s[32 * b + i], s[32 * b + i + 1]))
            dm_bs[b] = ps[0] + ps[1]
        dm_l = Float32(0.0) if const_expr(is_first) else row_sum * acc_scale
        for b in cutlass.range_constexpr(4):
            dm_l = dm_l + dm_bs[b] * dm_w[b]
        return row_max, dm_l


    # -----------------------------------------------------------------------------------------------------------------
    @cute.jit
    def softmax2(self, acc_qk, mbar, stats, sp_base, mLSE, mQS, mKVW, m_block, head_idx, kv_head_idx, batch_idx,
                 n_block_max, cnt):
        """C6: tile-parity softmax (see __init__). wg = warp // 4 owns tiles j = wg, wg + 2, ..."""
        tidx = cute.arch.thread_idx()[0] % 128
        wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) // 4
        tSAcc = acc_qk[(None, None), 0, 0]
        ld_atom = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), Float32)
        thr_ld = tcgen05.make_tmem_copy(ld_atom, tSAcc).get_slice(tidx)
        tStS = thr_ld.partition_S(tSAcc)
        tScS = thr_ld.partition_D(cute.make_identity_tensor((128, 128)))
        row = m_block * self.m_block_size + tidx
        qsc = Float32(mQS[batch_idx, head_idx, row // 32]) * Float32(math.log2(math.e))
        lane = (cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4) * 32
        m_last = -Float32.inf
        l_last = Float32(0.0)
        if wg == 0:
            m_last, l_last = self.softmax_step2(Int32(0), n_block_max - 1, True, thr_ld, tStS, tScS, mbar, stats,
                                                sp_base, tidx, lane, qsc, mKVW, kv_head_idx, batch_idx, cnt)
            for i in cutlass.range((cnt - 1) // 2, unroll=1):
                j = 2 * i + 2
                m_last, l_last = self.softmax_step2(j, n_block_max - 1 - j, False, thr_ld, tStS, tScS, mbar, stats,
                                                    sp_base, tidx, lane, qsc, mKVW, kv_head_idx, batch_idx, cnt)
        else:
            for i in cutlass.range(cnt // 2, unroll=1):
                j = 2 * i + 1
                m_last, l_last = self.softmax_step2(j, n_block_max - 1 - j, False, thr_ld, tStS, tScS, mbar, stats,
                                                    sp_base, tidx, lane, qsc, mKVW, kv_head_idx, batch_idx, cnt)
        # the owner of the last tile (j = cnt - 1) finalizes (hd128 correction epilogue formulas)
        if wg == ((cnt - 1) & 1):
            row_sum = l_last
            zero_or_nan = row_sum == Float32(0.0) or row_sum != row_sum
            scale = cute.arch.rcp_approx(row_sum if not zero_or_nan else Float32(1.0))
            LN2_ = math.log(2.0)
            lse = (m_last * Float32(1.0) + (cute.math.log2(row_sum, fastmath=True) - Float32(0.0))) * LN2_ \
                if not zero_or_nan else -Float32.inf
            mLSE[batch_idx, head_idx, row] = lse
            st_shared_f32(stats + 4 * (2 * 128 + tidx), scale)
            dm_utils.mbar_arrive(mbar + 64)

    @cute.jit
    def softmax_step2(self, j, n_block, is_first: cutlass.Constexpr, thr_ld, tStS, tScS, mbar, stats, sp_base, tidx,
                      lane, qsc, mKVW, kv_head_idx, batch_idx, cnt):
        slot = j & 1
        dm_kw = Int32(mKVW[batch_idx, kv_head_idx, 2 * n_block])
        dm_vw = Int32(mKVW[batch_idx, kv_head_idx, 2 * n_block + 1])
        dm_ks = [dm_utils.byte_pow2_f32(dm_kw, b) for b in range(4)]
        tr_w = (tidx // 32) if const_expr(self.trace) else None
        self.tr(j, 10, None, tr_w == 0)
        dm_utils.mbar_wait_parity(mbar + 8 * slot, (j >> 1) & 1)          # S_full[slot]
        dm_utils.tc_fence_after()
        self.tr(j, 11, None, tr_w == 0)
        tStS_cur = cute.make_tensor(tStS.iterator + slot * 128, tStS.layout)
        s = cute.make_rmem_tensor(tScS.shape, Float32)
        if const_expr(self.split == 2 and self.pc_inj == 7):
            dm_utils.tc_fence_before()
            dm_utils.mbar_arrive(mbar + 96 + 8 * slot)          # DEBUG PC 7: S_empty BEFORE the load (must corrupt)
        cute.copy(thr_ld, tStS_cur, s)
        cute.arch.fence_view_async_tmem_load()
        if const_expr(self.trace):
            for q in cutlass.range_constexpr(4):
                self.tr(j, 16 + q, None, tr_w == q)
        if const_expr(self.split == 2 and self.pc_inj != 7):
            dm_utils.tc_fence_before()
            dm_utils.mbar_arrive(mbar + 96 + 8 * slot)          # S_empty[slot](j): S(j) is in registers (wait::ld above)
        if const_expr(self.is_causal and is_first):
            for c in cutlass.range_constexpr(128):
                if Int32(c) > tidx:
                    s[c] = -Float32.inf
        # ---- m-independent prefix (hd128 sm100-mx line, same expressions) ----
        dm_c = [qsc * dm_ks[b] for b in range(4)]
        dm_bmax = [None] * 4
        for b in cutlass.range_constexpr(4):
            b0 = 32 * b
            lm = [utils.fmax(s[b0 + 3 * t], s[b0 + 3 * t + 1], s[b0 + 3 * t + 2]) for t in range(4)]
            for i in cutlass.range_constexpr(12, 32, 2):
                t = ((i - 12) // 2) % 4
                lm[t] = utils.fmax(lm[t], s[b0 + i], s[b0 + i + 1])
            v = utils.fmax(utils.fmax(lm[0], lm[1], lm[2]), lm[3])
            dm_bmax[b] = v * dm_c[b]
        dm_xmax = utils.fmax(utils.fmax(dm_bmax[0], dm_bmax[1], dm_bmax[2]), dm_bmax[3])
        dm_mcand = dm_utils.ceil_f32(dm_xmax)
        dm_kb = [None] * 4
        dm_bias = [None] * 4
        for b in cutlass.range_constexpr(4):
            dm_kb[b] = cute.arch.fmax(dm_utils.floor_f32(dm_bmax[b]), Float32(-1e30))
            dm_bias[b] = Float32(7.0) - dm_kb[b]
        for b in cutlass.range_constexpr(4):
            for i in cutlass.range_constexpr(0, 32, 2):
                s[32 * b + i], s[32 * b + i + 1] = cute.arch.fma_packed_f32x2(
                    (s[32 * b + i], s[32 * b + i + 1]), (dm_c[b], dm_c[b]), (dm_bias[b], dm_bias[b]))
        prow = sp_base + slot * 16384 + tidx * 128
        for b in cutlass.range_constexpr(4):
            for i in cutlass.range_constexpr(32):
                s[32 * b + i] = ex2_ftz(s[32 * b + i])
            for q in cutlass.range_constexpr(2):
                k = 32 * b + 16 * q
                w = [e4m3x4(s[k + 4 * t], s[k + 4 * t + 1], s[k + 4 * t + 2], s[k + 4 * t + 3]) for t in range(4)]
                chunk = Int32(2 * b + q)
                dm_utils.st_shared_v4_b32(prow + ((chunk ^ (tidx & 7)) << 4), w[0], w[1], w[2], w[3])
        dm_bs = [None] * 4
        for b in cutlass.range_constexpr(4):
            ps = (s[32 * b], s[32 * b + 1])
            for i in cutlass.range_constexpr(2, 32, 2):
                ps = cute.arch.add_packed_f32x2(ps, (s[32 * b + i], s[32 * b + i + 1]))
            dm_bs[b] = ps[0] + ps[1]
        if const_expr(self.g1 & 8):
            sfb0 = (lane << 16) + slot * 128
            # G1d: SFB(j) = the V scale word (m-independent) before the mailbox wait.  Columns slot+64..71 of this thread's TMEM
            # lane: its own S(j) load completed (wait::ld above); PV(j-2)'s SF read completed before QK(j) (C0) -> S_full(j).
            tmem_st_x4(sfb0 + self.sfb_off, dm_vw, dm_vw, dm_vw, dm_vw)
            tmem_st_x4(sfb0 + self.sfb_off + 4, dm_vw, dm_vw, dm_vw, dm_vw)
        # ---- ordered tail: the previous tile's (m, l) from the other WG ----
        self.tr(j, 12, None, tr_w == 0)
        m_old = -Float32.inf
        row_sum = Float32(0.0)
        if const_expr(not is_first):
            pj = j - 1
            if const_expr(self.pc_inj != 4):
                dm_utils.mbar_wait_parity(mbar + 80 + 8 * (pj & 1), (pj >> 1) & 1)
        self.tr(j, 13, None, tr_w == 0)
        if const_expr(not is_first):
            m_old = ld_shared_f32(stats + 4 * (512 + (pj & 1) * 128 + tidx))
            row_sum = ld_shared_f32(stats + 4 * (768 + (pj & 1) * 128 + tidx))
        acc_scale = Float32(1.0)
        row_max = dm_mcand
        if const_expr(is_first):
            acc_scale = Float32(0.0)
        else:
            row_max = m_old
            if dm_mcand > m_old + Float32(self.rescale_T):
                row_max = dm_mcand
                acc_scale = dm_utils.exp2i_f32(m_old - dm_mcand)
        if const_expr(self.g1 & 1):
            # G1a: acc_scale(j) -> correction now (stats[slot] reuse: correction(j-2) read it before O_ready(j-2) -> PV(j-2) ->
            # (C0) QK(j) -> S_full(j), which this thread waited)
            if const_expr(self.pc_inj == 9):
                # DEBUG PC 9 (G1a): stats_full arrives BEFORE its payload, payload ~20 us later -> must corrupt (rescaling inputs)
                dm_utils.mbar_arrive(mbar + 48 + 8 * slot)
                for _d in cutlass.range_constexpr(20):
                    dm_utils.nanosleep_1us()
                st_shared_f32(stats + 4 * (slot * 128 + tidx), acc_scale)
            else:
                st_shared_f32(stats + 4 * (slot * 128 + tidx), acc_scale)
                dm_utils.mbar_arrive(mbar + 48 + 8 * slot)          # stats_full[slot]
        m_safe = row_max if row_max != -Float32.inf else Float32(0.0)
        dm_w = [None] * 4
        dm_sfa = [None] * 4
        for b in cutlass.range_constexpr(4):
            e = dm_kb[b] - Float32(7.0) - m_safe
            dm_sfa[b] = dm_utils.ue8m0_byte(e)
            dm_w[b] = dm_utils.byte_shl23_f32(dm_sfa[b])
        sfa_word = dm_utils.pack_bytes4(dm_sfa[0], dm_sfa[1], dm_sfa[2], dm_sfa[3])
        sfbase = (lane << 16) + slot * 128
        if const_expr(self.g1 & 8):
            tmem_st_x4(sfbase + self.sfa_off, sfa_word, sfa_word, sfa_word, sfa_word)
        else:
            for r in cutlass.range_constexpr(4):
                tmem_st1(sfbase + self.sfa_off + r, sfa_word)
            for r in cutlass.range_constexpr(8):
                tmem_st1(sfbase + self.sfb_off + r, dm_vw)
        cute.arch.fence_view_async_shared()
        tmem_wait_st()
        dm_utils.tc_fence_before()
        dm_utils.mbar_arrive(mbar + 16 + 8 * slot)          # P_full[slot]
        if const_expr(self.trace):
            self.tr(j, 14, None, tr_w == 0)
            for q in cutlass.range_constexpr(4):
                self.tr(j, 20 + q, None, tr_w == q)
        if const_expr(not (self.g1 & 1)):
            st_shared_f32(stats + 4 * (slot * 128 + tidx), acc_scale)
            dm_utils.mbar_arrive(mbar + 48 + 8 * slot)          # stats_full[slot]
        dm_l = Float32(0.0) if const_expr(is_first) else row_sum * acc_scale
        for b in cutlass.range_constexpr(4):
            dm_l = dm_l + dm_bs[b] * dm_w[b]
        # publish (m, l) of tile j for tile j + 1 (owned by the other WG)
        if j + 1 < cnt:
            if const_expr(self.pc_inj == 5):
                dm_utils.mbar_arrive(mbar + 80 + 8 * slot)
                for _d in cutlass.range_constexpr(20):          # widen the window: payload lands ~20 us after the arrive
                    dm_utils.nanosleep_1us()
            st_shared_f32(stats + 4 * (512 + slot * 128 + tidx), row_max)
            st_shared_f32(stats + 4 * (768 + slot * 128 + tidx), dm_l)
            if const_expr(self.pc_inj != 5):
                dm_utils.mbar_arrive(mbar + 80 + 8 * slot)      # state_full[slot] (release: orders the stores above)
        self.tr(j, 15, None, tr_w == 0)
        return row_max, dm_l

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
        if const_expr(self.c11):
            if wg == 0:
                self.softmax_step3(Int32(0), n_block_max - 1, True, 0, mbar, stats, sp_base, tidx, lane, qsc, mKVW, kv_head_idx,
                                   batch_idx, m_last, l_last, cnt=cnt)
                for i in cutlass.range(cnt - 1, unroll=1):
                    self.softmax_step3(i + 1, n_block_max - 2 - i, False, 0, mbar, stats, sp_base, tidx, lane, qsc, mKVW,
                                       kv_head_idx, batch_idx, m_last, l_last, cnt=cnt)
            else:
                self.softmax_step3(Int32(0), n_block_max - 1, True, 1, mbar, stats, sp_base, tidx, lane, qsc, mKVW, kv_head_idx,
                                   batch_idx, m_last, l_last, cnt=cnt)
                for i in cutlass.range(cnt - 1, unroll=1):
                    self.softmax_step3(i + 1, n_block_max - 2 - i, False, 1, mbar, stats, sp_base, tidx, lane, qsc, mKVW,
                                       kv_head_idx, batch_idx, m_last, l_last, cnt=cnt)
        elif wg == 0:
            m_last, l_last = self.softmax_step3(Int32(0), n_block_max - 1, True, 0, mbar, stats, sp_base, tidx, lane, qsc, mKVW,
                                                kv_head_idx, batch_idx, m_last, l_last, cnt=cnt)
            for i in cutlass.range(cnt - 1, unroll=1):
                m_last, l_last = self.softmax_step3(i + 1, n_block_max - 2 - i, False, 0, mbar, stats, sp_base, tidx, lane, qsc,
                                                    mKVW, kv_head_idx, batch_idx, m_last, l_last, cnt=cnt)
            row_sum = l_last
            zero_or_nan = row_sum == Float32(0.0) or row_sum != row_sum
            scale = cute.arch.rcp_approx(row_sum if not zero_or_nan else Float32(1.0))
            LN2_ = math.log(2.0)
            lse = (m_last * Float32(1.0) + (cute.math.log2(row_sum, fastmath=True) - Float32(0.0))) * LN2_ \
                if not zero_or_nan else -Float32.inf
            mLSE[batch_idx, head_idx, row] = lse
            st_shared_f32(stats + 4 * (2 * 128 + tidx), scale)
            dm_utils.mbar_arrive(mbar + 64)
        else:
            self.softmax_step3(Int32(0), n_block_max - 1, True, 1, mbar, stats, sp_base, tidx, lane, qsc, mKVW, kv_head_idx,
                               batch_idx, m_last, l_last, cnt=cnt)
            for i in cutlass.range(cnt - 1, unroll=1):
                self.softmax_step3(i + 1, n_block_max - 2 - i, False, 1, mbar, stats, sp_base, tidx, lane, qsc, mKVW,
                                   kv_head_idx, batch_idx, m_last, l_last, cnt=cnt)

    @cute.jit
    def softmax_step3(self, j, n_block, is_first: cutlass.Constexpr, h: cutlass.Constexpr, mbar, stats, sp_base, tidx, lane,
                      qsc, mKVW, kv_head_idx, batch_idx, m_old, row_sum, cnt=None):
        """C8 tile step of WG h (h = 0: blocks 0, 1 + the ordered tail; h = 1: blocks 2, 3 + publish)."""
        slot = j & 1
        part = stats + 4 * (1024 + slot * 512)                      # part[slot][k][128]: bmax2, bmax3, bs2, bs3
        b3 = slot                                                    # C10: per-tile softmax resources ring b3 = j % 3
        ph3 = (j >> 1) & 1
        pfull = mbar + 16 + 8 * slot
        partf = mbar + 80 + 8 * slot
        if const_expr(self.c10):
            b3 = j % 3
            ph3 = (j // 3) & 1
            part = stats + 4 * (1024 + b3 * self.part_stride)
            pfull = mbar + 128 + 8 * b3
            partf = mbar + 176 + 8 * b3
        dm_kw = Int32(mKVW[batch_idx, kv_head_idx, 2 * n_block])
        dm_vw = Int32(mKVW[batch_idx, kv_head_idx, 2 * n_block + 1])
        dm_ks = [dm_utils.byte_pow2_f32(dm_kw, b) for b in range(4)]
        tr_w = (tidx // 32) if const_expr(self.trace) else None
        self.tr(j, 10 + 8 * h, None, tr_w == 0)
        dm_utils.mbar_wait_parity(mbar + 8 * slot, (j >> 1) & 1)          # S_full[slot]
        dm_utils.tc_fence_after()
        self.tr(j, 11 + 8 * h, None, tr_w == 0)
        sfbase = (lane << 16) + slot * 128
        if const_expr(h == 0 and self.c8 == 1 and self.pc_inj == 12):
            for r in cutlass.range_constexpr(8):                             # DEBUG PC 12: SFB before WG1 loaded its columns
                tmem_st1(sfbase + self.sfb_off + r, dm_vw)
        if const_expr(h == 0 and self.c9 and self.pc_inj == 13):
            for r in cutlass.range_constexpr(4):                             # DEBUG PC 13 (C9): SFA (+112) before WG1 loaded
                tmem_st1(sfbase + self.sfa_off + r, Int32(0))
        if const_expr(h == 1 and self.c8 == 1 and self.pc_inj in (12, 13)):
            for _d in cutlass.range_constexpr(20):
                dm_utils.nanosleep_1us()
        c0 = 64 * h
        if const_expr(h == 1 and self.c8 == 2 and self.pc_inj == 12):
            for r in cutlass.range_constexpr(8):                             # DEBUG PC 12 (C8b): SFB before our S load
                tmem_st1(sfbase + self.sfb_off + r, dm_vw)
            tmem_wait_st()
        if const_expr(self.sempty and self.pc_inj == 7):
            dm_utils.tc_fence_before()
            dm_utils.mbar_arrive(mbar + 96 + 8 * slot)                      # DEBUG PC 7 (C9/C10): S_empty BEFORE our S load
        if const_expr(self.sempty and h == 1 and self.pc_inj in (6, 7, 14)):
            for _d in cutlass.range_constexpr(20):                           # DEBUG PC 6/7 (C9): widen the window (WG1 load late)
                dm_utils.nanosleep_1us()
        s0 = tmem_ld32(sfbase + c0)
        s1 = tmem_ld32(sfbase + c0 + 32)
        tmem_wait_ld()
        if const_expr(self.sempty and self.pc_inj != 7):
            dm_utils.tc_fence_before()
            dm_utils.mbar_arrive(mbar + 96 + 8 * slot)                      # C9/C10: S_empty[slot](j): our S(j) columns in registers
        if const_expr(self.c10 and h == 1):
            if const_expr(self.pc_inj == 7):
                dm_utils.tc_fence_before()
            dm_utils.mbar_arrive(mbar + 112 + 8 * slot)                     # C10: W1L[slot](j): WG1 loaded S(j) cols 64..127
        if const_expr(h == 1 and self.c8 == 2 and self.pc_inj != 12):
            for r in cutlass.range_constexpr(8):                             # C8b: SFB(j) in our own columns after our load
                tmem_st1(sfbase + self.sfb_off + r, dm_vw)
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
            dm_kb[bb] = cute.arch.fmax(dm_utils.floor_f32(dm_bmax[bb]), Float32(-1e30))
            dm_bias[bb] = Float32(7.0) - dm_kb[bb]
        for bb in cutlass.range_constexpr(2):
            for i in cutlass.range_constexpr(0, 32, 2):
                s[32 * bb + i], s[32 * bb + i + 1] = cute.arch.fma_packed_f32x2(
                    (s[32 * bb + i], s[32 * bb + i + 1]), (dm_c[bb], dm_c[bb]), (dm_bias[bb], dm_bias[bb]))
        prow = sp_base + b3 * 16384 + tidx * 128
        for bb in cutlass.range_constexpr(2):
            for i in cutlass.range_constexpr(32):
                s[32 * bb + i] = ex2_ftz(s[32 * bb + i])
        pst = []                                                             # (address, 4 code words) per 16 B chunk
        for bb in cutlass.range_constexpr(2):
            for q in cutlass.range_constexpr(2):
                k = 32 * bb + 16 * q
                w = [e4m3x4(s[k + 4 * t], s[k + 4 * t + 1], s[k + 4 * t + 2], s[k + 4 * t + 3]) for t in range(4)]
                chunk = Int32(2 * (2 * h + bb) + q)
                pst.append((prow + ((chunk ^ (tidx & 7)) << 4), w))
                if const_expr(not (h == 1 and self.pc_inj == 11)):
                    dm_utils.st_shared_v4_b32(pst[-1][0], w[0], w[1], w[2], w[3])
        dm_bs = [None] * 2
        for bb in cutlass.range_constexpr(2):
            ps = (s[32 * bb], s[32 * bb + 1])
            for i in cutlass.range_constexpr(2, 32, 2):
                ps = cute.arch.add_packed_f32x2(ps, (s[32 * bb + i], s[32 * bb + i + 1]))
            dm_bs[bb] = ps[0] + ps[1]
        self.tr(j, 12 + 8 * h, None, tr_w == 0)
        row_max = m_old
        dm_l = row_sum
        pk = 4 * h if const_expr(self.c11) else 0                           # C11: part[b][4h + k]
        if const_expr(h == 1 or self.c11):
            # ---- publish the high half: part(j) reuse at j+2 is ordered by WG0's read(j) -> P_full(j) -> PV(j) -> (C0)
            # QK(j+2) -> S_full(j+2), which this WG waits before writing part(j+2) ----
            if const_expr(self.pc_inj == 5 and h == 1):
                dm_utils.tc_fence_before()
                dm_utils.mbar_arrive(partf)
                for _d in cutlass.range_constexpr(20):
                    dm_utils.nanosleep_1us()
            st_shared_f32(part + 4 * ((0 + pk) * 128 + tidx), dm_bmax[0])
            st_shared_f32(part + 4 * ((1 + pk) * 128 + tidx), dm_bmax[1])
            st_shared_f32(part + 4 * ((2 + pk) * 128 + tidx), dm_bs[0])
            st_shared_f32(part + 4 * ((3 + pk) * 128 + tidx), dm_bs[1])
            if const_expr(not (self.pc_inj == 5 and h == 1)):
                dm_utils.tc_fence_before()                                   # our S loads (wait::ld) before WG0's SFB stores
                dm_utils.mbar_arrive(partf)                                  # partial_full[slot] (C10: [j%3])
            if const_expr(not (self.pc_inj == 11 and h == 1)):
                cute.arch.fence_view_async_shared()
                if const_expr(self.c8 == 2):
                    tmem_wait_st()                                           # C8b: our SFB stores complete before P_full
                dm_utils.tc_fence_before()
                dm_utils.mbar_arrive(pfull)                                  # P_full[slot] (our P codes; C10 [j%3])
            else:
                # DEBUG PC 11: P_full arrives BEFORE WG1's P stores (stores ~20 us later, then the proxy fence) -> must corrupt
                dm_utils.tc_fence_before()
                dm_utils.mbar_arrive(pfull)
                for _d in cutlass.range_constexpr(20):
                    dm_utils.nanosleep_1us()
                for (a, w) in pst:
                    dm_utils.st_shared_v4_b32(a, w[0], w[1], w[2], w[3])
                cute.arch.fence_view_async_shared()
            self.tr(j, 14 + 8 * h, None, tr_w == 0)
        else:
            # ---- ordered tail (WG0): exactly the C6 tail over the 4 blocks ----
            if const_expr(self.pc_inj != 4):
                dm_utils.mbar_wait_parity(partf, ph3)                           # partial_full[slot] (C10 [j%3])
            dm_utils.tc_fence_after()
            self.tr(j, 13, None, tr_w == 0)
            bmax2 = ld_shared_f32(part + 4 * (0 * 128 + tidx))
            bmax3 = ld_shared_f32(part + 4 * (1 * 128 + tidx))
            bs2 = ld_shared_f32(part + 4 * (2 * 128 + tidx))
            bs3 = ld_shared_f32(part + 4 * (3 * 128 + tidx))
            bmax = [dm_bmax[0], dm_bmax[1], bmax2, bmax3]
            bsum = [dm_bs[0], dm_bs[1], bs2, bs3]
            kb = [dm_kb[0], dm_kb[1], cute.arch.fmax(dm_utils.floor_f32(bmax2), Float32(-1e30)),
                  cute.arch.fmax(dm_utils.floor_f32(bmax3), Float32(-1e30))]
            dm_xmax = utils.fmax(utils.fmax(bmax[0], bmax[1], bmax[2]), bmax[3])
            dm_mcand = dm_utils.ceil_f32(dm_xmax)
            acc_scale = Float32(1.0)
            row_max = dm_mcand
            if const_expr(is_first):
                acc_scale = Float32(0.0)
            else:
                row_max = m_old
                if dm_mcand > m_old + Float32(self.rescale_T):
                    row_max = dm_mcand
                    acc_scale = dm_utils.exp2i_f32(m_old - dm_mcand)
            st_a = stats + 4 * (slot * 128 + tidx)
            st_b = mbar + 48 + 8 * slot
            if const_expr(self.c10):
                st_a = stats + 4 * (512 + b3 * 128 + tidx)                   # C10 stats3[j%3] (C6 mailbox area, unused by C8)
                st_b = mbar + 152 + 8 * b3
            if const_expr(self.pc_inj == 9):
                dm_utils.mbar_arrive(st_b)
                for _d in cutlass.range_constexpr(20):
                    dm_utils.nanosleep_1us()
                st_shared_f32(st_a, acc_scale)
            else:
                st_shared_f32(st_a, acc_scale)
                dm_utils.mbar_arrive(st_b)                                   # stats_full[slot] (G1a; C10 [j%3])
            m_safe = row_max if row_max != -Float32.inf else Float32(0.0)
            dm_w = [None] * 4
            dm_sfa = [None] * 4
            for b in cutlass.range_constexpr(4):
                e = kb[b] - Float32(7.0) - m_safe
                dm_sfa[b] = dm_utils.ue8m0_byte(e)
                dm_w[b] = dm_utils.byte_shl23_f32(dm_sfa[b])
            sfa_word = dm_utils.pack_bytes4(dm_sfa[0], dm_sfa[1], dm_sfa[2], dm_sfa[3])
            sfa_a = sfbase + self.sfa_off
            sfb_a = sfbase + self.sfb_off
            if const_expr(self.c10):
                # SF(j) -> slot (j+1)&1 + 64 / + 72 after WG1 loaded S(j+1) (W1L), or (last tile) own slot + 80 / + 88 (partial edge)
                if j + 1 < cnt:
                    if const_expr(self.pc_inj != 14):
                        dm_utils.mbar_wait_parity(mbar + 112 + 8 * ((j + 1) & 1), ((j + 1) >> 1) & 1)
                        dm_utils.tc_fence_after()
                sfa_a = (lane << 16) + self.sf_col10(j, cnt)
                sfb_a = sfa_a + 8
            for r in cutlass.range_constexpr(4):
                tmem_st1(sfa_a + r, sfa_word)
            if const_expr(self.c8 == 1 and self.pc_inj != 12):
                for r in cutlass.range_constexpr(8):
                    tmem_st1(sfb_a + r, dm_vw)
            cute.arch.fence_view_async_shared()
            tmem_wait_st()
            dm_utils.tc_fence_before()
            dm_utils.mbar_arrive(pfull)                                      # P_full[slot] (C10 [j%3])
            self.tr(j, 14, None, tr_w == 0)
            dm_l = Float32(0.0) if const_expr(is_first) else row_sum * acc_scale
            for b in cutlass.range_constexpr(4):
                dm_l = dm_l + bsum[b] * dm_w[b]
        return row_max, dm_l

    # -----------------------------------------------------------------------------------------------------------------
    @cute.jit
    def softmax4(self, mbar, sp_base, stats, mQS, mKVW, m_block, head_idx, kv_head_idx, batch_idx, n_block_max, cnt):
        """C12: WG wg = warp // 4 owns tiles j = wg, wg + 2, ... (prefix only, see __init__ c12)."""
        tidx = cute.arch.thread_idx()[0] % 128
        wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) // 4
        row = m_block * self.m_block_size + tidx
        qsc = Float32(mQS[batch_idx, head_idx, row // 32]) * Float32(math.log2(math.e))
        lane = (cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4) * 32
        if wg == 0:
            self.softmax_step4(Int32(0), n_block_max - 1, True, mbar, stats, sp_base, tidx, lane, qsc, mKVW, kv_head_idx, batch_idx)
            for i in cutlass.range((cnt - 1) // 2, unroll=1):
                j = 2 * i + 2
                self.softmax_step4(j, n_block_max - 1 - j, False, mbar, stats, sp_base, tidx, lane, qsc, mKVW, kv_head_idx,
                                   batch_idx)
        else:
            for i in cutlass.range(cnt // 2, unroll=1):
                j = 2 * i + 1
                self.softmax_step4(j, n_block_max - 1 - j, False, mbar, stats, sp_base, tidx, lane, qsc, mKVW, kv_head_idx,
                                   batch_idx)

    @cute.jit
    def softmax_step4(self, j, n_block, is_first: cutlass.Constexpr, mbar, stats, sp_base, tidx, lane, qsc, mKVW, kv_head_idx,
                      batch_idx, J=None):
        """C12 tile step: S(j) load -> S_empty / W1L, 4-block prefix (C6 softmax_step2 expressions), publish part[j%3] + P codes.
        C14 (pers): J = global tile index (slot / ring / phase); j stays item-local (trace row)."""
        Jg = j if const_expr(J is None) else J
        slot = Jg & 1
        b3 = Jg % 3
        part = stats + 4 * (1024 + b3 * self.part_stride)
        if const_expr(self.pc_inj == 17):
            part = stats + 4 * (1024 + (Jg & 1) * self.part_stride)         # DEBUG PC 17: part ring depth 2 (reuse race)
        sb = b3
        if const_expr(self.pc_inj == 16):
            sb = Jg & 1                                                      # DEBUG PC 16: sP ring depth 2 (reuse race)
        pfull = mbar + 128 + 8 * b3
        partf = mbar + 176 + 8 * b3
        dm_kw = Int32(mKVW[batch_idx, kv_head_idx, 2 * n_block])
        dm_ks = [dm_utils.byte_pow2_f32(dm_kw, b) for b in range(4)]
        tr_w = (tidx // 32) if const_expr(self.trace) else None
        tro = 8 * (cute.arch.make_warp_uniform(cute.arch.warp_idx()) // 4) if const_expr(self.trace) else 0
        self.tr(j, 10 + tro, None, tr_w == 0)
        dm_utils.mbar_wait_parity(mbar + 8 * slot, (Jg >> 1) & 1)         # S_full[slot]
        dm_utils.tc_fence_after()
        self.tr(j, 11 + tro, None, tr_w == 0)
        sbase = (lane << 16) + slot * 128
        if const_expr(self.pc_inj == 7):
            dm_utils.tc_fence_before()
            dm_utils.mbar_arrive(mbar + 96 + 8 * slot)                      # DEBUG PC 7: S_empty BEFORE the S load
            dm_utils.mbar_arrive(mbar + 112 + 8 * slot)
        if const_expr(self.pc_inj in (6, 7, 14)):
            for _d in cutlass.range_constexpr(20):                           # DEBUG PC 6/7/14: widen the window (late S load)
                dm_utils.nanosleep_1us()
        s0 = tmem_ld32(sbase)
        s1 = tmem_ld32(sbase + 32)
        s2 = tmem_ld32(sbase + 64)
        s3 = tmem_ld32(sbase + 96)
        tmem_wait_ld()
        if const_expr(self.pc_inj != 7):
            dm_utils.tc_fence_before()
            dm_utils.mbar_arrive(mbar + 96 + 8 * slot)                      # S_empty[slot](j): S(j) in registers (MMA: QK(j+2))
            dm_utils.mbar_arrive(mbar + 112 + 8 * slot)                     # W1L[slot](j) (correction: SF(j-1) into this slot)
        s = s0 + s1 + s2 + s3
        if const_expr(self.is_causal and is_first):
            for c in cutlass.range_constexpr(128):
                s[c] = -Float32.inf if Int32(c) > tidx else s[c]
        # ---- m-independent prefix (C6 softmax_step2, op for op) ----
        dm_c = [qsc * dm_ks[b] for b in range(4)]
        dm_bmax = [None] * 4
        for b in cutlass.range_constexpr(4):
            b0 = 32 * b
            lm = [utils.fmax(s[b0 + 3 * t], s[b0 + 3 * t + 1], s[b0 + 3 * t + 2]) for t in range(4)]
            for i in cutlass.range_constexpr(12, 32, 2):
                t = ((i - 12) // 2) % 4
                lm[t] = utils.fmax(lm[t], s[b0 + i], s[b0 + i + 1])
            v = utils.fmax(utils.fmax(lm[0], lm[1], lm[2]), lm[3])
            dm_bmax[b] = v * dm_c[b]
        dm_bias = [None] * 4
        for b in cutlass.range_constexpr(4):
            dm_bias[b] = Float32(7.0) - cute.arch.fmax(dm_utils.floor_f32(dm_bmax[b]), Float32(-1e30))
        for b in cutlass.range_constexpr(4):
            for i in cutlass.range_constexpr(0, 32, 2):
                s[32 * b + i], s[32 * b + i + 1] = cute.arch.fma_packed_f32x2(
                    (s[32 * b + i], s[32 * b + i + 1]), (dm_c[b], dm_c[b]), (dm_bias[b], dm_bias[b]))
        prow = sp_base + sb * 16384 + tidx * 128
        pst = []
        for b in cutlass.range_constexpr(4):
            for i in cutlass.range_constexpr(32):
                s[32 * b + i] = ex2_ftz(s[32 * b + i])
            for q in cutlass.range_constexpr(2):
                k = 32 * b + 16 * q
                w = [e4m3x4(s[k + 4 * t], s[k + 4 * t + 1], s[k + 4 * t + 2], s[k + 4 * t + 3]) for t in range(4)]
                chunk = Int32(2 * b + q)
                pst.append((prow + ((chunk ^ (tidx & 7)) << 4), w))
                if const_expr(self.pc_inj != 11):
                    dm_utils.st_shared_v4_b32(pst[-1][0], w[0], w[1], w[2], w[3])
        dm_bs = [None] * 4
        for b in cutlass.range_constexpr(4):
            ps = (s[32 * b], s[32 * b + 1])
            for i in cutlass.range_constexpr(2, 32, 2):
                ps = cute.arch.add_packed_f32x2(ps, (s[32 * b + i], s[32 * b + i + 1]))
            dm_bs[b] = ps[0] + ps[1]
        self.tr(j, 12 + tro, None, tr_w == 0)
        # ---- publish (C11 part layout: k 0,1,4,5 = bmax0..3, k 2,3,6,7 = bs0..3); reuse of part / sP at j+3: see __init__ c10 ----
        if const_expr(self.pc_inj == 5):
            dm_utils.tc_fence_before()
            dm_utils.mbar_arrive(partf)
            for _d in cutlass.range_constexpr(20):
                dm_utils.nanosleep_1us()
        for (k, v) in ((0, dm_bmax[0]), (1, dm_bmax[1]), (2, dm_bs[0]), (3, dm_bs[1]),
                       (4, dm_bmax[2]), (5, dm_bmax[3]), (6, dm_bs[2]), (7, dm_bs[3])):
            st_shared_f32(part + 4 * (k * 128 + tidx), v)
        if const_expr(self.pc_inj != 5):
            dm_utils.tc_fence_before()                                       # our S loads before the correction's last-tile SF stores
            dm_utils.mbar_arrive(partf)                                      # partial_full3[j%3]
        if const_expr(self.pc_inj != 11):
            cute.arch.fence_view_async_shared()
            dm_utils.tc_fence_before()
            dm_utils.mbar_arrive(pfull)                                      # P_full3[j%3] (our P codes)
        else:
            dm_utils.tc_fence_before()                                       # DEBUG PC 11: P_full before the P stores (+20 us)
            dm_utils.mbar_arrive(pfull)
            for _d in cutlass.range_constexpr(20):
                dm_utils.nanosleep_1us()
            for (a, w) in pst:
                dm_utils.st_shared_v4_b32(a, w[0], w[1], w[2], w[3])
            cute.arch.fence_view_async_shared()
        self.tr(j, 14 + tro, None, tr_w == 0)

    # -----------------------------------------------------------------------------------------------------------------
    @cute.jit
    def tail_c11(self, j, n_block, is_first: cutlass.Constexpr, mbar, stats, tidx, lane, obase, mKVW, kv_head_idx, batch_idx, cnt,
                 m_old, row_sum, J=None):
        """C11: the C8 WG0 ordered tail for tile j, run by the correction WG, then the lazy O rescale (C6 correction body).
        C14 (pers): J = global tile index (rings / slots / phases); j, cnt item-local."""
        Jg = j if const_expr(J is None) else J
        b3 = Jg % 3
        part = stats + 4 * (1024 + b3 * self.part_stride)
        if const_expr(self.pc_inj == 17):
            part = stats + 4 * (1024 + (Jg & 1) * self.part_stride)         # DEBUG PC 17: part ring depth 2
        dm_vw = Int32(mKVW[batch_idx, kv_head_idx, 2 * n_block + 1])
        if const_expr(self.pc_inj != 4):
            dm_utils.mbar_wait_parity(mbar + 176 + 8 * b3, (Jg // 3) & 1)     # partial_full3[j%3] (both WGs)
        dm_utils.tc_fence_after()
        if const_expr(self.pc_inj == 17):
            for _d in cutlass.range_constexpr(20):                           # DEBUG PC 17: read part(j) late
                dm_utils.nanosleep_1us()
        self.tr(j, 24, None, tidx == 0)
        bmax = [ld_shared_f32(part + 4 * (0 * 128 + tidx)), ld_shared_f32(part + 4 * (1 * 128 + tidx)),
                ld_shared_f32(part + 4 * (4 * 128 + tidx)), ld_shared_f32(part + 4 * (5 * 128 + tidx))]
        bsum = [ld_shared_f32(part + 4 * (2 * 128 + tidx)), ld_shared_f32(part + 4 * (3 * 128 + tidx)),
                ld_shared_f32(part + 4 * (6 * 128 + tidx)), ld_shared_f32(part + 4 * (7 * 128 + tidx))]
        kb = [cute.arch.fmax(dm_utils.floor_f32(bmax[b]), Float32(-1e30)) for b in range(4)]
        dm_xmax = utils.fmax(utils.fmax(bmax[0], bmax[1], bmax[2]), bmax[3])
        dm_mcand = dm_utils.ceil_f32(dm_xmax)
        acc_scale = Float32(1.0)
        row_max = dm_mcand
        if const_expr(is_first):
            acc_scale = Float32(0.0)
        else:
            row_max = m_old
            if dm_mcand > m_old + Float32(self.rescale_T):
                row_max = dm_mcand
                acc_scale = dm_utils.exp2i_f32(m_old - dm_mcand)
        m_safe = row_max if row_max != -Float32.inf else Float32(0.0)
        dm_w = [None] * 4
        dm_sfa = [None] * 4
        for b in cutlass.range_constexpr(4):
            e = kb[b] - Float32(7.0) - m_safe
            dm_sfa[b] = dm_utils.ue8m0_byte(e)
            dm_w[b] = dm_utils.byte_shl23_f32(dm_sfa[b])
        sfa_word = dm_utils.pack_bytes4(dm_sfa[0], dm_sfa[1], dm_sfa[2], dm_sfa[3])
        if j + 1 < cnt:
            if const_expr(self.pc_inj != 14):
                dm_utils.mbar_wait_parity(mbar + 112 + 8 * ((Jg + 1) & 1), ((Jg + 1) >> 1) & 1)   # W1L(j+1)
                dm_utils.tc_fence_after()
        sfa_a = (lane << 16) + (self.sf_col10(j, cnt) if const_expr(J is None) else self.sf_col14(Jg, j, cnt))
        if const_expr(self.pc_inj == 15):
            dm_utils.tc_fence_before()
            dm_utils.mbar_arrive(mbar + 128 + 8 * b3)                          # DEBUG PC 15: P_full before the SF stores
            for _d in cutlass.range_constexpr(20):
                dm_utils.nanosleep_1us()
        for r in cutlass.range_constexpr(4):
            tmem_st1(sfa_a + r, sfa_word)
        for r in cutlass.range_constexpr(8):
            tmem_st1(sfa_a + 8 + r, dm_vw)
        tmem_wait_st()
        if const_expr(self.pc_inj != 15):
            dm_utils.tc_fence_before()
            dm_utils.mbar_arrive(mbar + 128 + 8 * b3)                          # P_full3[j%3] (SF(j) in TMEM)
        self.tr(j, 25, None, tidx == 0)
        if const_expr(not is_first):
            dm_utils.mbar_wait_parity(mbar + 32, (Jg - 1) & 1)                 # O_full(PV(j-1))
            dm_utils.tc_fence_after()
            if const_expr(self.pc_inj == 10):
                dm_utils.tc_fence_before()
                dm_utils.mbar_arrive(mbar + 40)
                for _d in cutlass.range_constexpr(20):
                    dm_utils.nanosleep_1us()
            if cute.arch.vote_ballot_sync(acc_scale < Float32(1.0)) != 0:      # hd128 predicate
                for c in cutlass.range_constexpr(8):
                    o = list(tmem_ld32(obase + 32 * c))
                    tmem_wait_ld()
                    for k in cutlass.range_constexpr(0, 32, 2):
                        o[k], o[k + 1] = cute.arch.mul_packed_f32x2((o[k], o[k + 1]), (acc_scale, acc_scale))
                    tmem_st32(obase + 32 * c, o)
                tmem_wait_st()
            if const_expr(self.pc_inj != 10):
                dm_utils.tc_fence_before()
                dm_utils.mbar_arrive(mbar + 40)                                  # O_ready(j)
            self.tr(j, 26, None, tidx == 0)
        dm_l = Float32(0.0) if const_expr(is_first) else row_sum * acc_scale
        for b in cutlass.range_constexpr(4):
            dm_l = dm_l + bsum[b] * dm_w[b]
        return row_max, dm_l

    @cute.jit
    def correction_c11(self, mbar, stats, mO, mLSE, mKVW, m_block, head_idx, kv_head_idx, batch_idx, n_block_max, cnt, sk_base):
        tidx = cute.arch.thread_idx()[0] % 128
        lane = (cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4) * 32
        obase = (lane << 16) + self.tmem_o
        m_last = -Float32.inf
        l_last = Float32(0.0)
        m_last, l_last = self.tail_c11(Int32(0), n_block_max - 1, True, mbar, stats, tidx, lane, obase, mKVW, kv_head_idx,
                                       batch_idx, cnt, m_last, l_last)
        for i in cutlass.range(cnt - 1, unroll=1):
            m_last, l_last = self.tail_c11(i + 1, n_block_max - 2 - i, False, mbar, stats, tidx, lane, obase, mKVW, kv_head_idx,
                                           batch_idx, cnt, m_last, l_last)
        row = m_block * self.m_block_size + tidx
        row_sum = l_last
        zero_or_nan = row_sum == Float32(0.0) or row_sum != row_sum
        scale = cute.arch.rcp_approx(row_sum if not zero_or_nan else Float32(1.0))
        LN2_ = math.log(2.0)
        lse = (m_last * Float32(1.0) + (cute.math.log2(row_sum, fastmath=True) - Float32(0.0))) * LN2_ \
            if not zero_or_nan else -Float32.inf
        mLSE[batch_idx, head_idx, row] = lse
        if const_expr(self.epi == 10):
            dm_utils.mbar_arrive(mbar + 64)          # DEBUG PC epi 10 only: the load warp's poison trigger (final_full, unused by C11)
        self.epilogue(mbar, mO, m_block, head_idx, batch_idx, cnt, sk_base, tidx, obase, scale)

    # -----------------------------------------------------------------------------------------------------------------
    @cute.jit
    def correction(self, mbar, stats, mO, m_block, head_idx, batch_idx, cnt, sk_base):
        tidx = cute.arch.thread_idx()[0] % 128
        lane = (cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4) * 32
        obase = (lane << 16) + self.tmem_o
        for i in cutlass.range(cnt - 1, unroll=1):
            j = i + 1
            slot = j & 1
            if const_expr(self.c10):
                dm_utils.mbar_wait_parity(mbar + 152 + 8 * (j % 3), (j // 3) & 1)  # stats3[j%3] (C10)
            else:
                dm_utils.mbar_wait_parity(mbar + 48 + 8 * slot, (j >> 1) & 1)      # stats(j)
            self.tr(j, 24, None, tidx == 0)
            if const_expr(self.c10):
                acc_scale = ld_shared_f32(stats + 4 * (512 + (j % 3) * 128 + tidx))
            else:
                acc_scale = ld_shared_f32(stats + 4 * (slot * 128 + tidx))
            dm_utils.mbar_wait_parity(mbar + 32, (j - 1) & 1)                       # O_full(PV(j-1))
            dm_utils.tc_fence_after()
            self.tr(j, 25, None, tidx == 0)
            if const_expr(self.pc_inj == 10):
                # DEBUG PC 10: O_ready(j) arrives BEFORE the O rescale -> PV(j) may accumulate into unscaled O -> must corrupt
                dm_utils.tc_fence_before()
                dm_utils.mbar_arrive(mbar + 40)
                for _d in cutlass.range_constexpr(20):          # widen the window: the rescale lands ~20 us after O_ready
                    dm_utils.nanosleep_1us()
            if cute.arch.vote_ballot_sync(acc_scale < Float32(1.0)) != 0:      # hd128 predicate
                for c in cutlass.range_constexpr(8):
                    o = list(tmem_ld32(obase + 32 * c))
                    tmem_wait_ld()
                    for k in cutlass.range_constexpr(0, 32, 2):
                        o[k], o[k + 1] = cute.arch.mul_packed_f32x2((o[k], o[k + 1]), (acc_scale, acc_scale))
                    tmem_st32(obase + 32 * c, o)
                tmem_wait_st()
            if const_expr(self.pc_inj != 10):
                dm_utils.tc_fence_before()
                dm_utils.mbar_arrive(mbar + 40)                                      # O_ready(j)
            self.tr(j, 26, None, tidx == 0)
        dm_utils.mbar_wait_parity(mbar + 64, Int32(0))
        scale = ld_shared_f32(stats + 4 * (2 * 128 + tidx))
        self.epilogue(mbar, mO, m_block, head_idx, batch_idx, cnt, sk_base, tidx, obase, scale)

    @cute.jit
    def epilogue(self, mbar, mO, m_block, head_idx, batch_idx, cnt, sk_base, tidx, obase, scale):
        dm_utils.mbar_wait_parity(mbar + 32, (cnt - 1) & 1)                         # O_full(PV(last))
        dm_utils.tc_fence_after()
        N = mO.shape[1]
        H = mO.shape[2]
        row = m_block * self.m_block_size + tidx
        base = Int64(mO.iterator.toint()) + (Int64((batch_idx * N + row) * H + head_idx) * Int64(self.head_dim * 2))
        if const_expr(self.epi == 0):
            for c in cutlass.range_constexpr(8):
                o = tmem_ld32(obase + 32 * c)
                tmem_wait_ld()
                for q in cutlass.range_constexpr(4):
                    wv = [bf16x2(mul_rn(o[8 * q + 2 * t], scale), mul_rn(o[8 * q + 2 * t + 1], scale)) for t in range(4)]
                    st_global_v4_b32(base + Int64(64 * c + 16 * q), wv[0], wv[1], wv[2], wv[3])
        else:
            # F1a: O_full(last) = the last PV (last reader of the sK/sV ring; every TMA load into it was consumed) completed.
            # The ring was last written by TMA (async proxy): order our generic writes after it.
            fence_proxy_async_shared()
            RS = 528                                                     # 512 B row + 16 B pad: conflict-free both ways
            srow = sk_base + Int32(RS) * tidx
            ln = tidx % 32
            w0 = tidx - ln                                               # first row of this warp (TMEM lane quadrant)
            if const_expr(self.epi == 8):                                # DEBUG PC (deterministic): copy out BEFORE staging
                for r in cutlass.range(32, unroll=4):
                    rr = w0 + r
                    v = ld_shared_v4_b32(sk_base + Int32(RS) * rr + 16 * ln)
                    gaddr = Int64(mO.iterator.toint()) + (Int64((batch_idx * N + m_block * self.m_block_size + rr) * H
                                                                + head_idx) * Int64(self.head_dim * 2)) + Int64(16 * ln)
                    st_global_v4_b32(gaddr, v[0], v[1], v[2], v[3])
            if const_expr(self.epi == 2):
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
            else:
                for c in cutlass.range_constexpr(8):
                    o = tmem_ld32(obase + 32 * c)
                    tmem_wait_ld()
                    for q in cutlass.range_constexpr(4):
                        wv = [bf16x2(mul_rn(o[8 * q + 2 * t], scale), mul_rn(o[8 * q + 2 * t + 1], scale)) for t in range(4)]
                        st_shared_v4_b32(srow + (64 * c + 16 * q), wv[0], wv[1], wv[2], wv[3])
            bar_warp_sync()                                              # warp-local hand-off: a warp copies only its own rows
            if const_expr(self.epi == 10):                               # DEBUG PC: hold the staged rows (see load warp)
                for _ in cutlass.range(30, unroll=1):
                    dm_utils.nanosleep_1us()
            if const_expr(self.epi == 9):
                w0 = (w0 + 32) % 128                                     # DEBUG PC (racy): another warp's rows, no CTA barrier
            if const_expr(self.epi != 8):
                for r in cutlass.range(32, unroll=4):
                    rr = w0 + r
                    v = ld_shared_v4_b32(sk_base + Int32(RS) * rr + 16 * ln)
                    gaddr = Int64(mO.iterator.toint()) + (Int64((batch_idx * N + m_block * self.m_block_size + rr) * H + head_idx)
                                                          * Int64(self.head_dim * 2)) + Int64(16 * ln)
                    st_global_v4_b32(gaddr, v[0], v[1], v[2], v[3])

    # =================================================================================================================
    # C14 persistent CTAs (fork hd256-fwd6, DM_HD256_FWD_PERS=1; see __init__ pers).  Only traced when self.pers.
    # =================================================================================================================
    @cute.jit
    def item_t2(self, u, tsd):
        """T2 split-tail map of linear CTA id u -> (m_block, head, batch) (same expressions as kernel(), without PC 18)."""
        nb_div, hb_div, h_div, tm, nb = tsd
        n_m = cute.arch.grid_dim()[0]
        n_h = cute.arch.grid_dim()[1]
        n_hb = n_h * cute.arch.grid_dim()[2]
        pair = Int32(0)
        m_block = Int32(0)
        if u < n_hb * nb:
            pair, r = divmod(u, nb_div)
            m_block = n_m - 1 - r
        else:
            q, r = divmod(u - n_hb * nb, hb_div)
            pair = r
            m_block = tm - 1 - q
            if const_expr(self.pc_inj == 18):
                m_block = m_block + 1                   # DEBUG PC 18: non-bijective map (hole at m_block 0; in bounds)
                if m_block > n_m - 1:
                    m_block = n_m - 1
        batch_idx, head_idx = divmod(pair, h_div)
        return m_block, head_idx, batch_idx

    @cute.jit
    def clc_next(self, mbar, resp_base, k, issuer: cutlass.Constexpr):
        """Claim for item k+1 (slot k&1, use n = k>>1 of that slot).  issuer (load warp): one thread waits the slot's previous
        acks (n >= 1), orders the old generic reads before the async response write, arrives expect_tx(16) and issues try_cancel.
        Every consuming warp: wait full, decode (all lanes), warp sync, lane 0 acks empty.  Returns (u, valid)."""
        s = k & 1
        n = k >> 1
        full = mbar + 224 + 8 * s
        resp = resp_base + 16 * s
        if const_expr(issuer):
            if cute.arch.lane_idx() == 0:
                if n >= 1:
                    dm_utils.mbar_wait_parity(mbar + 240 + 8 * s, (n - 1) & 1)
                fence_proxy_async_shared()
                dm_utils.mbar_arrive_expect_tx(full, 16)
                clc_try_cancel(resp, full)
        dm_utils.mbar_wait_parity(full, n & 1)
        valid, x, y, z = clc_decode(resp)
        bar_warp_sync()
        if cute.arch.lane_idx() == 0:
            dm_utils.mbar_arrive(mbar + 240 + 8 * s)
        u = x + cute.arch.grid_dim()[0] * (y + cute.arch.grid_dim()[1] * z)
        return u, valid

    def kseq14(self, m, cnt):
        """C14 ring order of an item: K0, K1, V0, K2, then the C7c order (K(m) at 2m-2 for m >= 3); K2 at 3 when cnt >= 3."""
        return self.kseq(m) + Int32(Int32(m) == 2)

    def vseq14(self, m, cnt):
        """C14: V0 at 2 when cnt >= 2 (C7c: 3 for cnt >= 3), else the C7c sequence."""
        return self.vseq(m, cnt) - Int32(Int32(m) == 0) * Int32(Int32(cnt) >= 3)

    @cute.jit
    def roles_pers(self, warp_idx, mQ, mK, mV, mO, mLSE, mQS, mKVW, tma_atom_Q, tma_atom_K, tma_atom_V, sQ, sK, sV, sP,
                   thr_mma_qk, thr_mma_pv, tiled_mma_qk, tiled_mma_pv, acc_pv, pipeline_q, pipeline_kv, mbar, stats, sp_base,
                   tmem, tmem_alloc_barrier, resp_base, tsd):
        bidx, bidy, bidz = cute.arch.block_idx()
        u0 = bidx + cute.arch.grid_dim()[0] * (bidy + cute.arch.grid_dim()[1] * bidz)
        if warp_idx == self.load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            self.load_pers(mQ, mK, mV, tma_atom_Q, tma_atom_K, tma_atom_V, sQ, sK, sV, thr_mma_qk, thr_mma_pv, pipeline_q,
                           pipeline_kv, mbar, resp_base, u0, tsd)
        if warp_idx == 15:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            self.pv_pers(tiled_mma_pv, sV, sP, acc_pv, pipeline_kv, mbar, resp_base, u0, tsd)
        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            tmem.allocate(512)
            tmem_alloc_barrier.arrive_and_wait_unaligned()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            self.qk_pers(tiled_mma_qk, sQ, sK, pipeline_q, pipeline_kv, mbar, resp_base, u0, tsd)
            tmem.relinquish_alloc_permit()
            if const_expr(self.ord_td):
                dm_utils.tc_fence_before()
            tmem_alloc_barrier.arrive_and_wait_unaligned()
            if const_expr(self.ord_td):
                dm_utils.tc_fence_after()
            tmem.free(tmem_ptr)
        if warp_idx <= self.softmax_warp_ids[-1]:
            cute.arch.setmaxregister_increase(self.num_regs_softmax)
            tmem_alloc_barrier.arrive_and_wait_unaligned()
            self.softmax_pers(mbar, sp_base, stats, mQS, mKVW, resp_base, u0, tsd)
            if const_expr(self.ord_td):
                dm_utils.tc_fence_before()
            tmem_alloc_barrier.arrive_unaligned()
        if warp_idx >= self.correction_warp_ids[0] and warp_idx <= self.correction_warp_ids[-1]:
            if const_expr(self.num_regs_correction > 128):
                cute.arch.setmaxregister_increase(self.num_regs_correction)
            else:
                cute.arch.setmaxregister_decrease(self.num_regs_correction)
            tmem_alloc_barrier.arrive_and_wait_unaligned()
            self.correction_pers(mbar, stats, mO, mLSE, mKVW, Int32(sK.iterator.toint()), resp_base, u0, tsd)
            if const_expr(self.ord_td):
                dm_utils.tc_fence_before()
            tmem_alloc_barrier.arrive_unaligned()

    @cute.jit
    def load_one14(self, atom, tg, ts, pipeline_kv, mbar, blk, seq, gate, k):
        """C14 TMA load of global sequence seq; the item's gated sequence (the previous epilogue's staging stage) first waits
        EPI_done(k-1)."""
        if k >= 1 and seq == gate:
            if const_expr(self.pc_inj != 22):
                dm_utils.mbar_wait_parity(mbar + 216, (k - 1) & 1)          # EPI_done(k-1): staged O copied out
        self.load_one(atom, tg, ts, pipeline_kv, blk, seq)

    @cute.jit
    def load_pers(self, mQ, mK, mV, tma_atom_Q, tma_atom_K, tma_atom_V, sQ, sK, sV, thr_mma_qk, thr_mma_pv, pipeline_q,
                  pipeline_kv, mbar, resp_base, u0, tsd):
        from flash_attn.cute.pipeline import _make_state
        k = Int32(0)
        J0 = Int32(0)
        u = u0
        valid = Int32(1)
        while valid != 0:
            m_block, head_idx, batch_idx = self.item_t2(u, tsd)
            kv_head_idx = head_idx // self.qhead_per_kvhead
            cnt = m_block + 1
            n_block_max = cnt
            gQ = cute.local_tile(mQ[None, None, head_idx, batch_idx], (self.m_block_size, self.head_dim), (m_block, 0))
            gK = cute.local_tile(mK[None, None, kv_head_idx, batch_idx], (self.n_block_size, self.head_dim), (None, 0))
            gV = cute.local_tile(mV[None, None, kv_head_idx, batch_idx], (self.head_dim, self.n_block_size), (0, None))
            tQsQ, tQgQ = cpasync.tma_partition(tma_atom_Q, 0, cute.make_layout(1), cute.group_modes(sQ, 0, 3),
                                               cute.group_modes(thr_mma_qk.partition_A(gQ), 0, 3))
            tKsK, tKgK = cpasync.tma_partition(tma_atom_K, 0, cute.make_layout(1), cute.group_modes(sK, 0, 3),
                                               cute.group_modes(thr_mma_qk.partition_B(gK), 0, 3))
            tVsV, tVgV = cpasync.tma_partition(tma_atom_V, 0, cute.make_layout(1), cute.group_modes(sV, 0, 3),
                                               cute.group_modes(thr_mma_pv.partition_B(gV), 0, 3))
            base = 2 * J0
            gate = base + 1 + 2 * Int32(cnt >= 2)
            q_st = _make_state(Int32(0), (k & 1) ^ 1)
            if const_expr(self.pc_inj == 24):                              # DEBUG PC 24: Q first (not behind K0's stage)
                pipeline_q.producer_acquire(q_st)
                cute.copy(tma_atom_Q, tQgQ, tQsQ[None, 0], tma_bar_ptr=pipeline_q.producer_get_barrier(q_st))
            self.load_one14(tma_atom_K, tKgK, tKsK, pipeline_kv, mbar, n_block_max - 1, base, gate, k)
            if const_expr(self.pc_inj != 24):
                pipeline_q.producer_acquire(q_st)
                cute.copy(tma_atom_Q, tQgQ, tQsQ[None, 0], tma_bar_ptr=pipeline_q.producer_get_barrier(q_st))
            if cnt > 1:
                self.load_one14(tma_atom_K, tKgK, tKsK, pipeline_kv, mbar, n_block_max - 2, base + 1, gate, k)
            for i in cutlass.range(cnt, unroll=1):
                if i == 0:
                    self.load_one14(tma_atom_V, tVgV, tVsV, pipeline_kv, mbar, n_block_max - 1, base + self.vseq14(0, cnt), gate, k)
                    if cnt > 2:
                        self.load_one14(tma_atom_K, tKgK, tKsK, pipeline_kv, mbar, n_block_max - 3, base + 3, gate, k)
                else:
                    if i + 2 < cnt:
                        self.load_one14(tma_atom_K, tKgK, tKsK, pipeline_kv, mbar, n_block_max - 3 - i, base + self.kseq(i + 2),
                                        gate, k)
                    self.load_one14(tma_atom_V, tVgV, tVsV, pipeline_kv, mbar, n_block_max - 1 - i, base + self.vseq(i, cnt), gate, k)
            u_n, valid_n = self.clc_next(mbar, resp_base, k, True)
            J0 = J0 + cnt
            k = k + 1
            u = u_n
            valid = valid_n

    @cute.jit
    def qk_pers(self, tiled_mma_qk, sQ, sK, pipeline_q, pipeline_kv, mbar, resp_base, u0, tsd):
        """C14 warp 12: QK(J) for every tile of every item.  Guards: S_empty(J-2); PV completion dep(J) via O_done2:
        j >= 2: PV(J-3) (C13); j = 1: PV(J0-1) for k >= 1 (previous last tile's SF at +80 of this slot); j = 0: PV(J0-2)."""
        from flash_attn.cute.pipeline import _make_state
        tSrQ = tiled_mma_qk.make_fragment_A(sQ)
        tSrK = tiled_mma_qk.make_fragment_B(sK)
        S = self.kv_stage
        k = Int32(0)
        J0 = Int32(0)
        u = u0
        valid = Int32(1)
        while valid != 0:
            m_block, head_idx, batch_idx = self.item_t2(u, tsd)
            cnt = m_block + 1
            q_st = _make_state(Int32(0), k & 1)
            pipeline_q.consumer_wait(q_st)
            for j in cutlass.range(cnt, unroll=1):
                J = J0 + j
                if J >= 2:
                    if const_expr(self.pc_inj != 6):
                        dm_utils.mbar_wait_parity(mbar + 96 + 8 * (J & 1), ((J - 2) >> 1) & 1)   # S_empty(J-2)
                    else:
                        if J == 2:
                            dm_utils.mbar_wait_parity(mbar + 96, Int32(0))  # DEBUG PC 6: keep the first wait (as mma_c13_qk)
                dep = J - 3
                if j == 0:
                    dep = J0 - 2
                if j == 1 and k >= 1:
                    if const_expr(self.pc_inj != 23):
                        dep = J0 - 1
                    else:
                        dep = J0 - 2
                if dep >= 0:
                    dm_utils.mbar_wait_parity(mbar + 200 + 8 * (dep & 1), (dep >> 1) & 1)        # O_done2: PV(dep) done
                dm_utils.tc_fence_after()
                ks = 2 * J0 + self.kseq14(j, cnt)
                self.kv_wait(pipeline_kv, ks)
                if const_expr(self.pc_inj == 24):
                    if j == cnt - 1:
                        pipeline_q.consumer_release(q_st)                   # DEBUG PC 24: Q released before the last QK
                        for _d in cutlass.range_constexpr(20):
                            dm_utils.nanosleep_1us()
                self.qk_ptx(tSrQ, tSrK, sQ, sK, ks % S, J & 1)
                with cute.arch.elect_one():
                    dm_utils.tc_commit(mbar + 8 * (J & 1))                  # S_full[J&1]
                pipeline_kv.consumer_release_w_index(ks % S)
            if const_expr(self.pc_inj != 24):
                pipeline_q.consumer_release(q_st)
            u_n, valid_n = self.clc_next(mbar, resp_base, k, False)
            J0 = J0 + cnt
            k = k + 1
            u = u_n
            valid = valid_n

    @cute.jit
    def pv_pers(self, tiled_mma_pv, sV, sP, acc_pv, pipeline_kv, mbar, resp_base, u0, tsd):
        """C14 warp 15: PV(J) (mma_pv C10/C13 path with global J; zero-init at item-local j = 0)."""
        tOrP = tiled_mma_pv.make_fragment_A(sP)
        tOrV = tiled_mma_pv.make_fragment_B(sV)
        pv_op = cute.make_mma_atom(tiled_mma_pv.op).op
        S = self.kv_stage
        k = Int32(0)
        J0 = Int32(0)
        u = u0
        valid = Int32(1)
        while valid != 0:
            m_block, head_idx, batch_idx = self.item_t2(u, tsd)
            cnt = m_block + 1
            for j in cutlass.range(cnt, unroll=1):
                J = J0 + j
                vs = 2 * J0 + self.vseq14(j, cnt)
                pipeline_kv.consumer_wait_w_index_phase(vs % S, (vs // S) & 1)
                pb = J % 3
                dm_utils.mbar_wait_parity(mbar + 128 + 8 * pb, (J // 3) & 1)     # P_full3[J%3]
                if const_expr(self.pc_inj == 16):
                    pb = J & 1                                              # DEBUG PC 16: sP ring depth 2, PV(J) issued late
                    for _d in cutlass.range_constexpr(20):
                        dm_utils.nanosleep_1us()
                if j >= 1:
                    dm_utils.mbar_wait_parity(mbar + 40, (J - k - 1) & 1)        # O_ready(J): round J - k - 1
                dm_utils.tc_fence_after()
                if const_expr(self.pc_inj == 23):
                    if j == cnt - 1:
                        for _d in cutlass.range_constexpr(20):              # DEBUG PC 23: PV(last) late
                            dm_utils.nanosleep_1us()
                sfa_c = self.sf_col14(J, j, cnt)
                dm_utils.gemm_ptx_bs(
                    pv_op, self.tmem_o, tOrP[None, None, None, pb], tOrV[None, None, None, vs % S],
                    sP[None, None, None, pb], sV[None, None, None, vs % S], sfa_c, sfa_c + 8,
                    tuple((0, kk) for kk in range(4)), tuple((0, kk) for kk in range(4)), zero_init=(j == 0))
                with cute.arch.elect_one():
                    dm_utils.tc_commit(mbar + 32)                                # O_full(J)
                    dm_utils.tc_commit(mbar + 200 + 8 * (J & 1))                 # O_done2[J&1]
                if const_expr(self.pc_inj in (1, 2)):
                    if const_expr(self.pc_inj == 2):
                        dm_utils.mbar_wait_parity(mbar + 32, J & 1)
                        dm_utils.tc_fence_after()
                    lane0 = (cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4) * 32
                    for r in cutlass.range_constexpr(4):
                        tmem_st1((lane0 << 16) + sfa_c + r, Int32(-2139062144))   # DEBUG PC 1 / 2: poison SFA(J) (0x80808080)
                    tmem_wait_st()
                pipeline_kv.consumer_release_w_index(vs % S)
            u_n, valid_n = self.clc_next(mbar, resp_base, k, False)
            J0 = J0 + cnt
            k = k + 1
            u = u_n
            valid = valid_n

    @cute.jit
    def softmax_pers(self, mbar, sp_base, stats, mQS, mKVW, resp_base, u0, tsd):
        """C14 softmax WGs: WG h owns the tiles with global J&1 == h (item-local first tile with the diagonal mask)."""
        tidx = cute.arch.thread_idx()[0] % 128
        wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) // 4
        lane = (cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4) * 32
        k = Int32(0)
        J0 = Int32(0)
        u = u0
        valid = Int32(1)
        while valid != 0:
            m_block, head_idx, batch_idx = self.item_t2(u, tsd)
            kv_head_idx = head_idx // self.qhead_per_kvhead
            cnt = m_block + 1
            n_block_max = cnt
            row = m_block * self.m_block_size + tidx
            qsc = Float32(mQS[batch_idx, head_idx, row // 32]) * Float32(math.log2(math.e))
            if ((J0 + wg) & 1) == 0:                                   # this WG owns local tiles 0, 2, ...
                self.softmax_step4(Int32(0), n_block_max - 1, True, mbar, stats, sp_base, tidx, lane, qsc, mKVW, kv_head_idx,
                                   batch_idx, J=J0)
                for i in cutlass.range((cnt - 1) // 2, unroll=1):
                    j = 2 * i + 2
                    self.softmax_step4(j, n_block_max - 1 - j, False, mbar, stats, sp_base, tidx, lane, qsc, mKVW, kv_head_idx,
                                       batch_idx, J=J0 + j)
            else:
                for i in cutlass.range(cnt // 2, unroll=1):
                    j = 2 * i + 1
                    self.softmax_step4(j, n_block_max - 1 - j, False, mbar, stats, sp_base, tidx, lane, qsc, mKVW, kv_head_idx,
                                       batch_idx, J=J0 + j)
            u_n, valid_n = self.clc_next(mbar, resp_base, k, False)
            J0 = J0 + cnt
            k = k + 1
            u = u_n
            valid = valid_n

    @cute.jit
    def correction_pers(self, mbar, stats, mO, mLSE, mKVW, sk_base, resp_base, u0, tsd):
        """C14 correction WG: tails (global J), LSE, then (after reading the next claim) the staged epilogue into the stage
        of the next item's gated load, EPI_done."""
        tidx = cute.arch.thread_idx()[0] % 128
        lane = (cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4) * 32
        obase = (lane << 16) + self.tmem_o
        k = Int32(0)
        J0 = Int32(0)
        u = u0
        valid = Int32(1)
        while valid != 0:
            m_block, head_idx, batch_idx = self.item_t2(u, tsd)
            kv_head_idx = head_idx // self.qhead_per_kvhead
            cnt = m_block + 1
            n_block_max = cnt
            m_last = -Float32.inf
            l_last = Float32(0.0)
            m_last, l_last = self.tail_c11(Int32(0), n_block_max - 1, True, mbar, stats, tidx, lane, obase, mKVW, kv_head_idx,
                                           batch_idx, cnt, m_last, l_last, J=J0)
            for i in cutlass.range(cnt - 1, unroll=1):
                m_last, l_last = self.tail_c11(i + 1, n_block_max - 2 - i, False, mbar, stats, tidx, lane, obase, mKVW,
                                               kv_head_idx, batch_idx, cnt, m_last, l_last, J=J0 + i + 1)
            row = m_block * self.m_block_size + tidx
            row_sum = l_last
            zero_or_nan = row_sum == Float32(0.0) or row_sum != row_sum
            scale = cute.arch.rcp_approx(row_sum if not zero_or_nan else Float32(1.0))
            LN2_ = math.log(2.0)
            lse = (m_last * Float32(1.0) + (cute.math.log2(row_sum, fastmath=True) - Float32(0.0))) * LN2_ \
                if not zero_or_nan else -Float32.inf
            mLSE[batch_idx, head_idx, row] = lse
            if const_expr(self.pc_inj == 25):
                if tidx == 0:
                    mLSE[batch_idx, head_idx, row] = Float32(k) + Float32(1000.0) * Float32(dm_utils.smid())   # DEBUG 25: item k of CTA
            u_n, valid_n = self.clc_next(mbar, resp_base, k, False)
            J0n = J0 + cnt
            cnt_n = Int32(3)
            if valid_n != 0:
                mb_n, h_n, b_n = self.item_t2(u_n, tsd)
                cnt_n = mb_n + 1
            stage = (2 * J0n + 1 + 2 * Int32(cnt_n >= 2)) % self.kv_stage
            self.epilogue_pers(mbar, mO, m_block, head_idx, batch_idx, J0n - 1, sk_base + stage * (self.tma_bytes_kv), tidx,
                               obase, scale)
            J0 = J0n
            k = k + 1
            u = u_n
            valid = valid_n

    @cute.jit
    def epilogue_pers(self, mbar, mO, m_block, head_idx, batch_idx, Jlast, stg, tidx, obase, scale):
        """C14 O epilogue: O_full(PV(Jlast)), then per column half h (TMEM chunks 4h..4h+3, chunk c+1 loaded while c is
        converted): bf16 rows (mul.rn + cvt, = epi 2 bits) into this warp's 8 KiB of the stage (row r: 256 B, 16-B chunk q at
        q ^ (r & 15)), bar.warp.sync, coalesced copy-out (2 rows x 256 B per instruction), bar.warp.sync.  Then
        fence.proxy.async + EPI_done (the load warp's gated TMA into this stage)."""
        dm_utils.mbar_wait_parity(mbar + 32, Jlast & 1)                         # O_full(PV(last))
        dm_utils.tc_fence_after()
        fence_proxy_async_shared()                                              # ring last written by TMA (async proxy)
        N = mO.shape[1]
        H = mO.shape[2]
        ln = tidx % 32
        w0 = tidx - ln
        wbase = stg + Int32(256) * w0                                           # this warp's 32 rows x 256 B
        srow = wbase + Int32(256) * ln
        sw = ln & 15
        rsub = ln >> 4                                                          # copy-out: row 2r + rsub, chunk ln & 15
        gbase = Int64(mO.iterator.toint()) + (Int64((batch_idx * N + m_block * self.m_block_size + w0) * H + head_idx)
                                              * Int64(self.head_dim * 2)) + Int64(16 * (ln & 15))
        rstride = Int64(H * self.head_dim * 2)
        o = tmem_ld32(obase)
        tmem_wait_ld()
        for h in cutlass.range_constexpr(2):
            for cc in cutlass.range_constexpr(4):
                c = 4 * h + cc
                if const_expr(c < 7):
                    on = tmem_ld32(obase + 32 * (c + 1))
                for q in cutlass.range_constexpr(4):
                    wv = [bf16x2(mul_rn(o[8 * q + 2 * t], scale), mul_rn(o[8 * q + 2 * t + 1], scale)) for t in range(4)]
                    st_shared_v4_b32(srow + ((Int32(4 * cc + q) ^ sw) << 4), wv[0], wv[1], wv[2], wv[3])
                if const_expr(c < 7):
                    tmem_wait_ld()
                    o = on
            bar_warp_sync()
            if const_expr(self.pc_inj == 22 and h == 0):
                for _d in cutlass.range_constexpr(20):                          # DEBUG PC 22: copy-out late (TMA lands first)
                    dm_utils.nanosleep_1us()
            for r in cutlass.range(16, unroll=4):
                rr = 2 * r + rsub
                v = ld_shared_v4_b32(wbase + Int32(256) * rr + (((ln & 15) ^ (rr & 15)) << 4))
                st_global_v4_b32(gbase + Int64(rr) * rstride + Int64(256 * h), v[0], v[1], v[2], v[3])
            bar_warp_sync()
        fence_proxy_async_shared()
        dm_utils.mbar_arrive(mbar + 216)                                        # EPI_done
