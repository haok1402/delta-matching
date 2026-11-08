"""Small PTX helpers for the sm100-mx kernels (CuTe DSL inline asm)."""
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm


def _f32_unary(asm, x, loc=None, ip=None):
    return Float32(llvm.inline_asm(T.f32(), [Float32(x).ir_value(loc=loc, ip=ip)], asm, "=f,f",
                                   has_side_effects=False, is_align_stack=False,
                                   asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def floor_f32(x: Float32, *, loc=None, ip=None) -> Float32:
    return _f32_unary("cvt.rmi.f32.f32 $0, $1;", x, loc, ip)


@dsl_user_op
def ceil_f32(x: Float32, *, loc=None, ip=None) -> Float32:
    return _f32_unary("cvt.rpi.f32.f32 $0, $1;", x, loc, ip)


@dsl_user_op
def exp2i_f32(e: Float32, *, loc=None, ip=None) -> Float32:
    """2^e for an integer-valued fp32 e (exact); 0 if e < -126; e is clamped to <= 127."""
    return _f32_unary(
        "{\n\t.reg .s32 i;\n\t.reg .pred p;\n\t"
        "cvt.rni.s32.f32 i, $1;\n\t"
        "min.s32 i, i, 127;\n\t"
        "setp.lt.s32 p, i, -126;\n\t"
        "add.s32 i, i, 127;\n\t"
        "shl.b32 i, i, 23;\n\t"
        "selp.b32 i, 0, i, p;\n\t"
        "mov.b32 $0, i;\n\t}", e, loc, ip)


@dsl_user_op
def ue8m0_byte(e: Float32, *, loc=None, ip=None) -> Int32:
    """UE8M0 byte for 2^e (e integer-valued fp32): clamp(e + 127, 0, 254)."""
    return Int32(llvm.inline_asm(
        T.i32(), [Float32(e).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .s32 i;\n\t"
        "cvt.rni.s32.f32 i, $1;\n\t"
        "add.s32 i, i, 127;\n\t"
        "max.s32 i, i, 0;\n\t"
        "min.s32 i, i, 254;\n\t"
        "mov.b32 $0, i;\n\t}",
        "=r,f", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def pow2_exponent_byte(s: Float32, *, loc=None, ip=None) -> Int32:
    """Biased exponent field of a power-of-2 fp32 (= its UE8M0 byte)."""
    return Int32(llvm.inline_asm(
        T.i32(), [Float32(s).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b32 i;\n\tmov.b32 i, $1;\n\tbfe.u32 $0, i, 23, 8;\n\t}",
        "=r,f", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def pack_bytes4(b0: Int32, b1: Int32, b2: Int32, b3: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(llvm.inline_asm(
        T.i32(), [Int32(b).ir_value(loc=loc, ip=ip) for b in (b0, b1, b2, b3)],
        "{\n\t.reg .b32 t0, t1;\n\t"
        "prmt.b32 t0, $1, $2, 0x0040;\n\t"     # bytes: [b0.0, b1.0, -, -]
        "prmt.b32 t1, $3, $4, 0x0040;\n\t"
        "prmt.b32 $0, t0, t1, 0x5410;\n\t}",
        "=r,r,r,r,r", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def bitcast_i32_f32(x: Int32, *, loc=None, ip=None) -> Float32:
    return Float32(llvm.inline_asm(T.f32(), [Int32(x).ir_value(loc=loc, ip=ip)], "mov.b32 $0, $1;", "=f,r",
                                   has_side_effects=False, is_align_stack=False,
                                   asm_dialect=llvm.AsmDialect.AD_ATT))


def _bs_idesc(idesc_plain: int, k: int) -> int:
    """Block-scaled (kind::mxf8f6f4, E8M0 scales) instruction descriptor from the plain f8f6f4 one:
    bits [4,6) c_format -> b_sf_id, bit 23 -> scale_format E8M0, bits [29,31) -> a_sf_id (= k-step byte)."""
    d = idesc_plain & ~(0x3 << 4) & ~(0x7 << 29)
    return d | (1 << 23) | ((k & 3) << 29) | ((k & 3) << 4)


def gemm_ptx_bs_ts(
    op, acc_tmem_addr, tCrA, tCrB, sB, sfa_tmem_addr, sfb_tmem_addr,
    mbar_ptr=None, mbar_phase=None, split_arrive=None, zero_init=False, tA_addr=None, cta_group: int = 1,
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
    offset_a = [cute.crd2idx((0, 0, k), tCrA_layout) for k in range(nk)]
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
                    "@P1 bra DONE; \n\tbra     LAB_WAIT; \n\tDONE: \n\t")
    instr = f"tcgen05.mma.cta_group::{cta_group}.kind::mxf8f6f4.block_scale.scale_vec::1X"

    def one(k, pred):
        return (f"add.u32 smem_desc_b_lo, smem_desc_b_lo_start, {hex(offset_b[k])};\n\t"
                f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                f"@leader_thread {instr} [tmem_acc], [tmem_a + {hex(offset_a[k])}], smem_desc_b, "
                f"{hex(_bs_idesc(idesc_plain, k))}, [tmem_sfa], [tmem_sfb], {pred};\n\t")

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
def pow2_sb(amax: Float32, *, loc=None, ip=None) -> Int32:
    """Biased exponent (UE8M0 byte) of the smallest power of 2 s with amax <= 448 s (amax >= 0 fp32):
    amax = m 2^E -> E-8 if m <= 1.75 else E-7; amax == 0 -> 127 (s = 1). Clamped to [1, 254]."""
    return Int32(llvm.inline_asm(
        T.i32(), [Float32(amax).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b32 a, e, m;\n\t.reg .pred pm, pz;\n\t"
        "mov.b32 a, $1;\n\t"
        "bfe.u32 e, a, 23, 8;\n\t"
        "and.b32 m, a, 0x7FFFFF;\n\t"
        "setp.gt.u32 pm, m, 0x600000;\n\t"
        "sub.s32 e, e, 8;\n\t"
        "@pm add.s32 e, e, 1;\n\t"
        "max.s32 e, e, 1;\n\t"
        "min.s32 e, e, 254;\n\t"
        "setp.eq.b32 pz, a, 0;\n\t"
        "selp.b32 $0, 127, e, pz;\n\t}",
        "=r,f", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def inv_pow2_from_sb(sb: Int32, *, loc=None, ip=None) -> Float32:
    """2^-(sb-127) as fp32 (exact)."""
    return Float32(llvm.inline_asm(
        T.f32(), [Int32(sb).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b32 t;\n\tsub.s32 t, 254, $1;\n\tshl.b32 $0, t, 23;\n\t}",
        "=f,r", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def add_sb_clamp(a: Int32, b: Int32, *, loc=None, ip=None) -> Int32:
    """UE8M0 byte of the product of two powers of 2 given their biased exponents: clamp(a + b - 127, 0, 254)."""
    return Int32(llvm.inline_asm(
        T.i32(), [Int32(a).ir_value(loc=loc, ip=ip), Int32(b).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .s32 t;\n\tadd.s32 t, $1, $2;\n\tsub.s32 t, t, 127;\n\tmax.s32 t, t, 0;\n\tmin.s32 $0, t, 254;\n\t}",
        "=r,r,r", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def redux_max_abs_f32(x: Float32, *, loc=None, ip=None) -> Float32:
    """Warp-wide max |x| (redux.sync.max.abs.f32, sm_100a)."""
    return Float32(llvm.inline_asm(
        T.f32(), [Float32(x).ir_value(loc=loc, ip=ip)],
        "redux.sync.max.abs.f32 $0, $1, 0xffffffff;",
        "=f,f", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


def gemm_ptx_bs(
    op, acc_tmem_addr, tCrA, tCrB, sA, sB, sfa_base, sfb_base, sfa_map, sfb_map,
    zero_init=False, tA_addr=None, cta_group: int = 1,
):
    """Block-scaled (kind::mxf8f6f4, UE8M0) MMA over all k-steps of tCrA/tCrB, SS (sA given) or TS (A in TMEM at tA_addr).
    sfa_map / sfb_map: per k-step (column offset added to sfa_base / sfb_base, sf byte id 0..3) python tuples.
    Mirrors flash_attn.cute.blackwell_helpers.gemm_ptx_partial."""
    from flash_attn.cute import mma_sm100_desc as sm100_desc
    from flash_attn.cute.blackwell_helpers import i64_to_i32x2
    is_ts = sA is None
    idesc_plain: int = sm100_desc.mma_op_to_idesc(op)
    nk = cute.size(tCrA.shape[2])
    assert len(sfa_map) == nk and len(sfb_map) == nk

    def desc_base(s, dtype_width, major_mode):
        b = sm100_desc.make_smem_desc_base(
            cute.recast_layout(128, dtype_width, s.layout[0]), s.iterator.type.swizzle_type,
            sm100_desc.Major.K if major_mode == cute.nvgpu.tcgen05.mma.OperandMajorMode.K else sm100_desc.Major.MN)
        return i64_to_i32x2(b)

    b_lo, b_hi = desc_base(sB, op.b_dtype.width, op.b_major_mode)
    offset_b = [cute.crd2idx((0, 0, k), tCrB.layout) for k in range(nk)]
    start_b = Int32(b_lo | sm100_desc.make_smem_desc_start_addr(sB[None, None, 0].iterator))
    if is_ts:
        tCrA_layout = cute.recast_layout(32, tCrA.element_type.width, tCrA.layout)
        offset_a = [cute.crd2idx((0, 0, k), tCrA_layout) for k in range(nk)]
        a_arg = Int32(cute.arch.make_warp_uniform(tA_addr))
        a_hi = None
    else:
        a_lo, a_hi = desc_base(sA, op.a_dtype.width, op.a_major_mode)
        offset_a = [cute.crd2idx((0, 0, k), tCrA.layout) for k in range(nk)]
        a_arg = Int32(cute.arch.make_warp_uniform(Int32(a_lo | sm100_desc.make_smem_desc_start_addr(sA[None, None, 0].iterator))))
    pred_str = "p" if isinstance(zero_init, cutlass.Boolean) else "0" if zero_init else "1"
    acc_arg = (Int32(1) - Int32(zero_init)) if isinstance(zero_init, cutlass.Boolean) else Int32(not zero_init)
    input_args = [a_arg.ir_value(), Int32(cute.arch.make_warp_uniform(start_b)).ir_value(), acc_arg.ir_value(),
                  Int32(cute.arch.make_warp_uniform(acc_tmem_addr)).ir_value(),
                  Int32(cute.arch.make_warp_uniform(sfa_base)).ir_value(),
                  Int32(cute.arch.make_warp_uniform(sfb_base)).ir_value()]
    instr = f"tcgen05.mma.cta_group::{cta_group}.kind::mxf8f6f4.block_scale.scale_vec::1X"
    body = ("{\n\t.reg .pred leader_thread;\n\t.reg .pred p;\n\t.reg .b32 tmem_acc, a0, sfa0, sfb0, sfa, sfb;\n\t"
            ".reg .b32 b_lo0, b_lo, b_hi, a_lo, a_hi;\n\t.reg .b64 bdesc, adesc;\n\t"
            "elect.sync _|leader_thread, -1;\n\t"
            "mov.b32 tmem_acc, $3;\n\tmov.b32 a0, $0;\n\tmov.b32 sfa0, $4;\n\tmov.b32 sfb0, $5;\n\t"
            "mov.b32 b_lo0, $1;\n\t"
            f"mov.b32 b_hi, {hex(b_hi)};\n\t"
            + ("" if is_ts else f"mov.b32 a_hi, {hex(a_hi)};\n\t")
            + "setp.ne.b32 p, $2, 0;\n\t")
    for k in range(nk):
        ca, ia = sfa_map[k]
        cb, ib = sfb_map[k]
        idesc = (idesc_plain & ~(0x3 << 4) & ~(0x7 << 29)) | (1 << 23) | ((ia & 3) << 29) | ((ib & 3) << 4)
        pred = pred_str if k == 0 else "1"
        body += (f"add.u32 b_lo, b_lo0, {hex(offset_b[k])};\n\t"
                 "mov.b64 bdesc, {b_lo, b_hi};\n\t"
                 f"add.u32 sfa, sfa0, {hex(ca)};\n\tadd.u32 sfb, sfb0, {hex(cb)};\n\t")
        if is_ts:
            body += f"@leader_thread {instr} [tmem_acc], [a0 + {hex(offset_a[k])}], bdesc, {hex(idesc)}, [sfa], [sfb], {pred};\n\t"
        else:
            body += (f"add.u32 a_lo, a0, {hex(offset_a[k])};\n\tmov.b64 adesc, {{a_lo, a_hi}};\n\t"
                     f"@leader_thread {instr} [tmem_acc], adesc, bdesc, {hex(idesc)}, [sfa], [sfb], {pred};\n\t")
    body += "}\n"
    llvm.inline_asm(None, input_args, body, ",".join(["r"] * len(input_args)),
                    has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


def gemm_ptx_bs_w_idx(tiled_mma, acc, tCrA, tCrB, sA, sB, A_idx=None, B_idx=None, **kw):
    rA = tCrA if A_idx is None else tCrA[None, None, None, A_idx]
    rB = tCrB if B_idx is None else tCrB[None, None, None, B_idx]
    sA_cur = None if sA is None else (sA if A_idx is None else sA[None, None, None, A_idx])
    sB_cur = sB if B_idx is None else sB[None, None, None, B_idx]
    op = cute.make_mma_atom(tiled_mma.op).op
    gemm_ptx_bs(op, acc.iterator.toint(), rA, rB, sA_cur, sB_cur, **kw)
