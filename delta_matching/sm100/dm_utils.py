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
def exp2_h2_e4m3x4(x0: Float32, x1: Float32, x2: Float32, x3: Float32, *, loc=None, ip=None):
    """4 x (fp32 -> f16, ex2.approx.f16x2, -> e4m3).  Returns (word of 4 e4m3 codes [x0 in byte 0],
    f16x2 exp of (x0, x1), f16x2 exp of (x2, x3)) as Int32 bit patterns."""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32()]),
        [Float32(v).ir_value(loc=loc, ip=ip) for v in (x0, x1, x2, x3)],
        "{\n\t.reg .b32 h0, h1, e0, e1;\n\t.reg .b16 c0, c1;\n\t"
        "cvt.rn.f16x2.f32 h0, $4, $3;\n\t"
        "cvt.rn.f16x2.f32 h1, $6, $5;\n\t"
        "ex2.approx.f16x2 e0, h0;\n\t"
        "ex2.approx.f16x2 e1, h1;\n\t"
        "cvt.rn.satfinite.e4m3x2.f16x2 c0, e0;\n\t"
        "cvt.rn.satfinite.e4m3x2.f16x2 c1, e1;\n\t"
        "mov.b32 $0, {c0, c1};\n\t"
        "mov.b32 $1, e0;\n\tmov.b32 $2, e1;\n\t}",
        "=r,=r,=r,f,f,f,f", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)
    return (Int32(llvm.extractvalue(T.i32(), res, [0])), Int32(llvm.extractvalue(T.i32(), res, [1])),
            Int32(llvm.extractvalue(T.i32(), res, [2])))


@dsl_user_op
def hadd2(a: Int32, b: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(llvm.inline_asm(T.i32(), [Int32(a).ir_value(loc=loc, ip=ip), Int32(b).ir_value(loc=loc, ip=ip)],
                                 "add.rn.f16x2 $0, $1, $2;", "=r,r,r", has_side_effects=False, is_align_stack=False,
                                 asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def h2_sum_f32(a: Int32, *, loc=None, ip=None) -> Float32:
    """f32(lo) + f32(hi) of an f16x2."""
    return Float32(llvm.inline_asm(
        T.f32(), [Int32(a).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b16 l, h;\n\t.reg .f32 fl, fh;\n\tmov.b32 {l, h}, $1;\n\t"
        "cvt.f32.f16 fl, l;\n\tcvt.f32.f16 fh, h;\n\tadd.f32 $0, fl, fh;\n\t}",
        "=f,r", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


def hadd2_tree(hs):
    """Pairwise f16x2 tree sum of a Python list (plain Python, unrolled at trace time)."""
    while len(hs) > 1:
        hs = [hadd2(hs[2 * t], hs[2 * t + 1]) for t in range(len(hs) // 2)]
    return hs[0]


@dsl_user_op
def byte_pow2_f32(word: Int32, j: int, *, loc=None, ip=None) -> Float32:
    """2^(byte_j(word) - 127) as fp32 (byte j of a packed UE8M0 word)."""
    return Float32(llvm.inline_asm(
        T.f32(), [Int32(word).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b32 t;\n\t" f"bfe.u32 t, $1, {8 * j}, 8;\n\t" "shl.b32 $0, t, 23;\n\t}",
        "=f,r", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))
