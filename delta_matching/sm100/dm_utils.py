"""Small PTX helpers for the sm100-mx kernels (CuTe DSL inline asm)."""
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Float64, Int32, const_expr
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm


def _f32_unary(asm, x, loc=None, ip=None):
    return Float32(llvm.inline_asm(T.f32(), [Float32(x).ir_value(loc=loc, ip=ip)], asm, "=f,f",
                                   has_side_effects=False, is_align_stack=False,
                                   asm_dialect=llvm.AsmDialect.AD_ATT))


import os as _os_impl
_TCF_IMPL = _os_impl.environ.get("DM_TCF_IMPL", "asm")   # asm | nvvm (tcgen05 fence emission)

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
def byte_shl23_f32(b: Int32, *, loc=None, ip=None) -> Float32:
    """bitcast(b << 23): 2^(b - 127) for a UE8M0 byte b in [1, 254], +0 for b == 0 (= exp2i_f32 of the same e)."""
    return Float32(llvm.inline_asm(T.f32(), [Int32(b).ir_value(loc=loc, ip=ip)], "shl.b32 $0, $1, 23;", "=f,r",
                                   has_side_effects=False, is_align_stack=False,
                                   asm_dialect=llvm.AsmDialect.AD_ATT))


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
                    + ("tcgen05.fence::after_thread_sync; \n\t" if TCF[0] >= 1 else ""))
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
        offset_a = [cute.crd2idx((0, 0, k), tCrA_layout) for k in range(nk)] if offs_a is None else list(offs_a)
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


@dsl_user_op
def tmem_st_x1(addr: Int32, w: Int32, *, loc=None, ip=None):
    """tcgen05.st.32x32b.x1: this thread's lane, one 32-bit column at addr (lane base in bits 16+)."""
    llvm.inline_asm(None, [Int32(addr).ir_value(loc=loc, ip=ip), Int32(w).ir_value(loc=loc, ip=ip)],
                    "tcgen05.st.sync.aligned.32x32b.x1.b32 [$0], {$1};", "r,r",
                    has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def tmem_st_x4(addr: Int32, w0: Int32, w1: Int32, w2: Int32, w3: Int32, *, loc=None, ip=None):
    llvm.inline_asm(None, [Int32(x).ir_value(loc=loc, ip=ip) for x in (addr, w0, w1, w2, w3)],
                    "tcgen05.st.sync.aligned.32x32b.x4.b32 [$0], {$1, $2, $3, $4};", "r,r,r,r,r",
                    has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def bytes4_same(b: Int32, *, loc=None, ip=None) -> Int32:
    """Replicate a byte into all 4 bytes of a word."""
    return Int32(llvm.inline_asm(T.i32(), [Int32(b).ir_value(loc=loc, ip=ip)],
                                 "{\n\t.reg .b32 t;\n\tand.b32 t, $1, 0xff;\n\tmul.lo.u32 $0, t, 0x01010101;\n\t}",
                                 "=r,r", has_side_effects=False, is_align_stack=False,
                                 asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def st_async_peer_b32(local_addr: Int32, local_mbar: Int32, peer_rank: Int32, val: Int32, *, loc=None, ip=None):
    """Store a 32-bit value into the peer CTA's smem at the same offset as local_addr, completing 4 tx bytes on the
    peer's mbarrier at the offset of local_mbar (st.async + mapa)."""
    llvm.inline_asm(None, [Int32(x).ir_value(loc=loc, ip=ip) for x in (local_addr, local_mbar, peer_rank, val)],
                    "{\n\t.reg .b32 ra, rm;\n\t"
                    "mapa.shared::cluster.u32 ra, $0, $2;\n\t"
                    "mapa.shared::cluster.u32 rm, $1, $2;\n\t"
                    "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [ra], $3, [rm];\n\t}",
                    "r,r,r,r", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def mbar_arrive_expect_tx(mbar: Int32, nbytes: int, *, loc=None, ip=None):
    llvm.inline_asm(None, [Int32(mbar).ir_value(loc=loc, ip=ip)],
                    f"mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [$0], {nbytes};", "r",
                    has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def mbar_wait_parity_nh(mbar: Int32, phase: Int32, *, loc=None, ip=None):
    """mbar_wait_parity without the try_wait suspend-time hint (hardware default time limit)."""
    llvm.inline_asm(None, [Int32(mbar).ir_value(loc=loc, ip=ip), Int32(phase).ir_value(loc=loc, ip=ip)],
                    "{\n\t.reg .pred P1;\n\tDM_WAITN:\n\t"
                    "mbarrier.try_wait.parity.shared::cta.b64 P1, [$0], $1;\n\t"
                    "@P1 bra DM_DONEN;\n\tbra DM_WAITN;\n\tDM_DONEN:\n\t}",
                    "r,r", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def mbar_wait_parity(mbar: Int32, phase: Int32, *, loc=None, ip=None):
    llvm.inline_asm(None, [Int32(mbar).ir_value(loc=loc, ip=ip), Int32(phase).ir_value(loc=loc, ip=ip)],
                    "{\n\t.reg .pred P1;\n\tDM_WAIT:\n\t"
                    "mbarrier.try_wait.parity.shared::cta.b64 P1, [$0], $1, 10000000;\n\t"
                    "@P1 bra DM_DONE;\n\tbra DM_WAIT;\n\tDM_DONE:\n\t}",
                    "r,r", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def ld_shared_b32(addr: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(llvm.inline_asm(T.i32(), [Int32(addr).ir_value(loc=loc, ip=ip)], "ld.volatile.shared.b32 $0, [$1];",
                                 "=r,r", has_side_effects=True, is_align_stack=False,
                                 asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def st_shared_b32(addr: Int32, val: Int32, *, loc=None, ip=None):
    llvm.inline_asm(None, [Int32(addr).ir_value(loc=loc, ip=ip), Int32(val).ir_value(loc=loc, ip=ip)],
                    "st.volatile.shared.b32 [$0], $1;", "r,r",
                    has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)
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


@dsl_user_op
def byte_pow2_f32_rt(word: Int32, pos: Int32, *, loc=None, ip=None) -> Float32:
    """byte_pow2_f32 with a runtime bit position pos = 8 j (same bfe/shl -> same value)."""
    return Float32(llvm.inline_asm(
        T.f32(), [Int32(word).ir_value(loc=loc, ip=ip), Int32(pos).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b32 t;\n\tbfe.u32 t, $1, $2, 8;\n\tshl.b32 $0, t, 23;\n\t}",
        "=f,r,r", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def absf(x: Float32, *, loc=None, ip=None) -> Float32:
    """abs.f32 (ptxas folds it into the consuming FMNMX as an operand modifier)."""
    return _f32_unary("abs.f32 $0, $1;", x, loc, ip)


@dsl_user_op
def clock64(*, loc=None, ip=None) -> cutlass.Int64:
    return cutlass.Int64(llvm.inline_asm(T.i64(), [], "mov.u64 $0, %clock64;", "=l",
                                         has_side_effects=True, is_align_stack=False,
                                         asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def red_add_u64(addr: cutlass.Int64, val: cutlass.Int64, *, loc=None, ip=None):
    llvm.inline_asm(None, [cutlass.Int64(addr).ir_value(loc=loc, ip=ip), cutlass.Int64(val).ir_value(loc=loc, ip=ip)],
                    "red.global.add.u64 [$0], $1;", "l,l", has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def red_add_f64_from_f32(addr: cutlass.Int64, val: Float32, *, loc=None, ip=None):
    """debug: red.global.add.f64 of an fp32 value (exactly widened)."""
    llvm.inline_asm(None, [cutlass.Int64(addr).ir_value(loc=loc, ip=ip), Float32(val).ir_value(loc=loc, ip=ip)],
                    "{\n\t.reg .f64 d;\n\tcvt.f64.f32 d, $1;\n\tred.global.add.f64 [$0], d;\n\t}", "l,f",
                    has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def st_sys_u32(addr: cutlass.Int64, val: Int32, *, loc=None, ip=None):
    llvm.inline_asm(None, [cutlass.Int64(addr).ir_value(loc=loc, ip=ip), Int32(val).ir_value(loc=loc, ip=ip)],
                    "st.relaxed.sys.global.u32 [$0], $1;", "l,r", has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def clock32(*, loc=None, ip=None) -> Int32:
    return Int32(llvm.inline_asm(T.i32(), [], "mov.u32 $0, %clock;", "=r",
                                 has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


# 2^f on [0, 1), fp32 Horner, degree 5: max rel error 1.7e-7 (= MUFU ex2.approx accuracy); fitted IRLS minimax
POLY5_EX2 = (0.9999999403953552, 0.6931530833244324, 0.24015361070632935, 0.05582635849714279,
             0.008989292196929455, 0.001877596601843834)


@dsl_user_op
def ex2_poly5_2(x: Float32, y: Float32, *, loc=None, ip=None):
    """Accurate (1.7e-7 rel) 2^x for a pair on the FMA pipe (FA4's ex2_emulation_2 with a degree-5 polynomial).
    Inputs are clamped at -126 (results ~2^-126 instead of 0 below that; negligible vs any row sum)."""
    from flash_attn.cute import utils as fu
    fp32_round_int = float(2**23 + 2**22)
    # clamp at -126: c0 < 1 means 2^frac can be < 1, and floor(x) = -127 would wrap the exponent field
    xy = (cute.arch.fmax(x, -126.0), cute.arch.fmax(y, -126.0))
    r = cute.arch.add_packed_f32x2(xy, (fp32_round_int, fp32_round_int), rnd="rm")
    rb = cute.arch.sub_packed_f32x2(r, (fp32_round_int, fp32_round_int))
    fr = cute.arch.sub_packed_f32x2(xy, rb)
    pe = fu.evaluate_polynomial_2(*fr, POLY5_EX2, loc=loc, ip=ip)
    return fu.combine_int_frac_ex2(r[0], pe[0], loc=loc, ip=ip), fu.combine_int_frac_ex2(r[1], pe[1], loc=loc, ip=ip)


# degree-6 exp2 on [0,1) in the form 1 + f*q(f) (Lawson-minimax, fp32 coefficients; kda/tests/p5acc/fit.py):
# exhaustive over all fp32 exp arguments in (-64,-0.5] U [0.5,8): max rel 6.6e-8, mean 2.3e-8, 91.8% correctly rounded,
# e4m3 RN-code mismatch 4.1e-7 vs MUFU ex2.approx.ftz 1.46e-7 / 4.4e-8 / 57.5% / 4.9e-7 (kda/runs/p5acc_exact.log)
POLY6_EX2 = (1.0, float.fromhex("0x1.62e42cp-1"), float.fromhex("0x1.ebfd58p-3"), float.fromhex("0x1.c68912p-5"),
             float.fromhex("0x1.3d0b92p-7"), float.fromhex("0x1.46d648p-10"), float.fromhex("0x1.c54182p-13"))


def ex2_poly6_2(x: Float32, y: Float32):
    """2^x for a pair on the FMA pipe, at least as accurate as ex2.approx.ftz.f32 on every element metric; results
    below 2^-126 (x < -126, incl. -inf) are flushed to 0 exactly like ex2.approx.ftz."""
    import cutlass
    from flash_attn.cute import utils as fu
    fp32_round_int = float(2**23 + 2**22)
    xy = (cute.arch.fmax(x, -127.0), cute.arch.fmax(y, -127.0))
    r = cute.arch.add_packed_f32x2(xy, (fp32_round_int, fp32_round_int), rnd="rm")
    rb = cute.arch.sub_packed_f32x2(r, (fp32_round_int, fp32_round_int))
    fr = cute.arch.sub_packed_f32x2(xy, rb)
    pe = fu.evaluate_polynomial_2(*fr, POLY6_EX2)
    ox = fu.combine_int_frac_ex2(r[0], pe[0])
    oy = fu.combine_int_frac_ex2(r[1], pe[1])
    return (Float32(cutlass.select_(x < Float32(-126.0), Float32(0.0), ox)),
            Float32(cutlass.select_(y < Float32(-126.0), Float32(0.0), oy)))


@dsl_user_op
def cpasync_bulk_store_s2g(smem_ptr, gmem_ptr, store_bytes, *, loc=None, ip=None):
    """cp.async.bulk smem -> gmem plain store (bulk_group), same grouping/wait semantics as the reduce-add variant."""
    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [gmem_ptr.llvm_ptr, smem_ptr_i32, Int32(store_bytes).ir_value()],
        "cp.async.bulk.global.shared::cta.bulk_group [$0], [$1], $2;",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def globaltimer(*, loc=None, ip=None) -> cutlass.Int64:
    return cutlass.Int64(llvm.inline_asm(T.i64(), [], "mov.u64 $0, %globaltimer;", "=l",
                                         has_side_effects=True, is_align_stack=False,
                                         asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def smid(*, loc=None, ip=None) -> Int32:
    return Int32(llvm.inline_asm(T.i32(), [], "mov.u32 $0, %smid;", "=r",
                                 has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def mbar_arrive_rank(mbar: Int32, rank: Int32, *, loc=None, ip=None):
    """mbarrier arrive (release, cluster scope) on CTA `rank`'s mbarrier at the offset of local `mbar`."""
    llvm.inline_asm(None, [Int32(mbar).ir_value(loc=loc, ip=ip), Int32(rank).ir_value(loc=loc, ip=ip)],
                    "{\n\t.reg .b32 rm;\n\t"
                    "mapa.shared::cluster.u32 rm, $0, $1;\n\t"
                    "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [rm];\n\t}",
                    "r,r", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def mbar_arrive_local(mbar: Int32, *, loc=None, ip=None):
    llvm.inline_asm(None, [Int32(mbar).ir_value(loc=loc, ip=ip)],
                    "mbarrier.arrive.release.cta.shared::cta.b64 _, [$0];",
                    "r", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def atom_add_global_u32(addr: cutlass.Int64, val: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(llvm.inline_asm(T.i32(), [cutlass.Int64(addr).ir_value(loc=loc, ip=ip), Int32(val).ir_value(loc=loc, ip=ip)],
                                 "atom.relaxed.gpu.global.add.u32 $0, [$1], $2;", "=r,l,r",
                                 has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def mbar_wait_parity_cluster(mbar: Int32, phase: Int32, *, loc=None, ip=None):
    llvm.inline_asm(None, [Int32(mbar).ir_value(loc=loc, ip=ip), Int32(phase).ir_value(loc=loc, ip=ip)],
                    "{\n\t.reg .pred P1;\n\tDM_WAITC:\n\t"
                    "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [$0], $1, 10000000;\n\t"
                    "@P1 bra DM_DONEC;\n\tbra DM_WAITC;\n\tDM_DONEC:\n\t}",
                    "r,r", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def f32_to_f64_exact(x: Float32, *, loc=None, ip=None) -> Float64:
    """Exact fp32 -> fp64 conversion without F2F (which issues on the XU/MUFU pipe): integer bit assembly for normals,
    inf and NaN; an exact FP64-pipe path (2^52 magic subtraction, power-of-two scaling) for zero and subnormals.
    Bitwise identical to cvt.f64.f32 for every fp32 input (NaN payload: quiet bit and upper mantissa preserved)."""
    asm = (
        "{\n"
        ".reg .b32 a, e, m, s, hi, lo, hii, mg, shi, slo;\n"
        ".reg .b64 dn, ds;\n"
        ".reg .f64 t;\n"
        ".reg .pred pz, pi;\n"
        "mov.b32 a, $1;\n"
        "and.b32 s, a, 0x80000000;\n"
        "and.b32 e, a, 0x7f800000;\n"
        "and.b32 m, a, 0x007fffff;\n"
        "shr.u32 hi, a, 3;\n"
        "and.b32 hi, hi, 0x0fffffff;\n"
        "add.u32 hi, hi, 0x38000000;\n"
        "or.b32 hi, hi, s;\n"
        "shl.b32 lo, a, 29;\n"
        "shr.u32 hii, m, 3;\n"
        "or.b32 hii, hii, 0x7ff00000;\n"
        "or.b32 hii, hii, s;\n"
        "setp.ne.u32 pz, m, 0;\n"
        "selp.b32 mg, 0x00080000, 0, pz;\n"
        "or.b32 hii, hii, mg;\n"
        "setp.eq.u32 pi, e, 0x7f800000;\n"
        "selp.b32 hi, hii, hi, pi;\n"
        "mov.b64 dn, {lo, hi};\n"
        "mov.b32 mg, 0x43300000;\n"
        "mov.b64 ds, {m, mg};\n"
        "mov.b64 t, ds;\n"
        "sub.rn.f64 t, t, 0d4330000000000000;\n"
        "mul.rn.f64 t, t, 0d36A0000000000000;\n"
        "mov.b64 {slo, shi}, t;\n"
        "or.b32 shi, shi, s;\n"
        "mov.b64 ds, {slo, shi};\n"
        "setp.eq.u32 pz, e, 0;\n"
        "selp.b64 dn, ds, dn, pz;\n"
        "mov.b64 $0, dn;\n"
        "}\n"
    )
    return Float64(llvm.inline_asm(T.f64(), [Float32(x).ir_value(loc=loc, ip=ip)], asm, "=d,f",
                                   has_side_effects=False, is_align_stack=False,
                                   asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def f32_to_f64_fastpath(x: Float32, *, loc=None, ip=None) -> Float64:
    """fp32 -> fp64 by integer bit assembly only (ALU): exact for zero, normals, inf and NaN (quiet bit set like
    cvt.f64.f32). NOT valid for subnormals (exponent field 0, mantissa != 0): callers must route those to Float64(x)."""
    asm = (
        "{\n"
        ".reg .b32 a, e, m, s, hi, lo, hii, q;\n"
        ".reg .pred pz, pi, pq;\n"
        "mov.b32 a, $1;\n"
        "and.b32 s, a, 0x80000000;\n"
        "and.b32 e, a, 0x7f800000;\n"
        "and.b32 m, a, 0x007fffff;\n"
        "shr.u32 hi, a, 3;\n"
        "and.b32 hi, hi, 0x0fffffff;\n"
        "add.u32 hi, hi, 0x38000000;\n"
        "or.b32 hi, hi, s;\n"
        "shl.b32 lo, a, 29;\n"
        "shr.u32 hii, m, 3;\n"
        "or.b32 hii, hii, 0x7ff00000;\n"
        "or.b32 hii, hii, s;\n"
        "setp.ne.u32 pq, m, 0;\n"
        "selp.b32 q, 0x00080000, 0, pq;\n"
        "or.b32 hii, hii, q;\n"
        "setp.eq.u32 pi, e, 0x7f800000;\n"
        "selp.b32 hi, hii, hi, pi;\n"
        "setp.eq.u32 pz, e, 0;\n"
        "selp.b32 hi, s, hi, pz;\n"
        "selp.b32 lo, 0, lo, pz;\n"
        "mov.b64 $0, {lo, hi};\n"
        "}\n"
    )
    return Float64(llvm.inline_asm(T.f64(), [Float32(x).ir_value(loc=loc, ip=ip)], asm, "=d,f",
                                   has_side_effects=False, is_align_stack=False,
                                   asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def f32_is_subnormal(x: Float32, *, loc=None, ip=None) -> Int32:
    """1 if x is a nonzero fp32 subnormal (exponent field 0, mantissa != 0), else 0 (ALU only)."""
    return Int32(llvm.inline_asm(T.i32(), [Float32(x).ir_value(loc=loc, ip=ip)],
                                 "{\n.reg .b32 a, e, m;\n.reg .pred p, q;\nmov.b32 a, $1;\nand.b32 e, a, 0x7f800000;\n"
                                 "and.b32 m, a, 0x007fffff;\nsetp.eq.u32 p, e, 0;\nsetp.ne.and.u32 q, m, 0, p;\n"
                                 "selp.u32 $0, 1, 0, q;\n}\n", "=r,f",
                                 has_side_effects=False, is_align_stack=False,
                                 asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


# ---- fwdrw ss_pv helpers ----
@dsl_user_op
def mbar_init(mbar: Int32, count: int, *, loc=None, ip=None):
    llvm.inline_asm(None, [Int32(mbar).ir_value(loc=loc, ip=ip)], f"mbarrier.init.shared::cta.b64 [$0], {count};", "r",
                    has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def mbar_arrive(mbar: Int32, *, loc=None, ip=None):
    llvm.inline_asm(None, [Int32(mbar).ir_value(loc=loc, ip=ip)], "mbarrier.arrive.release.cta.shared::cta.b64 _, [$0];",
                    "r", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def tc_commit(mbar: Int32, *, loc=None, ip=None):
    """tcgen05.commit (cta_group::1): mbarrier arrives once all prior async tcgen05 ops of this thread (mma / cp / shift)
    have completed; tcgen05.ld / st are NOT tracked (they use tcgen05.wait::ld / ::st)."""
    llvm.inline_asm(None, [Int32(mbar).ir_value(loc=loc, ip=ip)],
                    "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [$0];", "r",
                    has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def st_shared_v4_b32(addr: Int32, a: Int32, b: Int32, c: Int32, d: Int32, *, loc=None, ip=None):
    llvm.inline_asm(None, [Int32(x).ir_value(loc=loc, ip=ip) for x in (addr, a, b, c, d)],
                    "st.shared.v4.b32 [$0], {$1, $2, $3, $4};", "r,r,r,r,r",
                    has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def bitcast_f32_i32(x: Float32, *, loc=None, ip=None) -> Int32:
    return Int32(llvm.inline_asm(T.i32(), [Float32(x).ir_value(loc=loc, ip=ip)], "mov.b32 $0, $1;", "=r,f",
                                 has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))

@dsl_user_op
def red_add_v4_f32(addr: cutlass.Int64, a: Float32, b: Float32, c: Float32, d: Float32, *, loc=None, ip=None):
    """red.global.add.v4.f32 (sm_90+): 4 contiguous fp32 adds, 16-byte aligned address."""
    llvm.inline_asm(None, [cutlass.Int64(addr).ir_value(loc=loc, ip=ip)] + [Float32(x).ir_value(loc=loc, ip=ip) for x in (a, b, c, d)],
                    "red.relaxed.gpu.global.add.v4.f32 [$0], {$1, $2, $3, $4};", "l,f,f,f,f", has_side_effects=True,
                    is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def st_v4_f32(addr: cutlass.Int64, a: Float32, b: Float32, c: Float32, d: Float32, *, loc=None, ip=None):
    llvm.inline_asm(None, [cutlass.Int64(addr).ir_value(loc=loc, ip=ip)] + [Float32(x).ir_value(loc=loc, ip=ip) for x in (a, b, c, d)],
                    "st.global.v4.f32 [$0], {$1, $2, $3, $4};", "l,f,f,f,f", has_side_effects=True,
                    is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def fence_acq_rel_gpu(*, loc=None, ip=None):
    llvm.inline_asm(None, [], "fence.acq_rel.gpu;", "", has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def tc_fence_after(*, loc=None, ip=None):
    """tcgen05.fence::after_thread_sync: order later tcgen05 ops after a preceding thread sync (mbarrier wait)."""
    if _TCF_IMPL == "nvvm":
        from cutlass._mlir.dialects import nvvm as _nv
        _nv.tcgen05_fence(_nv.Tcgen05FenceKind.AFTER_THREAD_SYNC, loc=loc, ip=ip)
        return
    llvm.inline_asm(None, [], "tcgen05.fence::after_thread_sync;", "", has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def tc_fence_before(*, loc=None, ip=None):
    """tcgen05.fence::before_thread_sync: order earlier tcgen05 ops before a following thread sync (mbarrier arrive)."""
    if _TCF_IMPL == "nvvm":
        from cutlass._mlir.dialects import nvvm as _nv
        _nv.tcgen05_fence(_nv.Tcgen05FenceKind.BEFORE_THREAD_SYNC, loc=loc, ip=ip)
        return
    llvm.inline_asm(None, [], "tcgen05.fence::before_thread_sync;", "", has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT)


# ---- kda-sm100-fwdfence: tcgen05 thread-sync fences on every cross-thread hand-off (default on) ----
# PTX orders tcgen05 ops (ld/st/mma/cp) of one thread against another thread only through
# tcgen05.fence::before_thread_sync (producer, before the sync) and tcgen05.fence::after_thread_sync (consumer, after it).
# FA4-derived pipelines (cutlass.pipeline / flash_attn.cute.pipeline) emit no such fences. While a DM kernel body is
# traced with TCF on, every mbarrier / named-barrier wait gets fence::after and every arrive gets fence::before.
# On sm_100a (CUDA 13.0) both fences emit no SASS: they only forbid ptxas from moving tcgen05 ops across the sync
# (kda/docs/fence-audit.md). Knobs: fwd DM_FWD_OPTS tcf=0, bwd DM_BWD_OPTS notcf, pass0 exp variant +ntcf.
TCF = [False]
import os as _os_tcf
_TCF_MASK = int(_os_tcf.environ.get("DM_TCF_MASK", "15"))   # DEBUG bisect: 1 waits(after) 2 releases(before) 4 relay 8 bars


def tcf_begin(on):
    """0 = off (94ec380 code), 1 = minimal (default: tcgen05 pipelines / barriers marked with tcf_mark, explicit
    relay sites), 2 = blanket (every mbarrier / named-barrier wait and arrive, every sync_warp / elect_one)."""
    TCF[0] = int(on)


def tcf_end():
    TCF[0] = False


def _tcf_wrap(f, before, after, mode=2):
    if getattr(f, "_dm_tcf", False):
        return f

    def w(*a, **kw):
        if TCF[0] >= mode and before:
            tc_fence_before()
        r = f(*a, **kw)
        if TCF[0] >= mode and after:
            tc_fence_after()
        return r
    w._dm_tcf = True
    w.__name__ = getattr(f, "__name__", "w")
    w.__doc__ = f.__doc__
    return w


def _tcf_install():
    import cutlass.cute.arch as _A
    g = globals()
    for n in ("mbar_wait_parity", "mbar_wait_parity_nh", "mbar_wait_parity_cluster"):
        g[n] = _tcf_wrap(g[n], False, True)
    for n in ("mbar_arrive", "mbar_arrive_local", "mbar_arrive_rank", "mbar_arrive_expect_tx"):
        g[n] = _tcf_wrap(g[n], True, False)
    for n, b, a in (("mbarrier_wait", False, True), ("mbarrier_try_wait", False, True),
                    ("mbarrier_conditional_try_wait", False, True), ("mbarrier_test_wait", False, True),
                    ("mbarrier_arrive", True, False), ("mbarrier_arrive_and_expect_tx", True, False),
                    ("cp_async_mbarrier_arrive_noinc", True, False),
                    ("barrier", True, True), ("barrier_arrive", True, False)):
        f = getattr(_A, n, None)
        if f is not None:
            setattr(_A, n, _tcf_wrap(f, b, a))
    # warp -> elected-leader relays (Codex review 1): every lane fences its own tcgen05 ops before the warp sync, so the
    # leader's later arrive publishes the whole warp's ops. elect.sync already requires the full warp, so the added
    # bar.warp.sync is legal wherever elect_one is.
    _sw = _A.sync_warp
    _tcf_raw_sync_warp[0] = _sw
    _A.sync_warp = _tcf_wrap(_sw, True, True)
    _el = _A.elect_one
    if not getattr(_el, "_dm_tcf", False):
        def elect_one(*a, **kw):
            if TCF[0] >= 2:
                tc_fence_before()
                _sw()
            return _el(*a, **kw)
        elect_one._dm_tcf = True
        elect_one.__doc__ = _el.__doc__
        _A.elect_one = elect_one
    # flash_attn NamedBarrier.arrive_and_wait_unaligned calls nvvm.barrier_cta_sync directly (bypasses cute.arch.barrier)
    try:
        import flash_attn.cute.pipeline as _fp
        _nb = _fp.NamedBarrier
        for n, b, a in (("arrive_and_wait_unaligned", True, True), ("arrive_unaligned", True, False)):
            f = _nb.__dict__.get(n)
            if f is not None:
                setattr(_nb, n, _tcf_wrap(f, b, a))
    except Exception:   # pragma: no cover
        pass


# ---- minimal mode (TCF == 1): fences only on objects marked by tcf_mark (tcgen05 pipelines + barriers that relay
# tcgen05 state). Marking swaps the instance's class to a subclass (like flash_attn's create()); the wrappers check it.
# kinds: "ua" = PipelineUmmaAsync (MMA producer via tcgen05.commit -> thread consumer), "au" = PipelineAsyncUmma
# (thread producer -> MMA consumer, released via tcgen05.commit), "bar" = named barrier.
_TCF_SUB = {}


def tcf_mark(kind, *objs):
    for o in objs:
        if o is None:
            continue
        c = type(o)
        k = (c, kind)
        if k not in _TCF_SUB:
            _TCF_SUB[k] = type(c.__name__, (c,), {"_dm_tcf": kind})
        object.__setattr__(o, "__class__", _TCF_SUB[k])
    return objs


def _tcf_mwrap(f, role):
    if getattr(f, "_dm_tcfm", False):
        return f

    def w(self, *a, **kw):
        k = getattr(type(self), "_dm_tcf", None) if TCF[0] == 1 and not _TCF_SKIP[0] else None
        if k == "bar" and not (_TCF_MASK & 8):
            k = None
        before = after = False
        if k is not None:
            if role == "wait":
                after = True
            elif role == "rel":     # thread arrive; tcgen05.commit arrives need no before-fence
                before = not ((k == "ua" and f.__name__.startswith("producer_commit"))
                              or (k == "au" and f.__name__.startswith("consumer_release")))
            elif role == "both":
                before = after = True
        if before and k != "bar" and _TCF_MASK & 16 and k == "au":
            before = False   # DEBUG bisect: no before-fence on thread->MMA commits
        if before and k != "bar" and _TCF_MASK & 32 and k == "ua":
            before = False   # DEBUG bisect: no before-fence on thread releases of MMA-produced buffers
        if before and (_TCF_MASK & 2 or k == "bar"):
            tc_fence_before()
        r = f(self, *a, **kw)
        if after and (_TCF_MASK & 1 or k == "bar"):
            tc_fence_after()
        return r
    w._dm_tcfm = True
    w.__name__ = f.__name__
    return w


def _tcf_install_min():
    import cutlass.pipeline as _cp
    import flash_attn.cute.pipeline as _fp
    roles = {"consumer_wait": "wait", "producer_acquire": "wait", "consumer_release": "rel", "producer_commit": "rel",
             "arrive": "rel", "arrive_unaligned": "rel", "arrive_w_index": "rel", "wait": "wait",
             "wait_unaligned": "wait", "arrive_and_wait": "both", "arrive_and_wait_unaligned": "both",
             "arrive_and_wait_w_index": "both", "sync": "both"}
    seen = set()
    for mod in (_cp, _fp):
        for n in dir(mod):
            c = getattr(mod, n)
            if not isinstance(c, type) or c in seen:
                continue
            seen.add(c)
            if not (n.startswith("Pipeline") or n == "NamedBarrier"):
                continue
            for m, r in roles.items():
                f = c.__dict__.get(m)
                if f is not None and callable(f):
                    setattr(c, m, _tcf_mwrap(f, r))


_TCF_SKIP = [False]


class tcf_site:
    """DEBUG bisect: `with tcf_site("id"):` suppresses minimal-mode fences inside if id is in env DM_TCF_SKIP."""
    def __init__(self, sid):
        self.on = sid in _os_tcf.environ.get("DM_TCF_SKIP", "").split(",")

    def __enter__(self):
        if self.on:
            _TCF_SKIP[0] = True

    def __exit__(self, *a):
        _TCF_SKIP[0] = False


def tcf_relay():
    """Minimal mode: every lane fences its tcgen05 ops, then the warp syncs, before an elected leader's arrive."""
    if TCF[0] == 1 and _TCF_MASK & 4:
        tc_fence_before()
        _tcf_raw_sync_warp[0]()


_tcf_raw_sync_warp = [None]


_tcf_install()
_tcf_install_min()
