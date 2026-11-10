"""
FP8 (e4m3) producers of the SM100 attention: Q, K, V and dO codes with one
power-of-2 scale per 32 tokens over all of head_dim, the K mean, and the stale
delta.  CuTe DSL kernels; D (128 or 256) is a compile-time parameter.

Quantization of a (B, N, H, D) bf16 tensor x, per (batch, head, 32-token block):
    y = (x - mean[b, h]) * prescale   (fp32; K centred by its mean, Q
        multiplied by the softmax scale)
    amax = max |y| = m * 2^E with m in [1, 2):
    scale = 2^(E-8) if m <= 1.75 else 2^(E-7), the smallest power of 2 with
        amax <= 448 * scale (1 when amax == 0)
    code = rn_e4m3(y * (1 / scale)), saturating, never above 448
The scale's exponent biased by 127 is the UE8M0 byte that the tcgen05
block-scaled MMAs read.

The K mean is div.full(the in-order sum of the ROWS-row partials, N); a partial
is the in-order sum of its 64-row chunk sums; a chunk sum is a balanced pairwise
tree over its rows, grouped by the K layout (16-byte aligned rows or not).  The
stale delta is rowsum(dO * O) with a fixed in-thread tree and butterfly.  Every
fp32 operation of these sums is inline PTX with an explicit rounding mode, so
the compilers can neither contract nor reassociate it: results are bitwise
reproducible.  SIMT kernels: no TMA, tcgen05 or pipelines.
"""
import threading

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from delta_matching.sm100.cute_ptx_sm100 import abs_f32, add_f32, inline_asm, ld_shared_f32, max_f32, st_global_v4

__all__ = ["fp8_quant_qkv", "fp8_quant_dout", "compile_qkv_producers", "compile_row_quant"]

BLOCK = 32                 # tokens per E8M0 scale block, the rows of one quantization tile
ROWS_PER_PARTIAL = 1024    # K-mean partial length (halved until it divides N)
TILE_CONFIG = {128: (2, False), 256: (4, True)}    # D -> (warps per 32-row tile, re-convert rows per pass)
PAIRED_WARPS = {256: 4}    # D=256: Q/V/K-mean producers as 2 warp pairs per CTA; D=128 uses the chunked producers
COMPILE_LOCK = threading.Lock()
COMPILED = {}
PERSISTENT = []
LAUNCHERS = {}
COUNTERS = {}
RETIRED_COUNTERS = []


# PTX wrappers and building blocks.


def ir_value(value, value_type, loc, ip):
    return value_type(value).ir_value(loc=loc, ip=ip)


def struct_fields(result, field_type, count, loc, ip):
    llvm_type = T.f32() if field_type is Float32 else T.i32()
    return [field_type(llvm.extractvalue(llvm_type, result, [i], loc=loc, ip=ip)) for i in range(count)]


@dsl_user_op
def ld_v4(addr: Int64, *, loc=None, ip=None):
    """
    16-byte global load as 4 b32 words.
    """
    result = inline_asm(llvm.StructType.get_literal([T.i32()] * 4), [ir_value(addr, Int64, loc, ip)], "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,l", loc=loc, ip=ip)
    return struct_fields(result, Int32, 4, loc, ip)


@dsl_user_op
def ld_bf16_f32(addr: Int64, *, loc=None, ip=None) -> Float32:
    return Float32(inline_asm(T.f32(), [ir_value(addr, Int64, loc, ip)], "{\n\t.reg .b16 h;\n\tld.global.b16 h, [$1];\n\tcvt.f32.bf16 $0, h;\n\t}", "=f,l", loc=loc, ip=ip))


def make_bf16x2_to_f32(tag):
    @dsl_user_op
    def convert(word: Int32, *, loc=None, ip=None):
        result = inline_asm(llvm.StructType.get_literal([T.f32()] * 2), [ir_value(word, Int32, loc, ip)], "{\n\t.reg .b16 l, h;\n\tmov.b32 {l, h}, $2;\n\tcvt.f32.bf16 $0, l;\n\tcvt.f32.bf16 $1, h;\n\t}" + (f" // {tag}" if tag else ""), "=f,=f,r", loc=loc, ip=ip)
        return struct_fields(result, Float32, 2, loc, ip)
    return convert


BF16X2_CONVERTERS = {}


def bf16x2_to_f32(word, tag=""):
    """
    A b32 word of 2 bf16 (low half first) as 2 f32.  The tag makes the asm text unique per
    conversion pass, so LLVM cannot merge a re-conversion with an earlier one (that would keep every
    converted row live and cost occupancy).
    """
    convert = BF16X2_CONVERTERS.get(tag)
    if convert is None:
        convert = BF16X2_CONVERTERS[tag] = make_bf16x2_to_f32(tag)
    return convert(word)


@dsl_user_op
def f32x2_op(opcode: str, *operands, loc=None, ip=None):
    """
    A packed f32x2 instruction on (lo, hi) pairs (FADD2 / FMUL2 / FFMA2): each lane rounds exactly
    as the scalar instruction, with half the instructions.
    """
    values = []
    for pair in operands:
        values += [ir_value(pair[0], Float32, loc, ip), ir_value(pair[1], Float32, loc, ip)]
    num_operands = len(operands)
    regs = ", ".join(f"x{i}" for i in range(num_operands))
    text = "{\n\t.reg .b64 " + regs + ", z;\n\t"
    for i in range(num_operands):
        text += f"mov.b64 x{i}, {{${2 + 2 * i}, ${3 + 2 * i}}};\n\t"
    text += f"{opcode} z, " + regs + ";\n\tmov.b64 {$0, $1}, z;\n\t}"
    result = inline_asm(llvm.StructType.get_literal([T.f32()] * 2), values, text, "=f,=f," + ",".join(["f"] * (2 * num_operands)), loc=loc, ip=ip)
    return tuple(struct_fields(result, Float32, 2, loc, ip))


@dsl_user_op
def shfl_xor_f32(value: Float32, offset: int, *, loc=None, ip=None) -> Float32:
    return Float32(inline_asm(T.f32(), [ir_value(value, Float32, loc, ip)], f"shfl.sync.bfly.b32 $0, $1, {offset}, 31, -1;", "=f,f", side=True, loc=loc, ip=ip))


@dsl_user_op
def pow2_from_byte(exponent_byte: Int32, *, loc=None, ip=None) -> Float32:
    """
    The fp32 power of 2 whose biased exponent is exponent_byte.
    """
    return Float32(inline_asm(T.f32(), [ir_value(exponent_byte, Int32, loc, ip)], "shl.b32 $0, $1, 23;", "=f,r", loc=loc, ip=ip))


@dsl_user_op
def st_f32_pred(addr: Int64, value: Float32, pred: Int32, *, loc=None, ip=None):
    inline_asm(None, [ir_value(addr, Int64, loc, ip), ir_value(value, Float32, loc, ip), ir_value(pred, Int32, loc, ip)], "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $2, 0;\n\t@p st.global.f32 [$0], $1;\n\t}", "l,f,r", side=True, loc=loc, ip=ip)


@dsl_user_op
def st_v4_f32_pred(addr: Int64, a: Float32, b: Float32, c: Float32, d: Float32, pred: Int32, *, loc=None, ip=None):
    inline_asm(None, [ir_value(addr, Int64, loc, ip)] + [ir_value(x, Float32, loc, ip) for x in (a, b, c, d)] + [ir_value(pred, Int32, loc, ip)], "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $5, 0;\n\t@p st.global.v4.f32 [$0], {$1, $2, $3, $4};\n\t}", "l,f,f,f,f,r", side=True, loc=loc, ip=ip)


@dsl_user_op
def st_b32_pred(addr: Int64, word: Int32, pred: Int32, *, loc=None, ip=None):
    inline_asm(None, [ir_value(addr, Int64, loc, ip), ir_value(word, Int32, loc, ip), ir_value(pred, Int32, loc, ip)], "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $2, 0;\n\t@p st.global.b32 [$0], $1;\n\t}", "l,r,r", side=True, loc=loc, ip=ip)


@dsl_user_op
def sts_f32_pred(addr: Int32, value: Float32, pred: Int32, *, loc=None, ip=None):
    inline_asm(None, [ir_value(addr, Int32, loc, ip), ir_value(value, Float32, loc, ip), ir_value(pred, Int32, loc, ip)], "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $2, 0;\n\t@p st.shared.f32 [$0], $1;\n\t}", "r,f,r", side=True, loc=loc, ip=ip)


@dsl_user_op
def sts_i32_pred(addr: Int32, value: Int32, pred: Int32, *, loc=None, ip=None):
    inline_asm(None, [ir_value(addr, Int32, loc, ip), ir_value(value, Int32, loc, ip), ir_value(pred, Int32, loc, ip)], "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $2, 0;\n\t@p st.shared.b32 [$0], $1;\n\t}", "r,r,r", side=True, loc=loc, ip=ip)


@dsl_user_op
def lds_i32(addr: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(inline_asm(T.i32(), [ir_value(addr, Int32, loc, ip)], "ld.shared.b32 $0, [$1];", "=r,r", side=True, loc=loc, ip=ip))


@dsl_user_op
def bar_sync(*, loc=None, ip=None):
    inline_asm(None, [], "bar.sync 0;", "", side=True, loc=loc, ip=ip)


@dsl_user_op
def fence_acq_rel_gpu(*, loc=None, ip=None):
    inline_asm(None, [], "fence.acq_rel.gpu;", "", side=True, loc=loc, ip=ip)


@dsl_user_op
def atom_add_acq_rel_gpu(addr: Int64, value: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(inline_asm(T.i32(), [ir_value(addr, Int64, loc, ip), ir_value(value, Int32, loc, ip)], "atom.add.acq_rel.gpu.global.s32 $0, [$1], $2;", "=r,l,r", side=True, loc=loc, ip=ip))


@dsl_user_op
def ld_cg_f32(addr: Int64, *, loc=None, ip=None) -> Float32:
    """
    L2 load (bypasses L1): sums published by other CTAs.
    """
    return Float32(inline_asm(T.f32(), [ir_value(addr, Int64, loc, ip)], "ld.global.cg.f32 $0, [$1];", "=f,l", side=True, loc=loc, ip=ip))


@dsl_user_op
def alloc_smem(num_bytes: int, *, loc=None, ip=None) -> Int32:
    """
    The 32-bit shared address of a static buffer of num_bytes.
    """
    from cutlass.utils import SmemAllocator
    tensor = SmemAllocator().allocate_tensor(Float32, cute.make_layout(num_bytes // 4), byte_alignment=16)
    return Int32(tensor.iterator.toint())


def sum_tree(values):
    """
    Balanced pairwise sum, level by level: ((v0 + v1) + (v2 + v3)) + ...  (a power-of-2 count).
    """
    while len(values) > 1:
        values = [add_f32(values[2 * i], values[2 * i + 1]) for i in range(len(values) // 2)]
    return values[0]


def sum_tree_f32x2(pairs):
    """
    sum_tree on (lo, hi) pairs with packed adds.
    """
    while len(pairs) > 1:
        pairs = [f32x2_op("add.rn.f32x2", pairs[2 * i], pairs[2 * i + 1]) for i in range(len(pairs) // 2)]
    return pairs[0]


def max_tree(values):
    # max is exact and order-free (maxNum); the tree is only for instruction-level parallelism
    while len(values) > 1:
        values = [max_f32(values[2 * i], values[2 * i + 1]) for i in range(len(values) // 2)]
    return values[0]


def dot16_tree(x, other):
    """
    sum_j x_j * other_j over 16 columns, packed over column pairs (lo = even columns, hi = odd): a_k
    = fma(x(4k, 4k+1), other(4k, 4k+1), x(4k+2, 4k+3) * other(4k+2, 4k+3)), then (a0 + a1) + (a2 +
    a3), then lo + hi.
    """
    x_pairs = [(x[2 * i], x[2 * i + 1]) for i in range(8)]
    other_pairs = [(other[2 * i], other[2 * i + 1]) for i in range(8)]
    partial = [f32x2_op("fma.rn.f32x2", x_pairs[2 * k], other_pairs[2 * k], f32x2_op("mul.rn.f32x2", x_pairs[2 * k + 1], other_pairs[2 * k + 1])) for k in range(4)]
    total = f32x2_op("add.rn.f32x2", f32x2_op("add.rn.f32x2", partial[0], partial[1]), f32x2_op("add.rn.f32x2", partial[2], partial[3]))
    return add_f32(total[0], total[1])


def load_row16(addr, vec):
    """
    16 consecutive bf16 at byte address addr as 16 f32.
    """
    if vec:
        words = ld_v4(addr) + ld_v4(addr + Int64(16))
        values = []
        for word in words:
            values += bf16x2_to_f32(word)
        return values
    return [ld_bf16_f32(addr + Int64(2 * j)) for j in range(16)]


def load_row8(addr, vec):
    """
    8 consecutive bf16 at byte address addr as 8 f32.
    """
    if vec:
        values = []
        for word in ld_v4(addr):
            values += bf16x2_to_f32(word)
        return values
    return [ld_bf16_f32(addr + Int64(2 * j)) for j in range(8)]


def pair_barrier(pair, num_pairs):
    """
    The barrier of one warp pair: CTA-wide bar 0 for a one-pair CTA, else named barrier 1 + pair.
    """
    if num_pairs == 1:
        return bar_sync
    return lambda: inline_asm(None, [ir_value(Int32(1) + pair, Int32, None, None)], "bar.sync $0, 64;", "r", side=True)


# Quantization tiles: 32 rows x D of one tensor, Triton-style blocked layout (16 columns per thread).


class TileThread:
    """
    A thread's place in a 32-row x D tile on num_warps warps: lanes split a row in 16-column groups.
    """

    def __init__(self, D, tidx, num_warps=2):
        self.D, self.num_warps = D, num_warps
        self.lanes_per_row = D // 16
        self.rows_per_warp = 32 // self.lanes_per_row
        self.rows_per_pass = self.rows_per_warp * num_warps
        self.num_passes = BLOCK // self.rows_per_pass
        self.warp = tidx // 32
        self.lane = tidx % 32
        self.row_lane = self.lane // self.lanes_per_row
        self.col_lane = self.lane % self.lanes_per_row
        self.col0 = self.col_lane * 16

    def row(self, pass_idx):
        return self.warp * self.rows_per_warp + self.row_lane + pass_idx * self.rows_per_pass


class TileTarget:
    """
    Where a tile's codes, scale and UE8M0 byte go: (batch, head) of a (B, N, H, D) code tensor, its
    (B, H, N / 32) scales and, when byte_slot is not None, byte (2 * (block // 4) + byte_slot) * 4 +
    block % 4 of the head's packed words.
    """

    def __init__(self, code_base, scale_base, byte_base, byte_slot, N, H, head_idx, batch_head, batch_idx, row0):
        self.code_base, self.scale_base, self.byte_base, self.byte_slot = code_base, scale_base, byte_base, byte_slot
        self.N, self.H, self.head_idx, self.batch_head, self.batch_idx, self.row0 = N, H, head_idx, batch_head, batch_idx, row0


def load_tile_words(tile: TileThread, x_base, batch_idx, head_idx, row0, stride_b, stride_n, stride_h, vec):
    """
    Issue every load of the thread's rows before any use.  Per pass: 8 b32 words (bf16 pairs,
    16-byte path) or 16 f32.
    """
    first_addr = x_base + (Int64(batch_idx) * Int64(stride_b) + Int64(head_idx) * Int64(stride_h) + Int64(row0 + tile.row(0)) * Int64(stride_n) + Int64(tile.col0)) * 2
    pass_step = Int64(stride_n) * (tile.rows_per_pass * 2)    # strength-reduced: rows of pass p are row(0) + p * rows_per_pass
    addrs = [first_addr]
    for _ in range(1, tile.num_passes):
        addrs.append(addrs[-1] + pass_step)
    if not vec:
        return [load_row16(addr, vec) for addr in addrs]
    return [ld_v4(addr) + ld_v4(addr + Int64(16)) for addr in addrs]


def unpack_row(words, packed, tag=""):
    if not packed:
        return words
    values = []
    for word in words:
        values += bf16x2_to_f32(word, tag)
    return values


def row_values(words, packed, mean=None, prescale=None, reconvert=True):
    """
    pass -> the thread's 16 values (x - mean) * prescale of that pass.  Without reconvert every pass
    reuses one asm text, so LLVM keeps the converted rows live (fewer instructions, more registers).
    """
    def values_of(pass_idx, tag=""):
        values = unpack_row(words[pass_idx], packed, tag if reconvert else "")
        if mean is not None:
            values = [value for i in range(8) for value in f32x2_op("sub.rn.f32x2", (values[2 * i], values[2 * i + 1]), (mean[2 * i], mean[2 * i + 1]))]
        if prescale is not None:
            values = [value for i in range(8) for value in f32x2_op("mul.rn.f32x2", (values[2 * i], values[2 * i + 1]), (prescale, prescale))]
        return values
    return values_of


def reduce_tile_amax(tile: TileThread, values_of, s_amax, barrier):
    """
    The tile's amax: in-thread tree, warp butterfly, then the warps through shared memory.
    values_of is called again by the store pass: only the packed loads stay live (the recomputation
    is the same instructions, same values).
    """
    amax = max_tree([max_tree([abs_f32(value) for value in values_of(pass_idx, "a")]) for pass_idx in range(tile.num_passes)])
    for offset in (16, 8, 4, 2, 1):
        amax = max_f32(amax, shfl_xor_f32(amax, offset))
    sts_f32_pred(s_amax + tile.warp * 4, amax, Int32(tile.lane == 0))
    barrier()
    return max_tree([ld_shared_f32(s_amax + 4 * i) for i in range(tile.num_warps)])


def quantize_tile(tile: TileThread, values_of, s_amax, target: TileTarget, barrier=bar_sync):
    """
    Codes, scale and (byte_slot not None) UE8M0 byte of one tile.
    """
    amax = reduce_tile_amax(tile, values_of, s_amax, barrier)
    # scale byte = ((bits >> 23) & 0xFF) - 8 + ((bits & 0x7FFFFF) > 0x600000), clamped to [1, 254]; 127 unless amax > 0
    scale_byte = Int32(inline_asm(T.i32(), [ir_value(amax, Float32, None, None)], "{\n\t.reg .b32 a, m, e;\n\t.reg .pred p, q;\n\t"
                                "mov.b32 a, $1;\n\tshr.u32 e, a, 23;\n\tand.b32 e, e, 255;\n\tsub.s32 e, e, 8;\n\t"
                                "and.b32 m, a, 8388607;\n\tsetp.gt.u32 p, m, 6291456;\n\tselp.s32 m, 1, 0, p;\n\tadd.s32 e, e, m;\n\t"
                                "max.s32 e, e, 1;\n\tmin.s32 e, e, 254;\n\t"
                                "setp.gt.f32 q, $1, 0f00000000;\n\tselp.s32 $0, e, 127, q;\n\t}", "=r,f"))
    scale = pow2_from_byte(scale_byte)
    rcp = pow2_from_byte(Int32(254) - scale_byte)
    block_idx = target.row0 // BLOCK
    store_tile_codes(tile, values_of, rcp, target)
    store_tile_scale(tile, scale, scale_byte, block_idx, target)
    return scale


def store_tile_codes(tile: TileThread, values_of, rcp, target: TileTarget):
    D, N, H, head_idx, batch_idx, row0 = tile.D, target.N, target.H, target.head_idx, target.batch_idx, target.row0
    code_addr = target.code_base + ((Int64(batch_idx) * Int64(N) + Int64(row0 + tile.row(0))) * Int64(H) + Int64(head_idx)) * Int64(D) + Int64(tile.col0)
    pass_step = Int64(H) * (tile.rows_per_pass * D)
    for pass_idx in range(tile.num_passes):
        values = values_of(pass_idx, "c")
        scaled = []
        for i in range(8):
            scaled += list(f32x2_op("mul.rn.f32x2", (values[2 * i], values[2 * i + 1]), (rcp, rcp)))
        # 4 fp32 -> 4 e4m3 bytes, byte 0 first (cvt packs its first operand into the high byte)
        words = [Int32(inline_asm(T.i32(), [ir_value(x, Float32, None, None) for x in scaled[4 * k:4 * k + 4]], "{\n\t.reg .b16 l, h;\n\tcvt.rn.satfinite.e4m3x2.f32 l, $2, $1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 h, $4, $3;\n\tmov.b32 $0, {l, h};\n\t}", "=r,f,f,f,f")) for k in range(4)]
        st_global_v4(code_addr, *words)
        code_addr = code_addr + pass_step


def store_tile_scale(tile: TileThread, scale, scale_byte, block_idx, target: TileTarget):
    is_first = Int32(tile.warp == 0) * Int32(tile.lane == 0)
    st_f32_pred(target.scale_base + (Int64(target.batch_head) * Int64(target.N // BLOCK) + Int64(block_idx)) * 4, scale, is_first)
    if target.byte_slot is not None:
        byte_addr = target.byte_base + (Int64(target.batch_head) * Int64(target.N // 64) + Int64(2 * (block_idx // 4) + target.byte_slot)) * 4 + Int64(block_idx % 4)
        inline_asm(None, [ir_value(byte_addr, Int64, None, None), ir_value(scale_byte, Int32, None, None), ir_value(is_first, Int32, None, None)], "{\n\t.reg .pred p;\n\t.reg .b16 h;\n\tsetp.ne.s32 p, $2, 0;\n\tcvt.u16.u32 h, $1;\n\t@p st.global.b8 [$0], h;\n\t}", "l,r,r", side=True)


def load_mean(mean_base, batch_head, D, tile: TileThread):
    mean_addr = mean_base + (Int64(batch_head) * Int64(D) + Int64(tile.col0)) * 4
    return [Float32(inline_asm(T.f32(), [ir_value(mean_addr + Int64(4 * j), Int64, None, None)], "ld.global.f32 $0, [$1];", "=f,l")) for j in range(16)]


def store_stale_delta(tile: TileThread, x_values_of, other_values_of, delta_base, batch_head, N, row0):
    """
    rowsum(x * other) per row: dot16_tree in the thread, then a butterfly over the row's lanes
    (largest offset first); the row's first lane stores.
    """
    for pass_idx in range(tile.num_passes):
        total = dot16_tree(x_values_of(pass_idx, "d"), other_values_of(pass_idx, "o"))
        offset = tile.lanes_per_row // 2
        while offset >= 1:
            total = add_f32(total, shfl_xor_f32(total, offset))
            offset //= 2
        st_f32_pred(delta_base + (Int64(batch_head) * Int64(N) + Int64(row0 + tile.row(pass_idx))) * 4, total, Int32(tile.col_lane == 0))


class RowQuant:
    """
    One 32-row tile per CTA, grid (N / 32, B * H): codes and scales of x, optionally centred by a
    mean and writing UE8M0 bytes (K), optionally the stale delta rowsum(x * other) (dO with O).
    """

    def __init__(self, D, vec_x, has_mean, byte_slot, delta, vec_other, num_warps, reconvert):
        assert D in (128, 256) and delta in (None, "stale") and num_warps in (2, 4)
        self.D, self.vec_x, self.has_mean, self.byte_slot, self.delta, self.vec_other, self.num_warps, self.reconvert = D, vec_x, has_mean, byte_slot, delta, vec_other, num_warps, reconvert

    @cute.jit
    def __call__(self, x_base: Int64, other_base: Int64, mean_base: Int64, code_base: Int64, scale_base: Int64, delta_base: Int64, byte_base: Int64, N: Int32, H: Int32, B: Int32, x_stride_b: Int32, x_stride_n: Int32, x_stride_h: Int32, other_stride_b: Int32, other_stride_n: Int32, other_stride_h: Int32, stream: cuda.CUstream):
        self.kernel(x_base, other_base, mean_base, code_base, scale_base, delta_base, byte_base, N, H, x_stride_b, x_stride_n, x_stride_h, other_stride_b, other_stride_n, other_stride_h).launch(grid=[N // BLOCK, B * H, 1], block=[32 * self.num_warps, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, x_base: Int64, other_base: Int64, mean_base: Int64, code_base: Int64, scale_base: Int64, delta_base: Int64, byte_base: Int64, N: Int32, H: Int32, x_stride_b: Int32, x_stride_n: Int32, x_stride_h: Int32, other_stride_b: Int32, other_stride_n: Int32, other_stride_h: Int32):
        D = self.D
        tidx, _, _ = cute.arch.thread_idx()
        block_n, batch_head, _ = cute.arch.block_idx()
        s_amax = alloc_smem(16)
        tile = TileThread(D, tidx, self.num_warps)
        batch_idx, head_idx = batch_head // H, batch_head % H
        row0 = block_n * BLOCK
        x_words = load_tile_words(tile, x_base, batch_idx, head_idx, row0, x_stride_b, x_stride_n, x_stride_h, self.vec_x)
        other_words = self.load_other_words(tile, other_base, batch_idx, head_idx, row0, other_stride_b, other_stride_n, other_stride_h)
        mean = self.load_tile_mean(mean_base, batch_head, tile)
        x_values_of = row_values(x_words, self.vec_x, mean, None, self.reconvert)
        if const_expr(self.delta is not None):
            # delta first: the other tensor's words die before the quantization passes (registers); independent values
            store_stale_delta(tile, x_values_of, row_values(other_words, self.vec_other, reconvert=self.reconvert), delta_base, batch_head, N, row0)
        target = TileTarget(code_base, scale_base, byte_base, self.byte_slot, N, H, head_idx, batch_head, batch_idx, row0)
        quantize_tile(tile, x_values_of, s_amax, target)

    def load_other_words(self, tile, other_base, batch_idx, head_idx, row0, stride_b, stride_n, stride_h):
        if self.delta is None:
            return None
        return load_tile_words(tile, other_base, batch_idx, head_idx, row0, stride_b, stride_n, stride_h, self.vec_other)

    def load_tile_mean(self, mean_base, batch_head, tile):
        return load_mean(mean_base, batch_head, self.D, tile) if self.has_mean else None


# K-mean chunks: 64 rows x D on 2 warps, 8 columns per thread.


class ChunkThread:
    """
    A thread's place in a 64-row K chunk on 2 warps: lanes split a row in 8-column groups; the
    remaining lane bits and the warp form num_groups row groups of rows_per_group rows each.
    Aligned K layout: row = rep * num_groups + group (interleaved), otherwise row = group *
    rows_per_group + rep (contiguous).
    """

    def __init__(self, D, tidx, aligned):
        self.D, self.aligned = D, aligned
        self.lanes_per_row = D // 8
        self.rows_per_warp = 32 // self.lanes_per_row
        self.num_groups = self.rows_per_warp * 2
        self.rows_per_group = 64 // self.num_groups
        self.warp = tidx // 32
        self.lane = tidx % 32
        self.row_lane = self.lane // self.lanes_per_row
        self.col_lane = self.lane % self.lanes_per_row
        self.col0 = self.col_lane * 8
        self.group = self.warp * self.rows_per_warp + self.row_lane

    def row(self, rep):
        return rep * self.num_groups + self.group if self.aligned else self.group * self.rows_per_group + rep


def sum_chunk_columns(chunk: ChunkThread, k_base, batch_idx, head_idx, row0, stride_b, stride_n, stride_h, vec, s_partial, barrier=bar_sync):
    """
    The chunk's 8 column sums of this thread's columns (valid in warp 0, lanes with row_lane 0): a
    tree over the thread's rows, a lane butterfly, then the two warps through shared memory
    (s_partial, 2 * D floats).
    """
    first_addr = k_base + (Int64(batch_idx) * Int64(stride_b) + Int64(head_idx) * Int64(stride_h) + Int64(chunk.col0) + Int64(row0 + chunk.row(0)) * Int64(stride_n)) * 2
    rep_step = Int64(stride_n) * (2 * (chunk.num_groups if chunk.aligned else 1))
    addrs = [first_addr]
    for _ in range(1, chunk.rows_per_group):
        addrs.append(addrs[-1] + rep_step)
    if vec:
        # loads in groups of at most 16 rows; a balanced tree over 2^k leaves is the tree of its two halves' trees, so
        # the per-group trees combined by a tree are exactly the tree over all rows
        group_size = min(chunk.rows_per_group, 16)
        group_sums = []
        for group_start in range(0, chunk.rows_per_group, group_size):
            words = [ld_v4(addr) for addr in addrs[group_start:group_start + group_size]]
            sums = []
            for word_idx in range(4):              # one column pair at a time: only the packed words stay live
                sums += list(sum_tree_f32x2([tuple(bf16x2_to_f32(words[r][word_idx])) for r in range(group_size)]))
            group_sums.append(sums)
        column_sums = []
        for j in range(4):
            column_sums += list(sum_tree_f32x2([(sums[2 * j], sums[2 * j + 1]) for sums in group_sums]))
    else:
        rows = [load_row8(addr, vec) for addr in addrs]
        column_sums = [sum_tree([rows[rep][j] for rep in range(chunk.rows_per_group)]) for j in range(8)]
    if chunk.rows_per_warp == 2:
        column_sums = [add_f32(value, shfl_xor_f32(value, 16)) for value in column_sums]
    is_publisher = Int32(chunk.row_lane == 0)
    for j in range(8):
        sts_f32_pred(s_partial + (chunk.warp * chunk.D + chunk.col0 + j) * 4, column_sums[j], is_publisher)
    barrier()
    totals = []
    for j in range(8):
        totals.append(add_f32(ld_shared_f32(s_partial + (0 * chunk.D + chunk.col0 + j) * 4), ld_shared_f32(s_partial + (1 * chunk.D + chunk.col0 + j) * 4)))
    barrier()
    return totals


@cute.jit
def fold_partial_sum(chunk_base: Int64, partial_base: Int64, partial_idx: Int32, num_chunks: Int32, D: cutlass.Constexpr, tidx: Int32):
    """
    partial = ((0 + chunk 0) + chunk 1) + ... of the partial's chunk sums (read through L2: other
    CTAs wrote them).
    """
    acc_sum = cute.make_rmem_tensor(D // 64, Float32)
    for j in cutlass.range_constexpr(D // 64):
        acc_sum[j] = Float32(0.0)
    first_addr = chunk_base + (Int64(partial_idx) * Int64(num_chunks) * Int64(D) + Int64(tidx)) * 4
    for chunk_idx in cutlass.range(num_chunks, unroll=1):
        for j in cutlass.range_constexpr(D // 64):
            acc_sum[j] = add_f32(acc_sum[j], ld_cg_f32(first_addr + (Int64(chunk_idx) * Int64(D) + Int64(64 * j)) * 4))
    for j in cutlass.range_constexpr(D // 64):
        st_f32_pred(partial_base + (Int64(partial_idx) * Int64(D) + Int64(tidx + 64 * j)) * 4, acc_sum[j], Int32(1))


@cute.jit
def fold_head_mean(partial_base: Int64, mean_base: Int64, batch_head: Int32, num_partials: Int32, D: cutlass.Constexpr, tidx: Int32, N: Int32):
    """
    mean = div.full(((0 + partial 0) + partial 1) + ..., N) of one head; the partials, written by
    other CTAs, are read through L2.
    """
    acc_sum = cute.make_rmem_tensor(D // 64, Float32)
    for j in cutlass.range_constexpr(D // 64):
        acc_sum[j] = Float32(0.0)
    first_addr = partial_base + (Int64(batch_head) * Int64(num_partials) * Int64(D) + Int64(tidx)) * 4
    for group in cutlass.range(num_partials // 8, unroll=1):
        values = [[ld_cg_f32(first_addr + ((Int64(group) * 8 + Int64(i)) * Int64(D) + Int64(64 * j)) * 4) for j in range(D // 64)] for i in range(8)]
        for i in cutlass.range_constexpr(8):
            for j in cutlass.range_constexpr(D // 64):
                acc_sum[j] = add_f32(acc_sum[j], values[i][j])
    for partial_idx in cutlass.range(num_partials - (num_partials // 8) * 8, unroll=1):
        for j in cutlass.range_constexpr(D // 64):
            acc_sum[j] = add_f32(acc_sum[j], ld_cg_f32(first_addr + ((Int64(num_partials // 8) * 8 + Int64(partial_idx)) * Int64(D) + Int64(64 * j)) * 4))
    tokens = Float32(inline_asm(T.f32(), [ir_value(N, Int32, None, None)], "cvt.rn.f32.s32 $0, $1;", "=f,r"))
    for j in cutlass.range_constexpr(D // 64):
        st_f32_pred(mean_base + (Int64(batch_head) * Int64(D) + Int64(tidx + 64 * j)) * 4, Float32(inline_asm(T.f32(), [ir_value(acc_sum[j], Float32, None, None), ir_value(tokens, Float32, None, None)], "div.full.f32 $0, $1, $2;", "=f,f,f")), Int32(1))


def quantize_qkv_tile(D, tile_tidx, work_idx, work_start, pair, num_pairs, tiles_per_head, num_heads, x_base, strides, vec, prescale, s_amax, code_base, scale_base, byte_base, byte_slot, N):
    """
    One Q (prescale, no byte) or V (byte slot 1) tile of the producers: tile (work_idx - work_start)
    * num_pairs + pair.
    """
    tile = TileThread(D, tile_tidx, 2)
    tile_idx = work_idx if work_start is None else work_idx - work_start
    if pair is not None:
        tile_idx = tile_idx * num_pairs + pair
    batch_head = tile_idx // tiles_per_head
    batch_idx = batch_head // num_heads
    head_idx = batch_head % num_heads
    row0 = (tile_idx % tiles_per_head) * BLOCK
    words = load_tile_words(tile, x_base, batch_idx, head_idx, row0, strides[0], strides[1], strides[2], vec)
    values_of = row_values(words, vec, None, prescale) if prescale is not None else row_values(words, vec)
    amax_slot = s_amax + pair * 16 if pair is not None else s_amax
    quantize_tile(tile, values_of, amax_slot, TileTarget(code_base, scale_base, byte_base, byte_slot, N, num_heads, head_idx, batch_head, batch_idx, row0), pair_barrier(pair, num_pairs))


class QkvProducersChunked:
    """
    Q / V tiles and the K mean at D=128, every CTA 2 warps; 1-D grid [K chunks | Q tiles | V tiles].
    A chunk CTA sums one 64-row chunk of one partial and publishes it (fence, then an atomic arrival
    on the partial's counter); the CTA completing a partial folds its chunk sums in chunk order, so
    the sum does not depend on the arrival order, and the CTA completing a head's last partial folds
    the partials and writes the mean.  The completing CTA resets each counter.
    """

    def __init__(self, D, vec_q, vec_k, vec_v, k_aligned):
        assert D in (128, 256)
        self.D, self.vec_q, self.vec_k, self.vec_v, self.k_aligned = D, vec_q, vec_k, vec_v, k_aligned

    @cute.jit
    def __call__(self, q_base: Int64, k_base: Int64, v_base: Int64, q_code_base: Int64, v_code_base: Int64, q_scale_base: Int64, v_scale_base: Int64, byte_base: Int64, partial_base: Int64, chunk_base: Int64, counter_base: Int64, mean_base: Int64, N: Int32, H: Int32, Hk: Int32, B: Int32, q_stride_b: Int32, q_stride_n: Int32, q_stride_h: Int32, k_stride_b: Int32, k_stride_n: Int32, k_stride_h: Int32, v_stride_b: Int32, v_stride_n: Int32, v_stride_h: Int32, softmax_scale: Float32, rows_per_partial: Int32, stream: cuda.CUstream):
        tiles_per_head = N // BLOCK
        num_ctas = B * H * tiles_per_head + B * Hk * tiles_per_head + B * Hk * (N // rows_per_partial) * (rows_per_partial // 64)
        self.kernel(q_base, k_base, v_base, q_code_base, v_code_base, q_scale_base, v_scale_base, byte_base, partial_base, chunk_base, counter_base, mean_base, N, H, Hk, B, q_stride_b, q_stride_n, q_stride_h, k_stride_b, k_stride_n, k_stride_h, v_stride_b, v_stride_n, v_stride_h, softmax_scale, rows_per_partial).launch(grid=[num_ctas, 1, 1], block=[64, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, q_base: Int64, k_base: Int64, v_base: Int64, q_code_base: Int64, v_code_base: Int64, q_scale_base: Int64, v_scale_base: Int64, byte_base: Int64, partial_base: Int64, chunk_base: Int64, counter_base: Int64, mean_base: Int64, N: Int32, H: Int32, Hk: Int32, B: Int32, q_stride_b: Int32, q_stride_n: Int32, q_stride_h: Int32, k_stride_b: Int32, k_stride_n: Int32, k_stride_h: Int32, v_stride_b: Int32, v_stride_n: Int32, v_stride_h: Int32, softmax_scale: Float32, rows_per_partial: Int32):
        D = self.D
        tidx, _, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        s_amax = alloc_smem(2 * D * 4)
        s_arrivals = alloc_smem(16)
        tiles_per_head = N // BLOCK
        num_q_tiles = B * H * tiles_per_head
        num_v_tiles = B * Hk * tiles_per_head
        num_partials = N // rows_per_partial
        num_chunks = rows_per_partial // 64
        num_chunk_ctas = B * Hk * num_partials * num_chunks
        # chunk CTAs first: the longest dependency chain starts first
        work_idx = bx - num_chunk_ctas
        if bx < num_chunk_ctas:
            work_idx = work_idx + num_q_tiles + num_v_tiles + num_chunk_ctas
        if work_idx < num_q_tiles:
            quantize_qkv_tile(D, tidx, work_idx, None, None, 1, tiles_per_head, H, q_base, (q_stride_b, q_stride_n, q_stride_h), self.vec_q, softmax_scale, s_amax, q_code_base, q_scale_base, None, None, N)
        elif work_idx < num_q_tiles + num_v_tiles:
            quantize_qkv_tile(D, tidx, work_idx, num_q_tiles, None, 1, tiles_per_head, Hk, v_base, (v_stride_b, v_stride_n, v_stride_h), self.vec_v, None, s_amax, v_code_base, v_scale_base, byte_base, 1, N)
        else:
            self.chunk_role(tidx, work_idx - num_q_tiles - num_v_tiles, num_chunks, num_partials, B, Hk, N, rows_per_partial, k_base, k_stride_b, k_stride_n, k_stride_h, s_amax, s_arrivals, chunk_base, partial_base, counter_base, mean_base)

    @cute.jit
    def chunk_role(self, tidx: Int32, chunk_cta: Int32, num_chunks: Int32, num_partials: Int32, B: Int32, Hk: Int32, N: Int32, rows_per_partial: Int32, k_base: Int64, k_stride_b: Int32, k_stride_n: Int32, k_stride_h: Int32, s_partial: Int32, s_arrivals: Int32, chunk_base: Int64, partial_base: Int64, counter_base: Int64, mean_base: Int64):
        """
        Sum and publish one chunk; fold the partial, then the head mean, if this CTA completes it.
        """
        D = self.D
        partial_idx = chunk_cta // num_chunks
        chunk_idx = chunk_cta % num_chunks
        batch_head = partial_idx // num_partials
        partial_in_head = partial_idx % num_partials
        batch_idx = batch_head // Hk
        head_idx = batch_head % Hk
        chunk = ChunkThread(D, tidx, self.k_aligned)
        totals = sum_chunk_columns(chunk, k_base, batch_idx, head_idx, partial_in_head * rows_per_partial + chunk_idx * 64, k_stride_b, k_stride_n, k_stride_h, self.vec_k, s_partial)
        chunk_addr = chunk_base + ((Int64(partial_idx) * Int64(num_chunks) + Int64(chunk_idx)) * Int64(D) + Int64(chunk.col0)) * 4
        is_publisher = Int32(chunk.warp == 0) * Int32(chunk.row_lane == 0)
        st_v4_f32_pred(chunk_addr, totals[0], totals[1], totals[2], totals[3], is_publisher)
        st_v4_f32_pred(chunk_addr + Int64(16), totals[4], totals[5], totals[6], totals[7], is_publisher)
        # every writing thread fences before the barrier; thread 0's acq_rel atomic then publishes the whole CTA's stores
        fence_acq_rel_gpu()
        bar_sync()
        if tidx == 0:
            arrivals = atom_add_acq_rel_gpu(counter_base + Int64(partial_idx) * 4, Int32(1)) + Int32(1)
            sts_i32_pred(s_arrivals, arrivals, Int32(1))
        bar_sync()
        if lds_i32(s_arrivals) == num_chunks:
            fence_acq_rel_gpu()
            fold_partial_sum(chunk_base, partial_base, partial_idx, num_chunks, D, tidx)
            st_b32_pred(counter_base + Int64(partial_idx) * 4, Int32(0), Int32(tidx == 0))
            # second level: counter B * Hk * num_partials + head counts the head's completed partials
            fence_acq_rel_gpu()
            bar_sync()
            if tidx == 0:
                head_arrivals = atom_add_acq_rel_gpu(counter_base + (Int64(B) * Int64(Hk) * Int64(num_partials) + Int64(batch_head)) * 4, Int32(1)) + Int32(1)
                sts_i32_pred(s_arrivals + 4, head_arrivals, Int32(1))
            bar_sync()
            if lds_i32(s_arrivals + 4) == num_partials:
                fence_acq_rel_gpu()
                fold_head_mean(partial_base, mean_base, batch_head, num_partials, D, tidx, N)
                st_b32_pred(counter_base + (Int64(B) * Int64(Hk) * Int64(num_partials) + Int64(batch_head)) * 4, Int32(0), Int32(tidx == 0))


class QkvProducersPaired:
    """
    Q / V tiles and the K mean at D=256: 1-D grid [K partials | Q tile pairs | V tile pairs], each
    CTA num_warps / 2 warp pairs.  Tile CTAs give each pair its own 32-row tile; a partial CTA runs
    one chunk stream per pair and pair 0 folds the chunk sums in chunk order.  The CTA completing a
    head's last partial folds the partials and writes the mean; it also resets the head's counter.
    """

    def __init__(self, D, vec_q, vec_k, vec_v, k_aligned, num_warps):
        assert D in (128, 256) and num_warps == 4
        self.D, self.vec_q, self.vec_k, self.vec_v, self.k_aligned, self.num_warps = D, vec_q, vec_k, vec_v, k_aligned, num_warps

    @cute.jit
    def __call__(self, q_base: Int64, k_base: Int64, v_base: Int64, q_code_base: Int64, v_code_base: Int64, q_scale_base: Int64, v_scale_base: Int64, byte_base: Int64, partial_base: Int64, counter_base: Int64, mean_base: Int64, N: Int32, H: Int32, Hk: Int32, B: Int32, q_stride_b: Int32, q_stride_n: Int32, q_stride_h: Int32, k_stride_b: Int32, k_stride_n: Int32, k_stride_h: Int32, v_stride_b: Int32, v_stride_n: Int32, v_stride_h: Int32, softmax_scale: Float32, rows_per_partial: Int32, stream: cuda.CUstream):
        tiles_per_head = N // BLOCK
        num_pairs = self.num_warps // 2
        num_ctas = (B * H * tiles_per_head) // num_pairs + (B * Hk * tiles_per_head) // num_pairs + B * Hk * (N // rows_per_partial)
        self.kernel(q_base, k_base, v_base, q_code_base, v_code_base, q_scale_base, v_scale_base, byte_base, partial_base, counter_base, mean_base, N, H, Hk, B, q_stride_b, q_stride_n, q_stride_h, k_stride_b, k_stride_n, k_stride_h, v_stride_b, v_stride_n, v_stride_h, softmax_scale, rows_per_partial).launch(grid=[num_ctas, 1, 1], block=[32 * self.num_warps, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, q_base: Int64, k_base: Int64, v_base: Int64, q_code_base: Int64, v_code_base: Int64, q_scale_base: Int64, v_scale_base: Int64, byte_base: Int64, partial_base: Int64, counter_base: Int64, mean_base: Int64, N: Int32, H: Int32, Hk: Int32, B: Int32, q_stride_b: Int32, q_stride_n: Int32, q_stride_h: Int32, k_stride_b: Int32, k_stride_n: Int32, k_stride_h: Int32, v_stride_b: Int32, v_stride_n: Int32, v_stride_h: Int32, softmax_scale: Float32, rows_per_partial: Int32):
        D = self.D
        num_pairs = self.num_warps // 2
        tidx, _, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        s_amax = alloc_smem(self.num_warps * D * 4)
        s_chunk_sums = alloc_smem(16 * D * 4)
        pair = tidx // 64
        tiles_per_head = N // BLOCK
        num_q_ctas = (B * H * tiles_per_head) // num_pairs
        num_v_ctas = (B * Hk * tiles_per_head) // num_pairs
        num_partials = N // rows_per_partial
        num_partial_ctas = B * Hk * num_partials
        # partial CTAs first: the longest dependency chain starts first
        work_idx = bx - num_partial_ctas
        if bx < num_partial_ctas:
            work_idx = work_idx + num_q_ctas + num_v_ctas + num_partial_ctas
        if work_idx < num_q_ctas:
            quantize_qkv_tile(D, tidx % 64, work_idx, None, pair, num_pairs, tiles_per_head, H, q_base, (q_stride_b, q_stride_n, q_stride_h), self.vec_q, softmax_scale, s_amax, q_code_base, q_scale_base, None, None, N)
        elif work_idx < num_q_ctas + num_v_ctas:
            quantize_qkv_tile(D, tidx % 64, work_idx, num_q_ctas, pair, num_pairs, tiles_per_head, Hk, v_base, (v_stride_b, v_stride_n, v_stride_h), self.vec_v, None, s_amax, v_code_base, v_scale_base, byte_base, 1, N)
        else:
            self.partial_role(tidx, pair, work_idx - num_q_ctas - num_v_ctas, num_partials, Hk, N, rows_per_partial, k_base, k_stride_b, k_stride_n, k_stride_h, s_amax, s_chunk_sums, partial_base, counter_base, mean_base)

    @cute.jit
    def partial_role(self, tidx: Int32, pair: Int32, partial_cta: Int32, num_partials: Int32, Hk: Int32, N: Int32, rows_per_partial: Int32, k_base: Int64, k_stride_b: Int32, k_stride_n: Int32, k_stride_h: Int32, s_partial: Int32, s_chunk_sums: Int32, partial_base: Int64, counter_base: Int64, mean_base: Int64):
        """
        One partial: pair p sums chunks p, p + num_pairs, ... with the 2-warp tree (its own shared
        region and barrier), pair 0 folds the chunk sums in chunk order; then the head mean if this
        CTA completes the head.
        """
        D = self.D
        num_pairs = self.num_warps // 2
        batch_head = partial_cta // num_partials
        partial_in_head = partial_cta % num_partials
        batch_idx = batch_head // Hk
        head_idx = batch_head % Hk
        chunk = ChunkThread(D, tidx % 64, self.k_aligned)
        is_publisher = Int32(chunk.warp == 0) * Int32(chunk.row_lane == 0)
        num_chunks = rows_per_partial // 64
        for round_idx in cutlass.range((num_chunks + num_pairs - 1) // num_pairs, unroll=1):
            chunk_idx = round_idx * num_pairs + pair
            if chunk_idx < num_chunks:
                totals = sum_chunk_columns(chunk, k_base, batch_idx, head_idx, partial_in_head * rows_per_partial + chunk_idx * 64, k_stride_b, k_stride_n, k_stride_h, self.vec_k, s_partial + pair * (2 * D * 4), pair_barrier(pair, num_pairs))
                for j in cutlass.range_constexpr(8):
                    sts_f32_pred(s_chunk_sums + (chunk_idx * D + chunk.col0 + j) * 4, totals[j], is_publisher)
        bar_sync()
        acc_sum = cute.make_rmem_tensor(8, Float32)
        for j in cutlass.range_constexpr(8):
            acc_sum[j] = Float32(0.0)
        for chunk_idx in cutlass.range(num_chunks, unroll=1):
            for j in cutlass.range_constexpr(8):
                acc_sum[j] = add_f32(acc_sum[j], ld_shared_f32(s_chunk_sums + (chunk_idx * D + chunk.col0 + j) * 4))
        partial_addr = partial_base + ((Int64(batch_head) * Int64(num_partials) + Int64(partial_in_head)) * Int64(D) + Int64(chunk.col0)) * 4
        is_writer = is_publisher * Int32(pair == 0)
        st_v4_f32_pred(partial_addr, acc_sum[0], acc_sum[1], acc_sum[2], acc_sum[3], is_writer)
        st_v4_f32_pred(partial_addr + Int64(16), acc_sum[4], acc_sum[5], acc_sum[6], acc_sum[7], is_writer)
        # every writing thread fences before the barrier; thread 0's acq_rel atomic then publishes the whole CTA's stores
        fence_acq_rel_gpu()
        bar_sync()
        if tidx == 0:
            head_arrivals = atom_add_acq_rel_gpu(counter_base + Int64(batch_head) * 4, Int32(1)) + Int32(1)
            sts_i32_pred(s_chunk_sums, head_arrivals, Int32(1))
        bar_sync()
        if lds_i32(s_chunk_sums) == num_partials:
            fence_acq_rel_gpu()
            if tidx < 64:
                fold_head_mean(partial_base, mean_base, batch_head, num_partials, D, tidx, N)
            st_b32_pred(counter_base + Int64(batch_head) * 4, Int32(0), Int32(tidx == 0))


# Host side: compile cache and launchers.


def persistent_cache():
    """
    The compile cache under the package fingerprint (FA4 and delta_matching/sm100 sources, compile
    settings).
    """
    if not PERSISTENT:
        from flash_attn.cute import cache_utils
        from delta_matching.sm100.compile_cache_sm100 import fingerprint
        if cache_utils.CUTE_DSL_CACHE_ENABLED:
            PERSISTENT.append(cache_utils.JITPersistentCache(cache_utils.get_cache_path() / "dm_sm100" / fingerprint("quant") / "quant"))
        else:
            PERSISTENT.append(cache_utils.JITCache())
    return PERSISTENT[0]


def compile_kernel(kernel, signature):
    """
    Compile with every argument an Int64 address ("l"), Int32 ("i") or Float32 ("f") and an explicit
    stream: the host passes raw pointers and the current stream, which skips TVM-FFI tensor
    marshalling.
    """
    from delta_matching.sm100 import launch_cache_sm100
    fake = {"l": Int64(0), "i": Int32(0), "f": Float32(0.0)}
    args = [fake[kind] for kind in signature]
    args.append(cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False))
    launch_cache_sm100.EPOCH[0] += 1
    return cute.compile(kernel, *args, options="--enable-tvm-ffi")


def get_compiled(key, make_kernel, signature):
    fn = COMPILED.get(key)
    if fn is None:
        with COMPILE_LOCK:
            fn = COMPILED.get(key)
            if fn is None:
                cache = persistent_cache()
                fn = cache.get(key)
                if fn is None:
                    fn = compile_kernel(make_kernel(), signature)
                    cache[key] = fn
                COMPILED[key] = fn
    return fn


def vector_loads_ok(x):
    """
    True when 16-byte vector loads are legal: the base and every outer stride (bytes) are multiples
    of 16.
    """
    elem_bytes = x.element_size()
    return x.data_ptr() % 16 == 0 and all((stride * elem_bytes) % 16 == 0 for stride in x.stride()[:3])


def k_layout_aligned(k, num_kv_heads):
    """
    Selects the K-mean chunk grouping: pointer % 16 and the batch / token strides % 16 elements, the
    head stride too unless there is one KV head.
    """
    stride_b, stride_n, stride_h = k.stride()[:3]
    return k.data_ptr() % 16 == 0 and stride_b % 16 == 0 and stride_n % 16 == 0 and (num_kv_heads == 1 or stride_h % 16 == 0)


def current_stream(x):
    return torch._C._cuda_getCurrentRawStream(x.get_device())


def compile_row_quant(D, x, other, has_mean, byte_slot, delta):
    """
    The 32-row quantization kernel for x: optional mean (K centring), UE8M0 byte slot (0 for K, None
    for no bytes), delta None or "stale" (rowsum(x * other) of the bf16 tensor other).
    """
    vec_x = vector_loads_ok(x)
    vec_other = vector_loads_ok(other) if delta is not None else False
    num_warps, reconvert = TILE_CONFIG[D]
    key = ("row_quant", D, vec_x, has_mean, byte_slot, delta, vec_other, num_warps, reconvert)
    return get_compiled(key, lambda: RowQuant(D, vec_x, has_mean, byte_slot, delta, vec_other, num_warps, reconvert), ["l"] * 7 + ["i"] * 9)


def launch_row_quant(x, codes, scales, mean=None, other=None, delta=None, delta_out=None, byte_words=None, byte_slot=None):
    B, N, H, D = x.shape
    x_ptr = x.data_ptr()
    other_ptr = other.data_ptr() if other is not None else 0
    key = ("row_quant", x.shape, x.stride(), x.dtype, x_ptr & 15, mean is not None, byte_slot, delta, None if other is None else (other.stride(), other.dtype, other_ptr & 15))
    fn = LAUNCHERS.get(key)
    if fn is None:
        fn = LAUNCHERS[key] = compile_row_quant(D, x, other, mean is not None, byte_slot, delta)
    other_strides = other.stride() if other is not None else (0, 0, 0)
    x_strides = x.stride()
    fn(x_ptr, other_ptr, mean.data_ptr() if mean is not None else 0, codes.data_ptr(), scales.data_ptr(), delta_out.data_ptr() if delta_out is not None else 0, byte_words.data_ptr() if byte_words is not None else 0, N, H, B, x_strides[0], x_strides[1], x_strides[2], other_strides[0], other_strides[1], other_strides[2], current_stream(x))


def compile_qkv_producers(q, k, v):
    """
    The Q / V / K-mean kernel for these inputs (D=128: chunked, D=256: warp pairs); both leave the
    final K mean behind, so the K tiles that follow only quantize.
    """
    D, num_kv_heads = q.shape[-1], k.shape[2]
    vec_q, vec_k, vec_v, k_aligned = vector_loads_ok(q), vector_loads_ok(k), vector_loads_ok(v), k_layout_aligned(k, num_kv_heads)
    if D == 128:
        return get_compiled(("qkv_chunked", D, vec_q, vec_k, vec_v, k_aligned), lambda: QkvProducersChunked(D, vec_q, vec_k, vec_v, k_aligned), ["l"] * 12 + ["i"] * 13 + ["f", "i"])
    num_warps = PAIRED_WARPS[D]
    return get_compiled(("qkv_paired", D, vec_q, vec_k, vec_v, k_aligned, num_warps), lambda: QkvProducersPaired(D, vec_q, vec_k, vec_v, k_aligned, num_warps), ["l"] * 11 + ["i"] * 13 + ["f", "i"])


def kmean_counters(q, stream, num_counters):
    """
    The int32 K-mean arrival counters of (device, stream), zero at rest: each counter is reset by
    the CTA that completes it, and kernel completion orders that before the next launch on the
    stream. Grown-from buffers stay alive, because a captured CUDA graph may still point at them.
    """
    key = (q.get_device(), stream)
    counters = COUNTERS.get(key)
    if counters is None or counters.numel() < num_counters:
        if counters is not None:
            RETIRED_COUNTERS.append(counters)
        counters = COUNTERS[key] = torch.zeros(max(num_counters, 1024), device=q.device, dtype=torch.int32)
    return counters


def launch_resetting_counters(counters, launch):
    """
    A kernel cannot stop half-way without a sticky CUDA error, and a failed launch never ran; any
    exception from the launch call still re-zeroes the counters, so the next call starts from zero.
    """
    try:
        launch()
    except BaseException:
        counters.zero_()
        raise


def fp8_quant_qkv(q, k, v, softmax_scale: float):
    """
    Q, K, V (B, N, H or Hk, D) bf16 with unit stride on D, N % 128 == 0, D 128 or 256.  Returns
    (q_codes, q_scales, k_codes, k_scales, v_codes, v_scales, kv_bytes, k_mean): codes (B, N, H or
    Hk, D) e4m3 contiguous (Q multiplied by softmax_scale, K centred by k_mean); scales (B, H or Hk,
    N / 32) fp32 powers of 2; kv_bytes (B, Hk, N / 64) int32 packed UE8M0 bytes (word 2t holds the K
    bytes of 128-key tile t, word 2t + 1 the V bytes); k_mean (B, Hk, D) fp32.
    """
    B, N, H, D = q.shape
    num_kv_heads = k.shape[2]
    q_strides, k_strides, v_strides = q.stride(), k.stride(), v.stride()
    q_ptr, k_ptr, v_ptr = q.data_ptr(), k.data_ptr(), v.data_ptr()
    assert N % 128 == 0 and q_strides[-1] == 1 and k_strides[-1] == 1 and v_strides[-1] == 1
    assert q.dtype == k.dtype == v.dtype == torch.bfloat16
    rows_per_partial = ROWS_PER_PARTIAL
    while N % rows_per_partial:
        rows_per_partial //= 2
    num_partials = N // rows_per_partial
    device = q.device
    stream = current_stream(q)
    e4m3 = torch.float8_e4m3fn
    q_codes = torch.empty(B, N, H, D, device=device, dtype=e4m3)
    k_codes = torch.empty(B, N, num_kv_heads, D, device=device, dtype=e4m3)
    v_codes = torch.empty(B, N, num_kv_heads, D, device=device, dtype=e4m3)
    q_scales = torch.empty(B, H, N // 32, device=device, dtype=torch.float32)
    k_scales = torch.empty(B, num_kv_heads, N // 32, device=device, dtype=torch.float32)
    v_scales = torch.empty(B, num_kv_heads, N // 32, device=device, dtype=torch.float32)
    kv_bytes = torch.empty(B, num_kv_heads, N // 64, device=device, dtype=torch.int32)
    partials = torch.empty(B * num_kv_heads, num_partials, D, device=device, dtype=torch.float32)
    k_mean = torch.empty(B, num_kv_heads, D, device=device, dtype=torch.float32)
    key = (q.shape, k.shape, q_strides, k_strides, v_strides, q_ptr & 15, k_ptr & 15, v_ptr & 15)
    fn = LAUNCHERS.get(key)
    if fn is None:
        fn = LAUNCHERS[key] = compile_qkv_producers(q, k, v)
    sizes = (N, H, num_kv_heads, B, q_strides[0], q_strides[1], q_strides[2], k_strides[0], k_strides[1], k_strides[2], v_strides[0], v_strides[1], v_strides[2], float(softmax_scale), rows_per_partial, stream)
    if D == 128:
        chunk_sums = torch.empty(B * num_kv_heads * num_partials * (rows_per_partial // 64), D, device=device, dtype=torch.float32)
        counters = kmean_counters(q, stream, B * num_kv_heads * num_partials + B * num_kv_heads)
        launch_resetting_counters(counters, lambda: fn(q_ptr, k_ptr, v_ptr, q_codes.data_ptr(), v_codes.data_ptr(), q_scales.data_ptr(), v_scales.data_ptr(), kv_bytes.data_ptr(), partials.data_ptr(), chunk_sums.data_ptr(), counters.data_ptr(), k_mean.data_ptr(), *sizes))
    else:
        counters = kmean_counters(q, stream, B * num_kv_heads)
        launch_resetting_counters(counters, lambda: fn(q_ptr, k_ptr, v_ptr, q_codes.data_ptr(), v_codes.data_ptr(), q_scales.data_ptr(), v_scales.data_ptr(), kv_bytes.data_ptr(), partials.data_ptr(), counters.data_ptr(), k_mean.data_ptr(), *sizes))
    launch_row_quant(k, k_codes, k_scales, mean=k_mean, byte_words=kv_bytes, byte_slot=0)
    return q_codes, q_scales, k_codes, k_scales, v_codes, v_scales, kv_bytes, k_mean


def fp8_quant_dout(dout, out=None):
    """
    dO (B, N, H, D) bf16 with unit stride on D, D 128 or 256.  Returns (codes, scales, delta): codes
    (B, N, H, D) e4m3, scales (B, H, N / 32) fp32 powers of 2, and delta (B, H, N) fp32 =
    rowsum(dO * O) when out (O, bf16, same shape) is given (stale mode), else None.
    """
    B, N, H, D = dout.shape
    assert dout.stride(-1) == 1 and (out is None or out.stride(-1) == 1)
    assert dout.dtype == torch.bfloat16 and (out is None or (out.dtype == torch.bfloat16 and out.shape == dout.shape))
    codes = torch.empty(B, N, H, D, device=dout.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty(B, H, N // BLOCK, device=dout.device, dtype=torch.float32)
    delta = torch.empty(B, H, N, device=dout.device, dtype=torch.float32) if out is not None else None
    launch_row_quant(dout, codes, scales, other=out, delta="stale" if out is not None else None, delta_out=delta)
    return codes, scales, delta
