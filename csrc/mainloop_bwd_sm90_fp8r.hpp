/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/
/******************************************************************************
 * ## Delta-Matching-128 v2 (M20): PASS 1 "P1-R" = the delta_matching fp8 split backward's dK/dV pass at head_dim 128
 * in FA3's OWN hd128 backward geometry (kBlockM 64, kBlockN 128, SdP_swapAB, Mma_dKV_is_RS, AtomLayoutNdKV 2), selected
 * by FA3FP11_HD128_P1R=1 (set in the release build; the P1-A collective in mainloop_bwd_sm90_fp8v10.hpp stays the
 * d256 path).
 *
 * What is the same as P1-A (the certified hd128 port), per element and per tile, so that dq / dk / dv / psi / the dS'
 * codes are EXPECTED BITWISE (measured, not argued; the wgmma operand-swap symmetry was probed first):
 *   - the block-scale contract ([128, 64]: q8s / do8s per 128 rows = 2 m-blocks, k8s / v8s per 64 keys = one WG),
 *   - the exp2 line with the F1 cliff clamp, the dS line with the k8s fold, the psi rule (tile amax / 448 across both
 *     WGs, the F2 zero-tile sentinel, the F2b emission floor), the e4m3 conversions, the dS' workspace + psi contract pass 2
 *     consumes ([BH, nbm, nbn, 64, 128] tiles, the same bytes), the M4 dV window (rescale by the clamped f ratio, promote
 *     every 8 m-tiles into the smem overlay in true units) and the plan-A dK per-tile temp + fold, the exit k8s unfold.
 * What changes (schedule / geometry / instruction count only):
 *   - S^T = K Q^T and dP^T = V dO^T per WG (A = the WG's 64 keys of the natural K / V tiles, B = the natural Q / dO stage):
 *     the same m64n64k32 atoms over the same k order, so every element is the bitwise transpose of P1-A's S / dP (probe).
 *   - dV = P^T dO and dK_tmp = dS'^T Q as RS wgmma (m64n128k32 x 2, the same atoms as P1-A's dV^T / dK^T gemms with M and
 *     N swapped): the A operands are the e4m3 codes STRAIGHT FROM THE REGISTER FRAGMENTS (permute_Cregs_fp8 register
 *     renaming, then the FA3 Aregs view), against the dOt / Qt twins (B, K-major in rows).  The twins' pi row order
 *     (SIGMA = [0,1,8,9,2,3,10,11,4,5,12,13,6,7,14,15] per 16) is EXACTLY the k order the fp8 A fragment carries (register
 *     h, k slot 4t'+j <-> accumulator column 8(j>>1) + 2t' + (j&1)), so the M17 producers and the op schema are unchanged
 *     and the contraction runs over the same k sequence as P1-A.
 *   - No P~^T / dS'^T exchange tiles, no byte scatters, no cross-WG PdS barriers, no smem A-operand reads for dV / dK.
 *   - The ONE transpose left (dS' natural for pass 2) is done per WG with 16 shfl + 16 prmt + 2 stmatrix.trans per thread
 *     into a private 64 x 64 SW64 half tile with its own TMA box store: no cross-WG synchronization on that path.
 *   - LSE / dPsum are read from the stage's smem rows per column (swapAB: 16 per thread), never held in registers.
 *   - The psi cross-WG max keeps one named barrier per tile (parity double-buffered slots: no other CTA-wide barrier
 *     orders the slot reads any more).
 *   - Producer warpgroup: FA3's 24 / 240 register split (no transposer warps).
 * DBG (FA3FP11_DBG): the same rows / rowsabs / cks instrumentation contract as P1-A (per (bh, n-tile, WG, row) dS row
 * sums in true units; the WG's 64-key sum is reduced 2 keys per thread -> 8 lanes -> 4 warps in a fixed order).
 ******************************************************************************/
#pragma once

#include <climits>
#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/barrier.h>
#include "cutlass/pipeline/pipeline.hpp"

#include "cute/tensor.hpp"

#include "cutlass/gemm/collective/builders/sm90_common.inl"

#include "named_barrier.hpp"
#include "seqlen.h"
#include "block.h"
#include "mask.h"
#include "softmax.h"
#include "utils.h"

static_assert(FA3FP11_M17_LSEK == 1, "Delta-Matching-128 v2 (P1-R) is written for the in-kernel lse*log2e (FA3FP11_M17_LSEK=1) build line");
// ## M20 micro-levers (bitwise-class schedule changes; the release build sets RFSKIP, DSDEFER stays off):
//   FA3FP11_P1R_DSDEFER: each WG's dS' half tile is double-buffered and its TMA box store is issued ONE TILE LATER, right
//     after the next tile's psi barrier (which every thread reaches only after its stmatrix writes + proxy fence), so the two
//     per-tile WG-local named barriers disappear; the elected thread drains store(t-2) before that barrier.
//   FA3FP11_P1R_RFSKIP: the dV window rescale (64 FMUL per thread) is skipped when the clamped ratio is exactly 1.0f (every
//     second m-block at hd128: one dO scale per 128 rows).  x * 1.0f is exact for normal x; the promote's FFMA flushes a
//     denormal window value either way.
// kda-pass1 KDA_P1_DSDEFER (bitwise schedule): the M20 DSDEFER design re-implemented on the H1b source: buffer m & 1,
// the elected thread drains store(m-2) before the tile's psi barrier and issues store(m-1) right after it; no
// WG-local named barrier on the dS' path (the last tile's store gets one barrier after the loop).
#ifndef KDA_P1_DSDEFER
#define KDA_P1_DSDEFER 0
#endif
// kda-pass1 KDA_P1_PV4 (bitwise): the dV overlay as float4 slots, slot4(j, thread) = j * NumMmaThreads + thread
// (16 LDS.128 / STS.128 per promote instead of 64 + 64 scalar accesses; same values, same FMA order).
#ifndef KDA_P1_PV4
#define KDA_P1_PV4 0
#endif
#if KDA_P1_DSDEFER
#undef FA3FP11_P1R_DSDEFER
#define FA3FP11_P1R_DSDEFER 1
#endif
#ifndef FA3FP11_P1R_DSDEFER
#define FA3FP11_P1R_DSDEFER 0
#endif
// ## M21 P1R_MASKDIAG (bitwise-class; the release build sets it): the causal mask is applied only to the m-blocks whose rows can
// see a masked key of this CTA's 128 keys (the two diagonal m-blocks; the seqlen_k tail masks every tile), instead of every tile.
// ## M21 TIMING PROBE ONLY (numerics wrong: psi from this warpgroup's half alone, no cross-WG barrier): the upper bound of
// decoupling the two warpgroups.  Never in a shipped build.
#ifndef FA3FP11_P1R_NOPSIBAR
#define FA3FP11_P1R_NOPSIBAR 0
#endif
// ## M23 candidate B, HALFPSI (a NEW NUMERICS ARM, default off): psi per (tile, warpgroup half) = the amax of
// this warpgroup's 64-key half tile / 448 (WG-local max over its four warps, a WG-local named barrier), published at
// ptr_psi[(tile) * 2 + wg]; no cross-warpgroup exchange or CTA-wide barrier per tile.  The dK fold and the F2b floor were
// per warpgroup already.  Pass 2 (fq8_dq_pass2_hd128.h, HALFPSI) consumes the two scales with two accumulators.
#ifndef FA3FP11_HALFPSI
#define FA3FP11_HALFPSI 0
#endif
// ## M21 P1R_UNROLL2 (bitwise-class, default off): the m-block loop unrolled by the Q / Qt stage period (2) so that every
// smem stage index (Q, LSE, Qt; dO / dPsum / dOt are single-stage) is a compile-time constant: the wgmma descriptors and the
// per-column statistic addresses become immediates instead of per-tile integer / uniform arithmetic.  Same arithmetic,
// same order, same pipeline waits (the runtime pipeline states still drive the barriers).
#ifndef FA3FP11_P1R_UNROLL2
#define FA3FP11_P1R_UNROLL2 0
#endif
// ## bf16-dP (NEW ARM, default off): SageAttention's SageBwd exempts exactly one of the seven attention
// matmuls from quantization, (5) dP = dO V^T, which it keeps in raw 16-bit while quantizing the other six
// (arXiv 2603.02170: "SageBwd retains dP = dOV^T in FP16 precision, while quantizing the remaining four matrix
// multiplications ... This design choice avoids error amplification through dS, dQ, and dK").  At SAGEDP=1 pass 1
// reads the RAW bf16 V and dO for that one gemm and its accumulator is already in true dP units, so the code-scale
// product drops out of the dS' line.  Everything else, including the delta (which stays the STALE rowsum(dO*O) of
// the quantized forward, as in SageBwd), is bit-identical to the certified line.  This is a NEW ARM, never a
// bitwise change.  At 0 the whole thing compiles out.
#ifndef FA3FP11_P1R_SAGEDP
#define FA3FP11_P1R_SAGEDP 0
#endif

// ## kda-pass1 candidate switches (development; 0 = the accepted code, byte for byte):
//   KDA_P1_REDUX: the psi amax as a balanced fmaxf tree per thread, one redux.sync.max.u32 per warp (the per-thread amax
//     is a non-negative, non-NaN float, so the unsigned bit order is the float order) and a balanced tree over the 8 warp
//     slots, instead of a 32-long dependent FMNMX chain, 5 shuffle rounds and an 8-long chain.  max is exact and the
//     NaN-ignoring fmaxf set maximum is order-free: bitwise.
//   KDA_P1_DVZI: the first dV gemm of every promote window runs with scale-d = 0 (the window's accumulator is not
//     re-zeroed after the promote and not rescaled at the window start).  Bitwise if the wgmma result with C = +0
//     equals its zero-init result (verified by the equivalence suite).
//   KDA_P1_LSEPRE: the host forms lse * log2e (one rounded fp32 multiply, the kernel's __fmul_rn) for the hd128 pass 1
//     once per call; the kernel reads it instead of 16 FMUL per thread per m-block step.  Bitwise.
//   KDA_P1_EARLY: software-pipelined S / dP issue: the S and dP gemms of m-block m+1 are issued at the END of step m
//     (after the dK retire), so their tensor time runs under the dK fold, the dV promote and the loop overhead instead
//     of being waited for at the top of step m+1.  1 = fold, then S + dP, then the promote; 2 = fold half, S, fold half,
//     dP; 3 = S, fold, dP.  Same gemms, operands, values and order of arithmetic: bitwise.
#ifndef KDA_P1_REDUX
#define KDA_P1_REDUX 0
#endif
#ifndef KDA_P1_EARLY
#define KDA_P1_EARLY 0
#endif
//   KDA_P1_HALFPSI (a NEW NUMERICS ARM, output-changing): psi per (tile, warpgroup half) = the amax of this
//     warpgroup's 64-key half / 448 behind a WG-local named barrier; no cross-warpgroup psi rendezvous.  Published as
//     psi[BH, nbm, nbn, 2]; pass 2 consumes it with KDA_P2_HALFPSI (fq8_dq_pass2_hd128.h).
#ifndef KDA_P1_HALFPSI
#define KDA_P1_HALFPSI 0
#endif
//   KDA_EXPB (output-changing, pass 0 + pass 1 together so the matched P stays identical): the 2^8 offset folded into the
//     per-row bias, b = fl(fl(lse * log2e) - 8), P^ = exp2(min(fl(s * dsc - b), 12)): one rounding instead of two and no
//     per-element FADD.  KDA_P1_PROMOTE: the dV promote window in m-blocks (8 = the accepted 512 rows).
#ifndef KDA_EXPB
#define KDA_EXPB 0
#endif
#ifndef KDA_P1_PROMOTE
#define KDA_P1_PROMOTE 8
#endif
//   KDA_P1_PP (needs KDA_P1_HALFPSI: no cross-warpgroup psi rendezvous): warpgroup ping-pong.  1 = a one-time start
//     offset (warpgroup 1 starts after warpgroup 0's first dS line); 2 = turn barriers (ids 2 / 3) around each
//     warpgroup's S + dP issue; 3 = turns around both gemm batches (S + dP, dV + dK; implies DVLATE).
//   KDA_P1_DVLATE: the dV gemm issued right before the dK gemm (the pre-H1a placement) instead of before the psi
//     reduction.  Bitwise.
#ifndef KDA_P1_PP
#define KDA_P1_PP 0
#endif
#ifndef KDA_P1_DVLATE
#define KDA_P1_DVLATE (KDA_P1_PP == 3)
#endif
static_assert(KDA_P1_PP == 0 || KDA_P1_HALFPSI, "KDA_P1_PP needs KDA_P1_HALFPSI");
// KDA_ABL (TIMING-ONLY ablations, outputs are garbage; never shipped): bit 0 no exp line (P = the scaled score),
// 1 no dS' transpose / stmatrix / TMA store, 2 no dK fold, 3 no dV promote, 4 psi = 1 (no amax, no psi barrier),
// 5 no dV window rescale, 6 no S / dP gemms, 7 no dV / dK gemms, 8 no dS line, 9 no P / dS' e4m3 conversions.
#ifndef KDA_ABL
#define KDA_ABL 0
#endif
// KDA_P1_TOPFILL (bitwise schedule): the dV window work moves into the bubble where the warpgroup waits for its own S
// gemm: 1 = the promote of a window that ended at the previous step AND this step's dV rescale run right after the
// S / dP issue (the last window still promotes at the last step); 2 = only the rescale moves.  Same values, same order.
#ifndef KDA_P1_TOPFILL
#define KDA_P1_TOPFILL 0
#endif
// KDA_P1_FOLDSPLIT (bitwise schedule): the second half of the dK fold of step m runs in step m+1's S-wait bubble (32
// fp32 temps cross the step; each element's folds stay in tile order).  KDA_P1_XPTOP (bitwise, needs DSDEFER): the dS'
// shuffles / stmatrix of step m run in step m+1's S-wait bubble (the 8 packed words cross the step).
// KDA_P1_UNROLL2 (bitwise): the unmasked m-block loop runs pairs of steps with compile-time Q / Qt stage indices.
#ifndef KDA_P1_FOLDSPLIT
#define KDA_P1_FOLDSPLIT 0
#endif
#ifndef KDA_P1_XPTOP
#define KDA_P1_XPTOP 0
#endif
#ifndef KDA_P1_UNROLL2
#define KDA_P1_UNROLL2 0
#endif
static_assert(!KDA_P1_XPTOP || KDA_P1_DSDEFER, "KDA_P1_XPTOP needs KDA_P1_DSDEFER");
#define KDA_A(b) ((KDA_ABL >> (b)) & 1)
#ifndef KDA_P1_DVZI
#define KDA_P1_DVZI 0
#endif
#ifndef KDA_P1_LSEPRE
#define KDA_P1_LSEPRE 0
#endif

#include "kda_prof.h"

namespace flash {

using namespace cute;

template <int Stages, int Stages_dO, int Stages_dOt, bool Is_causal_>
struct CollectiveMainloopBwdSm90Fp8R {
    using Element = cutlass::float_e4m3_t;
    // ## bf16-dP: SageBwd's single exemption. dP = dO V^T is computed in bf16 from the RAW
    // (pre-quantization) dO and V instead of their e4m3 codes; S, dV, dQ, dK, the delta and the dS' path are
    // untouched, so this is exactly matmul (5) of the seven attention
    // matmuls.  V and raw dO feed NOTHING else in this collective (dV reads the dO twin), so the fp8 tiles are
    // replaced rather than duplicated: +16 KB for V, +8 KB per dO stage.  At FA3FP11_P1R_SAGEDP=0 every line
    // below collapses to the certified head_dim-128 code.
#if FA3FP11_P1R_SAGEDP
    using ElementDp = cutlass::bfloat16_t;
#else
    using ElementDp = Element;
#endif
    using ElementAccum = float;
    using ArchTag = cutlass::arch::Sm90;
    using ClusterShape = Shape<_1, _1, _1>;

    static constexpr int kBlockM = 64;
    static constexpr int kBlockN = 128;
    static constexpr int kHeadDim = 128;
    static constexpr int kBlockMh = kBlockM;    // interface parity with the v10 collective (the launch header reads it)
    using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
    static constexpr int kQuantKeys = 64;       // one K/V scale group per warpgroup column half
    static constexpr int kQuantRowsQ = 128;     // one Q/dO scale block per two m-blocks
    static constexpr int kQShift = 1;
    static constexpr int kStages = Stages;
    static constexpr int kStages_dO = Stages_dO;
    static constexpr int kStages_qt = Stages;
    static constexpr int kStages_dOt = Stages_dOt;
    static_assert(kStages >= kStages_dO, "pass 1: the dO pipeline cannot be deeper than the Q pipeline");
    static constexpr bool Is_causal = Is_causal_;
    static constexpr bool Is_local = false;
    static constexpr bool Has_softcap = false;
    static constexpr bool Varlen = false;
    static constexpr bool DeltaPass0 = false;
    static constexpr int kAbl = 0;
    static constexpr bool Q_dO_same_stages = kStages == kStages_dO;
    // When the stage counts match, Q and dO SHARE one pipeline object and therefore one transaction count.
    // That is only sound while raw dO has Q's element size, which bf16-dP breaks (bf16 dO vs e4m3 Q).
    static_assert(!(FA3FP11_P1R_SAGEDP && Q_dO_same_stages),
                  "bf16-dP needs FA3FP11_HD128_DOST != FA3FP11_HD128_QST: a shared Q/dO pipeline would size the bf16 dO TMA with Q's byte count");
    static constexpr bool SdP_swapAB = true;
    static constexpr bool dKV_swapAB = false;
    static constexpr int NumMmaWarpGroups = 2;
    static constexpr int AtomLayoutNdKV = 2;
    static constexpr int NumMmaThreads = NumMmaWarpGroups * cutlass::NumThreadsPerWarpGroup;
    static constexpr int NumProducerThreads = cutlass::NumThreadsPerWarp;
    // FA3's bwd split for two MMA warpgroups: the producer warpgroup only issues TMA.
    static constexpr uint32_t LoadRegisterRequirement = 24;
    static constexpr uint32_t MmaRegisterRequirement = 240;

    using SeqlenInfo_t = flash::SeqlenInfoQK<Varlen, kBlockM>;
    using BlockMN_t = flash::BlockMN<SeqlenInfo_t, kBlockM, kBlockN, Is_causal, Is_local>;

    // S^T / dP^T: (M = kBlockN keys, N = kBlockM rows, K = d); the two warpgroups tile M (64 keys each).
    using TileShapeAtomSdP = Shape<Int<kBlockN>, Int<kBlockM>, Int<kHeadDim>>;
    using AtomLayoutSdP = Layout<Shape<Int<NumMmaWarpGroups>, _1, _1>>;
    using TiledMmaSdP = decltype(cute::make_tiled_mma(
        cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShapeAtomSdP, GMMA::Major::K, GMMA::Major::K>(),
        AtomLayoutSdP{}));
    // dP^T = V dO^T: same tile shape and atom layout, ElementDp operands (== TiledMmaSdP at SAGEDP=0).
    using TiledMmadP = decltype(cute::make_tiled_mma(
        cute::GMMA::ss_op_selector<ElementDp, ElementDp, ElementAccum, TileShapeAtomSdP, GMMA::Major::K, GMMA::Major::K>(),
        AtomLayoutSdP{}));
    // dV / dK: (M = kBlockN keys, N = d, K = kBlockM rows), RS (A from registers), the two warpgroups tile M.
    using TileShapeAtomdKV = Shape<Int<kBlockN>, Int<kHeadDim>, Int<kBlockM>>;
    using AtomLayoutdKV = Layout<Shape<Int<AtomLayoutNdKV>, _1, _1>>;
    using TiledMmadKV = decltype(cute::make_tiled_mma(
        cute::GMMA::rs_op_selector<Element, Element, ElementAccum, TileShapeAtomdKV, GMMA::Major::K, GMMA::Major::K>(),
        AtomLayoutdKV{}));
    static_assert(CUTE_STATIC_V(size(TiledMmaSdP{})) == NumMmaThreads && CUTE_STATIC_V(size(TiledMmadKV{})) == NumMmaThreads);

    // Smem layouts (K-major everywhere; fp8 wgmma reads no MN-major operand).
    using SmemLayoutAtomQdO = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
                                       Int<kBlockM>, Int<kHeadDim>>());
    using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQdO{}, make_shape(Int<kBlockM>{}, Int<kHeadDim>{}, Int<kStages>{})));
    using SmemLayoutAtomdOdp = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, ElementDp,
                                        Int<kBlockM>, Int<kHeadDim>>());
    using SmemLayoutdO = decltype(tile_to_shape(SmemLayoutAtomdOdp{}, make_shape(Int<kBlockM>{}, Int<kHeadDim>{}, Int<kStages_dO>{})));
    using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
                                     Int<kBlockN>, Int<kHeadDim>>());
    using SmemLayoutK = decltype(tile_to_shape(SmemLayoutAtomK{}, select<1, 2>(TileShape_MNK{})));
    using SmemLayoutAtomVdp = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, ElementDp,
                                       Int<kBlockN>, Int<kHeadDim>>());
    using SmemLayoutV = decltype(tile_to_shape(SmemLayoutAtomVdp{}, select<1, 2>(TileShape_MNK{})));
    using SmemLayoutAtomQdOt = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
                                        Int<kHeadDim>, Int<kBlockM>>());
    using SmemLayoutQt = decltype(tile_to_shape(SmemLayoutAtomQdOt{}, make_shape(Int<kHeadDim>{}, Int<kBlockM>{}, Int<kStages_qt>{})));
    using SmemLayoutdOt = decltype(tile_to_shape(SmemLayoutAtomQdOt{}, make_shape(Int<kHeadDim>{}, Int<kBlockM>{}, Int<kStages_dOt>{})));
    using SmemLayoutLSE = cute::Layout<cute::Shape<Int<kBlockM>, Int<kStages>>, cute::Stride<_1, Int<cute::round_up(kBlockM, 64)>>>;
    // swapAB: the (keys, rows, stage) view broadcasting the per-row statistic along keys (FA3's SmemLayoutLSEMma).
    using SmemLayoutLSEMma = cute::Layout<cute::Shape<Int<kBlockN>, Int<kBlockM>, Int<kStages>>, cute::Stride<_0, _1, Int<cute::round_up(kBlockM, 64)>>>;
    // The dS' natural HALF tile of one warpgroup: kBlockM rows x 64 keys, 64-byte rows (SW64), stored by its own TMA box.
    static constexpr int kDsHalfN = kBlockN / NumMmaWarpGroups;
    using SmemLayoutAtomDS = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
                                      Int<kBlockM>, Int<kDsHalfN>>());
    using SmemLayoutDS = decltype(tile_to_shape(SmemLayoutAtomDS{}, make_shape(Int<kBlockM>{}, Int<kDsHalfN>{})));
    static constexpr int kDsHalfElems = kBlockM * kDsHalfN;
    static_assert(cute::cosize_v<SmemLayoutDS> == kDsHalfElems, "P1-R: the dS' half tile must be dense");
    // dV persistent accumulator overlay: kBlockN x kHeadDim fp32, slot(i, thread) = i * NumMmaThreads + thread (conflict-free).
    static constexpr int kDvOverlayFloats = kBlockN * kHeadDim;

    using GmemTiledCopyQdO = cute::SM90_TMA_LOAD;
    using GmemTiledCopyKV = cute::SM90_TMA_LOAD;
    using ShapeQKV = cute::Shape<int32_t, int32_t, int32_t, int32_t>;  // (seqlen, d, head, batch)
    using StrideQKV = cute::Stride<int64_t, _1, int64_t, int64_t>;
    using ShapeLSE = cute::Shape<int32_t, int32_t, int32_t>;  // (seqlen, head, batch)
    using StrideLSE = cute::Stride<_1, int64_t, int64_t>;
    using ShapedQaccum = cute::Shape<int32_t, int32_t, int32_t>;
    using StridedQaccum = cute::Stride<_1, int64_t, int64_t>;

    using TMA_QdO = decltype(make_tma_copy_A_sm90(
        GmemTiledCopyQdO{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, StrideQKV{}),
        take<0, 2>(SmemLayoutQ{}), TileShape_MNK{}, ClusterShape{}));
    using TMA_K = decltype(make_tma_copy_B_sm90(
        GmemTiledCopyKV{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, StrideQKV{}),
        SmemLayoutK{}, TileShape_MNK{}, ClusterShape{}));
    using TMA_V = decltype(make_tma_copy_B_sm90(
        GmemTiledCopyKV{},
        make_tensor(make_gmem_ptr(static_cast<ElementDp const*>(nullptr)), ShapeQKV{}, StrideQKV{}),
        SmemLayoutV{}, TileShape_MNK{}, ClusterShape{}));
    using TMA_dOdp = decltype(make_tma_copy_A_sm90(
        GmemTiledCopyQdO{},
        make_tensor(make_gmem_ptr(static_cast<ElementDp const*>(nullptr)), ShapeQKV{}, StrideQKV{}),
        take<0, 2>(SmemLayoutdO{}), TileShape_MNK{}, ClusterShape{}));
    // The twins are [d, seq, head, batch] (seq contiguous, pi-ordered inside every 16-position window).
    using TileShape_TwinM = Shape<Int<kHeadDim>, _1, Int<kBlockM>>;
    using TMA_Qt = decltype(make_tma_copy_A_sm90(
        cute::SM90_TMA_LOAD{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, StrideQKV{}),
        take<0, 2>(SmemLayoutQt{}), TileShape_TwinM{}, ClusterShape{}));
    using TMA_dOt = decltype(make_tma_copy_A_sm90(
        cute::SM90_TMA_LOAD{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, StrideQKV{}),
        take<0, 2>(SmemLayoutdOt{}), TileShape_TwinM{}, ClusterShape{}));
    // The dS' tile store: [BH, nbm, nbn, 64, 128] gmem (tile-contiguous), one (64 x 64) box per warpgroup.
    using ShapeDS8 = cute::Shape<Int<kBlockM>, Int<kBlockN>, int32_t, int32_t, int32_t>;
    using StrideDS8 = cute::Stride<Int<kBlockN>, _1, int64_t, int64_t, int64_t>;
    using TMA_dS8 = decltype(make_tma_copy(
        cute::SM90_TMA_STORE{},
        make_tensor(make_gmem_ptr(static_cast<Element*>(nullptr)), ShapeDS8{}, StrideDS8{}),
        SmemLayoutDS{}));

    using MainloopPipeline = typename cutlass::PipelineTmaAsync<kStages>;
    using PipelineState = typename MainloopPipeline::PipelineState;
    using MainloopPipeline_dO = typename cutlass::PipelineTmaAsync<kStages_dO>;
    using PipelineState_dO = typename MainloopPipeline_dO::PipelineState;
    using MainloopPipelineQt = typename cutlass::PipelineTmaAsync<kStages_qt>;
    using PipelineStateQt = typename MainloopPipelineQt::PipelineState;
    using MainloopPipelineDOt = typename cutlass::PipelineTmaAsync<kStages_dOt>;
    using PipelineStateDOt = typename MainloopPipelineDOt::PipelineState;

    static constexpr uint32_t TmaTransactionBytesQ = static_cast<uint32_t>(size(take<0, 2>(SmemLayoutQ{})) * cutlass::sizeof_bits_v<Element> / 8);
    static constexpr uint32_t TmaTransactionBytesK = static_cast<uint32_t>(size(SmemLayoutK{}) * cutlass::sizeof_bits_v<Element> / 8);
    static constexpr uint32_t TmaTransactionBytesV = static_cast<uint32_t>(size(SmemLayoutV{}) * cutlass::sizeof_bits_v<ElementDp> / 8);
    // The dO pipeline carried Q's transaction count while both were e4m3; under SAGEDP raw dO is bf16 and needs its own.
    static constexpr uint32_t TmaTransactionBytesdO = static_cast<uint32_t>(size(take<0, 2>(SmemLayoutdO{})) * cutlass::sizeof_bits_v<ElementDp> / 8);
    static constexpr uint32_t TmaTransactionBytesLSE = static_cast<uint32_t>(size(select<0>(SmemLayoutLSE{})) * cutlass::sizeof_bits_v<ElementAccum> / 8);
    static constexpr uint32_t TmaTransactionBytesQt = static_cast<uint32_t>(size(take<0, 2>(SmemLayoutQt{})) * cutlass::sizeof_bits_v<Element> / 8);
    static constexpr uint32_t TmaTransactionBytesdOt = static_cast<uint32_t>(size(take<0, 2>(SmemLayoutdOt{})) * cutlass::sizeof_bits_v<Element> / 8);

    static constexpr size_t SmemAlignmentQKVdO = 128;

    // The epilogue unions its bf16 dK / dV staging (2 x 32 KB) over the FIRST 64 KB of this struct; the persistent dV
    // overlay (read back at the mainloop exit, before the epilogue) and the dS' half tiles sit past the 80-KB mark.
    struct TensorStorage : cute::aligned_struct<1024> {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>, SmemAlignmentQKVdO> smem_k;
        cute::array_aligned<ElementDp, cute::cosize_v<SmemLayoutV>, SmemAlignmentQKVdO> smem_v;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>, SmemAlignmentQKVdO> smem_q;
        cute::array_aligned<ElementDp, cute::cosize_v<SmemLayoutdO>, SmemAlignmentQKVdO> smem_do;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQt>, SmemAlignmentQKVdO> smem_qt;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutdOt>, SmemAlignmentQKVdO> smem_dot;
        cute::array_aligned<ElementAccum, kDvOverlayFloats, 128> smem_dvacc;
        cute::array_aligned<Element, NumMmaWarpGroups * (1 + FA3FP11_P1R_DSDEFER) * kDsHalfElems, 1024> smem_ds;   // SW64 half tiles per WG (x2 under DSDEFER); 1024-B aligned: the swizzle anchors to the address
        cute::array_aligned<ElementAccum, cute::cosize_v<SmemLayoutLSE>, 128> smem_lse;
        cute::array_aligned<ElementAccum, cute::cosize_v<SmemLayoutLSE>, 128> smem_dpsum;
        cute::array_aligned<ElementAccum, 16, 16> smem_psired;   // 8 warps x 2 tile parities
#ifdef FA3FP11_DBG
        cute::array_aligned<ElementAccum, NumMmaWarpGroups * 4 * 2 * kBlockM, 16> smem_dbg_rows;   // [wg][warp][sum|abs][row]
#endif
    };
    static_assert(offsetof(TensorStorage, smem_dvacc) >= 64 * 1024, "P1-R: the dV overlay must sit past the epilogue's 64-KB staging window");

    // Host side kernel arguments: byte-compatible with the v10 collective's Arguments (the launch header brace-initializes it).
    struct Arguments {
        Element const* const ptr_Q;
        ShapeQKV const shape_Q;
        StrideQKV const stride_Q;
        Element const* const ptr_K;
        ShapeQKV const shape_K;
        StrideQKV const stride_K;
        ElementDp const* const ptr_V;
        ShapeQKV const shape_V;
        StrideQKV const stride_V;
        ElementDp const* const ptr_dO;
        ShapeQKV const shape_dO;
        StrideQKV const stride_dO;
        Element const* const ptr_Qt;
        ShapeQKV const shape_Qt;
        StrideQKV const stride_Qt;
        Element const* const ptr_Kt;
        ShapeQKV const shape_Kt;
        StrideQKV const stride_Kt;
        Element const* const ptr_dOt;
        ShapeQKV const shape_dOt;
        StrideQKV const stride_dOt;
        float const* const ptr_q8s;
        float const* const ptr_k8s;
        float const* const ptr_v8s;
        float const* const ptr_do8s;
        int const nblk_q;
        int const nblk_k;
        ElementAccum* const ptr_dQaccum;
        ShapedQaccum const shape_dQaccum;
        StridedQaccum const stride_dQaccum;
        float const* const ptr_LSE_log2;
        ShapeLSE const shape_LSE;
        StrideLSE const stride_LSE_log2;
        float const* const ptr_dPsum;
        StrideLSE const stride_dPsum;
        float const softmax_scale;
        int const window_size_left, window_size_right, attention_chunk;
        float const softcap_val;
        int const num_batch;
        int* const dq_semaphore;
        int const* const cu_seqlens_q;
        int const* const cu_seqlens_k;
        int const* const seqused_q;
        int const* const seqused_k;
        Element* const ptr_dS8;
        float* const ptr_psi;
        int const nbm;
        int const nbn;
        float* const ptr_dltpart;
        float* const ptr_dbg_rows;
        float* const ptr_dbg_rowsabs;
        unsigned int* const ptr_dbg_cks;
    };

    struct Params {
        ShapeQKV const shape_Q;
        ShapeQKV const shape_K;
        ShapeQKV const shape_V;
        ShapeQKV const shape_dO;
        cutlass::FastDivmod qhead_per_khead_divmod;
        TMA_QdO tma_load_Q;
        TMA_dOdp tma_load_dO;
        TMA_K tma_load_K;
        TMA_V tma_load_V;
        TMA_Qt tma_load_Qt;
        TMA_dOt tma_load_dOt;
        float const* const ptr_q8s;
        float const* const ptr_k8s;
        float const* const ptr_v8s;
        float const* const ptr_do8s;
        int const nblk_q;
        int const nblk_k;
        float const* const ptr_LSE_log2;
        ShapeLSE const shape_LSE;
        StrideLSE const stride_LSE_log2;
        float const* const ptr_dPsum;
        StrideLSE const stride_dPsum;
        float const softmax_scale;
        int const window_size_left, window_size_right;
        cutlass::FastDivmod attention_chunk_divmod;
        int const num_batch;
        int const* const cu_seqlens_q;
        int const* const cu_seqlens_k;
        int const* const seqused_q;
        int const* const seqused_k;
        TMA_dS8 tma_store_dS8;
        float* const ptr_psi;
        int const nbm;
        int const nbn;
        float* const ptr_dbg_rows;
        float* const ptr_dbg_rowsabs;
        unsigned int* const ptr_dbg_cks;
    };

    static Params
    to_underlying_arguments(Arguments const& args) {
        Tensor mQ = make_tensor(make_gmem_ptr(args.ptr_Q), args.shape_Q, args.stride_Q);
        TMA_QdO tma_load_Q = make_tma_copy_A_sm90(GmemTiledCopyQdO{}, mQ, SmemLayoutQ{}(_, _, _0{}), TileShape_MNK{}, ClusterShape{});
        Tensor mdO = make_tensor(make_gmem_ptr(args.ptr_dO), args.shape_dO, args.stride_dO);
        TMA_dOdp tma_load_dO = make_tma_copy_A_sm90(GmemTiledCopyQdO{}, mdO, SmemLayoutdO{}(_, _, _0{}), TileShape_MNK{}, ClusterShape{});
        Tensor mK = make_tensor(make_gmem_ptr(args.ptr_K), args.shape_K, args.stride_K);
        TMA_K tma_load_K = make_tma_copy_B_sm90(GmemTiledCopyKV{}, mK, SmemLayoutK{}, TileShape_MNK{}, ClusterShape{});
        Tensor mV = make_tensor(make_gmem_ptr(args.ptr_V), args.shape_V, args.stride_V);
        TMA_V tma_load_V = make_tma_copy_B_sm90(GmemTiledCopyKV{}, mV, SmemLayoutV{}, TileShape_MNK{}, ClusterShape{});
        Tensor mQt = make_tensor(make_gmem_ptr(args.ptr_Qt), args.shape_Qt, args.stride_Qt);
        TMA_Qt tma_load_Qt = make_tma_copy_A_sm90(cute::SM90_TMA_LOAD{}, mQt, take<0, 2>(SmemLayoutQt{}), TileShape_TwinM{}, ClusterShape{});
        Tensor mdOt = make_tensor(make_gmem_ptr(args.ptr_dOt), args.shape_dOt, args.stride_dOt);
        TMA_dOt tma_load_dOt = make_tma_copy_A_sm90(cute::SM90_TMA_LOAD{}, mdOt, take<0, 2>(SmemLayoutdOt{}), TileShape_TwinM{}, ClusterShape{});
        Tensor mDS8 = make_tensor(make_gmem_ptr(args.ptr_dS8),
            make_shape(Int<kBlockM>{}, Int<kBlockN>{}, args.nbn, args.nbm, args.num_batch * get<2>(args.shape_Q)),
            make_stride(Int<kBlockN>{}, _1{}, int64_t(kBlockM) * kBlockN,
                        int64_t(kBlockM) * kBlockN * args.nbn,
                        int64_t(kBlockM) * kBlockN * args.nbn * args.nbm));
        TMA_dS8 tma_store_dS8 = make_tma_copy(cute::SM90_TMA_STORE{}, mDS8, SmemLayoutDS{});
        cutlass::FastDivmod attention_chunk_divmod(args.attention_chunk >= 1 ? args.attention_chunk : 1);
        attention_chunk_divmod.divisor = args.attention_chunk;
        return {args.shape_Q, args.shape_K, args.shape_V, args.shape_dO,
                cutlass::FastDivmod(cute::ceil_div(get<2>(args.shape_Q), get<2>(args.shape_K))),
                tma_load_Q, tma_load_dO, tma_load_K, tma_load_V, tma_load_Qt, tma_load_dOt,
                args.ptr_q8s, args.ptr_k8s, args.ptr_v8s, args.ptr_do8s, args.nblk_q, args.nblk_k,
                args.ptr_LSE_log2, args.shape_LSE, args.stride_LSE_log2, args.ptr_dPsum, args.stride_dPsum,
                args.softmax_scale, args.window_size_left, args.window_size_right, attention_chunk_divmod,
                args.num_batch, args.cu_seqlens_q, args.cu_seqlens_k, args.seqused_q, args.seqused_k,
                tma_store_dS8, args.ptr_psi, args.nbm, args.nbn,
                args.ptr_dbg_rows, args.ptr_dbg_rowsabs, args.ptr_dbg_cks};
    }

    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& params) {
        cute::prefetch_tma_descriptor(params.tma_load_Q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_dO.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_K.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_V.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_Qt.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_dOt.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_store_dS8.get_tma_descriptor());
    }

    KT_MEMBERS   // the producer's tile counter (timeline-only build)
    // Producer (the M17 GT issue order per m-block: dO(m), dOt(m), Q(m+1), Qt(m+1); K / V one-shot on barrier_KV).
    template <typename SchedulerPrefetch, typename SharedStorage>
    CUTLASS_DEVICE void
    load_gt(Params const& params,
            MainloopPipeline pipeline_q, MainloopPipeline_dO pipeline_do,
            MainloopPipelineQt pipeline_qt, MainloopPipelineDOt pipeline_dot,
            PipelineState& smem_pipe_write, PipelineState_dO& smem_pipe_write_do,
            PipelineStateQt& smem_pipe_write_qt, PipelineStateDOt& smem_pipe_write_dot,
            SharedStorage &shared_storage, SchedulerPrefetch const& scheduler_prefetch,
            cute::tuple<int32_t, int32_t, int32_t> block_coord) {
        auto [n_block, bidh, bidb] = block_coord;
        KTP_BEGIN(0);
        SeqlenInfo_t seqlen_info{bidb, get<0>(params.shape_Q), size<0>(params.shape_K),
                                 params.cu_seqlens_q, params.cu_seqlens_k, params.seqused_q, params.seqused_k};
        auto [m_block_min, m_block_max] = BlockMN_t::get_m_block_min_max(
            seqlen_info, n_block, bidb, params.window_size_left, params.window_size_right, 0 /*sink_token_length*/);
        if constexpr (Is_causal) {
            if (m_block_max <= m_block_min) { scheduler_prefetch(); return; }
        }
        Tensor sQ = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutQ{});
        Tensor sdO = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_do.data()), SmemLayoutdO{});
        Tensor sK = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutK{});
        Tensor sV = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutV{});
        Tensor sLSE = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_lse.data()), SmemLayoutLSE{});
        Tensor sdPsum = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dpsum.data()), SmemLayoutLSE{});
        Tensor sQt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_qt.data()), SmemLayoutQt{});
        Tensor sdOt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dot.data()), SmemLayoutdOt{});

        int bidh_kv = params.qhead_per_khead_divmod.divide(bidh);
        Tensor mQ = params.tma_load_Q.get_tma_tensor(params.shape_Q)(_, _, bidh, bidb);
        Tensor mdO = params.tma_load_dO.get_tma_tensor(params.shape_dO)(_, _, bidh, bidb);
        Tensor mK = params.tma_load_K.get_tma_tensor(params.shape_K)(_, _, bidh_kv, bidb);
        Tensor mV = params.tma_load_V.get_tma_tensor(params.shape_V)(_, _, bidh_kv, bidb);
        auto shape_Qt = make_shape(get<1>(params.shape_Q), get<0>(params.shape_Q), get<2>(params.shape_Q), get<3>(params.shape_Q));
        auto shape_dOt = make_shape(get<1>(params.shape_dO), get<0>(params.shape_dO), get<2>(params.shape_dO), get<3>(params.shape_dO));
        Tensor mQt = params.tma_load_Qt.get_tma_tensor(shape_Qt)(_, _, bidh, bidb);
        Tensor mdOt = params.tma_load_dOt.get_tma_tensor(shape_dOt)(_, _, bidh, bidb);
        Tensor mLSE = make_tensor(make_gmem_ptr(params.ptr_LSE_log2), params.shape_LSE, params.stride_LSE_log2)(_, bidh, bidb);
        Tensor mdPsum = make_tensor(make_gmem_ptr(params.ptr_dPsum), params.shape_LSE, params.stride_dPsum)(_, bidh, bidb);

        Tensor gQ = local_tile(domain_offset(make_coord(seqlen_info.offset_q, _0{}), mQ), select<0, 2>(TileShape_MNK{}), make_coord(_, _0{}));
        Tensor gdO = local_tile(domain_offset(make_coord(seqlen_info.offset_q, _0{}), mdO), select<0, 2>(TileShape_MNK{}), make_coord(_, _0{}));
        Tensor gK = local_tile(domain_offset(make_coord(seqlen_info.offset_k, _0{}), mK), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));
        Tensor gV = local_tile(domain_offset(make_coord(seqlen_info.offset_k, _0{}), mV), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));
        Tensor gQt = local_tile(mQt, make_shape(Int<kHeadDim>{}, Int<kBlockM>{}), make_coord(_0{}, _));
        Tensor gdOt = local_tile(mdOt, make_shape(Int<kHeadDim>{}, Int<kBlockM>{}), make_coord(_0{}, _));
        Tensor gLSE = local_tile(domain_offset(make_coord(seqlen_info.offset_q_padded), mLSE), select<0>(TileShape_MNK{}), make_coord(_));
        Tensor gdPsum = local_tile(domain_offset(make_coord(seqlen_info.offset_q_padded), mdPsum), select<0>(TileShape_MNK{}), make_coord(_));

        Tensor sK_x = make_tensor(sK.data(), make_layout(sK.layout(), Layout<_1>{}));
        Tensor gK_x = make_tensor(gK.data(), make_layout(gK.layout(), Layout<_1>{}));
        Tensor sV_x = make_tensor(sV.data(), make_layout(sV.layout(), Layout<_1>{}));
        Tensor gV_x = make_tensor(gV.data(), make_layout(gV.layout(), Layout<_1>{}));
        auto block_tma_Q = params.tma_load_Q.get_slice(_0{});
        auto block_tma_dO = params.tma_load_dO.get_slice(_0{});
        Tensor tQgQ = group_modes<0, 3>(block_tma_Q.partition_S(gQ));
        Tensor tQsQ = group_modes<0, 3>(block_tma_Q.partition_D(sQ));
        Tensor tdOgdO = group_modes<0, 3>(block_tma_dO.partition_S(gdO));
        Tensor tdOsdO = group_modes<0, 3>(block_tma_dO.partition_D(sdO));
        auto block_tma_Qt = params.tma_load_Qt.get_slice(_0{});
        auto block_tma_dOt = params.tma_load_dOt.get_slice(_0{});
        Tensor tQtgQt = group_modes<0, 3>(block_tma_Qt.partition_S(gQt));
        Tensor tQtsQt = group_modes<0, 3>(block_tma_Qt.partition_D(sQt));
        Tensor tdOtgdOt = group_modes<0, 3>(block_tma_dOt.partition_S(gdOt));
        Tensor tdOtsdOt = group_modes<0, 3>(block_tma_dOt.partition_D(sdOt));
        auto [tKgK, tKsK] = tma_partition(params.tma_load_K, _0{}, Layout<_1>{}, group_modes<0, 2>(sK_x), group_modes<0, 2>(gK_x));
        auto [tVgV, tVsV] = tma_partition(params.tma_load_V, _0{}, Layout<_1>{}, group_modes<0, 2>(sV_x), group_modes<0, 2>(gV_x));
        auto bulk_copy = Copy_Traits<SM90_BULK_COPY_AUTO>{};
        uint16_t const mcast_mask_qdo = 0;

        int m_block = m_block_min;
        int lane_predicate = cute::elect_one_sync();
        if (lane_predicate) {
            pipeline_q.producer_acquire(smem_pipe_write);
            copy(params.tma_load_Q.with(*pipeline_q.producer_get_barrier(smem_pipe_write), mcast_mask_qdo, TMA::CacheHintSm90::EVICT_LAST),
                 tQgQ(_, m_block), tQsQ(_, smem_pipe_write.index()));
            copy(bulk_copy.with(*pipeline_q.producer_get_barrier(smem_pipe_write)), gLSE(_, m_block), sLSE(_, smem_pipe_write.index()));
            pipeline_qt.producer_acquire(smem_pipe_write_qt);
            copy(params.tma_load_Qt.with(*pipeline_qt.producer_get_barrier(smem_pipe_write_qt), mcast_mask_qdo, TMA::CacheHintSm90::EVICT_LAST),
                 tQtgQt(_, m_block), tQtsQt(_, smem_pipe_write_qt.index()));
        }
        if (lane_predicate) {
            shared_storage.pipelines.barrier_KV.arrive_and_expect_tx(TmaTransactionBytesK + TmaTransactionBytesV);
            copy(params.tma_load_K.with(reinterpret_cast<cutlass::arch::ClusterTransactionBarrier::ValueType&>(shared_storage.pipelines.barrier_KV), 0), tKgK, tKsK);
            copy(params.tma_load_V.with(reinterpret_cast<cutlass::arch::ClusterTransactionBarrier::ValueType&>(shared_storage.pipelines.barrier_KV), 0), tVgV, tVsV);
            KTP_H(8);
            #pragma unroll 1
            for (; m_block < m_block_max - 1; ++m_block) {
                KTP_VAL(30, smem_pipe_write.index()); KTP_VAL(31, smem_pipe_write.phase()); KTP(1);
                PipelineState_dO smem_pipe_write_do_cur = cute::conditional_return<Q_dO_same_stages>(smem_pipe_write, smem_pipe_write_do);
                pipeline_do.producer_acquire(smem_pipe_write_do_cur);
                KTP(2);
                copy(params.tma_load_dO.with(*pipeline_do.producer_get_barrier(smem_pipe_write_do_cur), mcast_mask_qdo, TMA::CacheHintSm90::EVICT_LAST),
                     tdOgdO(_, m_block), tdOsdO(_, smem_pipe_write_do_cur.index()));
                copy(bulk_copy.with(*pipeline_do.producer_get_barrier(smem_pipe_write_do_cur)), gdPsum(_, m_block), sdPsum(_, smem_pipe_write_do_cur.index()));
                KTP(3);
                if constexpr (!Q_dO_same_stages) { ++smem_pipe_write_do; }
                pipeline_dot.producer_acquire(smem_pipe_write_dot);
                KTP(4);
                copy(params.tma_load_dOt.with(*pipeline_dot.producer_get_barrier(smem_pipe_write_dot), mcast_mask_qdo, TMA::CacheHintSm90::EVICT_LAST),
                     tdOtgdOt(_, m_block), tdOtsdOt(_, smem_pipe_write_dot.index()));
                KTP(5);
                ++smem_pipe_write_dot;
                ++smem_pipe_write;
                pipeline_q.producer_acquire(smem_pipe_write);
                KTP(7);
                copy(params.tma_load_Q.with(*pipeline_q.producer_get_barrier(smem_pipe_write), mcast_mask_qdo, TMA::CacheHintSm90::EVICT_LAST),
                     tQgQ(_, m_block + 1), tQsQ(_, smem_pipe_write.index()));
                copy(bulk_copy.with(*pipeline_q.producer_get_barrier(smem_pipe_write)), gLSE(_, m_block + 1), sLSE(_, smem_pipe_write.index()));
                KTP(8);
                ++smem_pipe_write_qt;
                pipeline_qt.producer_acquire(smem_pipe_write_qt);
                KTP(9);
                copy(params.tma_load_Qt.with(*pipeline_qt.producer_get_barrier(smem_pipe_write_qt), mcast_mask_qdo, TMA::CacheHintSm90::EVICT_LAST),
                     tQtgQt(_, m_block + 1), tQtsQt(_, smem_pipe_write_qt.index()));
                KTP(10); KTP_STEP();
            }
        }
        scheduler_prefetch();
        if (lane_predicate) {
            PipelineState_dO smem_pipe_write_do_cur = cute::conditional_return<Q_dO_same_stages>(smem_pipe_write, smem_pipe_write_do);
            pipeline_do.producer_acquire(smem_pipe_write_do_cur);
            copy(params.tma_load_dO.with(*pipeline_do.producer_get_barrier(smem_pipe_write_do_cur), mcast_mask_qdo, TMA::CacheHintSm90::EVICT_LAST),
                 tdOgdO(_, m_block), tdOsdO(_, smem_pipe_write_do_cur.index()));
            copy(bulk_copy.with(*pipeline_do.producer_get_barrier(smem_pipe_write_do_cur)), gdPsum(_, m_block), sdPsum(_, smem_pipe_write_do_cur.index()));
            if constexpr (!Q_dO_same_stages) { ++smem_pipe_write_do; }
            pipeline_dot.producer_acquire(smem_pipe_write_dot);
            copy(params.tma_load_dOt.with(*pipeline_dot.producer_get_barrier(smem_pipe_write_dot), mcast_mask_qdo, TMA::CacheHintSm90::EVICT_LAST),
                 tdOtgdOt(_, m_block), tdOtsdOt(_, smem_pipe_write_dot.index()));
            ++smem_pipe_write_dot;
            ++smem_pipe_write;
            ++smem_pipe_write_qt;
        }
        if constexpr (Q_dO_same_stages) { smem_pipe_write_do = smem_pipe_write; }
        KTP_END();
    }

    CUTLASS_DEVICE void
    load_tail_gt(MainloopPipeline pipeline_q, MainloopPipeline_dO pipeline_do,
                 MainloopPipelineQt pipeline_qt, MainloopPipelineDOt pipeline_dot,
                 PipelineState& smem_pipe_write, PipelineState_dO& smem_pipe_write_do,
                 PipelineStateQt& smem_pipe_write_qt, PipelineStateDOt& smem_pipe_write_dot) {
        if (cute::elect_one_sync()) {
            pipeline_q.producer_tail(smem_pipe_write);
            pipeline_do.producer_tail(smem_pipe_write_do);
            pipeline_qt.producer_tail(smem_pipe_write_qt);
            pipeline_dot.producer_tail(smem_pipe_write_dot);
        }
    }

    CUTLASS_DEVICE void mma_init() {}

    template <typename SharedStorage, typename FrgTensordKV>
    CUTLASS_DEVICE bool
    mma(Params const& params,
        MainloopPipeline pipeline_q, MainloopPipeline_dO pipeline_do,
        MainloopPipelineQt pipeline_qt, MainloopPipelineDOt pipeline_dot,
        PipelineState& smem_pipe_read, PipelineState_dO& smem_pipe_read_do,
        PipelineStateQt& smem_pipe_read_qt, PipelineStateDOt& smem_pipe_read_dot,
        FrgTensordKV& tdKrdK, FrgTensordKV& tdVrdV,
        int thread_idx, int &work_idx,
        cute::tuple<int32_t, int32_t, int32_t> block_coord, SharedStorage& shared_storage) {
        static_assert(is_rmem<FrgTensordKV>::value, "dK and dV tensor must be rmem resident.");
        int const n_block = get<0>(block_coord);
        int const bidh = get<1>(block_coord);
        int const bidb = get<2>(block_coord);
        SeqlenInfo_t seqlen_info{bidb, get<0>(params.shape_Q), size<0>(params.shape_K),
                                 params.cu_seqlens_q, params.cu_seqlens_k, params.seqused_q, params.seqused_k};
        auto [m_block_min, m_block_max] = BlockMN_t::get_m_block_min_max(
            seqlen_info, n_block, bidb, params.window_size_left, params.window_size_right, 0 /*sink_token_length*/);
        if constexpr (Is_causal) {
            if (m_block_max <= m_block_min) { return false; }
        }
        Tensor sQ = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutQ{});
        Tensor sdO = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_do.data()), SmemLayoutdO{});
        Tensor sK = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutK{});
        Tensor sV = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutV{});
        Tensor sQt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_qt.data()), SmemLayoutQt{});
        Tensor sdOt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dot.data()), SmemLayoutdOt{});
        Tensor sLSEMma = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_lse.data()), SmemLayoutLSEMma{});
        Tensor sdPsumMma = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dpsum.data()), SmemLayoutLSEMma{});

        int const warp_group_idx = __shfl_sync(0xFFFFFFFF, thread_idx / cutlass::NumThreadsPerWarpGroup, 0);
        int const wg_thread = thread_idx % cutlass::NumThreadsPerWarpGroup;
        int const lane = thread_idx % cutlass::NumThreadsPerWarp;
        int const warp_in_wg = wg_thread / cutlass::NumThreadsPerWarp;
        uint32_t const wg_barrier_id = static_cast<uint32_t>(BwdNamedBarriers::dQFullWG1) + warp_group_idx;   // ids 5, 6: WG-local
#if KDA_P1_PP >= 2
        uint32_t const fq8_turn_me = static_cast<uint32_t>(BwdNamedBarriers::dQEmptyWG1) + warp_group_idx;        // ids 2, 3
        uint32_t const fq8_turn_other = static_cast<uint32_t>(BwdNamedBarriers::dQEmptyWG1) + (1 - warp_group_idx);
        auto turn_sync = [&]() { cutlass::arch::NamedBarrier::sync(NumMmaThreads, fq8_turn_me); };
        auto turn_arrive = [&]() { cutlass::arch::NamedBarrier::arrive(NumMmaThreads, fq8_turn_other); };
#else
        auto turn_sync = [&]() {};
        auto turn_arrive = [&]() {};
#endif
        KP_BEGIN();
        KT_BEGIN(0);
        // This warpgroup's dS' natural half tile (kBlockM rows x 64 keys, SW64) and its gmem box.
        Tensor sdS = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_ds.data() + warp_group_idx * (1 + FA3FP11_P1R_DSDEFER) * kDsHalfElems), SmemLayoutDS{});
        Tensor sdS1 = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_ds.data() + (warp_group_idx * (1 + FA3FP11_P1R_DSDEFER) + FA3FP11_P1R_DSDEFER) * kDsHalfElems), SmemLayoutDS{});   // the second buffer (== sdS without DSDEFER)

        Layout warp_group_thread_layout = make_layout(make_shape(Int<NumMmaWarpGroups>{}), make_stride(Int<cutlass::NumThreadsPerWarpGroup>{}));
        TiledMmaSdP tiled_mma_SdP;
        TiledMmadP tiled_mma_dP;
        TiledMmadKV tiled_mma_dKV;
        auto wg_mma_SdP = tiled_mma_SdP.get_slice(warp_group_thread_layout(warp_group_idx));
        auto wg_mma_dP = tiled_mma_dP.get_slice(warp_group_thread_layout(warp_group_idx));
        auto thread_mma_SdP = tiled_mma_SdP.get_thread_slice(thread_idx);
        auto wg_mma_dKV = tiled_mma_dKV.get_slice(warp_group_thread_layout(warp_group_idx));

        Tensor tSrK = wg_mma_SdP.partition_fragment_A(sK);          // this WG's 64 keys (MMA, MMA_M, MMA_K)
        Tensor tSrQ = wg_mma_SdP.partition_fragment_B(sQ);          // (MMA, MMA_N, MMA_K, PIPE)
        Tensor tdPrV = wg_mma_dP.partition_fragment_A(sV);
        Tensor tdPrdO = wg_mma_dP.partition_fragment_B(sdO);
        Tensor tdVrdOt = wg_mma_dKV.partition_fragment_B(sdOt);     // (MMA, MMA_N, MMA_K, PIPE): (d, rows)
        Tensor tdKrQt = wg_mma_dKV.partition_fragment_B(sQt);
        // Per-column (query row) statistics of the swapAB fragment: (2, V, MMA_N, PIPE) -> (16, PIPE); read from smem at use.
        Tensor tLSEsLSE = group_modes<0, 3>(thread_mma_SdP.partition_C(sLSEMma)(make_coord(_, _0{}, _), _0{}, _, _));
        Tensor tLSEsdPsum = group_modes<0, 3>(thread_mma_SdP.partition_C(sdPsumMma)(make_coord(_, _0{}, _), _0{}, _, _));
        static_assert(decltype(size<0>(tLSEsLSE))::value == 16, "P1-R: 16 query rows per thread in the swapAB fragment");

        auto consumer_wait = [](auto& pipeline, auto& smem_pipe_read) {
            auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
            pipeline.consumer_wait(smem_pipe_read, barrier_token);
        };
        int const seqlen_q = seqlen_info.seqlen_q;
        int const seqlen_k = seqlen_info.seqlen_k;
        flash::Mask<kBlockM, kBlockN, false /*PackGQA*/, TiledMmaSdP, SdP_swapAB> mask(
            thread_idx, seqlen_q, seqlen_k, params.window_size_left, params.window_size_right, 0 /*sink_token_length*/,
            params.attention_chunk_divmod, params.qhead_per_khead_divmod);

        int m_block = m_block_min;
        clear(tdKrdK);
        clear(tdVrdV);
        static_assert(decltype(size(tdVrdV))::value == 64 && decltype(size(tdKrdK))::value == 64, "P1-R: 64-value dKV fragments per thread");
        float* const fq8_smem_dv = shared_storage.tensors.mainloop.smem_dvacc.data() + thread_idx;   // slot(i) = i * NumMmaThreads + thread
#if KDA_P1_PV4
        float4* const fq8_smem_dv4 = reinterpret_cast<float4*>(shared_storage.tensors.mainloop.smem_dvacc.data()) + thread_idx;
        #pragma unroll
        for (int j = 0; j < size(tdVrdV) / 4; ++j) { fq8_smem_dv4[j * NumMmaThreads] = make_float4(0.f, 0.f, 0.f, 0.f); }
        auto dv_promote4 = [&](float f) {
            #pragma unroll
            for (int j = 0; j < size(tdVrdV) / 4; ++j) {
                float4 v = fq8_smem_dv4[j * NumMmaThreads];
                v.x = fmaf(tdVrdV(4 * j), f, v.x); v.y = fmaf(tdVrdV(4 * j + 1), f, v.y);
                v.z = fmaf(tdVrdV(4 * j + 2), f, v.z); v.w = fmaf(tdVrdV(4 * j + 3), f, v.w);
                fq8_smem_dv4[j * NumMmaThreads] = v;
            }
#if !KDA_P1_DVZI
            #pragma unroll
            for (int i = 0; i < size(tdVrdV); ++i) { tdVrdV(i) = 0.0f; }
#endif
        };
#else
        #pragma unroll
        for (int i = 0; i < size(tdVrdV); ++i) { fq8_smem_dv[i * NumMmaThreads] = 0.0f; }
#endif

        cutlass::ConsumerToken barrier_token = static_cast<cutlass::BarrierStatus>(shared_storage.pipelines.barrier_KV.try_wait(work_idx % 2));
        if (barrier_token == cutlass::BarrierStatus::WaitAgain) { shared_storage.pipelines.barrier_KV.wait(work_idx % 2); }

        // Scales: this WG's 64-key group (k8s / v8s), the Q / dO scale per 128 rows (m-block >> 1).
        int const fq8_bh = bidb * get<2>(params.shape_Q) + bidh;
        int const fq8_bh_kv = bidb * get<2>(params.shape_K) + params.qhead_per_khead_divmod.divide(bidh);
        float const fq8_k8s_wg = params.ptr_k8s[fq8_bh_kv * params.nblk_k + 2 * n_block + warp_group_idx];
        [[maybe_unused]] float const fq8_v8s_wg = params.ptr_v8s[fq8_bh_kv * params.nblk_k + 2 * n_block + warp_group_idx];   // dP descale; unread under bf16-dP
        float const* fq8_q8s_base = params.ptr_q8s + fq8_bh * params.nblk_q;
        float const* fq8_do8s_base = params.ptr_do8s + fq8_bh * params.nblk_q;
        auto fq8_qblk = [](int m) { return m >> kQShift; };
        float fq8_q8sc_next = fq8_q8s_base[fq8_qblk(m_block_min)];
        float fq8_do8sc_next = fq8_do8s_base[fq8_qblk(m_block_min)];
        float fq8_f_prev = 1.0f;     // dV window basis (2^-8 * do8sc(i))
        float fq8_g_max = 0.0f;      // row-global max of the dK basis (the F2b emission floor anchor)
        float* fq8_psired = shared_storage.tensors.mainloop.smem_psired.data();
        static constexpr float kFq8MoE = 8.0f;
        static constexpr float kFq8MoInv = 1.0f / 256.0f;
        constexpr int kFq8PromoteTiles = KDA_P1_PROMOTE;   // the 512-row dV promote window (pinned at 8, as P1-A)
        auto fq8_clamp_ratio = [](float r) { return fminf(fmaxf(r, 1e-8f), 1e8f); };

        // The dS' natural transpose bookkeeping (per lane, per CTA): lane holds key pair ip = lane / 4 of its warp's 16
        // keys; the two source lanes hold keys 2 ip (srcA) and 2 ip + 1 (srcB) in slot h = ip / 4.
        int const fq8_ip = lane >> 2;
        int const fq8_srcA = 8 * (fq8_ip & 3) + (lane & 3);
        int const fq8_srcB = fq8_srcA + 4;
        uint32_t const fq8_sel = fq8_ip < 4 ? 0x5140u : 0x7362u;
        // stmatrix.x4.trans row addresses: instruction s covers rows 32 s + lane of this warp's 16-key chunk.
        uint32_t const fq8_st_addr0 = cute::cast_smem_ptr_to_uint(&sdS(lane, 16 * warp_in_wg));
        uint32_t const fq8_st_addr1 = cute::cast_smem_ptr_to_uint(&sdS(32 + lane, 16 * warp_in_wg));
        uint32_t const fq8_st_addr0b = cute::cast_smem_ptr_to_uint(&sdS1(lane, 16 * warp_in_wg));
        uint32_t const fq8_st_addr1b = cute::cast_smem_ptr_to_uint(&sdS1(32 + lane, 16 * warp_in_wg));
        Tensor mdS8 = params.tma_store_dS8.get_tma_tensor(
            make_shape(Int<kBlockM>{}, Int<kBlockN>{}, params.nbn, params.nbm, params.num_batch * get<2>(params.shape_Q)));
        auto blk_tma_dS8 = params.tma_store_dS8.get_slice(_0{});
        Tensor tdS8s = blk_tma_dS8.partition_S(sdS);
        Tensor tdS8s1 = blk_tma_dS8.partition_S(sdS1);
        // The WG's box store of tile m (buffer m & 1 under DSDEFER; the single buffer otherwise), issued by the elected thread.
        auto fq8_issue_ds_store = [&](int m) {
            Tensor gdS8 = mdS8(_, _, n_block, m, fq8_bh);
            Tensor gdS8h = local_tile(gdS8, Shape<Int<kBlockM>, Int<kDsHalfN>>{}, make_coord(_0{}, warp_group_idx));
            Tensor tdS8g = blk_tma_dS8.partition_D(gdS8h);
            if (FA3FP11_P1R_DSDEFER && (m & 1)) { cute::copy(params.tma_store_dS8, tdS8s1, tdS8g); }
            else { cute::copy(params.tma_store_dS8, tdS8s, tdS8g); }
            cute::tma_store_arrive();
        };

#if KDA_P1_EARLY
        // The S / dP accumulators live across steps: step m issues m+1's gemms after its dK retire.
        Tensor tSrS = partition_fragment_C(tiled_mma_SdP, select<1, 0>(TileShape_MNK{}));
        Tensor tdPrdP = partition_fragment_C(tiled_mma_dP, select<1, 0>(TileShape_MNK{}));
        auto issue_S = [&]() {
            // The zero-init gemm reads no old value, but its "+f" operands would keep the previous step's S live
            // across the loop: redefine the fragment first (no instruction).
            #pragma unroll
            for (int i = 0; i < size(tSrS); ++i) { asm volatile("" : "=f"(tSrS(i))); }
            consumer_wait(pipeline_q, smem_pipe_read);
            flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma_SdP, tSrK, tSrQ(_, _, _, smem_pipe_read.index()), tSrS);
        };
        auto issue_dP = [&]() {
            PipelineState_dO rd = cute::conditional_return<Q_dO_same_stages>(smem_pipe_read, smem_pipe_read_do);
            #pragma unroll
            for (int i = 0; i < size(tdPrdP); ++i) { asm volatile("" : "=f"(tdPrdP(i))); }
            consumer_wait(pipeline_do, rd);
            flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma_dP, tdPrV, tdPrdO(_, _, _, rd.index()), tdPrdP);
        };
#endif
        bool fq8_promote_pending = false;
#if KDA_P1_FOLDSPLIT
        float fq8_khi[32];                 // the deferred upper half of the previous step's dK temp
        float fq8_khi_g = 0.0f;
        bool fq8_khi_pending = false;
        auto fold_hi = [&]() {
            #pragma unroll
            for (int i = 0; i < 32; ++i) { tdKrdK(32 + i) = fmaf(fq8_khi[i], fq8_khi_g, tdKrdK(32 + i)); }
            fq8_khi_pending = false;
        };
#endif
#if KDA_P1_XPTOP
        uint32_t fq8_xp[8];                // the previous step's packed dS' words, transposed in this step's bubble
        bool fq8_xp_pending = false;
        int fq8_xp_m = 0;
        auto xpose_store = [&](uint32_t (&pk)[8], int m) {
            #pragma unroll
            for (int w = 0; w < 8; ++w) {
                uint32_t const a = __shfl_sync(0xffffffff, pk[w], fq8_srcA);
                uint32_t const b = __shfl_sync(0xffffffff, pk[w], fq8_srcB);
                pk[w] = __byte_perm(a, b, fq8_sel);
            }
            uint32_t const a0 = (m & 1) ? fq8_st_addr0b : fq8_st_addr0;
            uint32_t const a1 = (m & 1) ? fq8_st_addr1b : fq8_st_addr1;
            asm volatile("stmatrix.sync.aligned.x4.trans.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n"
                         :: "r"(a0), "r"(pk[0]), "r"(pk[1]), "r"(pk[2]), "r"(pk[3]));
            asm volatile("stmatrix.sync.aligned.x4.trans.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n"
                         :: "r"(a1), "r"(pk[4]), "r"(pk[5]), "r"(pk[6]), "r"(pk[7]));
            cutlass::arch::fence_view_async_shared();
        };
#endif
        auto dv_promote = [&]() {
#if KDA_P1_PV4
            dv_promote4(fq8_f_prev); return;
#endif
            #pragma unroll
            for (int i = 0; i < size(tdVrdV); ++i) {
                float* const slot = fq8_smem_dv + i * NumMmaThreads;
                *slot = fmaf(tdVrdV(i), fq8_f_prev, *slot);
#if !KDA_P1_DVZI
                tdVrdV(i) = 0.0f;
#endif
            }
        };
        auto bwd_step = [&](int m_block, auto mask_fn, auto stage_c) {
            constexpr int kSt = decltype(stage_c)::value;   // compile-time Q / Qt stage under UNROLL2 (-1 = runtime)
            Tensor fq8_tmpK = make_fragment_like(tdKrdK);
            float fq8_g_fold = 0.0f;
#if !KDA_P1_EARLY
            // (kBlockN, kBlockM) = the swapAB fragment: rows = keys, columns = query rows.
            Tensor tSrS = partition_fragment_C(tiled_mma_SdP, select<1, 0>(TileShape_MNK{}));
            KT_VAL(30, smem_pipe_read.index()); KT_VAL(31, smem_pipe_read.phase()); KT(0);
            consumer_wait(pipeline_q, smem_pipe_read);
            turn_sync();
            KP(3);
            if constexpr (KDA_A(6)) { clear(tSrS); warpgroup_arrive(); warpgroup_commit_batch(); }
            else if constexpr (kSt >= 0) {
                flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma_SdP, tSrK, tSrQ(_, _, _, Int<kSt>{}), tSrS);
            } else {
                flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma_SdP, tSrK, tSrQ(_, _, _, smem_pipe_read.index()), tSrS);
            }
#endif
            float const fq8_q8sc = fq8_q8sc_next;
            float const fq8_do8sc = fq8_do8sc_next;
            if (m_block + 1 < m_block_max) {
                fq8_q8sc_next = fq8_q8s_base[fq8_qblk(m_block + 1)];
                fq8_do8sc_next = fq8_do8s_base[fq8_qblk(m_block + 1)];
            }
            float const fq8_s_descale_log2 = fq8_q8sc * fq8_k8s_wg * float(M_LOG2E);
            // dP accumulator -> TRUE dP units.  With e4m3 operands that is the product of their code scales;
            // under bf16-dP the operands are the raw bf16 tensors, so the accumulator is already true dP and the
            // only surviving factor is pass 1's k8s fold (dS' = 2^8 * P * (dP_true - delta) * k8s_wg).
#if FA3FP11_P1R_SAGEDP
            float const fq8_dp_descale = 1.0f;
#else
            float const fq8_dp_descale = fq8_do8sc * fq8_v8s_wg;
#endif
            float const fq8_dp_descale_f = fq8_dp_descale * fq8_k8s_wg;
#if !KDA_P1_EARLY
            Tensor tdPrdP = partition_fragment_C(tiled_mma_dP, select<1, 0>(TileShape_MNK{}));
#endif
            // dS rides on tdPrdP under tSrS's rowcol layout below, so the two accumulators must agree.
            static_assert(cute::is_same_v<decltype(partition_fragment_C(TiledMmadP{}, select<1, 0>(TileShape_MNK{}))),
                                          decltype(partition_fragment_C(TiledMmaSdP{}, select<1, 0>(TileShape_MNK{})))>,
                          "bf16-dP: the bf16 dP accumulator fragment must match the fp8 S one");
            PipelineState_dO smem_pipe_read_do_cur = cute::conditional_return<Q_dO_same_stages>(smem_pipe_read, smem_pipe_read_do);
#if !KDA_P1_EARLY
            KT(21);
            consumer_wait(pipeline_do, smem_pipe_read_do_cur);
            KT(22);
            if constexpr (KDA_A(6)) { clear(tdPrdP); warpgroup_arrive(); warpgroup_commit_batch(); }
            else if constexpr (kSt >= 0) {
                flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma_dP, tdPrV, tdPrdO(_, _, _, Int<0>{}), tdPrdP);
            } else {
                flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma_dP, tdPrV, tdPrdO(_, _, _, smem_pipe_read_do_cur.index()), tdPrdP);
            }
            turn_arrive();
#endif
            KP(4);
#if KDA_P1_XPTOP
            if (fq8_xp_pending) { xpose_store(fq8_xp, fq8_xp_m); fq8_xp_pending = false; }
#endif
#if KDA_P1_FOLDSPLIT
            if (fq8_khi_pending) { fold_hi(); }
#endif
#if KDA_P1_TOPFILL
            // In the S / dP gemm bubble: the deferred promote (previous window), then this step's window rescale.
#if KDA_P1_TOPFILL == 1
            if (fq8_promote_pending) { dv_promote(); fq8_promote_pending = false; }
#endif
            float const fq8_f_cur_t = kFq8MoInv * fq8_do8sc;
            float const fq8_rf_t = fq8_clamp_ratio(fq8_f_prev / fq8_f_cur_t);
#if KDA_P1_DVZI
            bool const fq8_dv_fresh_t = m_block == m_block_min || (m_block & (kFq8PromoteTiles - 1)) == 0;
            if (!fq8_dv_fresh_t && fq8_rf_t != 1.0f) {
#else
            if (fq8_rf_t != 1.0f) {
#endif
            #pragma unroll
            for (int i = 0; i < size(tdVrdV); ++i) { tdVrdV(i) *= fq8_rf_t; }
            }
#endif
            warpgroup_wait<1>();   // S retired
            KP(5);
            auto const stage_q = [&] { if constexpr (kSt >= 0) { return Int<kSt>{}; } else { return smem_pipe_read.index(); } }();
            // rowcol view: (16 query rows, 2 keys) per thread.
            Tensor scores = make_tensor(tSrS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SdP_swapAB>(tSrS.layout()));
            mask_fn(tSrS, m_block);
            #pragma unroll
            for (int mi = 0; mi < size<0>(scores); ++mi) {
#if KDA_P1_LSEPRE
                float const lse_scaled = tLSEsLSE(mi, stage_q);   // the host-formed __fmul_rn(lse, log2e) (- 8 under EXPB)
#elif KDA_EXPB
                float const lse_scaled = __fadd_rn(__fmul_rn(tLSEsLSE(mi, stage_q), float(M_LOG2E)), -kFq8MoE);
#else
                float const lse_scaled = __fmul_rn(tLSEsLSE(mi, stage_q), float(M_LOG2E));   // LSEK: the host prep's exact multiply
#endif
                #pragma unroll
                for (int ni = 0; ni < size<1>(scores); ++ni) {
                    // P-hat = 2^8 * exp(S - lse) with the F1 cliff clamp (P1-A's line, verbatim).
#if KDA_A(0)
                    scores(mi, ni) = scores(mi, ni) * fq8_s_descale_log2 - lse_scaled;
#elif KDA_EXPB
                    scores(mi, ni) = exp2f(fminf(scores(mi, ni) * fq8_s_descale_log2 - lse_scaled, kFq8MoE + 4.0f));
#else
                    scores(mi, ni) = exp2f(fminf(scores(mi, ni) * fq8_s_descale_log2 - lse_scaled + kFq8MoE, kFq8MoE + 4.0f));
#endif
                }
            }
            // The stage's LSE rows were read by the loop above and the S gemm has retired: raw Q's readers are done.
            pipeline_q.consumer_release(smem_pipe_read);
            KP(6);
            warpgroup_wait<0>();   // dP retired
            KP(7);
            auto const stage_do = [&] { if constexpr (kSt >= 0) { return Int<0>{}; } else { return smem_pipe_read_do_cur.index(); } }();
            Tensor dS = make_tensor(tdPrdP.data(), scores.layout());
#ifdef FA3FP11_DBG
            if (params.ptr_dbg_cks != nullptr) {
                unsigned int fq8_xp = 0u, fq8_xd = 0u;
                #pragma unroll
                for (int i = 0; i < size(tSrS); ++i) { fq8_xp ^= __float_as_uint(tSrS(i)); }
                #pragma unroll
                for (int i = 0; i < size(tdPrdP); ++i) { fq8_xd ^= __float_as_uint(tdPrdP(i)); }
                #pragma unroll
                for (int off = 16; off > 0; off /= 2) {
                    fq8_xp ^= __shfl_xor_sync(0xffffffff, fq8_xp, off);
                    fq8_xd ^= __shfl_xor_sync(0xffffffff, fq8_xd, off);
                }
                if (lane == 0) {
                    int64_t const fq8_nt = int64_t(params.num_batch) * get<2>(params.shape_Q) * params.nbm * params.nbn;
                    int64_t const fq8_ti = (int64_t(fq8_bh) * params.nbm + m_block) * params.nbn + n_block;
                    atomicXor(params.ptr_dbg_cks + fq8_ti, fq8_xp);
                    atomicXor(params.ptr_dbg_cks + fq8_nt + fq8_ti, fq8_xd);
                }
            }
            if (params.ptr_dbg_rows != nullptr) {
                // Per-(tile, WG, query row) sums of the realized dS in TRUE units: 2 keys per thread, then the 8 lanes of
                // a warp that share the column (lane bits 2..4), then the 4 warps in a fixed order through smem.
                float* dbg_smem = shared_storage.tensors.mainloop.smem_dbg_rows.data() + warp_group_idx * (4 * 2 * kBlockM);
                Tensor tScS = thread_mma_SdP.partition_C(cute::make_identity_tensor(select<1, 0>(TileShape_MNK{})));
                Tensor tScS_rc = make_tensor(tScS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SdP_swapAB>(tScS.layout()));
                #pragma unroll
                for (int mi = 0; mi < size<0>(dS); ++mi) {
                    float const dsum = tLSEsdPsum(mi, stage_do);
                    float ra = 0.f, rb = 0.f;
                    #pragma unroll
                    for (int ni = 0; ni < size<1>(dS); ++ni) {
                        float const t = scores(mi, ni) * (dS(mi, ni) * fq8_dp_descale - dsum);
                        ra += t; rb += fabsf(t);
                    }
                    #pragma unroll
                    for (int off = 4; off < 32; off *= 2) {
                        ra += __shfl_xor_sync(0xffffffff, ra, off);
                        rb += __shfl_xor_sync(0xffffffff, rb, off);
                    }
                    if ((lane >> 2) == 0) {
                        int const row = get<1>(tScS_rc(mi, _0{}));   // the query row of this column
                        dbg_smem[(warp_in_wg * 2 + 0) * kBlockM + row] = ra;
                        dbg_smem[(warp_in_wg * 2 + 1) * kBlockM + row] = rb;
                    }
                }
                cutlass::arch::NamedBarrier::sync(cutlass::NumThreadsPerWarpGroup, wg_barrier_id);
                if (wg_thread < kBlockM) {
                    float sa = 0.f, sb = 0.f;
                    #pragma unroll
                    for (int w = 0; w < 4; ++w) {
                        sa += dbg_smem[(w * 2 + 0) * kBlockM + wg_thread];
                        sb += dbg_smem[(w * 2 + 1) * kBlockM + wg_thread];
                    }
                    int64_t const gi = ((int64_t(fq8_bh) * params.nbn + n_block) * 2 + warp_group_idx) * int64_t(seqlen_q) + m_block * kBlockM + wg_thread;
                    params.ptr_dbg_rows[gi] = sa;
                    params.ptr_dbg_rowsabs[gi] = sb;
                }
                cutlass::arch::NamedBarrier::sync(cutlass::NumThreadsPerWarpGroup, wg_barrier_id);
            }
#endif
            #pragma unroll
            for (int mi = 0; mi < size<0>(dS); ++mi) {
                float const dP_sum_cur_f = tLSEsdPsum(mi, stage_do) * fq8_k8s_wg;
                #pragma unroll
                for (int ni = 0; ni < size<1>(dS); ++ni) {
                    // dS' = 2^8 * P * (dP_true - delta) * k8s_wg (the k8s fold; P1-A's line, verbatim).
#if KDA_A(8)
                    dS(mi, ni) = dS(mi, ni) + dP_sum_cur_f;
#else
                    dS(mi, ni) = scores(mi, ni) * (dS(mi, ni) * fq8_dp_descale_f - dP_sum_cur_f);
#endif
                }
            }
            // The stage's dPsum rows were read above and the dP gemm has retired: raw dO's readers are done.
            pipeline_do.consumer_release(smem_pipe_read_do_cur);
#if KDA_P1_PP == 1
            if (warp_group_idx == 0 && m_block == m_block_min) {
                cutlass::arch::NamedBarrier::arrive(NumMmaThreads, static_cast<uint32_t>(BwdNamedBarriers::dQEmptyWG1));
            }
#endif
            KP(8);
            // ## r12 STACK, part H1a (M24 ported onto the H1b source): the dS line above was the last reader of the fp32 P
            // fragment, and dV = P^T dO does not depend on psi, so the P~ codes, the dV window rescale, the dOt wait and
            // the dV gemm are issued here, before the cross-warpgroup psi barrier: the barrier wait, the dS' conversion,
            // the H1b pre-pack, the dK issue and the transpose / TMA store then run while dV is in flight.  Same
            // instructions, values and accumulation order (schedule only); FIFO retire unchanged (dV, then dK).
            // P~ codes as the RS A operand of dV (register renaming + the FA3 Aregs view; no smem).
            flash::permute_Cregs_fp8(tSrS);
            Tensor tdVrP_acc = make_tensor(tSrS.data(), flash::convert_layout_acc_Aregs<TiledMmadKV>(tSrS.layout()));
            Tensor rP = make_tensor_like<Element>(tdVrP_acc);
            if constexpr (KDA_A(9)) { Tensor r32 = recast<uint32_t>(rP); for (int i = 0; i < size(r32); ++i) { r32(i) = __float_as_uint(tdVrP_acc(4 * i)); } }
            else { flash::convert_type_out(tdVrP_acc, rP); }
            // dV window rescale (v10 / P1-A verbatim): the previous tile's dV gemm retired at the end of its step.
            float const fq8_f_cur = kFq8MoInv * fq8_do8sc;
            float const fq8_rf = fq8_clamp_ratio(fq8_f_prev / fq8_f_cur);
#if KDA_P1_DVZI
            // The first tile of a promote window starts from scale-d = 0 (the accumulator was promoted, not re-zeroed).
            bool const fq8_dv_fresh = m_block == m_block_min || (m_block & (kFq8PromoteTiles - 1)) == 0;
            if (!KDA_P1_TOPFILL && !fq8_dv_fresh && fq8_rf != 1.0f) {
#else
            if (!KDA_P1_TOPFILL && !KDA_A(5) && fq8_rf != 1.0f) {   // WG-uniform: the ratio is exactly 1.0f on the second m-block of every 128-row dO scale block
#endif
            #pragma unroll
            for (int i = 0; i < size(tdVrdV); ++i) { tdVrdV(i) *= fq8_rf; }
            }
            fq8_f_prev = fq8_f_cur;
            // dV += P^T dO (RS): issued now, retired first (FIFO) after the dK gemm below.
            auto issue_dV = [&]() {
            KT(23);
            consumer_wait(pipeline_dot, smem_pipe_read_dot);
            KT(24);
#if KDA_P1_DVZI
            tiled_mma_dKV.accumulate_ = fq8_dv_fresh ? GMMA::ScaleOut::Zero : GMMA::ScaleOut::One;
#endif
            if constexpr (KDA_A(7)) { warpgroup_arrive(); warpgroup_commit_batch(); }
            else if constexpr (kSt >= 0) {
                flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiled_mma_dKV, rP, tdVrdOt(_, _, _, Int<0>{}), tdVrdV);
            } else {
                flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiled_mma_dKV, rP, tdVrdOt(_, _, _, smem_pipe_read_dot.index()), tdVrdV);
            }
            };
#if !KDA_P1_DVLATE
            issue_dV();
#endif
            // psi = amax(|dS_tile|) / 448 across BOTH warpgroups (the tile is column-split per WG); parity-buffered slots.
#if KDA_P1_REDUX == 2   // the per-thread chain as shipped, one redux.sync per warp
            float fq8_amax = 0.f;
            #pragma unroll
            for (int i = 0; i < size(tdPrdP); ++i) { fq8_amax = fmaxf(fq8_amax, fabsf(tdPrdP(i))); }
            fq8_amax = __uint_as_float(__reduce_max_sync(0xffffffffu, __float_as_uint(fq8_amax)));
#elif KDA_P1_REDUX == 3   // a per-thread tree, the 5 shuffle rounds as shipped
            float fq8_amax;
            {
                float t[16];
                #pragma unroll
                for (int i = 0; i < 16; ++i) { t[i] = fmaxf(fabsf(tdPrdP(2 * i)), fabsf(tdPrdP(2 * i + 1))); }
                #pragma unroll
                for (int i = 0; i < 8; ++i) { t[i] = fmaxf(t[i], t[i + 8]); }
                #pragma unroll
                for (int i = 0; i < 4; ++i) { t[i] = fmaxf(t[i], t[i + 4]); }
                t[0] = fmaxf(t[0], t[2]); t[1] = fmaxf(t[1], t[3]);
                fq8_amax = fmaxf(0.f, fmaxf(t[0], t[1]));
            }
            #pragma unroll
            for (int off = 16; off > 0; off /= 2) { fq8_amax = fmaxf(fq8_amax, __shfl_xor_sync(0xffffffff, fq8_amax, off)); }
#elif KDA_P1_REDUX
            float fq8_amax;
            {
                static_assert(decltype(size(tdPrdP))::value == 32, "P1-R: 32 dS values per thread");
                float t[16];
                #pragma unroll
                for (int i = 0; i < 16; ++i) { t[i] = fmaxf(fabsf(tdPrdP(2 * i)), fabsf(tdPrdP(2 * i + 1))); }
                #pragma unroll
                for (int i = 0; i < 8; ++i) { t[i] = fmaxf(t[i], t[i + 8]); }
                #pragma unroll
                for (int i = 0; i < 4; ++i) { t[i] = fmaxf(t[i], t[i + 4]); }
                t[0] = fmaxf(t[0], t[2]); t[1] = fmaxf(t[1], t[3]);
                t[0] = fmaxf(t[0], t[1]);
                fq8_amax = fmaxf(0.f, t[0]);   // the chain's 0.f start: an all-NaN thread gives 0, never NaN
                fq8_amax = __uint_as_float(__reduce_max_sync(0xffffffffu, __float_as_uint(fq8_amax)));
            }
#else
            float fq8_amax = 0.f;
            #pragma unroll
            for (int i = 0; i < size(tdPrdP); ++i) { fq8_amax = fmaxf(fq8_amax, fabsf(tdPrdP(i))); }
            #pragma unroll
            for (int off = 16; off > 0; off /= 2) { fq8_amax = fmaxf(fq8_amax, __shfl_xor_sync(0xffffffff, fq8_amax, off)); }
#endif
            float* const fq8_psi_slots = fq8_psired + (m_block & 1) * 8;
#if KDA_A(4)
            fq8_amax = 448.0f;
#else
            if (lane == 0) { fq8_psi_slots[thread_idx / cutlass::NumThreadsPerWarp] = fq8_amax; }
#endif
            KP(9);
#if KDA_P1_DSDEFER
            if (wg_thread == 0) { cute::tma_store_wait<0>(); }   // store(m-2), issued after the previous tile's psi barrier
#endif
#if KDA_A(4)
#elif KDA_P1_HALFPSI
            cutlass::arch::NamedBarrier::sync(cutlass::NumThreadsPerWarpGroup, wg_barrier_id);
            {
                float4 const s4 = *reinterpret_cast<float4 const*>(fq8_psi_slots + 4 * warp_group_idx);
                fq8_amax = fmaxf(fmaxf(s4.x, s4.y), fmaxf(s4.z, s4.w));
            }
#else
            cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<uint32_t>(BwdNamedBarriers::dQEmptyWG3) /*id 4*/);
#endif
            KP(10);
#if KDA_P1_DSDEFER
            // Every thread of this warpgroup wrote + fenced tile m-1's codes before this barrier: issue its box store now.
            if (wg_thread == 0 && m_block > m_block_min) { fq8_issue_ds_store(m_block - 1); }
#endif
#if KDA_P1_HALFPSI || KDA_A(4)
#elif KDA_P1_REDUX == 1
            {
                float4 const s0 = *reinterpret_cast<float4 const*>(fq8_psi_slots);
                float4 const s1 = *reinterpret_cast<float4 const*>(fq8_psi_slots + 4);
                fq8_amax = fmaxf(fmaxf(fmaxf(s0.x, s0.y), fmaxf(s0.z, s0.w)), fmaxf(fmaxf(s1.x, s1.y), fmaxf(s1.z, s1.w)));
            }
#else
            #pragma unroll
            for (int w = 0; w < NumMmaThreads / cutlass::NumThreadsPerWarp; ++w) { fq8_amax = fmaxf(fq8_amax, fq8_psi_slots[w]); }
#endif
            bool const fq8_tile_zero = fq8_amax * (1.0f / 448.0f) < 1e-30f;
            float const fq8_psi_raw = fmaxf(fq8_amax * (1.0f / 448.0f), 1e-30f);
            float const fq8_psi = fmaxf(fq8_psi_raw, fq8_g_max * (256.0f * 1e-8f) / fq8_q8sc);
            float const fq8_psi_inv = fq8_tile_zero ? 0.0f : 1.0f / fq8_psi;
#if KDA_P1_HALFPSI
            if (wg_thread == 0) {
                params.ptr_psi[((int64_t(fq8_bh) * params.nbm + m_block) * params.nbn + n_block) * 2 + warp_group_idx] = fq8_tile_zero ? 0.0f : fq8_psi;
            }
#else
            if (thread_idx == 0) {
                params.ptr_psi[(int64_t(fq8_bh) * params.nbm + m_block) * params.nbn + n_block] = fq8_tile_zero ? 0.0f : fq8_psi;
            }
#endif
            // dS' codes = e4m3(dS / psi) as the RS A operand of dK.
            #pragma unroll
            for (int i = 0; i < size(tdPrdP); ++i) { tdPrdP(i) *= fq8_psi_inv; }
            flash::permute_Cregs_fp8(tdPrdP);
            Tensor tdKrdS_acc = make_tensor(tdPrdP.data(), flash::convert_layout_acc_Aregs<TiledMmadKV>(tdPrdP.layout()));
            Tensor rdS = make_tensor_like<Element>(tdKrdS_acc);
            if constexpr (KDA_A(9)) { Tensor r32 = recast<uint32_t>(rdS); for (int i = 0; i < size(r32); ++i) { r32(i) = __float_as_uint(tdKrdS_acc(4 * i)); } }
            else { flash::convert_type_out(tdKrdS_acc, rdS); }
            static_assert(decltype(size(rdS))::value == 32, "P1-R: 32 dS' codes per thread per tile");
            KP(11);
            // ## M25 (H1b): pre-pack the 8 natural dS' words from rdS NOW, before rdS becomes the in-flight A operand of the
            // dK gemm; the shuffles, stmatrix and TMA store below run from these copies while dK is outstanding.
            // A-register word 2i = (h0: g 2i, g 2i+1), word 2i+1 = (h1: g 2i, g 2i+1); W[g] = (h0 pair(g) | h1 pair(g) << 16).
            // One 8-word array: the deferred block overwrites each packed word in place with its final stmatrix word.
            uint32_t fq8_pk[8];
            {
                Tensor rdS32 = recast<uint32_t>(rdS);
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    fq8_pk[2 * i] = __byte_perm(rdS32(2 * i), rdS32(2 * i + 1), 0x5410);       // g = 2i
                    fq8_pk[2 * i + 1] = __byte_perm(rdS32(2 * i), rdS32(2 * i + 1), 0x7632);   // g = 2i + 1
                }
            }
            KP(12);
            // The dK fold basis and the F2b anchor (the dV window rescale moved above, H1a).
            fq8_g_fold = 0.0f;
            if (!fq8_tile_zero) {
                float const fq8_g_cur = fq8_psi * kFq8MoInv * fq8_q8sc;
                fq8_g_fold = fq8_g_cur;
                fq8_g_max = fmaxf(fq8_g_max, fq8_g_cur);
            }
            // dK_tmp = dS'^T Q (RS, zero-init); FIFO retire: dV (issued above, H1a) first, then dK.
            KP(14);
#if KDA_P1_PP == 3
            turn_sync();
#endif
#if KDA_P1_DVLATE
            issue_dV();
#endif
            KT(25);
            consumer_wait(pipeline_qt, smem_pipe_read_qt);
            KT(26);
            if constexpr (KDA_A(7)) { for (int i = 0; i < size(fq8_tmpK); ++i) { fq8_tmpK(i) = __uint_as_float(recast<uint32_t>(rdS)(i % 8)); } warpgroup_arrive(); warpgroup_commit_batch(); }
            else if constexpr (kSt >= 0) {
                flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma_dKV, rdS, tdKrQt(_, _, _, Int<kSt>{}), fq8_tmpK);
            } else {
                flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma_dKV, rdS, tdKrQt(_, _, _, smem_pipe_read_qt.index()), fq8_tmpK);
            }
#if KDA_P1_PP == 3
            turn_arrive();
#endif
            KP(15);
            // ## M25 (H1b): dS' natural (rows x keys) for pass 2, the ONE transpose, from the pre-packed words while the dV
            // and dK gemms are outstanding (sdS is not an operand of either: they read the dOt / Qt stages).  The key pair
            // this lane stores comes from lanes srcA / srcB (slot h = ip / 4); stmatrix.trans writes 16 keys x 8 rows.
#if KDA_P1_XPTOP
            #pragma unroll
            for (int w = 0; w < 8; ++w) { fq8_xp[w] = fq8_pk[w]; }
            fq8_xp_pending = true; fq8_xp_m = m_block;
            if constexpr (false) {
#else
            if constexpr (!KDA_A(1)) {
#endif
                #pragma unroll
                for (int w = 0; w < 8; ++w) {   // one word at a time: two shuffles, then overwrite it with its output
                    uint32_t const a = __shfl_sync(0xffffffff, fq8_pk[w], fq8_srcA);
                    uint32_t const b = __shfl_sync(0xffffffff, fq8_pk[w], fq8_srcB);
                    fq8_pk[w] = __byte_perm(a, b, fq8_sel);
                }
#if KDA_P1_DSDEFER
                // Buffer m & 1 was drained (store(m-2)) before this tile's psi barrier; its store is issued next tile.
                uint32_t const a0 = (m_block & 1) ? fq8_st_addr0b : fq8_st_addr0;
                uint32_t const a1 = (m_block & 1) ? fq8_st_addr1b : fq8_st_addr1;
                asm volatile("stmatrix.sync.aligned.x4.trans.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n"
                             :: "r"(a0), "r"(fq8_pk[0]), "r"(fq8_pk[1]), "r"(fq8_pk[2]), "r"(fq8_pk[3]));
                asm volatile("stmatrix.sync.aligned.x4.trans.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n"
                             :: "r"(a1), "r"(fq8_pk[4]), "r"(fq8_pk[5]), "r"(fq8_pk[6]), "r"(fq8_pk[7]));
                cutlass::arch::fence_view_async_shared();
#else
                // The previous tile's TMA box store of this WG must have drained before the tile is overwritten.
                if (wg_thread == 0) { cute::tma_store_wait<0>(); }
                cutlass::arch::NamedBarrier::sync(cutlass::NumThreadsPerWarpGroup, wg_barrier_id);
                asm volatile("stmatrix.sync.aligned.x4.trans.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n"
                             :: "r"(fq8_st_addr0), "r"(fq8_pk[0]), "r"(fq8_pk[1]), "r"(fq8_pk[2]), "r"(fq8_pk[3]));
                asm volatile("stmatrix.sync.aligned.x4.trans.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n"
                             :: "r"(fq8_st_addr1), "r"(fq8_pk[4]), "r"(fq8_pk[5]), "r"(fq8_pk[6]), "r"(fq8_pk[7]));
                cutlass::arch::fence_view_async_shared();
                cutlass::arch::NamedBarrier::sync(cutlass::NumThreadsPerWarpGroup, wg_barrier_id);
                if (wg_thread == 0) { fq8_issue_ds_store(m_block); }
#endif
            }
            KP(13);
            warpgroup_wait<1>();
            pipeline_dot.consumer_release(smem_pipe_read_dot);
            ++smem_pipe_read_dot;
            KP(16);
            warpgroup_wait<0>();
            pipeline_qt.consumer_release(smem_pipe_read_qt);
            ++smem_pipe_read_qt;
            KP(17);
            ++smem_pipe_read;
            if constexpr (!Q_dO_same_stages) { ++smem_pipe_read_do; }
            // plan-A dK fold: one fp32 FMA per element per tile in the tile's own g basis.
#if KDA_P1_EARLY
            bool const fq8_more = m_block + 1 < m_block_max;
            if (fq8_more) { turn_sync(); }
#if KDA_P1_EARLY == 3
            if (fq8_more) { issue_S(); }
#endif
            #pragma unroll
            for (int i = 0; i < size(tdKrdK) / 2; ++i) { if (!KDA_A(2)) tdKrdK(i) = fmaf(fq8_tmpK(i), fq8_g_fold, tdKrdK(i)); }
#if KDA_P1_EARLY == 2 || KDA_P1_EARLY == 5
            if (fq8_more) { issue_S(); }
#endif
            #pragma unroll
            for (int i = size(tdKrdK) / 2; i < size(tdKrdK); ++i) { if (!KDA_A(2)) tdKrdK(i) = fmaf(fq8_tmpK(i), fq8_g_fold, tdKrdK(i)); }
#if KDA_P1_EARLY == 1
            if (fq8_more) { issue_S(); }
#endif
#if KDA_P1_EARLY <= 3
            if (fq8_more) { issue_dP(); turn_arrive(); }
#endif
#elif KDA_P1_FOLDSPLIT
            #pragma unroll
            for (int i = 0; i < 32; ++i) { tdKrdK(i) = fmaf(fq8_tmpK(i), fq8_g_fold, tdKrdK(i)); }
            #pragma unroll
            for (int i = 0; i < 32; ++i) { fq8_khi[i] = fq8_tmpK(32 + i); }
            fq8_khi_g = fq8_g_fold; fq8_khi_pending = true;
            if (m_block == m_block_max - 1) { fold_hi(); }
#else
            #pragma unroll
            for (int i = 0; i < size(tdKrdK); ++i) { if (!KDA_A(2)) tdKrdK(i) = fmaf(fq8_tmpK(i), fq8_g_fold, tdKrdK(i)); }
#endif
            // dV window boundary (geometry-fixed): promote in TRUE units into the overlay, re-zero the window.
#if KDA_P1_TOPFILL == 1
            if (!KDA_A(3) && (m_block & (kFq8PromoteTiles - 1)) == kFq8PromoteTiles - 1 && m_block != m_block_max - 1) {
                fq8_promote_pending = true;
            } else
#endif
            if (!KDA_A(3) && ((m_block & (kFq8PromoteTiles - 1)) == kFq8PromoteTiles - 1 || m_block == m_block_max - 1)) {
#if KDA_P1_PV4
                dv_promote4(fq8_f_prev);
#else
                #pragma unroll
                for (int i = 0; i < size(tdVrdV); ++i) {
                    float* const slot = fq8_smem_dv + i * NumMmaThreads;
                    *slot = fmaf(tdVrdV(i), fq8_f_prev, *slot);
#if !KDA_P1_DVZI
                    tdVrdV(i) = 0.0f;
#endif
                }
#endif
            }
#if KDA_P1_EARLY == 4   // after the promote: only the loop overhead and the next step's scale loads overlap
            if (fq8_more) { issue_S(); issue_dP(); turn_arrive(); }
#elif KDA_P1_EARLY == 5   // S before the promote, dP after it
            if (fq8_more) { issue_dP(); turn_arrive(); }
#endif
            KP(18);
            KP_STEP_DONE();
        };

        auto mask_fn = [&](auto& tSrS, int m_block) { mask.template apply<true /*Seqlenk_mask*/, Is_causal, Is_local>(tSrS, m_block, n_block); };
        auto no_mask_fn = [&](auto& tSrS, int m_block) {};
        // Tiles whose rows all lie at or past this CTA's last key (m_block * 64 + (seqlen_k - seqlen_q) >= (n_block + 1) * 128)
        // have no maskable element; the seqlen_k tail (never at the training shapes) keeps the mask everywhere.
        constexpr int kbm_ = kBlockM, kbn_ = kBlockN;
        int const fq8_mask_m_end = ((n_block + 1) * kbn_ > seqlen_k) ? INT32_MAX
                                   : ((n_block + 1) * kbn_ - (seqlen_k - seqlen_q) + kbm_ - 1) / kbm_;
        KT_HEADER(n_block, get<1>(block_coord), bidb, m_block_min);
        KP(2);
#if KDA_P1_PP == 1
        if (warp_group_idx == 1) {
            cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<uint32_t>(BwdNamedBarriers::dQEmptyWG1));
        }
#elif KDA_P1_PP >= 2
        if (warp_group_idx == 1) { turn_arrive(); }   // warpgroup 0 takes the first turn
#endif
#if KDA_P1_EARLY
        turn_sync();
        issue_S();
        issue_dP();
        turn_arrive();
#endif
        CUTLASS_PRAGMA_NO_UNROLL
        for (; m_block < m_block_max && m_block < fq8_mask_m_end; ++m_block) { bwd_step(m_block, mask_fn, Int<-1>{}); }
        CUTLASS_PRAGMA_NO_UNROLL
#if KDA_P1_UNROLL2
        static_assert(kStages == 2 && kStages_dO == 1 && kStages_dOt == 1, "UNROLL2: written for Q / Qt 2 stages, dO / dOt 1");
        for (; m_block < m_block_max; ) {
            if (smem_pipe_read.index() == 0 && m_block + 1 < m_block_max) {
                bwd_step(m_block, no_mask_fn, Int<0>{}); bwd_step(m_block + 1, no_mask_fn, Int<1>{}); m_block += 2;
            } else {
                bwd_step(m_block, no_mask_fn, Int<-1>{}); ++m_block;
            }
        }
#else
        for (; m_block < m_block_max; ++m_block) { bwd_step(m_block, no_mask_fn, Int<-1>{}); }
#endif

#if KDA_P1_XPTOP
        if (fq8_xp_pending) { xpose_store(fq8_xp, fq8_xp_m); fq8_xp_pending = false; }
#endif
#if KDA_P1_DSDEFER
        // The last tile's codes: one WG barrier (every thread's stmatrix + fence), then its store.
        cutlass::arch::NamedBarrier::sync(cutlass::NumThreadsPerWarpGroup, wg_barrier_id);
        if (wg_thread == 0) { fq8_issue_ds_store(m_block_max - 1); }
#endif
        // Drain the last dS' box store (non-issuing threads return immediately).
        cute::tma_store_wait<0>();
        // Exit: the k8s unfold of dK (this WG's keys are one scale group) and the dV read-back from the overlay.
        float const fq8_ik = params.softmax_scale / fq8_k8s_wg;
        #pragma unroll
        for (int i = 0; i < size(tdKrdK); ++i) { tdKrdK(i) *= fq8_ik; }
        #pragma unroll
#if KDA_P1_PV4
        for (int j = 0; j < size(tdVrdV) / 4; ++j) {
            float4 const v = fq8_smem_dv4[j * NumMmaThreads];
            tdVrdV(4 * j) = v.x; tdVrdV(4 * j + 1) = v.y; tdVrdV(4 * j + 2) = v.z; tdVrdV(4 * j + 3) = v.w;
        }
#else
        for (int i = 0; i < size(tdVrdV); ++i) { tdVrdV(i) = fq8_smem_dv[i * NumMmaThreads]; }
#endif
        if constexpr (Q_dO_same_stages) { smem_pipe_read_do = smem_pipe_read; }
        KP(19);
        KP_FLUSH(0, params.ptr_dbg_cks);
        ++work_idx;
        return true;
    }
};

}  // namespace flash
