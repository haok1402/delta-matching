/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar,
 * Pradeep Ramani, Tri Dao.   (derived from flash-attention hopper/softmax.h)
 ******************************************************************************
 * ## FWDPORT (kda-fwdport): the forward softmax of the FA3-port experiments.
 *
 * One struct, three independent knobs, so each delta-matching addition to FA3's fp8 forward can be built and timed
 * on its own:
 *   kBasis   0: FA3's constant basis (codes e4m3(2^8 exp(s - m)), one row max, Max_offset 8)
 *            1: the faithful per-row, per-64-key-group basis of softmax_fq8f.h (codes e4m3(448 exp(s - nu_g)),
 *               nu_g = max(gm_g, m - B ln2)), the group partial sums folded with sigma_g onto the running max.
 *   kDescale 0: none (the S codes are used raw: timing only)
 *            1: the caller multiplies every S element by q8sc*k8sc_g before the softmax (the shipped FMUL form)
 *            2: DFOLD: the group maxima are taken on the raw codes and descaled once per row and group (bitwise the
 *               shipped maxima: RN(x g) is monotone in x), the exp argument is fma(x, RN(g l2), -off) (round-off class)
 *   kGroups  any number of 64-key groups per tile (2 / 3 / 4 at BN 128 / 192 / 256).
 * The struct keeps SoftmaxFq8F's method names (max_get_scale_grouped, online_softmax_grouped, basis<G>,
 * basis_last, promote_o, rescale_o, finalize) so the mainloop calls one interface.
 ******************************************************************************/

#pragma once

#include <cmath>
#include <utility>

#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>

#include "utils.h"
#include "softmax.h"


namespace flash {

using namespace cute;

template <int... Is, typename F>
__device__ __forceinline__ void port_static_for(std::integer_sequence<int, Is...>, F&& f) { (f(cute::Int<Is>{}), ...); }
template <int N, typename F>
__device__ __forceinline__ void port_for(F&& f) { port_static_for(std::make_integer_sequence<int, N>{}, f); }

// thread-local reduction over the rowcol columns [kCols*G, kCols*G + kCols) (any rowcol width)
template <int G, int kCols, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void port_reduce_group_(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1>& summary,
                                                   Operator& op) {
    #pragma unroll
    for (int ni = kCols * G; ni < kCols * G + kCols; ++ni) {
        #pragma unroll
        for (int mi = 0; mi < size<0>(tensor); ++mi) {
            summary(mi) = (ni == kCols * G) ? tensor(mi, ni) : op(summary(mi), tensor(mi, ni));
        }
    }
}

template <int kNRows, int kGroups, int kRowcol, int kBasis, int kDescale, int kPhatFloorB = 12>
struct SoftmaxPort {
    static_assert(kBasis == 0 || kBasis == 1);
    static_assert(kDescale >= 0 && kDescale <= 2);
    static constexpr int kColsPerGroup = kGroups > 0 ? kRowcol / kGroups : kRowcol;
    static_assert((kBasis == 0 && kDescale == 0) || kColsPerGroup * kGroups == kRowcol, "groups must tile the rowcol width");
    using TensorT = decltype(make_tensor<float>(Shape<Int<kNRows>>{}));
    static constexpr float kLog2_448 = 8.8073549220576042f;
    static constexpr float kFloorB = float(kPhatFloorB);
    static constexpr float kMaxOffset = 8.f;   // FA3's fp8 Max_offset (kBasis 0)
    static constexpr int kGS = kGroups > 0 ? kGroups : 1;

    TensorT row_max, row_sum;
    TensorT ms_[kGS];          // kBasis 1: the per-group log2 basis of the codes in tOrP
    float gsc_[kGS];           // kDescale 2: per-group descale q8sc * k8sc_g (natural units)
    float const softmax_scale_log2;

    CUTLASS_DEVICE SoftmaxPort(float const l2) : softmax_scale_log2(l2) {
        #pragma unroll
        for (int g = 0; g < kGS; ++g) { gsc_[g] = 1.f; }
    }

    template <bool Check_inf>
    __forceinline__ __device__ float max_l2(int mi) const {
        return Check_inf ? (row_max(mi) == -INFINITY ? 0.f : row_max(mi) * softmax_scale_log2)
                         : row_max(mi) * softmax_scale_log2;
    }

    template <bool Is_first, bool Check_inf = false, typename Tensor0>
    __forceinline__ __device__ TensorT max_get_scale_grouped(Tensor0& acc_s) {
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        static_assert(CUTE_STATIC_V(size<0>(scores)) == kNRows);
        MaxOp<float> max_op;
        TensorT tile_max;
        if constexpr (kBasis == 0 && kDescale != 2) {
            // FA3: one thread-local row max over the whole tile, one quad all-reduce
            flash::template reduce_max</*zero_init=*/true>(scores, tile_max);
        } else {
            // group maxima (on the raw codes with DFOLD), descaled once per row and group
            port_for<kGroups>([&](auto g_c) {
                constexpr int g = decltype(g_c)::value;
                port_reduce_group_<g, kColsPerGroup>(scores, ms_[g], max_op);
                if constexpr (kDescale == 2) {
                    #pragma unroll
                    for (int mi = 0; mi < kNRows; ++mi) { ms_[g](mi) *= gsc_[g]; }
                }
            });
            if constexpr (kBasis == 0) {
                // row basis: one all-reduce of the descaled tile max
                #pragma unroll
                for (int mi = 0; mi < kNRows; ++mi) {
                    float t = ms_[0](mi);
                    #pragma unroll
                    for (int g = 1; g < kGroups; ++g) { t = fmaxf(t, ms_[g](mi)); }
                    tile_max(mi) = t;
                }
                flash::quad_allreduce_(tile_max, tile_max, max_op);
            } else {
                #pragma unroll
                for (int g = 0; g < kGroups; ++g) { flash::quad_allreduce_(ms_[g], ms_[g], max_op); }
                #pragma unroll
                for (int mi = 0; mi < kNRows; ++mi) {
                    float t = ms_[0](mi);
                    #pragma unroll
                    for (int g = 1; g < kGroups; ++g) { t = fmaxf(t, ms_[g](mi)); }
                    tile_max(mi) = t;
                }
            }
        }
        TensorT scores_scale;
        #pragma unroll
        for (int mi = 0; mi < kNRows; ++mi) {
            if constexpr (Is_first) {
                row_max(mi) = tile_max(mi);
                scores_scale(mi) = 1.f;
            } else {
                float const prev = row_max(mi);
                row_max(mi) = fmaxf(prev, tile_max(mi));
                float const cur = !Check_inf ? row_max(mi) : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
                scores_scale(mi) = exp2f((prev - cur) * softmax_scale_log2);
                row_sum(mi) *= scores_scale(mi);
            }
            if constexpr (kBasis == 1) {
                float const floor_l2 = max_l2<Check_inf>(mi) - kFloorB;
                #pragma unroll
                for (int g = 0; g < kGroups; ++g) { ms_[g](mi) = fmaxf(ms_[g](mi) * softmax_scale_log2, floor_l2) - kLog2_448; }
            }
        }
        return scores_scale;
    }

    template <bool Is_first, bool Check_inf = false, typename Tensor0>
    __forceinline__ __device__ void online_softmax_grouped(Tensor0& acc_s) {
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        static_assert(CUTE_STATIC_V(size<0>(scores)) == kNRows);
        constexpr int kCols = decltype(size<1>(scores))::value;
        float c_[kGS];   // exp-argument multiplier per group
        #pragma unroll
        for (int g = 0; g < kGS; ++g) { c_[g] = kDescale == 2 ? gsc_[g] * softmax_scale_log2 : softmax_scale_log2; }
        if constexpr (kBasis == 0) {
            #pragma unroll
            for (int mi = 0; mi < kNRows; ++mi) {
                float const off = Check_inf ? (row_max(mi) == -INFINITY ? 0.f : row_max(mi) * softmax_scale_log2 - kMaxOffset)
                                            : row_max(mi) * softmax_scale_log2 - kMaxOffset;
                #pragma unroll
                for (int ni = 0; ni < kCols; ++ni) {
                    float const c = kDescale == 2 ? c_[ni / kColsPerGroup] : softmax_scale_log2;
                    scores(mi, ni) = exp2f(scores(mi, ni) * c - off);
                }
            }
            flash::reduce_sum</*zero_init=*/Is_first, /*warp_reduce=*/false>(scores, row_sum);
        } else {
            #pragma unroll
            for (int mi = 0; mi < kNRows; ++mi) {
                #pragma unroll
                for (int ni = 0; ni < kCols; ++ni) {
                    scores(mi, ni) = exp2f(scores(mi, ni) * c_[ni / kColsPerGroup] - ms_[ni / kColsPerGroup](mi));
                }
            }
            SumOp<float> sum_op;
            TensorT acc, part;
            cute::fill(acc, 0.f);
            port_for<kGroups>([&](auto g_c) {
                constexpr int g = decltype(g_c)::value;
                port_reduce_group_<g, kColsPerGroup>(scores, part, sum_op);
                #pragma unroll
                for (int mi = 0; mi < kNRows; ++mi) { acc(mi) += exp2f(ms_[g](mi) - max_l2<Check_inf>(mi)) * part(mi); }
            });
            #pragma unroll
            for (int mi = 0; mi < kNRows; ++mi) { row_sum(mi) = Is_first ? acc(mi) : row_sum(mi) + acc(mi); }
        }
    }

    template <int G>
    __forceinline__ __device__ TensorT basis(float v8sc_g) const {
        TensorT c;
        #pragma unroll
        for (int mi = 0; mi < kNRows; ++mi) {
            if constexpr (kBasis == 1) { c(mi) = v8sc_g * exp2f(ms_[G](mi) - max_l2</*Check_inf=*/true>(mi)); }
            else { c(mi) = v8sc_g; }
        }
        return c;
    }
    __forceinline__ __device__ TensorT basis_last(float v8sc_3) const { return basis<kGS - 1>(v8sc_3); }

    __forceinline__ __device__ TensorT finalize(float const final_scale = 1.f) {
        SumOp<float> sum_op;
        quad_allreduce_(row_sum, row_sum, sum_op);
        TensorT scores_scale;
        #pragma unroll
        for (int mi = 0; mi < size(row_sum); ++mi) {
            float sum = row_sum(mi);
            float inv_sum = (sum == 0.f || sum != sum) ? 0.f : 1.f / sum;
            scores_scale(mi) = inv_sum * final_scale;
            if constexpr (kBasis == 0) { sum *= 1.f / 256.f; }
            row_sum(mi) = (sum == 0.f || sum != sum) ? -INFINITY
                                                     : row_max(mi) * (softmax_scale_log2 * float(M_LN2)) + __logf(sum);
        }
        return scores_scale;
    }

    template <typename Tensor1, typename Tensor2>
    __forceinline__ __device__ void promote_o(Tensor1& acc_o, Tensor2 const& acc_d, TensorT const& w) {
        Tensor o_ = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
        Tensor d_ = make_tensor(acc_d.data(), flash::convert_layout_acc_rowcol(acc_d.layout()));
        #pragma unroll
        for (int mi = 0; mi < size<0>(o_); ++mi) {
            #pragma unroll
            for (int ni = 0; ni < size<1>(o_); ++ni) { o_(mi, ni) = fmaf(d_(mi, ni), w(mi), o_(mi, ni)); }
        }
    }

    template <typename Tensor1>
    __forceinline__ __device__ void rescale_o(Tensor1& acc_o, TensorT const& s) {
        Tensor o_ = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
        #pragma unroll
        for (int mi = 0; mi < size<0>(o_); ++mi) {
            #pragma unroll
            for (int ni = 0; ni < size<1>(o_); ++ni) { o_(mi, ni) *= s(mi); }
        }
    }
};

}  // namespace flash
