// ## KDA_PROF (profiling-only build, DM_PROF=1; never shipped): clock64() cycles per phase of each consumer warpgroup's
// pass-1 tile loop, accumulated by the warpgroup's first thread in shared memory and flushed with one atomicAdd per slot
// into the dbg_cks buffer (DBG-only otherwise), viewed as uint64 [variant][warpgroup][32]: variant 0 = the hd128 P1-R
// kernel, 1 = the d256 V10 kernel.  The probes only read the clock: the arithmetic and its order are unchanged.
// The enclosing scope provides wg_thread and warp_group_idx.  The static shared counters are exactly 1 KB so the dynamic
// shared-memory base after them stays 1024-byte aligned (a 512 B array broke the swizzled TMA layouts: wrong results).
#pragma once
#ifndef KDA_PROF
#define KDA_PROF 0
#endif
#if KDA_PROF
#define KP(slot) do { long long const kp_now_ = clock64(); \
    if (wg_thread == 0) { kp_s[warp_group_idx][slot] += (unsigned long long)(kp_now_ - kp_t); } kp_t = kp_now_; } while (0)
#define KP_BEGIN() \
    __shared__ alignas(128) unsigned long long kp_s[2][64]; /* 1024 B: keeps the dynamic smem base 1024-aligned (SW128) */ \
    if (wg_thread == 0) { _Pragma("unroll 1") for (int kp_i_ = 0; kp_i_ < 32; ++kp_i_) { kp_s[warp_group_idx][kp_i_] = 0ull; } } \
    long long kp_t = clock64(); long long const kp_t0 = kp_t
#define KP_STEP_DONE() do { if (wg_thread == 0) { kp_s[warp_group_idx][0] += 1ull; } } while (0)
#define KP_FLUSH(variant, ptr) do { if (wg_thread == 0) { \
    kp_s[warp_group_idx][1] += (unsigned long long)(kp_t - kp_t0); \
    unsigned long long* const kp_g_ = reinterpret_cast<unsigned long long*>(ptr) + ((variant) * 2 + warp_group_idx) * 32; \
    _Pragma("unroll 1") for (int kp_i_ = 0; kp_i_ < 20; ++kp_i_) { atomicAdd(kp_g_ + kp_i_, kp_s[warp_group_idx][kp_i_]); } } } while (0)
#else
#define KP(slot) do {} while (0)
#define KP_BEGIN() do {} while (0)
#define KP_STEP_DONE() do {} while (0)
#define KP_FLUSH(variant, ptr) do {} while (0)
#endif

// ## KDA_TRACE (timeline-only build, DM_TRACE=1; never shipped): absolute clock64() event times for a fixed sample of
// CTAs (linear block id lb with lb % KT_STRIDE == 0 and lb / KT_STRIDE < KT_S), the first KT_M m-block steps of each
// sampled CTA's FIRST tile, for three actors (consumer warpgroups 0 / 1 = their first thread, the producer = its elected
// lane).  Every (variant, sample, step, actor, event) has its own uint64 slot (no atomics), plus a per-sample header.
// Timestamps are comparable only within a CTA (one SM).  Event 30 / 31 of a step hold the pipeline stage index / phase.
#ifndef KDA_TRACE
#define KDA_TRACE 0
#endif
#if KDA_TRACE
#if KDA_PROF
#error "KDA_TRACE and KDA_PROF are separate builds"
#endif
#define KT_S 8
#define KT_M 12
#define KT_E 32
#define KT_HDR 16
#define KT_STRIDE 17
#define KT_EVENTS (2 * KT_S * KT_M * 3 * KT_E)
#define KT_WORDS (KT_EVENTS + 2 * KT_S * KT_HDR)
__device__ __forceinline__ int kt_linear_block() { return blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z); }
__device__ __forceinline__ int kt_sample_idx() {
    int const lb = kt_linear_block();
    return (lb % KT_STRIDE == 0 && lb / KT_STRIDE < KT_S) ? lb / KT_STRIDE : -1;
}
__device__ __forceinline__ unsigned long long* kt_slot(void* buf, int var, int s, int j, int actor) {
    return reinterpret_cast<unsigned long long*>(buf) + ((((var * KT_S + s) * KT_M + j) * 3 + actor) * KT_E);
}
__device__ __forceinline__ unsigned long long* kt_hdr(void* buf, int var, int s) {
    return reinterpret_cast<unsigned long long*>(buf) + KT_EVENTS + (var * KT_S + s) * KT_HDR;
}
__device__ __forceinline__ unsigned long long kt_smid() { unsigned r; asm volatile("mov.u32 %0, %%smid;" : "=r"(r)); return r; }
// consumers (scope: wg_thread, warp_group_idx, work_idx, params)
#undef KP
#undef KP_BEGIN
#undef KP_STEP_DONE
#undef KP_FLUSH
#define KP(slot) do { if ((slot) >= 3 && (slot) <= 18) { KT(slot); } } while (0)
#define KP_BEGIN() do {} while (0)
#define KP_STEP_DONE() do { ++kt_j; } while (0)
#define KP_FLUSH(variant, ptr) do {} while (0)
#define KT_BEGIN(var) constexpr int kt_var = (var); int const kt_s = kt_sample_idx(); \
    bool const kt_rec = kt_s >= 0 && work_idx == 0 && wg_thread == 0; int kt_j = 0; void* const kt_buf_ = params.ptr_dbg_cks
#define KT(e) do { if (kt_rec && kt_j < KT_M) { kt_slot(kt_buf_, kt_var, kt_s, kt_j, warp_group_idx)[(e)] = (unsigned long long)clock64(); } } while (0)
#define KT_VAL(e, v) do { if (kt_rec && kt_j < KT_M) { kt_slot(kt_buf_, kt_var, kt_s, kt_j, warp_group_idx)[(e)] = (unsigned long long)(v); } } while (0)
#define KT_HEADER(nb, h, b, mmin) do { if (kt_rec && warp_group_idx == 0) { unsigned long long* const kh_ = kt_hdr(kt_buf_, kt_var, kt_s); \
    kh_[0] = 1ull; kh_[1] = (nb); kh_[2] = (h); kh_[3] = (b); kh_[4] = kt_smid(); kh_[5] = (mmin); kh_[6] = kt_linear_block(); \
    kh_[7] = (unsigned long long)clock64(); } } while (0)
// producer (scope: params, the collective's kt_ptile_ member; runs on the elected lane)
#define KTP_BEGIN(var) constexpr int ktp_var = (var); int const ktp_s = kt_sample_idx(); bool const ktp_rec = ktp_s >= 0 && kt_ptile_ == 0; int ktp_j = 0
#define KTP(e) do { if (ktp_rec && ktp_j < KT_M) { kt_slot(params.ptr_dbg_cks, ktp_var, ktp_s, ktp_j, 2)[(e)] = (unsigned long long)clock64(); } } while (0)
#define KTP_VAL(e, v) do { if (ktp_rec && ktp_j < KT_M) { kt_slot(params.ptr_dbg_cks, ktp_var, ktp_s, ktp_j, 2)[(e)] = (unsigned long long)(v); } } while (0)
#define KTP_H(i) do { if (ktp_rec) { kt_hdr(params.ptr_dbg_cks, ktp_var, ktp_s)[(i)] = (unsigned long long)clock64(); } } while (0)
#define KTP_STEP() do { ++ktp_j; } while (0)
#define KTP_END() do { ++kt_ptile_; } while (0)
#define KT_MEMBERS int kt_ptile_ = 0;
#else
#define KT_BEGIN(var) do {} while (0)
#define KT(e) do {} while (0)
#define KT_VAL(e, v) do {} while (0)
#define KT_HEADER(nb, h, b, mmin) do {} while (0)
#define KTP_BEGIN(var) do {} while (0)
#define KTP(e) do {} while (0)
#define KTP_VAL(e, v) do {} while (0)
#define KTP_H(i) do {} while (0)
#define KTP_STEP() do {} while (0)
#define KTP_END() do {} while (0)
#define KT_MEMBERS
#endif

// ## KDA_TRACE forward section: the same sparse-sample idea for the FORWARD kernel (separate per-translation-unit buffer,
// since the forward params carry no debug pointer; read through the prof_fwd_buffer op).  Sample CTAs as above (the
// persistent grid's blocks 0, 17, ..., 119), first tile only, the first KF_M n-block steps; actors: consumer warpgroups
// 0 / 1 (first thread) and the producer warpgroup's thread 0.  Layout: [KF_S][KF_M][3][KF_E] + [KF_S][KF_HDR].
#if KDA_TRACE
#define KF_S 8
#define KF_M 12
#define KF_E 32
#define KF_HDR 16
#define KF_EVENTS (KF_S * KF_M * 3 * KF_E)
#define KF_WORDS (KF_EVENTS + KF_S * KF_HDR)
static __device__ unsigned long long kda_fwd_buf[KF_WORDS];
#define KF_BEGIN() int const kf_s = kt_sample_idx(); \
    bool const kf_rec = kf_s >= 0 && work_idx == 0 && (thread_idx % cutlass::NumThreadsPerWarpGroup) == 0; int kf_j = 0
#define KF(e) do { if (kf_rec && kf_j < KF_M) { kda_fwd_buf[((kf_s * KF_M + kf_j) * 3 + warp_group_idx) * KF_E + (e)] = (unsigned long long)clock64(); } } while (0)
#define KF_VAL(e, v) do { if (kf_rec && kf_j < KF_M) { kda_fwd_buf[((kf_s * KF_M + kf_j) * 3 + warp_group_idx) * KF_E + (e)] = (unsigned long long)(v); } } while (0)
#define KF_STEP() do { ++kf_j; } while (0)
#define KF_HEADER(mb, h, b, nmax, nmin) do { if (kf_rec && warp_group_idx == 0) { unsigned long long* const kh_ = kda_fwd_buf + KF_EVENTS + kf_s * KF_HDR; \
    kh_[0] = 1ull; kh_[1] = (mb); kh_[2] = (h); kh_[3] = (b); kh_[4] = kt_smid(); kh_[5] = (nmax); kh_[6] = (nmin); \
    kh_[7] = (unsigned long long)clock64(); kh_[8] = kt_linear_block(); } } while (0)
#define KFP_BEGIN() int const kfp_s = kt_sample_idx(); bool const kfp_rec = kfp_s >= 0 && work_idx == 0 && thread_idx == 0; int kfp_j = 0
#define KFP(e) do { if (kfp_rec && kfp_j < KF_M) { kda_fwd_buf[((kfp_s * KF_M + kfp_j) * 3 + 2) * KF_E + (e)] = (unsigned long long)clock64(); } } while (0)
#define KFP_VAL(e, v) do { if (kfp_rec && kfp_j < KF_M) { kda_fwd_buf[((kfp_s * KF_M + kfp_j) * 3 + 2) * KF_E + (e)] = (unsigned long long)(v); } } while (0)
#define KFP_STEP() do { ++kfp_j; } while (0)
#else
#define KF_BEGIN() do {} while (0)
#define KF(e) do {} while (0)
#define KF_VAL(e, v) do {} while (0)
#define KF_STEP() do {} while (0)
#define KF_HEADER(mb, h, b, nmax, nmin) do {} while (0)
#define KFP_BEGIN() do {} while (0)
#define KFP(e) do {} while (0)
#define KFP_VAL(e, v) do {} while (0)
#define KFP_STEP() do {} while (0)
#endif
