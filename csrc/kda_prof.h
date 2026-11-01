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
