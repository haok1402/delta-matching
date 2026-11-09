// p0rw2 (track R): pass 0 rewritten around MUFU occupancy ("D8"). CUDA C++ / inline PTX, sm_100a.
// delta[b,h,i] = sum_j P_ij dP_ij / sum_j P_ij, P = ex2.approx.ftz(fma(S, c_j, -l2_i)), dP = (dO8 V8^T) * a_j  -- the
// dm_common contract, bitwise equal to DMPass0Sm100 (exp "p3_16_0+ma+h2+pf+sq+ts"): same per-element ops, same den chains
// (4 f32x2 chains over pairs v%4 of each 32-col chunk, chunks 0..3, reset per key tile), same num chain
// (num = fma((acc0+acc1)+(acc2+acc3), a_j, num) per chunk), per-tile TwoSum, same fp64 final combine.
// Structure: CTA = (b, h, pair of 128-row Q tiles = 2 stages). 20 warps: 0-15 compute (stage s = (w/4)&1, row group
// h = w/8: rows 32q + 16h .. +15 of TMEM quadrant q = w%4), 16 MMA, 17 TMA load, 18-19 idle. tcgen05.ld.16x256b gives
// lane k = t%4 the column pairs with v%4 == k, so each lane carries den/num chain k of its two rows (rA, rA+8); the
// 4-lane combines (a+b)+(c+d) use xor-shuffles (exact). TMEM: S_0 0..127, S_1 128..255, dP_0 256..383, dP_1 384..511.
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>

namespace p0rw2 {

#ifndef P0RW2_HFU
#define P0RW2_HFU 1
#endif
#ifndef P0RW2_KVS
#define P0RW2_KVS 5
#endif
constexpr int HFU = P0RW2_HFU;
constexpr int KVS = P0RW2_KVS;            // K/V ring stages
constexpr float LOG2E = 1.4426950408889634f;
constexpr int NWARPS = 20, NTHR = NWARPS * 32;
constexpr int CR = 104, OR = 64;            // compute / other registers (CTA pool 20 x 96: 16*104 + 4*64 = 1920)

struct __align__(8) Bars {
  uint64_t qfull, kvfull[KVS], kvempty[KVS], sfull[2], sempty[2], dfull[2], dempty[2], vfull[KVS], qempty;
};
constexpr int SMEM_Q = 0, SMEM_DO = 32768, SMEM_KV = 65536, KV_STRIDE = 32768;   // K at +0, V at +16384 of a slot
constexpr int SMEM_BYTES = SMEM_KV + KVS * KV_STRIDE + 1024;                      // + 1024 alignment slack

__device__ __forceinline__ uint32_t su32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ unsigned long long pk(float a, float b) { unsigned long long r; asm("mov.b64 %0, {%1, %2};" : "=l"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ void upk(unsigned long long r, float& a, float& b) { asm("mov.b64 {%0, %1}, %2;" : "=f"(a), "=f"(b) : "l"(r)); }
// production PTX flavours (DSL dump): fma.rn.f32x2 / add.rn.f32x2 (no ftz), ex2.approx.ftz.f32, mul.f32, neg.f32
__device__ __forceinline__ unsigned long long ffma2(unsigned long long a, unsigned long long b, unsigned long long c) {
  unsigned long long d; asm("fma.rn.f32x2 %0, %1, %2, %3;" : "=l"(d) : "l"(a), "l"(b), "l"(c)); return d; }
__device__ __forceinline__ unsigned long long fadd2(unsigned long long a, unsigned long long b) {
  unsigned long long d; asm("add.rn.f32x2 %0, %1, %2;" : "=l"(d) : "l"(a), "l"(b)); return d; }
__device__ __forceinline__ float ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ float fmul(float a, float b) { float d; asm("mul.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b)); return d; }
__device__ __forceinline__ float fsub(float a, float b) { float d; asm("sub.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b)); return d; }
__device__ __forceinline__ float fneg(float a) { float d; asm("neg.f32 %0, %1;" : "=f"(d) : "f"(a)); return d; }

__device__ __forceinline__ void mbar_init(uint64_t* b, int cnt) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(su32(b)), "r"(cnt)); }
__device__ __forceinline__ void mbar_arrive(uint64_t* b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(su32(b)) : "memory"); }
__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(su32(b)), "r"(bytes) : "memory"); }
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t par) {
#ifdef P0RW2_OLDWAIT
  uint32_t ok = 0;
  while (!ok) asm volatile("{\n.reg .pred P;\nmbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2;\nselp.u32 %0, 1, 0, P;\n}" : "=r"(ok) : "r"(su32(b)), "r"(par) : "memory");
#else
  // DSL idiom: suspend-hinted try_wait in an asm-local loop (no C-level branch on a selp result)
  asm volatile("{\n.reg .pred P1;\nLAB_WAIT:\nmbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, 10000000;\n@P1 bra.uni DONE;\nbra.uni LAB_WAIT;\nDONE:\n}"
               :: "r"(su32(b)), "r"(par) : "memory");
#endif
}
__device__ __forceinline__ void tc_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
#ifdef P0RW2_TRACE
__device__ unsigned long long* g_trace;
#define TACQ(b, par, slot) do { unsigned long long _t = clock64(); mbar_wait(b, par); tc_after(); if ((threadIdx.x & 31) == 0) atomicAdd(&g_trace[slot], clock64() - _t); } while (0)
#define TWAIT(b, par, slot) do { unsigned long long _t = clock64(); mbar_wait(b, par); if ((threadIdx.x & 31) == 0) atomicAdd(&g_trace[slot], clock64() - _t); } while (0)
#else
#define TACQ(b, par, slot) acq(b, par)
#define TWAIT(b, par, slot) mbar_wait(b, par)
#endif
__device__ __forceinline__ void acq(uint64_t* b, uint32_t par) { mbar_wait(b, par); tc_after(); }
__device__ __forceinline__ void rel(uint64_t* b, int lane) { tc_before(); __syncwarp(); if (lane == 0) mbar_arrive(b); }
__device__ __forceinline__ void commit(uint64_t* b) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(su32(b)) : "memory"); }
__device__ __forceinline__ void wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }

__device__ __forceinline__ void tma4(uint32_t dst, const CUtensorMap* tm, int c0, int c1, int c2, int c3, uint64_t* bar) {
  asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, %5}], [%6];"
               :: "r"(dst), "l"((uint64_t)tm), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(su32(bar)) : "memory");
}
// tcgen05 SMEM descriptor, K-major SWIZZLE_128B: 8-row atoms of 128 B, SBO = 1024 B, LBO unused (1), version bit 46.
__device__ __forceinline__ uint64_t desc_sw128(uint32_t saddr) {
  uint64_t d = 0;
  d |= (uint64_t)((saddr >> 4) & 0x3FFF);
  d |= (uint64_t)1 << 16;
  d |= (uint64_t)(1024 >> 4) << 32;
  d |= (uint64_t)1 << 46;
  d |= (uint64_t)2 << 61;
  return d;
}
// kind::f8f6f4, A/B e4m3 K-major, D f32, M = 128, N = 128
constexpr uint32_t IDESC = (1u << 4) | ((uint32_t)(128 >> 3) << 17) | ((uint32_t)(128 >> 4) << 24);
__device__ __forceinline__ void mma_128(uint32_t tD, uint32_t sA, uint32_t sB) {
#ifdef P0RW2_NOMMA
  return;
#endif
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const uint64_t a = desc_sw128(sA + 32 * k), b = desc_sw128(sB + 32 * k);
    const uint32_t acc = k > 0;
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\ntcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n}"
                 :: "r"(tD), "l"(a), "l"(b), "r"(IDESC), "r"(acc));
  }
}
#define LD16X4(addr, r) asm volatile("tcgen05.ld.sync.aligned.16x256b.x4.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];" \
  : "=f"(r[0]),"=f"(r[1]),"=f"(r[2]),"=f"(r[3]),"=f"(r[4]),"=f"(r[5]),"=f"(r[6]),"=f"(r[7]),"=f"(r[8]),"=f"(r[9]),"=f"(r[10]),"=f"(r[11]),"=f"(r[12]),"=f"(r[13]),"=f"(r[14]),"=f"(r[15]) : "r"(addr))

__device__ __forceinline__ unsigned long long shx(unsigned long long v, int m) {
  float a, b; upk(v, a, b);
  return pk(__shfl_xor_sync(0xffffffffu, a, m), __shfl_xor_sync(0xffffffffu, b, m));
}
// (x_0 + x_1) + (x_2 + x_3) over the 4 lanes of a row group (lane k holds x_k); every lane gets the same bits
__device__ __forceinline__ unsigned long long sum4l(unsigned long long x, int k) {
  const unsigned long long o1 = shx(x, 1);
  const unsigned long long y = (k & 1) ? fadd2(o1, x) : fadd2(x, o1);   // x_{2m} + x_{2m+1}, operand order as today
  const unsigned long long o2 = shx(y, 2);
  return (k & 2) ? fadd2(o2, y) : fadd2(y, o2);
}
// TwoSum(s, e) += b exactly as DMPass0Sm100 (+ts)
__device__ __forceinline__ void two_sum2(unsigned long long& s, unsigned long long& e, unsigned long long b) {
  float s0, s1; upk(s, s0, s1);
  const unsigned long long sn = fadd2(s, b);
  float n0, n1; upk(sn, n0, n1);
  const unsigned long long bb = fadd2(sn, pk(fneg(s0), fneg(s1)));
  float bb0, bb1; upk(bb, bb0, bb1);
  const unsigned long long a = fadd2(s, pk(fneg(fsub(n0, bb0)), fneg(fsub(n1, bb1))));
  const unsigned long long br = fadd2(b, pk(fneg(bb0), fneg(bb1)));
  e = fadd2(e, fadd2(a, br));
  s = sn;
}

template <bool MASK>
__device__ __forceinline__ void tile_body(Bars& bars, float* r, const float* cj, const float* aj, uint32_t cS, uint32_t cD,
                                          int s, int it, int lane, int k, int kbase, int rA, int rB,
                                          unsigned long long nlA, unsigned long long nlB,
                                          unsigned long long& denA, unsigned long long& denB,
                                          unsigned long long& numA, unsigned long long& numB) {
      TACQ(&bars.sfull[s], it & 1, 1);
#pragma unroll HFU
      for (int hf = 0; hf < 2; ++hf) {
        LD16X4(cS + 64 * hf, (&r[0])); LD16X4(cS + 64 * hf + 32, (&r[16]));
        wait_ld();
        if (hf == 1) rel(&bars.sempty[s], lane);
#pragma unroll
        for (int jj = 0; jj < 2; ++jj) {
          const int j = 2 * hf + jj;
#pragma unroll
          for (int g = 0; g < 4; ++g) {
            float* q = &r[16 * jj + 4 * g];
            if (MASK) {
              const int col = kbase + 32 * j + 8 * g + 2 * k;
              if (col > rA) q[0] = -__int_as_float(0x7f800000);
              if (col + 1 > rA) q[1] = -__int_as_float(0x7f800000);
              if (col > rB) q[2] = -__int_as_float(0x7f800000);
              if (col + 1 > rB) q[3] = -__int_as_float(0x7f800000);
            }
            float x0, x1;
            upk(ffma2(pk(q[0], q[1]), pk(cj[j], cj[j]), nlA), x0, x1); q[0] = ex2(x0); q[1] = ex2(x1);
            denA = fadd2(denA, pk(q[0], q[1]));
            upk(ffma2(pk(q[2], q[3]), pk(cj[j], cj[j]), nlB), x0, x1); q[2] = ex2(x0); q[3] = ex2(x1);
            denB = fadd2(denB, pk(q[2], q[3]));
          }
        }
        if (hf == 0) TACQ(&bars.dfull[s], it & 1, 2);
#pragma unroll
        for (int jj = 0; jj < 2; ++jj) {
          const int j = 2 * hf + jj;
          float d[16];
          LD16X4(cD + 32 * j, d);
          wait_ld();
          if (j == 3) rel(&bars.dempty[s], lane);
          unsigned long long accA = 0, accB = 0;
#pragma unroll
          for (int g = 0; g < 4; ++g) {
            const float* q = &r[16 * jj + 4 * g];
            accA = ffma2(pk(q[0], q[1]), pk(d[4 * g], d[4 * g + 1]), accA);
            accB = ffma2(pk(q[2], q[3]), pk(d[4 * g + 2], d[4 * g + 3]), accB);
          }
          numA = ffma2(sum4l(accA, k), pk(aj[j], aj[j]), numA);
          numB = ffma2(sum4l(accB, k), pk(aj[j], aj[j]), numB);
        }
      }
}

struct Params {
  const float *qs, *ks, *vs, *dos, *lse;
  float* delta;
  unsigned long long* trace;   // P0RW2_TRACE builds: [0] CTA clk, [1] compute sfull wait, [2] compute dfull wait, [3] MMA kvfull wait, [4] MMA sempty+dempty wait, [5] load kvempty wait, [6] compute warps clk
  int B, N, H, Hk, causal, npairs;
};

__global__ void __launch_bounds__(NTHR, 1) k_p0rw2(const __grid_constant__ CUtensorMap tmQ, const __grid_constant__ CUtensorMap tmK,
                                                   const __grid_constant__ CUtensorMap tmV, const __grid_constant__ CUtensorMap tmDO,
                                                   const Params p) {
  extern __shared__ uint8_t smraw[];
  __shared__ __align__(8) Bars bars;
  __shared__ uint32_t s_taddr;
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, quad = warp % 4;
  const uint32_t sbase = (su32(smraw) + 1023u) & ~1023u;
  // work tile, heaviest (largest q pair) first
  const int BH = p.B * p.H;
  const int mp = p.npairs - 1 - (int)(blockIdx.x / BH), bh = (int)(blockIdx.x % BH);
  const int b = bh / p.H, h = bh % p.H, hk = h / (p.H / p.Hk);
  const int NB = p.N / 128;
  int n_max = NB;
  if (p.causal) n_max = min(NB, 2 * mp + 2);
  if (warp == 0) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" :: "r"(su32(&s_taddr)));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  if (tid == 0) {
    mbar_init(&bars.qfull, 1);
    for (int i = 0; i < KVS; ++i) { mbar_init(&bars.kvfull[i], 1); mbar_init(&bars.kvempty[i], 1); }
    for (int s = 0; s < 2; ++s) { mbar_init(&bars.sfull[s], 1); mbar_init(&bars.sempty[s], 8); mbar_init(&bars.dfull[s], 1); mbar_init(&bars.dempty[s], 8); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  tc_before(); __syncthreads(); tc_after();
  const uint32_t tb = s_taddr;
#if defined(P0RW2_TRACE) || defined(P0RW2_CYC)
  const unsigned long long tcta0 = clock64();
#endif
#ifdef P0RW2_TRACE
  g_trace = p.trace;
#endif

  if (warp < 16) {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(CR));
    const int s = (warp / 4) & 1, hh = warp / 8, k = lane & 3;
    const int m_block = 2 * mp + s;
    const int rA = m_block * 128 + quad * 32 + 16 * hh + lane / 4, rB = rA + 8;
    const int rAc = min(rA, p.N - 1), rBc = min(rB, p.N - 1);
    const uint32_t la = (uint32_t)(quad * 32 + 16 * hh) << 16;
    const uint32_t cS = tb + la + 128u * s, cD = tb + la + 256u + 128u * s;
    const size_t hrow = (size_t)(b * p.H + h);
    const float qsv = p.qs[hrow * (p.N / 32) + rAc / 32];          // rA, rB share the 32-row scale block
    const float dosv = p.dos[hrow * (p.N / 32) + rAc / 32];
    const float l2A = fmul(p.lse[hrow * p.N + rAc], LOG2E), l2B = fmul(p.lse[hrow * p.N + rBc], LOG2E);
    const unsigned long long nlA = pk(fneg(l2A), fneg(l2A)), nlB = pk(fneg(l2B), fneg(l2B));
    const float* ksr = p.ks + (size_t)(b * p.Hk + hk) * (p.N / 32);
    const float* vsr = p.vs + (size_t)(b * p.Hk + hk) * (p.N / 32);
    unsigned long long snA = 0, enA = 0, sdA = 0, edA = 0, snB = 0, enB = 0, sdB = 0, edB = 0;
    float r[32];
    float pks[4], pvs[4];   // raw K/V scales of the current tile, prefetched one tile ahead (as +pf)
#pragma unroll
    for (int j = 0; j < 4; ++j) { pks[j] = ksr[4 * (n_max - 1) + j]; pvs[j] = vsr[4 * (n_max - 1) + j]; }
    for (int it = 0; it < n_max; ++it) {
      const int n = n_max - 1 - it;
      float cj[4], aj[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) { cj[j] = fmul(fmul(pks[j], qsv), LOG2E); aj[j] = fmul(pvs[j], dosv); }
      const int nn = n > 0 ? n - 1 : 0;
#pragma unroll
      for (int j = 0; j < 4; ++j) { pks[j] = ksr[4 * nn + j]; pvs[j] = vsr[4 * nn + j]; }
      const bool diag = p.causal && (n + 1) * 128 > m_block * 128;
      const int kbase = n * 128;
      unsigned long long denA = 0, denB = 0, numA = 0, numB = 0;
      if (diag) tile_body<true>(bars, r, cj, aj, cS, cD, s, it, lane, k, kbase, rA, rB, nlA, nlB, denA, denB, numA, numB);
      else tile_body<false>(bars, r, cj, aj, cS, cD, s, it, lane, k, kbase, rA, rB, nlA, nlB, denA, denB, numA, numB);
      two_sum2(snA, enA, numA); two_sum2(sdA, edA, sum4l(denA, k));
      two_sum2(snB, enB, numB); two_sum2(sdB, edB, sum4l(denB, k));
    }
    if (k == 0) {
      float a0, a1, e0, e1, c0, c1, f0, f1;
      upk(snA, a0, a1); upk(enA, e0, e1); upk(sdA, c0, c1); upk(edA, f0, f1);
      double tn = ((double)a0 + (double)a1) + ((double)e0 + (double)e1), td = ((double)c0 + (double)c1) + ((double)f0 + (double)f1);
      if (rA < p.N) p.delta[hrow * p.N + rA] = td != 0.0 ? (float)(tn / td) : 0.f;
      upk(snB, a0, a1); upk(enB, e0, e1); upk(sdB, c0, c1); upk(edB, f0, f1);
      tn = ((double)a0 + (double)a1) + ((double)e0 + (double)e1); td = ((double)c0 + (double)c1) + ((double)f0 + (double)f1);
      if (rB < p.N) p.delta[hrow * p.N + rB] = td != 0.0 ? (float)(tn / td) : 0.f;
    }
  } else {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" :: "n"(OR));
    if (warp == 17 && lane == 0) {   // TMA loads: Q_0, Q_1, dO_0, dO_1 once; K/V ring over n descending
      mbar_expect_tx(&bars.qfull, 4 * 16384);
      for (int s = 0; s < 2; ++s) {
        tma4(sbase + SMEM_Q + 16384 * s, &tmQ, 0, (2 * mp + s) * 128, h, b, &bars.qfull);
        tma4(sbase + SMEM_DO + 16384 * s, &tmDO, 0, (2 * mp + s) * 128, h, b, &bars.qfull);
      }
      for (int it = 0; it < n_max; ++it) {
        const int n = n_max - 1 - it, slot = it % KVS;
        if (it >= KVS) TWAIT(&bars.kvempty[slot], ((it / KVS) - 1) & 1, 5);
#ifdef P0RW2_NOTMA
        mbar_arrive(&bars.kvfull[slot]);
#else
        mbar_expect_tx(&bars.kvfull[slot], 32768);
        tma4(sbase + SMEM_KV + KV_STRIDE * slot, &tmK, 0, n * 128, hk, b, &bars.kvfull[slot]);
        tma4(sbase + SMEM_KV + KV_STRIDE * slot + 16384, &tmV, 0, n * 128, hk, b, &bars.kvfull[slot]);
#endif
      }
    }
    if (warp == 16 && lane == 0) {   // MMA issue, order S_0, dP_0, S_1, dP_1 per key tile
      TWAIT(&bars.qfull, 0, 7);
      tc_after();
      for (int it = 0; it < n_max; ++it) {
        const int slot = it % KVS;
        if (it == 0) TWAIT(&bars.kvfull[slot], (it / KVS) & 1, 7);
        else TWAIT(&bars.kvfull[slot], (it / KVS) & 1, 3);
        tc_after();
        const uint32_t sK = sbase + SMEM_KV + KV_STRIDE * slot, sV = sK + 16384;
        for (int s = 0; s < 2; ++s) {
          if (it > 0) TACQ(&bars.sempty[s], (it - 1) & 1, 4);
          mma_128(tb + 128u * s, sbase + SMEM_Q + 16384 * s, sK);
          commit(&bars.sfull[s]);
          if (it > 0) TACQ(&bars.dempty[s], (it - 1) & 1, 4);
          mma_128(tb + 256u + 128u * s, sbase + SMEM_DO + 16384 * s, sV);
          commit(&bars.dfull[s]);
        }
        commit(&bars.kvempty[slot]);
      }
    }
  }
#ifdef P0RW2_TRACE
  if (warp < 16 && lane == 0) atomicAdd(&g_trace[6], clock64() - tcta0);
#endif
  tc_before();
  __syncthreads();
#if defined(P0RW2_TRACE) || defined(P0RW2_CYC)
  if (tid == 0) atomicAdd(&p.trace[0], clock64() - tcta0);
#endif
  if (warp == 0) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" :: "r"(tb));
}


// ------------------------------------------------------------------------------------------------------------------
// IMPL 1 ("today in C++", iteration c): the DSL's h2 schedule. 12 warps: 0-7 compute (stage s = w/4, one row per thread,
// row = 32q + lane, tcgen05.ld.32x32b), 8 MMA, 9 TMA load, 10-11 idle. Per tile and half h: S chunks 2h, 2h+1 -> exp/den
// -> (h == 0: wait dP) -> per chunk: dP load, num.  Registers: compute 208, other 64 (pool 12 x 168).
#define LD32X32(addr, r) asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];" \
  : "=f"(r[0]),"=f"(r[1]),"=f"(r[2]),"=f"(r[3]),"=f"(r[4]),"=f"(r[5]),"=f"(r[6]),"=f"(r[7]),"=f"(r[8]),"=f"(r[9]),"=f"(r[10]),"=f"(r[11]),"=f"(r[12]),"=f"(r[13]),"=f"(r[14]),"=f"(r[15]), \
    "=f"(r[16]),"=f"(r[17]),"=f"(r[18]),"=f"(r[19]),"=f"(r[20]),"=f"(r[21]),"=f"(r[22]),"=f"(r[23]),"=f"(r[24]),"=f"(r[25]),"=f"(r[26]),"=f"(r[27]),"=f"(r[28]),"=f"(r[29]),"=f"(r[30]),"=f"(r[31]) : "r"(addr))
#ifndef P0RW2_TW
#define P0RW2_TW 12
#endif
constexpr int T_NWARPS = P0RW2_TW, T_NTHR = T_NWARPS * 32;
#ifndef P0RW2_TCR
#define P0RW2_TCR 208
#endif
constexpr int T_CR = P0RW2_TCR;
constexpr int T_OR = T_NWARPS == 16 ? (T_CR == 224 ? 32 : 48) : 64;

// pair-typed TMEM load: 32 columns -> 16 packed f32x2 registers (mov.b64 inside the asm so ptxas keeps aligned pairs)
#define LDP16(addr, q) asm volatile("{\n.reg .b32 t<32>;\n" \
  "tcgen05.ld.sync.aligned.32x32b.x32.b32 {t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,t16,t17,t18,t19,t20,t21,t22,t23,t24,t25,t26,t27,t28,t29,t30,t31}, [%16];\n" \
  "mov.b64 %0, {t0,t1};\nmov.b64 %1, {t2,t3};\nmov.b64 %2, {t4,t5};\nmov.b64 %3, {t6,t7};\nmov.b64 %4, {t8,t9};\nmov.b64 %5, {t10,t11};\n" \
  "mov.b64 %6, {t12,t13};\nmov.b64 %7, {t14,t15};\nmov.b64 %8, {t16,t17};\nmov.b64 %9, {t18,t19};\nmov.b64 %10, {t20,t21};\n" \
  "mov.b64 %11, {t22,t23};\nmov.b64 %12, {t24,t25};\nmov.b64 %13, {t26,t27};\nmov.b64 %14, {t28,t29};\nmov.b64 %15, {t30,t31};\n}" \
  : "=l"(q[0]),"=l"(q[1]),"=l"(q[2]),"=l"(q[3]),"=l"(q[4]),"=l"(q[5]),"=l"(q[6]),"=l"(q[7]),"=l"(q[8]),"=l"(q[9]),"=l"(q[10]),"=l"(q[11]),"=l"(q[12]),"=l"(q[13]),"=l"(q[14]),"=l"(q[15]) \
  : "r"(addr))
#define LDP8(addr, q) asm volatile("{\n.reg .b32 t<16>;\n" \
  "tcgen05.ld.sync.aligned.32x32b.x16.b32 {t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15}, [%8];\n" \
  "mov.b64 %0, {t0,t1};\nmov.b64 %1, {t2,t3};\nmov.b64 %2, {t4,t5};\nmov.b64 %3, {t6,t7};\n" \
  "mov.b64 %4, {t8,t9};\nmov.b64 %5, {t10,t11};\nmov.b64 %6, {t12,t13};\nmov.b64 %7, {t14,t15};\n}" \
  : "=l"(q[0]),"=l"(q[1]),"=l"(q[2]),"=l"(q[3]),"=l"(q[4]),"=l"(q[5]),"=l"(q[6]),"=l"(q[7]) : "r"(addr))
// (P0, P1) = (ex2(x0), ex2(x1)) on a packed pair, result packed (ex2.approx.ftz.f32 each half)
__device__ __forceinline__ unsigned long long ex2p(unsigned long long x) {
  unsigned long long y;
  asm("{\n.reg .f32 a, b;\nmov.b64 {a, b}, %1;\nex2.approx.ftz.f32 a, a;\nex2.approx.ftz.f32 b, b;\nmov.b64 %0, {a, b};\n}" : "=l"(y) : "l"(x));
  return y;
}
// causal mask on a packed pair: element e (column col + e) -> -inf if col + e > row
__device__ __forceinline__ unsigned long long maskp(unsigned long long x, int col, int row) {
  float a, b; upk(x, a, b);
  if (col > row) a = -__int_as_float(0x7f800000);
  if (col + 1 > row) b = -__int_as_float(0x7f800000);
  return pk(a, b);
}

template <bool MASK>
__device__ __forceinline__ void tile_today(Bars& bars, unsigned long long* r2, const float* cj, const float* aj, uint32_t cS, uint32_t cD,
                                           int s, int it, int lane, int kbase, int row, unsigned long long nl,
                                           unsigned long long* den, unsigned long long& num) {
  TACQ(&bars.sfull[s], it & 1, 1);
#pragma unroll
  for (int hf = 0; hf < 2; ++hf) {
    LDP16(cS + 64 * hf, (&r2[32 * hf])); LDP16(cS + 64 * hf + 32, (&r2[32 * hf + 16]));
    wait_ld();
    if (hf == 1) rel(&bars.sempty[s], lane);
#pragma unroll
    for (int j = 2 * hf; j < 2 * hf + 2; ++j) {
      const unsigned long long cc = pk(cj[j], cj[j]);
#pragma unroll
      for (int v = 0; v < 16; ++v) {
        unsigned long long x = r2[16 * j + v];
        if (MASK) x = maskp(x, kbase + 32 * j + 2 * v, row);
        x = ex2p(ffma2(x, cc, nl));
        r2[16 * j + v] = x;
        den[v % 4] = fadd2(den[v % 4], x);
      }
    }
    if (hf == 0) TACQ(&bars.dfull[s], it & 1, 2);
#pragma unroll
    for (int j = 2 * hf; j < 2 * hf + 2; ++j) {
      unsigned long long acc[4] = {0, 0, 0, 0};
#ifdef P0RW2_DX16
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        unsigned long long d2[8];
        LDP8(cD + 32 * j + 16 * q, d2);
        wait_ld();
        if (j == 3 && q == 1) rel(&bars.dempty[s], lane);
#pragma unroll
        for (int v = 8 * q; v < 8 * q + 8; ++v) acc[v % 4] = ffma2(r2[16 * j + v], d2[v - 8 * q], acc[v % 4]);
      }
#else
      unsigned long long d2[16];
      LDP16(cD + 32 * j, d2);
      wait_ld();
      if (j == 3) rel(&bars.dempty[s], lane);
#pragma unroll
      for (int v = 0; v < 16; ++v) acc[v % 4] = ffma2(r2[16 * j + v], d2[v], acc[v % 4]);
#endif
      num = ffma2(fadd2(fadd2(acc[0], acc[1]), fadd2(acc[2], acc[3])), pk(aj[j], aj[j]), num);
    }
  }
}

__global__ void __launch_bounds__(T_NTHR, 1) k_today(const __grid_constant__ CUtensorMap tmQ, const __grid_constant__ CUtensorMap tmK,
                                                     const __grid_constant__ CUtensorMap tmV, const __grid_constant__ CUtensorMap tmDO,
                                                     const Params p) {
  extern __shared__ uint8_t smraw[];
  __shared__ __align__(8) Bars bars;
  __shared__ uint32_t s_taddr;
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, quad = warp % 4;
  const uint32_t sbase = (su32(smraw) + 1023u) & ~1023u;
  const int BH = p.B * p.H;
  const int mp = p.npairs - 1 - (int)(blockIdx.x / BH), bh = (int)(blockIdx.x % BH);
  const int b = bh / p.H, h = bh % p.H, hk = h / (p.H / p.Hk);
  const int NB = p.N / 128;
  int n_max = NB;
  if (p.causal) n_max = min(NB, 2 * mp + 2);
  if (warp == 0) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" :: "r"(su32(&s_taddr)));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  if (tid == 0) {
    mbar_init(&bars.qfull, 1);
#ifdef P0RW2_TWOISS
    for (int i = 0; i < KVS; ++i) { mbar_init(&bars.kvfull[i], 1); mbar_init(&bars.kvempty[i], 2); mbar_init(&bars.vfull[i], 1); }
#else
    for (int i = 0; i < KVS; ++i) { mbar_init(&bars.kvfull[i], 1); mbar_init(&bars.kvempty[i], 1); mbar_init(&bars.vfull[i], 1); }
#endif
    for (int s = 0; s < 2; ++s) { mbar_init(&bars.sfull[s], 1); mbar_init(&bars.sempty[s], 4); mbar_init(&bars.dfull[s], 1); mbar_init(&bars.dempty[s], 4); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  tc_before(); __syncthreads(); tc_after();
  const uint32_t tb = s_taddr;
#if defined(P0RW2_TRACE) || defined(P0RW2_CYC)
  const unsigned long long tcta0 = clock64();
#endif
#ifdef P0RW2_TRACE
  g_trace = p.trace;
#endif
  if (warp < 8) {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(T_CR));
    const int s = warp / 4;
    const int m_block = 2 * mp + s;
    const int row = m_block * 128 + quad * 32 + lane, rowc = min(row, p.N - 1);
    const uint32_t la = (uint32_t)(quad * 32) << 16;
    const uint32_t cS = tb + la + 128u * s, cD = tb + la + 256u + 128u * s;
    const size_t hrow = (size_t)(b * p.H + h);
    const float qsv = p.qs[hrow * (p.N / 32) + rowc / 32], dosv = p.dos[hrow * (p.N / 32) + rowc / 32];
    const float l2 = fmul(p.lse[hrow * p.N + rowc], LOG2E);
    const unsigned long long nl = pk(fneg(l2), fneg(l2));
    const float* ksr = p.ks + (size_t)(b * p.Hk + hk) * (p.N / 32);
    const float* vsr = p.vs + (size_t)(b * p.Hk + hk) * (p.N / 32);
    unsigned long long sn = 0, en = 0, sd = 0, ed = 0;
    float pks[4], pvs[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) { pks[j] = ksr[4 * (n_max - 1) + j]; pvs[j] = vsr[4 * (n_max - 1) + j]; }
    unsigned long long r[64];
    for (int it = 0; it < n_max; ++it) {
      const int n = n_max - 1 - it;
      float cj[4], aj[4];
#ifdef P0RW2_NOPF
#pragma unroll
      for (int j = 0; j < 4; ++j) { cj[j] = fmul(fmul(ksr[4 * n + j], qsv), LOG2E); aj[j] = fmul(vsr[4 * n + j], dosv); }
#else
#pragma unroll
      for (int j = 0; j < 4; ++j) { cj[j] = fmul(fmul(pks[j], qsv), LOG2E); aj[j] = fmul(pvs[j], dosv); }
      const int nn = n > 0 ? n - 1 : 0;
#pragma unroll
      for (int j = 0; j < 4; ++j) { pks[j] = ksr[4 * nn + j]; pvs[j] = vsr[4 * nn + j]; }
#endif
      const bool diag = p.causal && (n + 1) * 128 > m_block * 128;
      unsigned long long den[4] = {0, 0, 0, 0}, num = 0;
#ifdef P0RW2_NOCOMP
      TACQ(&bars.sfull[s], it & 1, 1); rel(&bars.sempty[s], lane); TACQ(&bars.dfull[s], it & 1, 2); rel(&bars.dempty[s], lane);
      if (it < 0)
#endif
      if (diag) tile_today<true>(bars, r, cj, aj, cS, cD, s, it, lane, n * 128, row, nl, den, num);
      else tile_today<false>(bars, r, cj, aj, cS, cD, s, it, lane, n * 128, row, nl, den, num);
      two_sum2(sn, en, num);
      two_sum2(sd, ed, fadd2(fadd2(den[0], den[1]), fadd2(den[2], den[3])));
    }
    float a0, a1, e0, e1, c0, c1, f0, f1;
    upk(sn, a0, a1); upk(en, e0, e1); upk(sd, c0, c1); upk(ed, f0, f1);
    const double tn = ((double)a0 + (double)a1) + ((double)e0 + (double)e1), td = ((double)c0 + (double)c1) + ((double)f0 + (double)f1);
    if (row < p.N) p.delta[hrow * p.N + row] = td != 0.0 ? (float)(tn / td) : 0.f;
  } else {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" :: "n"(T_OR));
    if (warp == 9 && lane == 0) {
      mbar_expect_tx(&bars.qfull, 4 * 16384);
      for (int s = 0; s < 2; ++s) {
        tma4(sbase + SMEM_Q + 16384 * s, &tmQ, 0, (2 * mp + s) * 128, h, b, &bars.qfull);
        tma4(sbase + SMEM_DO + 16384 * s, &tmDO, 0, (2 * mp + s) * 128, h, b, &bars.qfull);
      }
      for (int it = 0; it < n_max; ++it) {
        const int n = n_max - 1 - it, slot = it % KVS;
        if (it >= KVS) TWAIT(&bars.kvempty[slot], ((it / KVS) - 1) & 1, 5);
        mbar_expect_tx(&bars.kvfull[slot], 16384);
        tma4(sbase + SMEM_KV + KV_STRIDE * slot, &tmK, 0, n * 128, hk, b, &bars.kvfull[slot]);
        mbar_expect_tx(&bars.vfull[slot], 16384);
        tma4(sbase + SMEM_KV + KV_STRIDE * slot + 16384, &tmV, 0, n * 128, hk, b, &bars.vfull[slot]);
      }
    }
#ifdef P0RW2_TWOISS
    // two issuers: warp 8 issues S (K slots), warp 10 issues dP (V slots); each commits the slot's empty barrier (count 2)
    if ((warp == 8 || warp == 10) && lane == 0) {
      const bool isS = warp == 8;
      TWAIT(&bars.qfull, 0, 7);
      tc_after();
      for (int it = 0; it < n_max; ++it) {
        const int slot = it % KVS;
        if (isS) TWAIT(&bars.kvfull[slot], (it / KVS) & 1, 3); else TWAIT(&bars.vfull[slot], (it / KVS) & 1, 3);
        tc_after();
        const uint32_t sK = sbase + SMEM_KV + KV_STRIDE * slot, sV = sK + 16384;
        for (int s = 0; s < 2; ++s) {
          if (isS) {
            if (it > 0) TACQ(&bars.sempty[s], (it - 1) & 1, 4);
            mma_128(tb + 128u * s, sbase + SMEM_Q + 16384 * s, sK);
            commit(&bars.sfull[s]);
          } else {
            if (it > 0) TACQ(&bars.dempty[s], (it - 1) & 1, 4);
            mma_128(tb + 256u + 128u * s, sbase + SMEM_DO + 16384 * s, sV);
            commit(&bars.dfull[s]);
          }
        }
        commit(&bars.kvempty[slot]);
      }
    }
#else
    if (warp == 8 && lane == 0) {
      TWAIT(&bars.qfull, 0, 7);
      tc_after();
      for (int it = 0; it < n_max; ++it) {
        const int slot = it % KVS;
        TWAIT(&bars.kvfull[slot], (it / KVS) & 1, 3);
        tc_after();
        const uint32_t sK = sbase + SMEM_KV + KV_STRIDE * slot, sV = sK + 16384;
        for (int s = 0; s < 2; ++s) {
          if (it > 0) TACQ(&bars.sempty[s], (it - 1) & 1, 4);
          mma_128(tb + 128u * s, sbase + SMEM_Q + 16384 * s, sK);
          commit(&bars.sfull[s]);
        }
        TWAIT(&bars.vfull[slot], (it / KVS) & 1, 3);
        tc_after();
        for (int s = 0; s < 2; ++s) {
          if (it > 0) TACQ(&bars.dempty[s], (it - 1) & 1, 4);
          mma_128(tb + 256u + 128u * s, sbase + SMEM_DO + 16384 * s, sV);
          commit(&bars.dfull[s]);
        }
        commit(&bars.kvempty[slot]);
      }
    }
#endif
  }
#if defined(P0RW2_TRACE) || defined(P0RW2_CYC)
  if (tid == 0) atomicAdd(&p.trace[0], clock64() - tcta0);
#endif
  tc_before();
  __syncthreads();
  if (warp == 0) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" :: "r"(tb));
}

// IMPL 2: k_today made persistent: grid = #SMs, static striding over the LPT-ordered work-tile list (heaviest pair
// first, then (b, h)); all barrier phases run on CTA-global counters; Q/dO reloaded per work tile after qempty.
__device__ __forceinline__ void work_decode(const Params& p, int w, int& mp, int& b, int& h, int& hk, int& n_max) {
  const int BH = p.B * p.H;
  mp = p.npairs - 1 - w / BH;
  const int bh = w % BH;
  b = bh / p.H; h = bh % p.H; hk = h / (p.H / p.Hk);
  const int NB = p.N / 128;
  n_max = p.causal ? min(NB, 2 * mp + 2) : NB;
}
__global__ void __launch_bounds__(T_NTHR, 1) k_pers(const __grid_constant__ CUtensorMap tmQ, const __grid_constant__ CUtensorMap tmK,
                                                    const __grid_constant__ CUtensorMap tmV, const __grid_constant__ CUtensorMap tmDO,
                                                    const Params p) {
  extern __shared__ uint8_t smraw[];
  __shared__ __align__(8) Bars bars;
  __shared__ uint32_t s_taddr;
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, quad = warp % 4;
  const uint32_t sbase = (su32(smraw) + 1023u) & ~1023u;
  const int ntiles = p.B * p.H * p.npairs;
  if (warp == 0) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" :: "r"(su32(&s_taddr)));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  if (tid == 0) {
    mbar_init(&bars.qfull, 1); mbar_init(&bars.qempty, 1);
    for (int i = 0; i < KVS; ++i) { mbar_init(&bars.kvfull[i], 1); mbar_init(&bars.kvempty[i], 1); mbar_init(&bars.vfull[i], 1); }
    for (int s = 0; s < 2; ++s) { mbar_init(&bars.sfull[s], 1); mbar_init(&bars.sempty[s], 4); mbar_init(&bars.dfull[s], 1); mbar_init(&bars.dempty[s], 4); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  tc_before(); __syncthreads(); tc_after();
  const uint32_t tb = s_taddr;
#if defined(P0RW2_TRACE) || defined(P0RW2_CYC)
  const unsigned long long tcta0 = clock64();
#endif
  if (warp < 8) {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(T_CR));
    const int s = warp / 4;
    const uint32_t la = (uint32_t)(quad * 32) << 16;
    const uint32_t cS = tb + la + 128u * s, cD = tb + la + 256u + 128u * s;
    unsigned long long r[64];
    int gi = 0;   // CTA-global key-tile counter (S/dP slot phases)
    for (int w = blockIdx.x; w < ntiles; w += gridDim.x) {
      int mp, b, h, hk, n_max; work_decode(p, w, mp, b, h, hk, n_max);
      const int m_block = 2 * mp + s;
      const int row = m_block * 128 + quad * 32 + lane, rowc = min(row, p.N - 1);
      const size_t hrow = (size_t)(b * p.H + h);
      const float qsv = p.qs[hrow * (p.N / 32) + rowc / 32], dosv = p.dos[hrow * (p.N / 32) + rowc / 32];
      const float l2 = fmul(p.lse[hrow * p.N + rowc], LOG2E);
      const unsigned long long nl = pk(fneg(l2), fneg(l2));
      const float* ksr = p.ks + (size_t)(b * p.Hk + hk) * (p.N / 32);
      const float* vsr = p.vs + (size_t)(b * p.Hk + hk) * (p.N / 32);
      unsigned long long sn = 0, en = 0, sd = 0, ed = 0;
      float pks[4], pvs[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) { pks[j] = ksr[4 * (n_max - 1) + j]; pvs[j] = vsr[4 * (n_max - 1) + j]; }
      for (int it = 0; it < n_max; ++it, ++gi) {
        const int n = n_max - 1 - it;
        float cj[4], aj[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) { cj[j] = fmul(fmul(pks[j], qsv), LOG2E); aj[j] = fmul(pvs[j], dosv); }
        const int nn = n > 0 ? n - 1 : 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) { pks[j] = ksr[4 * nn + j]; pvs[j] = vsr[4 * nn + j]; }
        const bool diag = p.causal && (n + 1) * 128 > m_block * 128;
        unsigned long long den[4] = {0, 0, 0, 0}, num = 0;
        if (diag) tile_today<true>(bars, r, cj, aj, cS, cD, s, gi, lane, n * 128, row, nl, den, num);
        else tile_today<false>(bars, r, cj, aj, cS, cD, s, gi, lane, n * 128, row, nl, den, num);
        two_sum2(sn, en, num);
        two_sum2(sd, ed, fadd2(fadd2(den[0], den[1]), fadd2(den[2], den[3])));
      }
      float a0, a1, e0, e1, c0, c1, f0, f1;
      upk(sn, a0, a1); upk(en, e0, e1); upk(sd, c0, c1); upk(ed, f0, f1);
      const double tn = ((double)a0 + (double)a1) + ((double)e0 + (double)e1), td = ((double)c0 + (double)c1) + ((double)f0 + (double)f1);
      if (row < p.N) p.delta[hrow * p.N + row] = td != 0.0 ? (float)(tn / td) : 0.f;
    }
  } else {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" :: "n"(T_OR));
    if (warp == 9 && lane == 0) {   // loads
      int gk = 0, wt = 0;
      for (int w = blockIdx.x; w < ntiles; w += gridDim.x, ++wt) {
        int mp, b, h, hk, n_max; work_decode(p, w, mp, b, h, hk, n_max);
        if (wt > 0) mbar_wait(&bars.qempty, (wt - 1) & 1);
        mbar_expect_tx(&bars.qfull, 4 * 16384);
        for (int s = 0; s < 2; ++s) {
          tma4(sbase + SMEM_Q + 16384 * s, &tmQ, 0, (2 * mp + s) * 128, h, b, &bars.qfull);
          tma4(sbase + SMEM_DO + 16384 * s, &tmDO, 0, (2 * mp + s) * 128, h, b, &bars.qfull);
        }
        for (int it = 0; it < n_max; ++it, ++gk) {
          const int n = n_max - 1 - it, slot = gk % KVS;
          if (gk >= KVS) mbar_wait(&bars.kvempty[slot], ((gk / KVS) - 1) & 1);
          mbar_expect_tx(&bars.kvfull[slot], 16384);
          tma4(sbase + SMEM_KV + KV_STRIDE * slot, &tmK, 0, n * 128, hk, b, &bars.kvfull[slot]);
          mbar_expect_tx(&bars.vfull[slot], 16384);
          tma4(sbase + SMEM_KV + KV_STRIDE * slot + 16384, &tmV, 0, n * 128, hk, b, &bars.vfull[slot]);
        }
      }
    }
    if (warp == 8 && lane == 0) {   // MMA
      int gk = 0, wt = 0;
      for (int w = blockIdx.x; w < ntiles; w += gridDim.x, ++wt) {
        int mp, b, h, hk, n_max; work_decode(p, w, mp, b, h, hk, n_max);
        mbar_wait(&bars.qfull, wt & 1);
        tc_after();
        for (int it = 0; it < n_max; ++it, ++gk) {
          const int slot = gk % KVS;
          mbar_wait(&bars.kvfull[slot], (gk / KVS) & 1);
          tc_after();
          const uint32_t sK = sbase + SMEM_KV + KV_STRIDE * slot, sV = sK + 16384;
          for (int s = 0; s < 2; ++s) {
            if (gk > 0) acq(&bars.sempty[s], (gk - 1) & 1);
            mma_128(tb + 128u * s, sbase + SMEM_Q + 16384 * s, sK);
            commit(&bars.sfull[s]);
          }
          mbar_wait(&bars.vfull[slot], (gk / KVS) & 1);
          tc_after();
          for (int s = 0; s < 2; ++s) {
            if (gk > 0) acq(&bars.dempty[s], (gk - 1) & 1);
            mma_128(tb + 256u + 128u * s, sbase + SMEM_DO + 16384 * s, sV);
            commit(&bars.dfull[s]);
          }
          commit(&bars.kvempty[slot]);
        }
        commit(&bars.qempty);   // Q/dO of this work tile no longer read once these MMAs complete
      }
    }
  }
#if defined(P0RW2_TRACE) || defined(P0RW2_CYC)
  if (tid == 0) atomicAdd(&p.trace[0], clock64() - tcta0);
#endif
  tc_before();
  __syncthreads();
  if (warp == 0) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" :: "r"(tb));
}
}  // namespace p0rw2

// ---------------------------------------------------------------- host (C ABI, loaded with ctypes) -----------------
typedef CUresult (*EncodeFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                             const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                             CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static EncodeFn get_encode() {
  static EncodeFn fn = nullptr;
  if (!fn) {
    cudaDriverEntryPointQueryResult q;
    void* p = nullptr;
    if (cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q) != cudaSuccess) return nullptr;
    fn = (EncodeFn)p;
  }
  return fn;
}
// (B, N, H, 128) 1-byte rows -> 4D map, box 128 x 128 x 1 x 1, SWIZZLE_128B
static int make_map(CUtensorMap* m, void* ptr, int B, int N, int H) {
  const cuuint64_t dims[4] = {128, (cuuint64_t)N, (cuuint64_t)H, (cuuint64_t)B};
  const cuuint64_t strides[3] = {(cuuint64_t)H * 128, 128, (cuuint64_t)N * H * 128};
  const cuuint32_t box[4] = {128, 128, 1, 1}, es[4] = {1, 1, 1, 1};
  EncodeFn f = get_encode();
  if (!f) return -1;
  return (int)f(m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4, ptr, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

extern "C" int p0rw2_delta(void* q8, const float* qs, void* k8, const float* ks, void* v8, const float* vs, void* do8,
                           const float* dos, const float* lse, float* delta, int B, int N, int H, int Hk, int causal,
                           void* stream, unsigned long long* trace) {
  if (N % 128 || H % Hk) return -2;
  CUtensorMap tq, tk, tv, td;
  int e = make_map(&tq, q8, B, N, H) | make_map(&tk, k8, B, N, Hk) | make_map(&tv, v8, B, N, Hk) | make_map(&td, do8, B, N, H);
  if (e) return -3;
  p0rw2::Params p;
  p.qs = qs; p.ks = ks; p.vs = vs; p.dos = dos; p.lse = lse; p.delta = delta; p.trace = trace;
  p.B = B; p.N = N; p.H = H; p.Hk = Hk; p.causal = causal; p.npairs = (N / 128 + 1) / 2;
  static bool attr = false;
#ifndef P0RW2_IMPL
#define P0RW2_IMPL 0
#endif
  auto kern = P0RW2_IMPL == 2 ? p0rw2::k_pers : P0RW2_IMPL == 1 ? p0rw2::k_today : p0rw2::k_p0rw2;
  const int nthr = P0RW2_IMPL >= 1 ? p0rw2::T_NTHR : p0rw2::NTHR;
  if (!attr) {
    if (cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, p0rw2::SMEM_BYTES) != cudaSuccess) return -4;
    attr = true;
  }
  int grid = B * H * p.npairs;
  if (P0RW2_IMPL == 2) {
    static int nsm = 0;
    if (!nsm) { int dev; cudaGetDevice(&dev); cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev); }
    grid = grid < nsm ? grid : nsm;
  }
  kern<<<grid, nthr, p0rw2::SMEM_BYTES, (cudaStream_t)stream>>>(tq, tk, tv, td, p);
  return (int)cudaGetLastError();
}
