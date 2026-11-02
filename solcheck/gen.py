"""Synthetic step-kernel generator (kda-solcheck).

Emits a CUDA source with one persistent kernel (one CTA per SM) whose step loop performs a declared instruction mix:
wgmma groups (shape, SS / RS, accumulator, waits) and per-step counts of ALU / MUFU / LSU instructions, with no data
dependence on any real algorithm.  Modes:
  free : ALU work runs on private registers, never on accumulators; wgmma groups of step j are waited for one step
         later (wait_group 1): the tensor pipe never waits on ALU work (ideal overlap upper bound for the mix).
  dep  : the algorithm's structural dependences: ALU work after group g reads / rewrites g's accumulators (wait
         first), F2FP packs accumulators into the RS operands of the next group (the attention step's order).
The host main times `launches` launches with CUDA events and reads per-CTA clock64 / globaltimer stamps, so the
report has wall time, SM cycles and the effective SM clock.

    python solcheck/gen.py --cfg cfg.json --out dir      # writes dir/<name>.cu
"""
import argparse, json, os

WG_THREADS = 128


def acc_regs(n):
    return n // 2           # m64nN f32 accumulator registers per thread


def wgmma_ss(n, acc, zero, kidx, dA, dB):
    regs = acc_regs(n)
    outs = ", ".join(f"%{i}" for i in range(regs))
    cons = ", ".join(f'"+f"({acc}[{i}])' for i in range(regs))
    sd = f"%{regs + 2}" if zero == "rt" else ("0" if zero else "1")
    return (f'asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, {sd}, 0;\\n'
            f'wgmma.mma_async.sync.aligned.m64n{n}k32.f32.e4m3.e4m3 {{{outs}}}, %{regs}, %{regs+1}, p, 1, 1;\\n}}\\n"'
            f' : {cons} : "l"({dA} + {kidx * 2}), "l"({dB} + {kidx * 2}){", " + chr(34) + "r" + chr(34) + "(zflag)" if zero == "rt" else ""});')


def wgmma_rs(n, acc, zero, kidx, areg, dB):
    regs = acc_regs(n)
    outs = ", ".join(f"%{i}" for i in range(regs))
    cons = ", ".join(f'"+f"({acc}[{i}])' for i in range(regs))
    sd = f"%{regs + 5}" if zero == "rt" else ("0" if zero else "1")
    a = ", ".join(f"%{regs + i}" for i in range(4))
    ain = ", ".join(f'"r"({areg}[{4 * kidx + i}])' for i in range(4))
    return (f'asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, {sd}, 0;\\n'
            f'wgmma.mma_async.sync.aligned.m64n{n}k32.f32.e4m3.e4m3 {{{outs}}}, {{{a}}}, %{regs+4}, p, 1, 1;\\n}}\\n"'
            f' : {cons} : {ain}, "l"({dB} + {kidx * 2}){", " + chr(34) + "r" + chr(34) + "(zflag)" if zero == "rt" else ""});')


FENCE = 'asm volatile("wgmma.fence.sync.aligned;\\n" ::: "memory");'
COMMIT = 'asm volatile("wgmma.commit_group.sync.aligned;\\n" ::: "memory");'


def wait(n):
    return f'asm volatile("wgmma.wait_group.sync.aligned {n};\\n" ::: "memory");'


def fence_acc(acc, n):
    # keep the compiler from moving accumulator reads across the wait (cutlass warpgroup_fence_operand)
    return " ".join(f'asm volatile("" : "+f"({acc}[{i}]) :: "memory");' for i in range(acc_regs(n)))


# --------------------------------------------------------------------------------------------------- ALU emission
def op_code(op, dst, k):
    """One instruction of class op on float register expression dst (int register for int classes)."""
    c = "c%d" % (k % 2)
    if op == "FFMA":
        return f'asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"({dst}) : "f"({c}), "f"(c1));'
    if op == "FMUL":
        return f'asm volatile("mul.rn.f32 %0, %0, %1;" : "+f"({dst}) : "f"({c}));'
    if op == "FADD":
        return f'asm volatile("add.rn.f32 %0, %0, %1;" : "+f"({dst}) : "f"({c}));'
    if op == "FMNMX":
        return f'asm volatile("max.f32 %0, %0, %1;" : "+f"({dst}) : "f"({c}));'
    if op == "MUFU.EX2":
        return f'asm volatile("ex2.approx.ftz.f32 %0, %0;" : "+f"({dst}));'
    if op == "MUFU.RCP":
        return f'asm volatile("rcp.approx.ftz.f32 %0, %0;" : "+f"({dst}));'
    if op == "FSEL":
        return f'asm volatile("{{.reg .pred q; setp.ne.u32 q, %1, 0; selp.f32 %0, %0, %2, q;}}" : "+f"({dst}) : "r"(ione), "f"({c}));'
    if op == "SHFL":
        return f'asm volatile("shfl.sync.bfly.b32 %0, %0, %1, 31, -1;" : "+f"({dst}) : "r"(1 << ({k} % 5)));'
    if op == "LDS":      # LDS.64 whose two words feed two FFMAs (counted as LDS + 2 FFMA by cfgs.py)
        return (f'asm volatile("{{.reg .f32 t, u; ld.shared.v2.f32 {{t, u}}, [%1]; fma.rn.f32 %0, %0, t, %2; fma.rn.f32 %0, %0, u, %2;}}" '
                f': "+f"({dst}) : "r"(lds_addr + {(k * 264) % 8192}), "f"(c1));')
    if op == "STS":
        return f'asm volatile("st.shared.f32 [%0], %1;" :: "r"(sts_addr + {(k * 4) % 4096}), "f"({dst}) : "memory");'
    if op == "ALU":      # integer ALU-pipe filler (LOP3)
        return f'asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"({dst}) : "r"(ione), "r"(itwo));'
    if op == "IMAD":     # integer FMA-pipe filler
        return f'asm volatile("mad.lo.u32 %0, %0, %1, %2;" : "+r"({dst}) : "r"(ione), "r"(itwo));'
    raise ValueError(op)


INT_OPS = ("ALU", "IMAD")


def interleave(counts):
    """Spread op classes evenly: Bresenham-style merge by fractional position."""
    items = []
    for op, n in counts.items():
        for i in range(n):
            items.append(((i + 0.5) / n, op))
    items.sort()
    return [op for _, op in items]


def alu_block(counts, fregs, iregs, k0=0, f2fp=0, f2fp_src=None, f2fp_dst=None):
    """counts: {op: n}; fregs: float register expressions to cycle over; f2fp: number of half-packs from f2fp_src
    (float exprs) into f2fp_dst (uint32 exprs, two halves each)."""
    lines = []
    seq = interleave(counts)
    fi = ii = 0
    for k, op in enumerate(seq):
        if op in INT_OPS:
            lines.append(op_code(op, iregs[ii % len(iregs)], k0 + k)); ii += 1
        else:
            lines.append(op_code(op, fregs[fi % len(fregs)], k0 + k)); fi += 1
    for r in range((f2fp + 1) // 2):   # two cvt.rn.satfinite.e4m3x2.f32 (F2FP) per 32-bit destination
        d = f2fp_dst[r % len(f2fp_dst)]
        ns = len(f2fp_src)
        a, b, c, e = (f2fp_src[(4 * r + i + (4 * r) // ns) % ns] for i in range(4))
        lines.append(f'asm volatile("{{.reg .b16 l, h; cvt.rn.satfinite.e4m3x2.f32 l, %1, %2; cvt.rn.satfinite.e4m3x2.f32 h, %3, %4; '
                     f'mov.b32 %0, {{l, h}};}}" : "=r"({d}) : "f"({a}), "f"({b}), "f"({c}), "f"({e}));')
    return lines


RAND_FILL = r"""  const uint32_t zflag = (uint32_t)(out[5] != 0.f);   // 0 at run time: zero-init groups restart from 0 each step
  if (RAND_SMEM) {   // random e4m3 operands, |x| in [2^-2, 2^1) (tensor-core power depends on operand toggling)
    for (int i = threadIdx.x; i < 200 * 1024 / 4; i += NTHREADS) {
      uint32_t w = 0;
      #pragma unroll
      for (int b = 0; b < 4; ++b) {
        uint32_t h = (uint32_t)(i * 4 + b) * 2654435761u; h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
        uint32_t e = (h & 0x80u) | ((5u + ((h >> 8) & 3u)) << 3) | ((h >> 12) & 7u);
        w |= e << (8 * b);
      }
      reinterpret_cast<uint32_t*>(smem)[i] = w;
    }
    __syncthreads();
  }"""


# --------------------------------------------------------------------------------------------------- kernel
def kernel_src(cfg):
    name = cfg["name"]; nwg = cfg["wgs"]; mode = cfg["mode"]
    accs = cfg["accs"]                      # {acc_name: n}
    pregs = cfg.get("pregs", {})            # {name: count} uint32 RS operand registers
    groups = cfg["groups"]                  # [{"ops": [[n, "SS"|"RS", acc, zero_first, areg]], "wait_after": int|None,
                                            #   "alu": {op: n}, "f2fp": n, "f2fp_dst": name, "pre_alu_wait": int}]
    nthreads = nwg * WG_THREADS + (WG_THREADS if cfg.get("producer") else 0)
    L = []
    L.append("#include <cstdint>\n#include <cstdio>\n#include <cstdlib>\n#include <vector>\n#include <algorithm>\n#include <cuda_runtime.h>")
    L.append(f"#define NWG {nwg}\n#define NTHREADS {nthreads}\n#define RAND_SMEM {int(cfg.get('rand_smem', 1))}")
    L.append(f'extern "C" __global__ void __launch_bounds__(NTHREADS, 1) {name}(float* out, long long* stamps, int steps, int tiles) {{')
    L.append("  extern __shared__ __align__(1024) uint8_t smem[];")
    L.append("  const int wg = threadIdx.x / 128;")
    if cfg.get("producer"):
        L.append("  if (wg == NWG) { %s return; }" % (f'asm volatile("setmaxnreg.dec.sync.aligned.u32 {cfg["producer_regs"]};\\n");' if cfg.get("setmaxnreg") else ""))
        if cfg.get("setmaxnreg"):
            L.append(f'  asm volatile("setmaxnreg.inc.sync.aligned.u32 {cfg["setmaxnreg"]};\\n");')
    L.append("  const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(smem);")
    L.append(RAND_FILL)
    # descriptors: 128B swizzle, SBO 1024 B, LBO 16 B; distinct smem regions per operand role
    L.append("  auto mkdesc = [&](uint32_t off) -> uint64_t { uint32_t a = sbase + off; return (uint64_t)((a & 0x3FFFF) >> 4) | (1ull << 16) | (64ull << 32) | (1ull << 62); };")
    for i, d in enumerate(cfg.get("descs", ["dA", "dB", "dC"])):
        L.append(f"  uint64_t {d} = mkdesc({i * 32768} + wg * 0);")
    L.append("  float c0 = out[0] + 1.0f, c1 = out[1] + 0.5f;")
    L.append("  uint32_t ione = (uint32_t)(threadIdx.x | 1), itwo = (uint32_t)steps;")
    L.append("  uint32_t lds_addr = sbase + 98304 + (threadIdx.x % 128) * 8, sts_addr = sbase + 98304 + 16384 + wg * 4096 + (threadIdx.x % 128) * 4;")
    for a, n in accs.items():
        L.append(f"  float {a}[{acc_regs(n)}];")
        L.append(f"  #pragma unroll\n  for (int i = 0; i < {acc_regs(n)}; ++i) {a}[i] = 0.f;")
    fregs_decl = cfg.get("fregs", {}) if mode == "dep" else {}
    for a, n in fregs_decl.items():
        L.append(f"  float {a}[{n}];")
        L.append(f"  #pragma unroll\n  for (int i = 0; i < {n}; ++i) {a}[i] = c0 * (float)i;")
    for p, n in pregs.items():
        L.append(f"  uint32_t {p}[{n}];")
        L.append(f"  #pragma unroll\n  for (int i = 0; i < {n}; ++i) {p}[i] = (0x30303030u | ((i * 2654435761u + threadIdx.x * 40503u) & 0x87878787u)) ^ ((uint32_t)steps & 0x08080808u);")
    if mode == "free":
        L.append("  uint32_t Pscr[16];\n  #pragma unroll\n  for (int i = 0; i < 16; ++i) Pscr[i] = i;")
    nx = cfg.get("xregs_free", 32) if mode == "free" else cfg.get("xregs", 16) if mode == "free" or not any(g.get("reads") for g in groups) else 0
    ni = cfg.get("iregs", 4)
    L.append(f"  float X[{max(nx, 1)}]; uint32_t I[{ni}];")
    L.append(f"  #pragma unroll\n  for (int i = 0; i < {nx}; ++i) X[i] = c0 * (float)(i + threadIdx.x);")
    L.append(f"  #pragma unroll\n  for (int i = 0; i < {ni}; ++i) I[i] = threadIdx.x * 7 + i;")
    L.append("  { long long t0 = clock64(); unsigned long long g0; asm volatile(\"mov.u64 %0, %%globaltimer;\" : \"=l\"(g0)); "
             "if (threadIdx.x == 0) { stamps[4 * blockIdx.x] = t0; stamps[4 * blockIdx.x + 2] = (long long)g0; } }")
    if any(g.get("pp_sync") for g in groups):
        L.append('  if (wg == NWG - 1) asm volatile("bar.arrive 1, 256;" ::: "memory");')
    L.append("  __syncwarp();")
    L.append("  #pragma unroll 1\n  for (int it = 0; it < steps; ++it) {")
    iregs = [f"I[{i}]" for i in range(ni)]
    for gi, g in enumerate(groups):
        if g.get("only") is not None:
            L.append("    if (" + " || ".join(f"wg == {w}" for w in g["only"]) + ") {")
        if mode == "dep" and g.get("pre_wait") is not None:
            L.append("    " + wait(g["pre_wait"]))
        if g.get("bar"):
            L.append(f'    asm volatile("bar.sync %0, %1;" :: "r"({g["bar"][0]} + 0), "r"({g["bar"][1]}) : "memory");')
        if g.get("pp_sync"):     # FA3-style warpgroup ping-pong: wait for my turn (named barrier 1 + wg, 256 threads)
            L.append('    asm volatile("bar.sync %0, 256;" :: "r"(1 + wg) : "memory");')
        chunks = None
        if g.get("interleave") and mode == "free" and g.get("ops"):
            body = alu_block(g.get("alu", {}), [f"X[{i}]" for i in range(nx)], [f"I[{i}]" for i in range(ni)], k0=gi * 1000)
            k = len(g["ops"])
            chunks = [body[len(body) * i // k: len(body) * (i + 1) // k] for i in range(k)]
            g = dict(g, alu={})
        if g.get("ops"):
            L.append("    " + FENCE)
            for oi, (n, kind, acc, zero, areg) in enumerate(g["ops"]):
                if chunks:
                    L.extend("    " + x for x in chunks[oi])
                kidx = g.get("kbase", 0) + oi
                desc_b = g.get("descB", "dB")
                zi = cfg.get("zero_init", "rt")      # "rt": zero-init decided at run time (zflag = 0), opaque to ptxas
                zero = ("rt" if zi == "rt" else bool(zi)) if zero else False
                if kind == "SS":
                    L.append("    " + wgmma_ss(n, acc, zero if oi == 0 else False, kidx, g.get("descA", "dA"), desc_b))
                else:
                    L.append("    " + wgmma_rs(n, acc, zero if oi == 0 else False, oi if g.get("rs_k") is None else g["rs_k"][oi], areg, desc_b))
            L.append("    " + COMMIT)
        if g.get("pp_arrive"):   # hand the tensor core to the next warpgroup
            L.append('    asm volatile("bar.arrive %0, 256;" :: "r"(1 + (wg + 1) % NWG) : "memory");')
        if mode == "dep":
            if g.get("wait_after") is not None:
                L.append("    " + wait(g["wait_after"]))
                for a in g.get("reads", []):
                    L.append("    " + fence_acc(a, accs[a]))
            src = []
            for a in g.get("reads", []):
                src += [f"{a}[{i}]" for i in range(acc_regs(accs[a]))]
            fregs = src or [f"X[{i}]" for i in range(nx)]
            src = src or fregs
        else:
            fregs = [f"X[{i}]" for i in range(nx)]
            src = fregs
        dname = g.get("f2fp_dst") if mode == "dep" else "Pscr"
        dst = [f"{dname}[{i}]" for i in range(pregs.get(dname, 16))] if g.get("f2fp") else None
        for fd, fs, fn in (g.get("fold", []) if mode == "dep" else []):
            for i in range(fn):    # dst += src (the forward's O += PV fold)
                L.append(f'    asm volatile("add.rn.f32 %0, %0, %1;" : "+f"({fd}[{i}]) : "f"({fs}[{i}]));')
        if mode == "free" and g.get("fold"):
            g = dict(g, alu=dict(g.get("alu", {}), FADD=g.get("alu", {}).get("FADD", 0) + sum(f[2] for f in g["fold"])))
        if g.get("alu_regs") and mode == "dep":
            fregs = [f"{a}[{i}]" for a in g["alu_regs"] for i in range(acc_regs(accs[a]) if a in accs else cfg["fregs"][a])]
        for line in alu_block(g.get("alu", {}), fregs, iregs, k0=gi * 1000, f2fp=g.get("f2fp", 0), f2fp_src=src, f2fp_dst=dst):
            L.append("    " + line)
        if mode == "dep" and g.get("post_wait") is not None:
            L.append("    " + wait(g["post_wait"]))
        if g.get("only") is not None:
            L.append("    }")
    if mode == "free":
        L.append("    " + wait(cfg.get("free_depth", 1)))
    L.append("  }")
    L.append("  " + wait(0))
    L.append("  long long t1 = clock64(); unsigned long long g1; asm volatile(\"mov.u64 %0, %%globaltimer;\" : \"=l\"(g1));")
    L.append("  float s = 0.f;")
    for a, n in accs.items():
        L.append(f"  #pragma unroll\n  for (int i = 0; i < {acc_regs(n)}; ++i) s += {a}[i];")
    for a, n in fregs_decl.items():
        L.append(f"  #pragma unroll\n  for (int i = 0; i < {n}; ++i) s += {a}[i];")
    for p, n in pregs.items():
        L.append(f"  #pragma unroll\n  for (int i = 0; i < {n}; ++i) s += (float)({p}[i] & 7);")
    L.append(f"  #pragma unroll\n  for (int i = 0; i < {nx}; ++i) s += X[i];")
    if mode == "free":
        L.append("  #pragma unroll\n  for (int i = 0; i < 16; ++i) s += (float)(Pscr[i] & 7);")
    L.append(f"  #pragma unroll\n  for (int i = 0; i < {ni}; ++i) s += (float)(I[i] & 3);")
    L.append("  if (s == 1234.5f) out[4 + threadIdx.x] = s;")
    L.append("  if (threadIdx.x == 0) { stamps[4 * blockIdx.x + 1] = t1; stamps[4 * blockIdx.x + 3] = (long long)g1; }")
    L.append("}")
    return "\n".join(L)


HOST = r"""
int main(int argc, char** argv) {
  int steps = argc > 1 ? atoi(argv[1]) : 2000;
  int launches = argc > 2 ? atoi(argv[2]) : 20;
  int reps = argc > 3 ? atoi(argv[3]) : 5;
  int dev = 0; cudaDeviceProp p; cudaGetDeviceProperties(&p, dev);
  int sms = p.multiProcessorCount;
  int smem = 200 * 1024;
  cudaFuncSetAttribute(KNAME, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  float* out; long long* st; cudaMalloc(&out, 1 << 20); cudaMemset(out, 0, 1 << 20);
  cudaMalloc(&st, sizeof(long long) * 4 * sms);
  std::vector<long long> h(4 * sms);
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  if (steps <= 0) {           // auto: size one launch to about -steps ms (default 2 ms)
    float want = steps < 0 ? (float)-steps : 2.f, ms = 0.f; int probe = 64;
    for (;;) {
      cudaEventRecord(a); KNAME<<<sms, NTHREADS, smem>>>(out, st, probe, 0); cudaEventRecord(b); cudaEventSynchronize(b);
      cudaEventElapsedTime(&ms, a, b);
      if (ms > want / 8 || probe > (1 << 24)) break;
      probe *= 4;
    }
    steps = std::max(16, (int)(probe * want / ms));
  }
  for (int w = 0; w < 3; ++w) KNAME<<<sms, NTHREADS, smem>>>(out, st, steps, 0);
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) { printf("{\"error\": \"%s\"}\n", cudaGetErrorString(e)); return 1; }
  for (int r = 0; r < reps; ++r) {
    cudaEventRecord(a);
    for (int l = 0; l < launches; ++l) KNAME<<<sms, NTHREADS, smem>>>(out, st, steps, 0);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b);
    cudaMemcpy(h.data(), st, sizeof(long long) * 4 * sms, cudaMemcpyDeviceToHost);
    double cyc = 0, ns = 0; long long cmax = 0;
    for (int i = 0; i < sms; ++i) { long long c = h[4*i+1] - h[4*i]; cyc += c; cmax = std::max(cmax, c); ns += (double)(h[4*i+3] - h[4*i+2]); }
    cyc /= sms; ns /= sms;
    printf("{\"kernel\": \"%s\", \"rep\": %d, \"steps\": %d, \"launches\": %d, \"sms\": %d, \"ms\": %.6f, \"cyc_per_launch\": %.1f, "
           "\"cyc_max\": %lld, \"ns_last\": %.1f, \"mhz_last\": %.1f}\n",
           KSTR, r, steps, launches, sms, ms, cyc, cmax, ns, cyc / ns * 1e3);
  }
  e = cudaDeviceSynchronize();
  if (e != cudaSuccess) { printf("{\"error\": \"%s\"}\n", cudaGetErrorString(e)); return 1; }
  return 0;
}
"""


def source(cfg):
    return kernel_src(cfg) + f'\n#define KNAME {cfg["name"]}\n#define KSTR "{cfg["name"]}"\n' + HOST


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfgs = json.load(open(a.cfg))
    os.makedirs(a.out, exist_ok=True)
    for c in (cfgs if isinstance(cfgs, list) else [cfgs]):
        open(os.path.join(a.out, c["name"] + ".cu"), "w").write(source(c))
        print(c["name"])


if __name__ == "__main__":
    main()
