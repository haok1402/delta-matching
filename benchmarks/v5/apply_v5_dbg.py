"""Apply the residual-v5 DBG extension to a source tree (DBG builds only; production code paths are untouched).

    python apply_v5_dbg.py <tree> [--api-from <delta_matching_api.cpp>]

1. csrc/delta_matching_api.cpp (optionally replaced first by --api-from, e.g. HEAD's host file with the round-13
   twin-taking DBG op, for a tree that predates it): inside #ifdef FA3FP11_DBG only, the DBG row buffer grows by three
   fp64 planes placed right after `rows` in one allocation, and a new op bwd_fp8_t_dbg5 returns them as a 9th output
   rowsx [3, BH, nbn, 2, s_q] fp64: plane 0 = sum_j P^_j dP_j, plane 1 = sum_j P^_j, plane 2 = sum_j |P^_j dP_j|, per
   (n-tile, warpgroup, row), from pass 1's realized fp32 P^ (2^8 exp(S - lse), before e4m3) and fp32 dP_true =
   fl(dP_acc * do8s * v8s) (the same fp32 values pass 0 multiplies), products exact in fp64, sums in fp64.
2. csrc/mainloop_bwd_sm90_fp8r.hpp (hd128 pass 1) and csrc/mainloop_bwd_sm90_fp8v10.hpp (d256 pass 1): the writes of
   those planes, inside the existing `#ifdef FA3FP11_DBG ... if (params.ptr_dbg_rows != nullptr)` blocks.
"""
import argparse, os, re, shutil, sys

API_HELPER = r'''
// ## r21 residual-v5 (DBG only): rows and three fp64 planes in ONE allocation; the kernels locate the planes at
// ptr_dbg_rows + BH * nbn * 2 * s_q floats (8-byte aligned: that count is even).
static std::tuple<at::Tensor, at::Tensor> v5_rows_alloc(int64_t bh, int nbn, int sq, at::TensorOptions fopts) {
    int64_t const np = bh * nbn * 2 * int64_t(sq);
    at::Tensor buf = torch::zeros({7 * np}, fopts);
    at::Tensor rows = buf.narrow(0, 0, np).view({bh, nbn, 2, sq});
    at::Tensor rowsx = buf.narrow(0, np, 6 * np).view(at::kDouble).view({3, bh, nbn, 2, sq});
    return {rows, rowsx};
}
'''

ROWS_OLD = "at::Tensor rows = torch::zeros({int64_t(b) * h, nbn, 2, sq}, fopts);"
ROWS_NEW = "auto [rows, rowsx] = v5_rows_alloc(int64_t(b) * h, nbn, sq, fopts);"

T8 = "std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>"
T9 = "std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>"
SIG = '''(
    at::Tensor dout8, at::Tensor q8, at::Tensor k8, at::Tensor v8,
    at::Tensor lse, at::Tensor delta, at::Tensor kt8, at::Tensor qt8, at::Tensor dot8,
    at::Tensor q8s, at::Tensor k8s, at::Tensor v8s, at::Tensor do8s,
    double softmax_scale, bool is_causal
)'''

FP8R_ANCHOR = '''                    params.ptr_dbg_rows[gi] = sa;
                    params.ptr_dbg_rowsabs[gi] = sb;
                }
                cutlass::arch::NamedBarrier::sync(cutlass::NumThreadsPerWarpGroup, wg_barrier_id);
'''
FP8R_EXT = '''                // ## r21 residual-v5 (DBG only): fp64 sums of the SAME fp32 P^ / dP_true values (exact products):
                // plane 0 sum P^ dP_true, plane 1 sum P^, plane 2 sum |P^ dP_true|; 8 lanes then 4 warps (fixed order).
                {
                    int64_t const v5_np = int64_t(params.num_batch) * get<2>(params.shape_Q) * params.nbn * 2 * int64_t(seqlen_q);
                    double* const v5_g = reinterpret_cast<double*>(params.ptr_dbg_rows + v5_np);
                    double* const v5_s = reinterpret_cast<double*>(dbg_smem);   // 4 warps x kBlockM doubles = this WG's slot
                    #pragma unroll 1
                    for (int v5_pl = 0; v5_pl < 3; ++v5_pl) {
                        #pragma unroll
                        for (int mi = 0; mi < size<0>(dS); ++mi) {
                            double v5_a = 0.0;
                            #pragma unroll
                            for (int ni = 0; ni < size<1>(dS); ++ni) {
                                double const v5_p = double(scores(mi, ni));
                                double const v5_pd = v5_p * double(dS(mi, ni) * fq8_dp_descale);
                                v5_a += v5_pl == 0 ? v5_pd : (v5_pl == 1 ? v5_p : fabs(v5_pd));
                            }
                            #pragma unroll
                            for (int off = 4; off < 32; off *= 2) { v5_a += __shfl_xor_sync(0xffffffff, v5_a, off); }
                            if ((lane >> 2) == 0) {
                                int const row = get<1>(tScS_rc(mi, _0{}));
                                v5_s[warp_in_wg * kBlockM + row] = v5_a;
                            }
                        }
                        cutlass::arch::NamedBarrier::sync(cutlass::NumThreadsPerWarpGroup, wg_barrier_id);
                        if (wg_thread < kBlockM) {
                            double v5_t = 0.0;
                            #pragma unroll
                            for (int w = 0; w < 4; ++w) { v5_t += v5_s[w * kBlockM + wg_thread]; }
                            int64_t const gi = ((int64_t(fq8_bh) * params.nbn + n_block) * 2 + warp_group_idx) * int64_t(seqlen_q) + m_block * kBlockM + wg_thread;
                            v5_g[v5_pl * v5_np + gi] = v5_t;
                        }
                        cutlass::arch::NamedBarrier::sync(cutlass::NumThreadsPerWarpGroup, wg_barrier_id);
                    }
                }
'''

V10_A1 = "fq8_rb = flash::Allreduce<4>::run(fq8_rb, fq8_dbg_sum);\n"
V10_E1 = '''fq8_rb = flash::Allreduce<4>::run(fq8_rb, fq8_dbg_sum);
                    // ## r21 residual-v5 (DBG only): fp64 sums of the same fp32 P^ / dP_true (exact products), quad tree.
                    double v5_a0 = 0.0, v5_a1 = 0.0, v5_a2 = 0.0;
                    #pragma unroll
                    for (int ni = 0; ni < size<1>(dS); ++ni) {
                        double const v5_p = double(scores(mi, ni));
                        double const v5_pd = v5_p * double(dS(mi, ni) * fq8_dp_descale);
                        v5_a0 += v5_pd; v5_a1 += v5_p; v5_a2 += fabs(v5_pd);
                    }
                    #pragma unroll
                    for (int off = 2; off > 0; off /= 2) {
                        v5_a0 += __shfl_xor_sync(0xffffffff, v5_a0, off);
                        v5_a1 += __shfl_xor_sync(0xffffffff, v5_a1, off);
                        v5_a2 += __shfl_xor_sync(0xffffffff, v5_a2, off);
                    }
'''
V10_A2 = "params.ptr_dbg_rowsabs[fq8_gi] = fq8_rb;\n"
V10_E2 = '''params.ptr_dbg_rowsabs[fq8_gi] = fq8_rb;
                        {   // ## r21 residual-v5 (DBG only)
                            int64_t const v5_np = int64_t(params.num_batch) * get<2>(params.shape_Q) * params.nbn * 2 * int64_t(seqlen_q);
                            double* const v5_g = reinterpret_cast<double*>(params.ptr_dbg_rows + v5_np);
                            v5_g[fq8_gi] = v5_a0; v5_g[v5_np + fq8_gi] = v5_a1; v5_g[2 * v5_np + fq8_gi] = v5_a2;
                        }
'''


def sub_exact(s, old, new, count, what):
    n = s.count(old)
    if n != count:
        raise SystemExit(f"{what}: anchor found {n} times, expected {count}")
    return s.replace(old, new)


def patch_api(path):
    s = open(path).read()
    if "bwd_fp8_t_dbg5" in s:
        raise SystemExit(f"{path}: already patched")
    # helper right after the FA3FP11_DBG opening of the DBG ops section
    head = "#ifdef FA3FP11_DBG\n// ## FP11-M5 DBG ops"
    s = sub_exact(s, head, "#ifdef FA3FP11_DBG" + API_HELPER + "// ## FP11-M5 DBG ops", 1, "api DBG section")
    s = sub_exact(s, ROWS_OLD, ROWS_NEW, 2, "api rows alloc")
    # bwd_fp8_t_dbg -> bwd_fp8_t_dbg5 (9 outputs) + the unchanged 8-output wrapper
    old_def = T8 + " bwd_fp8_t_dbg" + SIG + " {"
    s = sub_exact(s, old_def, T9 + " bwd_fp8_t_dbg5" + SIG + " {", 1, "api bwd_fp8_t_dbg def")
    s = sub_exact(s, "    return {dq, dk, dv, rows, rowsabs, cks, ds8, psi};\n}\n",
                  "    return {dq, dk, dv, rows, rowsabs, cks, ds8, psi, rowsx};\n}\n\n"
                  + T8 + " bwd_fp8_t_dbg" + SIG + " {\n"
                  "    auto r = bwd_fp8_t_dbg5(dout8, q8, k8, v8, lse, delta, kt8, qt8, dot8, q8s, k8s, v8s, do8s, softmax_scale, is_causal);\n"
                  "    return {std::get<0>(r), std::get<1>(r), std::get<2>(r), std::get<3>(r), std::get<4>(r), std::get<5>(r),\n"
                  "            std::get<6>(r), std::get<7>(r)};\n}\n", 1, "api bwd_fp8_t_dbg return")
    # bwd_fp8_dbg keeps its 6 outputs (rowsx unused there but allocated so the kernel's writes stay in bounds)
    s = sub_exact(s, "    return {dq, dk, dv, rows, rowsabs, cks};\n}\n",
                  "    (void)rowsx;\n    return {dq, dk, dv, rows, rowsabs, cks};\n}\n", 1, "api bwd_fp8_dbg return")
    sch = ('    m.def("bwd_fp8_t_dbg(Tensor dout8, Tensor q8, Tensor k8, Tensor v8, Tensor lse, Tensor delta,"\n'
           '          " Tensor kt8, Tensor qt8, Tensor dot8, Tensor q8s, Tensor k8s, Tensor v8s, Tensor do8s,"\n'
           '          " float softmax_scale, bool is_causal) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");\n')
    s = sub_exact(s, sch, sch + sch.replace("bwd_fp8_t_dbg(", "bwd_fp8_t_dbg5(").replace(
        "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)",
        "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)"), 1, "api schema")
    s = sub_exact(s, '    m.impl("bwd_fp8_t_dbg", &bwd_fp8_t_dbg);\n',
                  '    m.impl("bwd_fp8_t_dbg", &bwd_fp8_t_dbg);\n    m.impl("bwd_fp8_t_dbg5", &bwd_fp8_t_dbg5);\n', 1, "api impl")
    open(path, "w").write(s)


def patch_fp8r(path):
    s = open(path).read()
    s = sub_exact(s, FP8R_ANCHOR, FP8R_ANCHOR + FP8R_EXT, 1, "fp8r DBG rows block")
    open(path, "w").write(s)


def patch_v10(path):
    s = open(path).read()
    s = sub_exact(s, V10_A1, V10_E1, 2, "v10 Allreduce anchor")
    s = sub_exact(s, V10_A2, V10_E2, 2, "v10 rowsabs store anchor")
    open(path, "w").write(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tree")
    ap.add_argument("--api-from", default=None)
    a = ap.parse_args()
    c = os.path.join(a.tree, "csrc")
    if a.api_from:
        shutil.copyfile(a.api_from, os.path.join(c, "delta_matching_api.cpp"))
    patch_api(os.path.join(c, "delta_matching_api.cpp"))
    patch_fp8r(os.path.join(c, "mainloop_bwd_sm90_fp8r.hpp"))
    patch_v10(os.path.join(c, "mainloop_bwd_sm90_fp8v10.hpp"))
    print("patched", a.tree)


if __name__ == "__main__":
    sys.exit(main())
