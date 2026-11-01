# apply one what-if variant to the working tree (reverted by the caller with git checkout)
import sys, re
v = sys.argv[1]
def sub(path, old, new):
    s = open(path).read(); assert old in s, (path, old); open(path, "w").write(s.replace(old, new))
if v == "d256_dost2":
    sub("csrc/flash_bwd_fp8v10_launch.h", "static constexpr int Stages_dO_d256 = 1;", "static constexpr int Stages_dO_d256 = 2;")
elif v == "hd128_dost2":
    sub("delta_matching/_buildcfg.py", '"FA3FP11_HD128_DOST": 1', '"FA3FP11_HD128_DOST": 2')
    sub("delta_matching/_buildcfg.py", '"FA3FP11_HD128_DOTST": 1', '"FA3FP11_HD128_DOTST": 2')
    for f in ("DOST", "DOTST"):
        s = open("csrc/dm_flagset.h").read()
        s = s.replace(f"(FA3FP11_HD128_{f}) != 1", f"(FA3FP11_HD128_{f}) != 2"); open("csrc/dm_flagset.h", "w").write(s)
elif v == "hd128_qst3":
    sub("delta_matching/_buildcfg.py", '"FA3FP11_HD128_QST": 2', '"FA3FP11_HD128_QST": 3')
    s = open("csrc/dm_flagset.h").read(); s = s.replace("(FA3FP11_HD128_QST) != 2", "(FA3FP11_HD128_QST) != 3"); open("csrc/dm_flagset.h", "w").write(s)
elif v == "hd128_wgpsi_probe":   # TIMING PROBE ONLY (numerics wrong): psi from this warpgroup's own 64-key half
    p = "csrc/mainloop_bwd_sm90_fp8r.hpp"
    sub(p, "cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<uint32_t>(BwdNamedBarriers::dQEmptyWG3) /*id 4*/);",
           "cutlass::arch::NamedBarrier::sync(cutlass::NumThreadsPerWarpGroup, wg_barrier_id);")
    sub(p, "for (int w = 0; w < NumMmaThreads / cutlass::NumThreadsPerWarp; ++w) { fq8_amax = fmaxf(fq8_amax, fq8_psi_slots[w]); }",
           "for (int w = 0; w < 4; ++w) { fq8_amax = fmaxf(fq8_amax, fq8_psi_slots[4 * warp_group_idx + w]); }")
else:
    raise SystemExit(f"unknown variant {v}")
print("applied", v)
