# Forward kernel profile (round 7)

Target: the shipped forward instantiations (D128 128x256 tiles, D256 128x128 tiles, two stages, two consumer
warpgroups, PV-first step), D128 / D256, causal / full, MHA / GQA, B4 N8192. Nsight Compute is unavailable on this
cluster (RmProfilingAdminOnly=1), so the profile is built from (1) a sparse in-kernel trace, (2) static SASS of the
shipped build against FA3's fp8 forward, (3) timing-only ablations. Design diff vs FA3: `forward_design_diff.md`.

## 1. Sparse trace (DM_TRACE forward section): a failed diagnostic

`csrc/kda_prof.h` forward section + probes in `csrc/mainloop_fwd_sm90_fp8f.hpp` (commit ccaee99, accessor fix e1edb89),
decoder `benchmarks/trace_fwd.py` (validated decode: header, all consumer events present and ordered, stage / phase
range, >= 3 steps per sample). Default build of the instrumented source: SASS identical to DO2 on all 22 kernel
functions (per-function comparison, `results/r7/sass_cmp.py`; one anonymous-namespace name hash differs by build path).

The trace build is NOT usable as a timeline:

- overhead (forward kernel ms, 2 rotating reps, `results/r7/trace_overhead_*.json`): D128 causal 1.562 -> 1.839
  (+18%), D128 full 2.913 -> 3.239 (+11%), D256 causal 1.489 -> 2.209 (+48%), D256 full 2.911 -> 3.049 (+5%);
  resources: forward STACK 32 / 24 / 24 / 24 -> 48 / 272 / 40 / 96 bytes (REG 168, LOCAL 0);
- arithmetic: the trace build's forward O and LSE differ from DO2 on every one of the 18 case families of the
  default route (`results/r7/trace7_vs_do2_compare.txt`; e.g. d128_bhnd causal: 11 of 1048576 O elements, 951 of 8192
  LSE rows), so the probes change code generation, not only timing (most likely FMA contraction across the probe
  points). The pass-1 trace (round 4) was bitwise; this one is not.

Its decoded regions are kept for the record only (`results/r7/fwd_trace_summary_PERTURBED.md`); no conclusion below
depends on them.

## 2. Static SASS (accepted DO2 build vs FA3 fp8 forward)

See `forward_design_diff.md`. Per steady-state step of ours: four PV sub-gemm groups, each `WARPGROUP.ARRIVE`, 2 x
QGMMA, `WARPGROUP.DEPBAR.LE gsb0, 0x0`, then the rescale of the 64-register temp (64 FMUL.FTZ by the precomputed per-row
rho) before the next `ARRIVE`: the tensor pipe of this warpgroup is idle during every rescale. The four k-scale
`LDG.E` are issued after the QK retire. Whole-kernel opcode counts (D128 full): FMUL 828 vs FA3 140, FFMA 414 vs 232,
FMNMX 326 vs 230, MUFU.EX2 290 vs 226.

## 3. Ablations (timing only; `results/r7/forward_whatif.md`, `forward_whatif.json`, patch `results/r7/whatif.patch`)

One build per knob from the same source (knob 0 = DO2 source, matches the DO2 library within 0.3%), 3 rotating
repetitions, forward kernel ms, change vs knob 0:

| knob | D128 causal MHA | D128 causal GQA4 | D128 full | D256 causal MHA | D256 causal GQA4 | D256 full |
|---|---:|---:|---:|---:|---:|---:|
| 1 no S descale (FMUL + k-scale loads) | -4.6% | -3.4% | -9.0% | -2.9% | -3.4% | -3.2% |
| 2 no rho rescale between the PV sub-gemms | -8.1% | -8.1% | -7.7% | -19.9% | -19.8% | -20.2% |
| 3 knob 2 + no inner PV waits | -4.6% | -3.6% | -1.9% | -21.1% | -22.3% | -24.0% |
| 4 k-scale prefetch at step start (expected bitwise; no equivalence run) | -1.1% | -0.8% | +0.2% | -3.1% | -3.2% | -1.3% |

Reading:

- The rho rescales between the serialized PV sub-gemms are the largest single cost: ~20% of the D256 forward and ~8%
  of D128. Removing the waits as well adds little at D256 and is slower at D128, so the cost is the rescale work on
  the PV critical path, not the wait instructions themselves.
- The S descale costs 3-9% (largest at D128 full, where there is no mask work to share the step).
- The exposed k-scale load latency is worth ~3% at D256 causal and ~1% elsewhere.
- Knobs 1-3 change numerics: they time specific code-removal builds, not achievable speedups of a numerically valid
  implementation, and they do not show that any one design is the only way to recover them. Knob 4 is expected bitwise
  by construction (same operands) but was not run through the equivalence suite.

## Hypothesis selected (round 7): PVIL

Interleave two half-width PV temps as two independent rho chains so each temp's rescale overlaps the other's
sub-gemm (D128: NSPLIT 1 -> 2, one pair of 32-register temps; D256: NSPLIT 2 -> 4, two pairs). Register ledger for the
PV phase (fp32 regs / thread): D128 tOrO 64 + two temps 2 x 32 + P codes 32 = 160 (shipped 64 + 64 + 32 = 160);
D256 tOrO 128 + 2 x 32 + 16 = 208 (shipped 128 + 64 + 16 = 208); the QK phase is unchanged (temps dead). Dependency
ledger: each temp sees the shipped op sequence on its own elements (zero-init g0, rho1, g1, rho2, g2, rho3, g3,
promote with basis_last); `warpgroup_wait<1>` retires exactly the older of the two outstanding sub-gemms before its
temp is rescaled; the last temp's g3 stays in flight to the caller's `wait<0>` + `fq8_promote_last`, as shipped.
Expected bitwise (a wgmma output element does not depend on the atom N width; the M16 probe recorded identical
digits at NSPLIT 4). Certified flag set changes: FA3FP8F_TP_NSPLIT 2 -> 4, FA3FP11_HD128_NSPLIT 1 -> 2,
FA3FP11_FWD_PVIL 0 -> 1. Knob 2's removal timing (~20% D256 / ~8% D128) sizes the opportunity; it is not a bound
for this design.
