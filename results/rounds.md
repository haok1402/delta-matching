# Rounds log (kda-sm90)

Every kernel hypothesis, accepted or not. Gate: pytest; byte equivalence against the frozen baseline
(`benchmarks/equivalence.py`, four routes); >= 3% on the target kernel's affected main-grid subset in a paired,
order-alternated A/B (`benchmarks/ab.py`, 3 reps), confirmed by a second fresh job; no shape over +1% fwd+bwd beyond its
own repetition spread (main and held-out grids); bench.py D128 / D256 not worse than 1%; no new spills / stack /
serialization.

Stop rule (Hao, after round 1): per kernel. After two consecutive failed profile-based rounds on one kernel, move to the
next kernel in the waste ranking; stop when each of the top four kernels (pass 1, forward, pass 2, torch glue) has two
consecutive failures or is within 10% of its modeled limit. Non-bitwise changes are allowed when the numerics contract
holds (AC-2.3). Hypotheses are based on a profile of the kernel.

| kernel | consecutive failed rounds | rounds |
|---|---:|---|
| pass 1 | 1 (H1a, pre-profile) | 1 |
| forward | 0 | - |
| pass 2 | 0 | - |
| torch glue | 0 | - |

## Round 1: H1a, pass 1 hd128, dV gemm issued before the cross-warpgroup psi barrier: REJECTED

- Change (`results/r1/h1a.patch`, `csrc/mainloop_bwd_sm90_fp8r.hpp`): after the dS line (the last reader of the fp32 P
  fragment), the P~ conversion, the dV window rescale / f_prev update, the dOt wait and the dV RS wgmma are issued before
  the psi amax / named barrier; the dK wgmma is issued after the dS' conversion as before; FIFO retirement unchanged
  (wait<1> retires dV then releases dOt, wait<0> retires dK). wgmma ledger per tile: S, dP (wait<1>, wait<0>), dV,
  dK (wait<1>, wait<0>), same groups and order as the baseline, dV issued earlier.
- Resources: REG 168, STACK 0 (causal) / 8 (non-causal), LOCAL 0 for the two pass-1 instantiations: identical to the
  baseline library (which also has STACK 8 on the non-causal instantiation); no ptxas spill / serialization
  diagnostics.
- Correctness: pytest 49 passed / 3 skipped; equivalence vs the frozen baseline, default route, 72 cases: BITWISE
  EQUAL (library sha256 36f86c61... vs 07fee0f2...).
- Performance (job 21157817, `results/r1/ab_main_full`, `results/r1/ab_holdout_full`): pass 1 on the D128 subset
  245.518 -> 242.412 ms (-1.27%, spreads 0.43% / 0.60%); pass 1 over the whole main grid -0.43%; e2e fwd+bwd D128
  -0.61%; held-out pass 1 -0.51%, e2e -0.20%, no held-out shape over the gate; two main shapes at +1.14% / +1.04%,
  both within their own repetition spread. bench.py N8192 matched fwd+bwd: D128 4.524 / 4.481 -> 4.460 / 4.448 ms,
  D256 8.750 / 8.724 -> 8.778 / 8.684 ms.
- Verdict: below the 3% gate on the target subset; reverted. Reading: hiding the psi barrier wait behind the dV gemm
  recovers about 1% of pass 1, so the per-tile cross-warpgroup synchronization is not pass 1's main cost; the
  per-element issue work (exp2, dS, conversions, the dS' transpose) is the more likely limiter.

## Round 2: pass-1 profile (no kernel change)

Nsight Compute is unavailable (RmProfilingAdminOnly=1 on the nodes), so pass 1 was profiled with a clock64 probe build
(`DM_PROF=1`, bitwise to the baseline, ~+20% overhead) plus what-if builds: `results/profile_pass1.md`, raw data in
`results/r2/`. Findings: pass 1 is issue-bound and serialized (tensor pipe busy <= 23-26% of each m-block step; the two
warpgroups run in lockstep), the MUFU exp2 floor is half the tensor ideal at hd128, the d256 single dO stage exposes
load latency (a second dO stage: -2.4 .. -5.0% d256 pass 1, bitwise), deeper hd128 stages are slower, and decoupling
the warpgroups (timing probe) is worth ~4-5% at hd128 before any ping-pong scheduling. Ranked profile-based hypotheses
are in the profile report; the next round takes P1-d256-DO2.
