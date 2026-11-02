# kda-sm90: delta-matching fp8 attention on H100 — interim report

Status: interim (round 17; updated after every accepted round). Branch `kda-sm90`; accepted build `r17_pk` (kda-sm90
7a06674 + tooling). Every number below is traced to a committed file; per-round detail and every screened variant are in
`results/rounds.md`, `results/screens/` and the helpers' `CANDIDATES.md` branches; rules in
`docs/kda-sm90/rule-change-r10.md`.

## Scope and goal

- Method: the matched-delta (Delta-Matching) fp8 attention, D128, causal.
- Acceptance shapes (Hao, round 16): the training shapes of the paper's 1.67B GDN/GQA hybrid (Hq 16 / Hk 4) and
  Qwen3-30B-A3B attention (Hq 32 / Hk 4) at B2 N8192 and B1 N65536; equal weight (geometric mean of per-shape ratios).
- Acceptance: target kernel >= 3% faster in two fresh jobs, no shape slower beyond spread, bench within 1%; bitwise on
  72 cases x 4 routes, or (named numerics changes) the residual-v4 numerics contract on all four routes.
- Goal (Hao, round 17): forward, pass 0, pass 1, pass 2 each >= 90% of the corrected floor sol4c (below), on the
  fixed-reference calibration; first milestone: forward as fast as FA3's fp8 forward.

## Before / after (four acceptance shapes, summed medians, ms; original = main 3d4cd5b, now = r17_pk)

Source: `results/progress/ab.md` (job 21181967, 3 reps each, same job).

| kernel | original ms | now ms | change |
|---|---:|---:|---:|
| forward | 75.302 | 67.493 | -10.4% |
| pass 0 (matched delta) | 63.185 | 52.880 | -16.3% |
| pass 1 (dK / dV, dS') | 198.397 | 180.046 | -9.3% |
| pass 2 (dQ) | 41.889 | 37.873 | -9.6% |
| glue (copies / GQA fold) | 5.010 | 3.181 | -36.5% |
| producers fwd / bwd, twin | 2.452 | 2.489 | +1.5% |
| end-to-end forward | 77.476 | 67.362 | -13.1% |
| end-to-end fwd+bwd | 385.356 | 342.276 | -11.2% |

## Floors: sol4-v2 and sol4c (accepted build r17_pk, fixed-reference calibration 1481.2 TFLOP/s fp8, 3026 GB/s)

Source: `results/r17/accepted_pk/sol4c.md` (`benchmarks/sol4.py`).

| kernel | measured ms | sol4-v2 floor | % of sol4-v2 | sol4c floor | % of sol4c | target 90% |
|---|---:|---:|---:|---:|---:|---|
| pass 1 | 179.293 | 73.665 (tensor) | 41.1% | 118.815 | 66.3% | not met |
| forward | 67.089 | 36.815 (tensor) | 54.9% | 52.668 | 78.5% | not met |
| pass 0 | 52.715 | 36.920 (tensor) | 70.0% | 49.557 | 94.0% | met |
| pass 2 | 37.807 | 35.592 (bytes) | 94.1% | 35.592 | 94.1% | met |

- sol4-v2: per kernel the max of four floors: tensor (tile-granular FLOPs / calibrated fp8 GEMM peak), MUFU and issue
  (loop-only lower bounds from a control-flow ledger of the emitted step loops), bytes (route-aware compulsory traffic /
  calibrated copy bandwidth). Inputs are bound (calibration digest, library hash, exact grid).
- sol4c (Hao, after kda-solcheck): sol4-v2 / the fraction an IDEAL kernel with exactly the accepted step-loop
  instruction mix reaches on this H100 (forward 0.699, pass 1 0.620, pass 0 0.745; best of five schedules; tensor and
  ALU work couple at ~C_alu + 0.55 C_tensor; loaded clock 1.66-1.80 GHz; IPC <= ~3.5; origin/kda-sm90-solcheck
  results/solcheck/README.md). 90% of sol4-v2 itself is not reachable with these instruction mixes.
- The same-job fp8 GEMM calibration varies ~5% between jobs; goal attainment uses the highest observed peak (1481.2).

## FA3 and paper comparisons

- Forward kernel vs FA3's fp8 forward kernel on the four shapes: 1.155x (r17_pk; was 1.32x before round 17).
- Paper protocol (Appendix C.2; `results/paper_c2.md`, job 21181967), B2 N8192, TFLOP/s fwd / bwd / fwd+bwd:

| config | accepted | original | FA3 BF16 | cuDNN BF16 | paper (matched) |
|---|---|---|---|---|---|
| 16 / 4 | 582.8 / 486.4 / 510.5 | 516.4 / 442.3 / 461.0 | 697.0 / 499.7 / 543.6 | 590.3 / 483.5 / 509.6 | 661.0 / 421.5 / 470.2 |
| 32 / 4 | 639.0 / 500.3 / 533.3 | 564.4 / 452.6 / 479.6 | 723.1 / 519.0 / 564.5 | (see paper_c2.md) | - |

  The protocol's per-call forward window includes ~0.3 ms of idle-GPU launch latency (host work in the op is 0.015 ms);
  the forward kernel itself ran at 704 TFLOP/s already at r14_h3. The paper's FA3 BF16 numbers (260.7 fwd+bwd at 16/4)
  do not reproduce on this H100 (549 here), so its "1.80x FA3 BF16" ratio does not either; cuDNN BF16 matches the paper.

## Accepted changes (in order; details and evidence in results/rounds.md)

| round | change | kernel | gain (gate / confirmation) | class |
|---|---|---|---|---|
| 3 | DO2 (dO pipeline depth 2) | pass 1 | (historical grid) | bitwise |
| 9 | P2ST3: pass-2 hd128 pipeline depth 3 | pass 2 | -7.5% / -6.6% | bitwise |
| 10 | PSIPF: psi loaded one iteration early | pass 2 | -3.21% / -3.14% | bitwise |
| 12 | H1a + H1b stack: dV before the psi barrier, dS' words pre-packed | pass 1 | -4.03% / -3.99% | bitwise |
| 14 | H3: producers read Q / K / V through their own strides | glue | -67.75% / -67.75% | bitwise |
| 17 | C2: P0RTR transposed quad reduce + P0SPF 8-byte scale loads (kda-pass0) | pass 0 | -5.21% / -5.27% | bitwise |
| 17 | C3: P0LPT longest-first dispatch + P0W4 four warpgroups per CTA (kda-pass0) | pass 0 | -7.67% / -7.64% | bitwise |
| 17 | S17-p1-39: early dV rescale + deferred double-buffered dS' stores | pass 1 | -4.68% / -4.87% | bitwise |
| 17 | C3g: BM192 three-warpgroup forward + lazy O rescale + descale fold on unmasked steps (kda-fwd) | forward | -12.89% / -12.68% | named numerics |
| 17 | pass-0 stack: P0NML mask-free loop + register split + P0ALL one all-heads call (kda-pass0) | pass 0 | -3.19% / -3.43% | bitwise |
| 17 | EXPBX64: exp-line bias fold off the masked diagonal, passes 0 + 1 (kda-p1packed) | pass 0 | -4.50% / -4.04% | named numerics |

## What was tried and why it failed (summary; full ledgers in results/screens and the helpers' CANDIDATES.md)

- Forward (~90 variants incl. kda-fwd S1-S54): every bitwise variant other than the accepted ones within noise or
  slower; probes put the rho rescales at ~9% and the S descale at ~5% of the forward; scheduler-barrier alternatives
  +1..12%; QK splits and in-step branches spill at 160 registers; held bitwise stack PRETURN + k/v-scale staging -2.4%.
- Pass 1 (~180 variants incl. kda-pass1's ~135): PINGPONG (round 6), MUFU-poly v1 (+5.6%), per-warpgroup psi
  (pass 2's second accumulator +7..9%), freeing registers for a pipelined S / dP (kda-p1split, +3.5-4%); per element
  8 instructions remain, each fixed by the numerics contract, without an sm90 shortcut, neutral, or failing
  residual-v4 (EXPB in pass 1, PW16 / PW32, windowed dK); held SHFL8 -1.1% (bitwise).
- Pass 0: output-changing cuts (FOLD -13%, LAZY, DEFER, NORM) fail residual-v4 per-row non-regression; bitwise levers
  exhausted after the accepted ones; meets sol4c.
- Pass 2: at ~94% of its bytes floor since round 10.
- Tooling / rules: four-floor model sol4-v1 -> v2 (control-flow ledger, bound inputs, route-aware glue), the numerics
  contract residual-v2 -> v3 -> v4 (per-row non-regression at round-off plus the measured LSE +-1 ulp envelope,
  degenerate / ill-conditioned exemptions fixed by the accepted build, error tolerances 1e-5 matched / 1e-3 stale-full,
  mean noise 1e-6, inherited-defect repair), equivalence manifest v4.

## Open (round 18 onward, Hao's direction)

1. Structural pass-1 redesign reusing pass 0's work (pass 0 stores what pass 1 needs, or a two-phase pass 1 that
   computes the matched delta itself), from kda-fuse's measured structure.
2. Cheaper forms of the exp clamp and the fp32 dK fold as named numerics changes under residual-v4.
3. Completing the held stacks (forward -2.4%, pass 1 -1.1%) to >= 3%.
4. FA3 fp8 forward parity (now 1.155x).
