# S17-psi: per-warpgroup psi (HALFPSI) at head_dim 128: worker log

Base: d07cdcd (accepted build). A = .local/frozen/r14_h3. Every timing below comes from `screen.sh`: a 2-rep A/B on B2 N8192
Hk4 causal-matched, D128, using per-kernel medians. Noise is about +-1%. Correctness is the default-route compare against
.local/equiv/base3_default.pt (72 cases, the full suite).

## Design (as implemented; numerics are identical in 01-08)

- **Pass 1** (csrc/mainloop_bwd_sm90_fp8r.hpp, `#if FA3FP11_HALFPSI`):
  - psi per (tile, warpgroup half) = max over this warpgroup's own four warp slots `fq8_psi_slots[4*wg + w]`.
    The slots are parity-buffered, as before.
  - This sits behind the WG-local named barrier `wg_barrier_id` (ids 5 / 6). The CTA-wide id-4 barrier is gone from the
    per-tile path.
  - Published by `wg_thread == 0` at `ptr_psi[tile*2 + wg]`. The floors, the zero sentinel, the dK fold and the F2b anchor
    are unchanged; they were already per warpgroup.
- **Pass 2** (csrc/fq8_dq_pass2_hd128.h, `#if FA3FP11_HALFPSI`):
  - Two 64x128 fp32 accumulators: `acc` covers half 0 (keys 0-63, k-blocks 0-1) and `acc_h1` covers half 1 (k-blocks 2-3).
  - Each half has its own window psi (`psi_w0` / `psi_w1`). Per half there is an F2 sentinel and an F2b dominance check
    against the row-global psi_max over both halves. psi_max is snapshotted before the tile, so the two predicates are
    symmetric.
  - In-window ratio rescale per half. scale_d = 0 on the window's first taken tile.
  - Geometry-fixed promote into smem_dq in true units, in the order `acc*psi_w0` then `acc_h1*psi_w1`.
  - psi is read as float2 with the r10 prefetch. The M10 `__syncthreads` release is unconditional, as before.
  - The single-psi code is kept under `#else`.
- **Host / config:**
  - csrc/delta_matching_api.cpp: the psi tensor is `[BH, nbm, nbn, 2]` at hd128 through the existing
    `FA3FP11_HALFPSI && head_size == 128`. Both DBG ops (`bwd_fp8_t_dbg`, `bwd_fp8_dbg_ds`) return that same tensor; the
    DBG comment is updated.
  - delta_matching/_buildcfg.py: `-DFA3FP11_HALFPSI=1` in EXTRA_DEFINES.
  - csrc/dm_flagset.h: the guard now requires `FA3FP11_HALFPSI == 1`.
  - D256 (fq8_dq_pass2.h, the fp8v10 mainloop) is untouched. Its psi stays `[.., 1]`.
- Note: benchmarks/numerics.py (the Q2/Q3 gate) needs `psi_halves=2` for this build at hd128. I did not edit benchmarks/.

## Correctness (identical for 01-08: the same compare lines and the same worse pairs)

- Compare: 120 mismatching tensors (dq/dk only, 30 cases x {dq, dQ, dk, dK}). dv is bitwise.
- **Error vs fp32 larger in 2 (case, tensor) pairs, not 0**:
  - d128_bhnd/causal/matched dQ 0.0634922 -> 0.0635133 (+0.033%)
  - d128_bhnd/causal/stale dK 0.066883 -> 0.066908 (+0.037%)
- Full distribution (04, errs.py): of 144 pairs, 34 better, 108 equal, 2 worse. The geomean error ratio is 0.99943.
  - Best: d128_min/full/stale dQ -0.69%, d128_min/causal/matched dQ -0.44%, d128_min/causal/matched dK -0.38%.
  - So it is a net accuracy gain, but strictly it FAILS the "0 larger" rule, by +0.03%, at rounding-noise level.
- pytest tests/ (04): 49 passed, 3 skipped (2 need two GPUs, 1 needs flash_attn). This includes determinism and finite
  checks. 07: 49 passed, 3 skipped (S17-psi-07/pytest.txt).
- The dumps (default.pt) were removed from .local/screens by someone else before my cross-compare job ran. 04/05/07 are
  numerically identical by construction (schedule-only deltas), and their compare lines vs the baseline are identical.
  A direct 04-vs-07 dump compare was not run.

## Variants (pass1 / pass2 / pass1+pass2 ms / e2e, H16 and H32)

| id | change | ptxas | REG (pass 2) | H16 p1 / p2 / p1+p2 / e2e | H32 p1 / p2 / p1+p2 / e2e | verdict |
|---|---|---|---|---|---|---|
| 01 | first implementation: both halves predicated inside ONE wgmma batch, runtime scale_d | C7515: all pass-2 QGMMAs serialized | 178 | -1.20 / +15.75 / +2.02 / +2.25 | -1.74 / +16.11 / +1.56 / +2.48 | reject (serialized) |
| 02 | one complete stage per taken half, runtime scale_d, wait after the merge | C7515 | 178 | -1.08 / +12.53 / +1.52 / +0.95 | -0.84 / +13.33 / +1.80 / -1.04 | reject |
| 03 | unconditional 4-wgmma fast path, runtime scale_d | C7515 | 178/180 | -1.17 / +14.55 / +1.80 / +0.46 | -0.95 / +14.49 / +1.91 / +1.79 | reject |
| 04 | each (take0, take1, first0, first1) combination is a self-contained stage with COMPILE-TIME scale_d and wait<0> inside its branch (the baseline flash::gemm shape) | clean | 178/180 | -1.48 / +7.41 / +0.20 / +0.19 | -1.42 / +7.89 / +0.30 / +1.27 | correct, net neutral-to-slower |
| 05 | 04 + predicates, ratios and both rescales hoisted ahead of the TMA full-barrier wait | clean | 178/180 | -1.04 / +8.45 / +0.76 / +0.42 | -1.38 / +8.71 / +0.50 / +0.68 | no gain |
| 06 | 05 + pass-1 barrier merge: store drain before the psi barrier, first transpose barrier dropped | clean | 178/180 | +0.60 / +8.19 / +2.05 / +0.47 | +0.19 / +8.69 / +1.77 / +0.46 | reject (pass-1 gain lost) |
| 07 | 05 + M23c OVERLAP: half 1's rescale under half 0's in-flight wgmmas (healthy path) | clean | 178/180 | -1.24 / +8.06 / +0.53 / +0.08 | -1.82 / +8.60 / +0.12 / +1.69 | no gain |
| 08 | 07 + pass-1 early amax: warp amax + slot write moved before the dV block | clean | 178/180 | +2.29 / +8.51 / +3.47 / +2.78 | +2.31 / +8.79 / +3.52 / +1.99 | reject |
| 09 | 05 + power-of-two psi: pass 1 rounds psi up to 2^k, pass 2 skips ratio==1 rescales | clean | 178/180 | -0.72 / +3.88 / +0.15 / -0.30 | -2.02 / +3.74 / -0.95 / -0.63 | reject: numerics (many pairs worse by up to +1.4%, e.g. d128_gqa4/causal/stale dK 0.06466 -> 0.06558) |

Pass-1 REG stays 168. The pass-1 kernel's STACK goes 8 -> 0, as in round 6. Pass-2 REG goes from 114/116 (base) to
178/180, with STACK 0 throughout and no spills. Pass-2 occupancy is still smem-limited at 2 CTAs/SM.

Absolute ms for 04: H16 pass1 1.840 -> 1.813, pass2 0.430 -> 0.461; H32 pass1 3.693 -> 3.640, pass2 0.839 -> 0.905.

### Timing probes (numerically WRONG; timing attribution only; built on 05; no compare)

| id | probe | H16 p1 / p2 | H32 p1 / p2 |
|---|---|---|---|
| 90 | both in-register rescales removed | -1.25 / -0.12 | -1.13 / -0.26 |
| 91 | all 4 wgmmas into `acc` (acc_h1 still rescaled and promoted) | -1.16 / +4.53 | -1.01 / +4.83 |
| 92 | acc_h1 dropped from the promote | -1.21 / +6.64 | -0.88 / +7.23 |

Reading: the whole pass-2 regression is the second accumulator's per-tile 64-FMUL ratio rescale. Without the rescales,
pass 2 is at parity (90). Two wgmma chains and the doubled promote cost almost nothing. Hiding that rescale under the
other half's wgmma (07) or ahead of the TMA wait (05) does not help.

## Verdict

HALFPSI at hd128 is implemented completely and correctly (deterministic, finite, net-better accuracy; 2 of 144 pairs are
worse by +0.03%). It is **not faster**. Best structure (04/07):
- pass 1 is -1.2..-1.8%;
- pass 2 is +7.4..+8.6% because of the second accumulator's per-tile rescale;
- pass1+pass2 is +0.1..+0.8%, at or within noise and not below it.

Round 2's -4.2..-4.8% pass-1 probe predates H1a/H1b. H1a already moved the dV issue under the psi barrier, so the
decoupling is now worth about 1.4% of pass 1.

None of 01-09 is recommended. The lever's remaining headroom is bounded by probe 90 (about -1% pass1+pass2 if the extra
rescale were free). The only rescale-free idea tried, power-of-two psi (09), costs accuracy.

Files: patches at .local/screens/S17-psi-NN/variant.patch (NN = 01..09, probes 90-92). The patch scripts are in
.local/screens/S17-psi-tools/ (p02..p09, p1merge, p1early, errs.py, ms.py).
