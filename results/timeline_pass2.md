# Pass 2 profile (round 9)

Target: the shipped dQ pass 2, `csrc/fq8_dq_pass2_hd128.h` (hd128: 64-row x 128-d dQ tile, 128-key dS' tiles,
kPromoteTiles 4, TMA depth `FA3FP11_HD128_P2ST` = 2) and `csrc/fq8_dq_pass2.h` (d256: 64-key tiles, kPromoteTiles 8,
depth 2). One CTA per (64-row m-block, bh), ONE 128-thread math warpgroup, `__launch_bounds__(128, 1)`.

Method (profiling limitation): no hardware counters (RmProfilingAdminOnly=1) and no in-kernel clock probes as timing
evidence (the round-7 forward probe build changed codegen and arithmetic). The profile is the emitted code order, the
per-symbol resources / occupancy arithmetic, and timing-only ablation builds (`results/r9/wi/whatif.patch`), each
timed per kernel (CUPTI, `benchmarks/time_cats.py`, 8 fwd+bwd configs, B4 N8192) in fresh processes, 3 rotating
repetitions (job 21165303). Ablation timings are observations of code-removal builds, not bounds.

## Emitted order and resources

Resources (accepted DO2): hd128 REG 114, d256 REG 167, STACK 0, LOCAL 0 (both instantiations each). Launch dynamic
smem (sizeof(SharedStorage): array payloads + barriers, rounded to 1024 B; review-verified from the host code) hd128
84992 B at depth 2, 109568 B at depth 3, d256 109568 B, each plus 1024 B static (cuobjdump SHARED): two CTAs per SM in
all three, i.e. TWO math warpgroups per SM in total. (The first version of this file quoted the 83968 / 108544 B array
payloads as CTA totals; the two-CTA conclusion is unchanged.)

Per tile, both headers emit one serial chain per CTA (`results/r9/wi/pass2_hd128_do2_order.txt`, hd128 instantiation of DO2):
`SYNCS.PHASECHK.TRANS64.TRYWAIT` (TMA full barrier) -> `LDG.E` of psi_row[nb] -> predicate / ratio (rescale of
the accumulator: 64 `FMUL` at hd128, 128 at d256) -> the tile's wgmma (hd128: 4 x `QGMMA.64x128x32`; d256: 2 x
`QGMMA.64x256x32`) -> `WARPGROUP.DEPBAR.LE gsb0, 0x0` -> `BAR.SYNC` (the
unconditional release) -> `SYNCS.ARRIVE.TRANS64` + 2 x `UTMALDG` (re-arm slot nb % depth for tile nb + depth). The psi
load is issued only after the barrier wait, so its latency sits between the TMA arrival and the gemm on every tile;
with depth 2 only one further tile is in flight while a tile is consumed.

## Ablations (pass-2 ms; change vs knob 0)

| variant | D128 causal matched Hk16 | D128 causal matched Hk4 | D128 full stale Hk16 | D128 causal stale Hk16 | D256 causal matched Hk8 | D256 causal matched Hk2 | D256 full stale Hk8 | D256 causal stale Hk8 | sum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| accepted DO2 (control) | 0.973 (+0.0%) | 0.960 (+0.0%) | 1.811 (+0.1%) | 0.974 (+0.1%) | 0.889 (+0.4%) | 0.882 (-0.4%) | 1.612 (+0.0%) | 0.889 (+0.1%) | 8.991 (+0.04%) |
| knob 0 (= DO2 source) | 0.973 (+0.0%) | 0.960 (+0.0%) | 1.810 (+0.0%) | 0.973 (+0.0%) | 0.886 (+0.0%) | 0.886 (+0.0%) | 1.611 (+0.0%) | 0.888 (+0.0%) | 8.987 (+0.00%) |
| 1 psi constant (no load; numerics wrong) | 0.936 (-3.8%) | 0.915 (-4.7%) | 1.787 (-1.2%) | 0.937 (-3.7%) | 0.834 (-5.8%) | 0.830 (-6.3%) | 1.524 (-5.4%) | 0.836 (-5.8%) | 8.600 (-4.31%) |
| 2 no in-window rescale (numerics wrong) | 0.958 (-1.5%) | 0.943 (-1.7%) | 1.795 (-0.8%) | 0.958 (-1.5%) | 0.836 (-5.6%) | 0.829 (-6.5%) | 1.503 (-6.7%) | 0.838 (-5.7%) | 8.661 (-3.63%) |
| 3 hd128 TMA depth 3 (flag; d256 unchanged) | 0.887 (-8.8%) | 0.885 (-7.8%) | 1.647 (-9.0%) | 0.890 (-8.5%) | 0.890 (+0.4%) | 0.885 (-0.1%) | 1.612 (+0.0%) | 0.888 (-0.0%) | 8.584 (-4.49%) |
| 4 PSIPF: psi loaded one tile early (bitwise on the default route) | 0.945 (-2.8%) | 0.925 (-3.6%) | 1.793 (-0.9%) | 0.944 (-3.0%) | 0.851 (-4.0%) | 0.843 (-4.9%) | 1.559 (-3.3%) | 0.849 (-4.4%) | 8.709 (-3.09%) |

Pass 1 is unchanged in every build (+-0.3% on its sum). These are observations of whole code-removal builds:
knob 1 removes the psi load AND the data-dependent scale / skip / state work, knob 2 can also change scheduling and
register use, so neither isolates an exclusive latency or bounds what a valid change can recover. Knob 3 (one more
TMA stage at hd128, -8..-9% of the hd128 pass 2 at unchanged occupancy) and knob 4 (PSIPF, -3.1% on depth 2) are
real candidates and are measured as separate hypotheses; knob 4's default-route equality is a saved log
(`p2wi4_compare.txt`), not an all-route certification of a later implementation.

## Hypotheses, in gate order (one per round, each its own commit)

1. P2ST3 (round 9): hd128 pass-2 TMA depth 2 -> 3 (`FA3FP11_HD128_P2ST` 3; certified flag-set change). Footprint:
   the hd128 pass 2 (D128 subset `pass2`); d256 code unchanged. Mechanism: one more tile in flight hides the TMA
   latency of the serial per-tile chain. Budget: dynamic smem 84992 -> 109568 B per CTA (+1024 B static; two CTAs
   per SM still fit),
   registers unchanged. Ledger: unchanged except the ring depth: wait(nb) on slot nb % 3 with parity (nb / 3) & 1,
   gemm with wg_wait 0, the unconditional `__syncthreads()`, then issue(nb + 3) into the slot just consumed; the
   prologue issues min(3, nb_end) tiles; kPromoteTiles (4) >= depth (3) still holds (static_assert). Arithmetic,
   windows, psi state, skip predicates, promotion and epilogue untouched: bitwise expected.
2. PSIPF (next pass-2 round): psi_row[nb + 1] loaded before the barrier wait of tile nb (one register), both headers;
   bitwise on the default route already (knob 4 build, `results/r9/wi/p2wi4_compare.txt`).
