# S17 pass-1 (D128 P1-R) screening log

Base: d07cdcd, A = `.local/frozen/r14_h3`. Every variant was screened with `.local/jobs/screen.sh` (2-rep A/B, B2 N8192
causal-matched, H16/Hk4 and H32/Hk4; noise about +-1%). bitwise = default-route equivalence vs `base3_default.pt` (72
cases); PROBE = timing only, wrong numerics, not checked. REG/STACK = the two `Fp8R` instantiations (non-causal, causal) of
pass 1 (base 168/8, 168/0). diag = ptxas spill / C7510 / C7511 / serialization lines in the build log. Patches:
`.local/screens/S17-p1-NN/variant.patch`. The edit recipes are in `.local/screens/S17-p1-tools/edits.py`. Worktrees are
sparse (csrc/ and delta_matching/ only; the build needs nothing else). "c" rows rerun the same library (copied jit dir).

| tag | description | bitwise | H16 pass1 | H32 pass1 | other kernels (max abs) | REG/STACK | diag | verdict |
|---|---|---|---:|---:|---:|---|---|---|
| 01 | EXPFENCE: warpgroup_fence_operand(tSrS) before the dP retire wait | rc 0 | +0.02% | +0.03% | 0.13% | 168/8 168/0 | 0 | null: SASS identical (opcode md5 = base; the fence emits nothing) |
| 02 | DVEARLY: P~ conversion + dOt wait + dV gemm issued right after exp2, before the dP retire | rc 0 | +5.00% | +4.72% | 0.22% | 168/8 168/0 | 0 | reject |
| 03 | AMAXTREE: psi amax as 4 independent max chains | rc 0 | +0.96% | +1.60% | 0.37% | 168/8 168/0 | 0 | reject alone (neutral on top of 39, see 42) |
| 04 | DSDBUF: dS' half tiles double-buffered, drain wait<1> (both WG barriers kept) | rc 0 | +2.73% | +2.33% | 0.32% | 168/8 168/0 | 0 | reject |
| 05 | DSDBUF2: as 04 + drain right after the issue, first WG barrier dropped | rc 0 | +0.98% | +1.36% | 0.27% | 168/8 168/0 | 0 | reject |
| 06 | UNROLL2: unmasked loop dispatches on the Q stage (compile-time descriptors) | rc 0 | +9.14% | +9.02% | 0.30% | 168/40 168/48 | 0 | reject (code bloat, stack) |
| 07 | PROBE NOPSIBAR: no cross-WG psi barrier / slot reads | probe | -6.17% | -5.93% | 0.47% | 168/8 168/0 | 0 | probe |
| 08 | PROBE NODSSTORE: no dS' shuffles / stmatrix / WG barriers / fence / TMA store | probe | -10.25% | -11.13% | 3.02% (pass2) | 168/8 168/0 | 0 | probe |
| 09 | PROBE NOPROMOTE: dV window promoted only at the last tile | probe | -3.73% | -4.07% | 0.33% | 168/8 168/0 | 0 | probe |
| 10 | PROBE NOEXP: no MUFU.EX2 (argument kept) | probe | -4.78% | -5.94% | 1.58% | 168/8 168/0 | 0 | probe |
| 11 | PROBE NOFOLD: dK fold 1 of 64 FFMA | probe | -4.65% | -5.22% | 0.42% | 168/8 168/8 | 0 | probe |
| 12 | PROBE NOPSI: no amax, shuffles, slots, barrier (psi from a constant) | probe | -10.77% | -9.73% | 0.80% | 168/8 168/8 | 0 | probe |
| 13 | FLAG PROBE QST 2 -> 3 | rc 0 | +0.51% | +0.82% | 0.52% | 168/8 168/0 | 0 | reject |
| 14 | FLAG PROBE DOST 1 -> 2 (Q / dO share the pipeline) | rc 0 | -1.87% | -2.98% | 0.83% | 168/0 168/0 | 0 | flag probe: helps (release-fixed flag) |
| 15 | FLAG PROBE DOTST 1 -> 2 | rc 0 | +1.31% | +1.19% | 0.16% | 168/8 168/8 | 0 | reject |
| 16 | 02 + DOTST 2 (flag) | rc 0 | +6.20% | +6.40% | 0.20% | 168/8 168/8 | 0 | reject (dOt latency is not why 02 lost) |
| 17 | MICRO: next-scale LDG only on odd m; psi slots as 2 x LDS.128 | rc 0 | +4.40% | +4.74% | 0.21% | 168/8 168/0 | 0 | reject |
| 18 | SEARLY: S(m+1) issued at the end of step m (in flight across the back-edge) | rc 0 | +19.80% | +18.94% | 0.72% | 168/80 168/88 | C7511 x2 | reject (wgmma serialized) |
| 19 | PROMOTE-EARLY: dV promote between the dV and dK retires | rc 0 | +0.74% | +1.28% | 0.43% | 168/8 168/0 | 0 | reject |
| 20 | 18 + 19 | rc 0 | +20.00% | +19.39% | 0.86% | 168/80 168/80 | C7511 | reject |
| **21** | **RESCALE-EARLY: dV window rescale (64 FMUL, every 2nd tile) moved from after the dS line into the S-retire bubble (after the dP issue)** | rc 0 | **-2.39%** | **-2.24%** | 0.27% | 168/8 168/0 | 0 (C7517 info) | **win (lever 1)** |
| 21c | rerun of 21 | rc 0 | -2.39% | -1.87% | 0.56% | same | - | confirms (H32 at the gate edge) |
| 22 | PROBE NOTMASTORE: no TMA store / drain (stmatrix, barriers, fence kept) | probe | -4.38% | -5.66% | 3.03% (pass2) | 168/8 168/0 | 0 | probe |
| 23 | PROBE NOWGBAR: no WG-local barriers around the stmatrix | probe | -1.13% | -1.82% | 0.78% | 168/8 168/0 | 0 | probe |
| 24 | PROBE NOFENCE: no fence.proxy.async (MEMBAR.ALL.CTA + FENCE.VIEW.ASYNC) | probe | -2.50% | -2.17% | 0.59% | 168/8 168/0 | 0 | probe |
| 25 | PROMOTE-V4: dV overlay as float4 slots (16 LDS.128/STS.128 per promote, not 64+64) | rc 0 | +0.29% | +0.72% | 0.50% | 168/8 168/8 | 0 | neutral |
| 26 | 19 + 25 | rc 0 | +0.79% | +0.04% | 1.05% | 168/8 168/8 | 0 | neutral |
| 27 | SEARLY2: half fold, S(m+1) issue, half fold, retire S(m+1) in the same iteration | rc 0 | +30.33% | +30.24% | 0.78% | 168/80 168/88 | C7511 | reject (registers) |
| 29 | 21 + STATS-EARLY: LSE*log2e and dPsum*k8s rows read / scaled in the S bubble | rc 0 | -2.81% | -2.49% | 0.46% | 168/8 168/8 | 0 | ~ 21 (+0.4%, noise) |
| 30 | 21 + 25 | rc 0 | -2.84% | -1.66% | 1.05% | 168/8 168/8 | 0 | ~ 21 |
| 31 | 21 + EXPPIN: never-taken STS.128 of the P~ codes pins exp2 above the dP retire wait | rc 0 | +0.42% | +0.50% | 0.23% | 168/8 168/0 | 0 | reject (overlapping exp2 with dP costs 2.6% vs 21) |
| 32 | 21 + FENCECLEAR: next-scale LDGs and psi STG moved after the dS' TMA store | rc 0 | +3.74% | +3.48% | 0.24% | 168/8 168/16 | 0 | reject |
| **33** | **21 + DSDEFER-CTA: dS' halves double-buffered by m parity; thread 0 stores BOTH halves of tile m right after the psi barrier of tile m+1 (every thread's stmatrix + proxy fence of m precede it); thread 0 drains store(m-2) before that barrier; both per-tile WG-local barriers removed; last tile after a post-loop 256-thread rendezvous** | rc 0 | **-4.04%** | **-3.89%** | 0.15% | 168/0 168/0 | 0 | **win** |
| 33c | rerun of 33 | rc 0 | -3.96% | -3.46% | 0.59% | same | - | confirms |
| 34 | 29 + PROMOTE-DEFER (window promote in the next step's S bubble) | rc 0 | -1.09% | -2.05% | 0.45% | 168/8 168/16 | 0 | reject |
| 35 | 33 + STATS-EARLY | rc 0 | -1.40% | -0.78% | 0.85% | 168/8 168/24 | 0 | reject (worse than 33) |
| 36 | 33 + FLAG PROBE DOST 2 | rc 0 | -6.49% | -7.00% | 0.55% | 168/0 168/0 | 0 | flag probe: strong |
| 36c | rerun of 36 | rc 0 | -6.40% | -6.16% | 0.49% | same | - | confirms |
| 37 | DSDEFER-CTA alone (no RESCALE-EARLY) | rc 0 | -2.39% | -2.43% | 0.12% | 168/0 168/0 | 0 | win alone (lever 2) |
| 38 | 35 + FENCE-LATE (tile m's proxy fence just before tile m+1's slot store) | rc 0 | -1.68% | -1.35% | 0.30% | 168/8 168/24 | 0 | reject |
| **39** | **21 + DSDEFER-SPLIT: as 33, but each WG's elected thread stores / drains its OWN half (two balanced issuers)** | rc 0 | **-5.23%** | **-5.05%** | 0.29% | 168/8 168/0 | 0 (C7517 info) | **BEST (bitwise, release flags)** |
| 39c | rerun of 39 | rc 0 | -5.36% | -5.21% | 0.27% | same | - | confirms |
| 40 | 39 + FLAG PROBE DOST 2 | rc 0 | -6.27% | -5.70% | 0.91% | 168/0 168/0 | 0 | flag probe (+~0.8% over 39; 36 was larger) |
| 41 | 39 + STATS-EARLY | rc 0 | -3.65% | -3.42% | 0.45% | 168/16 168/16 | 0 | reject |
| 42 | 39 + AMAXTREE | rc 0 | -5.68% | -5.84% | 0.50% | 168/8 168/0 | 0 | ~ 39 (+0.4%, noise) |
| 42c | rerun of 42 | rc 0 | -5.18% | -5.64% | 0.50% | same | - | = 39 within noise |
| 44 | 39 + AMAXTREE + PROMOTE-V4 | rc 0 | -5.85% | -5.70% | 0.38% | 168/8 168/8 | 0 | ~ 39 (one job) |

## Winners

- **39 = RESCALE-EARLY + DSDEFER-SPLIT** (bitwise, release flag set unchanged): pass 1 -5.23 / -5.05% and -5.36 / -5.21%
  in two jobs; other kernels within +-0.3%; REG 168, STACK 8 / 0 (= base), LOCAL 0, no spill / C7510 / C7511 / serialization.
  ptxas prints C7517 (an injected warpgroup.wait, informational) for both instantiations. It comes from RESCALE-EARLY:
  tdVrdV is touched before the first wgmma of the step. The injected wait is in the masked loop. In the unmasked loop the
  64 rescale FMULs sit directly after the dP QGMMAs, before the S-retire DEPBAR. The two levers add up: 21 = -2.2%,
  37 = -2.4%, 33 = -4.0%, and the split issuer gives about -1.2% more (39).
- 42 / 44 (39 + AMAXTREE (+ PROMOTE-V4)) are equal to 39 within noise. Prefer 39 (smaller).
- Flag probe (release-fixed flag, not shippable without a flag-set decision): DOST 1 -> 2 gives -1.9 / -3.0% alone (14),
  -6.4..-7.0% on top of 33 (36, 36c) and -6.3 / -5.7% on top of 39 (40). Bitwise rc 0 on all of these.

## Where pass-1 time goes (probes, % of pass 1, B2 N8192 H16 / H32)

| removed (PROBE) | H16 | H32 |
|---|---:|---:|
| whole psi path: amax + shuffles + slots + cross-WG barrier (12) | -10.8% | -9.7% |
| cross-WG psi barrier only (07) | -6.2% | -5.9% |
| whole dS' transpose + store path (08) | -10.3% | -11.1% |
|   of which TMA store + drain (22) | -4.4% | -5.7% |
|   of which proxy fence = MEMBAR.ALL.CTA + FENCE.VIEW.ASYNC (24) | -2.5% | -2.2% |
|   of which the two WG-local barriers (23) | -1.1% | -1.8% |
| MUFU exp2 (10) | -4.8% | -5.9% |
| dK fold, 64 FFMA (11) | -4.7% | -5.2% |
| dV promote every 8 tiles (09) | -3.7% | -4.1% |

Readings: the per-tile synchronization (psi rendezvous plus the dS' store sequence) costs more than the arithmetic does.
No single arithmetic block is worth more than about 5%. Overlap experiments are fragile. Pinning exp2 above the dP
retire (02, 31) and issuing S early (18, 20, 27) lose time: the S-early variants run out of registers (C7511 wgmma
serialization). Small instruction trims (03, 17, 25) do not move the measurement. What wins is (a) putting independent
vector work into the S-retire bubble (21) and (b) taking synchronization off the per-tile path (DSDEFER: no WG barriers,
the drain off the critical path, the store issued under the next tile's work). The rest of the dS' cost (the fence, the
memory traffic of the store) and the psi barrier (about 6%; 07) need either a flag change (DOST 2) or a numerics change
(per-WG psi: S17-psi worker).
