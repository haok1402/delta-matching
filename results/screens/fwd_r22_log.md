# S22-fwd: bitwise forward screens on the r21_clean tile (128 x 256, 2 consumer WGs + producer WG)

Base: HEAD eabba10 (the sources of accepted r21_clean). A = `.local/frozen/r21_clean` in every screen. The script is
`.local/rounds/r22-fwd/screen.sh`, a copy of `.local/jobs/screen.sh` with REF = `.local/equiv4/r21clean_default.pt`, a fresh
v4 dump of r21_clean made in job 21206382 (library sha 81245944d4ec). Each screen is 2 reps at B2 N8192, H16/4 and H32/4,
D128 causal matched. Trees are in `.local/wt/S22-fwd-NN` and patches (against HEAD) are in
`.local/screens/S22-fwd-NN/variant.patch`. The edit generator is `.local/rounds/r22-fwd/edit.py`, and the SASS listings are in
`.local/rounds/r22-fwd/`. "bitwise" is the default-route compare against r21_clean (72 cases). The REG/STACK column is the
causal / non-causal D128 BN256 forward symbols (r21_clean: 168/24, 168/24, with setmaxnreg 24 / 240 / 240).

Scope (integrator update during the round): kda-fwd owns the softmax / convert SASS cuts and has already screened ST3
(3 stages, no smem fit), PREW4, A1SKIP and A1DEF. WGID / SCPOST / PRETURN4 are already active at eabba10. So this round
covers only the round 7-16 levers and the 2-WG turn schedule.

## Step loop of r21_clean (unmasked, `r21_unmasked.txt`)

1273 instructions and 149 MUFU per warp per step (256 keys).
- FMUL 411: descale 128, rho 3 x 64, alpha 64.
- FFMA 212: exp code 128, promote 64.
- FMNMX 168, EX2 146, FADD 138, F2FP 64.
- QGMMA 12, and about 120 uniform / int / sync instructions.

The turn, from BAR.SYNC to BAR.ARV, is 4 x (2 QGMMA 64x128x32, then DEPBAR), with 3 x 64 FMUL rho rescales between them,
then 64 FFMA promote, the V release, the K wait, 4 QGMMA 64x256x32 QK, and BAR.ARV. ptxas already sinks the rho
RCP/EX2/clamps under the in-flight sub-gemms.

## Variants

| tag | change | bitwise | REG/STACK | fwd H16 | fwd H32 | verdict |
|---|---|---|---|---:|---:|---|
| S22-fwd-00 | null (HEAD = r21_clean sources), noise check | rc 0, 72 cases | 168/24, 168/24 | +0.03% | -1.22% (passes -1.0..-1.3%: drift) | noise about +-1% |
| S22-fwd-01 | PVIL4 (round-7 PVIL revisited on top of RHOPRE / PRETURN4): NSPLIT 2. Two 64-column slab temps run as interleaved rho chains (wait<1>, each slab's rescale under the other slab's sub-gemm), with rho / w precomputed | rc 0 | 168/24, 168/24 | **+2.16%** | **+1.92%** | reject. The SASS is as designed (QGMMA 64x64, DEPBAR.LE 1); the loop is 1307 vs 1273 instructions |
| S22-fwd-02 | PVIL4 + DEFERB: slab b's promote and the V release moved after the QK issue and the ring-turn arrive (wait<1> under QK) | rc 0 | **168/80, 168/96** (spills: O 64 + Db 32 + S 128 > 240) | +17.66% | +17.77% | reject |
| S22-fwd-03 | QKH: QK as two 64x128 halves (TiledMmaQKH, logical_divide of sK), with the promote issued between them | **rc 0** (QK halves are bitwise) | **168/160, 168/176** | +24.25% | +23.80% | reject. ptxas issues half 1 right after half 0, sinks the promote FFMAs past both, and spills O across the loop (32 LDL + 32 STL per step) |
| S22-fwd-04 | QKH with the promote as `asm volatile` fma.rn.ftz.f32 | rc 0 | 168/160, 168/176 | +24.37% | +23.40% | reject. Volatile asm does not pin the order either: the same spill and sinking |
| S22-fwd-05 | VREL: V released right after the last PV sub-gemm retires (before the promote / K wait) | rc 0 | 168/24 | +0.10% | +1.63% (passes +1.7%: drift) | reject (neutral) |
| S22-fwd-06 | PVIL4 + QKH (slab b's promote between the QK halves) | - | - | - | - | cancelled after 01 (PVIL slower) and 03 (QKH spills) |
| S22-fwd-07 | ARV-after-promote: the scheduler-barrier arrive after the promote, before the K wait / QK issue (SCHEDPV on the 2-WG tile) | rc 0 | 168/24 | +1.29% | +0.18% (passes -1.3%) | reject (slower) |
| S22-fwd-08 | ARV-before-promote: the arrive right after the last PV sub-gemm retires | rc 0 | 168/24 | +1.34% | +1.75% (passes +0.7%) | reject |
| S22-fwd-09 | G0PRE: the zero-init g0 PV sub-gemm issued before the scheduler-barrier sync (its wait is after the sync). The SASS confirms the QGMMAs before BAR.SYNC | rc 0 | 168/24 | -0.53% | +0.02% (passes +0.4%) | reject (neutral) |
| S22-fwd-10 | G01PRE: g0, rho1 and g1 before the sync | rc 0 | 168/24 | +0.28% | +0.77% (passes +0.5%) | reject (neutral) |

Every built variant is BITWISE EQUAL to r21_clean on the default route, and none spills except 02 / 03 / 04.

## Reading

- The turn schedule of the 2-WG tile is insensitive to bitwise edits. These changes do not move the forward beyond noise:
  - g0 or g0+g1 before the sync (09, 10);
  - an earlier V release (05).
  The arrive placement can only lose (07, 08). In the measured regime the scheduler-barrier sync is not where a warpgroup
  waits. The per-WG serial chain (turn, then QK latency, then softmax) sets the period, and its non-turn part is the
  softmax ALU / MUFU work (kda-fwd's scope).
- Shortening the PV chain by interleaving half-width slabs (01) costs +2%. It adds instructions (+34 per step: waits,
  descriptors, n64 atoms) and buys no tensor overlap that matters.
- Overlapping the promote with QK work is register-infeasible as ptxas schedules it. Either S (128) + O (64) + a temp
  slab are live together (02), or ptxas issues the two QK halves back to back and sinks the promote (03, 04, the same
  failure as S17-fwd-16/17/18). The register budget is the ISA maximum (24 + 2 x 240 = 504 of 512), so there is no split
  headroom.
- QK-first (FA3 IntraWGOverlap) at BN256 needs O 64 + S 128 + D 64 + P 32 = 288 > 240 registers, or 256 with NSPLIT 2,
  so it is infeasible. KSPF is obsolete (the scales come from smem via SMSC / SCPOST). ODEFER / RHOPRE are already
  covered (RHOPRE is in; ODEFER failed by ptxas hoisting).

## Verdict

No bitwise piece >= 1% in items (1) and (2) on this tile, so there is no stack and no 4-rep confirmation was run (no
candidate). Item (3) was handed to kda-fwd by the integrator. Levers in my scope are exhausted.
