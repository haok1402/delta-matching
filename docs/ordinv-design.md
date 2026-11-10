# ordinv: tcgen05 ordering inventory of the validated hd128 kernels (kda-sm100-ordinv, 2026-11-10)
(Part I = inventory + opt-in fixes (d830df0). Part II at the end = LAND CANDIDATE ORD-land bbfea5a: fixes default per lead decision
+ accuracy-audit 06:4x ruling; supersedes the "opt-in / not default" statements of Part I.)

Request: accuracy-audit RULINGS 05:3x (inbox/dm-validator.md), item 4 (+ item 1 rule). Scope: hd128 det forward (dm_fwd),
pass 0 (dm_delta, two-lane `+tl`), backward main (dm_bwd_sm100.py, 2-CTA pipelined, incl. the dsqw helper) of 7d63434 +
d3nh 2973034 (line numbers below = 2973034, wt/kda-sm100-d3-land). Fix branch: **kda-sm100-ordfix d830df0** (on 2973034,
pushed), every fix opt-in, default unchanged.

## 1. The rule (PTX ISA, verbatim source in kda/runs/ordinv/ptxdoc/)
- Async tcgen05 ops (`mma, cp, shift, ld, st`) "may execute and complete in a different order than they were issued"; only
  listed same-warp pairs are pipelined.
- **PTX 9.5 §9.7.18.6.2** (current) lists: `mma (A and metadata reads) -> mma (D writes)` (same N); `mma (A/metadata reads) ->
  cp/shift (writes)`; `mma (C reads | D writes) -> mma (D writes | C reads)` (same N **and accumulator address and shape**);
  `cp/shift (writes) -> mma (A/metadata reads)`; `st -> ld`, `st -> st`, `ld -> st` (same warp). "Instructions issued by
  different warps are always non-pipelined." Metadata = the `.sp` sparsity operand (`[sp-meta-tmem]`); **scale-factor
  reads (`[scale-A-tmem]`, `[scale-B-tmem]` of block-scaled mma) are not named in any pair.**
- **PTX 9.0 §9.7.16.6.2** (CUDA 13.0 docs, the wording the audit and Codex fwd-01 cite) is narrower: mma -> mma only for
  the same N, accumulator and shape, and has no ld/st pairs.
- Completion: `tcgen05.commit` tracks *all* prior mma/cp/shift of the issuing thread (§9.7.18.6.2.1.1), `wait::ld/st` the
  thread's ld/st. Register dependency on a `tcgen05.ld` result is **not** a memory ordering (§9.7.18.6.4.5): anti-dependency
  hazards need `tcgen05.wait::ld`. Cross-thread: `[op; wait; fence::before; arrive] -> [wait; fence::after; op]` (§9.7.18.6.4.4).

Classification used below: **G** = guaranteed by a listed pipelined pair or by a completion (commit -> mbarrier, wait::ld/st)
chain; **G9.5** = guaranteed only by a pair that PTX 9.5 lists and PTX 9.0 does not; **NG** = not guaranteed (issue order or
register dependency only). Cross-thread hand-offs with commit/wait + fences (the fwdfence class, kda/docs/fence-audit.md,
default since b5e57e9/927eb0f) count as G and are not re-listed except where a wait is missing.

## 2. Inventory (every region read by one tcgen05 op and later written by a different tcgen05 op)
TMEM maps: fwd (slice, q_stage 2) S0 0..127, S1 128..255, O0 256..383, O1 384..511; within S_s: P codes of key block q at
S_s+32q (8 cols), SFA S_s+16, SFB S_s+20 (`dm_sl_sfcol`, :484). bwd (2-CTA pipelined): S 0..127 [P 0..31, SFA_dv 32..47,
SFB_dv 48..51, SF_dq 52..57, dQ 64..127], dV 128..255, dP 256..383 [dS 256..287, SFA_dk 288..303, SFB_dk 304..307], dK 384..511.

| # | kernel | file:line (2973034) | read op -> later write op | region | ordering today | PTX |
|---|---|---|---|---|---|---|
| **F1** | fwd | dm_fwd_sm100.py:2340 -> :2374 (loop), :2408 -> next tile :2302 | PV_s(k) (mxf8f6f4 block-scaled, acc O_s) **SFA/SFB reads** -> QK_s(k+1) (f8f6f4, acc S_s) D writes | S_s+16..23 | issue order only (QK issued right after PV, no wait; cross-tile likewise) | **NG** (SF reads not in any pair; different acc/shape/kind) |
| F2 | fwd | :2340 -> :2374 | PV_s(k) **A reads** (P codes, TMEM) -> QK_s(k+1) D writes | S_s+32q..+7 | issue order = pipelined pair | G9.5 (A reads -> D writes, same N); NG under 9.0 wording. Covered by the F1 fix |
| F3 | fwd | :2340 loop | PV_s(k) C/D -> PV_s(k+1) C/D | O_s | pair | G (same N, acc, shape) |
| **F4** | fwd | :3059 (slice WG 0 `tcgen05.ld` S_s cols 0..31) -> :3093 kb arrive -> correction :3228 `tcgen05.st` SF | S_s+16..23 | ld -> (reg use, fence::before, mbarrier) -> other warp's st; **no tcgen05.wait::ld** (base fwd PTX: 2 wait::ld, none in the slice step) | **NG** (cross-thread WAR needs wait::ld, §9.7.18.6.4.4/.5) |
| F5 | fwd | :3059 -> :3102 (same slice warp) | ld S -> st P codes, same columns | S_s+32q..+7 | same-warp ld -> st | G9.5 (ld -> st pair); with the F4 fix also wait::ld |
| F6 | fwd | :4225 -> :4231 (correction rescale) | ld O chunk -> st O chunk, same warp | O_s | same-warp ld -> st | G9.5 |
| F7 | fwd | QK_s commit / PV_s(k-1) -> slice ld, correction ld/st O, epilogue; P/SF/O st -> PV | S_s, O_s | commit -> mbarrier (commit cumulative over PV_s(k-1)), wait::st + fences | G |
| P1 | pass 0 | dm_pass0_sm100.py:2064 QK_s, :2073 dP_s -> tl_loop ld (:2715..2751) -> next QK/dP | S_s, dP_s (disjoint; no SF, no TMEM A operand, no st) | commit -> mbarrier; wait::ld (`fence_view_async_tmem_load`) before release | G |
| **B1** | bwd | dm_bwd_sm100.py:3836 (prologue), :3957 (loop) -> :3877 | dV(t) (block-scaled, acc dV) **SFA/SFB_dv reads** -> S^T(t+1) (f8f6f4, acc S) D writes | S region 32..51 | issue order only (dQ-empty wait before S^T(t+1) tracks dQ(t-1), which is committed *before* dV(t) is issued) | **NG** |
| B2 | bwd | :3957 -> :3877 | dV(t) **A reads** (P codes, TMEM) -> S^T(t+1) D writes | S region 0..31 | pair | G9.5; covered by the B1 fix |
| **B3** | bwd | :3898 (loop), :3988 (tail) -> :3910 | dK(t) (block-scaled, acc dK) **SFA/SFB_dk reads** -> dP^T(t+1) (f8f6f4, acc dP) D writes | dP region 288..307 | issue order only (dP^T(t+1) issued right after dK(t)) | **NG** |
| B4 | bwd | :3898 -> :3910 | dK(t) **A reads** (dS codes, TMEM) -> dP^T(t+1) D writes | dP region 256..287 | pair | G9.5; covered by the B3 fix |
| B5 | bwd | dQ(t-1) :3937 SF_dq reads / D writes 64..127 -> S^T(t+1) :3877 | S region 52..127 | dQ full commit -> reduce warps drain -> dQ empty wait | G |
| B6 | bwd | S^T(t+1) D writes 64..127 -> compute ld S -> dQ(t) D writes 64..127 (:3937) | S region 64..127 | commit -> ld + wait::ld -> pair barrier/dsq mbar 10 -> pipeline_dS commit -> MMA wait | G |
| B7 | bwd | dV / dK / dQ C/D chains | dV, dK, dQ | pairs (same acc/shape) or completion | G |
| B8 | bwd | compute ld S (warps w, w+4) -> other warp st P codes / SF_dv (:5235); ld dP -> st fp32 dS (:5570); dsqw ld fp32 dS (:5889) -> st dS codes / SFA_dk (:5926) | S, dP regions | wait::ld (`fence_view_async_tmem_load`) + 64-thread pair barrier / mbarrier 9 + fences; dsqw: same warp, wait::ld before st | G |
| B9 | bwd | P/SF/dS st -> dV/dK/dQ MMAs; MMA -> compute/reduce/epilogue ld | all | wait::st + fences + mbarrier / commit | G |
| X | all | SMEM operands (Q/K/V/dO/sdS/sP) | MMA reads -> TMA / st.shared rewrites | tcgen05.commit to the pipeline empty barrier | G (no tcgen05 op writes SMEM) |

**Count: 20 sites (F1-F7, P1, B1-B9, X).** Not guaranteed: **3 in the strict NG sense plus 1 cross-thread wait gap = 4**
(F1, F4, B1, B3). Guaranteed only under PTX 9.5's wider pair list: 5 (F2, F5, F6, B2, B4); all 5 become completion-ordered
with the fixes below except F6 (same-warp ld -> st O in the correction rescale; a `wait::ld` there would close it under 9.0 too,
not built). Fully guaranteed: 11 (F3, F7, P1, B5-B9, X). Pass 0 has no site. The dsqw helper has none (its ld -> st is same warp
with an explicit wait::ld).

F1 is the hd128 twin of hd256's pre-C0 SF overwrite, but closer: QK_s(k+1) follows PV_s(k) directly (distance 1, the
hd256 slot was reused at j+2). B1/B3 are the same class in the backward (scale factors parked in a region the next
different-accumulator MMA overwrites). No output error has ever been observed (all bitwise stress clean) and the hd256
positive control shows the class only misbehaves for non-MMA writers (pc1 = tcgen05.st after issue), so these remain
"issue order in practice", which the 05:3x ruling does not accept as evidence.

## 3. Fixes (branch kda-sm100-ordfix d830df0, all opt-in; default build PTX byte-identical to 2973034)
| knob | sites | what it does |
|---|---|---|
| `DM_FWD_OPTS=ordw=1` | F1 (+F2) | `tcgen05.commit` after every PV_s -> new mbarrier ord[s] (2 x 8 B smem, only with the knob); the MMA warp waits completion #k (+ `fence::after_thread_sync`) before QK_s(k+1), and before a later work tile's first QK_s. Phase-safe: PV_s(k-1) is complete when PV_s(k) issues (P_full(k) needs S_s(k) loaded, whose commit tracks PV_s(k-1)), so at most one completion is outstanding. = hd256 C0 (f42496c) pattern. |
| `DM_FWD_OPTS=ordld=1` | F4 (+F5) | `tcgen05.wait::ld` in every slice warp after the S load, before its fence::before + kb arrival. |
| `DM_BWD_OPTS=ordw` (`ordw=v` / `ordw=k`: one site) | B1 (+B2), B3 (+B4) | cta_group::2 `tcgen05.commit` after every dV (-> ord[0]) and dK (-> ord[1]) of the leader MMA warp (mbarriers dm_xchg[11,12], only with the knob); wait (+ fence::after) before the next S^T resp. dP^T, incl. prologue/tile boundaries. Asserted to the 2-CTA pipelined schedule without `rdo`. |
| (none) | F6 | not built (G9.5; a wait::ld in correction_rescale would be the fix) |

Evidence (kda/runs/ordinv/): PTX dumps (ptx/, ptx.log): base 2973034 vs ordfix default **byte-identical** for fwd, pass 0, bwd
at 32/4 8k causal det; knobs on: fwd +4 commits, wait::ld 2 -> 8; bwd +4 commits. Outputs: every knob arm **bitwise** vs the
knob-off build in the same process (O, LSE; dQ, dK, dV), all finite, 3 shards x 2 shapes (ab_{32,16}_{1,2,3}.log).

## 4. Cost (same-process A/B, causal det, B4 N8k, 1 GB200, kda/tests/ordinv/ab.py v2: per group random order each round,
each timed arm preceded by an untimed base call, 40 rounds x 20 iters; paired per-round median ratio vs base; shards 2 / 3)
| arm | 32/4 | 16/4 |
|---|---|---|
| fwd base (ms) | 1.297 / 1.298 | 0.664 / 0.666 |
| fwd ordw=1 (F1) | **+0.99% / +0.96%** | **+0.93% / +0.93%** |
| fwd ordld=1 (F4) | -0.02% / +0.07% (noise) | +0.15% / +0.06% |
| fwd ordw=1,ordld=1 | +1.04% / +1.01% | +1.10% / +0.98% |
| fwd base#2 (self-check) | +0.03% / +0.06% | 0.00% / 0.00% |
| pass 0 (no fix; ms) | 1.245 / 1.250 | 0.611 / 0.619 |
| bwd base (ms) | 3.928 / 3.924 | 1.986 / 1.991 |
| bwd ordw=v (B1) | +0.57% / 0.00% | -0.21% / +0.78% |
| bwd ordw=k (B3) | +1.40% / +1.45% | +0.60% / +0.73% |
| bwd ordw (B1+B3) | **+1.35% / +0.70%** | **+0.53% / +0.92%** |
| bwd base#2 (self-check) | +0.57% / +0.01% | -0.10% / +0.21% |
Reading: all formal fixes cost about 1% per kernel at 8k (fwd ~+1.0%, bwd main ~+0.5..1.4%, B3 the larger part), i.e. about
+0.8..1.0% of fwd+pass0+bwd at the 8k headline shapes. Shard 1 (rotation without the predecessor control) is discarded:
it showed a 3-7% arm-position bias (self-check fwd 1.07 / bwd 0.97).
Not done (no land request is made, the lead/user decide): fa4-bench harness numbers, 32k/64k/128k timing, ncu, Codex review
of the diff, t_small/t_odd/identity (the knobs change no arithmetic: bitwise outputs above and an unchanged default PTX).

## 5. Options if the 1% matters (not built)
- F1: PV_s(k) could wait on the *other* stage's progress instead (issue PV0, PV1, then QK0, QK1 with each wait one PV older):
  likely worse (QK0(k+1) then waits for P1(k)). Or moving SFs out of S_s: TMEM is full (2 x 128 S + 2 x 128 O).
- B3: the dP region is the only free space for SFA_dk; a wait placed after dQ(t) (reorder dK, dQ, dP) hides it only if dS(t)
  is ready, at the price of dP(t+1) latency.
- An NVIDIA statement that scale-factor reads belong to "A and metadata reads" would make F1/B1/B3 G9.5 without any wait
  (Codex hd256-fwd-02 raised the same question).

---------------------------------------------------------------------------------------------------------------------
# PART II: LAND CANDIDATE ORD-land = kda-sm100-ordfix bbfea5a (parent 2973034), fixes DEFAULT (2026-11-10 07:xx)
Decisions: lead (kda-sm100, "adopt as DEFAULT") and accuracy-audit RULING 06:4x (dm-validator.md): SF reads are not "metadata";
F1/F4/B1/B3 = latent races, fixes REQUIRED as default; the PTX-9.5-only pairs (F2, F5, F6, B2, B4) are ACCEPTED (PTX ISA 9.5
§9.7.18.6.2 lists them as pipelined pairs, no version gate); F6 may stay. So this candidate fixes exactly F1, F4, B1, B3.

## II.1 Commits (branch kda-sm100-ordfix, pushed)
d830df0 opt-in knobs (Part I) -> 569ca71 DEFAULT ON (fwd ordw + ordld; bwd ordw on the 2-CTA pipelined schedule, both dV and dK), debug-only
off knobs DM_FWD_OPTS=ordw=0,ordld=0 / DM_BWD_OPTS=noordw (= the pre-ordinv build), per-site bwd ordw=v/k removed, planted controls ->
a683b1a controls per Codex P3 (fwd boundary plant, F4 regression arm ordpc=5) -> **bbfea5a** bwd ordpc2 (B1 control after dV issue).
`rdo` (debug order) now asserts unless combined with noordw. The 1-CTA / serial bwd schedules are not used at hd128 (FA4 picks 2-CTA
for every hdim >= 128) and are not covered (Codex: do not describe them as covered).
PTX (kda/runs/ordinv/ptx/, ptx2.log, 32/4 8k det): bbfea5a off-knobs == 2973034 byte for byte (fwd, pass 0, bwd); bbfea5a default ==
d830df0 knobs-on (the A/B'd build). Results measured on a683b1a / 569ca71 carry over (identical default PTX).

## II.2 Planted controls (kda/tests/ordinv/pc.py; 8k B4, GEMM + NaN-realloc perturbation, every call bitwise vs ref call 0;
logs runs/ordinv/pc_{32,16,32_growmax}.log (a683b1a) and pc2_* (bbfea5a); accuracy-audit reproduced the fwd ones with audit/fwd_race.py)
| site | plant (debug knob) | fix OFF + plant | fix ON + plant | fix OFF, no plant |
|---|---|---|---|---|
| F1 | ordpc=1: MMA warp tcgen05.st 0x80808080 into SFA of S_s right before QK_s(k+1) (+ at the work-tile boundary) | **150/150, 150/150, 100/100 caught** | 0 | 0 |
| F4 | ordpc=4: slice warps re-load S cols 16..23 and poison l on a change; without the fix the re-load is placed at the latest PTX-legal point (after the kb arrival + 2 us) | **150/150, 150/150, 100/100 caught** | 0 | - |
| F4 (regression, Codex P3) | ordpc=5: same re-load before the arrival in both arms, only wait::ld differs | 0 (the real gap does not fire on this HW) | 0 | - |
| B1 | ordpc1: SFA_dv poison at the S^T(t+1) issue point | 0/150 (window closed in practice by the dQ-empty wait; still not PTX-guaranteed) | 0 | 0 |
| B1 | ordpc2: SFA_dv poison right after dV(t) issue (hd256 pc1/pc2 analogue) | **150/150, 150/150, 100/100 caught** | 0 (poison after the dV completion wait) | - |
| B3 | ordpc3: SFA_dk poison right before dP^T(t+1) | **150/150, 150/150, 100/100 caught** | 0 | 0 |
(32/4 randn, 16/4 randn, 32/4 growmax.) The fwd forward grid is one tile per CTA at these shapes, so the boundary plant does not execute there.

## II.3 Codex review (kda/runs/ordinv/codex_review.md, 569ca71): no correctness defect in the 4 fixes (parity incl. tile boundaries,
empty / one-block tiles, q_stage=1 path, cta_group::2 commit form and issuing thread, mbarrier allocation/init, no new deadlock, no other
uncovered reuse in fwd / pass 0 / bwd+dsqw). Two P3 items on the controls: both addressed (a683b1a, bbfea5a; table above).

## II.4 Gates (DM_WT = ordfix, logs kda/runs/ordinv/)
- Bitwise vs 2973034 (bw/, bw_summary.txt; separate processes per tree): O, LSE, delta, dQ, dK, dV sha256 equal at all 8 headline shapes
  x randn + growmax (16/16), finite, repeatable. accuracy-audit independently: 24/24 incl. ramp (hashcmp, 569ca71).
- In-process knob A/B: every arm bitwise (ab_*, e2e_* BITS lines).
- t_small det + nd PASS (pre_small.log); t_odd no NaN (pre_odd.log); t_bits noordw vs default 4/4 bitwise (pre_bits.log); identity
  (t_ident_bwd BASE noordw vs default, 4 shapes x 4 families x det/nd) IDENT_RESULT PASS (pre_ident.log).
- det stress (fwd+bwd e2e under GEMM load, grads bitwise vs ref): 8k 32/4 + 16/4 randn 400/400 + growmax 300/300 calls, bad 0
  (pre_stress_*.log); planted-control runs' ref arms 0/150-200 (fwd O/LSE and bwd dQ/dK/dV).
- ncu (kda/profile/ordinv-r1/REPORT.md): fwd duration +1.2% / +1.7%, bwd main +1.3% / +1.1% vs 2973034 (32/4 / 16/4 8k);
  instructions +0.8% fwd / +0.1% bwd; FA4-det fwd 1.965 / 0.988 ms, bwd 6.753 / 3.445 ms (ncu clocks).

## II.5 Cost (det, causal; base = off knobs = 2973034 PTX; each timed arm preceded by an untimed base call; random order per round)
Kernel level at 8k B4 (kda/tests/ordinv/ab.py, 40 rounds x 20 iters, 2 shards, self-check <= 0.6%, paired median): fwd +0.96..0.99% (32/4),
+0.93% (16/4) [ordw]; ordld noise; both +1.0..1.1%; bwd main +0.70..1.35% (32/4), +0.53..0.92% (16/4).
All 8 headline shapes (kda/tests/ordinv/e2e.py, 4 shards, 8 rounds; median of the paired cand/base ratio over shards whose base/base#2
self-check is within 1%, (n ok / 4)); e2e = dm_attn_func fwd+bwd incl. quantization and pass 0 (pass 0 code is unchanged):
| shape | e2e per call | e2e back-to-back | fwd b2b | bwd b2b |
|---|---|---|---|---|
| 32/4 8k B4 | 1.0178 (4/4) | 1.0073 (4/4) | 1.0171 (1/4) | 1.0073 (1/4) |
| 32/4 32k | 1.0089 (3/4) | 1.0072 (4/4) | 1.0128 (4/4) | 1.0076 (4/4) |
| 32/4 64k | 1.0065 (4/4) | 1.0106 (4/4) | 1.0128 (4/4) | 1.0155 (3/4) |
| 32/4 128k | 1.0051 (4/4) | 1.0037 (4/4) | 1.0062 (4/4) | 1.0046 (4/4) |
| 16/4 8k B4 | 1.0050 (3/4) | 1.0098 (4/4) | 1.0134 (4/4) | 1.0045 (4/4) |
| 16/4 32k | 1.0108 (4/4) | 1.0097 (4/4) | 1.0407 (3/4) | 1.0046 (4/4) |
| 16/4 64k | 1.0033 (4/4) | 1.0060 (4/4) | 1.0081 (4/4) | 1.0145 (3/4) |
| 16/4 128k | 1.0045 (4/4) | 1.0038 (4/4) | 1.0073 (4/4) | 1.0061 (4/4) |
Summary: e2e fwd+bwd +0.3..1.8% per call, +0.4..1.1% back-to-back (geomean about +0.8%). Per-kernel per-call medians at 8 rounds are too
noisy at some shapes (self-check > 1%: see e2e_summary.txt; the 16/4 32k fwd b2b 1.04 rests on 3 shards); the 8k kernel numbers above
(40 rounds) are the reliable per-kernel figures. Not done: fa4-bench harness (the lead submits; fa4-bench re-runs the det headline).

---------------------------------------------------------------------------------------------------------------------
# PART III: cross-CTA mbarrier scope (accuracy-audit RULING 08:2x item 1(b)) - branch kda-sm100-ordfix2 (on bbfea5a)

> **USER DECISION (2026-11-10 ~13:00): formal gap at X3-X7, X10, X11 ACCEPTED for now; bbfea5a/d5beca5 = validated headline; pwq/ordfix2 is NOT the product default (its 'default' below is branch-local only); no new hand-offs of this kind; kda-sm100-xexempt seeks a cheap formal fix (<= ~5% e2e). Full note: section III.U.**

Rule: a hand-off between threads of DIFFERENT CTAs of the cluster that publishes data written by ordinary / async-proxy-fenced stores,
tcgen05.st or ld results reused needs .release.cluster arrive + .acquire.cluster wait (or fence.acq_rel.cluster around .cta ones).
Exempt: TMA complete_tx, tcgen05.commit (incl. multicast). PTX facts: `mbarrier.arrive` without .sem/.scope = .release.cta;
`mbarrier.try_wait` without = .acquire.cta. cutlass-dsl's cute.arch.mbarrier_arrive keeps scope .cta even for a peer-CTA target
(its own comment: "Remote CTA arrivals should be cluster-scoped for formal cross-CTA memory-model ordering ... Pass scope=CLUSTER").

## III.1 Inventory (default det hd128 bwd = 2-CTA pipelined, dsqw; leader = CTA rank 0; PTX of bbfea5a, kda/runs/ordinv/ptx/bwd_v2_def.ptx)
fwd and pass 0: cluster size 1 (FORCE use_2cta_instrs=False): no cross-CTA hand-off (their shared::cluster forms address the own CTA).
| # | barrier (dm_bwd_sm100.py) | producer -> consumer (CTAs) | data published | sem/scope (bbfea5a PTX) | class |
|---|---|---|---|---|---|
| X1 | Q / Qt / Kt / dO PipelineTmaUmma (:2052-2092) full / empty | TMA (cta_group::2) -> leader MMA; MMA tcgen05.commit multicast -> TMA producer | Q, K, dO tiles | complete_tx / tcgen05.commit | exempt |
| X2 | S_P / dP / dQ / dKV full (PipelineUmmaAsync producer_commit) | leader MMA -> compute / dsqw / reduce / epilogue of both CTAs | S, dP, dQ, dK/dV accumulators | tcgen05.commit.cta_group::2 multicast | exempt |
| X3 | S_P empty (consumer_release :5006/:5013/:5338) | 8 compute warps of BOTH CTAs -> leader MMA (dV, next S^T) | P^T codes + SFA_dv/SFB_dv (tcgen05.st, own TMEM) read by the pair dV MMA; WAR of S loads | mbarrier.arrive.shared::cluster (.release.cta) / try_wait (.acquire.cta) | **NOT exempt** |
| X4 | dP empty (dsqw consumer_release :6014) | dsqw warps of BOTH CTAs -> leader MMA (dK, next dP^T) | dS codes + SFA_dk (tcgen05.st); WAR of fp32 dS / dP loads | .release.cta / .acquire.cta | **NOT exempt** |
| X5 | dS full (PipelineAsyncUmma producer_commit :6062) | dsqw warps of BOTH CTAs -> leader MMA (dQ) | dQ SFs (tcgen05.st), own sdS half (st.shared + fence.proxy.async) | .release.cta / .acquire.cta | **NOT exempt** |
| X6 | dQ empty (reduce consumer_release :6411/:6475) | reduce warps of BOTH CTAs -> leader MMA (next S^T over the dQ columns) | WAR of dQ tcgen05.ld (ld results reused) | .release.cta / .acquire.cta | **NOT exempt** |
| X7 | dKV empty (epilogue consumer_release :6709/:6768) | epilogue warps of BOTH CTAs -> leader MMA | WAR of dK/dV TMEM loads (a later write only with pers>1 tiles) | .release.cta / .acquire.cta | **NOT exempt** (form) |
| X8 | dS peer exchange dS_cluster_full (dsqw :6044 arrive.expect_tx remote + :6046 cp.async.bulk s2cluster) | dsqw tidx0 of CTA r -> relay warp of CTA r^1 | sdS half (st.shared + fence.proxy.async in the source CTA, then the bulk copy) | data via bulk-copy complete_tx (the remote expect_tx arrive carries no data) | exempt |
| X9 | dQ SF word exchange dm_xchg[slot] (:6040 st.async + complete_tx) | dsqw tidx0 of CTA r -> CTA r^1 (local wait) | 1 word | st.async complete_tx | exempt |
| X10 | dS relay dS_cluster_leader (relay warp :2554 mbarrier_arrive(ptr, 0)) | relay warp of BOTH CTAs (CTA1: remote) -> leader MMA (dQ) | forwards "peer's dS half landed in my sdS" (ordinary-thread relay of an async completion) | .release.cta / .acquire.cta | **NOT exempt** |
| X11 | 2-CTA TMEM dealloc handshake (TmemAllocator.free, :2392) | MMA warp of CTA1 -> CTA0 dealloc | WAR: all TMEM accesses done before tcgen05.dealloc | .release.cta / .acquire.cta | **NOT exempt** |
| X12 | cluster init (fence.mbarrier_init.release.cluster + barrier.cluster.arrive.relaxed / wait) | all -> all | mbarrier init | prescribed PTX init pattern | exempt |
| X13 | pdyn / pers barriers (:277 mbar_arrive_rank, dm_xchg[2]) | not in the default build | - | already .release.cluster / local | n/a |
PTX count (bbfea5a, 32/4 8k det): 10 remote `mbarrier.arrive.shared::cluster` = the sites X3-X7, X10, X11 (some inlined in more than one code copy; attributed by warp
role order in the PTX, line info does not resolve the inlined helpers), all .release.cta,
1 remote arrive.expect_tx (X8, exempt). **7 non-exempt hand-offs (X3-X7, X10, X11) = latent violations** of the 08:2x rule; the consumer of all
of them is the leader's MMA warp (the dealloc wait for X11).

## III.2 Fix variants (branch kda-sm100-ordfix2, pushed; head 527721c; default = pwq since 8080515; DM_BWD_OPTS=noxscope = debug-only
previous build, PTX byte-identical to bbfea5a; fwd and pass 0 PTX unchanged in every variant)
Mechanism: dm_utils.XS / _xs_install() rewrites, while the bwd kernel is traced, the remote cute.arch.mbarrier_arrive calls (cutlass pipeline
consumer_release / producer_commit of the peer CTA's warps, the dS relay, the dealloc handshake) and the MMA warp's mbarrier waits.
| variant (DM_BWD_OPTS) | arrive side | wait side (leader MMA warp + dealloc) | bwd main 8k 32/4 / 16/4 (paired, private cache) |
|---|---|---|---|
| noxscope (= bbfea5a) | .release.cta remote | .acquire.cta | 1.000 |
| xs:wq | unchanged | try_wait.parity.acquire.cluster | 1.003-1.020 / 1.003-1.004 (acquire side is cheap) |
| xs:awf (dd33930/abc5b60 default) | .release.cluster | + fence.acq_rel.cluster after every wait | 2.34 / 2.34 |
| xs:awq | .release.cluster | acquire.cluster | 1.54 / 1.55 |
| **xs:pwq (DEFAULT 8080515)** | .release.cluster only when the target is the OTHER CTA (mapa + %cluster_ctarank predicate), .release.cta for own-CTA targets | acquire.cluster | **1.47-1.49 / 1.49** |
| xs:fawq (ruling's fence form) | per arriving thread fence.acq_rel.cluster + .cta arrive | acquire.cluster | 1.58 / 1.59 |
| xs:rly (hd256 KV2 relay) | CTA-1 warps .release.cta (count 2) on their own (unused) barrier copy; CTA 1's idle MMA warp relays ONE .release.cluster arrive per site and phase | acquire.cluster | per-lane blocking relay 1.94 / 1.97; single-thread polling relay 2.05 / 2.05 |
Per-role attribution of the cluster-scope arrives (xsr:, awq): relay +12-14%, compute (S_P, dKV) +28-30%, dsqw (dP, dS) +41-45%, reduce (dQ) +29%,
dealloc +1%. SASS (kda/runs/ordinv/x3p/sass/): every .release.cluster arrive / fence.acq_rel.cluster lowers to MEMBAR.ALL.GPU (noxscope 2, pwq 12);
the .acquire.cluster try_wait adds none. Reading: on hd128 the per-iteration time is ~2 us and each cluster-scope release costs ~0.5 us on the
critical path; the KV2 relay (hd256: +7-11%, much longer iterations) serializes 4 such releases per iteration in one thread and is slower here.
No formal variant below +47% bwd main was found.

## III.3 Evidence for the default 8080515 (pwq) - accuracy-audit 09:1x private-cache rule (fresh FLASH_ATTENTION_CUTE_DSL_CACHE_DIR per tree
under kda/runs/ordinv/x3p/, AOT + the GPU processes of the same run; DM kernel compiles that miss the fake-tensor AOT key happen inside the GPU process
before gpupool_ready into the same fresh cache and are logged '[ordinv] compile')
- Static control (the planted control for a scope bug; a runtime plant is not feasible because the weaker scope has no observable effect on this
  hardware): kda/tests/ordinv/xscope_check.py on the PTX compiled into the private caches: bbfea5a FAIL (10 unscoped remote arrives, no cluster
  acquire), 8080515 PASS (10 remote arrive sites as .release.cluster/.cta pairs, 24 acquire.cluster waits), abc5b60 PASS (awf form)
  (x3p/ptx/, x3p/ptx4/xscope_check.txt).
- Bitwise vs bbfea5a, separate trees and caches: O, LSE, delta, dQ, dK, dV equal at 8 headline shapes x randn + growmax: abc5b60 16/16
  (x3p/bw/summary.txt), 8080515 16/16 (x3p/bw4/). xs:rly bitwise on t_bits 4/4 (x3p/bisect/tbits_rly.log).
- Gates on 8080515 (x3p/gates4/): t_small PASS det+nd, t_odd no NaN, t_bits noxscope 4/4, identity PASS, det stress 8k 32/4 randn 400 + 16/4
  growmax 300 calls bad 0 (same on abc5b60: x3p/gates/).
- A/B e2e det fwd+bwd, 8080515 vs noxscope, private cache (x3p/ab4/summary.txt; fwd column = A/A since fwd is identical): bwd main 1.44-1.54x,
  e2e per call 1.25-1.33x, back-to-back 1.24-1.29x at all 8 shapes. (abc5b60 awf: bwd 2.2-2.5x, e2e 1.79-1.89x; x3p/ab/summary.txt.)
- The relay planted control xrp:<k> (forward one phase early) exists for the relay variant but was not run: the relay is not adopted.

## III.4 Status
Cheapest formal variant = pwq, default on kda-sm100-ordfix2 (8080515 == 527721c default PTX). Its cost (+25..33% e2e det, +44..54% bwd main) is far
above the ~5% threshold -> ESCALATION to the user via the lead (accuracy-audit 10:3x), with the KV2 precedent: the same relay that costs +7-11% on hd256
costs 2x here. Until decided: bbfea5a / bb1f2f7 stay the validated headline (latent formal violation, no observed failure).


### III.U USER DECISION (2026-11-10 ~13:00, via lead) - Part III cross-CTA scope (7 sites X3-X7, X10, X11)
- The formal gap is ACCEPTED for now and documented here. Correctness currently rests on the per-site writer fences
  (wait::st/ld + tcgen05.fence::before_thread_sync / fence.proxy.async + bar.warp.sync) before each remote .release.cta arrive;
  accuracy-audit's micro shows the UNFENCED form returns stale data on GB200 (31/36 runs), so those fences are load-bearing.
- bbfea5a / d5beca5 stay the validated hd128 headline under accuracy-audit's interim position (VALIDATED.md:499).
- pwq / ordfix2 (+25-33% e2e) is NOT made default.
- kda-sm100-xexempt continues (red.async / exempt hardware-completion designs under the 5 rasy conditions); one lands only if formally
  correct AND <= ~5% e2e det per call. Status: rasy +5.5..8.8% (8k/32k), rrasy (relay) being measured.
- Rule: no new hd128 work may add cross-CTA remote .cta arrive hand-offs of this kind (kda/docs/fork-rules.md top).
