# Delta-Matching sm90 kernel evolution toward speed of light (branch kda-sm90)

## Goal Description

Make the Delta-Matching fp8 attention kernels faster on H100 (sm90) without changing their numerics: the forward
(`fwd_fp8`), the matched-delta pass 0, pass 1 (dK / dV and the dS' tiles), pass 2 (dQ), the Triton producers, the twins
and the torch glue around them. Work proceeds in rounds of one measured hypothesis each, ordered by the time a kernel
wastes against its modeled speed of light (largest gap first, re-ranked after every round), each round reviewed by
Codex through the RLCR loop. Accepted rounds are committed one per improvement on `kda-sm90` with the measurement in the
message and pushed to `origin kda-sm90`. The campaign stops after two consecutive rounds without an accepted gain, or
when every kernel is within 10% of its modeled speed of light, and ends with `results/REPORT.md`.

Baseline (results/baseline.md, sum over the main grid; measured / modeled SOL / wasted): pass 1 507.7 / 129.9 / 377.9 ms
(26%), forward 187.0 / 64.7 / 122.3 ms (35%), pass 2 106.1 / 62.8 / 43.3 ms (59%), torch glue 36.6 / 3.0 / 33.6 ms,
pass 0 36.1 / 16.2 / 19.9 ms (45%), producers + twins 27.4 / 17.7 / 9.7 ms. FA3's fp8 forward is about 25% faster than
ours and FA3's whole bf16 fwd+bwd beats our fp8 fwd+bwd at every shape.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: Measurement and equivalence tooling exists, is committed, and is trustworthy.
  - AC-1.1: `benchmarks/equivalence.py` runs a fixed shape set (D 128 / 256; MHA, GQA 4, a GQA 3 or MQA case; B 1 and
    B > 1; causal matched, causal stale, full stale; the shortest legal N, a 512-row promotion boundary, a held-out
    shape; plus stress inputs: zeros, constant K, scaled q/k (x3, x10), large outliers) with one library / package in its
    own process, dumps O, dQ, dK, dV, LSE and the producer codes / scales / twins and the matched delta, and compares two
    dumps byte for byte (uint8 / int views, so signed zeros count).
    - Positive: two dumps of the unmodified baseline in two processes compare bitwise equal (determinism).
    - Negative: a dump made with a deliberately perturbed library (for example a different softmax scale fed to one
      op, or a one-ulp change injected into dV) is reported as a mismatch naming the tensor and the first index.
  - AC-1.2: `benchmarks/kernels.py` accounting includes the producer layout copies and the GQA fold under glue, psi
    traffic, the k-mean traffic and tile-granular dS' bytes (visited 64 x kbn tiles, not exact pairs), and the column
    is labelled "modeled SOL"; `benchmarks/summarize.py --base` fails when a shape present in one file is missing from
    the other.
    - Positive: re-summarizing the baseline JSON with the corrected model reproduces the measured times and gives a
      non-zero modeled glue for MHA.
    - Negative: comparing a JSON with a dropped shape against the baseline exits non-zero.
  - AC-1.3: A paired A/B timing mode runs the baseline library and the candidate in the same fresh job, alternating
    order over at least 3 repetitions, and reports per-kernel medians and spread.
    - Positive: baseline vs itself shows |change| below 1% on the per-kernel sums.
    - Negative: an artificially slowed candidate (for example an extra kernel launch or sleep in a hook) is reported
      as a regression.
- AC-2: Every accepted round preserves the numerics contract.
  - AC-2.1: `python -m pytest tests/` passes on the candidate build (expected skips only: the two-GPU routing test and
    the FA2 cross-check when flash_attn 2 is absent).
    - Positive: 49 passed, 3 skipped as on the baseline.
    - Negative: a candidate that breaks the plain/API bitwise route test or the fp32 bounds fails the round.
  - AC-2.2: For schedule / layout / launch changes, all AC-1.1 outputs and intermediates are byte-identical to the
    frozen baseline (and to the previous accepted revision), across the unsplit route, forced `DM_BWD_HEAD_GROUPS=kv`
    and `head`, and `DM_BWD_STREAMS=2`.
    - Positive: equivalence reports zero mismatches on every shape and route.
    - Negative: any single mismatching byte rejects a change that was declared bitwise.
  - AC-2.3: A change that is not bitwise is accepted only as an explicitly named numerics change: its commit says so,
    the certified flag set / version stamp is updated if flags change, outputs stay deterministic, the relative error
    against fp32 is not larger than the baseline's on any equivalence shape, and the matched-mode rowsum(dS) residual
    (DM_DBG build, normalized by rowsum |dS|) stays at fp32 round-off.
    - Positive: such a change documents all four measurements in the commit message and results/.
    - Negative: a change whose error vs fp32 rises on any shape, or whose rowsum residual grows beyond fp32 round-off,
      is rejected.
- AC-3: Every accepted round is a measured, confirmed speedup with no regressions.
  - AC-3.1: The target kernel's time summed over the main-grid shapes the change affects (for example hd128 only for an
    hd128-only change) drops by at least 3% versus the previous accepted revision, in the paired A/B of AC-1.3, and the
    gain is confirmed by a second fresh job.
    - Positive: two separate jobs both show >= 3% on the target kernel sum.
    - Negative: a gain seen in one job only, or below 3%, is not accepted.
  - AC-3.2: No held-out shape and no main-grid shape regresses by more than 1% in end-to-end fwd+bwd (beyond the
    measured run-to-run spread), and `benchmarks/bench.py` at D 128 and D 256 (N 8192) does not regress by more than 1%.
    - Positive: the summarize comparison shows every held-out change <= +1%.
    - Negative: a held-out shape at +1.5% end to end rejects the round even if the target kernel improved.
  - AC-3.3: The build shows no new register spills, local-memory stack, function outlining or wgmma serialization
    (ptxas / cuobjdump resource usage and SASS inspected for the changed kernels).
    - Positive: resource usage of the changed kernel matches the baseline's spill-free profile.
    - Negative: a candidate with "wgmma serialization" warnings or non-zero spill stores is rejected or reworked.
- AC-4: Git protocol.
  - Positive: each accepted improvement is its own commit on `kda-sm90` authored by Hao Kang <haok1402@gmail.com>, with
    the measurement in the message, pushed with `git push origin kda-sm90` right after the commit; rejected rounds are
    recorded in `results/rounds.md` (hypothesis, patch summary, evidence, reason) without code changes left behind.
  - Negative: no push to main or upstream, no commit with another author identity, no uncommitted source change left
    at the end of a round.
- AC-5: Final report `results/REPORT.md`: per-kernel before / after (ms, TFLOPS, % of modeled SOL) on the main and
  held-out grids, FA3 comparisons, what worked, what did not (with the rejected rounds), and what remains.
  - Positive: every number in the report traces to a committed JSON in results/.
  - Negative: a claim without a recorded measurement is not allowed.

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)

Several accepted rounds across pass 1 (hd128 and d256), the forward producers and glue, pass 2, the forward and pass 0,
each a schedule / layout / launch change that is byte-identical to the baseline, plus the tooling of AC-1 and the final
report. Larger architectural changes (new workspace layouts, a persistent pass 1, fused producers) are in scope when
they stay deterministic and bitwise and fit one round each.

### Lower Bound (Minimum Acceptable Scope)

The AC-1 tooling, at least one attempted round on the largest-gap kernel (pass 1), every round recorded with its
evidence, and `results/REPORT.md` describing the before / after state and what remains, even if no round is accepted.

### Allowed Choices

- Can use: reordering of independent instructions and asynchronous wgmma issue / wait points, split-phase barriers
  (mbarrier) that preserve the same values, pipeline depth and shared-memory staging changes, launch geometry and
  scheduling, strided addressing in the Triton producers, removing host-side copies, CUDA graphs only where the API
  already supports them, new benchmark / equivalence scripts under benchmarks/ and results under results/.
- Cannot use: atomics or non-deterministic reductions, changes to the block-scale contract, quantization rules, psi /
  F1 / F2 / F2b guards, rounding modes or summation orders (unless explicitly a numerics change under AC-2.3), changes
  to the public API contract, pushes to main or upstream, GPU work outside sbatch jobs.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

Round template: (1) state the hypothesis, the code path it touches, the expected affected subset and a register /
dependency budget; (2) implement on a clean tree; (3) JIT-build into its own directory (`DM_JIT_DIR`) via an sbatch
job, inspect resource usage; (4) pytest + equivalence vs the frozen baseline package in separate processes; (5) paired
A/B timing on main + held-out grids and bench.py, then a second fresh job; (6) accept (commit + push + results JSON) or
reject (record in results/rounds.md, revert).

Hypothesis backlog, ranked by modeled waste and re-ranked after each round:
- H1a (pass 1 hd128, `csrc/mainloop_bwd_sm90_fp8r.hpp`): issue the dV gemm (P conversion, dV window rescale, dOt wait)
  before the cross-warpgroup psi barrier, after every reader of the fp32 P fragment is done; the barrier wait then
  overlaps tensor-core work. Same instructions per element, so bitwise.
- H1b (same file): pre-pack the 8 dS' transpose words from `rdS` before issuing the dK gemm, then run the shuffles,
  stmatrix, WG barriers and TMA store while dK is in flight (wgmma A registers must not be read in flight). Needs an
  explicit register-liveness check at 240 regs/thread (the 24 / 240 split leaves almost no headroom).
- H3 (producers, `delta_matching/producers.py`, `quant.py`): read the strided (B, N, H, D) inputs directly in the Q / K /
  V quantizers and the k-mean kernels instead of `reshape(BH, N, D).contiguous()` (three full copies per call: 0.57 ms at
  32k tokens, H 16, D 128). Keep the 64-row partial k-mean and its serial fold order. Bitwise by construction; verify.
- H2 (pass 1 d256, `csrc/mainloop_bwd_sm90_fp8v10.hpp`): its shared-memory P^T / dS'^T exchanges and CTA-wide barriers
  need their own design (not a copy of H1).
- H4 (pass 2): pipeline depth / occupancy / K-twin reuse for the dS' stream; keep the unconditional release barrier
  that fixed a previous hang.
- H5 (forward): register-constrained (PV-first ordering at 240 regs); needs a register plan before any overlap change.
- H6 (pass 0), H7 (persistent pass 1 with K/V empty/full handshakes and epilogue ordering), split-path copies in
  `delta_matching/split.py`, a coalesced GQA dK/dV fold kernel (access-pattern only; the reshape is already a view),
  H8 (fused twins: the source records them as slower; only with a new coalesced design).

### Relevant References

- `csrc/mainloop_bwd_sm90_fp8r.hpp` - pass 1 at hd128: `mma()` / `bwd_step` (S, dP, exp, dS, psi barrier, conversions,
  dS' transpose + TMA store, dV / dK wgmma, dK fold, dV promote)
- `csrc/flash_bwd_kernel_sm90_fp8r.h` - pass 1 hd128 kernel wrapper (producer warp, 24 / 240 register split)
- `csrc/mainloop_bwd_sm90_fp8v10.hpp`, `csrc/flash_bwd_kernel_sm90_fp8v10.h` - pass 1 at d256
- `csrc/flash_bwd_fp8v10_launch.h` - pass 1 / pass 2 launch glue and tile shapes
- `csrc/fq8_dq_pass2_hd128.h`, `csrc/fq8_dq_pass2.h` - pass 2 (dQ)
- `csrc/mainloop_fwd_sm90_fp8f.hpp`, `csrc/softmax_fq8f.h`, `csrc/flash_fwd_fp8f_launch.h` - forward
- `csrc/fp11_delta_v6.h`, `csrc/fp11_delta_v4.h` - pass 0
- `delta_matching/producers.py`, `quant.py`, `doq.py`, `twins.py` - Triton producers
- `delta_matching/attn.py`, `core.py`, `split.py` - routing, GQA fold, head-group split
- `delta_matching/_buildcfg.py`, `csrc/dm_flagset.h` - the certified flag set
- `benchmarks/kernels.py`, `benchmarks/summarize.py`, `results/baseline.md` - measurement
- `tests/test_dm_attn.py`, `tests/test_api.py` - correctness gates

## Dependencies and Sequence

### Milestones
1. Tooling: equivalence dumps and byte comparison, corrected accounting, paired A/B timing, a frozen baseline package
   (library + Python sources) for separate-process comparisons.
2. Pass 1 hd128 rounds (H1a, then H1b): the largest modeled gap.
3. Re-rank, then the next rounds from the backlog (H3 producers, H2 pass 1 d256, H4 pass 2, H5 forward, H6 pass 0,
   H7 and the rest), one hypothesis per round.
4. Final report when the stop rule triggers.

Milestone 1 precedes every round (the gates depend on it). Within milestone 2, H1b builds on the H1a schedule when H1a
is accepted, otherwise on the baseline. Later rounds each compare against the latest accepted revision and the frozen
baseline.

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | benchmarks/equivalence.py: dumps + byte compare, stress inputs, routes; frozen baseline package under .local | AC-1.1, AC-2.2 | coding | - |
| task2 | kernels.py accounting fixes (glue copies, psi, k-mean, tile-granular dS'), summarize strictness, paired A/B mode | AC-1.2, AC-1.3 | coding | - |
| task3 | Round H1a: dV gemm issued before the psi barrier in pass 1 hd128; build, resources, gates, A/B, confirm | AC-2, AC-3, AC-4 | coding | task1, task2 |
| task4 | Review the H1a / H1b register-liveness and wgmma group ledger | AC-3.3 | analyze | task3 |
| task5 | Round H1b: pre-packed dS' transpose + TMA store overlapping the dK gemm | AC-2, AC-3, AC-4 | coding | task3, task4 |
| task6 | Re-rank the backlog from the latest measurements | AC-3 | analyze | task5 |
| task7 | Next rounds from the backlog (H3, H2, H4, H5, H6, H7, ...), one hypothesis each, until the stop rule | AC-2, AC-3, AC-4 | coding | task6 |
| task8 | results/rounds.md log of every round, accepted or not | AC-4, AC-5 | coding | task3 |
| task9 | results/REPORT.md final report | AC-5 | coding | task7, task8 |

## Claude-Codex Deliberation

### Agreements
- H1 is legal only if every dependency of the dV gemm moves with it and the fp32 P fragment's readers finish first;
  wgmma A / accumulator registers must not be accessed in flight, so the transpose must use pre-packed copies.
- The 24 / 240 register split leaves almost no headroom; every round needs resource / spill / serialization evidence.
- Bitwise changes and numerics changes are separate classes; bitwise claims need byte comparisons of outputs and
  intermediates, in separate processes, across routes and stress inputs.
- The SOL model is an optimistic reference ("modeled SOL"), and its byte accounting needs the corrections of AC-1.2.
- H2 (d256), H4 (pass 2), H5 (forward), H7 (persistence) and H8 (fused twins) each need their own design; the source
  documents constraints (pass 2's unconditional release barrier, the forward's register-limited PV-first order, fused
  twins measured slower).
- Rejected rounds are recorded with their evidence.

### Resolved Disagreements
- GQA fold copy (draft H3 claimed `permute(...).reshape(B, Hk, rep, N, D)` copies): Codex showed the reshape is a view;
  the measured copies are the producers' `.contiguous()` of Q / K / V (0.57 ms MHA vs 0.20 ms GQA 4 matches 48 vs 24
  head-tensors). Resolution: H3 targets the producer copies only; a GQA fold kernel is a separate, access-pattern-only
  backlog item.
- Non-bitwise escape hatch: the user's brief allows a change of the certified flag set when the commit says so and shows
  the numerics unchanged or better; Codex asks for a distinct identity and stricter criteria. Resolution: AC-2.3
  (named numerics change, flag set / stamp updated, per-shape error not larger, rowsum residual at round-off), with
  bitwise schedule changes preferred.
- Meaning of "3% on the target kernel": resolved as the target kernel's time summed over the affected main-grid subset,
  paired A/B against the previous accepted revision, confirmed in a fresh job (AC-3.1).
- Stop rule: Codex suggests calibrated practical ceilings instead of 10% of SOL and not stopping after two failed rounds
  in unrelated kernels; the user's brief fixes the stop rule (two consecutive rounds without an accepted gain, or every
  kernel within 10% of its SOL). Resolution: keep the user's rule; report the modeled SOL and its limits honestly.
- Workload weighting for the ranking: the user's brief ranks by wasted time on the shape grid; resolution: the
  equal-weight main grid (all modes) as committed in results/baseline.md.

### Convergence Status
- Final Status: `partially_converged` (direct mode: one Codex first-pass analysis, no convergence rounds)

## Pending User Decisions

- DEC-1: Scope of architectural changes (new workspace layouts, deterministic fusion of pass 1 / pass 2, persistent
  pass 1)
  - Claude Position: in scope when deterministic, bitwise and one round each; prioritized after schedule-only rounds.
  - Codex Position: N/A - open question (keep larger architectural changes separate from schedule-only rounds).
  - Tradeoff Summary: larger changes can remove the quadratic dS' traffic but cost many rounds and risk numerics.
  - Decision Status: Resolved by the user's brief (standing approval for the agent's own next steps; one hypothesis per
    round): in scope, one round each, after the schedule-only rounds.

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers
- These terms are for plan documentation only, not for the resulting codebase
- Use descriptive, domain-appropriate naming in code instead
- Follow the existing source comment style (short development tags such as `## M24` are the repository's own convention
  for marking a change in csrc/ comments)

### Environment Notes
- Every GPU step is an sbatch job (1 GPU, `--wait`), sourcing `.local/env.sh` (CUDA 13.0.2 toolkit, cuda-compat 580
  libcuda first on LD_LIBRARY_PATH, the dm venv). Builds: `python -m delta_matching.jit` with `DM_JIT_DIR` per variant.
- Nsight Compute counters are unavailable (ERR_NVGPUCTRPERM); evidence is CUPTI timing, SASS and resource usage.
- Pushes go to `origin kda-sm90` only, after checking the commit author.

## Output File Convention

This template is used to produce the main output file (e.g., `plan.md`).

### Translated Language Variant

When `alternative_plan_language` resolves to a supported language name through merged config loading, a translated variant of the output file is also written after the main file. Humanize loads config from merged layers in this order: default config, optional user config, then optional project config; `alternative_plan_language` may be set at any of those layers. The variant filename is constructed by inserting `_<code>` (the ISO 639-1 code from the built-in mapping table) immediately before the file extension:

- `plan.md` becomes `plan_<code>.md` (e.g. `plan_zh.md` for Chinese, `plan_ko.md` for Korean)
- `docs/my-plan.md` becomes `docs/my-plan_<code>.md`
- `output` (no extension) becomes `output_<code>`

The translated variant file contains a full translation of the main plan file's current content in the configured language. All identifiers (`AC-*`, task IDs, file paths, API names, command flags) remain unchanged, as they are language-neutral.

When `alternative_plan_language` is empty, absent, set to `"English"`, or set to an unsupported language, no translated variant is written. Humanize does not auto-create `.humanize/config.json` when no project config file is present.

--- Original Design Draft Start ---

# Draft: evolve the delta_matching sm90 kernels toward speed of light

## Goal

Make the Delta-Matching fp8 attention kernels (forward, matched-delta pass 0, pass 1 dK/dV, pass 2 dQ, the Triton
producers, the twins and the torch glue around them) faster on H100 (sm90) without changing their numerics, one
measured hypothesis per round, on branch `kda-sm90`.

## Where we start (measured, results/baseline.md, commit "Add per-kernel speed-of-light benchmark")

Environment: H100 SXM, driver 535 + cuda-compat 580 (CUDA 13.0.2), torch 2.12.0+cu130, Triton 3.7.0, FA3 built from the
pinned flash-attention commit. Sustained SM clock about 1950 MHz under this load -> dense fp8 peak 2109 TFLOPS; HBM3
3.35 TB/s. Nsight Compute counters are not accessible on this cluster (ERR_NVGPUCTRPERM), so evidence comes from
CUPTI kernel timings (benchmarks/kernels.py), SASS / resource usage (cuobjdump) and controlled A/B builds.

Waste ranking over the main grid (D 128/256, N 2k-32k at 32k tokens, MHA/GQA4, causal matched, causal stale, full stale):

| kernel | measured | SOL | % of SOL | wasted |
|---|---:|---:|---:|---:|
| pass 1 (dK, dV, dS' tiles) | 507.7 ms | 129.9 ms | 26% | 377.9 ms |
| forward | 187.0 ms | 64.7 ms | 35% | 122.3 ms |
| pass 2 (dQ) | 106.1 ms | 62.8 ms | 59% | 43.3 ms |
| torch glue (layout copies, GQA fold) | 36.6 ms | 3.0 ms | 8% | 33.6 ms |
| pass 0 (matched delta) | 36.1 ms | 16.2 ms | 45% | 19.9 ms |
| producers + twins | 27.4 ms | 17.7 ms | | 9.7 ms |

Reference points: FlashAttention-3 fp8 forward is about 25% faster than our forward; FA3's whole bf16 fwd+bwd is faster
than our fp8 fwd+bwd at every shape (for example D128 N16k causal: 12.7 vs 16.3 ms).

## Constraints (the numerics contract)

- README.md and csrc/README.md define the contract. The flag set in delta_matching/_buildcfg.py (enforced by
  csrc/dm_flagset.h) is the kernel's numerical identity. Prefer changes whose outputs are BITWISE identical to the
  baseline library (schedule, layout, launch, addressing changes). If a change alters the certified flag set, the commit
  says so and shows the numerics are unchanged or better.
- Outputs must stay deterministic; the matched-mode identity rowsum(dS) = 0 must hold to fp32 round-off; errors against
  fp32 must not exceed the current values.
- No pushes to main or upstream; the branch kda-sm90 is pushed to origin after every accepted commit.
- All GPU work runs as sbatch jobs (1 GPU, --wait); builds through Slurm.

## Acceptance of a round

1. `python -m pytest tests/` passes on the candidate build.
2. An equivalence check against the baseline library (DM_SO) on a fixed set of shapes (D 128/256, MHA/GQA, causal
   matched/stale, full stale, a held-out shape): O, dQ, dK, dV, LSE bitwise equal, or, if not bitwise, determinism plus
   relative error vs fp32 no worse than baseline and the matched rowsum(dS) residual at fp32 round-off (DM_DBG build).
3. The target kernel is at least 3% faster on the main grid, confirmed by a second fresh job; no regression above 1% on
   the held-out shapes or on benchmarks/bench.py at D 128 and D 256 (end to end).
4. One commit per accepted improvement with the measurement in the message; push to origin kda-sm90.

Stop when two consecutive rounds give no accepted gain, or every kernel is within 10% of its speed of light; then write
results/REPORT.md (per-kernel before/after, % of SOL, what worked, what did not, what remains).

## Hypothesis backlog (ranked by wasted time; one per round, re-ranked after each round)

H1. Pass 1 at hd128 (mainloop_bwd_sm90_fp8r.hpp), schedule overlap inside each tile. Today each warpgroup runs, per
64-row m-block: S and dP wgmma -> exp / dS elementwise -> psi amax -> CTA-wide named barrier (cross-WG psi) -> P and dS'
conversions -> dS' transpose (16 shfl + 16 prmt) -> 2 WG barriers + stmatrix + TMA store -> dV and dK wgmma -> wait ->
dK fold. The wgmma are asynchronous: issue the dV gemm (P^T dO, which does not depend on psi) before the psi barrier,
and run the dS' transpose / stmatrix / TMA store after issuing the dK gemm (pre-packing the transpose words before the
issue, since wgmma A registers must not be read while in flight). Same instructions per element, same order of
arithmetic -> bitwise. Expected: hides the barrier wait and the transpose behind tensor-core time.
H2. Pass 1 at d256 (mainloop_bwd_sm90_fp8v10.hpp): the analogous overlap / barrier reduction, after reading its tile loop.
H3. Forward producers and glue: the Q/K/V quantizers and the k-mean read `x.reshape(BH, N, D).contiguous()` of the
(B, N, H, D) inputs, which is three full transposition copies per call (0.57 ms at 32k tokens, H 16, D 128); read the
strided input directly (as doq_m17 already does for dO). The GQA dK/dV fold `dk.permute(...).reshape(B, Hk, rep, N, D)
.sum(2)` copies before reducing; reduce on the contiguous (B, N, Hk, rep, D) view instead (bitwise: the same fp32 sum of
the same `rep` values per element, verified). Expected about 3-6% of end-to-end fwd+bwd and 15-40% of end-to-end fwd.
H4. Pass 2 (fq8_dq_pass2*.h), 59-76% of the HBM bound at D 128: pipeline depth / occupancy / TMA box shape for the dS'
stream; L2 promotion; schedule only.
H5. Forward (mainloop_fwd_sm90_fp8f.hpp): intra-warpgroup overlap / ping-pong schedule of the per-32-key P basis work,
persistent scheduling details; schedule only.
H6. Pass 0 (fp11_delta_v6.h / fp11_delta_v4.h) analogous schedule work.
H7. Persistent / LPT scheduling and epilogue overlap for pass 1 (non-persistent SingleTileBwdLPTScheduler today).
H8. Launch overhead: twins fused into the producers (DM_FUSED_TWIN path is self-checked), fewer small launches.

## Tooling to add

- benchmarks/equivalence.py: dump O / dQ / dK / dV / LSE for a fixed shape set with a given library (DM_SO) and compare
  two dumps bitwise; also report relative errors vs fp32 and determinism.
- benchmarks/kernels.py and benchmarks/summarize.py (exist): main + held-out grids, before/after comparison.
- results/: each accepted round adds its measured JSON and a summary; results/REPORT.md at the end.

--- Original Design Draft End ---
