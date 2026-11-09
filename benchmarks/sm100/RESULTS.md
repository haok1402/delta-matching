# Results: kda-sm100 candidates vs FA4 (hd128, GB200) — maintained by fa4-bench

## HEADLINE (target: causal fwd+bwd, N = 8k / 32k / 64k x H16/16, H32/8 = 6 shapes, aim 1.10x FA4), updated 2026-11-09 06:10

Status source of truth: sm100/validate/VALIDATED.md. **43db3c7 = PROMOTION PASS in BOTH modes (validator 06:00), the newest
validated commit, no OPEN accuracy items** (= a970641 + the deterministic=False switch; pass 0 / delta / P / dP / dS
unchanged, so rowsum(dS)=0 at round-off in both modes). It supersedes a970641.

**MAIN (user, non-det vs non-det): 43db3c7 non-deterministic vs FA4 non-deterministic default = 0.936x FA4** causal fwd+bwd,
geomean over the 6 target shapes, nvl72013-T18, 4-GPU median per shape. **Target 1.10x: +17.6% still needed.**
Reference (det vs det): 43db3c7 deterministic vs FA4 deterministic = 0.957x.

### Modes used in every column (user 05:20: determinism not required)

| column | FA4 side | our side (DM) |
|---|---|---|
| **MAIN: non-det vs FA4** | FA4 default backward = non-deterministic (atomic dQ). It is FA4's fastest: the 05:19 sweep found tile_n 64 / pack_gqa / no-2CTA all within +-2.4% noise (results/fa4sweep.s0.jsonl) | dm_attn_func(..., deterministic=False) = arm `dm_nd` (bench/cands/kda_full_nd.py): unordered dQ atomics, GQA dK/dV without semaphores |
| reference: det vs FA4-det | FA4 deterministic=True (3-10% slower than its default) | dm_attn_func default (deterministic) = arm `dm_matched` |

| MAIN: our non-det vs FA4 non-det (median of 4 GPUs) | 8k | 32k | 64k |
|---|---|---|---|
| H16/16 | 0.805 | 0.962 | 1.010 |
| H32/8  | 0.860 | 1.000 | 0.997 |
| **gap to 1.10x** H16/16 | +36.6% | +14.3% | +8.9% |
| **gap to 1.10x** H32/8 | +27.9% | +10.0% | +10.3% |
| bwd alone, H16/16 / H32/8 | 0.777 / 0.821 | 0.891 / 0.959 | 0.964 / 0.966 |
| fwd alone, H16/16 / H32/8 | 0.874 / 0.894 | 0.934 / 1.031 | 1.126 / 1.181 |
| reference: our det vs FA4-det, H16/16 / H32/8 | 0.834 / 0.893 | 0.961 / 1.013 | 1.033 / 1.025 |

- Our non-det vs our det in the same runs: fwd+bwd time ratio 0.996..1.012 per shape, i.e. the 2-CTA backward gains
  nothing from non-determinism yet; kda-sm100 expects the real gain from the 1-CTA backward (nd1). Our det vs FA4 non-det:
  0.930x.
- vs a970641 (0.926x, our det vs FA4 non-det, same node): consistent. GPU spread <= 0.056 (32k H16: 0.918-0.974). No SUSPECT
  flags. Sanity errors identical to a970641.
- Next: 6a313b0 (forward trims, bitwise; pending validation) measured +0.4% fwd+bwd vs a970641 (fwd +2.9%), same lease.

Full tables (python bench/headline.py hl_43db3c7 dm_nd; FA4-det reference table = python bench/headline.py hl_43db3c7 dm_matched):

arm=dm_nd  files=results/hl_43db3c7.[sg]*.jsonl  host=nvl72013-T18

vs fa4_bf16  [FA4 default backward = NON-DETERMINISTIC (atomic dQ), its fastest setting on sm100]  (speedup = fa4_bf16_ms / dm_nd_ms; dm_nd = DM build as committed)

| causal | pass | H/Hk | 8k | 32k | 64k | geomean (6 shapes) |
|---|---|---|---|---|---|---|
| 1 | fwd_bwd | 16/16 | 0.805 | 0.962 | 1.010 | **0.936** |
| 1 | fwd_bwd | 32/8 | 0.860 | 1.000 | 0.997 |  |
| 1 | fwd | 16/16 | 0.874 | 0.934 | 1.126 | 1.000 |
| 1 | fwd | 32/8 | 0.894 | 1.031 | 1.181 |  |
| 1 | bwd | 16/16 | 0.777 | 0.891 | 0.964 | 0.893 |
| 1 | bwd | 32/8 | 0.821 | 0.959 | 0.966 |  |

vs fa4_bf16_det  [FA4 deterministic=True backward]  (speedup = fa4_bf16_det_ms / dm_nd_ms; dm_nd = DM build as committed)

| causal | pass | H/Hk | 8k | 32k | 64k | geomean (6 shapes) |
|---|---|---|---|---|---|---|
| 1 | fwd_bwd | 16/16 | 0.831 | 0.973 | 1.036 | **0.963** |
| 1 | fwd_bwd | 32/8 | 0.898 | 1.022 | 1.037 |  |
| 1 | fwd | 16/16 | 0.868 | 0.947 | 1.120 | 1.003 |
| 1 | fwd | 32/8 | 0.894 | 1.045 | 1.188 |  |
| 1 | bwd | 16/16 | 0.815 | 0.949 | 1.000 | 0.945 |
| 1 | bwd | 32/8 | 0.898 | 1.005 | 1.018 |  |

gap to the 1.10x target (causal fwd+bwd vs fa4_bf16): speedup still needed = 1.10 / current

| H/Hk | 8k | 32k | 64k | geomean |
|---|---|---|---|---|
| 16/16 | +36.6% | +14.3% | +8.9% | +17.6% |
| 32/8 | +27.9% | +10.0% | +10.3% |  |

spread across runs/GPUs (min..max of the per-GPU ratio, n = 4), causal fwd+bwd vs fa4_bf16:

| H/Hk | 8k | 32k | 64k |
|---|---|---|---|
| 16/16 | 0.791..0.818 | 0.918..0.974 | 1.000..1.014 |
| 32/8 | 0.833..0.873 | 0.976..1.011 | 0.986..1.004 |

---

## HISTORY 2026-11-09 03:10: NO COMPLIANT HEADLINE (resolved 04:05 by 77f470e)

**Every commit since 3122cc4 (P0-v3) breaks the zero-sum invariant and is NON-COMPLIANT, NOT PROMOTABLE.** The affected
commits are 3122cc4, 30ec6cf, 9c690e1, 69f570a, 5b50ab2, e34bc6b, 3fec800/4ea8bd5, cc1dbf2 and e646cb1, and all their speedups below.
- Rule (user): "You must not break the zero sum invariant. Any operations that trade away the accuracy is banned."
- Evidence: dm-validator found it (validate/runs/ve646/full.json, gate log validate/runs/gate/e646_full.log) and
  accuracy-audit confirmed it (audit/runs/zs_own_p.log). Pass 0 builds delta with a different exp mix (FA4 deg-3 polynomial
  / MUFU) than the backward's P. **Start point (dm-validator, emulation identity on ordinary randn inputs): 21b6b15
  compliant (3.0e-7); 3122cc4 is the first violation (9.5e-5; it contains the f8ed80a poly mix); every commit through
  e646cb1 is ~1e-4.** On adversarial families the kernel-side rowsum(dS) reaches 5e-5..2e-4, against ~1e-7 round-off.
  (Correction 03:40: an earlier version of this note said "~2e-7 on ordinary randn". That value was a different, kernel-own-P
  measurement and does not show the violation; the emulation metric above does.)
- Why my bench missed it: its sanity gate checks rel. L2 of O/dQ/dK/dV vs fp32 on randn inputs only. It does not test the
  rowsum(dS) identity or adversarial inputs, so these headlines passed it. Headline eligibility now requires the
  validator's new identity gate (validator/validate.py) in addition to VALIDATED.md.
- Next: kda-sm100-fix0 (pass 0 = bitwise the backward's P, deg-3 mix dropped; kda expects -1..3% bwd). Once it passes the
  gate I re-measure the headline (4-GPU median) and report the pass-0 slowdown vs e646cb1 separately (same-lease A/B).
- The numbers below are kept as measured, but they are labelled NON-COMPLIANT. The last validated commit before 3122cc4
  (21b6b15) has an OOB bug, so there is no compliant headline right now.




### Pre-measured, PENDING validation: 0c3f69d (pass 0 fp64 flush every tile; supersedes 49ff526), 04:08
Same-lease A/B vs the headline 77f470e (nvl72131-T16, 1 GPU): **matched causal fwd+bwd -1.6% (kernel time +1.1..1.4% on every
shape; bwd +0.7..1.8%, fwd unchanged)**. Single-GPU ratio 0.924x -> 0.909x FA4 (0.949x vs det). kda predicted +0.6-1.5%;
this is the price of closing the 14 OPEN identity items. No SUSPECT flags. The 4-GPU headline follows once it is in VALIDATED.md.

| arm | causal | pass | 77f470e | 0c3f69d | change | 0c3f69d vs FA4-det | per-shape speedup of cand (min..max) | n |
|---|---|---|---|---|---|---|---|---|
| dm_matched | 1 | bwd | 0.887 | 0.872 | -1.7% | 0.927 | 0.982..0.993 | 6 |
| dm_matched | 1 | fwd | 0.995 | 0.995 | -0.0% | 0.996 | 1.000..1.000 | 6 |
| dm_matched | 1 | fwd_bwd | 0.924 | 0.909 | -1.6% | 0.949 | 0.986..0.989 | 6 |
| dm_stale | 1 | bwd | 1.193 | 1.189 | -0.4% | 1.264 | 0.995..1.002 | 6 |
| dm_stale | 1 | fwd | 0.995 | 0.995 | -0.0% | 0.996 | 0.999..1.000 | 6 |
| dm_stale | 1 | fwd_bwd | 1.152 | 1.149 | -0.2% | 1.199 | 0.996..1.007 | 6 |

### Pre-measured, PENDING the identity gate: 49ff526 (P0-v7: fp64-flushed row sums in pass 0; supersedes 77f470e), 04:00
Same-lease A/B vs e646cb1 (nvl72131-T16, 1 GPU): fwd+bwd 0.966x -> **0.925x FA4** single-GPU (0.955x vs FA4-det); kernel
time +1.8..6.4% per shape vs e646cb1 (the total cost of all accuracy fixes). fwd -9.5%, bwd -3.4%. Same as bd94e1f within
noise (cross-run). No SUSPECT flags. The 4-GPU headline runs as soon as 49ff526 passes.

| arm | causal | pass | e646cb1 | 49ff526 | change | 49ff526 vs FA4-det | per-shape speedup of cand (min..max) | n |
|---|---|---|---|---|---|---|---|---|
| dm_matched | 1 | bwd | 0.914 | 0.883 | -3.4% | 0.933 | 0.954..1.000 | 6 |
| dm_matched | 1 | fwd | 1.104 | 1.000 | -9.5% | 1.002 | 0.895..0.934 | 6 |
| dm_matched | 1 | fwd_bwd | 0.966 | 0.925 | -4.2% | 0.955 | 0.940..0.982 | 6 |
| dm_stale | 1 | bwd | 1.191 | 1.200 | +0.7% | 1.267 | 0.998..1.036 | 6 |
| dm_stale | 1 | fwd | 1.104 | 1.000 | -9.4% | 1.002 | 0.896..0.931 | 6 |
| dm_stale | 1 | fwd_bwd | 1.184 | 1.162 | -1.9% | 1.200 | 0.972..0.999 | 6 |

### Pre-measured, PENDING the identity gate: bd94e1f, superseded by 77f470e = same production code + debug hooks behind const_expr flags (accuracy fix = fix0 pass 0 bitwise-backward P + all-MUFU forward), 03:32
Same-lease A/B vs e646cb1, nvl72131-T16 GPU 2, 6 target shapes:
- **Cost of the accuracy fix:** fwd+bwd kernel time +1.8..+4.8% per shape (geomean ratio 0.958x -> 0.925x FA4, -3.4%).
  fwd -8.6% (all-MUFU exp), bwd -2.0% (pass 0 MUFU). kda predicted +3-5%.
- **bd94e1f vs FA4 (single GPU): 0.925x** (0.961x vs FA4-det). Per shape (H16/16 | H32/8): 8k 0.805 | 0.852,
  32k 0.946 | 0.986, 64k 0.991 | 0.988. Gap to 1.10x: geomean +18.9% (8k +36.6% | +29.1%).
- Sanity (rel. L2 vs fp32) is unchanged to 4 digits. No SUSPECT flags. The identity itself is dm-validator's gate.
- This becomes the headline (4-GPU median via run_headline.sh) once bd94e1f passes the identity gate and is in VALIDATED.md.
- Staged 4-GPU headline for 77f470e (now the headline, see top) (03:40, nvl72131-T16, median of 4 GPUs; published as the headline only after the
  gate PASS): 0.925x FA4 (0.961x vs FA4-det).** Per shape (H16/16 | H32/8): 8k 0.814 | 0.851, 32k 0.938 | 0.979,
  64k 0.995 | 0.989. Gap to 1.10x +18.9%. GPU spread <= 0.035. No SUSPECT flags. Raw: results/hl_77f470e.g*.jsonl.

| arm | causal | pass | e646cb1 | bd94e1f | change | bd94e1f vs FA4-det | per-shape speedup of cand (min..max) | n |
|---|---|---|---|---|---|---|---|---|
| dm_matched | 1 | bwd | 0.908 | 0.890 | -2.0% | 0.943 | 0.964..0.998 | 6 |
| dm_matched | 1 | fwd | 1.097 | 1.003 | -8.6% | 0.997 | 0.896..0.920 | 6 |
| dm_matched | 1 | fwd_bwd | 0.958 | 0.925 | -3.4% | 0.961 | 0.952..0.982 | 6 |
| dm_stale | 1 | bwd | 1.181 | 1.197 | +1.4% | 1.269 | 1.000..1.030 | 6 |
| dm_stale | 1 | fwd | 1.094 | 1.003 | -8.3% | 0.997 | 0.897..0.926 | 6 |
| dm_stale | 1 | fwd_bwd | 1.176 | 1.153 | -1.9% | 1.198 | 0.972..0.990 | 6 |

## LAST MEASURED (NON-COMPLIANT) HEADLINE (user target: causal fwd+bwd, N = 8k / 32k / 64k x H16/16, H32/8 = 6 shapes, aim 1.10x FA4), updated 2026-11-09 02:55

**[NON-COMPLIANT, NOT PROMOTABLE: breaks the zero-sum invariant] e646cb1 (= cc1dbf2 + msplit; matched / Delta-Matching, deterministic bwd), nvl72131-T16, 4-GPU median per
shape: geomean over the 6 target shapes = 0.960x FA4 (0.994x vs FA4 deterministic). Target 1.10x: +14.6% still needed.**

| matched causal fwd+bwd vs FA4 (median of 4 GPUs) | 8k | 32k | 64k |
|---|---|---|---|
| H16/16 | 0.853 | 0.985 | 1.015 |
| H32/8  | 0.889 | 1.016 | 1.014 |
| **gap to 1.10x** H16/16 (cut in our time needed) | +29.0% (-22.5%) | +11.7% (-10.5%) | +8.4% (-7.7%) |
| **gap to 1.10x** H32/8 | +23.7% (-19.2%) | +8.3% (-7.6%) | +8.5% (-7.8%) |
| bwd alone, H16/16 / H32/8 | 0.819 / 0.836 | 0.918 / 0.959 | 0.955 / 0.964 |
| fwd alone, H16/16 / H32/8 | 0.947 / 0.994 | 1.038 / 1.170 | 1.230 / 1.278 |

- vs the previous headline cc1dbf2 (0.951x, same node and method, 30 min earlier): kernel time is +0.6..1.6% faster on
  every shape (median over GPUs), and the geomean rises +0.9%.
- 8k is the biggest gap (+24-29%). The bwd (0.82-0.96x) is the lever.
- GPU spread <= 0.032 per shape (32k H16: 0.969-1.001). No SUSPECT flags. Validated 02:23 (validate/VALIDATED.md).

Full tables (python bench/headline.py hl_e646cb1):

arm=dm_matched  files=results/hl_e646cb1.[sg]*.jsonl  host=nvl72131-T16

vs fa4_bf16  (speedup = fa4_bf16_ms / dm_matched_ms)

| causal | pass | H/Hk | 8k | 32k | 64k | geomean (6 shapes) |
|---|---|---|---|---|---|---|
| 1 | fwd_bwd | 16/16 | 0.853 | 0.985 | 1.015 | **0.960** |
| 1 | fwd_bwd | 32/8 | 0.889 | 1.016 | 1.014 |  |
| 1 | fwd | 16/16 | 0.947 | 1.038 | 1.230 | 1.102 |
| 1 | fwd | 32/8 | 0.994 | 1.170 | 1.278 |  |
| 1 | bwd | 16/16 | 0.819 | 0.918 | 0.955 | 0.907 |
| 1 | bwd | 32/8 | 0.836 | 0.959 | 0.964 |  |

vs fa4_bf16_det  (speedup = fa4_bf16_det_ms / dm_matched_ms)

| causal | pass | H/Hk | 8k | 32k | 64k | geomean (6 shapes) |
|---|---|---|---|---|---|---|
| 1 | fwd_bwd | 16/16 | 0.881 | 1.009 | 1.045 | **0.994** |
| 1 | fwd_bwd | 32/8 | 0.936 | 1.058 | 1.052 |  |
| 1 | fwd | 16/16 | 0.945 | 1.042 | 1.221 | 1.099 |
| 1 | fwd | 32/8 | 0.994 | 1.156 | 1.277 |  |
| 1 | bwd | 16/16 | 0.856 | 0.968 | 1.003 | 0.961 |
| 1 | bwd | 32/8 | 0.915 | 1.008 | 1.026 |  |

gap to the 1.10x target (causal fwd+bwd vs fa4_bf16): speedup still needed = 1.10 / current

| H/Hk | 8k | 32k | 64k | geomean |
|---|---|---|---|---|
| 16/16 | +29.0% | +11.7% | +8.4% | +14.6% |
| 32/8 | +23.7% | +8.3% | +8.5% |  |

spread across runs/GPUs (min..max of the per-GPU ratio, n = 4), causal fwd+bwd vs fa4_bf16:

| H/Hk | 8k | 32k | 64k |
|---|---|---|---|
| 16/16 | 0.838..0.860 | 0.969..1.001 | 1.008..1.023 |
| 32/8 | 0.880..0.898 | 1.013..1.026 | 1.003..1.015 |

---

All rows are measured by attnbench.py. The candidate and FA4 are timed in the same process on the same GPU, interleaved
in 6 rounds, with every GPU busy (4 shards). Speedup = FA4_ms / cand_ms, geomean over 12 shapes (H16/16 + H32/8,
N 1k-32k). "e2e" means the time starts from bf16 q/k/v, so the quantization producers are included, as in training.

> **Stale mode is reported for reference only.** The headline is **matched (Delta-Matching), causal, fwd+bwd**.

| date | cand (sha) | pass | causal geomean (min-max) | non-causal geomean (min-max) | FA4 fp8 fwd same run | rel.err O vs fp32 | raw |
|---|---|---|---|---|---|---|---|
| 2026-11-08 17:38 | C1-v1 fwd e2e (811db9e) | fwd | 0.902 (0.624-1.170) | 0.936 (0.705-1.161) | 1.173 / 1.030 | 5.0-5.7e-2 | results/c1v1_fwd.s*.jsonl |
| 2026-11-08 17:53 | B1 fwd+bwd, stale mode, deterministic bwd (kda/cands/b1_cur.py @ wt/kda-sm100 7ed21b0) | fwd_bwd | 0.639 (0.368-1.003) vs FA4; 0.661 vs FA4-det | 0.728 (0.438-1.089); 0.746 vs FA4-det | - | O 5e-2, dQ/dK 6.7e-2, dV 5e-2 | results/b1_stale.s*.jsonl |
| 2026-11-08 18:23 | FULL matched (api.dm_attn_func, pinned snap 6a31419: C1-v2 fwd + P0 kernel + B1 det bwd) | fwd_bwd | **0.729** (0.537-0.891); 0.756 vs FA4-det | 0.771 (0.587-0.945); 0.793 vs FA4-det | - | O 5e-2, dQ/dK 6.5e-2, dV 5e-2 | results/full_6a31419.s*.jsonl |
| 2026-11-08 18:23 | FULL stale  [reference only] (same build) | fwd_bwd | 0.686 (0.385-1.047); 0.711 vs FA4-det | 0.800 (0.462-1.169); 0.823 vs FA4-det | - | same | same |
| 2026-11-08 18:53 | FULL matched (snap 02016db: stale delta fused into dO quantizer) | fwd_bwd | **0.730** (0.538-0.893); 0.758 vs FA4-det | 0.772 (0.589-0.942); 0.796 vs FA4-det | - | no SUSPECT | results/full_02016db.s*.jsonl |
| 2026-11-08 18:53 | FULL stale  [reference only] (snap 02016db) | fwd_bwd | 0.861 (0.592-1.097); 0.894 vs FA4-det | 0.945 (0.672-1.186); 0.975 vs FA4-det | - | no SUSPECT | same |
| 2026-11-08 19:16 | FULL matched (snap 5a912b2: P0-v2, deg-5 FMA ex2 polynomial in pass 0) | fwd_bwd | **0.729** (0.540-0.893); 0.760 vs FA4-det | 0.789; 0.809 vs FA4-det | - | no SUSPECT | results/full_5a912b2.s*.jsonl |
| 2026-11-08 19:16 | FULL stale  [reference only] (snap 5a912b2) | fwd_bwd | 0.858; 0.894 vs FA4-det | 0.950; 0.973 vs FA4-det | - | no SUSPECT | same |
| 2026-11-08 19:40 | FULL matched (snap d227693: B1-v3, delta smem loads hoisted; numerics bitwise = 5a912b2) | fwd_bwd | **0.763** (+4.7% vs 5a912b2); 0.791 vs FA4-det | 0.802 (+1.7%); 0.826 vs FA4-det | - | no SUSPECT | results/full_d227693.s*.jsonl |
| 2026-11-08 19:40 | FULL stale  [reference only] (snap d227693) | fwd_bwd | 0.906; 0.940 vs FA4-det | 0.971; 1.000 vs FA4-det | - | no SUSPECT | same |
| 2026-11-08 20:10 | **VALIDATED** FULL matched (snap 21b6b15, in validate/VALIDATED.md; = d227693 + debug hook) on nvl72053-T07 | fwd_bwd | **0.745** (0.551-0.931); 0.776 vs FA4-det | 0.796; 0.821 vs FA4-det | - | no SUSPECT | results/full_21b6b15.s*.jsonl |
| 2026-11-08 20:10 | FULL stale  [reference only] (snap 21b6b15) | fwd_bwd | 0.884; 0.921 vs FA4-det | 0.966; 0.996 vs FA4-det | - | no SUSPECT | same |
| 2026-11-08 21:20 | [NON-COMPLIANT: zero-sum invariant] VALIDATED FULL matched, snap 3122cc4 (= a911afe NaN fix + P0-v3 exp mix + OOB clamp), nvl72053-T07 GPUs 0,1,3 | fwd_bwd | **0.752** (0.551-0.94); 0.781 vs FA4-det | 0.805; 0.829 vs FA4-det | - | no SUSPECT | results/full_3122cc4g3.s*.jsonl |
| 2026-11-08 21:20 | [NON-COMPLIANT: zero-sum invariant] FULL stale  [reference only], snap 3122cc4 | fwd_bwd | 0.890; 0.924 vs FA4-det | 0.968; 0.996 vs FA4-det | - | no SUSPECT | same |
| 2026-11-08 21:55 | [NON-COMPLIANT: zero-sum invariant] VALIDATED FULL matched, snap 30ec6cf (= 3122cc4 + aaca3a0 P0-v4 + 0c3f1e1), nvl72053-T07 GPUs 0-3, same lease as 3122cc4 | fwd_bwd | **0.757** (+1.0% vs 3122cc4 same lease); 0.789 vs FA4-det | 0.807; 0.832 vs FA4-det | - | no SUSPECT | results/ab_3122cc4_30ec6cf_30ec6cf.s*.jsonl |
| 2026-11-08 22:25 | [NON-COMPLIANT: zero-sum invariant] VALIDATED FULL matched, snap 9c690e1 (= 30ec6cf + e2e glue + bwd2; bitwise = 30ec6cf), nvl72053-T07 GPUs 1-3, same lease as 30ec6cf | fwd_bwd | **0.794** (0.581-0.961; +5.2% vs 30ec6cf same lease); 0.827 vs FA4-det | 0.850; 0.876 vs FA4-det | - | no SUSPECT | results/ab_30ec6cf_9c690e1_9c690e1.s*.jsonl |
| 2026-11-08 22:25 | [NON-COMPLIANT: zero-sum invariant] FULL stale  [reference only], snap 9c690e1 | fwd_bwd | 0.943; 0.982 vs FA4-det | 1.031; 1.063 vs FA4-det | - | no SUSPECT | same |
| 2026-11-08 23:30 | [NON-COMPLIANT: zero-sum invariant] VALIDATED (current headline)** FULL matched, snap 5b50ab2 (= 69f570a + posmaj GQA schedule; bitwise = 69f570a), nvl72052-T18 GPUs 2-3, same lease as 69f570a | fwd_bwd | **0.817** (0.626-0.988; +1.6% vs 69f570a same lease); 0.854 vs FA4-det | 0.895; 0.924 vs FA4-det | - | no SUSPECT | results/ab_69f570a_5b50ab2_5b50ab2.s*.jsonl |
| 2026-11-08 23:30 | [NON-COMPLIANT: zero-sum invariant] FULL stale  [reference only], snap 5b50ab2 | fwd_bwd | 0.975; 1.019 vs FA4-det | 1.092; 1.128 vs FA4-det | - | no SUSPECT | same |

### Notes on C1-v1 (fwd only, no bwd yet)
- It is faster than FA4 only at long sequences: 1.05-1.17x at 16k-32k, where FA4 hits the power cap and its clock drops.
  It is slower at <= 4k: 0.62-0.72x at 1k. kda's own screening said geomean 0.993 for the kernel alone. Two reasons it
  reads lower here: (1) e2e includes the producer (quant_qkv2), which costs about 0.08-0.1 ms per call (the dominant
  loss at 1k-2k); (2) kda's screening FA4 times at long shapes were 20-29% higher than in this harness
  (e.g. H32/8 N16k non-causal: 6.62 ms in kda/benchmark.csv vs 5.14-5.30 ms here), which inflated its long-shape ratios.
- Error vs the fp32 reference is 5e-2, the same as FA4's own fp8 fwd (5e-2), against 2.3e-3 for bf16. Pass/fail on
  accuracy is the validator's call.
- FA4's fp8 fwd in the same runs: 1.17x causal / 1.03x non-causal. C1-v1 does not beat FA4's own fp8 fwd yet.

### Notes on B1 (stale mode; matched still uses a torch delta, so it was not timed)
- bwd alone: 0.630x causal / 0.745x non-causal vs FA4 (default, non-deterministic bwd). FA4 with deterministic=True
  is 5-6% slower than default on bwd (0.946-0.948x), so the like-for-like ratio (deterministic vs deterministic) is
  0.661x / 0.746x for fwd+bwd. **The headline uses FA4 default (non-deterministic), the stronger baseline**; the
  deterministic comparison is reported next to it.
- Causal fwd+bwd: 1k 0.37-0.43x, 8k 0.70-0.75x, 32k 0.95-1.00x. It reaches parity only at 32k (and 1.09x at 32k
  non-causal, GQA). Short shapes are dominated by fixed costs: host time is about 1.1 ms per fwd+bwd call against
  0.4 ms for FA4 (separate producer launches), plus the producers' GPU time.
- The fwd inside this composite is 0.64x FA4, below C1-v1 e2e (0.90x). b1_cur.py quantizes with quant_rows x3 + kmean
  instead of the fused quant_qkv2.

### Notes on FULL @6a31419 (first complete Delta-Matching sm100 build; matched causal is the headline)
- Matched, causal, fwd+bwd vs FA4: 1k 0.71x / 0.54x (H16/16 / H32/8), 8k 0.74x / 0.74x, 32k 0.86x / 0.89x.
  fwd 0.90x, bwd 0.68x: **the bwd is the gap.** It is 856-1000 TF against FA4's 1250-1380 TF at >= 8k.
- The P0 matched-delta pass costs about 20% of the bwd at long shapes (32k causal bwd: matched 11.0 vs stale 9.0 ms
  for H16/16; 22.2 vs 18.2 for H32/8). At short shapes, stale is SLOWER than matched (1k causal bwd: 1.21 vs 0.76 ms;
  32/8: 2.77 vs 1.89), so the stale delta path has a large fixed cost (probably the unfused torch rowsum(dO*O) plus
  extra producers).
- Stale wins at 32k only: 1.05x causal H32/8, 1.17x non-causal H32/8.
- The candidate was loaded from a pinned clean snapshot (bench/snap/6a31419); the live worktree had uncommitted edits.

### Notes on FULL @02016db
- Stale improved from 0.686x to 0.861x causal fwd+bwd (0.800x to 0.945x non-causal). The stale bwd at 8k causal H16/16
  went from 2.97 to 2.41 ms, and 1k from 1.21 to 0.65 ms. Matched is unchanged at 0.730x (as expected: kernels unchanged).
- The matched-vs-stale gap is now the P0 (matched delta) pass: bwd 8k causal 3.07 vs 2.41 ms (+27%), 32k 10.98 vs
  8.44 ms (+30%). The stale bwd main kernel alone runs at 1141 TF (8k) and 1303 TF (32k) against FA4's 1304/1313 TF,
  so at long shapes the bwd main kernel is already at parity; **P0 is the main lever for the headline (matched).**

### Notes on FULL @5a912b2 vs 02016db
- Matched causal fwd+bwd 0.730x -> 0.729x (per-shape candidate time 1.000-1.005x faster: no change). Matched
  non-causal 0.772x -> 0.789x (+2.2%; the bwd is 1.2-2.7% faster on every shape). Stale and fwd unchanged (+-0.5%).
- The commit is mainly an accuracy change (more exact ex2 in pass 0). It does not make the causal matched pre-pass
  measurably faster: P0 still costs +27-29% of the bwd (8k causal: 3.06 vs 2.41 ms; 32k: 10.96 vs 8.44 ms).

### Delta d227693 vs 5a912b2 (python delta.py 5a912b2 d227693)
The bwd is 3-8% faster on every causal shape; the fwd is unchanged.

| arm | causal | pass | 5a912b2 | d227693 | change | d227693 vs FA4-det | per-shape speedup of cand (min..max) | n |
|---|---|---|---|---|---|---|---|---|
| dm_matched | 0 | bwd | 0.731 | 0.748 | +2.3% | 0.783 | 1.012..1.033 | 12 |
| dm_matched | 0 | fwd | 0.939 | 0.937 | -0.2% | 0.937 | 1.000..1.004 | 12 |
| dm_matched | 0 | fwd_bwd | 0.789 | 0.802 | +1.7% | 0.826 | 1.010..1.027 | 12 |
| dm_matched | 1 | bwd | 0.682 | 0.721 | +5.7% | 0.762 | 1.031..1.084 | 12 |
| dm_matched | 1 | fwd | 0.899 | 0.899 | -0.0% | 0.900 | 1.000..1.001 | 12 |
| dm_matched | 1 | fwd_bwd | 0.729 | 0.763 | +4.7% | 0.791 | 1.024..1.068 | 12 |
| dm_stale | 0 | bwd | 0.929 | 0.958 | +3.1% | 1.003 | 1.014..1.044 | 12 |
| dm_stale | 0 | fwd | 0.939 | 0.936 | -0.2% | 0.936 | 0.998..1.002 | 12 |
| dm_stale | 0 | fwd_bwd | 0.950 | 0.971 | +2.3% | 1.000 | 1.011..1.038 | 12 |
| dm_stale | 1 | bwd | 0.838 | 0.898 | +7.2% | 0.950 | 1.038..1.110 | 12 |
| dm_stale | 1 | fwd | 0.899 | 0.898 | -0.1% | 0.899 | 0.997..1.001 | 12 |
| dm_stale | 1 | fwd_bwd | 0.858 | 0.906 | +5.6% | 0.940 | 1.027..1.084 | 12 |

### 21b6b15 (validated) and the node effect on the ratio
- Same node (nvl72053-T07), back to back: 21b6b15 vs d227693 = +0.1% matched causal fwd+bwd (no performance change, as
  expected for a debug-hook commit).
- Across nodes: d227693 measured 0.763x on nvl72063-T13 and 0.744x on nvl72053-T07. The DM kernel time is the same on
  both nodes (+-0.4%), but **FA4 causal is ~3% faster on nvl72053-T07** (fwd_bwd geomean 1.029x, up to 1.066x per
  shape); non-causal FA4 is the same on both. So a speedup ratio carries about +-2-3% node dependence. Rule from now on:
  commit-to-commit deltas only from runs on the same node, back to back (the previous commit is re-run if the node
  changed). Headline = the latest same-node number, with the node named.
- **Current validated headline (21b6b15): matched causal fwd+bwd 0.745x FA4 (0.776x vs FA4 deterministic) on
  nvl72053-T07; 0.763x on nvl72063-T13.**

| arm | causal | pass | d227693n2 | 21b6b15 | change | 21b6b15 vs FA4-det | per-shape speedup of cand (min..max) | n |
|---|---|---|---|---|---|---|---|---|
| dm_matched | 0 | bwd | 0.747 | 0.746 | -0.1% | 0.783 | 0.991..1.003 | 12 |
| dm_matched | 0 | fwd | 0.939 | 0.927 | -1.3% | 0.925 | 0.966..1.006 | 12 |
| dm_matched | 0 | fwd_bwd | 0.801 | 0.796 | -0.6% | 0.821 | 0.986..1.002 | 12 |
| dm_matched | 1 | bwd | 0.702 | 0.701 | -0.1% | 0.748 | 1.000..1.002 | 12 |
| dm_matched | 1 | fwd | 0.877 | 0.878 | +0.1% | 0.879 | 0.997..1.001 | 12 |
| dm_matched | 1 | fwd_bwd | 0.744 | 0.745 | +0.1% | 0.776 | 0.999..1.001 | 12 |
| dm_stale | 0 | bwd | 0.955 | 0.959 | +0.4% | 1.006 | 1.000..1.009 | 12 |
| dm_stale | 0 | fwd | 0.939 | 0.927 | -1.3% | 0.925 | 0.966..1.006 | 12 |
| dm_stale | 0 | fwd_bwd | 0.969 | 0.966 | -0.3% | 0.996 | 0.992..1.003 | 12 |
| dm_stale | 1 | bwd | 0.875 | 0.874 | -0.1% | 0.933 | 0.999..1.002 | 12 |
| dm_stale | 1 | fwd | 0.877 | 0.879 | +0.2% | 0.880 | 0.999..1.000 | 12 |
| dm_stale | 1 | fwd_bwd | 0.883 | 0.884 | +0.1% | 0.921 | 0.999..1.001 | 12 |

> **2026-11-08 20:12 caveat (dm-validator):** 21b6b15 and all B1 builds give NaN dQ when N has an odd number of
> 128-blocks. The main-suite shapes (N = 1k..32k) are all even, so the timings above stand, but 21b6b15 is no longer a valid
> headline. Fix candidate: a911afe (validation running). The headline moves to a911afe once it is in validate/VALIDATED.md.

### a911afe (NaN-dQ fix + causal mask skip), pre-measured 2026-11-08 20:50, PENDING VALIDATION (not a headline yet)
Same node as 21b6b15 (nvl72053-T07), 3 GPUs: matched causal fwd+bwd 0.742x FA4 (0.774x vs det); no change vs 21b6b15
(per-shape candidate time 0.998-1.006x). The claimed +2% on causal MHA is not visible: the H16/16 causal bwd is
0.998-1.005x on every N. It becomes the headline once it is listed in validate/VALIDATED.md.

| arm | causal | pass | 21b6b15 | a911afe | change | a911afe vs FA4-det | per-shape speedup of cand (min..max) | n |
|---|---|---|---|---|---|---|---|---|
| dm_matched | 0 | bwd | 0.746 | 0.742 | -0.6% | 0.779 | 1.000..1.004 | 12 |
| dm_matched | 0 | fwd | 0.927 | 0.920 | -0.8% | 0.920 | 0.998..1.004 | 12 |
| dm_matched | 0 | fwd_bwd | 0.796 | 0.793 | -0.4% | 0.817 | 0.999..1.004 | 12 |
| dm_matched | 1 | bwd | 0.701 | 0.701 | -0.0% | 0.746 | 0.997..1.006 | 12 |
| dm_matched | 1 | fwd | 0.878 | 0.872 | -0.7% | 0.871 | 0.996..1.008 | 12 |
| dm_matched | 1 | fwd_bwd | 0.745 | 0.742 | -0.3% | 0.774 | 0.998..1.006 | 12 |
| dm_stale | 0 | bwd | 0.959 | 0.953 | -0.6% | 1.002 | 0.998..1.005 | 12 |
| dm_stale | 0 | fwd | 0.927 | 0.920 | -0.8% | 0.920 | 0.998..1.004 | 12 |
| dm_stale | 0 | fwd_bwd | 0.966 | 0.962 | -0.4% | 0.990 | 0.997..1.004 | 12 |
| dm_stale | 1 | bwd | 0.874 | 0.874 | -0.0% | 0.930 | 0.996..1.006 | 12 |
| dm_stale | 1 | fwd | 0.879 | 0.872 | -0.8% | 0.872 | 0.996..1.008 | 12 |
| dm_stale | 1 | fwd_bwd | 0.884 | 0.881 | -0.3% | 0.918 | 0.998..1.006 | 12 |

### 3122cc4 (validated; supersedes 21b6b15 / a911afe, which have OOB reads)
- **Headline: matched causal fwd+bwd 0.752x FA4 (0.781x vs FA4 deterministic), nvl72053-T07.** The same node gave 0.742x
  (a911afe) and 0.745x (21b6b15) earlier.
- Kernel performance has not changed since d227693: per-shape candidate time vs a911afe is 0.998-1.004x (fwd
  0.989-1.000x). P0-v3 (-3% on pass 0 alone) is about 0.5% of the matched bwd, below the noise.
- The ratio still moved +1.3% (3-GPU run) or +1.7% (1-GPU run on GPU0) because **FA4's time depends on which GPUs the
  shards land on** (a911afe on GPUs 1,2,3; 3122cc4 on 0,1,3 or 0 alone). The DM kernel's time is the same on every GPU.
  So treat headline ratios as +-1.5% (GPU placement) and +-2-3% (node). For commit-to-commit deltas, the trustworthy
  metric is the per-shape candidate-time column (same node). The ratio deltas above are within placement noise.

| arm | causal | pass | a911afe | 3122cc4g3 | change | 3122cc4g3 vs FA4-det | per-shape speedup of cand (min..max) | n |
|---|---|---|---|---|---|---|---|---|
| dm_matched | 0 | bwd | 0.742 | 0.756 | +1.9% | 0.792 | 1.005..1.016 | 12 |
| dm_matched | 0 | fwd | 0.920 | 0.929 | +1.0% | 0.930 | 0.994..1.003 | 12 |
| dm_matched | 0 | fwd_bwd | 0.793 | 0.805 | +1.5% | 0.829 | 1.003..1.012 | 12 |
| dm_matched | 1 | bwd | 0.701 | 0.709 | +1.2% | 0.754 | 1.000..1.006 | 12 |
| dm_matched | 1 | fwd | 0.872 | 0.882 | +1.1% | 0.882 | 0.989..1.000 | 12 |
| dm_matched | 1 | fwd_bwd | 0.742 | 0.752 | +1.3% | 0.781 | 0.998..1.004 | 12 |
| dm_stale | 0 | bwd | 0.953 | 0.960 | +0.7% | 1.007 | 0.996..1.000 | 12 |
| dm_stale | 0 | fwd | 0.920 | 0.929 | +1.0% | 0.930 | 0.996..1.001 | 12 |
| dm_stale | 0 | fwd_bwd | 0.962 | 0.968 | +0.6% | 0.996 | 0.996..1.004 | 12 |
| dm_stale | 1 | bwd | 0.874 | 0.881 | +0.8% | 0.936 | 0.996..1.001 | 12 |
| dm_stale | 1 | fwd | 0.872 | 0.882 | +1.1% | 0.882 | 0.992..1.002 | 12 |
| dm_stale | 1 | fwd_bwd | 0.881 | 0.890 | +1.0% | 0.924 | 0.992..1.000 | 12 |

### aaca3a0 (P0-v4, branch kda-sm100-p0r3) vs 3122cc4: same-lease A/B/A, PENDING VALIDATION (2026-11-08 21:50)
New method (bench/run_ab.sh): A, B, then A again in ONE lease, shard i always on the same GPU (nvl72053-T07, GPUs 0,1,2),
so placement and node are identical. A vs A-repeat gives the noise floor: per-shape candidate time 0.993-1.005.
- matched causal bwd: **+0.3..+1.0% faster on every shape** (above the noise floor, below the expected ~2%). For
  example H16/16 8k 2.894 -> 2.868 ms, 32k 10.314 -> 10.215 ms. The fwd+bwd ratio is 0.752x -> 0.756x (+0.5%).
- non-causal fwd 0.999..1.036 (+2% geomean): unexpected for a pass-0-only change. Worth checking whether the fwd shares
  the changed exp code. Matched non-causal bwd and stale are unchanged.

B vs A:
| arm | causal | pass | 3122cc4 | aaca3a0 | change | aaca3a0 vs FA4-det | per-shape speedup of cand (min..max) | n |
|---|---|---|---|---|---|---|---|---|
| dm_matched | 0 | bwd | 0.757 | 0.757 | -0.0% | 0.794 | 0.995..1.008 | 12 |
| dm_matched | 0 | fwd | 0.927 | 0.945 | +2.0% | 0.943 | 0.999..1.036 | 12 |
| dm_matched | 0 | fwd_bwd | 0.806 | 0.811 | +0.6% | 0.836 | 0.999..1.013 | 12 |
| dm_matched | 1 | bwd | 0.711 | 0.716 | +0.7% | 0.761 | 1.003..1.010 | 12 |
| dm_matched | 1 | fwd | 0.879 | 0.880 | +0.1% | 0.881 | 1.000..1.004 | 12 |
| dm_matched | 1 | fwd_bwd | 0.752 | 0.756 | +0.5% | 0.787 | 1.003..1.008 | 12 |
| dm_stale | 0 | bwd | 0.960 | 0.958 | -0.2% | 1.006 | 0.997..1.006 | 12 |
| dm_stale | 0 | fwd | 0.926 | 0.944 | +2.0% | 0.943 | 1.000..1.036 | 12 |
| dm_stale | 0 | fwd_bwd | 0.968 | 0.973 | +0.5% | 1.003 | 0.996..1.010 | 12 |
| dm_stale | 1 | bwd | 0.883 | 0.882 | -0.0% | 0.938 | 0.997..1.002 | 12 |
| dm_stale | 1 | fwd | 0.879 | 0.880 | +0.1% | 0.881 | 0.999..1.004 | 12 |
| dm_stale | 1 | fwd_bwd | 0.890 | 0.889 | -0.1% | 0.926 | 0.998..1.001 | 12 |

A vs A-repeat (noise floor):
| arm | causal | pass | 3122cc4 | 3122cc4_r2 | change | 3122cc4_r2 vs FA4-det | per-shape speedup of cand (min..max) | n |
|---|---|---|---|---|---|---|---|---|
| dm_matched | 0 | bwd | 0.757 | 0.759 | +0.3% | 0.794 | 0.996..1.005 | 12 |
| dm_matched | 0 | fwd | 0.927 | 0.928 | +0.2% | 0.928 | 0.995..1.002 | 12 |
| dm_matched | 0 | fwd_bwd | 0.806 | 0.809 | +0.4% | 0.832 | 0.993..1.004 | 12 |
| dm_matched | 1 | bwd | 0.711 | 0.712 | +0.2% | 0.753 | 0.994..1.001 | 12 |
| dm_matched | 1 | fwd | 0.879 | 0.880 | +0.2% | 0.881 | 0.996..1.002 | 12 |
| dm_matched | 1 | fwd_bwd | 0.752 | 0.753 | +0.2% | 0.783 | 0.997..1.000 | 12 |
| dm_stale | 0 | bwd | 0.960 | 0.964 | +0.4% | 1.009 | 0.998..1.012 | 12 |
| dm_stale | 0 | fwd | 0.926 | 0.928 | +0.2% | 0.927 | 0.997..1.002 | 12 |
| dm_stale | 0 | fwd_bwd | 0.968 | 0.972 | +0.4% | 1.000 | 0.992..1.003 | 12 |
| dm_stale | 1 | bwd | 0.883 | 0.884 | +0.1% | 0.936 | 0.991..1.000 | 12 |
| dm_stale | 1 | fwd | 0.879 | 0.880 | +0.1% | 0.880 | 0.992..1.001 | 12 |
| dm_stale | 1 | fwd_bwd | 0.890 | 0.892 | +0.2% | 0.927 | 0.995..1.001 | 12 |

### 30ec6cf (validated; superseded by 9c690e1) vs 3122cc4, same lease, GPUs 0-3 of nvl72053-T07 (2026-11-08 21:55)
**matched causal fwd+bwd 0.757x FA4 (0.789x vs FA4 deterministic)**, +1.0% vs 3122cc4. Per-shape candidate time is
1.005-1.019x, above the 0.993-1.005 noise floor, from both bwd (+0.2..1.0%) and fwd (+0.2..5.0%, causal). The A-repeat
was cancelled to free the GPUs for kda-sm100; the noise floor is from the 3122cc4/aaca3a0 A/B/A run.

| arm | causal | pass | 3122cc4 | 30ec6cf | change | 30ec6cf vs FA4-det | per-shape speedup of cand (min..max) | n |
|---|---|---|---|---|---|---|---|---|
| dm_matched | 0 | bwd | 0.756 | 0.757 | +0.1% | 0.794 | 0.996..1.001 | 12 |
| dm_matched | 0 | fwd | 0.928 | 0.941 | +1.4% | 0.940 | 0.999..1.041 | 12 |
| dm_matched | 0 | fwd_bwd | 0.806 | 0.807 | +0.2% | 0.832 | 0.995..1.012 | 12 |
| dm_matched | 1 | bwd | 0.708 | 0.713 | +0.7% | 0.759 | 1.002..1.010 | 12 |
| dm_matched | 1 | fwd | 0.878 | 0.893 | +1.8% | 0.895 | 1.002..1.050 | 12 |
| dm_matched | 1 | fwd_bwd | 0.749 | 0.757 | +1.0% | 0.789 | 1.005..1.019 | 12 |
| dm_stale | 0 | bwd | 0.961 | 0.962 | +0.2% | 1.010 | 0.992..1.007 | 12 |
| dm_stale | 0 | fwd | 0.928 | 0.941 | +1.5% | 0.940 | 0.999..1.041 | 12 |
| dm_stale | 0 | fwd_bwd | 0.969 | 0.971 | +0.2% | 1.001 | 0.997..1.013 | 12 |
| dm_stale | 1 | bwd | 0.879 | 0.880 | +0.0% | 0.936 | 1.000..1.001 | 12 |
| dm_stale | 1 | fwd | 0.878 | 0.894 | +1.9% | 0.896 | 1.002..1.057 | 12 |
| dm_stale | 1 | fwd_bwd | 0.887 | 0.892 | +0.5% | 0.930 | 1.000..1.016 | 12 |

### f5a04c6 (e2e glue, merge of 50ded49) vs 30ec6cf, same-lease A/B/A, GPUs 1-3 nvl72053-T07, PENDING VALIDATION (22:11)
Noise floor (A vs A-repeat): per-shape 0.994-1.004.
- **matched causal fwd+bwd 0.753x -> 0.776x FA4 (+3.0%), 0.807x vs FA4-det.** Every shape is faster (1.013-1.057).
- matched causal bwd per N: H16/16 1k..32k +7.8/+5.7/+4.6/+4.0/+3.4/+3.1%; H32/8 +5.2/+3.8/+2.9/+2.1/+1.7/+1.6%.
  This confirms kda's fork screening (bwd +3.4-5% MHA, +1.6-2.9% GQA). The fwd is unchanged.
- matched non-causal fwd+bwd +1.7%. Stale +1.9..3.3%. No SUSPECT flags.
- Becomes the headline when it is in validate/VALIDATED.md (I will reuse this run if the node is unchanged).

B vs A:
| arm | causal | pass | 30ec6cf | f5a04c6 | change | f5a04c6 vs FA4-det | per-shape speedup of cand (min..max) | n |
|---|---|---|---|---|---|---|---|---|
| dm_matched | 0 | bwd | 0.752 | 0.770 | +2.4% | 0.810 | 1.001..1.063 | 12 |
| dm_matched | 0 | fwd | 0.936 | 0.933 | -0.4% | 0.934 | 0.995..1.003 | 12 |
| dm_matched | 0 | fwd_bwd | 0.805 | 0.819 | +1.7% | 0.845 | 1.000..1.046 | 12 |
| dm_matched | 1 | bwd | 0.711 | 0.736 | +3.5% | 0.783 | 1.016..1.078 | 12 |
| dm_matched | 1 | fwd | 0.889 | 0.886 | -0.3% | 0.889 | 0.996..1.001 | 12 |
| dm_matched | 1 | fwd_bwd | 0.753 | 0.776 | +3.0% | 0.807 | 1.013..1.057 | 12 |
| dm_stale | 0 | bwd | 0.955 | 0.983 | +2.9% | 1.033 | 0.999..1.072 | 12 |
| dm_stale | 0 | fwd | 0.936 | 0.933 | -0.4% | 0.934 | 0.995..1.002 | 12 |
| dm_stale | 0 | fwd_bwd | 0.968 | 0.987 | +1.9% | 1.017 | 0.998..1.050 | 12 |
| dm_stale | 1 | bwd | 0.876 | 0.913 | +4.2% | 0.971 | 1.019..1.089 | 12 |
| dm_stale | 1 | fwd | 0.889 | 0.887 | -0.3% | 0.889 | 0.996..1.000 | 12 |
| dm_stale | 1 | fwd_bwd | 0.887 | 0.917 | +3.3% | 0.953 | 1.014..1.063 | 12 |

A vs A-repeat:
| arm | causal | pass | 30ec6cf | 30ec6cf_r2 | change | 30ec6cf_r2 vs FA4-det | per-shape speedup of cand (min..max) | n |
|---|---|---|---|---|---|---|---|---|
| dm_matched | 0 | bwd | 0.752 | 0.754 | +0.2% | 0.791 | 0.998..1.004 | 12 |
| dm_matched | 0 | fwd | 0.936 | 0.938 | +0.1% | 0.937 | 1.000..1.003 | 12 |
| dm_matched | 0 | fwd_bwd | 0.805 | 0.804 | -0.2% | 0.830 | 0.994..1.002 | 12 |
| dm_matched | 1 | bwd | 0.711 | 0.712 | +0.1% | 0.755 | 0.999..1.001 | 12 |
| dm_matched | 1 | fwd | 0.889 | 0.889 | -0.1% | 0.889 | 1.000..1.003 | 12 |
| dm_matched | 1 | fwd_bwd | 0.753 | 0.754 | +0.1% | 0.785 | 1.000..1.002 | 12 |
| dm_stale | 0 | bwd | 0.955 | 0.957 | +0.3% | 1.004 | 0.995..1.008 | 12 |
| dm_stale | 0 | fwd | 0.936 | 0.937 | +0.1% | 0.936 | 0.999..1.004 | 12 |
| dm_stale | 0 | fwd_bwd | 0.968 | 0.967 | -0.1% | 0.999 | 0.998..1.003 | 12 |
| dm_stale | 1 | bwd | 0.876 | 0.877 | +0.1% | 0.931 | 0.999..1.002 | 12 |
| dm_stale | 1 | fwd | 0.889 | 0.889 | -0.1% | 0.889 | 0.998..1.005 | 12 |
| dm_stale | 1 | fwd_bwd | 0.887 | 0.888 | +0.1% | 0.924 | 0.999..1.001 | 12 |

### 9c690e1 (merge of f5a04c6 e2e glue + d3a5837 bwd2) vs 30ec6cf, same-lease A/B/A, GPUs 1-3 nvl72053-T07, VALIDATED 22:16 by dm-validator = CURRENT HEADLINE (22:22)
Noise floor (A vs A-repeat): per-shape 0.993-1.007.
- **matched causal fwd+bwd 0.755x -> 0.794x FA4 (+5.2%), 0.827x vs FA4-det.** Every shape is faster (1.034-1.068).
- matched causal bwd per N: H16/16 +9.5/+8.9/+8.8/+8.1/+8.5/+8.4% (1k..32k); H32/8 +5.8/+5.0/+4.6/+4.9/+4.3/+4.4%. fwd unchanged.
- matched causal fwd+bwd vs FA4 per N: H16/16 0.795/0.798/0.790/0.820/0.849/0.931; H32/8 0.581/0.642/0.709/0.822/0.924/0.961.
  Short GQA (1k-2k) is the weakest point.
- matched non-causal 0.850x (+5.6%). Stale [reference only] now beats FA4 on non-causal (1.031x fwd+bwd, bwd 1.044x).
- Sanity errors vs fp32 are identical to 30ec6cf (O 5.0-5.7e-2, dQ/dK 6.4-6.7e-2, dV 4.9-5.5e-2): the speed-up does not
  come from changed numerics. No SUSPECT flags.
- This becomes the headline once it is in validate/VALIDATED.md (same node: this run is reused).

B vs A:
| arm | causal | pass | 30ec6cf | 9c690e1 | change | 9c690e1 vs FA4-det | per-shape speedup of cand (min..max) | n |
|---|---|---|---|---|---|---|---|---|
| dm_matched | 0 | bwd | 0.752 | 0.807 | +7.4% | 0.850 | 1.046..1.106 | 12 |
| dm_matched | 0 | fwd | 0.937 | 0.935 | -0.2% | 0.934 | 0.987..1.007 | 12 |
| dm_matched | 0 | fwd_bwd | 0.805 | 0.850 | +5.6% | 0.876 | 1.027..1.079 | 12 |
| dm_matched | 1 | bwd | 0.712 | 0.759 | +6.6% | 0.807 | 1.043..1.095 | 12 |
| dm_matched | 1 | fwd | 0.887 | 0.887 | +0.1% | 0.887 | 0.992..1.007 | 12 |
| dm_matched | 1 | fwd_bwd | 0.755 | 0.794 | +5.2% | 0.827 | 1.034..1.068 | 12 |
| dm_stale | 0 | bwd | 0.956 | 1.044 | +9.3% | 1.100 | 1.054..1.139 | 12 |
| dm_stale | 0 | fwd | 0.937 | 0.936 | -0.1% | 0.934 | 0.987..1.006 | 12 |
| dm_stale | 0 | fwd_bwd | 0.968 | 1.031 | +6.5% | 1.063 | 1.033..1.099 | 12 |
| dm_stale | 1 | bwd | 0.878 | 0.949 | +8.1% | 1.009 | 1.053..1.112 | 12 |
| dm_stale | 1 | fwd | 0.887 | 0.888 | +0.0% | 0.887 | 0.992..1.006 | 12 |
| dm_stale | 1 | fwd_bwd | 0.889 | 0.943 | +6.0% | 0.982 | 1.041..1.082 | 12 |

A vs A-repeat:
| arm | causal | pass | 30ec6cf | 30ec6cf_r2 | change | 30ec6cf_r2 vs FA4-det | per-shape speedup of cand (min..max) | n |
|---|---|---|---|---|---|---|---|---|
| dm_matched | 0 | bwd | 0.752 | 0.751 | -0.1% | 0.791 | 0.999..1.006 | 12 |
| dm_matched | 0 | fwd | 0.937 | 0.932 | -0.5% | 0.935 | 0.996..1.007 | 12 |
| dm_matched | 0 | fwd_bwd | 0.805 | 0.803 | -0.2% | 0.829 | 0.993..1.004 | 12 |
| dm_matched | 1 | bwd | 0.712 | 0.711 | -0.2% | 0.756 | 0.999..1.005 | 12 |
| dm_matched | 1 | fwd | 0.887 | 0.887 | +0.0% | 0.888 | 0.996..1.007 | 12 |
| dm_matched | 1 | fwd_bwd | 0.755 | 0.755 | +0.0% | 0.784 | 1.000..1.005 | 12 |
| dm_stale | 0 | bwd | 0.956 | 0.953 | -0.2% | 1.004 | 0.995..1.001 | 12 |
| dm_stale | 0 | fwd | 0.937 | 0.932 | -0.5% | 0.935 | 0.996..1.006 | 12 |
| dm_stale | 0 | fwd_bwd | 0.968 | 0.965 | -0.3% | 0.997 | 0.997..1.001 | 12 |
| dm_stale | 1 | bwd | 0.878 | 0.876 | -0.2% | 0.931 | 0.999..1.004 | 12 |
| dm_stale | 1 | fwd | 0.887 | 0.888 | +0.0% | 0.889 | 0.998..1.007 | 12 |
| dm_stale | 1 | fwd_bwd | 0.889 | 0.889 | -0.0% | 0.923 | 0.999..1.005 | 12 |

### 69f570a (bwd mskipu: CTA-uniform causal-mask skip below the diagonal) vs 9c690e1, same-lease A/B/A, GPUs 0-3 nvl72053-T07, VALIDATED 23:02 (superseded by 5b50ab2 at 23:17) (22:40)
Noise floor 0.997-1.002. matched causal bwd +1.1% (every shape 1.004-1.012); matched causal fwd+bwd 0.797x -> 0.802x
FA4 (+0.6%; 0.836x vs det). Non-causal unchanged (as expected for a causal-mask change). No SUSPECT flags.
(9c690e1 measured 0.797x here on GPUs 0-3 vs 0.794x on GPUs 1-3 before: placement noise, as documented.)

B vs A:
| arm | causal | pass | 9c690e1 | 69f570a | change | 69f570a vs FA4-det | per-shape speedup of cand (min..max) | n |
|---|---|---|---|---|---|---|---|---|
| dm_matched | 0 | bwd | 0.811 | 0.813 | +0.2% | 0.851 | 0.997..1.003 | 12 |
| dm_matched | 0 | fwd | 0.941 | 0.939 | -0.2% | 0.938 | 0.999..1.002 | 12 |
| dm_matched | 0 | fwd_bwd | 0.853 | 0.851 | -0.3% | 0.878 | 0.989..1.006 | 12 |
| dm_matched | 1 | bwd | 0.762 | 0.770 | +1.1% | 0.815 | 1.004..1.012 | 12 |
| dm_matched | 1 | fwd | 0.892 | 0.893 | +0.1% | 0.893 | 0.999..1.003 | 12 |
| dm_matched | 1 | fwd_bwd | 0.797 | 0.802 | +0.6% | 0.836 | 1.002..1.009 | 12 |
| dm_stale | 0 | bwd | 1.049 | 1.050 | +0.0% | 1.099 | 0.986..1.001 | 12 |
| dm_stale | 0 | fwd | 0.941 | 0.939 | -0.2% | 0.937 | 0.997..1.003 | 12 |
| dm_stale | 0 | fwd_bwd | 1.037 | 1.033 | -0.3% | 1.066 | 0.989..1.003 | 12 |
| dm_stale | 1 | bwd | 0.953 | 0.965 | +1.2% | 1.021 | 1.004..1.015 | 12 |
| dm_stale | 1 | fwd | 0.893 | 0.893 | +0.1% | 0.893 | 0.994..1.000 | 12 |
| dm_stale | 1 | fwd_bwd | 0.946 | 0.953 | +0.7% | 0.993 | 1.003..1.012 | 12 |

A vs A-repeat:
| arm | causal | pass | 9c690e1 | 9c690e1_r2 | change | 9c690e1_r2 vs FA4-det | per-shape speedup of cand (min..max) | n |
|---|---|---|---|---|---|---|---|---|
| dm_matched | 0 | bwd | 0.811 | 0.812 | +0.1% | 0.852 | 0.997..1.001 | 12 |
| dm_matched | 0 | fwd | 0.941 | 0.940 | -0.2% | 0.936 | 0.998..1.001 | 12 |
| dm_matched | 0 | fwd_bwd | 0.853 | 0.855 | +0.2% | 0.879 | 0.998..1.002 | 12 |
| dm_matched | 1 | bwd | 0.762 | 0.762 | -0.1% | 0.810 | 0.999..1.002 | 12 |
| dm_matched | 1 | fwd | 0.892 | 0.893 | +0.0% | 0.894 | 0.997..1.000 | 12 |
| dm_matched | 1 | fwd_bwd | 0.797 | 0.796 | -0.1% | 0.831 | 0.999..1.000 | 12 |
| dm_stale | 0 | bwd | 1.049 | 1.050 | +0.0% | 1.102 | 0.986..1.008 | 12 |
| dm_stale | 0 | fwd | 0.941 | 0.940 | -0.1% | 0.936 | 0.999..1.002 | 12 |
| dm_stale | 0 | fwd_bwd | 1.037 | 1.037 | -0.0% | 1.066 | 0.988..1.000 | 12 |
| dm_stale | 1 | bwd | 0.953 | 0.953 | -0.1% | 1.012 | 0.999..1.000 | 12 |
| dm_stale | 1 | fwd | 0.893 | 0.893 | +0.0% | 0.894 | 0.997..1.000 | 12 |
| dm_stale | 1 | fwd_bwd | 0.946 | 0.945 | -0.1% | 0.986 | 0.998..1.001 | 12 |

### 5b50ab2 (validated 23:17, CURRENT HEADLINE) vs 69f570a (validated 23:02), same-lease A/B, new node nvl72052-T18 GPUs 2-3 (23:30)
- **Headline: matched causal fwd+bwd 0.817x FA4 (0.854x vs FA4 deterministic), nvl72052-T18.** +1.6% vs 69f570a.
- The posmaj schedule only affects GQA, as designed. H32/8 matched causal bwd speedup per N: +11.0/+8.0/+5.1/+2.6/+0.9/+0.1%
  (1k..32k), so the short-GQA weak spot improved. H16/16 is unchanged (1.000).
- 5b50ab2 matched causal fwd+bwd vs FA4 per N: H16/16 0.79/0.80/0.80/0.81/0.87/0.97; H32/8 0.63/0.68/0.75/0.85/0.96/0.99.
- Node note: 69f570a matched causal is 0.804x here vs 0.802x on nvl72053-T07 (consistent), but non-causal is 0.885x here
  vs 0.851x there. FA4 non-causal is slower on this node, so non-causal ratios are not comparable across nodes.
- Stale [reference only] is 1.09x FA4 non-causal and 0.975x causal on this node.

| arm | causal | pass | 69f570a | 5b50ab2 | change | 5b50ab2 vs FA4-det | per-shape speedup of cand (min..max) | n |
|---|---|---|---|---|---|---|---|---|
| dm_matched | 0 | bwd | 0.842 | 0.853 | +1.3% | 0.892 | 0.998..1.068 | 12 |
| dm_matched | 0 | fwd | 0.984 | 0.987 | +0.3% | 0.986 | 0.999..1.006 | 12 |
| dm_matched | 0 | fwd_bwd | 0.885 | 0.895 | +1.1% | 0.924 | 0.994..1.055 | 12 |
| dm_matched | 1 | bwd | 0.771 | 0.788 | +2.2% | 0.835 | 0.999..1.110 | 12 |
| dm_matched | 1 | fwd | 0.880 | 0.881 | +0.1% | 0.881 | 0.998..1.003 | 12 |
| dm_matched | 1 | fwd_bwd | 0.804 | 0.817 | +1.6% | 0.854 | 0.999..1.090 | 12 |
| dm_stale | 0 | bwd | 1.091 | 1.107 | +1.5% | 1.158 | 0.992..1.082 | 12 |
| dm_stale | 0 | fwd | 0.983 | 0.986 | +0.3% | 0.985 | 0.998..1.004 | 12 |
| dm_stale | 0 | fwd_bwd | 1.075 | 1.092 | +1.6% | 1.128 | 0.998..1.064 | 12 |
| dm_stale | 1 | bwd | 0.966 | 0.991 | +2.6% | 1.050 | 1.000..1.124 | 12 |
| dm_stale | 1 | fwd | 0.880 | 0.881 | +0.2% | 0.882 | 0.999..1.004 | 12 |
| dm_stale | 1 | fwd_bwd | 0.957 | 0.975 | +1.9% | 1.019 | 1.000..1.099 | 12 |

Noise floor for 5b50ab2 vs 69f570a (A vs A-repeat, same lease): per-shape 0.992-1.008
| arm | causal | pass | 69f570a | 69f570a_r2 | change | 69f570a_r2 vs FA4-det | per-shape speedup of cand (min..max) | n |
|---|---|---|---|---|---|---|---|---|
| dm_matched | 0 | bwd | 0.842 | 0.845 | +0.4% | 0.881 | 0.995..1.008 | 12 |
| dm_matched | 0 | fwd | 0.984 | 0.986 | +0.2% | 0.985 | 0.997..1.005 | 12 |
| dm_matched | 0 | fwd_bwd | 0.885 | 0.886 | +0.2% | 0.917 | 0.992..1.001 | 12 |
| dm_matched | 1 | bwd | 0.771 | 0.769 | -0.3% | 0.818 | 0.999..1.000 | 12 |
| dm_matched | 1 | fwd | 0.880 | 0.881 | +0.2% | 0.881 | 0.998..1.004 | 12 |
| dm_matched | 1 | fwd_bwd | 0.804 | 0.804 | -0.0% | 0.838 | 1.000..1.001 | 12 |
| dm_stale | 0 | bwd | 1.091 | 1.092 | +0.1% | 1.138 | 0.987..1.001 | 12 |
| dm_stale | 0 | fwd | 0.983 | 0.985 | +0.2% | 0.983 | 0.994..1.004 | 12 |
| dm_stale | 0 | fwd_bwd | 1.075 | 1.078 | +0.3% | 1.115 | 0.994..1.003 | 12 |
| dm_stale | 1 | bwd | 0.966 | 0.964 | -0.2% | 1.024 | 1.000..1.004 | 12 |
| dm_stale | 1 | fwd | 0.880 | 0.882 | +0.2% | 0.882 | 0.999..1.004 | 12 |
| dm_stale | 1 | fwd_bwd | 0.957 | 0.957 | +0.0% | 0.997 | 1.000..1.007 | 12 |

### e34bc6b (kda-sm100-gqa: bwd gloop for GQA) vs 5b50ab2, same-lease A/B, GPUs 0-3 nvl72052-T18, PENDING VALIDATION (23:38)
- matched causal fwd+bwd 0.828x -> **0.858x FA4 (+3.6%)**, 0.895x vs det. The gain is entirely short GQA: H32/8 matched
  causal bwd 1k +38.3%, 2k +18.9%; 4k-32k and all MHA unchanged (1.000), so gloop is only selected at N <= 2k.
- Per N vs FA4 (matched causal fwd+bwd): H16/16 0.81/0.80/0.84/0.81/0.87/0.97; H32/8 0.85/0.78/0.80/0.87/0.95/0.99.
- Sanity errors are identical to 5b50ab2 (numerics unchanged). No SUSPECT flags.
- (5b50ab2 measured 0.828x on GPUs 0-3 here vs 0.817x on GPUs 2-3 earlier: placement noise.)

| arm | causal | pass | 5b50ab2 | e34bc6b | change | e34bc6b vs FA4-det | per-shape speedup of cand (min..max) | n |
|---|---|---|---|---|---|---|---|---|
| dm_matched | 0 | bwd | 0.842 | 0.869 | +3.3% | 0.907 | 0.996..1.256 | 12 |
| dm_matched | 0 | fwd | 0.962 | 0.964 | +0.2% | 0.965 | 0.999..1.002 | 12 |
| dm_matched | 0 | fwd_bwd | 0.884 | 0.904 | +2.3% | 0.931 | 0.985..1.197 | 12 |
| dm_matched | 1 | bwd | 0.800 | 0.834 | +4.3% | 0.884 | 0.999..1.383 | 12 |
| dm_matched | 1 | fwd | 0.911 | 0.910 | -0.1% | 0.910 | 0.999..1.004 | 12 |
| dm_matched | 1 | fwd_bwd | 0.828 | 0.858 | +3.6% | 0.895 | 0.999..1.293 | 12 |
| dm_stale | 0 | bwd | 1.092 | 1.136 | +4.0% | 1.186 | 0.994..1.325 | 12 |
| dm_stale | 0 | fwd | 0.962 | 0.964 | +0.2% | 0.965 | 0.996..1.002 | 12 |
| dm_stale | 0 | fwd_bwd | 1.077 | 1.106 | +2.7% | 1.139 | 0.991..1.238 | 12 |
| dm_stale | 1 | bwd | 1.008 | 1.060 | +5.2% | 1.122 | 0.999..1.478 | 12 |
| dm_stale | 1 | fwd | 0.911 | 0.910 | -0.1% | 0.910 | 0.999..1.003 | 12 |
| dm_stale | 1 | fwd_bwd | 0.988 | 1.030 | +4.2% | 1.073 | 0.999..1.346 | 12 |

Noise floor for e34bc6b vs 5b50ab2 (A vs A-repeat, same lease):
| arm | causal | pass | 5b50ab2 | 5b50ab2_r2 | change | 5b50ab2_r2 vs FA4-det | per-shape speedup of cand (min..max) | n |
|---|---|---|---|---|---|---|---|---|
| dm_matched | 0 | bwd | 0.842 | 0.844 | +0.2% | 0.883 | 0.997..1.002 | 12 |
| dm_matched | 0 | fwd | 0.962 | 0.968 | +0.5% | 0.966 | 0.999..1.006 | 12 |
| dm_matched | 0 | fwd_bwd | 0.884 | 0.883 | -0.2% | 0.912 | 0.983..1.002 | 12 |
| dm_matched | 1 | bwd | 0.800 | 0.800 | -0.1% | 0.846 | 0.999..1.002 | 12 |
| dm_matched | 1 | fwd | 0.911 | 0.908 | -0.3% | 0.908 | 0.999..1.002 | 12 |
| dm_matched | 1 | fwd_bwd | 0.828 | 0.831 | +0.3% | 0.866 | 0.999..1.002 | 12 |
| dm_stale | 0 | bwd | 1.092 | 1.095 | +0.3% | 1.146 | 0.999..1.002 | 12 |
| dm_stale | 0 | fwd | 0.962 | 0.966 | +0.4% | 0.965 | 0.995..1.002 | 12 |
| dm_stale | 0 | fwd_bwd | 1.077 | 1.076 | -0.1% | 1.112 | 0.994..1.000 | 12 |
| dm_stale | 1 | bwd | 1.008 | 1.006 | -0.1% | 1.064 | 0.994..1.002 | 12 |
| dm_stale | 1 | fwd | 0.911 | 0.909 | -0.2% | 0.909 | 0.999..1.003 | 12 |
| dm_stale | 1 | fwd_bwd | 0.988 | 0.990 | +0.2% | 1.033 | 0.994..1.001 | 12 |

### 3fec800 (P0-v5: pass-0 two-halves + p3_16_6 exp mix) vs cfb8d13 (= e34bc6b kernels), focus suite, same lease, nvl72052-T18 GPUs 0-1, VALIDATED 00:52 (merged as 4ea8bd5, kernels identical) (00:55)
- matched causal fwd+bwd 0.950x -> **0.962x FA4 (+1.3%)**, 0.995x vs FA4-det. The matched bwd is +1.7..2.7% on EVERY focus
  shape (causal and non-causal). The fwd is unchanged, stale is unchanged (pass-0 only), and sanity errors are bitwise
  equal to cfb8d13. No SUSPECT flags.
- Per N (matched causal fwd+bwd vs FA4): H16/16 0.880 / 0.984 / 1.016, H32/8 0.891 / 1.004 / 1.007 (8k/32k/64k).
  At 32k-64k it is now >= FA4, and 8k is the remaining gap (0.88-0.89x).
- **Placement caveat (measured):** cfb8d13 has the same kernel code as e34bc6b (no diff in delta_matching/). It gave 0.950x
  here (GPUs 0-1) vs 0.935x for e34bc6b on GPUs 2-3 an hour earlier. The DM kernel's times are identical (+-0.3% per shape),
  but **FA4's times differ by up to 7% between these GPU pairs** (8k H16/16 fwd/bwd). FA4 bf16 is more power-bound, so it
  is more sensitive to each GPU's power/thermal envelope. The same-lease ratio is the right commit-to-commit comparison;
  the absolute headline ratio carries about +-1.5% (geomean) from GPU placement, up to +-7% on a single 8k shape.

arm=dm_matched  files=results/focus_cfb8d13_3fec800_3fec800.s*.jsonl  host=nvl72052-T18

vs fa4_bf16  (speedup = fa4_bf16_ms / dm_matched_ms)

| causal | pass | H/Hk | 8k | 32k | 64k | geomean (6 shapes) |
|---|---|---|---|---|---|---|
| 1 | fwd_bwd | 16/16 | 0.880 | 0.984 | 1.016 | **0.962** |
| 1 | fwd_bwd | 32/8 | 0.891 | 1.004 | 1.007 |  |
| 1 | fwd | 16/16 | 0.984 | 1.082 | 1.247 | 1.130 |
| 1 | fwd | 32/8 | 1.036 | 1.179 | 1.286 |  |
| 1 | bwd | 16/16 | 0.846 | 0.913 | 0.964 | 0.910 |
| 1 | bwd | 32/8 | 0.841 | 0.954 | 0.949 |  |
| 0 | fwd_bwd | 16/16 | 0.912 | 1.018 | 1.011 | 0.989 |
| 0 | fwd_bwd | 32/8 | 0.965 | 1.036 | 1.000 |  |
| 0 | fwd | 16/16 | 0.969 | 1.095 | 1.200 | 1.114 |
| 0 | fwd | 32/8 | 1.002 | 1.199 | 1.249 |  |
| 0 | bwd | 16/16 | 0.852 | 0.949 | 0.961 | 0.927 |
| 0 | bwd | 32/8 | 0.900 | 0.983 | 0.926 |  |

vs fa4_bf16_det  (speedup = fa4_bf16_det_ms / dm_matched_ms)

| causal | pass | H/Hk | 8k | 32k | 64k | geomean (6 shapes) |
|---|---|---|---|---|---|---|
| 1 | fwd_bwd | 16/16 | 0.904 | 0.997 | 1.047 | **0.995** |
| 1 | fwd_bwd | 32/8 | 0.945 | 1.041 | 1.045 |  |
| 1 | fwd | 16/16 | 0.982 | 1.086 | 1.229 | 1.127 |
| 1 | fwd | 32/8 | 1.038 | 1.179 | 1.278 |  |
| 1 | bwd | 16/16 | 0.877 | 0.969 | 1.008 | 0.963 |
| 1 | bwd | 32/8 | 0.921 | 1.004 | 1.009 |  |
| 0 | fwd_bwd | 16/16 | 0.945 | 1.053 | 1.055 | 1.023 |
| 0 | fwd_bwd | 32/8 | 0.998 | 1.052 | 1.037 |  |
| 0 | fwd | 16/16 | 0.971 | 1.065 | 1.220 | 1.106 |
| 0 | fwd | 32/8 | 0.998 | 1.173 | 1.238 |  |
| 0 | bwd | 16/16 | 0.920 | 1.005 | 1.025 | 0.977 |
| 0 | bwd | 32/8 | 0.942 | 0.991 | 0.985 |  |

B vs A:
| arm | causal | pass | cfb8d13 | 3fec800 | change | 3fec800 vs FA4-det | per-shape speedup of cand (min..max) | n |
|---|---|---|---|---|---|---|---|---|
| dm_matched | 0 | bwd | 0.908 | 0.927 | +2.1% | 0.977 | 1.018..1.027 | 6 |
| dm_matched | 0 | fwd | 1.096 | 1.114 | +1.6% | 1.106 | 0.993..1.035 | 6 |
| dm_matched | 0 | fwd_bwd | 0.971 | 0.989 | +1.9% | 1.023 | 1.013..1.029 | 6 |
| dm_matched | 1 | bwd | 0.888 | 0.910 | +2.4% | 0.963 | 1.017..1.023 | 6 |
| dm_matched | 1 | fwd | 1.130 | 1.130 | +0.0% | 1.127 | 0.998..1.009 | 6 |
| dm_matched | 1 | fwd_bwd | 0.950 | 0.962 | +1.3% | 0.995 | 1.005..1.019 | 6 |
| dm_stale | 0 | bwd | 1.221 | 1.221 | -0.0% | 1.286 | 0.997..1.001 | 6 |
| dm_stale | 0 | fwd | 1.095 | 1.112 | +1.6% | 1.104 | 0.998..1.034 | 6 |
| dm_stale | 0 | fwd_bwd | 1.216 | 1.216 | +0.0% | 1.257 | 0.995..1.009 | 6 |
| dm_stale | 1 | bwd | 1.172 | 1.178 | +0.5% | 1.247 | 0.999..1.005 | 6 |
| dm_stale | 1 | fwd | 1.128 | 1.131 | +0.2% | 1.128 | 1.000..1.016 | 6 |
| dm_stale | 1 | fwd_bwd | 1.180 | 1.179 | -0.1% | 1.219 | 0.995..1.000 | 6 |

### cc1dbf2 (kda-sm100-8k: gloop auto-threshold 2048 -> 4096) vs 4ea8bd5, same lease, nvl72131-T16 GPU 2, VALIDATED 02:05 (01:52)
- **H32/8 4k matched fwd+bwd +4.4% causal, +4.8% non-causal** (kernel time). Every other shape unchanged (0.992-1.004),
  as designed. Matches the expected -4.6% time.
- Headline geomean (8 causal shapes, single GPU): 0.906x -> **0.912x FA4 (+0.6%)**, 0.948x vs FA4-det. H32/8 4k: 0.772 -> 0.805.
- No SUSPECT flags.

| arm | causal | pass | 4ea8bd5 | cc1dbf2 | change | cc1dbf2 vs FA4-det | per-shape speedup of cand (min..max) | n |
|---|---|---|---|---|---|---|---|---|
| dm_matched | 0 | bwd | 0.892 | 0.897 | +0.6% | 0.944 | 0.990..1.061 | 8 |
| dm_matched | 0 | fwd | 1.070 | 1.069 | -0.0% | 1.065 | 1.000..1.005 | 8 |
| dm_matched | 0 | fwd_bwd | 0.948 | 0.953 | +0.5% | 0.982 | 0.992..1.048 | 8 |
| dm_matched | 1 | bwd | 0.863 | 0.868 | +0.6% | 0.918 | 0.997..1.056 | 8 |
| dm_matched | 1 | fwd | 1.042 | 1.042 | -0.0% | 1.040 | 0.999..1.005 | 8 |
| dm_matched | 1 | fwd_bwd | 0.906 | 0.912 | +0.6% | 0.948 | 0.996..1.044 | 8 |
| dm_stale | 0 | bwd | 1.162 | 1.175 | +1.1% | 1.236 | 0.996..1.081 | 8 |
| dm_stale | 0 | fwd | 1.067 | 1.067 | +0.0% | 1.063 | 1.000..1.010 | 8 |
| dm_stale | 0 | fwd_bwd | 1.160 | 1.168 | +0.7% | 1.203 | 0.996..1.060 | 8 |
| dm_stale | 1 | bwd | 1.102 | 1.110 | +0.7% | 1.175 | 0.996..1.073 | 8 |
| dm_stale | 1 | fwd | 1.039 | 1.039 | -0.1% | 1.036 | 0.997..1.003 | 8 |
| dm_stale | 1 | fwd_bwd | 1.096 | 1.105 | +0.9% | 1.149 | 0.996..1.056 | 8 |

### e646cb1 (cc1dbf2 gloop-to-4k + msplit) vs 4ea8bd5, same lease (chunked per N), nvl72131-T16 GPU 2, PENDING VALIDATION (02:30)
- matched causal fwd+bwd geomean over the 8 target shapes 0.908x -> **0.919x FA4 (+1.3%)**, 0.956x vs FA4-det. Kernel
  time is +0.3..+4.2% on every shape; the bwd gains +1.9% (msplit plus GQA 4k gloop) and the fwd is unchanged.
- Per N (H16/16 | H32/8): 4k 0.821 | 0.805, 8k 0.843 | 0.892, 32k 0.987 | 1.015, 64k 1.007 | 1.017. No SUSPECT flags.

| arm | causal | pass | 4ea8bd5 | e646cb1 | change | e646cb1 vs FA4-det | per-shape speedup of cand (min..max) | n |
|---|---|---|---|---|---|---|---|---|
| dm_matched | 1 | bwd | 0.861 | 0.878 | +1.9% | 0.927 | 1.006..1.054 | 8 |
| dm_matched | 1 | fwd | 1.044 | 1.042 | -0.2% | 1.040 | 0.983..1.002 | 8 |
| dm_matched | 1 | fwd_bwd | 0.908 | 0.919 | +1.3% | 0.956 | 1.003..1.042 | 8 |
| dm_stale | 1 | bwd | 1.100 | 1.129 | +2.6% | 1.192 | 1.008..1.072 | 8 |
| dm_stale | 1 | fwd | 1.040 | 1.041 | +0.0% | 1.039 | 0.999..1.004 | 8 |
| dm_stale | 1 | fwd_bwd | 1.101 | 1.118 | +1.6% | 1.162 | 1.005..1.054 | 8 |

### 49ff526 vs 77f470e, same lease (nvl72131-T16, 1 GPU), 04:20
| arm | causal | pass | 77f470e | 49ff526 | change | 49ff526 vs FA4-det | per-shape speedup of cand (min..max) | n |
|---|---|---|---|---|---|---|---|---|
| dm_matched | 1 | bwd | 0.888 | 0.875 | -1.5% | 0.930 | 0.983..0.991 | 6 |
| dm_matched | 1 | fwd | 0.998 | 1.000 | +0.3% | 0.994 | 1.000..1.000 | 6 |
| dm_matched | 1 | fwd_bwd | 0.920 | 0.913 | -0.9% | 0.950 | 0.992..0.996 | 6 |
| dm_stale | 1 | bwd | 1.194 | 1.188 | -0.4% | 1.264 | 0.997..1.001 | 6 |
| dm_stale | 1 | fwd | 0.998 | 1.001 | +0.3% | 0.994 | 1.000..1.000 | 6 |
| dm_stale | 1 | fwd_bwd | 1.150 | 1.148 | -0.2% | 1.195 | 1.000..1.002 | 6 |

### [FAILED validation 05:15: pass-0 crash at N=128 (IndexError, dm_pass0_sm100.py:1470, +pf/+sq path); not promotable] Pre-measured 29881d0 (P0-v8 bitwise pass 0 +pf+sq) vs headline 0c3f69d, same lease, new node nvl72013-T18 GPU 1 (04:58)
- matched causal fwd+bwd **+1.1%** (kernel time 1.004..1.011 on every shape; bwd +0.9..1.5%, fwd unchanged).
  Single-GPU ratio 0.913x -> 0.923x FA4 (0.957x vs det). kda expected ~+0.9%.
- Sanity errors bitwise identical to 0c3f69d. No SUSPECT flags.
- 0c3f69d re-anchored on the new node: 0.913x single GPU (4-GPU median on nvl72131-T16: 0.914x).

| arm | causal | pass | 0c3f69d | 29881d0 | change | 29881d0 vs FA4-det | per-shape speedup of cand (min..max) | n |
|---|---|---|---|---|---|---|---|---|
| dm_matched | 1 | bwd | 0.871 | 0.882 | +1.3% | 0.938 | 1.009..1.015 | 6 |
| dm_matched | 1 | fwd | 0.993 | 0.994 | +0.2% | 0.996 | 0.999..1.000 | 6 |
| dm_matched | 1 | fwd_bwd | 0.913 | 0.923 | +1.1% | 0.957 | 1.004..1.011 | 6 |
| dm_stale | 1 | bwd | 1.191 | 1.192 | +0.0% | 1.267 | 0.996..1.003 | 6 |
| dm_stale | 1 | fwd | 0.993 | 0.994 | +0.1% | 0.996 | 0.999..1.000 | 6 |
| dm_stale | 1 | fwd_bwd | 1.153 | 1.157 | +0.3% | 1.199 | 0.998..1.006 | 6 |

### Pre-measured, PENDING validation: a970641 (N=128 fix of P0-v8 + dqrd; supersedes 29881d0 / e1ac44f) vs headline 0c3f69d, same lease, nvl72013-T18 GPU 1 (05:22)
matched causal fwd+bwd **+1.9%** (kernel time 1.005..1.013 on every shape; bwd +0.8..1.5%, fwd unchanged); single-GPU ratio
0.915x -> 0.933x FA4 (0.963x vs FA4-det). Sanity bitwise identical to 0c3f69d. No SUSPECT flags.
