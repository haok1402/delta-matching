# Delta-Matching validator (sm100 and any CUDA GPU)

One command judges a candidate attention kernel against the pure-torch Delta-Matching reference
(`delta_matching/reference.py`) and exact attention, and prints PASS / FAIL with numbers:

```bash
python validator/validate.py /abs/path/cand.py[:attn] --shapes quick      # ~1 min;  --shapes full (~10 min), bench (FA4 bench shapes)
python validator/train_check.py /abs/path/cand.py[:attn] --steps 400      # slow tier: short LM training vs bf16 attention
```
The last stdout line before PASS/FAIL is `SUMMARY_JSON {...}` (one line, for candidates.jsonl). Exit code 0 = PASS.
On hsgpark run it through the pool: `gpupool run --gpus 1 --who <you> -- sm100/validate/validate.sh ...` (uses the
shared env sm100/env/venv). The pool kills a reservation idle for 90 s: a candidate that JIT-compiles on first call
(CuTe DSL, Triton, torch cpp_extension) should compile into a persistent cache in a `--gpus 0` run first, or pass
`--warmup 3m` to `gpupool run`. All cases run in one candidate process, so compilation happens once per validation.

## Candidate contract
A Python file (any language underneath) defining
- `attn(q, k, v, softmax_scale=None, causal=True, mode="matched") -> o` with `delta_matching.dm_attn_func`'s contract:
  q (B, N, H, D) bf16 CUDA, k / v (B, N, Hk, D), D = 128, H % Hk == 0, softmax scale D**-0.5; o (B, N, H, D) bf16;
  `o.backward(do)` fills q / k / v grads (bf16). Must accept strided views (q, k, v split from one packed qkv).
- `NUMERICS = "<preset>"` (default "sm90"): the quantization scheme, a key of `reference.PRESETS`
  (`sm90`, `mxfp8` = `mxfp8-all`, `mxfp8-p`; add a preset for a new scheme - the validator judges against it).
- optional `SUPPORTS = {"seqlen_multiple": 128, "noncausal": True, "matched_noncausal": False}`.
- `debug(q, k, v, do, causal, mode) -> {"delta": (B, H, N) fp32, "lse": (B, H, N) fp32 natural log}`: REQUIRED for a PASS in
  matched mode. `delta` must be the delta the backward actually consumes (in dP units) and `lse` the one it recomputes P
  from, for these exact inputs (the validator binds both to the gradients, so a hook that reports something else fails).

## What is checked, per case (shape x input family x causal)
Each case runs the candidate in a separate process (`runner.py`, which never imports the reference; a candidate that
imports it fails) on inputs drawn from a seed chosen at run time. Then, per output O, dQ, dK, dV and per mode:
| check | rule | why |
|---|---|---|
| shape / dtype / finite | bf16, right shape, no NaN / Inf | |
| accuracy | err(cand, exact) <= 1.25 x err(emulation, exact) + 2e-3 | the emulation of the declared numerics is what a correct fp8 kernel achieves on these very inputs; 25% covers accumulation-order code flips |
| blocks | worst (batch, head, 128-row block) error <= 1.5 x the emulation's worst + 5e-3 | catches skipped heads / tail tiles / wrong masks hidden by a global norm |
| faithful | dist(cand, emulation) <= 0.5 x err(emulation, exact) | the candidate implements the numerics it declares (honest kernels sit ~1e-3 away) |
| not_too_accurate | err(cand, exact) >= 0.5 x err(emulation, exact) | a kernel doing bf16 math while claiming fp8 speed fails here |
| fingerprint | dist(cand, emulation) < dist(cand, bf16 emulation) | same, independently of tolerances |
| debug_lse (matched) | candidate LSE vs the emulation: max abs diff <= 1e-3 + 1e-5 x |LSE| | the backward's P comes from it |
| rowsum_dS_direct (matched) | max over rows of |delta sum_j P - sum_j P dP| / sum_j |P dP| <= 1e-3, with P = exp(S - LSE_candidate) over the bitwise-reproducible quantized operands | THE Delta-Matching identity, measured directly: pass-0 style deltas sit at 1e-7..3e-5, a 1% delta error at 1e-2, stale at 1e-2..1. The dK-sum check below cannot see P mismatches of 1e-3 (dS quantization noise dominates it) |
| debug_binds_grads (matched) | candidate dQ / dK vs the emulated backward driven by the candidate's reported delta and LSE: <= 3e-3 + 0.05 x fp8 error | the hook cannot report a delta the gradients do not use (a 3% lie reads 0.02-0.08) |
| rowsum_dS_identity (matched) | sum_j dK_j metric <= 2 x the emulation's + 1e-4 | matched delta makes rowsum(dS) = 0, so sum over keys of dK = Q^T rowsum(dS) ~ 0; stale shows ~2-40x more |
| modes_share_O / dV | bitwise equal between matched and stale | the modes differ only in delta |
| modes_differ_dQ | matched != stale | a matched mode that silently runs stale fails |
| repeat / views bitwise | identical outputs for a repeat call and for strided-view inputs | determinism, stride handling |
| stress (finite / bitwise every call) | 4 (quick) or 8 (full, bench) repeated calls on large MHA shapes (H16 N4096/8192, causal and full): every call finite and bitwise equal to the first | races show up intermittently (B1-v0's 2-CTA backward produced NaN rows in 1-6 of 6 calls) |
| no_stale_cache | after overwriting the input buffers in place the result follows the new inputs | output caching keyed on pointers |
The fingerprint/faithful/not_too_accurate checks apply where fp8 noise dominates (emulation error > 8e-3).
Input families: randn, q/k scaled x3 / x10 (logit range), channel outliers x30 + huge tokens, huge / tiny magnitudes,
values on e4m3 code boundaries, peaked attention, uniform attention (q = 0), zero dO, constant V (dQ = dK = 0 exactly).
Shapes: MHA / GQA / MQA, causal / non-causal, non-power-of-two seqlens, up to 16k (full); `--shapes bench` validates
exactly the shapes `sm100/bench/attnbench.py` times, so no benchmarked shape is unvalidated.
All-masked rows cannot occur (self-attention, causal or full, no mask argument).

Self-test: `tests/test_validator.py` (the emulation passes; bf16 math, a skipped head, a garbage tail block, pointer
caching, non-determinism, mode-dependent forward and stale-as-matched all fail).

## The reference (delta_matching/reference.py)
`dm_reference(q, k, v, do, causal, mode, preset)` reproduces every quantization point of a kernel exactly (scales,
rounding, saturation, block shapes) and computes the matmuls in fp32 (no TF32). The `sm90` preset follows the shipped
Hopper kernel (sm100/validate/NUMERICS-hd128.md): Q / dO 128-row and K / V 64-row fp32 block scales (K centered by its
mean, Q pre-scaled by D**-0.5), forward P coded per (row, 64-key group) under a descending 256-key online max, backward
P as e4m3(2^8 P), dS per 64x128 tile on 2^8 ks_j dS, matched delta = sum_j P_ij dP_ij with the unquantized P (which is
why rowsum(dS) = delta (1 - sum P) ~ fp32 round-off). Not modelled: MMA accumulation order, ex2/rcp approximations, the
F2b 1e-8 dominance floor; they move a correct kernel ~1e-3 away from the emulation.

> **TOY SETUP.** Everything in the training sections below (train_check.py, longtrain.py) uses small byte-level models
> on a few-MB to 349 MB source-code corpus with an odd held-out split. These runs are smoke tests that the kernels run
> inside a training loop without breaking (NaNs, a grossly wrong gradient such as delta = 0). They do NOT support
> conclusions about real LLM training - in particular not about matched vs stale or fp8 vs bf16 quality - and must not
> be cited as evidence (user, 2026-11-09). The validation verdicts rest on the per-call tiers above.

## Training tier calibration (toy smoke test; GB200, 2026-11-08; `validator/train_check.py none --arms bf16,emul,emul_stale,emul_nodelta`)
Final-20% mean loss, same seed (same init and batches), sm90 emulation:
| steps | bf16 | fp8 matched (emul) | fp8 stale | delta = 0 (broken control) | bf16, other seed |
|---|---|---|---|---|---|
| 400 | 1.2871 | 1.2893 | 1.2888 | (4.0 avg over steps 0-200 of the 2000 run) | |
| 2000 | 0.6038 | 0.6004 | 0.5971 | 8.4513 (diverges) | 0.6511 |
Same-seed fp8 arms sit within 0.007 nats of bf16; a broken backward is off by > 2 nats within 200 steps. The budget
max(2%, 0.03 nats) of the same-seed bf16 loss therefore separates them with wide margins (seed-to-seed noise, 0.05, is
why every arm uses the same seed). matched vs stale is NOT separable at this scale: stale's drift shows over long
training runs, so this tier judges breakage only (a smoke test), not the matched-vs-stale benefit.

### Stale vs matched in toy training (sm100-mx, GB200, 2026-11-08) — observation only, see the TOY SETUP note
Kernel 6a31419 / 5a912b2: matched healthy on seeds 0-3 (0.59-0.64, at or below bf16); stale diverges on every seed
(3.1-3.3, onset steps 400-600) and under every forward variant (MUFU exps, eager rescale). The emulation's stale is
healthy (seeds 0-3) only because the emulation is exactly self-consistent (forward and backward use bitwise the same fp32
S and exact exp). Perturbing the emulation's FORWARD S by 1e-4 relative (the size of an MMA accumulation-order
difference between a forward and a backward kernel) makes emulated stale diverge too (3.95; 1e-3: 3.26), while emulated
matched stays at bf16 level (0.598 / 0.599). In this toy setup, stale's divergence therefore did not point to a kernel bug (kernel and emulation agree per call); no
claim is made about real training. Data: sm100/validate/runs/{seeds,v5a9,noise}.
Random backward-P error (P_bwd *= 1 + U(-a, a), emulation, 2000 steps): matched trains at bf16 level for a up to 2e-3
under both sm100-mx and sm100-mx-x (validate/runs/bwdp/). Under sm100-mx-x such noise reads 1e-3..4e-2 on the direct
identity check, i.e. that check is stricter than training needs for random (non-systematic) error; kept strict by design.

### Toy training at N = 8192 (batch 2, 2000 steps, kernel 3fec800, GB200, 2026-11-09) - observation only
| seed | bf16 | matched | stale |
|---|---|---|---|
| 0 | 0.6710 | 0.6810 (+0.010) | 3.120 (diverged from ~step 400) |
| 1 | 0.8211 | 0.8281 (+0.007) | 3.199 (diverged from ~step 200) |
Smoke-test reading only: matched ran without breakage at 8k. (runs/train8k/)

### Long training (toy; validate/longtrain/REPORT.md)
Stopped by the user at ~12k steps. Labelled a toy setup there as well; not evidence about real training.
