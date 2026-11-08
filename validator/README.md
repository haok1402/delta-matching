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
- optional `debug(q, k, v, do, causal, mode) -> dict` with any of `delta` (B, H, N), `lse` (B, H, N), `rowsum_dS`.

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
| rowsum_dS_identity (matched) | sum_j dK_j metric <= 2 x the emulation's + 1e-4 | matched delta makes rowsum(dS) = 0, so sum over keys of dK = Q^T rowsum(dS) ~ 0; stale shows ~2-40x more |
| modes_share_O / dV | bitwise equal between matched and stale | the modes differ only in delta |
| modes_differ_dQ | matched != stale | a matched mode that silently runs stale fails |
| repeat / views bitwise | identical outputs for a repeat call and for strided-view inputs | determinism, stride handling |
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

## Training tier calibration (GB200, 2026-11-08; `validator/train_check.py none --arms bf16,emul,emul_stale,emul_nodelta`)
Final-20% mean loss, same seed (same init and batches), sm90 emulation:
| steps | bf16 | fp8 matched (emul) | fp8 stale | delta = 0 (broken control) | bf16, other seed |
|---|---|---|---|---|---|
| 400 | 1.2871 | 1.2893 | 1.2888 | (4.0 avg over steps 0-200 of the 2000 run) | |
| 2000 | 0.6038 | 0.6004 | 0.5971 | 8.4513 (diverges) | 0.6511 |
Same-seed fp8 arms sit within 0.007 nats of bf16; a broken backward is off by > 2 nats within 200 steps. The budget
max(2%, 0.03 nats) of the same-seed bf16 loss therefore separates them with wide margins (seed-to-seed noise, 0.05, is
why every arm uses the same seed). matched vs stale is NOT separable at this scale: stale's drift shows over long
training runs, so this tier judges breakage, not the matched-vs-stale benefit (reported, not judged).
