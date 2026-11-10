# hd128 accuracy-floor investigation: bit-identical inputs for the real sm90 kernel (accuracy-audit, 2026-11-10)

Question: is the sm100 kernel (kda-sm100 @ 7d63434, per-32-token pow2 scales, per-(row,32-key) pow2 P exponent)
less accurate than the paper's sm90 kernel ([128,64] fp32 scales, per-(row,64-key) P basis) on hd128?
Emulations say yes (O +5..+45% depending on family). Part 2 = the REAL sm90 kernel on H100 on the SAME bf16 inputs.

## Cases (18)
causal, D=128, shapes (B,N,H,Hk) = (1,2048,8,2) and (1,4096,16,4), families
randn outliers nearsat scaled3 scaled10 peaked bigmag audit_ramp audit_growmax (definitions in gen_inputs.py;
make_inputs is verbatim dm-validator validate.py). One CPU torch.Generator seeded SEED=13 per case.

## Steps for sm90-ref
1. `python gen_inputs.py OUTDIR` (CPU only; ~450 MB of .pt). It must print `sha256 vs manifest: ALL MATCH`.
   Inputs are bf16 q (B,N,H,D), k/v (B,N,Hk,D), do (B,N,H,D) in `in_N<N>_H<H>_<Hk>_<family>.pt` (dict q,k,v,do,causal).
   If it reports MISMATCH (different torch randn on CPU), stop and tell accuracy-audit (we then ship the tensors another way).
2. For each file: run the sm90 kernel from kda-sm90-clean, **deterministic mode, matched delta**, causal=True,
   softmax scale D**-0.5, forward + backward with grad_output = do. Collect O, dQ, dK, dV and the forward LSE
   (natural log, (B,H,N); if the kernel stores log2, convert: lse_nat = lse2 * ln 2).
3. Metrics: `metrics.exact(q,k,v,do)` (fp64, on GPU) and `metrics.errors(dict(O=..,dQ=..,dK=..,dV=..,LSE=..), ref)`:
   relative L2 vs exact for O/dQ/dK/dV; LSE_maxabs = min over raw and K-centred exact LSE (both kernels centre K).
4. Write `results_sm90.json` = {case_name: {"kernel_sm90_<sha>": errors}} (same layout as results_sm100.json) and push it
   to this branch (or report the path). Please record the kda-sm90-clean commit, GPU, and the env (torch/CUDA).

## Already here
- results_sm100.json: real sm100 kernel (7d63434, det) and the sm100-mx / sm90 emulations (dm-validator reference.py),
  same inputs, same metrics. Kernel = its emulation to ~4 digits.
- manifest.json: sha256 of every tensor (torch 2.14.1+cu130 CPU generator).
