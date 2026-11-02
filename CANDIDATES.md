# kda-fwdport: the forward rebuilt from FA3's fp8 forward

Helper kda-fwdport. The route: start from FA3's fp8 forward (hopper/, e4m3 path) and add one at a time what
delta-matching needs, measuring the cost of each addition and trying several ways to express it. Branches:
`kda-sm90-fwdport` (experiment scaffolding, this file) and `kda-sm90-fwdport-smsc` (the candidate).
Shapes: D128 causal matched, B2 N8192 and B1 N65536 at H/Hk 16/4 and 32/4. Times are the forward kernel (CUPTI),
geometric mean over the four shapes; screens are 2 alternating reps (noise about +-0.5%), candidates 3 reps.

## What delta-matching needs from the forward (delta_matching/core.py, attn.py, fwd_fp8)

- Outputs: O (bf16) and the natural-log LSE (b, h, s). The backward reads only LSE (and O in stale mode); it recomputes P
  itself, so nothing about the forward's P basis is saved or read by the backward.
- Numerics: Q/K/V e4m3 codes with per-block scales (Q per 128 rows, K/V per 64 keys) -> per-element S descale
  q8sc*k8sc_g; the faithful per-row, per-64-key P basis (codes e4m3(448 exp(s - nu_g)), each group max exactly 448);
  the group products weighted per row and group (w_g = v8sc_g 2^(ms_g - m l2)) in fp32 with a zero-init per-tile temp
  (accumulating fp8 wgmma products straight into the running O re-rounds it per 32-key atom: +1.4e-2 O error, fails).
- The LSE comes from unquantized p, so the P basis touches O only; any change that moves LSE moves dQ/dK/dV too.

## Scaffolding (commit chain on kda-sm90-fwdport, `DM_FWDPORT=<kVar>` selects a variant at run time)

`csrc/mainloop_fwd_sm90_fp8port.hpp` + `softmax_port.h`: FA3's consumer step in two orders with independent knobs,
kVar = BM*1e7 + SP*1e6 + LAZY*1e5 + SCHED*1e4 + BN*1e3 + DESC*100 + BASIS*10 + PV:
SCHED 0 FA3 order (QK(n) + PV(n-1) issued together, softmax(n) under PV(n-1)) / 1 PV-first (shipped);
BN 0:256 1:224 2:192 3:128; DESC 0 none / 1 FMUL per element (shipped) / 2 DFOLD (exp FFMA constant per group);
BASIS 0 FA3 constant 2^8 basis / 1 per-group; PV 0 one gemm into O (FA3) / 1 exact Horner temp (shipped) / 2 unweighted
temp + promote / 3 two-temp group fold; LAZY deferred alpha; SP k-scale loads 0 at use / 1 step start / 2 none (ablation)
/ 3 smem-staged with the K TMA; BM 0:128 (2 WGs) 1:192 (3 WGs). Validation: 10111 (PV-first, FMUL, basis, Horner)
reproduces the accepted forward byte for byte (O on the four shapes; 72 cases x 4 outputs of the default route).

## Cost ledger (geo-mean vs FA3 fp8, same job; jobs 21177333 (t1), 21177789 (t2), 21178059 (t4), 21179674 (t6b))

| step | variant | vs FA3 |
|---|---|---:|
| FA3 replica in our harness (BN224, FA3 order, no scales) | 1000 | -3.0% |
| FA3 math at BN256, FA3 order | 0 | +10.9% (wgmma serialized, STACK 96: S 128 + O 64 + P 32 > 240) |
| FA3 math at BN192, FA3 order | 2000 | -1.2% |
| FA3 math at BN256, PV-first | 10000 | +0.4% (order and tile cost nothing) |
| + S descale, FMUL per element | 10100 | +18.3% |
| + S descale, DFOLD | 10200 | +7.9% |
| + DFOLD, no scale loads (ablation) | 2010200 | +0.9% |
| + DFOLD, scales smem-staged | 3010200 | +5.8% |
| + per-group basis (FMUL / DFOLD / DFOLD+smem) | 10110 / 10210 / 3010210 | +24.3% / +16.4% / +13.8% |
| + exact weights, Horner (FMUL = accepted / DFOLD) | 10111 / 10211 | +32.5% / +29.0% |
| + exact weights, Horner, smem-staged scales | 3010111 / 3010211 | +27.8% / +27.9% |
| + exact weights, two-temp fold (DFOLD / +smem) | 10213 / 3010213 | +24.3% / +23.2% |
| accepted r17_p1 forward (A) | - | +32.1% |
| A + smem-staged scales (shipped code, bitwise) | b6 default | +26.9% |

Reading: on FA3's own math the PV-first order and the 256-key tile are free; the per-tile global k-scale loads are the
single largest addition (load latency on the softmax's critical path, about 7% of FA3); the per-group basis costs
6-8% (group all-reduces and the row-max floor join are latency, not issue: +80 instructions per step); the exact group
weighting costs 8-14%. Expressions tried per lever:
- descale: FMUL (shipped), DFOLD into the exp FFMA, loads at use / at step start in registers / smem-staged / none.
  Smem staging is the only bitwise one that pays.
- group weights: Horner in one temp (shipped), unweighted temp + promote (cost probe), two alternating zero-init temps
  folded O += w_g T_g (faster with DFOLD, -4.7%, but 4 fp32 roundings onto O per tile move O by up to 9e-6 and stale
  dQ/dK by up to 2.5e-4: fails residual-v4), FA3 order with a temp at BN 256 / 192 / 128 (spills: +36 .. +690%).
- alpha deferred (lazy reference basis, rebase in the PV phase): no gain in the 2-WG layout (+3..8%); the vote branch
  spilled where S was live.
- P quantization / V scale folds: a non-dyadic change of basis re-randomizes P rounding and loses the exact group
  maximum (systematically worse O); only dyadic shifts are exact, and they cannot absorb v8sc_g or the fractional part
  of ms_g, so the weights stay on the output side.

## Candidates

### SMSC: K/V scales staged into smem with the K TMA (bitwise)

- Commit 80b8151 on `kda-sm90-fwdport-smsc` (on kda-fwd's C4 b7d70a0 = r17_fwd C3g + PRETURN). D128 PV-first only;
  d256 untouched. The producer issues two 16 B `cp.async.bulk` (the aligned window k8s / v8s[4*(n/2), +4)) on the K stage
  barrier (K expect-tx +32 B); consumers read the tile's pair from smem right after the K wait. The V(t) pipeline's byte
  count is set from TmaTransactionBytesV explicitly (it was inherited from K's params: raising K's count hung V).
- Bitwise: equivalence.py BITWISE EQUAL vs C4, FULL SUITE, 72 cases on default and kv (job 21179759); O sha equal on the
  four shapes in every timing job.
- Resources: REG 128 / STACK 16 on the causal D128 symbol (unchanged); the non-causal D128 symbol STACK 16 -> 40
  (not timed; warning class); no spill / C7510 / serialization diagnostics.
- A/B vs C4 (forward geo-mean, 3 reps): job 21179673 -1.41% (per shape -1.5 / -1.6 / -1.4 / -1.2%), job 21179761
  -1.19% (-1.5 / -1.7 / -0.7 / -0.9%). kda-fwd's acceptance A/B of C4 + SMSC vs r17_fwd: -2.46% (21179751), -2.42%
  (21179752), bitwise.
- On the old base (r17_p1, 2 WGs) the same change is -3.9% (job 21179674): C3g's third warpgroup already hides most of the
  load latency.
- Status: READY for stacking (bitwise, >= 1% in two jobs); C4 + SMSC alone is 2.4%, short of 3%.

## The same ledger in the accepted layout (BM192, three consumer warpgroups, BN128, 3 stages; jobs 21179940, 21180532)

| step | variant | vs FA3 |
|---|---|---:|
| FA3 math, 3 WGs, PV-first | 10013000 | **-7.0%** (-11% at 64K) |
| FA3 math, 3 WGs, FA3 order | 10003000 | +19.0% (S + O + P over 160 regs: serialized) |
| + DFOLD descale | 10013200 | -1.1% |
| + per-group basis | 10013210 | +3.0% |
| + zero-init temp + promote, unweighted (wrong numerics, cost probe) | 10013212 | +12.7% |
| + exact Horner weights, port (spills at 160 regs, STACK 64) | 10013211 | +31.0% |
| accepted forward C4 (C3g + PRETURN) | c4 | +14.8% |
| C4 + SMSC | c4s | +13.4% |
| V-scale folded into the codes (BASIS 2: one tile basis per row, codes scaled by v8sc_g/vmax, one PV chain, no rho) | 10013222 | +13.7% |

BASIS 2 at 2 WGs (3010222) is -9.4% vs the exact Horner (3010211), but it fails residual-v4 badly (job 21180533): O error
up on all 22 causal D128 cases by +2e-3 .. +5.5e-3 (5-12% relative), stale dQ / dK up to +1.4e-2. It loses the exact
448-mapping of every group maximum but one, as expected. The P-quantization / V-scale fold is therefore measured,
not only argued, to be out.

## Final summary (kda-fwdport)

- Ledger: on FA3's own math, our PV-first order and 256-key tile are free (+0.4%), and the 3-WG BM192 layout is 7% faster
  than FA3 itself. The delta-matching additions cost, in that layout: S descale about 6% (mostly scale-load latency),
  per-group basis about 4%, and the zero-init per-tile temp + promote (the fp8-accumulate precision contract) about 10%.
  The exact rho weighting on top is about 2% in C4.
- Delivered: SMSC (k/v scales staged into smem with the K TMA), bitwise on default and kv (72 cases each),
  -1.41% / -1.19% vs C4 in two jobs; with C4, -2.46% / -2.42% vs r17_fwd (kda-fwd's A/B). READY for stacking.
- Not reachable under residual-v4 with the levers tried: "within 5% of FA3" (C4 + SMSC is 1.134x FA3). Every form that
  drops the exact per-group weights or the zero-init tile temp fails numerics:
  - BASIS 2: +2e-3 .. 5e-3 on O.
  - Direct accumulation into O: +1.4e-2 on O (historical measurement in the mainloop comments).
  - Two-temp fold: within 1e-5 on O, but fails on stale dQ / dK.
  Every exact form that keeps FA3's issue order spills at 240 / 160 registers.
- What remains grounded: the temp + promote costs about 10% in the 3-WG layout. Taking the running O out of registers
  (kda-fwd's O-in-smem plan) is the one structural lever left that could let FA3's QK(n) + PV(n-1) issue order hold the temp.
  The remaining descale cost after SMSC (2010200 vs 3010200: 5 points on plain FA3 math) is the per-tile scale
  arithmetic in the max / exp-constant chain.
