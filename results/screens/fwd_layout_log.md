# S20-fwd: D128 causal forward, layout space (round 20)

Base: HEAD e15af8e (sources = c8c2f48 = accepted r19_fwd). A = `.local/frozen/r19_fwd` in every screen
(`screen.sh`, 2 reps, B2 N8192 H16/4 and H32/4). Trees are in `.local/wt/S20-fwd-NN`, and patches (against HEAD) are in
`.local/screens/S20-fwd-NN/variant.patch`. The bitwise check is the default route against `.local/equiv4/r17pk_default.pt`.
The REG/STACK column is the causal D128 forward symbol at launch (r19_fwd: REG 128 / STACK 24, non-causal BM192 56).

## Fixed points (from the numerics, the register file and the ledger)

- **The P-hat group is 64 keys, fixed.** The group is the K/V quantization block (one k8s / v8s per 64 keys,
  `kQuantKeys`), and its codes use v8s_g. Any other group size changes the quantizer contract. Merging groups is
  FWDSHB / BASIS 2, which fail residual-v4 by 100-400x.
  - BN64: one group per tile, so there is no rho inside a tile. The promote (64 FFMA per 64 keys) replaces it, so per 128
    keys it is 128 vector ops, the same as rho 64 + promote 64 at BN128. The per-tile bookkeeping doubles, and the
    ledger measured this (S47 b3n64: about 9 points slower than BN128).
  - BN96: not group-aligned (1.5 groups per tile), so not expressible without changing the quantization.
  - BN192 / BN256 at 3 WGs: S alone is 96 / 128 registers, which with O 64 exceeds 160. BN256 at 2 WGs was measured
    at +8.3% (FWDSHB2) and +9.97% (S2).
- **The register file:** 3 consumer WGs x 160 + a 32-register producer = 128 x 512 = 65536, already the whole file.
  - The consumers cannot reach 168 with a producer warpgroup (24 + 3 x 168 = 528 > 512).
  - 4 consumer WGs at 120 registers: O 64 + the tile temp D 64 = 128 > 120 before S or any scalar. Infeasible
    (D is the contract's zero-init per-tile temp).
- **The quad max shuffles (integrator note: the 8 SHFL + 8 FMNMX quad all-reduce is worth -1.93%):** no layout reduces
  them.
  - A row is spread over the 4 lanes of a quad in every wgmma m64nN accumulator, so each row and 64-key group needs
    log2(4) = 2 shuffle rounds, for any BN.
  - BN only changes groups per tile, not shuffles per key.
  - Transposed products (S^T) would move the row reduction across warps, which is worse.
  - More independent WG chains is the only layout way to hide the shuffle latency, and the register file forbids it
    (S20-fwd-04, regtest, 05, 06 below).

## Variants

| tag | layout | regs / stack (causal D128) | bitwise / v4 | fwd H16 | fwd H32 | verdict |
|---|---|---|---|---:|---:|---|
| S20-fwd-01 | K/V stages 3 -> 2 (FWDST=2) | 128/24 (unchanged) | BITWISE EQUAL, 72 cases | +0.97% | +0.36% | reject |
| S20-fwd-02 | K/V stages 3 -> 4 (FWDST=4) | 128/24 | BITWISE EQUAL, 72 cases | +0.14% | +0.60% | reject (as S51) |
| S20-fwd-03 | NOVT ablation: the producer's in-smem V transpose skipped (WRONG numerics, timing only). Sizes "V^T layout from the quantizer / producer warp" | 128/24 | n/a | -2.47% | -2.26% | upper bound < 3.5%; not pursued |
| S20-fwd-04 | register split 56 / 152 (producer / consumer), sensitivity probe | 128/96 (non-causal 56) | BITWISE EQUAL | +5.66% | +5.76% | 8 registers less = +5.7%: the 160 budget binds |
| regtest | 416-thread CTA (3 consumer WGs + 1 producer WARP), setmaxnreg.dec 24 on the warp, .inc 160 / 168 on the consumers (`.local/rounds/r20-fwd/regtest/`, job 21198777) | ptxas launch 128 | n/a | - | - | **hangs at 160 and at 168** (timeout): registers released by setmaxnreg.dec only return to the CTA's own pool. A producer warp gives consumers at most (416 x 128 - 32 x 24) / 384 = 136 registers. The FA3-style producer warp frees nothing here |
| S20-fwd-05 | 2 CTAs / SM: BM64, 1 consumer WG at 232 registers + producer WG at 24, 2 stages (113 KB smem / CTA), grid 2 x SMs | 128/32 | rc 1: 1 case differs (d128_mqa causal stale dK 0.06579 -> 0.06580; m-block coverage changes the stale-mode path) | +23.70% | +22.57% | reject: 2 unsynchronized chains / SM < 3 ring-ordered chains |
| S20-fwd-06 | BM128, 2 consumer WGs at 240 registers (BM192=0), PV-first on the r19 stack | 168/24-32 | rc 1 (same single stale dK case) | +18.33% | +17.44% | reject: the third chain is worth ~18% (consistent with S35/C3) |

Timing for 05 and 06 is far outside the bar, so no DBG / residual-v4 capture was run for them.

## Not built (analysis)

- **Softmax / PV warp specialization (FA4-like).** At BM128: 2 softmax WGs (S 64 + stats, ~96 registers) and 2 PV WGs
  (O 64 + D 64 + scalars, ~144 registers), with P handed through smem (SS wgmma) and rho / w / rebase per row through
  smem. It would be bitwise in principle.
  - Budget: 24 + 2 x 96 + 2 x 144 = 504, i.e. zero slack on the PV side.
  - Bound: kda-solcheck's ideal schedules for the exact mix give spec 69.9% vs ping-pong 66.8% of sol4. That is about
    +4.6% at the ideal. It must also pay P STS + fence + barriers per tile, and it drops from 3 to 2 softmax chains.
    The expected real gain is 0 to ~4%, at the cost of a new consumer mainloop (~400+ lines).
  - Not screened in this round; it is the only remaining structural option with nonzero expected value.
- **Pre-transposed V (quantizer writes V^T):** bounded by S20-fwd-03 at -2.3 .. -2.5% even if free. It needs a P register
  permutation (S30 VCOL measured +3.6%) or a key-permuted V^T layout (not bitwise: changes the wgmma k order).
- **Rejected by the ledger, not repeated:** BN64 (S47), BN256 (S2 / FWDSHB2), stages (S51), OSMEM (S45), ring / token
  variants (TOK2*, TT2/3, QKRING, ARVEARLY).

## Verdict

No layout candidate. Every layout either:

- removes a chain (2 WGs: +18%; 2 CTAs / SM: +23%),
- loses registers (a producer warp caps the consumers at 136; 152 registers alone costs +5.7%),
- breaks the 64-key group, or
- is bounded below the 3.5% bar (V transpose: <= 2.5%; warp specialization: <= ~4.6% ideal, unbuilt).

The 3-WG x 160-register BM192 x BN128 layout is a local optimum of the layout space under the numerics contract.
