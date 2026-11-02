# S19-fwd: D128 causal forward vs FA3 fp8 forward, per-step SASS diff and instruction attacks

Base: HEAD 24275e3 (accepted r18_c4). The held bitwise stack C4 PRETURN + SMSC (origin/kda-sm90-fwdport-smsc 80b8151,
`git diff c1ceece 80b8151 -- csrc`) applies cleanly to HEAD, because the forward files are unchanged since c1ceece. S19-fwd-00 is
that stack. The variants are built on S19-fwd-00, and their A/B uses A = S19-fwd-00 (`ACCEPTED=` in screen.sh) unless marked
otherwise. Tools and listings are in `.local/rounds/r19-fwd/` (loop extracts `*_unmasked.txt`, `*_path.txt`, `*_turn.txt`;
`classes.py`, `opc.py`; A/B runs in `ab/`). The variant generator is `.local/screens/S19-fwd-scripts/edit.py`. Patches are
`.local/screens/S19-fwd-NN/variant.patch` (against HEAD, stack included).

## 1. Kernels and step loops

| | ours (r18_c4 / stack S19-fwd-00) | FA3 fp8 (flash_attn_3/_C.abi3.so, cubin flash_fwd_hdim128_e4m3_sm90) |
|---|---|---|
| symbol | FlashAttnFwdSm90Fq8F ... ILi192EEENS7_ILi128EEE ... causal, DynamicPersistent | FlashAttnFwdSm90 ... C<128>,C<224>,C<128> ... causal (Lb1) |
| tile / WGs | 192 x 128, 3 consumer WGs, 64 rows x 128 keys per WG step | 128 x 224, 2 consumer WGs, 64 x 224 per WG step |
| unmasked step loop (cheapest path, sol4) | r18_c4 0x7d50-0xa770: 599 instr, 75 MUFU; stack: 591 | 0x8b10-0xb530: 661 instr, 114 MUFU |
| per 128-key step | 599 / 591 | 377.7 |
| per score per thread | 9.36 / 9.23 | 5.90 (ratio 1.57x) |
| masked (diagonal) loop | 787 (stack): + 64 descale FMUL, 64 ISETP + 72 FSEL mask | 903 / 224 keys |

## 2. Opcode classes per warp per step (cheapest path), normalized to 128 keys

| class | ours (stack, 128 keys) | FA3 (224 keys) | FA3 per 128 keys | ours - FA3 |
|---|---:|---:|---:|---:|
| FFMA | 140 | 116 | 66.3 | +73.7 |
| FMUL | 85 | 66 | 37.7 | +47.3 |
| FMNMX | 82 | 116 | 66.3 | +15.7 |
| MUFU | 75 | 114 | 65.1 | +9.9 |
| FADD | 72 | 112 | 64.0 | +8.0 |
| F2FP | 32 | 56 | 32.0 | 0 |
| uniform (U*, R2UR, VOTEU) | 42 | 39 | 22.3 | +19.7 |
| int / move / S2R | 21 | 13 | 7.4 | +13.6 |
| QGMMA | 8 | 11 | 6.3 | +1.7 |
| SHFL | 8 | 4 | 2.3 | +5.7 |
| WARPGROUP arrive/depbar | 6 | 4 | 2.3 | +3.7 |
| BAR / SYNCS | 6 | 4 | 2.3 | +3.7 |
| FSETP / FSEL | 6 | 0 | 0 | +6.0 |
| control / pred (BRA, VOTE, PLOP3) | 6 | 6 | 3.4 | +2.6 |
| LDS (scales) | 2 | 0 | 0 | +2.0 |
| total | 591 | 661 | 377.7 | +213.3 |

## 3. Attribution of ours (stack, 591 per step), by source

| source (mainloop_fwd_sm90_fp8f.hpp / softmax_fq8f.h) | instr | where |
|---|---:|---|
| FA3 core: raw group max FMNMX 60, code FFMA 64 + MUFU 64, F2FP 32, row-sum FADD 60 | 280 | softmax |
| rho rescale of the tile temp (rescale_o(tOrD, rho1)) FMUL | 64 | ring turn |
| promote O += w D (promote_o) FFMA | 64 | ring turn |
| rho_1 scalars (fq8_rho: RCP + FMUL v ratio, FADD, MUFU, FMUL, 2x2 clamps) | 12 | pre-turn (ptxas sinks the 4 clamps into the turn) |
| promote weight (basis_ref: FADD, MUFU, FMUL) | 6 | pre-turn / turn |
| lazy rebase check (max_l2<true>: FMUL, FSETP, FSEL; FADD, FSETP, PLOP3, VOTE, BRA) | 13 | pre-turn |
| group-max finish, 2 groups x 2 rows (SHFL 8, FMNMX 8); FA3 needs ~4.6 per 128 keys | 16 | softmax |
| descale fold / basis scalars (q*k, FLT_MIN floor, ms*=g, tile/row max, floor FFMA, ms*l2, fmax, -log2 448, g*l2) | 28 | softmax |
| scores_scale for row_sum (FADD, FMUL, MUFU) | 6 | softmax |
| row-sum fold onto the running max (FFMA 4, MUFU 4, FFMA 4, FFMA 2) | 14 | softmax |
| scale reads (SMSC LDS 2, R2UR 2) | 4 | ring turn (after the K wait) |
| QGMMA 8, WARPGROUP 6, BAR/SYNCS 6 | 20 | turn |
| integer / uniform / control: wgmma descriptors (8 atoms), pipeline state, barrier-id remat (S2R x2 + SHF/IADD3/ISETP/SEL = 8), loop-carried MOVs 8 | 64 | spread |

Gap to FA3 per 128 keys (+213): the per-64-key P basis (rho + promote) is +128, minus FA3's per-tile alpha rescale (-37, which
lazy removes); per-tile scalar bookkeeping that FA3 does not have or amortizes over 224 keys is about +88 (rho / weight / rebase /
2-group reductions / descale-basis / fold / scale reads); integer / uniform / control is about +36; the gemm/barrier
instructions of the shorter tile are about +9. The kda-fwd estimate of "~120 bookkeeping" is about right (~119 float/sync
bookkeeping plus ~64 int/uniform here).

Ring-turn view (coordinator note: the forward is ring-turn-bound; softmax has slack): the stack's turn (BAR.SYNC .. BAR.ARV,
`s00_turn.txt`) is 191 instructions: 64 FMUL rho + 64 FFMA promote + 8 QGMMA + 55 other (descriptor uniforms 21, the
SMSC LDS/R2UR + q*k of the next tile, the BAR.ARV id remat S2R/SHF/ISETP/SEL, 4 rho clamps, weight FMUL, V release / K wait).
Only the 55 "other" can be cut bitwise. The rho and promote form the method's floor (every reformulation is rejected in
ledger S1-S54 / FWDSHB).

## 4. Variants (all bitwise unless stated; fwd % = B vs A, 2-rep screen unless "4-rep")

| tag | change | default-route bitwise vs r17_pk byte ref | step path / turn | fwd H16 | fwd H32 | REG/STACK (causal D128) | verdict |
|---|---|---|---|---:|---:|---|---|
| S19-fwd-00 | stack C4 PRETURN + SMSC on HEAD (A = r18_c4) | rc 0, 72 cases | 591 / 191 | -2.73% | -2.75% | 128/16 (non-causal BM192 40, as on 80b8151) | reference for the rest |
| S19-fwd-01 | WGID: scheduler-barrier ids from the shfl'd warp_group_idx, computed once (no S2R remat) | rc 0 | 584 / **201** | -0.36% | +0.90% (passes +0.9..1.3% drift) | unchanged | reject: ptxas sinks the pre-turn rho/weight MUFUs past the now-early BAR.SYNC, so the turn grows by 10 |
| S19-fwd-02 | REBNI: lazy-rebase check uses max_l2<Check_inf> (no -inf guard on unmasked steps; the consumed row max is finite there) | rc 0 | 587 / 191 | -0.31% | -0.36% | unchanged | piece (pre-turn) |
| S19-fwd-03 | UNROLL2: `#pragma unroll 2` on the unmasked loop | rc 1 (d256 dK) | - | +12.61% | +12.37% | STACK 152-544, C7512 serialized | reject |
| S19-fwd-04 / 05 | WGID + REBNI (two screens) | rc 0 / rc 0 | 580 / 201 | -0.36 / -0.42% | -1.18 / -1.37% (passes -0.6..-0.8% drift) | unchanged | below bar |
| S19-fwd-06 / 07 | = 04 vs r18_c4 (stack + piece) | (= 04) | | -3.07 / -3.03% | -2.26 / -1.95% (passes +0.9..1.2% drift) | | not >= 3% at H32 |
| S19-fwd-08 | = 04 vs stack, 4-rep | | | -0.32% | +1.08% (passes +1.3..1.7% drift) | | inconclusive / no |
| S19-fwd-09 | = 04 vs r18_c4, 4-rep | | | -3.14% | -2.55% | | below bar at H32 |
| **S19-fwd-10** | **REBNI + SCPOST**: the SMSC scale read (LDS x2, R2UR, the q*k use) is issued after the ring-turn arrive (after the QK issue), not after the K wait inside the turn | rc 0, 72 cases | 587 / **184** (no LDS/R2UR in the turn) | -0.62% | -2.00% (passes -1.2..-1.4% drift) | unchanged (128/16; non-causal 40) | **candidate piece** |
| S19-fwd-11 | = 10 vs stack, 4-rep (passes flat) | | | **-0.70%** (per rep -0.73 -0.49 -1.18 -0.82) | **-0.61%** (-2.31 -0.63 -0.16 +0.14) | | piece >= 0.6% on both shapes |
| S19-fwd-12 | = 10 vs r18_c4, 4-rep (stack + piece; passes flat) | | | **-3.44%** (-3.40 -3.59 -3.33 -3.37) | **-3.61%** (-3.24 -4.18 -2.41 -4.54) | | full bitwise stack clears 3% |

Not retried (ledger): QK split / PV interleave (IL2, PVIL), ring changes (ARVEARLY, TT2/3, QKRING, KWEARLY, KTOK), VPF,
stages, BN64/BN128, shared basis, OSMEM. UNROLL2 (new) spills.

## 5. Candidate

C4 PRETURN + SMSC + S19-fwd-10 (REBNI + SCPOST), all bitwise (default route BITWISE EQUAL on 72 cases for 00 and 10), with no
resource change on the causal D128 symbol. vs r18_c4: fwd -3.44% (H16) and -3.61% (H32) in a 4-rep job (21193636/7), with the
unchanged kernels flat. The 2-rep screens of the same stack were -2.7% (00, stack alone). The piece alone vs the stack: -0.70% /
-0.61% (4-rep), and -0.62% / about -0.7% drift-corrected in the 2-rep screen. Patch: `.local/screens/S19-fwd-10/variant.patch`
(HEAD -> stack + piece); the piece alone on top of the stack is `.local/screens/S19-fwd-10/piece_on_stack.patch`. Gating still
needs the acceptance A/B (ab.py --grid paper, >= 3 reps, two jobs) and the 4-route bitwise gate. REBNI's -inf guard removal is
safe for causal / non-causal without local attention (Is_local is disabled in this build; the consumed tile's row max is
finite on every unmasked step). Re-check this if local attention is ever enabled.

## 6. Remaining gap

After the piece, the turn is 184 instructions, 128 of them rho + promote (the method's floor under the current contract),
8 QGMMA, and about 48 uniform/sync. Further bitwise instruction cuts in the turn are worth at most ~1%. The softmax-side
bookkeeping (~90 instructions) does not pay (softmax has slack). FA3 parity needs the rho / promote out of the turn, which
every measured layout so far rules out (registers at 160 per thread, numerics).
