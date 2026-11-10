# Delta-Matching sm90 fp8 attention: head_dim 128 numerics (branch `main`)

This is a read-only analysis of `delta-matching@main` (before PR #2). `dm_flagset.h` reduces the sources to the certified flag set,
so every `#if` below is resolved with those values.
Paths are relative to the repo root, and `file:N` means a line on `main`.
Notation: `fl(x)` means RN-even rounding to fp32. `L2 = float(M_LOG2E) = 1.4426950216293335`.
`R448 = fp32(1/448) = 0.0022321429569274187`. `sm = fp32(128**-0.5)`.

## 0. Flag resolution at hd128 (corrections to the brief)

| flag | effect at hd128 |
|---|---|
| `FA3FP11_HD128_BN256=1` | Forward tiles are **128 rows x 256 keys** (`flash_fwd_fp8f_launch.h:37-40`). That gives **4 P-hat groups of 64 keys** per tile, not 32-key groups (`mainloop_fwd_sm90_fp8f.hpp:216-222`: kQuantKeys=64, kGroups=4, kColsPerGroup=16, kAtomsPerGroup=2). |
| `FA3FP8F_TP_NSPLIT=2` | **d256 only.** hd128 uses `FA3FP11_HD128_NSPLIT` (default 1, pinned by the flagset): one 128-wide PV temp (`mainloop_fwd_sm90_fp8f.hpp:213-224`). |
| `FA3FP8F_TP_GPP=4` | **Not consulted at hd128.** The `kGroups==4 && kAtomsPerGroup==2` branch comes before the GPP branches and hard-codes one promote per tile (`mainloop_fwd_sm90_fp8f.hpp:1507-1515`). The numerics are the same as GPP=4. |
| `FA3FP8F_TP_PVFIRST=1` | d256 only. hd128 uses `FA3FP11_HD128_PVFIRST=1`; both give the PV-first schedule (`:509`). This is issue order only. |
| `FA3FP8F_PHAT_FLOOR_B=12` | Applies (`flash_fwd_kernel_sm90_fq8f.h:429-430`). |
| `HD128_P1R=1` | Pass 1 is `mainloop_bwd_sm90_fp8r.hpp` with kBlockM 64 and kBlockN 128 (`flash_bwd_fp8v10_launch.h:50-63,169`). |
| `P1R_MASKDIAG=1`, `P1R_RFSKIP=1` | Bitwise-neutral: the mask is skipped where it is a no-op, and `x*1.0f` is skipped. |
| `P1R_SAGEDP=0` | dP is computed on the e4m3 codes. |
| `HD128_P0V5=1`, `P0V6=1` | Pass 0 is `fp11_delta_v6.h` (included by `flash_delta_hdim128_e4m3_fp11v5_sm90.cu:5`). v4 is not used at hd128. |
| `HALFPSI` | Off (`delta_matching_api.cpp:306`). There is one psi per 64x128 dS' tile. |

Build: `--use_fast_math` (`_buildcfg.py:117`). As a result, `exp2f` becomes `ex2.approx.ftz.f32`, `__logf` becomes
`lg2.approx.ftz * ln2`, fp32 `/` is an approximate divide, and denormals are flushed. Section 8 lists every site this touches.

## 1. Producers (Python, API route) — confirmations and corrections

The route is `attn.py:212` → `producers.quant_qkv_m17(q,k,v, blocks=(128,64))`. Note that `FUSED_TWIN=0` by default, so Q
goes through the frozen `quant._quant_kernel` (`producers.py:164-171`). On the backward side, `attn.py:239` →
`producers.doq_m17(dO, O|None, H, 128, want_delta=(mode!="matched"))`.

- **Q** (`quant.py:59-75`, MODE 0, BLK=128 rows, D=128):
  - `x = fl(bf16->fp32(q) * fp32(sm))`.
  - `s = max(amax|x|, 1e-30) / 448`. This is a **Triton `/`, which lowers to `div.full.f32` (≤2 ulp), not IEEE.** Hypothesis to
    verify on a GPU: it equals `fl(amax*R448)` whenever `rcp(448)` rounds to R448. Otherwise call the Triton kernel itself.
  - `code = e4m3_rne(clamp(div_rn(x, s), ±448))`. `div_rn` is IEEE, which matches torch `x/s` on fp32 CUDA.
  - Confirmed: Q is pre-multiplied by sm, and the kernels never apply sm to S again (`delta_matching_api.cpp:167`
    sets scale_softmax=1).
- **K** (MODE 1, 64 rows/scale):
  - `x = fl(k - km)`, where `km[bh,d] = fold_j(tl.sum(K[64j:64j+64,d])) / N` (`quant.py:33-37`, `producers.py:41-65`).
  - The fold over j is sequential fp32. **However, the order inside `tl.sum` over 64 rows is Triton's reduction tree.**
    `/N` is `div.full`. As a result, km is not guaranteed bitwise in torch. Reuse `producers.kmean_m17` / `quant._kmean_kernel`.
  - Scales and codes are formed as for Q.
- **V** (MODE 2, 64 rows/scale): plain.
- **dO** (`producers.py:262-281`, 128 rows/scale at hd128): **`s = fl(max(amax,1e-30) * R448)`, a multiply, unlike Q/K/V.**
  `code = e4m3(clamp(div_rn(x,s),±448))`.
- **Stale delta** (`producers.py:289-297`): `dlt = Σ_chunk tl.sum(x*o, axis=1)` over the two 64-column chunks, all in fp32.
  - `x` is the **raw bf16 dO**, not the codes (`DELTA_FROM_CODES=0`, `producers.py:300-322`).
  - The order inside `tl.sum` is Triton's tree, so this is ±1 ulp in torch. Reuse the Triton producer for bitwise.
- **Twins**: `kTt` is a plain transpose (`quant.py:76-79`). `Qt` and `dOt` are pi-ordered (`twins.py`). These are data movement
  only: they change only the k order inside an MMA, which torch cannot reproduce anyway.
- **Scale tensor layout**: `[BH, N/blk]` fp32, viewed as `(B, H(_k), nblk)`.

## 2. Forward (`fwd_fp8`; hd128 = 128x256 tiles, 2 consumer WGs x 64 rows)

**Outputs** (`delta_matching_api.cpp:151-152`):
- O is `(B,N,H,D)` bf16.
- LSE is `(B,H,N)` fp32, in **natural log and true units**, over the sm-scaled, K-centered scores. There is no offset
  (`softmax_fq8f.h:325-338`).

**Per (bh, m-block of 128 rows)**
- Scales:
  - `qs = q8s[bh, m]` (`mainloop_fwd_sm90_fp8f.hpp:1343-1345`): one Q scale per 128 rows, shared by both WGs.
  - `ks_g = k8s[bh_kv, 4t+g]` and `vs_g = v8s[bh_kv, 4t+g]` for g=0..3 (`:1364-1376`, `:1390-1394`).
  - `bh_kv = b*Hk + h/(H/Hk)` (`:1177`, `:1349`).
- Key tiles t (256 keys) are visited in **descending** order: `n_block = n_block_max-1 … 0` (`:1261`, `:1754-1767`).
  - Under causal masking, `n_block_max = ceil(min(128(m+1), N)/256)`.
  - Every row has at least one live key in its first (diagonal) tile, so `row_max` is never -inf after tile 1.
- Tile step (PV-first schedule, `:1666-1709`; prologue `:1424-1453`; tail `:1779-1803`):
  1. `Sacc = wgmma_fp8(Q8[rows], K8[tile]^T)`: fp32 accumulator, 4 k32 atoms over D (`:1688`).
  2. Mask (FA3 `mask.h`): `Sacc[i,j] = -inf` if `j > i`, plus the seqlen_k tail.
     - It is applied to the raw accumulator, **before** the descale (`:1432-1433`, `:1691-1692`).
     - Masked tiles are those with `n_block >= n_block_min_causal_local_mask`; these also run with `Check_inf=true`. The
       results are identical either way.
  3. Descale (`fq8_descale`, `:1364-1376`): `S = fl(Sacc * fl(qs*ks_g))`, elementwise, with g = 64-key group.
     - On `main`, `-inf * fl(qs*ks)` gives NaN if `qs*ks` underflows to 0 (all-zero Q and K blocks: 1e-30/448 squared). That
       is a known bug, fixed on the sm90 speed-up branch (PR #2) (§6).
  4. Group maxima and the running max (`softmax_fq8f.h:186-237`):
     - `gm_g = max_{j in g} S` (fmaxf, order-free).
     - First tile: `m = max_g gm_g`. Otherwise `m_new = max(m, max_g gm_g)`, `alpha = ex2(fl(fl(m_old - m_new) * L2))`,
       and `l_thread *= alpha` (`:221-227`).
     - Then `ms_g = fl(fmaxf(fl(gm_g*L2), floor) - fl(log2 448))`, with `floor = fl(m*L2) - 12` (`:230-234`, `:153`).
       **The `m*L2 - 12` may or may not be contracted to an FMA** (see §8).
  5. P-hat (`:255`): `Pf = ex2(fma(S, L2, -ms_g))`. The FMA contraction on main is confirmed by the PR #2 forward commit's text,
     "fma(RN(x g), log2e, -m)".
     - The **effective scale per (row, 64-key group) is `2^ms_g`**: `C = e4m3(448 * exp(S - nu_g))`, with
       `nu_g = max(gm_g, m - 12 ln2)`.
     - This basis is **not a power of two**. It maps the group max to 448 unless the floor binds, in which case the group is
       more than 2^12 below the running max.
     - A fully masked group has `gm_g = -inf`. The floor keeps `ms_g` finite, so `Pf = 0`.
     - `P8 = cvt.rn.satfinite.e4m3(Pf)` (`:1704`, via cutlass `NumericArrayConverter`). In torch, use `clamp(±448)` then
       `.to(e4m3fn)`.
  6. Row sum, from **unquantized** `Pf` (`softmax_fq8f.h:264-289`). The sums are per thread:
     - Thread c = lane%4 of a quad holds, for each of its 2 rows, the columns `64g + 8k + 2c + e` (k=0..7, e=0..1).
       Within a group they are summed sequentially in the order (k, e): `part_g = (((x0 + x1) + x2) + …)`.
     - The thread's tile contribution is `acc = Σ_{g=0..3} fma(ex2(ms_g - m*L2), part_g, acc)`, starting from acc=0 and
       running g in order. `m*L2` here is also contraction-ambiguous.
     - The thread state is `l_c = (first tile) acc`, else `l_c*alpha + acc`.
     - The quad reduce happens **only at finalize**, using `Allreduce<4>`: `(l_c + l_{c^2}) + (l_{c^1} + l_{c^3})`
       (`:327`; FA3 `utils.h:79-98`).
  7. PV for the tile. It is deferred and runs at the start of the next step, using **this tile's `ms_` and `m`**
     (`mainloop_fwd_sm90_fp8f.hpp:1507-1515`, `:1672-1681`):

         D  = mma(P8[:,g0], V8[g0])                 # zero-init tensor-core accumulator, 2 k32 atoms per group
         D  = fl(D * rho_1); D += mma(P8[:,g1], V8[g1])
         D  = fl(D * rho_2); D += mma(P8[:,g2], V8[g2])
         D  = fl(D * rho_3); D += mma(P8[:,g3], V8[g3])
         O  = fma(D, w3, O)                         # promote on CUDA cores, fp32
         rho_g = clamp(fl(fl(vs_{g-1}/vs_g) * ex2(ms_{g-1} - ms_g)), 1e-30, 1e30)    # :1410-1417 (div is approximate)
         w3    = fl(vs_3 * ex2(ms_3 - m*L2))                                         # softmax_fq8f.h:309-316 (Check_inf=true)

     - The **V scale enters only through rho and w3**, as per-row factors on the accumulator, never inside the codes.
     - The next step then computes its max and applies `O = fl(O*alpha)` (`:1701`), so O always holds the true
       numerator in units of exp(-m).
  8. Finalize (`softmax_fq8f.h:325-338`; `mainloop_fwd_sm90_fp8f.hpp:1791-1803`):
     - `l = quad_reduce(l_c)`.
     - `inv = (l==0||nan) ? 0 : 1.f/l`. This is an approximate reciprocal under fast-math.
     - `LSE = (l==0||nan) ? -inf : fma(m, fl(L2*fl(ln2)), __logf(l))`. Since `fl(L2*fl(ln2)) == 1.0f` exactly, this reduces to
       `LSE = fl(m + fl(lg2.approx(l) * fl(ln2)))`.
     - Then the last tile's promote runs, then `O *= inv`, then bf16 RN in the epilogue.
     - Write order: **promote the last tile before the 1/l scale**.
- Lazy-rescale thresholds: **none on main**. Alpha is applied every tile (`:1701`). The only "laziness" is FA3's normal
  online max.
- Fully masked rows cannot occur in causal self-attention. If one did, the result would be O=0 and LSE=-inf
  (`softmax_fq8f.h:302-307`, `flash_fwd_kernel_sm90_fq8f.h:461-462`).

## 3. Backward pass 1 (dK, dV, dS' codes, psi) — `mainloop_bwd_sm90_fp8r.hpp`

**Geometry**
- One CTA per (bh, n-block of 128 keys). WG w ∈ {0,1} owns keys `[128n+64w, +64)`.
- m-blocks of 64 rows are visited **ascending**, from `m_min = 2n` (causal) to `N/64 - 1` (`:587-590`; FA3 `block.h:83-101`;
  window_right=-1 gives (128n+1)/64 = 2n).
- Scales:
  - `ks = k8s[bh_kv, 2n+w]` and `vs = v8s[bh_kv, 2n+w]` (`:654-655`).
  - `qs = q8s[bh, m>>1]` and `ds = do8s[bh, m>>1]` (`:658-660`, `:707-712`).
- Inputs:
  - The LSE is the forward's natural-log LSE.
  - delta is the stale or matched value in **true dP units** (`delta_matching_api.cpp:459-470`).
  - The mainloop's `softmax_scale` is **1.0f** (`flash_bwd_fp8v10_launch.h:116`).

**Per tile (m, n, w)**
```
c_s  = fl(fl(qs*ks)*L2)                                   # :713
dpdf = fl(fl(ds*vs)*ks)                                   # :720-722
Sacc = wgmma(K8[64 keys], Q8[64 rows]^T)                  # swapAB; claimed bitwise = transpose of the non-swapped S (:17-18)
Sacc = -inf where key > row                               # :739 (MASKDIAG: only m < 2n+2, :936-944, no-op elsewhere)
lse2 = fl(LSE_i * L2)                                     # :742 (__fmul_rn)
P'   = ex2( min( fl(fma(Sacc, c_s, -lse2) + 8), 12 ) )    # :746  F1 "cliff clamp": P' <= 2^12 (P <= 16)
dPacc= wgmma(V8[64 keys], dO8[64 rows]^T)
dS'  = fl( P' * fma(dPacc, dpdf, -fl(delta_i*ks)) )        # :816-820  = 2^8 * k * P*(dP - delta) in true units (FMA contraction of the inner term assumed)
amax = max |dS'| over the whole 64x128 tile (both WGs; named barrier) # :826-835
tile_zero = fl(amax*R448) < 1e-30                         # :836   (F2)
psi  = max( max(fl(amax*R448), 1e-30), fl(fl(gmax*F)/qs) ) # :837-838 (F2b floor; F = fp32(256*fp32(1e-8)) = 2.56e-6; '/' approximate)
psi_inv = tile_zero ? 0 : 1/psi                            # :839 (approximate rcp/div)
psi_out[bh,m,n] = tile_zero ? 0 : psi                      # :841
P8   = e4m3_satfinite(P')                                 # :845-847  (P' may exceed 448 only if P > 1.75; it saturates)
dS8  = e4m3_satfinite(fl(dS' * psi_inv))                   # :850-854 ; stored natural [BH, N/64, N/128, 64, 128] (api:301)
f_cur = fl(ds/256); rf = clamp(f_prev/f_cur, 1e-8, 1e8)    # :885-886 (approximate div); f_prev starts at 1.0f
if rf != 1: Vacc = fl(Vacc*rf)                             # :887-890 ; f_prev = f_cur
g_fold = tile_zero ? 0 : fl(fl(psi*2^-8)*qs);  gmax = max(gmax, g_fold) if !tile_zero   # :892-897 (gmax: tiles < m only)
Vacc += wgmma(P8^T, dO8)          # tensor-core accumulate, NOT zero-init (:899-904)
Ktmp  = wgmma(dS8^T, Q8)          # zero-init per tile (:905-910)
Kacc  = fma(Ktmp, g_fold, Kacc)   # :921  (fp32, the "plan-A fold")
if (m & 7) == 7 or m == m_max-1:  # :923-930  dV window = 8 m-tiles (512 rows), geometry-fixed
    dVs = fma(Vacc, f_prev, dVs); Vacc = 0
```
- At exit (`:948-952`): `dK = bf16(fl(Kacc * fl(1.0f/ks)))`, where the division is approximate because `softmax_scale` is a
  runtime 1.0f. `dV = bf16(dVs)`. Both are written per **query head**, layout `(B, N, H, D)` (`delta_matching_api.cpp:288-290`).
- **What psi is:** the per-(bh, 64-row m-block, 128-key n-block) dS' quantization scale, in dS' units
  (`2^8 * k8s * dS_true`, with the k8s fold per 64-key half).
  - True dS is recovered as `code*psi/(2^8*k8s(j))` (`delta_matching_api.cpp:666-667`).
  - `psi = 0` is the F2 zero-tile sentinel.
  - **F2b** raises psi so that this tile's dK basis `g = psi*qs/256` is ≥ 1e-8 x the largest previous g in this key block.
    The effect is to coarsen negligible tiles, not to skip them.
- **How sm and K-centering enter:**
  - Q codes already carry sm, so dK has no extra factor, and dQ gets `sm/256` in pass 2.
  - The kernel returns the gradient with respect to **centered** K. The mean-subtraction Jacobian term
    `-(1/N)Σ_j dKc_j = -(sm/N)Σ_i Q_i·rowsum_i(dS)` is **not applied**. It vanishes only if `rowsum(dS) = 0`, which is the
    reason for the matched delta.

## 4. Pass 2 (dQ) — `fq8_dq_pass2_hd128.h`

**Geometry:** one CTA per (bh, 64-row mb), launched longest-first (`:209`). nb (128-key tiles) runs ascending over
`[0, nb_end)`. Under causal masking, `nb_end = mb/2 + 1` (`:215`).

```
psi_w = 0; psi_max = 0
for nb:
  psi = psi_out[bh, mb, nb]
  if psi != 0 and not (psi < fl(psi_max*1e-8f)):               # :302  F2 sentinel + F2b dominance skip
      if psi_w != 0: acc = fl(acc * clamp(psi_w/psi, 1e-8, 1e8)); acc += wgmma(dS8[mb,nb], K8[nb])   # :303-309
      else:          acc  = wgmma(dS8[mb,nb], K8[nb])          # zero-init (:317)
      psi_w = psi; psi_max = max(psi_max, psi)
  if (nb & 3) == 3 or nb == nb_end-1:                            # :359 window = 4 tiles = 512 keys
      if psi_w != 0: dQs = fma(acc, psi_w, dQs); psi_w = 0       # :367
dQ = bf16(fl(dQs * fl(sm/256)))                                  # :391, :426
```
- Here `K8` is the K-centered e4m3 codes read through the plain-transposed `kTt`. The k8s factor is already folded into the
  dS' codes, which is why one psi can span two K scale groups.
- dQ is bf16, layout `(B,N,H,D)`.
- Skipped tiles are transparent: they neither start nor end a window.

## 5. Pass 0 (matched delta) — `fp11_delta_v6.h`

**Geometry**
- Grid `(N/128, B*H)` (`:512`). CTA p owns rows `[128p, 128p+128)`: WG w owns m-block `2p+w`.
- Scales are `qs = q8s[p]` and `ds = do8s[p]` (`:324-325`).
- Under causal masking, the n-blocks of 64 keys are `0 .. 2p+1`, visited **ascending** (`:217-223`), in stages of 2.
- `lse2 = fl(LSE*L2)` (`:331`).

**Per 64x64 tile (nb)**
- `dsc = fl(fl(qs*ks_nb)*L2)` and `dp = fl(ds*vs_nb)` (`:411-414`). This is the same expression order as pass 1.
- The mask is applied only where it can bind (`:342-345`).
- `P' = ex2(min(fl(fma(Sacc,dsc,-lse2)+8),12))` (`:353`). This is **pass 1's line verbatim**, on the same fp32 `Sacc`, and is
  **not** quantized.
- `dPacc = wgmma(dO8, V8)`.

**Row sum (`:360-375`):** per thread c (quad lane), per row:
```
a0 = 0; for the 8 elems in keys 0..31 (cols 8k+2c+e, order k then e): a0 = fma(P', fl(dPacc*dp), a0)
a1 = same for keys 32..63
a0 = (a0_c + a0_{c^2}) + (a0_{c^1} + a0_{c^3})      # Allreduce<4>
acc_fp64 += double(fl(a0*2^-8)); acc_fp64 += double(fl(a1*2^-8))
delta_i = float(acc_fp64)                            # :460, one fp32 rounding
```
- **Which values are summed:** the fp32 **unquantized** P' (not the e4m3 P8 that feeds dV) times the fp32 dP in true units.
  - These are exactly the fp32 operands of pass 1's dS line, assuming the S/dP wgmma results are bitwise equal between
    pass 0's (Q·K) and pass 1's swapAB (K·Q) geometry. The source asserts this was "measured, not argued".
  - The sum is in true dP units. k8s is not included; pass 1 applies it.
- **Result:** `delta_i = Σ_j P_ij dP_ij` with the backward's own P and dP. Pass 1 then gives
  `rowsum_j(dS_ij) = Σ P dP - delta·Σ P = delta·(1 - Σ_j P_ij)` (up to the different fp32 roundings in the dS line).
  - This is zero to the extent that the backward-recomputed P sums to 1.
  - That holds to ~fp32/ex2 round-off, because the LSE comes from the same S codes.
  - Stale mode instead feeds `rowsum(dO_bf16 * O_bf16)`, which violates the identity by the fp8 error.
- Fully masked tiles add `+0`. A CTA with no keys writes delta=0 (`:224-230`).

## 6. GQA, determinism, layouts, shape rules

**GQA**
- K and V are read per KV head: `bh_kv = b*Hk + h/(H/Hk)` in every pass (`mainloop_bwd_sm90_fp8r.hpp:653`, `fq8_dq_pass2_hd128.h:231`, `mainloop_fwd_sm90_fp8f.hpp:1177,1349`).
- dK and dV come back **per query head** in bf16. The Python side folds them with `dk.reshape(B,Hk,rep,N,D).sum(2)`, a torch
  bf16 sum with an fp32 accumulator and a single rounding (`attn.py:266-269`). To reproduce it, call the same torch op.

**Determinism:** no atomics, fixed orders, one writer per output (`core.py:11`). Splitting the backward per head group is
bitwise identical (`split.py:272-276`).

**Layouts and dtypes**
- O, dq, dk, dv: `(B,N,H,D)` bf16, returned as `(B,H,N,D)` views.
- LSE and delta: `(B,H,N)` fp32.
- Workspaces: `ds8 [BH, N/64, N/128, 64, 128]` e4m3 and `psi [BH, N/64, N/128, 1]` fp32.

**Shape rules at hd128**
- The API requires N % 256 == 0 (`core.py:142-144`). The forward requires seqlen_k % 256 because of BN256
  (`delta_matching_api.cpp:136-140`). Pass 1 requires seqlen_k % 128 and pass 0 requires 128-multiples.
- Matched mode requires causal. Self-attention only. B*H ≤ 65535.

**Block boundaries that affect bits**
- Forward: 256-key tiles (promote per tile), 64-key P groups, 128-row Q scale blocks.
- Pass 1: dV window of 8 m-tiles aligned at `m & 7`; the dK fold is per 64-row tile.
- Pass 2: window of 4 n-tiles aligned at `nb & 3`.

## 7. `the sm90 speed-up branch` vs `main` (`git log main..the sm90 speed-up branch`, 8 commits)

| commit | change | changes output bits? |
|---|---|---|
| Forward (D128) | hd128 forward: **BM192 x BN128** (BN256→0, so 2 groups and one rho per 128-key tile; Q scale index per WG). **FWDLAZY**: O is kept against a lagging reference `mref` and rebased only when `abs(m*L2 - mref) > 8`, with factor `min(ex2(mref-m*L2), 256)`; the final scale includes `ex2(mref - m*L2)`. **FWDDFOLD**: on unmasked steps the exp line is `ex2(fma(Sacc, fl(g*L2), -ms))`, with ms built from `max(raw)*g`, replacing `ex2(fma(fl(Sacc*g), L2, -ms))`. The descale is floored at FLT_MIN. The remaining pieces (RHOPRE, SMSC, …) are bitwise. | **Yes**: P codes, O and LSE, and therefore every backward output. |
| Passes 0 + 1 exp line (D128) | **EXPBX64** (passes 0 and 1): on 64x64 tiles that pass 0 does not mask, `P' = ex2(fma(Sacc, dsc, -b))` with `b = fma(LSE, L2, -8)` and no clamp. Masked tiles keep main's line. Removing the clamp alone is bitwise. | **Yes**: P', delta, dS', dV, dK, dQ. |
| Pass 0 row sums; pass 1 dV window | **P0DEFER/P0LAZY**: per-lane `a = fma(P', dPacc, a)` with no per-element descale; `fl(fl(a0*dp0 + a1*dp1) * 2^-8)` added in fp64 per lane; one fp64 quad reduce at the end (xor 1, then xor 2). **PW32**: the dV promote window becomes 32 m-tiles (2048 rows). | **Yes**: delta (matched) and dV, so also dK and dQ. |
| No -inf * 0 fix | No `-inf*0`: the descale is floored at FLT_MIN (`1.17549435e-38`) on maskable tiles (pass 0, pass 1, d256 forward). | Only when `fl(qs*ks) < FLT_MIN` (for example all-zero Q and K blocks), where main produces NaN, or `P' = 2^12` on masked entries in the backward. |
| Pass 2, pass 1 schedule, producers, pass 0 dispatch | Pipeline depth, psi prefetch, schedule, strided producers, pass-0 dispatch. | No (bitwise, per the commit text). |

To emulate the sm90 speed-up branch (PR #2) you need a separate pseudocode branch. Everything below describes `main`.

## 8. Torch-like pseudocode (main, hd128)

Helper functions:
- `ex2` = `ex2.approx.ftz.f32`.
- `fma32(a,b,c)` = fp32 FMA. Compute it in fp64 and round to fp32; exact except for ~2^-29-probability double rounding.
- `e4m3(x)` = `clamp(x, -448, 448).to(float8_e4m3fn)`, which matches `cvt.rn.satfinite`.
- `mma(A8,B8)` = matmul of the dequantized code values. The products are exact; the accumulation is §9 item 1.

```python
L2 = f32(1.4426950216293335); R448 = f32(1/448); LOG2_448 = f32(8.807354922057604); sm = f32(128**-0.5)

# ---------------- producers ----------------
Q8,qs  = quant(fl(q*sm), rows=128, scale=lambda a: divfull(max(a,1e-30),448))
km     = kmean_triton(k)                       # reuse the Triton kernel for bitwise
K8,ks  = quant(fl(k-km), rows=64, ...);  V8,vs = quant(v, rows=64, ...)

# ---------------- forward, per (bh, 128-row block m) ----------------
for t in reversed(range(T_max(m))):            # 256-key tiles, descending
    Sacc = mma(Q8[r], K8[t].T); Sacc[causal] = -inf
    S    = fl(Sacc * fl(qs[m]*ks[4t+g]))        # per 64-key group g
    gm   = groupmax(S)                          # [rows,4]
    if first: M = gm.max(-1)
    else:     Mn = max(M, gm.max(-1)); alpha = ex2(fl(fl(M-Mn)*L2)); lc *= alpha; M = Mn
              O = fma32(Dprev, w3prev, O)       # promote of the PREVIOUS tile, before alpha (PV-first)
              O = fl(O*alpha)
    ms   = fl(max(fl(gm*L2), fl(M*L2)-12) - LOG2_448)        # floor: FMA-contraction ambiguous
    Pf   = ex2(fma32(S, L2, -ms[g]))
    lc   = per_thread_rowsum(Pf, ms, M) (+ lc)                # quad-lane layout of section 2.6
    P8   = e4m3(Pf)
    D    = pv_chain(P8, V8[t], vs[4t:4t+4], ms)               # rho rescales between groups, TC accumulate
    w3   = fl(vs[4t+3]*ex2(ms[3] - M*L2))
    Dprev, w3prev = D, w3
l = quad_allreduce(lc); LSE = fl(M + fl(lg2approx(l)*f32(ln2))); O = fma32(D, w3, O); O = bf16(fl(O*rcp(l)))

# ---------------- dO producer ----------------
dO8,dos = quant(dO, rows=128, scale=lambda a: fl(max(a,1e-30)*R448))
delta   = stale: tl_rowsum(dO_bf16*O_bf16)    |   matched: pass0()

# ---------------- pass 0 (matched), per row i in 64-row block, n-blocks of 64 keys ascending ----------------
#   P' = ex2(min(fl(fma32(Sacc, fl(fl(qs*ks)*L2), -fl(LSE*L2)) + 8), 12))
#   per quad lane: a0,a1 fma chains of P' * fl(dPacc*fl(ds*vs)); Allreduce4; fp64 += fl(a*2^-8); delta = f32(acc)

# ---------------- pass 1, per 128-key block n, WG w, m = 2n.. ascending ----------------  (section 3 verbatim)
# ---------------- pass 2, per 64-row block mb, nb = 0..mb//2 ascending ----------------    (section 4 verbatim)
dk, dv = per-query-head bf16 -> torch .sum over rep (GQA)
```

## 9. Where torch cannot be bit-exact, and expected tolerance

1. **The fp8 wgmma accumulator** (S, dP, PV, dV, dK_tmp, dQ). Hopper fp8 MMA accumulates with roughly 13-14 effective
   mantissa bits per k32 step (the kernel's own comments: `mainloop_fwd_sm90_fp8f.hpp:73-81`, `fq8_dq_pass2_hd128.h:17-23`).
   The exact alignment and truncation rule is undocumented.
   - The chain lengths are: S/dP 4 atoms; PV 8 atoms per tile with rescales; dV up to 8 tiles x 2 atoms;
     dQ up to 4 tiles x 4 atoms.
   - fp32 or fp64 torch matmuls differ at about 1e-4 relative per chain.
   - **This dominates.** It flips e4m3 P and dS' codes near rounding midpoints. My rough estimate is 1e-3..1e-2 of codes;
     this is an estimate, not a measurement.
2. **The MUFU approximations**: `ex2.approx.ftz` (P', alpha, rho, w, sigma), `lg2.approx` (LSE), `rcp.approx`/`div.full`
   (1/l, 1/psi, `vs_{g-1}/vs_g`, `f_prev/f_cur`, `psi_w/psi`, `gmax*F/qs`, `1/ks`). These are ±1-2 ulp versus torch.
   - Workaround: these single instructions can be reproduced with `triton tl.inline_asm_elementwise` (for example
     `ex2.approx.ftz.f32`), which is torch-callable but not pure torch.
3. **FMA contraction** is chosen by nvcc.
   - Confirmed by later commit text: the forward exp line (`fma(S,L2,-ms)`) and the backward exp line
     (`fma(Sacc,c,-lse2)+8`).
   - Assumed but not verified: the dS line inner FMA (`fp8r:820`), the forward floor `m*L2-12` (`softmax_fq8f.h:230`), and
     `ms - m*L2` inside ex2 (`:271`, `:313`).
   - The fold `acc += e*part` is very likely an FFMA.
   - Resolving these needs `cuobjdump -sass` of a built `_C*.so`. None exists in the checkout, and there is no cuobjdump
     on the login node.
4. **Triton producers**: the K-mean `tl.sum` tree, the `div.full` scale for Q/K/V, and the stale-delta `tl.sum` order. Call
   the Triton kernels; they are deterministic.
5. **Per-thread summation orders** (forward l, pass-0 a0/a1) can be reproduced exactly in torch with the lane map of
   §2.6 / §5. They only buy bits if items 1-2 are also matched.

**Tolerance to expect without emulating items 1-2.** These figures are estimates; validate them on a GPU.
- Bitwise: all scales and Q/K/V/dO codes (given item 4).
- O and LSE: about 1 bf16 ulp per element; normwise relative error ~1e-4..1e-3.
- dV: about the same.
- dK and dQ in matched mode: normwise ~1e-3. They are more sensitive, because dS' codes and psi follow the small residual
  `dP - delta`, and F2b / tile-zero decisions can flip on near-threshold tiles.
- Delta (matched): ~1e-6 relative.
- For scale, the repo's own fp32-reference bounds are O/dV 8e-2 and dQ/dK 1.2e-1 (`tests/test_dm_attn.py:12-15`). That is
  the fp8 floor, about 100x larger than the emulation gap.
