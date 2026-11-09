# FA4 baseline on GB200 (sm100), head dim 128 — the target to beat

**2026-11-09: causal only (user). Non-causal rows below are reference only.**

## FOCUS (user, 2026-11-08 23:50): N = 8k, 32k, 64k only. The headline is the geomean over these 6 shapes.

Suite `focus` (attnbench.py --suite focus, the default of run_ab.sh / run_suite4.sh): H/Hk in {16/16, 32/8},
N in {8k, 32k, 64k}, B = max(1, 32768 // N) = 4 / 1 / 1, causal and non-causal. 64k is a single 64k-token sequence:
the long-context training case. At B=1 it still has 512 128-row tiles x H heads (>= 8192 CTA tiles), so all 148 SMs
are busy. Memory: q/k/v/o/dO/grads at 64k H32 is ~3.5 GB, which fits easily. FA4, cuDNN and FA4-fp8 all run at 64k.
Sanity at N > 8k (new): O and dQ are checked exactly in fp32 on the last 512 query rows (the full context), and dK / dV
against FA4 bf16 (FA4's own O/dQ error there is 2.4e-3; FA4-fp8 O 5.4e-2).
Measured nvl72052-T18 GPUs 2-3 with the filler on the other GPUs (results/focus_baseline.s*.jsonl), ms (TFLOPS):

| H/Hk | N | B | causal | FA4 fwd | FA4 bwd | **FA4 fwd+bwd** | FA4-det fwd+bwd | cuDNN fwd+bwd | FA4-fp8 fwd (kernel only) |
|---|---|---|---|---|---|---|---|---|---|
| 16/16 | 8k | 4 | 1 | 0.688 (1599) | 2.014 (1365) | 2.714 (1418) | 2.823 (1363) | 2.722 (1414) | 0.629 (1747) |
| 16/16 | 32k | 1 | 1 | 2.499 (1760) | 8.215 (1338) | 11.327 (1359) | 11.704 (1315) | 11.851 (1299) | 2.241 (1962) |
| 16/16 | 64k | 1 | 1 | 11.757 (1496) | 34.682 (1268) | 47.867 (1286) | 48.532 (1269) | 48.780 (1262) | 8.814 (1996) |
| 32/8 | 8k | 4 | 1 | 1.289 (1706) | 4.342 (1266) | 5.899 (1305) | 6.140 (1253) | 6.122 (1257) | 1.205 (1826) |
| 32/8 | 32k | 1 | 1 | 5.483 (1604) | 17.442 (1261) | 23.863 (1290) | 24.312 (1266) | 24.745 (1244) | 4.407 (1996) |
| 32/8 | 64k | 1 | 1 | 24.253 (1451) | 70.024 (1256) | 94.507 (1303) | 97.978 (1257) | 97.971 (1257) | 17.986 (1956) |
| 16/16 | 8k | 4 | 0 | 1.401 (1569) | 3.875 (1419) | 5.377 (1431) | 5.741 (1341) | 5.751 (1338) | 1.350 (1629) |
| 16/16 | 32k | 1 | 0 | 5.487 (1603) | 16.669 (1319) | 24.277 (1268) | 24.404 (1262) | 24.787 (1242) | 4.997 (1760) |
| 16/16 | 64k | 1 | 0 | 24.722 (1423) | 69.237 (1270) | 96.785 (1272) | 100.504 (1225) | 101.538 (1213) | 18.814 (1870) |
| 32/8 | 8k | 4 | 0 | 2.665 (1650) | 8.242 (1334) | 11.639 (1323) | 11.940 (1289) | 12.370 (1244) | 2.594 (1695) |
| 32/8 | 32k | 1 | 0 | 11.812 (1489) | 36.470 (1206) | 49.875 (1235) | 50.541 (1218) | 51.018 (1207) | 9.641 (1825) |
| 32/8 | 64k | 1 | 0 | 50.996 (1380) | 139.715 (1259) | 192.939 (1277) | 200.650 (1227) | 203.538 (1210) | 37.982 (1853) |

- At 64k, FA4 bf16 falls to 1.45-1.50 PF fwd and 1.25-1.27 PF bwd (SM clock 1.33-1.43 GHz under the power cap).
  FA4's fp8 fwd is 1.34x its bf16 fwd at 64k, against 1.07-1.10x at 8k. The longer the sequence, the more power-bound
  the kernel, and the more a lower-energy fp8 kernel gains.
- cuDNN is 0.96-0.97x FA4 on fwd+bwd, and FA4-det is 0.97x. FA4 default stays the baseline.

**4k added (user, 2026-11-09).** N=4096, B=8, measured on GPUs 1-2 (median of 2), causal / non-causal, ms (TFLOPS):
| H/Hk | FA4 fwd | FA4 fwd+bwd | FA4-det fwd+bwd | cuDNN fwd+bwd | FA4-fp8 fwd |
|---|---|---|---|---|---|
| 16/16 causal | 0.399 (1378) | 1.567 (1228) | 1.628 (1182) | 1.534 (1254) | 0.348 (1581) |
| 32/8 causal | 0.699 (1573) | 2.989 (1288) | 3.115 (1236) | 3.144 (1224) | 0.641 (1717) |
| 16/16 non-causal | 0.691 (1591) | 2.626 (1466) | 2.739 (1405) | 2.693 (1429) | 0.696 (1579) |
| 32/8 non-causal | 1.284 (1713) | 5.353 (1438) | 5.487 (1403) | 5.827 (1321) | 1.382 (1591) |

---

Owner: fa4-bench. Measured 2026-11-08 on nvl72063-T13 (gpupool node), torch 2.14.1+cu130, FA4 = flash-attn-4 @ b2edfab9
(Dao-AILab main, CuTe DSL, cutlass-dsl 4.8.1), cuDNN 9.24. Raw: results/baseline_exclusive.jsonl, results/baseline_filler-on.jsonl.

## How to measure a candidate (do exactly this; numbers from anywhere else are not comparable)

    G=~/workspace/haok1402/sm100/gpupool/bin/gpupool; cd ~/workspace/haok1402/sm100/bench
    $G run --gpus 1 --who <you> --shell -- "source ../env/activate.sh && python attnbench.py --arms fa4 \
        --cand mine=/abs/path/cand.py:attn --suite main --tag <tag> --out results/<name>.jsonl --resume"
    python report.py results/<name>.jsonl            # per-shape ms/TFLOPS + geomean speedup over fa4_bf16
    # full suite on all 4 GPUs (one shard per GPU, all busy):  ./run_suite4.sh <tag> results/<name>.jsonl "--arms fa4 --cand mine=/abs/cand.py:attn"
    #   then: python report.py results/<name>.s*.jsonl
    # headline numbers: a few shapes with --exclusive-node --timeout 15m (the other 3 GPUs idle meanwhile; keep it short)

The candidate and FA4 are timed in the same process on the same GPU, interleaved in 6 rounds; the speedup is the ratio of
medians from that run. Do not compare a candidate number against the absolute tables below (GPU-to-GPU spread ~4%).

## Shapes (suite `main`) and why

hd128; seqlen 1k, 2k, 4k, 8k, 16k, 32k with 32k tokens per batch (B = 32768/N). H/Hk = 16/16 (FlashAttention's own
benchmark config at hd128, so numbers are comparable to published FA4 results) and 32/8 (Llama-3-8B GQA, the realistic
training case). Every shape has >= 4096 128-row tiles: all 148 SMs are filled. Causal and non-causal. Matched mode needs
causal, so **the primary target is causal fwd+bwd**; non-causal is reported against the "stale" mode.
`--suite full` adds 64/8 (Llama-3-70B ratio) and 32/32.

## Timing method (attnbench.py)

L2 flushed (4x L2) before every call; the GPU is pre-loaded with a spin so the host is ahead (as in a training step):
CUDA events then measure GPU execution, not Python launch latency (host time per call is recorded separately and a
launch-bound flag is set if host > 0.9x GPU time). Cross-check: CUPTI summed kernel time (torch.profiler); agrees with the
events within ~1-2%. Adaptive iterations (>= 30 calls and >= 1 s per arm/pass), median with p10/p90 (p10-p90 spread
typically < 2%). bwd = autograd.grad of a forward done outside the timed window. TFLOPS use the FA convention
(fwd 4·B·H·N²·D, /2 causal; bwd 2.5x; fwd+bwd 3.5x), so a kernel doing extra work gets no credit for it. A sanity gate
compares O/dQ/dK/dV against an fp32 reference (rel. L2; FA4 bf16 ~2.3e-3, FA4 fp8 fwd ~5e-2) and checks that the output
responds to the input; failing arms are marked SUSPECT and left out of speedups.
Clocks cannot be locked (no root). They are recorded instead: the GPUs run at 2062 MHz on short shapes and fall to
1400-1650 MHz on long non-causal shapes because of the ~1.2 kW power cap. That is how they behave in training too.

## Neighbors' filler on vs exclusive node: which numbers are honest

FA4 times with the 3 neighbors running the matmul filler (GPU1) vs a quiet node (GPU0): median ratio 0.991, range
0.966-1.060 over 72 points. That is within the GPU-to-GPU spread, so the neighbors don't measurably slow FA4 here; the
GPU's own power cap dominates. **Honest numbers:** (1) speedups = same-run interleaved ratios, valid with filler on; (2)
absolute headline tables = `--exclusive-node` (below). Screening during evolution can use plain `--gpus 1`.

## Targets: FA4 bf16, exclusive node (ms, TFLOPS in parentheses)

| H/Hk | N | B | causal | fwd ms (TF) | bwd ms (TF) | fwd+bwd ms (TF) | FA4 fp8 fwd ms (TF) |
|---|---|---|---|---|---|---|---|
| 16/16 | 1024 | 32 | 1 | 0.161 (853) | 0.569 (604) | 0.721 (668) | 0.135 (1021) |
| 16/16 | 2048 | 16 | 1 | 0.245 (1121) | 0.778 (883) | 1.016 (947) | 0.206 (1333) |
| 16/16 | 4096 | 8 | 1 | 0.407 (1351) | 1.199 (1146) | 1.589 (1211) | 0.350 (1572) |
| 16/16 | 8192 | 4 | 1 | 0.705 (1560) | 2.064 (1332) | 2.770 (1389) | 0.630 (1745) |
| 16/16 | 16384 | 2 | 1 | 1.336 (1646) | 3.832 (1435) | 5.183 (1485) | 1.178 (1867) |
| 16/16 | 32768 | 1 | 1 | 2.539 (1733) | 7.735 (1421) | 10.844 (1420) | 2.240 (1963) |
| 32/8 | 1024 | 32 | 1 | 0.255 (1076) | 0.948 (725) | 1.193 (806) | 0.214 (1282) |
| 32/8 | 2048 | 16 | 1 | 0.413 (1331) | 1.391 (988) | 1.787 (1077) | 0.356 (1545) |
| 32/8 | 4096 | 8 | 1 | 0.712 (1544) | 2.302 (1194) | 3.015 (1276) | 0.641 (1714) |
| 32/8 | 8192 | 4 | 1 | 1.329 (1655) | 4.274 (1286) | 5.589 (1377) | 1.204 (1827) |
| 32/8 | 16384 | 2 | 1 | 2.516 (1748) | 8.216 (1338) | 11.289 (1364) | 2.287 (1923) |
| 32/8 | 32768 | 1 | 1 | 5.228 (1682) | 16.690 (1318) | 23.245 (1324) | 4.390 (2004) |
| 16/16 | 1024 | 32 | 0 | 0.223 (1235) | 0.730 (942) | 0.942 (1022) | 0.198 (1388) |
| 16/16 | 2048 | 16 | 0 | 0.393 (1400) | 1.131 (1215) | 1.501 (1282) | 0.366 (1502) |
| 16/16 | 4096 | 8 | 0 | 0.712 (1544) | 1.957 (1405) | 2.665 (1444) | 0.696 (1580) |
| 16/16 | 8192 | 4 | 0 | 1.351 (1628) | 3.667 (1499) | 5.055 (1522) | 1.349 (1630) |
| 16/16 | 16384 | 2 | 0 | 2.586 (1701) | 7.362 (1494) | 10.598 (1452) | 2.601 (1691) |
| 16/16 | 32768 | 1 | 0 | 5.204 (1690) | 15.181 (1449) | 21.942 (1403) | 4.988 (1763) |
| 32/8 | 1024 | 32 | 0 | 0.395 (1393) | 1.257 (1094) | 1.617 (1190) | 0.362 (1519) |
| 32/8 | 2048 | 16 | 0 | 0.695 (1583) | 2.096 (1311) | 2.791 (1379) | 0.686 (1603) |
| 32/8 | 4096 | 8 | 0 | 1.310 (1678) | 3.813 (1442) | 5.197 (1481) | 1.333 (1650) |
| 32/8 | 8192 | 4 | 0 | 2.504 (1756) | 7.686 (1431) | 10.706 (1438) | 2.598 (1693) |
| 32/8 | 16384 | 2 | 0 | 5.138 (1712) | 16.366 (1344) | 23.152 (1330) | 5.020 (1752) |
| 32/8 | 32768 | 1 | 0 | 10.932 (1609) | 33.571 (1310) | 46.425 (1326) | 9.616 (1830) |

## Reference arms (exclusive node, geomean of FA4_ms / arm_ms over the 12 shapes)

| arm | fwd causal | fwd non-causal | bwd causal | bwd non-causal | fwd+bwd causal | fwd+bwd non-causal |
|---|---|---|---|---|---|---|
| cuDNN SDPA bf16 | 0.959 | 1.001 | 0.981 | 0.929 | 0.975 | 0.959 |
| FA4 fp8 fwd (kernel only, quantization untimed) | 1.149 | 1.038 | - | - | - | - |

FA4 is the fastest bf16 baseline (cuDNN is within ~2-7%; it wins on short causal bwd, e.g. 1k/2k MHA by ~10%).
**FA4's fp8 forward is only 1.04-1.15x its bf16 forward, even though GB200's dense fp8 is 2x bf16.** At hd128 the
forward is limited by the softmax (MUFU exp2 and FMA), not by the tensor cores. An fp8 Delta-Matching kernel will only
beat FA4 by a large margin if it also cuts the softmax and exp cost (e.g. FA4's polynomial exp2 emulation on FMA, fewer
rescales) and the bwd's non-MMA work. The bwd (2.5x the fwd FLOPs, ~1.3-1.4 PF) is where most of the fwd+bwd time goes.

## Full per-pass tables (exclusive node)

| H/Hk | N | B | causal | pass | fa4_bf16 ms (TF) | sdpa_cudnn_bf16 ms (TF) | fa4_fp8_fwd ms (TF) |
|---|---|---|---|---|---|---|---|
| 16/16 | 1024 | 32 | 1 | fwd | 0.161 (853) | 0.166 (826) | 0.135 (1021) |
| 16/16 | 2048 | 16 | 1 | fwd | 0.245 (1121) | 0.253 (1085) | 0.206 (1333) |
| 16/16 | 4096 | 8 | 1 | fwd | 0.407 (1351) | 0.415 (1324) | 0.350 (1572) |
| 16/16 | 8192 | 4 | 1 | fwd | 0.705 (1560) | 0.714 (1540) | 0.630 (1745) |
| 16/16 | 16384 | 2 | 1 | fwd | 1.336 (1646) | 1.347 (1632) | 1.178 (1867) |
| 16/16 | 32768 | 1 | 1 | fwd | 2.539 (1733) | 2.565 (1715) | 2.240 (1963) |
| 32/8 | 1024 | 32 | 1 | fwd | 0.255 (1076) | 0.292 (941) | 0.214 (1282) |
| 32/8 | 2048 | 16 | 1 | fwd | 0.413 (1331) | 0.454 (1211) | 0.356 (1545) |
| 32/8 | 4096 | 8 | 1 | fwd | 0.712 (1544) | 0.754 (1458) | 0.641 (1714) |
| 32/8 | 8192 | 4 | 1 | fwd | 1.329 (1655) | 1.379 (1595) | 1.204 (1827) |
| 32/8 | 16384 | 2 | 1 | fwd | 2.516 (1748) | 2.592 (1697) | 2.287 (1923) |
| 32/8 | 32768 | 1 | 1 | fwd | 5.228 (1682) | 5.407 (1627) | 4.390 (2004) |
| 16/16 | 1024 | 32 | 0 | fwd | 0.223 (1235) | 0.219 (1257) | 0.198 (1388) |
| 16/16 | 2048 | 16 | 0 | fwd | 0.393 (1400) | 0.380 (1445) | 0.366 (1502) |
| 16/16 | 4096 | 8 | 0 | fwd | 0.712 (1544) | 0.690 (1594) | 0.696 (1580) |
| 16/16 | 8192 | 4 | 0 | fwd | 1.351 (1628) | 1.314 (1673) | 1.349 (1630) |
| 16/16 | 16384 | 2 | 0 | fwd | 2.586 (1701) | 2.529 (1739) | 2.601 (1691) |
| 16/16 | 32768 | 1 | 0 | fwd | 5.204 (1690) | 5.370 (1638) | 4.988 (1763) |
| 32/8 | 1024 | 32 | 0 | fwd | 0.395 (1393) | 0.395 (1393) | 0.362 (1519) |
| 32/8 | 2048 | 16 | 0 | fwd | 0.695 (1583) | 0.694 (1585) | 0.686 (1603) |
| 32/8 | 4096 | 8 | 0 | fwd | 1.310 (1678) | 1.320 (1665) | 1.333 (1650) |
| 32/8 | 8192 | 4 | 0 | fwd | 2.504 (1756) | 2.509 (1753) | 2.598 (1693) |
| 32/8 | 16384 | 2 | 0 | fwd | 5.138 (1712) | 5.380 (1635) | 5.020 (1752) |
| 32/8 | 32768 | 1 | 0 | fwd | 10.932 (1609) | 11.333 (1552) | 9.616 (1830) |
| 16/16 | 1024 | 32 | 1 | bwd | 0.569 (604) | 0.508 (676) | - |
| 16/16 | 2048 | 16 | 1 | bwd | 0.778 (883) | 0.730 (942) | - |
| 16/16 | 4096 | 8 | 1 | bwd | 1.199 (1146) | 1.158 (1187) | - |
| 16/16 | 8192 | 4 | 1 | bwd | 2.064 (1332) | 2.065 (1331) | - |
| 16/16 | 16384 | 2 | 1 | bwd | 3.832 (1435) | 3.922 (1402) | - |
| 16/16 | 32768 | 1 | 1 | bwd | 7.735 (1421) | 8.276 (1329) | - |
| 32/8 | 1024 | 32 | 1 | bwd | 0.948 (725) | 1.080 (636) | - |
| 32/8 | 2048 | 16 | 1 | bwd | 1.391 (988) | 1.496 (919) | - |
| 32/8 | 4096 | 8 | 1 | bwd | 2.302 (1194) | 2.397 (1147) | - |
| 32/8 | 8192 | 4 | 1 | bwd | 4.274 (1286) | 4.286 (1283) | - |
| 32/8 | 16384 | 2 | 1 | bwd | 8.216 (1338) | 8.716 (1261) | - |
| 32/8 | 32768 | 1 | 1 | bwd | 16.690 (1318) | 17.486 (1258) | - |
| 16/16 | 1024 | 32 | 0 | bwd | 0.730 (942) | 0.709 (970) | - |
| 16/16 | 2048 | 16 | 0 | bwd | 1.131 (1215) | 1.143 (1202) | - |
| 16/16 | 4096 | 8 | 0 | bwd | 1.957 (1405) | 2.037 (1350) | - |
| 16/16 | 8192 | 4 | 0 | bwd | 3.667 (1499) | 3.839 (1432) | - |
| 16/16 | 16384 | 2 | 0 | bwd | 7.362 (1494) | 8.160 (1347) | - |
| 16/16 | 32768 | 1 | 0 | bwd | 15.181 (1449) | 17.096 (1286) | - |
| 32/8 | 1024 | 32 | 0 | bwd | 1.257 (1094) | 1.461 (941) | - |
| 32/8 | 2048 | 16 | 0 | bwd | 2.096 (1311) | 2.349 (1170) | - |
| 32/8 | 4096 | 8 | 0 | bwd | 3.813 (1442) | 4.243 (1296) | - |
| 32/8 | 8192 | 4 | 0 | bwd | 7.686 (1431) | 8.604 (1278) | - |
| 32/8 | 16384 | 2 | 0 | bwd | 16.366 (1344) | 17.519 (1255) | - |
| 32/8 | 32768 | 1 | 0 | bwd | 33.571 (1310) | 35.191 (1250) | - |
| 16/16 | 1024 | 32 | 1 | fwd_bwd | 0.721 (668) | 0.669 (719) | - |
| 16/16 | 2048 | 16 | 1 | fwd_bwd | 1.016 (947) | 0.978 (983) | - |
| 16/16 | 4096 | 8 | 1 | fwd_bwd | 1.589 (1211) | 1.562 (1232) | - |
| 16/16 | 8192 | 4 | 1 | fwd_bwd | 2.770 (1389) | 2.794 (1377) | - |
| 16/16 | 16384 | 2 | 1 | fwd_bwd | 5.183 (1485) | 5.318 (1447) | - |
| 16/16 | 32768 | 1 | 1 | fwd_bwd | 10.844 (1420) | 11.277 (1365) | - |
| 32/8 | 1024 | 32 | 1 | fwd_bwd | 1.193 (806) | 1.364 (705) | - |
| 32/8 | 2048 | 16 | 1 | fwd_bwd | 1.787 (1077) | 1.937 (993) | - |
| 32/8 | 4096 | 8 | 1 | fwd_bwd | 3.015 (1276) | 3.170 (1214) | - |
| 32/8 | 8192 | 4 | 1 | fwd_bwd | 5.589 (1377) | 5.774 (1333) | - |
| 32/8 | 16384 | 2 | 1 | fwd_bwd | 11.289 (1364) | 11.706 (1315) | - |
| 32/8 | 32768 | 1 | 1 | fwd_bwd | 23.245 (1324) | 23.806 (1293) | - |
| 16/16 | 1024 | 32 | 0 | fwd_bwd | 0.942 (1022) | 0.922 (1043) | - |
| 16/16 | 2048 | 16 | 0 | fwd_bwd | 1.501 (1282) | 1.511 (1274) | - |
| 16/16 | 4096 | 8 | 0 | fwd_bwd | 2.665 (1444) | 2.736 (1407) | - |
| 16/16 | 8192 | 4 | 0 | fwd_bwd | 5.055 (1522) | 5.296 (1453) | - |
| 16/16 | 16384 | 2 | 0 | fwd_bwd | 10.598 (1452) | 11.168 (1378) | - |
| 16/16 | 32768 | 1 | 0 | fwd_bwd | 21.942 (1403) | 22.666 (1358) | - |
| 32/8 | 1024 | 32 | 0 | fwd_bwd | 1.617 (1190) | 1.838 (1047) | - |
| 32/8 | 2048 | 16 | 0 | fwd_bwd | 2.791 (1379) | 3.051 (1261) | - |
| 32/8 | 4096 | 8 | 0 | fwd_bwd | 5.197 (1481) | 5.541 (1389) | - |
| 32/8 | 8192 | 4 | 0 | fwd_bwd | 10.706 (1438) | 11.461 (1343) | - |
| 32/8 | 16384 | 2 | 0 | fwd_bwd | 23.152 (1330) | 23.259 (1324) | - |
| 32/8 | 32768 | 1 | 0 | fwd_bwd | 46.425 (1326) | 46.919 (1312) | - |

Speedup over fa4_bf16 (geomean of fa4_bf16_ms / arm_ms over shapes):

| arm | pass | causal | geomean | min | max | n |
|---|---|---|---|---|---|---|
| fa4_fp8_fwd | fwd | 0 | 1.038 | 0.964 | 1.137 | 12 |
| fa4_fp8_fwd | fwd | 1 | 1.149 | 1.100 | 1.198 | 12 |
| sdpa_cudnn_bf16 | bwd | 0 | 0.929 | 0.860 | 1.030 | 12 |
| sdpa_cudnn_bf16 | bwd | 1 | 0.981 | 0.877 | 1.119 | 12 |
| sdpa_cudnn_bf16 | fwd | 0 | 1.001 | 0.955 | 1.033 | 12 |
| sdpa_cudnn_bf16 | fwd | 1 | 0.959 | 0.874 | 0.992 | 12 |
| sdpa_cudnn_bf16 | fwd_bwd | 0 | 0.959 | 0.880 | 1.021 | 12 |
| sdpa_cudnn_bf16 | fwd_bwd | 1 | 0.975 | 0.875 | 1.077 | 12 |
