# A/B (acceptance run): grid all, 3 reps alternating A/B order, median per config

- A = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/r10_psipf (library sha256 e1119aaaa89ca07e)
- B = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/wt/mpoly (library sha256 76e806cb65ad66b8)

### Per-kernel sums, all configurations

| kernel | A ms (sum of medians) | B ms | change | A rep spread | B rep spread |
|---|---:|---:|---:|---:|---:|
| fwd | 203.834 | 203.990 | +0.08% | 0.14% | 0.31% |
| prod_fwd | 16.333 | 16.336 | +0.02% | 0.21% | 0.08% |
| prod_bwd | 8.559 | 8.557 | -0.03% | 0.09% | 0.04% |
| pass0 | 39.196 | 39.335 | +0.35% | 0.65% | 0.56% |
| pass1 | 540.767 | 555.684 | +2.76% | 0.22% | 0.32% |
| pass2 | 107.935 | 107.960 | +0.02% | 0.09% | 0.06% |
| twin | 6.298 | 6.299 | +0.01% | 0.11% | 0.04% |
| glue | 42.008 | 42.006 | -0.00% | 0.23% | 0.09% |
| e2e_fwd | 249.579 | 249.258 | -0.13% | 0.19% | 0.11% |
| e2e_fwdbwd | 962.975 | 978.812 | +1.64% | 0.28% | 0.07% |

### Acceptance predicates (target pass1, D128 tuned)

- target over all D128 shapes: 265.271 -> 280.097 ms (+5.59%; needs <= -3.0%)
- D128 shapes slower beyond spread: [[128, 1, 16, 4, 32768, 'causal', 'matched'], [128, 1, 16, 4, 32768, 'causal', 'stale'], [128, 1, 16, 4, 32768, 'full', 'stale'], [128, 1, 16, 16, 32768, 'causal', 'matched'], [128, 1, 16, 16, 32768, 'causal', 'stale'], [128, 1, 16, 16, 32768, 'full', 'stale'], [128, 1, 24, 8, 12288, 'causal', 'stale'], [128, 1, 24, 8, 12288, 'full', 'stale'], [128, 2, 16, 4, 16384, 'causal', 'matched'], [128, 2, 16, 4, 16384, 'causal', 'stale'], [128, 2, 16, 4, 16384, 'full', 'stale'], [128, 2, 16, 16, 16384, 'causal', 'matched'], [128, 2, 16, 16, 16384, 'causal', 'stale'], [128, 2, 16, 16, 16384, 'full', 'stale'], [128, 2, 32, 32, 3072, 'causal', 'matched'], [128, 2, 32, 32, 3072, 'causal', 'stale'], [128, 2, 32, 32, 3072, 'full', 'stale'], [128, 3, 12, 4, 6144, 'causal', 'matched'], [128, 3, 12, 4, 6144, 'causal', 'stale'], [128, 3, 12, 4, 6144, 'full', 'stale'], [128, 4, 16, 4, 8192, 'causal', 'matched'], [128, 4, 16, 4, 8192, 'causal', 'stale'], [128, 4, 16, 4, 8192, 'full', 'stale'], [128, 4, 16, 16, 8192, 'causal', 'matched'], [128, 4, 16, 16, 8192, 'causal', 'stale'], [128, 4, 16, 16, 8192, 'full', 'stale'], [128, 8, 16, 4, 4096, 'causal', 'matched'], [128, 8, 16, 4, 4096, 'causal', 'stale'], [128, 8, 16, 4, 4096, 'full', 'stale'], [128, 8, 16, 16, 4096, 'causal', 'matched'], [128, 8, 16, 16, 4096, 'causal', 'stale'], [128, 8, 16, 16, 4096, 'full', 'stale'], [128, 16, 16, 4, 2048, 'causal', 'matched'], [128, 16, 16, 4, 2048, 'causal', 'stale'], [128, 16, 16, 4, 2048, 'full', 'stale'], [128, 16, 16, 16, 2048, 'causal', 'stale'], [128, 16, 16, 16, 2048, 'full', 'stale']]
- other head dims: target 275.496 -> 275.588 ms (+0.03%, slower beyond spread: False); shapes slower beyond spread: none
- checks {'target_gain': False, 'tune_shapes': False, 'other_shapes': True, 'other_target': True} -> **REJECT**

### Shapes over the 1.0% fwd+bwd gate

- (128, 1, 16, 4, 32768, 'causal', 'matched'): +2.37% (A reps [31.227, 31.081, 31.083], B reps [31.785, 31.821, 32.001]; spreads 0.47% / 0.68%) BEYOND SPREAD
- (128, 1, 16, 4, 32768, 'causal', 'stale'): +3.62% (A reps [25.709, 25.977, 25.849], B reps [26.784, 26.55, 26.84]; spreads 1.04% / 1.08%) BEYOND SPREAD
- (128, 1, 16, 4, 32768, 'full', 'stale'): +3.42% (A reps [50.746, 50.792, 50.908], B reps [52.531, 52.387, 52.553]; spreads 0.32% / 0.32%) BEYOND SPREAD
- (128, 1, 16, 16, 32768, 'causal', 'matched'): +2.97% (A reps [31.125, 31.209, 31.314], B reps [32.222, 32.0, 32.135]; spreads 0.61% / 0.69%) BEYOND SPREAD
- (128, 1, 16, 16, 32768, 'causal', 'stale'): +3.79% (A reps [25.958, 25.973, 25.961], B reps [26.944, 26.964, 26.64]; spreads 0.06% / 1.20%) BEYOND SPREAD
- (128, 1, 16, 16, 32768, 'full', 'stale'): +3.45% (A reps [50.706, 51.358, 51.043], B reps [52.595, 52.802, 52.847]; spreads 1.28% / 0.48%) BEYOND SPREAD
- (128, 1, 24, 8, 12288, 'causal', 'matched'): +2.74% (A reps [6.933, 7.08, 6.846], B reps [7.166, 7.123, 7.086]; spreads 3.38% / 1.13%) (within spread)
- (128, 1, 24, 8, 12288, 'causal', 'stale'): +3.41% (A reps [5.85, 5.857, 5.913], B reps [6.056, 6.023, 6.075]; spreads 1.07% / 0.86%) BEYOND SPREAD
- (128, 1, 24, 8, 12288, 'full', 'stale'): +2.78% (A reps [11.131, 11.211, 11.095], B reps [11.441, 11.393, 11.457]; spreads 1.04% / 0.56%) BEYOND SPREAD
- (128, 2, 16, 4, 16384, 'causal', 'matched'): +5.03% (A reps [15.842, 16.015, 16.084], B reps [16.821, 16.416, 16.944]; spreads 1.51% / 3.14%) BEYOND SPREAD
- (128, 2, 16, 4, 16384, 'causal', 'stale'): +3.15% (A reps [13.214, 13.337, 13.399], B reps [13.807, 13.75, 13.757]; spreads 1.38% / 0.41%) BEYOND SPREAD
- (128, 2, 16, 4, 16384, 'full', 'stale'): +3.85% (A reps [25.614, 25.687, 25.829], B reps [26.676, 26.653, 26.733]; spreads 0.84% / 0.30%) BEYOND SPREAD
- (128, 2, 16, 16, 16384, 'causal', 'matched'): +2.76% (A reps [16.099, 16.18, 16.21], B reps [16.692, 16.613, 16.626]; spreads 0.69% / 0.47%) BEYOND SPREAD
- (128, 2, 16, 16, 16384, 'causal', 'stale'): +3.25% (A reps [13.536, 13.548, 13.494], B reps [13.978, 13.865, 13.976]; spreads 0.40% / 0.80%) BEYOND SPREAD
- (128, 2, 16, 16, 16384, 'full', 'stale'): +3.54% (A reps [25.739, 25.841, 26.044], B reps [26.756, 26.83, 26.695]; spreads 1.18% / 0.50%) BEYOND SPREAD
- (128, 2, 32, 32, 3072, 'causal', 'matched'): +1.81% (A reps [1.689, 1.684, 1.685], B reps [1.716, 1.712, 1.716]; spreads 0.28% / 0.27%) BEYOND SPREAD
- (128, 2, 32, 32, 3072, 'causal', 'stale'): +1.95% (A reps [1.49, 1.49, 1.487], B reps [1.519, 1.519, 1.522]; spreads 0.21% / 0.20%) BEYOND SPREAD
- (128, 2, 32, 32, 3072, 'full', 'stale'): +3.15% (A reps [2.304, 2.321, 2.306], B reps [2.384, 2.363, 2.379]; spreads 0.76% / 0.88%) BEYOND SPREAD
- (128, 3, 12, 4, 6144, 'causal', 'matched'): +2.09% (A reps [3.03, 3.035, 3.038], B reps [3.099, 3.098, 3.1]; spreads 0.25% / 0.07%) BEYOND SPREAD
- (128, 3, 12, 4, 6144, 'causal', 'stale'): +3.19% (A reps [2.598, 2.598, 2.596], B reps [2.693, 2.666, 2.681]; spreads 0.08% / 0.99%) BEYOND SPREAD
- (128, 3, 12, 4, 6144, 'full', 'stale'): +2.85% (A reps [4.624, 4.614, 4.624], B reps [4.763, 4.756, 4.749]; spreads 0.22% / 0.28%) BEYOND SPREAD
- (128, 4, 16, 4, 8192, 'causal', 'matched'): +2.28% (A reps [8.545, 8.64, 8.61], B reps [8.806, 8.891, 8.805]; spreads 1.10% / 0.97%) BEYOND SPREAD
- (128, 4, 16, 4, 8192, 'causal', 'stale'): +4.20% (A reps [7.281, 7.184, 7.227], B reps [7.464, 7.539, 7.53]; spreads 1.34% / 1.00%) BEYOND SPREAD
- (128, 4, 16, 4, 8192, 'full', 'stale'): +3.16% (A reps [13.319, 13.412, 13.462], B reps [13.836, 13.801, 13.838]; spreads 1.07% / 0.27%) BEYOND SPREAD
- (128, 4, 16, 16, 8192, 'causal', 'matched'): +3.46% (A reps [8.672, 8.703, 8.87], B reps [9.022, 9.004, 8.989]; spreads 2.27% / 0.37%) BEYOND SPREAD
- (128, 4, 16, 16, 8192, 'causal', 'stale'): +3.43% (A reps [7.469, 7.368, 7.418], B reps [7.692, 7.673, 7.632]; spreads 1.37% / 0.78%) BEYOND SPREAD
- (128, 4, 16, 16, 8192, 'full', 'stale'): +3.32% (A reps [13.564, 13.528, 13.566], B reps [13.904, 14.014, 14.021]; spreads 0.28% / 0.83%) BEYOND SPREAD
- (128, 8, 16, 4, 4096, 'causal', 'matched'): +1.74% (A reps [4.951, 4.952, 4.958], B reps [5.036, 5.072, 5.038]; spreads 0.15% / 0.72%) BEYOND SPREAD
- (128, 8, 16, 4, 4096, 'causal', 'stale'): +2.27% (A reps [4.319, 4.323, 4.32], B reps [4.414, 4.461, 4.418]; spreads 0.11% / 1.07%) BEYOND SPREAD
- (128, 8, 16, 4, 4096, 'full', 'stale'): +2.65% (A reps [7.33, 7.272, 7.289], B reps [7.482, 7.52, 7.482]; spreads 0.81% / 0.51%) BEYOND SPREAD
- (128, 8, 16, 16, 4096, 'causal', 'matched'): +1.89% (A reps [5.133, 5.121, 5.11], B reps [5.227, 5.215, 5.217]; spreads 0.46% / 0.24%) BEYOND SPREAD
- (128, 8, 16, 16, 4096, 'causal', 'stale'): +1.78% (A reps [4.508, 4.511, 4.484], B reps [4.588, 4.608, 4.587]; spreads 0.58% / 0.47%) BEYOND SPREAD
- (128, 8, 16, 16, 4096, 'full', 'stale'): +2.41% (A reps [7.473, 7.433, 7.443], B reps [7.623, 7.619, 7.625]; spreads 0.54% / 0.08%) BEYOND SPREAD
- (128, 16, 16, 4, 2048, 'causal', 'matched'): +2.18% (A reps [3.195, 3.206, 3.191], B reps [3.256, 3.266, 3.264]; spreads 0.45% / 0.30%) BEYOND SPREAD
- (128, 16, 16, 4, 2048, 'causal', 'stale'): +2.07% (A reps [2.868, 2.858, 2.855], B reps [2.916, 2.918, 2.918]; spreads 0.46% / 0.07%) BEYOND SPREAD
- (128, 16, 16, 4, 2048, 'full', 'stale'): +2.16% (A reps [4.301, 4.296, 4.286], B reps [4.389, 4.387, 4.39]; spreads 0.35% / 0.06%) BEYOND SPREAD
- (128, 16, 16, 16, 2048, 'causal', 'matched'): +1.31% (A reps [3.387, 3.541, 3.484], B reps [3.733, 3.53, 3.497]; spreads 4.41% / 6.67%) (within spread)
- (128, 16, 16, 16, 2048, 'causal', 'stale'): +1.83% (A reps [3.047, 3.052, 3.05], B reps [3.098, 3.106, 3.11]; spreads 0.17% / 0.38%) BEYOND SPREAD
- (128, 16, 16, 16, 2048, 'full', 'stale'): +2.06% (A reps [4.502, 4.454, 4.444], B reps [4.53, 4.557, 4.546]; spreads 1.31% / 0.59%) (within spread)
- (256, 2, 8, 2, 16384, 'causal', 'matched'): +3.41% (A reps [15.587, 15.579, 16.257], B reps [15.774, 16.362, 16.118]; spreads 4.35% / 3.65%) (within spread)
- (256, 4, 8, 8, 8192, 'causal', 'matched'): +1.76% (A reps [8.561, 8.595, 8.513], B reps [8.711, 8.765, 8.609]; spreads 0.96% / 1.79%) (within spread)

### Per shape (fwd+bwd, ms)

| shape | attn | mode | A median | B median | change | A spread | B spread |
|---|---|---|---:|---:|---:|---:|---:|
| D128 B1 H16/4 N32768 | causal | matched | 31.083 | 31.821 | +2.37% | 0.47% | 0.68% | slower beyond spread
| D128 B1 H16/4 N32768 | causal | stale | 25.849 | 26.784 | +3.62% | 1.04% | 1.08% | slower beyond spread
| D128 B1 H16/4 N32768 | full | stale | 50.792 | 52.531 | +3.42% | 0.32% | 0.32% | slower beyond spread
| D128 B1 H16/16 N32768 | causal | matched | 31.209 | 32.135 | +2.97% | 0.61% | 0.69% | slower beyond spread
| D128 B1 H16/16 N32768 | causal | stale | 25.961 | 26.944 | +3.79% | 0.06% | 1.20% | slower beyond spread
| D128 B1 H16/16 N32768 | full | stale | 51.043 | 52.802 | +3.45% | 1.28% | 0.48% | slower beyond spread
| D128 B1 H24/8 N12288 | causal | matched | 6.933 | 7.123 | +2.74% | 3.38% | 1.13% |
| D128 B1 H24/8 N12288 | causal | stale | 5.857 | 6.056 | +3.41% | 1.07% | 0.86% | slower beyond spread
| D128 B1 H24/8 N12288 | full | stale | 11.131 | 11.441 | +2.78% | 1.04% | 0.56% | slower beyond spread
| D128 B2 H16/4 N16384 | causal | matched | 16.015 | 16.821 | +5.03% | 1.51% | 3.14% | slower beyond spread
| D128 B2 H16/4 N16384 | causal | stale | 13.337 | 13.757 | +3.15% | 1.38% | 0.41% | slower beyond spread
| D128 B2 H16/4 N16384 | full | stale | 25.687 | 26.676 | +3.85% | 0.84% | 0.30% | slower beyond spread
| D128 B2 H16/16 N16384 | causal | matched | 16.180 | 16.626 | +2.76% | 0.69% | 0.47% | slower beyond spread
| D128 B2 H16/16 N16384 | causal | stale | 13.536 | 13.976 | +3.25% | 0.40% | 0.80% | slower beyond spread
| D128 B2 H16/16 N16384 | full | stale | 25.841 | 26.756 | +3.54% | 1.18% | 0.50% | slower beyond spread
| D128 B2 H32/32 N3072 | causal | matched | 1.685 | 1.716 | +1.81% | 0.28% | 0.27% | slower beyond spread
| D128 B2 H32/32 N3072 | causal | stale | 1.490 | 1.519 | +1.95% | 0.21% | 0.20% | slower beyond spread
| D128 B2 H32/32 N3072 | full | stale | 2.306 | 2.379 | +3.15% | 0.76% | 0.88% | slower beyond spread
| D128 B3 H12/4 N6144 | causal | matched | 3.035 | 3.099 | +2.09% | 0.25% | 0.07% | slower beyond spread
| D128 B3 H12/4 N6144 | causal | stale | 2.598 | 2.681 | +3.19% | 0.08% | 0.99% | slower beyond spread
| D128 B3 H12/4 N6144 | full | stale | 4.624 | 4.756 | +2.85% | 0.22% | 0.28% | slower beyond spread
| D128 B4 H16/4 N8192 | causal | matched | 8.610 | 8.806 | +2.28% | 1.10% | 0.97% | slower beyond spread
| D128 B4 H16/4 N8192 | causal | stale | 7.227 | 7.530 | +4.20% | 1.34% | 1.00% | slower beyond spread
| D128 B4 H16/4 N8192 | full | stale | 13.412 | 13.836 | +3.16% | 1.07% | 0.27% | slower beyond spread
| D128 B4 H16/16 N8192 | causal | matched | 8.703 | 9.004 | +3.46% | 2.27% | 0.37% | slower beyond spread
| D128 B4 H16/16 N8192 | causal | stale | 7.418 | 7.673 | +3.43% | 1.37% | 0.78% | slower beyond spread
| D128 B4 H16/16 N8192 | full | stale | 13.564 | 14.014 | +3.32% | 0.28% | 0.83% | slower beyond spread
| D128 B8 H16/4 N4096 | causal | matched | 4.952 | 5.038 | +1.74% | 0.15% | 0.72% | slower beyond spread
| D128 B8 H16/4 N4096 | causal | stale | 4.320 | 4.418 | +2.27% | 0.11% | 1.07% | slower beyond spread
| D128 B8 H16/4 N4096 | full | stale | 7.289 | 7.482 | +2.65% | 0.81% | 0.51% | slower beyond spread
| D128 B8 H16/16 N4096 | causal | matched | 5.121 | 5.217 | +1.89% | 0.46% | 0.24% | slower beyond spread
| D128 B8 H16/16 N4096 | causal | stale | 4.508 | 4.588 | +1.78% | 0.58% | 0.47% | slower beyond spread
| D128 B8 H16/16 N4096 | full | stale | 7.443 | 7.623 | +2.41% | 0.54% | 0.08% | slower beyond spread
| D128 B16 H16/4 N2048 | causal | matched | 3.195 | 3.264 | +2.18% | 0.45% | 0.30% | slower beyond spread
| D128 B16 H16/4 N2048 | causal | stale | 2.858 | 2.918 | +2.07% | 0.46% | 0.07% | slower beyond spread
| D128 B16 H16/4 N2048 | full | stale | 4.296 | 4.389 | +2.16% | 0.35% | 0.06% | slower beyond spread
| D128 B16 H16/16 N2048 | causal | matched | 3.484 | 3.530 | +1.31% | 4.41% | 6.67% |
| D128 B16 H16/16 N2048 | causal | stale | 3.050 | 3.106 | +1.83% | 0.17% | 0.38% | slower beyond spread
| D128 B16 H16/16 N2048 | full | stale | 4.454 | 4.546 | +2.06% | 1.31% | 0.59% | slower beyond spread
| D256 B1 H8/2 N32768 | causal | matched | 30.002 | 30.186 | +0.61% | 0.64% | 0.51% |
| D256 B1 H8/2 N32768 | causal | stale | 25.636 | 25.584 | -0.20% | 0.92% | 0.39% |
| D256 B1 H8/2 N32768 | full | stale | 50.907 | 50.765 | -0.28% | 0.21% | 0.19% |
| D256 B1 H8/8 N32768 | causal | matched | 30.428 | 30.300 | -0.42% | 1.61% | 0.66% |
| D256 B1 H8/8 N32768 | causal | stale | 25.784 | 25.812 | +0.11% | 0.15% | 0.22% |
| D256 B1 H8/8 N32768 | full | stale | 50.992 | 51.014 | +0.04% | 0.83% | 0.34% |
| D256 B1 H12/4 N12288 | causal | matched | 6.803 | 6.811 | +0.11% | 0.69% | 1.70% |
| D256 B1 H12/4 N12288 | causal | stale | 5.909 | 5.913 | +0.07% | 0.68% | 0.31% |
| D256 B1 H12/4 N12288 | full | stale | 11.136 | 11.065 | -0.64% | 0.57% | 0.90% |
| D256 B2 H8/2 N16384 | causal | matched | 15.587 | 16.118 | +3.41% | 4.35% | 3.65% |
| D256 B2 H8/2 N16384 | causal | stale | 13.389 | 13.468 | +0.59% | 0.67% | 0.19% |
| D256 B2 H8/2 N16384 | full | stale | 25.753 | 25.840 | +0.34% | 0.47% | 0.61% |
| D256 B2 H8/8 N16384 | causal | matched | 15.772 | 15.786 | +0.09% | 0.32% | 0.41% |
| D256 B2 H8/8 N16384 | causal | stale | 13.528 | 13.555 | +0.20% | 0.67% | 0.67% |
| D256 B2 H8/8 N16384 | full | stale | 26.018 | 25.881 | -0.53% | 0.30% | 1.50% |
| D256 B2 H16/1 N5120 | causal | matched | 3.387 | 3.389 | +0.04% | 0.15% | 0.32% |
| D256 B2 H16/1 N5120 | causal | stale | 3.015 | 3.015 | -0.00% | 0.78% | 0.15% |
| D256 B2 H16/1 N5120 | full | stale | 5.412 | 5.416 | +0.07% | 0.15% | 0.87% |
| D256 B3 H6/2 N6144 | causal | matched | 2.965 | 2.964 | -0.02% | 0.33% | 0.45% |
| D256 B3 H6/2 N6144 | causal | stale | 2.638 | 2.623 | -0.55% | 0.19% | 0.17% |
| D256 B3 H6/2 N6144 | full | stale | 4.662 | 4.663 | +0.02% | 0.09% | 0.06% |
| D256 B4 H8/2 N8192 | causal | matched | 8.402 | 8.400 | -0.02% | 2.06% | 1.15% |
| D256 B4 H8/2 N8192 | causal | stale | 7.308 | 7.293 | -0.20% | 0.83% | 0.79% |
| D256 B4 H8/2 N8192 | full | stale | 13.411 | 13.433 | +0.16% | 1.39% | 0.85% |
| D256 B4 H8/8 N8192 | causal | matched | 8.561 | 8.711 | +1.76% | 0.96% | 1.79% |
| D256 B4 H8/8 N8192 | causal | stale | 7.440 | 7.478 | +0.52% | 0.66% | 0.59% |
| D256 B4 H8/8 N8192 | full | stale | 13.620 | 13.570 | -0.37% | 1.98% | 0.42% |
| D256 B8 H8/2 N4096 | causal | matched | 4.822 | 4.821 | -0.02% | 0.55% | 0.91% |
| D256 B8 H8/2 N4096 | causal | stale | 4.317 | 4.315 | -0.04% | 0.12% | 0.17% |
| D256 B8 H8/2 N4096 | full | stale | 7.350 | 7.356 | +0.08% | 0.48% | 0.20% |
| D256 B8 H8/8 N4096 | causal | matched | 5.001 | 4.980 | -0.42% | 0.53% | 0.24% |
| D256 B8 H8/8 N4096 | causal | stale | 4.486 | 4.492 | +0.13% | 0.16% | 0.15% |
| D256 B8 H8/8 N4096 | full | stale | 7.500 | 7.494 | -0.08% | 0.13% | 0.16% |
| D256 B16 H8/2 N2048 | causal | matched | 3.086 | 3.086 | +0.02% | 0.22% | 0.30% |
| D256 B16 H8/2 N2048 | causal | stale | 2.818 | 2.817 | -0.04% | 0.58% | 0.09% |
| D256 B16 H8/2 N2048 | full | stale | 4.318 | 4.315 | -0.07% | 0.65% | 0.24% |
| D256 B16 H8/8 N2048 | causal | matched | 3.283 | 3.272 | -0.33% | 0.41% | 0.37% |
| D256 B16 H8/8 N2048 | causal | stale | 2.994 | 2.992 | -0.08% | 0.13% | 0.03% |
| D256 B16 H8/8 N2048 | full | stale | 4.442 | 4.439 | -0.06% | 0.70% | 0.11% |

Worst single-shape fwd+bwd change (B vs A): +5.03%
