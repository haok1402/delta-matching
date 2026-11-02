# Forward sparse timeline (lb % 17 == 0, lb / 17 < 8; first 12 n-block steps of the first tile)

Build /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/trace7/delta_matching/_C.so (sha256 fd9cce219d12ca10). Bracketed regions of the instrumented kernel; medians over interior sampled steps (the first and last recorded step of each sample excluded); cycles.

## D128 B4 H16/16 N8192 causal: 8 samples, 160 warpgroup-steps

| region | median cycles | share of step |
|---|---:|---:|
| V wait | 117 | 1.9% |
| scheduler barrier | 39 | 0.6% |
| PV chain issue + in-chain promotes | 990 | 16.0% |
| PV retire | 171 | 2.8% |
| last promote + V release | 386 | 6.2% |
| K wait | 72 | 1.2% |
| QK issue + arrive | 158 | 2.5% |
| QK retire + K release | 468 | 7.5% |
| mask + descale | 280 | 4.5% |
| grouped max / scale | 616 | 9.9% |
| O rescale | 300 | 4.8% |
| grouped online softmax | 1706 | 27.5% |
| P conversion | 252 | 4.1% |
| v-scale load + step end | 100 | 1.6% |
| step (warpgroup) | 6208 | |

WG1 - WG0 step-start offset: 2054 cycles; share of each warpgroup's vector regions (mask .. P conversion) inside the other warpgroup's outstanding PV / QK windows: 24%.
Producer (thread 0): load_V bracket 185, load_K bracket 3925, V transpose bracket 1602, producer step 6150 cycles.

## D128 B4 H16/16 N8192 full: 8 samples, 160 warpgroup-steps

| region | median cycles | share of step |
|---|---:|---:|
| V wait | 69 | 1.2% |
| scheduler barrier | 39 | 0.7% |
| PV chain issue + in-chain promotes | 926 | 16.7% |
| PV retire | 150 | 2.7% |
| last promote + V release | 187 | 3.4% |
| K wait | 69 | 1.2% |
| QK issue + arrive | 158 | 2.8% |
| QK retire + K release | 454 | 8.2% |
| mask + descale | 285 | 5.1% |
| grouped max / scale | 612 | 11.0% |
| O rescale | 111 | 2.0% |
| grouped online softmax | 1798 | 32.4% |
| P conversion | 249 | 4.5% |
| v-scale load + step end | 72 | 1.3% |
| step (warpgroup) | 5556 | |

WG1 - WG0 step-start offset: 1444 cycles; share of each warpgroup's vector regions (mask .. P conversion) inside the other warpgroup's outstanding PV / QK windows: 24%.
Producer (thread 0): load_V bracket 198, load_K bracket 3738, V transpose bracket 1316, producer step 5570 cycles.

## D128 B4 H16/4 N8192 causal: 8 samples, 160 warpgroup-steps

| region | median cycles | share of step |
|---|---:|---:|
| V wait | 116 | 1.9% |
| scheduler barrier | 38 | 0.6% |
| PV chain issue + in-chain promotes | 1012 | 16.4% |
| PV retire | 172 | 2.8% |
| last promote + V release | 382 | 6.2% |
| K wait | 71 | 1.1% |
| QK issue + arrive | 162 | 2.6% |
| QK retire + K release | 468 | 7.6% |
| mask + descale | 294 | 4.8% |
| grouped max / scale | 657 | 10.6% |
| O rescale | 280 | 4.5% |
| grouped online softmax | 1742 | 28.2% |
| P conversion | 258 | 4.2% |
| v-scale load + step end | 89 | 1.4% |
| step (warpgroup) | 6174 | |

WG1 - WG0 step-start offset: 1910 cycles; share of each warpgroup's vector regions (mask .. P conversion) inside the other warpgroup's outstanding PV / QK windows: 25%.
Producer (thread 0): load_V bracket 200, load_K bracket 4010, V transpose bracket 1551, producer step 6300 cycles.

## D256 B4 H8/8 N8192 causal: 8 samples, 160 warpgroup-steps

| region | median cycles | share of step |
|---|---:|---:|
| V wait | 112 | 1.4% |
| scheduler barrier | 37 | 0.5% |
| PV chain issue + in-chain promotes | 2086 | 26.2% |
| PV retire | 127 | 1.6% |
| last promote + V release | 1704 | 21.4% |
| K wait | 66 | 0.8% |
| QK issue + arrive | 159 | 2.0% |
| QK retire + K release | 483 | 6.1% |
| mask + descale | 433 | 5.4% |
| grouped max / scale | 421 | 5.3% |
| O rescale | 1130 | 14.2% |
| grouped online softmax | 737 | 9.3% |
| P conversion | 72 | 0.9% |
| v-scale load + step end | 266 | 3.3% |
| step (warpgroup) | 7960 | |

WG1 - WG0 step-start offset: 4176 cycles; share of each warpgroup's vector regions (mask .. P conversion) inside the other warpgroup's outstanding PV / QK windows: 24%.
Producer (thread 0): load_V bracket 290, load_K bracket 5002, V transpose bracket 2236, producer step 8106 cycles.

## D256 B4 H8/8 N8192 full: 8 samples, 160 warpgroup-steps

| region | median cycles | share of step |
|---|---:|---:|
| V wait | 75 | 1.4% |
| scheduler barrier | 42 | 0.8% |
| PV chain issue + in-chain promotes | 1740 | 33.0% |
| PV retire | 97 | 1.8% |
| last promote + V release | 134 | 2.5% |
| K wait | 69 | 1.3% |
| QK issue + arrive | 156 | 3.0% |
| QK retire + K release | 452 | 8.6% |
| mask + descale | 209 | 4.0% |
| grouped max / scale | 492 | 9.3% |
| O rescale | 195 | 3.7% |
| grouped online softmax | 1034 | 19.7% |
| P conversion | 139 | 2.6% |
| v-scale load + step end | 142 | 2.7% |
| step (warpgroup) | 5264 | |

WG1 - WG0 step-start offset: 2384 cycles; share of each warpgroup's vector regions (mask .. P conversion) inside the other warpgroup's outstanding PV / QK windows: 45%.
Producer (thread 0): load_V bracket 309, load_K bracket 2752, V transpose bracket 1790, producer step 5282 cycles.

## D256 B4 H8/2 N8192 causal: 8 samples, 160 warpgroup-steps

| region | median cycles | share of step |
|---|---:|---:|
| V wait | 120 | 1.6% |
| scheduler barrier | 39 | 0.5% |
| PV chain issue + in-chain promotes | 1863 | 24.3% |
| PV retire | 125 | 1.6% |
| last promote + V release | 1534 | 20.0% |
| K wait | 68 | 0.9% |
| QK issue + arrive | 167 | 2.2% |
| QK retire + K release | 478 | 6.2% |
| mask + descale | 425 | 5.5% |
| grouped max / scale | 434 | 5.7% |
| O rescale | 1112 | 14.5% |
| grouped online softmax | 736 | 9.6% |
| P conversion | 76 | 1.0% |
| v-scale load + step end | 270 | 3.5% |
| step (warpgroup) | 7676 | |

WG1 - WG0 step-start offset: 3958 cycles; share of each warpgroup's vector regions (mask .. P conversion) inside the other warpgroup's outstanding PV / QK windows: 25%.
Producer (thread 0): load_V bracket 310, load_K bracket 4590, V transpose bracket 2223, producer step 7724 cycles.

