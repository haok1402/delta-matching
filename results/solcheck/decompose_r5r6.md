| kernel | best variant | % of sol4 | C_tensor | C_alu alone (x / spec_noT) | C_best | kappa | f_eff MHz | 8192 f_eff / calib | C_tensor / C_best | ALU mix alone: IPC / MUFU lanes / bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| fwd | v_fwd_spec | 69.9% | 1536 | 1974 / 2085 | 2909 | 0.54 | 1789 | 1.347 | 0.528 | 3.53 / 14.6 / mufu 94% |
| pass1 | v_pass1_ilv | 62.0% | 1024 | 1377 / 1529 | 2163 | 0.62 | 1780 | 1.340 | 0.473 | 3.56 / 6.5 / issue 92% |
| pass0 | v_pass0_ilv | 74.5% | 1024 | 1131 / 1192 | 1724 | 0.52 | 1691 | 1.274 | 0.594 | 3.52 / 14.5 / mufu 94% |
