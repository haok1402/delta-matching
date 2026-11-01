# Pass-1 sparse timeline (lb % 17 == 0, lb / 17 < 8; first 12 m-block steps of the first tile)

Build: /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/trace4/delta_matching/_C.so (sha256 321bf8318b8a132a), route {'groups': 'off', 'streams': 1, 'ws_limit': 8589934592.0}.

## D128 B4 H16/16 N8192 causal (matched): 8 sampled CTAs

80 interior steps; medians in cycles (SM clock):

| quantity | median |
|---|---:|
| step_wg0 | 3849 |
| step_wg1 | 3866 |
| offset | -61 |
| wait_Q | 82 |
| wait_dO | 80 |
| wait_dOt | 62 |
| wait_Qt | 85 |
| wait_S | 9 |
| wait_dP | 9 |
| wait_dV | 20 |
| wait_dK | 253 |
| psi_barrier | 7 |
| vec_wg0 | 2471 |
| vec_wg0_under_wg1_mma | 708 |
| vec_wg1 | 2574 |
| vec_wg1_under_wg0_mma | 1078 |
| p_acq_dO | 878 |
| p_acq_dOt | 1613 |
| p_acq_Q | 148 |
| p_acq_Qt | 142 |
| q_ready_lead | 3978 |

Share of WG0's vector work while WG1 has wgmma in flight: 27%; WG1 under WG0's wgmma: 40%.

## D128 B4 H16/16 N8192 full (stale): 8 sampled CTAs

80 interior steps; medians in cycles (SM clock):

| quantity | median |
|---|---:|
| step_wg0 | 3938 |
| step_wg1 | 3944 |
| offset | -49 |
| wait_Q | 65 |
| wait_dO | 84 |
| wait_dOt | 64 |
| wait_Qt | 82 |
| wait_S | 26 |
| wait_dP | 9 |
| wait_dV | 23 |
| wait_dK | 255 |
| psi_barrier | 7 |
| vec_wg0 | 2398 |
| vec_wg0_under_wg1_mma | 534 |
| vec_wg1 | 2608 |
| vec_wg1_under_wg0_mma | 1022 |
| p_acq_dO | 986 |
| p_acq_dOt | 1674 |
| p_acq_Q | 152 |
| p_acq_Qt | 144 |
| q_ready_lead | 4056 |

Share of WG0's vector work while WG1 has wgmma in flight: 22%; WG1 under WG0's wgmma: 37%.

## D256 B4 H8/8 N8192 causal (matched): 8 sampled CTAs

80 interior steps; medians in cycles (SM clock):

| quantity | median |
|---|---:|
| step_wg0 | 3801 |
| step_wg1 | 3801 |
| offset | -61 |
| wait_Q | 70 |
| wait_dO | 70 |
| wait_dOt | 96 |
| wait_Qt | 82 |
| wait_S | 29 |
| wait_dP | 23 |
| wait_dV | 27 |
| wait_dK | 221 |
| psi_barrier | 22 |
| vec_wg0 | 1914 |
| vec_wg0_under_wg1_mma | 0 |
| vec_wg1 | 2174 |
| vec_wg1_under_wg0_mma | 664 |
| p_acq_dO | 127 |
| p_acq_dOt | 2860 |
| p_acq_Q | 186 |
| p_acq_Qt | 87 |
| q_ready_lead | 4076 |

Share of WG0's vector work while WG1 has wgmma in flight: 4%; WG1 under WG0's wgmma: 27%.

## D256 B4 H8/8 N8192 full (stale): 8 sampled CTAs

80 interior steps; medians in cycles (SM clock):

| quantity | median |
|---|---:|
| step_wg0 | 3846 |
| step_wg1 | 3844 |
| offset | -44 |
| wait_Q | 78 |
| wait_dO | 68 |
| wait_dOt | 66 |
| wait_Qt | 82 |
| wait_S | 29 |
| wait_dP | 22 |
| wait_dV | 27 |
| wait_dK | 223 |
| psi_barrier | 22 |
| vec_wg0 | 2001 |
| vec_wg0_under_wg1_mma | 320 |
| vec_wg1 | 1984 |
| vec_wg1_under_wg0_mma | 614 |
| p_acq_dO | 120 |
| p_acq_dOt | 2889 |
| p_acq_Q | 151 |
| p_acq_Qt | 92 |
| q_ready_lead | 4056 |

Share of WG0's vector work while WG1 has wgmma in flight: 14%; WG1 under WG0's wgmma: 27%.

