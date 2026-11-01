# Pass-1 sparse timeline (lb % 17 == 0, lb / 17 < 8; first 12 m-block steps of the first tile)

Build: /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/trace4/delta_matching/_C.so (sha256 321bf8318b8a132a), route {'groups': 'off', 'streams': 1, 'ws_limit': 8589934592.0}.

## D128 B4 H16/16 N8192 causal (matched): 8 sampled CTAs

80 interior steps; medians in cycles (SM clock):

| quantity | median |
|---|---:|
| step_wg0 | 3849 |
| step_wg1 | 3866 |
| offset | -61 |
| region_Q_wait | 82 |
| region_dO_wait | 80 |
| region_dOt_wait | 62 |
| region_Qt_wait | 85 |
| region_S_retire | 9 |
| region_dP_retire | 9 |
| region_dV_retire_release | 20 |
| region_dK_retire_release_state | 253 |
| region_psi_barrier | 7 |
| vec_regions_wg0 | 2471 |
| vec_regions_wg0_in_wg1_outstanding | 708 |
| vec_regions_wg1 | 2574 |
| vec_regions_wg1_in_wg0_outstanding | 1078 |
| p_region_dO_acquire | 878 |
| p_region_dOt_acquire_bracket | 1613 |
| p_region_Q_acquire_bracket | 148 |
| p_region_Qt_acquire_bracket | 142 |
| q_issue_lead | 3978 |

Share of WG0's instrumented vector regions inside WG1's software-outstanding wgmma windows (issue to observed retire / release): 27%; WG1 in WG0's: 40%.

## D128 B4 H16/16 N8192 full (stale): 8 sampled CTAs

80 interior steps; medians in cycles (SM clock):

| quantity | median |
|---|---:|
| step_wg0 | 3938 |
| step_wg1 | 3944 |
| offset | -49 |
| region_Q_wait | 65 |
| region_dO_wait | 84 |
| region_dOt_wait | 64 |
| region_Qt_wait | 82 |
| region_S_retire | 26 |
| region_dP_retire | 9 |
| region_dV_retire_release | 23 |
| region_dK_retire_release_state | 255 |
| region_psi_barrier | 7 |
| vec_regions_wg0 | 2398 |
| vec_regions_wg0_in_wg1_outstanding | 534 |
| vec_regions_wg1 | 2608 |
| vec_regions_wg1_in_wg0_outstanding | 1022 |
| p_region_dO_acquire | 986 |
| p_region_dOt_acquire_bracket | 1674 |
| p_region_Q_acquire_bracket | 152 |
| p_region_Qt_acquire_bracket | 144 |
| q_issue_lead | 4056 |

Share of WG0's instrumented vector regions inside WG1's software-outstanding wgmma windows (issue to observed retire / release): 22%; WG1 in WG0's: 37%.

## D256 B4 H8/8 N8192 causal (matched): 8 sampled CTAs

80 interior steps; medians in cycles (SM clock):

| quantity | median |
|---|---:|
| step_wg0 | 3801 |
| step_wg1 | 3801 |
| offset | -61 |
| region_Q_wait | 70 |
| region_dO_wait | 70 |
| region_dOt_wait | 96 |
| region_Qt_wait | 82 |
| region_S_retire | 29 |
| region_dP_retire | 23 |
| region_dV_retire_release | 27 |
| region_dK_retire_release_state | 221 |
| region_psi_barrier | 22 |
| vec_regions_wg0 | 1914 |
| vec_regions_wg0_in_wg1_outstanding | 0 |
| vec_regions_wg1 | 2174 |
| vec_regions_wg1_in_wg0_outstanding | 664 |
| p_region_dO_acquire | 127 |
| p_region_dOt_acquire_bracket | 2860 |
| p_region_Q_acquire_bracket | 186 |
| p_region_Qt_acquire_bracket | 87 |
| q_issue_lead | 4076 |

Share of WG0's instrumented vector regions inside WG1's software-outstanding wgmma windows (issue to observed retire / release): 4%; WG1 in WG0's: 27%.

## D256 B4 H8/8 N8192 full (stale): 8 sampled CTAs

80 interior steps; medians in cycles (SM clock):

| quantity | median |
|---|---:|
| step_wg0 | 3846 |
| step_wg1 | 3844 |
| offset | -44 |
| region_Q_wait | 78 |
| region_dO_wait | 68 |
| region_dOt_wait | 66 |
| region_Qt_wait | 82 |
| region_S_retire | 29 |
| region_dP_retire | 22 |
| region_dV_retire_release | 27 |
| region_dK_retire_release_state | 223 |
| region_psi_barrier | 22 |
| vec_regions_wg0 | 2001 |
| vec_regions_wg0_in_wg1_outstanding | 320 |
| vec_regions_wg1 | 1984 |
| vec_regions_wg1_in_wg0_outstanding | 614 |
| p_region_dO_acquire | 120 |
| p_region_dOt_acquire_bracket | 2889 |
| p_region_Q_acquire_bracket | 151 |
| p_region_Qt_acquire_bracket | 92 |
| q_issue_lead | 4056 |

Share of WG0's instrumented vector regions inside WG1's software-outstanding wgmma windows (issue to observed retire / release): 14%; WG1 in WG0's: 27%.

