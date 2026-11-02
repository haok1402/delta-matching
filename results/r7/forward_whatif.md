# Forward what-if ablations (job 21163152; timing only except knob 4)

Forward kernel ms per config (B4 N8192), median of 3 rotating-order repetitions, change vs knob 0.

| variant | D128 H16/16 causal | D128 H16/4 causal | D128 H16/16 full | D256 H8/8 causal | D256 H8/2 causal | D256 H8/8 full |
|---|---:|---:|---:|---:|---:|---:|
| accepted DO2 (control) | 1.562 (-0.1%, n=3, spread 0.0%) | 1.538 (-0.2%, n=3, spread 0.7%) | 2.908 (-0.2%, n=3, spread 0.1%) | 1.477 (+0.2%, n=3, spread 1.0%) | 1.474 (+0.3%, n=3, spread 0.8%) | 2.884 (+0.3%, n=3, spread 0.7%) |
| what-if build, knob 0 (= DO2 source) | 1.563 (+0.0%, n=3, spread 0.1%) | 1.541 (+0.0%, n=3, spread 1.0%) | 2.913 (+0.0%, n=3, spread 0.1%) | 1.474 (+0.0%, n=3, spread 1.0%) | 1.469 (+0.0%, n=3, spread 0.1%) | 2.877 (+0.0%, n=3, spread 1.2%) |
| 1: no S descale | 1.491 (-4.6%, n=3, spread 0.0%) | 1.490 (-3.4%, n=3, spread 0.1%) | 2.649 (-9.0%, n=3, spread 0.0%) | 1.432 (-2.9%, n=3, spread 0.3%) | 1.419 (-3.4%, n=3, spread 0.8%) | 2.786 (-3.2%, n=3, spread 0.3%) |
| 2: no rho rescale in the PV chain | 1.437 (-8.1%, n=3, spread 0.1%) | 1.417 (-8.1%, n=3, spread 0.8%) | 2.688 (-7.7%, n=3, spread 0.1%) | 1.181 (-19.9%, n=3, spread 1.0%) | 1.178 (-19.8%, n=3, spread 0.2%) | 2.297 (-20.2%, n=3, spread 1.1%) |
| 3: no rho, no inner PV waits | 1.492 (-4.6%, n=3, spread 0.0%) | 1.487 (-3.6%, n=3, spread 0.9%) | 2.856 (-1.9%, n=3, spread 0.2%) | 1.163 (-21.1%, n=3, spread 1.1%) | 1.141 (-22.3%, n=3, spread 1.0%) | 2.185 (-24.0%, n=3, spread 1.1%) |
| 4: k-scale prefetch at step start (bitwise) | 1.546 (-1.1%, n=3, spread 0.0%) | 1.529 (-0.8%, n=3, spread 0.1%) | 2.920 (+0.2%, n=3, spread 0.1%) | 1.428 (-3.1%, n=3, spread 0.3%) | 1.422 (-3.2%, n=3, spread 0.4%) | 2.839 (-1.3%, n=3, spread 0.3%) |
