# Original baseline vs the current accepted build (Hao, round 17)

`ab.py run --a .local/frozen/base (main 3d4cd5b) --b <accepted root> --grid paper --reps 3`, the gate's method (e2e fwd
and fwd+bwd medians, producers included): A_median.json = original, B_median.json = accepted (the files hold the latest row). Rerun after every
accepted round.

| when | accepted build | e2e fwd | e2e fwd+bwd | job |
|---|---|---:|---:|---|
| round 17 (before C2) | r14_h3 (d07cdcd) | -1.93% | -3.42% | 21172405 |
| round 17 (C2 accepted) | r17_c2 (c563139) |  -1.90%  |  -3.99%  | 21172815 |

Round 17 per kernel (sum over the four shapes): pass 2 -9.85%, pass 1 -4.24%, glue -36.6%, forward kernel +1.29%,
pass 0 +1.48% (forward / pass-0 sources unchanged since the baseline: a build / codegen difference, under investigation).
