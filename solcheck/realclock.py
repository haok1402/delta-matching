"""SM clock / power of the real kernels under sustained load (nvidia-smi sampled by the caller; this script prints
phase markers with wall-clock times).  Loops the accepted forward alone, then the backward alone (retain_graph), at one
acceptance shape, for --secs seconds each.

    python solcheck/realclock.py --secs 8 [--B 1 --N 65536 --H 32 --Hk 4]
"""
import argparse, json, time
import torch
import delta_matching as dm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=8); ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--N", type=int, default=65536); ap.add_argument("--H", type=int, default=32)
    ap.add_argument("--Hk", type=int, default=4)
    a = ap.parse_args()
    torch.manual_seed(0)
    q = torch.randn(a.B, a.N, a.H, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(a.B, a.N, a.Hk, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(a.B, a.N, a.Hk, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    do = torch.randn(a.B, a.N, a.H, 128, device="cuda", dtype=torch.bfloat16)
    for phase in ("fwd", "bwd"):
        o = dm.dm_attn_func(q, k, v, causal=True, mode="matched")
        torch.cuda.synchronize()
        n, t0 = 0, time.time()
        ev0 = torch.cuda.Event(enable_timing=True); ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        while time.time() - t0 < a.secs:
            for _ in range(4):
                if phase == "fwd":
                    dm.dm_attn_func(q, k, v, causal=True, mode="matched")
                else:
                    torch.autograd.grad(o, (q, k, v), do, retain_graph=True)
                n += 1
            torch.cuda.synchronize()
        ev1.record(); ev1.synchronize()
        t1 = time.time()
        print(json.dumps({"phase": phase, "t0": t0, "t1": t1, "calls": n, "ms_per_call": ev0.elapsed_time(ev1) / n,
                          "shape": [a.B, a.N, a.H, a.Hk]}), flush=True)
        time.sleep(2)


if __name__ == "__main__":
    main()
