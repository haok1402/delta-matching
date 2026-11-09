"""FA4 bwd per-call time distribution at H16/16 N32k causal: is the bimodality from the flush, the spin, or inherent?"""
import statistics, time, torch
try:
    from gpupool_hook import gpupool_ready, gpupool_done
except ImportError:
    gpupool_ready = gpupool_done = lambda: None
from flash_attn.cute import flash_attn_func as fa4
import pynvml
pynvml.nvmlInit(); h = pynvml.nvmlDeviceGetHandleByIndex(0)
dev = "cuda"; B, N, H, D = 1, 32768, 16, 128
q, k, v = (torch.randn(B, N, H, D, device=dev, dtype=torch.bfloat16).requires_grad_() for _ in range(3))
do = torch.randn(B, N, H, D, device=dev, dtype=torch.bfloat16)
flush = torch.empty(1 << 30, dtype=torch.uint8, device=dev)
for _ in range(5):
    o = fa4(q, k, v, causal=True)[0]; torch.autograd.grad(o, (q, k, v), do)
torch.cuda.synchronize(); gpupool_ready()
for fl in (True, False):
    for spin in (True, False):
        ts, clk = [], []
        for i in range(200):
            o = fa4(q, k, v, causal=True)[0]
            if fl: flush.zero_()
            if spin: torch.cuda._sleep(int(2e6))
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record(); torch.autograd.grad(o, (q, k, v), do); e1.record(); e1.synchronize()
            ts.append(e0.elapsed_time(e1)); clk.append(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM))
        s = sorted(ts)
        fast = [t for t in ts if t < (s[0] + s[-1]) / 2]
        print(f"flush={fl} spin={spin}: median {statistics.median(ts):.3f} p10 {s[20]:.3f} p90 {s[180]:.3f} min {s[0]:.3f} "
              f"fast-share {len(fast)/len(ts):.2f} | clk median {statistics.median(clk)} min {min(clk)} max {max(clk)}", flush=True)
        # correlation of time with clock
        hi = [t for t, c in zip(ts, clk) if c >= statistics.median(clk)]; lo = [t for t, c in zip(ts, clk) if c < statistics.median(clk)]
        if hi and lo: print(f"   time at clk>=median {statistics.median(hi):.3f}  clk<median {statistics.median(lo):.3f}")
gpupool_done()
