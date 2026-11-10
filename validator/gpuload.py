"""Concurrent GPU load to perturb timing: matmul + memcpy loop until killed."""
try:
    from gpu_share_hook import share_done as _d; _d()
except ImportError:
    pass
import torch
a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16); b = torch.empty(1 << 30, dtype=torch.uint8, device="cuda")
while True:
    for _ in range(20):
        a = (a @ a).clamp_(-1, 1)
    b.random_(0, 256); torch.cuda.synchronize()
