import torch, delta_matching as dm
for causal in (True, False):
    q = torch.zeros(1, 1024, 4, 128, device="cuda", dtype=torch.bfloat16)
    o = dm.dm_attn_func(q, q.clone(), q.clone(), causal=causal, mode="stale")
    bad = ~torch.isfinite(o.float())
    print("causal", causal, "O nonfinite", int(bad.sum()), "of", o.numel(), "first at", bad.nonzero()[:1].tolist())
from delta_matching import producers as P
print([n for n in dir(P) if "quant" in n])
