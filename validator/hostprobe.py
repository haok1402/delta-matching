"""Host fast-path probe (record-and-replay safety).  python hostprobe.py cand.py out.pt
Scenarios (det mode, matched + stale); every output saved; compare two candidates' files bitwise with --cmp a.pt b.pt.
 S1 repeat: same inputs x3 (call 2/3 replay).      S2 new contents, same shape (replay with different data).
 S3 keep outputs of call A, run call B (different data), re-save A (output buffers must not be reused/aliased).
 S4 side stream: inputs produced on stream s (no sync), call inside torch.cuda.stream(s), consume on s.
 S5 two streams concurrently, different data, same shape.   S6 non-contiguous q/k/v views (packed qkv), same shape.
 S7 back-to-back calls without sync, outputs of all kept.   S8 shape change and back (key switch).
 S9 odd-block shapes (9 / 13 / 39 / 11 blocks), causal + non-causal, matched + stale, 3 calls each."""
try:
    from gpupool_hook import gpupool_done as _d; _d()
except ImportError:
    pass
import sys, importlib.util, torch
if sys.argv[1] == "--cmp":
    a, b = torch.load(sys.argv[2]), torch.load(sys.argv[3]); bad = 0
    for k in a:
        eq = [torch.equal(x, y) for x, y in zip(a[k], b[k])]
        if not all(eq):
            bad += 1
        print(k, "OK" if all(eq) else f"DIFF {eq}")
    print("RESULT", "PASS" if bad == 0 and a.keys() == b.keys() else f"FAIL {bad}")
    sys.exit(0 if bad == 0 else 1)
s = importlib.util.spec_from_file_location("c", sys.argv[1]); C = importlib.util.module_from_spec(s); s.loader.exec_module(C)
try:
    from gpupool_hook import gpupool_done as _d; _d()
except ImportError:
    pass
OUT = {}
def inputs(seed, B=2, N=2048, H=16, Hk=4):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return [torch.randn(B, N, h, 128, device="cuda", generator=g).bfloat16() for h in (H, Hk, Hk, H)]
def fb(q, k, v, do, causal=True, mode="matched"):
    l = [x.detach().clone().requires_grad_() for x in (q, k, v)]
    o = C.attn(*l, causal=causal, mode=mode); o.backward(do)
    return [o.detach(), l[0].grad, l[1].grad, l[2].grad]
cpu = lambda ts: [t.cpu() for t in ts]  # noqa: E731
for mode in ("matched", "stale"):
    for causal in (True, False):
        t = f"{mode}_{'c' if causal else 'nc'}"
        x = inputs(1)
        for i in range(3):
            OUT[f"S1_{t}_{i}"] = cpu(fb(*x, causal, mode)); torch.cuda.synchronize()
        OUT[f"S2_{t}"] = cpu(fb(*inputs(2), causal, mode)); torch.cuda.synchronize()
        A = fb(*inputs(3), causal, mode); B_ = fb(*inputs(4), causal, mode); torch.cuda.synchronize()
        OUT[f"S3_{t}_A"], OUT[f"S3_{t}_B"] = cpu(A), cpu(B_)
        st = torch.cuda.Stream()
        with torch.cuda.stream(st):
            xs = inputs(5)                      # produced on st, no sync
            r = fb(*xs, causal, mode)
            r = [y * 1 for y in r]              # consumed on st
        torch.cuda.synchronize(); OUT[f"S4_{t}"] = cpu(r)
        s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
        x6, x7 = inputs(6), inputs(7); torch.cuda.synchronize()
        with torch.cuda.stream(s1):
            r1 = fb(*x6, causal, mode)
        with torch.cuda.stream(s2):
            r2 = fb(*x7, causal, mode)
        torch.cuda.synchronize(); OUT[f"S5_{t}_1"], OUT[f"S5_{t}_2"] = cpu(r1), cpu(r2)
        q, k, v, do = inputs(8)
        qkv = torch.cat([q, k, v], dim=2); qv, kv, vv = qkv.split([q.shape[2], k.shape[2], v.shape[2]], dim=2)
        l = [qv.detach().requires_grad_(), kv.detach().requires_grad_(), vv.detach().requires_grad_()]
        o = C.attn(qv, kv, vv, causal=causal, mode=mode)
        g = torch.autograd.grad(o, (qv,), do, allow_unused=True) if False else None
        OUT[f"S6_{t}"] = cpu([o.detach()]); torch.cuda.synchronize()
        rs = [fb(*inputs(10 + j), causal, mode) for j in range(4)]; torch.cuda.synchronize()
        for j, r in enumerate(rs):
            OUT[f"S7_{t}_{j}"] = cpu(r)
        OUT[f"S8_{t}_small"] = cpu(fb(*inputs(20, B=1, N=384, H=4, Hk=1), causal, mode))
        OUT[f"S8_{t}_back"] = cpu(fb(*inputs(1), causal, mode)); torch.cuda.synchronize()
# S9: odd numbers of 128-row blocks (abf6f87 replay bug: 1152 = 9, 1664 = 13, 4992 = 39 blocks), 3 calls each
for (B, N, H, Hk) in ((1, 1152, 4, 4), (3, 1664, 8, 4), (1, 4992, 32, 8), (2, 1408, 8, 2)):
    for mode in ("matched", "stale"):
        for causal in (True, False):
            x = inputs(30 + N, B=B, N=N, H=H, Hk=Hk)
            for i in range(3):
                OUT[f"S9_B{B}_N{N}_H{H}_{Hk}_{mode}_{int(causal)}_{i}"] = cpu(fb(*x, causal, mode)); torch.cuda.synchronize()
torch.save(OUT, sys.argv[2]); print("saved", len(OUT))
