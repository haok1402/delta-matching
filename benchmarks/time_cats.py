"""Per-category device ms (CUPTI) of fwd+bwd for a fixed config set, one build per process (profile_pass1.select_build).

    python benchmarks/time_cats.py --root <build root> --out t.json [--cats pass2,pass1]
"""
import argparse, json, os, sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kernels as K  # noqa: E402
import profile_pass1 as PP  # noqa: E402

# (D, B, H, Hk, N, attn, mode): causal matched / causal stale / full stale, MHA and GQA, both head dims
CONFIGS = [(128, 4, 16, 16, 8192, "causal", "matched"), (128, 4, 16, 4, 8192, "causal", "matched"),
           (128, 4, 16, 16, 8192, "full", "stale"), (128, 4, 16, 16, 8192, "causal", "stale"),
           (256, 4, 8, 8, 8192, "causal", "matched"), (256, 4, 8, 2, 8192, "causal", "matched"),
           (256, 4, 8, 8, 8192, "full", "stale"), (256, 4, 8, 8, 8192, "causal", "stale")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--cats", default="pass2"); ap.add_argument("--iters", type=int, default=10)
    a = ap.parse_args()
    meta = PP.select_build(a.root)
    import delta_matching as dm
    cats = a.cats.split(",")
    recs = []
    for D, B, H, Hk, N, attn, mode in CONFIGS:
        g = torch.Generator(device="cuda").manual_seed(0)
        q = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16, generator=g).requires_grad_()
        k = torch.randn(B, N, Hk, D, device="cuda", dtype=torch.bfloat16, generator=g).requires_grad_()
        v = torch.randn(B, N, Hk, D, device="cuda", dtype=torch.bfloat16, generator=g).requires_grad_()
        do = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16, generator=g)

        def step():
            o = dm.dm_attn_func(q, k, v, causal=attn == "causal", mode=mode)
            o.backward(do)
            q.grad = k.grad = v.grad = None
        names = K.kernel_times(step, a.iters)
        ms = {c: sum(t for n, t in names.items() if K.classify(n) == c) for c in cats}
        if any(ms[c] <= 0 for c in cats):
            raise SystemExit(f"missing category timings {ms} for {(D, B, H, Hk, N, attn, mode)}")
        recs.append({"cfg": [D, B, H, Hk, N, attn, mode], "ms": ms})
        print(f"{recs[-1]['cfg']}: " + " ".join(f"{c} {v:.3f} ms" for c, v in ms.items()), flush=True)
    with open(a.out, "w") as f:
        json.dump({"meta": meta, "records": recs}, f, indent=1)


if __name__ == "__main__":
    main()
