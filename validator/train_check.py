#!/usr/bin/env python
"""Slow tier: a short training run of a small causal LM with the candidate's attention, against bf16 attention.

  python validator/train_check.py <candidate spec> [--steps 400] [--device cuda:0] [--arms bf16,matched,stale,emul]

Model: a pre-norm GPT (6 layers, d_model 512, 4 heads of dim 128, GQA off, RoPE, qk-norm off so the logit range grows as
it would in a real model), byte-level, sequence 2048, batch 8, AdamW lr 1e-3 cosine, bf16 autocast.  Data: the bytes of
the Python standard library sources on this machine (offline, deterministic order by seed).  Every arm starts from the
same weights and sees the same batches; only the attention function differs:
  bf16     torch SDPA (the control)
  matched  candidate, mode="matched";   stale  candidate, mode="stale"
  emul     the torch emulation of the candidate's NUMERICS preset (matched), as the expected fp8 curve (slow, optional)
PASS when no arm has a non-finite loss and the matched arm's mean loss over the last 20% of steps is within
max(2%, 0.03 nats) of bf16's (the fp8 noise budget of a correct fp8 attention at this scale; a wrong delta or a broken
gradient drifts well past that within a few hundred steps).  The stale - matched gap is reported, not judged.
"""
import argparse
import glob
import importlib.util
import json
import math
import os
import sys
import sysconfig
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))


def load_spec(spec):
    target, _, func = spec.partition(":")
    func = func or "attn"
    if target.endswith(".py"):
        sys.path.insert(0, os.path.dirname(os.path.abspath(target)))
        ms = importlib.util.spec_from_file_location("dm_candidate", target)
        mod = importlib.util.module_from_spec(ms)
        sys.modules["dm_candidate"] = mod
        ms.loader.exec_module(mod)
    else:
        mod = importlib.import_module(target)
    return mod, getattr(mod, func)


def corpus(device):
    root = sysconfig.get_paths()["stdlib"]
    files = sorted(glob.glob(os.path.join(root, "**", "*.py"), recursive=True))
    data = b"".join(open(f, "rb").read() for f in files)
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).to(device)


def rope(x, base=10000.0):
    B, N, H, D = x.shape
    f = base ** (-torch.arange(0, D, 2, device=x.device, dtype=torch.float32) / D)
    t = torch.arange(N, device=x.device, dtype=torch.float32)[:, None] * f[None]
    c, s = t.cos()[None, :, None], t.sin()[None, :, None]
    x1, x2 = x.float()[..., ::2], x.float()[..., 1::2]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2).to(x.dtype)


class Block(nn.Module):
    def __init__(self, d, h, attn):
        super().__init__()
        self.h, self.attn = h, attn
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d, bias=False), nn.Linear(d, d, bias=False)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d, bias=False), nn.GELU(), nn.Linear(4 * d, d, bias=False))

    def forward(self, x):
        B, N, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, N, 3, self.h, d // self.h).unbind(2)
        q, k = rope(q), rope(k)
        a = self.attn(q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16))
        x = x + self.o(a.reshape(B, N, d))
        return x + self.ff(self.n2(x))


class GPT(nn.Module):
    def __init__(self, attn, L=6, d=512, h=4, V=256):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.blocks = nn.ModuleList([Block(d, h, attn) for _ in range(L)])
        self.nf, self.head = nn.LayerNorm(d), nn.Linear(d, V, bias=False)

    def forward(self, x):
        h = self.emb(x)
        for b in self.blocks:
            h = b(h)
        return self.head(self.nf(h))


def sdpa(q, k, v):
    return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                          is_causal=True).transpose(1, 2)


def train(attn, data, a, dev):
    torch.manual_seed(a.seed)
    model = GPT(attn).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)
    g = torch.Generator(device="cpu").manual_seed(a.seed + 1)
    losses = []
    for step in range(a.steps):
        lr = a.lr * min(1.0, (step + 1) / 50) * 0.5 * (1 + math.cos(math.pi * step / a.steps))
        for pg in opt.param_groups:
            pg["lr"] = lr
        idx = torch.randint(0, data.numel() - a.seq - 1, (a.batch,), generator=g)
        x = torch.stack([data[i:i + a.seq] for i in idx.tolist()]).long()
        y = torch.stack([data[i + 1:i + a.seq + 1] for i in idx.tolist()]).long()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(model(x).float().view(-1, 256), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if not math.isfinite(losses[-1]):
            break
    return losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--arms", default="bf16,matched,stale")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    dev = torch.device(a.device)
    torch.cuda.set_device(dev)
    data = corpus(dev)
    mod, fn = load_spec(a.candidate) if a.candidate != "none" else (None, None)
    arms = {"bf16": sdpa}
    if fn is not None:
        arms["matched"] = lambda q, k, v: fn(q, k, v, causal=True, mode="matched")
        arms["stale"] = lambda q, k, v: fn(q, k, v, causal=True, mode="stale")
    if "emul" in a.arms:
        ms = importlib.util.spec_from_file_location("dm_ref_train", os.path.join(os.path.dirname(HERE), "delta_matching",
                                                                                  "reference.py"))
        R = importlib.util.module_from_spec(ms); ms.loader.exec_module(R)
        preset = getattr(mod, "NUMERICS", "sm90") if mod else "sm90"

        class _E(torch.autograd.Function):
            @staticmethod
            def forward(ctx, q, k, v):
                ctx.save_for_backward(q, k, v)
                return R.dm_reference(q, k, v, torch.zeros_like(q), True, "stale", preset)["O"].contiguous()

            @staticmethod
            def backward(ctx, do):
                q, k, v = ctx.saved_tensors
                r = R.dm_reference(q, k, v, do.contiguous(), True, "matched", preset)
                return r["dQ"].to(q.dtype), r["dK"].to(k.dtype), r["dV"].to(v.dtype)
        arms["emul"] = _E.apply
    res = {}
    for name in a.arms.split(","):
        if name not in arms:
            continue
        t = time.time()
        res[name] = train(arms[name], data, a, dev)
        tail = res[name][-max(1, a.steps // 5):]
        print(f"{name:8s} final-20% mean loss {sum(tail) / len(tail):.4f}  ({time.time() - t:.0f} s)", flush=True)
    tailm = {n: sum(l[-max(1, a.steps // 5):]) / max(1, len(l[-max(1, a.steps // 5):])) for n, l in res.items()}
    finite = all(all(math.isfinite(x) for x in l) and len(l) == a.steps for l in res.values())
    ok = finite
    if "matched" in tailm:
        budget = max(0.02 * tailm["bf16"], 0.03)
        ok &= tailm["matched"] - tailm["bf16"] <= budget
        print(f"matched - bf16 = {tailm['matched'] - tailm['bf16']:+.4f} (budget {budget:.4f})")
    if "stale" in tailm and "matched" in tailm:
        print(f"stale - matched = {tailm['stale'] - tailm['matched']:+.4f} (reported, not judged)")
    out = {"pass": ok, "tail_mean": tailm, "steps": a.steps, "losses": res}
    if a.json:
        json.dump(out, open(a.json, "w"))
    print("SUMMARY_JSON " + json.dumps({"pass": ok, "tail_mean": tailm, "steps": a.steps}))
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
