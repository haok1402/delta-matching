#!/usr/bin/env python
"""Long training comparison of attention arms on one GPU each: bf16 attention vs fp8 Delta-Matching (matched / stale).

  python validator/longtrain.py --arm bf16|matched|stale --cand /abs/cand.py --out DIR [--seed 0] [--steps 40000]

Same model init, same data order (the batch of step t is a pure function of (seed, t)), so arms differ only in attention.
Resumable: a checkpoint (model, optimizer, step) is written atomically every --ckpt-every steps and loaded on start, so
the run continues across gpupool time limits and node handovers (run it under `gpupool run --retry`).  Per-step loss and
periodic held-out eval loss go to OUT/<arm>_s<seed>/log.jsonl; report with validator/longtrain_report.py.

Model: pre-norm byte-level GPT, RoPE, no qk-norm (the logit range grows as in a real model: the fp8 stress case),
d_model 1536, 16 layers, 12 heads of dim 128, seq 8192, batch 4, AdamW (6e-4, warmup 1000, cosine to 10%), bf16 autocast.
Corpus: validate/longtrain/corpus.bin (349 MB of source text built by build_corpus.py); last 1% held out.
"""
try:                    # gpupool: untimed; share the GPU with the filler when we are not using it
    from gpupool_hook import gpupool_done as _gp_done
    _gp_done()
except ImportError:
    pass
import argparse
import importlib.util
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

CORPUS = os.path.expanduser("~/workspace/haok1402/sm100/validate/longtrain/corpus.bin")


def load_cand(path):
    ms = importlib.util.spec_from_file_location("dm_candidate", path)
    mod = importlib.util.module_from_spec(ms)
    sys.modules["dm_candidate"] = mod
    ms.loader.exec_module(mod)
    return mod


def bf16_attn():
    try:
        from flash_attn.cute import flash_attn_func as fa4

        def f(q, k, v):
            o = fa4(q, k, v, causal=True)
            return o[0] if isinstance(o, (tuple, list)) else o
        f.name = "fa4-bf16"
        return f
    except Exception:  # noqa: BLE001
        def f(q, k, v):
            return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                                  is_causal=True).transpose(1, 2)
        f.name = "sdpa-bf16"
        return f


def rope(x, base=10000.0):
    B, N, H, D = x.shape
    f = base ** (-torch.arange(0, D, 2, device=x.device, dtype=torch.float32) / D)
    t = torch.arange(N, device=x.device, dtype=torch.float32)[:, None] * f[None]
    c, s = t.cos()[None, :, None], t.sin()[None, :, None]
    x1, x2 = x.float()[..., ::2], x.float()[..., 1::2]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2).to(torch.bfloat16)


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
        a = self.attn(rope(q), rope(k), v.to(torch.bfloat16).contiguous())
        x = x + self.o(a.reshape(B, N, d))
        return x + self.ff(self.n2(x))


class GPT(nn.Module):
    def __init__(self, attn, L, d, h, V=256):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.blocks = nn.ModuleList([Block(d, h, attn) for _ in range(L)])
        self.nf, self.head = nn.LayerNorm(d), nn.Linear(d, V, bias=False)

    def forward(self, x):
        h = self.emb(x)
        for b in self.blocks:
            h = b(h)
        return self.head(self.nf(h))


def batch(data, lo, hi, B, N, seed, step, dev):
    g = torch.Generator().manual_seed(seed * 1_000_003 + step)
    idx = torch.randint(lo, hi - N - 1, (B,), generator=g).tolist()
    x = torch.stack([data[i:i + N] for i in idx]).to(dev, non_blocking=True).long()
    y = torch.stack([data[i + 1:i + N + 1] for i in idx]).to(dev, non_blocking=True).long()
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["bf16", "matched", "stale"])
    ap.add_argument("--cand", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--seq", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--dmodel", type=int, default=1536)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--eval-every", type=int, default=500)
    a = ap.parse_args()
    dev = torch.device("cuda")
    torch.zeros(1, device=dev)
    try:
        from gpupool_hook import gpupool_done
        gpupool_done()
    except ImportError:
        pass
    run = os.path.join(a.out, f"{a.arm}_s{a.seed}")
    os.makedirs(run, exist_ok=True)
    raw = open(CORPUS, "rb").read()
    data = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    n = data.numel()
    split = int(n * 0.99)
    if a.arm == "bf16":
        attn = bf16_attn()
        attn_name = attn.name
    else:
        mod = load_cand(a.cand)
        mode = a.arm
        attn = lambda q, k, v: mod.attn(q, k, v, causal=True, mode=mode)  # noqa: E731
        attn_name = f"{os.path.basename(a.cand)}:{mode}"
    torch.manual_seed(a.seed)
    H = a.dmodel // 128
    model = GPT(attn, a.layers, a.dmodel, H).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
    step = 0
    ck = os.path.join(run, "ckpt.pt")
    if os.path.exists(ck):
        st = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        step = st["step"]
        print(f"resumed {run} at step {step}", flush=True)
    else:
        json.dump({"arm": a.arm, "attn": attn_name, "args": vars(a), "params": sum(p.numel() for p in model.parameters()),
                   "corpus_bytes": n}, open(os.path.join(run, "config.json"), "w"), indent=1)
    log = open(os.path.join(run, "log.jsonl"), "a")
    import signal
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))   # node handover: checkpoint, then exit non-zero

    def save():
        log.flush()
        tmp = ck + ".tmp"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, tmp)
        os.replace(tmp, ck)
    t0 = time.time()
    while step < a.steps:
        if stop["flag"]:
            save()
            print(f"SIGTERM: checkpointed {run} at step {step}", flush=True)
            sys.exit(143)
        lr = a.lr * min(1.0, (step + 1) / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, step / a.steps))))
        for pg in opt.param_groups:
            pg["lr"] = lr
        x, y = batch(data, 0, split, a.batch, a.seq, a.seed, step, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(model(x).float().view(-1, 256), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        rec = {"step": step, "loss": loss.item(), "gnorm": gn.item(), "lr": lr, "t": round(time.time() - t0, 2)}
        if not math.isfinite(rec["loss"]):
            rec["nonfinite"] = True
        if step % a.eval_every == 0 or step == a.steps:
            model.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                ev = []
                for j in range(4):
                    xe, ye = batch(data, split, n, a.batch, a.seq, 12345, j, dev)
                    ev.append(F.cross_entropy(model(xe).float().view(-1, 256), ye.view(-1)).item())
            rec["eval"] = sum(ev) / len(ev)
            model.train()
        log.write(json.dumps(rec) + "\n")
        if step % 50 == 0:
            log.flush()
        if step % a.ckpt_every == 0 or step == a.steps:
            save()
        if step % 100 == 0:
            print(f"{a.arm} s{a.seed} step {step} loss {rec['loss']:.4f} {(time.time() - t0) / max(1, step):.3f}s/step",
                  flush=True)
    log.flush()
    print("DONE", run, flush=True)


if __name__ == "__main__":
    main()
