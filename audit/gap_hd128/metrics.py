"""Metric definitions for the hd128 accuracy-floor investigation (accuracy-audit). Identical on both machines.
Reference: fp64 causal softmax attention on the SAME bf16 inputs (no rounding), scale D**-0.5, GQA by repeat_interleave on the head axis.
Metric: relative L2 error ||x - ref|| / ||ref|| over the whole tensor, for O (B,N,H,D), dQ (B,N,H,D), dK (B,N,Hk,D), dV (B,N,Hk,D);
plus LSE max abs error (natural log) when the kernel exports it. Inputs: q, k, v (B,N,H|Hk,D) bf16, do (B,N,H,D) bf16; dO is the
upstream gradient of O.
usage (library): ref = exact(q, k, v, do); errs = errors(outs, ref) with outs = dict(O=..., dQ=..., dK=..., dV=..., [LSE=(B,H,N)])"""
import torch


def exact(q, k, v, do, causal=True):
    B, N, H, D = q.shape
    rep = H // k.shape[2]
    qd, kd, vd = (t.double().requires_grad_() for t in (q, k, v))
    s = torch.einsum("bqhd,bkhd->bhqk", qd, kd.repeat_interleave(rep, 2)) * D ** -0.5
    if causal:
        s = s.masked_fill(~torch.ones(N, N, dtype=torch.bool, device=q.device).tril(), float("-inf"))
    o = torch.einsum("bhqk,bkhd->bqhd", s.softmax(-1), vd.repeat_interleave(rep, 2))
    o.backward(do.double())
    with torch.no_grad():    # LSE over K centred by its per-head token mean (both the sm90 and sm100 kernels centre K before quantizing)
        kc = (k.double() - k.double().mean(1, keepdim=True)).repeat_interleave(rep, 2)
        sc = torch.einsum("bqhd,bkhd->bhqk", q.double(), kc) * D ** -0.5
        if causal:
            sc = sc.masked_fill(~torch.ones(N, N, dtype=torch.bool, device=q.device).tril(), float("-inf"))
    return dict(O=o.detach(), LSE=torch.logsumexp(s.detach(), -1), LSEc=torch.logsumexp(sc, -1), dQ=qd.grad, dK=kd.grad, dV=vd.grad)


def errors(outs, ref):
    rel = lambda a, b: ((a.double() - b.double()).norm() / b.double().norm()).item()  # noqa: E731
    e = {m: rel(outs[m], ref[m]) for m in ("O", "dQ", "dK", "dV")}
    if outs.get("LSE") is not None:
        l = outs["LSE"].double().reshape(ref["LSE"].shape)
        e["LSE_maxabs"] = min((l - ref["LSE"]).abs().max().item(), (l - ref["LSEc"]).abs().max().item())   # raw or K-centred
    return e
