"""Bit-identical input generator for the hd128 accuracy-floor investigation (accuracy-audit, 2026-11-10).
make_inputs() is copied VERBATIM from dm-validator validator/validate.py @ 364413b. CPU torch generator only.
usage: python gen_inputs.py OUTDIR   -> writes in_N<N>_H<H>_<Hk>_<family>.pt and checks sha256 against manifest.json"""
import hashlib, json, math, os, sys
import torch
D = 128
BF = torch.bfloat16
SEED = 13
SHAPES = [(1, 2048, 8, 2), (1, 4096, 16, 4)]          # (B, N, H, Hk), causal
FAMILIES = ["randn", "outliers", "nearsat", "scaled3", "scaled10", "peaked", "bigmag", "audit_ramp", "audit_growmax"]


def make_inputs(kind, B, N, H, Hk, gen, dev):
    def rn(*s):
        return torch.randn(*s, generator=gen, device=dev, dtype=torch.float32)
    q, k, v, do = rn(B, N, H, D), rn(B, N, Hk, D), rn(B, N, Hk, D), rn(B, N, H, D)
    if kind.startswith("stress60mix_"):               # alternating stress: primary inputs randn, alt = the named family
        kind = "randn"
    if kind.startswith("stress60_"):                  # stress60 on a named input family, e.g. stress60_ramp
        kind = kind[len("stress60_"):]
    if kind.startswith("audit_"):
        # accuracy-audit's rescale families (sm100/audit/families.py, CORRECTED after kda-sm100-sspv2's planted-bug
        # control: the forward visits key blocks from the HIGHEST index down).  q ~ sqrt(D) u + noise, k_j = s_j u + noise:
        # ramp / ramp_th / ramp_sm rise towards key 0 by 12 / 6 / 3 nats per 128 keys (rescale every tile / just above /
        # below the lazy threshold); growmax: spike keys every 512, height growing towards key 0 (+10 nats each);
        # ramp_up: rising with key index = seen first = NO rescale (control)
        fam = kind[len("audit_"):]
        u = rn(D)
        u = u / u.norm()
        j = torch.arange(N, device=dev, dtype=torch.float32)
        if fam in ("ramp", "ramp_th", "ramp_sm"):
            sj = {"ramp": 12.0, "ramp_th": 6.0, "ramp_sm": 3.0}[fam] / 128.0 * (N - 1 - j)
        elif fam == "ramp_up":
            sj = 12.0 / 128.0 * j
        elif fam == "growmax":
            sj = torch.zeros(N, device=dev)
            pos = torch.arange(256, N, 512, device=dev)
            sj[pos] = 10.0 * torch.arange(len(pos), 0, -1, device=dev, dtype=torch.float32)
        else:
            raise ValueError(kind)
        q = math.sqrt(D) * u.expand(B, N, H, D) + 0.5 * rn(B, N, H, D)
        k = sj[None, :, None, None] * u + 0.5 * rn(B, N, Hk, D)
        v, do = rn(B, N, Hk, D), rn(B, N, H, D)
        return [x.to(BF).contiguous() for x in (q, k, v, do)]
    if kind in ("randn", "stress", "stress20", "stress60"):
        pass
    elif kind == "ramp":
        # online-softmax RESCALE path (kda-sm100-sspv2: randn never triggers it).  The forward visits key blocks from the
        # HIGHEST index down (FA4 n_block loop), so logits must rise towards key 0 for the running max to grow on every
        # visited tile (kda-sm100-sspv2 12:05: the ascending version never rescaled).  Channel 0: q = 8, k = +28 .. -28
        j = torch.arange(N, device=dev, dtype=torch.float32)
        q[..., 0] = 8.0
        k[..., 0] = (28.0 * (1.0 - 2.0 * j / max(N - 1, 1)))[None, :, None]
    elif kind == "growmax":
        # growing-max rows in steps, in the forward's visiting order (high key index first): k channel 0 is a staircase
        # rising towards key 0 (16 steps), q channel 0 a per-row slope in [2, 14], so some rows jump the max by many log2
        # units at each step and others barely move
        j = torch.arange(N, device=dev, dtype=torch.float32)
        k[..., 0] = (2.5 * torch.floor(16.0 * (N - 1 - j) / N) - 20.0)[None, :, None]
        q[..., 0] = 2.0 + 12.0 * torch.rand(B, N, H, generator=gen, device=dev)
    elif kind.startswith("scaled"):                   # larger logit range: q, k * s
        s = float(kind[6:])
        q, k = q * s, k * s
    elif kind == "outliers":                          # activation outliers: a few hot channels, a few huge tokens
        ch = torch.randint(0, D, (4,), generator=gen, device=dev)
        q[..., ch] *= 30; k[..., ch] *= 30
        tok = torch.randint(0, N, (max(1, N // 512),), generator=gen, device=dev)
        v[:, tok] *= 100; do[:, tok] *= 50
    elif kind == "bigmag":                            # huge and tiny magnitudes (scale handling, no overflow/underflow)
        q, k, v, do = q * 3e3, k * 1e-3, v * 2e4, do * 1e-4
    elif kind == "nearsat":                           # values on and next to e4m3 code boundaries after scaling
        def grid(x):
            g = torch.tensor([448., 416., 432., 440., 464., 240., 1., 1.0625, 0.0078125, 2 ** -9], device=dev)
            idx = torch.randint(0, len(g), x.shape, generator=gen, device=dev)
            sign = torch.where(torch.rand(x.shape, generator=gen, device=dev) < 0.5, -1.0, 1.0)
            return g[idx] * sign / 448.0 * 4.0
        q, k, v, do = grid(q), grid(k), grid(v), grid(do)
    elif kind == "peaked":                            # attention concentrated on a few keys (P near one-hot)
        k = k.clone()
        hot = torch.randint(0, N, (max(1, N // 256),), generator=gen, device=dev)
        k[:, hot] = q[:, hot][:, :, :Hk] * 4
        q = q * 2
    elif kind == "uniform":                           # zero logits: P uniform, LSE = log(#keys)
        q = torch.zeros_like(q)
    elif kind == "zero_do":
        do = torch.zeros_like(do)
    elif kind == "lowrank_v":                         # V constant across keys: O = v0, dQ = dK = 0 exactly
        v = v[:, :1].expand_as(v).contiguous()
    else:
        raise ValueError(kind)
    return [x.to(BF) for x in (q, k, v, do)]


INPUT_CACHE = os.environ.get("DMVAL_INPUT_CACHE", os.path.expanduser("~/workspace/haok1402/sm100/validate/inputcache"))



def sha(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def generate(outdir=None):
    man = {}
    for B, N, H, Hk in SHAPES:
        for fam in FAMILIES:
            g = torch.Generator().manual_seed(SEED)
            q, k, v, do = make_inputs(fam, B, N, H, Hk, g, torch.device("cpu"))
            name = f"in_N{N}_H{H}_{Hk}_{fam}"
            man[name] = {n: sha(t) for n, t in zip(("q", "k", "v", "do"), (q, k, v, do))}
            if outdir:
                torch.save(dict(q=q, k=k, v=v, do=do, causal=True), os.path.join(outdir, name + ".pt"))
    return man


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else None
    if out:
        os.makedirs(out, exist_ok=True)
    man = generate(out)
    ref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manifest.json")
    if os.path.exists(ref_path):
        ref = json.load(open(ref_path))["inputs"]
        bad = [n for n in man if man[n] != ref.get(n)]
        print("sha256 vs manifest:", "ALL MATCH" if not bad else f"MISMATCH {bad}")
    else:
        json.dump({"torch": torch.__version__, "inputs": man}, open(ref_path, "w"), indent=1)
        print("manifest written")
