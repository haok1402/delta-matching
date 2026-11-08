"""Pure-PyTorch reference of Delta-Matching attention (forward + both backward delta modes), for any device.

No Triton, no compiled library, no relative imports: the file can be loaded standalone by path (the validator does)
and runs on CPU (small shapes) as well as on sm90 / sm100.

  attention_fp32(q, k, v, do, causal)            exact attention in fp32 (fp64 with dtype=torch.float64), autograd grads
  dm_reference(q, k, v, do, causal, mode, preset) emulation of an fp8 Delta-Matching kernel: every quantization point of
                                                  the kernel is reproduced in torch; returns O, LSE, dQ, dK, dV, delta and
                                                  diagnostics (row sums of dS, column sums of dK)
  PRESETS                                         the quantization schemes: "sm90" (the shipped Hopper kernel), "mxfp8"
                                                  (OCP MX block-32 e4m3 with e8m0 scales along every matmul's contraction
                                                  dim, the sm100 tcgen05 block-scaled format), "bf16" (no fp8, a control)

Layout: q (B, N, H, D), k / v (B, N, Hk, D), do (B, N, H, D) - flash-attention layout, any float dtype (bf16 in practice).
Outputs in the same layout, fp32 (the caller rounds to bf16 to compare against a bf16 kernel output).

What the emulation is exact about and what it is not: quantization (scales, rounding, saturation, which tensor is
quantized with which block) is reproduced exactly; fp32 matmul accumulation order (tensor-core MMA trees) and the
hardware exp2 approximation are not, so a correct kernel sits at fp32 round-off * (fp8 code flips this causes) from
the emulation, far below the fp8 floor against exact attention.  The validator relies on exactly that gap.
"""

import math
from dataclasses import dataclass, field, replace

import torch

E4M3 = torch.float8_e4m3fn
E4M3_MAX = 448.0
TINY = 1e-30
INV_E4M3_MAX = float(torch.tensor(1.0 / 448.0, dtype=torch.float32))   # fp32(1/448): the dO producer multiplies by it
LOG2E = 1.4426950408889634


# ----------------------------------------------------------------------------------------------------------------- quant
def e4m3_round(x: torch.Tensor) -> torch.Tensor:
    """Round fp32 values to the nearest e4m3 value (RNE, saturating at +-448), returned as fp32."""
    return x.clamp(-E4M3_MAX, E4M3_MAX).to(E4M3).to(torch.float32)


def quant_blockrows(x: torch.Tensor, blk: int, mul: bool = False):
    """The sm90 producers' quantizer: x (..., N, D) fp32 -> (codes fp32-valued e4m3, scales (..., N // blk) fp32).
    One scale per block of `blk` rows (all D columns): s = max(absmax, 1e-30) / 448, code = e4m3_rne(x / s) (IEEE
    division, clamped).  Dequantized value = code * s."""
    *lead, N, D = x.shape
    assert N % blk == 0, (N, blk)
    xb = x.reshape(*lead, N // blk, blk * D)
    amax = xb.abs().amax(-1)
    s = amax.clamp_min(TINY) * INV_E4M3_MAX if mul else amax.clamp_min(TINY) / E4M3_MAX
    codes = e4m3_round(xb / s.unsqueeze(-1)).reshape(*lead, N, D)
    return codes, s


def e8m0_scale(amax: torch.Tensor, rounding: str = "ceil") -> torch.Tensor:
    """The power-of-two (e8m0) scale of an MX block from its absmax (fp32 result).
    "ceil":  smallest 2^e with amax / 2^e <= 448 (no saturation; NVIDIA Transformer Engine / cuBLAS MXFP8 recipe).
    "floor": OCP MX v1.0: 2^(floor(log2 amax) - 8) (e4m3 emax = 8); values above 448 * 2^e saturate.
    An all-zero block gets 2^-127 (the smallest e8m0 value)."""
    a = amax.to(torch.float32)
    if rounding == "ceil":
        e = torch.ceil(torch.log2(a / E4M3_MAX))
        # guard log2 rounding: make sure a / 2^e <= 448 and a / 2^(e-1) > 448
        e = torch.where(a / torch.exp2(e) > E4M3_MAX, e + 1, e)
        e = torch.where(a / torch.exp2(e - 1) <= E4M3_MAX, e - 1, e)
    elif rounding == "floor":
        m, ex = torch.frexp(a)            # a = m * 2^ex, m in [0.5, 1): floor(log2 a) = ex - 1, exact
        e = (ex - 1 - 8).to(torch.float32)
    else:
        raise ValueError(rounding)
    e = torch.where(a > 0, e, torch.full_like(e, -127.0)).clamp(-127, 127)
    return torch.exp2(e)


def quant_mx(x: torch.Tensor, axis: int, block: int = 32, rounding: str = "ceil"):
    """MXFP8 (e4m3 elements, one e8m0 scale per `block` consecutive elements along `axis`, the matmul's contraction
    dim).  Returns (codes fp32-valued e4m3, scales fp32 powers of two, shape of x with axis divided by block)."""
    axis = axis % x.dim()
    xt = x.movedim(axis, -1)
    *lead, K = xt.shape
    assert K % block == 0, (K, block)
    xb = xt.reshape(*lead, K // block, block)
    s = e8m0_scale(xb.abs().amax(-1), rounding)
    codes = e4m3_round(xb / s.unsqueeze(-1))
    return codes.reshape(*lead, K).movedim(-1, axis), s.movedim(-1, axis)


def dq_mx(codes: torch.Tensor, s: torch.Tensor, axis: int, block: int = 32) -> torch.Tensor:
    return codes * s.repeat_interleave(block, dim=axis % codes.dim())


def fake_mx(x: torch.Tensor, axis: int, block: int = 32, rounding: str = "ceil") -> torch.Tensor:
    c, s = quant_mx(x, axis, block, rounding)
    return dq_mx(c, s, axis, block)


def fake_blockrows(x: torch.Tensor, blk: int, mul: bool = False) -> torch.Tensor:
    c, s = quant_blockrows(x, blk, mul)
    *lead, N, D = x.shape
    return (c.reshape(*lead, N // blk, blk, D) * s[..., None, None]).reshape(x.shape)


# ----------------------------------------------------------------------------------------------------------------- spec
@dataclass(frozen=True)
class Preset:
    """A quantization scheme of an fp8 attention kernel.  Every quantization site of the forward and backward:
      q, k, v, do     the producer quantizers of the inputs (k after centering, q after the softmax-scale bake)
      p_fwd           P in the forward's PV matmul;     p_bwd   P in dV = P^T dO
      dp              dP = dO V^T is fp32 out of the MMA (only its inputs are quantized)
      ds              dS in dK = dS^T Q and dQ = dS K
    Each site is a scheme name understood by `_site`: "none" (fp32), "bf16", "blockrows:<rows>", "mx:<axis>", or a
    preset-specific scheme ("sm90_pfwd", "sm90_pbwd", "sm90_ds")."""
    name: str
    q: str
    k: str
    v: str
    do: str
    p_fwd: str
    p_bwd: str
    ds: str
    # separate quantizations of the same tensor for its two uses (MX quantizes along the contraction dim, so the
    # transposed operand of the other matmul gets its own scales): q2 for dK = dS^T Q, k2 for dQ = dS K,
    # v2 unused (V is only in PV and dP), do2 for dV = P^T dO, ds2 for dQ (ds is the dK operand)
    q2: str | None = None
    k2: str | None = None
    do2: str | None = None
    ds2: str | None = None
    center_k: bool = True        # K centered by its per-head mean before quantization (softmax-invariant)
    mx_rounding: str = "ceil"
    notes: str = ""


PRESETS: dict[str, Preset] = {}


def _register(p: Preset):
    PRESETS[p.name] = p
    return p


# the shipped sm90 hd128 kernel (Delta-Matching-128 v3); the P / dS sites are filled in from NUMERICS-hd128.md
_register(Preset("sm90", q="blockrows:128", k="blockrows:64", v="blockrows:64", do="blockrows_mul:128",
                 p_fwd="sm90_pfwd", p_bwd="sm90_pbwd", ds="sm90_ds",
                 notes="the Hopper kernel, head_dim 128, branch main (sm100/validate/NUMERICS-hd128.md): Q / dO 128-row and "
                       "K / V 64-row fp32 block scales; forward P e4m3 with a per (row, 64-key group) basis under a "
                       "descending 256-key-tile online max; backward P as e4m3(2^8 P); dS per 64x128 tile scale psi on "
                       "2^8 ks_j dS; matched delta = sum_j P dP with the unquantized P"))
# MXFP8 everywhere: each operand quantized in blocks of 32 along the contraction dim of the matmul that consumes it.
#   S = Q K^T     contracts D:      q mx:-1, k mx:-1
#   O = P V       contracts keys:   p_fwd mx:-1 (P rows along keys), v mx:-3 (V along seqlen; (B,N,H,D) -> axis 1)
#   dV = P^T dO   contracts queries: p_bwd along queries, do2 along seqlen
#   dP = dO V^T   contracts D:      do mx:-1, v (D-blocked copy) -> k-like; we reuse 'v' only for PV, see _vD
#   dK = dS^T Q   contracts queries: ds along queries, q2 along seqlen
#   dQ = dS K     contracts keys:   ds2 along keys, k2 along seqlen
_register(Preset("mxfp8", q="mx:D", k="mx:D", v="mx:N", do="mx:D", p_fwd="mx:keys", p_bwd="mx:queries",
                 ds="mx:queries", q2="mx:N", k2="mx:N", do2="mx:N", ds2="mx:keys",
                 notes="OCP MX e4m3, block 32 along every contraction dim, e8m0 scales (ceil rounding: no saturation)"))
# kda-sm100's proposal: inputs as sm90, P and dS e4m3 with a UE8M0 scale per row per 32 keys (ceil rounding)
_register(Preset("mxfp8-p", q="blockrows:128", k="blockrows:64", v="blockrows:64", do="blockrows_mul:128",
                 p_fwd="mx:keys", p_bwd="mx:keys", ds="mx:keys",
                 notes="sm90 input quantizers; P (PV, dV) and dS (dK, dQ) MX e4m3 per row per 32 keys"))
PRESETS["mxfp8-all"] = PRESETS["mxfp8"]
_register(Preset("bf16", q="bf16", k="bf16", v="bf16", do="bf16", p_fwd="bf16", p_bwd="bf16", ds="bf16",
                 notes="no fp8 anywhere: bf16 operands, fp32 accumulation (the control a cheating candidate matches)"))
_register(Preset("fp32", q="none", k="none", v="none", do="none", p_fwd="none", p_bwd="none", ds="none",
                 center_k=False, notes="no rounding at all (sanity: equals attention_fp32 up to accumulation order)"))


# ----------------------------------------------------------------------------------------------------------------- sites
# Tensors inside the engine are (..., rows, cols): inputs (B, H, N, D); P / dS (B, H, Nq, Nk).
_AXES = {"D": -1, "N": -2, "keys": -1, "queries": -2}


class _Ctx:
    """What a site quantizer may need besides the tensor (row max of the forward, causal mask, ...)."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


SITE_FNS = {}     # preset-specific schemes: name -> fn(x, ctx, preset) -> fake-quantized fp32 tensor


def _site(scheme: str, x: torch.Tensor, ctx: _Ctx, preset: Preset) -> torch.Tensor:
    if scheme == "none":
        return x
    if scheme == "bf16":
        return x.to(torch.bfloat16).to(torch.float32)
    if scheme.startswith("blockrows_mul:"):
        return fake_blockrows(x, int(scheme.split(":")[1]), mul=True)
    if scheme.startswith("blockrows:"):
        return fake_blockrows(x, int(scheme.split(":")[1]))
    if scheme.startswith("mx:"):
        return fake_mx(x, _AXES[scheme.split(":")[1]], 32, preset.mx_rounding)
    if scheme.startswith("e4m3"):          # plain e4m3 with a fixed power-of-two multiplier: "e4m3*256"
        mul = float(scheme.split("*")[1]) if "*" in scheme else 1.0
        return e4m3_round(x * mul) / mul
    if scheme in SITE_FNS:
        return SITE_FNS[scheme](x, ctx, preset)
    raise ValueError(f"unknown quantization scheme {scheme!r}")


def kmean_sm90(k: torch.Tensor) -> torch.Tensor:
    """Per-head mean over the sequence in the producers' order: fp32 sums of 64-row chunks folded serially, / N.
    k (..., N, D) fp32 -> (..., 1, D).  (The within-chunk 64-row sum is a Triton tree; matches to fp32 round-off.)"""
    N = k.shape[-2]
    if N % 64:
        return k.mean(-2, keepdim=True)
    parts = k.reshape(*k.shape[:-2], N // 64, 64, k.shape[-1]).sum(-2)
    acc = torch.zeros_like(parts[..., 0, :])
    for j in range(parts.shape[-2]):
        acc = acc + parts[..., j, :]
    return (acc / N).unsqueeze(-2)


class _NoTF32:
    def __enter__(self):
        self.a = torch.backends.cuda.matmul.allow_tf32
        self.b = torch.backends.cudnn.allow_tf32
        self.c = torch.get_float32_matmul_precision()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

    def __exit__(self, *a):
        torch.backends.cuda.matmul.allow_tf32 = self.a
        torch.backends.cudnn.allow_tf32 = self.b
        torch.set_float32_matmul_precision(self.c)


# ----------------------------------------------------------------------------------------------------------------- exact
def attention_fp32(q, k, v, do, causal=True, dtype=torch.float32):
    """Exact attention (no quantization) in fp32 or fp64 on the given inputs, gradients by autograd.  GQA by repeating
    K / V.  Returns dict O, LSE, dQ, dK, dV in the input layout (B, N, H, D)."""
    B, N, H, D = q.shape
    Hk = k.shape[2]
    with _NoTF32():
        qf = q.to(dtype).transpose(1, 2).detach().requires_grad_(True)
        kf = k.to(dtype).transpose(1, 2).detach().requires_grad_(True)
        vf = v.to(dtype).transpose(1, 2).detach().requires_grad_(True)
        kr = kf.repeat_interleave(H // Hk, dim=1)
        vr = vf.repeat_interleave(H // Hk, dim=1)
        s = (qf @ kr.transpose(-1, -2)) * (D ** -0.5)
        if causal:
            mask = torch.ones(N, N, device=q.device, dtype=torch.bool).tril()
            s = s.masked_fill(~mask, float("-inf"))
        lse = torch.logsumexp(s, -1)
        o = torch.softmax(s, dim=-1) @ vr
        o.backward(do.to(dtype).transpose(1, 2))
    return {"O": o.detach().transpose(1, 2), "LSE": lse.detach(), "dQ": qf.grad.transpose(1, 2),
            "dK": kf.grad.transpose(1, 2), "dV": vf.grad.transpose(1, 2)}


# ----------------------------------------------------------------------------------------------------------------- engine
def dm_reference(q, k, v, do, causal=True, mode="matched", preset="sm90", q_block=1024, aux=False,
                 o_for_stale=None):
    """Emulate an fp8 Delta-Matching kernel in torch.

    q (B, N, H, D), k / v (B, N, Hk, D), do (B, N, H, D); mode "matched" | "stale"; preset a PRESETS name or Preset.
    Returns dict O (bf16-rounded, as the kernel stores it), LSE (B, H, N) natural log, dQ, dK, dV (fp32, input layout),
    delta (B, H, N), and with aux=True: rowsum_dS (sum_j dS_ij before dS quantization), rowsum_dSq (after), colsum_dK
    (sum_j dK_j per head, (B, Hk, D)).  o_for_stale: the O the stale delta reads (default: this emulation's bf16 O; a
    validator passes the candidate's own O to isolate the backward).
    Memory: O(q_block * N) per (batch, head); causal key ranges are truncated per block."""
    p = PRESETS[preset] if isinstance(preset, str) else preset
    if mode not in ("matched", "stale"):
        raise ValueError(mode)
    B, N, H, D = q.shape
    Hk = k.shape[2]
    rep = H // Hk
    sm = D ** -0.5
    dev = q.device
    f32 = torch.float32
    with _NoTF32():
        qh = q.to(f32).transpose(1, 2) * sm                  # (B, H, N, D), the softmax scale baked into Q
        kh = k.to(f32).transpose(1, 2)
        if p.center_k:
            kh = kh - kmean_sm90(kh)
        vh = v.to(f32).transpose(1, 2)
        doh = do.to(f32).transpose(1, 2)
        c0 = _Ctx(role="input")
        Qa = _site(p.q, qh, c0, p)                           # operand of S = Q K^T
        Ka = _site(p.k, kh, c0, p)
        kscale = quant_blockrows(kh, 64)[1].repeat_interleave(64, -1)[..., :N] if N % 64 == 0 else None   # (B, Hk, N)
        Va = _site(p.v, vh, c0, p)                           # operand of O = P V
        Vd = _site(p.v if p.q2 is None else p.k, vh, c0, p)  # operand of dP = dO V^T (MX: D-blocked like K)
        dOa = _site(p.do, doh, c0, p)                        # operand of dP = dO V^T
        Qb = Qa if p.q2 is None else _site(p.q2, qh, c0, p)  # operand of dK = dS^T Q
        Kb = Ka if p.k2 is None else _site(p.k2, kh, c0, p)  # operand of dQ = dS K
        dOb = dOa if p.do2 is None else _site(p.do2, doh, c0, p)   # operand of dV = P^T dO

        O = torch.empty(B, H, N, D, device=dev, dtype=f32)
        LSE = torch.empty(B, H, N, device=dev, dtype=f32)
        dQ = torch.zeros(B, H, N, D, device=dev, dtype=f32)
        dK = torch.zeros(B, Hk, N, D, device=dev, dtype=f32)
        dV = torch.zeros(B, Hk, N, D, device=dev, dtype=f32)
        delta = torch.empty(B, H, N, device=dev, dtype=f32)
        rs = torch.empty(B, H, N, device=dev, dtype=f32) if aux else None
        rsq = torch.empty(B, H, N, device=dev, dtype=f32) if aux else None
        qb = min(q_block, N)
        assert qb % 128 == 0 or qb == N
        kidx = torch.arange(N, device=dev)
        # forward first (the stale delta needs the stored bf16 O)
        for i0 in range(0, N, qb):
            i1 = min(i0 + qb, N)
            nk = i1 if causal else N
            S = Qa[:, :, i0:i1] @ Ka[:, :, :nk].repeat_interleave(rep, 1).transpose(-1, -2)
            if causal:
                S = S.masked_fill(kidx[None, :nk] > kidx[i0:i1, None], float("-inf"))
            m = S.amax(-1, keepdim=True)
            P = torch.exp(S - m)
            l = P.sum(-1, keepdim=True)
            ctx = _Ctx(role="p_fwd", rowmax=m, rows=(i0, i1), causal=causal, S=S)
            Pq = _site(p.p_fwd, P, ctx, p)
            O[:, :, i0:i1] = (Pq @ Va[:, :, :nk].repeat_interleave(rep, 1)) / l
            LSE[:, :, i0:i1] = (m + torch.log(l)).squeeze(-1)
        Ob = O.to(torch.bfloat16)
        Ost = (o_for_stale.to(f32).transpose(1, 2) if o_for_stale is not None else Ob.to(f32))
        for i0 in range(0, N, qb):
            i1 = min(i0 + qb, N)
            nk = i1 if causal else N
            Kr = Ka[:, :, :nk].repeat_interleave(rep, 1)
            S = Qa[:, :, i0:i1] @ Kr.transpose(-1, -2)
            if causal:
                S = S.masked_fill(kidx[None, :nk] > kidx[i0:i1, None], float("-inf"))
            P = torch.exp(S - LSE[:, :, i0:i1, None])
            ctx = _Ctx(role="p_bwd", lse=LSE[:, :, i0:i1, None], rows=(i0, i1), causal=causal, S=S)
            Pq = _site(p.p_bwd, P, ctx, p)
            dVp = Pq.transpose(-1, -2) @ dOb[:, :, i0:i1]                         # (B, H, nk, D)
            dV[:, :, :nk] += dVp.reshape(B, Hk, rep, nk, D).sum(2)
            dP = dOa[:, :, i0:i1] @ Vd[:, :, :nk].repeat_interleave(rep, 1).transpose(-1, -2)
            if mode == "stale":
                dl = (doh[:, :, i0:i1] * Ost[:, :, i0:i1]).sum(-1, keepdim=True)
            else:
                dl = _matched_delta(P, Pq, dP, p)
            delta[:, :, i0:i1] = dl.squeeze(-1)
            dS = _ds_operand(P, Pq, p) * (dP - dl)
            ctx = _Ctx(role="ds", rows=(i0, i1), causal=causal,
                       kscale=None if kscale is None else kscale[:, :, :nk].repeat_interleave(rep, 1).unsqueeze(-2).squeeze(-2))
            dSq = _site(p.ds, dS, ctx, p)
            dSq2 = dSq if p.ds2 is None else _site(p.ds2, dS, ctx, p)
            if aux:
                rs[:, :, i0:i1] = dS.sum(-1)
                rsq[:, :, i0:i1] = dSq.sum(-1)
            dKp = dSq.transpose(-1, -2) @ Qb[:, :, i0:i1]
            dK[:, :, :nk] += dKp.reshape(B, Hk, rep, nk, D).sum(2)
            dQ[:, :, i0:i1] = (dSq2 @ Kb[:, :, :nk].repeat_interleave(rep, 1)) * sm
    out = {"O": Ob.transpose(1, 2), "LSE": LSE, "dQ": dQ.transpose(1, 2), "dK": dK.transpose(1, 2),
           "dV": dV.transpose(1, 2), "delta": delta}
    if aux:
        out.update(rowsum_dS=rs, rowsum_dSq=rsq, colsum_dK=dK.sum(2))
    return out


def _matched_delta(P, Pq, dP, p: Preset):
    """The matched delta: rowsum of the backward's own (P, dP) pair, P the unquantized fp32 P of the dS line (sm90
    pass 0 sums exactly the operands of pass 1's dS line), accumulated in fp64 and rounded once."""
    fn = SITE_FNS.get(p.name + ":matched_delta")
    if fn is not None:
        return fn(P, Pq, dP, p)
    return (P.double() * dP.double()).sum(-1, keepdim=True).float()


def _ds_operand(P, Pq, p: Preset):
    """The P that multiplies (dP - delta) in dS: the unquantized fp32 P (the sm90 kernel's pass 1 dS line, and the pass-0
    delta sums the same P, which is what makes rowsum(dS) = delta * (1 - rowsum P) vanish at fp32 round-off)."""
    fn = SITE_FNS.get(p.name + ":ds_p")
    return fn(P, Pq, p) if fn is not None else P


# ------------------------------------------------------------------------------------------------- sm90 site schemes
def _rowblock_absmax(x, blk=32):
    """e4m3 with one fp32 scale absmax / 448 per row per `blk` keys (columns)."""
    *lead, R_, C = x.shape
    xb = x.reshape(*lead, R_, C // blk, blk)
    s = xb.abs().amax(-1, keepdim=True).clamp_min(TINY) / E4M3_MAX
    return (e4m3_round(xb / s) * s).reshape(x.shape)


LN2 = 0.6931471805599453


def _pad_to(x, rows, cols, value=0.0):
    R_, C = x.shape[-2:]
    pr, pc = (-R_) % rows, (-C) % cols
    if pr or pc:
        x = torch.nn.functional.pad(x, (0, pc, 0, pr), value=value)
    return x


def sm90_pfwd(P, ctx, p):
    """The forward's P codes (NUMERICS-hd128.md section 2): keys in 256-key tiles visited in DESCENDING order with an online
    max m_t (the max over tiles >= t); per (row, 64-key group g) the basis nu_g = max(groupmax_g S, m_t - 12 ln2) and
    code = e4m3(448 * exp(S - nu_g)).  P here is exp(S - m) with m the final row max; returns the dequantized P-hat in the
    same units.  (The row sum l is taken from the unquantized P, which the engine already does.)"""
    S = ctx.S
    R_, C = S.shape[-2:]
    Sp = _pad_to(S, 1, 256, float("-inf"))
    T = Sp.shape[-1] // 256
    g = Sp.reshape(*Sp.shape[:-1], T, 4, 64)
    gm = g.amax(-1)                                                    # (..., R, T, 4)
    tm = gm.amax(-1)                                                   # (..., R, T)
    mt = tm.flip(-1).cummax(-1).values.flip(-1)                        # running max when tile t is processed
    nu = torch.maximum(gm, mt.unsqueeze(-1) - 12 * LN2)                # (..., R, T, 4)
    m = ctx.rowmax.unsqueeze(-1).unsqueeze(-1)                         # (..., R, 1, 1, 1)
    nu = torch.where(torch.isfinite(nu), nu, m.squeeze(-1))            # tiles with no live key (padding, causal tail)
    pf = torch.exp(g - nu.unsqueeze(-1)) * E4M3_MAX
    ph = e4m3_round(pf) / E4M3_MAX * torch.exp(nu.unsqueeze(-1) - m)
    return ph.reshape(Sp.shape)[..., :C]


def sm90_ds(dS, ctx, p):
    """dS codes of pass 1 (NUMERICS-hd128.md section 3): dS' = 2^8 * ks_j * dS (ks: the 64-key K scale of key j) is
    quantized with one scale psi = max(amax / 448, 1e-30) per 64-row x 128-key tile; dS-hat = e4m3(dS' / psi) * psi /
    (2^8 ks_j).  (The F2b dominance floor, which coarsens tiles 1e-8 below the key block's largest, is not modelled.)"""
    ks = ctx.kscale                                                    # (..., C) per key
    X = dS * ks.unsqueeze(-2)
    R_, C = X.shape[-2:]
    Xp = _pad_to(X, 64, 128)
    Rp, Cp = Xp.shape[-2:]
    t = Xp.reshape(*Xp.shape[:-2], Rp // 64, 64, Cp // 128, 128)
    psi = t.abs().amax(dim=(-3, -1), keepdim=True) * (1.0 / E4M3_MAX)
    zero = psi < TINY
    psi = psi.clamp_min(TINY)
    q = torch.where(zero, torch.zeros_like(t), e4m3_round(t / psi) * psi)
    return q.reshape(Xp.shape)[..., :R_, :C] / ks.unsqueeze(-2)


SITE_FNS["sm90_pfwd"] = sm90_pfwd
SITE_FNS["sm90_pbwd"] = lambda P, ctx, p: e4m3_round(P.clamp(max=16.0) * 256.0) / 256.0     # P' = 2^8 P, fixed scale
SITE_FNS["sm90_ds"] = sm90_ds
