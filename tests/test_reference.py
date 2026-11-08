"""The pure-torch Delta-Matching reference (delta_matching/reference.py): runs on any CUDA GPU (sm100 included) and, for
small shapes, on CPU.  It is the ground truth the validator judges kernels against, so it is tested here against exact
attention, against its own structural identities, and (on a Hopper GPU with the built extension) against the shipped
sm90 kernel."""
import importlib.util
import math
import os

import pytest
import torch

_P = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "delta_matching", "reference.py")
_S = importlib.util.spec_from_file_location("dm_reference_under_test", _P)
R = importlib.util.module_from_spec(_S)
_S.loader.exec_module(R)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
BIG = DEV == "cuda"
BF = torch.bfloat16
D = 128
FP8_PRESETS = ["sm90", "mxfp8"]
# fp8 floor bounds on Gaussian inputs (README "Accuracy": O, dV ~0.04, dQ, dK ~0.06; bound = 2x)
TOL = {"O": 8e-2, "dV": 8e-2, "dQ": 1.2e-1, "dK": 1.2e-1}
NAMES = ("O", "dQ", "dK", "dV")


def rel(a, b):
    a = a.double(); b = b.double()
    return ((a - b).norm() / b.norm().clamp_min(1e-300)).item()


def inputs(B, N, H, Hk, seed=0, scale=1.0, dev=DEV):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = (torch.randn(B, N, H, D, generator=g) * scale).to(dev, BF)
    k = (torch.randn(B, N, Hk, D, generator=g) * scale).to(dev, BF)
    v = torch.randn(B, N, Hk, D, generator=g).to(dev, BF)
    do = torch.randn(B, N, H, D, generator=g).to(dev, BF)
    return q, k, v, do


SMALL = [(1, 256, 4, 4), (1, 256, 4, 2), (2, 384, 4, 1)]


# ------------------------------------------------------------------------------------------------------- quantizers
def test_e4m3_round_is_torch_rne_saturating():
    x = torch.tensor([0.0, 1.0, 1.0625, 1.03125, 1.09375, 447.9, 448.0, 460.0, 1e6, -1e6, 2 ** -9, 2 ** -10, 3 * 2 ** -10])
    y = R.e4m3_round(x)
    assert y[6] == 448 and y[7] == 448 and y[8] == 448 and y[9] == -448      # saturating clamp, never NaN
    assert y[3] == 1.0 and y[4] == 1.125                                      # ties to even (1.03125 -> 1, 1.09375 -> 1.125)
    assert y[10] == 2 ** -9 and y[11] == 0 and y[12] == 2 ** -8               # subnormals (min 2^-9), ties to even


def test_blockrows_quantizer_contract():
    x = torch.randn(2, 3, 512, D, dtype=torch.float64).float() * 7
    c, s = R.quant_blockrows(x, 128)
    assert s.shape == (2, 3, 4)
    assert torch.equal(s, x.reshape(2, 3, 4, -1).abs().amax(-1) / 448)
    assert c.abs().max() <= 448 and torch.equal(R.e4m3_round(c), c)
    xr = R.fake_blockrows(x, 128)
    assert rel(xr, x) < 2 ** -4                                              # e4m3: 3 mantissa bits
    z = torch.zeros(1, 128, D)
    assert torch.equal(R.fake_blockrows(z, 128), z)                          # all-zero block: scale 1e-30 / 448, no NaN


@pytest.mark.parametrize("rounding", ["ceil", "floor"])
def test_mx_quantizer(rounding):
    g = torch.Generator().manual_seed(1)
    x = torch.randn(4, 256, D, generator=g) * torch.logspace(-6, 6, 4)[:, None, None]
    for axis in (-1, -2):
        c, s = R.quant_mx(x, axis, 32, rounding)
        assert torch.equal(torch.exp2(torch.log2(s).round()), s)              # powers of two (e8m0)
        assert torch.equal(R.e4m3_round(c), c)
        xr = R.dq_mx(c, s, axis)
        assert rel(xr, x) < 2 ** -4
        amax = x.movedim(axis, -1).reshape(*x.movedim(axis, -1).shape[:-1], -1, 32).abs().amax(-1).movedim(-1, axis)
        if rounding == "ceil":                                                # no saturation, and the tightest such scale
            assert (amax / s <= 448).all() and (amax / (s / 2) > 448).all()
        else:                                                                 # OCP: 2^(floor(log2 amax) - 8)
            assert torch.equal(s, torch.exp2(torch.floor(torch.log2(amax)) - 8))


def test_kmean_order_matches_mean():
    k = torch.randn(2, 3, 1024, D)
    assert torch.allclose(R.kmean_sm90(k), k.mean(-2, keepdim=True), rtol=0, atol=1e-6)


# --------------------------------------------------------------------------------------------- the attention engine
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("shape", SMALL)
def test_fp32_preset_is_exact_attention(shape, causal):
    B, N, H, Hk = shape
    q, k, v, do = inputs(B, N, H, Hk, seed=2)
    ex = R.attention_fp32(q, k, v, do, causal, torch.float64)
    for mode in ("matched", "stale"):
        o = R.dm_reference(q, k, v, do, causal, mode, "fp32", q_block=128)
        assert rel(o["O"], ex["O"]) < 4e-3                                     # bf16 rounding of the stored O
        assert rel(o["dV"], ex["dV"]) < 1e-5
        assert rel(o["dQ"], ex["dQ"]) < (1e-5 if mode == "matched" else 1e-2)  # stale reads the bf16 O
        assert rel(o["dK"], ex["dK"]) < (1e-5 if mode == "matched" else 1e-2)
        assert torch.allclose(o["LSE"].double(), ex["LSE"], atol=1e-5)


@pytest.mark.parametrize("preset", FP8_PRESETS)
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("shape", SMALL + ([(1, 2048, 8, 2), (2, 1024, 4, 4)] if BIG else []))
def test_fp8_presets_within_fp8_floor(preset, shape, causal):
    B, N, H, Hk = shape
    q, k, v, do = inputs(B, N, H, Hk, seed=3)
    ex = R.attention_fp32(q, k, v, do, causal, torch.float64 if N <= 1024 else torch.float32)
    for mode in ("matched", "stale"):
        o = R.dm_reference(q, k, v, do, causal, mode, preset)
        errs = {n: rel(o[n], ex[n]) for n in NAMES}
        for n, e in errs.items():
            assert math.isfinite(e) and e < TOL[n], (preset, mode, errs)
            assert e > 5e-3, f"{preset} {n} error {e} is below any fp8 floor: is the quantization on?"


@pytest.mark.parametrize("preset", FP8_PRESETS)
@pytest.mark.parametrize("causal", [True, False])
def test_matched_rowsum_identity(preset, causal):
    """matched: rowsum(dS) = 0 at fp32 round-off (relative to sum_j |P_ij dP_ij|); stale: at the fp8 error level."""
    q, k, v, do = inputs(1, 512, 4, 2, seed=4)
    lvl = {}
    for mode in ("matched", "stale"):
        o = R.dm_reference(q, k, v, do, causal, mode, preset, aux=True)
        lvl[mode] = o["rowsum_dS"].abs().max().item()
    assert lvl["matched"] < 1e-5, lvl
    assert lvl["stale"] > 100 * lvl["matched"], lvl


@pytest.mark.parametrize("preset", FP8_PRESETS)
def test_modes_share_forward_and_dv(preset):
    q, k, v, do = inputs(1, 512, 4, 2, seed=5)
    a = R.dm_reference(q, k, v, do, True, "matched", preset)
    b = R.dm_reference(q, k, v, do, True, "stale", preset)
    assert torch.equal(a["O"], b["O"]) and torch.equal(a["dV"], b["dV"]) and torch.equal(a["LSE"], b["LSE"])
    assert not torch.equal(a["dQ"], b["dQ"]) and not torch.equal(a["dK"], b["dK"])


def test_deterministic():
    q, k, v, do = inputs(1, 512, 4, 2, seed=6)
    a = R.dm_reference(q, k, v, do, True, "matched", "sm90")
    b = R.dm_reference(q, k, v, do, True, "matched", "sm90")
    assert all(torch.equal(a[n], b[n]) for n in NAMES)


@pytest.mark.parametrize("preset", FP8_PRESETS)
def test_gqa_equals_repeated_mha(preset):
    """GQA = MHA on repeated K / V with dK / dV summed over the group (quantization is per KV head, so the repeated
    copies quantize identically)."""
    q, k, v, do = inputs(1, 256, 8, 2, seed=7)
    g = R.dm_reference(q, k, v, do, True, "matched", preset)
    m = R.dm_reference(q, k.repeat_interleave(4, 2), v.repeat_interleave(4, 2), do, True, "matched", preset)
    assert torch.equal(g["O"], m["O"])
    assert torch.allclose(g["dQ"], m["dQ"], rtol=1e-5, atol=1e-6)
    assert torch.allclose(g["dK"], m["dK"].reshape(1, 256, 2, 4, D).sum(3), rtol=1e-4, atol=1e-5)
    assert torch.allclose(g["dV"], m["dV"].reshape(1, 256, 2, 4, D).sum(3), rtol=1e-4, atol=1e-5)


def test_q_block_invariance():
    """Blocking the query rows (memory bound) does not change the result beyond fp32 matmul round-off."""
    q, k, v, do = inputs(1, 512, 4, 2, seed=8)
    a = R.dm_reference(q, k, v, do, True, "matched", "sm90", q_block=128)
    b = R.dm_reference(q, k, v, do, True, "matched", "sm90", q_block=512)
    for n in NAMES:
        assert rel(a[n], b[n]) < 1e-5, n


def test_non_contiguous_inputs():
    B, N, H, Hk = 1, 256, 4, 2
    qkv = torch.randn(B, N, H + 2 * Hk, D).to(DEV, BF)
    do = torch.randn(B, N, H, D).to(DEV, BF)
    qv, kv, vv = qkv.split([H, Hk, Hk], dim=2)
    a = R.dm_reference(qv, kv, vv, do, True, "matched", "sm90")
    b = R.dm_reference(qv.contiguous(), kv.contiguous(), vv.contiguous(), do, True, "matched", "sm90")
    assert all(torch.equal(a[n], b[n]) for n in NAMES)


@pytest.mark.parametrize("kind", ["scale10", "outliers", "bigmag", "tiny", "uniform", "zero_do"])
def test_adversarial_inputs_stay_finite_and_bounded(kind):
    B, N, H, Hk = 1, 512, 4, 2
    q, k, v, do = inputs(B, N, H, Hk, seed=9)
    q, k, v, do = (x.float() for x in (q, k, v, do))
    if kind == "scale10":
        q, k = q * 10, k * 10
    elif kind == "outliers":
        q[..., :3] *= 30; k[..., :3] *= 30; v[:, ::97] *= 100
    elif kind == "bigmag":
        q, k, v, do = q * 3e3, k * 1e-3, v * 2e4, do * 1e-4
    elif kind == "tiny":
        q, k, v, do = q * 1e-20, k * 1e-20, v * 1e-20, do * 1e-20
    elif kind == "uniform":
        q = torch.zeros_like(q)
    elif kind == "zero_do":
        do = torch.zeros_like(do)
    q, k, v, do = (x.to(BF) for x in (q, k, v, do))
    ex = R.attention_fp32(q, k, v, do, True, torch.float64)
    for preset in FP8_PRESETS:
        for mode in ("matched", "stale"):
            o = R.dm_reference(q, k, v, do, True, mode, preset)
            for n in NAMES:
                assert torch.isfinite(o[n]).all(), (kind, preset, mode, n)
                if ex[n].norm() > 0 and kind not in ("scale10", "outliers", "tiny"):
                    assert rel(o[n], ex[n]) < 0.15, (kind, preset, mode, n, rel(o[n], ex[n]))
            if kind == "zero_do":
                assert all(o[n].abs().max() == 0 for n in ("dQ", "dK", "dV"))


@pytest.mark.skipif(not BIG, reason="long sequence needs a GPU")
@pytest.mark.parametrize("N", [8192, 16384])
def test_long_sequence(N):
    q, k, v, do = inputs(1, N, 4, 1, seed=10)
    ex = R.attention_fp32(q, k, v, do, True, torch.float32)
    o = R.dm_reference(q, k, v, do, True, "matched", "sm90", aux=True)
    for n in NAMES:
        assert rel(o[n], ex[n]) < TOL[n]
    assert o["rowsum_dS"].abs().max() < 1e-5


# ------------------------------------------------------------------------------ the Hopper kernel against the emulation
def _hopper_kernel():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (9, 0):
        return None
    try:
        import delta_matching as dm
        from delta_matching.core import _ops
        _ops()
        return dm
    except Exception:  # noqa: BLE001
        return None


@pytest.mark.skipif(_hopper_kernel() is None, reason="needs a Hopper GPU and the built sm90 extension")
@pytest.mark.parametrize("mode", ["matched", "stale"])
@pytest.mark.parametrize("shape", [(1, 2048, 4, 4), (1, 2048, 8, 2)])
def test_sm90_kernel_matches_emulation(shape, mode):
    """The emulation reproduces the shipped kernel: kernel-to-emulation distance is a small fraction of the fp8 floor
    (what is left is MMA accumulation order and ex2.approx flipping a few codes)."""
    dm = _hopper_kernel()
    B, N, H, Hk = shape
    q, k, v, do = inputs(B, N, H, Hk, seed=11)
    leaves = [x.clone().requires_grad_(True) for x in (q, k, v)]
    o = dm.dm_attn_func(*leaves, causal=True, mode=mode)
    o.backward(do)
    kern = {"O": o.detach(), "dQ": leaves[0].grad, "dK": leaves[1].grad, "dV": leaves[2].grad}
    em = R.dm_reference(q, k, v, do, True, mode, "sm90", o_for_stale=kern["O"])
    ex = R.attention_fp32(q, k, v, do, True, torch.float32)
    for n in NAMES:
        d, floor = rel(kern[n], em[n]), rel(em[n], ex[n])
        print(f"{n}: kernel-emulation {d:.2e}, emulation-exact {floor:.2e}")
        assert d < 0.2 * floor, (n, d, floor)
