"""sm100 (Blackwell, sm_100) ring context-parallel blocks at head_dim 256 (delta_matching.sm100.ring_hd256, numerics sm100-mx-d256).
A block attends Nq queries to Nk keys; causal blocks are bottom-right aligned (query i sees keys <= i + Nk - Nq).
  bottom-right   causal fwd_block / delta_block / delta_nd + delta_finish / bwd_block bitwise equal to the last Nq rows of the square
                 causal dm_attn_func path at Nk whose dO is zero on the first Nk - Nq rows (those rows add exact zeros to dK / dV),
                 run with the square's lse / delta of the rows;
  non-causal     Nq != Nk blocks against exact attention and the torch emulation of the spec (delta_matching/reference.py, preset
                 sm100-mx-d256), with the bounds of test_sm100_hd256.py;
  fp32 outputs   fp32 dq / dk / dv are accumulators (+=): bf16 of one call into zeros == the bf16 outputs, two calls == 2x one call,
                 nonzero content gets exactly one fp32 add;
  key ranges     DM_BWD_KRANGE=1 under a small DM_BWD_WS_LIMIT_GB: the exact-carry key-range plan is bitwise the whole block;
  dmk            the KV2 tile-class peel (DM_RING_DMK=1, the default) is bitwise DM_RING_DMK=0;
  identity       one causal sequence run as its ring decomposition (chunk j = rows [j c, (j + 1) c) against keys [0, (j + 1) c)),
                 matched delta from each chunk's own fwd_block lse and delta_block: rowsum(dS) = 0 to fp32 round-off on every row,
                 from the backward's in-kernel pre-quantization dS (the dbg-16 build of the KV2 launch that bwd_block runs) and from
                 the spec emulation with the kernel's lse / delta.
Needs an sm_100 GPU (B200 / GB200) with the CuTe DSL (nvidia-cutlass-dsl, flash-attn cute); skipped elsewhere.  The first run compiles
the kernels into the CuTe DSL cache (FLASH_ATTENTION_CUTE_DSL_CACHE_DIR), later runs load them."""
import os
from contextlib import contextmanager

import pytest
import torch


def _skip_reason():
    if not torch.cuda.is_available():
        return "needs a CUDA GPU"
    if torch.cuda.get_device_capability(0)[0] != 10:
        return f"sm100 kernel test: needs an sm_100 GPU; found {torch.cuda.get_device_name(0)}"
    try:
        import delta_matching.sm100.attention  # noqa: F401
        import delta_matching.sm100.ring_hd256  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return f"delta_matching.sm100 cannot be imported ({type(e).__name__}: {e})"
    return None


REASON = _skip_reason()
pytestmark = pytest.mark.skipif(REASON is not None, reason=REASON or "")
D = 256
SM = D ** -0.5          # softmax scale baked into the Q codes (= 2^-4, exact)
H, HK = 16, 4
ENV_KNOBS = ("DM_BWD_WS_LIMIT_GB", "DM_H256_DQW_GB", "DM_BWD_KRANGE", "DM_RING_KEYRANGE", "DM_RING_DMK", "DM_BWD_OSTR",
             "DM_BWD_HEAD_GROUPS", "DM_H256_DQR_CARRY")
OUT_NAMES = ("dq", "dk", "dv", "dq fp32", "dk fp32", "dv fp32")


@contextmanager
def _env(**kw):
    old = {k: os.environ.get(k) for k in ENV_KNOBS}
    try:
        for k in ENV_KNOBS:
            os.environ.pop(k, None)
        for k, v in kw.items():
            os.environ[k] = str(v)
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _inputs(B, Nq, Nk, seed, family="randn"):
    """bf16 q, dO (B, Nq, H, 256) and k, v (B, Nk, Hk, 256).  ramp / growmax (Nq == Nk, as in test_sm100_hd256.py): logits that grow
    towards key 0, so the online-softmax rescale fires on many tiles."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    q, k, v, do = (torch.randn(B, n, h, D, device="cuda", generator=g) for n, h in ((Nq, H), (Nk, HK), (Nk, HK), (Nq, H)))
    N = Nq
    j = torch.arange(N, device="cuda", dtype=torch.float32)
    if family == "ramp":
        assert Nq == Nk
        q[..., 0] = 8.0
        k[..., 0] = (28.0 * (1.0 - 2.0 * j / max(N - 1, 1)))[None, :, None]
    elif family == "growmax":
        assert Nq == Nk
        k[..., 0] = (2.5 * torch.floor(16.0 * (N - 1 - j) / N) - 20.0)[None, :, None]
        q[..., 0] = 2.0 + 12.0 * torch.rand(B, N, H, generator=g, device="cuda")
    else:
        assert family == "randn", family
    return [x.bfloat16().contiguous() for x in (q, k, v, do)]


def _codes(q, k, v, do):
    """The producers' operands (any Nq, Nk) from quant.quant_rows_ref, the pure-torch quantizer the CuTe producers are bitwise
    equal to: e4m3 codes + one pow2 scale per 32 tokens, Q with the softmax scale baked, K centred by its fp32 token mean."""
    from delta_matching.sm100.fwd import pack_kv_scales
    from delta_matching.sm100.quant import quant_rows_ref
    mean = k.float().mean(dim=1)
    q8, qs = quant_rows_ref(q, SM)
    k8, ks = quant_rows_ref(k, 1.0, mean)
    v8, vs = quant_rows_ref(v, 1.0)
    do8, dos = quant_rows_ref(do, 1.0)
    return dict(q8=q8, qs=qs, k8=k8, ks=ks, v8=v8, vs=vs, kvw=pack_kv_scales(ks, vs), do8=do8, dos=dos, mean=mean)


def _rows(c, a, b):
    """the query rows [a, b) of a code dict"""
    out = dict(c)
    out.update(q8=c["q8"][:, a:b].contiguous(), qs=c["qs"][..., a // 32:b // 32].contiguous(),
               do8=c["do8"][:, a:b].contiguous(), dos=c["dos"][..., a // 32:b // 32].contiguous())
    return out


def _keys(c, n):
    """the keys [0, n) of a code dict (kvw: two words per 128-key tile)"""
    out = dict(c)
    out.update(k8=c["k8"][:, :n].contiguous(), ks=c["ks"][..., :n // 32].contiguous(), v8=c["v8"][:, :n].contiguous(),
               vs=c["vs"][..., :n // 32].contiguous(), kvw=c["kvw"][..., :n // 64].contiguous())
    return out


def _ops(c):
    return c["q8"], c["qs"], c["k8"], c["ks"], c["v8"], c["vs"], c["do8"], c["dos"]


def _fwd_delta(c, causal):
    """(o, lse) of the block and its matched delta (the block holds all keys of its rows)"""
    from delta_matching.sm100 import ring_hd256 as R
    o, lse = R.fwd_block(c["q8"], c["qs"], c["k8"], c["kvw"], c["v8"], causal)
    return o, lse, R.delta_block(*_ops(c), lse, causal)


def _block(B, Nq, Nk, causal, seed):
    """codes of one block with its lse / delta: causal = the last Nq rows of an Nk-token sequence, non-causal = Nq queries vs Nk keys"""
    if causal:
        c = _rows(_codes(*_inputs(B, Nk, Nk, seed)), Nk - Nq, Nk)
    else:
        c = _codes(*_inputs(B, Nq, Nk, seed))
    _, lse, delta = _fwd_delta(c, causal)
    return c, lse, delta


def _bwd_variants(c, lse, delta, causal):
    """bwd_block outputs: bf16 dq / dk / dv (written) and fp32 dq / dk / dv (+= onto fixed nonzero content)"""
    from delta_matching.sm100 import ring_hd256 as R
    bf = R.bwd_block(*_ops(c), lse, delta, causal)
    g = torch.Generator(device="cuda").manual_seed(1)
    f32 = [torch.randn(t.shape, device="cuda", generator=g) for t in bf]
    R.bwd_block(*_ops(c), lse, delta, causal, dq=f32[0], dk=f32[1], dv=f32[2])
    return list(bf) + f32


def _same(a, b):
    a, b = a.contiguous(), b.contiguous()
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def _rel(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm().clamp_min(1e-300)).item()


def _spy_kv2(monkeypatch):
    """records the opts string of every KV2 launch (ring_hd256 and the key-range path call bwd_hd256kv2.kv2_launch)"""
    from delta_matching.sm100 import bwd_hd256kv2
    seen, orig = [], bwd_hd256kv2.kv2_launch

    def spy(*a, **kw):
        seen.append(kw.get("opts") or "")
        return orig(*a, **kw)
    monkeypatch.setattr(bwd_hd256kv2, "kv2_launch", spy)
    return seen


def _cap_for_key_ranges(B, Nq, Nk, causal, min_ranges=3):
    """(cap GiB, plan): the largest cap on a fine geometric grid under which bwd_block plans exact key ranges (all heads, ascending,
    covering the block) with at least min_ranges ranges; None if there is none."""
    from delta_matching.sm100 import ring_hd256 as R
    n_q, n_k = Nq // 128, Nk // 128
    with _env():
        top = R.plan_bytes(B, H, HK, n_q, n_k, causal, [(0, HK, 0, n_k)]) / 2 ** 30
    for i in range(600):
        cap = top * 0.98 ** i
        with _env(DM_BWD_KRANGE=1, DM_BWD_WS_LIMIT_GB=repr(cap)):
            p = R.plan(B, H, HK, n_q, n_k, causal)
        if p is None:
            return None
        if all(c[:2] == (0, HK) for c in p) and len(p) >= min_ranges:
            assert p[0][2] == 0 and p[-1][3] == n_k and all(a[3] == b[2] for a, b in zip(p, p[1:])), p
            return cap, p
    return None


# --------------------------------------------------------------------------------------------------------- bottom-right causal
BR_CASES = [  # B, Nq, Nk
    (1, 1024, 1024),
    (1, 1024, 2048),
    (2, 1024, 2048),
    (1, 1024, 4096),
    (2, 1024, 4096),
]


@pytest.mark.parametrize("B,Nq,Nk", BR_CASES)
def test_bottom_right_causal_bitwise_square_rows(B, Nq, Nk):
    """A causal block of the last Nq rows against all Nk keys == the square causal dm_attn_func path at Nk (dO zero on the first
    Nk - Nq rows), row for row and bit for bit: O / lse of the rows, delta (delta_block and delta_nd + delta_finish) with the square's
    lse, dQ of the rows and dK / dV of every key with the square's lse / delta."""
    from delta_matching.sm100 import ring_hd256 as R
    from delta_matching.sm100.attention import dm_attn_debug, dm_attn_func
    from delta_matching.sm100.quant import quant_do_delta, quant_qkv2
    a = Nk - Nq
    q, k, v, do = _inputs(B, Nk, Nk, seed=B * Nk + Nq)
    do[:, :a] = 0
    qa, ka, va = (x.clone().requires_grad_() for x in (q, k, v))
    o_sq = dm_attn_func(qa, ka, va, causal=True, mode="matched", deterministic=True)
    o_sq.backward(do)
    dbg = dm_attn_debug(q, k, v, do, causal=True, mode="matched")
    lse, delta = dbg["lse"][..., a:].contiguous(), dbg["delta"][..., a:].contiguous()
    q8, qs, k8, ks, v8, vs, kvw, _ = quant_qkv2(q, k, v, SM)          # the producers dm_attn_func runs: the square's codes
    do8, dos, _ = quant_do_delta(do, None)
    c = _rows(dict(q8=q8, qs=qs, k8=k8, ks=ks, v8=v8, vs=vs, kvw=kvw, do8=do8, dos=dos), a, Nk)

    o_b, lse_b = R.fwd_block(c["q8"], c["qs"], k8, kvw, v8, True)
    assert _same(o_b, o_sq.detach()[:, a:]), "fwd_block O"
    assert _same(lse_b, lse), "fwd_block lse"
    assert _same(R.delta_block(*_ops(c), lse, True), delta), "delta_block"
    nd = torch.zeros(B, H, Nq, 2, device="cuda", dtype=torch.float64)
    R.delta_nd(*_ops(c), lse, True, nd)
    assert _same(R.delta_finish(nd), delta), "delta_nd + delta_finish"
    dq, dk, dv = R.bwd_block(*_ops(c), lse, delta, True)
    assert _same(dq, qa.grad[:, a:]), "bwd_block dq"
    assert _same(dk, ka.grad), "bwd_block dk"
    assert _same(dv, va.grad), "bwd_block dv"
    assert all(x.float().abs().amax().item() > 0 for x in (o_b, dq, dk, dv))
    if a:     # the comparison sees the row alignment: the rows one tile earlier differ
        assert not _same(o_b, o_sq.detach()[:, a - 128:Nk - 128]) and not _same(dq, qa.grad[:, a - 128:Nk - 128])


# ------------------------------------------------------------------------------------------------------- non-causal Nq != Nk
EMU_DIST = {"O": 5e-3, "dQ": 1e-3, "dK": 1.2e-3, "dV": 2e-3}     # rel L2(kernel, emulation): the bounds of test_sm100_hd256.py
FLOOR = {"O": 6e-2, "dQ": 7.5e-2, "dK": 7.5e-2, "dV": 6e-2}     # rel L2(kernel, exact) on Gaussian inputs: the fp8 floor bounds
NC_CASES = [  # B, Nq, Nk
    (1, 1024, 2048),
    (2, 2048, 1024),
    (1, 1152, 2688),
]


def _by_key_squares(fn, q, do, Nk):
    """fn(q', do') of an Nk x Nk non-causal square (exact attention / spec emulation on the block's keys) for an Nq x Nk block: the
    queries padded to a multiple of Nk with zero rows whose dO is zero (they add exact zeros to dK / dV), split into Nk-row squares;
    rows concatenated, dK / dV summed."""
    B, Nq = q.shape[:2]
    n = -(-Nq // Nk) * Nk
    if n > Nq:
        q, do = (torch.cat([x, x.new_zeros(B, n - Nq, *x.shape[2:])], dim=1) for x in (q, do))
    parts = [fn(q[:, i:i + Nk].contiguous(), do[:, i:i + Nk].contiguous()) for i in range(0, n, Nk)]
    out = {name: torch.cat([p[name] for p in parts], dim=1)[:, :Nq] for name in ("O", "dQ")}
    out["LSE"] = torch.cat([p["LSE"] for p in parts], dim=-1)[..., :Nq]
    out.update({name: sum(p[name] for p in parts) for name in ("dK", "dV")})
    if "delta" in parts[0]:
        out["delta"] = torch.cat([p["delta"] for p in parts], dim=-1)[..., :Nq]
    return out


@pytest.mark.parametrize("B,Nq,Nk", NC_CASES)
def test_noncausal_rect_vs_exact_and_spec_emulation(B, Nq, Nk):
    """Non-causal Nq != Nk block (fwd_block, delta_block, bwd_block): every output at most 1% (+1e-5) further from exact fp64 attention
    than the spec's torch emulation, within the test_sm100_hd256.py distance of the emulation and the Gaussian fp8 floor; LSE equals
    the emulation's to fp32 round-off; delta_nd + delta_finish of the single block is bitwise delta_block."""
    from delta_matching import reference as REF
    from delta_matching.sm100 import ring_hd256 as R
    q, k, v, do = _inputs(B, Nq, Nk, seed=Nq + Nk + B)
    c = _codes(q, k, v, do)
    o, lse, delta = _fwd_delta(c, False)
    nd = torch.zeros(B, H, Nq, 2, device="cuda", dtype=torch.float64)
    R.delta_nd(*_ops(c), lse, False, nd)
    assert _same(R.delta_finish(nd), delta), "delta_nd + delta_finish"
    dq, dk, dv = R.bwd_block(*_ops(c), lse, delta, False)
    em = _by_key_squares(lambda qq, dd: REF.dm_reference(qq, k, v, dd, False, "matched", "sm100-mx-d256", q_block=512,
                                                         k_mean=c["mean"]), q, do, Nk)
    ex = _by_key_squares(lambda qq, dd: REF.attention_fp32(qq, k, v, dd, False, dtype=torch.float64), q, do, Nk)
    for name, got in zip(("O", "dQ", "dK", "dV"), (o, dq, dk, dv)):
        assert torch.isfinite(got.float()).all(), name
        err, emu_err = _rel(got, ex[name]), _rel(em[name].bfloat16(), ex[name])
        assert err <= 1.01 * emu_err + 1e-5, f"{name}: kernel err {err:.4e} vs spec emulation {emu_err:.4e}"
        dist = _rel(got, em[name].bfloat16())
        assert dist <= EMU_DIST[name], f"{name}: kernel vs spec emulation rel L2 {dist:.3e} > {EMU_DIST[name]:.1e}"
        assert err <= FLOOR[name], f"{name}: err {err:.4e} above the fp8 floor bound {FLOOR[name]}"
    el = em["LSE"].double()
    assert ((lse.double() - el).abs() <= 1e-5 + 1e-6 * el.abs()).all()
    dist = _rel(delta, em["delta"])      # fp32 round-off level (measured <= 2.4e-7)
    assert dist <= 1e-6, f"delta_block vs emulation delta rel L2 {dist:.3e}"


# ------------------------------------------------------------------------------------------------------------ fp32 accumulate
ACC_CASES = [  # B, Nq, Nk, causal
    (1, 1024, 2048, True),
    (2, 1024, 4096, True),
    (2, 2048, 1024, False),
]


@pytest.mark.parametrize("B,Nq,Nk,causal", ACC_CASES)
def test_fp32_accumulators(B, Nq, Nk, causal):
    """fp32 dk / dv (the ring's mode, with bf16 dq) and fp32 dq / dk / dv: bf16 of one call into zeros == the bf16 path bitwise; a
    second call into the same buffers == 2x one call; nonzero initial content gets exactly one fp32 add."""
    from delta_matching.sm100 import ring_hd256 as R
    c, lse, delta = _block(B, Nq, Nk, causal, seed=B + Nq + 3 * Nk)
    ops = _ops(c)
    bf = R.bwd_block(*ops, lse, delta, causal)

    def zeros(t):
        return torch.zeros(t.shape, device="cuda", dtype=torch.float32)
    dk1, dv1 = zeros(bf[1]), zeros(bf[2])
    dq_bf = R.bwd_block(*ops, lse, delta, causal, dk=dk1, dv=dv1)[0]
    assert _same(dq_bf, bf[0]), "bf16 dq next to fp32 dk / dv"
    assert _same(dk1.bfloat16(), bf[1]) and _same(dv1.bfloat16(), bf[2]), "bf16(fp32 dk / dv) != bf16 dk / dv"
    one = [zeros(t) for t in bf]
    R.bwd_block(*ops, lse, delta, causal, dq=one[0], dk=one[1], dv=one[2])
    for name, x, y in zip(OUT_NAMES, one, bf):
        assert _same(x.bfloat16(), y), f"bf16(fp32 {name}) != bf16 {name}"
    assert _same(one[1], dk1) and _same(one[2], dv1), "fp32 dk / dv depend on the dq dtype"
    two = [x.clone() for x in one]
    R.bwd_block(*ops, lse, delta, causal, dq=two[0], dk=two[1], dv=two[2])
    for name, x, y in zip(OUT_NAMES, two, one):
        assert _same(x, 2 * y), f"{name}: two += calls != 2x one call"
    g = torch.Generator(device="cuda").manual_seed(5)
    init = [torch.randn(t.shape, device="cuda", generator=g) for t in one]
    acc = [x.clone() for x in init]
    R.bwd_block(*ops, lse, delta, causal, dq=acc[0], dk=acc[1], dv=acc[2])
    for name, x, i, y in zip(OUT_NAMES, acc, init, one):
        assert _same(x, i + y), f"{name}: += onto nonzero content is not one fp32 add"


# ---------------------------------------------------------------------------------------------------------------- key ranges
KR_CASES = [  # B, Nq, Nk, causal
    (2, 1024, 4096, True),
    (1, 1024, 8192, True),
    (2, 1024, 4096, False),
]


@pytest.mark.parametrize("B,Nq,Nk,causal", KR_CASES)
def test_key_range_plan_bitwise_whole_block(B, Nq, Nk, causal, monkeypatch):
    """DM_BWD_KRANGE=1 with a workspace cap below one KV-head group: bwd_block runs >= 3 exact-carry key ranges (KV2 krg on causal
    blocks + Dqg kr), and dq / dk / dv (bf16 written, fp32 +=) are bitwise the whole block's at the default cap."""
    from delta_matching.sm100 import ring_hd256 as R
    n_q, n_k = Nq // 128, Nk // 128
    c, lse, delta = _block(B, Nq, Nk, causal, seed=7 * B + Nk)
    with _env():
        assert R.plan(B, H, HK, n_q, n_k, causal) == [(0, HK, 0, n_k)]
        ref = _bwd_variants(c, lse, delta, causal)
    found = _cap_for_key_ranges(B, Nq, Nk, causal)
    assert found is not None, "no cap yields a key-range plan"
    cap, chunks = found
    seen = _spy_kv2(monkeypatch)
    with _env(DM_BWD_KRANGE=1, DM_BWD_WS_LIMIT_GB=repr(cap)):
        assert R.plan(B, H, HK, n_q, n_k, causal) == chunks
        out = _bwd_variants(c, lse, delta, causal)
    assert len(seen) == 2 * B * len(chunks), (len(seen), B, chunks)          # one KV2 launch per range and batch element per call
    assert all(("krg" in o.split()) == causal and "rect" in o.split() for o in seen), seen
    for name, x, y in zip(OUT_NAMES, out, ref):
        assert _same(x, y), f"{name}: key ranges {[ch[2:] for ch in chunks]} (cap {cap:.4f} GiB) differ from the whole block"


# ---------------------------------------------------------------------------------------------------------------------- dmk
DMK_CASES = [  # B, Nq, Nk, key ranges
    (1, 1024, 1024, False),
    (2, 1024, 2048, False),
    (1, 1024, 4096, False),
    (2, 1024, 4096, True),
]


@pytest.mark.parametrize("B,Nq,Nk,krange", DMK_CASES)
def test_dmk_peel_bitwise(B, Nq, Nk, krange, monkeypatch):
    """Causal blocks (square, bottom-right, and a key-range plan): DM_RING_DMK=1 (KV2 opt dmk, the default) and DM_RING_DMK=0 give
    bitwise the same bf16 and fp32 dq / dk / dv; the knob reaches every KV2 launch."""
    c, lse, delta = _block(B, Nq, Nk, True, seed=11 * B + Nk)
    knobs = {}
    if krange:
        found = _cap_for_key_ranges(B, Nq, Nk, True)
        assert found is not None, "no cap yields a key-range plan"
        knobs = dict(DM_BWD_KRANGE=1, DM_BWD_WS_LIMIT_GB=repr(found[0]))
    seen = _spy_kv2(monkeypatch)
    out = {}
    for dmk in ("1", "0"):
        seen.clear()
        with _env(DM_RING_DMK=dmk, **knobs):
            out[dmk] = _bwd_variants(c, lse, delta, True)
        assert seen and all(("dmk" in o.split()) == (dmk == "1") for o in seen), (dmk, seen)
        assert all(("krg" in o.split()) == krange for o in seen), seen
    for name, x, y in zip(OUT_NAMES, out["1"], out["0"]):
        assert _same(x, y), f"{name}: DM_RING_DMK=1 differs from DM_RING_DMK=0"


# ------------------------------------------------------------------------------------------------------------------ identity
def _kv2_identity(c, lse, delta, causal):
    """per query row (B, H, Nq): fp64 sums of sum_k dS and sum_k |P dP| over the block's keys, from the dbg-16 build (same arithmetic,
    compiled-in sums of the pre-quantization fp32 dS) of the KV2 launch ring_hd256.bwd_block runs for this block (whole-block plan:
    opts dqt, + rect when Nq != Nk, + dmk on causal blocks at the default DM_RING_DMK)."""
    from delta_matching.sm100 import bwd_hd256kv2
    B, Nq, Hq, _ = c["q8"].shape
    n_q, n_k = Nq // 128, c["k8"].shape[1] // 128
    opts = bwd_hd256kv2.DmHd256Kv2.DEFAULT_OPTS + " dqt" + (" rect" if n_k != n_q else "") + (" dmk" if causal else "")
    tiles = bwd_hd256kv2.dqw_tiles(B, Hq, n_q, causal, n_k)
    ws = (torch.empty(tiles, 128, 128, device="cuda", dtype=torch.uint8), torch.empty(tiles, 1, 4, device="cuda", dtype=torch.uint8))
    sums = bwd_hd256kv2.kv2_launch(*_ops(c), lse, delta, causal, dbg=16, opts=opts, dqw=ws)[2]["hash"]
    sums = sums.view(torch.float64).view(B, Hq, Nq, 4)
    return sums[..., 0], sums[..., 1]


ID_CASES = [  # B, chunk c, family
    (1, 1024, "randn"),
    (1, 1024, "ramp"),
    (2, 1024, "growmax"),
    (1, 2048, "randn"),
]


@pytest.mark.parametrize("B,c,family", ID_CASES)
def test_zero_sum_identity_across_ring_chunks(B, c, family):
    """A causal sequence of N = 2c tokens as the ring decomposition: chunk j (rows [j c, (j + 1) c)) attends keys [0, (j + 1) c) as one
    causal block (chunk 0 a square, chunk 1 bottom-right), lse from fwd_block, matched delta from delta_block.  On every row of both
    chunks max |sum_k dS| / sum_k |P dP| <= 1e-5 and its mean <= 2e-7 (test_sm100_hd256.py's identity bounds), from
      the kernel   the backward's own pre-quantization dS (KV2 dbg-16 build of the chunk's bwd_block launch); a delta off by 2^-10
                   relative fails the bounds (the check is live);
      the spec     the torch emulation (reference.dm_reference, aux rowsum_dS) of the whole sequence with the chunks' kernel lse / delta
                   (causal rows of the square see exactly their chunk's keys): P / dP of the spec, delta of the kernel."""
    from delta_matching import reference as REF
    N = 2 * c
    q, k, v, do = _inputs(B, N, N, seed=13 + B, family=family)
    full = _codes(q, k, v, do)           # one K mean over the whole sequence (the ring's global mean)
    rel, ctl, lses, deltas = [], [], [], []
    for j in range(2):
        blk = _keys(_rows(full, j * c, (j + 1) * c), (j + 1) * c)
        _, lse, delta = _fwd_delta(blk, True)
        assert torch.isfinite(lse).all() and torch.isfinite(delta).all()
        lses.append(lse)
        deltas.append(delta)
        rowsum, den = _kv2_identity(blk, lse, delta, True)
        live = den > 0
        assert live.float().mean().item() > 0.99, f"chunk {j}: {live.float().mean().item():.3f} live rows"
        rel.append((rowsum.abs() / den.clamp_min(1e-300))[live])
        rowsum2, den2 = _kv2_identity(blk, lse, delta * (1 + 2 ** -10), True)
        ctl.append((rowsum2.abs() / den2.clamp_min(1e-300))[den2 > 0])
    rel, ctl = torch.cat(rel), torch.cat(ctl)
    assert rel.max().item() <= 1e-5 and rel.mean().item() <= 2e-7, ("kernel", rel.max().item(), rel.mean().item())
    assert ctl.mean().item() > 1e-5, ctl.mean().item()
    em = REF.dm_reference(q, k, v, do, True, "matched", "sm100-mx-d256", q_block=512, aux=True, k_mean=full["mean"],
                          lse_override=torch.cat(lses, -1), delta_override=torch.cat(deltas, -1))
    den = em["sum_absPdP"].double()
    live = den > 0
    assert live.float().mean().item() > 0.99
    rel = (em["rowsum_dS"].double().abs() / den.clamp_min(1e-300))[live]
    assert rel.max().item() <= 1e-5 and rel.mean().item() <= 2e-7, ("spec emulation", rel.max().item(), rel.mean().item())
