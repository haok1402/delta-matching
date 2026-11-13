"""
Ring context-parallel block entry points at head_dim 256 (sm100-mx-d256, deterministic).

A block attends one query range (Nq) to one key range (Nk). Causal blocks are bottom-right aligned (FA4 convention:
query i sees keys <= i + Nk - Nq, Nk - Nq >= 0 a multiple of 128): with the K/V of all ranks in global token order, the
local chunk j attends [0, (j+1) c) as ONE causal block. Non-causal blocks may have any Nq, Nk (several sources' keys
concatenated on the token axis). Codes and scales are the quant.py producers' (e4m3, one pow2 scale per 32 tokens).

fwd_block    (o bf16, lse) of the block (natural-log lse over the block's keys).
delta_block  delta (B, H, Nq) of a block that holds ALL keys of its rows (pass 0, num/den).
delta_nd     nd (B, H, Nq, 2) fp64 += the block's per-row [num, den] (merged lse), delta_finish(nd) after the last
             block (split-key fallback; bitwise pass 0's own divide for a single block).
bwd_block    dQ of the block: bf16 dq (written) or fp32 dq (+=); dK / dV (B, Nk, Hk, 256) bf16 of the block
             (GQA-folded), written into caller tensors when given (e.g. a send-buffer slot).
All accumulating launches must run on one stream in a fixed order (the accumulators are read-add-write, no atomics).
"""
import torch

from .fwd_hd256 import dm_fwd_hd256_mx
from .pass0_hd256 import delta_nd_hd256, delta_block_hd256, delta_finish  # noqa: F401
from . import bwd_hd256kv2
from . import bwd_hd256mx
from .dm_bwd_hd256dqg_sm100 import DmHd256Dqg
from .attention_bwd_hd256_sm100 import workspace_limit_bytes, TILE_BYTES, DQG_STAGES
from .attention_bwd_hd256_sm100 import key_range_chunks, key_range_bytes, key_ranges_setting, backward_key_ranges
from .launch_cache_sm100 import env

D = 256


def fwd_block(q8, qs, k8, kvw, v8, causal, out=None):
    """(o (B, Nq, H, 256) bf16, lse (B, H, Nq) fp32) of one block. out = (o, lse) caller buffers."""
    rect = k8.shape[1] != q8.shape[1]
    return dm_fwd_hd256_mx(q8, qs, k8, kvw, v8, causal, rect=rect, out=out)


def delta_block(q8, qs, k8, ks, v8, vs, do8, dos, lse, causal):
    """matched delta (B, H, Nq) fp32 of a block that holds all keys of its rows."""
    return delta_block_hd256(q8, qs, k8, ks, v8, vs, do8, dos, lse, causal)


def delta_nd(q8, qs, k8, ks, v8, vs, do8, dos, lse, causal, nd):
    """nd (B, H, Nq, 2) fp64 += this block's [num, den] per query row (lse: the merged lse of the rows)."""
    return delta_nd_hd256(q8, qs, k8, ks, v8, vs, do8, dos, lse, causal, nd)


def dqg_launch(W, Wsb, k8, ks, B, Nq, H, causal, dq, n_k=None):
    """dQ from the workspace: dq fp32 -> dq += dS K x 2^-4; dq bf16 -> written (rect when n_k != Nq / 128)."""
    Hk = k8.shape[2]
    f8 = torch.float8_e4m3fn
    acc = dq.dtype == torch.float32
    assert dq.shape == (B, Nq, H, D) and dq.dtype in (torch.float32, torch.bfloat16) and dq.is_contiguous()
    rect = n_k is not None and n_k != Nq // 128
    mA = W.view(f8).permute(2, 1, 0)
    mKt = k8.view(f8).permute(3, 1, 2, 0)
    key = ("dqg", bool(causal), DQG_STAGES, "", True) + (("acc",) if acc else ()) + (("rect",) if rect else ())
    bwd_hd256mx._run(key, lambda: DmHd256Dqg(bool(causal), DQG_STAGES, "", True, False, rect=rect, acc=acc),
                     [mA, mKt, Wsb, ks, dq], [0, 0, 2, 2, 3], [Nq // 128, H, Hk, n_k if rect else 0], mode="dqg")
    return dq


def _chunk_bytes(B, H, Hk, n_q, n_k, causal, chunk, direct, kv_bytes, dq_bytes):
    """Device bytes a chunk needs besides the caller's tensors: its dS workspace (W + Wsb) and, when it cannot write
    the caller's dK / dV / dQ directly, its temporaries (dK + dV of its keys x KV heads, dQ of its q heads)."""
    g0, g1, k0, k1 = chunk
    rep = H // Hk
    nb = bwd_hd256kv2.dqw_tiles(B, (g1 - g0) * rep, n_q, causal, k1 - k0) * TILE_BYTES
    if not direct:
        nb += 2 * B * (k1 - k0) * 128 * (g1 - g0) * D * kv_bytes
    if (g0, g1) != (0, Hk):
        nb += B * n_q * 128 * (g1 - g0) * rep * D * dq_bytes
    return nb


def _direct(B, Hk, n_k, chunk):
    g0, g1, k0, k1 = chunk
    return (g0, g1) == (0, Hk) and (B == 1 or (k0, k1) == (0, n_k))     # key-range slices are contiguous at B == 1


def plan_bytes(B, H, Hk, n_q, n_k, causal, chunks, kv_bytes=2, dq_bytes=2):
    """Peak extra device bytes of bwd_block for this plan (one workspace sized for the largest chunk, reused; one
    chunk's temporaries live at a time). Excludes the caller's inputs / outputs."""
    if _exact_ranges(Hk, n_k, chunks):
        return key_range_bytes(H, n_q, n_k, causal, [(c[2], c[3]) for c in chunks])
    ws = max(bwd_hd256kv2.dqw_tiles(B, (g1 - g0) * (H // Hk), n_q, causal, k1 - k0) for g0, g1, k0, k1 in chunks) * TILE_BYTES
    tmp = max(_chunk_bytes(B, H, Hk, n_q, n_k, causal, c, _direct(B, Hk, n_k, c), kv_bytes, dq_bytes)
              - bwd_hd256kv2.dqw_tiles(B, (c[1] - c[0]) * (H // Hk), n_q, causal, c[3] - c[2]) * TILE_BYTES for c in chunks)
    return ws + tmp


def plan(B, H, Hk, n_q, n_k, causal, cap=None, kv_bytes=2, dq_bytes=2):
    """Chunks (g0, g1, k0, k1) in tiles: KV-head groups [g0, g1) x key tiles [k0, k1), ascending (= the unchunked
    reduction order of dQ; dK / dV of a key are formed in exactly one chunk). Preference: whole block; one KV group per
    chunk (bitwise the whole block).  DM_RING_KEYRANGE=1 (non-causal only): key ranges (with all heads at B == 1, else
    per KV group), dQ = fp32 sum of the ranges' partials (deterministic, not bitwise the whole block).
    The cap DM_BWD_WS_LIMIT_GB bounds plan_bytes(): workspace AND the chunk temporaries (kv_bytes / dq_bytes = element
    size of the caller's dK / dV and dQ)."""
    cap = workspace_limit_bytes() if cap is None else cap
    rep = H // Hk

    def fits(chunks):
        return plan_bytes(B, H, Hk, n_q, n_k, causal, chunks, kv_bytes, dq_bytes) <= cap

    for chunks in ([(0, Hk, 0, n_k)], [(g, g + 1, 0, n_k) for g in range(Hk)]):
        if fits(chunks):
            return chunks
    if key_ranges_setting():
        # DM_BWD_KRANGE=1: key ranges with all heads and the EXACT dQ carry (bitwise the whole block; KV2 opt krg + Dqg
        # opt kr, attention_bwd_hd256_sm100.backward_key_ranges), causal and non-causal, any dq / dk / dv dtype; replaces
        # the DM_RING_KEYRANGE sum-of-partials ranges. Workspace (largest range) + fp32 dQ carry <= the cap.
        ranges = key_range_chunks(B, H, n_q, n_k, causal, cap)
        return None if ranges is None else [(0, Hk, k0, k1) for k0, k1 in ranges]
    if causal or env("DM_RING_KEYRANGE", "0") != "1":
        return None
    for nh in ((Hk,) if B == 1 else ()) + (1,):            # key ranges with all heads at B == 1, then per KV group
        for per in range(n_k, 0, -1):
            chunks = [(g, g + nh, k0, min(n_k, k0 + per)) for g in range(0, Hk, nh) for k0 in range(0, n_k, per)]
            if fits(chunks):
                return chunks
    return None


def _exact_ranges(Hk, n_k, chunks):
    """the plan is DM_BWD_KRANGE exact key ranges (all heads, at least one range shorter than the block)."""
    return key_ranges_setting() and all(c[:2] == (0, Hk) for c in chunks) and any(c[2:] != (0, n_k) for c in chunks)


def _aligned16(t):
    return t.data_ptr() % 16 == 0


def bwd_block(q8, qs, k8, ks, v8, vs, do8, dos, lse, delta, causal, dq=None, dk=None, dv=None):
    """(dq, dk, dv) of the block. dq: None / bf16 (B, Nq, H, 256) -> written; fp32 -> dq += this block's dQ.
    dk / dv (B, Nk, Hk, 256), contiguous caller tensors when given: bf16 -> written (rounded once from the fp32 TMEM
    accumulator); fp32 -> dk += / dv += this block's fp32 dK / dV (KV2 dkacc: read-add-write, deterministic).
    Caller tensors must be 16-byte aligned (vector epilogues). lse / delta: of the rows (merged / global).
    Extra device memory: plan_bytes(...) <= DM_BWD_WS_LIMIT_GB (workspace + chunk temporaries)."""
    B, Nq, H, Dq = q8.shape
    Nk, Hk = k8.shape[1], k8.shape[2]
    rep = H // Hk
    assert Dq == D and H % Hk == 0 and Nq % 128 == 0 and Nk % 128 == 0 and (not causal or Nk >= Nq)
    dev = q8.device
    if dq is None:
        dq = torch.empty(B, Nq, H, D, device=dev, dtype=torch.bfloat16)
    assert dq.shape == (B, Nq, H, D) and dq.dtype in (torch.float32, torch.bfloat16) and dq.is_contiguous()
    acc = dq.dtype == torch.float32
    n_q, n_k = Nq // 128, Nk // 128
    if dk is None:
        dk = torch.empty(B, Nk, Hk, D, device=dev, dtype=torch.bfloat16)
    if dv is None:
        dv = torch.empty(B, Nk, Hk, D, device=dev, dtype=torch.bfloat16)
    assert dk.shape == (B, Nk, Hk, D) and dv.shape == dk.shape and dk.is_contiguous() and dv.is_contiguous()
    assert dk.dtype == dv.dtype and dk.dtype in (torch.float32, torch.bfloat16)
    assert _aligned16(dq) and _aligned16(dk) and _aligned16(dv), "bwd_block: dq / dk / dv must be 16-byte aligned"
    kacc = dk.dtype == torch.float32
    chunks = plan(B, H, Hk, n_q, n_k, causal, kv_bytes=dk.element_size(), dq_bytes=dq.element_size())
    assert chunks is not None, f"ring bwd_block: no chunk fits DM_BWD_WS_LIMIT_GB ({B}, {H}, {Hk}, {Nq}, {Nk}, {causal})"
    if _exact_ranges(Hk, n_k, chunks):
        return backward_key_ranges(q8, qs, k8, ks, v8, vs, do8, dos, lse, delta, causal, [c[2:] for c in chunks], dq, dk, dv)
    assert acc or all(c[2:] == (0, n_k) for c in chunks), "key-range chunks need an fp32 dq (accumulate)"
    ws_tiles = max(bwd_hd256kv2.dqw_tiles(B, (g1 - g0) * rep, n_q, causal, k1 - k0) for g0, g1, k0, k1 in chunks)
    W_all = torch.empty(ws_tiles, 128, 128, device=dev, dtype=torch.uint8)     # one workspace, reused by every chunk
    Wsb_all = torch.empty(ws_tiles, 1, 4, device=dev, dtype=torch.uint8)       # (stream order covers the reuse)
    for g0, g1, k0, k1 in chunks:
        nk_c = k1 - k0
        rect = nk_c != n_q
        opts = bwd_hd256kv2.DmHd256Kv2.DEFAULT_OPTS + " dqt" + (" rect" if rect else "")
        heads = slice(g0 * rep, g1 * rep)
        keys = slice(k0 * 128, k1 * 128)
        whole_heads = (g0, g1) == (0, Hk)
        nt = bwd_hd256kv2.dqw_tiles(B, (g1 - g0) * rep, n_q, causal, nk_c)
        W, Wsb = W_all[:nt], Wsb_all[:nt]
        kq = (k8[:, keys, g0:g1], ks[:, g0:g1, k0 * 4:k1 * 4], v8[:, keys, g0:g1], vs[:, g0:g1, k0 * 4:k1 * 4])
        qq = (q8[:, :, heads], qs[:, heads], do8[:, :, heads], dos[:, heads], lse[:, heads], delta[:, heads])
        direct = _direct(B, Hk, n_k, (g0, g1, k0, k1))
        if direct:
            out = (dk[:, keys], dv[:, keys])
            opts = opts + (" dkacc" if kacc else "")
        else:
            out = (torch.empty(B, nk_c * 128, g1 - g0, D, device=dev, dtype=dk.dtype),
                   torch.empty(B, nk_c * 128, g1 - g0, D, device=dev, dtype=dk.dtype))
            opts = opts + (" dkf32" if kacc else "")
        bwd_hd256kv2.kv2_launch(qq[0], qq[1], kq[0], kq[1], kq[2], kq[3], qq[2], qq[3], qq[4], qq[5], causal,
                                opts=opts, dqw=(W, Wsb), out=out)
        if not direct:
            if kacc:
                dk[:, keys, g0:g1] += out[0]
                dv[:, keys, g0:g1] += out[1]
            else:
                dk[:, keys, g0:g1].copy_(out[0])
                dv[:, keys, g0:g1].copy_(out[1])
        del out                                                                  # one chunk's temporaries at a time
        if whole_heads:
            dqg_launch(W, Wsb, kq[0], kq[1], B, Nq, H, causal, dq, n_k=nk_c)
        else:
            tmp = (torch.zeros if acc else torch.empty)(B, Nq, (g1 - g0) * rep, D, device=dev, dtype=dq.dtype)
            dqg_launch(W, Wsb, kq[0], kq[1], B, Nq, (g1 - g0) * rep, causal, tmp, n_k=nk_c)
            if acc:
                dq[:, :, heads] += tmp
            else:
                dq[:, :, heads].copy_(tmp)
            del tmp
    return dq, dk, dv
