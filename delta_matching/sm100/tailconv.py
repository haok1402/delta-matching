"""rw8k tconv: tail converter = FA4's backward postprocess (fp32 accumulator * scale -> bf16), launched with PDL right
after the deterministic backward main kernel, each CTA converting its tile as soon as the tile is FINAL.

Readiness (no new hand-off: the det dQ / dK / dV lock counters the backward already maintains):
  dQ tile (b, h, m), slot r of the 2-CTA pair: final count = number of KV cluster tiles that write it
      = min(ceil(N / ctn), ceil((m + 1) * tile_m / ctn)) causal, ceil(N / ctn) otherwise   (ctn = cluster tile n = 256)
  GQA dK / dV tile (b, hk, n), slot r: final count = G (query heads per KV head)
  (verified on the hardware: kda/tests/rw8k/semdump.py, 8k / 1152 / 4k non-causal / 2k MHA).
Protocol per converter CTA: thread 0 polls every slot of its tile with ld.acquire.gpu until >= final (bounded: trap after
~2^26 polls, never a silent hang) -> bar.sync (CTA) -> every thread fence.proxy.async.global (the writers' bulk reduce-adds
are async-proxy writes) -> FA4's unchanged conversion code -> griddepcontrol.wait (deferred, before exit).
The writers (backward, tconv knob) execute fence.proxy.async.global before every det semaphore release, and the backward
issues griddepcontrol.launch_dependents after its TMEM allocation (the converter needs no TMEM).
Semaphore reset / ABA: the counters are zeroed per call by the backward host path (prestage side stream, which waits on the
main stream, i.e. after the previous call's converter) before the next backward main kernel; the converter only reads.
Controls (DM_TCONV_CTL, debug): early = poll for final - 1 (reads before the last writer released -> wrong bits, caught);
drop = poll for final + 1 (the flag never arrives -> bounded wait traps).
Same arithmetic and same loads/stores as FA4's postprocess: dQ / dK / dV bitwise equal to the serial postprocess."""
import inspect
import os

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, const_expr

import flash_attn.cute.flash_bwd_postprocess as _fpp
from . import dm_utils

_SRC = inspect.getsource(_fpp)


def _patch(src: str) -> str:
    subs = [
        ("class FlashAttentionBackwardPostprocess:", "class DMTailConv:"),
        # __call__: two extra arguments (readiness counters, kind) before the stream
        ("        mCuTotalMBlocks: Optional[cute.Tensor] = None,\n        # Always keep stream as the last parameter",
         "        mCuTotalMBlocks: Optional[cute.Tensor] = None,\n        mDmSem: Optional[cute.Tensor] = None,\n"
         "        # Always keep stream as the last parameter"),
        ("            tile_sched_params,\n            TileScheduler,\n        ).launch(\n            grid=grid_dim,\n"
         "            block=[self.num_threads, 1, 1],\n            smem=smem_size,\n            stream=stream,\n        )",
         "            tile_sched_params,\n            TileScheduler,\n            mDmSem,\n        ).launch(\n            grid=grid_dim,\n"
         "            block=[self.num_threads, 1, 1],\n            smem=smem_size,\n            stream=stream,\n"
         "            use_pdl=True,\n        )"),
        ("        tile_sched_params: ParamsBase,\n        TileScheduler: cutlass.Constexpr[Callable],\n    ):\n"
         "        # ///////////////////////////////////////////////////////////////////////////////\n"
         "        # Get shared memory buffer",
         "        tile_sched_params: ParamsBase,\n        TileScheduler: cutlass.Constexpr[Callable],\n"
         "        mDmSem: Optional[cute.Tensor],\n    ):\n"
         "        # ///////////////////////////////////////////////////////////////////////////////\n"
         "        # Get shared memory buffer"),
        ("        m_block, head_idx, batch_idx, _ = work_tile.tile_idx\n",
         "        m_block, head_idx, batch_idx, _ = work_tile.tile_idx\n"
         "        cute.arch.griddepcontrol_launch_dependents()   # rw8k: the next converter (dK / dV) may launch now\n"
         "        if work_tile.is_valid_tile:\n"
         "            self._dm_ready(mDmSem, m_block, head_idx, batch_idx, tidx, mdQ.shape[1])\n"),
    ]
    for a, b in subs:
        if src.count(a) != 1:
            raise RuntimeError(f"tailconv patch anchor not unique/found ({src.count(a)}): {a[:80]!r}")
        src = src.replace(a, b)
    # deferred griddepcontrol.wait at the very end of the kernel (the module ends with the kernel body)
    src = src.rstrip("\n") + "\n        cute.arch.griddepcontrol_wait()\n"
    return src


_ns = dict(vars(_fpp))
_FN = _fpp.__file__.replace(".py", "_dm_tailconv.py")
_PSRC = _patch(_SRC)
import linecache  # noqa: E402
linecache.cache[_FN] = (len(_PSRC), None, _PSRC.splitlines(True), _FN)   # the DSL's AST pass re-reads the source
exec(compile(_PSRC, _FN, "exec"), _ns)
_Base = _ns["DMTailConv"]


class DMTailConv(_Base):
    """kind 'q': dQ (tile_m rows, causal-dependent final count); kind 'kv': GQA dK / dV (final count = G)."""

    def __init__(self, *a, dm_kind="q", dm_causal=True, dm_ctn=256, dm_G=1, dm_ctl="", **k):
        super().__init__(*a, **k)
        self.dm_kind, self.dm_causal, self.dm_ctn, self.dm_G, self.dm_ctl = dm_kind, dm_causal, dm_ctn, dm_G, dm_ctl

    @cute.jit
    def _dm_ready(self, mDmSem, m_block, head_idx, batch_idx, tidx, seqlen):
        if const_expr(mDmSem is not None):
            if const_expr(self.dm_kind == "q"):
                nk = (seqlen + self.dm_ctn - 1) // self.dm_ctn
                fin = nk
                if const_expr(self.dm_causal):
                    fin = cutlass.min(nk, ((m_block + 1) * self.tile_m + self.dm_ctn - 1) // self.dm_ctn)
            else:
                fin = Int32(self.dm_G)
            if const_expr(self.dm_ctl == "early"):
                fin = fin - 1          # CONTROL: reads before the last writer released (must be caught by the bit check)
            if const_expr(self.dm_ctl == "drop"):
                fin = fin + 1          # CONTROL: the flag never arrives (must trap, not hang)
            if tidx == 0:
                nh = cute.size(mDmSem, mode=[1])
                nt = cute.size(mDmSem, mode=[2])
                ns = cute.size(mDmSem, mode=[3])
                base = mDmSem.iterator.toint() + Int64(4) * Int64(((batch_idx * nh + head_idx) * nt + m_block) * ns)
                for r in cutlass.range_constexpr(cute.size(mDmSem, mode=[3])):
                    dm_utils.poll_acquire_ge_or_trap(base + Int64(4 * r), fin, Int32(1 << 26))
            cute.arch.barrier()
            dm_utils.fence_proxy_async_global()


def _cache():
    from flash_attn.cute import cache_utils
    from .fwd import _fingerprint
    if cache_utils.CUTE_DSL_CACHE_ENABLED:
        return cache_utils.JITPersistentCache(cache_utils.get_cache_path() / "dm_sm100" / _fingerprint() / "tconv")
    return cache_utils.JITCache()


_CACHE = None


def _compile(key):
    """Fake-tensor compile (mirrors interface._compile_bwd_postprocess, dense only) + the readiness tensor."""
    from flash_attn.cute.interface import make_fake_bwd_tensors
    (dtype, hdim, block_size, num_threads, atom_layout, swap_ab, use_2cta_instrs, cluster_size, arch, hdim_multiple_of,
     accum_order, dm_kind, dm_causal, dm_ctn, dm_G, ctl) = key
    mQ, mK, mV, mO, mdO, mdQ, mdK, mdV, mLSE, mLSElog2, mPdPsum, mdQaccum, mdKaccum, mdVaccum, mScaleP = \
        make_fake_bwd_tensors(dtype, has_gqa=True, varlen_q=False, varlen_k=False)
    from quack.compile_utils import make_fake_tensor as _ft
    mSem = _ft(Int32, (cute.sym_int(), cute.sym_int(), cute.sym_int(), 2), divisibility=1)
    conv = DMTailConv(dtype, hdim, arch, block_size, num_threads, atom_layout, swap_ab, use_2cta_instrs=use_2cta_instrs,
                      cluster_size=cluster_size, hdim_multiple_of=hdim_multiple_of, accum_order=accum_order,
                      pack_factor=1, dm_kind=dm_kind, dm_causal=dm_causal, dm_ctn=dm_ctn, dm_G=dm_G, dm_ctl=ctl)
    return cute.compile(conv, mdQaccum, mdQ, cutlass.Float32(0.0), None, None, None, None, mSem,
                        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True), options="--enable-tvm-ffi")


def tconv_convert(sem, accum, output, scale, cu_seqlens, seqused, arch, dtype, hdim, block_size, num_threads,
                  atom_layout, swap_ab, use_2cta_instrs=False, cluster_size=1, cu_total_m_blocks=None,
                  sink_tensors=None, hdim_multiple_of=32, accum_order=None, pack_factor=1, *, fake_mode,
                  dm_kind, dm_causal, dm_ctn, dm_G):
    """Drop-in for interface._bwd_postprocess_convert with the readiness semaphore and the kind / counts."""
    global _CACHE
    assert cu_seqlens is None and seqused is None and sink_tensors is None and pack_factor == 1, "tconv: dense only"
    assert cu_total_m_blocks is None
    if _CACHE is None:
        _CACHE = _cache()
    key = (dtype, hdim, block_size, num_threads, atom_layout, swap_ab, use_2cta_instrs, cluster_size, arch,
           hdim_multiple_of, accum_order, dm_kind, bool(dm_causal), int(dm_ctn), int(dm_G),
           os.environ.get("DM_TCONV_CTL", ""))
    if key not in _CACHE:
        _CACHE[key] = _compile(key)
    if not fake_mode:
        _CACHE[key](accum, output, scale, None, None, None, None, sem)
