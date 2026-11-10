"""
Compile-cache keys of the sm100 kernels: every compile-time input outside FA4's own key (compile_env) and the
persistent-cache fingerprint (FA4 sources, this package's sources, the opts string and compile_env).
"""
import hashlib
from pathlib import Path

from flash_attn.cute import cache_utils

from delta_matching.sm100.launch_cache_sm100 import env
from delta_matching.sm100.tcgen05_fence_sm100 import TCF_IMPL, TCF_MASK

__all__ = ["PACKAGE_DIR", "STATIC_ENV", "compile_env", "fingerprint"]

# every source file of this package is hashed into the fingerprint, so a moved helper can never be served from a stale entry
PACKAGE_DIR = Path(__file__).resolve().parent


STATIC_ENV = []


def compile_env():
    """every compile-time input of the DM kernels that FA4's compile key (dtype / head dims / causal /
    GQA ratio / config incl. registers / deterministic / arch / ...) and the opts string do NOT carry, rendered as one
    string that is hashed into the persistent cache directory (fingerprint) and into the in-process cache memos:
      DM_ABL (bwd ablation knobs, read in DMBackwardSm100.__init__), DM_TCF_SKIP (read while tracing), the tcgen05 fence
      emission settings in effect (DM_TCF_IMPL / DM_TCF_MASK, read at tcgen05_fence_sm100 import), the pass-0 register splits in effect
      (DM_P0_C3REGS / DM_P0_TLREGS, read at pass0 import; also in the config), and the CuTe DSL compile settings (CUTE_DSL_LINEINFO,
      _COMPILER_OPT, _DEBUG, ... = the DSL's own cache_key_str() minus arch, which FA4's key has, and the runtime lib path).
    Opts strings (DM_FWD_OPTS, effective DM_BWD_OPTS incl. b6=, DM_P0_EXP variant) are hashed by the callers."""
    from . import pass0 as _p0
    if not STATIC_ENV:
        try:
            from cutlass.cutlass_dsl import CuTeDSL
            dsl = ";".join(x for x in CuTeDSL._get_dsl().envar.cache_key_str().split(";")
                           if x and not x.startswith(("arch=", "shared_libs=")))
        except Exception as e:  # noqa: BLE001
            dsl = f"unknown:{type(e).__name__}"
        # NOT hashed: the SM count (AOT fakes 148, GB200 reports 152): measured not to change the compiled object (whole .o
        # byte-identical for fwd / pass 0 / bwd), and hashing it would make every GPU run miss
        # the AOT entries (compile inside a lease).
        STATIC_ENV.append(f"dsl={dsl}|tcf_impl={TCF_IMPL}|tcf_mask={TCF_MASK}|"
                           f"p0regs={sorted(_p0.REGS_C3.items())}{sorted(_p0.REGS_TL.items())}")
    return f"{STATIC_ENV[0]}|abl={env('DM_ABL', '')}|tcf_skip={env('DM_TCF_SKIP', '')}"


def fingerprint(opts=""):
    h = hashlib.sha256(cache_utils._compute_source_fingerprint().encode())
    h.update(opts.encode())   # kernel variant knobs (DM_FWD_OPTS, read in DMForwardSm100.__init__)
    h.update(compile_env().encode())   # compile-time inputs outside FA4's key and the opts
    for f in sorted(PACKAGE_DIR.glob("*.py")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:24]
