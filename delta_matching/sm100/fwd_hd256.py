"""Host entry of the hd256 forward ([64,32] fp32 contract, numerics scheme DM_HD256_NUMERICS, default "a").

dm_fwd_hd256(q8, qs, k8, ks, v8, vs, causal) -> (O (B,N,H,256) bf16, LSE (B,H,N) fp32 natural log)
  q8 (B,N,H,256) e4m3 (sm = 2^-4 baked), k8 / v8 (B,N,Hk,256) e4m3; qs (B,H,N/64), ks / vs (B,Hk,N/32) fp32
  (quant_hd256.quant_qkv_hd256). N % 128 == 0.
Compile cache: persistent (FLASH_ATTENTION_CUTE_DSL_CACHE_DIR) under a fingerprint of this file and the kernel file;
the key depends only on (causal, H/Hk, kv_stage, numerics) - shapes are dynamic. Under torch FakeTensorMode the call
only compiles (AOT, no GPU) and returns empty fakes.
"""
import hashlib
import os
from pathlib import Path

import torch
import cutlass.cute as cute
from flash_attn.cute import cache_utils
from flash_attn.cute.cute_dsl_utils import to_cute_tensor
from flash_attn.cute.testing import is_fake_mode

from .dm_fwd_hd256_sm100 import DMFwdHd256Sm100

_HERE = Path(__file__).resolve().parent
_CACHE = None


def _cache():
    global _CACHE
    if _CACHE is None:
        h = hashlib.sha256(cache_utils._compute_source_fingerprint().encode())
        for f in ("dm_fwd_hd256_sm100.py", "fwd_hd256.py", "dm_utils.py"):
            h.update((_HERE / f).read_bytes())
        if cache_utils.CUTE_DSL_CACHE_ENABLED:
            _CACHE = cache_utils.JITPersistentCache(cache_utils.get_cache_path() / "dm_sm100" / h.hexdigest()[:24] / "fwd_hd256")
        else:
            _CACHE = cache_utils.JITCache()
    return _CACHE


def numerics():
    return os.environ.get("DM_HD256_NUMERICS", "a")


def dm_fwd_hd256(q8, qs, k8, ks, v8, vs, causal: bool, kv_stage: int = 4):
    B, N, H, D = q8.shape
    Hk = k8.shape[2]
    assert D == 256 and N % 128 == 0 and H % Hk == 0
    assert q8.dtype == torch.float8_e4m3fn and k8.dtype == q8.dtype and v8.dtype == q8.dtype
    assert qs.shape == (B, H, N // 64) and ks.shape == (B, Hk, N // 32) and vs.shape == (B, Hk, N // 32)
    q8, k8, v8, qs, ks, vs = (t.contiguous() for t in (q8, k8, v8, qs, ks, vs))
    o = torch.empty(B, N, H, D, device=q8.device, dtype=torch.bfloat16)
    lse = torch.empty(B, H, N, device=q8.device, dtype=torch.float32)
    num = numerics()
    key = ("fwd_hd256", bool(causal), H // Hk, kv_stage, num)
    cache = _cache()
    if key not in cache:
        kern = DMFwdHd256Sm100(bool(causal), H // Hk, kv_stage=kv_stage, numerics=num)
        args = [to_cute_tensor(t) for t in (q8, k8, v8, o)]
        args += [to_cute_tensor(lse, assumed_align=4)]
        args += [to_cute_tensor(t, assumed_align=4) for t in (qs, ks, vs)]
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        cache[key] = cute.compile(kern, *args, stream, options="--enable-tvm-ffi")
    if is_fake_mode():
        return o, lse
    u8 = torch.uint8
    cache[key](q8.view(u8), k8.view(u8), v8.view(u8), o, lse, qs, ks, vs)
    return o, lse
