"""Host entry of the hd256 forward, MXFP8 (sm100-mx at D=256, validator preset sm100-mx-d256; kernel dm_fwd_hd256_mx_sm100.py).

dm_fwd_hd256_mx(q8, qs, k8, kvw, v8, causal) -> (O (B,N,H,256) bf16, LSE (B,H,N) fp32 natural log)
  q8 (B,N,H,256) e4m3 (sm = 2^-4 baked), k8 / v8 (B,N,Hk,256) e4m3; qs (B,H,N/32) fp32 pow2, kvw (B,Hk,N/64) int32 packed
  UE8M0 K/V words (quant.quant_qkv2(q, k, v, 256**-0.5)). N % 128 == 0.
Compile cache: persistent (FLASH_ATTENTION_CUTE_DSL_CACHE_DIR) under fwd._fingerprint(); shapes are dynamic. Under torch
FakeTensorMode the call only compiles (AOT, no GPU) and returns empty fakes.
"""
import torch
import cutlass.cute as cute
from flash_attn.cute import cache_utils
from flash_attn.cute.cute_dsl_utils import to_cute_tensor
from flash_attn.cute.testing import is_fake_mode

from .dm_fwd_hd256_mx_sm100 import DMFwdHd256MxSm100

_CACHE = None


def _cache():
    global _CACHE
    if _CACHE is None:
        # cache key: the same fingerprint as every other DM kernel = FA4 source fingerprint +
        # ALL delta_matching/sm100/*.py bytes + fwd.compile_env() (DSL lineinfo/opt/debug, SM count, fence settings, ...).
        # (Was: only 4 files and no compile settings -> a change in dm_common*.py / quant*.py or the DSL settings kept the key.)
        from .compile_cache_sm100 import fingerprint
        if cache_utils.CUTE_DSL_CACHE_ENABLED:
            _CACHE = cache_utils.JITPersistentCache(cache_utils.get_cache_path() / "dm_sm100" / fingerprint() / "fwd_hd256")
        else:
            _CACHE = cache_utils.JITCache()
    return _CACHE


def dm_fwd_hd256_mx(q8, qs, k8, kvw, v8, causal: bool):
    """q8 (B,N,H,256) e4m3 (sm baked), k8 / v8 (B,N,Hk,256) e4m3; qs (B,H,N/32) fp32 pow2 (quant.quant_qkv2).
    kvw: (B,Hk,N/64) int32 packed UE8M0 K/V words (fwd.pack_kv_scales). Returns O (B,N,H,256) bf16, LSE (B,H,N) fp32 natural log."""
    B, N, H, D = q8.shape
    Hk = k8.shape[2]
    assert D == 256 and N % 128 == 0 and H % Hk == 0
    assert q8.dtype == torch.float8_e4m3fn and k8.dtype == q8.dtype and v8.dtype == q8.dtype
    assert qs.shape == (B, H, N // 32) and kvw.shape == (B, Hk, N // 64) and kvw.dtype == torch.int32
    q8, k8, v8, qs, kvw = (t.contiguous() for t in (q8, k8, v8, qs, kvw))
    o = torch.empty(B, N, H, D, device=q8.device, dtype=torch.bfloat16)
    lse = torch.empty(B, H, N, device=q8.device, dtype=torch.float32)
    key = ("fwd_hd256_mx", bool(causal), H // Hk)
    cache = _cache()
    if key not in cache:
        kern = DMFwdHd256MxSm100(bool(causal), H // Hk)
        args = [to_cute_tensor(t) for t in (q8, k8, v8, o)]
        args += [to_cute_tensor(lse, assumed_align=4), to_cute_tensor(qs, assumed_align=4), to_cute_tensor(kvw, assumed_align=4)]
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        cache[key] = cute.compile(kern, *args, stream, options="--enable-tvm-ffi")
    if is_fake_mode():
        return o, lse
    u8 = torch.uint8
    cache[key](q8.view(u8), k8.view(u8), v8.view(u8), o, lse, qs, kvw)
    return o, lse
