"""Track R (kda-sm100-p0rw2): pass 0 rewritten in CUDA C++ around MUFU occupancy (csrc/p0rw2.cu, design D8).
dm_delta_rw2(...) has the same contract as pass0.dm_delta and is meant to be bitwise equal to the DSL pass 0
(exp p3_16_0+ma+h2+pf+sq+ts). Opt-in: DM_P0_IMPL=rw2 (pass0.dm_delta dispatches here).
Build (no GPU): python -m delta_matching.sm100.p0rw2 build  -> csrc/build_p0rw2/libp0rw2.so"""
import ctypes
import os
import subprocess
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "csrc", "p0rw2.cu")
_LIB = None


TRACE = os.environ.get("DM_P0RW2_TRACE", "0") == "1"


def _so_path():
    d = os.environ.get("DM_P0RW2_BUILD", os.path.join(_HERE, "csrc", "build_p0rw2"))
    tag = os.environ.get("DM_P0RW2_TAG", "")
    return os.path.join(d, ("libp0rw2_trace" if TRACE else "libp0rw2") + tag + ".so")


def build():
    so = _so_path()
    os.makedirs(os.path.dirname(so), exist_ok=True)
    nvcc = os.environ.get("DM_NVCC", os.path.expanduser("~/workspace/haok1402/sm100/env/cuda-13.0/bin/nvcc"))
    cmd = [nvcc, "-O3", "-std=c++17", "-gencode=arch=compute_100a,code=sm_100a", "-lineinfo", "-Xptxas", "-v",
           "-shared", "-Xcompiler", "-fPIC", "-o", so, _SRC, "-lcudart"] + (["-DP0RW2_TRACE"] if TRACE else []) + os.environ.get("DM_P0RW2_DEFS", "").split()
    r = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(r.stdout + r.stderr)
    if r.returncode:
        raise RuntimeError("p0rw2 build failed")
    return so


_LIBS = {}


def _lib(tag=None):
    global _LIB
    if tag is not None:
        if tag not in _LIBS:
            d = os.environ.get("DM_P0RW2_BUILD", os.path.join(_HERE, "csrc", "build_p0rw2"))
            L = ctypes.CDLL(os.path.join(d, ("libp0rw2_trace" if TRACE else "libp0rw2") + tag + ".so"))
            L.p0rw2_delta.restype = ctypes.c_int
            _LIBS[tag] = L
        return _LIBS[tag]
    if _LIB is None:
        so = _so_path()
        if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(_SRC):
            raise RuntimeError(f"{so} missing or stale: build it AOT (python -m delta_matching.sm100.p0rw2 build)")
        _LIB = ctypes.CDLL(so)
        _LIB.p0rw2_delta.restype = ctypes.c_int
    return _LIB


TRACE_BUF = None


def dm_delta_rw2(q8, qs, k8, ks, v8, vs, do8, dos, lse, causal, tag=None):
    global TRACE_BUF
    B, N, H, D = q8.shape
    Hk = k8.shape[2]
    assert D == 128 and N % 128 == 0
    q8, k8, v8, do8 = (t.contiguous() for t in (q8, k8, v8, do8))
    qs, ks, vs, dos = (t.contiguous().float() for t in (qs, ks, vs, dos))
    lse = lse.contiguous().float()
    out = torch.empty((B, H, N), dtype=torch.float32, device=q8.device)
    P = ctypes.c_void_p
    rc = _lib(tag).p0rw2_delta(P(q8.data_ptr()), P(qs.data_ptr()), P(k8.data_ptr()), P(ks.data_ptr()), P(v8.data_ptr()),
                            P(vs.data_ptr()), P(do8.data_ptr()), P(dos.data_ptr()), P(lse.data_ptr()), P(out.data_ptr()),
                            B, N, H, Hk, int(bool(causal)), P(torch.cuda.current_stream().cuda_stream),
                            P(TRACE_BUF.data_ptr() if TRACE_BUF is not None else 0))
    if rc:
        raise RuntimeError(f"p0rw2_delta failed rc={rc}")
    return out


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "build":
    print("built", build())
