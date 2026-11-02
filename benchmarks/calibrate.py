"""Same-allocation peak calibration for the four-floor model (benchmarks/sol4.py; Hao's audit direction 5).

    python benchmarks/calibrate.py --out calib.json

- fp8 tensor peak: sustained cuBLASLt e4m3 GEMM throughput (torch._scaled_mm, M = N = K = 8192, bf16 output,
  per-tensor scales), CUDA-event timing, 20 warm-ups, 50 timed repetitions in 5 batches of 10; the peak is the
  median batch throughput (FLOP = 2 M N K).
- copy bandwidth: device-to-device copy of a 4 GiB buffer (far beyond the 50 MB L2), 5 warm-ups, 20 timed copies;
  bytes moved = read + write; median.
- the SM clock is sampled with nvidia-smi while the GEMM loop runs.
Everything (raw timings, sizes, GPU, clock, torch / CUDA versions, job id) goes into the JSON, whose sha256 binds it to
the timing data that uses it.
"""
import argparse, hashlib, json, os, statistics, subprocess, sys, threading, time

import torch


def _events(fn, reps):
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); b.synchronize()
        ts.append(a.elapsed_time(b))
    return ts


def gemm_fp8(n=8192, warm=20, batches=5, per_batch=10):
    a = torch.randn(n, n, device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn(n, n, device="cuda").to(torch.float8_e4m3fn).t()        # column-major B, as cuBLASLt fp8 needs
    one = torch.ones((), device="cuda")

    def run():
        for _ in range(per_batch):
            torch._scaled_mm(a, b, scale_a=one, scale_b=one, out_dtype=torch.bfloat16)
    for _ in range(warm):
        torch._scaled_mm(a, b, scale_a=one, scale_b=one, out_dtype=torch.bfloat16)
    torch.cuda.synchronize()
    ms = _events(run, batches)
    flop = 2.0 * n ** 3 * per_batch
    tflops = [flop / (t * 1e-3) / 1e12 for t in ms]
    return {"M": n, "N": n, "K": n, "dtype": "e4m3 x e4m3 -> bf16", "per_batch": per_batch, "batch_ms": ms,
            "tflops_per_batch": tflops, "peak_tflops": statistics.median(tflops)}


def copy_bw(gib=4, warm=5, reps=20):
    n = gib * (1 << 30)
    src = torch.empty(n, dtype=torch.uint8, device="cuda").random_(0, 255)
    dst = torch.empty_like(src)
    for _ in range(warm):
        dst.copy_(src)
    torch.cuda.synchronize()
    ms = _events(lambda: dst.copy_(src), reps)
    gbs = [2.0 * n / (t * 1e-3) / 1e9 for t in ms]
    return {"bytes_per_copy": 2 * n, "buffer_gib": gib, "copy_ms": ms, "gbs": gbs, "peak_gbs": statistics.median(gbs)}


def sample_clock(stop, out):
    while not stop.is_set():
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits", "-i",
                                os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]],
                               capture_output=True, text=True, timeout=5)
            out.append(float(r.stdout.strip().splitlines()[0]))
        except Exception:
            pass
        time.sleep(0.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _paths import results_path
    a.out = results_path(a.out)
    clocks, stop = [], threading.Event()
    th = threading.Thread(target=sample_clock, args=(stop, clocks), daemon=True); th.start()
    g = gemm_fp8()
    stop.set(); th.join()
    c = copy_bw()
    p = torch.cuda.get_device_properties(0)
    out = {"version": 1, "gpu": p.name, "device_uuid": str(getattr(p, "uuid", "")) or None, "sms": p.multi_processor_count, "torch": torch.__version__,
           "cuda": torch.version.cuda, "job": os.environ.get("SLURM_JOB_ID"), "host": os.uname().nodename,
           "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "sm_clock_mhz_samples": clocks,
           "sm_clock_mhz": statistics.median(clocks) if clocks else None, "gemm_fp8": g, "copy": c}
    blob = json.dumps(out, sort_keys=True)
    out["sha256"] = hashlib.sha256(blob.encode()).hexdigest()
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"fp8 GEMM {g['peak_tflops']:.1f} TFLOPS, copy {c['peak_gbs']:.1f} GB/s, SM clock "
          f"{out['sm_clock_mhz']} MHz, sha256 {out['sha256'][:16]}")


if __name__ == "__main__":
    main()
