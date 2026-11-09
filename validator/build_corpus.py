"""Build an offline byte-level training corpus from source / text files on this machine (Python packages, CUDA headers,
flash-attention / CUTLASS sources).  Deterministic order; one uint8 file + a held-out tail for evaluation."""
import os
import sys

ROOTS = [os.path.expanduser(p) for p in (
    "~/workspace/haok1402/sm100/env/venv/lib/python3.12/site-packages",
    "~/workspace/haok1402/sm100/env/cuda-13.0",
    "~/workspace/haok1402/sm100/ref",
    "~/workspace/haok1402/sm100/kda-ref",
)]
EXT = (".py", ".h", ".hpp", ".cuh", ".cu", ".cpp", ".cc", ".c", ".md", ".rst", ".txt", ".pyi", ".inl")
out = sys.argv[1]
total = 0
seen = set()
with open(out + ".tmp", "wb") as f:
    for root in ROOTS:
        for dp, dn, fn in sorted(os.walk(root, followlinks=False)):
            dn.sort()
            for name in sorted(fn):
                if not name.endswith(EXT):
                    continue
                p = os.path.join(dp, name)
                try:
                    st = os.stat(p)
                    if st.st_size > 2_000_000 or st.st_size < 64 or (st.st_ino, st.st_dev) in seen:
                        continue
                    seen.add((st.st_ino, st.st_dev))
                    b = open(p, "rb").read()
                except OSError:
                    continue
                if b.count(0) > 0:
                    continue
                f.write(b)
                f.write(b"\n")
                total += len(b) + 1
os.replace(out + ".tmp", out)
print("corpus bytes", total, out)
