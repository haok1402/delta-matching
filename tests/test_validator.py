"""The validation harness judges itself: the torch emulation wrapped as a candidate passes, and every cheat in
validator/selftest/cheats.py fails.  Needs a CUDA GPU; ~2 minutes."""
import os
import subprocess
import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V = os.path.join(ROOT, "validator")


def _run(cand, env_extra, only="B1_H8_Hk2_N1024_c_randn"):
    env = dict(os.environ, DMVAL_SELFTEST="1", **env_extra)
    cmd = [sys.executable, os.path.join(V, "validate.py"), os.path.join(V, "selftest", cand), "--shapes", "quick",
           "--seed", "5"] + (["--only", only] if only else [])
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1800)
    return p.returncode, p.stdout + p.stderr


@pytest.mark.parametrize("preset", ["sm90", "mxfp8", "mxfp8-p", "mx-v1"])
def test_honest_emulation_passes(preset):
    rc, out = _run("emul.py", {"SELFTEST_PRESET": preset}, only=None if preset == "sm90" else "N1024")
    assert rc == 0 and "\nPASS" in out, out[-3000:]


@pytest.mark.parametrize("cheat", ["bf16", "skiphead", "lastblock", "cache", "nondet", "modes", "stale_as_matched"])
def test_cheats_fail(cheat):
    rc, out = _run("cheats.py", {"CHEAT": cheat})
    assert rc != 0 and "\nFAIL" in out, out[-3000:]


@pytest.mark.parametrize("variant", ["delta_off", "debug_lies"])
def test_delta_cheats_fail(variant):
    rc, out = _run("emul.py", {"SELFTEST_VARIANT": variant})
    assert rc != 0 and "\nFAIL" in out, out[-3000:]


def test_sanitizer_tier_passes_the_emulation():
    env = dict(os.environ, DMVAL_SELFTEST="1")
    p = subprocess.run([sys.executable, os.path.join(V, "sanitize.py"), os.path.join(V, "selftest", "emul.py"),
                        "--timeout", "600"], capture_output=True, text=True, env=env, timeout=900)
    assert p.returncode == 0 and "PASS" in p.stdout, p.stdout[-2000:] + p.stderr[-2000:]
