"""pytest plugin used by benchmarks/gate.py: inside the candidate's own pytest process, load the delta_matching ops and
fail the session unless the imported package, the loaded library and its sha256 are exactly the gate's candidate
(KDA_EXPECT_PKG / KDA_EXPECT_LIB / KDA_EXPECT_SHA256).  Loaded with `-p gate_identity` (benchmarks/ on PYTHONPATH)."""
import hashlib, os

import pytest


def check_identity(pkg_dir, lib_path, want_pkg, want_lib, want_sha):
    """Raise ValueError unless (pkg_dir, lib_path) is the expected build; returns the library sha256."""
    h = hashlib.sha256()
    with open(lib_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    got = (os.path.realpath(pkg_dir), os.path.realpath(lib_path), h.hexdigest())
    want = (os.path.realpath(want_pkg), os.path.realpath(want_lib), want_sha)
    if got != want:
        raise ValueError(f"pytest loaded {got}, the gate expects {want}")
    return got[2]


def pytest_sessionstart(session):
    want = [os.environ.get(k) for k in ("KDA_EXPECT_PKG", "KDA_EXPECT_LIB", "KDA_EXPECT_SHA256")]
    if not all(want):
        pytest.exit("gate_identity: KDA_EXPECT_PKG / KDA_EXPECT_LIB / KDA_EXPECT_SHA256 are required", returncode=3)
    import delta_matching as dm
    dm.core._ops()
    try:
        sha = check_identity(os.path.dirname(dm.__file__), dm.core.loaded_library(), *want)
    except ValueError as e:
        pytest.exit(f"gate_identity: {e}", returncode=3)
    print(f"[gate_identity] candidate package and library verified (sha256 {sha[:16]})")
