"""CPU-only tests of the measurement gates (benchmarks/ab.py, equivalence.py, kernels.py accounting).

    python -m pytest benchmarks/test_tooling.py -q
"""
import copy, os, subprocess, sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ab  # noqa: E402
import equivalence as EQ  # noqa: E402
import kernels as K  # noqa: E402


# ---------------------------------------------------------------- accounting (hand-computed byte ledger)
def test_accounting_terms_hd128_gqa():
    D, B, H, Hk, N = 128, 1, 4, 2, 1024
    f, b = K.accounting(D, B, H, Hk, N, True, "matched")
    BHq, BHk, ND = 4, 2, N * D
    tiles = K.visited_tiles(N, 64, 128, True) * BHq
    # causal 64 x 128 tiles over 16 m-blocks: ceil((mb+1)*64/128) = 1,1,2,2,...,8,8 -> 72 per head
    assert K.visited_tiles(N, 64, 128, True) == 72 and K.visited_tiles(N, 64, 128, False) == 16 * 8
    Sq, Skv = 4 * BHq * N // 128, 8 * BHk * N // 64
    assert b["fwd"] == BHq * ND + 2 * BHk * ND + 2 * BHq * ND + 4 * BHq * N + Sq + Skv
    assert b["pass0"] == 2 * BHq * ND + 2 * BHk * ND + 8 * BHq * N + 2 * Sq + Skv        # no K twin; scale reads
    assert b["pass1"] == 4 * BHq * ND + 2 * BHk * ND + 8 * BHq * N + 4 * BHq * ND + tiles * 64 * 128 + 4 * tiles + 2 * Sq + Skv
    kmean = 2 * BHk * (N // 64) * D * 4 + 2 * BHk * D * 4
    scales = 4 * (BHq * N // 128 + 2 * BHk * N // 64)
    assert b["prod_fwd"] == 2 * (BHq + 2 * BHk) * ND + 2 * BHk * ND + (BHq + 3 * BHk) * ND + kmean + scales
    assert b["glue"] == 2 * 2 * (BHq + 2 * BHk) * ND + 2 * 2 * BHq * ND + 2 * 2 * BHk * ND
    assert f["pass1"] == 8 * (N * (N + 1) / 2) * BHq * D and f["pass2"] == f["pass1"] / 4


def test_accounting_d256_partials_and_mha_glue():
    D, B, H, N = 256, 2, 2, 256
    _, b = K.accounting(D, B, H, H, N, True, "matched")
    BHq, ND = 4, N * D
    part = 2 * K.visited_tiles(N, 64, 64, True) * BHq * 64 * 2 * 4
    Sq, Skv = 4 * BHq * N // 64, 8 * BHq * N // 32
    assert b["pass0"] == 2 * BHq * ND + 2 * BHq * ND + 8 * BHq * N + 2 * Sq + Skv + part
    assert b["glue"] == 2 * 2 * 3 * BHq * ND                                               # MHA: copies only
    _, b2 = K.accounting(D, B, H, H, N, True, "matched", layout="bhnd")
    assert b2["glue"] == 0


# ---------------------------------------------------------------- A/B gates
def rec(D=128, N=2048, attn="causal", mode="matched", t=1.0, B=16, H=16, Hk=16):
    ms = {c: (t / len(K.CATS) if (c != "pass0" or mode == "matched") else 0.0) for c in K.CATS}
    names = {"fwd": "FlashAttnFwdSm90Fq8F", "pass1": "FlashAttnBwdSm90Fp8R", "pass2": "fq8p2h128", "pass0": "DeltaV6",
             "twin": "_twin_pi_kernel", "prod_bwd": "_doq_m17_kernel", "prod_fwd": "_quant_kernel", "glue": "direct_copy"}
    kern = {names[c]: v for c, v in ms.items() if v > 0}
    return {"D": D, "B": B, "H": H, "Hk": Hk, "N": N, "attn": attn, "mode": mode, "e2e_fwd_ms": t / 3,
            "e2e_fwdbwd_ms": t, "ms": ms, "kernels": kern}


def test_median_records_rejects_missing_repetition():
    with pytest.raises(SystemExit):
        ab.median_records([[rec()], []])


def test_validate_run_rejects_wrong_library_and_missing_config(tmp_path):
    build = ("/x/delta_matching", "/x/delta_matching/_C.so", "a" * 64)
    good = {"meta": {"package": build[0], "library": build[1], "library_sha256": build[2]},
            "records": [rec(), rec(attn="full", mode="stale")]}
    exp = [ab.key(r) for r in good["records"]]
    ab.validate_run(good, build, exp, "ok")
    bad = copy.deepcopy(good); bad["meta"]["library_sha256"] = "b" * 64
    with pytest.raises(SystemExit):
        ab.validate_run(bad, build, exp, "sha")
    short = copy.deepcopy(good); short["records"] = short["records"][:1]
    with pytest.raises(SystemExit):
        ab.validate_run(short, build, exp, "missing")
    dup = copy.deepcopy(good); dup["records"].append(rec())
    with pytest.raises(SystemExit):
        ab.validate_run(dup, build, exp, "dup")
    nan = copy.deepcopy(good); nan["records"][0]["ms"]["pass1"] = float("nan")
    with pytest.raises(SystemExit):
        ab.validate_run(nan, build, exp, "nan")


def test_validate_run_rejects_zero_required_kernels():
    build = ("/x/delta_matching", "/x/delta_matching/_C.so", "a" * 64)
    meta = {"package": build[0], "library": build[1], "library_sha256": build[2]}
    stale = rec(attn="full", mode="stale")
    assert stale["ms"]["pass0"] == 0.0
    ab.validate_run({"meta": meta, "records": [stale]}, build, [ab.key(stale)], "stale without pass0 is fine")
    for mutate, why in ((lambda r: (r["ms"].__setitem__("pass1", 0.0), r["kernels"].pop("FlashAttnBwdSm90Fp8R")), "zero pass1"),
                        (lambda r: (r["ms"].__setitem__("pass0", 0.0), r["kernels"].pop("DeltaV6")), "zero matched pass0"),
                        (lambda r: r.__setitem__("kernels", {}), "empty kernel events"),
                        (lambda r: r["ms"].__setitem__("pass2", 5.0), "category total disagrees with events"),
                        (lambda r: [r["ms"].__setitem__(c, 0.0) for c in K.CATS], "all categories zero")):
        r = rec()
        mutate(r)
        with pytest.raises(SystemExit):
            ab.validate_run({"meta": meta, "records": [r]}, build, [ab.key(r)], why)


def test_child_env_drops_inherited_dm_so(monkeypatch):
    monkeypatch.setenv("DM_SO", "/candidate/override.so")
    monkeypatch.setenv("KDA_BENCH_SLOWDOWN_CYCLES", "5")
    env = ab.child_env("/frozen/A", "/frozen/A/delta_matching/_C.so")
    assert env["DM_SO"] == "/frozen/A/delta_matching/_C.so" and env["PYTHONPATH"] == "/frozen/A"
    assert "KDA_BENCH_SLOWDOWN_CYCLES" not in env


def test_reps_below_three_rejected_without_smoke(tmp_path):
    p = subprocess.run([sys.executable, os.path.join(HERE, "ab.py"), "run", "--a", "x", "--b", "y", "--reps", "1",
                        "--out-dir", str(tmp_path)], capture_output=True, text=True)
    assert p.returncode != 0 and "reps >= 3" in (p.stdout + p.stderr)


def test_single_shape_regression_is_listed_despite_harmless_total():
    # 20 shapes; B is 0.5% faster on 19 and 5% slower on one: the grid total barely moves, the shape must be flagged
    shapes = [(128, 2 ** i, 1024 * (i + 1)) for i in range(20)]
    raw_a, raw_b = [], []
    for rep in range(3):
        jit = 1 + 0.001 * rep
        raw_a.append([rec(N=N, B=Bb, t=10.0 * jit) for (_, Bb, N) in shapes])
        raw_b.append([rec(N=N, B=Bb, t=(10.5 if i == 7 else 9.95) * jit) for i, (_, Bb, N) in enumerate(shapes)])
    res, _, _ = ab.analyze(raw_a, raw_b, ab.SUBSETS["D128"], 1.0)
    assert abs(res["all"]["e2e_fwdbwd"]["change_pct"]) < 1.0
    flagged = res["over_gate"]
    assert len(flagged) == 1 and flagged[0]["key"][4] == shapes[7][2] and flagged[0]["beyond_spread"]
    assert "subset" in res and res["subset"]["pass1"]["A"] > 0


# ---------------------------------------------------------------- equivalence manifests
def fake_dump(path, route="kv", only=None, drop=(), suite=None):
    cases = EQ.selected_cases(route, only)
    res = {"manifest": {"version": EQ.MANIFEST_VERSION, "route": route, "route_env": EQ.ROUTES[route],
                        "suite": suite or ("full" if only is None else "partial"), "only": only,
                        "cases": dict(cases), "tensors": {}},
           "meta": {"library_sha256": "0" * 64}, "cases": {}, "errors": {}}
    for k_, desc in cases:
        res["cases"][k_] = {n: torch.zeros(4) for n in EQ.required_tensors(desc) if n not in drop}
    torch.save(res, path)
    return path


def cmp(a, b, *extra):
    return subprocess.run([sys.executable, os.path.join(HERE, "equivalence.py"), "compare", str(a), str(b), *extra],
                          capture_output=True, text=True)


def test_compare_full_partial_and_rejections(tmp_path):
    a = fake_dump(tmp_path / "a.pt"); b = fake_dump(tmp_path / "b.pt")
    r = cmp(a, b)
    assert r.returncode == 0 and "FULL SUITE" in r.stdout and "BITWISE EQUAL" in r.stdout
    # jointly missing dV: both sides lack a required tensor
    a2 = fake_dump(tmp_path / "a2.pt", drop=("dV",)); b2 = fake_dump(tmp_path / "b2.pt", drop=("dV",))
    r = cmp(a2, b2)
    assert r.returncode != 0 and "MISSING required" in r.stdout
    # mismatched routes
    c = fake_dump(tmp_path / "c.pt", route="head")
    assert cmp(a, c).returncode != 0
    # partial dumps need --partial and are labelled
    p1 = fake_dump(tmp_path / "p1.pt", only="d128_gqa4"); p2 = fake_dump(tmp_path / "p2.pt", only="d128_gqa4")
    assert cmp(p1, p2).returncode != 0
    r = cmp(p1, p2, "--partial")
    assert r.returncode == 0 and "PARTIAL" in r.stdout


def test_empty_selection_rejected():
    assert EQ.selected_cases("default", "no_such_case") == []


def test_routes_are_complete():
    for r, env in EQ.ROUTES.items():
        assert set(env) == {"DM_BWD_HEAD_GROUPS", "DM_BWD_STREAMS", "DM_BWD_WS_LIMIT_GB"}
    assert EQ.ROUTES["default"]["DM_BWD_HEAD_GROUPS"] == "off" and EQ.ROUTES["kv"]["DM_BWD_STREAMS"] == "1"
