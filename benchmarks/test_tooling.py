"""CPU-only tests of the measurement gates (benchmarks/ab.py, equivalence.py, kernels.py accounting).

    python -m pytest benchmarks/test_tooling.py -q
"""
import copy, hashlib, json, os, subprocess, sys

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
                        (lambda r: [r["ms"].__setitem__(c, 0.0) for c in K.CATS], "all categories zero"),
                        (lambda r: (r["ms"].__setitem__("pass1", 1e-9), r["kernels"].__setitem__("FlashAttnBwdSm90Fp8R", float("nan"))), "NaN event"),
                        (lambda r: r["kernels"].__setitem__("FlashAttnBwdSm90Fp8R", float("inf")), "inf event"),
                        (lambda r: r["kernels"].__setitem__("FlashAttnBwdSm90Fp8R", True), "bool event"),
                        (lambda r: r["kernels"].__setitem__("FlashAttnBwdSm90Fp8R", "1.0"), "string event"),
                        (lambda r: r["kernels"].__setitem__("", 0.1), "empty event name"),
                        (lambda r: (r["ms"].__setitem__("pass1", 1e-9), r["kernels"].__setitem__("FlashAttnBwdSm90Fp8R", -1.0),
                                    r["kernels"].__setitem__("FlashAttnBwdSm90Fp8R_2", 1.000000001)), "cancelling negative event")):
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


# ---------------------------------------------------------------- equivalence: schema-validated captured dumps
def fake_dump(path, route="kv", only="d128_gqa4", mutate=None, suite=None):
    """A dump that satisfies the independently derived schema (zero tensors of the expected shapes / dtypes)."""
    cases = EQ.selected_cases(route, only)
    res = {"manifest": {"version": EQ.MANIFEST_VERSION, "route": route, "route_env": dict(EQ.ROUTES[route]),
                        "resolved_route": {"groups": EQ.ROUTES[route]["DM_BWD_HEAD_GROUPS"],
                                           "streams": int(EQ.ROUTES[route]["DM_BWD_STREAMS"]),
                                           "ws_limit": float(EQ.ROUTES[route]["DM_BWD_WS_LIMIT_GB"]) * 2 ** 30},
                        "suite": suite or ("partial" if only else "full"), "only": only, "cases": dict(cases),
                        "tensors": {}},
           "meta": {"library_sha256": "0" * 64}, "cases": {}, "errors": {}, "heads_per_call": {}}
    for k_, desc in cases:
        sch = EQ.expected_schema(desc, route)
        res["cases"][k_] = {n: torch.zeros(shape, dtype=getattr(torch, dt.split(".")[1])) for n, (shape, dt) in sch.items()}
        res["manifest"]["tensors"][k_] = {n: [list(shape), dt] for n, (shape, dt) in sch.items()}
        res["heads_per_call"][k_] = EQ.heads_per_call(route, desc["H"], desc["Hk"])
    if mutate:
        mutate(res)
    torch.save(res, path)
    return path


def cmp(a, b, *extra):
    return subprocess.run([sys.executable, os.path.join(HERE, "equivalence.py"), "compare", str(a), str(b), *extra],
                          capture_output=True, text=True)


def first_case(res):
    return next(iter(res["cases"]))


def test_schema_groups_follow_the_route():
    desc = dict(EQ.selected_cases("kv", "d128_gqa4")[0][1])          # H 8, Hk 2 -> 4 query heads per KV group
    assert {n.split(".")[0] for n in EQ.expected_schema(desc, "kv") if n.startswith("g")} == {"g00-04", "g04-08"}
    assert {n.split(".")[0] for n in EQ.expected_schema(desc, "head") if n.startswith("g")} == {f"g{h:02d}-{h + 1:02d}" for h in range(8)}
    assert {n.split(".")[0] for n in EQ.expected_schema(desc, "default") if n.startswith("g")} == {"g00-08"}
    mqa = dict(EQ.selected_cases("kv", "d128_mqa")[0][1])             # MQA: one KV group = all heads
    assert EQ.heads_per_call("kv", mqa["H"], mqa["Hk"]) == mqa["H"]
    st = dict(EQ.selected_cases("kv", "d128_gqa4")[1][1])             # stale: no pass-0 tensors, a doq delta
    assert st["mode"] == "stale" and "doq.delta" in EQ.expected_schema(st, "kv")
    assert not any("delta_fp8" in n for n in EQ.expected_schema(st, "kv"))


def test_valid_partial_dump_compares_only_with_partial_flag(tmp_path):
    a = fake_dump(tmp_path / "a.pt"); b = fake_dump(tmp_path / "b.pt")
    r = cmp(a, b)
    assert r.returncode == 2 and "partial dump" in r.stdout
    r = cmp(a, b, "--partial")
    assert r.returncode == 0 and "PARTIAL" in r.stdout and "BITWISE EQUAL" in r.stdout, r.stdout


def test_captured_intermediate_byte_flip_is_reported(tmp_path):
    a = fake_dump(tmp_path / "a.pt")

    def flip(res):
        res["cases"][first_case(res)]["g04-08.bwd.delta"].view(-1).view(torch.uint8)[5] ^= 1
    b = fake_dump(tmp_path / "b.pt", mutate=flip)
    r = cmp(a, b, "--partial")
    assert r.returncode == 1 and "g04-08.bwd.delta" in r.stdout and "first at" in r.stdout, r.stdout


@pytest.mark.parametrize("why,mutate", [
    ("jointly empty tensors", lambda res: [t.update({n: torch.empty(0, dtype=torch.uint8) for n in t}) for t in res["cases"].values()]),
    ("wrong shape", lambda res: res["cases"][first_case(res)].__setitem__("quant.Q8", torch.zeros(3, 3, dtype=torch.float8_e4m3fn))),
    ("wrong dtype", lambda res: res["cases"][first_case(res)].__setitem__("fwd.LSE", res["cases"][first_case(res)]["fwd.LSE"].double())),
    ("manifest table not a dict", lambda res: res["manifest"].__setitem__("tensors", "x")),
    ("manifest entry contradicts", lambda res: res["manifest"]["tensors"][first_case(res)].__setitem__("api.O", [[1], "torch.bfloat16"])),
    ("wrong resolved route", lambda res: res["manifest"].__setitem__("resolved_route", {"groups": "off", "streams": 1, "ws_limit": 8 * 2.0 ** 30})),
    ("wrong route env", lambda res: res["manifest"]["route_env"].__setitem__("DM_BWD_STREAMS", "2")),
    ("missing group", lambda res: [t.pop(n) for t in res["cases"].values() for n in list(t) if n.startswith("g04-08")]),
    ("extra group", lambda res: res["cases"][first_case(res)].__setitem__("g08-12.bwd.dq", torch.zeros(1, dtype=torch.bfloat16))),
    ("wrong heads per call", lambda res: res["heads_per_call"].__setitem__(first_case(res), 8)),
    ("full suite with a subset of cases", lambda res: res["manifest"].__setitem__("suite", "full")),
])
def test_jointly_invalid_dumps_are_rejected(tmp_path, why, mutate):
    a = fake_dump(tmp_path / "a.pt", mutate=mutate); b = fake_dump(tmp_path / "b.pt", mutate=mutate)
    r = cmp(a, b, "--partial")
    assert r.returncode == 2 and "INVALID" in r.stdout, (why, r.stdout)


def test_mismatched_routes_rejected(tmp_path):
    a = fake_dump(tmp_path / "a.pt", route="kv"); b = fake_dump(tmp_path / "b.pt", route="head")
    assert cmp(a, b, "--partial").returncode == 2


def test_empty_selection_and_complete_routes():
    assert EQ.selected_cases("default", "no_such_case") == []
    for env in EQ.ROUTES.values():
        assert set(env) == {"DM_BWD_HEAD_GROUPS", "DM_BWD_STREAMS", "DM_BWD_WS_LIMIT_GB"}
    assert EQ.ROUTES["default"]["DM_BWD_HEAD_GROUPS"] == "off" and EQ.ROUTES["kv"]["DM_BWD_STREAMS"] == "1"


def test_every_route_covers_every_fixed_case():
    # independent of selected_cases(): every (shape, mode) of the fixed suite, including the minimum-length and packed
    # MHA cases, on every route
    want = {(name, attn, mode) for (name, *_rest) in EQ.SHAPES for (attn, mode) in EQ.MODES}
    assert len(want) == 72 and {"d128_min", "d256_min", "d128_packed", "d256_packed"} <= {w[0] for w in want}
    for route in EQ.ROUTES:
        got = {(d["name"], d["attn"], d["mode"]) for _, d in EQ.selected_cases(route)}
        assert got == want, (route, sorted(want - got)[:4])


@pytest.mark.parametrize("why,mutate", [
    ("None tensor", lambda res: res["cases"][first_case(res)].__setitem__("api.dV", None)),
    ("non-tensor value", lambda res: res["cases"][first_case(res)].__setitem__("quant.Q8", [1, 2, 3])),
    ("malformed descriptor", lambda res: res["manifest"]["cases"].__setitem__(first_case(res), {"name": "x"})),
])
def test_malformed_payloads_return_invalid(tmp_path, why, mutate):
    a = fake_dump(tmp_path / "a.pt", mutate=mutate); b = fake_dump(tmp_path / "b.pt", mutate=mutate)
    r = cmp(a, b, "--partial")
    assert r.returncode == 2 and "INVALID" in r.stdout and "Traceback" not in r.stderr, (why, r.stdout, r.stderr[-300:])


# ---------------------------------------------------------------- gate runner: every failing step stops the gate
STUB = r"""#!{py}
import os, sys
fail = os.environ.get("STUB_FAIL_ON", "")
args = sys.argv[1:]
line = " ".join(args)
def done(tag):
    sys.exit(42 if tag and tag in fail.split(",") else 0)
FUNCS = ["_Z9KernelX_causal", "_Z9KernelX_full", "_Z5Other"]
if sys.argv[0].endswith("cuobjdump"):
    if "cuobjdump" in fail.split(","):
        done("cuobjdump")
    shape = os.environ.get("STUB_RES", "ok")
    lib = open(args[-1], "rb").read()
    funcs = [f for f in FUNCS if not (shape == "nomatch" and "KernelX" in f)]
    if "--dump-resource-usage" in args:
        print("Fatbin elf code:\n================\narch = sm_90")
        for f in funcs:
            if shape == "onlysass" and f.endswith("full"):
                continue
            print(f"\tFunction {f}:")
            bad = shape == "badrow" and f.endswith("causal")
            print("\tREG:168 STACK:0" if bad else "\tREG:168 STACK:8 SHARED:0 LOCAL:0 CONSTANT[0]:904 TEXTURE:0 SURFACE:0 SAMPLER:0")
    else:
        for f in funcs:
            print(f"\n\tcode for sm_90\n\t\tFunction : {f}")
            n = 10 if shape == "truncated" and f.endswith("full") else 200
            for i in range(n):
                op = "QGMMA.64x256x32.F32.E4M3.E4M3" if i % 20 == 0 else "FFMA R1, R2, R3, R4"
                print(f"        /*{16 * i:04x}*/                   {op} ;  /* 0x000000000000 */")
    done("cuobjdump_ok")
if args[:2] == ["-m", "delta_matching.jit"]:
    jit = os.environ["DM_JIT_DIR"]; os.makedirs(jit, exist_ok=True)
    open(os.path.join(jit, "_C.so"), "wb").write(b"lib")
    os.makedirs("delta_matching", exist_ok=True)
    link = os.path.join("delta_matching", "_C.so")
    if os.path.lexists(link):
        os.remove(link)
    os.symlink(os.path.join(jit, "_C.so"), link)
    done("jit")
if args[:2] == ["-m", "pytest"] and "test_tooling" in line:
    done("tooling")
if args[:2] == ["-m", "pytest"]:
    open(os.environ["STUB_ENVLOG"], "w").write(repr({k: os.environ.get(k) for k in
        ("DM_SO", "PYTHONPATH", "KDA_EXPECT_PKG", "KDA_EXPECT_LIB", "KDA_EXPECT_SHA256", "PYTEST_ADDOPTS")}))
    if "-p" not in args or args[args.index("-p") + 1] != "gate_identity":
        sys.exit(7)
    junit = [x.split("=", 1)[1] for x in args if x.startswith("--junitxml=")][0]
    report = os.environ.get("STUB_JUNIT", "ok")
    cases = ['<testcase classname="t" name="p%d"/>' % i for i in range(49 if report != "short" else 30)]
    cases += ['<testcase classname="t" name="s%d"><skipped message="needs two GPUs"/></testcase>' % i for i in range(2)]
    cases.append('<testcase classname="t" name="fa"><skipped message="could not import \'flash_attn\': no module"/></testcase>')
    if report == "badskip":
        cases.append('<testcase classname="t" name="x"><skipped message="needs a CUDA device"/></testcase>')
    if report != "missing":
        open(junit, "w").write('<testsuites><testsuite name="pytest">' + "".join(cases) + "</testsuite></testsuites>")
    done("pytest")
if "equivalence.py" in line and "dump" in args:
    out = args[args.index("--out") + 1]
    if "dump" not in fail.split(","):
        open(out, "wb").write(b"x")
    done("dump")
if "equivalence.py" in line:
    done("compare")
if "ab.py" in line:
    d = args[args.index("--out-dir") + 1]; os.makedirs(d, exist_ok=True)
    if "ab" not in fail.split(","):
        open(os.path.join(d, "ab.json"), "w").write("{}")
    done("ab")
if "bench.py" in line:
    done("bench")
sys.exit(0)
"""


def run_gate(tmp_path, mode, fail, extra=None, kernel="X"):
    stub = tmp_path / "python"; stub.write_text(STUB.replace("{py}", sys.executable)); stub.chmod(0o755)
    cub = tmp_path / "cuobjdump"; cub.write_text(STUB.replace("{py}", sys.executable)); cub.chmod(0o755)
    wt = tmp_path / "wt"; (wt / "delta_matching").mkdir(parents=True)
    if mode == "confirm":
        (wt / "delta_matching" / "_C.so").write_bytes(b"lib")
    prev = tmp_path / "prev.so"; prev.write_bytes(b"prev")
    dumps = ",".join(str(tmp_path / f"d{i}.pt") for i in range(4))
    env = dict(os.environ, GATE_PYTHON=str(stub), GATE_CUOBJDUMP=str(cub), STUB_FAIL_ON=fail,
               STUB_ENVLOG=str(tmp_path / "pytest_env.txt"), **(extra or {}))
    return subprocess.run([sys.executable, os.path.join(HERE, "gate.py"), "--tag", "t", "--worktree", str(wt),
                           "--prev-root", str(tmp_path), "--prev-lib", str(prev), "--subset", "D128", "--kernel", kernel,
                           "--mode", mode, "--base-dumps", dumps, "--prev-dumps", dumps,
                           "--run-dir", str(tmp_path / f"run_{mode}_{fail or 'none'}")], env=env, capture_output=True, text=True)


@pytest.mark.parametrize("mode,fail,step", [
    ("full", "tooling", "tooling_tests"), ("full", "jit", "build"), ("full", "cuobjdump", "resources_prev"),
    ("full", "pytest", "pytest"), ("full", "dump", "dump_default"), ("full", "compare", "compare_base_default"),
    ("full", "ab", "ab_main"), ("full", "bench", "bench_A_rep0_D128"),
    ("confirm", "ab", "ab_main"), ("confirm", "bench", "bench_A_rep0_D128"), ("timing", "jit", "build")])
def test_gate_stops_on_any_failing_step(tmp_path, mode, fail, step):
    r = run_gate(tmp_path, mode, fail)
    assert r.returncode == 1 and f"step {step} failed" in r.stdout, (mode, fail, r.stdout[-800:], r.stderr[-400:])


@pytest.mark.parametrize("mode", ["full", "confirm", "timing"])
def test_gate_passes_when_every_step_passes(tmp_path, mode):
    r = run_gate(tmp_path, mode, "")
    assert r.returncode == 0 and "all steps passed" in r.stdout, (r.stdout[-800:], r.stderr[-400:])


def test_gate_refuses_a_reused_run_directory(tmp_path):
    (tmp_path / "run_full_none").mkdir()
    assert run_gate(tmp_path, "full", "").returncode != 0


@pytest.mark.parametrize("shape,why", [("nomatch", "no function matches"), ("badrow", "malformed"),
                                       ("onlysass", "different functions"), ("truncated", "truncated SASS")])
def test_gate_rejects_missing_or_malformed_resource_evidence(tmp_path, shape, why):
    r = run_gate(tmp_path, "full", "", extra={"STUB_RES": shape})
    assert r.returncode == 1 and why in r.stdout and "step pytest" not in r.stdout, r.stdout[-800:]


def test_gate_rejects_a_misspelled_kernel_selector(tmp_path):
    r = run_gate(tmp_path, "full", "", kernel="NoSuchKernel")
    assert r.returncode == 1 and "no function matches" in r.stdout, r.stdout[-800:]


def test_gate_records_both_target_instantiations(tmp_path):
    r = run_gate(tmp_path, "full", "")
    rec = json.load(open(tmp_path / "run_full_none" / "gate.json"))
    for side in ("prev", "cand"):
        assert sorted(rec["resources"][side]["rows"]) == ["_Z9KernelX_causal", "_Z9KernelX_full"]
        assert rec["resources"][side]["counts"]["QGMMA"] == 20


@pytest.mark.parametrize("report,why", [("short", "30 passed"), ("badskip", "skips outside"), ("missing", "no junit")])
def test_gate_rejects_an_incomplete_candidate_test_report(tmp_path, report, why):
    r = run_gate(tmp_path, "full", "", extra={"STUB_JUNIT": report})
    assert r.returncode == 1 and why in r.stdout, r.stdout[-800:]


def test_candidate_pytest_is_pinned_to_the_fresh_build_in_a_hostile_environment(tmp_path):
    hostile = {"DM_SO": str(tmp_path / "prev.so"), "PYTHONPATH": "/elsewhere", "PYTEST_ADDOPTS": "-k nothing"}
    r = run_gate(tmp_path, "full", "", extra=hostile)
    assert r.returncode == 0, r.stdout[-800:]
    seen = eval(open(tmp_path / "pytest_env.txt").read())
    lib = str(tmp_path / "run_full_none" / "jit" / "_C.so")
    assert seen["DM_SO"] == lib == seen["KDA_EXPECT_LIB"]
    assert seen["PYTHONPATH"].split(os.pathsep)[0] == str(tmp_path / "wt") and seen["PYTEST_ADDOPTS"] is None
    assert seen["KDA_EXPECT_PKG"] == str(tmp_path / "wt" / "delta_matching")
    assert seen["KDA_EXPECT_SHA256"] == hashlib.sha256(b"lib").hexdigest()


FAKE_CORE = """
import os
_lib = None
def _ops():
    global _lib
    _lib = os.environ.get("FAKE_LOADED") or os.environ["DM_SO"]
def loaded_library():
    return _lib
"""


@pytest.mark.parametrize("case", ["match", "other_lib", "bad_sha", "other_pkg", "unset"])
def test_identity_plugin_checks_inside_the_pytest_process(tmp_path, case):
    pkg = tmp_path / "wt" / "delta_matching"; pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("from . import core\n"); (pkg / "core.py").write_text(FAKE_CORE)
    lib = tmp_path / "cand.so"; lib.write_bytes(b"cand"); other = tmp_path / "prev.so"; other.write_bytes(b"prev")
    (tmp_path / "wt" / "tests").mkdir(); (tmp_path / "wt" / "tests" / "test_x.py").write_text("def test_x():\n    pass\n")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(tmp_path / "wt"), HERE]), DM_SO=str(lib),
               KDA_EXPECT_PKG=str(pkg), KDA_EXPECT_LIB=str(lib), KDA_EXPECT_SHA256=hashlib.sha256(b"cand").hexdigest())
    if case == "other_lib":
        env["FAKE_LOADED"] = str(other)
    elif case == "bad_sha":
        env["KDA_EXPECT_SHA256"] = "0" * 64
    elif case == "other_pkg":
        env["KDA_EXPECT_PKG"] = str(tmp_path / "elsewhere" / "delta_matching")
    elif case == "unset":
        env.pop("KDA_EXPECT_SHA256")
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q", "-p", "gate_identity", "-p", "no:cacheprovider"],
                       cwd=tmp_path / "wt", env=env, capture_output=True, text=True)
    if case == "match":
        assert r.returncode == 0 and "1 passed" in r.stdout, r.stdout[-600:]
    else:
        assert r.returncode == 3 and "gate_identity" in r.stdout + r.stderr and "passed" not in r.stdout, r.stdout[-600:]
