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
    if shape == "newfunc" and lib == b"lib":
        funcs = funcs + ["_Z9KernelX_w4"]   # the candidate adds a target kernel (e.g. a 4-warpgroup CTA variant)
    ninstr = 5000 if shape == "long" else 200
    if "--dump-resource-usage" in args:
        print("Fatbin elf code:\n================\narch = sm_90")
        for f in funcs:
            if shape == "onlysass" and f.endswith("full"):
                continue
            print(f"\tFunction {f}:")
            bad = shape == "badrow" and f.endswith("causal")
            stack = 16 if shape == "stackgrow" and lib == b"lib" and f.endswith("causal") else 8
            print("\tREG:168 STACK:0" if bad else f"\tREG:168 STACK:{stack} SHARED:0 LOCAL:0 CONSTANT[0]:904 TEXTURE:0 SURFACE:0 SAMPLER:0")
    elif "--dump-elf" in args:
        print("Fatbin elf code:\n================\narch = sm_90a\nSections:\nIndex Offset   Size ES Align Type Flags Link Info Name")
        for k, f in enumerate(funcs):
            if shape == "noelf" and f.endswith("full"):
                continue
            for _ in range(2 if shape == "dupelf" and f.endswith("causal") else 1):
                print(f"   {k + 10:2x}   {4096 * (k + 1):x}  {16 * ninstr:x}  0 80   PROGBITS   6    3   {k + 3:x} .text.{f}")
    else:
        for f in funcs:
            for _ in range(2 if shape == "dupfunc" and f.endswith("causal") else 1):
                print(f"\n\tcode for sm_90\n\t\tFunction : {f}")
                full = f.endswith("full")
                idx = list(range(ninstr))
                if full and shape == "truncated":
                    idx = idx[:10]
                if full and shape == "tail":
                    idx = idx[:100]
                if full and shape == "gap":
                    idx = idx[:50] + idx[51:]
                if full and shape == "dupoff":
                    idx = idx[:51] + [50] + idx[51:]
                for i in idx:
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
if args[:2] == ["-m", "pytest"] and any(x.endswith("test_tooling.py") for x in args):
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
        acc = os.environ.get("STUB_DECISION", "accept") == "accept"
        if "--target" not in args or args[args.index("--grid") + 1] != "paper":
            sys.exit(9)
        open(os.path.join(d, "ab.json"), "w").write(
            '{"decision": {"accept": %s, "checks": {"target_gain": %s}, "tune_dim": 128, "target_tune_change_pct": -4.0}}'
            % (("true", "true") if acc else ("false", "false")))
    done("ab")
if "bench.py" in line:
    if args[args.index("--modes") + 1] != "matched":
        sys.exit(9)
    out = args[args.index("--json-out") + 1]
    slow = os.environ.get("STUB_BENCH_SLOW") == "1" and "bench_B_" in os.path.basename(out)
    t = 20.0 if slow else 10.0
    if "bench" not in fail.split(","):
        open(out, "w").write('{"rows": {"delta_matching (matched)": {"fwd_ms": 1.0, "fwdbwd_ms": %s}, '
                             '"delta_matching (stale)": {"fwd_ms": 1.0, "fwdbwd_ms": %s}}}' % (t, t))
    done("bench")
sys.exit(0)
"""


def run_gate(tmp_path, mode, fail, extra=None, kernel="X", args=()):
    stub = tmp_path / "python"; stub.write_text(STUB.replace("{py}", sys.executable)); stub.chmod(0o755)
    cub = tmp_path / "cuobjdump"; cub.write_text(STUB.replace("{py}", sys.executable)); cub.chmod(0o755)
    wt = tmp_path / "wt"; (wt / "delta_matching").mkdir(parents=True, exist_ok=True)
    if mode == "confirm":
        (wt / "delta_matching" / "_C.so").write_bytes(b"lib")
    prev = tmp_path / "prev.so"; prev.write_bytes(b"prev")
    dumps = ",".join(str(tmp_path / f"d{i}.pt") for i in range(4))
    env = dict(os.environ, GATE_PYTHON=str(stub), GATE_CUOBJDUMP=str(cub), STUB_FAIL_ON=fail,
               STUB_ENVLOG=str(tmp_path / "pytest_env.txt"), **(extra or {}))
    return subprocess.run([sys.executable, os.path.join(HERE, "gate.py"), "--tag", "t", "--worktree", str(wt),
                           "--prev-root", str(tmp_path), "--prev-lib", str(prev), "--target", "pass2", "--kernel", kernel,
                           "--mode", mode, "--base-dumps", dumps, "--prev-dumps", dumps,
                           "--run-dir", str(tmp_path / f"run_{mode}_{fail or 'none'}")] + list(args), env=env, capture_output=True, text=True)


@pytest.mark.parametrize("mode,fail,step", [
    ("full", "tooling", "tooling_tests"), ("full", "jit", "build"), ("full", "cuobjdump", "resources_prev"),
    ("full", "pytest", "pytest"), ("full", "dump", "dump_default"), ("full", "compare", "compare_base_default"),
    ("full", "ab", "ab_all"), ("full", "bench", "bench_A_rep0_D128"),
    ("confirm", "ab", "ab_all"), ("confirm", "bench", "bench_A_rep0_D128"), ("timing", "jit", "build")])
def test_gate_stops_on_any_failing_step(tmp_path, mode, fail, step):
    r = run_gate(tmp_path, mode, fail, args=["--tooling-tests"] if fail == "tooling" else [])
    assert r.returncode == 1 and f"step {step} failed" in r.stdout, (mode, fail, r.stdout[-800:], r.stderr[-400:])


@pytest.mark.parametrize("mode", ["full", "confirm", "timing"])
def test_gate_passes_when_every_step_passes(tmp_path, mode):
    r = run_gate(tmp_path, mode, "")
    assert r.returncode == 0 and "all steps passed" in r.stdout, (r.stdout[-800:], r.stderr[-400:])


def test_gate_refuses_a_reused_run_directory(tmp_path):
    (tmp_path / "run_full_none").mkdir()
    assert run_gate(tmp_path, "full", "").returncode != 0


@pytest.mark.parametrize("shape,why", [("nomatch", "no function matches"), ("badrow", "malformed"),
                                       ("onlysass", "different functions"), ("truncated", "incomplete SASS"),
                                       ("tail", "incomplete SASS"), ("gap", "incomplete SASS"), ("dupoff", "incomplete SASS"),
                                       ("dupfunc", "duplicate SASS function"), ("noelf", "ELF .text sections"),
                                       ("dupelf", "duplicate ELF")])
def test_gate_rejects_missing_or_malformed_resource_evidence(tmp_path, shape, why):
    r = run_gate(tmp_path, "full", "", extra={"STUB_RES": shape})
    assert r.returncode == 1 and why in r.stdout and "step pytest" not in r.stdout, r.stdout[-800:]


def test_gate_new_target_functions_need_the_explicit_flag(tmp_path):
    r = run_gate(tmp_path, "full", "", extra={"STUB_RES": "newfunc"})
    assert r.returncode == 1 and "functions in the previous build" in r.stdout, r.stdout[-800:]
    d = tmp_path / "flag"; d.mkdir()
    r = run_gate(d, "full", "", extra={"STUB_RES": "newfunc"}, args=["--allow-new-functions"])
    assert r.returncode == 0 and "new kernels allowed" in r.stdout, r.stdout[-800:]


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


def test_gate_counts_instructions_past_offset_0x10000(tmp_path):
    r = run_gate(tmp_path, "full", "", extra={"STUB_RES": "long"})
    rec = json.load(open(tmp_path / "run_full_none" / "gate.json"))
    assert r.returncode == 0 and rec["resources"]["cand"]["counts"]["QGMMA"] == 2 * 250, r.stdout[-600:]


def test_resource_growth_is_recorded_but_not_fatal_in_timing_mode(tmp_path):
    r = run_gate(tmp_path, "timing", "", extra={"STUB_RES": "stackgrow"})
    rec = json.load(open(tmp_path / "run_timing_none" / "gate.json"))
    assert r.returncode == 0 and rec["resource_regressions"] == ["_Z9KernelX_causal: STACK 8 -> 16"], r.stdout[-600:]


def test_gate_rejects_when_the_acceptance_predicates_fail(tmp_path):
    r = run_gate(tmp_path, "full", "", extra={"STUB_DECISION": "reject"})
    assert r.returncode == 1 and "acceptance predicates not met" in r.stdout, r.stdout[-600:]


def _recs(t, p2, d256=(30.0, 1.0), slow=None, ratio=None):
    """One repetition: the eight acceptance shapes (kernels.PAPER_SET; e2e t + index, pass2 p2, `slow` = {index: extra
    e2e ms}, `ratio` = {index: pass2}) and one D256 shape (e2e, pass2)."""
    out = []
    for i, (D, B, H, Hk, N) in enumerate(K.PAPER_SET):
        ms = {c: 1.0 for c in K.CATS}; ms["pass2"] = (ratio or {}).get(i, p2)
        e2e = t + i + (slow or {}).get(i, 0.0)
        out.append({"D": D, "B": B, "H": H, "Hk": Hk, "N": N, "attn": "causal", "mode": "matched",
                    "e2e_fwd_ms": e2e / 2, "e2e_fwdbwd_ms": e2e, "ms": ms, "kernels": {}})
    ms = {c: 1.0 for c in K.CATS}; ms["pass2"] = d256[1]
    out.append({"D": 256, "B": 1, "H": 16, "Hk": 16, "N": 2048, "attn": "causal", "mode": "matched",
                "e2e_fwd_ms": d256[0] / 2, "e2e_fwdbwd_ms": d256[0], "ms": ms, "kernels": {}})
    return out


def _decide(a_runs, b_runs, target="pass2"):
    res, _, _ = ab.analyze(a_runs, b_runs)
    return ab.decision(a_runs, b_runs, res, target)


def test_paper_set_is_the_four_round16_training_shapes():
    assert sorted(K.PAPER_SET) == sorted((128, B, H, 4, N) for H in (16, 32) for B, N in ((2, 8192), (1, 65536)))
    assert K.GRIDS["paper"] == K.PAPER_SET and K.PAPER_GUARD == []


def test_decision_accepts_a_gain_with_noise_only():
    A = [_recs(10.0 + 0.02 * i, 1.0, (30.0 + 0.02 * i, 1.0)) for i in range(3)]
    B = [_recs(10.03, 0.95, (30.0 + 0.01 * i, 1.0)) for i in range(3)]   # +0.01 ms vs a 0.04 ms range
    d = _decide(A, B)
    assert d["accept"] and abs(d["target_tune_change_pct"] + 5.0) < 1e-6 and d["tuned_configs"] == 4, d


def test_decision_rejects_small_gain_slow_shape_and_reports_d256():
    A = [_recs(10.0 + 0.01 * i, 1.0, (30.0 + 0.01 * i, 1.0)) for i in range(3)]
    small = _decide(A, [_recs(10.0, 0.98) for _ in range(3)])
    assert not small["accept"] and not small["checks"]["target_gain"]
    last = len(K.PAPER_SET) - 1                     # B1 N65536 H32 / 4: in the acceptance set since round 16
    slow = _decide(A, [_recs(10.0, 0.9, slow={last: 0.5}) for _ in range(3)])
    assert not slow["accept"] and slow["tune_shapes_slower_beyond_spread"] == [list(K.PAPER_SET[last]) + ["causal", "matched"]]
    other = _decide(A, [_recs(10.0, 0.9, d256=(31.0, 1.2)) for _ in range(3)])
    assert other["accept"] and other["other_shapes_slower_beyond_spread"] and other["target_other_slower_beyond_spread"]


def test_decision_weights_every_shape_equally():
    A = [_recs(10.0 + 0.01 * i, 1.0) for i in range(3)]
    one = _decide(A, [_recs(10.0, 1.0, ratio={0: 0.88}) for _ in range(3)])       # one shape -12%: gm 0.88 ** (1/4)
    assert one["accept"] and abs(one["target_tune_change_pct"] - 100 * (0.88 ** 0.25 - 1)) < 1e-9, one
    assert abs(one["target_tune_sum_change_pct"] + 3.0) < 1e-9                    # the plain sum, recorded only


def test_decision_needs_the_complete_matched_set():
    A = [_recs(10.0 + 0.01 * i, 1.0) for i in range(3)]
    B = [_recs(10.0, 0.9) for _ in range(3)]
    for runs in (A, B):
        for recs in runs:
            recs[1]["mode"] = "stale"   # a stale shape is not an acceptance shape
    d = _decide(A, B)
    assert not d["accept"] and d["tuned_configs"] == 3 and not d["checks"]["complete_set"]


def test_slower_within_spread_is_not_flagged():
    A = [_recs(10.0 + 0.2 * i, 1.0) for i in range(3)]
    B = [_recs(10.3, 0.9) for _ in range(3)]
    res, _, _ = ab.analyze(A, B)
    assert not res["shapes"][0]["slower_beyond_spread"]   # +0.1 ms median vs a 0.4 ms A range


def test_gate_rejects_a_bench_regression_even_when_every_child_succeeds(tmp_path):
    r = run_gate(tmp_path, "full", "", extra={"STUB_BENCH_SLOW": "1"})
    rec = json.load(open(tmp_path / "run_full_none" / "gate.json"))
    assert r.returncode == 1 and "bench.py fwd+bwd regression" in r.stdout, r.stdout[-600:]
    assert rec["bench_decision"]["changes_pct"]["D128 delta_matching (matched)"] == 100.0


def test_gate_records_the_bench_decision_on_success(tmp_path):
    r = run_gate(tmp_path, "confirm", "")
    rec = json.load(open(tmp_path / "run_confirm_none" / "gate.json"))
    assert r.returncode == 0 and rec["bench_decision"]["ok"] and list(rec["bench_decision"]["changes_pct"]) == \
        ["D128 delta_matching (matched)"]


def test_stack_growth_is_a_warning_and_the_ab_still_runs(tmp_path):
    r = run_gate(tmp_path, "full", "", extra={"STUB_RES": "stackgrow"})
    rec = json.load(open(tmp_path / "run_full_none" / "gate.json"))
    assert r.returncode == 0 and "WARNING: per-symbol stack" in r.stdout and "ab_all: rc 0" in r.stdout
    assert rec["resource_regressions"] == ["_Z9KernelX_causal: STACK 8 -> 16"]


def test_prebuilt_library_is_not_rebuilt_and_build_diagnostics_fail(tmp_path):
    lib = tmp_path / "pre.so"; lib.write_bytes(b"lib")
    log = tmp_path / "build.log"; log.write_text("BUILD-OK\n")
    r = run_gate(tmp_path, "timing", "jit", args=["--lib", str(lib), "--build-log", str(log)])
    assert r.returncode == 0 and "step build" not in r.stdout and "build:" not in r.stdout, r.stdout[-600:]
    log.write_text("ptxas warning : (C7510) wgmma serialized\n")
    (tmp_path / "x").mkdir()
    r = run_gate(tmp_path / "x", "timing", "", args=["--lib", str(lib), "--build-log", str(log)])
    assert r.returncode == 1 and "build diagnostics" in r.stdout, r.stdout[-600:]


def _steps(tmp_path, name):
    return [st["step"] for st in json.load(open(tmp_path / name / "gate.json"))["steps"]]


def test_tooling_tests_do_not_run_by_default(tmp_path):
    r = run_gate(tmp_path, "full", "tooling")
    assert r.returncode == 0 and "tooling_tests" not in _steps(tmp_path, "run_full_tooling"), r.stdout[-400:]


def test_tooling_tests_run_when_requested(tmp_path):
    r = run_gate(tmp_path, "full", "", args=["--tooling-tests"])
    assert r.returncode == 0 and _steps(tmp_path, "run_full_none")[0] == "tooling_tests", r.stdout[-400:]


@pytest.mark.parametrize("kind", ["regular", "symlink"])
def test_lib_pointing_at_the_package_entry_is_kept_intact(tmp_path, kind):
    wt = tmp_path / "wt"; (wt / "delta_matching").mkdir(parents=True, exist_ok=True)
    log = tmp_path / "build.log"; log.write_text("BUILD-OK\n")
    if kind == "regular":
        entry = wt / "delta_matching" / "_C.so"; entry.write_bytes(b"lib")
        target = entry
    else:
        target = tmp_path / "jitdir_C.so"; target.write_bytes(b"lib")
        (wt / "delta_matching" / "_C.so").symlink_to(target)
    r = run_gate(tmp_path, "timing", "jit", args=["--lib", str(wt / "delta_matching" / "_C.so"), "--build-log", str(log)])
    assert r.returncode == 0 and "build" not in _steps(tmp_path, "run_timing_jit"), r.stdout[-600:]
    assert target.read_bytes() == b"lib" and os.path.realpath(wt / "delta_matching" / "_C.so") == os.path.realpath(target)


@pytest.mark.parametrize("text,fatal", [
    ("ptxas info    : 0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n", False),
    ("nvcc -o /tmp/spill_test/serialize_kernels.o -c x.cu\n", False),
    ("ptxas info    : 32 bytes stack frame, 16 bytes spill stores, 16 bytes spill loads\n", True),
    ("ptxas warning : (C7510) Potential Performance Loss: wgmma.mma_async instructions are serialized\n", True),
    ("ptxas warning : wgmma serialized due to function call\n", True),
    ("ptxas warning : Registers are spilled to local memory in function 'X'\n", True),
    ("ptxas info    : 0 bytes stack frame, 8 bytes spill stores\n", True),
    ("ptxas info    : 4 bytes spill loads\n", True)])
def test_build_diagnostics_parse_counts_not_words(tmp_path, text, fatal):
    log = tmp_path / "b.log"; log.write_text("BUILD-OK\n" + text)
    sys.path.insert(0, HERE)
    import gate as G
    assert bool(G.build_diagnostics(str(log))) == fatal


def test_full_gate_fails_on_a_spill_warning_only_build_log(tmp_path):
    lib = tmp_path / "pre.so"; lib.write_bytes(b"lib")
    log = tmp_path / "build.log"; log.write_text("BUILD-OK\nptxas warning : Registers are spilled to local memory in function 'X'\n")
    r = run_gate(tmp_path, "full", "", args=["--lib", str(lib), "--build-log", str(log)])
    assert r.returncode == 1 and "build diagnostics" in r.stdout, r.stdout[-600:]


# ---- numerics.py (schema v2): synthetic captures ----
import hashlib as _hl  # noqa: E402
import numerics as NUM  # noqa: E402

_H = lambda t: _hl.sha256(t.encode()).hexdigest()


def _sig(tag):
    return {"bytes": _H("b" + tag), "nan_mask": _H("n"), "inf_mask": _H("i"), "finite": _H("f" + tag), "n_nan": 0, "n_inf": 0}


def _cap(dbg=False, route="default", err=1e-3, code_sha="c0de", lib="L"):
    cases = EQ.selected_cases(route)
    out = {"schema": NUM.SCHEMA, "route": route, "dbg": dbg,
           "identity": {"library_sha256": _H(lib + str(dbg)), "python_sha256": _H("py"), "code_sha": code_sha,
                        "flags": {"F": 1}, "dbg_build": dbg, "prof": False, "trace": False},
           "descriptors_sha256": _hl.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest(), "cases": {},
           "resolved_route": {"groups": "auto", "streams": 1}}
    for key, desc in cases:
        c = {"errors": {o: err for o in NUM.OUTPUTS}, "outputs": {o: _sig(key + o) for o in NUM.OUTPUTS},
             "reference_finite": True, "input_id": _H("in" + key), "ref_id": _H("ref" + key)}
        if dbg:
            sch = NUM.expected_group_schema(desc, route)
            gs = []
            for gi in range(sch["groups"]):
                g = {"heads": sch["heads"], "problems": [], "visited_tiles": sch["visited_tiles"],
                     "region_hashes": [_H(f"{key}{gi}a"), _H(f"{key}{gi}b")]}
                if desc["mode"] == "matched":
                    g.update(residual_plan=0.5, residual_centered=1e-6, zero_den_nonzero_num=0)
                gs.append(g)
            c["groups"] = gs
        out["cases"][key] = c
    return out


def _rows(route="default"):
    """residual-v2 sidecar: every matched group's rows at half the derived bound; row 0 (one key) exempt."""
    import torch
    side = {}
    for key, desc in EQ.selected_cases(route):
        if desc["mode"] != "matched":
            continue
        sch = NUM.expected_group_schema(desc, route)
        n = desc["N"]
        keys = (torch.arange(n) + 1).clamp(max=n).to(torch.int64).view(1, -1).expand(desc["B"] * sch["heads"], n).contiguous()
        for gi in range(sch["groups"]):
            den = torch.ones(keys.shape, dtype=torch.float64)
            num = keys.double() * 2.0 ** -24
            num[:, 0] = 1.0                       # one visible key: ratio 1, exempt (degenerate)
            side[f"{key}|{gi}"] = {"num": num, "den": den, "keys": keys.clone()}
    return side


def _numcheck(tmp_path, mutate=None, ws="same", manifest=None, extra=(), rows_mutate=None):
    import torch
    files = {"base": _cap(lib="base"), "prev": _cap(lib="prev"), "cand": _cap(), "cand-repeat": _cap(),
             "cand-dbg": _cap(True), "cand-dbg-repeat": _cap(True), "prev-dbg": _cap(True, lib="prev")}
    sides = {t: _rows() for t in ("cand-dbg", "cand-dbg-repeat", "prev-dbg")}
    if rows_mutate:
        rows_mutate(sides)
    for t, sd in sides.items():
        sp = tmp_path / f"{t}.json.rows.pt"; torch.save(sd, sp)
        files[t]["rows_sidecar"] = {"file": sp.name, "sha256": EQ.sha256(str(sp)), "version": NUM.RESIDUAL_VERSION}
    man = manifest if manifest is not None else {"route": "default", "base_library_sha256": files["base"]["identity"]["library_sha256"],
                                                  "allow_repair": False, "cases": {}}
    if mutate:
        mutate(files, man)
    args = ["check", "--route", "default", "--ws", ws] + list(extra)
    for tag, d in files.items():
        if d is None:
            continue
        p = tmp_path / f"{tag}.json"; p.write_text(json.dumps(d)); args += [f"--{tag}", str(p)]
    mp = tmp_path / "defects.json"; mp.write_text(json.dumps(man)); args += ["--defects", str(mp)]
    return subprocess.run([sys.executable, os.path.join(HERE, "numerics.py")] + args, capture_output=True, text=True)


def _c(f, tag, key="d128_gqa4/causal/matched"):
    return f[tag]["cases"][key]


def test_numerics_check_passes_a_self_certificate(tmp_path):
    r = _numcheck(tmp_path, extra=("--report", str(tmp_path / "rep.json")))
    assert r.returncode == 0 and "PASS" in r.stdout, r.stdout[-800:]
    rep = json.load(open(tmp_path / "rep.json"))
    assert rep["ac23_eligible"] is True and "residual-v4" in rep["residual_versions"]["rows"]
    k = rep["cases"]["d128_gqa4/causal/matched"]
    assert k["exempt"] > 0 and k["exempt"] == k["prev_exempt"] and k["violations"] == 0


@pytest.mark.parametrize("why,mutate", [
    ("missing cases", lambda f, m: f["cand"]["cases"].pop("d128_min/causal/matched")),
    ("non-finite error", lambda f, m: _c(f, "cand")["errors"].update(dQ=float("nan"))),
    ("non-finite error", lambda f, m: _c(f, "cand-dbg")["outputs"]["dK"].update(n_nan=3)),
    ("error 2.0", lambda f, m: _c(f, "cand")["errors"].update(dK=2e-3)),
    ("nondeterministic", lambda f, m: _c(f, "cand-repeat")["outputs"]["O"].update(bytes=_H("x"))),
    ("DBG-build run's outputs differ", lambda f, m: _c(f, "cand-dbg-repeat")["outputs"]["O"].update(bytes=_H("x"))),
    ("capture required", lambda f, m: f.update({"cand-dbg": None})),
    ("groups", lambda f, m: [_c(f, t)["groups"].pop() for t in ("cand-dbg", "cand-dbg-repeat", "prev-dbg")]),
    ("coverage / region hashes invalid", lambda f, m: [_c(f, t)["groups"][0].update(visited_tiles=0, region_hashes=[None, None])
                                                      for t in ("cand-dbg", "cand-dbg-repeat", "prev-dbg")]),
    ("workspaces differ between", lambda f, m: _c(f, "cand-dbg-repeat")["groups"][0].update(region_hashes=[_H("p"), _H("q")])),
    ("declared unchanged", lambda f, m: [_c(f, t)["groups"][0].update(region_hashes=[_H("p"), _H("q")]) for t in ("cand-dbg", "cand-dbg-repeat")]),
    ("out of range", lambda f, m: [_c(f, t)["groups"][0].update(residual_plan=1.5) for t in ("cand-dbg", "cand-dbg-repeat")]),
    ("out of range", lambda f, m: [_c(f, t)["groups"][0].update(residual_centered=float("nan")) for t in ("cand-dbg", "cand-dbg-repeat")]),
    ("missing input / reference identity", lambda f, m: _c(f, "cand").update(input_id=None)),
    ("missing output hashes", lambda f, m: _c(f, "cand")["outputs"]["dQ"].update(bytes=None)),
    ("missing build stamp identity", lambda f, m: [f[t]["identity"].update(flags=None) for t in f]),
    ("instrumented", lambda f, m: f["cand"]["identity"].update(trace=True)),
    ("not a DBG build of the previous", lambda f, m: f["prev-dbg"]["identity"].update(code_sha="older")),
    ("different backward routes", lambda f, m: f["cand"].update(resolved_route={"groups": "kv", "streams": 1})),
    ("row sidecar", lambda f, m: f["cand-dbg"]["rows_sidecar"].update(sha256=_H("tampered"))),
    ("row sidecar", lambda f, m: f["prev-dbg"].pop("rows_sidecar")),
    ("zero-denominator", lambda f, m: [_c(f, t)["groups"][0].update(zero_den_nonzero_num=2) for t in ("cand-dbg", "cand-dbg-repeat")]),
    ("missing library", lambda f, m: [f[t]["identity"].update(library_sha256=None) for t in f]),
    ("not from the candidate's source", lambda f, m: [f[t]["identity"].update(code_sha="other") for t in ("cand-dbg", "cand-dbg-repeat")]),
    ("different numerical flags", lambda f, m: [f[t]["identity"].update(flags={"F": 2}) for t in ("cand-dbg", "cand-dbg-repeat")]),
    ("not a DBG build", lambda f, m: f["cand-dbg"]["identity"].update(dbg_build=False)),
    ("non-finite fp32 reference", lambda f, m: _c(f, "prev").update(reference_finite=False)),
    ("input / reference identities", lambda f, m: _c(f, "prev").update(ref_id=_H("other"))),
    ("workspace", lambda f, m: _c(f, "cand-dbg")["groups"][0]["problems"].append("NaN e4m3 code on a visited tile")),
])
def test_numerics_check_rejects(tmp_path, why, mutate):
    r = _numcheck(tmp_path, mutate)
    assert r.returncode == 1 and why in r.stdout, (why, r.stdout[-800:])


_G = "d128_gqa4/causal/matched|0"


def _both(fn):
    return lambda s: [fn(s[t]) for t in ("cand-dbg", "cand-dbg-repeat")]


@pytest.mark.parametrize("why,rows_mutate", [
    # one row worsens past its accepted value + round-off while the group's plan maximum stays 1 (row 0, exempt)
    ("above their accepted residual", _both(lambda s: s[_G]["num"].__setitem__((0, 7), 8 * 2.0 ** -22))),
    ("exemption set grows", _both(lambda s: (s[_G]["num"].__setitem__((0, 9), 0.0), s[_G]["den"].__setitem__((0, 9), 0.0)))),
    ("zero-denominator rows with a nonzero sum", _both(lambda s: s[_G]["den"].__setitem__((0, 9), 0.0))),
    ("nondeterministic", lambda s: s["cand-dbg-repeat"][_G]["num"].__setitem__((0, 5), 0.0)),
    ("visible-key counts differ", _both(lambda s: s[_G]["keys"].__setitem__((0, 5), 99))),
    ("row records missing", _both(lambda s: s.pop(_G))),
])
def test_numerics_residual_v2_rejects(tmp_path, why, rows_mutate):
    r = _numcheck(tmp_path, rows_mutate=rows_mutate)
    assert r.returncode == 1 and why in r.stdout, (why, r.stdout[-800:])


def _err(f, key, o, v, tags=("cand", "cand-repeat", "cand-dbg", "cand-dbg-repeat")):
    for t in tags:
        _c(f, t, key)["errors"][o] = v


def test_numerics_v4_error_tolerance_and_mean(tmp_path):
    k2 = "d128_min/causal/matched"
    up_down = lambda f, m: (_err(f, "d128_gqa4/causal/matched", "dK", 1e-3 + 5e-6), _err(f, k2, "dK", 1e-3 - 6e-6))
    assert _numcheck(tmp_path, up_down).returncode == 0                     # +5e-6 in tolerance, mean not rising
    d = tmp_path / "mean"; d.mkdir()
    r = _numcheck(d, lambda f, m: _err(f, "d128_gqa4/causal/matched", "dK", 1e-3 + 5e-6))
    assert r.returncode == 1 and "mean dK error" in r.stdout, r.stdout[-600:]
    d = tmp_path / "big"; d.mkdir()
    r = _numcheck(d, lambda f, m: (_err(f, "d128_gqa4/causal/matched", "dK", 1e-3 + 2e-5), _err(f, k2, "dK", 1e-3 - 5e-5)))
    assert r.returncode == 1 and "+ 1e-05" in r.stdout, r.stdout[-600:]


def test_numerics_v4_stale_cases_have_the_correctness_tolerance(tmp_path):
    st = [k for k, d in EQ.selected_cases("default") if d["mode"] != "matched"][:2]
    ok = lambda f, m: (_err(f, st[0], "dQ", 1e-3 + 5e-4), _err(f, st[1], "dQ", 1e-3 - 6e-4))
    assert _numcheck(tmp_path, ok).returncode == 0                          # +5e-4 on a stale case, mean not rising
    d = tmp_path / "x"; d.mkdir()
    r = _numcheck(d, lambda f, m: (_err(f, st[0], "dQ", 1e-3 + 2e-3), _err(f, st[1], "dQ", 1e-3 - 3e-3 + 1e-9)))
    assert r.returncode == 1 and "+ 0.001" in r.stdout, r.stdout[-600:]


def test_numerics_v4_ill_conditioned_rows_exempt_but_fixed_by_the_accepted_build(tmp_path):
    def ill(s):                               # row 7 has residual 0.5 in the accepted build: ill-conditioned
        for t in s:
            s[t][_G]["num"][0, 7] = 0.5
        for t in ("cand-dbg", "cand-dbg-repeat"):
            s[t][_G]["num"][0, 7] = 0.9       # moves freely (reported)
    r = _numcheck(tmp_path, rows_mutate=ill, extra=("--report", str(tmp_path / "r.json")))
    assert r.returncode == 0, r.stdout[-600:]
    c = json.load(open(tmp_path / "r.json"))["cases"]["d128_gqa4/causal/matched"]
    assert c["ill_conditioned"] == 1 and abs(c["ill_worst_move"] - 0.4) < 1e-12
    d = tmp_path / "new"; d.mkdir()          # a well-conditioned accepted row cannot become ill-conditioned
    r = _numcheck(d, rows_mutate=_both(lambda s: s[_G]["num"].__setitem__((0, 7), 0.5)))
    assert r.returncode == 1 and "above their accepted residual" in r.stdout, r.stdout[-600:]


def test_numerics_residual_v3_absolute_bound_is_reported_not_enforced(tmp_path):
    def both_high(s):                         # accepted AND candidate above the absolute bound, equal: passes (v3)
        for t in s:
            s[t][_G]["num"][0, 7] = 8 * 2.0 ** -22
    r = _numcheck(tmp_path, rows_mutate=both_high, extra=("--report", str(tmp_path / "r.json")))
    assert r.returncode == 0, r.stdout[-600:]
    assert json.load(open(tmp_path / "r.json"))["cases"]["d128_gqa4/causal/matched"]["violations"] == 1


def test_numerics_residual_v2_regression_vs_the_accepted_row(tmp_path):
    import torch
    def prev_better(s):                       # the accepted build had row 7 at 0; the candidate's is still in bound
        s["prev-dbg"][_G]["num"][0, 7] = 0.0
    r = _numcheck(tmp_path, rows_mutate=prev_better)
    assert r.returncode == 0, r.stdout[-600:]  # +k 2^-24 <= round-off: not a regression
    d = tmp_path / "x"; d.mkdir()
    def exempt_shrinks(s):                    # an exempt accepted row becoming non-exempt (in bound) is allowed
        for t in ("cand-dbg", "cand-dbg-repeat"):
            s[t][_G]["num"][0, 0] = 0.0
    assert _numcheck(d, rows_mutate=exempt_shrinks).returncode == 0
    assert torch.equal(NUM.residual_v2(torch.tensor([[1.0]], dtype=torch.float64), torch.tensor([[1.0]], dtype=torch.float64),
                                       torch.tensor([[1]]))[0], torch.tensor([[True]]))


def _nan_case(f, m, key="d128_zeros/causal/matched", sigs=("base", "prev", "cand", "cand-repeat", "cand-dbg",
                                                              "cand-dbg-repeat", "prev-dbg")):
    for t in sigs:
        o = f[t]["cases"][key]
        o["errors"]["O"] = float("nan"); o["outputs"]["O"].update(n_nan=7, nan_mask=_H("mask"), finite=_H("fin"))
    m["cases"][key] = {"outputs": {"O": {"nan_mask": _H("mask"), "inf_mask": _H("i"), "finite": _H("fin"), "n_nan": 7, "n_inf": 0}},
                       "workspace_problems": []}


def test_numerics_known_defects_pass_qualified_and_are_bound(tmp_path):
    r = _numcheck(tmp_path, _nan_case)
    assert r.returncode == 0 and "qualified by known defects" in r.stdout, r.stdout[-600:]
    for i, extra in enumerate([lambda f, m: f["cand"]["cases"]["d128_zeros/causal/matched"]["outputs"]["O"].update(nan_mask=_H("moved")),
                               lambda f, m: f["cand-repeat"]["cases"]["d128_zeros/causal/matched"]["outputs"]["O"].update(finite=_H("changed")),
                               lambda f, m: m.update(base_library_sha256=_H("someone else")),
                               lambda f, m: f["cand-dbg"]["cases"]["d128_zeros/causal/matched"]["groups"][0]["problems"].append("psi (1,) shape")]):
        d = tmp_path / f"k{i}"; d.mkdir()
        r = _numcheck(d, lambda f, m: (_nan_case(f, m), extra(f, m)))
        assert r.returncode == 1, (i, r.stdout[-600:])


def test_numerics_gate_needs_all_four_routes(tmp_path):
    for n in ("base", "cdbg", "pdbg"):
        (tmp_path / n / "delta_matching").mkdir(parents=True)
        (tmp_path / n / "delta_matching" / "_C.so").write_bytes(b"x")
    r = run_gate(tmp_path, "numerics", "", args=["--routes", "default", "--base-root", str(tmp_path / "base"),
                                                 "--cand-dbg-root", str(tmp_path / "cdbg"), "--prev-dbg-root",
                                                 str(tmp_path / "pdbg"), "--defects", str(tmp_path / "m.json")])
    assert r.returncode == 1 and "exactly the four routes" in r.stdout, r.stdout[-600:]


def test_check_workspace_schema_nan_codes_and_per_half_sentinel():
    desc = {"D": 128, "B": 1, "H": 2, "Hk": 2, "N": 256, "attn": "causal"}
    sch = NUM.expected_group_schema(desc, "default")
    ds8 = torch.zeros(sch["ds8"], dtype=torch.float8_e4m3fn); psi = torch.ones(sch["psi"])
    rows = torch.zeros(sch["rows"]); ra = torch.zeros(sch["rows"])
    probs, regions, nvis = NUM.check_workspace(ds8, psi, rows, ra, sch, True)
    assert not probs and len(regions) == 2 and nvis == sch["visited_tiles"]
    bad = ds8.view(torch.uint8).clone(); bad[0, 0, 0, 0, 0] = 0x7F
    assert any("NaN e4m3" in p for p in NUM.check_workspace(bad.view(torch.float8_e4m3fn), psi, rows, ra, sch, True)[0])
    empty = torch.zeros([0] + sch["ds8"][1:], dtype=torch.float8_e4m3fn)
    assert NUM.check_workspace(empty, psi, rows, ra, sch, True)[1] is None
    sch2 = NUM.expected_group_schema(desc, "default", psi_halves=2)
    psi2 = torch.ones(sch2["psi"]); psi2[0, 0, 0, 1] = 0.0
    codes = ds8.view(torch.uint8).clone(); codes[0, 0, 0, 3, 100] = 0x38   # key 100 is in half 1 (64..127)
    p = NUM.check_workspace(codes.view(torch.float8_e4m3fn), psi2, rows, ra, sch2, True)[0]
    assert any("zero-psi key half 1" in x for x in p), p
    incons = ra.clone(); r2 = rows.clone(); r2[0, 0, 0, 0] = 1.0
    assert any("inconsistent" in x for x in NUM.check_workspace(ds8, psi, r2, incons, sch, True)[0])


def test_row_residual_and_workspace_validation():
    rows = torch.zeros(1, 2, 2, 2)
    rows[0, 0, 0] = torch.tensor([1.0, 2.0]); rows[0, 1, 0] = torch.tensor([-1.0, -2.0])
    ratio, bad, _ = NUM.row_residual(rows, rows.abs())
    assert ratio == 0.0 and bad == 0
    rows[0, 1, 1, 0] = 0.5
    assert abs(NUM.row_residual(rows, rows.abs())[0] - 0.2) < 1e-12



def test_row_residual_plan_is_ill_conditioned_and_the_diagnostic_is_not():
    rows = torch.zeros(1, 1, 2, 1); rows[0, 0, 0, 0] = 1e-6          # a one-key row: one tiny residue
    r, _, worst = NUM.row_residual(rows, rows.abs())
    assert r == 1.0 and worst["num"] == worst["den"] > 0             # the plan's ratio: always 1 on such a row
    delta = torch.full((1, 1), 1.0)
    assert NUM.row_residual(rows, rows.abs(), delta)[0] < 1e-8         # residue / (|t| + 2^8 |delta|)



# ---- sol4-v2 (four-floor model, round 16) ----
import sol4 as S4  # noqa: E402

_TOY = {"symbol": "toy", "unit": (64, 64), "wgs": 1, "step_elems_per_wg": 64 * 64, "gemm_flop_per_elem": 64,
        "work_tile_axis": 0, "peel": 1}                       # one QGMMA.64x64x32 (262144 FLOPs) per step


def _sass(lines, name="toy"):
    return f"\t\tFunction : {name}\n" + "".join(f"        /*{a:04x}*/  {t} ;\n" for a, t in lines)


def _ledger(lines):
    return S4.kernel_ledger(S4.parse_sass(_sass(lines))["toy"], _TOY)


def test_sol4_ledger_skips_rare_arm_and_never_sums_exclusive_arms():
    L = [(0x00, "MOV R0, R1"), (0x10, "QGMMA.64x64x32.F32.E4M3.E4M3 R24, gdesc[UR4], R24"), (0x20, "@P0 BRA 0x60"),
         (0x30, "FMUL R1, R1, R2"), (0x40, "FMUL R1, R1, R2"), (0x50, "FMUL R1, R1, R2"),   # promotion: rarely taken
         (0x60, "@P1 BRA 0x90"), (0x70, "FADD R1, R1, R2"), (0x80, "BRA 0xa0"),             # arm A (2 instructions)
         (0x90, "FMUL R1, R1, R2"),                                                         # arm B (1 instruction)
         (0xa0, "MUFU.EX2 R3, R3"), (0xb0, "@P2 MUFU.RCP R4, R4"), (0xc0, "@P3 BRA 0x0"), (0xd0, "EXIT")]
    led = _ledger(L)
    loop = [r for r in led["loops"] if r["innermost"]][0]
    assert loop["static_instr"] == 13 and loop["static_mufu"] == 2
    assert led["per_step_instr"] == 8      # 0x00 0x10 0x20 0x60 0x90 0xa0 0xb0 0xc0 (hand count)
    assert led["per_step_mufu"] == 1       # EX2 counted, the predicated RCP not (lane participation unknown)


def test_sol4_ledger_counts_one_of_two_exclusive_qgmma_arms():
    L = [(0x00, "MOV R0, R1"), (0x10, "@P0 BRA 0x40"), (0x20, "QGMMA.64x64x32.F32.E4M3.E4M3 R24, gdesc[UR4], RZ, !UPT"),
         (0x30, "BRA 0x50"), (0x40, "QGMMA.64x64x32.F32.E4M3.E4M3 R24, gdesc[UR4], R24"), (0x50, "@P1 BRA 0x0")]
    led = _ledger(L)
    loop = [r for r in led["loops"] if r["innermost"]][0]
    assert loop["static_qgmma"] == 2 and loop["steps"] == 1 and led["per_step_instr"] == 4   # 0x00 0x10 0x40 0x50


def test_sol4_ledger_charges_the_full_work_path_and_flags_data_dependent_skips():
    L = [(0x00, "@P0 BRA 0x30"), (0x10, "FMUL R1, R1, R2"), (0x20, "QGMMA.64x64x32.F32.E4M3.E4M3 R24, gdesc[UR4], R24"),
         (0x30, "@P1 BRA 0x0")]
    led = _ledger(L)
    loop = [r for r in led["loops"] if r["innermost"]][0]
    assert led["per_step_instr"] == 4 and loop["path_instr_any"] == 2 and loop["data_dependent_skip"]


def test_sol4_ledger_picks_the_innermost_step_loop_and_rejects_unsupported_flow():
    inner = [(0x10, "MOV R0, R1"), (0x20, "QGMMA.64x64x32.F32.E4M3.E4M3 R24, gdesc[UR4], R24"), (0x30, "@P1 BRA 0x10")]
    L = [(0x00, "MOV R0, R1")] + inner + [(0x40, "QGMMA.64x64x32.F32.E4M3.E4M3 R24, gdesc[UR4], R24"), (0x50, "@P2 BRA 0x0")]
    led = _ledger(L)
    assert led["step_loops"] == [(0x10, 0x30)] and led["per_step_instr"] == 3
    with pytest.raises(S4.ModelError, match="no step loop"):
        _ledger([(0x00, "BRX R2"), (0x10, "QGMMA.64x64x32.F32.E4M3.E4M3 R24, gdesc[UR4], R24"), (0x20, "@P1 BRA 0x0")])
    with pytest.raises(S4.ModelError, match="no step loop"):     # QGMMA work not a whole declared step
        _ledger([(0x00, "QGMMA.64x32x32.F32.E4M3.E4M3 R24, gdesc[UR4], R24"), (0x10, "@P1 BRA 0x0")])


def _calib4(gbs=3000.0, job="7"):
    c = {"version": 1, "sms": 100, "sm_clock_mhz": 2000.0, "gemm_fp8": {"peak_tflops": 1000.0},
         "copy": {"peak_gbs": gbs}, "job": job, "host": "h"}
    c["sha256"] = S4.calib_digest(c)
    return c


def _counts4(lib="L", version=S4.COUNTS_VERSION):
    k = lambda i, m: {"per_step_instr": i, "per_step_mufu": m}
    return {"version": version, "declared": json.loads(json.dumps(S4.KERNELS_D128_CAUSAL)), "library_sha256": lib,
            "kernels": {"fwd": k(1000, 128), "pass1": k(600, 32), "pass0": k(280, 32), "pass2": k(70, 0)}}


def _rec4(B=1, H=16, Hk=4, N=4096, names=("_quant_strided_kernel",)):
    return {"D": 128, "B": B, "H": H, "Hk": Hk, "N": N, "attn": "causal", "mode": "matched",
            "ms": {c: 1.0 for c in S4.CATS}, "kernels": {n: 0.1 for n in names}}


def _runs4(calib, recs=None, lib="L", job="7"):
    recs = recs if recs is not None else [_rec4(B, H, Hk, N) for D, B, H, Hk, N in K.PAPER_SET]
    for r in recs:
        if S4.heads_per_call(r["B"], r["H"], r["Hk"], r["N"]) < r["H"]:
            r["kernels"]["CatArrayBatchedCopy_x"] = 0.1
    return {"meta": {"calibration_sha256": calib["sha256"], "library_sha256": lib, "job": job}, "records": recs}


def test_sol4_hand_computed_fwd_floors_with_peeled_steps():
    f, _ = S4.floors(_rec4(B=1, H=1, Hk=1), _calib4(), _counts4())
    f = f["fwd"]["floors_ms"]
    units = S4.visited_units(4096, 128, 256, True)
    assert units == sum(min(16, -(-((mb + 1) * 128) // 256)) for mb in range(32))
    assert abs(f["tensor"] - units * 128 * 256 * 512 / 1e15 * 1e3) < 1e-12
    loop_steps = units * 2 - 32 * 2                      # 2 warpgroups; the first n-block of each of 32 m-blocks peeled
    assert abs(f["issue"] - loop_steps * 4 * 1000 / (4 * 100 * 2e9) * 1e3) < 1e-12
    assert abs(f["mufu"] - loop_steps * 4 * 128 * 32 / (16 * 100 * 2e9) * 1e3) < 1e-12


def test_sol4_glue_ledger_follows_the_route():
    ND = 32768 * 128
    split = S4.glue_bytes(128, 1, 16, 16, 32768, "strided", 1)           # MHA 32k, B1: per-head slices are views
    assert split == 3 * 2 * 2 * 16 * ND                                   # only the output concatenation
    grp = S4.glue_bytes(128, 1, 16, 4, 32768, "strided", 4)               # per-KV-group calls: dO8 / Q8 copied
    assert grp == 2 * 2 * 16 * ND + 3 * 2 * 2 * 16 * ND + 2 * 2 * 16 * ND + 2 * 2 * 4 * ND   # + the GQA fold
    b2 = S4.glue_bytes(128, 2, 32, 8, 16384, "strided", 4)                # B > 1: every slice is copied
    ND2, calls = 16384 * 128, 8
    assert b2 == (2 * 2 * 64 * ND2 + 2 * 4 * 64 * 16384 + 2 * 2 * 64 * ND2 + 2 * 2 * 2 * ND2 * calls + 2 * 2 * ND2 * calls
                  + 3 * 2 * 2 * 64 * ND2 + 2 * 2 * 64 * ND2 + 2 * 2 * 16 * ND2)
    assert S4.glue_bytes(128, 2, 16, 16, 8192, "strided", 16) == 0       # unsplit MHA, strided producers
    fold = 2 * 2 * 8 * 16 * 4096 * 128 + 2 * 2 * 8 * 4 * 4096 * 128
    assert S4.glue_bytes(128, 8, 16, 4, 4096, "strided", 16) == fold
    assert S4.glue_bytes(128, 8, 16, 4, 4096, "copy", 16) == fold + 2 * 2 * (8 * 16 + 2 * 8 * 4) * 4096 * 128
    r = S4.record_route(_rec4(1, 16, 16, 32768, names=("_quant_strided_kernel", "CatArrayBatchedCopy_x")))
    assert r == {"producers": "strided", "heads_per_call": 1, "recorded": False}
    assert S4.record_route(_rec4(names=("_quant_kernel",)))["producers"] == "copy"     # the contiguous fallback
    with pytest.raises(S4.ModelError, match="concatenation"):
        S4.record_route(_rec4(1, 16, 16, 32768))                                       # split rule vs. no cat kernel
    bad = dict(_rec4(), route={"producers": "copy", "heads_per_call": 16, "streams": 1})
    with pytest.raises(S4.ModelError, match="disagrees"):
        S4.record_route(bad)


def test_sol4_inputs_bound_before_modeling():
    c = _calib4()
    assert len(S4.validate_inputs(_runs4(c), c, _counts4())) == 4
    tampered = dict(c, copy={"peak_gbs": 1.0})                            # content changed, old digest kept
    with pytest.raises(S4.ModelError, match="does not match its sha256"):
        S4.validate_inputs(_runs4(c), tampered, _counts4())
    other = _calib4(gbs=2000.0)
    with pytest.raises(S4.ModelError, match="another calibration"):
        S4.validate_inputs(_runs4(c), other, _counts4())
    with pytest.raises(S4.ModelError, match="not the timed library"):
        S4.validate_inputs(_runs4(c), c, _counts4(lib="M"))
    with pytest.raises(S4.ModelError, match="counts version"):
        S4.validate_inputs(_runs4(c), c, _counts4(version="not-sol4"))
    with pytest.raises(S4.ModelError, match="job differ"):
        S4.validate_inputs(_runs4(c, job="8"), c, _counts4())
    runs = _runs4(c); runs["records"].pop()                               # a configuration missing
    with pytest.raises(S4.ModelError, match="exactly once"):
        S4.validate_inputs(runs, c, _counts4())
    runs = _runs4(c); runs["records"][-1] = dict(runs["records"][0])      # a configuration duplicated
    with pytest.raises(S4.ModelError, match="exactly once"):
        S4.validate_inputs(runs, c, _counts4())
    runs = _runs4(c); runs["records"][0]["ms"]["fwd"] = float("nan")
    with pytest.raises(S4.ModelError, match="malformed"):
        S4.validate_inputs(runs, c, _counts4())


def test_sol4_memory_kernels_report_issue_unmodeled():
    c = _calib4()
    res = S4.model_records(S4.validate_inputs(_runs4(c), c, _counts4()), c, _counts4())
    t = res["totals"]["acceptance"]["kernels"]
    assert t["glue"]["floors_ms"]["issue"] is None and t["fwd"]["floors_ms"]["issue"] > 0
    assert "| glue |" in "\n".join(S4.table(res)) and "n/a" in "\n".join(S4.table(res))


def test_median_records_aggregates_fa3_over_the_repetitions():
    def rec(fp8, bf16):
        return {"D": 128, "B": 1, "H": 1, "Hk": 1, "N": 4096, "attn": "causal", "mode": "matched", "e2e_fwd_ms": 1.0,
                "e2e_fwdbwd_ms": 2.0, "ms": {c: 1.0 for c in K.CATS},
                "fa3": {"fp8_fwd_kernel_ms": fp8, "bf16_fwd_ms": bf16, "note": "x"}}
    med = ab.median_records([[rec(5.0, 1.0)], [rec(1.0, 2.0)], [rec(2.0, 3.0)]])[0]
    assert med["fa3"]["fp8_fwd_kernel_ms"] == 2.0 and med["fa3"]["bf16_fwd_ms"] == 2.0 and med["fa3"]["note"] == "x"
    assert abs(med["fa3_spread_pct"]["fp8_fwd_kernel_ms"] - 200.0) < 1e-9
    with pytest.raises(SystemExit, match="missing"):
        ab.median_records([[rec(1.0, 1.0)], [dict(rec(1.0, 1.0), fa3=None)], [rec(1.0, 1.0)]])


def test_gate_lib_stamp_handling(tmp_path):
    # package-resident library with its own stamp: no SameFileError, stamp intact
    wt = tmp_path / "wt"; (wt / "delta_matching").mkdir(parents=True)
    lib = wt / "delta_matching" / "_C.so"; lib.write_bytes(b"lib")
    (wt / "delta_matching" / "_build_info.py").write_text("BUILD_INFO = {}\n")
    log = tmp_path / "build.log"; log.write_text("BUILD-OK\n")
    r = run_gate(tmp_path, "timing", "jit", args=["--lib", str(lib), "--build-log", str(log)])
    assert r.returncode == 0 and (wt / "delta_matching" / "_build_info.py").exists(), r.stdout[-600:]
    # external library without a sidecar: an unrelated package stamp is removed
    (tmp_path / "x").mkdir(); ext = tmp_path / "x" / "ext.so"; ext.write_bytes(b"lib")
    (tmp_path / "y").mkdir()
    wt2 = tmp_path / "y" / "wt"; (wt2 / "delta_matching").mkdir(parents=True)
    (wt2 / "delta_matching" / "_build_info.py").write_text("BUILD_INFO = {'stale': 1}\n")
    r = run_gate(tmp_path / "y", "timing", "jit", args=["--lib", str(ext), "--build-log", str(log)])
    assert r.returncode == 0 and not (wt2 / "delta_matching" / "_build_info.py").exists(), r.stdout[-600:]


def test_numerics_gate_accepts_only_residual_v2_eligible_reports(tmp_path):
    (tmp_path / "delta_matching").mkdir(); (tmp_path / "delta_matching" / "_C.so").write_bytes(b"prev")   # --prev-root
    for n in ("base", "cdbg", "pdbg"):
        (tmp_path / n / "delta_matching").mkdir(parents=True)
        (tmp_path / n / "delta_matching" / "_C.so").write_bytes(b"x")
    r = run_gate(tmp_path, "numerics", "", args=["--base-root", str(tmp_path / "base"), "--cand-dbg-root",
                                                 str(tmp_path / "cdbg"), "--prev-dbg-root", str(tmp_path / "pdbg"),
                                                 "--defects", str(tmp_path / "m.json")])
    assert r.returncode == 1 and "not AC-2.3 eligible under residual-v4" in r.stdout, r.stdout[-600:]
