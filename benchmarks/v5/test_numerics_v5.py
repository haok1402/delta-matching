"""CPU unit tests of the residual-v5 check on synthetic captures (python -m pytest test_numerics_v5.py -q)."""
import os, sys, tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numerics_v5 as V  # noqa: E402

ULP = 2.0 ** -23


def _rec(num, den, sk=None, causal=True):
    num = torch.tensor(num, dtype=torch.float64).view(1, -1)
    den = torch.tensor(den, dtype=torch.float64).view(1, -1)
    sq = num.shape[1]
    pdp = torch.full_like(num, 256.0 * 0.5)
    return {"num": num, "den": den, "pdp": pdp, "psum": torch.full_like(num, 256.0), "pdpabs": pdp.abs(),
            "du": torch.full((1, sq), 0.5, dtype=torch.float32), "causal": causal, "sk": sk or sq, "groups": 1}


def _cap(cases):
    return {"version": V.VERSION, "suite": "cases", "route": "default", "library": "x", "library_sha256": "0" * 64,
            "package": "p", "cases": {k: dict(r, desc={"k": k}, outputs={"O": "h"}) for k, r in cases.items()}}


def _run(orig, cand, form="v5"):
    with tempfile.TemporaryDirectory() as d:
        po, pc = os.path.join(d, "o.pt"), os.path.join(d, "c.pt")
        torch.save(_cap(orig), po); torch.save(_cap(cand), pc)

        class A: pass
        a = A(); a.orig, a.cand, a.report, a.worst, a.env, a.form = po, pc, None, 5, None, form
        return V.check(a)


BASE = ([0.0, 1e-7, 3e-7, 2e-6], [1.0, 1.0, 1.0, 1.0])   # rows 0..3: k = 1..4 visible keys (causal)


def test_equal_passes():
    assert _run({"c": _rec(*BASE)}, {"c": _rec(*BASE)}) == 0


def test_better_passes():
    assert _run({"c": _rec(*BASE)}, {"c": _rec([0.0, 0.0, 0.0, 1e-6], BASE[1])}) == 0


def test_within_roundoff_passes():
    num = list(BASE[0]); num[3] += 3.9 * ULP          # row 3: k = 4, allowance 4 * 2^-23
    assert _run({"c": _rec(*BASE)}, {"c": _rec(num, BASE[1])}) == 0


def test_worse_beyond_roundoff_fails():
    num = list(BASE[0]); num[3] += 4.5 * ULP
    assert _run({"c": _rec(*BASE)}, {"c": _rec(num, BASE[1])}) == 1


def test_v4_exempt_degenerate_row_worse_fails():
    # row 0 has one visible key (degenerate, exempt under v2..v4): residual 0 in the original, 1 in the candidate
    num = list(BASE[0]); num[0] = 0.5
    den = list(BASE[1]); den[0] = 0.5
    assert _run({"c": _rec(*BASE)}, {"c": _rec(num, den)}) == 1


def test_v4_ill_conditioned_row_worse_fails():
    # row 3's ORIGINAL residual is above 1e-3 (ill-conditioned, exempt under v4); any rise beyond round-off fails in v5
    o = _rec([0.0, 0.0, 0.0, 2e-3], BASE[1])
    c = _rec([0.0, 0.0, 0.0, 2e-3 + 1e-5], BASE[1])
    assert _run({"c": o}, {"c": c}) == 1


def test_zero_den_nonzero_num_fails():
    num = list(BASE[0]); num[2] = 1e-30
    den = list(BASE[1]); den[2] = 0.0
    o = _rec(*BASE); o["den"][0, 2] = 0.0; o["num"][0, 2] = 0.0
    assert _run({"c": o}, {"c": _rec(num, den)}) == 1


def test_layout_mismatch_fails():
    assert _run({"c": _rec(*BASE)}, {"c": _rec(BASE[0][:3], BASE[1][:3])}) == 1


def test_missing_case_fails():
    assert _run({"c": _rec(*BASE), "d": _rec(*BASE)}, {"c": _rec(*BASE)}) == 1


def test_delta_report():
    o = _rec(*BASE)
    c = _rec(*BASE); c["du"] = c["du"] + 1e-3
    r = V.case_check("c", o, c)
    assert abs(r["delta_cand"]["max_abs"] - 1e-3) < 1e-6 and r["delta_orig"]["max_abs"] == 0.0


def test_reduce_group():
    rows = torch.tensor([[[[1.0, -2.0]], [[-1.0, 0.5]]]]).permute(0, 1, 2, 3)   # [1, nbn 2, wg 1, sq 2]
    rows = torch.cat([rows, torch.zeros_like(rows)], 2)                          # wg 2
    rx = torch.stack([rows.double(), rows.double().abs(), rows.double().abs()])
    g = V.reduce_group(rows, rows.abs(), rx, torch.zeros(1, 1, 2))
    assert g["num"].tolist() == [[0.0, 1.5]] and g["den"].tolist() == [[2.0, 2.5]]


def test_envelope_is_diagnostic_only():
    num = list(BASE[0]); num[3] += 1e-4
    o, c = _rec(*BASE), _rec(num, BASE[1])
    r = V.case_check("c", o, c, env=[_rec(num, BASE[1]), _rec(*BASE)])
    assert r["worse"] == 1 and r["worse_beyond_orig_lse_envelope"] == 0 and r["problems"]


# ---- v5-abs: |sum t|_cand <= |sum t|_orig + c k 2^-23 S, S = sum|P^ dP| + |delta| sum P^  (_rec: 128 + 0.5 * 256 = 256)
S_REC = 256.0


def test_abs_scale():
    assert V.abs_scale(_rec(*BASE)).tolist() == [[S_REC] * 4]


def test_abs_equal_passes():
    assert _run({"c": _rec(*BASE)}, {"c": _rec(*BASE)}, "v5-abs") == 0


def test_abs_within_bound_passes_even_if_ratio_worse():
    # row 3 (k 4): rise 3.9 * 2^-23 * S in |sum t| -> abs passes; the ratio rises far beyond k 2^-23 -> v5 fails
    num = list(BASE[0]); num[3] += 3.9 * ULP * S_REC
    assert _run({"c": _rec(*BASE)}, {"c": _rec(num, BASE[1])}, "v5-abs") == 0
    assert _run({"c": _rec(*BASE)}, {"c": _rec(num, BASE[1])}, "v5") == 1


def test_abs_beyond_bound_fails_and_reports_c():
    num = list(BASE[0]); num[3] += 8.0 * ULP * S_REC
    assert _run({"c": _rec(*BASE)}, {"c": _rec(num, BASE[1])}, "v5-abs") == 1
    r = V.case_check("c", _rec(*BASE), _rec(num, BASE[1]), form="v5-abs")
    assert r["abs_worse_c1"] == 1 and abs(r["abs_c_needed"] - 2.0) < 1e-9


def test_abs_degenerate_row_no_exemption():
    # row 0 (k 1, exempt under v4): a rise of 2 k 2^-23 S fails
    num = list(BASE[0]); num[0] += 2.0 * ULP * S_REC
    assert _run({"c": _rec(*BASE)}, {"c": _rec(num, BASE[1])}, "v5-abs") == 1


def test_abs_zero_scale_rise_is_infinite_c():
    o = _rec(*BASE); o["pdpabs"][0, 0] = 0.0; o["psum"][0, 0] = 0.0
    num = list(BASE[0]); num[0] = 1e-30
    r = V.case_check("c", o, _rec(num, BASE[1]), form="v5-abs")
    assert r["abs_worse_c1"] == 1 and r["abs_c_needed"] == float("inf")


def test_abs_total_c_is_max_over_cases(tmp_path):
    num = list(BASE[0]); num[3] += 0.6 * ULP * S_REC
    po, pc = tmp_path / "o.pt", tmp_path / "c.pt"
    torch.save(_cap({"c": _rec(*BASE), "d": _rec(*BASE)}), po)
    torch.save(_cap({"c": _rec(num, BASE[1]), "d": _rec(num, BASE[1])}), pc)
    class A: pass
    a = A(); a.orig, a.cand, a.worst, a.env, a.form, a.report = str(po), str(pc), 5, None, "v5-abs", str(tmp_path / "r.json")
    assert V.check(a) == 0
    import json
    assert abs(json.load(open(a.report))["totals"]["abs_c_needed"] - 0.15) < 1e-9   # 0.6 / k 4, not summed
