"""Assumption microbenchmarks for the sol4 floors (generator configs, same format as cfgs.py).

    python solcheck/micro.py --cfgs cfgs.json --out micro.json

  t_*   : tensor only: wgmma rate per shape (SS n64 / n128 / n256, RS n128) and warpgroups per SM (1..4), plus each
          kernel's real wgmma groups without ALU work (dep structure: its waits stay)
  a_*   : pipe rates without wgmma: one op class at a time (MUFU.EX2, FFMA, FMUL, FADD, FMNMX, F2FP, LOP3, IMAD, SHFL,
          LDS.64 + 2 FFMA) and pairs (EX2 + FFMA, EX2 + FMNMX, FFMA + FMNMX) on 4 warpgroups
  x_*   : each kernel's ALU mix alone (its wgmma removed, same warpgroups): issue-limited rate of the mix
  w_*   : tensor + one ALU class: wgmma SS n128 stream with N EX2 / FFMA per wgmma (concurrency of the tensor pipe with
          MUFU / FMA issue)
  b_*   : wait / barrier costs: one SS n64 / n128 wgmma + wait 0 per step (round-trip latency); ping-pong named
          barrier pair per step without work
"""
import argparse, copy, json


def g_tensor(n, kind, count, acc="A", areg=None, zero=True):
    return {"ops": [[n, kind, acc, zero, areg]] * count}


def build(cfgs):
    out = []
    # --- tensor only, single shapes, free (wait depth 1)
    for wgs in (1, 2, 3, 4):
        for n, kind in ((64, "SS"), (128, "SS"), (256, "SS"), (128, "RS")):
            if n == 256 and wgs > 2:
                continue
            c = {"name": f"t_{kind}{n}_w{wgs}", "mode": "free", "wgs": wgs, "accs": {"A": n},
                 "pregs": {"P": 16}, "xregs_free": 1, "free_depth": 0,
                 "groups": [g_tensor(n, kind, 8, "A", "P")], "kind": "tensor", "flop_per_step_wg": 8 * 2 * 64 * n * 32}
            out.append(c)
            if n <= 128:   # two accumulators, wait_group 1: the warpgroup always has a group in flight
                out.append({"name": f"t_{kind}{n}_w{wgs}_dbl", "mode": "dep", "wgs": wgs, "accs": {"A": n, "B": n},
                            "pregs": {"P": 16}, "xregs": 1, "kind": "tensor", "flop_per_step_wg": 8 * 2 * 64 * n * 32,
                            "groups": [dict(g_tensor(n, kind, 4, "A", "P"), wait_after=1),
                                       dict(g_tensor(n, kind, 4, "B", "P"), wait_after=1)]})
    # --- each kernel's wgmma groups alone (dep structure, ALU removed)
    for c0 in cfgs:
        if c0["mode"] != "dep":
            continue
        c = copy.deepcopy(c0); c["name"] = "t_" + c0["name"][4:]; c["kind"] = "tensor_of"
        for g in c["groups"]:
            g.pop("alu", None); g.pop("f2fp", None); g.pop("fold", None)
        out.append(c)
        # its ALU mix alone
        x = copy.deepcopy(c0); x["name"] = "x_" + c0["name"][4:]; x["kind"] = "alu_of"
        for g in x["groups"]:
            g.pop("ops", None)
        out.append(x)
    # --- pipe rates
    single = ["MUFU.EX2", "FFMA", "FMUL", "FADD", "FMNMX", "ALU", "IMAD", "SHFL", "LDS", "FSEL"]
    for op in single:
        out.append({"name": "a_" + op.replace(".", "_"), "mode": "free", "wgs": 4, "accs": {}, "xregs_free": 32,
                    "groups": [{"alu": {op: 256}}], "kind": "alu"})
    out.append({"name": "a_F2FP", "mode": "free", "wgs": 4, "accs": {}, "xregs_free": 32,
                "groups": [{"f2fp": 256}], "kind": "alu"})
    for a, b in (("MUFU.EX2", "FFMA"), ("MUFU.EX2", "FMNMX"), ("FFMA", "FMNMX"), ("FFMA", "ALU"), ("MUFU.EX2", "ALU")):
        for ra in (1, 4, 8):
            out.append({"name": f"a_{a.replace('.', '_')}_{b}_1to{ra}", "mode": "free", "wgs": 4, "accs": {},
                        "xregs_free": 32, "groups": [{"alu": {a: 64, b: 64 * ra}}], "kind": "alu"})
    # --- tensor + ALU class (free): SS n128 x 8 per step + k ALU per step
    for op in ("MUFU.EX2", "FFMA", "FMNMX"):
        for k in (64, 128, 256, 512):
            out.append({"name": f"w_{op.replace('.', '_')}_{k}", "mode": "free", "wgs": 2, "accs": {"A": 128},
                        "xregs_free": 32, "free_depth": 0, "groups": [dict(g_tensor(128, "SS", 8), alu={op: k})],
                        "kind": "mixed", "flop_per_step_wg": 8 * 2 * 64 * 128 * 32})
    # --- latency of one wgmma round trip / barriers
    for n in (64, 128):
        for wgs in (1, 2):
            out.append({"name": f"b_wait_SS{n}_w{wgs}", "mode": "dep", "wgs": wgs, "accs": {"A": n}, "xregs": 1,
                        "groups": [dict(g_tensor(n, "SS", 1), wait_after=0, reads=["A"])], "kind": "latency",
                        "flop_per_step_wg": 2 * 64 * n * 32})
            out.append({"name": f"b_wait4_SS{n}_w{wgs}", "mode": "dep", "wgs": wgs, "accs": {"A": n}, "xregs": 1,
                        "groups": [dict(g_tensor(n, "SS", 4), wait_after=0, reads=["A"])], "kind": "latency",
                        "flop_per_step_wg": 4 * 2 * 64 * n * 32})
    for wgs in (2, 3):
        out.append({"name": f"b_pingpong_w{wgs}", "mode": "dep", "wgs": wgs, "accs": {}, "xregs": 1,
                    "groups": [{"pp_sync": True, "pp_arrive": True, "alu": {"FFMA": 1}}], "kind": "latency"})
    # --- schedule variants of the synthetic step kernels (same per-step work), upper-bound probes
    byname = {c["name"]: c for c in cfgs}
    for cat, W in (("fwd", 3), ("pass1", 2), ("pass0", 4)):
        dep, free = byname[f"syn_{cat}_dep"], byname[f"syn_{cat}_free"]
        # v_*_ilv: free mode, each group's ALU work interleaved between its wgmma issues
        v = copy.deepcopy(free); v["name"] = f"v_{cat}_ilv"; v["kind"] = "variant"; v["equiv"] = [dep["name"], 1]
        for g in v["groups"]:
            if g.get("ops"):
                g["interleave"] = True
        if cat == "fwd":   # put the softmax ALU work under both wgmma groups
            g0, g1 = v["groups"]
            half = {k: n // 2 for k, n in g1["alu"].items()}
            g0["alu"] = dict(g0.get("alu", {}), **half); g0["alu"]["FADD"] = g0["alu"].get("FADD", 0) + 64
            g1["alu"] = {k: n - half[k] for k, n in g1["alu"].items()}
        out.append(v)
        # v_*_spec: warp specialization: warpgroup 0 issues all W warpgroups' wgmma (alternating waits), warpgroups
        # 1..W run the ALU mix on private registers (no dependence): hardware concurrency of the two streams
        ops = [op for g in dep["groups"] for op in g.get("ops", [])]
        alu, f2fp = {}, 0
        for g in dep["groups"]:
            for k, n in g.get("alu", {}).items():
                alu[k] = alu.get(k, 0) + n
            f2fp += g.get("f2fp", 0)
            for fd, fs, fn in g.get("fold", []):
                alu["FADD"] = alu.get("FADD", 0) + fn
        tg = []
        for w in range(W):
            tg.append({"ops": [[n, kind, "T", True, "P"] for n, kind, _, _, _ in ops], "wait_after": 1, "only": [0]})
        nmax = max(n for n, *_ in ops)
        v = {"name": f"v_{cat}_spec", "mode": "dep", "wgs": W + 1, "accs": {"T": nmax}, "pregs": {"P": 16},
             "xregs": 24, "kind": "variant", "equiv": [dep["name"], W],
             "groups": tg + [{"alu": alu, "f2fp": f2fp, "f2fp_dst": "Q", "only": list(range(1, W + 1))}]}
        v["pregs"]["Q"] = 16
        if cat == "fwd":   # 4 warpgroups -> 128 registers; RS here makes ptxas serialize (C7520): PV as SS
            v["xregs"] = 8   # (SS and RS n128 run at the same 8192 FLOP / clk, t_SS128 vs t_RS128)
            for g in v["groups"][:-1]:
                g["ops"] = [[n, "SS", acc, z, ar] for n, k, acc, z, ar in g["ops"]]
        out.append(v)
    # forward: FA3-style ping-pong of the three warpgroups' GEMM issue (named barriers, as the real BM192 kernel)
    v = copy.deepcopy(byname["syn_fwd_dep"]); v["name"] = "v_fwd_pp"; v["kind"] = "variant"; v["equiv"] = ["syn_fwd_dep", 1]
    v["groups"][0]["pp_sync"] = True; v["groups"][1]["pp_arrive"] = True
    out.append(v)
    # --- the same with all-zero operand data (tensor-core power / clock depends on operand toggling)
    for c0 in list(cfgs) + [c for c in out if c["name"] in ("t_SS128_w2", "t_SS64_w4", "t_RS128_w3", "t_fwd_dep",
                                                              "t_pass1_dep", "t_pass0_dep")]:
        if c0["mode"] != "dep" and not c0["name"].startswith("t_"):
            continue
        z = copy.deepcopy(c0); z["name"] = c0["name"] + "_z"; z["rand_smem"] = 0
        out.append(z)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfgs", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    m = build(json.load(open(a.cfgs)))
    json.dump(m, open(a.out, "w"), indent=1)
    print(len(m), "microbenchmarks")


if __name__ == "__main__":
    main()
