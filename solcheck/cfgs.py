"""Synthetic step-kernel configurations from the measured step-loop mix (solcheck/mix.py output).

    python solcheck/cfgs.py --mix solcheck/mix_r17p0s.json --out solcheck/cfgs.json [--filler-adjust adj.json]

Opcode classes of the real loop -> synthetic instructions:
  explicit: FFMA FMUL FADD FMNMX MUFU.EX2 MUFU.RCP F2FP(cvt e4m3x2) SHFL FSEL LDS STS
  ALU-pipe filler (LOP3): ISETP FSETP LOP3 SHF PRMT SEL PLOP3 LEA IADD3 VOTE F2F DADD and half of the uniform /
      misc ops (U*, R2UR, S2R, S2UR, MOV, LDG, BSSY, BSYNC, NOP, MEMBAR, FENCE, STSM, SYNCS, BAR)
  FMA-pipe filler (IMAD): IMAD VIADD and the other half of the uniform / misc ops
  structural (emitted by the generator itself): QGMMA, WARPGROUP, the loop branch
`adjust` adds / removes ALU filler per kernel so that the compiled loop's instruction count matches the real one.
"""
import argparse, json

EXPLICIT = {"FFMA", "FMUL", "FADD", "FMNMX", "MUFU.EX2", "MUFU.RCP", "SHFL", "FSEL", "LDS", "STS"}
ALU = {"ISETP", "FSETP", "LOP3", "SHF", "PRMT", "SEL", "PLOP3", "LEA", "IADD3", "VOTE", "F2F", "DADD"}
FMAI = {"IMAD", "VIADD"}
STRUCT = {"WARPGROUP", "BRA"}


def classify(h):
    exp, alu, fmai, misc, f2fp = {}, 0, 0, 0, 0
    for op, c in h.items():
        if op.startswith("QGMMA"):
            continue
        if op in EXPLICIT:
            exp[op] = exp.get(op, 0) + c
        elif op == "F2FP":
            f2fp += c
        elif op in ALU:
            alu += c
        elif op in FMAI:
            fmai += c
        elif op in STRUCT:
            continue
        else:
            misc += c
    exp = {k: int(round(v)) for k, v in exp.items()}
    if exp.get("LDS"):            # each synthetic LDS.64 feeds two FFMAs
        exp["FFMA"] = exp.get("FFMA", 0) - 2 * exp["LDS"]
    alu = int(round(alu + misc / 2)); fmai = int(round(fmai + misc / 2))
    return exp, alu, fmai, int(round(f2fp))


def split(d, frac):
    a = {k: int(round(v * frac)) for k, v in d.items()}
    b = {k: v - a[k] for k, v in d.items()}
    return a, b


def build(mix, adjust=None):
    adjust = adjust or {}
    out = []
    for cat in ("fwd", "pass1", "pass0"):
        l = min(mix[cat], key=lambda l: l["instr"] / l["steps"])
        h = l["hist_per_step"]
        exp, alu, fmai, f2fp = classify(h)
        target = l["instr"] / l["steps"]
        for mode in ("dep", "free"):
            alu_m = alu + adjust.get(f"syn_{cat}_{mode}", 0)
            common = {"mode": mode, "target_instr": target, "cat": cat, "free_depth": 0 if cat == "fwd" else 1}
            if cat == "fwd":
                # real BM192 step (SASS): PV_{j-1} = P V (RS, zero-init into the shared accumulator A), wait, O += A
                # (64 FADD), S_j = Q K^T (SS, zero-init into A), wait, softmax -> P (fp8, 16 registers)
                ops = dict(exp, ALU=alu_m, IMAD=fmai)
                ops["FADD"] = ops.get("FADD", 0) - 64
                c = dict(common, name=f"syn_fwd_{mode}", wgs=3, accs={"A": 128}, fregs={"O": 64}, pregs={"P": 16},
                         xregs=16, groups=[
                             {"ops": [[128, "RS", "A", True, "P"]] * 4, "wait_after": 0, "reads": ["A"],
                              "fold": [["O", "A", 64]]},
                             {"ops": [[128, "SS", "A", True, None]] * 4, "wait_after": 0, "reads": ["A"],
                              "alu": ops, "f2fp": f2fp, "f2fp_dst": "P"}])
            elif cat == "pass1":
                mufu = {k: v for k, v in exp.items() if k.startswith("MUFU")}
                rest = {k: v for k, v in exp.items() if not k.startswith("MUFU")}
                r1, r2 = split(dict(rest, ALU=alu_m, IMAD=fmai), 0.5)
                c = dict(common, name=f"syn_pass1_{mode}", wgs=2, accs={"S": 64, "dP": 64, "dV": 128, "dK": 128},
                         pregs={"P": 8, "dS": 8}, xregs=16, groups=[
                             {"ops": [[64, "SS", "S", True, None]] * 4},
                             {"ops": [[64, "SS", "dP", True, None]] * 4, "wait_after": 1, "reads": ["S"],
                              "alu": dict(r1, **mufu), "f2fp": f2fp // 2, "f2fp_dst": "P"},
                             {"wait_after": 0, "reads": ["dP"], "alu": r2, "f2fp": f2fp - f2fp // 2, "f2fp_dst": "dS"},
                             {"ops": [[128, "RS", "dV", False, "P"]] * 2},
                             {"ops": [[128, "RS", "dK", True, "dS"]] * 2, "post_wait": 0}])
            else:
                mufu = {k: v for k, v in exp.items() if k.startswith("MUFU")}
                rest = {k: v for k, v in exp.items() if not k.startswith("MUFU")}
                r1, r2 = split(dict(rest, ALU=alu_m, IMAD=fmai), 0.5)
                c = dict(common, name=f"syn_pass0_{mode}", wgs=4, accs={"S": 64, "dP": 64}, xregs=12, groups=[
                    {"ops": [[64, "SS", "S", True, None]] * 4},
                    {"ops": [[64, "SS", "dP", True, None]] * 4, "wait_after": 1, "reads": ["S"],
                     "alu": dict(r1, **mufu)},
                    {"wait_after": 0, "reads": ["dP"], "alu": r2}])
            out.append(c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", required=True); ap.add_argument("--out", required=True); ap.add_argument("--adjust")
    a = ap.parse_args()
    adj = json.load(open(a.adjust)) if a.adjust else None
    cf = build(json.load(open(a.mix)), adj)
    json.dump(cf, open(a.out, "w"), indent=1)
    for c in cf:
        print(c["name"], [g.get("alu") for g in c["groups"] if g.get("alu")])


if __name__ == "__main__":
    main()
