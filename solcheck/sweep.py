"""Sensitivity sweeps of the synthetic step kernels (dep structure = the real kernels' order):
  s_<cat>_i<d>   : ALU-pipe filler changed by d instructions per warp per step (d in -150 .. +100)
  s_<cat>_m<p>   : p% of the MUFU.EX2 replaced by FFMA (same instruction count)
    python solcheck/sweep.py --cfgs cfgs.json --out sweep.json
"""
import argparse, copy, json


def scale_groups(c, op, delta):
    gs = [g for g in c["groups"] if g.get("alu", {}).get(op, 0) > 0] or [g for g in c["groups"] if g.get("alu")]
    left = delta
    for i, g in enumerate(gs):
        d = left if i == len(gs) - 1 else int(round(delta / len(gs)))
        g["alu"][op] = max(0, g["alu"].get(op, 0) + d); left -= d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfgs", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = []
    for c0 in json.load(open(a.cfgs)):
        if c0["mode"] != "dep":
            continue
        cat = c0["cat"]
        for d in (-150, -100, -50, 50, 100):
            c = copy.deepcopy(c0); c["name"] = f"s_{cat}_i{'m' if d < 0 else 'p'}{abs(d)}"; c["kind"] = "variant"
            c.pop("target_instr", None)
            # remove FMA-pipe and ALU-pipe instructions in proportion (FFMA / FMUL / FADD first, then ALU filler)
            if d < 0:
                for op in ("ALU", "IMAD", "FMUL", "FFMA"):
                    have = sum(g.get("alu", {}).get(op, 0) for g in c["groups"])
                    take = min(have, -d)
                    if take:
                        scale_groups(c, op, -take); d += take
                    if d == 0:
                        break
            else:
                scale_groups(c, "ALU", d)
            out.append(c)
        for p in (25, 50, 100):
            c = copy.deepcopy(c0); c["name"] = f"s_{cat}_m{p}"; c["kind"] = "variant"; c.pop("target_instr", None)
            for g in c["groups"]:
                m = g.get("alu", {}).get("MUFU.EX2", 0)
                k = m * p // 100
                if k:
                    g["alu"]["MUFU.EX2"] = m - k; g["alu"]["FFMA"] = g["alu"].get("FFMA", 0) + k
            out.append(c)
    json.dump(out, open(a.out, "w"), indent=1)
    print(len(out), "sweep kernels")


if __name__ == "__main__":
    main()
