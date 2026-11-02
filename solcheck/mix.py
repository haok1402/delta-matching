"""Instruction mix of each kernel's sol4 step loop: the opcode histogram along the cheapest full-work path (the
path sol4-v2 charges), per warp per step.

    python solcheck/mix.py --sass lib.sass [--out mix.json]
"""
import argparse, collections, heapq, json, math, os, re, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmarks"))
import sol4  # noqa: E402


def cheapest_path(body, lo_i, hi_i, need_flops):
    lo, hi = body[lo_i][0], body[hi_i][0]
    index = {body[i][0]: i for i in range(lo_i, hi_i + 1)}

    def succ(i):
        addr, pred, op, rest = body[i]
        if i == hi_i:
            return []
        if op.startswith("BRA"):
            t = sol4._target(op, rest)
            out = []
            if lo <= t <= hi:
                j = hi_i if t == lo else index.get(t)
                out.append(j)
            if pred:
                out.append(i + 1)
            return out
        if op == "EXIT":
            return [i + 1] if pred else []
        return [i + 1]

    f0 = sol4.qgmma_flops(body[lo_i][2])
    start = (lo_i, f0)
    dist, prev = {start: 1}, {}
    pq = [(1, lo_i, f0)]
    while pq:
        d, i, f = heapq.heappop(pq)
        if i == hi_i and f == need_flops:
            path, s = [], (i, f)
            while s in prev or s == start:
                path.append(s[0])
                if s == start:
                    break
                s = prev[s]
            return path[::-1]
        if d > dist.get((i, f), math.inf) or i == hi_i:
            continue
        for j in succ(i):
            if j > hi_i:
                continue
            nf = f + sol4.qgmma_flops(body[j][2])
            if nf > need_flops:
                continue
            if d + 1 < dist.get((j, nf), math.inf):
                dist[(j, nf)] = d + 1; prev[(j, nf)] = (i, f); heapq.heappush(pq, (d + 1, j, nf))
    raise RuntimeError("no path")


def classify(op):
    base = op.split(".")[0]
    if base in ("QGMMA", "HGMMA"):
        return op
    if base == "MUFU":
        return op
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sass", required=True); ap.add_argument("--out")
    a = ap.parse_args()
    funcs = sol4.parse_sass(open(a.sass).read())
    res = {}
    for cat, k in sol4.KERNELS_D128_CAUSAL.items():
        name = [n for n in funcs if re.search(k["symbol"], n)][0]
        body = funcs[name]
        led = sol4.kernel_ledger(body, k)
        idx = {b[0]: i for i, b in enumerate(body)}
        per = []
        for (s, e), r in zip(led["step_loops"], [r for r in led["loops"] if r.get("innermost")]):
            p = cheapest_path(body, idx[s], idx[e], r["path_flops_max"])
            h = collections.Counter(classify(body[i][2]) for i in p)
            steps = r["steps"]
            per.append({"loop": [hex(s), hex(e)], "steps": steps, "instr": len(p),
                        "hist_per_step": {o: c / steps for o, c in h.most_common()}})
        res[cat] = per
        print(f"== {cat}")
        for l in per:
            print(f"  loop {l['loop']} steps {l['steps']} instr/step {l['instr']/l['steps']:.0f}")
            print("   ", ", ".join(f"{o}:{c:g}" for o, c in l["hist_per_step"].items()))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
