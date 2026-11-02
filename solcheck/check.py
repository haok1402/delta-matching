"""Compare each compiled synthetic kernel's step loop with the real kernel's: per-step opcode classes and total.

    python solcheck/check.py --cfgs cfgs.json --dir builddir [--adjust-out adj.json]
"""
import argparse, collections, json, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmarks"))
import sol4  # noqa: E402

def loop_hist(sass_text):
    funcs = sol4.parse_sass(sass_text)
    (name, body), = funcs.items()
    best = None
    for hi_i, t in sol4.back_edges(body):
        lo_i = next(i for i, b in enumerate(body) if b[0] == t)
        seg = body[lo_i:hi_i + 1]
        if any(o.startswith("QGMMA") for _, _, o, _ in seg) and (best is None or seg[0][0] > best[0][0]):
            best = seg
    h = collections.Counter(o if o.startswith(("QGMMA", "MUFU")) else o.split(".")[0] for _, _, o, _ in best)
    return len(best), h

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfgs", required=True); ap.add_argument("--dir", required=True); ap.add_argument("--adjust-out")
    a = ap.parse_args()
    adj = {}
    prev = json.load(open(a.adjust_out)) if a.adjust_out and os.path.exists(a.adjust_out) else {}
    for c in json.load(open(a.cfgs)):
        n, h = loop_hist(open(os.path.join(a.dir, c["name"] + ".sass")).read())
        print(f"{c['name']}: loop {n} instr (real {c['target_instr']:.0f}, diff {n - c['target_instr']:+.0f})")
        print("   ", ", ".join(f"{o}:{v}" for o, v in h.most_common()))
        adj[c["name"]] = prev.get(c["name"], 0) + int(round(c["target_instr"] - n))
    if a.adjust_out:
        json.dump(adj, open(a.adjust_out, "w"))
main()
