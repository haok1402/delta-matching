"""Markdown tables from check reports: python summarize.py <report dir> [--cases]"""
import json, os, sys
d = sys.argv[1]
for f in ("cases_default", "cases_kv", "cases_head", "cases_streams2", "accept", "stress"):
    p = os.path.join(d, f + ".json")
    if not os.path.isfile(p):
        continue
    r = json.load(open(p))
    t = r["totals"]
    print(f"\n#### {f}: {r['result']} ({t['worse']} of {t['rows']} rows worse than the original)\n")
    print("| case | rows | cand max | cand median | orig max | orig median | worse | worse (v4-exempt) | max excess | "
          "outputs = orig | max abs(du - dr) / (2^-8 sum abs(P dP)) cand / orig | max abs(2^-8 sum P - 1) cand |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k, e in r["cases"].items():
        if "rows" not in e:
            continue
        dc, do = e["delta_cand"], e["delta_orig"]
        print(f"| {k} | {e['rows']} | {e['cand_max']:.3g} | {e['cand_median']:.3g} | {e['orig_max']:.3g} | "
              f"{e['orig_median']:.3g} | {e['worse']} | {e['worse_among_v4_exempt']} | {e['max_excess']:.3g} | "
              f"{e['outputs_equal']} | {dc['max_rel_to_sumabs']:.3g} / {do['max_rel_to_sumabs']:.3g} | {dc['max_norm_err']:.3g} |")
