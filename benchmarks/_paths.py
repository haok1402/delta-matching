"""Where JSON results live (Hao, round 20): never in the repo.  Any JSON output path under <repo>/results/ is
redirected to the same relative path under KDA_RESULTS (default ~/workspace/haok1402/kda-results); markdown stays in
results/.  results/progress/*.json (the dashboard's two medians) is the one exception."""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KDA_RESULTS = os.environ.get("KDA_RESULTS", os.path.expanduser("~/workspace/haok1402/kda-results"))


def results_path(p, is_dir=False):
    """Map a JSON output path (or an output directory) under <repo>/results/ to KDA_RESULTS; other paths unchanged."""
    if not p:
        return p
    ap = os.path.abspath(p)
    res = os.path.join(REPO, "results") + os.sep
    if not ap.startswith(res):
        return p
    rel = ap[len(res):]
    if rel.startswith("progress" + os.sep) or (not is_dir and not rel.endswith(".json")):
        return p
    out = os.path.join(KDA_RESULTS, rel)
    os.makedirs(out if is_dir else os.path.dirname(out), exist_ok=True)
    if out != p:
        print(f"[results] {p} -> {out} (JSON results live outside the repo)", flush=True)
    return out
