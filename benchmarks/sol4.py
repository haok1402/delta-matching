"""Four-floor speed-of-light model, version "sol4-v2" (Hao's audit direction 5; round-16 repair of sol4-v1, whose
round-15 artifacts stay as historical evidence and are reproducible from commit b4611ce).

    python benchmarks/sol4.py counts --lib <_C.so> --out counts.json         # emitted-code loop ledger (CPU only)
    python benchmarks/sol4.py model --runs median.json --calib calib.json --counts counts.json --out sol4.json [--md t.md]

Per kernel category and configuration: four floors, their maximum and the controlling floor.
  tensor = tile-granular matmul FLOPs (whole diagonal tiles, as the kernels compute them) / calibrated fp8 GEMM peak
  MUFU   = loop-only lower bound of executed MUFU lane-ops (all MUFU opcodes) / (16 per clock per SM x SMs x clock)
  issue  = loop-only lower bound of executed warp instructions / (4 per clock per SM x SMs x clock)
  bytes  = modeled compulsory HBM bytes / calibrated device-copy bandwidth; the glue ledger follows the route the timed
           call actually took (producer backend from its kernel names, backward head split from delta_matching.split)
The MUFU / issue floors are LOWER BOUNDS built from a control-flow ledger of the emitted step loops (sol4-v1 charged the
static loop interval, i.e. every arm of every in-loop branch, as executed):
  * the loop body is a graph of instructions (fall-through, in-loop branch targets; out-of-loop targets of predicated
    branches fall through; unpredicated out-of-loop branches and EXIT end a path); one iteration's cost is the CHEAPEST
    path from the loop head to its back edge (Dijkstra), so rarely-taken arms (pass-1 promotion / dO-scale rescale) and
    mutually exclusive arms (pass-2 first / continued accumulation) are never summed; predicated instructions still
    issue (counted for issue), predicated MUFU are excluded from the MUFU bound (lane participation unknown);
  * the step loops of a kernel are the INNERMOST loops whose cheapest-path QGMMA FLOPs are a whole multiple of the
    declared per-warpgroup step FLOPs, and every path must carry the same QGMMA FLOPs (else the mapping is rejected);
    the per-step bound is the minimum over those loops (masked / unmasked bodies alike);
  * steps outside the loops (one peeled step per warpgroup per work tile, declared) and all prologue / epilogue /
    producer-warp / scheduler work are NOT counted: the floors are loop-only lower bounds (stated per record).
Triton producers, twins and glue are memory kernels: bytes floor only, tensor / MUFU 0, issue "n/a" (not modeled).
Inputs are bound before any floor is computed: calibration digest recomputed, the runs' calibration id and timed library
hash matched, counts version / library hash matched, the exact sixteen d128m configurations each once.
"""
import argparse, hashlib, heapq, json, math, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

MODEL_VERSION = "sol4-v2"
COUNTS_VERSION = "sol4-counts-v2"
MUFU_PER_CLK_SM = 16
ISSUE_PER_CLK_SM = 4
CATS = ("fwd", "prod_fwd", "prod_bwd", "pass0", "pass1", "pass2", "twin", "glue")

# D128 causal CUDA kernels of the target grid: symbol pattern; unit = (rows, keys) of one visited tile; wgs = consumer
# warpgroups that each run steps_per_unit loop steps of step_elems_per_wg score elements per unit; gemm_flop_per_elem =
# 2 x GEMMs x D; work_tile_axis = which unit side a CTA work tile spans (0: a row block loops over keys, 1: a key block
# loops over rows); peel = steps per warpgroup per work tile issued outside the step loop (fwd: the first n-block,
# mainloop_fwd_sm90_fp8f.hpp, before fwd_step; others: assumed 1, which only lowers the bound).
KERNELS_D128_CAUSAL = {
    "fwd": {"symbol": r"FlashAttnFwdSm90Fq8F.*ILi192EEENS7_ILi128EEESB_EEELi128E.*DynamicPersistent", "unit": (192, 128),
            "wgs": 3, "step_elems_per_wg": 64 * 128, "gemm_flop_per_elem": 2 * 2 * 128, "work_tile_axis": 0, "peel": 1},
            # round 24: BM192 (192 x 128 tiles, three consumer warpgroups) re-accepted under the relative error clause
    "pass1": {"symbol": r"INS1_20FlashAttnBwdSm90Fp8R.*SingleTileBwdLPTScheduler", "unit": (64, 128), "wgs": 2,
              "step_elems_per_wg": 64 * 64, "gemm_flop_per_elem": 4 * 2 * 128, "work_tile_axis": 1, "peel": 1},
    "pass0": {"symbol": r"DeltaV6ILb1ELi4E", "unit": (256, 64), "wgs": 4,   # round 17 C3: P0W4 (4 WGs / 256-row CTA)
              "step_elems_per_wg": 64 * 64, "gemm_flop_per_elem": 2 * 2 * 128, "work_tile_axis": 0, "peel": 1},
    "pass2": {"symbol": r"fq8_dq_pass2_kernel_hd128ILb1E", "unit": (64, 128), "wgs": 1,
              "step_elems_per_wg": 64 * 128, "gemm_flop_per_elem": 1 * 2 * 128, "work_tile_axis": 0, "peel": 1},
}
SPLIT_WS_LIMIT = 8 * 2 ** 30
# sol4c (Hao, round 17; kda-solcheck, origin/kda-sm90-solcheck b5be598, results/solcheck/README.md): the fraction of
# sol4-v2 an IDEAL kernel with exactly the accepted step-loop instruction mix reaches on this H100 (best of five
# schedules; tensor and ALU work couple at ~C_alu + 0.55 C_tensor, loaded clock 1.66-1.80 GHz, IPC <= ~3.5).
# sol4c floor = sol4-v2 floor / fraction; pass 2 and the memory kernels keep their sol4-v2 (bytes) floor.
IDEAL_FRACTION = {"fwd": 0.699, "pass1": 0.620, "pass0": 0.745}      # delta_matching.split default DM_BWD_WS_LIMIT_GB = 8


class ModelError(Exception):
    pass


# ----------------------------------------------------------------------------------------------- emitted-code ledger
_INS = re.compile(r"/\*([0-9a-f]{4,6})\*/\s+(@!?U?P\w+\s+)?([A-Z][A-Za-z0-9_.]*)([^;]*);")


def parse_sass(text):
    """{function: [(addr, predicated, opcode, operands)]} from cuobjdump -sass text."""
    funcs, cur = {}, None
    for line in text.splitlines():
        m = re.search(r"Function : (\S+)", line)
        if m:
            cur = m.group(1); funcs[cur] = []
            continue
        m = _INS.search(line)
        if cur and m:
            pred = m.group(2)
            funcs[cur].append((int(m.group(1), 16), bool(pred) and pred.strip() not in ("@PT",), m.group(3), m.group(4)))
    return funcs


def _target(op, rest):
    m = re.search(r"0x([0-9a-f]+)", rest)
    return int(m.group(1), 16) if m else None


def qgmma_flops(op):
    m = re.match(r"[QH]GMMA\.(\d+)x(\d+)x(\d+)", op)
    if not m:
        return 0
    a, b, c = map(int, m.groups())
    return 2 * a * b * c


def is_mufu(op):
    return op.startswith("MUFU.")


def back_edges(body):
    return [(i, _target(op, rest)) for i, (addr, _, op, rest) in enumerate(body)
            if op.startswith("BRA") and _target(op, rest) is not None and _target(op, rest) < addr]


def cheapest(body, lo_i, hi_i, weight, need_flops=None):
    """Cheapest path cost (sum of weight(instr)) from instruction lo_i to the back edge hi_i, one iteration; with
    need_flops, only over paths whose QGMMA FLOPs total exactly need_flops (the iterations that do the step's tensor
    work, as the tensor floor assumes)."""
    lo, hi = body[lo_i][0], body[hi_i][0]
    index = {body[i][0]: i for i in range(lo_i, hi_i + 1)}

    def succ(i):
        addr, pred, op, rest = body[i]
        if i == hi_i:
            return []
        if op.startswith(("BRX", "JMX", "CALL", "RET", "JMP")):
            raise ModelError(f"unsupported control flow {op} at {addr:#x} inside a step loop")
        if op.startswith("BRA"):
            t = _target(op, rest)
            if t is None:
                raise ModelError(f"branch without a static target at {addr:#x}")
            out = []
            if lo <= t <= hi:
                j = hi_i if t == lo else index.get(t)   # a jump to the head ends the iteration
                if j is None:
                    raise ModelError(f"branch at {addr:#x} targets {t:#x}, not an instruction of the loop")
                out.append(j)
            if pred:
                out.append(i + 1)    # not taken; an unpredicated jump out of the loop ends the path
            return out
        if op == "EXIT":
            return [i + 1] if pred else []
        return [i + 1]

    f0 = qgmma_flops(body[lo_i][2]) if need_flops is not None else 0
    start = (lo_i, f0)
    dist = {start: weight(body[lo_i])}
    pq = [(dist[start], lo_i, f0)]
    while pq:
        d, i, f = heapq.heappop(pq)
        if i == hi_i and (need_flops is None or f == need_flops):
            return d
        if d > dist.get((i, f), math.inf) or i == hi_i:
            continue
        for j in succ(i):
            if j > hi_i:
                continue
            nf = f + qgmma_flops(body[j][2]) if need_flops is not None else 0
            if need_flops is not None and nf > need_flops:
                continue
            nd = d + weight(body[j])
            if nd < dist.get((j, nf), math.inf):
                dist[(j, nf)] = nd; heapq.heappush(pq, (nd, j, nf))
    raise ModelError(f"no path through the loop {lo:#x}-{hi:#x}" + (f" with {need_flops} QGMMA FLOPs" if need_flops else ""))


def loop_ledger(body, step_flops):
    """Every back-edge loop with its static totals and cheapest-path costs; the step loops are the innermost loops whose
    largest per-iteration QGMMA work is a whole number of declared steps; their cost is the cheapest path doing exactly
    that work (mutually exclusive arms are never summed; rarely taken arms are not charged)."""
    idx_of = {a: i for i, (a, *_ ) in enumerate(body)}
    out = []
    for hi_i, t in back_edges(body):
        if t not in idx_of:
            continue
        lo_i = idx_of[t]
        seg = body[lo_i:hi_i + 1]
        rec = {"start": body[lo_i][0], "end": body[hi_i][0],
               "static_instr": len(seg), "static_mufu": sum(is_mufu(o) for _, _, o, _ in seg),
               "static_qgmma": sum(qgmma_flops(o) > 0 for _, _, o, _ in seg)}
        try:
            rec["path_flops_min"] = cheapest(body, lo_i, hi_i, lambda x: qgmma_flops(x[2]))
            rec["path_flops_max"] = fmax = _max_flops(body, lo_i, hi_i)
            if fmax > 0 and fmax % step_flops == 0:
                rec["path_instr"] = cheapest(body, lo_i, hi_i, lambda x: 1, need_flops=fmax)
                rec["path_mufu"] = cheapest(body, lo_i, hi_i, lambda x: 1 if is_mufu(x[2]) and not x[1] else 0,
                                            need_flops=fmax)
                rec["path_instr_any"] = cheapest(body, lo_i, hi_i, lambda x: 1)   # incl. data-dependent skips
        except ModelError as e:
            rec["error"] = str(e)
        out.append(rec)
    for r in out:
        f = r.get("path_flops_max", 0)
        r["step_loop_candidate"] = "error" not in r and "path_instr" in r
        r["steps"] = f // step_flops if r["step_loop_candidate"] else 0
        # a cheaper path with less QGMMA work is a data-dependent skip (pass 2: psi zero-tile sentinel / dominance);
        # the floors charge the full-work path for every visited tile, as the tensor floor does
        r["data_dependent_skip"] = r["step_loop_candidate"] and r["path_flops_min"] < f
    for r in out:
        r["innermost"] = r["step_loop_candidate"] and not any(
            o is not r and o["step_loop_candidate"] and r["start"] <= o["start"] and o["end"] <= r["end"]
            and (o["start"], o["end"]) != (r["start"], r["end"]) for o in out)
    return out


def _max_flops(body, lo_i, hi_i):
    """Largest QGMMA FLOPs on any acyclic path through one iteration (static QGMMA total bounds it; exact via DP over
    instructions in address order, which is a topological order once inner back edges are ignored)."""
    lo, hi = body[lo_i][0], body[hi_i][0]
    index = {body[i][0]: i for i in range(lo_i, hi_i + 1)}
    best = {lo_i: qgmma_flops(body[lo_i][2])}
    for i in range(lo_i, hi_i + 1):
        if i not in best or i == hi_i:
            continue
        addr, pred, op, rest = body[i]
        nxt = []
        if op.startswith("BRA"):
            t = _target(op, rest)
            if t is not None and lo <= t <= hi:
                j = hi_i if t == lo else index.get(t)
                if j is not None and j > i:
                    nxt.append(j)       # inner back edges (spin waits) are not followed
            if pred:
                nxt.append(i + 1)
        elif op == "EXIT":
            nxt = [i + 1] if pred else []
        else:
            nxt = [i + 1]
        for j in nxt:
            if j <= hi_i:
                v = best[i] + qgmma_flops(body[j][2])
                if v > best.get(j, -1):
                    best[j] = v
    if hi_i not in best:
        raise ModelError("no acyclic path through the loop")
    return best[hi_i]


def kernel_ledger(body, k):
    step_flops = k["step_elems_per_wg"] * k["gemm_flop_per_elem"]
    led = loop_ledger(body, step_flops)
    steps = [r for r in led if r.get("innermost")]
    if not steps:
        raise ModelError("no step loop whose QGMMA work matches the declared step on every path")
    per_step_instr = min(r["path_instr"] / r["steps"] for r in steps)
    per_step_mufu = min(r["path_mufu"] / r["steps"] for r in steps)
    return {"loops": led, "step_loops": [(r["start"], r["end"]) for r in steps], "step_flops_per_wg": step_flops,
            "per_step_instr": per_step_instr, "per_step_mufu": per_step_mufu}


def counts(a):
    text = subprocess.run(["cuobjdump", "-sass", a.lib], capture_output=True, text=True, check=True).stdout
    res = counts_from_text(text)
    res.update(library=os.path.realpath(a.lib), library_sha256=_sha(a.lib))
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


def counts_from_text(text, kernels=None):
    funcs = parse_sass(text)
    res = {"version": COUNTS_VERSION, "declared": kernels or KERNELS_D128_CAUSAL, "kernels": {}}
    for cat, k in (kernels or KERNELS_D128_CAUSAL).items():
        names = [n for n in funcs if re.search(k["symbol"], n)]
        if len(names) != 1:
            raise ModelError(f"{cat}: {len(names)} functions match {k['symbol']!r}")
        led = kernel_ledger(funcs[names[0]], k)
        res["kernels"][cat] = dict(led, function=names[0])
        print(f"{cat:6s} step loops {[(hex(s), hex(e)) for s, e in led['step_loops']]}: >= {led['per_step_instr']:.1f} "
              f"instr, >= {led['per_step_mufu']:.1f} MUFU per warp per step")
    return res


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------- input binding (B18)
def calib_digest(calib):
    body = {k: v for k, v in calib.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def _pos(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0


GRID_PARTS = {   # grid -> (its configurations, the part each belongs to)
    "paper": lambda K: (K.PAPER_SET, lambda key: "acceptance"),
    "d128m": lambda K: (K.D128M, lambda key: "target" if key[4] in (4096, 8192) else "guard"),
}


def validate_inputs(runs, calib, cnt, grid="paper"):
    import kernels as K
    if calib.get("version") != 1 or not isinstance(calib.get("sha256"), str):
        raise ModelError("calibration: unsupported version or missing sha256")
    if calib_digest(calib) != calib["sha256"]:
        raise ModelError("calibration: the content does not match its sha256")
    try:
        vals = (calib["gemm_fp8"]["peak_tflops"], calib["copy"]["peak_gbs"], calib["sm_clock_mhz"], calib["sms"])
    except (KeyError, TypeError):
        raise ModelError("calibration: peaks / clock / SM count missing")
    if not all(_pos(v) for v in vals):
        raise ModelError("calibration: peaks, clock and SM count must be finite and positive")
    meta = runs.get("meta") or {}
    if meta.get("calibration_sha256") != calib["sha256"]:
        raise ModelError("the timing runs are bound to another calibration (meta.calibration_sha256)")
    for f in ("job", "host", "device_uuid"):   # same-allocation identity, where both sides record it
        if meta.get(f) is not None and calib.get(f) is not None and str(meta[f]) != str(calib[f]):
            raise ModelError(f"timing and calibration {f} differ ({meta[f]} vs {calib[f]})")
    if cnt.get("version") != COUNTS_VERSION:
        raise ModelError(f"counts version {cnt.get('version')!r} is not {COUNTS_VERSION}")
    if cnt.get("declared") != json.loads(json.dumps(KERNELS_D128_CAUSAL)):
        raise ModelError("counts were made with other kernel declarations")
    if not meta.get("library_sha256") or cnt.get("library_sha256") != meta["library_sha256"]:
        raise ModelError("the counts library is not the timed library")
    recs = runs.get("records") or []
    keys = [(r.get("D"), r.get("B"), r.get("H"), r.get("Hk"), r.get("N"), r.get("attn"), r.get("mode")) for r in recs]
    shapes, _ = GRID_PARTS[grid](K)
    want = sorted((D, B, H, Hk, N, "causal", "matched") for D, B, H, Hk, N in shapes)
    if sorted(keys) != want:
        raise ModelError(f"the runs must hold the {len(want)} {grid} configurations exactly once "
                         f"({len(keys)} records, {len(set(keys))} distinct)")
    for r in recs:
        if not isinstance(r.get("ms"), dict) or not all(_pos(r["ms"].get(c)) or r["ms"].get(c) == 0 for c in CATS):
            raise ModelError(f"malformed category times in {r.get('B'), r.get('H'), r.get('Hk'), r.get('N')}")
    return recs


# ----------------------------------------------------------------------------------------------- the model
def visited_units(N, bm, bn, causal):
    nbm, nbn = N // bm, N // bn
    if not causal:
        return nbm * nbn
    return sum(min(nbn, -(-((mb + 1) * bm) // bn)) for mb in range(nbm))


def heads_per_call(B, H, Hk, N, limit=SPLIT_WS_LIMIT, groups="auto"):
    """delta_matching.split._bwd_heads_per_call (default environment)."""
    if groups in ("off", "0", "1"):
        return H
    if groups == "kv":
        return H // Hk
    if groups == "head":
        return 1
    ws = float(B) * N * N
    for hp in (H, H // Hk, 1):
        if ws * hp <= limit:
            return hp
    return 1


def glue_bytes(D, B, H, Hk, N, producers, hp):
    """Glue ledger of the route actually taken (B19).
    producers 'copy': the (B, N, H, D) -> (B*H, N, D) bf16 q / k / v copies (read + write); 'strided' (H3): none.
    GQA fold (Hk < H): dK / dV per query head read (bf16), per KV head written.
    Backward head split (hp < H, delta_matching.split): per call, contiguous() copies (read + write) of the slices that
    are not already contiguous.  The operands are (B, N, h, D) views of (B, H, N, D) memory (and (B, H, ...) tensors),
    so a slice of one head with B == 1 is contiguous and copies nothing: dO8 / Q8 (fp8) copy unless hp == 1 and B == 1;
    K8 / V8 (one KV head per call, fp8), LSE (fp32), Qt / dOt (fp8) and the K twin (fp8, one KV head per call) copy only
    when B > 1 (K8 / V8 / K twin also need Hk > 1); then the three (B, N, H, D) bf16 outputs are concatenated (read +
    write).  Per-block scales (< 0.1% of these bytes) are not modeled."""
    ND, BHq, BHk = N * D, B * H, B * Hk
    b = 0
    if producers == "copy":
        b += 2 * 2 * (BHq + 2 * BHk) * ND
    elif producers != "strided":
        raise ModelError(f"unknown producer backend {producers!r}")
    if Hk < H:
        b += 2 * 2 * BHq * ND + 2 * 2 * BHk * ND
    if hp < H:
        calls = H // hp
        if not (hp == 1 and B == 1):
            b += 2 * 2 * BHq * ND                              # dO8, Q8 slices
        if B > 1:
            b += 2 * 4 * BHq * N + 2 * 2 * BHq * ND            # LSE; Qt, dOt
            if Hk > 1:
                b += 2 * 2 * B * ND * calls + 2 * B * ND * calls   # K8, V8; K twin: one KV head per call
        b += 3 * 2 * 2 * BHq * ND                              # torch.cat of dq, dk, dv
    return b


def record_route(rec):
    """The route the timed call took: recorded (kernels.py >= round 16) or derived from its kernel names and the split
    rule (round-15 records); a recorded route must agree with the derived one."""
    names = rec.get("kernels") or {}
    if not names:
        raise ModelError("record without kernel names: the producer backend cannot be established")
    prod = "strided" if any("_strided_kernel" in n for n in names) else "copy"
    hp = heads_per_call(rec["B"], rec["H"], rec["Hk"], rec["N"])
    cat_seen = any("CatArrayBatchedCopy" in n for n in names)
    if cat_seen != (hp < rec["H"]):
        raise ModelError(f"split rule says heads per call {hp} of {rec['H']} but the timed call "
                         f"{'ran' if cat_seen else 'did not run'} the output concatenation")
    r = rec.get("route")
    if r:
        if r.get("producers") != prod or r.get("heads_per_call") != hp or r.get("streams", 1) != 1 \
                or r.get("layout", "bnhd") != "bnhd":
            raise ModelError(f"recorded route {r} disagrees with the derived one (producers {prod}, heads {hp}) or is "
                             "outside the model (streams > 1, other layout)")
    return {"producers": prod, "heads_per_call": hp, "recorded": bool(r)}


def floors(rec, calib, cnt):
    """sol4-v2 floors (ms) for one D128 causal matched record."""
    import kernels as K
    peak_f = calib["gemm_fp8"]["peak_tflops"] * 1e12
    peak_b = calib["copy"]["peak_gbs"] * 1e9
    clk = calib["sm_clock_mhz"] * 1e6
    sms = calib["sms"]
    D, B, H, Hk, N = rec["D"], rec["B"], rec["H"], rec["Hk"], rec["N"]
    if D != 128 or rec["attn"] != "causal" or rec["mode"] != "matched":
        raise ModelError("sol4-v2 covers the D128 causal matched configurations")
    route = record_route(rec)
    _, b = K.accounting(D, B, H, Hk, N, True, rec["mode"], "bnhd")
    b = dict(b, glue=glue_bytes(D, B, H, Hk, N, route["producers"], route["heads_per_call"]))
    out = {}
    for c in CATS:
        fl = {"bytes": b[c] / peak_b * 1e3, "tensor": 0.0, "mufu": 0.0, "issue": None}
        if c in KERNELS_D128_CAUSAL:
            k = KERNELS_D128_CAUSAL[c]
            kc = cnt["kernels"].get(c)
            if not kc:
                raise ModelError(f"emitted-code ledger for {c} is required")
            bm, bn = k["unit"]
            units = visited_units(N, bm, bn, True) * B * H
            fl["tensor"] = units * bm * bn * k["gemm_flop_per_elem"] / peak_f * 1e3
            steps_per_unit = bm * bn / k["wgs"] / k["step_elems_per_wg"]
            work_tiles = (N // (bm if k["work_tile_axis"] == 0 else bn)) * B * H
            loop_steps = max(0.0, units * k["wgs"] * steps_per_unit - work_tiles * k["wgs"] * k["peel"])
            warp_steps = loop_steps * 4
            fl["issue"] = warp_steps * kc["per_step_instr"] / (ISSUE_PER_CLK_SM * sms * clk) * 1e3
            fl["mufu"] = warp_steps * kc["per_step_mufu"] * 32 / (MUFU_PER_CLK_SM * sms * clk) * 1e3
        vals = {n: v for n, v in fl.items() if v is not None}
        ctl = max(vals, key=vals.get)
        ms = rec["ms"][c]
        solc = vals[ctl] / IDEAL_FRACTION.get(c, 1.0)
        out[c] = {"floors_ms": fl, "sol_ms": vals[ctl], "controlling": ctl, "measured_ms": ms,
                  "pct_of_floor": 100 * vals[ctl] / ms if ms > 0 else None, "issue_modeled": fl["issue"] is not None,
                  "sol4c_ms": solc, "pct_of_sol4c": 100 * solc / ms if ms > 0 else None}
    return out, route


def model(a):
    runs_raw = open(a.runs, "rb").read(); cnt_raw = open(a.counts, "rb").read()
    runs = json.loads(runs_raw); calib = json.load(open(a.calib)); cnt = json.loads(cnt_raw)
    recs = validate_inputs(runs, calib, cnt, a.grid)
    res = model_records(recs, calib, cnt, a.grid)
    res.update(runs=a.runs, runs_sha256=hashlib.sha256(runs_raw).hexdigest(), counts_sha256=hashlib.sha256(cnt_raw).hexdigest())
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    lines = table(res)
    if a.md:
        open(a.md, "w").write("\n".join(lines))
    print("\n".join(lines))


def model_records(recs, calib, cnt, grid="paper"):
    import kernels as K
    _, part_of = GRID_PARTS[grid](K)
    res = {"version": MODEL_VERSION, "grid": grid, "calibration_sha256": calib["sha256"], "counts_library_sha256": cnt["library_sha256"],
           "bound": "issue / MUFU: loop-only lower bounds (cheapest path per step loop iteration); peeled steps, prologue, "
                    "epilogue, producer warps not counted; memory kernels: bytes only, issue not modeled",
           "configs": []}
    for r in recs:
        fl, route = floors(r, calib, cnt)
        key = (r["D"], r["B"], r["H"], r["Hk"], r["N"])
        c = {"key": list(key), "part": part_of(key), "route": route, "floors": fl}
        if r.get("fa3"):
            c["fa3"] = r["fa3"]; c["fa3_spread_pct"] = r.get("fa3_spread_pct")
        res["configs"].append(c)
    tot = {}
    for part in dict.fromkeys(c["part"] for c in res["configs"]):
        sel = [c for c in res["configs"] if c["part"] == part]
        t = {}
        for c in CATS:
            m = sum(x["floors"][c]["measured_ms"] for x in sel); s = sum(x["floors"][c]["sol_ms"] for x in sel)
            sc = sum(x["floors"][c]["sol4c_ms"] for x in sel)
            fl = {n: (None if any(x["floors"][c]["floors_ms"][n] is None for x in sel)
                      else sum(x["floors"][c]["floors_ms"][n] for x in sel)) for n in ("tensor", "mufu", "issue", "bytes")}
            known = {n: v for n, v in fl.items() if v is not None}
            t[c] = {"measured_ms": m, "sol_ms": s, "waste_ms": m - s, "pct_of_floor": 100 * s / m if m else None,
                    "sol4c_ms": sc, "pct_of_sol4c": 100 * sc / m if m else None,
                    "floors_ms": fl, "controlling": max(known, key=known.get)}
        fa = [x.get("fa3", {}).get("fp8_fwd_kernel_ms") for x in sel]
        tot[part] = {"configs": len(sel), "kernels": t,
                     "fa3_fp8_fwd_kernel_ms": sum(fa) if fa and all(_pos(v) for v in fa) else None}
        if tot[part]["fa3_fp8_fwd_kernel_ms"]:
            tot[part]["fwd_over_fa3_fp8"] = t["fwd"]["measured_ms"] / tot[part]["fa3_fp8_fwd_kernel_ms"]
    res["totals"] = tot
    return res


def table(res):
    lines = []
    f3 = lambda v: "n/a" if v is None else f"{v:.3f}"
    for part in res["totals"]:
        T = res["totals"][part]; t = T["kernels"]
        lines += [f"### {part} ({T['configs']} configurations, model {res['version']}, calibration "
                  f"{res['calibration_sha256'][:12]}; issue / MUFU are loop-only lower bounds)", "",
                  "| kernel | measured ms | tensor | MUFU | issue | bytes | floor (max) | controlling | % of floor | sol4c ms | % of sol4c | waste ms |",
                  "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|"]
        for c in sorted(CATS, key=lambda c: -t[c]["waste_ms"]):
            v = t[c]; fl = v["floors_ms"]
            lines.append(f"| {c} | {v['measured_ms']:.3f} | {f3(fl['tensor'])} | {f3(fl['mufu'])} | {f3(fl['issue'])} | "
                         f"{f3(fl['bytes'])} | {v['sol_ms']:.3f} | {v['controlling']} | {v['pct_of_floor']:.1f}% | {v['sol4c_ms']:.3f} | "
                         f"{v['pct_of_sol4c']:.1f}% | {v['waste_ms']:.3f} |")
        if T.get("fa3_fp8_fwd_kernel_ms"):
            lines += ["", f"FA3 fp8 forward kernel (per-configuration medians, summed): {T['fa3_fp8_fwd_kernel_ms']:.3f} ms; "
                          f"our forward / FA3 fp8 = {T['fwd_over_fa3_fp8']:.4f}"]
        lines.append("")
    return lines


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("counts"); c.add_argument("--lib", required=True); c.add_argument("--out", required=True)
    m = sp.add_parser("model")
    for n in ("runs", "calib", "counts", "out"):
        m.add_argument(f"--{n}", required=True)
    m.add_argument("--md", default=None); m.add_argument("--grid", choices=tuple(GRID_PARTS), default="paper")
    a = ap.parse_args()
    from _paths import results_path
    a.out = results_path(a.out)
    try:
        counts(a) if a.cmd == "counts" else model(a)
    except ModelError as e:
        raise SystemExit(f"sol4: {e}")


if __name__ == "__main__":
    main()
