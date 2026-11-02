"""Simulate the H1b dS' transpose (per warp, 32 lanes): which source (lane, rdS word, byte) lands in each target
(lane, stmatrix word, byte).  Then look for an 8-shuffle schedule with lane-independent register choices."""
import itertools, collections

def byte_perm(a, b, s):
    src = list(a) + list(b)          # bytes 0..3 of a, 4..7 of b
    return [src[(s >> (4 * i)) & 0x7] for i in range(4)]

rd = {L: [[(L, i, b) for b in range(4)] for i in range(8)] for L in range(32)}
pk = {}
for L in range(32):
    p = []
    for i in range(4):
        p.append(byte_perm(rd[L][2 * i], rd[L][2 * i + 1], 0x5410))
        p.append(byte_perm(rd[L][2 * i], rd[L][2 * i + 1], 0x7632))
    pk[L] = p
T = {}
for L in range(32):
    ip = L >> 2
    srcA = 8 * (ip & 3) + (L & 3); srcB = srcA + 4
    sel = 0x5140 if ip < 4 else 0x7362
    T[L] = [byte_perm(pk[srcA][w], pk[srcB][w], sel) for w in range(8)]
# per target lane: which sources and which source words / bytes
for L in (0, 1, 4, 16, 17):
    print(L, [[(s, i, b) for (s, i, b) in T[L][w]] for w in range(2)])
# needed bytes per (target, source)
need = collections.defaultdict(list)
for L in range(32):
    for w in range(8):
        for pos, (s, i, b) in enumerate(T[L][w]):
            need[(L, s)].append((w, pos, i, b))
cnt = collections.Counter(len(v) for v in need.values())
print("bytes per (target, source):", cnt, "sources per target:", collections.Counter(sum(1 for (t, s) in need if t == L) for L in range(32)))
own = sum(1 for (t, s) in need if t == s)
print("self pairs:", own)
# for a source lane, which own words feed which target
S = 0
print("source 0 sends:", sorted((t, w, pos, i, b) for (t, s), v in need.items() if s == S for (w, pos, i, b) in v)[:20])

# ---- candidate 8-shuffle scheme ----
ok = True
for L in range(32):
    h = 1 if L >= 16 else 0
    ip = L >> 2
    srcA = 8 * (ip & 3) + (L & 3); srcB = srcA + 4
    R = {}
    for ph in range(2):
        for j in range(4):
            src = srcA if (ph ^ h) == 0 else srcB
            S = src; xs = ((S >> 2) & 1) ^ ph       # the provider's choice (its own lane bits)
            provided = byte_perm(rd[S][2 * j], rd[S][2 * j + 1], 0x7654 if xs else 0x3210)
            R[(ph, j)] = provided
    for j in range(4):
        r0, r1 = R[(0, j)], R[(1, j)]
        o0 = byte_perm(r0, r1, 0x1504 if h else 0x5140)
        o1 = byte_perm(r0, r1, 0x3726 if h else 0x7362)
        if o0 != T[L][2 * j] or o1 != T[L][2 * j + 1]:
            ok = False
            print("mismatch", L, j, o0, T[L][2 * j], o1, T[L][2 * j + 1]); break
print("8-shuffle scheme reproduces the transpose:", ok)
