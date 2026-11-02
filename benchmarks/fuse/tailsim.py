"""Greedy-dispatch tail model: CTAs in launch order go to the first free slot; work in units (steps), plus a fixed
per-CTA overhead. Prints the makespan efficiency = total work / (slots * makespan) per kernel and shape."""
import heapq, math, sys
SMS = 132
def sim(works, slots, ovh):
    h = [0.0] * slots; heapq.heapify(h); tot = 0.0
    for w in works:
        t = heapq.heappop(h); heapq.heappush(h, t + w + ovh); tot += w + ovh
    mk = max(h); return tot / (slots * mk), mk
def pass1(B, H, N, ovh=0.9):
    nbn = N // 128; nbm = N // 64
    size_one = N * 768; sw = 1 if 40*2**20 < size_one else 2 ** int(math.log2(40*2**20 // size_one))
    hb = B * H; works = []
    for t in range(nbn * hb):
        sec, l2 = divmod(t, sw * nbn)
        if sec < hb // sw: blk, r = divmod(l2, sw)
        else: blk, r = divmod(l2, hb % sw)
        works.append(nbm - 2 * blk)
    return sim(works, SMS, ovh)
def pass0(B, H, N, ovh=0.5):   # grid (bh, p) with p = npairs-1-blockIdx.y, x fastest; stages of 128 keys
    npairs = N // 256; works = []
    for y in range(npairs):
        p = npairs - 1 - y
        works += [2 * p + 2] * (B * H)
    return sim(works, SMS, ovh)
def pass2(B, H, N, ovh=0.3):   # grid (nbm, bh), mb = nbm-1-x, x fastest; tiles of 128 keys; 2 CTAs / SM
    nbm = N // 64; works = []
    for bh in range(B * H):
        works += [ (nbm - 1 - x) // 2 + 1 for x in range(nbm)]
    return sim(works, 2 * SMS, ovh)
for (B, H, N, calls) in ((2, 16, 8192, 1), (2, 32, 8192, 1), (1, 1, 65536, 16), (1, 1, 65536, 32)):
    r = [f(B, H, N) for f in (pass0, pass1, pass2)]
    print(f"B{B} H{H} N{N} x{calls} calls: pass0 eff {r[0][0]:.4f}  pass1 eff {r[1][0]:.4f}  pass2 eff {r[2][0]:.4f}")
