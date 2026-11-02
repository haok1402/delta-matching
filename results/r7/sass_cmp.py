"""Per-function SASS comparison of two libraries: instruction bodies by function; anonymous-namespace hashes in
names (_GLOBAL__N__<hash>_) are normalised (they depend on the build path).  Exit 0 iff every body is identical."""
import collections, hashlib, re, subprocess, sys
def per(lib):
    out = subprocess.run(["cuobjdump", "-sass", lib], capture_output=True, text=True, check=True).stdout
    d, f = collections.defaultdict(list), None
    for l in out.splitlines():
        m = re.search(r"Function : (\S+)", l)
        if m: f = re.sub(r"_GLOBAL__N__[0-9a-f]+_", "_GLOBAL__N__X_", m.group(1)); continue
        if f and re.search(r"/\*[0-9a-f]{4,6}\*/", l): d[f].append(l.strip())
    return {k: hashlib.md5("\n".join(v).encode()).hexdigest() for k, v in d.items()}
a, b = per(sys.argv[1]), per(sys.argv[2])
diff = sorted(f for f in set(a) | set(b) if a.get(f) != b.get(f))
print(f"{len(a)} / {len(b)} functions, {len(diff)} differ" + "".join(f"\n  {f[:140]}" for f in diff))
sys.exit(1 if diff else 0)
