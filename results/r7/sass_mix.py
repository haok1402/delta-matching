import collections, re, subprocess, sys
def funcs(lib, pat):
    out = subprocess.run(["cuobjdump", "-sass", lib], capture_output=True, text=True).stdout
    d, f = {}, None
    for l in out.splitlines():
        m = re.search(r"Function : (\S+)", l)
        if m: f = m.group(1) if re.search(pat, m.group(1)) else None; d.setdefault(f, []) if f else None; continue
        if f:
            m = re.search(r"/\*[0-9a-f]{4,6}\*/\s+(@!?U?P\w+\s+)?([A-Z0-9_.]+)", l)
            if m: d[f].append(m.group(2))
    return d
CLS = [("QGMMA", "QGMMA"), ("MUFU.EX2", "MUFU.EX2"), ("MUFU other", "MUFU"), ("FFMA", "FFMA"), ("FMUL", "FMUL"), ("FADD", "FADD"),
       ("FMNMX", "FMNMX"), ("SHFL", "SHFL"), ("F2FP", "F2FP"), ("F2F/I2F", "F2F|I2F"), ("PRMT", "PRMT"), ("FSEL/SEL", "FSEL|^SEL"),
       ("FSETP/ISETP", "SETP"), ("WARPGROUP", "WARPGROUP"), ("BAR", "^BAR"), ("LDS/STS", "^LDS|^STS|^LDSM|^STSM"),
       ("LDG/STG", "^LDG|^STG"), ("LDL/STL", "^LDL|^STL"), ("UTMA/SYNCS", "UTMA|SYNCS"), ("IMAD/IADD/LOP", "^IMAD|^IADD|^LOP|^SHF|^LEA"), ("MOV", "^MOV|^UMOV"), ("BRA", "BRA")]
for lib, pat, tag in [(sys.argv[1], sys.argv[2], "A"), (sys.argv[3], sys.argv[4], "B")]:
    for f, ops in funcs(lib, pat).items():
        c = collections.Counter()
        for o in ops:
            for name, rx in CLS:
                if re.search(rx, o): c[name] += 1; break
            else: c["other"] += 1
        print(tag, f[-60:], len(ops), dict(c))
