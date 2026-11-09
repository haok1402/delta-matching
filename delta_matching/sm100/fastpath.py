"""Host fast path (kda-sm100-host2): record-and-replay of FA4 host entries.

The first call of a (shape, strides, dtypes, options) key goes through FA4's full host interface (_flash_attn_fwd /
_flash_attn_bwd: config selection, validation, tensor conversion, compile-cache lookup) with the compile caches wrapped
so that every compiled-kernel invocation is RECORDED (compiled fn + argument structure).  A plan is then built that maps
every tensor argument either to one of the caller's input tensors (same storage, recreated as the same view of the
first input on that storage; the aliasing pattern of the inputs is part of the key), or to a fresh allocation whose
provenance was observed while recording: torch.empty(_like) / torch.zeros(_like) (hooked during the recording call;
empty results sentinel-filled) and untouched by host code before the first kernel that uses it.  Later calls with the
same key skip the host interface and just call the recorded compiled functions with the new tensors.  Kernel launches,
kernel arguments' values and launch order are identical, so results are bitwise identical to the slow path.  Any
buffer of other provenance (made by another op, written by host code, never seen by a kernel) disables the fast path
for that key (WHY records the reason).
Disable globally with DM_FASTPATH=0.
"""
import os
import threading

import torch

ENABLED = os.environ.get("DM_FASTPATH", "1") != "0"
_TLS = threading.local()

try:                        # os.environ.get without the Mapping machinery (same lookup + decode; ~10x cheaper)
    _ED, _ENC, _DEC = os.environ._data, os.environ.encodekey, os.environ.decodevalue
except AttributeError:
    _ED = None
_EK = {}


def env(k, d=None):
    """== os.environ.get(k, d), read live on every call (tests / debug hooks change the env at runtime)."""
    if _ED is None:
        return os.environ.get(k, d)
    kb = _EK.get(k)
    if kb is None:
        kb = _EK[k] = _ENC(k)
    v = _ED.get(kb)
    return d if v is None else _DEC(v)


class TLaunch:
    """Cached direct launch of a Triton JITFunction: kernel(grid, *args, **kw) == jitfn[grid](*args, **kw).
    The first call per key goes through JITFunction.run (binding, specialization, compile-cache lookup) and returns the
    CompiledKernel; later calls with the same key (grid, every non-tensor arg by value, every tensor's dtype + 16-byte
    alignment of its pointer = all that Triton specializes on, kwargs, device) call the kernel's launcher directly with
    the same bound-argument order, swapping in the new tensors.  Same compiled kernel, same arguments -> same launch."""

    def __init__(self, jf):
        self.jf = jf
        self.c = {}

    def __call__(self, grid, *args, **kw):
        if not ENABLED:
            return self.jf[grid](*args, **kw)
        dev = torch.cuda.current_device()
        k = (grid, dev, tuple(kw.values()), tuple(kw), tuple((a.dtype, a.data_ptr() % 16) if isinstance(a, torch.Tensor)
                                                          else (type(a), a) for a in args))
        e = self.c.get(k)
        if e is None:
            from triton import knobs
            kernel = self.jf.run(*args, grid=grid, warmup=False, **kw)
            bound = list(self.jf.device_caches[dev][4](*args, **kw)[0].values())
            ok = (kernel is not None and len(bound) >= len(args) and all(b is a for a, b in zip(args, bound))
                  and not self.jf.pre_run_hooks)
            g = tuple(grid) + (1,) * (3 - len(grid))
            self.c[k] = (kernel.run, kernel.function, kernel.packed_metadata, kernel, tuple(bound[len(args):]), g,
                         knobs.runtime) if ok else False
            return
        if e is False:
            return self.jf[grid](*args, **kw)
        run, fn, pm, kernel, tail, g, rt = e
        if getattr(rt.launch_enter_hook, "calls", 1) or getattr(rt.launch_exit_hook, "calls", 1):
            return self.jf[grid](*args, **kw)     # launch hooks installed (profilers): Triton's own path
        stream = torch._C._cuda_getCurrentRawStream(dev)
        run(g[0], g[1], g[2], stream, fn, pm, None, None, None, *args, *tail)


class RecCache:
    """Wraps an FA4 compile cache; while recording, compiled functions fetched from it are wrapped to log calls."""

    def __init__(self, cache):
        self._c = cache

    def __contains__(self, k):
        return k in self._c

    def __getitem__(self, k):
        fn = self._c[k]
        rec = getattr(_TLS, "rec", None)
        if rec is None:
            return fn

        def proxy(*args, **kw):
            rec.log(fn, args, kw)
            return fn(*args, **kw)
        return proxy

    def __setitem__(self, k, v):
        self._c[k] = v

    def get(self, k, default=None):
        return self[k] if k in self else default

    def __getattr__(self, n):
        return getattr(self._c, n)


def _leaves(x, out):
    if isinstance(x, torch.Tensor):
        out.append(x)
    elif isinstance(x, (tuple, list)):
        for y in x:
            _leaves(y, out)
    elif isinstance(x, dict):
        for y in x.values():
            _leaves(y, out)
    return out


def _sptr(t):
    return t.untyped_storage().data_ptr()


_SENT = 0xFF          # recording only: torch.empty results are filled with this byte (NaN for float dtypes)


class _Recorder:
    def __init__(self, input_ptrs):
        self.calls = []
        self.zero = {}            # storage ptr -> bool (all bytes zero when the first kernel touching it was called)
        self.input_ptrs = input_ptrs
        self.kind = {}            # storage ptr -> "e" / "z": allocated by torch.empty(_like) / torch.zeros(_like)
        self.state = {}           # storage ptr -> "e" (still all-sentinel) / "z" (still all-zero) / False, at first use
        self.keep = []            # hooked allocations stay alive while recording: their pointers can't be reused

    def log(self, fn, args, kw):
        for t in _leaves((args, kw), []):
            p = _sptr(t)
            if p in self.input_ptrs or p in self.zero:
                continue
            st = t.untyped_storage()
            b = torch.empty(0, dtype=torch.uint8, device=t.device).set_(st)
            self.zero[p] = bool((b == 0).all().item()) if b.numel() else True
            k = self.kind.get(p)
            if k == "e":
                self.state[p] = "e" if (not b.numel() or bool((b == _SENT).all().item())) else False
            elif k == "z":
                self.state[p] = "z" if self.zero[p] else False
        self.calls.append((fn, args, kw))

    def hooks(self):
        """Patch torch allocation entry points (recording call only) to learn how each kernel buffer was created."""
        orig = {n: getattr(torch, n) for n in ("empty", "zeros", "empty_like", "zeros_like")}

        def mk(n, kind):
            o = orig[n]

            def w(*a, **k):
                t = o(*a, **k)
                if isinstance(t, torch.Tensor) and t.is_cuda and type(t) is torch.Tensor:
                    if kind == "e" and t.numel():
                        t.view(-1).view(torch.uint8).fill_(_SENT) if t.is_contiguous() else None
                    if kind == "z" or t.is_contiguous():
                        self.kind[_sptr(t)] = kind
                        self.keep.append(t)
                return t
            return w
        for n, kind in (("empty", "e"), ("empty_like", "e"), ("zeros", "z"), ("zeros_like", "z")):
            setattr(torch, n, mk(n, kind))
        return orig


def _view_spec(t, base):
    """Spec to recreate t from base (same storage): (dtype, size, stride, byte offset from base's data_ptr)."""
    return (t.dtype, tuple(t.shape), tuple(t.stride()), t.data_ptr() - base.data_ptr())


def _make_view(base, spec):
    dtype, size, stride, boff = spec
    b = base if base.dtype == dtype else base.view(dtype)
    es = b.element_size()
    if boff % es:
        raise RuntimeError("fastpath: misaligned view")
    return torch.as_strided(b, size, stride, b.storage_offset() + boff // es)


class Plan:
    """Replays recorded compiled calls. inputs: name -> tensor (caller's tensors); allocs recreated per call."""

    def __init__(self, rec: _Recorder, inputs: dict, ret):
        in_by_ptr = {}
        for name, t in inputs.items():
            if t is not None:
                in_by_ptr.setdefault(_sptr(t), (name, t))
        allocs = {}            # storage ptr -> [nbytes, device, zero, returned]
        ret_leaves = _leaves(ret, [])
        ret_ptrs = {_sptr(t) for t in ret_leaves}

        def spec_of(t):
            p = _sptr(t)
            if p in in_by_ptr:
                name, base = in_by_ptr[p]
                return ("in", name, _view_spec(t, base))
            if p not in allocs:
                st = t.untyped_storage()
                k = rec.state.get(p)
                # only buffers of known provenance are replayable: created by a hooked torch.empty(_like) /
                # torch.zeros(_like) and untouched by host code until the first kernel that uses them (re-created the
                # same way). Anything else (made by another torch op, written by host code, never seen by a kernel)
                # makes the plan non-replayable (its bytes would depend on the recording call's data).
                zero = {"z": True, "e": False, False: "modified"}.get(k, "unknown")
                allocs[p] = [st.nbytes(), t.device, zero, p in ret_ptrs, st.data_ptr()]
            a = allocs[p]
            return ("alloc", p, (t.dtype, tuple(t.shape), tuple(t.stride()), t.data_ptr() - a[4]))

        def conv(x):
            if isinstance(x, torch.Tensor):
                return spec_of(x)
            if isinstance(x, tuple) and hasattr(x, "_fields"):
                return ("nt", type(x), tuple(conv(y) for y in x))
            if isinstance(x, tuple):
                return ("tuple", tuple(conv(y) for y in x))
            if isinstance(x, list):
                return ("list", [conv(y) for y in x])
            if isinstance(x, dict):
                return ("dict", {k: conv(v) for k, v in x.items()})
            return ("const", x)

        self.calls = [(fn, conv(args), conv(kw)) for fn, args, kw in rec.calls]
        self.ret = conv(ret)
        # safety: every allocation must be zero-initialised (we zero it) or be an output (overwritten by the kernels)
        self.ok = True
        self.why = []
        for p, (nb, dev, zero, returned, _) in allocs.items():
            if zero == "modified":
                self.ok = False      # host code wrote into a fresh buffer before the kernel: not replayable
                self.why.append(("host-modified", nb))
            elif zero == "unknown":
                self.ok = False
                self.why.append(("unknown-provenance", nb, returned))
            elif zero is None and not returned:
                self.ok = False      # a returned-only tensor never passed to a kernel / unknown provenance
                self.why.append(("unknown-not-returned", nb))
            elif zero is False and not returned:
                self.ok = False
                self.why.append(("nonzero-scratch", nb))
        self.allocs = {p: (nb, dev, zero is True) for p, (nb, dev, zero, returned, _) in allocs.items()}
        self.inputs_tpl = {n: (t.dtype, tuple(t.shape), tuple(t.stride())) for n, t in inputs.items() if t is not None}
        self._run = self._codegen()

    def _codegen(self):
        """Specialised replay function: straight-line Python that rebuilds exactly the recorded arguments."""
        C, T = [], []
        env = {"torch": torch, "_mv": _make_view, "C": C, "T": T}
        pre, memo = [], {}
        # allocations: typed directly when one contiguous view covers the whole storage, else a uint8 buffer
        views_by_alloc = {}

        def collect(s):
            k = s[0]
            if k == "alloc":
                views_by_alloc.setdefault(s[1], []).append(s[2])
            elif k in ("tuple", "list"):
                for y in s[1]:
                    collect(y)
            elif k == "nt":
                for y in s[2]:
                    collect(y)
            elif k == "dict":
                for y in s[1].values():
                    collect(y)
        for fn, a, kw in self.calls:
            collect(a), collect(kw)
        collect(self.ret)
        abase = {}
        for n_, (p, (nb, dev, zero)) in enumerate(self.allocs.items()):
            env[f"D{n_}"] = dev
            whole = None
            for (dt, size, stride, boff) in views_by_alloc.get(p, []):
                es = torch.empty(0, dtype=dt).element_size()
                numel = 1
                for x in size:
                    numel *= x
                cstride, acc = [], 1
                for x in reversed(size):
                    cstride.append(acc)
                    acc *= x
                if boff == 0 and numel * es == nb and tuple(reversed(cstride)) == tuple(stride):
                    whole = (dt, size)
                    break
            fn_ = "torch.zeros" if zero else "torch.empty"
            if whole is not None:
                env[f"DT{n_}"] = whole[0]
                pre.append(f"    B{n_} = {fn_}({tuple(whole[1])!r}, dtype=DT{n_}, device=D{n_})")
            else:
                pre.append(f"    B{n_} = {fn_}({nb}, dtype=torch.uint8, device=D{n_})")
            abase[p] = (f"B{n_}", whole)

        def tens(key, expr):
            if key not in memo:
                v = f"v{len(memo)}"
                pre.append(f"    {v} = {expr}")
                memo[key] = v
            return memo[key]

        def gen(s):
            k = s[0]
            if k == "in":
                name, spec = s[1], s[2]
                dt, size, stride, boff = spec
                t_dt, t_size, t_stride = self.inputs_tpl[name]
                base = f"I[{name!r}]"
                if boff == 0 and size == t_size and stride == t_stride:
                    if dt == t_dt:
                        return base
                    if torch.empty(0, dtype=dt).element_size() == torch.empty(0, dtype=t_dt).element_size():
                        C.append(dt)
                        return tens(("in", name, spec), f"{base}.view(C[{len(C) - 1}])")
                C.append(spec)
                return tens(("in", name, spec), f"_mv({base}, C[{len(C) - 1}])")
            if k == "alloc":
                bvar, whole = abase[s[1]]
                dt, size, stride, boff = s[2]
                if whole is not None and boff == 0 and whole == (dt, size):
                    return bvar
                C.append(s[2])
                return tens(("alloc", s[1], s[2]), f"_mv({bvar}, C[{len(C) - 1}])")
            if k == "nt":
                T.append(s[1])
                return f"T[{len(T) - 1}](" + "".join(gen(y) + ", " for y in s[2]) + ")"
            if k == "tuple":
                return "(" + "".join(gen(y) + ", " for y in s[1]) + ")"
            if k == "list":
                return "[" + ", ".join(gen(y) for y in s[1]) + "]"
            if k == "dict":
                return "{" + ", ".join(f"{kk!r}: {gen(v)}" for kk, v in s[1].items()) + "}"
            C.append(s[1])
            return f"C[{len(C) - 1}]"
        body = []
        for i_, (fn, a, kw) in enumerate(self.calls):
            env[f"F{i_}"] = fn
            body.append(f"    F{i_}(*{gen(a)}, **{gen(kw)})")
        ret = gen(self.ret)
        src = "def _run(I):\n" + "\n".join(pre + body) + f"\n    return {ret}\n"
        self.src = src
        exec(compile(src, "<dm_fastpath_plan>", "exec"), env)
        return env["_run"]

    def run(self, inputs: dict):
        return self._run(inputs)


_PLANS = {}
WHY = {}


def key_of(tag, inputs: dict, extra):
    parts = [tag, extra]
    first = {}           # storage ptr -> (index, data_ptr) of the first input on that storage
    for i, (name, t) in enumerate(inputs.items()):   # call sites build the dict in a fixed order; order is in the key
        if t is None:
            parts.append((name, None))
        else:
            dp = t.data_ptr()
            sp = t.untyped_storage().data_ptr()
            f = first.get(sp)
            if f is None:
                f = first[sp] = (i, dp)
            # alias signature: (first input sharing this storage, byte offset from it); plans rebuild aliased inputs
            # as views of that first input, so a call with different aliasing must not hit the same plan
            parts.append((name, t.shape, t.stride(), t.dtype, t.get_device(), dp % 128, f[0], dp - f[1]))
    return tuple(parts)


def call(tag, inputs: dict, extra, slow, pre=None):
    """slow(): runs the normal host path (must fetch compiled fns through RecCache-wrapped caches) and returns ret."""
    if not ENABLED or type(next(iter(inputs.values()))) is not torch.Tensor:    # fake / subclass tensors: normal path
        return slow()
    k = key_of(tag, inputs, extra)
    plan = _PLANS.get(k)
    if plan is not None:
        if plan is False:
            return slow()
        if pre is not None:
            pre()
        return plan.run(inputs)
    if any(isinstance(t, torch._subclasses.fake_tensor.FakeTensor) for t in inputs.values() if t is not None):
        return slow()          # fake tensors (AOT compile): never planned (no plan exists for their keys)
    rec = _Recorder({_sptr(t) for t in inputs.values() if t is not None})
    _TLS.rec = rec
    orig = rec.hooks()
    try:
        ret = slow()
    finally:
        for n, f in orig.items():
            setattr(torch, n, f)
        _TLS.rec = None
    try:
        p = Plan(rec, inputs, ret)
        _PLANS[k] = p if (p.ok and p.calls) else False
        WHY[k] = p.why if p.calls else ["no-calls"]
    except Exception as e:
        _PLANS[k] = False
        WHY[k] = [repr(e)]
    return ret
