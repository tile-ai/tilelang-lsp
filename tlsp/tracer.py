"""Call-tracer fallback — captures TileLang DSL calls while `get_tir` runs.

Used when IR spans are unavailable (old tilelang, `TILELANG_ENABLE_IR_SPAN=0`,
or zero span coverage) and as a cross-check for the span path. The tracer
monkeypatches the public `tilelang.language` entry points, records the user
frame's line (the DSL's AST rewriting preserves filenames/line numbers), then
delegates to the original function.

Alloc/param records keep the returned Buffer object: the DSL frame binds the
source lvalue name to it *after* the call returns, so names are resolved
lazily at join time.
"""

from __future__ import annotations

import functools
import sys
from dataclasses import dataclass, field
from typing import Any

from . import protocol
from .collect import alloc_label, buffer_label, kernel_label, short_layout

# name in tilelang.language -> record kind
_ALLOC_FUNCS = {
    "alloc_shared",
    "alloc_local",
    "alloc_fragment",
    "alloc_var",
    "alloc_barrier",
    "alloc_tmem",
    "alloc_reducer",
    "alloc_descriptor",
    "alloc_wgmma_desc",
    "alloc_tcgen05_smem_desc",
    "alloc_tcgen05_instruction_desc",
    "alloc_tcgen05_instr_desc",
}
_PARAM_FUNCS = {"Tensor", "empty"}
_OP_FUNCS = {
    "copy",
    "gemm",
    "clear",
    "fill",
    "reduce",
    "atomic_add",
    "atomic_max",
    "atomic_min",
    "cumsum",
    "cummax",
    "print",
    "annotate_layout",
}
_LOOP_FUNCS = {"Pipelined", "Parallel"}
_KERNEL_FUNCS = {"Kernel"}


@dataclass
class TraceRecord:
    kind: str  # "alloc" | "param" | "op" | "loop" | "kernel"
    func_name: str
    line: int
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    result: Any = None  # returned Buffer for alloc/param records


class CallTracer:
    def __init__(self, source_file: str):
        self.source_file = source_file
        self.records: list[TraceRecord] = []
        self._originals: dict[str, Any] = {}
        self._module = None

    # -- install / uninstall ------------------------------------------------

    def install(self) -> None:
        import tilelang.language as lang

        self._module = lang
        for name in _ALLOC_FUNCS | _PARAM_FUNCS | _OP_FUNCS | _LOOP_FUNCS | _KERNEL_FUNCS:
            original = getattr(lang, name, None)
            if not callable(original) or name in self._originals:
                continue
            self._originals[name] = original
            setattr(lang, name, self._wrap(name, original))

    def uninstall(self) -> None:
        if self._module is not None:
            for name, original in self._originals.items():
                setattr(self._module, name, original)
        self._originals = {}
        self._module = None

    def __enter__(self) -> CallTracer:
        self.install()
        return self

    def __exit__(self, *exc) -> None:
        self.uninstall()

    # -- wrapping -------------------------------------------------------------

    def _wrap(self, name: str, original):
        tracer = self

        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            frame = sys._getframe(1)
            line = frame.f_lineno
            filename = frame.f_code.co_filename
            result = original(*args, **kwargs)
            if filename == tracer.source_file:
                kind = tracer._kind_of(name)
                tracer.records.append(
                    TraceRecord(
                        kind=kind,
                        func_name=name,
                        line=line,
                        args=args,
                        kwargs=dict(kwargs),
                        result=result,
                    )
                )
            return result

        return wrapper

    @staticmethod
    def _kind_of(name: str) -> str:
        if name in _ALLOC_FUNCS:
            return "alloc"
        if name in _PARAM_FUNCS:
            return "param"
        if name in _OP_FUNCS:
            return "op"
        if name in _LOOP_FUNCS:
            return "loop"
        return "kernel"


# ---------------------------------------------------------------------------
# records -> hints
# ---------------------------------------------------------------------------


def _record_buffer(record: TraceRecord) -> Any | None:
    buf = record.result
    if buf is not None and hasattr(buf, "dtype") and hasattr(buf, "shape"):
        return buf
    return None


def _norm_dtype(dtype: Any) -> str | None:
    if dtype is None:
        return None
    text = str(dtype)
    return text[2:] if text.startswith("T.") else text


def _shape_of(obj: Any) -> Any:
    """Shape of a Buffer/BufferRegion or a literal tuple/list argument."""
    if obj is None:
        return None
    if isinstance(obj, (tuple, list)):
        return tuple(str(d) if not isinstance(d, (int, float)) else d for d in obj)
    shape = getattr(obj, "shape", None)
    if shape is not None:
        try:
            return tuple(int(d) if str(d).isdigit() else str(d) for d in shape)
        except Exception:
            return str(shape)
    return None


def _buffers_in_args(args: tuple) -> list[Any]:
    found: list[Any] = []

    def walk(obj, depth=0):
        if depth > 4:
            return
        buf = getattr(obj, "buffer", None)
        if buf is not None and hasattr(buf, "name"):
            if not any(buf.same_as(b) for b in found):
                found.append(buf)
            return
        if hasattr(obj, "dtype") and hasattr(obj, "shape") and hasattr(obj, "name"):
            if not any(obj.same_as(b) for b in found):
                found.append(obj)
            return

    for arg in args:
        walk(arg)
    return found


def _shape_concrete(shape: Any) -> bool:
    if not isinstance(shape, (tuple, list)):
        return shape is None
    return all(isinstance(d, (int, float)) for d in shape)


def hints_from_records(
    records: list[TraceRecord],
    layout_by_name: dict[str, Any] | None = None,
) -> list[dict]:
    """Turn trace records into hint dicts (same shape as the span collector)."""
    layout_by_name = layout_by_name or {}
    hints: list[dict] = []
    seen = set()

    def add(line, kind, label, detail=None, buffer=None):
        if not label:
            return
        key = (kind, line, buffer or label)
        if key in seen:
            return
        seen.add(key)
        hints.append(protocol.make_hint(line=line, kind=kind, label=label, detail=detail, buffer=buffer))

    # Alloc/param sites are recorded twice on some code paths (the JIT binder
    # evaluates annotations once symbolically, then again with concrete
    # kwargs). Group by (kind, line) and keep the most concrete record.
    best: dict[tuple, tuple] = {}  # (kind, line) -> (score, index, record)
    for i, rec in enumerate(records):
        if rec.kind not in ("alloc", "param"):
            continue
        buf = _record_buffer(rec)
        shape = None
        if buf is not None:
            shape = _shape_of(buf)
        elif rec.args:
            shape = _shape_of(rec.args[0])
        score = (buf is not None, _shape_concrete(shape), i)
        key = (rec.kind, rec.line)
        if key not in best or score > best[key][0]:
            best[key] = (score, i, rec)

    for rec in records:
        if rec.kind in ("alloc", "param"):
            key = (rec.kind, rec.line)
            if best.get(key, (None, -1, None))[2] is not rec:
                continue
            buf = _record_buffer(rec)
            if buf is not None:
                name = getattr(buf, "name", None) or None
                layout = layout_by_name.get(name)
                add(rec.line, rec.kind, alloc_label(buf, layout), detail=None if layout is None else f"{name}: {layout}", buffer=name)
            else:
                # Non-buffer results (e.g. alloc_var scalars): render from args.
                shape = _shape_of(rec.args[0]) if rec.args else None
                dtype = _norm_dtype(rec.kwargs.get("dtype"))
                if dtype is None and len(rec.args) > 1:
                    dtype = _norm_dtype(rec.args[1])
                add(rec.line, rec.kind, protocol.format_buffer_label(shape, dtype, rec.kwargs.get("scope")))
        elif rec.kind == "op":
            buffers = _buffers_in_args(rec.args)
            parts = []
            details = []
            for buf in buffers:
                layout = layout_by_name.get(buf.name)
                text = buffer_label(buf)
                if layout is not None:
                    text += f" ⇒ {short_layout(layout)}"
                details.append(f"{buf.name}: {text}")
                parts.append(buf.name)
            label = f"⟨{rec.func_name}"
            if parts:
                label += ": " + ", ".join(parts[:3])
            label += "⟩"
            add(rec.line, "op", label, detail="\n".join(details) or None)
        elif rec.kind == "loop":
            if rec.func_name == "Pipelined":
                stages = rec.kwargs.get("num_stages")
                label = "⟨pipelined" + (f", stages={stages}" if stages is not None else "") + "⟩"
            else:  # Parallel
                extents = ", ".join(str(a) for a in rec.args[:4])
                label = f"⟨parallel ({extents})⟩"
            add(rec.line, "loop", label)
        elif rec.kind == "kernel":
            grid = [a for a in rec.args if isinstance(a, (int,))]
            grid = grid or [str(a) for a in rec.args]
            threads = rec.kwargs.get("threads")
            add(rec.line, "kernel", kernel_label(grid, threads))

    hints.sort(key=lambda h: (h["line"], h["kind"]))
    return hints
