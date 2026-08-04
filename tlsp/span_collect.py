"""Span-based hint collection — the primary analysis path (tilelang >= 0.1.13).

Span injection is merged upstream (tilelang main, PyPI 0.1.13). Walks a PrimFunc
at any lowering stage and emits hint dicts located purely via IR spans
(`tilelang.ir.get_stmt_span` / `get_buffer_span`). Works on the fresh `get_tir`
output (no layouts) as well as on the post-LayoutInference module (with
`layout_map` annotations), which is what powers the partial mode when a pass
fails mid-prologue.

Only locations inside the current source file are annotated: statements inside
macros carry the macro definition's span, which may live in another file.
"""

from __future__ import annotations

from typing import Any

from . import protocol
from .collect import (
    alloc_label,
    buffer_label,
    full_layout_text,
    kernel_label,
    loop_detail,
    loop_hint_label,
    short_layout,
)
from .static_fallback import StaticIndex

TILE_OP_PREFIX = "tl.tileop."


def _imports():
    from tilelang import tvm  # noqa: F401
    from tvm import tirx
    from tvm.tirx.stmt_functor import post_order_visit
    from tilelang.ir import get_buffer_span, get_stmt_span, span_to_location

    return tirx, post_order_visit, get_buffer_span, get_stmt_span, span_to_location


class SpanCollector:
    def __init__(self, source_file: str, index: StaticIndex | None = None):
        self.source_file = source_file
        self.index = index or StaticIndex()
        self.hints: list[dict] = []
        self.layout_by_name: dict[str, Any] = {}
        self.buffer_by_name: dict[str, Any] = {}
        self._seen: set = set()
        (
            self._tirx,
            self._post_order_visit,
            self._get_buffer_span,
            self._get_stmt_span,
            self._span_to_location,
        ) = _imports()

    # -- location helpers ---------------------------------------------------

    def _stmt_line(self, node: Any) -> int | None:
        loc = self._span_to_location(self._get_stmt_span(node))
        if loc and loc[0] == self.source_file:
            return loc[1]
        return None

    def _buffer_line(self, buffer: Any) -> int | None:
        loc = self._span_to_location(self._get_buffer_span(buffer))
        if loc and loc[0] == self.source_file:
            return loc[1]
        return None

    def _add(self, line: int | None, kind: str, label: str, detail: str | None = None, buffer: str | None = None):
        if line is None or not label:
            return
        key = (kind, line, buffer or label)
        if key in self._seen:
            return
        self._seen.add(key)
        self.hints.append(
            protocol.make_hint(
                line=line,
                kind=kind,
                label=label,
                detail=detail,
                buffer=buffer,
            )
        )

    # -- main entry -----------------------------------------------------------

    def collect(self, func: Any, fresh_func: Any = None) -> list[dict]:
        """Collect hints from `func` (any IR state).

        `fresh_func` (the parse-time PrimFunc) is only used to recover the
        kernel-launch line when the post-pass tree lost its spans.
        """
        tirx = self._tirx

        # Pass 1: harvest layout_maps (buffer name -> layout) for op details.
        def harvest(node):
            if isinstance(node, tirx.SBlock) and node.annotations:
                layout_map = node.annotations.get("layout_map")
                if layout_map:
                    for buf, layout in layout_map.items():
                        self.layout_by_name.setdefault(buf.name, layout)
                        self.buffer_by_name.setdefault(buf.name, buf)

        self._post_order_visit(func.body, harvest)

        # Pass 2: emit hints.
        def visit(node):
            if isinstance(node, tirx.SBlock):
                self._visit_block(node)
            elif isinstance(node, tirx.For):
                self._visit_for(node)
            elif isinstance(node, tirx.Evaluate):
                self._visit_evaluate(node)

        self._post_order_visit(func.body, visit)

        self._collect_params(func)
        self._collect_kernel(func, fresh_func)

        self.hints.sort(key=lambda h: (h["line"], h["kind"]))
        return self.hints

    # -- per-node visitors ----------------------------------------------------

    def _visit_block(self, block: Any):
        # a) layout_map entries -> alloc lines (with inferred layout)
        layout_map = (block.annotations or {}).get("layout_map") if block.annotations else None
        mapped = set()
        if layout_map:
            for buf, layout in layout_map.items():
                mapped.add(buf.name)
                self.buffer_by_name.setdefault(buf.name, buf)
                line = self._buffer_line(buf)
                self._add(line, protocol.HINT_ALLOC, alloc_label(buf, layout), detail=full_layout_text(buf, layout), buffer=buf.name)
        # b) alloc_buffers without a layout entry -> shape-only alloc hints
        for buf in block.alloc_buffers or []:
            if buf.name in mapped:
                continue
            self.buffer_by_name.setdefault(buf.name, buf)
            line = self._buffer_line(buf)
            self._add(line, protocol.HINT_ALLOC, alloc_label(buf), detail=buffer_label(buf), buffer=buf.name)

    def _visit_for(self, node: Any):
        if not node.annotations:
            return
        label = loop_hint_label(node.annotations)
        if not label:
            return
        line = self._stmt_line(node)
        self._add(line, protocol.HINT_LOOP, label, detail=loop_detail(node.annotations))

    def _visit_evaluate(self, node: Any):
        call = getattr(node, "value", None)
        op = getattr(call, "op", None)
        name = getattr(op, "name", "") if op is not None else ""
        if not name.startswith(TILE_OP_PREFIX):
            return
        line = self._stmt_line(node)
        if line is None:
            return
        op_name = name[len(TILE_OP_PREFIX) :]
        buffers = self._call_buffers(call)
        label, detail = self._op_text(op_name, buffers)
        self._add(line, protocol.HINT_OP, label, detail=detail)

    def _call_buffers(self, call: Any) -> list[Any]:
        """Buffers referenced by a tile-op call, in arg order, deduped.

        Tile-op args are themselves Calls (e.g. tl.region(...)) whose children
        are BufferLoad/BufferRegion nodes; tvm Arrays are not list/tuple, so
        child traversal is duck-typed.
        """
        found: list[Any] = []

        def add(buf):
            if buf is not None and hasattr(buf, "name") and not any(buf.same_as(b) for b in found):
                found.append(buf)

        def walk(obj, depth=0):
            if depth > 8 or obj is None or isinstance(obj, (str, int, float, bool)):
                return
            buf = getattr(obj, "buffer", None)
            if buf is not None:
                add(buf)
                return
            for attr in ("args", "indices", "value", "region"):
                sub = getattr(obj, attr, None)
                if sub is None or isinstance(sub, (str, int, float, bool)):
                    continue
                try:
                    iterator = iter(sub)
                except TypeError:
                    walk(sub, depth + 1)
                else:
                    for item in iterator:
                        walk(item, depth + 1)

        walk(call)
        return found

    def _op_text(self, op_name: str, buffers: list[Any]) -> tuple[str, str | None]:
        label = f"⟨{op_name}"
        primary = None
        detail_lines: list[str] = []
        for buf in buffers:
            layout = self.layout_by_name.get(buf.name)
            text = buffer_label(buf)
            if layout is not None:
                text += f" ⇒ {short_layout(layout)}"
            detail_lines.append(f"{buf.name}: {text}")
            if primary is None and layout is not None:
                primary = f"{buf.name} ⇒ {short_layout(layout)}"
        if primary:
            label += f": {primary}"
        elif buffers:
            label += ": " + ", ".join(b.name for b in buffers[:3])
        label += "⟩"
        return label, "\n".join(detail_lines) or None

    # -- params / kernel --------------------------------------------------------

    def _collect_params(self, func: Any):
        for _var, buf in func.buffer_map.items() if func.buffer_map else []:
            self.buffer_by_name.setdefault(buf.name, buf)
            line = self._buffer_line(buf)
            if line is None:
                # Parameter buffer has no span: old span_dev builds had an
                # upstream stamping bug (fixed upstream since).
                # Fall back to the AST name index.
                line = self.index.params.get(buf.name)
            layout = self.layout_by_name.get(buf.name)
            self._add(
                line,
                protocol.HINT_PARAM,
                alloc_label(buf, layout),
                detail=None if layout is None else full_layout_text(buf, layout),
                buffer=buf.name,
            )

    def _collect_kernel(self, func: Any, fresh_func: Any):
        tirx = self._tirx
        grid_dims: list[str] = []
        threads: str | None = None
        line = None

        def launch_chain(root_func):
            """Peel SBlockRealize/SBlock to the thread_extent AttrStmt chain."""
            nonlocal line
            dims: list[tuple[str, str]] = []
            body = root_func.body
            for _ in range(16):
                if body is None:
                    break
                tname = type(body).__name__
                if isinstance(body, tirx.AttrStmt) and body.attr_key == "thread_extent":
                    node = body.node
                    tag = str(getattr(node, "thread_tag", "") or "")
                    if not tag:
                        tag = str(getattr(getattr(node, "var", None), "name", node))
                    dims.append((tag, str(body.value)))
                    if line is None:
                        line = self._stmt_line(body)
                    body = body.body
                    continue
                if tname == "SBlockRealize":
                    body = getattr(body, "block", None)
                    continue
                if tname == "SBlock":
                    body = getattr(body, "body", None)
                    continue
                if tname == "SeqStmt":
                    seq = getattr(body, "seq", None)
                    body = seq[0] if seq is not None and len(seq) else None
                    continue
                break
            return dims

        dims = launch_chain(func)
        if not dims and fresh_func is not None:
            dims = launch_chain(fresh_func)
        for tag, extent in dims:
            if tag.startswith("blockIdx"):
                grid_dims.append(extent)
            elif tag == "threadIdx.x":
                threads = extent

        if line is None:
            # The parse-time root block ("tilelang_root") is stamped with the
            # `with T.Kernel(...)` line by the frame-exit stamper.
            for candidate in (func, fresh_func):
                if candidate is None:
                    continue
                found_line = None

                def find_root_block(node):
                    nonlocal found_line
                    if found_line is not None:
                        return
                    if isinstance(node, tirx.SBlock) and "root" in (node.name_hint or ""):
                        found_line = self._stmt_line(node)

                self._post_order_visit(candidate.body, find_root_block)
                if found_line is not None:
                    line = found_line
                    break

        if line is None and not grid_dims and threads is None:
            return
        label = kernel_label(grid_dims, threads)
        self._add(line, protocol.HINT_KERNEL, label)


def collect_hints(func: Any, source_file: str, fresh_func: Any = None, index: StaticIndex | None = None) -> list[dict]:
    """Module-level convenience wrapper."""
    return SpanCollector(source_file, index=index).collect(func, fresh_func=fresh_func)
