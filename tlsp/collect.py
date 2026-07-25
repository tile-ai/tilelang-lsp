"""Text rendering for Layout/Fragment objects and buffers, short and full form.

Shared by the span collector and the tracer join so every mode emits the same
label/detail text. Only string shaping lives here; nothing imports tilelang at
module scope so the module also loads in tilelang-less contexts.
"""

from __future__ import annotations

from typing import Any, Optional

from . import protocol

_SHORT_LIMIT = 72


def _collapse(text: str) -> str:
    return " ".join(text.split())


def short_layout(layout: Any) -> str:
    """One-line layout summary for inlay hint labels."""
    if layout is None:
        return ""
    try:
        text = _collapse(str(layout))
    except Exception:
        return type(layout).__name__
    if len(text) > _SHORT_LIMIT:
        text = text[: _SHORT_LIMIT - 1].rstrip() + "…"
    return text


def full_layout_text(buffer: Any, layout: Any) -> str:
    """Multi-line detail for hover: buffer header + full layout dump."""
    lines = []
    if buffer is not None:
        lines.append(f"{getattr(buffer, 'name', '?')}: {buffer_label(buffer)}")
    if layout is not None:
        try:
            lines.append(str(layout))
        except Exception:
            lines.append(type(layout).__name__)
    return "\n".join(lines)


def shape_str(buffer: Any) -> str:
    try:
        shape = list(buffer.shape)
        dims = []
        for d in shape:
            try:
                dims.append(str(int(d)))
            except (TypeError, ValueError):
                dims.append(str(d))
        return "(" + ", ".join(dims) + ")" if len(dims) != 1 else f"({dims[0]},)"
    except Exception:
        return "?"


def buffer_label(buffer: Any) -> str:
    """'⟨(128, 32) float16 @shared.dyn⟩' from a tirx Buffer."""
    try:
        scope = buffer.scope()
    except Exception:
        scope = None
    return protocol.format_buffer_label(
        shape_str(buffer), getattr(buffer, "dtype", None), scope
    )


def alloc_label(buffer: Any, layout: Any = None) -> str:
    label = buffer_label(buffer)
    if layout is not None:
        label += f" ⇒ {short_layout(layout)}"
    return label


def loop_hint_label(annotations: Any) -> Optional[str]:
    """'⟨pipelined, stages=3⟩' / '⟨parallel ⇒ Fragment(...)⟩' from For annotations."""
    if not annotations:
        return None
    items = dict(annotations.items()) if hasattr(annotations, "items") else {}
    parts = []
    if "num_stages" in items:
        stages = items["num_stages"]
        try:
            stages = int(stages)
        except (TypeError, ValueError):
            pass
        parts.append(f"pipelined, stages={stages}")
    if "parallel_loop_layout" in items:
        parts.append(f"parallel ⇒ {short_layout(items['parallel_loop_layout'])}")
    if not parts:
        return None
    return "⟨" + "; ".join(parts) + "⟩"


def loop_detail(annotations: Any) -> str:
    if not annotations:
        return ""
    lines = []
    for key, value in (annotations.items() if hasattr(annotations, "items") else []):
        lines.append(f"{key} = {value}")
    return "\n".join(lines)


def kernel_label(grid_dims: list, threads: Optional[Any]) -> str:
    parts = []
    if grid_dims:
        parts.append("grid (" + ", ".join(str(d) for d in grid_dims) + ")")
    if threads is not None:
        parts.append(f"threads={threads}")
    return "⟨" + ", ".join(parts) + "⟩" if parts else "⟨kernel⟩"
