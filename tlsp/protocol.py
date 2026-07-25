"""NDJSON wire protocol between the LSP server and the analyzer subprocess.

One JSON object per line on stdin/stdout. The analyzer speaks first with a
`hello` message describing its capabilities, then answers `analyze` requests
until `shutdown`.

Result schema (version 2), all lines 1-based; the server converts to 0-based
LSP positions::

    {
      "version": 2,
      "meta": {"mode": "full|partial|tracer|static",
               "analysis_ms": 430, "tilelang_version": "...", "warnings": []},
      "hints": [{"line": 22, "kind": "alloc|param|op|loop|kernel",
                 "buffer": "C_local"|null, "label": "...", "detail": "...|null"}],
      "diagnostics": [{"line": 16, "col": 1, "end_line": 16,
                       "end_col": 1048576, "severity": "error|warning",
                       "source": "tilelang|tilelang-static", "message": "..."}],
      "symbols": [{"name": "matmul", "kind": "function", "line": 7}]
    }
"""

from __future__ import annotations

import json
import sys
from typing import Any, BinaryIO, Optional

SCHEMA_VERSION = 2

# Hint kinds produced by analyzers.
HINT_ALLOC = "alloc"
HINT_PARAM = "param"
HINT_OP = "op"
HINT_LOOP = "loop"
HINT_KERNEL = "kernel"
HINT_KINDS = (HINT_ALLOC, HINT_PARAM, HINT_OP, HINT_LOOP, HINT_KERNEL)

# Analysis modes, best to worst.
MODE_FULL = "full"
MODE_PARTIAL = "partial"
MODE_TRACER = "tracer"
MODE_STATIC = "static"

SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"

# Span semantics are line-level: diagnostics cover the whole line.
LINE_END_COL = 1 << 20


def format_shape(shape: Any) -> str:
    """'(128, 32)'-style rendering shared by every hint producer."""
    if shape is None:
        return "?"
    if isinstance(shape, (list, tuple)):
        inner = ", ".join(format_shape(d) for d in shape)
        if len(shape) == 1 and isinstance(shape, tuple):
            inner += ","
        return "(" + inner + ")"
    return str(shape).replace("'", "")


def format_buffer_label(shape: Any, dtype: Any = None, scope: Any = None) -> str:
    """'⟨(128, 32) float16 @shared.dyn⟩' — the canonical buffer summary."""
    parts = []
    if shape is not None and shape != "?":
        parts.append(format_shape(shape))
    if dtype:
        parts.append(str(dtype).replace("'", ""))
    if scope:
        parts.append("@" + str(scope).replace("'", ""))
    return "⟨" + " ".join(parts) + "⟩" if parts else "⟨?⟩"


def make_hint(
    line: int,
    kind: str,
    label: str,
    detail: Optional[str] = None,
    buffer: Optional[str] = None,
) -> dict:
    return {"line": line, "kind": kind, "buffer": buffer, "label": label, "detail": detail}


def make_diagnostic(
    line: int,
    message: str,
    severity: str = SEVERITY_ERROR,
    source: str = "tilelang",
    col: int = 1,
    end_line: Optional[int] = None,
    end_col: int = LINE_END_COL,
) -> dict:
    return {
        "line": line,
        "col": col,
        "end_line": end_line if end_line is not None else line,
        "end_col": end_col,
        "severity": severity,
        "source": source,
        "message": message,
    }


def make_result(
    mode: str,
    hints: Optional[list] = None,
    diagnostics: Optional[list] = None,
    symbols: Optional[list] = None,
    analysis_ms: int = 0,
    tilelang_version: Optional[str] = None,
    warnings: Optional[list] = None,
) -> dict:
    return {
        "version": SCHEMA_VERSION,
        "meta": {
            "mode": mode,
            "analysis_ms": analysis_ms,
            "tilelang_version": tilelang_version,
            "warnings": warnings or [],
        },
        "hints": hints or [],
        "diagnostics": diagnostics or [],
        "symbols": symbols or [],
    }


def hello_result(
    tilelang_version: Optional[str],
    span_supported: bool,
    error: Optional[str] = None,
) -> dict:
    return {
        "method": "hello",
        "result": {
            "analyzer_version": SCHEMA_VERSION,
            "tilelang_version": tilelang_version,
            "span_supported": span_supported,
            "error": error,
        },
    }


def analyze_request(req_id: int, params: dict) -> dict:
    return {"id": req_id, "method": "analyze", "params": params}


def ok_response(req_id: int, result: dict) -> dict:
    return {"id": req_id, "result": result}


def err_response(req_id: Any, kind: str, message: str) -> dict:
    return {"id": req_id, "error": {"kind": kind, "message": message}}


def write_message(obj: dict, stream=None) -> None:
    """Write one NDJSON message and flush."""
    stream = stream if stream is not None else sys.stdout
    line = json.dumps(obj, ensure_ascii=False)
    stream.write(line + "\n")
    stream.flush()


def read_message(stream=None) -> Optional[dict]:
    """Read one NDJSON message; None on EOF, ValueError on bad JSON."""
    stream = stream if stream is not None else sys.stdin
    while True:
        line = stream.readline()
        if line == "":
            return None
        if not line.strip():
            continue
        return json.loads(line)


class BinaryChannel:
    """NDJSON channel over binary streams (subprocess pipes)."""

    def __init__(self, reader: BinaryIO, writer: BinaryIO):
        self._reader = reader
        self._writer = writer

    def write(self, obj: dict) -> None:
        data = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
        self._writer.write(data)
        self._writer.flush()

    def read(self) -> Optional[dict]:
        while True:
            line = self._reader.readline()
            if line == b"":
                return None
            if not line.strip():
                continue
            return json.loads(line.decode("utf-8"))
