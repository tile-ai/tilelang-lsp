"""Collect VerifyParallelLoop's aggregated data-race warning from stderr.

The race checker (src/transform/verify_parallel_loop.cc in tilelang) reports
suspected races through TVM's C++ LOG(WARNING) and lets the pipeline continue,
so nothing surfaces in the analysis result by itself. LOG(WARNING) writes to
OS-level fd 2 — bypassing sys.stderr — which is why capture has to dup the fd.

The aggregated block looks like::

    Warning: Data race detected: 2 potential race(s)
      [1] `C_local(0, 0)` is written by multiple threads in loop (local_y, local_x)
      --> /path/to/kernel.py:22:1
      Example:
      <z3 model lines>
      [2] `...` is written by multiple threads in loop (...)
      --> /path/to/kernel.py:30:1
      ...
    If you believe this is a false positive, pass `PassKey...`

Each numbered entry becomes one warning diagnostic, located via the same
`--> file:line:col` marker the error paths use (see tlsp.markers).
"""

from __future__ import annotations

import contextlib
import os
import re
import threading

from . import protocol
from .markers import parse_location_marker

# One race entry header: "  [1] `buf(i, j)` is written by multiple threads in loop (vars)"
_ENTRY_RE = re.compile(r"^\s*\[\d+]\s+(?P<what>`.+`\s+is written by multiple threads in loop\s+.+?)\s*$")
_FOOTER = "If you believe this is a false positive"


@contextlib.contextmanager
def capture_os_stderr():
    """Yield a list receiving the bytes written to fd 2 inside the block.

    A drainer thread keeps the pipe from filling up (z3 models can be long).
    Restoring fd 2 closes the pipe's last write end, so the drainer sees EOF
    and the joined bytes are complete.
    """
    saved = os.dup(2)
    read_fd, write_fd = os.pipe()
    chunks: list[bytes] = []

    def _drain() -> None:
        with os.fdopen(read_fd, "rb") as f:
            chunks.append(f.read())

    os.dup2(write_fd, 2)
    os.close(write_fd)
    drainer = threading.Thread(target=_drain, daemon=True)
    drainer.start()
    try:
        yield chunks
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        drainer.join(timeout=2.0)


def race_diagnostics(stderr_text: str, path: str) -> list[dict]:
    """Turn aggregated race reports into warning diagnostics, one per entry.

    Entries whose marker points at another file (or carries none) land on
    line 1, mirroring the L2 fallback.
    """
    if "is written by multiple threads" not in stderr_text:
        return []
    abs_path = os.path.abspath(path)
    lines = stderr_text.splitlines()
    diags: list[dict] = []
    i = 0
    while i < len(lines):
        entry = _ENTRY_RE.match(lines[i])
        if entry is None:
            i += 1
            continue
        j = i + 1
        while j < len(lines) and _ENTRY_RE.match(lines[j]) is None and _FOOTER not in lines[j]:
            j += 1
        line = 1
        loc = parse_location_marker("\n".join(lines[i:j]))
        if loc is not None:
            file, lineno, _col = loc
            try:
                if os.path.abspath(file) == abs_path:
                    line = lineno
            except Exception:
                pass
        diags.append(
            protocol.make_diagnostic(
                line=line,
                message=f"Data race: {entry.group('what')}"[:600],
                severity=protocol.SEVERITY_WARNING,
            )
        )
        i = j
    return diags
