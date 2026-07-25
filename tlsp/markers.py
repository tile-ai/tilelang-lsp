"""Parse the `--> file:line:col` location marker embedded in compiler errors.

span_dev C++ passes append a machine-readable location line to error messages
when the relevant IR node carries a source span (see `src/span_utils.h` in the
tilelang repo). `tilelang.errors.enrich_error` renders it as a clang-style
snippet for the CLI; the LSP only needs the position.

The canonical regex lives in `tilelang.errors._MARKER_RE`; we import it when
tilelang is available and fall back to an identical local copy so this module
stays usable in tilelang-less (static-only) contexts.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

_FALLBACK_PATTERN = (
    # The file part is lazy so the trailing ":line[:col]" is split off from
    # the right: greedy matching would swallow ":line" into the file group,
    # while lazy matching still accommodates paths with spaces
    # ("/path with spaces/") and Windows drive letters ("C:\work\kernel.py")
    # via backtracking. Keep in sync with tilelang.errors._MARKER_RE.
    r"\n\s*-->\s+(?P<file>.+?):(?P<line>\d+)(?::(?P<col>\d+))?\s*$"
)


def _load_marker_re() -> "re.Pattern[str]":
    # re.MULTILINE is required: after enrich_error renders the clang-style
    # snippet, the `-->` header is no longer at end-of-string. The upstream
    # regex is compiled without it, so we always use a local MULTILINE build
    # of the same pattern (it matches both bare and enriched forms).
    try:
        from tilelang.errors import _MARKER_RE as upstream

        if upstream.flags & re.MULTILINE:
            return upstream
        return re.compile(upstream.pattern, upstream.flags | re.MULTILINE)
    except Exception:
        return re.compile(_FALLBACK_PATTERN, re.MULTILINE)


MARKER_RE = _load_marker_re()


def parse_location_marker(message: str) -> Optional[Tuple[str, int, int]]:
    """Extract (file, line, col) from a compiler error message, or None.

    Col defaults to 1 when the marker carries no column (line-level spans).
    Only the marker position is consumed; any rendered snippet after it is
    ignored.
    """
    if not isinstance(message, str):
        return None
    match = MARKER_RE.search(message)
    if match is None:
        return None
    return (match.group("file"), int(match.group("line")), int(match.group("col") or 1))


def strip_marker(message: str) -> str:
    """Return the message with the marker line (and any rendered snippet) removed.

    The first line of a TVM error is the human-readable summary; the marker and
    the clang-style snippet `enrich_error` may have appended are display noise
    for an LSP diagnostic.
    """
    if not isinstance(message, str):
        return str(message)
    # The marker (plus the snippet enrich_error may render after it) is always
    # the tail of the message, so cutting at the match start removes both.
    match = MARKER_RE.search(message)
    if match is not None:
        message = message[: match.start()]
    return message.strip()
