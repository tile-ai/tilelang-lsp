"""Persistent analyzer worker: `python -m tlsp.analyzer`.

Speaks NDJSON on stdio (see tlsp.protocol). Imports tilelang once and serves
`analyze` requests until `shutdown`, so the ~2.6s import cost is amortized
across edits. stdout is reserved for protocol messages; all logging goes to
stderr.

Also usable as a one-shot CLI for debugging:

    python -m tlsp.analyzer --once path/to/kernel.py --target c \
        --kwargs '{"M": 1024}'
"""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__, protocol


def _serve() -> int:
    from . import pipeline

    # Announce capabilities before any request.
    protocol.write_message(protocol.hello_result(
        tilelang_version=pipeline.tilelang_version(),
        span_supported=pipeline.span_supported(),
        error=pipeline._TILELANG_ERR,
    ))

    while True:
        try:
            request = protocol.read_message()
        except (ValueError, TypeError) as exc:
            protocol.write_message(protocol.err_response(None, "bad-request", str(exc)))
            continue
        if request is None:  # EOF: server went away
            return 0

        method = request.get("method")
        req_id = request.get("id")
        if method == "shutdown":
            return 0
        if method != "analyze":
            protocol.write_message(
                protocol.err_response(req_id, "unknown-method", str(method))
            )
            continue

        params = request.get("params") or {}
        try:
            result = pipeline.analyze(
                path=params["path"],
                kernel=params.get("kernel"),
                kwargs=params.get("kwargs"),
                target_str=params.get("target", "cuda"),
            )
            protocol.write_message(protocol.ok_response(req_id, result))
        except Exception as exc:  # analyze() itself is total; this is paranoia
            protocol.write_message(
                protocol.err_response(req_id, "internal", f"{type(exc).__name__}: {exc}")
            )


def _once(argv) -> int:
    parser = argparse.ArgumentParser(prog="tlsp.analyzer")
    parser.add_argument("--once", metavar="PATH", required=True)
    parser.add_argument("--target", default="cuda")
    parser.add_argument("--kernel", default=None)
    parser.add_argument("--kwargs", default=None,
                        help="JSON object of extra kernel kwargs")
    args = parser.parse_args(argv)

    from . import pipeline

    kwargs = json.loads(args.kwargs) if args.kwargs else None
    result = pipeline.analyze(
        path=args.once, kernel=args.kernel, kwargs=kwargs, target_str=args.target
    )
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


def main() -> None:
    if "--once" in sys.argv:
        raise SystemExit(_once([a for a in sys.argv[1:]]))
    raise SystemExit(_serve())


if __name__ == "__main__":
    main()
