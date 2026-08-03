"""LSP server entry point: `python -m tlsp [--target cuda]`."""

import argparse
import logging
import sys


def main() -> None:
    parser = argparse.ArgumentParser(prog="tlsp")
    parser.add_argument(
        "--target",
        default=None,
        help="default compilation target for analysis (cuda, c, llvm, metal, webgpu); overridden by "
        "@tilelang.jit(target=...) or a '# <typecheck> target=...' directive in the source",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args, _unknown = parser.parse_known_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from .server import create_server

    target = args.target or "cuda"
    create_server(target=target).start_io()


if __name__ == "__main__":
    main()
