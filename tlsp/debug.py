"""CUDA-GDB launcher for TileLang kernels: 'python -m tlsp.debug -- kernel.py'.

Before running the user script, this module enables TileLang's
'tl.emit_line_directives' pass config so generated device code maps statements
back to Python source lines. It adds '-G' for full device debug information, or
'-lineinfo' for line-only debugging, while preserving the quoting of existing
compile flags. Caches are disabled by default so cached autotune results cannot
bypass the debug build.

Generated sources are kept under '<debug-dir>/generated' through both
'debug_root_path' (JIT source dumps) and 'cuda.kernels_output_dir' (NVCC's
actual source input). The module imports TileLang lazily so argument parsing
and unit tests do not require a CUDA-capable installation.
"""

from __future__ import annotations

import argparse
import functools
import importlib
import os
import runpy
import shlex
import sys
from pathlib import Path
from typing import Any

_LINE_DIRECTIVE_ATTRIBUTE = "TL_EMIT_LINE_DIRECTIVES"
_CUDA_OUTPUT_DIR_ATTRIBUTE = "CUDA_KERNELS_OUTPUT_DIR"
_DEVICE_DEBUG_FLAGS = ("-G", "--device-debug")
_LINE_INFO_FLAGS = ("-lineinfo", "--generate-line-info")


def _log(message: str) -> None:
    print(f"[tlsp.debug] {message}", file=sys.stderr, flush=True)


def normalize_compile_flags(flags: Any) -> list[str]:
    """Wrap flag groups without changing their quoting for TileLang's backends."""
    if flags is None:
        return []
    if isinstance(flags, str):
        return [flags]
    return [str(flag) for flag in flags]


def _has_compile_flag(flags: Any, candidates: tuple[str, ...]) -> bool:
    for group in normalize_compile_flags(flags):
        try:
            tokens = shlex.split(group)
        except ValueError:
            tokens = group.split()
        if any(token in candidates for token in tokens):
            return True
    return False


def _pass_config_key(attribute: str) -> str | None:
    try:
        from tilelang.transform import PassConfigKey
    except ImportError:
        return None
    return getattr(PassConfigKey, attribute, None)


def line_directive_key() -> str | None:
    """Return the TileLang 0.1.15+ pass-config key for Python line mapping."""
    return _pass_config_key(_LINE_DIRECTIVE_ATTRIBUTE)


def cuda_output_dir_key() -> str | None:
    """Return the pass-config key that preserves NVCC's actual source input."""
    return _pass_config_key(_CUDA_OUTPUT_DIR_ATTRIBUTE)


def merge_debug_options(
    kwargs: dict[str, Any],
    *,
    device_debug: bool,
    generated_dir: str | None,
    line_directives: bool = True,
    allow_debug_root_path: bool = True,
) -> dict[str, Any]:
    """Return debug options without mutating the caller or changing flag quoting."""
    merged = dict(kwargs)
    pass_configs = dict(merged.get("pass_configs") or {})

    key = line_directive_key() if line_directives else None
    if key is not None:
        pass_configs[key] = True

    output_key = cuda_output_dir_key() if generated_dir else None
    if output_key is not None:
        pass_configs[output_key] = generated_dir

    if pass_configs or "pass_configs" in merged:
        merged["pass_configs"] = pass_configs

    flags = normalize_compile_flags(merged.get("compile_flags"))
    if device_debug and not _has_compile_flag(flags, _DEVICE_DEBUG_FLAGS):
        flags.append("-G")
        merged["compile_flags"] = flags
    elif not device_debug and key is not None and not _has_compile_flag(flags, _LINE_INFO_FLAGS):
        flags.append("-lineinfo")
        merged["compile_flags"] = flags

    if allow_debug_root_path and generated_dir:
        merged["debug_root_path"] = generated_dir

    return merged


def install_debug_hooks(*, device_debug: bool, generated_dir: str, line_directives: bool = True) -> None:
    """Patch 'tilelang.jit' and 'tilelang.compile' for this process.

    The user's script runs after these wrappers are installed, covering normal
    decorators and direct compile calls. Grouped autotune and direct JITKernel
    construction bypass these entry points and are not supported by this MVP.
    """
    import tilelang

    jit_module = importlib.import_module("tilelang.jit")
    original_jit = getattr(tilelang, "jit", None)
    if not callable(original_jit):
        raise RuntimeError("tilelang.jit is not callable; this TileLang version is unsupported")

    @functools.wraps(original_jit)
    def debug_jit(*args: Any, **kwargs: Any) -> Any:
        options = merge_debug_options(
            kwargs,
            device_debug=device_debug,
            generated_dir=generated_dir,
            line_directives=line_directives,
        )
        return original_jit(*args, **options)

    original_compile = getattr(tilelang, "compile", None)
    if not callable(original_compile):
        original_compile = getattr(jit_module, "compile", None)

    tilelang.jit = debug_jit
    jit_module.jit = debug_jit

    if callable(original_compile):

        @functools.wraps(original_compile)
        def debug_compile(*args: Any, **kwargs: Any) -> Any:
            options = merge_debug_options(
                kwargs,
                device_debug=device_debug,
                generated_dir=generated_dir,
                line_directives=line_directives,
                allow_debug_root_path=False,
            )
            return original_compile(*args, **options)

        tilelang.compile = debug_compile
        jit_module.compile = debug_compile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tlsp.debug",
        description="Run a TileLang script under CUDA-GDB with Python line mapping and full device debug info.",
    )
    parser.add_argument(
        "--debug-dir",
        default=".tlsp/debug",
        help="directory for debug artifacts (default: .tlsp/debug, relative to the current directory)",
    )
    parser.add_argument(
        "--no-device-debug",
        action="store_true",
        help="do not add -G to the device compile flags (line mapping only, faster compile)",
    )
    parser.add_argument(
        "--no-line-directives",
        action="store_true",
        help="do not enable tl.emit_line_directives (debugger shows generated CUDA instead of Python)",
    )
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="keep TileLang's kernel cache (the default disables it to guarantee a debug build)",
    )
    parser.add_argument("script", help="user script to run (place after -- to pass script arguments)")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="arguments forwarded to the user script")
    return parser


def parse_runner_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_runner_args(argv)

    script = Path(args.script).expanduser()
    if not script.is_file():
        raise SystemExit(f"tlsp.debug: script not found: {script}")
    script = script.resolve()

    debug_dir = Path(args.debug_dir).expanduser().resolve()
    generated_dir = debug_dir / "generated"

    if not args.keep_cache:
        os.environ["TILELANG_DISABLE_CACHE"] = "1"
    if not args.no_line_directives:
        os.environ["TILELANG_ENABLE_IR_SPAN"] = "1"

    key = None if args.no_line_directives else line_directive_key()
    if key is None and not args.no_line_directives:
        raise SystemExit(
            "tlsp.debug: Python line mapping requires TileLang >= 0.1.15 with tl.emit_line_directives; "
            "upgrade TileLang or pass --no-line-directives to debug generated CUDA instead"
        )

    generated_dir.mkdir(parents=True, exist_ok=True)
    install_debug_hooks(
        device_debug=not args.no_device_debug,
        generated_dir=str(generated_dir),
        line_directives=key is not None,
    )

    if not args.no_device_debug:
        device_info = "-G"
    elif key is not None:
        device_info = "-lineinfo"
    else:
        device_info = "off"
    _log(f"device debug info: {device_info}")
    _log(f"Python line mapping: {'on' if key is not None else 'off'}")
    _log(f"generated sources: {generated_dir}")
    _log(f"running {script}")

    # Match 'python script.py': argv[0] is the script and its directory is on
    # sys.path before the current working directory.
    sys.argv = [str(script), *args.args]
    sys.path.insert(0, str(script.parent))
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
