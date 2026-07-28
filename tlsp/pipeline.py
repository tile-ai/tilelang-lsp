"""Analysis orchestration: capability detection and the four-tier fallback chain.

    full    get_tir OK + prologue through LayoutInference OK
            -> span collection with inferred layouts
    partial get_tir OK, pass k failed
            -> span collection on the last-good IR state + L2 diagnostic
    tracer  spans unavailable (old tilelang / switch off / zero coverage)
            -> traced call lines; layout_map joined by buffer name when the
               prologue still completes
    static  get_tir failed (import error / NameError / mid-editing)
            -> pure-AST hints + L1 diagnostic

Every stage of the prologue keeps the last successful module so a failing
pass never costs the hints collected so far — that is the "editing-phase
fault tolerance" this analyzer is built for.
"""

from __future__ import annotations

import ast
import os
import sys
import time
import types
from typing import Any

from . import protocol
from .config import find_kernel_defs, resolve_config
from .markers import parse_location_marker, strip_marker
from .static_fallback import analyze_static, index_for_function


def _tilelang():
    """Import tilelang lazily and cache the outcome (success or failure)."""
    global _TLIMPL, _TLIMPL_DONE
    if _TLIMPL_DONE:
        return _TLIMPL
    _TLIMPL_DONE = True
    try:
        import tilelang  # noqa

        import tilelang.ir  # noqa

        _TLIMPL = tilelang
    except Exception as exc:  # pragma: no cover - depends on environment
        _TLIMPL = None
        globals()["_TILELANG_ERR"] = f"{type(exc).__name__}: {exc}"
    return _TLIMPL


_TLIMPL = None
_TLIMPL_DONE = False
_TILELANG_ERR: str | None = None


def tilelang_version() -> str | None:
    tl = _tilelang()
    return getattr(tl, "__version__", "unknown") if tl is not None else None


def span_supported() -> bool:
    tl = _tilelang()
    if tl is None:
        return False
    try:
        import tilelang.ir as tl_ir

        if not hasattr(tl_ir, "get_buffer_span"):
            return False
        try:
            from tilelang import env

            return bool(env.is_span_enable())
        except Exception:
            return True  # very old versions have no switch; spans may exist
    except Exception:
        return False


def load_user_module(path: str):
    """Exec the user file in a fresh module namespace.

    Deliberately bypasses the import loader: its timestamp-based .pyc cache
    only compares second-level mtime and file size, so a quick same-length
    save would silently analyze stale bytecode.
    """
    with open(path, encoding="utf-8") as f:
        source = f.read()
    module = types.ModuleType("_tlsp_user_kernel")
    module.__file__ = path
    # dataclasses/typing look the defining module up in sys.modules.
    sys.modules["_tlsp_user_kernel"] = module
    try:
        exec(compile(source, path, "exec"), module.__dict__)
    finally:
        sys.modules.pop("_tlsp_user_kernel", None)
    return module


def _discover_kernels(module) -> dict[str, Any]:
    from tilelang.jit import JITImpl

    found: dict[str, Any] = {}
    for name in dir(module):
        if name.startswith("__"):
            continue
        obj = getattr(module, name)
        if isinstance(obj, JITImpl):
            found[name] = obj
    return found


def _kernel_to_tir(kernel: Any, kwargs: dict[str, Any]):
    from tilelang import tvm
    from tilelang.jit import JITImpl

    if isinstance(kernel, JITImpl):
        return kernel.get_tir(**kwargs)
    if isinstance(kernel, tvm.tirx.PrimFunc):
        return kernel
    if callable(kernel):
        func = kernel(**kwargs)
        if not isinstance(func, tvm.tirx.PrimFunc):
            raise TypeError(f"factory returned {type(func).__name__}, expected a PrimFunc")
        return func
    raise TypeError(f"don't know how to turn {type(kernel).__name__} into TIR")


def _l1_diagnostic(exc: BaseException, path: str) -> dict:
    """Locate a user-code exception by its innermost user-file frame."""
    abs_path = os.path.abspath(path)
    line = 1
    tb = exc.__traceback__
    last_user_line = None
    while tb is not None:
        frame = tb.tb_frame
        try:
            if os.path.abspath(frame.f_code.co_filename) == abs_path:
                last_user_line = tb.tb_lineno
        except Exception:
            pass
        tb = tb.tb_next
    if last_user_line is not None:
        line = last_user_line
    message = f"{type(exc).__name__}: {strip_marker(str(exc)).splitlines()[0] if str(exc) else ''}"
    if isinstance(exc, TypeError) and "required argument" in message:
        message += " — provide kernel arguments via a `# <typecheck> name=value, ...` comment or a `<kernel>.compile(...)` call"
    return protocol.make_diagnostic(line=line, message=message[:600])


def _l2_diagnostic(exc: BaseException, path: str, stage_name: str) -> dict:
    """Locate a pass failure via the `--> file:line:col` span marker."""
    abs_path = os.path.abspath(path)
    text = str(exc)
    loc = parse_location_marker(text)
    message = strip_marker(text).splitlines()[0] if strip_marker(text) else text
    message = f"[{stage_name}] {type(exc).__name__}: {message}"[:600]
    if loc is not None:
        file, line, _col = loc
        try:
            if os.path.abspath(file) == abs_path:
                return protocol.make_diagnostic(line=line, message=message)
        except Exception:
            pass
    return protocol.make_diagnostic(line=1, message=message)


def _main_func(module: Any):
    funcs = [f for f in module.functions.values() if type(f).__name__ == "PrimFunc" or hasattr(f, "buffer_map")]
    return funcs[0] if funcs else None


def _collect_layout_maps(module: Any) -> dict[str, Any]:
    """buffer name -> layout, merged over all SBlock layout_map annotations."""
    from tilelang import tvm  # noqa: F401
    from tvm import tirx
    from tvm.tirx.stmt_functor import post_order_visit

    merged: dict[str, Any] = {}
    func = _main_func(module)
    if func is None:
        return merged

    def visit(node):
        if isinstance(node, tirx.SBlock) and node.annotations:
            layout_map = node.annotations.get("layout_map")
            if layout_map:
                for buf, layout in layout_map.items():
                    merged.setdefault(getattr(buf, "name", str(buf)), layout)

    post_order_visit(func.body, visit)
    return merged


def _run_prologue(mod: Any, target: Any, stages: list[tuple[str, Any]]):
    """Run stages one by one; return (last_good_mod, layout_done, failure)."""
    last_good = mod
    layout_done = False
    failure: tuple[str, BaseException] | None = None
    with target:
        for name, stage in stages:
            try:
                last_good = stage(last_good)
                if name == "LayoutInference":
                    layout_done = True
            except Exception as exc:
                failure = (name, exc)
                break
    return last_good, layout_done, failure


def analyze(
    path: str,
    kernel: str | None = None,
    kwargs: dict[str, Any] | None = None,
    target_str: str = "cuda",
) -> dict:
    """Analyze one TileLang source file. Never raises."""
    t0 = time.time()
    abs_path = os.path.abspath(path)
    try:
        with open(abs_path, encoding="utf-8") as f:
            source = f.read()
    except OSError as exc:
        return protocol.make_result(
            mode=protocol.MODE_STATIC,
            diagnostics=[protocol.make_diagnostic(1, f"cannot read file: {exc}")],
            analysis_ms=int((time.time() - t0) * 1000),
        )

    static = analyze_static(source)
    cfg = resolve_config(source)
    if kernel is not None:
        cfg.kernel = kernel
    if kwargs:
        cfg.kwargs.update(kwargs)
    warnings = list(cfg.warnings)

    try:
        kernel_defs = find_kernel_defs(ast.parse(source))
    except SyntaxError:
        kernel_defs = {}
    symbols = [{"name": name, "kind": "function", "line": line} for name, line in kernel_defs.items()]

    def finish(mode, hints, diagnostics, extra_warnings=None):
        return protocol.make_result(
            mode=mode,
            hints=hints,
            diagnostics=diagnostics + static.diagnostics,
            symbols=symbols,
            analysis_ms=int((time.time() - t0) * 1000),
            tilelang_version=tilelang_version(),
            warnings=warnings + (extra_warnings or []),
        )

    if cfg.kernel is None:
        # Nothing to elaborate; static hints only, no error (file may simply
        # contain no @tilelang.jit kernel — checked before importing tilelang
        # so plain Python files never see a "tilelang missing" diagnostic).
        return finish(protocol.MODE_STATIC, static.hints, [])

    tl = _tilelang()
    if tl is None:
        diag = protocol.make_diagnostic(
            1,
            f"tilelang is not importable in this environment "
            f"({_TILELANG_ERR}); static hints only. Analysis runs under the "
            f"Python interpreter that hosts the tlsp server — point "
            f"`tilelang.pythonPath` at an environment with tilelang installed.",
        )
        return finish(protocol.MODE_STATIC, static.hints, [diag])

    # -- elaborate: load module + get_tir ------------------------------------
    from .prologue import build_target, prologue_stages
    from .span_collect import SpanCollector
    from .tracer import CallTracer, hints_from_records

    use_spans = span_supported()
    tracer: CallTracer | None = None
    module = None
    func = None
    records = []

    def elaborate(with_tracer: bool):
        nonlocal module, func, tracer, records
        module = load_user_module(abs_path)
        kernels = _discover_kernels(module)
        chosen = cfg.kernel if cfg.kernel in kernels else (next(iter(kernels)) if kernels else None)
        if chosen is None:
            raise LookupError(f"no @tilelang.jit kernel found (wanted {cfg.kernel!r})")
        kernel_obj = kernels[chosen]
        cfg.kernel = chosen
        if with_tracer:
            tracer = CallTracer(abs_path)
            tracer.install()
        try:
            func = _kernel_to_tir(kernel_obj, cfg.kwargs)
        finally:
            if tracer is not None:
                records = list(tracer.records)
                tracer.uninstall()

    try:
        elaborate(with_tracer=not use_spans)
    except Exception as exc:
        return finish(protocol.MODE_STATIC, static.hints, [_l1_diagnostic(exc, abs_path)])

    # Kernel-scoped AST name index for the param-hint fallback (file-global
    # indexes misattribute names shared between kernels).
    scoped_index = index_for_function(source, cfg.kernel) if cfg.kernel else static.index

    # Span coverage self-check: zero buffer coverage -> rerun with the tracer.
    if use_spans:
        try:
            import tilelang.ir as tl_ir

            cov = tl_ir.span_coverage(func)
            if cov["buffers"][1] > 0 and cov["buffers"][0] == 0:
                use_spans = False
                warnings.append("span coverage is 0; fell back to tracer mode")
                try:
                    elaborate(with_tracer=True)
                except Exception as exc:
                    return finish(protocol.MODE_STATIC, static.hints, [_l1_diagnostic(exc, abs_path)])
        except Exception:
            pass

    # -- prologue ------------------------------------------------------------
    try:
        target = build_target(target_str)
    except Exception as exc:
        # Unknown/misconfigured target: still give span-level hints.
        hints = SpanCollector(abs_path, index=scoped_index).collect(func, fresh_func=func) if use_spans else hints_from_records(records)
        diag = protocol.make_diagnostic(1, f"cannot build target {target_str!r}: {exc}")
        return finish(protocol.MODE_PARTIAL, hints, [diag])

    from tilelang import tvm

    mod = tvm.IRModule({func.attrs["global_symbol"]: func})
    try:
        stages, stage_warnings = prologue_stages(target)
    except Exception as exc:
        # Even pass-list construction can fail on a mismatched build:
        # degrade to span-level hints instead of an internal error.
        hints = SpanCollector(abs_path, index=scoped_index).collect(func, fresh_func=func) if use_spans else hints_from_records(records)
        diag = protocol.make_diagnostic(1, f"cannot build prologue passes: {exc}")
        return finish(protocol.MODE_PARTIAL, hints, [diag])
    last_good, layout_done, failure = _run_prologue(mod, target, stages)

    # -- collect ---------------------------------------------------------------
    if use_spans:
        final_func = _main_func(last_good) or func
        hints = SpanCollector(abs_path, index=scoped_index).collect(final_func, fresh_func=func)
    else:
        layout_by_name = _collect_layout_maps(last_good) if layout_done else {}
        hints = hints_from_records(records, layout_by_name)

    if failure is not None:
        stage_name, exc = failure
        diag = _l2_diagnostic(exc, abs_path, stage_name)
        mode = protocol.MODE_PARTIAL if use_spans else protocol.MODE_TRACER
        return finish(mode, hints, [diag], extra_warnings=stage_warnings)

    mode = protocol.MODE_FULL if use_spans else protocol.MODE_TRACER
    return finish(mode, hints, [], extra_warnings=stage_warnings)
