"""Resolve which kernel to analyze, with which kwargs and target, from source text only.

Kwargs priority (per key, highest first):

1. ``# <typecheck> M=1024, N=1024, kernel=matmul`` comment directives
2. kwargs harvested from ``<kernel>.compile(...)`` / direct ``<kernel>(...)`` calls
3. function signature defaults

Analysis target priority: a ``# <typecheck> target=c`` directive beats
``target=`` at a ``<kernel>.compile(...)`` call site, which beats a literal
``@tilelang.jit(target=...)`` decorator; all are optional and fall back to
the server's default target.

``T.const("M, N")`` symbols may stay unbound; the analyzer reports a precise
diagnostic when required params are missing.
"""

from __future__ import annotations

import ast
import io
import tokenize
from dataclasses import dataclass, field
from typing import Any

DIRECTIVE_MARK = "<typecheck>"


@dataclass
class Directive:
    lineno: int
    kernel: str | None
    kwargs: dict[str, Any]
    target: str | None = None


@dataclass
class KernelConfig:
    kernel: str | None
    kwargs: dict[str, Any] = field(default_factory=dict)
    origin: str = "none"  # "directive" | "call" | "defaults" | "none"
    warnings: list[str] = field(default_factory=list)
    target: str | None = None  # resolved analysis target (directive > call site > decorator); None = server default


def _literal(node: ast.AST) -> tuple[bool, Any]:
    """Best-effort constant extraction.

    Returns (ok, value). ``T.float16``/bare names degrade to their dotted
    string form ("float16" / "name"); anything else is rejected.
    """
    try:
        return True, ast.literal_eval(node)
    except (ValueError, SyntaxError):
        pass
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        # T.float16 -> "float16" (get_tir accepts dtype names as strings)
        return True, node.attr
    if isinstance(node, ast.Name):
        return True, node.id
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        ok, val = _literal(node.operand)
        if ok and isinstance(val, (int, float)):
            return True, -val if isinstance(node.op, ast.USub) else val
    return False, None


def _parse_kwarg_text(text: str) -> tuple[dict[str, Any], str | None, str | None, list[str]]:
    """Parse ``M=1024, kernel=matmul, target=c`` style text.

    Returns (kwargs, kernel, target, warnings): ``kernel``/``target`` steer the
    analysis itself, so they are pulled out instead of forwarded to the kernel.
    """
    kwargs: dict[str, Any] = {}
    kernel: str | None = None
    target: str | None = None
    warnings: list[str] = []
    text = text.strip()
    if not text:
        return kwargs, kernel, target, warnings
    try:
        call = ast.parse(f"__f({text})", mode="eval").body
    except SyntaxError:
        warnings.append(f"cannot parse <typecheck> directive: {text!r}")
        return kwargs, kernel, target, warnings
    if not isinstance(call, ast.Call):
        return kwargs, kernel, target, warnings
    for kw in call.keywords:
        if kw.arg is None:
            continue
        ok, val = _literal(kw.value)
        if not ok:
            warnings.append(f"skipped non-literal value for {kw.arg!r} in directive")
            continue
        if kw.arg == "kernel":
            kernel = str(val)
        elif kw.arg == "target":
            if isinstance(val, str):
                target = val
            else:
                warnings.append(f"ignored non-string target {val!r} in directive")
        else:
            kwargs[kw.arg] = val
    return kwargs, kernel, target, warnings


def parse_directives(source: str) -> tuple[list[Directive], list[str]]:
    """Extract all ``# <typecheck> ...`` comment directives (tokenize-accurate)."""
    directives: list[Directive] = []
    warnings: list[str] = []
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for tok in tokens:
            if tok.type != tokenize.COMMENT:
                continue
            text = tok.string.lstrip("#").strip()
            if not text.startswith(DIRECTIVE_MARK):
                continue
            kwargs, kernel, target, warns = _parse_kwarg_text(text[len(DIRECTIVE_MARK) :])
            warnings.extend(warns)
            directives.append(Directive(lineno=tok.start[0], kernel=kernel, kwargs=kwargs, target=target))
    except tokenize.TokenError:
        # Unterminated constructs mid-editing; return what we have.
        pass
    return directives, warnings


def _is_jit_decorator(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "jit"
    if isinstance(node, ast.Attribute):
        return node.attr == "jit"
    if isinstance(node, ast.Call):
        return _is_jit_decorator(node.func)
    return False


def find_kernel_defs(tree: ast.Module) -> dict[str, int]:
    """Map of ``@tilelang.jit`` function name -> def line, in document order."""
    found: dict[str, int] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(_is_jit_decorator(d) for d in node.decorator_list):
            found[node.name] = node.lineno
    return found


def kernel_symbols(source: str) -> list[dict]:
    """``documentSymbol`` payload for the ``@tilelang.jit`` kernels in *source*."""
    try:
        kernel_defs = find_kernel_defs(ast.parse(source))
    except SyntaxError:
        kernel_defs = {}
    return [{"name": name, "kind": "function", "line": line} for name, line in kernel_defs.items()]


def decorator_target(tree: ast.Module, kernel_name: str) -> str | None:
    """Literal ``target=`` from the kernel's ``@tilelang.jit(...)`` decorator, if any."""
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name != kernel_name:
            continue
        for dec in node.decorator_list:
            if not isinstance(dec, ast.Call) or not _is_jit_decorator(dec.func):
                continue
            for kw in dec.keywords:
                if kw.arg == "target":
                    ok, val = _literal(kw.value)
                    if ok and isinstance(val, str):
                        return val
        return None
    return None


def harvest_call_kwargs(tree: ast.Module, kernel_name: str) -> dict[str, Any]:
    """Collect literal kwargs from ``<kernel>.compile(...)`` or ``<kernel>(...)`` calls.

    Later call sites override earlier ones (``main()`` conventions put the
    canonical invocation last).
    """
    kwargs: dict[str, Any] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_compile = (
            isinstance(func, ast.Attribute) and func.attr == "compile" and isinstance(func.value, ast.Name) and func.value.id == kernel_name
        )
        is_direct = isinstance(func, ast.Name) and func.id == kernel_name
        if not (is_compile or is_direct):
            continue
        for kw in node.keywords:
            if kw.arg is None:
                continue
            ok, val = _literal(kw.value)
            if ok:
                kwargs[kw.arg] = val
    return kwargs


def signature_defaults(tree: ast.Module, kernel_name: str) -> dict[str, Any]:
    """Literal default values from the kernel function signature."""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == kernel_name:
            args = node.args
            positional = args.args
            defaults = args.defaults
            result: dict[str, Any] = {}
            for arg, default in zip(positional[len(positional) - len(defaults) :], defaults):
                ok, val = _literal(default)
                if ok:
                    result[arg.arg] = val
            for arg, default in zip(args.kwonlyargs, args.kw_defaults):
                if default is None:
                    continue
                ok, val = _literal(default)
                if ok:
                    result[arg.arg] = val
            return result
    return {}


def _choose_kernel(directives: list[Directive], kernel_defs: dict[str, int]) -> str | None:
    named = [d for d in directives if d.kernel]
    for d in reversed(named):
        if d.kernel in kernel_defs:
            return d.kernel
    if kernel_defs:
        return next(iter(kernel_defs))
    return named[-1].kernel if named else None


def _choose_directive(directives: list[Directive], kernel: str, def_line: int | None) -> Directive | None:
    """Nearest applicable directive above the kernel def (else nearest overall)."""
    applicable = [d for d in directives if d.kernel is None or d.kernel == kernel]
    if not applicable:
        return None
    if def_line is not None:
        above = [d for d in applicable if d.lineno < def_line]
        if above:
            return max(above, key=lambda d: d.lineno)
    return min(applicable, key=lambda d: abs(d.lineno - (def_line or d.lineno)))


def _directive_target(directives: list[Directive], kernel: str | None) -> str | None:
    """Last applicable directive carrying ``target=`` wins.

    Unlike kwargs (nearest directive only), target is merged across every
    applicable directive, so a bare ``# <typecheck> target=c`` line is not
    shadowed by a nearer target-less one. The mid-edit SyntaxError path uses
    the same rule.
    """
    target = None
    for d in directives:
        if kernel is not None and d.kernel is not None and d.kernel != kernel:
            continue
        if d.target:
            target = d.target
    return target


def resolve_config(source: str) -> KernelConfig:
    """Resolve (kernel, kwargs) for a TileLang source file."""
    warnings: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Mid-editing states: directives still work, AST harvest does not.
        directives, warns = parse_directives(source)
        kernel = None
        for d in reversed(directives):
            if d.kernel:
                kernel = d.kernel
                break
        merged: dict[str, Any] = {}
        for d in directives:
            if kernel is None or d.kernel in (None, kernel):
                merged.update(d.kwargs)
        return KernelConfig(
            kernel=kernel,
            kwargs=merged,
            origin="directive" if merged or kernel else "none",
            warnings=warns,
            target=_directive_target(directives, kernel),
        )

    kernel_defs = find_kernel_defs(tree)
    directives, warns = parse_directives(source)
    warnings.extend(warns)
    kernel = _choose_kernel(directives, kernel_defs)
    if kernel is None:
        return KernelConfig(kernel=None, origin="none", warnings=warnings)

    kwargs: dict[str, Any] = {}
    origin = "none"

    defaults = signature_defaults(tree, kernel)
    if defaults:
        kwargs.update(defaults)
        origin = "defaults"

    harvested = harvest_call_kwargs(tree, kernel)
    # target= steers the analysis; it is not a kernel kwarg (get_tir would choke).
    call_target = harvested.pop("target", None)
    if harvested:
        kwargs.update(harvested)
        origin = "call"

    directive = _choose_directive(directives, kernel, kernel_defs.get(kernel))
    if directive and directive.kwargs:
        kwargs.update(directive.kwargs)
        origin = "directive"

    target = _directive_target(directives, kernel)
    if target is None and isinstance(call_target, str):
        target = call_target
    if target is None:
        target = decorator_target(tree, kernel)
    return KernelConfig(kernel=kernel, kwargs=kwargs, origin=origin, warnings=warnings, target=target)
