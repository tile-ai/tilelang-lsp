"""Pure-AST static analysis — the last-resort analysis tier.

Runs without importing tilelang: parses the source, tracks variable bindings
well enough to evaluate shapes, and emits hints for ``T.alloc_*``/``T.Tensor``/
``T.empty`` buffers and ``T.copy`` calls, plus broadcast-mismatch warnings as
real diagnostics.

Used by the LSP server directly (instant feedback on every keystroke) and by
the analyzer subprocess when the module cannot be imported/elaborated.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from . import protocol


ALLOC_FUNCS: Dict[str, Dict[str, Any]] = {
    "alloc_shared": {"default_scope": "shared.dyn"},
    "alloc_local": {"default_scope": "local"},
    "alloc_fragment": {"default_scope": "local.fragment"},
    "alloc_var": {"default_scope": "local.var", "special_handling": "alloc_var"},
    "alloc_barrier": {
        "default_scope": "shared.barrier",
        "arg_idx_shape": 0,
        "fixed_dtype": "uint64",
    },
    "alloc_tmem": {"default_scope": "shared.tmem"},
    "alloc_reducer": {"default_scope": "local.fragment"},
    "alloc_descriptor": {
        "default_scope": "local.descriptor.wgmma",
        "special_handling": "alloc_descriptor",
    },
    "alloc_wgmma_desc": {
        "default_scope": "local.descriptor.wgmma",
        "fixed_shape": "(1,)",
    },
    "alloc_tcgen05_smem_desc": {
        "default_scope": "local.descriptor.tcgen05_smem",
        "fixed_shape": "(1,)",
    },
    "alloc_tcgen05_instruction_desc": {
        "default_scope": "local.descriptor.tcgen05_instr",
        "fixed_shape": "(1,)",
    },
    "alloc_tcgen05_instr_desc": {
        "default_scope": "local.descriptor.tcgen05_instr",
        "fixed_shape": "(1,)",
    },
    "Tensor": {"default_scope": "global"},
    "empty": {"default_scope": "global"},
    "match_buffer": {"default_scope": "match_buffer"},
}


def _norm_dtype(dtype: Any) -> Optional[str]:
    """'T.float16' -> 'float16'; keep plain names/strings as-is."""
    if dtype is None:
        return None
    if isinstance(dtype, str):
        return dtype[2:] if dtype.startswith("T.") else dtype
    return str(dtype)


@dataclass
class BufferInfo:
    shape: Any
    dtype: Optional[str] = None
    scope: Optional[str] = None

    def __eq__(self, other):
        if not isinstance(other, BufferInfo):
            return False
        s1 = tuple(self.shape) if isinstance(self.shape, (list, tuple)) else (self.shape,)
        s2 = tuple(other.shape) if isinstance(other.shape, (list, tuple)) else (other.shape,)
        return s1 == s2 and self.dtype == other.dtype and self.scope == other.scope

    def label(self) -> str:
        return protocol.format_buffer_label(self.shape, self.dtype, self.scope)


@dataclass
class StaticIndex:
    """Name -> line index reused by span/param fallbacks."""

    params: Dict[str, int] = field(default_factory=dict)
    allocs: Dict[str, int] = field(default_factory=dict)


@dataclass
class StaticResult:
    hints: List[dict] = field(default_factory=list)
    diagnostics: List[dict] = field(default_factory=list)
    index: StaticIndex = field(default_factory=StaticIndex)


class ShapeEvaluator(ast.NodeVisitor):
    def __init__(self):
        self.variables: Dict[str, Any] = {}
        self.result = StaticResult()

    # -- statements ---------------------------------------------------------

    def visit_FunctionDef(self, node: ast.FunctionDef):
        old_vars = self.variables.copy()

        for arg in node.args.args + node.args.kwonlyargs:
            if arg.annotation is None:
                continue
            try:
                if isinstance(arg.annotation, ast.Call):
                    func_name = self._get_func_name(arg.annotation.func)
                    if func_name and func_name.split(".")[-1] in ("Tensor", "empty"):
                        buf_info = self._evaluate_alloc(
                            arg.annotation, ALLOC_FUNCS[func_name.split(".")[-1]]
                        )
                        if buf_info:
                            self.variables[arg.arg] = buf_info
                            self.result.index.params.setdefault(arg.arg, arg.lineno)
                            self._add_hint(arg.annotation, buf_info.label(),
                                           protocol.HINT_PARAM, buffer=arg.arg)
            except Exception:
                pass

        self.generic_visit(node)
        self.variables = old_vars

    def visit_AnnAssign(self, node: ast.AnnAssign):
        # New-style DSL: `A: T.Tensor((M, N), dtype)` — the annotation *is* the
        # tensor declaration call (node.value is None in this form).
        call = None
        if isinstance(node.annotation, ast.Call):
            call = node.annotation
        elif isinstance(node.value, ast.Call):
            call = node.value
        if call is not None:
            func_name = self._get_func_name(call.func)
            base = func_name.split(".")[-1] if func_name else None
            if base in ALLOC_FUNCS and isinstance(node.target, ast.Name):
                try:
                    buf_info = self._evaluate_alloc(call, ALLOC_FUNCS[base])
                    if buf_info:
                        self.variables[node.target.id] = buf_info
                        if base in ("Tensor", "empty"):
                            self.result.index.params.setdefault(node.target.id, node.lineno)
                            kind = protocol.HINT_PARAM
                        else:
                            self.result.index.allocs.setdefault(node.target.id, node.lineno)
                            kind = protocol.HINT_ALLOC
                        # visit_Call skips nodes parented to AnnAssign, so emit
                        # the hint here for both annotation- and value-form.
                        self._add_hint(call, buf_info.label(), kind, buffer=node.target.id)
                except Exception:
                    pass
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        evaluated = None
        try:
            evaluated = self._evaluate(node.value)
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.variables[target.id] = evaluated
                elif isinstance(target, ast.Tuple) and isinstance(evaluated, tuple):
                    if len(target.elts) == len(evaluated):
                        for var, val in zip(target.elts, evaluated):
                            if isinstance(var, ast.Name):
                                self.variables[var.id] = val
        except Exception:
            pass

        # Index `X = T.alloc_*(...)` / `X = T.empty(...)` sites by name.
        if isinstance(node.value, ast.Call):
            func_name = self._get_func_name(node.value.func)
            base = func_name.split(".")[-1] if func_name else None
            if base in ALLOC_FUNCS:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if base in ("Tensor", "empty"):
                            self.result.index.params.setdefault(target.id, node.lineno)
                        else:
                            self.result.index.allocs.setdefault(target.id, node.lineno)

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        func_name = self._get_func_name(node.func)
        base_name = func_name.split(".")[-1] if func_name else None

        if base_name in ALLOC_FUNCS:
            # Annotation calls are already hinted by visit_FunctionDef/AnnAssign.
            if not isinstance(node.parent, (ast.arg, ast.AnnAssign)):
                try:
                    buf_info = self._evaluate_alloc(node, ALLOC_FUNCS[base_name])
                    if buf_info:
                        kind = (protocol.HINT_PARAM
                                if base_name in ("Tensor", "empty")
                                else protocol.HINT_ALLOC)
                        buffer = self._assigned_name(node)
                        self._add_hint(node, buf_info.label(), kind, buffer=buffer)
                except Exception:
                    pass
        elif base_name == "copy":
            self._handle_copy(node)

        self.generic_visit(node)

    # -- copy / mismatch ----------------------------------------------------

    def _handle_copy(self, node: ast.Call):
        if len(node.args) < 2:
            return
        try:
            src_info = self._evaluate(node.args[0])
            dst_info = self._evaluate(node.args[1])
            if not isinstance(src_info, BufferInfo) or not isinstance(dst_info, BufferInfo):
                return
            src_is_0dim = not src_info.shape or src_info.shape == ()
            dst_is_0dim = not dst_info.shape or dst_info.shape == ()
            if src_is_0dim and not dst_is_0dim:
                label = f"copy ptr({self._fmt_shape(dst_info.shape)}) ⇒ {dst_info.label()}"
            elif dst_is_0dim and not src_is_0dim:
                label = f"copy {src_info.label()} ⇒ ptr({self._fmt_shape(src_info.shape)})"
            elif src_info == dst_info:
                label = f"copy {src_info.label()}"
            else:
                label = f"copy {src_info.label()} ⇒ {dst_info.label()}"
            self._add_hint(node, label, protocol.HINT_OP, buffer=None)
        except Exception:
            pass

    # -- evaluation ---------------------------------------------------------

    def _evaluate(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Call):
            func_name = self._get_func_name(node.func)
            if func_name:
                base_name = func_name.split(".")[-1]
                if base_name in ALLOC_FUNCS:
                    return self._evaluate_alloc(node, ALLOC_FUNCS[base_name])

        if isinstance(node, ast.Subscript):
            return self._evaluate_subscript(node)

        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Tuple):
            return tuple(self._evaluate(elt) for elt in node.elts)
        elif isinstance(node, ast.List):
            return [self._evaluate(elt) for elt in node.elts]
        elif isinstance(node, ast.Name):
            return self.variables.get(node.id, node.id)
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                return f"{node.value.id}.{node.attr}"
            return f"?.{node.attr}"
        elif isinstance(node, ast.BinOp):
            left = self._evaluate(node.left)
            right = self._evaluate(node.right)

            if isinstance(left, BufferInfo) and isinstance(right, BufferInfo):
                is_scalar_op = (not left.shape or left.shape == ()) or (
                    not right.shape or right.shape == ()
                )
                if not is_scalar_op and not self._check_broadcast(left.shape, right.shape):
                    msg = (f"shape mismatch {self._fmt_shape(left.shape)} vs "
                           f"{self._fmt_shape(right.shape)} (broadcast incompatible)")
                    self.result.diagnostics.append(protocol.make_diagnostic(
                        line=node.lineno,
                        message=msg,
                        severity=protocol.SEVERITY_WARNING,
                        source="tilelang-static",
                    ))
                return left

            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                if isinstance(node.op, ast.Div):
                    return left / right
                if isinstance(node.op, ast.FloorDiv):
                    return left // right
                if isinstance(node.op, ast.Mod):
                    return left % right

            op_map = {
                ast.Add: "+", ast.Sub: "-", ast.Mult: "*",
                ast.Div: "/", ast.FloorDiv: "//", ast.Mod: "%",
            }
            op_str = op_map.get(type(node.op), "?")
            return f"({left} {op_str} {right})"

        raise ValueError("Cannot evaluate")

    def _evaluate_alloc(self, node: ast.Call, config: Dict[str, Any]) -> Optional[BufferInfo]:
        special_handling = config.get("special_handling")
        shape_val: Any = "?"
        dtype_val = None
        scope_val = config.get("default_scope")

        if special_handling == "alloc_var":
            if node.args:
                try:
                    dtype_val = self._evaluate(node.args[0])
                except Exception:
                    pass
                shape_val = (1,)
                if len(node.args) > 1:
                    try:
                        arg1 = self._evaluate(node.args[1])
                        if isinstance(arg1, str) and "local" in arg1:
                            scope_val = arg1
                    except Exception:
                        pass

        elif special_handling == "alloc_descriptor":
            shape_val = (1,)
            kind = "wgmma"
            if node.args:
                try:
                    kind = self._evaluate(node.args[0])
                except Exception:
                    pass
            else:
                for kw in node.keywords:
                    if kw.arg == "kind":
                        try:
                            kind = self._evaluate(kw.value)
                        except Exception:
                            pass
                        break
            if kind in ["wgmma", "tcgen05_smem", "tcgen05_instr"]:
                scope_val = f"local.descriptor.{kind}"

            if len(node.args) > 1:
                try:
                    dtype_val = self._evaluate(node.args[1])
                except Exception:
                    pass
            if dtype_val is None:
                for kw in node.keywords:
                    if kw.arg == "dtype":
                        try:
                            dtype_val = self._evaluate(kw.value)
                        except Exception:
                            pass
                        break
        else:
            arg_idx_shape = config.get("arg_idx_shape", 0)
            arg_idx_dtype = config.get("arg_idx_dtype", 1)
            fixed_dtype = config.get("fixed_dtype")
            fixed_shape = config.get("fixed_shape")

            if fixed_shape:
                shape_val = self._parse_shape_str(fixed_shape)
            elif len(node.args) > arg_idx_shape:
                try:
                    shape_val = self._evaluate(node.args[arg_idx_shape])
                    if config.get("default_scope") == "shared.barrier":
                        shape_val = (shape_val,)
                except Exception:
                    pass

            if fixed_dtype:
                dtype_val = fixed_dtype
            elif len(node.args) > arg_idx_dtype:
                try:
                    dtype_val = self._evaluate(node.args[arg_idx_dtype])
                except Exception:
                    pass

            if dtype_val is None and not fixed_dtype:
                for kw in node.keywords:
                    if kw.arg == "dtype":
                        try:
                            dtype_val = self._evaluate(kw.value)
                        except Exception:
                            pass
                        break

        for kw in node.keywords:
            if kw.arg == "scope":
                try:
                    scope_val = self._evaluate(kw.value)
                except Exception:
                    pass
                break

        return BufferInfo(shape=shape_val, dtype=_norm_dtype(dtype_val), scope=scope_val)

    def _evaluate_subscript(self, node: ast.Subscript) -> Any:
        try:
            value_info = self._evaluate(node.value)
            if not isinstance(value_info, BufferInfo):
                return "?"

            original_shape = value_info.shape
            if not isinstance(original_shape, (tuple, list)):
                return value_info

            slice_items: List[ast.AST] = []
            if isinstance(node.slice, ast.Tuple):
                slice_items = list(node.slice.elts)
            else:
                slice_items = [node.slice]

            new_shape: List[Any] = []
            current_dim = 0

            for item in slice_items:
                if current_dim >= len(original_shape):
                    break
                dim_size = original_shape[current_dim]
                if isinstance(item, ast.Slice):
                    try:
                        lower = self._evaluate(item.lower) if item.lower else 0
                        upper = self._evaluate(item.upper) if item.upper else dim_size
                        step = self._evaluate(item.step) if item.step else 1
                        if (
                            isinstance(lower, (int, float))
                            and isinstance(upper, (int, float))
                            and isinstance(step, (int, float))
                        ):
                            new_shape.append((upper - lower + (step - 1)) // step)
                        else:
                            new_shape.append(f"({upper}-{lower})//{step}")
                    except Exception:
                        new_shape.append("?")
                    current_dim += 1
                else:
                    current_dim += 1

            while current_dim < len(original_shape):
                new_shape.append(original_shape[current_dim])
                current_dim += 1

            return BufferInfo(
                shape=tuple(new_shape), dtype=value_info.dtype, scope=value_info.scope
            )
        except Exception:
            return "?"

    # -- helpers ------------------------------------------------------------

    def _add_hint(self, node: ast.AST, label: str, kind: str,
                  buffer: Optional[str] = None):
        line = node.end_lineno if hasattr(node, "end_lineno") else node.lineno
        self.result.hints.append(protocol.make_hint(
            line=line, kind=kind, label=label, buffer=buffer,
        ))

    def _assigned_name(self, node: ast.Call) -> Optional[str]:
        """Name of the variable this call is assigned to (via parent links)."""
        parent = getattr(node, "parent", None)
        if isinstance(parent, ast.Assign) and len(parent.targets) == 1:
            target = parent.targets[0]
            if isinstance(target, ast.Name):
                return target.id
        if isinstance(parent, ast.AnnAssign) and isinstance(parent.target, ast.Name):
            return parent.target.id
        return None

    def _get_func_name(self, node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                return f"{node.value.id}.{node.attr}"
        elif isinstance(node, ast.Name):
            return node.id
        return None

    def _check_broadcast(self, shape1: Any, shape2: Any) -> bool:
        s1 = list(shape1) if isinstance(shape1, (list, tuple)) else [shape1]
        s2 = list(shape2) if isinstance(shape2, (list, tuple)) else [shape2]
        while len(s1) < len(s2):
            s1.insert(0, 1)
        while len(s2) < len(s1):
            s2.insert(0, 1)
        for d1, d2 in zip(s1, s2):
            if isinstance(d1, (int, float)) and isinstance(d2, (int, float)):
                if d1 != d2 and d1 != 1 and d2 != 1:
                    return False
        return True

    def _parse_shape_str(self, shape_str: str) -> Any:
        try:
            return ast.literal_eval(shape_str)
        except Exception:
            return shape_str

    def _fmt_shape(self, shape: Any) -> str:
        return str(shape).replace("'", "")


def _attach_parents(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node  # type: ignore[attr-defined]


def parse_with_fallback(source: str) -> Optional[ast.Module]:
    """Parse; on SyntaxError comment out the offending line (up to 20 attempts)."""
    try:
        tree = ast.parse(source)
        _attach_parents(tree)
        return tree
    except SyntaxError:
        pass
    lines = source.splitlines(keepends=True)
    for _ in range(20):
        try:
            tree = ast.parse("".join(lines))
            _attach_parents(tree)
            return tree
        except SyntaxError as e:
            if e.lineno is None:
                return None
            idx = e.lineno - 1
            if 0 <= idx < len(lines):
                nl = "\n" if lines[idx].endswith("\n") else ""
                lines[idx] = f"# [syntax error]{nl}"
            else:
                return None
    return None


def analyze_static(source: str) -> StaticResult:
    """Run the pure-AST analysis. Never raises."""
    result = StaticResult()
    try:
        tree = parse_with_fallback(source)
    except Exception:
        return result
    if tree is None:
        return result
    try:
        evaluator = ShapeEvaluator()
        evaluator.result = result
        evaluator.visit(tree)
    except Exception:
        pass
    return result


def index_for_function(source: str, func_name: str) -> StaticIndex:
    """Name -> line index restricted to one kernel's body.

    The file-global index maps a buffer name to its first declaration, which
    is wrong when several kernels share parameter names (A/B). The span
    collector's param fallback needs the lines *of the analyzed kernel*.
    """
    index = StaticIndex()
    try:
        tree = parse_with_fallback(source)
    except Exception:
        return index
    if tree is None:
        return index
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            evaluator = ShapeEvaluator()
            evaluator.visit(node)
            return evaluator.result.index
    return index
