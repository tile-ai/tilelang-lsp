import sys
import ast
import logging
from typing import List, Optional, Dict, Any, Union, Tuple
from dataclasses import dataclass
from pygls.lsp.server import LanguageServer
from lsprotocol.types import (
    TEXT_DOCUMENT_INLAY_HINT,
    InlayHint,
    InlayHintKind,
    InlayHintParams,
    Position,
    TextDocumentIdentifier,
)

# Setup logging
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(__name__)

server = LanguageServer("tlsp", "v0.1")

# Configuration for allocation functions
ALLOC_FUNCS = {
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
    "match_buffer": {"default_scope": "match_buffer"},
}


@dataclass
class BufferInfo:
    shape: Tuple[Any, ...]
    dtype: Optional[str] = None
    scope: Optional[str] = None

    def __eq__(self, other):
        if not isinstance(other, BufferInfo):
            return False
        s1 = (
            tuple(self.shape)
            if isinstance(self.shape, (list, tuple))
            else (self.shape,)
        )
        s2 = (
            tuple(other.shape)
            if isinstance(other.shape, (list, tuple))
            else (other.shape,)
        )
        return s1 == s2 and self.dtype == other.dtype and self.scope == other.scope

    def __str__(self):
        parts = []
        if self.shape is not None:
            parts.append(str(self.shape).replace("'", ""))
        if self.dtype:
            parts.append(str(self.dtype))
        if self.scope:
            parts.append(str(self.scope))
        return " | ".join(parts)


class ShapeEvaluator(ast.NodeVisitor):
    def __init__(self):
        self.variables: Dict[str, Any] = {}
        self.hints: List[InlayHint] = []

    def visit_FunctionDef(self, node: ast.FunctionDef):
        old_vars = self.variables.copy()

        # Parse arguments with T.Tensor annotation
        for arg in node.args.args:
            if arg.annotation:
                try:
                    if isinstance(arg.annotation, ast.Call):
                        func_name = self._get_func_name(arg.annotation.func)
                        if func_name and "Tensor" in func_name:
                            buf_info = self._evaluate_alloc(
                                arg.annotation, ALLOC_FUNCS["Tensor"]
                            )
                            if buf_info:
                                self.variables[arg.arg] = buf_info
                except:
                    pass

        self.generic_visit(node)
        self.variables = old_vars

    def visit_Assign(self, node: ast.Assign):
        evaluated = None
        eval_ok = False
        try:
            evaluated = self._evaluate(node.value)
            eval_ok = True
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

        # Add type hints for non-TileLang variables
        if self._is_tilelang_call(node.value):
            # TileLang alloc calls already get hints from visit_Call
            self.generic_visit(node)
            return

        # If the RHS resolves to a BufferInfo, show buffer properties
        if eval_ok and isinstance(evaluated, BufferInfo):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.hints.append(
                        InlayHint(
                            position=Position(
                                line=target.end_lineno - 1,
                                character=target.end_col_offset,
                            ),
                            label=f": {evaluated}",
                            kind=InlayHintKind.Type,
                            padding_left=True,
                        )
                    )
            self.generic_visit(node)
            return

        type_label = self._infer_type_label(node.value)
        if type_label:
            resolved = evaluated if eval_ok else None

            for target in node.targets:
                if isinstance(target, ast.Name):
                    label = self._format_var_hint(
                        type_label, resolved, node.value, target.id
                    )
                    self.hints.append(
                        InlayHint(
                            position=Position(
                                line=target.end_lineno - 1,
                                character=target.end_col_offset,
                            ),
                            label=label,
                            kind=InlayHintKind.Type,
                            padding_left=True,
                        )
                    )
                elif isinstance(target, ast.Tuple):
                    elem_types = self._infer_tuple_unpack_types(node.value)
                    elem_values = self._resolve_tuple_values(node.value)
                    for i, elt in enumerate(target.elts):
                        if not isinstance(elt, ast.Name):
                            continue
                        et = elem_types[i] if i < len(elem_types) else None
                        ev = elem_values[i] if i < len(elem_values) else None
                        if et:
                            label = self._format_var_hint(et, ev, None, elt.id)
                            self.hints.append(
                                InlayHint(
                                    position=Position(
                                        line=elt.end_lineno - 1,
                                        character=elt.end_col_offset,
                                    ),
                                    label=label,
                                    kind=InlayHintKind.Type,
                                    padding_left=True,
                                )
                            )

        self.generic_visit(node)

    def visit_Expr(self, node: ast.Expr):
        """Handle bare expression statements like a standalone `Q_shared` line."""
        inner = node.value
        # Single variable reference
        if isinstance(inner, ast.Name):
            val = self.variables.get(inner.id)
            if isinstance(val, BufferInfo):
                self.hints.append(
                    InlayHint(
                        position=Position(
                            line=inner.end_lineno - 1,
                            character=inner.end_col_offset,
                        ),
                        label=f": {val}",
                        kind=InlayHintKind.Type,
                        padding_left=True,
                    )
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        func_name = self._get_func_name(node.func)

        # 1. Handle Allocation Functions
        matched_config = None
        if func_name:
            name_parts = func_name.split(".")
            base_name = name_parts[-1]
            if base_name in ALLOC_FUNCS:
                matched_config = ALLOC_FUNCS[base_name]

        if matched_config:
            try:
                buf_info = self._evaluate_alloc(node, matched_config)
                if buf_info:
                    self._add_hint(node, f": {buf_info}")
            except Exception as e:
                logger.debug(f"Alloc hint error: {e}")

        # 2. Handle Operation Functions (T.copy, etc.)
        elif func_name and (func_name.endswith(".copy") or func_name == "copy"):
            if len(node.args) >= 2:
                try:
                    src_info = self._evaluate(node.args[0])
                    dst_info = self._evaluate(node.args[1])

                    if isinstance(src_info, BufferInfo) and isinstance(
                        dst_info, BufferInfo
                    ):
                        src_is_0dim = not src_info.shape or src_info.shape == ()
                        dst_is_0dim = not dst_info.shape or dst_info.shape == ()

                        if src_is_0dim and not dst_is_0dim:
                            src_str = (
                                f"ptr of size {self._format_shape(dst_info.shape)}"
                            )
                            if src_info.scope:
                                src_str += f" ({src_info.scope})"
                            label = f"copy: {src_str} -> {dst_info}"
                        elif dst_is_0dim and not src_is_0dim:
                            dst_str = (
                                f"ptr of size {self._format_shape(src_info.shape)}"
                            )
                            if dst_info.scope:
                                dst_str += f" ({dst_info.scope})"
                            label = f"copy: {src_info} -> {dst_str}"
                        elif src_info == dst_info:
                            label = f"copy: {src_info}"
                        else:
                            label = f"copy: {src_info} -> {dst_info}"
                        self._add_hint(node, label)
                except Exception as e:
                    logger.debug(f"Copy hint error: {e}")

        self.generic_visit(node)

    def _add_hint(self, node: ast.AST, label: str):
        line = node.end_lineno if hasattr(node, "end_lineno") else node.lineno
        col = (
            node.end_col_offset if hasattr(node, "end_col_offset") else node.col_offset
        )
        self.hints.append(
            InlayHint(
                position=Position(line=line - 1, character=col),
                label=label,
                kind=InlayHintKind.Type,
                padding_left=True,
            )
        )

    def _get_func_name(self, node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                return f"{node.value.id}.{node.attr}"
        elif isinstance(node, ast.Name):
            return node.id
        return None

    def _check_broadcast(
        self, shape1: Tuple[Any, ...], shape2: Tuple[Any, ...]
    ) -> bool:
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

    def _evaluate(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Call):
            func_name = self._get_func_name(node.func)
            if func_name:
                name_parts = func_name.split(".")
                base_name = name_parts[-1]
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

            # Shape Check
            if isinstance(left, BufferInfo) and isinstance(right, BufferInfo):
                is_scalar_op = (not left.shape or left.shape == ()) or (
                    not right.shape or right.shape == ()
                )
                if not is_scalar_op:
                    if not self._check_broadcast(left.shape, right.shape):
                        msg = f"WARNING: shape mismatch {self._format_shape(left.shape)} vs {self._format_shape(right.shape)}"
                        self._add_hint(node, f": {msg}")
                # Return left as result info (simplified)
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
                ast.Add: "+",
                ast.Sub: "-",
                ast.Mult: "*",
                ast.Div: "/",
                ast.FloorDiv: "//",
                ast.Mod: "%",
            }
            op_str = op_map.get(type(node.op), "?")
            return f"({left} {op_str} {right})"

        raise ValueError("Cannot evaluate")

    def _evaluate_alloc(
        self, node: ast.Call, config: Dict[str, Any]
    ) -> Optional[BufferInfo]:
        special_handling = config.get("special_handling")
        shape_val = "?"
        dtype_val = None
        scope_val = config.get("default_scope")

        if special_handling == "alloc_var":
            if node.args:
                try:
                    dtype_val = self._evaluate(node.args[0])
                except:
                    pass
                shape_val = (1,)
                if len(node.args) > 1:
                    try:
                        arg1 = self._evaluate(node.args[1])
                        if isinstance(arg1, str) and "local" in arg1:
                            scope_val = arg1
                    except:
                        pass

        elif special_handling == "alloc_descriptor":
            shape_val = (1,)
            kind = "wgmma"
            if node.args:
                try:
                    kind = self._evaluate(node.args[0])
                except:
                    pass
            else:
                for kw in node.keywords:
                    if kw.arg == "kind":
                        try:
                            kind = self._evaluate(kw.value)
                        except:
                            pass
                        break
            if kind in ["wgmma", "tcgen05_smem", "tcgen05_instr"]:
                scope_val = f"local.descriptor.{kind}"

            if len(node.args) > 1:
                try:
                    dtype_val = self._evaluate(node.args[1])
                except:
                    pass
            if dtype_val is None:
                for kw in node.keywords:
                    if kw.arg == "dtype":
                        try:
                            dtype_val = self._evaluate(kw.value)
                        except:
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
                except:
                    pass

            if fixed_dtype:
                dtype_val = fixed_dtype
            elif len(node.args) > arg_idx_dtype:
                try:
                    dtype_val = self._evaluate(node.args[arg_idx_dtype])
                except:
                    pass

            if dtype_val is None and not fixed_dtype:
                for kw in node.keywords:
                    if kw.arg == "dtype":
                        try:
                            dtype_val = self._evaluate(kw.value)
                        except:
                            pass
                        break

        for kw in node.keywords:
            if kw.arg == "scope":
                try:
                    scope_val = self._evaluate(kw.value)
                except:
                    pass
                break

        return BufferInfo(shape=shape_val, dtype=dtype_val, scope=scope_val)

    def _parse_shape_str(self, shape_str: str) -> Any:
        try:
            return ast.literal_eval(shape_str)
        except:
            return shape_str

    def _format_shape(self, shape: Any) -> str:
        return str(shape).replace("'", "")

    def _format_var_hint(
        self,
        type_label: str,
        resolved: Any,
        value_node: Optional[ast.AST],
        var_name: str,
    ) -> str:
        """Build a hint string like ': int = 30' or ': str' depending on context."""
        if resolved is None or isinstance(resolved, BufferInfo):
            return f": {type_label}"

        resolved_str = self._format_resolved(resolved)

        # Don't show value if it's the same as what's literally written in source
        # e.g. `x = 30` — the "30" is already visible
        if value_node and self._is_plain_literal(value_node):
            return f": {type_label}"

        # Don't show value if it equals the variable name (unresolved reference)
        if isinstance(resolved, str) and resolved == var_name:
            return f": {type_label}"

        # Truncate long values
        if len(resolved_str) > 40:
            resolved_str = resolved_str[:37] + "..."

        return f": {type_label} = {resolved_str}"

    def _format_resolved(self, value: Any) -> str:
        """Format a resolved value for display."""
        if isinstance(value, str):
            # Could be a variable reference like "T.float16" or actual string
            return repr(value) if not value.startswith(("T.", "torch.")) else value
        if isinstance(value, (int, float, bool)):
            return repr(value)
        if isinstance(value, (list, tuple)):
            inner = ", ".join(self._format_resolved(v) for v in value)
            if isinstance(value, tuple):
                return f"({inner},)" if len(value) == 1 else f"({inner})"
            return f"[{inner}]"
        return str(value)

    def _type_of_value(self, val: Any) -> Optional[str]:
        """Return a type label for a resolved Python value."""
        if isinstance(val, BufferInfo):
            return None  # handled by TileLang hints
        if isinstance(val, bool):
            return "bool"
        if isinstance(val, int):
            return "int"
        if isinstance(val, float):
            return "float"
        if isinstance(val, str):
            return "str"
        if isinstance(val, list):
            return "list"
        if isinstance(val, tuple):
            return "tuple"
        if isinstance(val, dict):
            return "dict"
        if isinstance(val, set):
            return "set"
        if val is None:
            return "None"
        return None

    def _is_plain_literal(self, node: ast.AST) -> bool:
        """Check if a node is a simple literal that the user can already see."""
        if isinstance(node, ast.Constant):
            return True
        # Negative numbers: -1 is UnaryOp(USub, Constant)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return isinstance(node.operand, ast.Constant)
        return False

    def _resolve_tuple_values(self, node: ast.AST) -> List[Any]:
        """Resolve individual values from a tuple expression."""
        if isinstance(node, ast.Tuple):
            result = []
            for elt in node.elts:
                try:
                    result.append(self._evaluate(elt))
                except Exception:
                    result.append(None)
            return result
        return []

    def _is_tilelang_call(self, node: ast.AST) -> bool:
        """Check if a node is a TileLang alloc/Tensor call."""
        if not isinstance(node, ast.Call):
            return False
        func_name = self._get_func_name(node.func)
        if not func_name:
            return False
        base_name = func_name.split(".")[-1]
        return base_name in ALLOC_FUNCS

    def _infer_type_label(self, node: ast.AST) -> Optional[str]:
        """Infer a human-readable type label for a value expression."""
        # Variable references: look up resolved value type
        if isinstance(node, ast.Name):
            val = self.variables.get(node.id)
            if val is not None:
                return self._type_of_value(val)
            return None

        # Attribute access: e.g. torch.float16, self.x
        if isinstance(node, ast.Attribute):
            try:
                val = self._evaluate(node)
                if val is not None:
                    return self._type_of_value(val)
            except Exception:
                pass
            return None

        # Literals
        if isinstance(node, ast.Constant):
            v = node.value
            if v is None:
                return "None"
            if isinstance(v, bool):
                return "bool"
            if isinstance(v, int):
                return "int"
            if isinstance(v, float):
                return "float"
            if isinstance(v, str):
                return "str"
            if isinstance(v, bytes):
                return "bytes"
            if isinstance(v, complex):
                return "complex"
            return None

        # f-strings
        if isinstance(node, ast.JoinedStr):
            return "str"

        # Unary ops: -1 -> int, not x -> bool
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Not):
                return "bool"
            inner = self._infer_type_label(node.operand)
            return inner

        # Boolean ops: x and y, x or y
        if isinstance(node, ast.BoolOp):
            return "bool"

        # Comparisons: x < y, x == y, x in y
        if isinstance(node, ast.Compare):
            return "bool"

        # Collection literals
        if isinstance(node, ast.List):
            elem_type = self._infer_collection_elem_type(node.elts)
            return f"list[{elem_type}]" if elem_type else "list"

        if isinstance(node, ast.Tuple):
            elem_types = [self._infer_type_label(e) or "?" for e in node.elts]
            return f"tuple[{', '.join(elem_types)}]"

        if isinstance(node, ast.Set):
            elem_type = self._infer_collection_elem_type(node.elts)
            return f"set[{elem_type}]" if elem_type else "set"

        if isinstance(node, ast.Dict):
            key_type = self._infer_collection_elem_type(
                [k for k in node.keys if k is not None]
            )
            val_type = self._infer_collection_elem_type(node.values)
            if key_type and val_type:
                return f"dict[{key_type}, {val_type}]"
            return "dict"

        # Comprehensions
        if isinstance(node, ast.ListComp):
            elem_type = self._infer_type_label(node.elt)
            return f"list[{elem_type}]" if elem_type else "list"
        if isinstance(node, ast.SetComp):
            elem_type = self._infer_type_label(node.elt)
            return f"set[{elem_type}]" if elem_type else "set"
        if isinstance(node, ast.DictComp):
            kt = self._infer_type_label(node.key)
            vt = self._infer_type_label(node.value)
            if kt and vt:
                return f"dict[{kt}, {vt}]"
            return "dict"
        if isinstance(node, ast.GeneratorExp):
            return "Generator"

        # Binary ops between numeric literals
        if isinstance(node, ast.BinOp):
            lt = self._infer_type_label(node.left)
            rt = self._infer_type_label(node.right)
            if lt and rt:
                if "float" in (lt, rt) or "complex" in (lt, rt):
                    return "float" if "complex" not in (lt, rt) else "complex"
                if lt == rt == "int":
                    if isinstance(node.op, ast.Div):
                        return "float"
                    return "int"
                if lt == rt == "str" and isinstance(node.op, ast.Mod):
                    return "str"
            return None

        # Function calls with known return types
        if isinstance(node, ast.Call):
            func_name = self._get_func_name(node.func)
            if func_name:
                builtin_types = {
                    "int": "int", "float": "float", "str": "str",
                    "bool": "bool", "list": "list", "dict": "dict",
                    "set": "set", "tuple": "tuple", "bytes": "bytes",
                    "bytearray": "bytearray", "complex": "complex",
                    "frozenset": "frozenset", "range": "range",
                    "len": "int", "abs": "int", "round": "int",
                    "min": "int", "max": "int", "sum": "int",
                    "sorted": "list", "reversed": "iterator",
                    "enumerate": "enumerate", "zip": "zip",
                    "map": "map", "filter": "filter",
                    "repr": "str", "chr": "str", "ord": "int",
                    "hex": "str", "oct": "str", "bin": "str",
                    "isinstance": "bool", "issubclass": "bool",
                    "hasattr": "bool", "callable": "bool",
                    "type": "type", "id": "int", "hash": "int",
                    "input": "str", "open": "TextIO",
                }
                base = func_name.split(".")[-1]
                if base in builtin_types:
                    return builtin_types[base]
            return None

        # Await expression
        if isinstance(node, ast.Await):
            return None

        # Starred expression
        if isinstance(node, ast.Starred):
            return None

        # IfExp: x if cond else y
        if isinstance(node, ast.IfExp):
            bt = self._infer_type_label(node.body)
            ot = self._infer_type_label(node.orelse)
            if bt and ot and bt == ot:
                return bt
            if bt and ot:
                return f"{bt} | {ot}"
            return bt or ot

        return None

    def _infer_collection_elem_type(self, elts: List[ast.AST]) -> Optional[str]:
        """If all elements in a collection share a type, return it."""
        if not elts:
            return None
        types = set()
        for e in elts:
            t = self._infer_type_label(e)
            if t is None:
                return None
            types.add(t)
        if len(types) == 1:
            return types.pop()
        return " | ".join(sorted(types))

    def _infer_tuple_unpack_types(self, node: ast.AST) -> List[Optional[str]]:
        """Infer types for each element of a tuple unpacking."""
        if isinstance(node, ast.Tuple):
            return [self._infer_type_label(e) for e in node.elts]
        # For function calls that return tuples, we can't easily infer
        return []

    def _evaluate_subscript(self, node: ast.Subscript) -> Any:
        try:
            value_info = self._evaluate(node.value)
            if not isinstance(value_info, BufferInfo):
                return "?"

            original_shape = value_info.shape
            if not isinstance(original_shape, (tuple, list)):
                return value_info

            slice_items = []
            if sys.version_info >= (3, 9):
                if isinstance(node.slice, ast.Tuple):
                    slice_items = node.slice.elts
                else:
                    slice_items = [node.slice]
            else:
                if isinstance(node.slice, ast.Index):
                    slice_items = [node.slice.value]
                elif isinstance(node.slice, ast.Slice):
                    slice_items = [node.slice]
                elif isinstance(node.slice, ast.ExtSlice):
                    slice_items = [
                        d.value if isinstance(d, ast.Index) else d
                        for d in node.slice.dims
                    ]
                else:
                    slice_items = [node.slice]

            new_shape = []
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
                            new_dim = (upper - lower + (step - 1)) // step
                            new_shape.append(new_dim)
                        else:
                            new_shape.append(f"({upper}-{lower})//{step}")
                    except:
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

        except Exception as e:
            logger.debug(f"Slice eval error: {e}")
            return "?"


@server.feature(TEXT_DOCUMENT_INLAY_HINT)
def inlay_hints(params: InlayHintParams) -> Optional[List[InlayHint]]:
    document = server.workspace.get_text_document(params.text_document.uri)
    source = document.source

    # Try parsing the full document first
    tree = None
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Fallback: try to parse valid portions by removing bad lines
        tree = _parse_with_fallback(source)

    if tree is None:
        return []

    try:
        evaluator = ShapeEvaluator()
        evaluator.visit(tree)
        return evaluator.hints
    except Exception as e:
        logger.error(f"Error evaluating hints: {e}")
        return []


def _parse_with_fallback(source: str) -> Optional[ast.Module]:
    """Try to parse as much of the source as possible by commenting out bad lines."""
    lines = source.splitlines(keepends=True)
    max_attempts = 20
    for _ in range(max_attempts):
        try:
            return ast.parse("".join(lines))
        except SyntaxError as e:
            if e.lineno is None:
                return None
            # Replace the offending line with a comment
            idx = e.lineno - 1
            if 0 <= idx < len(lines):
                # Preserve the newline
                nl = "\n" if lines[idx].endswith("\n") else ""
                lines[idx] = f"# [syntax error]{nl}"
            else:
                return None
    return None


if __name__ == "__main__":
    server.start_io()
