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
        try:
            value = self._evaluate(node.value)
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.variables[target.id] = value
                elif isinstance(target, ast.Tuple) and isinstance(value, tuple):
                    if len(target.elts) == len(value):
                        for var, val in zip(target.elts, value):
                            if isinstance(var, ast.Name):
                                self.variables[var.id] = val
        except Exception:
            pass
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
        elif isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.Str):
            return node.s
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
                        msg = f"Buffer shape mismatch {self._format_shape(left.shape)} vs {self._format_shape(right.shape)}"
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
    try:
        tree = ast.parse(document.source)
        evaluator = ShapeEvaluator()
        evaluator.visit(tree)
        return evaluator.hints
    except Exception as e:
        logger.error(f"Error parsing document: {e}")
        return None


if __name__ == "__main__":
    server.start_io()
