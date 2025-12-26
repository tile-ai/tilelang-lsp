import sys
import ast
import logging
from typing import List, Optional, Dict, Any, Union
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


class ShapeEvaluator(ast.NodeVisitor):
    def __init__(self):
        self.variables: Dict[str, Any] = {}
        self.hints: List[InlayHint] = []

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # We process each function independently for local variables
        # Create a new scope for the function
        old_vars = self.variables.copy()

        # Process arguments (basic support)
        # We might not know argument values, but we can track them as symbolic if needed.
        # For this simple implementation, we focus on assignments within the function.

        self.generic_visit(node)

        # Restore variables after function
        self.variables = old_vars

    def visit_Assign(self, node: ast.Assign):
        # Handle simple assignments: x = value
        # This helps us resolve shapes like A_shared_shape = (M, K)
        try:
            value = self._evaluate(node.value)
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.variables[target.id] = value
                elif isinstance(target, ast.Tuple) and isinstance(value, tuple):
                    # Unpacking: x, y = (1, 2)
                    if len(target.elts) == len(value):
                        for var, val in zip(target.elts, value):
                            if isinstance(var, ast.Name):
                                self.variables[var.id] = val
        except Exception:
            pass

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        # Check for T.alloc_shared, T.alloc_local
        func_name = self._get_func_name(node.func)
        target_funcs = [
            "T.alloc_shared",
            "T.alloc_local",
            "T.alloc_fragment",
            "T.match_buffer",
            "T.Tensor",
        ]
        if func_name in target_funcs:
            self._create_inlay_hint(node)

        self.generic_visit(node)

    def _get_func_name(self, node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                return f"{node.value.id}.{node.attr}"
        elif isinstance(node, ast.Name):
            return node.id
        return None

    def _evaluate(self, node: ast.AST) -> Any:
        # Simple evaluator for literals, tuples, and variable lookups
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Num):  # Python < 3.8
            return node.n
        elif isinstance(node, ast.Str):  # Python < 3.8
            return node.s
        elif isinstance(node, ast.Tuple):
            return tuple(self._evaluate(elt) for elt in node.elts)
        elif isinstance(node, ast.List):
            return [self._evaluate(elt) for elt in node.elts]
        elif isinstance(node, ast.Name):
            return self.variables.get(node.id, node.id)  # Return name if not found
        elif isinstance(node, ast.Attribute):
            # Support T.float16 etc.
            if isinstance(node.value, ast.Name):
                return f"{node.value.id}.{node.attr}"
            return f"?.{node.attr}"
        elif isinstance(node, ast.BinOp):
            left = self._evaluate(node.left)
            right = self._evaluate(node.right)
            # Try to actually compute if both are numbers
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

            # Symbolic representation
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

    def _format_shape(self, shape: Any) -> str:
        # Format the shape for display
        return str(shape).replace("'", "")

    def _create_inlay_hint(self, node: ast.Call):
        if not node.args:
            return

        func_name = self._get_func_name(node.func)

        # 1. Shape (Argument 0)
        shape_arg = node.args[0]
        try:
            shape_val = self._evaluate(shape_arg)
        except Exception:
            shape_val = "?"

        # 2. Dtype (Argument 1 or keyword 'dtype')
        dtype_val = None
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

        # 3. Scope (Keyword 'scope' or inferred from func name)
        scope_val = None
        for kw in node.keywords:
            if kw.arg == "scope":
                try:
                    scope_val = self._evaluate(kw.value)
                except Exception:
                    pass
                break

        if scope_val is None and func_name:
            if "shared" in func_name:
                scope_val = "shared"
            elif "local" in func_name:
                scope_val = "local"
            elif "fragment" in func_name:
                scope_val = "local.fragment"
            elif "Tensor" in func_name:
                scope_val = "global"

        try:
            # Format Label
            # Format: : shape | dtype | scope
            label_parts = [self._format_shape(shape_val)]
            if dtype_val:
                label_parts.append(str(dtype_val))
            if scope_val:
                label_parts.append(str(scope_val))

            label = ": " + " | ".join(label_parts)

            # Alternative: Place hint at the end of the Call `T.alloc_shared(...) : shape`
            line = node.end_lineno if hasattr(node, "end_lineno") else node.lineno
            col = (
                node.end_col_offset
                if hasattr(node, "end_col_offset")
                else node.col_offset
            )

            self.hints.append(
                InlayHint(
                    position=Position(line=line - 1, character=col),
                    label=label,
                    kind=InlayHintKind.Type,
                    padding_left=True,
                )
            )

        except Exception as e:
            logger.debug(f"Failed to create hint: {e}")


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
        return []


if __name__ == "__main__":
    server.start_io()
