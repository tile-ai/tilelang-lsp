import unittest
import ast
from tlsp.server import ShapeEvaluator, ALLOC_FUNCS


class TestShapeEvaluator(unittest.TestCase):
    def test_basic_static_shape(self):
        code = """
import tilelang.language as T
def test_func():
    A = T.alloc_shared((10, 20), T.float16)
    B = T.alloc_local((5,), T.int32)
    """
        tree = ast.parse(code)
        evaluator = ShapeEvaluator()
        evaluator.visit(tree)
        self.assertEqual(len(evaluator.hints), 2)
        self.assertEqual(
            evaluator.hints[0].label, ": (10, 20) | T.float16 | shared.dyn"
        )
        self.assertEqual(evaluator.hints[1].label, ": (5,) | T.int32 | local")

    def test_alloc_var(self):
        code = """
import tilelang.language as T
def test_func():
    v1 = T.alloc_var(T.int32, 1)
    v2 = T.alloc_var(T.float32, "local.custom")
    """
        tree = ast.parse(code)
        evaluator = ShapeEvaluator()
        evaluator.visit(tree)
        self.assertEqual(len(evaluator.hints), 2)
        self.assertEqual(evaluator.hints[0].label, ": (1,) | T.int32 | local.var")
        self.assertEqual(evaluator.hints[1].label, ": (1,) | T.float32 | local.custom")

    def test_alloc_barrier(self):
        code = """
import tilelang.language as T
def test_func():
    b = T.alloc_barrier(4)
    """
        tree = ast.parse(code)
        evaluator = ShapeEvaluator()
        evaluator.visit(tree)
        self.assertEqual(len(evaluator.hints), 1)
        self.assertEqual(evaluator.hints[0].label, ": (4,) | uint64 | shared.barrier")

    def test_alloc_descriptor(self):
        code = """
import tilelang.language as T
def test_func():
    d1 = T.alloc_descriptor("tcgen05_smem")
    d2 = T.alloc_wgmma_desc()
    """
        tree = ast.parse(code)
        evaluator = ShapeEvaluator()
        evaluator.visit(tree)
        self.assertEqual(len(evaluator.hints), 2)
        self.assertEqual(
            evaluator.hints[0].label, ": (1,) | local.descriptor.tcgen05_smem"
        )
        self.assertEqual(evaluator.hints[1].label, ": (1,) | local.descriptor.wgmma")

    def test_alloc_tmem(self):
        code = """
import tilelang.language as T
def test_func():
    t = T.alloc_tmem((10, 10), T.float16)
    """
        tree = ast.parse(code)
        evaluator = ShapeEvaluator()
        evaluator.visit(tree)
        self.assertEqual(len(evaluator.hints), 1)
        self.assertEqual(
            evaluator.hints[0].label, ": (10, 10) | T.float16 | shared.tmem"
        )

    def test_slicing(self):
        code = """
import tilelang.language as T
def test_func():
    A = T.alloc_shared((10, 20), T.float16)
    T.copy(A[0, :], A[1, :])
    """
        tree = ast.parse(code)
        evaluator = ShapeEvaluator()
        evaluator.visit(tree)
        self.assertEqual(len(evaluator.hints), 2)
        self.assertIn("copy: (20,) | T.float16 | shared.dyn", evaluator.hints[1].label)

    def test_slicing_dimension_reduction(self):
        code = """
import tilelang.language as T
def test_func():
    A = T.alloc_shared((10, 20, 30), T.float16)
    B = T.alloc_local((20, 30), T.float16)
    T.copy(A[0], B)
    """
        tree = ast.parse(code)
        evaluator = ShapeEvaluator()
        evaluator.visit(tree)
        self.assertEqual(len(evaluator.hints), 3)
        self.assertIn("copy: (20, 30) | T.float16", evaluator.hints[2].label)

    def test_slicing_complex(self):
        code = """
import tilelang.language as T
def test_func():
    A = T.alloc_shared((10, 20), T.float16)
    B = T.alloc_local((5, 20), T.float16)
    T.copy(A[0:5, :], B)
    """
        tree = ast.parse(code)
        evaluator = ShapeEvaluator()
        evaluator.visit(tree)
        self.assertEqual(len(evaluator.hints), 3)
        self.assertIn("copy: (5, 20) | T.float16", evaluator.hints[2].label)

    def test_slicing_symbolic(self):
        code = """
def alloc_var_add(N, block_N, dtype,):
    import tilelang.language as T
    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), threads=block_N) as bx:
            A_shared = T.alloc_shared([block_N], dtype)
            T.copy(A[bx * block_N], A_shared)
            T.copy(A_shared, B[bx * block_N])
    """
        tree = ast.parse(code)
        evaluator = ShapeEvaluator()
        evaluator.visit(tree)
        copy_hints = [h for h in evaluator.hints if "copy:" in h.label]
        self.assertEqual(len(copy_hints), 2)
        hint1 = copy_hints[0].label
        self.assertIn("ptr", hint1)
        hint2 = copy_hints[1].label
        self.assertIn("ptr", hint2)

    def test_shape_mismatch(self):
        code = """
import tilelang.language as T
def test_func():
    A = T.alloc_shared((2, 1), T.float16)
    B = T.alloc_shared((3, 1), T.float16)
    C = A + B
    """
        tree = ast.parse(code)
        evaluator = ShapeEvaluator()
        evaluator.visit(tree)
        # Expect 1 hint for shape mismatch (plus hints for A and B)
        # A hint, B hint, Warning hint
        warning_hints = [h for h in evaluator.hints if "WARNING" in h.label]
        self.assertEqual(len(warning_hints), 1)
        hint = warning_hints[0]
        self.assertIn("shape mismatch", hint.label)
        self.assertIn("(2, 1)", hint.label)
        self.assertIn("(3, 1)", hint.label)

    def test_shape_broadcast(self):
        code = """
import tilelang.language as T
def test_func():
    A = T.alloc_shared((2, 3), T.float16)
    B = T.alloc_shared((3,), T.float16)
    # (2, 3) + (3,) -> (2, 3) compatible
    C = A + B
    """
        tree = ast.parse(code)
        evaluator = ShapeEvaluator()
        evaluator.visit(tree)
        # Expect no warning hints
        warning_hints = [h for h in evaluator.hints if "WARNING" in h.label]
        self.assertEqual(len(warning_hints), 0)


if __name__ == "__main__":
    unittest.main()
