import unittest
import ast
from server import ShapeEvaluator


class TestShapeEvaluator(unittest.TestCase):
    def test_basic_static_shape(self):
        """测试基本的静态shape解析"""
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

        # 验证第一个hint是 (10, 20) | T.float16 | shared
        hint1 = evaluator.hints[0]
        self.assertEqual(hint1.label, ": (10, 20) | T.float16 | shared")

        # 验证第二个hint是 (5,) | T.int32 | local
        hint2 = evaluator.hints[1]
        self.assertEqual(hint2.label, ": (5,) | T.int32 | local")

    def test_variable_shape(self):
        """测试通过变量引用的shape解析"""
        code = """
import tilelang.language as T

def test_func():
    shape = (30, 40)
    A = T.alloc_shared(shape, dtype="float32")
    """

        tree = ast.parse(code)
        evaluator = ShapeEvaluator()
        evaluator.visit(tree)

        self.assertEqual(len(evaluator.hints), 1)
        self.assertEqual(evaluator.hints[0].label, ": (30, 40) | float32 | shared")

    def test_explicit_scope(self):
        """测试显式指定scope"""
        code = """
import tilelang.language as T

def test_func():
    A = T.alloc_buffer((10, 20), dtype=T.float16, scope="shared.dyn")
    """
        # 注意：我们需要确保 alloc_buffer 在 target_funcs 中，或者我们只测试 target_funcs 中的函数
        # server.py 中 target_funcs 没有 T.alloc_buffer，但我们可以临时测试 alloc_shared 的 scope 参数

        code = """
import tilelang.language as T

def test_func():
    A = T.alloc_shared((10, 20), T.float16, scope="shared.dyn")
    """
        tree = ast.parse(code)
        evaluator = ShapeEvaluator()
        evaluator.visit(tree)

        self.assertEqual(len(evaluator.hints), 1)
        self.assertEqual(
            evaluator.hints[0].label, ": (10, 20) | T.float16 | shared.dyn"
        )

    def test_arithmetic_shape(self):
        """测试包含算术运算的shape解析"""
        code = """
import tilelang.language as T

def test_func():
    M = 50
    K = 60
    A = T.alloc_shared((M, K), T.float16)
    B = T.alloc_shared((M * 2, K // 2), T.float16)
    """

        tree = ast.parse(code)
        evaluator = ShapeEvaluator()
        evaluator.visit(tree)

        self.assertEqual(len(evaluator.hints), 2)
        self.assertEqual(evaluator.hints[0].label, ": (50, 60) | T.float16 | shared")
        self.assertEqual(evaluator.hints[1].label, ": (100, 30) | T.float16 | shared")

    def test_tensor_shape(self):
        """测试T.Tensor的shape解析"""
        code = """
import tilelang.language as T

def test_func():
    A = T.Tensor((70, 80), T.float16)
    """

        tree = ast.parse(code)
        evaluator = ShapeEvaluator()
        evaluator.visit(tree)

        self.assertEqual(len(evaluator.hints), 1)
        self.assertEqual(evaluator.hints[0].label, ": (70, 80) | T.float16 | global")

    def test_complex_nested_shape(self):
        """测试复杂嵌套的shape解析"""
        code = """
import tilelang.language as T

def test_func():
    dim1 = 90
    dim2 = 100
    shape = (dim1, dim2, 3)
    A = T.alloc_shared(shape, T.float16)
    """

        tree = ast.parse(code)
        evaluator = ShapeEvaluator()
        evaluator.visit(tree)

        self.assertEqual(len(evaluator.hints), 1)
        self.assertEqual(
            evaluator.hints[0].label, ": (90, 100, 3) | T.float16 | shared"
        )


if __name__ == "__main__":
    unittest.main()
