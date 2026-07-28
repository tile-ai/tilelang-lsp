import unittest

from tlsp.static_fallback import analyze_static


def labels(result):
    return [h["label"] for h in result.hints]


class TestStaticAlloc(unittest.TestCase):
    def test_basic_static_shape(self):
        code = """
import tilelang.language as T
def test_func():
    A = T.alloc_shared((10, 20), T.float16)
    B = T.alloc_local((5,), T.int32)
"""
        result = analyze_static(code)
        self.assertEqual(len(result.hints), 2)
        self.assertEqual(result.hints[0]["label"], "⟨(10, 20) float16 @shared.dyn⟩")
        self.assertEqual(result.hints[0]["kind"], "alloc")
        self.assertEqual(result.hints[0]["buffer"], "A")
        self.assertEqual(result.hints[1]["label"], "⟨(5,) int32 @local⟩")

    def test_alloc_var(self):
        code = """
import tilelang.language as T
def test_func():
    v1 = T.alloc_var(T.int32, 1)
    v2 = T.alloc_var(T.float32, "local.custom")
"""
        result = analyze_static(code)
        self.assertEqual(len(result.hints), 2)
        self.assertEqual(result.hints[0]["label"], "⟨(1,) int32 @local.var⟩")
        self.assertEqual(result.hints[1]["label"], "⟨(1,) float32 @local.custom⟩")

    def test_alloc_barrier(self):
        code = """
import tilelang.language as T
def test_func():
    b = T.alloc_barrier(4)
"""
        result = analyze_static(code)
        self.assertEqual(len(result.hints), 1)
        self.assertEqual(result.hints[0]["label"], "⟨(4,) uint64 @shared.barrier⟩")

    def test_alloc_descriptor(self):
        code = """
import tilelang.language as T
def test_func():
    d1 = T.alloc_descriptor("tcgen05_smem")
    d2 = T.alloc_wgmma_desc()
"""
        result = analyze_static(code)
        self.assertEqual(len(result.hints), 2)
        self.assertEqual(result.hints[0]["label"], "⟨(1,) @local.descriptor.tcgen05_smem⟩")
        self.assertEqual(result.hints[1]["label"], "⟨(1,) @local.descriptor.wgmma⟩")

    def test_alloc_tmem(self):
        code = """
import tilelang.language as T
def test_func():
    t = T.alloc_tmem((10, 10), T.float16)
"""
        result = analyze_static(code)
        self.assertEqual(len(result.hints), 1)
        self.assertEqual(result.hints[0]["label"], "⟨(10, 10) float16 @shared.tmem⟩")

    def test_hint_line_is_alloc_line(self):
        code = "import tilelang.language as T\ndef test_func():\n    A = T.alloc_shared((10, 20), T.float16)\n"
        result = analyze_static(code)
        self.assertEqual(result.hints[0]["line"], 3)


class TestStaticCopyAndSlicing(unittest.TestCase):
    def test_slicing(self):
        code = """
import tilelang.language as T
def test_func():
    A = T.alloc_shared((10, 20), T.float16)
    T.copy(A[0, :], A[1, :])
"""
        result = analyze_static(code)
        self.assertEqual(len(result.hints), 2)
        self.assertEqual(result.hints[1]["kind"], "op")
        self.assertIn("copy", result.hints[1]["label"])
        self.assertIn("(20,)", result.hints[1]["label"])

    def test_slicing_dimension_reduction(self):
        code = """
import tilelang.language as T
def test_func():
    A = T.alloc_shared((10, 20, 30), T.float16)
    B = T.alloc_local((20, 30), T.float16)
    T.copy(A[0], B)
"""
        result = analyze_static(code)
        self.assertEqual(len(result.hints), 3)
        self.assertEqual(
            result.hints[2]["label"],
            "copy ⟨(20, 30) float16 @shared.dyn⟩ ⇒ ⟨(20, 30) float16 @local⟩",
        )

    def test_slicing_complex(self):
        code = """
import tilelang.language as T
def test_func():
    A = T.alloc_shared((10, 20), T.float16)
    B = T.alloc_local((5, 20), T.float16)
    T.copy(A[0:5, :], B)
"""
        result = analyze_static(code)
        self.assertEqual(len(result.hints), 3)
        self.assertIn("(5, 20)", result.hints[2]["label"])

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
        result = analyze_static(code)
        copy_hints = [h for h in result.hints if h["kind"] == "op"]
        self.assertEqual(len(copy_hints), 2)
        self.assertIn("ptr", copy_hints[0]["label"])
        self.assertIn("ptr", copy_hints[1]["label"])


class TestStaticDiagnostics(unittest.TestCase):
    def test_shape_mismatch_is_diagnostic(self):
        code = """
import tilelang.language as T
def test_func():
    A = T.alloc_shared((2, 1), T.float16)
    B = T.alloc_shared((3, 1), T.float16)
    C = A + B
"""
        result = analyze_static(code)
        warnings = [d for d in result.diagnostics if d["severity"] == "warning"]
        self.assertEqual(len(warnings), 1)
        self.assertIn("shape mismatch", warnings[0]["message"])
        self.assertIn("(2, 1)", warnings[0]["message"])
        self.assertIn("(3, 1)", warnings[0]["message"])
        self.assertEqual(warnings[0]["line"], 6)
        self.assertEqual(warnings[0]["source"], "tilelang-static")

    def test_shape_broadcast_ok(self):
        code = """
import tilelang.language as T
def test_func():
    A = T.alloc_shared((2, 3), T.float16)
    B = T.alloc_shared((3,), T.float16)
    C = A + B
"""
        result = analyze_static(code)
        self.assertEqual(result.diagnostics, [])


class TestStaticScope(unittest.TestCase):
    def test_generic_python_vars_produce_no_hints(self):
        code = """
import tilelang.language as T
def test_func():
    x = 30
    name = "hello"
    y = x + 1
    items = [1, 2, 3]
"""
        result = analyze_static(code)
        self.assertEqual(result.hints, [])
        self.assertEqual(result.diagnostics, [])

    def test_annassign_param_style(self):
        code = """
import tilelang.language as T
def kernel(M, dtype):
    A: T.Tensor((M, 64), dtype)
    B = T.empty((M, 64), dtype)
"""
        result = analyze_static(code)
        kinds = {h["kind"] for h in result.hints}
        self.assertEqual(kinds, {"param"})
        self.assertEqual(result.index.params.get("A"), 4)
        self.assertEqual(result.index.params.get("B"), 5)
        self.assertEqual(result.hints[0]["label"], "⟨(M, 64) dtype @global⟩")

    def test_signature_tensor_annotation(self):
        code = """
import tilelang.language as T
@T.prim_func
def main(A: T.Tensor((128,), T.float16)):
    pass
"""
        result = analyze_static(code)
        self.assertEqual(len(result.hints), 1)
        self.assertEqual(result.hints[0]["kind"], "param")
        self.assertEqual(result.hints[0]["label"], "⟨(128,) float16 @global⟩")

    def test_syntax_error_fallback(self):
        code = (
            "import tilelang.language as T\n"
            "def test_func():\n"
            "    A = T.alloc_shared((10, 20), T.float16)\n"
            "    def broken(:\n"
            "    B = T.alloc_local((5,), T.int32)\n"
        )
        result = analyze_static(code)
        got = labels(result)
        self.assertIn("⟨(10, 20) float16 @shared.dyn⟩", got)


if __name__ == "__main__":
    unittest.main()
