import unittest

from tlsp.config import (
    find_kernel_defs,
    harvest_call_kwargs,
    kernel_symbols,
    parse_directives,
    resolve_config,
    signature_defaults,
)
import ast


class TestDirectives(unittest.TestCase):
    def test_basic(self):
        src = "x = 1  # <typecheck> M=1024, N=1024, block_M=128\n"
        directives, warns = parse_directives(src)
        self.assertEqual(warns, [])
        self.assertEqual(len(directives), 1)
        self.assertEqual(directives[0].kwargs, {"M": 1024, "N": 1024, "block_M": 128})
        self.assertIsNone(directives[0].kernel)
        self.assertEqual(directives[0].lineno, 1)

    def test_kernel_selector_and_dtype_string(self):
        src = "# <typecheck> kernel=matmul, dtype=T.float16, name='f32'\n"
        directives, _ = parse_directives(src)
        self.assertEqual(directives[0].kernel, "matmul")
        self.assertEqual(directives[0].kwargs, {"dtype": "float16", "name": "f32"})

    def test_directive_inside_string_ignored(self):
        src = 's = "# <typecheck> M=1"\n'
        directives, _ = parse_directives(src)
        self.assertEqual(directives, [])

    def test_bad_directive_warns_not_raises(self):
        src = "# <typecheck> M==oops(\n"
        directives, warns = parse_directives(src)
        self.assertTrue(warns)


class TestKernelDefs(unittest.TestCase):
    def test_find_jit_kernels(self):
        src = "import tilelang\n@tilelang.jit\ndef k1(A): ...\n@tilelang.jit(target='c')\ndef k2(A): ...\ndef plain(): ...\n"
        defs = find_kernel_defs(ast.parse(src))
        self.assertEqual(defs, {"k1": 3, "k2": 5})

    def test_kernel_symbols(self):
        src = "import tilelang\n@tilelang.jit\ndef k1(A): ...\n"
        self.assertEqual(kernel_symbols(src), [{"name": "k1", "kind": "function", "line": 3}])

    def test_kernel_symbols_tolerates_syntax_error(self):
        self.assertEqual(kernel_symbols("def k(  # broken mid-edit\n"), [])


class TestHarvest(unittest.TestCase):
    SRC = (
        "import tilelang\n"
        "@tilelang.jit\n"
        "def matmul(A, block_M, block_N=64, dtype=T.float16): ...\n"
        "def main():\n"
        "    matmul.compile(M=1024, N=1024, block_M=128)\n"
    )

    def test_compile_harvest(self):
        tree = ast.parse(self.SRC)
        kwargs = harvest_call_kwargs(tree, "matmul")
        self.assertEqual(kwargs, {"M": 1024, "N": 1024, "block_M": 128})

    def test_signature_defaults(self):
        tree = ast.parse(self.SRC)
        self.assertEqual(
            signature_defaults(tree, "matmul"),
            {"block_N": 64, "dtype": "float16"},
        )


class TestResolveConfig(unittest.TestCase):
    def test_priority_directive_over_call_over_defaults(self):
        src = (
            "import tilelang\n"
            "# <typecheck> block_M=256\n"
            "@tilelang.jit\n"
            "def matmul(A, block_M, block_N=64): ...\n"
            "def main():\n"
            "    matmul.compile(M=1024, block_M=128)\n"
        )
        cfg = resolve_config(src)
        self.assertEqual(cfg.kernel, "matmul")
        self.assertEqual(cfg.origin, "directive")
        self.assertEqual(cfg.kwargs["block_M"], 256)  # directive wins
        self.assertEqual(cfg.kwargs["M"], 1024)  # from call
        self.assertEqual(cfg.kwargs["block_N"], 64)  # from defaults

    def test_kernel_selection_by_directive(self):
        src = "import tilelang\n@tilelang.jit\ndef k1(A): ...\n# <typecheck> kernel=k2, M=16\n@tilelang.jit\ndef k2(A, M): ...\n"
        cfg = resolve_config(src)
        self.assertEqual(cfg.kernel, "k2")
        self.assertEqual(cfg.kwargs, {"M": 16})

    def test_defaults_only(self):
        src = "import tilelang\n@tilelang.jit\ndef k(A, block_M=128): ...\n"
        cfg = resolve_config(src)
        self.assertEqual(cfg.kwargs, {"block_M": 128})
        self.assertEqual(cfg.origin, "defaults")

    def test_no_kernel(self):
        cfg = resolve_config("x = 1\n")
        self.assertIsNone(cfg.kernel)
        self.assertEqual(cfg.origin, "none")

    def test_syntax_error_still_reads_directives(self):
        src = "# <typecheck> kernel=matmul, M=1024\ndef matmul(A, M  # broken mid-edit\n"
        cfg = resolve_config(src)
        self.assertEqual(cfg.kernel, "matmul")
        self.assertEqual(cfg.kwargs, {"M": 1024})
        self.assertEqual(cfg.origin, "directive")


if __name__ == "__main__":
    unittest.main()
