import ast
import unittest

from tlsp.config import (
    find_kernel_defs,
    harvest_call_kwargs,
    kernel_symbols,
    parse_directives,
    resolve_config,
    signature_defaults,
)


def _cfg(body: str):
    """resolve_config with the standard ``import tilelang`` preamble."""
    return resolve_config("import tilelang\n" + body)


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
        directives, _ = parse_directives('s = "# <typecheck> M=1"\n')
        self.assertEqual(directives, [])

    def test_bad_directive_warns_not_raises(self):
        directives, warns = parse_directives("# <typecheck> M==oops(\n")
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
        cfg = _cfg(
            "# <typecheck> block_M=256\n"
            "@tilelang.jit\n"
            "def matmul(A, block_M, block_N=64): ...\n"
            "def main():\n"
            "    matmul.compile(M=1024, block_M=128)\n"
        )
        self.assertEqual(cfg.kernel, "matmul")
        self.assertEqual(cfg.origin, "directive")
        self.assertEqual(cfg.kwargs["block_M"], 256)  # directive wins
        self.assertEqual(cfg.kwargs["M"], 1024)  # from call
        self.assertEqual(cfg.kwargs["block_N"], 64)  # from defaults

    def test_kernel_selection_by_directive(self):
        cfg = _cfg("@tilelang.jit\ndef k1(A): ...\n# <typecheck> kernel=k2, M=16\n@tilelang.jit\ndef k2(A, M): ...\n")
        self.assertEqual(cfg.kernel, "k2")
        self.assertEqual(cfg.kwargs, {"M": 16})

    def test_defaults_only(self):
        cfg = _cfg("@tilelang.jit\ndef k(A, block_M=128): ...\n")
        self.assertEqual(cfg.kwargs, {"block_M": 128})
        self.assertEqual(cfg.origin, "defaults")

    def test_no_kernel(self):
        cfg = resolve_config("x = 1\n")
        self.assertIsNone(cfg.kernel)
        self.assertEqual(cfg.origin, "none")

    def test_syntax_error_still_reads_directives(self):
        cfg = resolve_config("# <typecheck> kernel=matmul, M=1024\ndef matmul(A, M  # broken mid-edit\n")
        self.assertEqual(cfg.kernel, "matmul")
        self.assertEqual(cfg.kwargs, {"M": 1024})
        self.assertEqual(cfg.origin, "directive")

    def test_syntax_error_still_reads_directive_target(self):
        cfg = resolve_config("# <typecheck> target=c\ndef k(A  # broken mid-edit\n")
        self.assertEqual(cfg.target, "c")

    # -- analysis target resolution (directive > call site > decorator) ---------

    def test_directive_target_pulled_out_of_kwargs(self):
        cfg = _cfg("# <typecheck> kernel=k, target=c, M=16\n@tilelang.jit\ndef k(A, M): ...\n")
        self.assertEqual(cfg.target, "c")
        self.assertNotIn("target", cfg.kwargs)

    def test_decorator_target(self):
        self.assertEqual(_cfg("@tilelang.jit(target='c')\ndef k(A): ...\n").target, "c")

    def test_directive_target_beats_decorator(self):
        self.assertEqual(_cfg("# <typecheck> target=llvm\n@tilelang.jit(target='c')\ndef k(A): ...\n").target, "llvm")

    def test_directive_target_beats_call_site(self):
        src = "# <typecheck> target=llvm\n@tilelang.jit\ndef k(A): ...\ndef main():\n    k(target='c')\n"
        self.assertEqual(_cfg(src).target, "llvm")

    def test_compile_call_target_steers_not_poisons(self):
        cfg = _cfg("@tilelang.jit\ndef k(A, M): ...\ndef main():\n    k.compile(M=16, target='c')\n")
        self.assertEqual(cfg.target, "c")
        self.assertNotIn("target", cfg.kwargs)

    def test_multi_directive_target_not_shadowed(self):
        # A nearer target-less directive must not hide an earlier target= one.
        cfg = _cfg("# <typecheck> target=c\n# <typecheck> M=16\n@tilelang.jit\ndef k(A, M): ...\n")
        self.assertEqual(cfg.target, "c")
        self.assertEqual(cfg.kwargs, {"M": 16})

    def test_last_directive_target_wins(self):
        self.assertEqual(_cfg("# <typecheck> target=c\n# <typecheck> target=llvm\n@tilelang.jit\ndef k(A): ...\n").target, "llvm")

    def test_target_defaults_to_none(self):
        self.assertIsNone(_cfg("@tilelang.jit\ndef k(A): ...\n").target)

    def test_non_string_directive_target_warns(self):
        cfg = _cfg("# <typecheck> target=123\n@tilelang.jit\ndef k(A): ...\n")
        self.assertIsNone(cfg.target)
        self.assertTrue(any("target" in w for w in cfg.warnings))


if __name__ == "__main__":
    unittest.main()
