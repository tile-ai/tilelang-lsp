import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tlsp import debug as tlsp_debug


class TestCompileFlags(unittest.TestCase):
    def test_none(self):
        self.assertEqual(tlsp_debug.normalize_compile_flags(None), [])

    def test_string_remains_one_raw_group(self):
        self.assertEqual(tlsp_debug.normalize_compile_flags("-O3 -DDEBUG"), ["-O3 -DDEBUG"])

    def test_sequence_is_copied_without_rewriting(self):
        flags = ["-O3", '-DFOO="hello world"']
        normalized = tlsp_debug.normalize_compile_flags(flags)
        self.assertEqual(normalized, flags)
        self.assertIsNot(normalized, flags)


class TestMergeDebugOptions(unittest.TestCase):
    def _merge(self, kwargs, **overrides):
        options = {
            "device_debug": True,
            "generated_dir": "/tmp/tlsp-debug/generated",
        }
        options.update(overrides)
        keys = {
            "TL_EMIT_LINE_DIRECTIVES": "tl.emit_line_directives",
            "CUDA_KERNELS_OUTPUT_DIR": "cuda.kernels_output_dir",
        }
        with mock.patch.object(tlsp_debug, "_pass_config_key", side_effect=keys.get):
            return tlsp_debug.merge_debug_options(kwargs, **options)

    def test_adds_line_directives_device_debug_and_output_dirs(self):
        merged = self._merge({})
        self.assertEqual(
            merged["pass_configs"],
            {
                "tl.emit_line_directives": True,
                "cuda.kernels_output_dir": "/tmp/tlsp-debug/generated",
            },
        )
        self.assertEqual(merged["compile_flags"], ["-G"])
        self.assertEqual(merged["debug_root_path"], "/tmp/tlsp-debug/generated")

    def test_preserves_options_but_owns_debug_output_dirs(self):
        merged = self._merge(
            {
                "target": "cuda",
                "pass_configs": {"tl.enable_fast_math": True, "cuda.kernels_output_dir": "/custom"},
                "compile_flags": "-O3",
                "debug_root_path": "/custom",
            }
        )
        self.assertEqual(merged["target"], "cuda")
        self.assertEqual(
            merged["pass_configs"],
            {
                "tl.enable_fast_math": True,
                "tl.emit_line_directives": True,
                "cuda.kernels_output_dir": "/tmp/tlsp-debug/generated",
            },
        )
        self.assertEqual(merged["compile_flags"], ["-O3", "-G"])
        self.assertEqual(merged["debug_root_path"], "/tmp/tlsp-debug/generated")

    def test_raw_string_flags_keep_quotes_and_spaces(self):
        raw = '-DFOO="hello world" -I"/tmp/include path"'
        merged = self._merge({"compile_flags": raw})
        self.assertEqual(merged["compile_flags"], [raw, "-G"])

    def test_does_not_duplicate_device_debug_flag(self):
        merged = self._merge({"compile_flags": ["-G"]})
        self.assertEqual(merged["compile_flags"], ["-G"])
        merged = self._merge({"compile_flags": "--device-debug -O3"})
        self.assertEqual(merged["compile_flags"], "--device-debug -O3")

    def test_device_debug_off_adds_lineinfo_without_splitting_flags(self):
        raw = '-O3 -DFOO="hello world"'
        merged = self._merge({"compile_flags": raw}, device_debug=False)
        self.assertEqual(merged["compile_flags"], [raw, "-lineinfo"])
        self.assertEqual(merged["debug_root_path"], "/tmp/tlsp-debug/generated")

        existing = self._merge({"compile_flags": "-O3 -lineinfo"}, device_debug=False)
        self.assertEqual(existing["compile_flags"], "-O3 -lineinfo")

        without_dump = self._merge({}, device_debug=False, allow_debug_root_path=False)
        self.assertNotIn("debug_root_path", without_dump)
        self.assertEqual(without_dump["compile_flags"], ["-lineinfo"])

    def test_line_directives_off_does_not_add_mapping_options(self):
        merged = self._merge({}, line_directives=False, device_debug=False)
        self.assertNotIn("tl.emit_line_directives", merged["pass_configs"])
        self.assertNotIn("compile_flags", merged)

    def test_compile_call_has_no_debug_root_path(self):
        merged = self._merge({}, allow_debug_root_path=False)
        self.assertNotIn("debug_root_path", merged)
        self.assertEqual(merged["pass_configs"]["cuda.kernels_output_dir"], "/tmp/tlsp-debug/generated")

    def test_kwargs_are_not_mutated(self):
        kwargs = {"pass_configs": {"existing": True}, "compile_flags": ["-O3"]}
        original = {"pass_configs": dict(kwargs["pass_configs"]), "compile_flags": list(kwargs["compile_flags"])}
        self._merge(kwargs)
        self.assertEqual(kwargs, original)

    def test_unknown_line_directive_key_only_skips_line_mapping(self):
        with mock.patch.object(tlsp_debug, "line_directive_key", return_value=None), mock.patch.object(
            tlsp_debug, "cuda_output_dir_key", return_value="cuda.kernels_output_dir"
        ):
            merged = tlsp_debug.merge_debug_options({}, device_debug=True, generated_dir="/tmp/x")
        self.assertEqual(merged["pass_configs"], {"cuda.kernels_output_dir": "/tmp/x"})
        self.assertEqual(merged["compile_flags"], ["-G"])


class TestRunnerArguments(unittest.TestCase):
    def test_script_and_arguments_after_separator(self):
        args = tlsp_debug.parse_runner_args(["--debug-dir", "out", "--", "kernel.py", "--foo", "bar"])
        self.assertEqual(args.debug_dir, "out")
        self.assertEqual(args.script, "kernel.py")
        self.assertEqual(args.args, ["--foo", "bar"])

    def test_script_arguments_without_separator(self):
        args = tlsp_debug.parse_runner_args(["kernel.py", "--foo"])
        self.assertEqual(args.script, "kernel.py")
        self.assertEqual(args.args, ["--foo"])

    def test_switches(self):
        args = tlsp_debug.parse_runner_args(["--no-device-debug", "--no-line-directives", "--keep-cache", "--", "kernel.py"])
        self.assertTrue(args.no_device_debug)
        self.assertTrue(args.no_line_directives)
        self.assertTrue(args.keep_cache)

    def test_missing_script_is_an_error(self):
        with self.assertRaises(SystemExit):
            tlsp_debug.parse_runner_args(["--debug-dir", "out"])


class TestDebugRunnerEndToEnd(unittest.TestCase):
    def _write_fake_project(self, root: Path, *, line_directives: bool = True) -> Path:
        package = root / "tilelang"
        package.mkdir()
        (package / "__init__.py").write_text("from .jit import jit, compile\n")
        (package / "jit.py").write_text(
            "def jit(*args, **kwargs):\n"
            "    def decorate(func):\n"
            "        return {'kind': 'jit', 'func': func.__name__, 'kwargs': kwargs}\n"
            "    if args and callable(args[0]):\n"
            "        return decorate(args[0])\n"
            "    return decorate\n\n"
            "def compile(*args, **kwargs):\n"
            "    return {'kind': 'compile', 'kwargs': kwargs}\n"
        )
        line_key = '    TL_EMIT_LINE_DIRECTIVES = "tl.emit_line_directives"\n' if line_directives else ""
        (package / "transform.py").write_text(
            "class PassConfigKey:\n" + line_key + '    CUDA_KERNELS_OUTPUT_DIR = "cuda.kernels_output_dir"\n'
        )
        script = root / "probe.py"
        script.write_text(
            "import os\n"
            "import sys\n"
            "import tilelang\n"
            "@tilelang.jit(pass_configs={'keep': 1}, compile_flags='-DFOO=\"hello world\"')\n"
            "def kernel():\n"
            "    pass\n"
            "print('JIT', kernel)\n"
            "print('COMPILE', tilelang.compile('f', compile_flags=['-O3']))\n"
            "print('ARGV', sys.argv)\n"
            "print('ENV', os.environ.get('TILELANG_ENABLE_IR_SPAN'), os.environ.get('TILELANG_DISABLE_CACHE'))\n"
        )
        return script

    def _run_launcher(self, root: Path, script: Path, debug_dir: Path, *runner_args: str) -> subprocess.CompletedProcess:
        repo_root = Path(__file__).resolve().parents[1]
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(root), str(repo_root)])
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["TILELANG_ENABLE_IR_SPAN"] = "0"
        env["TILELANG_DISABLE_CACHE"] = "0"
        return subprocess.run(
            [sys.executable, "-m", "tlsp.debug", "--debug-dir", str(debug_dir), *runner_args, "--", str(script), "--flag"],
            capture_output=True,
            text=True,
            env=env,
            cwd=root,
            timeout=120,
        )

    def test_launcher_patches_fake_tilelang_and_fixes_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = self._write_fake_project(root)
            debug_dir = (root / "debug").resolve()
            proc = self._run_launcher(root, script, debug_dir)
            self.assertEqual(proc.returncode, 0, proc.stderr)

            lines = proc.stdout.splitlines()
            jit_line = next(line for line in lines if line.startswith("JIT "))
            compile_line = next(line for line in lines if line.startswith("COMPILE "))
            generated_dir = str(debug_dir / "generated")
            self.assertIn("'func': 'kernel'", jit_line)
            self.assertIn(generated_dir, jit_line)
            self.assertIn("'-G'", jit_line)
            self.assertIn("'keep': 1", jit_line)
            self.assertIn("'tl.emit_line_directives': True", jit_line)
            self.assertIn("'cuda.kernels_output_dir'", jit_line)
            self.assertIn("'-DFOO=\"hello world\"'", jit_line)
            self.assertIn("'-G'", compile_line)
            self.assertIn("'cuda.kernels_output_dir'", compile_line)
            self.assertNotIn("debug_root_path", compile_line)
            self.assertIn(f"ARGV ['{script.resolve()}', '--flag']", proc.stdout)
            self.assertIn("ENV 1 1", proc.stdout)
            self.assertTrue((debug_dir / "generated").is_dir())

    def test_missing_line_mapping_fails_unless_explicitly_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = self._write_fake_project(root, line_directives=False)
            debug_dir = (root / "debug").resolve()

            failed = self._run_launcher(root, script, debug_dir)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("requires TileLang >= 0.1.15", failed.stderr)
            self.assertIn("--no-line-directives", failed.stderr)
            self.assertNotIn("JIT", failed.stdout)

            fallback = self._run_launcher(root, script, debug_dir, "--no-line-directives")
            self.assertEqual(fallback.returncode, 0, fallback.stderr)
            self.assertIn("Python line mapping: off", fallback.stderr)
            self.assertIn("JIT", fallback.stdout)
            self.assertNotIn("tl.emit_line_directives", fallback.stdout)


if __name__ == "__main__":
    unittest.main()
