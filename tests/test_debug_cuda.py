from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures"
KERNEL_SOURCE = FIXTURES / "elementwise_add.py"
DRIVER_SOURCE = FIXTURES / "cuda_debug_driver.py"
BREAKPOINT_LINE = 23
TEST_PYTHON = os.environ.get("TLSP_TEST_PYTHON")
CUDA_GDB = os.environ.get("TLSP_TEST_CUDA_GDB")
TILELANG_PATH = os.environ.get("TLSP_TEST_TILELANG")


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    paths = [str(REPO_ROOT)]
    if TILELANG_PATH:
        paths.insert(0, TILELANG_PATH)
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _output_section(output: str, name: str) -> str:
    return output.split(f"{name}_BEGIN", 1)[1].split(f"{name}_END", 1)[0]


@unittest.skipUnless(TEST_PYTHON and CUDA_GDB, "TLSP_TEST_PYTHON and TLSP_TEST_CUDA_GDB are required")
class TestCudaDebugIntegration(unittest.TestCase):
    def test_python_pending_breakpoint_and_numerics(self):
        self.assertIn("C_local[local_y, local_x]", KERNEL_SOURCE.read_text().splitlines()[BREAKPOINT_LINE - 1])
        with tempfile.TemporaryDirectory() as tmp:
            debug_dir = Path(tmp) / "debug"
            command = [
                CUDA_GDB,
                "-q",
                "-batch",
                "-ex",
                "set pagination off",
                "-ex",
                "set confirm off",
                "-ex",
                "set breakpoint pending on",
                "-ex",
                f"break {KERNEL_SOURCE}:{BREAKPOINT_LINE}",
                "-ex",
                "run",
                "-ex",
                'printf "TLSP_LINE_INFO_BEGIN\\n"',
                "-ex",
                "info line",
                "-ex",
                'printf "TLSP_LINE_INFO_END\\n"',
                "-ex",
                'printf "TLSP_CUDA_KERNELS_BEGIN\\n"',
                "-ex",
                "info cuda kernels",
                "-ex",
                'printf "TLSP_CUDA_KERNELS_END\\n"',
                "-ex",
                'printf "TLSP_THREAD_IDX_BEGIN\\n"',
                "-ex",
                "print threadIdx.x",
                "-ex",
                'printf "TLSP_THREAD_IDX_END\\n"',
                "-ex",
                "next",
                "-ex",
                'printf "TLSP_STEP_INFO_BEGIN\\n"',
                "-ex",
                "info line",
                "-ex",
                'printf "TLSP_STEP_INFO_END\\n"',
                "-ex",
                "disable",
                "-ex",
                "continue",
                "--args",
                TEST_PYTHON,
                "-m",
                "tlsp.debug",
                "--debug-dir",
                str(debug_dir),
                "--",
                str(DRIVER_SOURCE),
            ]
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                env=_child_env(),
                cwd=REPO_ROOT,
                timeout=300,
            )
            output = proc.stdout + proc.stderr
            details = output[-12000:]
            self.assertEqual(proc.returncode, 0, details)
            self.assertIn(f"elementwise_add.py:{BREAKPOINT_LINE}", output, details)

            line_info = _output_section(output, "TLSP_LINE_INFO")
            self.assertIn(f'Line {BREAKPOINT_LINE} of "', line_info, details)
            self.assertIn(str(KERNEL_SOURCE), line_info, details)
            kernels = _output_section(output, "TLSP_CUDA_KERNELS")
            self.assertIn("elementwise_add", kernels, details)
            thread_idx = _output_section(output, "TLSP_THREAD_IDX")
            self.assertRegex(thread_idx, r"\$\d+ = \d+", details)
            step_info = _output_section(output, "TLSP_STEP_INFO")
            self.assertIn(f'Line {BREAKPOINT_LINE + 1} of "', step_info, details)
            self.assertIn(str(KERNEL_SOURCE), step_info, details)
            self.assertIn("TLSP_CUDA_DEBUG_NUMERICS_OK", output, details)

            generated = debug_dir / "generated"
            sources = [*generated.rglob("*.c"), *generated.rglob("*.cu")]
            self.assertTrue(sources, f"no generated CUDA sources under {generated}\n{details}")
            generated_text = "\n".join(source.read_text() for source in sources)
            self.assertIn(f'#line {BREAKPOINT_LINE} "{KERNEL_SOURCE}"', generated_text, details)


if __name__ == "__main__":
    unittest.main()
