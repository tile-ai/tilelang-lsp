"""Regression tests for tlsp.pipeline.load_user_module (pure stdlib)."""

import os
import sys
import tempfile
import unittest

from tlsp.pipeline import load_user_module


class TestLoadUserModule(unittest.TestCase):
    def _write(self, dirpath, name, text):
        path = os.path.join(dirpath, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path

    def test_same_size_same_second_edit_reloads(self):
        """A quick same-length save must never analyze stale bytecode.

        The import loader's timestamp .pyc only compares second-level mtime
        and file size; load_user_module compiles from source instead.
        """
        with tempfile.TemporaryDirectory() as d:
            path = self._write(d, "mod.py", "VALUE = 1\n")
            mtime = os.stat(path).st_mtime
            self.assertEqual(load_user_module(path).VALUE, 1)

            self._write(d, "mod.py", "VALUE = 2\n")  # same length
            os.utime(path, (mtime, mtime))  # same second: defeats timestamp pyc
            self.assertEqual(load_user_module(path).VALUE, 2)

    def test_module_removed_from_sys_modules(self):
        with tempfile.TemporaryDirectory() as d:
            path = self._write(d, "mod.py", "X = 1\n")
            load_user_module(path)
            self.assertNotIn("_tlsp_user_kernel", sys.modules)

    def test_dataclass_with_dotted_classvar_annotation(self):
        # dataclasses resolves dotted ClassVar via sys.modules[cls.__module__].
        with tempfile.TemporaryDirectory() as d:
            path = self._write(
                d,
                "mod.py",
                "from __future__ import annotations\n"
                "import dataclasses\n"
                "import typing\n"
                "@dataclasses.dataclass\n"
                "class C:\n"
                "    x: typing.ClassVar[int] = 7\n",
            )
            module = load_user_module(path)
            self.assertEqual(module.C.x, 7)

    def test_syntax_error_raises(self):
        with tempfile.TemporaryDirectory() as d:
            path = self._write(d, "mod.py", "def broken(:\n")
            with self.assertRaises(SyntaxError):
                load_user_module(path)


if __name__ == "__main__":
    unittest.main()
