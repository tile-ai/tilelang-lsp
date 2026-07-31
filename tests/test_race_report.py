"""Unit tests for tlsp.race_report: stderr capture + aggregated warning parsing."""

import os
import unittest

from tlsp import protocol
from tlsp.race_report import capture_os_stderr, race_diagnostics

_THIS_FILE = os.path.abspath(__file__)

# Mirrors the real aggregated block emitted by verify_parallel_loop.cc.
SAMPLE_TWO_RACES = """\
[01:33:16] /tilelang/src/transform/verify_parallel_loop.cc:146: Warning: Data race detected: 2 potential race(s)
  [1] `C_local(0, 0)` is written by multiple threads in loop (local_y, local_x)
  --> {path}:22:1
  Example:
  A_shared[local_y, local_x] == A_shared[local_y_OTHER_, local_x_OTHER_]$1 = false
  bx = 0
  [2] `out_a(0)` is written by multiple threads in loop (ti)
  --> /somewhere/else/kernel.py:30:1
  Example:
  ti = 0
If you believe this is a false positive, pass `PassKey.TL_DISABLE_DATA_RACE_CHECK` to pass key to disable this check.
"""

SAMPLE_NO_SPAN = """\
Warning: Data race detected: 1 potential race(s)
  [1] `buf(0)` is written by multiple threads in loop (i)
  Example:
  i = 0
If you believe this is a false positive, pass `PassKey.TL_DISABLE_DATA_RACE_CHECK` to pass key to disable this check.
"""

# Pre-aggregation format (before tilelang commit bdb769a): no numbered entries,
# no span marker. Not parsed — stays invisible as before.
SAMPLE_LEGACY = """\
Warning: Data race detected: `buf(0)` is written by multiple threads in loop [i], Example:
i = 0
If you believe this is a false positive, pass `PassKey.TL_DISABLE_DATA_RACE_CHECK` to pass key to disable this check.
"""


class TestRaceDiagnostics(unittest.TestCase):
    def test_two_races_split_and_located(self):
        diags = race_diagnostics(SAMPLE_TWO_RACES.format(path=_THIS_FILE), _THIS_FILE)
        self.assertEqual(len(diags), 2)
        for d in diags:
            self.assertEqual(d["severity"], protocol.SEVERITY_WARNING)
            self.assertEqual(d["source"], "tilelang")
        self.assertEqual(diags[0]["line"], 22)
        self.assertIn("Data race: `C_local(0, 0)`", diags[0]["message"])
        self.assertIn("local_y, local_x", diags[0]["message"])
        # Second entry's marker points at another file: falls back to line 1.
        self.assertEqual(diags[1]["line"], 1)
        self.assertIn("out_a(0)", diags[1]["message"])

    def test_entry_without_span_lands_on_line_1(self):
        diags = race_diagnostics(SAMPLE_NO_SPAN, _THIS_FILE)
        self.assertEqual(len(diags), 1)
        self.assertEqual(diags[0]["line"], 1)
        self.assertEqual(diags[0]["severity"], protocol.SEVERITY_WARNING)

    def test_legacy_format_is_ignored(self):
        self.assertEqual(race_diagnostics(SAMPLE_LEGACY, _THIS_FILE), [])

    def test_unrelated_stderr_is_ignored(self):
        text = "Loading tilelang libs from dev root\nsome unrelated warning\n"
        self.assertEqual(race_diagnostics(text, _THIS_FILE), [])
        self.assertEqual(race_diagnostics("", _THIS_FILE), [])


class TestCaptureOsStderr(unittest.TestCase):
    def test_captures_fd_level_writes(self):
        with capture_os_stderr() as chunks:
            os.write(2, b"line-a\n")
            os.write(2, b"line-b\n")
        self.assertEqual(b"".join(chunks), b"line-a\nline-b\n")

    def test_drainer_keeps_large_output_unblocked(self):
        blob = b"x" * 200_000 + b"\n"
        with capture_os_stderr() as chunks:
            view = memoryview(blob)
            while view:
                view = view[os.write(2, view) :]
        self.assertEqual(b"".join(chunks), blob)


if __name__ == "__main__":
    unittest.main()
