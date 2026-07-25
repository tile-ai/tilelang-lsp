import unittest

from tlsp.markers import parse_location_marker, strip_marker


class TestParseLocationMarker(unittest.TestCase):
    def test_basic(self):
        msg = "Check failed: layout inconsistent\n  --> /abs/path/kernel.py:21:1"
        self.assertEqual(
            parse_location_marker(msg), ("/abs/path/kernel.py", 21, 1)
        )

    def test_no_col_defaults_1(self):
        msg = "error\n  --> /abs/kernel.py:7"
        self.assertEqual(parse_location_marker(msg), ("/abs/kernel.py", 7, 1))

    def test_path_with_spaces(self):
        msg = "error\n  --> /path with spaces/my kernel.py:21:3"
        self.assertEqual(
            parse_location_marker(msg), ("/path with spaces/my kernel.py", 21, 3)
        )

    def test_windows_drive_letter(self):
        msg = "error\n  --> C:\\work\\kernel.py:21:1"
        self.assertEqual(
            parse_location_marker(msg), ("C:\\work\\kernel.py", 21, 1)
        )

    def test_no_marker(self):
        self.assertIsNone(parse_location_marker("plain error message"))
        self.assertIsNone(parse_location_marker(""))

    def test_enriched_snippet_form(self):
        # enrich_error replaces the bare marker with a rendered snippet; the
        # header line still matches.
        msg = (
            "Check failed: bad layout\n"
            "  --> /abs/kernel.py:21:1\n"
            "   |\n"
            "21 |     C_local = T.alloc_fragment((128, 128), accum_dtype)\n"
            "   |     ^"
        )
        self.assertEqual(parse_location_marker(msg), ("/abs/kernel.py", 21, 1))


class TestStripMarker(unittest.TestCase):
    def test_strip_bare(self):
        msg = "Check failed: bad layout\n  --> /abs/kernel.py:21:1"
        self.assertEqual(strip_marker(msg), "Check failed: bad layout")

    def test_strip_snippet(self):
        msg = (
            "Check failed: bad layout\n"
            "  --> /abs/kernel.py:21:1\n"
            "   |\n"
            "21 |     C_local = ...\n"
            "   |     ^"
        )
        self.assertEqual(strip_marker(msg), "Check failed: bad layout")

    def test_strip_keeps_message_when_no_marker(self):
        self.assertEqual(strip_marker("boom"), "boom")


if __name__ == "__main__":
    unittest.main()
