"""Unit tests for the pygls front-end: position conversion, URI handling,
stale-result guard and debounce (needs pygls; skips cleanly without it)."""

import os
import tempfile
import time
import unittest

try:
    from lsprotocol.types import TextDocumentItem
    from pygls.workspace import Workspace

    from tlsp import protocol
    from tlsp.server import TileLangLanguageServer, _to_lsp_hints, _utf16_len

    HAVE_PYGLS = True
except ImportError:  # pragma: no cover
    HAVE_PYGLS = False


@unittest.skipUnless(HAVE_PYGLS, "pygls not installed")
class TestPositions(unittest.TestCase):
    def test_utf16_len(self):
        self.assertEqual(_utf16_len("abc"), 3)
        self.assertEqual(_utf16_len("中"), 1)  # BMP: one UTF-16 code unit
        self.assertEqual(_utf16_len("🙂"), 2)  # astral: surrogate pair

    def test_hint_position_excludes_newline(self):
        # pygls document.lines keeps the trailing '\n'.
        result = protocol.make_result(
            mode=protocol.MODE_STATIC,
            hints=[protocol.make_hint(1, "alloc", "x")])
        hints = _to_lsp_hints(result, ["abc\n"])
        self.assertEqual(hints[0].position.character, 3)

    def test_hint_range_filter(self):
        result = protocol.make_result(
            mode=protocol.MODE_STATIC,
            hints=[protocol.make_hint(1, "alloc", "a"),
                   protocol.make_hint(5, "alloc", "b")])
        hints = _to_lsp_hints(result, ["x\n"] * 5, visible=(0, 2))
        self.assertEqual([h.label for h in hints], ["a"])


@unittest.skipUnless(HAVE_PYGLS, "pygls not installed")
class TestServerGuards(unittest.TestCase):
    TILE_SRC = "import tilelang.language as T\nX = 1\n"

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        os.environ["TLSP_CACHE_DIR"] = self._tmp.name
        self.addCleanup(os.environ.pop, "TLSP_CACHE_DIR", None)
        self.addCleanup(self._tmp.cleanup)
        self.tlsp = TileLangLanguageServer(target="c")
        # pygls exposes the workspace only after initialize; inject one.
        self.tlsp.server.protocol._workspace = Workspace(None)
        # No real subprocesses / background consumers in unit tests.
        self.tlsp.analyzer.start = lambda: False
        self.tlsp._ensure_bg = lambda: None

    def tearDown(self):
        for timer in self.tlsp._debounce_timers.values():
            timer.cancel()

    def _put_doc(self, uri, source, version=1):
        self.tlsp.server.workspace.put_text_document(
            TextDocumentItem(uri=uri, language_id="python",
                             version=version, text=source))

    def test_schedule_uses_decoded_fs_path(self):
        uri = "file:///Users/t%20est/%E4%B8%AD%E6%96%87.py"
        self._put_doc(uri, self.TILE_SRC)
        self.tlsp.schedule_full_analysis(uri)
        _uri, path, _text = self.tlsp._bg_queue.get_nowait()
        self.assertEqual(path, "/Users/t est/中文.py")

    def test_schedule_skips_plain_python(self):
        uri = "file:///plain.py"
        self._put_doc(uri, "def f():\n    pass\n")
        self.tlsp.schedule_full_analysis(uri)
        self.assertTrue(self.tlsp._bg_queue.empty())

    def test_stale_full_result_not_published(self):
        uri = "file:///stale.py"
        path = "/stale.py"
        old_text = self.TILE_SRC
        self._put_doc(uri, old_text, version=1)
        fake = protocol.make_result(
            mode=protocol.MODE_FULL,
            hints=[protocol.make_hint(1, "alloc", "x")])
        self.tlsp.analyzer.analyze = lambda *a, **k: fake
        published = []
        self.tlsp._publish = lambda u, r: published.append((u, r))

        self.tlsp._full_analysis(uri, path, old_text)
        self.assertEqual(len(published), 1)

        # The user edited the document while the analysis was in flight.
        self._put_doc(uri, old_text + "Y = 2\n", version=2)
        self.tlsp._full_analysis(uri, path, old_text)
        self.assertEqual(len(published), 1)  # stale result discarded

    def test_debounce_latest_wins(self):
        uri = "file:///deb.py"
        self._put_doc(uri, self.TILE_SRC)
        calls = []
        self.tlsp._static_refresh = lambda u: calls.append(u)
        for _ in range(3):
            self.tlsp._debounce_static(uri)
        time.sleep(0.8)
        self.assertEqual(calls, [uri])


if __name__ == "__main__":
    unittest.main()
