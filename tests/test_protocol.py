import io
import json
import os
import tempfile
import time
import unittest

from tlsp import protocol
from tlsp.worker import AnalyzerClient, ResultCache


class TestNdjsonProtocol(unittest.TestCase):
    def test_roundtrip(self):
        buf = io.StringIO()
        obj = {"id": 1, "method": "analyze", "params": {"path": "/tmp/x.py", "中文": True}}
        protocol.write_message(obj, stream=buf)
        buf.seek(0)
        self.assertEqual(protocol.read_message(stream=buf), obj)

    def test_read_skips_blank_lines_and_eof(self):
        buf = io.StringIO("\n\n")
        self.assertIsNone(protocol.read_message(stream=buf))

    def test_binary_channel(self):
        read_buf = io.BytesIO(b'{"id": 7, "result": {"ok": true}}\n')
        write_buf = io.BytesIO()
        ch = protocol.BinaryChannel(read_buf, write_buf)
        ch.write({"method": "shutdown"})
        self.assertEqual(write_buf.getvalue(), b'{"method": "shutdown"}\n')
        self.assertEqual(ch.read(), {"id": 7, "result": {"ok": True}})


class TestAnalyzerClient(unittest.TestCase):
    """Round-trip against the real analyzer subprocess.

    Works with or without tilelang: the analyzer reports capabilities in its
    hello and serves static analysis when tilelang is missing.
    """

    def setUp(self):
        self.client = AnalyzerClient(target="c")

    def tearDown(self):
        self.client.shutdown()

    def test_hello(self):
        self.assertTrue(self.client.start())
        hello = self.client.hello
        self.assertIsNotNone(hello)
        self.assertEqual(hello["analyzer_version"], protocol.SCHEMA_VERSION)
        self.assertIn("span_supported", hello)

    def test_analyze_static_file(self):
        with tempfile.NamedTemporaryFile(
            "w", suffix=".py", delete=False
        ) as f:
            f.write(
                "import tilelang.language as T\n"
                "def k():\n"
                "    A = T.alloc_shared((8, 8), T.float16)\n"
            )
            path = f.name
        self.addCleanup(os.unlink, path)

        result = self.client.analyze(path)
        self.assertIsNotNone(result)
        self.assertEqual(result["version"], protocol.SCHEMA_VERSION)
        labels = [h["label"] for h in result["hints"]]
        self.assertIn("⟨(8, 8) float16 @shared.dyn⟩", labels)
        # tilelang missing here -> static mode with an explanatory diagnostic;
        # a tilelang-capable env may reach further modes.
        self.assertIn(result["meta"]["mode"],
                      (protocol.MODE_STATIC, protocol.MODE_FULL,
                       protocol.MODE_PARTIAL, protocol.MODE_TRACER))

    def test_analyze_missing_file(self):
        result = self.client.analyze("/nonexistent/definitely_missing.py")
        self.assertIsNotNone(result)
        self.assertTrue(result["diagnostics"])

    def test_crash_restart(self):
        self.assertTrue(self.client.start())
        # Kill the worker; the next request must restart it once and answer.
        self.client._proc.kill()
        time.sleep(0.2)
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write("x = 1\n")
            path = f.name
        self.addCleanup(os.unlink, path)
        result = self.client.analyze(path)
        self.assertIsNotNone(result)
        self.assertTrue(self.client.alive)


class TestResultCache(unittest.TestCase):
    def test_mem_and_disk_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["TLSP_CACHE_DIR"] = tmp
            try:
                cache = ResultCache()
                result = protocol.make_result(mode=protocol.MODE_STATIC,
                                              hints=[protocol.make_hint(1, "alloc", "x")])
                cache.put("/p/a.py", "content-1", "c", result, tl_version="0")
                hit = cache.get("/p/a.py", "content-1", "c", tl_version="0")
                self.assertIsNotNone(hit)
                self.assertEqual(hit["hints"][0]["label"], "x")
                # different content/target -> miss
                self.assertIsNone(cache.get("/p/a.py", "content-2", "c", tl_version="0"))
                self.assertIsNone(cache.get("/p/a.py", "content-1", "cuda", tl_version="0"))
                # fresh instance still hits via disk
                cache2 = ResultCache()
                self.assertIsNotNone(cache2.get("/p/a.py", "content-1", "c", tl_version="0"))
            finally:
                os.environ.pop("TLSP_CACHE_DIR", None)


if __name__ == "__main__":
    unittest.main()
