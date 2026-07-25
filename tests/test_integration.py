"""Integration tests against a real tilelang environment.

Gated by environment variables (tests skip cleanly without them):

    TLSP_TEST_PYTHON    python that can import tilelang (+ pygls for the LSP
                        session test), e.g. /path/to/tilelang/venv/bin/python
    TLSP_TEST_TILELANG  path of the tilelang repo, prepended to PYTHONPATH
                        (needed when the venv has no tilelang .pth/install)

On a machine without CUDA, use target "c" (these fixtures are CUDA-free).
"""

import json
import os
import subprocess
import sys
import threading
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES = os.path.join(REPO_ROOT, "tests", "fixtures")

TEST_PYTHON = os.environ.get("TLSP_TEST_PYTHON")
TILELANG_PATH = os.environ.get("TLSP_TEST_TILELANG")


def _child_env(extra=None):
    env = dict(os.environ)
    paths = [REPO_ROOT]
    if TILELANG_PATH:
        paths.insert(0, TILELANG_PATH)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(paths + ([existing] if existing else []))
    if extra:
        env.update(extra)
    return env


def run_analyzer_once(fixture, target="c", extra_env=None, timeout=180):
    proc = subprocess.run(
        [TEST_PYTHON, "-m", "tlsp.analyzer", "--once",
         os.path.join(FIXTURES, fixture), "--target", target],
        capture_output=True, text=True, env=_child_env(extra_env), timeout=timeout,
    )
    if proc.returncode != 0:
        raise AssertionError(f"analyzer exited {proc.returncode}:\n{proc.stderr[-2000:]}")
    return json.loads(proc.stdout)


def hint_lines(result, kind):
    return sorted(h["line"] for h in result["hints"] if h["kind"] == kind)


@unittest.skipUnless(TEST_PYTHON, "TLSP_TEST_PYTHON not set")
class TestAnalyzerIntegration(unittest.TestCase):
    def test_elementwise_full(self):
        r = run_analyzer_once("elementwise_add.py")
        self.assertEqual(r["meta"]["mode"], "full")
        self.assertEqual(r["diagnostics"], [])
        self.assertEqual(hint_lines(r, "param"), [10, 11, 12])
        self.assertEqual(hint_lines(r, "alloc"), [15, 16, 17, 18])
        self.assertEqual(hint_lines(r, "op"), [20, 21, 24, 25])
        self.assertEqual(hint_lines(r, "loop"), [22])
        kernel = [h for h in r["hints"] if h["kind"] == "kernel"]
        self.assertEqual(len(kernel), 1)
        self.assertEqual(kernel[0]["line"], 14)
        self.assertIn("threads=128", kernel[0]["label"])
        c_local = [h for h in r["hints"] if h.get("buffer") == "C_local"]
        self.assertTrue(any("Fragment" in h["label"] for h in c_local))
        self.assertTrue(any(h.get("detail") for h in c_local))

    def test_gemm_partial_layout_failure(self):
        r = run_analyzer_once("gemm_partial.py")
        self.assertEqual(r["meta"]["mode"], "partial")
        errors = [d for d in r["diagnostics"] if d["severity"] == "error"]
        self.assertEqual(len(errors), 1)
        # LayoutInference reports the C_local alloc line via the span marker.
        self.assertEqual(errors[0]["line"], 18)
        self.assertIn("LayoutInference", errors[0]["message"])
        self.assertEqual(hint_lines(r, "alloc"), [16, 17, 18])
        loop = [h for h in r["hints"] if h["kind"] == "loop"]
        self.assertEqual(len(loop), 1)
        self.assertIn("stages=3", loop[0]["label"])

    def test_broken_name_static(self):
        r = run_analyzer_once("broken_name.py")
        self.assertEqual(r["meta"]["mode"], "static")
        errors = [d for d in r["diagnostics"] if d["severity"] == "error"]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["line"], 18)
        self.assertIn("block_M_typo", errors[0]["message"])
        self.assertIn(16, hint_lines(r, "alloc"))

    def test_multi_kernel_directive(self):
        r = run_analyzer_once("multi_kernel.py")
        # The directive selects kernel_two (body starts at line 21).
        self.assertEqual(hint_lines(r, "param"), [22, 23])
        kernel = [h for h in r["hints"] if h["kind"] == "kernel"]
        self.assertEqual(len(kernel), 1)
        self.assertEqual(kernel[0]["line"], 24)
        self.assertIn("threads=64", kernel[0]["label"])

    def test_tracer_matches_span_lines(self):
        span = run_analyzer_once("elementwise_add.py")
        tracer = run_analyzer_once(
            "elementwise_add.py", extra_env={"TILELANG_ENABLE_IR_SPAN": "0"}
        )
        self.assertEqual(tracer["meta"]["mode"], "tracer")
        for kind in ("alloc", "op"):
            self.assertEqual(
                hint_lines(tracer, kind), hint_lines(span, kind),
                f"{kind} lines differ between tracer and span modes",
            )


# ---------------------------------------------------------------------------
# minimal LSP client for the end-to-end session test
# ---------------------------------------------------------------------------


class LSPClient:
    def __init__(self, argv, env):
        self.proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, env=env,
        )
        self._id = 0
        self.notifications = []
        self._lock = threading.Lock()

    def _send(self, obj):
        body = json.dumps(obj).encode("utf-8")
        self.proc.stdin.write(
            f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body
        )
        self.proc.stdin.flush()

    def request(self, method, params=None, timeout=120):
        self._id += 1
        req_id = self._id
        self._send({"jsonrpc": "2.0", "id": req_id, "method": method,
                    "params": params or {}})
        return self._wait_response(req_id, timeout)

    def notify(self, method, params=None):
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def _read_message(self):
        headers = {}
        while True:
            line = self.proc.stdout.readline().decode("ascii", "replace").strip()
            if not line:
                break
            if ":" in line:
                k, v = line.split(":", 1)
                headers[k.strip().lower()] = v.strip()
        length = int(headers["content-length"])
        return json.loads(self.proc.stdout.read(length).decode("utf-8"))

    def _wait_response(self, req_id, timeout):
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self._read_message()
            if msg.get("id") == req_id:
                return msg
            with self._lock:
                self.notifications.append(msg)
        raise TimeoutError(f"no response to request {req_id}")

    def wait_notification(self, method, timeout=120):
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                for i, msg in enumerate(self.notifications):
                    if msg.get("method") == method:
                        return self.notifications.pop(i)
            time.sleep(0.05)
        raise TimeoutError(f"no notification {method}")

    def close(self):
        try:
            self.request("shutdown", timeout=10)
            self.notify("exit")
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


@unittest.skipUnless(TEST_PYTHON, "TLSP_TEST_PYTHON not set")
class TestLSPSession(unittest.TestCase):
    def test_open_save_hover_hints(self):
        client = LSPClient([TEST_PYTHON, "-m", "tlsp", "--target", "c"],
                           _child_env())
        self.addCleanup(client.close)

        init = client.request("initialize", {
            "processId": None,
            "rootUri": None,
            "capabilities": {},
            "initializationOptions": {"target": "c"},
        })
        self.assertIn("capabilities", init["result"])
        client.notify("initialized", {})

        fixture = os.path.join(FIXTURES, "elementwise_add.py")
        uri = "file://" + fixture
        with open(fixture, encoding="utf-8") as f:
            text = f.read()
        client.notify("textDocument/didOpen", {
            "textDocument": {"uri": uri, "languageId": "python",
                             "version": 1, "text": text},
        })

        # Static hints are available immediately.
        hints = client.request("textDocument/inlayHint", {
            "textDocument": {"uri": uri},
            "range": {"start": {"line": 0, "character": 0},
                      "end": {"line": 100, "character": 0}},
        })
        static_labels = [h["label"] for h in hints["result"]]
        self.assertTrue(any("shared.dyn" in label for label in static_labels))

        # didSave triggers the full analysis (background thread + subprocess).
        client.notify("textDocument/didSave", {"textDocument": {"uri": uri}})

        # Poll inlayHint until the full-analysis hints land (the first
        # publishDiagnostics notification may be the didOpen static refresh).
        import time

        full_labels = []
        deadline = time.time() + 180
        while time.time() < deadline:
            hints = client.request("textDocument/inlayHint", {
                "textDocument": {"uri": uri},
                "range": {"start": {"line": 0, "character": 0},
                          "end": {"line": 100, "character": 0}},
            }, timeout=60)
            full_labels = [h["label"] for h in hints["result"]]
            if any("Fragment" in label for label in full_labels):
                break
            time.sleep(1.0)
        self.assertTrue(any("Fragment" in label for label in full_labels),
                        f"no layout hints after full analysis: {full_labels}")
        self.assertTrue(any("threads=128" in label for label in full_labels))

        # The server must push workspace/inlayHint/refresh after a background
        # analysis, otherwise real clients keep showing the stale static hints
        # (regression: pygls requires workspace_inlay_hint_refresh(None) — the
        # params argument is mandatory, a bare call raises TypeError).
        refresh = client.wait_notification("workspace/inlayHint/refresh", timeout=10)
        self.assertIn("id", refresh)

        # Hover on the C_local alloc line (line 17 -> 0-based 16).
        hover = client.request("textDocument/hover", {
            "textDocument": {"uri": uri},
            "position": {"line": 16, "character": 10},
        })
        self.assertIsNotNone(hover["result"])
        self.assertIn("Fragment", hover["result"]["contents"]["value"])


if __name__ == "__main__":
    unittest.main()
