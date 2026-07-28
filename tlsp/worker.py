"""Analyzer subprocess management and analysis-result caching.

Kept free of pygls imports so it can be unit-tested with a bare interpreter.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import subprocess
import sys
import threading
from collections import OrderedDict

from . import protocol
import contextlib

logger = logging.getLogger("tlsp.worker")

HELLO_TIMEOUT_S = 30.0
ANALYZE_TIMEOUT_S = 15.0
MEM_CACHE_CAP = 32
DISK_CACHE_CAP = 256

# ---------------------------------------------------------------------------
# analyzer subprocess
# ---------------------------------------------------------------------------


class AnalyzerClient:
    """Owns the persistent analyzer subprocess (start, request, restart)."""

    def __init__(self, target: str = "cuda"):
        self.target = target
        self._proc: subprocess.Popen | None = None
        self._queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._req_id = 0
        self.hello: dict | None = None
        self._restart_used = False

    # -- lifecycle -----------------------------------------------------------

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> bool:
        """Spawn the analyzer and wait for its hello. Idempotent."""
        with self._lock:
            return self._start_locked()

    def _start_locked(self) -> bool:
        if self.alive and self.hello is not None:
            return True
        self._kill_locked()
        try:
            env = dict(os.environ)
            # Make the tlsp package importable in the child even when the
            # server itself runs from an unusual cwd or an editable install.
            pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            python_path = env.get("PYTHONPATH", "")
            if pkg_root not in python_path.split(os.pathsep):
                env["PYTHONPATH"] = pkg_root + (os.pathsep + python_path if python_path else "")
            self._proc = subprocess.Popen(
                [sys.executable, "-u", "-m", "tlsp.analyzer"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )
        except Exception as exc:
            logger.error("cannot spawn analyzer: %s", exc)
            self._proc = None
            return False
        self._queue = queue.Queue()
        # Pass proc/queue explicitly: after a restart the old reader thread
        # keeps draining the old process and must signal EOF into the OLD
        # queue only — never into the new generation's queue.
        threading.Thread(target=self._read_loop, args=(self._proc, self._queue), daemon=True).start()
        threading.Thread(target=self._drain_stderr, args=(self._proc,), daemon=True).start()
        try:
            msg = self._queue.get(timeout=HELLO_TIMEOUT_S)
        except queue.Empty:
            logger.error("analyzer hello timeout")
            self._kill_locked()
            return False
        if msg is None or msg.get("method") != "hello":
            logger.error("analyzer hello failed: %r", msg)
            self._kill_locked()
            return False
        self.hello = msg.get("result") or {}
        logger.info("analyzer up: %s", self.hello)
        return True

    def shutdown(self) -> None:
        with self._lock:
            if self.alive:
                with contextlib.suppress(Exception):
                    self._write({"method": "shutdown"})
            self._kill_locked()

    def _kill_locked(self) -> None:
        proc, self._proc = self._proc, None
        self.hello = None
        if proc is not None:
            try:
                proc.kill()
                proc.wait(timeout=5)
            except Exception:
                pass
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                try:
                    if stream is not None:
                        stream.close()
                except Exception:
                    pass

    # -- io -------------------------------------------------------------------

    def _read_loop(self, proc: subprocess.Popen, q: queue.Queue) -> None:
        try:
            assert proc.stdout is not None
            for line in iter(proc.stdout.readline, b""):
                line = line.strip()
                if not line:
                    continue
                try:
                    q.put(json.loads(line.decode("utf-8")))
                except ValueError:
                    logger.warning("analyzer: bad JSON line: %r", line[:200])
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("analyzer read loop ended: %s", exc)
        finally:
            q.put(None)

    def _drain_stderr(self, proc: subprocess.Popen) -> None:
        try:
            assert proc.stderr is not None
            for line in iter(proc.stderr.readline, b""):
                logger.debug("analyzer: %s", line.decode("utf-8", "replace").rstrip())
        except Exception:
            pass

    def _write(self, obj: dict) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        data = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
        self._proc.stdin.write(data)
        self._proc.stdin.flush()

    # -- requests ---------------------------------------------------------------

    def analyze(self, path: str, kernel: str | None = None, kwargs: dict | None = None) -> dict | None:
        """Run a full analysis; None when the analyzer is unavailable."""
        with self._lock:
            if not self._start_locked():
                return None
            result = self._request_locked(
                {
                    "path": path,
                    "kernel": kernel,
                    "kwargs": kwargs or {},
                    "target": self.target,
                }
            )
            if result is None and not self._restart_used:
                # One restart per client lifetime for a wedged/crashed worker.
                self._restart_used = True
                logger.warning("analyzer request failed; restarting worker once")
                self._kill_locked()
                if self._start_locked():
                    result = self._request_locked(
                        {
                            "path": path,
                            "kernel": kernel,
                            "kwargs": kwargs or {},
                            "target": self.target,
                        }
                    )
            if result is not None:
                self._restart_used = False
            return result

    def _request_locked(self, params: dict) -> dict | None:
        self._req_id += 1
        req_id = self._req_id
        try:
            self._write(protocol.analyze_request(req_id, params))
        except Exception as exc:
            logger.warning("analyzer write failed: %s", exc)
            return None
        while True:
            try:
                msg = self._queue.get(timeout=ANALYZE_TIMEOUT_S)
            except queue.Empty:
                logger.warning("analyzer analyze timeout for %s", params.get("path"))
                return None
            if msg is None:
                return None  # worker died
            if msg.get("id") != req_id:
                continue  # stale response to an earlier timed-out request
            if "error" in msg:
                logger.warning("analyzer error: %s", msg["error"])
                return None
            return msg.get("result")


# ---------------------------------------------------------------------------
# caches
# ---------------------------------------------------------------------------


def _cache_dir() -> str:
    return os.environ.get("TLSP_CACHE_DIR", os.path.expanduser("~/.cache/tilelang_lsp"))


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ResultCache:
    """Memory LRU + content-hash-keyed disk cache of analysis results."""

    def __init__(self):
        self._mem: OrderedDict[str, dict] = OrderedDict()

    def _key(self, path: str, text: str, target: str, tl_version: str | None) -> str:
        raw = json.dumps([os.path.abspath(path), _content_hash(text), target, tl_version or ""])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, path: str, text: str, target: str, tl_version: str | None = None) -> dict | None:
        key = self._key(path, text, target, tl_version)
        hit = self._mem.get(key)
        if hit is not None:
            self._mem.move_to_end(key)
            return hit
        cache_file = os.path.join(_cache_dir(), key + ".json")
        try:
            with open(cache_file, encoding="utf-8") as f:
                result = json.load(f)
            self._mem[key] = result
            self._evict_mem()
            return result
        except (OSError, ValueError):
            return None

    def put(self, path: str, text: str, target: str, result: dict, tl_version: str | None = None) -> None:
        key = self._key(path, text, target, tl_version)
        self._mem[key] = result
        self._evict_mem()
        try:
            os.makedirs(_cache_dir(), exist_ok=True)
            tmp = os.path.join(_cache_dir(), key + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(result, f)
            os.replace(tmp, os.path.join(_cache_dir(), key + ".json"))
            self._evict_disk()
        except OSError:
            pass

    def _evict_mem(self) -> None:
        while len(self._mem) > MEM_CACHE_CAP:
            self._mem.popitem(last=False)

    def _evict_disk(self) -> None:
        try:
            files = [os.path.join(_cache_dir(), f) for f in os.listdir(_cache_dir()) if f.endswith(".json")]
            if len(files) <= DISK_CACHE_CAP:
                return
            files.sort(key=os.path.getmtime)
            for f in files[: len(files) - DISK_CACHE_CAP]:
                os.unlink(f)
        except OSError:
            pass
