"""pygls LSP front-end: debounce, caches, feature handlers, analyzer subprocess.

Analysis work is split by trigger:

- typing (didChange): pure-AST static analysis, in-process, milliseconds,
  debounced 500ms (latest-wins) — hints/diagnostics refresh as you pause;
- open/save (didOpen/didSave): full analysis dispatched to the persistent
  `tlsp.analyzer` subprocess (tilelang lives there), results cached by
  content hash both in memory and on disk under ~/.cache/tilelang_lsp.

The server never imports tilelang itself, so it starts instantly and survives
tilelang-less environments (static-only mode with an L3 diagnostic).
"""

from __future__ import annotations

import logging
import queue
import threading

from pygls.lsp.server import LanguageServer
from lsprotocol.types import (
    INITIALIZE,
    TEXT_DOCUMENT_DID_CHANGE,
    TEXT_DOCUMENT_DID_CLOSE,
    TEXT_DOCUMENT_DID_OPEN,
    TEXT_DOCUMENT_DID_SAVE,
    TEXT_DOCUMENT_DOCUMENT_SYMBOL,
    TEXT_DOCUMENT_HOVER,
    TEXT_DOCUMENT_INLAY_HINT,
    DidChangeTextDocumentParams,
    DidCloseTextDocumentParams,
    DidOpenTextDocumentParams,
    DidSaveTextDocumentParams,
    Diagnostic,
    DiagnosticSeverity,
    DocumentSymbolParams,
    Hover,
    HoverParams,
    InitializeParams,
    InlayHint,
    InlayHintKind,
    InlayHintParams,
    Location,
    MarkupContent,
    MarkupKind,
    Position,
    PublishDiagnosticsParams,
    Range,
    SymbolInformation,
    SymbolKind,
)

from . import __version__, protocol
from .config import kernel_symbols
from .static_fallback import analyze_static
from .worker import AnalyzerClient, ResultCache

logger = logging.getLogger("tlsp.server")


# ---------------------------------------------------------------------------
# LSP conversion helpers
# ---------------------------------------------------------------------------

_SEVERITY = {
    protocol.SEVERITY_ERROR: DiagnosticSeverity.Error,
    protocol.SEVERITY_WARNING: DiagnosticSeverity.Warning,
}


def _utf16_len(text: str) -> int:
    """Length in UTF-16 code units (the LSP position encoding)."""
    return len(text.encode("utf-16-le")) // 2


def _to_lsp_diagnostics(result: dict, lines: list) -> list:
    out = []
    for d in result.get("diagnostics", []):
        line = max(d["line"] - 1, 0)
        # pygls' document.lines keeps the trailing newline: strip it, then
        # measure in UTF-16 code units — plain len is wrong on both counts.
        text = lines[line].rstrip("\r\n") if line < len(lines) else ""
        start_col = _utf16_len(text) - _utf16_len(text.lstrip())
        out.append(
            Diagnostic(
                range=Range(
                    start=Position(line=line, character=start_col),
                    end=Position(line=line, character=max(_utf16_len(text), start_col + 1)),
                ),
                message=d["message"],
                severity=_SEVERITY.get(d.get("severity"), DiagnosticSeverity.Error),
                source=d.get("source", "tilelang"),
            )
        )
    return out


def _to_lsp_hints(result: dict, lines: list, visible: tuple | None = None) -> list:
    out = []
    for h in result.get("hints", []):
        line = h["line"] - 1
        if not (0 <= line < len(lines)):
            continue
        if visible is not None and not (visible[0] <= line <= visible[1]):
            continue
        out.append(
            InlayHint(
                position=Position(line=line, character=_utf16_len(lines[line].rstrip("\r\n"))),
                label=h["label"],
                kind=InlayHintKind.Type,
                padding_left=True,
                tooltip=h.get("detail") or None,
            )
        )
    return out


_SYMBOL_KIND = {"function": SymbolKind.Function}


def _to_lsp_symbols(result: dict, lines: list, uri: str) -> list:
    out = []
    for s in result.get("symbols", []):
        line = s["line"] - 1
        if not (0 <= line < len(lines)):
            continue
        # Whole-line range, mirroring the diagnostics: span semantics are
        # line-level, so start at the first non-whitespace character.
        text = lines[line].rstrip("\r\n")
        start_col = _utf16_len(text) - _utf16_len(text.lstrip())
        out.append(
            SymbolInformation(
                name=s["name"],
                kind=_SYMBOL_KIND.get(s.get("kind"), SymbolKind.Function),
                location=Location(
                    uri=uri,
                    range=Range(
                        start=Position(line=line, character=start_col),
                        end=Position(line=line, character=max(_utf16_len(text), start_col + 1)),
                    ),
                ),
            )
        )
    return out


# ---------------------------------------------------------------------------
# server
# ---------------------------------------------------------------------------


class TileLangLanguageServer:
    def __init__(self, target: str = "cuda"):
        self.server = LanguageServer("tlsp", __version__)
        self.analyzer = AnalyzerClient(target=target)
        self.cache = ResultCache()
        self.results: dict[str, dict] = {}  # uri -> latest protocol result
        self._bg_thread: threading.Thread | None = None
        self._bg_queue: queue.Queue = queue.Queue()
        self._debounce_timers: dict[str, threading.Timer] = {}
        self._register_features()

    # -- background full analysis ---------------------------------------------

    def _ensure_bg(self) -> None:
        if self._bg_thread is None or not self._bg_thread.is_alive():
            self._bg_thread = threading.Thread(target=self._bg_loop, daemon=True)
            self._bg_thread.start()

    def _bg_loop(self) -> None:
        while True:
            task = self._bg_queue.get()
            if task is None:
                return
            uri, path, text = task
            try:
                self._full_analysis(uri, path, text)
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("full analysis failed: %s", exc)

    def schedule_full_analysis(self, uri: str) -> None:
        document = self.server.workspace.get_text_document(uri)
        if document is None:
            return
        if "tilelang" not in document.source:
            return  # plain Python file: static hints only, no analyzer needed
        path = document.path  # pygls to_fs_path: decodes %XX, drives, UNC
        self._ensure_bg()
        # Drain stale queued tasks for this uri (latest-wins).
        kept = []
        while True:
            try:
                task = self._bg_queue.get_nowait()
            except queue.Empty:
                break
            if task[0] != uri:
                kept.append(task)
        for task in kept:
            self._bg_queue.put(task)
        self._bg_queue.put((uri, path, document.source))

    def _full_analysis(self, uri: str, path: str, text: str) -> None:
        # Idempotent start: guarantees hello (hence the tilelang version in
        # the cache key) is available even if the initialize-time warm-up
        # thread has not finished yet.
        self.analyzer.start()
        cached = self.cache.get(path, text, self.analyzer.target, (self.analyzer.hello or {}).get("tilelang_version"))
        if cached is not None:
            logger.info("cache hit for %s", path)
            self._publish_if_current(uri, text, cached)
            return

        result = self.analyzer.analyze(path)
        if result is None:
            # Analyzer unavailable/crashed: static fallback + L3 diagnostic.
            static = analyze_static(text)
            result = protocol.make_result(
                mode=protocol.MODE_STATIC,
                hints=static.hints,
                diagnostics=[
                    protocol.make_diagnostic(
                        1, "tilelang analyzer unavailable (crash/timeout); static hints only. See TileLang LSP output."
                    )
                ],
                tilelang_version=None,
            )
        else:
            self.cache.put(path, text, self.analyzer.target, result, (self.analyzer.hello or {}).get("tilelang_version"))
        self._publish_if_current(uri, text, result)

    # -- publish ----------------------------------------------------------------

    def _publish_if_current(self, uri: str, text: str, result: dict) -> None:
        """Publish only when the document still holds the analyzed text.

        The full analysis runs in the background; the user may keep editing
        or close the file meanwhile. A stale result must not overwrite newer
        static results (it stays in the content-hash cache either way).
        """
        document = self.server.workspace.get_text_document(uri)
        if document is None or document.source != text:
            logger.info("discarding stale full-analysis result for %s", uri)
            return
        self._publish(uri, result)

    def _publish(self, uri: str, result: dict) -> None:
        self.results[uri] = result
        document = self.server.workspace.get_text_document(uri)
        lines = document.lines if document is not None else []
        self.server.text_document_publish_diagnostics(PublishDiagnosticsParams(uri=uri, diagnostics=_to_lsp_diagnostics(result, lines)))
        try:
            self.server.workspace_inlay_hint_refresh(None)
        except Exception:
            logger.warning("inlay hint refresh failed", exc_info=True)

    # -- feature handlers ---------------------------------------------------------

    def _register_features(self) -> None:
        server = self.server

        @server.feature(INITIALIZE)
        def initialize(params: InitializeParams) -> None:
            # pygls computes server capabilities after this handler runs.
            # Warm up the analyzer: faster first analysis, and the disk-cache
            # key carries the tilelang version from the first file on.
            threading.Thread(target=self.analyzer.start, daemon=True).start()

        @server.feature(TEXT_DOCUMENT_DID_OPEN)
        def did_open(params: DidOpenTextDocumentParams) -> None:
            self._static_refresh(params.text_document.uri)
            self.schedule_full_analysis(params.text_document.uri)

        @server.feature(TEXT_DOCUMENT_DID_CHANGE)
        def did_change(params: DidChangeTextDocumentParams) -> None:
            # Debounce keystrokes (latest-wins, 500ms); the timer reads the
            # newest document text when it fires. The heavy full analysis
            # still waits for didSave.
            self._debounce_static(params.text_document.uri)

        @server.feature(TEXT_DOCUMENT_DID_SAVE)
        def did_save(params: DidSaveTextDocumentParams) -> None:
            self.schedule_full_analysis(params.text_document.uri)

        @server.feature(TEXT_DOCUMENT_DID_CLOSE)
        def did_close(params: DidCloseTextDocumentParams) -> None:
            timer = self._debounce_timers.pop(params.text_document.uri, None)
            if timer is not None:
                timer.cancel()
            self.results.pop(params.text_document.uri, None)
            self.server.text_document_publish_diagnostics(PublishDiagnosticsParams(uri=params.text_document.uri, diagnostics=[]))

        @server.feature(TEXT_DOCUMENT_INLAY_HINT)
        def inlay_hint(params: InlayHintParams) -> list:
            uri = params.text_document.uri
            result = self.results.get(uri)
            if result is None:
                self._static_refresh(uri)
                result = self.results.get(uri)
            if result is None:
                return []
            document = server.workspace.get_text_document(uri)
            lines = document.lines if document is not None else []
            visible = (params.range.start.line, params.range.end.line)
            return _to_lsp_hints(result, lines, visible)

        @server.feature(TEXT_DOCUMENT_HOVER)
        def hover(params: HoverParams) -> Hover | None:
            uri = params.text_document.uri
            result = self.results.get(uri)
            if result is None:
                return None
            line = params.position.line + 1
            details = [h["detail"] for h in result.get("hints", []) if h["line"] == line and h.get("detail")]
            if not details:
                return None
            body = "\n\n---\n\n".join(f"```text\n{d}\n```" for d in details)
            return Hover(contents=MarkupContent(kind=MarkupKind.Markdown, value=body))

        @server.feature(TEXT_DOCUMENT_DOCUMENT_SYMBOL)
        def document_symbol(params: DocumentSymbolParams) -> list:
            uri = params.text_document.uri
            result = self.results.get(uri)
            if result is None:
                return []
            document = server.workspace.get_text_document(uri)
            lines = document.lines if document is not None else []
            return _to_lsp_symbols(result, lines, uri)

    def _debounce_static(self, uri: str) -> None:
        old = self._debounce_timers.pop(uri, None)
        if old is not None:
            old.cancel()
        timer = threading.Timer(0.5, self._debounced_static_refresh, args=(uri,))
        timer.daemon = True
        self._debounce_timers[uri] = timer
        timer.start()

    def _debounced_static_refresh(self, uri: str) -> None:
        self._debounce_timers.pop(uri, None)
        try:
            self._static_refresh(uri)
        except Exception:
            logger.exception("static refresh failed for %s", uri)

    def _static_refresh(self, uri: str) -> None:
        document = self.server.workspace.get_text_document(uri)
        if document is None:
            return
        source = document.source
        if "tilelang" not in source:
            self.results[uri] = protocol.make_result(mode=protocol.MODE_STATIC)
            return
        static = analyze_static(source)
        previous = self.results.get(uri)
        # Keep compiler diagnostics of a previous full analysis; refresh only
        # the hints while typing. Full analysis re-publishes everything.
        diagnostics = list(static.diagnostics)
        if previous is not None and previous["meta"]["mode"] != protocol.MODE_STATIC:
            diagnostics = [d for d in previous.get("diagnostics", []) if d.get("source") != "tilelang-static"] + diagnostics
        result = protocol.make_result(
            mode=protocol.MODE_STATIC,
            hints=static.hints,
            diagnostics=diagnostics,
            # Keep the outline alive between full-analysis publishes.
            symbols=kernel_symbols(source),
            tilelang_version=(self.analyzer.hello or {}).get("tilelang_version"),
        )
        self.results[uri] = result
        self.server.text_document_publish_diagnostics(
            PublishDiagnosticsParams(
                uri=uri,
                diagnostics=_to_lsp_diagnostics(result, document.lines),
            )
        )

    # -- run ----------------------------------------------------------------------

    def start_io(self) -> None:
        try:
            self.server.start_io()
        finally:
            self.analyzer.shutdown()


def create_server(target: str = "cuda") -> TileLangLanguageServer:
    return TileLangLanguageServer(target=target)
