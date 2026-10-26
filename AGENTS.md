# AGENTS.md

Guidance for AI agents working in this repository.

## Project overview

`tlsp` is an LSP server for [TileLang](https://github.com/tile-ai/tilelang) that shows
inlay hints (buffer shapes/dtypes/scopes, kernel launch geometry) and diagnostics by
actually compiling the user's kernels in a background subprocess. When tilelang is
missing or compilation fails partway, the server degrades to a static AST-based mode
instead of going dark. `tilelang_lsp_extension/` is the VS Code client extension.

## Repository layout

- `tlsp/` — the LSP server package
  - `server.py` — pygls frontend: LSP handlers, debounce, version-bound publishing
  - `worker.py` — analyzer subprocess manager (queue, restart, disk cache)
  - `analyzer.py` — subprocess entry point (`python -m tlsp.analyzer`)
  - `pipeline.py` — analysis pipeline: user-module loading, mode selection (full / partial / static)
  - `prologue.py` — tilelang compile-stage prologue (target-aware lowering)
  - `span_collect.py`, `collect.py`, `markers.py`, `tracer.py` — span/hint collection
  - `race_report.py` — turns VerifyParallelLoop's stderr data-race warnings into diagnostics
  - `static_fallback.py` — AST-only analysis used when tilelang is unavailable or breaks early
  - `protocol.py` — wire protocol between server and worker
  - `config.py` — static config extraction from source
  - `debug.py` — CUDA-GDB launcher (`python -m tlsp.debug`) used by the extension's
    "Debug CUDA Kernel" command; enables TileLang line directives, `-G`, cache isolation
    and generated-source dumps. See `docs/cuda-debugging.md`.
- `tests/` — unittest suite + `fixtures/` (see gotchas below)
- `tilelang_lsp_extension/` — VS Code extension (TypeScript)
  - `src/cudaDebug.ts` — pure Nsight launch configuration and generated-source discovery;
    its Node tests live in `src/node-tests/` and run without VS Code or a GPU.

## Commands

```bash
# Unit tests (needs pygls installed; integration cases skip without env vars)
python -m unittest discover tests -v

# Integration tests (need a tilelang-capable python: PyPI tilelang>=0.1.13 or a
# local build; TLSP_TEST_TILELANG only when using a source checkout)
TLSP_TEST_PYTHON=/path/to/venv/bin/python TLSP_TEST_TILELANG=/path/to/tilelang \
  python -m unittest tests.test_integration -v

# Real CUDA-GDB acceptance (Linux + NVIDIA GPU; skipped without TLSP_TEST_CUDA_GDB)
TLSP_TEST_PYTHON=/path/to/venv/bin/python TLSP_TEST_TILELANG=/path/to/tilelang \
  TLSP_TEST_CUDA_GDB=/usr/local/cuda/bin/cuda-gdb \
  /path/to/venv/bin/python -m unittest tests.test_debug_cuda -v

# Lint everything (run before committing; mirrors the tilelang repo's setup)
pre-commit run --all-files

# VS Code extension (from tilelang_lsp_extension/)
npm ci && npm run compile && npm run lint && npm run test:cuda-debug
```

## Conventions and gotchas

- Python `>=3.8`. Every `tlsp` module except `__init__.py`/`__main__.py` starts with
  `from __future__ import annotations`;
  ruff is configured in `pyproject.toml` (line-length 140, rules `E,W,F,UP,FA,B,SIM`,
  target py38). Keep modern annotation syntax (`X | None`, `list[str]`) — it is safe
  because of the future import.
- Some `tests/fixtures/*.py` are **intentionally broken inputs** (NameError, failing
  LayoutInference, ...). Never "fix" those failures. Tests assert exact 1-based
  hint/diagnostic/debug lines, so changing fixture lines means updating
  `tests/test_integration.py` and possibly `tests/test_debug_cuda.py` (which pins
  `elementwise_add.py:23` and stepping to `:24`). `elementwise_add.py` and `cuda_debug_driver.py` are valid
  execution fixtures. Intentional undefined names use the ruff per-file-ignore in
  `pyproject.toml`.
- CUDA debugging requires TileLang >= 0.1.15 for Python line mapping. Normal jit/compile
  calls are covered; grouped autotune, direct JITKernel construction and compile
  subprocesses are not. GPU acceptance is opt-in through `TLSP_TEST_CUDA_GDB`; never
  claim the skipped test proves cuda-gdb breakpoints or stepping work.
- `tilelang_lsp_extension/test.py` is a manual playground with **deliberate syntax
  errors** for probing diagnostics in the editor. It is excluded from pre-commit at the
  top level of `.pre-commit-config.yaml`; do not fix or delete it.
- Integration tests run with `target="c"` (CPU lowering): kernel launches lower to
  serial loops, so expect `thread: 1` in fragment details, **not** the SIMT
  `threads=128`. Do not pass `lower_thread_binding=True` for `c`/`llvm` targets in
  `prologue.py` — that was a real bug.
- Analysis target comes from code, not an editor setting: `# <typecheck> target=...`
  beats `target=` at a `<kernel>.compile(...)` call site, which beats a literal
  `@tilelang.jit(target=...)` decorator, which beats the server `--target` default
  (cuda). Resolution lives in `config.resolve_config` (`_directive_target` merges
  target across applicable directives; kwargs stay nearest-directive-only); the
  `tilelang.target` VS Code setting was removed.
- Upstream tilelang keeps the data-race checker opt-in: `VerifyParallelLoop` only
  enters the prologue when `TILELANG_ENABLE_DATA_RACE_CHECK=1` (or pass config
  overrides). The VS Code extension turns it on by default via the
  `tilelang.serverEnv` setting (users can remove that entry to opt out); other
  clients must set the variable in the server process env themselves.
  `tests.test_integration.test_data_race_warning` pins the env var itself.
- The analyzer must never trust timestamp-based `.pyc` caches for user files:
  `pipeline.py` reads and compiles the source text itself.
- Background analysis results are bound to a document version before publishing;
  keep that invariant when touching `server.py`.
- Lint gate: `.pre-commit-config.yaml` (ruff, codespell, pymarkdown, plus an eslint
  `language: system` hook that reuses the extension's `node_modules`). CI
  (`.github/workflows/ci.yml`) runs pre-commit, the test suite on Python 3.12
  (unit + integration against PyPI `tilelang==0.1.13`), and the extension build.
- Never commit `node_modules/`, `out/`, `*.vsix`, `__pycache__/` (see `.gitignore`).
- Do not run `git commit`/`git push` unless the user explicitly asks.
