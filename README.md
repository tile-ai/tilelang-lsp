# TileLang LSP (tlsp)

[中文 README](./README_zh.md)

A Language Server Protocol (LSP) implementation for [TileLang](https://github.com/tile-ai/tilelang),
providing inlay hints for buffer shapes/dtypes/scopes and inferred layouts, hover details,
and precise diagnostics.

## Features

- **Inlay hints** at end of line, five kinds:
  - `alloc` — `X = T.alloc_*(...)` lines: shape/dtype/scope, plus the inferred
    `Layout`/`Fragment` once LayoutInference has run
  - `param` — kernel signature parameters: `⟨(M, K) float16 @global⟩`
  - `op` — `T.copy` / `T.gemm` / ... lines: short summary of the layouts involved
  - `loop` — `T.Pipelined` / `T.Parallel` lines: `⟨pipelined, stages=3⟩`, parallel loop layout
  - `kernel` — `with T.Kernel(...)` line: grid dims and thread count
- **Hover** on a hint line shows the full layout/fragment text.
- **Diagnostics** at three precision levels (see below).
- **Works while your kernel does not compile**: a four-level degradation chain still
  produces as many hints as possible (see "How it works").

Results are cached in memory (LRU) and on disk under `~/.cache/tilelang_lsp/`
(override with `TLSP_CACHE_DIR`), keyed by file content hash + target + tilelang version.

## Installation

1. Python 3.8+ with [tilelang](https://github.com/tile-ai/tilelang) importable
   (for full analysis; without it tlsp still serves static hints).
2. Install the server package:
   ```bash
   pip install -e .
   ```
3. Install the VS Code extension (see below).

## Usage

### VS Code

```bash
cd tilelang_lsp_extension
npm install
vsce package   # produces tilelang-type-decorator-*.vsix
```

Install the `.vsix` in VS Code. The extension launches `python -m tlsp` with the
configured interpreter. To develop/debug the extension: `npm run compile`, then open
`src/extension.ts` and press F5.

Settings (all under `tilelang.*`, changes restart the server automatically):

| Setting | Default | Meaning |
|---|---|---|
| `tilelang.pythonPath` | `"python"` | Python interpreter for the server (must be able to `import tlsp`; tilelang optional but recommended) |
| `tilelang.enableInlayHints` | `true` | Master switch for the server |
| `tilelang.serverArgs` | `[]` | Extra CLI args appended to `python -m tlsp` (e.g. `["--target", "c"]` to change the default analysis target) |
| `tilelang.serverEnv` | `{"TILELANG_ENABLE_DATA_RACE_CHECK": "1"}` | Extra environment variables for the server process (the analyzer subprocess inherits them). Default enables data-race diagnostics; remove that entry to opt out. Custom entries replace the whole default object — keep the race-check key if you still want those diagnostics. `PYTHONPATH` is managed by the extension |

### Neovim

```lua
vim.lsp.config['tlsp'] = {
  cmd = { 'python', '-m', 'tlsp' },  -- add '--target', 'c' on CUDA-free machines
  filetypes = { 'python' },
  root_markers = { 'pyproject.toml', '.git' },
}
vim.lsp.config['tlsp'].capabilities = vim.lsp.protocol.make_client_capabilities()
vim.lsp.enable('tlsp')
```

Outside VS Code, set `TILELANG_ENABLE_DATA_RACE_CHECK=1` in the server process
environment to enable data-race diagnostics (the VS Code extension enables it by
default via `tilelang.serverEnv`).

### The `<typecheck>` directive

tlsp needs concrete kernel arguments to elaborate TIR. It resolves them in this order:

1. A comment directive nearest to the kernel definition:
   ```python
   # <typecheck> M=1024, N=1024, K=1024, block_M=128, kernel=matmul
   ```
   Values are parsed with `ast.literal_eval`; dtype strings like `"float16"` work.
   `kernel=` selects which kernel to analyze (default: first `@tilelang.jit` found).
2. Kwargs harvested from `<name>.compile(...)` / direct kernel calls in the file.
3. Function signature defaults.

The analysis target comes from the code:
`# <typecheck> target=c` wins over `target=` at a `<kernel>.compile(...)` call
site, which wins over a literal `@tilelang.jit(target=...)` decorator, which in
turn wins over the server default (`--target`, `cuda` if unset).

Missing required parameters produce a diagnostic asking for a `<typecheck>` directive.
Symbolic variables declared via `T.const("M, N")` may stay unbound.

### CLI debugging

```bash
python -m tlsp.analyzer --once path/to/kernel.py --target c
```
prints the full analysis result (schema v2) as a single JSON line.

## How it works

On every open/save, the analyzer runs a four-level degradation chain and reports its mode:

| Mode | Condition | What you get |
|---|---|---|
| `full` | `get_tir` OK + frontend passes up to LayoutInference OK | All hints, alloc hints include inferred layouts |
| `partial` | `get_tir` OK, some pass failed | Hints collected from the last good IR state + an L2 diagnostic at the exact line (parsed from the `--> file:line:col` marker in the pass error) |
| `tracer` | IR spans unavailable (old tilelang, or `TILELANG_ENABLE_IR_SPAN=0`) | Hints from a monkeypatch call tracer over `T.*` entries; layouts joined by buffer name when passes succeed |
| `static` | `get_tir` failed (import error, NameError, half-written code) | Pure-AST hints (shape/dtype/scope) + shape-mismatch warnings + an L1 diagnostic from traceback remapping |

If the analyzer subprocess itself crashes or tilelang is missing entirely, the server
falls back to static mode with a file-level (L3) diagnostic explaining why.

While you type (before saving), hints come from the same static analysis running
in-process — so they stay responsive and correct for unparsable intermediate states.

## Development

```bash
python -m unittest discover tests -v    # unit tests (no tilelang needed)
```

Integration tests need a python that can import tilelang — a plain
`pip install tilelang>=0.1.13` environment is enough (they skip cleanly without
one):

```bash
TLSP_TEST_PYTHON=/path/to/venv/bin/python python -m unittest tests.test_integration -v
# using a source checkout instead? also point at it:
# TLSP_TEST_TILELANG=/path/to/tilelang/repo
```

## Demo

![demo](./tilelang_lsp_extension/assets/demo.png)

Diagnostics

![partial](./tilelang_lsp_extension/assets/partial.png)
