# tilelang_LSP_extension README

## Release Notes

### 0.0.1

Now, the extension supports interpreting variables defined with basic tilelang allocation languages such as T.alloc_shared(), T.alloc_fragment() and so on.
Basic variable lifetime analysis is supported.

---
## Requirements
The extension requires **npm** to compile the code and **vsce** to package and release.

## How to use the LSP VSCode extension

To package and release the extension, one can run the following bash code to compile the dependencies of releasing the extension:
```bash
cd tilelang_lsp_extension
npm install
```
Then under the folder /tilelang_lsp_extension/, run

```bash
vsce package
```
If there are some confirmations that show up in the console, please type in "yes".
Once the extension is packaged, an extension program "tilelang-type-decorator-(*version).vsix" will appear in the current folder, and you can install this extension with VSCode.

If you are a developer and you wish to debug the extension, you can run

```bash
cd tilelang_lsp_extension
npm run compile
```
and in VS Code, open src/extension.ts and press F5 to enter the debug mode.

## Configuration

Settings (under `tilelang.*`; changing any of them restarts the server):

- `tilelang.pythonPath` (default `"python"`): interpreter for the server. It must be
  able to `import tlsp` (install with `pip install -e .` in the tilelang-lsp repo);
  a tilelang-enabled interpreter unlocks full analysis.
- `tilelang.enableInlayHints` (default `true`): master switch.
- `tilelang.serverArgs` (default `[]`): extra args appended to `python -m tlsp`
  (e.g. `["--target", "c"]` to change the default analysis target).
- `tilelang.serverEnv` (default `{"TILELANG_ENABLE_DATA_RACE_CHECK": "1"}`): extra
  environment variables for the server process, inherited by the analyzer
  subprocess. The default enables data-race diagnostics; remove that entry to opt out.
  Custom entries replace the whole default object — keep the race-check key if you
  still want those diagnostics. `PYTHONPATH` is managed by the extension.

## CUDA debugging (experimental)

With [NVIDIA Nsight VS Code Edition](https://marketplace.visualstudio.com/items?itemName=nvidia.nsight-vscode-edition)
installed, **TileLang: Debug CUDA Kernel** launches the active Python script under `cuda-gdb`
via `python -m tlsp.debug`. TileLang's line directives and `-G` device debug info allow device
breakpoints and stepping to map back to Python source lines. Linux with an NVIDIA GPU (local
or Remote-SSH/WSL) and TileLang >= 0.1.15 are required. The extension contributes Python
breakpoint support for the editor; Nsight supplies the actual debug adapter.

Settings (also under `tilelang.*`):

- `tilelang.debug.cudaGdbPath` (default `""`): cuda-gdb path; empty uses the Nsight default.
- `tilelang.debug.deviceDebug` (default `true`): add `-G`; disable for line-only `-lineinfo`.
- `tilelang.debug.breakOnLaunch` (default `true`): stop at every kernel launch, including
  framework kernels; disable to rely only on Python pending breakpoints.
- `tilelang.debug.args` (default `[]`): arguments forwarded to the debugged script.
- `tilelang.debug.extraEnv` (default `{}`): extra debuggee environment variables; the tlsp
  module path is preserved when merging `PYTHONPATH`.
- `tilelang.debug.debugDir` (default `""`): overrides `.tlsp/debug` for generated artifacts.

**TileLang: Open Generated CUDA Source** recursively finds JIT `.c` and NVCC `.cu` files under
the debug directory. See `docs/cuda-debugging.md` in the repository for the support matrix,
current validation status, and Linux/GPU acceptance test. Non-GPU configuration tests run with
`npm run test:cuda-debug`.

## Working mechanism

The extension is a thin LSP client: it resolves a Python interpreter (configured →
ms-python selection → `python`/`python3`) and spawns `python -m tlsp` over stdio.

All analysis happens in the server (`tlsp` package): while you type it answers from
pure-AST static analysis; on save it dispatches to a resident analyzer subprocess that
runs tilelang's frontend passes and collects hints from IR spans, with a four-level
degradation chain (full → partial → tracer → static) so hints and precise diagnostics
survive kernels that do not compile yet. See the repository README for details.

## Following extension guidelines
1. Install the extension
2. Inlay decorations will be displayed on the screen

**Enjoy!**
