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
- `tilelang.target` (default `"cuda"`): analysis target (`cuda`/`c`/`llvm`/`metal`/`webgpu`).
  Use `"c"` on machines without CUDA.
- `tilelang.enableInlayHints` (default `true`): master switch.
- `tilelang.serverArgs` (default `[]`): extra args appended to `python -m tlsp`.

## Working mechanism

The extension is a thin LSP client: it resolves a Python interpreter (configured →
ms-python selection → `python`/`python3`), spawns `python -m tlsp` over stdio, and
forwards the analysis target via `initializationOptions`.

All analysis happens in the server (`tlsp` package): while you type it answers from
pure-AST static analysis; on save it dispatches to a resident analyzer subprocess that
runs tilelang's frontend passes and collects hints from IR spans, with a four-level
degradation chain (full → partial → tracer → static) so hints and precise diagnostics
survive kernels that do not compile yet. See the repository README for details.

## Following extension guidelines
1. Install the extension
2. Inlay decorations will be displayed on the screen

**Enjoy!**
