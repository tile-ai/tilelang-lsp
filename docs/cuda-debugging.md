# TileLang CUDA Debugging: Implementation and Validation

[中文版](./cuda-debugging_zh.md)

## Current status

This repository provides an experimental CUDA debugging entry point. It reuses the
`cuda-gdb` debug adapter from NVIDIA Nsight Visual Studio Code Edition and maps generated
TileLang device code back to the user's Python source lines. No custom DAP fork or TileLang
source changes are required.

Source-path analysis, launcher unit tests, extension configuration tests, TypeScript
compilation, and lint have passed. Real GPU acceptance has also passed on Ubuntu 24.04 with
an RTX 5090 (`sm_120a`), NVCC/CUDA-GDB 13.0, and TileLang 0.1.15:

- Python pending breakpoints resolve and stop after JIT loading;
- CUDA kernel and thread inspection works, including `threadIdx.x`;
- device stepping moves from Python line 23 to line 24 in the acceptance fixture;
- numerical checks pass after continuing execution;
- the CUBIN DWARF line table contains the original Python filename and line numbers.

Manual debugging through VS Code Remote-SSH and Nsight has also been confirmed. The UI
checklist below is retained for reproducibility.

Requirements:

- Linux locally or through Remote-SSH/WSL, an NVIDIA GPU, and CUDA Toolkit with nvcc and cuda-gdb;
- NVIDIA Nsight Visual Studio Code Edition for VS Code debugging;
- TileLang >= 0.1.15, including both `PassConfigKey.TL_EMIT_LINE_DIRECTIVES` and the
  corresponding native codegen implementation;
- a debuggee Python interpreter that can import both `tilelang` and `tlsp`.

## How it works

### Python line mapping

TileLang's `CodeGenTileLangCBase` emits the following directive for statements with spans when
`tl.emit_line_directives=True`:

```c
#line 42 "/path/to/kernel.py"
```

nvcc carries these logical filenames and line numbers into the device debug line table, allowing
cuda-gdb to display a device stop as `kernel.py:42`. Mapping is statement-level and best-effort:
one TileLang operation may expand to many CUDA statements. Generated statements without spans
inherit the preceding mapping, so adjacent Python lines can share or drift in complex kernels.

Four upstream codegen tests, including CUDA source generation, passed in the GPU environment.
They cover enabled mapping, the disabled default, repeated directives for the same Python line,
and CUDA Python line mapping.

### Device debug information

`compile_flags` accepted by `tilelang.compile` and `tilelang.jit` flow into
`tl.device_compile_flags` and then into the device compiler command. The launcher appends `-G`
by default to preserve local variables and full device debug information. A fixed host `-O3`
does not change the device-side behavior of `-G`.

Set `tilelang.debug.deviceDebug=false`, or pass `--no-device-debug` to the CLI, to append
`-lineinfo` instead. This provides faster line-only debugging. `-G` increases compilation and
execution overhead and can change the behavior of kernels that depend on optimized execution.

### JIT and pending breakpoints

TileLang kernels are generated and loaded while the Python script is running, so their symbols
do not exist when the debugger starts. The extension sets `set breakpoint pending on`, allowing
Python source breakpoints to resolve after the kernel is loaded.

`tilelang.debug.breakOnLaunch=true` (the default) also sets
`set cuda break_on_launch application`. This is a useful fallback when no source breakpoint is
available, but framework kernels such as PyTorch kernels may stop first. Continue to the target
TileLang kernel, or disable this option and rely only on Python pending breakpoints.

### Cache and generated sources

The launcher sets `TILELANG_DISABLE_CACHE=1` by default and forces
`TILELANG_ENABLE_IR_SPAN=1` when Python mapping is enabled. This prevents external settings or
autotune result caches from bypassing the debug build. `--keep-cache` restores caching, with the
corresponding reuse risk left to the user.

Generated artifacts are written under `.tlsp/debug/generated`:

- `debug_root_path` stores JIT `.c` source copies;
- `cuda.kernels_output_dir` stores the actual `.cu` and binary files used by NVCC on paths that
  use that compiler.

The launcher preserves quotes and spaces in original compile-flag strings. It parses only a
read-only copy to detect existing flags, preventing downstream splitting from corrupting macro
definitions or paths.

## Architecture and code entry points

```text
VS Code
  └─ TileLang: Debug CUDA Kernel
       └─ cuda-gdb DAP from Nsight VSCE
            └─ cuda-gdb
                 └─ python -m tlsp.debug --debug-dir .tlsp/debug -- kernel.py
                      ├─ Disable cache and enable IR spans
                      ├─ tl.emit_line_directives = True
                      ├─ compile_flags += -G (or -lineinfo)
                      ├─ Set JIT/NVCC artifact directories
                      └─ runpy.run_path(kernel.py)
```

- `tlsp/debug.py`: patches `tilelang.jit` and `tilelang.compile` before running the user script;
- `tilelang_lsp_extension/src/cudaDebug.ts`: pure launch configuration, directory resolution,
  and generated-source discovery;
- `tilelang_lsp_extension/src/extension.ts`: editor commands, Nsight activation, preflight
  checks, and debug-session startup;
- `tests/test_debug.py`: option, environment, and fake-TileLang subprocess tests that require
  neither TileLang nor a GPU;
- `tests/test_debug_cuda.py`: explicitly enabled real cuda-gdb acceptance;
- the extension contributes Python breakpoint support, while Nsight provides the DAP.

## Usage

### VS Code

1. Install CUDA Toolkit and Nsight on the Linux GPU host. Set `tilelang.pythonPath` to an
   interpreter that imports TileLang and tlsp. For Remote-SSH/WSL, install all three components
   on the remote side.
2. Optionally review these settings in the remote or workspace scope:

   ```json
   {
     "tilelang.pythonPath": "/path/to/venv/bin/python",
     "tilelang.debug.cudaGdbPath": "/usr/local/cuda/bin/cuda-gdb",
     "tilelang.debug.deviceDebug": true,
     "tilelang.debug.breakOnLaunch": false
   }
   ```

3. Open a Python script that actually executes a kernel, not merely one that defines it. Set a
   breakpoint with F9 on `with T.Kernel(...)`, `T.copy(...)`, or a device computation line.
4. Run **TileLang: Debug CUDA Kernel** from the Command Palette or editor menu. Do not use a
   regular Python debugger configuration.
5. If a framework kernel stops first, continue to the TileLang kernel. Inspect CUDA focus,
   `threadIdx`, local variables, registers, and disassembly. This session debugs device code,
   not Python host-interpreter stepping.
6. Run **TileLang: Open Generated CUDA Source** to inspect the `.c` and `.cu` artifacts. The
   command searches `.tlsp/debug/generated` recursively and offers multiple files in
   modification-time order.

Settings include `tilelang.debug.cudaGdbPath`, `deviceDebug`, `breakOnLaunch`, `args`, `extraEnv`,
and `debugDir`. `extraEnv.PYTHONPATH` preserves the tlsp module path and merges user and
inherited entries.

### Use cuda-gdb directly

Start the script through `tlsp.debug` under cuda-gdb, using an interpreter that can import both
`tilelang` and `tlsp`:

```bash
cuda-gdb -q --args /path/to/venv/bin/python -m tlsp.debug \
  --debug-dir .tlsp/debug -- /absolute/path/to/kernel.py
```

At the `(cuda-gdb)` prompt, enter the following commands. Replace `42` with the line containing
the desired TileLang device statement:

```gdb
set breakpoint pending on
break /absolute/path/to/kernel.py:42
run
info cuda kernels
info cuda threads
print threadIdx.x
next
disable
continue
```

Pending breakpoints resolve after the JIT kernel loads. `next` steps through device code at
Python-line granularity. `disable` prevents another stop for other CUDA blocks before continuing.
As a fallback, run `set cuda break_on_launch application` before `run`, but this also stops on
framework kernels.

Add `--no-device-debug` before `--` for faster line-only builds. Append user-script arguments
after the script path. The session still debugs device code, not the Python host interpreter.

The launcher can also run without VS Code or cuda-gdb for build and numerical checks:

```bash
python -m tlsp.debug --debug-dir /tmp/tlsp-debug -- /path/to/kernel.py
# Line-only debugging:
python -m tlsp.debug --no-device-debug -- /path/to/kernel.py
# Older TileLang: debug generated CUDA without Python line mapping:
python -m tlsp.debug --no-line-directives -- /path/to/kernel.py
```

By default, a missing `tl.emit_line_directives` capability produces an explicit error instead of
incorrectly suggesting that Python breakpoints should work.

## Validation

### Checks without a GPU

```bash
python -m unittest discover tests -v
cd tilelang_lsp_extension
npm run test:cuda-debug
npm run lint
```

### Automated Linux + NVIDIA GPU acceptance

Prepare the toolchain and run the following commands from the repository root:

```bash
nvidia-smi
nvcc --version
cuda-gdb --version

export TLSP_TEST_PYTHON=/path/to/venv/bin/python
# Also set this when using a TileLang source checkout not installed in that environment:
export TLSP_TEST_TILELANG=/path/to/tilelang
export TLSP_TEST_CUDA_GDB=/usr/local/cuda/bin/cuda-gdb

"$TLSP_TEST_PYTHON" -m unittest tests.test_debug_cuda -v
```

The test sets a Python pending breakpoint on a device computation line in
`tests/fixtures/elementwise_add.py`, without launch-stop, and asserts that:

- the JIT binary loads and execution stops at the original Python filename and line;
- `info cuda kernels` shows the target kernel and `print threadIdx.x` returns an integer;
- `next` advances from Python line 23 to line 24;
- disabling the breakpoint and continuing produces the numerical-success marker;
- generated sources contain the corresponding `#line`.

This automated test passed on the RTX 5090/CUDA 13.0 environment. A separate native CUDA smoke
test first verified cuda-gdb device breakpoints, thread variables, and numerical results, ruling
out a container or driver that fundamentally could not debug CUDA. The test is explicitly skipped
when `TLSP_TEST_CUDA_GDB` is unset on ordinary development machines and CI.

### VS Code manual checklist

- Confirm that the Python breakpoint actually stops after JIT loading; do not rely only on its
  initial `verified` state.
- At the stop, run `-exec info cuda kernels`, `-exec info cuda threads`, and
  `-exec print threadIdx.x` in the Debug Console.
- Step through device code and confirm that the editor shows the original Python file rather
  than a temporary `.cu` file.
- Confirm that **Open Generated CUDA Source** finds JIT `.c` and NVCC `.cu` files.
- If the breakpoint misses, inspect generated `#line` directives first. After the target kernel
  loads, run `-exec break /path/to/kernel.py:42` in the Debug Console to distinguish a pending
  breakpoint issue from missing device line information.

## Support boundaries

- The primary paths are regular CUDA `@tilelang.jit` and `tilelang.compile`; the default
  `tvm_ffi` backend has passed GPU acceptance. `par_compile` can indirectly use the compile
  hook but does not receive JITImpl source copies.
- Non-grouped autotune usually passes through these entry points. **Grouped autotune bypasses
  the hooks, can lose `-G` and line mapping, and is not currently supported.**
- Direct `JITKernel` construction, direct cache/lower calls, and separate Python compilation
  subprocesses are not managed by this monkeypatch. CuTeDSL and raw CUDA source kernels do not
  use this C-codegen line mapping.
- cython and NVRTC have relevant flag paths but were not included in the first end-to-end
  acceptance matrix. Do not extrapolate the default-backend result to every execution backend.
- Mapping has no column-level precision, and a `-G` build can behave differently from an
  optimized build.
- Use a separate Python debug session for Python host breakpoints; this session targets CUDA
  device code.

Next priorities are adding the automated test to an NVIDIA GPU CI runner, covering more
execution backends, improving upstream span granularity, providing an official debug
configuration entry point, and integrating compute-sanitizer diagnostics.
