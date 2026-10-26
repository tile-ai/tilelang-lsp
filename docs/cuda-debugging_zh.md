# TileLang CUDA 调试：实现与验证

[English version](./cuda-debugging.md)

## 当前状态

本仓库实现了实验性 CUDA 调试入口：复用 NVIDIA Nsight VS Code Edition 的
`cuda-gdb` 调试适配器，把 TileLang 生成的设备代码映射回用户编写的 Python 行。
无需自研或 fork DAP，也不需要修改 TileLang 源码。

已完成源码链路核验、启动器单元测试、扩展配置测试、TypeScript 编译和 lint，并在
Ubuntu 24.04 + RTX 5090（sm_120a）+ NVCC/CUDA-GDB 13.0 + TileLang 0.1.15 上完成真实
GPU 验收：Python pending breakpoint 命中、CUDA kernel/线程信息、`threadIdx.x`、
设备单步（Python 23 行到 24 行）和继续运行后的数值检查均通过。CUBIN 的 DWARF 行表
也确认包含原始 Python 文件和行号。

VS Code Remote-SSH + Nsight 的手动调试也已确认正常；下方保留 UI 验收步骤供复现。
Nsight 配置另由源码和 Node 测试核对。

运行前提：

- Linux（本机或 Remote-SSH/WSL）+ NVIDIA GPU + CUDA Toolkit（含 nvcc、cuda-gdb）；
- VS Code 安装 NVIDIA Nsight Visual Studio Code Edition；
- TileLang >= 0.1.15（包含 `PassConfigKey.TL_EMIT_LINE_DIRECTIVES` 和对应原生 codegen）；
- debuggee 的 Python 环境能 `import tilelang` 和 `import tlsp`。

## 工作原理

### Python 行映射

TileLang 的 `CodeGenTileLangCBase` 在 `tl.emit_line_directives=True` 时，为带 span 的
语句输出：

```c
#line 42 "/path/to/kernel.py"
```

nvcc 将这些逻辑文件名和行号带入设备调试行表，cuda-gdb 因而可以把设备停点显示为
`kernel.py:42`。映射是语句级、尽力而为的：一个 TileLang 算子可能展开成多条 CUDA
语句；无 span 的合成语句会继承前一条映射，复杂 kernel 中相邻 Python 行可能共享或漂移。

上游四项 codegen 测试（含 CUDA source）已在 GPU 环境通过，覆盖映射开启、默认关闭、
同一 Python 行的重复 directive 和 CUDA Python 行映射。

### 设备调试信息

`tilelang.compile` / `tilelang.jit` 的 `compile_flags` 会汇入
`tl.device_compile_flags`，最终传给设备编译器。启动器默认追加 `-G`，保留设备局部变量和
完整调试信息；固定存在的 host `-O3` 不会抵消 `-G` 关闭设备优化的语义。

`tilelang.debug.deviceDebug=false` 或 CLI 的 `--no-device-debug` 改为追加 `-lineinfo`，
用于只关心行号的较快调试。`-G` 会显著增加编译和运行开销，也可能改变依赖优化行为的
kernel 的表现。

### JIT 与 pending breakpoint

TileLang kernel 在 Python 脚本运行时才生成并加载 cubin，因此调试器启动时还没有目标符号。
扩展设置 `set breakpoint pending on`，让 Python 源行断点在 kernel 加载后解析。

`tilelang.debug.breakOnLaunch=true`（默认）还会设置
`set cuda break_on_launch application`。它可用于没有预设断点时的兜底，但 PyTorch 等框架
kernel 可能先停；继续到目标 TileLang kernel，或关闭该选项后仅依赖 Python pending breakpoint。

### 缓存与生成源码

启动器默认设置 `TILELANG_DISABLE_CACHE=1`，并在启用 Python 映射时强制
`TILELANG_ENABLE_IR_SPAN=1`，避免外部环境或 autotune 结果缓存绕过 debug 构建。
`--keep-cache` 可恢复缓存，但调试时应自行承担复用风险。

生成产物统一放在 `.tlsp/debug/generated`：

- `debug_root_path` 保存 JIT 的 `.c` 源码副本；
- `cuda.kernels_output_dir` 保存 NVCC 实际使用的 `.cu` / binary（适用于走该 NVCC 路径的编译）。

启动器会保留原始 compile flag 字符串的引号和空格，只解析只读副本来检测已有 flags，
避免下游再次拆分时破坏宏定义或路径。

## 架构与代码入口

```text
VS Code
  └─ TileLang: Debug CUDA Kernel
       └─ Nsight VSCE 的 cuda-gdb DAP
            └─ cuda-gdb
                 └─ python -m tlsp.debug --debug-dir .tlsp/debug -- kernel.py
                      ├─ 禁用缓存、启用 IR span
                      ├─ tl.emit_line_directives = True
                      ├─ compile_flags += -G（或 -lineinfo）
                      ├─ 设置 JIT / NVCC 生成产物目录
                      └─ runpy.run_path(kernel.py)
```

- `tlsp/debug.py`：在用户脚本前 patch `tilelang.jit` / `tilelang.compile`，再执行脚本；
- `tilelang_lsp_extension/src/cudaDebug.ts`：纯 launch 配置、目录解析和生成源码查找；
- `tilelang_lsp_extension/src/extension.ts`：编辑器命令、Nsight 激活和调试会话启动；
- `tests/test_debug.py`：不需要 TileLang/GPU 的选项、环境与 fake-TileLang 子进程测试；
- `tests/test_debug_cuda.py`：显式 opt-in 的真实 cuda-gdb 验收；
- 扩展声明 Python breakpoint，使用户可以在 `.py` 文件中按 F9；Nsight 仍负责 DAP。

## 使用

1. 在 Linux GPU 端安装 CUDA Toolkit 和 Nsight，并让 `tilelang.pythonPath` 指向能导入
   TileLang/tlsp 的解释器。Remote-SSH/WSL 时这些组件都必须位于远端。
2. 打开 kernel Python 文件，在 `with T.Kernel(...)`、`T.copy(...)` 或设备计算语句上按 F9。
3. 执行 **TileLang: Debug CUDA Kernel**。
4. 若先停在框架 kernel，Continue 到 TileLang kernel；随后检查 CUDA focus、`threadIdx`、
   局部变量、寄存器和 disassembly。这里调试的是设备代码，不是 Python 解释器宿主单步。
5. 用 **TileLang: Open Generated CUDA Source** 对照本次 `.c` / `.cu` 产物；该命令会递归查找
   `.tlsp/debug/generated`，有多个文件时按修改时间提供选择。

设置项：`tilelang.debug.cudaGdbPath`、`deviceDebug`、`breakOnLaunch`、`args`、`extraEnv`、
`debugDir`。`extraEnv.PYTHONPATH` 会保留 tlsp 模块路径并与用户/继承路径合并。

CLI 也可独立于 VS Code 使用：

```bash
python -m tlsp.debug --debug-dir /tmp/tlsp-debug -- /path/to/kernel.py
# 仅行号：
python -m tlsp.debug --no-device-debug -- /path/to/kernel.py
# 旧版 TileLang 仅调生成 CUDA（没有 Python 行映射）：
python -m tlsp.debug --no-line-directives -- /path/to/kernel.py
```

默认缺少 `tl.emit_line_directives` 会明确报错，防止用户误以为 Python 断点应当可用。

## 验证

### 无 GPU 检查

```bash
python -m unittest discover tests -v
cd tilelang_lsp_extension
npm run test:cuda-debug
npm run lint
```

### Linux + NVIDIA GPU 自动验收

在仓库根目录准备工具链，然后运行：

```bash
nvidia-smi
nvcc --version
cuda-gdb --version

export TLSP_TEST_PYTHON=/path/to/venv/bin/python
# 使用 TileLang 源码 checkout 且未安装到该环境时再设置：
export TLSP_TEST_TILELANG=/path/to/tilelang
export TLSP_TEST_CUDA_GDB=/usr/local/cuda/bin/cuda-gdb

"$TLSP_TEST_PYTHON" -m unittest tests.test_debug_cuda -v
```

该测试在 `tests/fixtures/elementwise_add.py` 的设备计算行设置 Python pending breakpoint，
不开启 launch-stop，随后断言：

- JIT binary 加载后实际停在原始 Python 文件和行号；
- `info cuda kernels` 能看到目标 kernel，`print threadIdx.x` 返回整数；
- `next` 从 Python 23 行前进到 24 行；
- 禁用断点并继续后，输出数值检查成功标记；
- 生成源码中存在对应的 `#line`。

上述自动测试已在 RTX 5090/CUDA 13.0 环境通过。另一个原生 CUDA smoke 也先验证了
cuda-gdb 设备断点、线程变量和数值结果，排除了容器或驱动本身不支持调试的可能。
普通开发机与 CI 未设置 `TLSP_TEST_CUDA_GDB` 时，该测试明确跳过。

### VS Code 手动验收

- Python 断点在 JIT 加载后实际命中；不要只凭初始 `verified` 状态判断成功；
- 停住时执行 `info cuda kernels`、`info cuda threads`、`print threadIdx.x`；
- 设备单步显示原始 Python 文件，而不是临时 `.cu`；
- Open Generated CUDA Source 能找到 JIT `.c` 和 NVCC `.cu`；
- 若断点不命中，先检查生成源码的 `#line`，再在目标 kernel 加载后用 Debug Console 执行
  `-exec break /path/to/kernel.py:42`，区分 pending 断点问题与行表缺失问题。

## 支持边界

- 重点支持常规 CUDA `@tilelang.jit` 和 `tilelang.compile`，默认 `tvm_ffi` 后端已通过
  GPU 验收。`par_compile` 可间接经过 compile hook，但不具有 JITImpl 的源码副本。
- 非 grouped autotune 通常会经过上述入口；**grouped autotune 绕过这些 hook，可能丢失
  `-G` 和行映射，当前不支持**。
- 直接构造 `JITKernel`、直接调用 cache/lower、独立 Python 编译子进程不受当前 monkeypatch
  管理。CuTeDSL 和原始 CUDA source kernel 不走这套 C codegen 行映射。
- cython/NVRTC 的参数路径存在，但未纳入首个端到端验收矩阵；不要把默认后端的验证结果
  外推到所有 execution backend。
- 映射没有列级精度，且 `-G` 构建可能与优化构建行为不同。
- Python 宿主断点请另开 Python 调试会话；当前会话面向 CUDA 设备代码。

后续优先事项是把自动测试纳入有 NVIDIA GPU 的 CI runner、覆盖更多 execution backend，
再考虑更细的上游 span 映射、官方 debug 配置入口及 compute-sanitizer 诊断集成。
