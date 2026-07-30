# DeepForge 用户指南

[English](user-guide.en.md)

本文面向使用 DeepForge CPU MVP 编译和执行 cuDNN Frontend 序列化 Graph 的用户。
设计和实现细节见 [设计文档](design/overview.md)。

## 1. 支持范围

DeepForge `0.1.0` 当前支持：

| 项目 | 支持范围 |
|---|---|
| 平台 | Linux x86-64 |
| 输入 | cuDNN Frontend `v1.24.0` 生成的 Graph JSON 或 canonical UBJSON |
| Graph schema | `json_version == "1.0"`，`cudnn_frontend_version == 12400` |
| 算子 | 恰好一个 `CONV_FPROP` |
| Tensor | 静态 rank-4、f32、非 virtual、非 pass-by-value |
| 布局 | packed NHWC X/Y，packed KRSC W |
| Conv | cross-correlation，stride 1，dilation 1，静态非负 padding |
| CPU 代码 | scalar、AVX2+FMA、AVX-512F+FMA，运行时自动分发 |
| 输出 | LLVM IR 或包含三个原生 object 的 `.dfo` artifact |

不支持动态 shape、group/depthwise Conv、bias/activation fusion、其他算子、GPU
执行、CUDA device pointer、bf16、AMX 或内部多线程。输入文件最大为 16 MiB。

CUDA Toolkit 和 cuDNN backend 不是依赖。项目只使用开源 `cudnn-frontend`
源码中的 serialization 协议和 vendored nlohmann/json header。

## 2. 准备环境

固定和验证过的依赖如下：

| 依赖 | 版本/要求 |
|---|---|
| LLVM/MLIR | 必须精确为 `llvmorg-22.1.8` |
| cuDNN Frontend source | 必须为 `v1.24.0` |
| nlohmann/json | Frontend vendored `3.11.3` |
| CMake | 至少 `3.27` |
| C++ compiler | 支持 C++20；已验证 GCC `13.4.0` |
| Ninja | 已验证 `1.10.1` |

请为当前环境自行选择依赖位置；DeepForge 不提供与具体机器绑定的默认值：

```bash
: "${LLVM_INSTALL_PREFIX:?请设置 LLVM_INSTALL_PREFIX}"
: "${CUDNN_FRONTEND_SOURCE_DIR:?请设置 CUDNN_FRONTEND_SOURCE_DIR}"
```

新环境只需下载 cuDNN Frontend 源码，不需要构建它：

```bash
git clone --branch v1.24.0 --depth 1 \
  https://github.com/NVIDIA/cudnn-frontend.git \
  "$CUDNN_FRONTEND_SOURCE_DIR"
```

LLVM/MLIR 的完整构建命令见 [中文 README](../README.zh-CN.md#构建)。CMake 通过
`CMAKE_PREFIX_PATH` 发现两个依赖，并拒绝非 `22.1.8` 的 MLIR 和版本不匹配的
Frontend/JSON header。

## 3. 构建和安装

构建完整编译器、CLI、benchmark 和测试：

```bash
cmake -S . -B build-full -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${LLVM_INSTALL_PREFIX};${CUDNN_FRONTEND_SOURCE_DIR}" \
  -DDEEPFORGE_BUILD_TESTS=ON \
  -DDEEPFORGE_BUILD_TOOLS=ON
cmake --build build-full -j
ctest --test-dir build-full --output-on-failure
```

可选安装：

```bash
cmake --install build-full --prefix install
export PATH="$PWD/install/bin:$PATH"
```

当前安装规则只发布 `deepforge-compile` 和 `deepforge-benchmark`。库 header、
静态库和 CMake package 尚未作为稳定 SDK 安装；嵌入式 API 应通过源码树的
CMake target 使用。

## 4. 准备输入

输入应由固定版本 Frontend 的以下接口产生：

- `Graph::serialize(nlohmann::json&)` 产生 JSON。
- `Graph::serialize(std::vector<uint8_t>&)` 产生 UBJSON。

DeepForge 不定义私有 JSON schema。可以先用仓库 fixture 验证安装：

```bash
cp test/fixtures/conv2d_f32_c17.json /tmp/graph.json
```

逻辑维度和 packed stride 必须满足：

| Tensor | 逻辑 dim | packed stride |
|---|---|---|
| X | `[N,C,H,W]` | `[H*W*C,1,W*C,C]` |
| W | `[K,C,R,S]` | `[R*S*C,1,S*C,C]` |
| Y | `[N,K,P,Q]` | `[P*Q*K,1,Q*K,K]` |

输出空间大小为：

```text
P = H + pre_h + post_h - R + 1
Q = W + pre_w + post_w - S + 1
```

X、W、Y 必须有显式且互不重复的 UID。UBJSON 中的非空
`pass_by_values`、`workspace_modifications` 或 `variant_pack_replacements` 会被
拒绝，因为它们包含当前 CPU MVP 未实现的执行语义。

## 5. 编译 artifact

在构建树中运行：

```bash
DEEPFORGE_COMPILE=build-full/tools/deepforge-compile

"$DEEPFORGE_COMPILE" /tmp/graph.json \
  --input-format=auto \
  --target=x86-64 \
  --emit=object \
  -o /tmp/conv2d.dfo
```

`--emit=object` 输出的是 `.dfo` 容器；容器内包含 scalar、AVX2 和 AVX-512
三个 object。省略 `-o` 时，输出路径为输入文件名替换成 `.dfo` 后缀。

编译 UBJSON：

```bash
"$DEEPFORGE_COMPILE" graph.ubjson \
  --input-format=ubjson \
  -o conv2d.dfo
```

退出码 `0` 表示成功，`1` 表示编译、I/O 或 artifact 错误，`2` 表示命令行参数
错误。输出文件通过同目录唯一临时文件和 rename 发布；并发写同一路径不会产生
半文件，但最终内容由最后一次成功发布决定。

## 6. 查看 IR

输出某个 CPU 变体的 LLVM IR：

```bash
"$DEEPFORGE_COMPILE" /tmp/graph.json \
  --emit=llvm-ir \
  --variant=avx2 \
  -o /tmp/conv2d-avx2.ll
```

编译 artifact 的同时保存各阶段 MLIR：

```bash
"$DEEPFORGE_COMPILE" /tmp/graph.json \
  --variant=avx512 \
  --dump-ir=imported:/tmp/imported.mlir \
  --dump-ir=bufferized:/tmp/bufferized.mlir \
  --dump-ir=llvm:/tmp/llvm.mlir \
  -o /tmp/conv2d.dfo
```

`--variant` 只选择 LLVM IR 输出或 `llvm` dump 的变体，不会减少 `.dfo` 中的三个
CPU 变体。

## 7. 检查 artifact

```bash
"$DEEPFORGE_COMPILE" --inspect /tmp/conv2d.dfo
```

输出包含格式版本、producer 版本、target triple、函数名、UID、workspace 大小和
三个 object 的符号、feature 要求及字节数。loader 要求 target triple 与当前主机
精确一致。

`.dfo` 包含会以当前进程权限执行的原生代码。FNV checksum 只能检测意外损坏，
不能证明来源可信；只应装载可信构建或可信发布渠道生成的 artifact。

## 8. C++ API

在 DeepForge 源码构建中，可以链接 `DeepForge::Compiler`：

```cmake
add_subdirectory(/path/to/DeepForge DeepForge-build)
target_link_libraries(my_app PRIVATE DeepForge::Compiler)
```

编译并执行 Graph 的最小示例：

```cpp
#include "DeepForge/Compiler/Codegen.h"

#include <cstdlib>
#include <iostream>
#include <vector>

static std::size_t elements(std::array<std::int64_t, 4> const& shape) {
    std::size_t count = 1;
    for (auto value : shape) count *= static_cast<std::size_t>(value);
    return count;
}

int main() {
    deepforge::compiler::CompileOptions options;
    deepforge::compiler::CompilationResult result;
    auto status = deepforge::compiler::compile_file("graph.json", options,
                                                     result);
    if (status.is_bad()) {
        std::cerr << status.message() << '\n';
        return 1;
    }

    std::vector<float> x(elements(result.metadata.x_shape), 1.0F);
    std::vector<float> w(elements(result.metadata.w_shape), 1.0F);
    std::vector<float> y(elements(result.metadata.y_shape));
    deepforge::runtime::VariantPack pack{
        {result.metadata.x_uid, x.data()},
        {result.metadata.w_uid, w.data()},
        {result.metadata.y_uid, y.data()},
    };

    auto workspace_size = result.executable->get_workspace_size();
    if (workspace_size < 0) return 1;
    void* workspace = workspace_size == 0
                          ? nullptr
                          : std::aligned_alloc(
                                64, static_cast<std::size_t>(workspace_size));
    if (workspace_size != 0 && workspace == nullptr) return 1;

    status = result.executable->execute(nullptr, pack, workspace);
    std::free(workspace);
    if (status.is_bad()) {
        std::cerr << status.message() << '\n';
        return 1;
    }
}
```

从文件恢复 executable：

```cpp
deepforge::compiler::ArtifactInfo info;
std::unique_ptr<deepforge::runtime::Executable> executable;
auto status = deepforge::compiler::load_artifact_executable(
    "conv2d.dfo", executable, &info);
```

运行时契约：

- variant-pack 必须提供 metadata 中 X、W、Y UID 对应的 host pointer；额外 UID
  被忽略。
- X、W、Y 至少按 `alignof(float)` 对齐，并具有完整 tensor 容量；API 不携带
  buffer length，运行时无法证明实际分配大小。
- X、W、Y 和 workspace 的有效地址区间不得重叠。
- workspace 大小来自 `get_workspace_size()`，非零时必须 64-byte 对齐。
- `FrontendHandle` 是为 Frontend 调用形状保留的 opaque `void*`，CPU runtime 不
  解引用它，可以传 null。
- 同一 `Executable` 的并发调用是安全的，但当前实现会在内部串行调用 kernel；
  每次调用仍须使用独立 output 和 workspace。

## 9. Benchmark

```bash
DEEPFORGE_BENCHMARK=build-full/tools/deepforge-benchmark
"$DEEPFORGE_BENCHMARK" --profile=all --iterations=3
```

profile 可选 `small`、`medium`、`large`、`all`，iterations 范围为 `[1,1000]`。
CSV 输出包含编译耗时、单次执行耗时、GFLOP/s，以及相对 scalar 结果的最大绝对和
相对误差。benchmark 是回归基线，不是跨机器可直接比较的性能承诺。

## 10. 常见错误

| 诊断 | 处理 |
|---|---|
| CMake 报 LLVM version mismatch | 将 `22.1.8` 前缀放在 `CMAKE_PREFIX_PATH` 首位，或显式设置 `MLIR_DIR` |
| 找不到 Frontend 或 JSON header | 将 `v1.24.0` checkout 加入 `CMAKE_PREFIX_PATH`，或显式设置 `DEEPFORGE_CUDNN_FRONTEND_INCLUDE_DIR` |
| `DFE_SCHEMA_VERSION_MISMATCH` | 使用 Graph JSON schema `1.0` |
| `DFE_FRONTEND_VERSION_MISMATCH` | 使用 cuDNN Frontend `v1.24.0` 重新序列化 |
| `DFE_UNSUPPORTED_NODE` | 将 Graph 限制为单个 `CONV_FPROP` |
| `DFE_INVALID_LAYOUT` | 检查 X/Y NHWC 和 W KRSC 的 packed stride |
| `DFE_INVALID_SHAPE` | 检查静态正维度和 Conv 输出公式 |
| `DFE_INVALID_VARIANT_PACK` | 检查 UID、host pointer、对齐、别名和 workspace |
| `DFE_UNSUPPORTED_CPU_FEATURE` | 不要强制执行主机不支持的变体；使用自动 `execute` |
| artifact target 不匹配 | 在目标主机或相同 target triple 环境重新编译 |

完整规范见 [MVP 兼容性与运行契约](design/contracts.md)，artifact 二进制字段见
[DFO Artifact 格式](artifact-format.md)。
