# DeepForge 用户指南

[English](user-guide.en.md)

本文面向使用 DeepForge CPU runtime 编译和执行 cuDNN Frontend 序列化 Graph 的用户。
设计和实现细节见 [设计文档](design/overview.md)。

## 1. 支持范围

DeepForge `0.1.0` 当前支持：

| 项目 | 支持范围 |
|---|---|
| 平台 | Linux x86-64 |
| 输入 | cuDNN Frontend `v1.24.0` 生成的 Graph JSON 或 canonical UBJSON |
| Graph schema | `json_version == "1.0"`，`cudnn_frontend_version == 12400` |
| 算子 | v1.24.0 全部 39 个 serialized tag 的已验证静态 CPU 子集 |
| 通用 Tensor | 静态 rank 1-64 f32 data、显式 UID；文档指定的 C4 metadata 可为 INT32/INT64；支持 virtual 中间值 |
| 通用布局 | 正且不重叠的任意 stride；无 reorder/ragged metadata |
| C5 特殊 storage | 文档指定端口支持 FLOAT16、BFLOAT16、FP8 E4M3/E5M2/E8M0、packed FP4 E2M1/INT4 及 FLOAT control |
| Conv | rank 3-5 FPROP/DGRAD/WGRAD、group channel、正 stride/dilation、非负非对称 padding、两种 math mode |
| CPU 代码 | scalar、AVX2+FMA、AVX-512F+FMA，运行时自动分发 |
| 输出 | LLVM IR 或包含三个原生 object 的 `.dfo` artifact |

基础操作子集如下：

| Tag | 当前约束 |
|---|---|
| `RESHAPE` | 仅 `LOGICAL`，元素数相同 |
| `TRANSPOSE` | 完整静态 permutation |
| `SLICE` | 半开区间不越界，stride 为正整数 |
| `CONCATENATE` | 编号输入、非负 axis、无 in-place mode |
| `POINTWISE` | v1.24.0 全部 50 个 mode，尾维对齐的 NumPy broadcast |
| `REDUCTION` | 全部 9 个 mode；输入输出 rank 相同，被归约维度为 1 |
| `MATMUL` | 相同且 >= 2 的 rank、batch broadcast、无 M/N/K override、padding value 为 0 |
| `RESAMPLE` | 3 个 pooling mode 加整数 `NEAREST`；支持 3 个 padding mode 且无 index 输出。`BILINEAR` 因 v1.24.0 序列化丢失 fraction denominator 而被拒绝 |

C3 操作族如下：

| Tags | 当前约束 |
|---|---|
| `CONV_FPROP`, `CONV_DGRAD`, `CONV_WGRAD` | logical `[N,C,spatial...]` rank 3-5；group 数为 `X.C / W.C`；输出 shape 必须匹配 padding、stride、dilation |
| `BATCHNORM`, `BATCHNORM_INFERENCE`, `DBN`, `DBN_WEIGHT` | per-channel 参数/统计量；training running-stat 端口全有或全无；`peer_stats` 必须为空 |
| `GENSTATS`, `BN_FINALIZE` | per-channel sum/square sum 和 training-stat finalize |
| `INSTANCE_NORM`, `INSTANCE_NORM_BPROP` | per-channel 参数和 per-instance/channel 保存统计量 |
| `LAYER_NORM`, `LAYER_NORM_BPROP` | 归一化轴由同 rank broadcast scale shape 推导 |
| `RMS_NORM`, `RMS_NORM_BPROP` | RMS 统计量由 scale shape 推导；serializer 允许时 bias/bias gradient 可选 |
| `ADA_LAYER_NORM`, `ADA_LAYER_NORM_BPROP` | 保留 batch 统计量和 adaptive 同 rank 参数的 layer normalization |

C4 sequence/attention 支持如下：

| Tags | 当前约束 |
|---|---|
| `RNG` | Bernoulli f32 输出；fixed seed 或 scalar INT64 `Seed`/`Offset`；使用确定性的 DeepForge CPU stream |
| `ROPE`, `ROPE_BWD` | f32 BHSD split-half rotation，旋转完整末维或最后一个偶数宽度子空间；支持 `[S,1,1,R]` frequency 和 output scale |
| `SDPA`, `SDPA_BWD` | f32 BHSD、GQA、scalar scale、broadcast bias、ALiBi causal mask、INT32 sequence length、top-left/bottom-right window、custom/probability dropout、forward row 输出和 backward Q/K/V/bias gradient |

C5 特殊操作支持如下：

| Tags | 当前约束 |
|---|---|
| `BLOCK_SCALE_QUANTIZE`, `BLOCK_SCALE_DEQUANTIZE` | 静态且可整除的 block、FLOAT compute；声明端口支持 f32/f16/bf16 value 和 FP8/FP4/INT4 storage；FP4 按低/高 nibble 打包 |
| `MATMUL_FP8` | A/B 为 FP8 E4M3/E5M2、scalar FLOAT descale/output scale、rank >= 2 batch broadcast；C 可为 FP8/f32/f16/bf16，`Amax_C` 为 scalar FLOAT；无 M/N/K override |
| `MOE_GROUPED_MATMUL`, `MOE_GROUPED_MATMUL_BWD` | `mode=NONE`、`top_k` 为 0/1、Token `[1,T,K]`、Weight `[E,K,N]`、INT32 offset `[E,1,1]`，data 共享 f32/f16/bf16 类型 |
| `SDPA_FP8_FWD`, `SDPA_FP8_BWD` | 静态 FP8 E4M3/E5M2 BHSD、GQA、scalar FLOAT scale/descale、两种 diagonal window、Stats/amax；无 padding、dropout、ALiBi |
| `SDPA_MXFP8_FWD`, `SDPA_MXFP8_BWD` | 静态 BHSD/GQA、32 元素 E8M0 block descale、f16/bf16/f32 输出或梯度、backward transpose-oriented 输入、Stats/amax；scale tensor 当前采用逻辑 `NONE` ordering，backward dS 使用文档声明的 f32 CPU reference approximation |

Paged/cache attention、block mask、sink token、packed/ragged attention 和动态
shape 延后。C5 FP8 attention 还将 padding、dropout、ALiBi、可选端口和 producer
生成的 `F8_128x4` scale reorder 延后到 C6。v1.24.0 标准 SDPA 的 bottom-right
causal 路径不与 bias、ALiBi 或 dropout 组合。CPU RNG 在 DeepForge variant 间可
复现，但不承诺匹配 cuDNN GPU Philox bit pattern。

Comparison、logical 和 generated-index pointwise 输出仍使用 f32 `0`/`1` 或 f32
index。连接 tensor 类型同时受两端操作支持时，C2-C5 tag 可在同一个图中混合。
不支持动态 shape、显式 alias、scalar pass-by-value、文档 C5 子集外的
ragged/reordered tensor、分布式 peer statistics、GPU 执行、CUDA device pointer、
AMX 或内部多线程。输入文件最大为 16 MiB。精确矩阵见
[schema 清单](cudnn-graph-schema-inventory.md#5-capability-含义)。

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

在仓库根目录构建完整编译器、CLI、benchmark 和测试：

```bash
./scripts/build.sh
```

可选安装：

```bash
./scripts/build.sh --install install
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

原有优化 `CONV_FPROP` 路径的逻辑维度和 packed stride 必须满足：

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

通用 C3 convolution 路径接受 rank 3-5 tensor 和正且不重叠的 stride。每个空间轴
的输出 extent 为：

```text
effective_filter = dilation * (filter_extent - 1) + 1
output_extent = 1 + (input_extent + pre + post - effective_filter) / stride
```

`W` 的 logical shape 为 `[K,C_per_group,filter...]`；group 数由
`X.C / C_per_group` 推导，且 `Y.K` 必须可被 group 数整除。

X、W、Y 必须有显式且互不重复的 UID。UBJSON 中的非空
`pass_by_values`、`workspace_modifications` 或 `variant_pack_replacements` 会被
拒绝，因为它们包含当前 CPU MVP 未实现的执行语义。

通用 C2-C5 图中每个被读写的非 virtual tensor 都必须出现在执行期 UID map 中。virtual
tensor 不放入 UID map，而是使用查询得到的 workspace。可写 buffer 不得与其他
argument 或 workspace 重叠。在 alias 语义实现前，`VIEW_ONLY`、
`in_place_index` 以及同一个 UID 同时作为某 node 输入输出都会被拒绝。tensor name
不能替代显式 UID。epsilon、momentum、accumulation count 等 normalization scalar
输入是与操作 rank 匹配、shape 全为 1 的显式 f32 tensor；scalar pass-by-value
serialization 仍延后。

SDPA 使用 rank-4 BHSD tensor。`SEQ_LEN_Q`/`SEQ_LEN_KV` 为 INT32
`[B,1,1,1]`；probability dropout 和 dynamic RNG 使用单元素 INT64 `Seed`/`Offset`
tensor。这些 metadata buffer 与 data 一样通过 UID map 提供 host pointer。Forward
`Stats` 保存 row log-sum-exp，并由 `SDPA_BWD` 读取；`Max`、`Sum_exp` 和
`RNG_DUMP` 是可选序列化输出。

## 5. 编译 artifact

在构建树中运行：

```bash
DEEPFORGE_COMPILE=build/tools/deepforge-compile

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

- variant-pack 必须提供 metadata 中每个非 virtual argument UID 对应的 host
  pointer；额外 UID 被忽略。
- 每个 argument 必须满足 metadata 记录的 alignment，并具有完整 tensor 容量；
  API 不携带 buffer length，运行时无法证明实际分配大小。
- 可写 argument 与 workspace 的有效地址区间不得重叠。
- workspace 大小来自 `get_workspace_size()`，非零时必须 64-byte 对齐。
- `FrontendHandle` 是为 Frontend 调用形状保留的 opaque `void*`，CPU runtime 不
  解引用它，可以传 null。
- 同一 `Executable` 的并发调用是安全的，但当前实现会在内部串行调用 kernel；
  每次调用仍须使用独立 output 和 workspace。

## 9. Benchmark

```bash
DEEPFORGE_BENCHMARK=build/tools/deepforge-benchmark
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
| `DFE_UNSUPPORTED_NODE` | 使用当前 capability matrix 列出的 tag |
| `DFE_UNSUPPORTED_OPERATION` | 移除延后 attribute、不支持的 peer statistics 或超出该 tag 已声明 CPU 子集的配置 |
| `DFE_INVALID_LAYOUT` | 检查 Conv packed stride，或基础 tensor 的正且不重叠 stride |
| `DFE_INVALID_SHAPE` | 检查静态维度、操作 shape 规则和 Conv 输出公式 |
| `DFE_INVALID_VARIANT_PACK` | 检查 UID、host pointer、对齐、别名和 workspace |
| `DFE_UNSUPPORTED_CPU_FEATURE` | 不要强制执行主机不支持的变体；使用自动 `execute` |
| artifact target 不匹配 | 在目标主机或相同 target triple 环境重新编译 |

规范子集见 [MVP 兼容性与运行契约](design/contracts.md) 和
[schema 清单](cudnn-graph-schema-inventory.md)，artifact 二进制字段见
[DFO Artifact 格式](artifact-format.md)。
