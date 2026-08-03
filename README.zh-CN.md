# DeepForge

[English](README.md) | [中文用户指南](docs/user-guide.zh-CN.md)

DeepForge 是一个基于 MLIR 的 CPU 编译器。它读取开源
`cudnn-frontend` 生成的序列化 Graph，将受支持的图降低为 LLVM IR 和
x86-64 目标代码，并以 cuDNN Frontend 的 UID variant-pack 方式执行。

**当前状态**：CPU MVP 的 P0-P6、MVP 后覆盖阶段 C0-C5 及前十二个 C6 增量已实现。从 strict
JSON/UBJSON importer、标准
Tensor/Linalg IR、唯一一次 One-Shot Bufferize 和静态 workspace planning，到
scalar/AVX2/AVX-512 LLVM object、CPUID 分发、Frontend-shaped runtime、可重新装载
的 `.dfo` artifact、CLI、benchmark 和 sanitizer 测试均已打通。v1.24.0 全部
39 个 serialized tag 都已有明确声明并经过测试的 CPU 执行子集；这些子集严格
小于 cuDNN backend 允许的完整配置空间。
优化 Conv 路径已具有可检查、target-aware 的 K-output unroll cost model，并保留
固定 baseline 回退。

**MVP**：静态、连续、f32 的单个 Conv2D FWD，目标为 x86-64，提供标量、
AVX2 和 AVX-512 三个代码变体。Machine Dialect、AMX、bf16、多线程及多算子
融合不在 MVP 内。

## 固定版本

| 依赖 | 固定版本/提交 | 状态 |
|---|---|---|
| LLVM/MLIR | `llvmorg-22.1.8` (`ca7933e47d3a`) | 已安装并验证 |
| cuDNN Frontend | `v1.24.0` (`c4a97621eca5`) | 已下载，作为 serialization reference |
| nlohmann/json | `3.11.3` | 随 Frontend 源码提供，路径见下表 |
| C++ language mode | `C++20` | CMake 强制启用 |

不支持以 LLVM `main`、MLIR 滚动版本或 LLVM 23 RC 构建。具体兼容范围见
[MVP 兼容性与运行契约](docs/design/contracts.md)。

## 外部工具链版本

下表是当前 P0 在 Linux x86-64 上验证过的工具链基线。带“固定”状态的版本属于
项目兼容性契约；带“验证”状态的版本是当前工作区成功构建所用的基线，CI 应优先
复用。CMake 的项目最低版本由 `CMakeLists.txt` 中的
`cmake_minimum_required` 约束。

| 工具/依赖 | 版本或约束 | 状态和用途 |
|---|---|---|
| 操作系统/架构 | Linux x86-64；glibc `2.35` | 已验证的主机基线 |
| Git | `2.34.1` | 已验证；获取固定 tag，不参与运行时 ABI |
| CMake | `4.4.0`；最低 `3.27` | 已验证；配置 LLVM/MLIR 和 DeepForge |
| Ninja | `1.10.1` | 已验证；构建生成器 |
| GNU C/C++ | `13.4.0`；C++ `20` | 已验证；主机代码和运行时编译 |
| GNU binutils/`ld` | `2.38` | 已验证；主机链接 |
| Python | `3.10.12`；LLVM 要求至少 `3.8` | 已验证；LLVM/MLIR 配置期工具 |
| zlib development package | `1.2.11` | 已验证；LLVM/MLIR 压缩支持 |
| LLVM/MLIR | `22.1.8`；X86 target | 固定且已安装；唯一支持的 MLIR/LLVM 工具链 |
| cuDNN Frontend | `v1.24.0` (`c4a97621eca5`) | 固定且已下载；serialization schema/fixture reference |
| nlohmann/json | `3.11.3` | 固定；Frontend vendored header，不依赖系统安装 |
| CUDA Toolkit/cuDNN backend | 不属于 MVP 依赖；当前未安装 | CPU importer/runtime 不需要；GPU execution 与 Frontend samples 延后 |

表中的“当前验证版本”不表示任意更高版本都兼容；变更 CMake、编译器、
binutils 或 parser 后，需要重新执行 P0 的 configure、MLIR smoke test 和
端到端测试。`LLVM/MLIR` 的版本检查是硬失败，不允许静默使用其他版本。

CMake 从 `PATH` 查找构建工具，DeepForge 不记录可执行文件的具体位置。在新环境
中先执行下面的探测命令；若 `cmake` 低于 `3.27`，应先调整 `PATH` 或安装满足
最低版本的 CMake：

```bash
command -v cmake ninja c++ ld python3 git
cmake --version | head -1
ninja --version
c++ --version | head -1
ld --version | head -1
python3 --version
git --version
```

CMake 不写死依赖位置。MLIR 通过 `find_package(MLIR CONFIG)` 发现，cuDNN
Frontend header 通过 `find_path` 发现。配置时将 LLVM/MLIR 安装前缀和 cuDNN
Frontend checkout 加入 `CMAKE_PREFIX_PATH`。非标准目录结构可以显式设置
`MLIR_DIR` 或 `DEEPFORGE_CUDNN_FRONTEND_INCLUDE_DIR`。

对 DeepForge 当前 CPU MVP，cuDNN Frontend 的依赖边界是“下载源码、读取协议和
复用 vendored JSON header”，不执行 Frontend 自身的 CMake 构建，也不链接 GPU
backend。Frontend 仓库的 C++ samples/tests 若要单独编译，仍需要 CUDA Toolkit；
其总头文件还会包含 `cudnn.h`。因此下载源码足够支持当前 importer/reference
工作，但不等于已经安装了可编译或可执行的 CUDA/cuDNN backend。

当前构建不会查找 CUDA。官方 `cudnnHandle_t`/`cudnn_frontend::error_t` exact
types、GPU execution 以及与 Frontend `Graph` object 的二进制互换不属于当前
CPU-only MVP，因此不锁定也不安装 CUDA/cuDNN backend 版本。

## 支持输入

输入必须满足 `json_version == "1.0"`、`cudnn_frontend_version == 12400`，单个
文件最大 16 MiB。当前可执行形式为：

- 原有优化的单节点 packed f32 rank-4 `CONV_FPROP` 路径；
- 使用 3 个 convolution tag、8 个 C2 基础 tag、14 个 C3
  normalization/statistics tag 和 5 个 C4 sequence/attention tag 的静态有序 DAG；
- 在精确约束内执行 block-scale conversion、FP8 matmul、FP8/MXFP8 attention 和
  MoE grouped matmul 的 9 个 C5 特殊 tag。精确约束见
  [schema capability matrix](docs/cudnn-graph-schema-inventory.md#5-capability-含义)。

通用路径支持 rank 3-5 grouped convolution、stride、dilation、非对称 padding、
FPROP/DGRAD/WGRAD 及 C2-C6 混合图；也支持 capability matrix 范围内的
normalization forward/backward、batch statistics、running-stat 更新、确定性
Bernoulli RNG、RoPE forward/backward 和 f32 SDPA forward/backward。C5 在 9 个
特殊 tag 所需端口加入软件 FLOAT16/BFLOAT16/FP8/FP4/INT4 conversion 和 packed
storage。C4 sequence metadata 使用 INT32 length 和 scalar INT64 seed/offset
tensor。支持 virtual workspace 中间值和正且不重叠的 strided layout。C6 已在文档
指定的 FP8 block-scale 和 E8M0 MXFP8 scale 端口解码 Frontend `F8_128x4` 物理布局，
并支持带 virtual 中间值、无 broadcast 的 exact-shape plain f32 纯 `POINTWISE` DAG，
以及单个 external plain A/B/C 标准 f32 `MATMUL` 的 runtime shape override。序列化
dimension 是上界，Frontend-shaped override array 提供不超过编译 storage bound 的正
runtime dimension/stride；MATMUL 还保持 M/N/K 关系和 batch broadcast。单个标准
f32 LOGICAL `RESHAPE` 也接受 rank 各自固定的 external plain X/Y descriptor；runtime
dimension 可缩小或重新分配元素，但 X/Y 元素总数必须相同。单个 dense 标准 f32
`SDPA` forward 也接受 external plain Q/K/V/O 及可选 row-output
descriptor；runtime B、Sq、Skv 可缩小，head、embedding、GQA 和跨 tensor 关系保持
固定并在执行前校验。该子集只支持无 mask 或 top-left causal，不包含其他可选
attention 特性。单独的 dynamic-shape context flag 会被保存在 plan metadata 中，
但不会让其他 operation 自动变为动态。标准 f32 `MATMUL` 与 `MATMUL_FP8` 还可
分别接收 producer 序列化的
external plain INT32
`M_override`、`N_override`、`K_override` tensor：rank 与 C 相同，末两个 matrix
dimension 为 1，每个 batch dimension 为 1 或对应的 C dimension。各 batch 的值在
序列化静态上界内选择输出与 reduction extent；标准 MATMUL 用有限 f32
`padding_value` 填充 M/N 无效区，MATMUL_FP8 填零。Runtime scalar
pass-by-value input 也可执行：tensor 必须是 external、plain、仅作为
input、所有 dimension 为 1 且没有内嵌 payload；调用者按普通 UID 提供 scalar 地址。
这与 Frontend 的 runtime 形式一致，不改变公开 ABI 或 artifact format。对应的 fused
形式要求 tensor 携带 v1.24.0 精确 `{index,value}` scalar variant，且根级
`pass_by_values` 包含完全相同的条目。DeepForge 从公开 argument table 移除该 UID，
并在每个生成 object 中将 graph-owned value lowering 为 private read-only global；
调用者既不提供也不能覆盖该值，现有 artifact format 直接将其作为 object data
持久化。静态 f32 SDPA
支持采用已校验 element-prefix offset 的 external ragged forward data/row output 和
backward data/gradient。Forward 还支持独立 paged K/V cache、带独立 prefix 的紧凑
INT32 page table、压缩 UINT8 block mask 和每个 query head 的 sink logit；backward
可输出 sink gradient。这些 storage 形式都要求 padding 和显式 sequence length。
固定使用的 v1.24.0 serialization schema 没有 paged backward page-table 端口，因此
该形式不属于当前输入契约。显式 alias、其他 tensor reorder、分布式 peer statistics
和 FP8/MXFP8 特殊 attention 路径的可选特性暂不可执行。

## 架构

```text
cuDNN Frontend serialized Graph (JSON or UBJSON)
        |
        v
DeepForge importer + support validation
        |
        v
MVP Conv: Tensor + Linalg    通用 C2-C6: MemRef + SCF + Math
        |  one-shot-bufferize once       |
        +----------------------+----------+
                               v
MemRef + Affine/SCF + Vector Dialect
        |  direct Conv2D cost model, C-vectorization, K-output unroll
        v
LLVM Dialect
        |  translate to LLVM IR, LLVM target code generation
        v
scalar / AVX2 / AVX-512 object code
        |
        v
DeepForge Executable::execute(uid_to_host_ptr, workspace)
```

DeepForge 逻辑集中在 import/support validation、语义 lowering、workspace planning
和运行时分发。IR 主干复用上游 MLIR 方言，不引入临时 `cudnn.*` 方言，也不经过
自定义 Machine Dialect。

## 设计文档

| 文档 | 内容 |
|---|---|
| [中文用户指南](docs/user-guide.zh-CN.md) | 构建、CLI、运行时 API、benchmark 和排错 |
| [English User Guide](docs/user-guide.en.md) | Build, CLI, runtime API, benchmark, and troubleshooting |
| [MVP 契约](docs/design/contracts.md) | 版本、输入格式、支持矩阵、Frontend-shaped ABI 和数值规则 |
| [总体架构](docs/design/overview.md) | 组件边界、IR 全景和关键决策 |
| [Tensor 层](docs/design/tensor-layer.md) | cuDNN Graph 导入、形状和 padding |
| [Linalg 层](docs/design/linalg-layer.md) | `linalg.conv_2d_nhwc_fhwc` 语义和布局 |
| [Loop+Schedule 层](docs/design/loop-schedule-layer.md) | Direct Conv SCF 循环、归约向量化、生效的 cost model 和延后 tiling 边界 |
| [Machine Dialect](docs/design/machine-dialect.md) | 延后原因和重新引入条件 |
| [Vector+LLVM 层](docs/design/vector-llvm-layer.md) | 完整 LLVM lowering 与 CPU 变体 |
| [Pass Pipeline](docs/design/pass-pipeline.md) | MVP pass 顺序、前后置条件和合法性检查 |
| [Conv2D 示例](docs/design/example-conv2d.md) | 端到端 IR 示例 |
| [DFO artifact](docs/artifact-format.md) | `.dfo` 二进制布局、ORC 装载和信任边界 |
| [性能基线](docs/benchmark-baseline.md) | small/medium/large profile 与复测方法 |
| [cuDNN Graph 全操作覆盖方案](docs/cudnn-graph-coverage-plan.md) | MVP 后的通用架构、操作阶段和验收门槛 |
| [cuDNN Graph Schema 清单](docs/cudnn-graph-schema-inventory.md) | v1.24.0 根字段、tensor、39 个 operation tag、端口、属性和 mode |

开发顺序、阶段退出条件和测试门见
[开发方案](docs/development-plan.md)；全操作扩展见
[cuDNN Graph 全操作覆盖方案](docs/cudnn-graph-coverage-plan.md)。

## 构建

固定版本依赖可被 CMake 发现后，在 DeepForge 仓库根目录执行：

```bash
./scripts/build.sh
```

只构建/测试 P1 importer，不查找 MLIR，也不需要 CUDA/cuDNN backend：

```bash
./scripts/build.sh --importer-only
```

脚本会根据自身位置确定源码目录，保留环境中已有的 `CMAKE_PREFIX_PATH`，并将
已设置的 `LLVM_INSTALL_PREFIX` 和 `CUDNN_FRONTEND_SOURCE_DIR` 作为额外搜索
前缀传给 CMake，不会假设依赖安装位置。常用选项如下：

```bash
./scripts/build.sh --build-type Debug --build-dir build-debug
./scripts/build.sh --sanitizer address --build-dir build-asan
./scripts/build.sh --jobs 8 --no-tests
./scripts/release-check.sh --jobs 2
./scripts/build.sh --help
```

首次准备依赖时，请自行选择源码、构建和安装目录。下面使用的变量不带项目默认
值，必须由当前环境显式设置：

```bash
: "${LLVM_SOURCE_DIR:?请设置 LLVM_SOURCE_DIR}"
: "${LLVM_BUILD_DIR:?请设置 LLVM_BUILD_DIR}"
: "${LLVM_INSTALL_PREFIX:?请设置 LLVM_INSTALL_PREFIX}"
: "${CUDNN_FRONTEND_SOURCE_DIR:?请设置 CUDNN_FRONTEND_SOURCE_DIR}"

# cuDNN Frontend serialization 参考源码（当前不构建、不安装 CUDA）
git clone --branch v1.24.0 --depth 1 \
  https://github.com/NVIDIA/cudnn-frontend.git \
  "$CUDNN_FRONTEND_SOURCE_DIR"

# LLVM/MLIR（只需构建一次）
git clone --branch llvmorg-22.1.8 --depth 1 \
  https://github.com/llvm/llvm-project.git \
  "$LLVM_SOURCE_DIR"
cmake -S "$LLVM_SOURCE_DIR/llvm" -B "$LLVM_BUILD_DIR" -G Ninja \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD=X86 \
  -DLLVM_BUILD_TESTS=OFF \
  -DMLIR_INCLUDE_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$LLVM_INSTALL_PREFIX"
cmake --build "$LLVM_BUILD_DIR" --target install -j

# P0 smoke test
export PATH="${LLVM_INSTALL_PREFIX}/bin:${PATH}"
llvm-config --version
mlir-opt --version
mlir-translate --version
llc --version
```

CMake 会校验 `LLVM_PACKAGE_VERSION`、cuDNN Frontend 和 vendored nlohmann/json；
版本不是 `22.1.8`、`1.24.0`、`3.11.3` 时直接失败。

## 编译和执行

生成可重新装载的三变体 artifact，并检查其 metadata：

```bash
build/tools/deepforge-compile test/fixtures/conv2d_f32_c17.json \
  --input-format=auto \
  --target=x86-64 \
  --emit=object \
  --dump-ir=imported:build/imported.mlir \
  --dump-ir=bufferized:build/bufferized.mlir \
  --dump-ir=llvm:build/llvm.mlir \
  -o build/conv2d.dfo

build/tools/deepforge-compile --inspect build/conv2d.dfo
build/tools/deepforge-benchmark \
  --profile=all --iterations=3 --schedule=both
```

`--emit=llvm-ir --variant=scalar|avx2|avx512` 可直接输出对应 LLVM IR。
`.dfo` 可以通过 `load_artifact_executable` 从文件或内存重新建立 ORC executable；
artifact 包含原生 object code，不是安全沙箱，只能装载可信来源的文件。

库接口如下：

```cpp
deepforge::compiler::CompileOptions options;
deepforge::compiler::CompilationResult compilation;
auto status = deepforge::compiler::compile_file("graph.json", options,
                                                compilation);
if (status.is_bad()) {
    // status.code() and status.message()
}

deepforge::runtime::VariantPack variant_pack = {
    {x_uid, x_host_ptr},
    {w_uid, w_host_ptr},
    {y_uid, y_host_ptr},
};

auto workspace = allocate_aligned(
    compilation.executable->get_workspace_size(), 64);
deepforge::runtime::FrontendHandle handle = nullptr;
status = compilation.executable->execute(handle, variant_pack,
                                         workspace.get());

std::unique_ptr<deepforge::runtime::Executable> loaded;
status = deepforge::compiler::load_artifact_executable("conv2d.dfo", loaded);
```

公开 execute 与 override workspace query 固定为 Frontend `v1.24.0` 的 handle、
UID map、workspace、override UID/shape/stride 调用形状；
CPU runtime 不解引用 handle。MVP 使用 CPU-only opaque handle 和 DeepForge status，
不把 CUDA/cuDNN backend 作为 public build dependency。
对外不暴露 memref descriptor 或生成 kernel 的裸指针签名，生成 kernel 是运行时
内部的隐藏符号。`DeepForge/Runtime/Executable.h` 本身不包含 MLIR、CUDA 或 cuDNN
header；编译器 API 仍按预期依赖固定的 MLIR 工具链。

## 路线图

| 阶段 | 内容 |
|---|---|
| P0-P2 | 已完成：工具链、strict importer、Tensor/Linalg IR |
| P3 | 已完成：One-Shot Bufferize 和静态 workspace |
| P4 | 已完成：scalar LLVM/object、JIT 和 runtime |
| P5 | 已完成：AVX2/AVX-512、tail、CPUID/XGETBV 分发 |
| P6 | 已完成：CLI、可装载 artifact、CI、benchmark 和质量门 |
| C0-C5 | 已完成：通用 graph/runtime 基础及全部 39 个 serialized tag 的已验证子集 |
| C6 | 进行中：`F8_128x4`、多节点 exact-pointwise、单个标准 f32 MATMUL、LOGICAL RESHAPE 与 dense SDPA-forward descriptor override、MATMUL M/N/K extent override、runtime/embedded scalar pass-by-value、标准 f32 SDPA ragged/packed/block-mask/sink metadata 及首个 direct-Conv cost model 已完成；剩余已交付子集之外的 dynamic 行为 |
| Optimize | 进行中：target-aware K-output unroll 已完成；外层 tiling、padding fusion、并行化继续由 benchmark 驱动 |
| Re-evaluate | 至少出现两个后端的共同抽象需求后，再评估 Machine Dialect |

## 参考

- [MLIR](https://mlir.llvm.org/)
- [cuDNN Frontend](https://github.com/NVIDIA/cudnn-frontend)
- [LLVM releases](https://github.com/llvm/llvm-project/releases)

## 许可

Apache 2.0，见 [LICENSE](LICENSE)。
