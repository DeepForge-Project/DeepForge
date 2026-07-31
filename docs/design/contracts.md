# MVP 兼容性与运行契约

[English](contracts.en.md)

本文档是 DeepForge 原始 Conv2D MVP 的规范性边界。MVP 后的算子扩展由本文、
[schema capability 清单](../cudnn-graph-schema-inventory.md) 和
[全操作覆盖方案](../cudnn-graph-coverage-plan.md) 共同约束；发生冲突时采用更严格的
适用契约。

## 1. 版本策略

| 组件 | 固定版本 | 策略 |
|---|---|---|
| LLVM/MLIR | `llvmorg-22.1.8` | CMake 必须精确匹配 |
| cuDNN Frontend | `v1.24.0` | importer schema/fixture 按该 tag 的 serializer 源码固化；其他 producer 版本在 MVP 中拒绝 |
| nlohmann/json | `3.11.3` | 与 Frontend vendored header 一致；JSON/UBJSON 均使用该 parser |
| CUDA/cuDNN backend | 无 | CPU-only MVP 不构建、不链接、不运行 GPU backend |
| cuDNN Graph JSON | `json_version == "1.0"` | 不匹配时拒绝并报告版本 |
| 平台 | Linux x86-64 | MVP 唯一执行平台 |

升级依赖必须单独提交，并重新生成序列化 fixture、IR golden test 和端到端数值
结果。不得静默接受未知 Graph schema。

## 2. cuDNN Frontend 输入契约

### 2.1 接受的载体

`deepforge-compile` 接受开源 cuDNN Frontend 原生生成的两种载体：

1. `Graph::serialize(nlohmann::json&)` 产生的 Graph JSON。
2. `Graph::serialize(std::vector<uint8_t>&)` 产生的 UBJSON blob。

在固定的 `v1.24.0` 源码中，vector 重载先调用 Graph JSON serialization，追加
plan/runtime metadata，再以 `nlohmann::json::to_ubjson` 把整个 JSON object 编码成
**单个 UBJSON 文档**。不存在额外 envelope，也不存在追加在 UBJSON 文档后的 GPU
plan bytes；`cudnn_backend_data` 是文档内的一个 JSON value。importer 必须以 strict
模式解析整个输入，并用同版本 parser 的默认 `to_ubjson` 结果做 canonical byte
校验，拒绝截断、非法 token、尾随 no-op、其他尾随字节或非默认 UBJSON 编码。

两种载体都必须满足 `json_version == "1.0"` 且数值字段
`cudnn_frontend_version == 12400`。`cudnn_backend_version` 只记录 producer 环境，
不参与 CPU 语义检查。UBJSON 可能比 Graph JSON 多出以下字段：

| 字段 | MVP 处理 |
|---|---|
| `cudnn_backend_data`, `behavior_notes` | 接受并忽略；不得解释为 CPU execution plan |
| `variant_pack_uids` | 若存在，必须与单 Conv 的非 virtual X/W/Y UID 集合一致 |
| `pass_by_values`, `workspace_modifications`, `variant_pack_replacements` | 必须不存在或为空；`variant_pack_replacements` 的空 object/array 均接受，非空表示 MVP 未支持的执行语义 |
| `fe_workspace_size` | 若存在必须为 0；CPU workspace 由 DeepForge 重新规划 |
| `tensors_to_dump` | 调试 metadata，接受并忽略 |

importer 直接使用结构化 JSON/UBJSON parser，不以字符串规则解析，也不定义
DeepForge 私有 JSON 格式。输入格式可通过 `--input-format=json|ubjson|auto`
指定；`auto` 先尝试严格 UTF-8 JSON，再尝试严格 UBJSON，只负责载体探测，不放宽
schema 校验。单个序列化 Graph 输入上限为 16 MiB；`parse_file` 在读取过程中也
执行该上限，避免为超限文件分配无界内存。

### 2.2 MVP serialized schema

固定版本 serializer 的 MVP 子集要求 root 为 object，并包含以下字段：

| 层级 | 必需形状 | MVP 约束 |
|---|---|---|
| root | `json_version`, `cudnn_frontend_version`, `graph_uid`, `context`, `nodes`, `tensors` | 版本分别为字符串 `"1.0"`、整数 `12400`；`nodes` 长度必须为 1 |
| node | `tag`, `inputs`, `outputs`, `compute_data_type`, `pre_padding`, `post_padding`, `stride`, `dilation`, `math_mode` | `tag == "CONV_FPROP"`；端口恰为 `X/W` 和 `Y`；dtype 为 `FLOAT`；二维参数长度为 2 |
| tensor map | object，key 为显式 UID 的十进制字符串 | key、`uid` 和 node reference 必须一致；MVP 只保留 X/W/Y 三个非 virtual tensor |
| tensor | `name`, `data_type`, `dim`, `stride`, `is_virtual`, `is_pass_by_value`, `reordering_type`, `uid`, `uid_assigned` | `FLOAT`、rank-4、`uid_assigned == true`、`is_virtual == false`、`is_pass_by_value == false`、reordering 为 `NONE` |

node reference 必须是 JSON/UBJSON integer UID；仅有 tensor name 的 fallback reference
不属于 MVP。JSON 与 UBJSON 只在载体层不同，解析后必须生成相同的 canonical model；
root 中列出的 execution/debug metadata 按 2.1 的规则处理，不能改变该 model 的语义。
importer 先在临时对象中完成全部校验，失败时不得部分修改调用者提供的输出 model。

### 2.3 “完整对接”与算子覆盖的区别

完整对接是指准确读取固定版本 cuDNN Frontend 的序列化协议、tensor UID、
dim、stride、data type、`is_virtual`、node input/output port 引用和 attributes。
`set_output(true)` 在该版本中序列化为 `is_virtual == false`，没有独立 output
字段；Y 由 `CONV_FPROP.outputs.Y` 确定。完整对接不代表 MVP 实现所有 cuDNN 算子。

解析器应能遍历任意合法节点，并对不支持的节点返回结构化诊断，例如：

```text
DFE_UNSUPPORTED_NODE: node[2] tag=POINTWISE is not supported by MVP
```

不得把不认识的节点跳过后继续编译。

### 2.4 UID 约束

输入、权重和输出 tensor 必须具有显式、互不重复且稳定的 UID。运行时
variant-pack 以 UID 绑定地址。仅有 tensor name、依赖反序列化时自动分配 UID
的图不满足可复现编译要求，应在 importer 阶段拒绝。

## 3. Conv2D MVP 支持矩阵

| 项目 | MVP 支持范围 |
|---|---|
| Graph | 恰好一个 `CONV_FPROP` node |
| Tensor rank | X、W、Y 均为静态 rank-4 |
| Data type | X、W、Y 和 compute 均为 f32 |
| Conv mode | cross-correlation |
| X logical dims | cuDNN `[N, C, H, W]` |
| W logical dims | cuDNN `[K, C, R, S]` |
| Y logical dims | cuDNN `[N, K, P, Q]` |
| Physical layout | packed NHWC input/output，packed KRSC filter |
| Spatial stride | `[1, 1]` |
| Dilation | `[1, 1]` |
| Padding | 静态、非负，允许 pre/post 不对称 |
| Grouping | 不支持 grouped/depthwise convolution |
| Fusion | 不支持 bias、activation 或其他节点融合 |
| Shapes | 所有维度大于 0；C/K 无整除要求 |

cuDNN 以逻辑 NCHW/KCRS 维度配合 stride 描述物理 NHWC/KRSC。importer 必须
验证精确 packed stride，并转换为 MLIR 的物理维度顺序：

| Tensor | cuDNN dim | cuDNN packed stride | MLIR tensor shape |
|---|---|---|---|
| X | `[N,C,H,W]` | `[H*W*C,1,W*C,C]` | `[N,H,W,C]` |
| W | `[K,C,R,S]` | `[R*S*C,1,S*C,C]` | `[K,R,S,C]` |
| Y | `[N,K,P,Q]` | `[P*Q*K,1,Q*K,K]` | `[N,P,Q,K]` |

因此 W 对应 Linalg `FHWC`，其中 `F` 是输出 feature，也就是 K：
`[F,H,W,C] == [K,R,S,C]`。`[R,S,C,K]` 属于 `HWCF`，不能传给
`linalg.conv_2d_nhwc_fhwc`。

输出空间维度必须满足：

```text
P = H + pre_h + post_h - R + 1
Q = W + pre_w + post_w - S + 1
```

若序列化 Y 的 dim 与推导结果不一致，编译失败。

所有维度乘积、packed stride、元素字节数和 workspace offset 都必须使用 checked
arithmetic 计算；任何超出 `int64_t`/`size_t` 可表示范围的输入返回
`DFE_DIMENSION_OVERFLOW`，不能在后续 MLIR index 或指针计算中静默回绕。

### 3.1 MVP 后 C2 扩展

C2 在不放宽 Conv 契约的前提下增加第二种可执行图：只由 `RESHAPE`、
`TRANSPOSE`、`SLICE`、`CONCATENATE`、`POINTWISE`、`REDUCTION`、
`MATMUL` 和 `RESAMPLE` 构成的静态 f32 DAG。精确 mode、shape、layout、
attribute 和 alias 约束以
[schema capability 清单](../cudnn-graph-schema-inventory.md#5-capability-含义)中
已验证行的声明为准。

C2 tensor 必须具有正的静态维度、正且不重叠的 stride、显式 UID、`NONE`
reorder，并且不是 pass-by-value 或 ragged tensor。virtual tensor 使用静态规划的
workspace。`VIEW_ONLY` reshape、in-place concatenate、同 UID
输入输出、MATMUL 维度 override 或非零 padding、RESAMPLE index 输出仍不支持。
RESAMPLE `BILINEAR` 也会被拒绝，因为 v1.24.0 序列化 fraction 表示省略了恢复
分数缩放语义所需的 denominator。第 4 节公开执行接口保持不变，差异只存在于
runtime 隐藏的 invocation adapter。

### 3.2 MVP 后 C3 扩展

C3 在同一个静态 f32 DAG 中加入 rank 3-5 的 `CONV_FPROP`、`CONV_DGRAD` 和
`CONV_WGRAD`。logical tensor 使用 `[N,C,spatial...]`，filter 使用
`[K,C_per_group,filter...]`。group 数由 `X.C / C_per_group` 推导；group 数和
`Y.K / groups` 都必须是正整数。空间 stride/dilation 为正，pre/post padding 非负，
支持 `CROSS_CORRELATION` 和反转 filter 的 `CONVOLUTION`。序列化输出的每个 extent
必须与 checked convolution 公式一致。

C3 还执行 schema capability 清单声明的 14 个 normalization/statistics 行。
Batch 参数和统计量 shape 为 `[1,C,1,...]`；instance 参数同样为 per-channel，保存
统计量为 `[N,C,1,...]`。Layer/RMS 在同 rank scale 非 1 的维度做归一化；adaptive
layer normalization 使用相同规则，但始终保留 batch 轴。

Forward normalization 使用 population variance 做归一化，并输出
`1/sqrt(variance + epsilon)` 作为序列化 inverse variance。running variance 在
reduction count 大于 1 时使用 sample variance；running-stat 更新公式为
`(1 - momentum) * previous + momentum * current`。Backward 行读取保存的统计量，
计算 data、scale 和 serializer 声明的 bias gradient。`GENSTATS`、`BN_FINALIZE`
和 `DBN_WEIGHT` 通过公开序列化端口输出 sum、等价 affine 和 gradient coefficient，
不引入私有 side channel。

scalar-like 输入必须是所有维度为 1、非 pass-by-value 的显式 f32 tensor。非空
分布式 `peer_stats` 被拒绝。BATCHNORM running-stat 端口必须全有或全无。执行时
epsilon 必须使 square-root 输入为正，`ACCUM_COUNT` 必须为正；这些数据值属于调用者
前置条件，而不是编译期 metadata。C2/C3 node 可通过 virtual workspace tensor
混合。dynamic shape、alias、pass-by-value、ragged/reordered storage 和非 f32 执行
仍延后。

## 4. 对外运行接口

MVP 对外只提供 cuDNN Frontend `v1.24.0` Graph execute 的调用形状，不提供第二套
handle-free API、descriptor 或裸 kernel ABI。CPU-only 形式如下：

```cpp
namespace deepforge::runtime {

using FrontendHandle = void *;
using VariantPack = std::unordered_map<int64_t, void *>;

class Executable {
public:
  int64_t get_workspace_size() const;
  import::Status execute(FrontendHandle handle,
                         VariantPack &uid_to_host_ptr,
                         void *workspace) const;
};

} // namespace deepforge::runtime
```

契约如下：

- 参数顺序和 map 类型与 Frontend 的 UID-map execute overload 保持一致；map 使用
  非 const 引用以保持调用兼容，但 DeepForge 不修改其中的键或地址。
- `handle` 仅为调用形状兼容而保留，可以为 null；CPU runtime 不解引用、不查询也
  不转发它。CPU-only 形式使用 opaque `void *`，不要求 CUDA Toolkit/cuDNN backend。
- variant-pack 中必须存在序列化图的 X、W、Y UID；额外 UID 被忽略。
- 指针是 CPU 可寻址 host pointer，不是 CUDA device pointer。
- X/W/Y 至少按 `alignof(float)` 对齐。生成代码不得无条件标注 64-byte 对齐。
- workspace 由调用者提供，按 64-byte 对齐；调用者必须先按
  `get_workspace_size()` 分配返回的字节数。若返回值为 0，workspace 可为 null。
- X、W、Y 和 workspace 的有效字节区间不得重叠。runtime 可以校验 UID、空指针、
  对齐、整数溢出和可计算的区间重叠；C++ 接口没有携带分配长度，不能便携地证明
  指针确实指向足够大的已分配对象，调用者仍必须保证每个 buffer 的容量。
  完成这些前置检查后，内部 kernel 才能使用 `noalias` 假设。
- Y 由调用者分配；DeepForge 不替换 variant-pack 中的地址。
- Executable 的编译 metadata 是不可变的；使用不同 workspace 的并发调用是安全的，
  同一个 workspace 不得被并发执行共享。
- `DeepForge/Runtime/Executable.h` 可在没有 MLIR、CUDA 或 cuDNN include path 时独立
  包含；MLIR ExecutionEngine、ORC 和 compiler metadata 只出现在内部 factory header。

CPU-only `deepforge::import::Status` 提供 `is_good/is_bad`、稳定错误码和 message；
它保持 Frontend-shaped 的错误检查方式，但不是
`cudnn_frontend::error_t` 的 binary-compatible 类型。runtime 错误映射到
`DFE_INVALID_VARIANT_PACK`、`DFE_UNSUPPORTED_CPU_FEATURE` 或
`DFE_GRAPH_EXECUTION_FAILED` 等稳定语义。生成 kernel 的私有参数签名和 memref
descriptor 仅存在于 runtime 内部；object 中的原始函数和
`_mlir_ciface_<symbol>` loader wrapper 均为 ELF `GLOBAL HIDDEN`。

这保证相同的 C++ 调用形状，不承诺能把 DeepForge `Executable` 与 NVIDIA
`Graph` 对象在二进制层互换；`std::unordered_map`、`std::string` 和编译器 ABI 仍受
README 中固定主机工具链约束。

## 5. Workspace 和 buffer ownership

所有中间 buffer 都来自调用者 workspace。MVP 禁止生成 kernel 内部调用
`malloc/free`：

1. Tensor/Linalg 变换完成后，仅运行一次 One-Shot Bufferize。
2. 对 `memref.alloc` 以及 One-Shot Bufferize 可能留下的
   `bufferization.alloc_tensor`，先 materialize 成静态 memref allocation；这
   是 bufferization 后的 lowering，不是第二次 bufferize。
3. `deepforge-workspace-plan` 为静态临时 buffer 计算带对齐的 offset。
4. 临时 `memref.alloc` 被改写为 workspace 上的 `memref.view/subview`。
5. 最终 IR 不得残留 owned allocation，因此不需要在 kernel 中 deallocate。

显式 padding 的临时张量是 MVP 的主要 workspace 消耗。后续可通过边界条件或
pad/conv fusion 消除，但不能改变对外接口。

## 6. 数值语义

MVP 使用 f32 输入、f32 权重和 f32 累加。允许在同一个输出元素的归约范围内：

- 使用 FMA；
- 按 SIMD lane 重排加法；
- 对 R/S/C 循环展开或分块。

不允许 `nnan`、`ninf`、`nsz`、近似倒数等与 Conv2D 无关的 fast-math 假设。
默认验证标准为：

```text
abs(actual - reference) <= 1e-4 + 1e-3 * abs(reference)
```

reference 使用相同 f32 输入并以 f64 累加。容差可由测试参数覆盖，但放宽必须在
测试报告中可见。NaN/Inf 的专项测试单独比较分类，不套用上述有限数容差。

## 7. CPU feature 分发和尾部

编译产物包含三个内部变体：

| 变体 | 向量宽度 | 前置条件 |
|---|---:|---|
| scalar | 1 | x86-64 baseline |
| AVX2 | 8 x f32 | AVX2 + FMA |
| AVX-512 | 16 x f32 | AVX-512F + FMA |

runtime 使用 CPUID 选择当前 CPU 支持的最高变体。每个 SIMD 变体对 `C % VF`
使用标量 cleanup loop，因此 C 和 K 不要求是向量宽度的倍数。不得在不支持的
CPU 上执行带更高 ISA 的函数。

## 8. 失败策略

以下情况均为编译期错误，不做静默 fallback：schema/版本不匹配、动态 shape、
非 packed stride、非 f32、未知 node、输出形状不一致、非 unit stride/dilation、
维度/字节数溢出。

CPU feature 不足不是错误，运行时回退到较低变体。AVX 变体的选择除 CPUID 外还
必须检查 OS 对 XMM/YMM/ZMM 状态的支持（例如 `OSXSAVE`/`XGETBV`），不能只看
硬件 feature bit。数值容差超限、缺 UID、空
指针或 buffer 重叠是执行/测试错误，必须返回可定位的错误码。workspace 指针
和大小由 `get_workspace_size()` 查询契约保证；MVP 接口沿用 cuDNN Frontend
的 execute 形状，因此没有额外的 size 参数。

### 8.1 Stable importer diagnostics

P1 importer 的 `Status::code()` 和 message 前缀使用以下稳定标识：

| 诊断码 | 含义 |
|---|---|
| `DFE_INVALID_ARGUMENT`, `DFE_IO_ERROR` | 空输入、无效参数或文件读取失败 |
| `DFE_PARSE_ERROR` | JSON/UBJSON 语法、截断、尾随字节或 canonical UBJSON 校验失败 |
| `DFE_SCHEMA_VERSION_MISMATCH`, `DFE_FRONTEND_VERSION_MISMATCH` | Graph schema 或 Frontend producer 版本不匹配 |
| `DFE_MISSING_FIELD`, `DFE_INVALID_FIELD_TYPE`, `DFE_INVALID_VALUE` | 固定 schema 的字段缺失、类型错误或值不满足约束 |
| `DFE_UNSUPPORTED_NODE`, `DFE_UNSUPPORTED_DATA_TYPE`, `DFE_UNSUPPORTED_EXECUTION_METADATA` | 输入协议合法，但语义超出 MVP |
| `DFE_DUPLICATE_UID`, `DFE_MISSING_UID` | UID 不唯一、缺失、name fallback 或引用不存在 |
| `DFE_INVALID_LAYOUT`, `DFE_INVALID_SHAPE` | packed layout、rank、静态维度或 Conv 输出形状不合法 |
| `DFE_DIMENSION_OVERFLOW` | 整数、维度乘积、packed stride 或字节数超出可表示范围 |

P2 继续复用上述稳定诊断码：canonical model 的物理 layout、padding 或 shape
错误分别返回 `DFE_INVALID_LAYOUT`、`DFE_INVALID_SHAPE` 或
`DFE_DIMENSION_OVERFLOW`；标准 MLIR verifier 或生成 module 结构不满足约束时返回
`DFE_INVALID_VALUE`，message 以 `DFE_INVALID_VALUE: mlir:` 开头。P2 不引入新的
MLIR 方言或自定义 operation，UID/shape/padding 等后续编译信息通过独立的
`Conv2DCompileMetadata` 传递。

runtime 和 artifact loader 增加以下固定编号；已有编号不得重排：

| 数值 | 诊断码 | 含义 |
|---:|---|---|
| 17 | `DFE_INVALID_VARIANT_PACK` | UID 缺失、空指针、对齐或可计算的 alias/workspace 前置条件失败 |
| 18 | `DFE_UNSUPPORTED_CPU_FEATURE` | 强制执行的 object 变体不受当前 CPUID/OS XSTATE 支持 |
| 19 | `DFE_GRAPH_EXECUTION_FAILED` | JIT/内部入口调用失败 |

`.dfo` 的格式、版本校验、ORC 装载和原生代码信任边界见
[DFO Artifact 格式](../artifact-format.md)。FNV checksum 只检测意外损坏，不构成
签名或执行隔离；runtime 只能装载可信来源的 artifact。

## 9. 兼容决策

- producer 固定为 cuDNN Frontend `v1.24.0`，其他版本在 MVP 中拒绝；
- 同时接受 Graph JSON 和单文档 UBJSON，不保留或重导出 GPU plan metadata；
- 公开 execute 固定为 handle + UID map + workspace 调用形状，不发布 handle-free
  overload；CPU runtime 始终忽略 handle。
- 对外 ABI 固定为 Frontend-shaped 的 CPU-only 形式：opaque `void *` handle、UID
  variant-pack、workspace 和 DeepForge status；不要求 CUDA/cuDNN headers；
- 官方 `cudnnHandle_t`/`cudnn_frontend::error_t` exact types、GPU execution 和
  Frontend samples 不属于 MVP，未来若引入必须单独进行 ABI/依赖评审。
