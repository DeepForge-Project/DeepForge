# cuDNN Graph 全操作覆盖方案

[English](cudnn-graph-coverage-plan.en.md)

## 1. 文档状态

这是已经确认的 MVP 后扩展方案。只有当对应阶段实现并通过验收时，才扩展当前
支持契约。现有静态、连续、f32
`CONV_FPROP` 路径在整个扩展过程中始终作为回归基线。

方案以固定版本 cuDNN Frontend `v1.24.0` 源码和 LLVM/MLIR
`llvmorg-22.1.8` 为依据。常规构建、测试和执行继续保持 CPU-only，不依赖
CUDA 或 cuDNN backend 动态库。

## 2. “所有操作”的口径

建议把兼容边界定义为 cuDNN Frontend serialized Graph 协议，而不是底层
cuDNN Backend API 中的所有 descriptor。

固定版本的 serializer 会输出 39 种不同的顶层 node tag：

```text
ADA_LAYER_NORM          ADA_LAYER_NORM_BPROP    BATCHNORM
BATCHNORM_INFERENCE     BLOCK_SCALE_DEQUANTIZE  BLOCK_SCALE_QUANTIZE
BN_FINALIZE             CONCATENATE             CONV_DGRAD
CONV_FPROP              CONV_WGRAD              DBN
DBN_WEIGHT              GENSTATS                INSTANCE_NORM
INSTANCE_NORM_BPROP     LAYER_NORM              LAYER_NORM_BPROP
MATMUL                  MATMUL_FP8              MOE_GROUPED_MATMUL
MOE_GROUPED_MATMUL_BWD  POINTWISE               REDUCTION
RESAMPLE                RESHAPE                 RMS_NORM
RMS_NORM_BPROP          RNG                     ROPE
ROPE_BWD                SDPA                    SDPA_BWD
SDPA_FP8_BWD            SDPA_FP8_FWD            SDPA_MXFP8_BWD
SDPA_MXFP8_FWD          SLICE                   TRANSPOSE
```

部分公开 Graph API（例如 softmax 和 mask helper）属于复合操作，没有独立的
序列化 tag。当这些 API 序列化后产生的全部 primitive node 都受支持时，才认为
对应复合 API 已覆盖。

v1.24.0 的 `Graph::deserialize` 只显式重建其中 12 种：
`CONV_FPROP`、`CONV_DGRAD`、`CONV_WGRAD`、`POINTWISE`、
`REDUCTION`、`MATMUL`、`RESAMPLE`、`SLICE`、`TRANSPOSE`、
`SDPA`、`SDPA_BWD` 和 `MOE_GROUPED_MATMUL`。因此 DeepForge 必须分别
记录三种状态：

1. **已识别**：完整解析固定版本的序列化 schema，不丢弃 node 或 attribute。
2. **可执行**：可以在 CPU 上完成验证、lowering、编译和运行。
3. **已验证**：声明的 shape、layout 和 data type 子集已经通过正例、负例、
   artifact、sanitizer 和数值测试。

只有 capability matrix 中不存在未知 tag，且所有范围内条目都达到声明的验证
级别，才能声称“支持所有操作”。已经识别但尚未实现的 node 必须返回稳定的
`unsupported operation` 诊断，不能静默忽略。

## 3. 兼容性契约

扩展过程中保持以下边界：

- 输入协议：cuDNN Frontend `v1.24.0`、`json_version == "1.0"`，支持
  JSON 和 UBJSON carrier；
- 执行目标：Linux x86-64 CPU；
- 对外调用形式：opaque handle、UID 到 host pointer 的 variant pack、调用方
  workspace 和 DeepForge status；
- 编译 IR：使用上游 Tensor、Linalg、Arith、Math、SCF/Affine、Vector、
  MemRef 和 LLVM dialect，不暴露或序列化 `cudnn.*` dialect；
- 内存所有权：生成 kernel 不分配内存，临时存储统一规划到调用方 workspace；
- 依赖发现：CMake 和构建脚本从调用方指定的 prefix 中查找依赖，不把本机绝对
  路径写入项目契约。

对外 execute API 保持源码兼容。生成 kernel 的内部 ABI 和 `.dfo` payload
需要泛化，但二者都不是公共 ABI。

## 4. 必要的架构改造

### 4.1 通用 canonical Graph

用与协议 carrier 无关的通用图模型替换当前单 Conv 模型：

- 可变 rank 的 dimensions 和 strides；
- 完整的 v1.24.0 data type 目录，以及经过溢出检查的 storage size 元数据；
- tagged node attribute variant 和有序 input/output port；
- 任意多节点 DAG、稳定 node 顺序和 UID 引用；
- graph input/output、virtual tensor、pass-by-value scalar 和 constant；
- 显式保留 dynamic dimension、ragged、reorder 和 workspace 元数据，即使执行
  支持被延后；
- 每种操作独立完成验证和 shape inference，不与 JSON 解析混在一起。

新增表驱动 capability registry，统一记录每种 operation/mode 的 schema 版本、
attribute、合法 data type/layout、lowering 状态和诊断。这样支持声明不会散落在
多个 importer switch 中。

### 4.2 标准 MLIR lowering

协议 node 在 bufferization 前完成规范化：

- view 和数据搬运映射到 Tensor/Linalg；
- pointwise 和 broadcast 映射到 Linalg、Arith 和 Math；
- reduction、matmul、convolution 和 pooling 在 indexing 语义可验证时使用
  named 或 generic Linalg；
- normalization、RoPE、softmax 和 attention 先分解为已经测试的 primitive；
- graph-level transform 完成后仍然只运行一次 One-Shot Bufferize。

初期不计划引入自定义高层 dialect。只有标准 IR 分解产生了已经测量且无法通过
canonical C++ graph model 解决的编译时间或优化问题时，才重新评估该决策。

### 4.3 通用 runtime 和 artifact

用以下结构替换固定 X/W/Y metadata 和三个 rank-4 f32 descriptor：

- 以 external tensor UID 为键的有序 kernel argument table；
- 每个 external tensor 的 element type、rank、dimensions、strides、byte range、
  alignment 和读写属性；
- 支持任意 graph signature 和 workspace view 的生成式 adapter；
- 基于读写属性而不是 Conv 固定名称的 overlap 检查；
- `.dfo` format v2，保存通用 tensor/argument table 和各 CPU variant symbol；C6
  后续为 dynamic policy metadata 将 writer 升到 v3，为 ragged storage reference
  升到 v4，再为 packed logical-sequence divisor 升到 v5，并保持旧版本读取兼容。

兼容规则继续读取 format-v1 Conv2D、format-v2 generic、format-v3 shape-override 和
format-v4 ragged-storage artifact；当前新编译 graph 写 format v5。未知 artifact
version 仍然直接报错，不能静默解释。

### 4.4 Cost model 所在层

Cost model 继续位于 schedule 和 target specialization 层，不参与判断序列化操作
是否可接受。每个 operation family 提供合法 schedule candidate 和 feature，cost
model 在语义 lowering 之后、最终 Vector/LLVM codegen 之前选择 candidate。当
不存在可用优化 candidate 时，始终保留 scalar correctness 路径。

## 5. 分阶段交付

### C0. 通用基础架构

**状态**：已于 2026-07-31 完成并通过验证。

**工作内容**：引入通用 tensor、node、多节点 DAG 验证、capability registry、
通用 compile metadata、runtime argument table 和 `.dfo` v2。把现有 Conv 路径迁移
到这些契约上，但不改变对外行为。

**退出条件**：现有测试全部原样通过；format-v1 artifact 仍可加载；多节点 graph
可以被解析并在具体未支持 node 上精确失败；runtime 的 UID、byte range、alignment
和 overlap 检查由 argument table 驱动。

### C1. 完整协议识别

**状态**：已于 2026-07-31 完成并通过验证。

**工作内容**：解析并结构化验证全部 39 个 tag、全部 50 个非 sentinel pointwise
mode、全部 9 个 reduction mode、每个 node port 和 v1.24.0 中每个序列化
attribute。增加自动生成或人工复核的 schema inventory 和公开 capability matrix。

**退出条件**：每个 tag 的 JSON 和 UBJSON 都得到等价 canonical graph；错误
attribute 的诊断确定；已知 tag 不会被报成 unknown；未 lowering 的 tag 明确报告
“已识别但不可执行”。

### C2. 基础可执行操作

**状态**：已于 2026-07-31 完成并通过验证。

**工作内容**：实现静态 shape f32 的 `RESHAPE`、`TRANSPOSE`、`SLICE`、
`CONCATENATE`、适用的全部 `POINTWISE` mode、适用的全部 `REDUCTION` mode、
`MATMUL` 和 `RESAMPLE`。在 Frontend 契约允许的范围实现 NumPy 风格 broadcast。

**退出条件**：单 node 和融合多 node graph 通过 scalar reference、artifact
round-trip、workspace、runtime validation、ASan 和 UBSan 测试。

### C3. Convolution 和训练操作族

**状态**：已于 2026-07-31 完成并通过验证。

**工作内容**：先泛化 `CONV_FPROP`，再加入 `CONV_DGRAD` 和 `CONV_WGRAD`；
加入 `GENSTATS`、`BN_FINALIZE`、`DBN`、`DBN_WEIGHT`、batch normalization、
instance normalization、layer normalization、RMS normalization、adaptive layer
normalization 及其已序列化的 backward variant。

**退出条件**：forward/backward shape inference 和 gradient 被独立检查；代表性
输入通过 finite-difference gradient test；这些操作可与 C2 操作组成多节点图。

交付子集覆盖 rank 3-5 grouped convolution、独立 FPROP/DGRAD/WGRAD reference、
normalization forward/backward reference、LayerNorm finite difference、running
statistics、C2/C3 混合图、artifact 重载以及完整 Release/ASan/UBSan 测试。

### C4. Sequence 和 attention 操作族

**状态**：已于 2026-07-31 完成并通过验证。

**工作内容**：先用 primitive decomposition 实现 `ROPE`、`ROPE_BWD`、`RNG`、
`SDPA` 和 `SDPA_BWD`。把 mask、causal、bias、dropout、seed/offset 和 sequence
length metadata 拆成明确的子阶段。

**退出条件**：确定性 RNG、attention reference、mask/边界 shape、backward
gradient 和 workspace 上界测试全部通过。

交付子集覆盖 fixed/tensor seed-offset Bernoulli RNG、full/partial RoPE
forward/adjoint，以及带 GQA、bias、ALiBi、sequence padding、两种 diagonal
alignment、sliding window、custom/probability dropout、row statistics、RNG dump
和 Q/K/V/bias gradient 的 f32 BHSD SDPA。测试包括独立 reference、有限差分、
全 mask row、artifact reload、CPU variant 一致性及 Release/ASan/UBSan。Paged/cache、
block mask、sink token 和 packed/ragged 路径仍归 C6；FP8/MXFP8 已在下述 C5
子集内交付。

### C5. Data type 和特殊操作

**状态**：已于 2026-07-31 完成并通过验证。

**工作内容**：执行支持从 f32 扩展到合法的 f64、f16、bf16、integer 和 boolean，
再加入 FP8/FP4/INT4 storage 和 conversion 语义。实现 block-scale
quantize/dequantize、`MATMUL_FP8`、FP8/MXFP8 SDPA variant，以及 MoE grouped
matmul forward/backward。

v1.24.0 enum 有 20 个非 sentinel data type。支持状态按 operation/data type
组合声明；识别 enum 不等于已经可执行。

**退出条件**：conversion 边界、saturation/rounding、特殊浮点值、packed storage、
accumulator type 和每种类型的数值误差均有测试。不支持的 host ISA 可以正确 fallback
或返回明确错误。

已交付 numeric layer 为 f64、f16、bf16、signed/unsigned integer、boolean、FP8
E4M3/E5M2/E8M0、packed FP4 E2M1 和 packed INT4 提供 CPU storage 与 f32
conversion。执行能力仍按 operation/port 声明，并未把这些类型开放给所有通用操作。
9 个特殊 tag 已有静态 validated 子集：block-scale conversion、FP8 matmul、MoE
grouped matmul forward/backward，以及 FP8/MXFP8 SDPA forward/backward。

C5 测试通过公开 UID variant-pack ABI 直接执行低精度 raw buffer，覆盖
round-to-nearest-even、saturation、NaN/Inf、packed virtual workspace、block
scale、amax、外部 backward Stats、MoE expert partition、4 个特殊 attention tag、
错误 transpose shape 以及完整 Release/ASan/UBSan。物理 `F8_128x4` reorder、
dynamic/ragged metadata、可选 FP8 attention 特性和精确 MXFP8 dS block
requantization 在 C5 结束时归入 C6。

### C6. 动态元数据、优化和发布验收

**状态**：进行中，六个独立验收增量已完成。第一个为 block-scale conversion 的
E4M3/E8M0 scale 和 MXFP8 的 E8M0 descale 端口实现 Frontend/CUTLASS
`F8_128x4` 物理 ordering。第二个为单个 exact-shape、external plain f32
`POINTWISE` 子集实现 Frontend-shaped runtime dimension/stride、artifact v3 policy
持久化和配套 workspace query。测试覆盖物理 offset/padding、runtime descriptor
执行、artifact reload、dimension/byte-span 上界、错误 override list 和静态 artifact
拒绝。第三个为静态 f32 SDPA forward 实现独立 ragged Q/K/V/O 和独立 paged K/V；
执行前校验 prefix 内容和紧凑 span，阻止非法 page ID 参与内存寻址，以 artifact v4
持久化 storage reference，并测试精确分配、独立 K/V page ordering、artifact reload、
错误 metadata 以及 Release/ASan/UBSan。第四个加入 ragged forward row output 和
backward tensor，校验 backward max-total hint，支持 K/V page table 各自独立紧凑
存储，并实现标准 f32 forward block mask 及 forward/backward sink token/dSink。
Artifact v5 记录 page-table block divisor；独立 reference、有限差分、非连续压缩
mask、精确分配、错误 metadata、reload 和 sanitizer 构成该增量的验收集。
第五个为原有优化的静态 packed f32 Conv 路径加入首个生效的 Loop/Schedule cost
model。它选择 target-aware K-output unroll，保留 baseline 回退，暴露 schedule 名称，
并通过决策、tail、数值、绑核 A/B、Release、ASan 和 UBSan 检查。通用 C2-C6 graph
仍使用 reference schedule。第六个为标准和 FP8 MATMUL 实现 producer 序列化的
per-batch INT32 M/N/K extent tensor，包括 broadcast metadata、标准 MATMUL 有限
padding、FP8 零 padding、artifact reload、错误 descriptor、Release、ASan 和 UBSan
覆盖。

**工作内容**：将动态行为扩展到已交付 pointwise/MATMUL 子集之外；加入已交付
scale 子集外的 reorder format；随后独立评估 fusion、threading、其他 vector
schedule 和各操作族 cost model。固定使用的 v1.24.0 serializer 没有 paged backward
page-table 端口，因此该能力属于未来 schema 版本的兼容工作，不是当前契约下的实现
任务。

**退出条件**：范围内 capability matrix 全部达到已验证状态；sanitizer 和兼容性
测试全部通过；scalar 与 optimized variant 在每种操作的容差内一致；中英文性能
基线和用户文档已经发布。

## 6. 阶段执行规则

每个阶段单独开发和 review：

1. 启用 capability row 前先更新 capability matrix 和规范契约；
2. lowering 之前先增加 parser 和 verifier 测试；
3. vector/fusion 之前先完成 scalar CPU correctness；
4. 运行 importer-only、完整 MLIR、artifact、E2E、ASan 和 UBSan 测试；
5. 中英文用户文档和设计文档同步更新；
6. 阶段退出条件全部通过后提交，再开始下一阶段。

部分支持必须精确表示成 operation、mode、attribute、shape class、layout 和 data
type 的组合。任何阶段都不能用宽泛的“支持某操作”描述更窄的实现。

## 7. 测试和 fixture 策略

常规 CPU CI 使用仓库内 JSON fixture，并通过固定版本 nlohmann/json 生成 UBJSON。
Fixture 覆盖每个 tag 和 attribute 分支，包括错误和未支持组合。数值 reference 使用
独立 scalar 实现，浮点比较通常使用更高精度 accumulator。

数值容差按 operation 和 data type 分别定义。Reduction 和 attention 使用与规模
相关的误差界；integer/boolean 要求精确相等；NaN、infinity、signed zero、
saturation 和 RNG 可复现性都有专门测试。

fixture 字段和公开 execute signature 直接对照固定版本的开源 Frontend 源码。
运行 CUDA/cuDNN producer 或比较 GPU 结果不属于 DeepForge CPU release gate，也不是
构建、测试、安装或运行依赖。

## 8. 主要风险

- **范围歧义**：Graph serializer tag、复合 Graph API 和 Backend descriptor 是三个
  不同集合，必须先确认第 10 节。
- **Schema 不对称**：v1.24.0 serializer 输出的 tag 多于公开
  `Graph::deserialize` switch 能重建的 tag，需要直接核对 node serializer 和 fixture。
- **组合数量**：mode、attribute、layout 和 data type 的组合增长远快于 tag 数量，
  必须使用 capability registry 和生成式测试矩阵管理支持声明。
- **低精度 CPU 语义**：FP8/FP4/INT4 在优化前可能先依赖软件 conversion 和 packed
  storage 实现。
- **Attention 展开**：primitive decomposition 可能增加编译时间和 workspace；C4
  先记录基线，再决定是否需要高层 IR。
- **Artifact 迁移**：通用 signature 需要 format v2，v1 兼容性必须通过测试证明。

## 9. 批准后的第一步

第一个实现增量只执行 C0：

1. 增加 capability registry，把当前 `CONV_FPROP` 标为已验证；
2. 用经过溢出检查的可变 rank metadata 替换固定 rank tensor metadata；
3. 引入通用 node container，同时保留 Conv attribute；
4. 增加通用 external argument table，并迁移 runtime validation；
5. 定义和测试 `.dfo` v2，同时保留 v1 reader；
6. 重新运行现有 correctness、artifact、sanitizer 和 CI gate。

该保持兼容性的 C0 迁移已在 operation lowering 开始前完成；此规则现在是历史
顺序约束，不是尚未关闭的 gate。

## 10. 已确认的项目决策

项目负责人已于 2026-07-31 接受以下默认值：

1. **范围**：最终覆盖全部 39 个 v1.24.0 serialized tag 和复合 Graph API，不覆盖
   与 Graph serialization 无关的底层 Backend descriptor。
2. **类型顺序**：所有操作先完成静态 f32 correctness，再在 C5 扩展合法 data type
   矩阵。
3. **Shape 顺序**：先支持任意 rank 的静态 shape；dynamic、ragged、reorder 和
   paged metadata 延后到 C6。
4. **Artifact 兼容**：在 generic、dynamic-policy、ragged 和 packed-sequence-divisor
   metadata 兼容迁移后，保留 `.dfo` v1-v4 reader，新编译结果写 v5。
5. **外部验证**：CPU CI 保持自包含，未来使用可选 GPU runner 做 producer 和差分
   发布验证。
