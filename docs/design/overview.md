# DeepForge 总体架构设计

[English](overview.en.md)

## 1. 项目定位

DeepForge 将固定版本 cuDNN Frontend 的序列化 Graph 编译为 CPU object code。
MVP 只实现静态、packed、f32 Conv2D FWD，但输入协议和运行接口直接沿用
cuDNN Frontend 的 Graph serialization 与 UID variant-pack 概念。
当前 MVP 后 C5 实现可执行全部 39 个 serialized tag 所声明的静态子集：3 个
convolution、8 个基础操作、14 个 normalization/statistics、5 个
sequence/attention 和 9 个低精度特殊操作。通用 data 仍为 f32，INT32/INT64
只用于文档指定的 sequence 端口；特殊路径按 operation 加入 f16/bf16/FP8/FP4/INT4
storage 与 conversion。公开 runtime 接口保持不变，内部只使用上游 MemRef、SCF、
Arith、Math 和 LLVM dialect。

规范性支持边界见 [contracts.md](contracts.md) 和
[schema capability 清单](../cudnn-graph-schema-inventory.md)。

## 2. 设计原则

1. **先正确再扩展**：标量 kernel 是所有优化路径的语义基线。
2. **协议兼容与算子覆盖分离**：完整解析 cuDNN schema，明确拒绝未实现节点。
3. **上游优先**：Tensor、Linalg、Affine/SCF、Vector、LLVM 均复用 MLIR。
4. **一次 bufferization**：tensor/Linalg 变换结束后运行一次 One-Shot Bufferize。
5. **显式合法性边界**：每阶段有 allowed-dialect 集合并运行 verifier。
6. **目标信息不改变语义**：当前只固定 ISA feature；未来 cache cost model 只能影响
   tiling 选择，不能被建模为可寻址内存。
7. **运行时隔离内部 ABI**：外部只有 Frontend-shaped handle + UID variant-pack +
   workspace，内部 kernel 签名可演化。
8. **分析按需重算**：结构变换后不保留过期 dependence/alias 结论。

## 3. 组件和 IR 边界

```text
+----------------------------------------------------------------+
| cudnn-frontend v1.24.0 serialized Graph                         |
| JSON graph object or single-document UBJSON graph + metadata    |
+-------------------------------+--------------------------------+
                                | structured parse + validation
                                v
+----------------------------------------------------------------+
| DeepForge Importer                                               |
| logical cuDNN dims/strides -> physical MLIR tensor types         |
| CONV_FPROP -> optional tensor.pad + fill + linalg.conv_2d_*     |
+-------------------------------+--------------------------------+
                                | no cudnn.* op leaves importer
                                v
+----------------------------------------------------------------+
| Tensor + Linalg Dialects                                         |
| destination-passing tensor IR, named Conv kept for MVP          |
+-------------------------------+--------------------------------+
                                | one-shot-bufferize exactly once
                                v
+----------------------------------------------------------------+
| MemRef + Affine/SCF + Vector                                     |
| direct-conv loop schedule, C-reduction SIMD, workspace views    |
+-------------------------------+--------------------------------+
                                | complete dialect conversions
                                v
+----------------------------------------------------------------+
| LLVM Dialect -> LLVM IR -> x86-64 object variants               |
+-------------------------------+--------------------------------+
                                | hidden internal entry points
                                v
+----------------------------------------------------------------+
| Runtime Executable                                               |
| UID binding, pointer validation, workspace, CPUID dispatch       |
+----------------------------------------------------------------+
```

上图描述原有优化 Conv 路径。canonical import 之后，通用 C2-C5 图走并行的标准
MLIR 路径，直接生成静态 MemRef/SCF/Arith/Math IR、规划 virtual tensor workspace
view，再进入同一 LLVM object pipeline。该路径包含 grouped convolution、
convolution gradient、normalization、sequence transform 和 attention
forward/backward，以及 C5 软件低精度 conversion、block scaling、FP8 matmul 和
MoE；它不运行 MVP Conv 的 Tensor/Linalg bufferization 或 direct-conv schedule。

### 3.1 Importer

Importer 是文件/对象模型到 MLIR 的边界，不是 MLIR pass。它负责：

- 探测 JSON 或 UBJSON；
- 以 nlohmann/json 3.11.3 严格解析完整文档，拒绝 UBJSON 尾随字节；
- 校验 `json_version`、frontend version 和必需字段；
- 忽略文档内 GPU-only plan metadata，拒绝非空的未支持执行语义字段；
- 解析 tensor/node 引用和稳定 UID；
- 校验适用的 capability 子集；
- 为优化 MVP Conv 构造标准 Tensor/Linalg IR，或为通用 C2-C5 图构造标准
  MemRef/SCF/Math IR。

不定义 `cudnn.conv_fwd` 临时 op。这样 One-Shot Bufferize 不会遇到没有
`BufferizableOpInterface` 的未知自定义 op。

### 3.2 Compiler pipeline

主 pipeline 由五个可单独测试的阶段组成：

```text
import-cudnn-graph
  -> named tensor/linalg verification
  -> one-shot-bufferize
  -> workspace planning/rewrite
  -> clone target variants + direct-conv SCF schedule
  -> affine/scf/vector/memref/func/cf/arith/index to LLVM
```

详细 pass 顺序见 [pass-pipeline.md](pass-pipeline.md)。

### 3.3 Runtime

Runtime 不解释 Graph，也不重新做 shape inference。编译产物记录：

- X/W/Y UID 与静态字节数；
- workspace 大小和对齐；
- 三个内部 kernel 入口；
- 各变体所需 CPU feature；
- 数值和 alias contract 版本。

execute 忽略兼容用 handle，只做 UID 绑定、指针元数据/可计算重叠检查、CPUID
（含 OS SIMD 状态）分发和内部调用；buffer 容量仍由调用者按契约保证。

## 4. Conv2D 核心表示

cuDNN 的 logical dims 与 MLIR physical shapes 不同：

```text
cuDNN X: dim[N,C,H,W], stride[HWC,1,WC,C]
MLIR X:  tensor<NxHxWxCxf32>

cuDNN W: dim[K,C,R,S], stride[RSC,1,SC,C]
MLIR W:  tensor<KxRxSxCxf32>

cuDNN Y: dim[N,K,P,Q], stride[PQK,1,QK,K]
MLIR Y:  tensor<NxPxQxKxf32>
```

因此使用 `linalg.conv_2d_nhwc_fhwc`，其中 filter 的 F 维就是 K。主路径
保留 cuDNN 的 KRSC packed buffer，不在 ABI 外要求调用者预打包权重。

MVP 向量化 C reduction：input 与同一个 K 的 filter 在 C 上都连续；SIMD FMA
后必须通过 `vector.reduction <add>` 合并 lane，再写一个 Y 标量。不能把 C lane
直接当作 K lane 存储。

## 5. 已解决的设计问题

| 原问题 | MVP 决策 |
|---|---|
| C lane 被错误存为 K | C 向量累加后做水平归约；端到端示例覆盖 |
| Vector 后仍残留多种方言 | 明确完整 LLVM conversion 顺序和最终 legality check |
| cuDNN Graph/C++/MLIR 入口混淆 | importer 只接官方 JSON/UBJSON serialization |
| 多次/过早 bufferize | Tensor/Linalg 变换后只运行一次 One-Shot Bufferize |
| ABI、stride、padding、alignment 未定义 | 统一写入 contracts.md；runtime 校验可计算的调用前条件 |
| L1/L2/L3 被当作地址空间 | cache 仅作为 tiling cost model 参数 |
| AMX 生命周期和 tile 配置未解决 | AMX 完全移出 MVP，后续优先使用上游 X86 dialect |
| 动态 shape/类型转换过度承诺 | MVP 只接受静态 f32；不使用 `tensor.cast` 改元素类型 |
| 分析结果跨变换失效 | analysis 由每个 transform 即时获取并自动失效 |
| tail/feature/test/version 缺失 | scalar cleanup、三变体 CPUID、测试矩阵、精确版本锁定 |

## 6. Target model

当前实现没有尚未使用的 `TargetProfile` 抽象，而是在 object codegen 处固定三种
`llvm::TargetMachine` 配置：

```text
host x86-64 triple + x86-64 baseline
host x86-64 triple + avx2,fma
host x86-64 triple + avx512f,fma
```

vector width 分别为 1、8、16。`.dfo` 记录精确 target triple 和 required features，
loader 拒绝与主机 triple 不一致的原生 artifact。

首个生效的性能 cost model 位于 Loop/Schedule 层。对优化的静态 packed f32 Conv
路径，它根据 target vector width 和 register budget 选择可检查的 K-output unroll
factor，并始终保留固定 baseline 回退；通用 C2-C6 graph 仍使用 reference schedule。
runtime 的 CPUID/XGETBV 分发只做 ISA capability/safety 检查，不是 cost model。
cache profile、software prefetch、外层 tiling 和 threading 均留到 benchmark 驱动的
Optimize 阶段。

## 7. Ownership 和 alias

- 每个 external tensor 都携带 read、write 或 read-write access metadata。
- 外部 buffer 和 workspace 不得重叠。
- runtime 完成可计算的区间重叠检查后，私有 kernel 才可获得 readonly/noalias
  属性；接口没有携带分配长度，不能把它描述成可移植的越界检测器。
- 临时 buffer 全部来自 workspace，kernel 不拥有外部地址，也不调用 free。
- 编译器中的 AliasAnalysis 只服务当前 pass；任何重排/融合后按需重算。

## 8. 延后范围

以下内容仍不属于冻结的 MVP baseline：Machine Dialect、AMX/bf16、cache address
space、OpenMP、多算子 fusion、动态 shape、非 packed stride、NCHW physical
layout、grouped/depthwise Conv、GPU backend。MVP 后 C2-C5 已独立加入正的任意
stride、grouped convolution、特殊端口 bf16 和 schema 清单声明的能力子集。C6
已加入文档指定 block/MXFP8 端口的 `F8_128x4` 物理 scale 解码，以及单个
exact-shape external f32 `POINTWISE` node 的 runtime override。标准和 FP8 MATMUL
还支持 producer 序列化的 per-batch INT32 M/N/K extent tensor，同时保持静态
allocation 上界。标准 f32 SDPA 已
支持 ragged forward data/row output 和 backward data/gradient、带紧凑 page table
的独立 paged K/V、forward block mask 及 forward/backward sink token。更广泛
dynamic 行为、其他 physical reorder metadata、threading 和广泛 fusion 仍待实现。
固定使用的 v1.24.0 serializer 无法表达 paged backward，因此它属于未来 schema
版本兼容工作。

延后不等于删除设计方向。每项在拥有明确语义、上游能力评估、正确性测试和性能
基线后单独引入。

## 9. 目录规划

```text
include/DeepForge/
  Import/          serialized Graph parser and validator
  Compiler/        pipeline and target profile
  Runtime/         Executable and variant-pack API
lib/
  Import/
  Transforms/      direct Conv schedule and workspace planning
  Compiler/
  Runtime/
tools/
  deepforge-compile/
test/
  Import/
  Transforms/
  E2E/
```

MVP 不创建 `include/DeepForge/Dialect/Machine` 或对应 TableGen 目录。
