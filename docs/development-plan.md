# DeepForge 开发方案

[English](development-plan.en.md)

本文是已经完成的历史 P0-P6 MVP 方案。当前 MVP 后开发按中英文
[C0-C6 全操作覆盖方案](cudnn-graph-coverage-plan.md)执行；下文 Conv 约束仍是
回归基线，不代表当前完整 capability。

## 1. 目标和基线

本方案把当前设计文档转成可执行的开发顺序。它针对的是 MVP，不扩大支持范围。
规范性约束以 [MVP 契约](design/contracts.md) 为准；本方案解决“先实现什么、
如何验收、何时允许进入下一阶段”，不替代各层设计文档。

当前基线：

- P0-P6 已完成：仓库已有 canonical Graph model、strict JSON/UBJSON importer、标准
  Tensor/Linalg module builder、唯一一次 One-Shot Bufferize、静态 workspace、三种
  x86-64 object、Frontend-shaped runtime、可重新装载的 `.dfo`、CLI、benchmark、
  CTest 和 sanitizer/CI 配置；
- LLVM/MLIR 固定为 `llvmorg-22.1.8`，不使用 `main` 或 LLVM 23 RC；
- cuDNN Frontend 以 `v1.24.0` 作为 serializer schema/fixture reference，schema 版本固定
  为 `json_version == "1.0"`；MVP 拒绝其他 producer 版本，扩展兼容范围须单独评审；
- MVP 只接受 Linux x86-64、静态 packed f32、单个 `CONV_FPROP`；stride/dilation
  为 1，不支持 grouped/depthwise、融合、动态 shape、NCHW physical layout；
- Machine Dialect、AMX、bf16、多线程和 GPU backend 延后；
- 对外固定为 Frontend-shaped 的 CPU-only ABI：opaque handle + UID variant-pack +
  workspace + DeepForge status；CUDA/cuDNN backend、官方 exact C++ types 和生成
  kernel 的 memref descriptor/裸函数签名不公开。
- P1 importer 是独立的 CPU-only target，可以用 `DEEPFORGE_ENABLE_MLIR=OFF`
  配置和测试；P2 之后的 compiler target 才要求 MLIR。

## 2. 开发原则

1. **先打通标量纵向链路**：必须先完成 Graph importer -> MLIR -> scalar object ->
   runtime execute，再做 AVX 优化。
2. **每个阶段都有可执行的退出条件**：没有通过当前阶段的 verifier、负例测试和
   golden 输出，不进入下一阶段。
3. **协议兼容和算子覆盖分离**：解析器读取官方 JSON/UBJSON 协议；不支持的 node
   返回诊断，不跳过节点，不定义私有 `cudnn.*` MLIR op。
4. **一次 bufferization**：Tensor/Linalg 结构变换完成后只运行一次 One-Shot
   Bufferize；后续只做 bufferization dialect 的合法化和 workspace planning。
5. **target-independent 资源计划**：workspace plan 在 ISA-specific vectorization
   前完成；vectorizer 不能改变 memref allocation 的数量、大小或生命周期。
6. **标准方言优先**：MVP 路径使用 Tensor/Linalg/MemRef/Affine/SCF/Vector/LLVM，
   不复制寄存器分配和 cache address space。
7. **正确性优先于性能**：scalar 是参考实现；所有变体都与同一 f64-accumulate
   reference 比较，容差按契约记录。

## 3. 工作包和依赖

| 编号 | 工作包 | 主要目录 | 前置依赖 | 主要产物 |
|---|---|---|---|---|
| P0 | 工具链和依赖固化 | `CMakeLists.txt`, `cmake/` | 无 | 可复现 MLIR 22.1.8 环境、版本清单 |
| P1 | cuDNN serialization importer | `lib/Import`, `include/DeepForge/Import`, `test/Import` | P0 | canonical Graph model、JSON/UBJSON parser |
| P2 | Tensor/Linalg IR 导入 | `lib/MLIR`, `include/DeepForge/Compiler`, `test/MLIR` | P1 | 标准 MLIR module、compile metadata、layout/padding verifier |
| P3 | Bufferization 和 workspace | `lib/Transforms`, `lib/Runtime`, `test/Transforms` | P2 | 一次 bufferization、静态 workspace plan、内部 adapter |
| P4 | Scalar lowering 和运行时 | `lib/Compiler`, `lib/Runtime`, `tools` | P3 | scalar LLVM/object、UID execute |
| P5 | AVX2/AVX-512 变体 | `lib/Transforms`, `lib/Compiler`, `lib/Runtime`, `test/E2E` | P4 | C-reduction SIMD、CPUID/OS-state dispatch |
| P6 | CLI、打包、质量门 | `tools`, `test`, `docs` | P4，P5 可并行收尾 | `.dfo` artifact、CI、benchmark 和发布检查 |

关键依赖关系：

```text
P0 -> P1 -> P2 -> P3 -> P4 -> P5 -> P6
             |          |      |
             +-- IR tests +-- runtime tests
```

P1 的 parser 单元测试和 P0 的工具链安装可以并行准备，但任何 MLIR lowering
实现都必须以 P0 的固定版本为依据。

## 4. 分阶段实施

### P0. 工具链和依赖固化

**目标**：让开发、测试和 CI 使用同一套 LLVM/MLIR，而不是边开发边追踪滚动版本。

**工作内容**：

- 按 README 的命令构建并安装 `llvmorg-22.1.8`，至少启用 MLIR 和 X86 target；
- 在配置阶段精确检查 `LLVM_PACKAGE_VERSION == 22.1.8`；
- 固化 Ninja、C++20 编译器、nlohmann/json `3.11.3` parser 和 cuDNN Frontend
  `v1.24.0` 源码/fixture reference 的来源；不构建或链接 CUDA/cuDNN backend；
- 记录 `mlir-opt`、`mlir-translate`、`llc` 的版本和 X86 target 可用性；
- 用一个最小 module 做 `mlir-opt` parse/verify 和 LLVM translation smoke test；
- 将 MLIR pass 名称和 C++ API 以 22.1.8 的 registry/header 为准，不依赖文档中未
  验证的 shell spelling。

**产物**：

- 调用者选择的本地安装前缀，通过 `CMAKE_PREFIX_PATH` 交给 CMake 发现；
- CMake configure 成功日志和版本清单；
- 中文 README 的[外部工具链版本表](../README.zh-CN.md#外部工具链版本)，记录 LLVM commit/tag、
  C++ 编译器、parser 和 fixture 版本。

**退出条件**：

```text
mlir-opt --version       -> 22.1.8
mlir-translate --version -> 22.1.8
llc --version            -> includes X86 backend
cmake -S . -B build ...  -> configure succeeds
```

在 P0 完成前，缺少 `MLIRConfig.cmake` 导致的 CMake 失败是预期状态，不开始调试
lowering 代码。

本工作区的 P0 已完成：`llvmorg-22.1.8` 已构建安装，`mlir-opt` parse/verify、
`mlir-translate --mlir-to-llvmir` smoke test、X86 `llc` 检查和 DeepForge CMake
configure 均已通过。外部工具的实际验证版本记录在 [中文 README 外部工具链版本](../README.zh-CN.md#外部工具链版本)。

### P1. cuDNN Frontend serialization importer

**目标**：把官方 Graph JSON/UBJSON 转为与载体无关的 canonical model，并对所有
MVP 边界做一次性校验。

**工作内容**：

- 实现结构化 JSON/UBJSON reader；按 `v1.24.0` 的真实实现把 vector serialization
  作为单个 strict UBJSON 文档解析，拒绝截断和尾随字节；
- 校验 `json_version == "1.0"`、`cudnn_frontend_version == 12400`、必需字段和
  整数范围；
- 接受并忽略内嵌 `cudnn_backend_data`/`behavior_notes`，校验
  `variant_pack_uids`，并将 `pass_by_values` 与 typed tensor payload 做一致性校验；
  对非空 `workspace_modifications`、`variant_pack_replacements` 或非零
  `fe_workspace_size` 返回不支持诊断；
- 解析 `context`、`tensors`、`nodes`、UID、dim、stride、dtype、`is_virtual`、
  node input/output port 引用和 Conv attributes；
- 为未知 node 建立稳定的 `DFE_UNSUPPORTED_*` 诊断；禁止忽略未识别节点；
- 建立 `SerializedGraph`、`TensorDesc`、`ConvFpropDesc` 等内部模型；该模型不带
  MLIR 类型，便于独立测试；
- 对 X/W/Y 验证精确 packed stride、f32、rank-4、静态正维度和 output shape；
- 保证 JSON 和 UBJSON 解码为 field-for-field 等价的 canonical model；canonical
  model 不依赖对象键顺序或载体编码，后续如需可复现 dump 再定义确定的打印顺序；
- 不链接 CUDA/cuDNN execution path，不尝试恢复文档内的 GPU execution plan。

**测试**：

- 当前先入库一份按固定版本 Frontend serializer 源码校验的最小 JSON fixture，测试
  使用 vendored nlohmann/json 3.11.3 将同一 JSON 编码为 UBJSON；fixture 字段和公开
  execute 调用形状直接对照固定的开源 Frontend 源码，CUDA/cuDNN producer 不是项目
  build、CI、runtime 或 release 的前置依赖；
- JSON/UBJSON canonical model 等价测试；
- 缺 UID、重复 UID、截断/尾随 UBJSON、坏 schema、未知 node、非 f32、动态 shape、非
  packed stride、非 unit stride/dilation、output shape 错误和乘法溢出负例；
- 不对称 padding 和 C/K 非向量倍数的正例。

**退出条件**：使用 `DEEPFORGE_ENABLE_MLIR=OFF` 时，P1 可以在没有 MLIR 的情况下
独立配置并运行 importer 测试；所有不支持输入都在 importer 阶段失败，不把错误推迟
到 lowering。

### P2. Tensor/Linalg IR 导入和验证

**目标**：把 canonical model 转成可被上游 MLIR verifier 和 One-Shot Bufferize
接受的 destination-passing IR。

**工作内容**：

- 将 cuDNN logical dims/strides 转为 X `[N,H,W,C]`、W `[K,R,S,C]`、Y
  `[N,P,Q,K]`；
- 对非零 padding 生成标准 `tensor.pad`，并生成 `linalg.fill` 和
  `linalg.conv_2d_nhwc_fhwc`；padding 全零时 Conv 直接读取 X，不定义
  `cudnn.conv_fwd`；
- Y 作为函数边界 destination 传入并显式 fill 为 zero，避免未初始化输出和
  返回 buffer ownership 不清；
- 保留 static shape、identity layout 和 f32 element type；
- 实现独立 verifier，重新计算 indexing maps、padding 和 P/Q；
- MVP 保留 named Conv op，禁用 generic generalization；未来若打开 generalization，
  先增加对应 indexing-map verifier 和 lowering pattern；
- 输出 import-stage MLIR dump 和 compile metadata。

**测试**：

- MLIR parser/verifier golden tests；
- `[K,R,S,C]` 与错误的 `[R,S,C,K]` 对照；
- 对称/不对称 padding、C/K tail、最小尺寸和边界尺寸；
- 检查不存在自定义 cuDNN op、动态维度或隐式 element type cast。

**退出条件**：每个正例都能被 MLIR 22.1.8 标准 parser/verifier 读回，且能直接
进入 One-Shot Bufferize；每个负例有稳定错误码。

本工作区的 P2 importer/verifier 已完成：`DeepForge::MLIRImport` 的
`import_conv2d` 将 canonical model 生成为单函数、静态 f32 的
可选 `tensor.pad` + `linalg.fill` + `linalg.conv_2d_nhwc_fhwc` module；
`verify_conv2d_module` 会独立复算物理形状、padding、destination 和 unit
stride/dilation。UID、物理 shape、padding、stride/dilation 和函数名通过
`Conv2DCompileMetadata` 单独返回，不写入自定义 MLIR operation/attribute。
测试包含固定 MLIR golden、标准 parser/verifier round-trip、对称/不对称 padding、
最小尺寸、C/K tail、错误 layout/shape、动态维度和 HWCF 负例；另有
`one-shot-bufferize` pass smoke。P3 已在该输入边界上继续实现。

### P3. One-Shot Bufferize、allocation materialization 和 workspace

**目标**：固定内存所有权和内部 adapter 形状，确保 kernel 不调用 malloc/free。

**工作内容**：

- Tensor/Linalg 结构变换结束后只调用一次 One-Shot Bufferize；
- 显式设置 function-boundary identity layout 和 `allow-unknown-ops=false`；
- 将残留 `bufferization.alloc_tensor`、pad materialization 等合法化为静态
  `memref.alloc`/fill/copy/subview；这不是第二次 bufferization；
- 处理等价 Y result，必要时使用 `drop-equivalent-buffer-results`；
- 实现 checked-size workspace planner：静态 allocation、生命周期、64-byte
  offset、对齐 padding 和总大小；
- 把临时 allocation 改为 workspace view，删除 owned dealloc；
- 生成内部 memref descriptor adapter，外部仍只见 opaque handle + UID map +
  workspace pointer；
- 实现 pointer UID/null/alignment/可计算 overlap/overflow 检查，并明确容量由
  调用者保证；
- 在 target-independent 阶段冻结 workspace layout，后续 vectorizer 不得改变它。

**测试**：

- bufferization 前后 dialect legality；
- 只执行一次的 instrumentation/assertion；
- `alloc_tensor` 和 pad allocation 均被 planner 接管；
- offset 对齐、生命周期复用、size overflow 和无残留 owned allocation；
- X/W/Y/workspace 重叠、空指针、错误对齐和 UID 缺失负例；
- 并发使用不同 workspace 的 runtime 测试。

**退出条件**：固定输入可得到确定的 workspace size/offsets；最终内部 IR 不含
Tensor 或 Bufferization allocation residue，且不依赖生成 kernel 内部分配。

本工作区的 P3 已完成：One-Shot Bufferize 由单一入口运行并记录次数；pad allocation
经 checked-size/liveness workspace planner 改写为 64-byte aligned `memref.view`；
独立 verifier 拒绝 Tensor/Bufferization residue、owned allocation 和不一致 plan。
runtime 在进入隐藏 kernel 前完成 UID、null、对齐、区间 alias 和 workspace 检查，
并发测试使用各自 workspace。

### P4. Scalar lowering、LLVM conversion 和 runtime

**目标**：先得到可独立验证的 baseline object code。

**工作内容**：

- `deepforge-lower-direct-conv` 只消费 named Conv，生成
  `n, oh, ow, k, r, s, c` 循环和单 output scalar accumulator；
- 用标准 Linalg-to-Affine/SCF 路径处理剩余 fill/copy/pad 辅助 op；
- 实现 `VF=1` scalar schedule，明确 C tail、OH/OW/K 边界；
- 按 22.1.8 的合法 conversion 顺序完成 Vector/SCF/Affine/Index/Arith/MemRef/
  Func/CF -> LLVM；
- 翻译 LLVM Dialect 为 LLVM IR，再用 X86 target machine 生成 baseline object；
- 实现 `compile(serialized_graph)`、Executable metadata 和 Frontend-shaped
  `execute(handle, uid_map, workspace)` 的最小闭环；
- 先只选择 scalar，不提前引入 AVX feature。

**测试**：

- LLVM legality 和 `llc` 编译 smoke test；
- scalar object 对随机输入、全零、单位权重、负值和 NaN/Inf 的 reference test；
- 小尺寸 exhaustive indexing test，覆盖 padding 边界；
- 通过 variant-pack 写入调用者提供的 Y，验证不替换地址；
- 错误输入在 importer/runtime 边界返回可定位 status。

**退出条件**：在 x86-64 baseline 上，从 source-validated JSON fixture 生成并执行
scalar object，数值满足契约，且没有 Tensor/Linalg/Affine/SCF/MemRef 等残留。

本工作区的 P4 已完成：named Conv 被直接降低为静态 SCF 标量循环，辅助 Linalg op
经上游 loop conversion 清除，再完整转换到 LLVM dialect/IR/object。开发期使用 MLIR
ExecutionEngine 执行 baseline object，最终 legality、随机/边界数值和运行时负例均由
CTest 覆盖。

### P5. AVX2、AVX-512 和 feature dispatch

**目标**：在不改变 Graph、ABI、workspace 和数值契约的前提下加入 SIMD 变体。

**工作内容**：

- 从同一个 target-independent planned IR 复制三个 target-specific module；
- AVX2 使用 `vector<8xf32>`，AVX-512 使用 `vector<16xf32>`；
- 只沿连续 C reduction 加载 X/W，使用 vector FMA 后水平 reduction 到一个 Y
  scalar；禁止未显式 packing 的 K-vectorization；
- 为 `C % VF` 生成 scalar cleanup，不依赖 masked load；
- 分别设置 `+avx2,+fma` 和 `+avx512f,+fma`，确认 scalar module 不带高 ISA；
- runtime 以 CPUID + OSXSAVE/XGETBV 检查选择最高安全变体，失败回退 scalar；
- 检查汇编/目标属性和非法指令安全性，不依赖固定 LLVM intrinsic 文本；
- 对每个变体执行同一套 f64 reference 和误差报告。

**测试**：

- C 为 `1, 7, 8, 15, 16, 17, 31, 32, 33` 的 tail 矩阵；
- K、OH、OW、N 非 tile 倍数；
- 运行时 feature matrix：baseline、AVX2、AVX2+FMA、AVX-512F、OS 不保存
  SIMD 状态；
- scalar/AVX2/AVX-512 输出逐元素比较和最大误差统计；
- 反汇编确认 scalar 路径没有 AVX 指令。

**退出条件**：每个硬件能力选择正确变体，不支持的 CPU 不执行高 ISA，所有变体
满足同一数值和 workspace 契约。

本工作区的 P5 已完成：AVX2/AVX-512 分别生成 `vector<8xf32>` 和
`vector<16xf32>` C-reduction，`C % VF` 使用标量 cleanup；CPUID、FMA、OSXSAVE 和
XGETBV 共同决定安全分发。测试覆盖 C/K `1,7,8,15,16,17,31,32,33`、N=2、
29x30 spatial 边界、NaN/Inf/极值，并以反汇编确认 scalar 无 VEX/EVEX、AVX2 无
ZMM、AVX-512 使用 ZMM。

### P6. CLI、artifact、CI 和性能基线

**目标**：把编译器从测试程序整理成可重复使用的开发产物。

**工作内容**：

- 完成 `deepforge-compile` 的 JSON/UBJSON、`--input-format`、`--target`、
  `--emit`、IR dump 和错误诊断选项；
- 定义 `.dfo` artifact：编译 metadata、UID/shape/stride、workspace size、
  numeric contract、variant symbol/feature 和 object sections；
- 固定公共 status/error code，内部 kernel 符号默认 hidden；
- 增加 LIT/CTest/端到端测试和 22.1.8 工具链 CI；
- 记录小、中、大 Conv2D 的 scalar/AVX2/AVX-512 correctness 与性能基线；
- 将 padding copy、tiling 和 vector width 作为可测量优化，不以 cache address
  space 或 Machine Dialect 作为性能前提。

**退出条件**：从一份按固定 serializer 源码验证的 serialized fixture 可以重现 artifact、IR dump、运行
结果和测试报告；失败输入有稳定诊断，构建版本偏离时配置阶段即失败。

本工作区的 P6 已完成：`deepforge-compile` 支持 JSON/UBJSON/auto、三阶段 MLIR dump、
LLVM IR 和 `.dfo`；loader 可从文件或内存把三个 object 交给 ORC 并复用同一 runtime
校验/分发。artifact 有严格版本、长度、尾随数据和 checksum 校验，独立 JSON/UBJSON
编译结果逐字节可复现，所有内部 object 符号均为 `GLOBAL HIDDEN`。CI workflow、
ASan/UBSan jobs、small/medium/large benchmark 和原生 artifact 信任边界均已入库。

## 5. 测试门和验收矩阵

每个阶段至少有以下四类测试，不能只依靠最终 E2E：

| 类别 | 重点 |
|---|---|
| Unit | parser、UID、shape/stride、checked arithmetic、error code |
| IR golden | importer IR、bufferized IR、workspace rewrite、final legality |
| Negative | schema、node、dtype、layout、shape、alias、feature 和 ABI 前置条件 |
| E2E | scalar/AVX2/AVX-512、边界 padding、C/K/spatial 边界、数值容差 |

最小正例矩阵：

- `N=1` 和 `N>1`；
- `R/S=1`、`3`，不对称 padding；
- `C/K` 小于、等于、大于 8/16；
- 非方形 H/W 和跨多个 vector-width 的 spatial loop 边界；
- 全零、随机有限值、极值、NaN/Inf 分类；
- JSON 与 UBJSON 两种官方载体。

数值参考固定为相同 f32 输入、f64 累加；默认判定：

```text
abs(actual - reference) <= 1e-4 + 1e-3 * abs(reference)
```

放宽容差必须作为测试参数和报告字段出现，不能隐式修改全局规则。

## 6. 公开接口决策

已冻结：公开 execute 只有 Frontend-shaped overload，参数顺序为 handle、
`std::unordered_map<int64_t, void*>&`、workspace；不发布 handle-free overload，CPU
runtime 不解引用 handle。MVP 使用 opaque `void *` handle 和 DeepForge status，
保持 CPU-only build，不要求 CUDA/cuDNN headers 或 backend。

官方 `cudnnHandle_t`/`cudnn_frontend::error_t` exact types、GPU execution 和
Frontend samples 属于后续范围，不能反向成为 CPU MVP 的构建前置条件。

## 7. 风险和应对

| 风险 | 应对 |
|---|---|
| 新开发机或 CI 没有 MLIR | P0 构建/安装固定 tag；所有 pass API 以该安装验证 |
| pass 名称或 API 在 22.1.8 变化 | 用 C++ pipeline builder + legality tests，不依赖手写命令拼接 |
| UBJSON 被误当作私有 framing 或接受尾随数据 | 按 `to_ubjson`/strict `from_ubjson` 语义实现，增加截断和尾随负例 |
| `alloc_tensor` 漏出 workspace planner | materialize 后统一检查所有静态 allocation，未知 allocation 直接失败 |
| AVX feature 泄漏到 scalar | 独立 target module、反汇编检查、CPUID + XGETBV |
| vector reduction 结果错误 | C-lane indexing invariant、scalar reference、tail 专项测试 |
| pointer 容量不可由 ABI 证明 | 明确调用者容量前置条件，不宣称 portable bounds checking |
| padding copy 过慢 | 先保留正确 baseline，使用 benchmark 决定后续 fusion/边界优化 |
| 过早引入 Machine/AMX | MVP 禁止相关目录、op 和 pass；设置独立后续评审门 |

## 8. Definition of Done

MVP 只有同时满足以下条件才算完成：

- LLVM/MLIR 版本精确为 `llvmorg-22.1.8`，构建和 CI 可复现；
- 按 Frontend v1.24.0 serializer 源码验证的 JSON/UBJSON fixture 能导入，未知/不支持输入能稳定失败；
- layout、padding、UID、workspace、alignment、alias 和数值契约均有文档和测试；
- One-Shot Bufferize 只运行一次，最终 LLVM conversion 无残留非法 dialect；
- scalar、AVX2、AVX-512 变体共享 public contract，feature dispatch 安全；
- C/OH/OW/K tails、NaN/Inf 分类和默认误差阈值均有 E2E 结果；
- artifact 和 runtime 不暴露 memref descriptor 或裸 kernel ABI；
- Machine Dialect、AMX、bf16、fusion、dynamic shape 未混入 MVP；
- CPU-only Frontend-shaped ABI 已选定，并有 source-compatibility 形状测试；
  CUDA/cuDNN backend 未进入 MVP 构建依赖。

## 9. 实施结果与后续顺序

P0-P6 和 MVP 后 C0-C5 已按下面顺序完成：

1. 已完成：安装并验证 LLVM/MLIR `llvmorg-22.1.8`，记录 P0 工具版本；
2. 已完成：按 Frontend `v1.24.0` serializer 源码固化 Conv2D JSON fixture，并由
   vendored parser 生成等价 UBJSON；CPU release gate 是 serialization 和 execute
   signature 的源码级一致性，不要求 CUDA/cuDNN producer；
3. 已完成：实现 P1 canonical model 和 strict JSON/UBJSON importer，并覆盖完整 schema
   与负向路径；
4. 已完成：实现 P2 destination-passing MLIR importer/verifier 和 metadata 输出；
5. 已完成：P3 One-Shot Bufferize、allocation materialization 和 workspace plan；
6. 已完成：P4 scalar、P5 SIMD 分发以及 P6 CLI/artifact/质量门。
7. 已完成：MVP 后 C0-C4，包括通用 graph execution、完整 39-tag schema 识别、
   8 个基础 tag、rank 3-5 convolution forward/backward、14 个
   normalization/statistics tag 和 5 个带 integer sequence metadata 的
   sequence/attention tag。
8. 已完成：C5 CPU 低精度 storage/conversion、packed FP4/INT4、block-scale
   quantize/dequantize、FP8 matmul、MoE grouped matmul forward/backward 和全部
   4 个 FP8/MXFP8 attention tag；公开 UID variant-pack ABI 与 CPU-only 依赖边界
   保持不变。
9. 已开始：C6 首个独立测试增量为 block-scale conversion 的 E4M3/E8M0 scale 和
   MXFP8 forward/backward 的 E8M0 descale 加入 Frontend/CUTLASS
   `F8_128x4` 物理解码。
10. 已完成：C6.2 为单个 exact-shape、external plain f32 `POINTWISE` 子集实现
    runtime shape override；dynamic memref descriptor、Frontend-shaped
    execute/workspace overload、artifact v3 持久化、artifact reload 执行及错误
    metadata/上界测试均已覆盖。
11. 已完成：C6.3 为静态 f32 SDPA forward 实现 ragged/paged storage。Q/K/V/O 可
    独立 ragged，K/V 可独立 paged；runtime prefix/content 校验、紧凑 alias span、
    独立 page table、非法 page 安全寻址、artifact v4 持久化/reload 和错误 metadata
    均通过 Release、ASan、UBSan 测试。
12. 已完成：C6.4 标准 f32 SDPA metadata，包括 ragged forward row output/backward
    argument、max-total 校验 hint、独立紧凑 K/V page table、压缩 forward block
    mask 及带 dSink 的 forward/backward sink token。Artifact v5 持久化 packed
    sequence divisor；独立 reference、有限差分、精确分配、错误 metadata、reload、
    Release、ASan 和 UBSan 测试覆盖该增量。
13. 已完成：C6.5 为优化的静态 packed f32 Conv 路径加入首个生效的 Loop/Schedule
    cost model。它选择合法且 target-aware 的 K-output unroll factor，暴露选择结果，
    保留 baseline policy，在独立 accumulator 间复用 X load，并由决策、tail、数值、
    benchmark A/B、Release、ASan 和 UBSan gate 覆盖。
14. 已完成：C6.6 为标准和 FP8 MATMUL 实现 producer 序列化的 INT32 M/N/K extent
    override。Per-batch broadcast metadata、标准 MATMUL 有限 f32 padding、FP8 零
    padding、artifact reload、错误 descriptor、Release、ASan 和 UBSan 均已覆盖，
    且不改变公开 execute ABI。
15. 已完成：C6.7 Frontend runtime scalar pass-by-value input。External、仅作为
    input、全 1 dimension 的 tensor 使用普通 UID-map pointer 和 artifact argument
    metadata；pointwise broadcast、normalization epsilon、FP8 MATMUL control、错误
    根级 metadata/descriptor、Release、ASan 和 UBSan 覆盖该增量，且不改变 ABI 或
    artifact version。
16. 已完成：C6.8 Frontend 内嵌/fused pass-by-value scalar constant。Importer 识别
    全部六种精确 `{index,value}` variant，并要求 tensor payload 与根级
    `pass_by_values` 一一对应且 bit-preserving 一致。编译器生成 private read-only
    global，从公开 metadata 排除 graph-owned UID，并通过现有 target object 持久化
    value，且该增量不改变 artifact format。Pointwise 不可覆盖性、artifact reload、normalization
    epsilon、INT64 RNG seed/offset、FP8 MATMUL control、错误 metadata、Release、ASan
    和 UBSan 覆盖该增量。
17. 已完成：C6.9 多节点 exact-shape pointwise override。无 broadcast 的 plain-f32
    纯 `POINTWISE` DAG 可将同一个 runtime shape 传播给 virtual 中间值；内部 packed
    view 使用按序列化最大值分配的 workspace，virtual UID 不进入公开 override ABI。
    Dynamic 执行、artifact reload、错误 graph/UID、Release、ASan 和 UBSan 覆盖该
    增量，且不改变 artifact version。
18. 已完成：C6.10 有界 MATMUL descriptor override。单个标准 f32 `MATMUL` 接受
    Frontend A/B/C UID、shape、stride array，且三个 tensor 必须是 external plain。
    Runtime 校验序列化 dimension/byte-span 上界、M/N/K 关系、batch broadcast 和合法
    partial override；动态执行使用 runtime C/K extent 与 singleton-batch 选择。
    Artifact v6 记录有序 A/B/C role UID，同时保持 v1-v5 可读。Reload、错误关系/图、
    Release、ASan 和 UBSan 覆盖该增量，公开 execute/workspace ABI 不变。
19. 已完成：C6.11 有界 SDPA-forward descriptor override。单个 dense 标准 f32
    `SDPA` 接受 external plain rank-4 Q/K/V/O 与可选 Stats/Max/Sum_exp 的
    Frontend UID、shape、stride array。Runtime 只允许 B、Sq、Skv 在序列化
    dimension/byte-span 上界内变化，并保持固定 head、embedding、GQA 与 row-output
    关系。Operation 可无 mask 或使用 top-left causal；其他可选 attention 特性、
    virtual role 和组合图均被拒绝。Artifact v7 记录有序 role，同时保持 v1-v6
    可读。非连续 stride 执行、reload、partial/错误 override、Release、ASan 和 UBSan
    覆盖该增量，公开 execute/workspace ABI 不变。
20. 已完成：C6.12 有界 LOGICAL RESHAPE descriptor override。单个标准 f32
    `RESHAPE` 接受 external plain X/Y 的 Frontend UID、shape、stride array，X/Y 各自
    的固定 rank 可以不同。Runtime 校验序列化 dimension/byte-span 上界、受支持的
    非重叠 stride，以及完整或 partial override 后相等的最终元素总数；动态执行保持
    lexicographic reshape 顺序。Artifact v8 记录有序 X/Y role，同时保持 v1-v7 可读。
    非连续 stride 执行、reload、错误 span/关系/图、Release、ASan 和 UBSan 覆盖该
    增量，公开 execute/workspace ABI 不变。

剩余 C6 功能工作是已交付 exact-pointwise、单个标准 f32 MATMUL、LOGICAL RESHAPE
与 dense 标准 f32 SDPA-forward descriptor-override 子集之外的 dynamic 行为。除 enum 转换外，固定
v1.24.0 中的 `F16x16` 只有
attribute round-trip 覆盖，可执行 `INT8x32`
helper 只存在于 legacy backend/filter 路径；两者都没有可达的现代 serialized Graph
port 加物理映射。它们是等待 producer 契约和生成 fixture 的兼容 gate，不应在本项目
内推测布局。固定使用的
v1.24.0 serializer 无法表达 paged backward，该能力仅作为未来
schema 版本的兼容工作跟踪。后续 benchmark 驱动优化仍由 Loop/Schedule 层负责，并对照保留的
baseline policy 逐项评估外层 tiling、padding fusion 和多线程。这些优化
不得反向改变已冻结的 serialization、ABI、workspace ownership、数值和 artifact
契约。
