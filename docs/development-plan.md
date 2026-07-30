# DeepForge 开发方案

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

- 本地安装前缀，例如 `$DEEPFORGE_LLVM_PREFIX`；当前工作区使用
  `/local_data/yanghesong/opt/llvm-22.1.8`；
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
  `variant_pack_uids`；对非空 `pass_by_values`、`workspace_modifications`、
  `variant_pack_replacements` 或非零 `fe_workspace_size` 返回不支持诊断；
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

- 当前先入库一份按固定版本 Frontend serializer 形状校验的最小 JSON fixture，测试
  使用 vendored nlohmann/json 3.11.3 将同一 JSON 编码为 UBJSON；后续可在独立的
  CUDA/cuDNN 环境追加官方 producer 生成的 fixture，但 producer 不是 DeepForge CPU
  build、CI 或 runtime 的前置依赖；
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

P0-P6 已按下面顺序完成：

1. 已完成：安装并验证 LLVM/MLIR `llvmorg-22.1.8`，记录 P0 工具版本；
2. 已完成：按 Frontend `v1.24.0` serializer 源码固化 Conv2D JSON fixture，并由
   vendored parser 生成等价 UBJSON；官方 producer 交叉生成的 fixture 保留为 MVP
   发布门，不阻塞 CPU-only 开发；
3. 已完成：实现 P1 canonical model、strict JSON/UBJSON importer 和 59 项测试检查；
4. 已完成：实现 P2 destination-passing MLIR importer/verifier 和 metadata 输出；
5. 已完成：P3 One-Shot Bufferize、allocation materialization 和 workspace plan；
6. 已完成：P4 scalar、P5 SIMD 分发以及 P6 CLI/artifact/质量门。

后续只进入 benchmark 驱动的 Optimize 阶段：先建立绑核、隔离负载的可重复测量，
再逐项评估外层 tiling、padding fusion 和多线程；这些优化不得反向改变已冻结的
serialization、ABI、workspace ownership、数值和 artifact 契约。
