# MVP Pass Pipeline 设计

[English](pass-pipeline.en.md)

## 1. 总览

本 pipeline 把“输入协议解析”“结构化变换”“bufferization”“目标代码转换”
分开。输入是 cuDNN Frontend serialization，输出是三个 LLVM module/object
变体。Machine Dialect 和 AMX 不在图中。

```text
cuDNN Frontend v1.24.0 JSON/UBJSON
  |
  +-- importer: parse + validate + create standard Tensor/Linalg IR
  |
  +-- [A] Tensor/Linalg transforms
  |       validate named Conv; outer tiling/generalization disabled in MVP
  |
  +-- [B] One-Shot Bufferize (exactly once)
  |       function boundaries + identity layout
  |
  +-- [C] MemRef/Loop/Schedule
  |       workspace views, direct SCF Conv lowering,
  |       Linalg-to-loops for helpers, C-reduction vectorization
  |
  +-- [D] Complete LLVM conversion
  |       lower-affine, vector/scf/index/arith/memref/func/cf -> LLVM
  |
  +-- [E] LLVM IR + target-specific object variants
          scalar, AVX2+FMA, AVX-512F+FMA
```

每阶段结束都运行 `verify` 和 allowed-dialect 检查。失败即停止，不静默把未知
op 留到最终 translator。

## 2. 阶段 0：Importer（编译器入口外）

### `deepforge-import-cudnn`

| 项目 | 规定 |
|---|---|
| 输入 | 官方 Graph JSON 或 UBJSON |
| 输出 | 标准 Tensor/Linalg MLIR module + `Conv2DCompileMetadata` |
| 依赖 | 固定 cuDNN Frontend serialization schema |
| 不做 | 不调用文档内的 cuDNN GPU execution plan，不生成 `cudnn.*` op |

Importer 先验证 graph version、节点 tag、tensor 引用、UID、dim/stride、dtype、
padding/stride/dilation/convolution mode，再构造 IR。它只对一个合法且受支持的
Conv FPROP 图产生 IR；其他合法图得到明确 `DFE_UNSUPPORTED_*` 诊断。

P2 的 C++ 入口是 `deepforge::compiler::import_conv2d`；默认函数名为
`deepforge_conv2d`。导入结果严格限制为静态 f32 的标准 `func.func`，并由
`verify_conv2d_module` 在上游 `mlir::verify` 之外重新检查 physical order、padding、
destination 和 indexing attributes。

## 3. 阶段 A：Tensor/Linalg 变换

当前实现顺序：

```text
verify static named linalg.conv_2d_nhwc_fhwc
keep the named Conv op unchanged until One-Shot Bufferize
```

当前 MVP 未启用外层 tiling，也没有 `(1,28,28,32)` 这样的生效参数。这样先保留
可审计的 direct-convolution baseline；外层 N/OH/OW/K tiling 进入 Optimize 阶段，
必须由 benchmark 证明收益。R/S/C 是 reduction 维度，不能把 C tile 当作 K
vector lane。

生效的 cost model 在后续 Loop/Schedule 部分执行，不属于 importer、Stage A 或
runtime。它输入 target 硬件事实，为优化 Conv 路径输出显式 K-output unroll
schedule；外层 tiling 仍延后。

`deepforge-lower-direct-conv` 的 MVP 输入必须仍是 named
`linalg.conv_2d_nhwc_fhwc`；若未来开启 `linalg-generalize-named-ops`，必须先
为 generic op 增加等价的 indexing-map verifier 和 lowering pattern，不能只打开
该选项后继续复用 named-op pass。

### 前置条件

- shape 全静态；
- X/W/Y 的物理 layout 已验证；
- output destination 已由调用者传入并 fill；
- 没有未注册的 tensor op。

### 后置条件

- 所有 tensor/linalg 变换合法；
- 仍可由标准 MLIR verifier 读回；
- padding metadata 已登记，bufferization 后产生的临时 allocation 有明确的
  workspace planner 归属；
- 可以进入唯一的 One-Shot Bufferize。

## 4. 阶段 B：One-Shot Bufferize

### `one-shot-bufferize`

这是 MVP 唯一的 tensor-to-memref bufferization。概念选项：

```text
bufferize-function-boundaries
function-boundary-type-conversion=identity-layout-map
allow-unknown-ops=false
```

具体命令行 spelling 以 LLVM/MLIR 22.1.8 pass registry 为准，pipeline builder
必须显式设置等价的 `BufferizationOptions`。

bufferization 前不得运行旧式 `tensor-bufferize`/`linalg-bufferize`；bufferization
后不得再次运行另一个全局 bufferize pass。

随后运行：

```text
convert-bufferization-to-memref
drop-equivalent-buffer-results  # when a result aliases the Y destination
canonicalize
```

如果 One-Shot Bufferize 产生 `bufferization.alloc_tensor` 或 tensor.pad 的
materialization，先用等价的静态 allocation lowering 将其变成 `memref.alloc`、
`memref.fill/copy` 和必要的 `memref.subview`。这一步仍属于 bufferization 后的
合法化，不得重新运行 One-Shot Bufferize。

## 5. 阶段 C：MemRef/Loop/Schedule

### C1. `deepforge-workspace-plan`

在创建 ISA-specific module 之前，扫描 materialize 后的静态 `memref.alloc`：

1. 为每个临时 allocation 分配 64-byte aligned offset；
2. 将 allocation 改写为 flat i8 workspace 上的 `memref.view`；
3. 删除对应 owned deallocation；
4. 若发现动态或无法静态规划的 allocation，直接报错。

workspace plan 之后，从同一 module 克隆三个变体。variant lowering 只能创建 SSA
scalar/vector value，不得新增或改变 memref allocation 的数量、大小或生命周期。

### C2. Direct Conv lowering

直接消费 bufferized 的 `linalg.conv_2d_nhwc_fhwc`，逻辑 loop 顺序为
`n, oh, ow, k, r, s, c`。SIMD schedule 可按 `KU` strip-mine K，并携带 `KU` 个
独立 accumulator，使一次 X load 服务相邻输出；step-one loop 处理 `K mod KU`。
实现为受控 C++ lowering 函数，而不是可从命令行任意组合的注册 pass；它只处理
support matrix 中已验证的 named Conv2D。

### C3. Auxiliary Linalg-to-loops

在主 Conv 被移除后，`convert-linalg-to-loops` 只处理剩余的 `linalg.fill`、copy 或
padding materialization，生成 SCF loop。静态 packed Conv2D 的主 body 不依赖该
通用 pass 的 pattern recognition。

### C4. Canonicalize

辅助 loop lowering 后运行 canonicalize。当前 MLIR pipeline 显式展开 cost model
选择的独立 K 输出，但不展开 R/S，也不承诺某个后端 unroll 结果；增加其他 unroll
必须继续满足数值契约并由 benchmark 验证。

### C5. `deepforge-vectorize-conv-reduction`

对 C reduction 生成 `vector<VFxf32>`：

```text
VF=16: AVX-512 variant
VF=8 : AVX2 variant
VF=1 : scalar variant
```

向量计算必须是：

```text
x = load X[...,c_vec]
for u in [0,KU):
  vacc[u] = fma(x, load W[k_base+u,...,c_vec], vacc[u])
  Y[...,k_base+u] = vector.reduction(add, vacc[u]) + scalar_tail[u]
```

K 不是 vector lane；`vacc[u]` 是独立的 C-lane vector。不允许真正的
`load W[..., k_vec]`，除非另有显式 weight pack pass 和新的 ABI/ownership 契约。
scalar 变体使用独立的标量 reduction 路径。

## 6. 阶段 D：完整 LLVM conversion

LLVM/MLIR 22.1.8 中已经验证的实现顺序如下：

```text
canonicalize
convert-vector-to-llvm
lower-affine
convert-scf-to-cf
expand-strided-metadata
convert-index-to-llvm
convert-arith-to-llvm
finalize-memref-to-llvm
convert-func-to-llvm
convert-cf-to-llvm
reconcile-unrealized-casts
```

说明：

- `lower-affine` 消除 Affine；
- `convert-scf-to-cf` 消除仍存在的 SCF；
- `convert-vector-to-llvm` 把 Vector 变成 LLVM-compatible vector operations；
- index/arith/memref/func/cf 必须各自有 conversion；
- `mlir-translate --mlir-to-llvmir` 只翻译 LLVM Dialect，不负责这些 conversion；
- 任何残留 Tensor/Linalg/Affine/SCF/MemRef/Arith/Index/Func/CF 或
  `unrealized_conversion_cast` 都是 pipeline error。

`convert-vector-to-llvm` 与 `lower-affine` 的具体相对顺序由 LLVM 22.1.8 的
conversion patterns 验证；实现通过 legality test 固定，而不是依赖一条手写
shell 命令碰巧成功。

## 7. 阶段 E：LLVM IR 与 object

MVP 在 target-independent workspace plan 完成后，从同一个 validated IR 复制出
三个 target-specific module，分别运行对应的 vectorization/lowering 并生成
object。这样不会让一个 module-level target feature 意外泄漏到 scalar 变体。
三个变体的目标属性为：

```text
baseline       -> x86-64 baseline
avx2           -> +avx2,+fma
avx512         -> +avx512f,+fma
```

runtime metadata 记录每个函数的 required features。公共 `Executable` 只暴露
Frontend-shaped handle + variant-pack + workspace execute；裸函数参数不导出。

## 8. Pass 依赖和分析失效

```text
Importer
  -> named Tensor/Linalg verification
  -> One-Shot Bufferize
  -> Workspace plan/rewrite
  -> clone scalar/AVX2/AVX-512 modules
  -> Direct Conv lowering
  -> integrated C-reduction vectorization for SIMD variants
  -> Linalg-to-SCF loops for remaining auxiliary ops
  -> Lower/Convert all standard dialects
  -> LLVM translation
```

workspace planner 在改写前按当前 operation order 计算静态 lifetime；结果冻结为
`WorkspacePlan`。后续 lowering 不新增 allocation。未来若加入 tiling、循环重排或
额外 allocation，必须让相关分析失效并重新计划，不能复用旧结论。

## 9. 失败策略

编译期错误：schema/version、超出已声明 pointwise、MATMUL、RESHAPE、REDUCTION、TRANSPOSE、SDPA-forward override
policy 的动态行为、layout、dtype、未知 node、unsupported access、无法
workspace-plan、非法
vectorization 或最终 dialect residue。

运行期回退只发生在 CPU feature：AVX-512 -> AVX2 -> scalar。它不掩盖输入契约
错误；只有按已声明 C6 override policy 编译的图可提供经过校验的 runtime dimension
和 stride。

## 10. 命令行形态

```bash
deepforge-compile graph.json \
  --input-format=auto \
  --target=x86-64 \
  --emit=object \
  -o conv2d.dfo
```

调试输出：

```bash
deepforge-compile graph.json --dump-ir=imported:imported.mlir \
  --dump-ir=bufferized:bufferized.mlir \
  --dump-ir=llvm:llvm.mlir -o graph.dfo
deepforge-compile graph.json --emit=llvm-ir -o conv2d.ll
```

每个 dump 都必须能通过对应阶段的 verifier；最终 `.ll` 不能作为中间阶段的
替代品。

## 11. 测试矩阵

| 层次 | 断言 |
|---|---|
| Import | JSON/UBJSON、UID、schema 和 support diagnostics |
| Tensor/Linalg | layout、padding、P/Q、indexing maps |
| Bufferization | 只执行一次、无 unknown tensor op、workspace offsets |
| Schedule | C lane 对齐、C tail、精确 N/OH/OW/K loop bounds、reduction correctness |
| LLVM legality | 无残留方言、无未 reconcile cast |
| Runtime | UID map、null/alignment/overlap、workspace contract、CPUID |
| E2E | scalar/AVX2/AVX512 与 f64 reference 满足容差 |
| Regression | 反例必须失败且包含稳定错误码 |
