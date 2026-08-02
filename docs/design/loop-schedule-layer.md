# Loop 与 Schedule 层设计

[English](loop-schedule-layer.en.md)

## 1. 职责

本层把已经 bufferize 的 Linalg Conv2D 变成显式 SCF 循环，为 SIMD 变体应用
C-reduction 向量化，并可展开相互独立的 K 输出以复用 X load。Schedule 是受控的
C++ lowering 阶段，不创建第二种 Schedule IR。当前实现不做外层 tiling 或显式
R/S unroll。

MVP 先做单线程 direct convolution。外层并行化和 OpenMP 等待有独立的同步、
线程池和 workspace 设计后再引入。

## 2. 循环语义

对 `linalg.conv_2d_nhwc_fhwc`，逻辑循环为：

```text
for n  in [0,N)       parallel
  for oh in [0,OH)    parallel
    for ow in [0,OW)  parallel
      for k in [0,K)  parallel
        acc = 0
        for r in [0,R) reduction
          for s in [0,S) reduction
            for c in [0,C) reduction
              acc += X[n,oh+r,ow+s,c] * W[k,r,s,c]
        Y[n,oh,ow,k] = acc
```

`N/OH/OW/K` 的迭代之间没有写冲突；`R/S/C` 属于同一个 output element 的
归约。任何 reorder、tile 或 vectorize 都必须保留这组 iterator semantics。

## 3. Pass 顺序

MVP 的 schedule 顺序固定为：

```text
bufferized linalg.conv_2d_nhwc_fhwc
  -> deepforge-workspace-plan
  -> clone scalar / AVX2 / AVX-512 modules
  -> 为每个 target variant 选择 direct-Conv schedule
  -> direct Conv lowering (consume the named Conv op)
       scalar: scalar SCF reduction
       AVX2:   vector<8xf32> + scalar C tail + 可选 K-output unroll
       AVX512: vector<16xf32> + scalar C tail + 可选 K-output unroll
  -> convert-linalg-to-loops (remaining fill/copy/pad ops)
  -> canonicalize
```

Direct Conv lowering 按已验证的 indexing maps 直接生成 SCF scalar/vector loop；
标准 `convert-linalg-to-loops` 只处理剩余的 fill/copy/pad materialization。不能先
把 Conv 丢给通用转换再依赖脆弱的 loop pattern recognition，也不把
`affine-super-vectorize` 的启发式行为当作 Conv2D 语义保证。

## 4. 正确的 C-reduction 向量化

cuDNN packed filter 的 C 维是连续的，所以 SIMD lane 表示 C。一个 schedule 可以
为相邻的 `KU` 个 K 输出维护独立 accumulator，并在它们之间复用一次 X load；这是
output-loop unroll，不是 K-vectorization：

```text
for n, oh, ow:
  for k_base = 0 to floor(K/KU)*KU step KU:
    vacc[0:KU] = vector.splat(0.0)  // KU 个 vector<VFxf32>
    tail_acc[0:KU] = 0.0            // KU 个独立 f32

    for r, s:
      for c_vec = 0 to C - VF step VF:
        x_vec = load X[n, oh+r, ow+s, c_vec : VF contiguous f32]
        for u = 0 to KU:
          w_vec = load W[k_base+u, r, s, c_vec : VF contiguous f32]
          vacc[u] = vector.fma(x_vec, w_vec, vacc[u])

      for c = floor(C/VF)*VF to C:
        x = X[n, oh+r, ow+s, c]
        for u = 0 to KU:
          tail_acc[u] += x * W[k_base+u, r, s, c]

    for u = 0 to KU:
      Y[n, oh, ow, k_base+u] = vector.reduce_add(vacc[u]) + tail_acc[u]

  使用同一 body 和 KU=1 处理 K mod KU 个输出
```

关键不变量：

1. `x_vec` 和 `w_vec` 的 lane `i` 表示同一个 C 索引；
2. 每个 `vacc[u]` 相互独立，并在所有 R/S/C block 间持续累加；
3. 每个 reduction 结果都是一个 output scalar，SIMD lane 永不表示 K；
4. C tail 使用标量 cleanup，不读越界，不需要 masked load；
5. K 不要求是 `KU` 的倍数，step-one cleanup loop 处理尾部。

如果未来改为真正的 K-lane-vectorization，必须先把 filter 从 `[K,R,S,C]` 显式 pack 成
`[R,S,C,K]`，并将 pack buffer 的 ownership、cache 和执行成本写入新契约。
仅交换 indexing map 会产生错误结果。

## 5. MLIR 形态

scalar variant 使用四层 step-one outer `scf.for`。SIMD variant 使用三层 N/OH/OW
loop、步长为 `KU` 的 K main loop 和 step-one K cleanup loop。R/S 通过 `iter_args`
携带 `KU` 组 vector/scalar accumulator；C loop 含 `vector.fma`、scalar cleanup，
每个输出有一个 `vector.reduction <add>`。可用
`deepforge-compile --dump-ir=llvm:<path>` 保存完整转换后 module。`vector.load` 不附加
runtime 无法证明的 64-byte alignment；X/W/Y 的公开要求只有 `alignof(float)`。

## 6. Tiling 和展开（Optimize 阶段）

N/OH/OW 保持 step-one loop，当前没有生效的 tile 参数或显式 R/S unroll。SIMD
lowering 生成精确的 C 和 K cleanup loop；所有 upper bound 都是静态值，不需要
masked 或越界访问。

后续候选 tiling/unroll 必须逐项通过正确性矩阵和绑核 benchmark。cost model 可以
使用 cache size、cache line 和向量寄存器压力估算工作集，但不能生成 L1/L2/L3
address space，也不能把数据“放入”某级 cache。软件 prefetch 当前关闭。

### 6.1 生效的 Cost Model

首个生效的 cost model 位于编译期 Loop/Schedule 层，并明确只服务于原有优化的静态、
连续 f32 单 `CONV_FPROP` 路径。通用 C2-C6 graph 仍使用 `generic-reference`；importer、
runtime 和 Machine Dialect 都不选择性能 schedule。

模型按 target 固定 `VF` 为 `1`、`8` 或 `16`，并考虑
`KU in {1,2,4,8}`。scalar 和 `baseline` policy 只允许 `KU=1`。SIMD auto policy 中，
候选须满足 `KU <= K`，且 `2*KU+4` 不超过 target vector register budget（AVX2 为
`16`，AVX-512 为 `32`）。确定性的估算公式为：

```text
channel_steps = R*S*(floor(C/VF) + C mod VF)
input_loads   = (floor(K/KU) + K mod KU) * channel_steps
weight_loads  = K * channel_steps
score         = 2*input_loads + weight_loads + 16*KU
```

公共的 N/OH/OW 因子不影响候选排序，因此省略。评分鼓励 X-load 复用，同时计入
register/IR 压力。baseline 始终是候选，score 相同时保留较小 schedule，因此不存在
无候选导致编译失败。选择结果以 `direct-c-vf<VF>-ku<KU>` 暴露在
`CompilationResult::variants` 和 benchmark CSV 中；
`CompileOptions::schedule_policy = kBaseline` 提供稳定的 A/B 和诊断回退。

CPUID/XGETBV runtime dispatch 仍只是 capability check，不是性能模型。模型不改变
Graph 语义、公开 ABI、workspace ownership、artifact format 或数值容差。cache
tiling、padding fusion 和 threading 仍需由后续 benchmark 驱动。

## 7. 依赖与合法性

当前 direct lowering 从已验证的 named Conv 语义构造循环，不缓存
DependenceAnalysis。未来每个改变循环结构的 pass 都必须在变换前查询当前
dependence/alias 信息，并让 PassManager 正确失效和重算分析。

Conv2D 的 `Y[n,oh,ow,k]` 写入互不别名，因此外层可以在未来并行化；归约
accumulator 只能在 C/R/S 内重排。浮点重排属于 contracts.md 允许的误差范围，
不能在没有数值开关的情况下启用更激进的 fast-math。

## 8. 并行化（后续）

MVP 不产生 `affine.parallel` 或 OpenMP runtime call。后续并行化需要同时定义：

- 线程分块和 scheduling policy；
- 每线程 workspace/临时 buffer；
- output 写入和异常传播；
- CPU feature dispatch 与线程池生命周期。

只把 `affine.for` 改成 `affine.parallel` 不足以完成该功能。

## 9. 退出条件

本层完成后：

- Linalg 已变成 SCF + MemRef + Arith/Vector；
- 所有 vector reduction 都有明确 reduction target；
- 没有 `affine.vector_load` 这类非上游临时 op；
- C cleanup 和 N/OH/OW/K 的精确 loop bound 可从 IR 看见；
- workspace 临时 buffer 已有静态 offset；
- 完整 LLVM conversion 后不残留 SCF/MemRef/Arith/Vector 等源方言。
