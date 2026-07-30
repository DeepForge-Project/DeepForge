# Loop 与 Schedule 层设计

## 1. 职责

本层把已经 bufferize 的 Linalg Conv2D 变成显式 SCF 循环，并为 SIMD 变体应用
C-reduction 向量化。Schedule 是受控的 C++ lowering 阶段，不创建第二种
Schedule IR。当前 MVP 不做外层 tiling 或显式 R/S unroll。

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
  -> direct Conv lowering (consume the named Conv op)
       scalar: scalar SCF reduction
       AVX2:   vector<8xf32> + scalar C tail
       AVX512: vector<16xf32> + scalar C tail
  -> convert-linalg-to-loops (remaining fill/copy/pad ops)
  -> canonicalize
```

Direct Conv lowering 按已验证的 indexing maps 直接生成 SCF scalar/vector loop；
标准 `convert-linalg-to-loops` 只处理剩余的 fill/copy/pad materialization。不能先
把 Conv 丢给通用转换再依赖脆弱的 loop pattern recognition，也不把
`affine-super-vectorize` 的启发式行为当作 Conv2D 语义保证。

## 4. 正确的 C-reduction 向量化

cuDNN packed filter 的 C 维是连续的，所以 MVP 沿 C 做 SIMD。每次只计算一个
输出 K，不能把向量 lane 当成 K：

```text
for n, oh, ow, k:
  vacc = vector.splat(0.0)       // vector<VFxf32>
  tail_acc = 0.0                 // f32

  for r, s:
    for c_vec = 0 to C - VF step VF:
      x_vec = load X[n, oh+r, ow+s, c_vec : VF contiguous f32]
      w_vec = load W[k, r, s, c_vec : VF contiguous f32]
      vacc = vector.fma(x_vec, w_vec, vacc)

    for c = floor(C/VF)*VF to C:
      tail_acc += X[n, oh+r, ow+s, c] * W[k, r, s, c]

  acc = vector.reduction(add, vacc) + tail_acc
  store Y[n, oh, ow, k] = acc
```

关键不变量：

1. `x_vec` 和 `w_vec` 的 lane `i` 表示同一个 C 索引；
2. `vacc` 在所有 R/S/C block 间持续累加；
3. reduction 结果是一个 output scalar；
4. C tail 使用标量 cleanup，不读越界，不需要 masked load；
5. K 不要求是 VF 的倍数，因为 K 仍是标量循环。

如果未来改为 K-vectorization，必须先把 filter 从 `[K,R,S,C]` 显式 pack 成
`[R,S,C,K]`，并将 pack buffer 的 ownership、cache 和执行成本写入新契约。
仅交换 indexing map 会产生错误结果。

## 5. MLIR 形态

实际 variant IR 使用四层 outer `scf.for`、R/S 的 `scf.for iter_args`、完整 C block
loop、标量 C tail、`vector.fma` 和 `vector.reduction <add>`。可用
`deepforge-compile --dump-ir=llvm:<path>` 保存完整转换后 module；转换前的 loop
形态由 compiler E2E 测试直接检查。`vector.load` 不附加 runtime 无法证明的
64-byte alignment；X/W/Y 的公开要求只有 `alignof(float)`。

## 6. Tiling 和展开（Optimize 阶段）

当前 MVP 的 N/OH/OW/K 循环步长都是 1，没有生效的 tile 参数，也没有显式 R/S
unroll。C tail 由 SIMD lowering 生成；N/OH/OW/K 使用精确静态 upper bound，因此
不存在当前实现中的 tile tail。

后续候选 tiling/unroll 必须逐项通过正确性矩阵和绑核 benchmark。cost model 可以
使用 cache size、cache line 和向量寄存器压力估算工作集，但不能生成 L1/L2/L3
address space，也不能把数据“放入”某级 cache。软件 prefetch 当前关闭。

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
