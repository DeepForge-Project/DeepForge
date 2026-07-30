# Machine Dialect：延后设计说明

[English](machine-dialect.en.md)

## 1. 当前状态

Machine Dialect **不属于 MVP**。当前代码和主 pipeline 不应注册、生成或依赖
`machine.*` op，也不创建对应的 TableGen 文件。

MVP 使用：

```text
Linalg -> Affine/SCF -> Vector -> LLVM
```

这条路径已经足够表达 scalar、AVX2 和 AVX-512 的 Conv2D schedule。添加一个
没有明确 lowering、寄存器语义和验证器的中间 dialect 只会隐藏未解决的问题。

## 2. 延后的原因

原设计把以下不同层次混在一起：

- SIMD 运算语义；
- ISA 指令选择；
- cache placement；
- AMX tile register/config 生命周期；
- 线程/并行资源。

Vector/LLVM 已能表达 MVP 的 SIMD 语义，LLVM backend 负责最终指令选择；L1/L2/L3
不是编译器可直接寻址的内存空间。因此这些内容暂不需要新 IR。

## 3. 何时重新评估

只有同时满足以下条件才开启 Machine Dialect 设计：

1. 至少两个后端（例如 CPU SIMD 和 GPU/AMX）需要相同的资源分配或调度语义；
2. 上游 dialect 无法表达该共同语义；
3. 能给出 verifier、canonical form、buffer/alias 规则和每个 target 的
   完整 lowering；
4. 有独立的 round-trip、legality、数值和性能测试；
5. 该抽象不把 cache 当作 address space，也不把 LLVM register allocation
   的职责提前复制到 DeepForge。

## 4. AMX 后续边界

AMX 不是“把 vector<16xf32> 换成 tile”这么简单。引入前必须定义：

- bf16 输入、f32 累加和允许误差；
- filter/input packing（尤其是 cuDNN KRSC 到 tile-friendly layout）；
- tilecfg 初始化、销毁、线程迁移和 OS XSTATE 权限；
- 8 个 tile register 的活跃区间和寄存器分配；
- tile load/store 的行 stride 与 workspace；
- AVX-512/scalar fallback 和 CPUID 分发。

MLIR 22.1.8 已有上游 X86 相关能力可供评估。后续优先采用上游 X86/LLVM
dialect/conversion，而不是先定义 `machine.amx_tile_load` 再手写未验证的
intrinsic。

## 5. Target profile 与 cache

MVP 的 target profile 是 compiler 配置，不是 Machine op：

```text
TargetProfile:
  triple = x86_64-unknown-linux-gnu
  vector widths = {1, 8, 16}
  cache line = 64 bytes
  cache capacities = cost-model hints
```

cache capacity 只影响 tiling 评分。软件 prefetch、NUMA placement、L1/L2/L3
分配和 DRAM 迁移都不由当前 IR 表示。

## 6. 迁移条件

未来若加入 Machine Dialect，必须先删除当前文档中所有“唯一自定义层”的承诺，
新增独立设计评审，并保持 MVP 的标准 Vector/LLVM 路径可用。Machine lowering
失败不得破坏 scalar fallback。
