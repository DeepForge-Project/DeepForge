# Linalg 层设计

[English](linalg-layer.en.md)

## 1. 职责

Linalg 层把 Tensor 层的 Conv2D 表达为可分析、可变换的结构化运算。MVP 保留
上游 named op 到 direct loop lowering 前，避免过早丢失卷积的
reduction/parallel 迭代信息。

## 2. Named Op 和布局

MVP 使用：

```text
linalg.conv_2d_nhwc_fhwc
```

其物理 operand 形状为：

```text
input  [N, H,  W,  C]
filter [K, R,  S,  C]   // F,H,W,C; F == output K
output [N, OH, OW, K]
```

计算语义：

```text
Y[n,oh,ow,k] =
  sum(r in [0,R), s in [0,S), c in [0,C))
    X[n, oh+r, ow+s, c] * W[k, r, s, c]
```

该布局正好匹配 cuDNN Frontend 的 packed KRSC filter：C 是 stride-1 维。旧
设计中把 `[R,S,C,K]` 标作 FHWC 是错误的；若以后选择 K-contiguous packed
weight，应改用 `linalg.conv_2d_nhwc_hwcf` 并增加显式 packing，不能只改名字。

## 3. Linalg 示例

```mlir
// %y_init is the function-boundary destination for the serialized Y UID.
%zero = arith.constant 0.0 : f32
%out = linalg.fill ins(%zero : f32)
    outs(%y_init : tensor<1x56x56x64xf32>)
    -> tensor<1x56x56x64xf32>
%result = linalg.conv_2d_nhwc_fhwc
    {dilations = dense<1> : tensor<2xi64>,
     strides = dense<1> : tensor<2xi64>}
    ins(%padded, %weight : tensor<1x58x58x64xf32>,
                           tensor<64x3x3x64xf32>)
    outs(%out : tensor<1x56x56x64xf32>)
    -> tensor<1x56x56x64xf32>
```

所有 output 都先显式 fill 为 zero。这样 bufferization 后的 output 是
destination-passing 的，且不会依赖未初始化内存。

## 4. Padding 策略

MVP 在 Tensor 层通过 `tensor.pad` 物化非零 padding，Linalg convolution 使用零
padding。padding 全零时 X 直接进入 named Conv，不产生无意义 copy 或 workspace。
选择该策略是为了让卷积 body 保持标准 named op，并把真实边界分支移到一个可测量的
workspace copy。

workspace planner 记录 padded shape 和字节数：

```text
pad_bytes = N * (H + pre_h + post_h) * (W + pre_w + post_w) * C * sizeof(f32)
```

后续如果 pad/conv fusion 能证明更快，可替换实现，但必须保持同一 graph 和
variant-pack 语义。

## 5. Tiling（延后）

当前 MVP 未运行 Linalg tiling；N/OH/OW/K 直接降低为步长 1 的静态循环。这样先
建立未分块的正确性和性能 baseline，避免把未经测量的 `(1,28,28,32)` 固化成
契约。R/S/C 始终是 reduction 维度。

Optimize 阶段若引入 tiling，合法性要求为：

1. 只在 named op 的 parallel/reduction 迭代器语义允许时变换；
2. 处理边界 tile，不假设 OH/OW/K 可整除；
3. 变换后重新运行 canonicalize 和 verifier；
4. 不缓存变换前的 dependence 结论。

## 6. Generalization 的使用条件

`linalg-generalize-named-ops` 不是默认步骤。只有在自定义 indexing map 或
direct-conv schedule 需要 generic body 时才使用，并在进入 Affine 前验证：

```text
iterator_types = [parallel, parallel, parallel, parallel,
                  reduction, reduction, reduction]
```

概念性 indexing maps：

```text
X: (n,oh,ow,k,r,s,c) -> (n,oh+r,ow+s,c)
W: (n,oh,ow,k,r,s,c) -> (k,r,s,c)
Y: (n,oh,ow,k,r,s,c) -> (n,oh,ow,k)
```

这份 map 也作为 Direct Conv lowering 的 reference；任何向量化变换都必须保持
相同的 reduction 维度，而不能把 reduction lane 当作 output K lane。

## 7. Bufferization 顺序

正确顺序是：

```text
Tensor/Linalg transforms
  -> canonicalize/cse
  -> one-shot-bufferize (exactly once)
  -> convert-bufferization-to-memref
```

One-Shot Bufferize 的 function-boundary 选项固定为可预测的 identity layout；
该 Conv Tensor/Linalg 路径保持静态。独立的 C6 generic pointwise 路径可使用内部
dynamic memref 签名，由 runtime adapter 提供经过校验的 dimension/stride，且该
descriptor 不进入 public API。

MVP 的 workspace planner 在 bufferization 后接管临时 allocation；它不能在
bufferization 前凭空假定 memref 地址，也不能在 LLVM lowering 后再猜测别名。

## 8. 退出条件

离开 Linalg 层时：

- 只剩标准 Tensor/Linalg/Arith/Func（以及可选的 Transform metadata）；
- 该 Tensor/Linalg Conv 路径的所有尺寸和 strides 已静态确定；
- output 已初始化；
- named Conv 的 parallel/reduction 语义仍然合法；
- One-Shot Bufferize 可以一次完成，不需要第二个 dialect-specific bufferize。

## 9. 未来扩展

以下能力单独设计，不从 MVP 的 layout 契约中隐式推导：K-contiguous weight
packing、NCHW layout、grouped/depthwise convolution、bias/activation fusion、
im2col+GEMM 和 bf16/AMX。每项都需要新的 serialized graph support matrix、
workspace 规则和数值测试。
