# Tensor 层设计

[English](tensor-layer.en.md)

## 1. 职责

Tensor 层是 cuDNN Frontend Graph 到结构化 MLIR 的第一段。它负责表达形状、
padding 和 destination-passing 语义，不负责选择 ISA，也不负责分配最终 host
指针。

MVP importer 直接生成标准 `tensor` 和 `linalg` op；不存在临时的
`cudnn.conv_fwd` MLIR op。这样进入 One-Shot Bufferize 时，所有 tensor op 都有
上游的 `BufferizableOpInterface`。

## 2. cuDNN 维度到 MLIR 维度

cuDNN Frontend 用 logical dimension 顺序配合 stride 描述 layout。MVP 只接受
以下三种 packed layout：

```text
X logical dim [N,C,H,W], stride [H*W*C, 1, W*C, C]
W logical dim [K,C,R,S], stride [R*S*C, 1, S*C, C]
Y logical dim [N,K,P,Q], stride [P*Q*K, 1, Q*K, K]
```

importer 把它们转成物理顺序的 ranked tensor：

```text
X: tensor<NxHxWxCxf32>
W: tensor<KxRxSxCxf32>
Y: tensor<NxPxQxKxf32>
```

这里的 W 是 `[K,R,S,C]`，即 Linalg 术语中的 FHWC；它不是
`[R,S,C,K]`。如果 serialized graph 的 stride 不是上面的精确 packed stride，
importer 返回 `DFE_INVALID_LAYOUT`。

## 3. 导入阶段的校验

导入时一次性完成所有静态推导，避免后续 pass 对动态 shape 作隐式猜测：

1. 检查 `json_version`、frontend version 和显式 UID。
2. 检查 graph 只有一个 `CONV_FPROP`，没有 pointwise、bias 或 activation 节点。
3. 检查 X/W/Y 是 rank-4、f32、所有维度大于 0。
4. 检查 stride、dilation、convolution mode 和 padding。
5. 按 padding 计算 P/Q，并与 Y 的 serialized dim 比较。
6. 检查所有引用 tensor 都存在且 UID 唯一。

动态 shape 不在 MVP 内。形状推导失败是导入错误，而不是“尽量静态化”后继续
编译。

## 4. Tensor IR 形式

以 `N=1,H=W=56,C=64,K=64,R=S=3,padding=1` 为例：

```mlir
func.func @conv2d_fwd_tensor(
    %x: tensor<1x56x56x64xf32>,
    %w: tensor<64x3x3x64xf32>,
    %y_init: tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32> {
  %zero = arith.constant 0.0 : f32
  %padded = tensor.pad %x low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%n: index, %h: index, %ww: index, %c: index):
    tensor.yield %zero : f32
  } : tensor<1x56x56x64xf32> to tensor<1x58x58x64xf32>

  %y = linalg.fill ins(%zero : f32)
      outs(%y_init : tensor<1x56x56x64xf32>)
      -> tensor<1x56x56x64xf32>
  %result = linalg.conv_2d_nhwc_fhwc
      {dilations = dense<1> : tensor<2xi64>,
       strides = dense<1> : tensor<2xi64>}
      ins(%padded, %w : tensor<1x58x58x64xf32>,
                       tensor<64x3x3x64xf32>)
      outs(%y : tensor<1x56x56x64xf32>)
      -> tensor<1x56x56x64xf32>
  return %result : tensor<1x56x56x64xf32>
}
```

`%y_init` 是调用者提供的 destination，对应 serialized graph 的 Y UID。这样
One-Shot Bufferize 可以复用 Y 的外部 buffer；runtime 不会在 kernel 内为输出
重新分配内存，也不会依赖返回 tensor 的 ownership。若 LLVM lowering 仍保留
等价的 memref result，必须用 `drop-equivalent-buffer-results` 或等价的内部
ABI rewrite 将它消除。

非零 padding 使用 `tensor.pad` 物化，填充值是 f32 zero；padding 全零时 Conv
直接读取 X，不生成临时 copy 或 workspace allocation。Linalg convolution 本身的
padding 始终为零；原始 pre/post padding 保存在编译 metadata 中，用于 workspace
大小和诊断。

## 5. 为什么不使用 `tensor.cast` 做精度转换

MVP 的输入、权重、输出和累加都是 f32，不做元素类型转换。`tensor.cast` 只能
改变兼容的形状/抽象布局，不能把 f32 变成 bf16；bf16/AMX 需要显式转换、累加
和数值契约，留到后续版本。

## 6. Pass 边界

### 6.1 Import

```text
serialized Graph
  -> validate + derive static shapes
  -> optional tensor.pad / linalg.fill / linalg.conv_2d_nhwc_fhwc
```

### 6.2 Tensor/Linalg 变换

允许在 bufferization 前运行：

```text
canonicalize
cse
tensor.expand_shape / extract_slice (only when a legal schedule needs them)
```

当前 MVP 为保持 named Conv baseline，不实际运行这些可选结构变换或 Linalg tiling；
它们是 Optimize 阶段的合法候选，不是现行 pipeline。generalization 仍然关闭。

MVP 不使用独立的 `tensor-bufferize` 或 `linalg-bufferize`。这些旧名称会造成
“先局部 bufferize、后再次 bufferize”的错误预期。

### 6.3 Bufferization 前置条件

- 不含 `cudnn.*` 或未知 tensor op；
- 所有 tensor shape 静态；
- Linalg op 的输入/输出类型和 indexing 关系已经确定；
- 函数边界策略已选定：编译器内部使用固定 layout 的 memref adapter；
- padding metadata 已登记；bufferization 产生的临时 pad allocation 必须在
  bufferization 后登记给 workspace planner。任何残留的
  `bufferization.alloc_tensor` 也必须先 materialize 成静态 `memref.alloc`，不能
  让 planner 只扫描一种 allocation。

## 7. 退出条件

Tensor/Linalg 结构变换阶段结束时必须满足：

- 仍可用标准 MLIR parser/printer 读回；
- 不含 cuDNN 专用 op；
- 没有动态维度；
- Conv2D 的物理 layout、padding 和输出形状可由 verifier 独立复算；
- 当前 compiler pipeline 随后且仅运行一次 `one-shot-bufferize`。

## 8. 测试

导入测试至少覆盖：

- JSON 和 UBJSON 两种载体；
- 对称和不对称 padding；
- C/K 非 8/16 倍数；
- UID 缺失/重复；
- 每种非 packed stride；
- 非 f32、动态 shape、非 unit stride/dilation、融合 node 的拒绝诊断。

当前实现位于 `lib/MLIR/Conv2DImporter.cpp`，默认生成函数名为
`deepforge_conv2d`。它保留 module 的标准 Tensor/Linalg 形式，并通过独立的
`Conv2DCompileMetadata` 传递 UID、物理 shape 和原始 padding；后续 workspace/runtime
阶段不需要从 MLIR operation 名称或自定义 attribute 反向猜测这些信息。
