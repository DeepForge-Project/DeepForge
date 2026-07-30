# Conv2D FWD 端到端示例

本文示例固定为：`N=1, H=W=56, C=64, K=64, R=S=3`，stride/dilation 为
`[1,1]`，四边 padding 为 `1`，所有数据类型为 f32。它展示的是 MVP 的正确
语义路径，不是某个 pass 的完整 printer 输出。

## 1. cuDNN Frontend Graph 输入

cuDNN Frontend 的 logical dimensions 和 packed strides：

```text
X:
  dim    = [1, 64, 56, 56]       // [N,C,H,W]
  stride = [64*56*56, 1, 64*56, 64]
  uid    = 101

W:
  dim    = [64, 64, 3, 3]        // [K,C,R,S]
  stride = [64*3*3, 1, 64*3, 64]
  uid    = 102

Y:
  dim    = [1, 64, 56, 56]       // [N,K,P,Q]
  stride = [64*56*56, 1, 64*56, 64]
  uid    = 103

CONV_FPROP:
  padding  = [1, 1]
  stride   = [1, 1]
  dilation = [1, 1]
  mode     = CROSS_CORRELATION
```

cuDNN 的 stride 使实际内存顺序为：

```text
X: [N,H,W,C]
W: [K,R,S,C] = [F,H,W,C]
Y: [N,P,Q,K]
```

这里不能把 W 写成 `[R,S,C,K]`。那是另一种 K-contiguous packing，既不符合
本输入的 stride，也不能直接配合 `linalg.conv_2d_nhwc_fhwc`。

## 2. Import 后的 Tensor/Linalg IR

padding 先物化，输出先初始化：

```mlir
func.func @conv2d_fwd(
    %x: tensor<1x56x56x64xf32>,
    %w: tensor<64x3x3x64xf32>,
    %y_init: tensor<1x56x56x64xf32>)
    -> tensor<1x56x56x64xf32> {
  %zero = arith.constant 0.0 : f32
  %padded = tensor.pad %x low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%n: index, %h: index, %ww: index, %c: index):
    tensor.yield %zero : f32
  } : tensor<1x56x56x64xf32> to tensor<1x58x58x64xf32>

  %out = linalg.fill ins(%zero : f32)
      outs(%y_init : tensor<1x56x56x64xf32>)
      -> tensor<1x56x56x64xf32>
  %result = linalg.conv_2d_nhwc_fhwc
      {dilations = dense<1> : tensor<2xi64>,
       strides = dense<1> : tensor<2xi64>}
      ins(%padded, %w : tensor<1x58x58x64xf32>,
                       tensor<64x3x3x64xf32>)
      outs(%out : tensor<1x56x56x64xf32>)
      -> tensor<1x56x56x64xf32>
  return %result : tensor<1x56x56x64xf32>
}
```

Importer metadata records that tensor arguments `%x`, `%w` and `%y_init` correspond
to UIDs 101, 102 and 103. It does not insert a custom `cudnn.conv_fwd` operation.

## 3. One-Shot Bufferize 后

概念性的 identity-layout 结果如下：

```mlir
func.func @conv2d_fwd_bufferized(
    %x: memref<1x56x56x64xf32>,
    %w: memref<64x3x3x64xf32>,
    %y: memref<1x56x56x64xf32>) {
  %padded = memref.alloc() : memref<1x58x58x64xf32>
  // bufferized tensor.pad: zero-fill and copy x into padded
  // linalg.fill + linalg.conv remain until loop lowering
  return
}
```

`deepforge-workspace-plan` 把 `%padded` 改为 workspace 中的静态 view，例如：

```text
workspace alignment = 64
padded offset       = 0
padded bytes        = 1 * 58 * 58 * 64 * 4
```

实际 offset 由 planner 按所有临时 buffer 的生命周期和对齐计算，示例中的 0
只是因为只有一个临时 buffer。最终 kernel 不调用 `malloc`。

## 4. Direct Conv 循环

对一个输出元素 `Y[n,oh,ow,k]`，正确的 SIMD 变换是沿 C reduction：

```text
for r = 0 .. 3:
  for s = 0 .. 3:
    for c = 0 .. 64 step 16:
      xv = X[n, oh+r, ow+s, c:c+16]
      wv = W[k, r, s, c:c+16]
      vacc = fma(xv, wv, vacc)

sum = horizontal_add(vacc)
Y[n,oh,ow,k] = sum + scalar_tail
```

本例 C=64，所以没有 tail；通用 MVP 仍会生成 `C % VF` 的标量 cleanup。K 是
标量循环，不能把 `W[k, ...]` 误写成 `W[..., k:k+16]`。

概念性 Vector IR：

```mlir
%vzero = arith.constant dense<0.0> : vector<16xf32>
%xv = vector.load %x[%n, %h, %ww, %c]
    : memref<1x58x58x64xf32>, vector<16xf32>
%wv = vector.load %w[%k, %r, %s, %c]
    : memref<64x3x3x64xf32>, vector<16xf32>
%vacc = vector.fma %xv, %wv, %vzero : vector<16xf32>
// 实际 IR 在所有 r/s/c block 中传递 %vacc
%sum = vector.reduction <add>, %vacc : vector<16xf32> into f32
memref.store %sum, %y[%n, %oh, %ow, %k]
    : memref<1x56x56x64xf32>
```

`%sum` 是一个 output scalar，因此使用 scalar `memref.store`；不能把 reduction
结果当成 vector 存储。

## 5. LLVM lowering

完成 `lower-affine`、`convert-vector-to-llvm`、`convert-scf-to-cf`、
`convert-index/arithmetic/memref/func/cf-to-llvm` 和 cast reconcile 后，概念性
LLVM IR 包含：

```llvm
%xv = load <16 x float>, ptr %x_ptr, align 4
%wv = load <16 x float>, ptr %w_ptr, align 4
%prod = fmul <16 x float> %xv, %wv
%vacc = fadd <16 x float> %vacc0, %prod
; vector reduction -> scalar fadd tree
store float %sum, ptr %y_ptr, align 4
```

在启用 `+avx512f,+fma` 的 object 变体中，LLVM backend 可以选择 zmm FMA；
DeepForge 不依赖某个 intrinsic 文本名称。scalar/AVX2 变体保持相同的内部
参数和输出语义。

## 6. Runtime variant-pack

```cpp
std::unordered_map<int64_t, void *> variant_pack = {
    {101, input_host},
    {102, weight_host},
    {103, output_host},
};

deepforge::runtime::FrontendHandle handle = nullptr;
auto status = executable.execute(handle, variant_pack, workspace_host);
```

Runtime 忽略兼容用 handle，按编译 metadata 查找三个 UID，检查非空、范围和
不重叠，然后按 CPUID 选择 AVX-512、AVX2 或 scalar 内部函数。它不会改变
variant-pack 中的地址，也不会要求调用者提供另一个 descriptor。

## 7. 正确性检查

reference 使用同样的 f32 输入，以 f64 累加：

```text
for n, oh, ow, k:
  ref = 0.0 (f64)
  for r, s, c:
    ref += (f64)X[n,oh+r,ow+s,c] * (f64)W[k,r,s,c]
```

scalar、AVX2 和 AVX-512 输出均检查：

```text
abs(actual - ref) <= 1e-4 + 1e-3 * abs(ref)
```

此外，fixture 测试确认：

- JSON 与 UBJSON 导入得到相同 tensor/node metadata；
- `[K,R,S,C]` filter indexing 正确；
- C tail、padding 边界和 output shape 正确；
- 最终 MLIR 没有 Tensor/Linalg/Affine/SCF/MemRef/Arith/Index/Func/CF 残留；
- 不支持 layout 或 node 时编译失败，而不是生成可能错误的代码。
