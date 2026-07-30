# Conv2D Forward End-to-End Example

[中文](example-conv2d.md)

This example uses `N=1, H=W=56, C=64, K=64, R=S=3`, stride and dilation
`[1,1]`, padding of 1 on every side, and f32 throughout. It demonstrates the
correct semantic path for the MVP rather than the complete printer output of a
particular pass.

## 1. cuDNN Frontend Graph Input

cuDNN Frontend logical dimensions and packed strides are:

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

These cuDNN strides produce the physical memory order:

```text
X: [N,H,W,C]
W: [K,R,S,C] = [F,H,W,C]
Y: [N,P,Q,K]
```

W must not be represented as `[R,S,C,K]`. That is a different K-contiguous
packing, does not match the input strides, and cannot be passed directly to
`linalg.conv_2d_nhwc_fhwc`.

## 2. Tensor/Linalg IR After Import

Padding is materialized first and the output is initialized:

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

Importer metadata records that `%x`, `%w`, and `%y_init` correspond to UIDs
101, 102, and 103. The importer does not insert a custom `cudnn.conv_fwd`
operation.

## 3. After One-Shot Bufferize

A conceptual identity-layout result is:

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

`deepforge-workspace-plan` rewrites `%padded` as a static workspace view, for
example:

```text
workspace alignment = 64
padded offset       = 0
padded bytes        = 1 * 58 * 58 * 64 * 4
```

The planner computes real offsets from all temporary-buffer lifetimes and
alignments. This example uses offset zero only because it has one temporary.
The final kernel never calls `malloc`.

## 4. Direct Conv Loops

For one output element `Y[n,oh,ow,k]`, the correct SIMD transform follows the C
reduction:

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

C is 64 here, so this instance has no tail. The general MVP still emits scalar
cleanup for `C % VF`. K remains a scalar loop; `W[k, ...]` must not become
`W[..., k:k+16]`.

Conceptual Vector IR is:

```mlir
%vzero = arith.constant dense<0.0> : vector<16xf32>
%xv = vector.load %x[%n, %h, %ww, %c]
    : memref<1x58x58x64xf32>, vector<16xf32>
%wv = vector.load %w[%k, %r, %s, %c]
    : memref<64x3x3x64xf32>, vector<16xf32>
%vacc = vector.fma %xv, %wv, %vzero : vector<16xf32>
// Real IR carries %vacc across all r/s/c blocks.
%sum = vector.reduction <add>, %vacc : vector<16xf32> into f32
memref.store %sum, %y[%n, %oh, %ow, %k]
    : memref<1x56x56x64xf32>
```

`%sum` is one output scalar, so the store is scalar. The reduction result must
not be treated as a vector output.

## 5. LLVM Lowering

After `lower-affine`, `convert-vector-to-llvm`, `convert-scf-to-cf`, the
index/arith/memref/func/cf-to-LLVM conversions, and cast reconciliation,
conceptual LLVM IR contains:

```llvm
%xv = load <16 x float>, ptr %x_ptr, align 4
%wv = load <16 x float>, ptr %w_ptr, align 4
%prod = fmul <16 x float> %xv, %wv
%vacc = fadd <16 x float> %vacc0, %prod
; vector reduction -> scalar fadd tree
store float %sum, ptr %y_ptr, align 4
```

With `+avx512f,+fma`, the LLVM backend may select a zmm FMA for the object
variant. DeepForge does not depend on a particular intrinsic name. Scalar and
AVX2 variants preserve the same internal arguments and output semantics.

## 6. Runtime Variant Pack

```cpp
std::unordered_map<int64_t, void *> variant_pack = {
    {101, input_host},
    {102, weight_host},
    {103, output_host},
};

deepforge::runtime::FrontendHandle handle = nullptr;
auto status = executable.execute(handle, variant_pack, workspace_host);
```

The runtime ignores the compatibility handle, resolves three UIDs from compile
metadata, checks null pointers, ranges, and non-overlap, then selects an
AVX-512, AVX2, or scalar internal function using CPUID. It neither changes
addresses in the variant pack nor requires another descriptor from the caller.

## 7. Correctness Check

The reference uses the same f32 input and accumulates in f64:

```text
for n, oh, ow, k:
  ref = 0.0 (f64)
  for r, s, c:
    ref += (f64)X[n,oh+r,ow+s,c] * (f64)W[k,r,s,c]
```

Scalar, AVX2, and AVX-512 outputs all satisfy:

```text
abs(actual - ref) <= 1e-4 + 1e-3 * abs(ref)
```

Fixture tests also verify:

- JSON and UBJSON import to identical tensor and node metadata;
- `[K,R,S,C]` filter indexing is correct;
- C tails, padding boundaries, and output shapes are correct;
- final MLIR contains no Tensor, Linalg, Affine, SCF, MemRef, Arith, Index,
  Func, or CF residue;
- unsupported layouts or nodes fail compilation instead of producing
  potentially incorrect code.
