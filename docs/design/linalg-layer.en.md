# Linalg Layer Design

[中文](linalg-layer.md)

## 1. Responsibility

The Linalg layer represents the Tensor-layer Conv2D as a structured operation
that can be analyzed and transformed. The MVP preserves the upstream named
operation until direct loop lowering so that convolution reduction and
parallel iterator semantics are not discarded prematurely.

## 2. Named Operation and Layout

The MVP uses:

```text
linalg.conv_2d_nhwc_fhwc
```

Its physical operand shapes are:

```text
input  [N, H,  W,  C]
filter [K, R,  S,  C]   // F,H,W,C; F is output K
output [N, OH, OW, K]
```

The computation is:

```text
Y[n,oh,ow,k] =
  sum(r in [0,R), s in [0,S), c in [0,C))
    X[n, oh+r, ow+s, c] * W[k, r, s, c]
```

This layout exactly matches the cuDNN Frontend packed KRSC filter, where C is
the unit-stride dimension. Earlier designs incorrectly labeled `[R,S,C,K]` as
FHWC. A future K-contiguous packed weight path must use
`linalg.conv_2d_nhwc_hwcf` and an explicit packing step; renaming the layout is
not sufficient.

## 3. Linalg Example

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

Every output is explicitly filled with zero first. After bufferization, the
output remains destination-passing and never depends on uninitialized memory.

## 4. Padding Strategy

The MVP materializes nonzero padding with `tensor.pad` in the Tensor layer and
runs the Linalg convolution with zero internal padding. When all padding is
zero, X is passed directly to the named Conv operation, with no unnecessary
copy or workspace allocation. This keeps the convolution body as a standard
named operation and moves boundary handling into a measurable workspace copy.

The workspace planner records the padded shape and byte count:

```text
pad_bytes = N * (H + pre_h + post_h) * (W + pre_w + post_w) * C * sizeof(f32)
```

A future pad/conv fusion may replace this implementation if measurements show
that it is faster, but it must preserve the same graph and variant-pack
semantics.

## 5. Tiling (Deferred)

The current MVP does not run Linalg tiling. N/OH/OW/K lower directly to static
loops with step one. This establishes an untiled correctness and performance
baseline instead of freezing an unmeasured tuple such as `(1,28,28,32)` into
the contract. R/S/C always remain reduction dimensions.

If the Optimize phase introduces tiling, it must:

1. transform only dimensions permitted by the named operation's parallel and
   reduction iterator semantics;
2. handle boundary tiles without assuming OH/OW/K divisibility;
3. rerun canonicalization and verification after the transform;
4. avoid retaining dependence results computed before the transform.

## 6. Conditions for Generalization

`linalg-generalize-named-ops` is not a default step. It may be used only when a
custom indexing map or direct-convolution schedule requires a generic body.
Before entering Affine, the following iterator types must be verified:

```text
iterator_types = [parallel, parallel, parallel, parallel,
                  reduction, reduction, reduction]
```

Conceptual indexing maps are:

```text
X: (n,oh,ow,k,r,s,c) -> (n,oh+r,ow+s,c)
W: (n,oh,ow,k,r,s,c) -> (k,r,s,c)
Y: (n,oh,ow,k,r,s,c) -> (n,oh,ow,k)
```

These maps are also the reference for direct Conv lowering. Every vectorizing
transform must preserve the same reduction dimensions and must not interpret
a reduction lane as an output-K lane.

## 7. Bufferization Order

The required order is:

```text
Tensor/Linalg transforms
  -> canonicalize/cse
  -> one-shot-bufferize (exactly once)
  -> convert-bufferization-to-memref
```

One-Shot Bufferize function-boundary options use a predictable identity
layout. If an internal function needs a dynamic descriptor, the runtime
adapter must handle it rather than exposing an unknown layout to the generated
kernel.

The MVP workspace planner takes ownership of temporary allocations after
bufferization. It cannot assume memref addresses before bufferization or infer
aliases after LLVM lowering.

## 8. Exit Conditions

On leaving the Linalg layer:

- only standard Tensor, Linalg, Arith, and Func dialects, plus optional
  transform metadata, remain;
- every size and stride is static;
- the output is initialized;
- the named Conv parallel and reduction semantics remain valid;
- One-Shot Bufferize can complete once, without a second dialect-specific
  bufferization pass.

## 9. Future Extensions

The following require separate designs and cannot be inferred from the MVP
layout contract: K-contiguous weight packing, NCHW layout, grouped or depthwise
convolution, bias or activation fusion, im2col plus GEMM, and bf16/AMX. Each
requires a new serialized-graph support matrix, workspace rules, and numeric
tests.
