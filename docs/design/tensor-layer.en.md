# Tensor Layer Design

[中文](tensor-layer.md)

## 1. Responsibility

The Tensor layer is the first structured MLIR stage after the cuDNN Frontend
Graph. It represents shapes, padding, and destination-passing semantics. It
does not select an ISA or allocate final host pointers.

The MVP importer directly generates standard `tensor` and `linalg` operations;
there is no temporary `cudnn.conv_fwd` MLIR operation. Consequently, every
tensor operation has an upstream `BufferizableOpInterface` when the module
reaches One-Shot Bufferize.

## 2. cuDNN Dimensions to MLIR Dimensions

cuDNN Frontend describes layout using logical dimension order plus strides.
The MVP accepts only these packed layouts:

```text
X logical dim [N,C,H,W], stride [H*W*C, 1, W*C, C]
W logical dim [K,C,R,S], stride [R*S*C, 1, S*C, C]
Y logical dim [N,K,P,Q], stride [P*Q*K, 1, Q*K, K]
```

The importer converts them to ranked tensors in physical order:

```text
X: tensor<NxHxWxCxf32>
W: tensor<KxRxSxCxf32>
Y: tensor<NxPxQxKxf32>
```

W is `[K,R,S,C]`, or FHWC in Linalg terminology. It is not `[R,S,C,K]`. If a
serialized graph does not use the exact packed strides above, the importer
returns `DFE_INVALID_LAYOUT`.

## 3. Import-Time Validation

All static derivation is completed during import so later passes never have to
guess about dynamic shapes:

1. Validate `json_version`, the Frontend version, and explicit UIDs.
2. Require exactly one `CONV_FPROP`, with no pointwise, bias, or activation
   node.
3. Require rank-4 f32 X/W/Y tensors with strictly positive dimensions.
4. Validate strides, dilation, convolution mode, and padding.
5. Derive P/Q from padding and compare them with the serialized Y dimensions.
6. Require every referenced tensor to exist and every UID to be unique.

Dynamic shapes are outside the MVP. Failed shape inference is an import error;
the importer does not continue after making a best-effort static assumption.

## 4. Tensor IR Form

For `N=1,H=W=56,C=64,K=64,R=S=3,padding=1`:

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

`%y_init` is the caller-provided destination for the serialized Y UID. This
lets One-Shot Bufferize reuse the external Y buffer. The runtime does not
allocate output memory inside the kernel and does not depend on ownership of a
returned tensor. If LLVM lowering retains an equivalent memref result, it must
be removed with `drop-equivalent-buffer-results` or an equivalent internal ABI
rewrite.

Nonzero padding is materialized with `tensor.pad` and an f32 zero fill. When
all padding is zero, Conv reads X directly and creates no temporary copy or
workspace allocation. The Linalg convolution itself always has zero padding.
Original pre- and post-padding values remain in compile metadata for workspace
calculation and diagnostics.

## 5. Why `tensor.cast` Is Not a Precision Conversion

MVP inputs, weights, outputs, and accumulation are all f32. `tensor.cast` can
change compatible shape or abstract layout information; it cannot convert f32
to bf16. A bf16/AMX path requires explicit conversions, accumulation rules, and
a new numeric contract, and remains deferred.

## 6. Pass Boundaries

### 6.1 Import

```text
serialized Graph
  -> validate + derive static shapes
  -> optional tensor.pad / linalg.fill / linalg.conv_2d_nhwc_fhwc
```

### 6.2 Tensor and Linalg Transforms

The following may run before bufferization:

```text
canonicalize
cse
tensor.expand_shape / extract_slice (only when a legal schedule needs them)
```

To preserve the named-Conv baseline, the current MVP does not run these
optional structural transforms or Linalg tiling. They are legal candidates for
the Optimize phase, not part of the active pipeline. Generalization also
remains disabled.

The MVP does not use separate `tensor-bufferize` or `linalg-bufferize` passes.
Those legacy names would incorrectly suggest local bufferization followed by a
second bufferization pass.

### 6.3 Bufferization Preconditions

- No `cudnn.*` or unknown tensor operations remain.
- Every tensor shape is static.
- Linalg input/output types and indexing relations are fixed.
- The function-boundary policy is selected: compiler internals use a
  fixed-layout memref adapter.
- Padding metadata is recorded. Temporary padding allocations produced by
  bufferization must be registered with the workspace planner afterward. Any
  remaining `bufferization.alloc_tensor` must first become a static
  `memref.alloc`; the planner cannot scan only one allocation form.

## 7. Exit Conditions

At the end of Tensor/Linalg structural transforms:

- the module round-trips through the standard MLIR parser and printer;
- there are no cuDNN-specific operations;
- there are no dynamic dimensions;
- a verifier can independently recompute the Conv2D physical layout, padding,
  and output shape;
- the compiler pipeline runs `one-shot-bufferize` next and exactly once.

## 8. Tests

Importer tests cover at least:

- both JSON and UBJSON carriers;
- symmetric and asymmetric padding;
- C/K values not divisible by 8 or 16;
- missing and duplicate UIDs;
- each non-packed stride pattern;
- rejection diagnostics for non-f32 data, dynamic shapes, non-unit stride or
  dilation, and fused nodes.

The current implementation is in `lib/MLIR/Conv2DImporter.cpp` and emits the
default function name `deepforge_conv2d`. It preserves a standard Tensor/Linalg
module and returns UID, physical shape, and original padding in a separate
`Conv2DCompileMetadata`. Later workspace and runtime stages therefore do not
infer metadata from MLIR operation names or custom attributes.
