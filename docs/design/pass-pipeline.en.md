# MVP Pass Pipeline Design

[中文](pass-pipeline.md)

## 1. Overview

This pipeline separates input-protocol parsing, structured transforms,
bufferization, and target-code conversion. Its input is cuDNN Frontend
serialization and its output is three LLVM module and object variants. Machine
Dialect and AMX are not part of the pipeline.

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

Every stage ends with `verify` and an allowed-dialect check. A failure stops the
pipeline; unknown operations are never silently left for the final translator.

## 2. Stage 0: Importer (Outside the Pass Pipeline)

### `deepforge-import-cudnn`

| Item | Requirement |
|---|---|
| Input | Official Graph JSON or UBJSON |
| Output | Standard Tensor/Linalg MLIR module plus `Conv2DCompileMetadata` |
| Dependency | Pinned cuDNN Frontend serialization schema |
| Excluded | Does not invoke an embedded cuDNN GPU execution plan or emit `cudnn.*` operations |

The importer validates Graph version, node tags, tensor references, UIDs,
dimensions and strides, data types, padding, stride, dilation, and convolution
mode before constructing IR. It emits IR only for one valid, supported Conv
FPROP graph; every other valid graph receives an explicit
`DFE_UNSUPPORTED_*` diagnostic.

The P2 C++ entry point is `deepforge::compiler::import_conv2d`, and its default
function name is `deepforge_conv2d`. Import results are restricted to a static
f32 standard `func.func`. In addition to upstream `mlir::verify`,
`verify_conv2d_module` checks physical order, padding, destination arguments,
and indexing attributes.

## 3. Stage A: Tensor/Linalg Transforms

The current implementation performs:

```text
verify static named linalg.conv_2d_nhwc_fhwc
keep the named Conv op unchanged until One-Shot Bufferize
```

The MVP enables no outer tiling and has no active tuple such as
`(1,28,28,32)`. This preserves an auditable direct-convolution baseline.
N/OH/OW/K tiling belongs to the Optimize phase and must demonstrate a benchmark
benefit. R/S/C are reduction dimensions; a C tile cannot be treated as a K
vector lane.

The active cost model runs later in the Loop/Schedule portion, not in the
importer, Stage A, or runtime. It consumes target facts and produces an explicit
K-output unroll schedule for the optimized Conv path. Outer tiling remains
deferred.

The MVP input to `deepforge-lower-direct-conv` must remain a named
`linalg.conv_2d_nhwc_fhwc`. Before enabling
`linalg-generalize-named-ops`, the compiler must add an equivalent indexing-map
verifier and lowering pattern for generic operations. Merely enabling the
option while reusing a named-op-only pass is invalid.

### Preconditions

- Every shape is static.
- Physical X/W/Y layouts are verified.
- The caller supplies and the IR fills the output destination.
- No unregistered tensor operations remain.

### Postconditions

- All Tensor/Linalg transforms are legal.
- The module round-trips through the standard MLIR verifier.
- Padding metadata is recorded, and temporary allocations produced after
  bufferization have a clear workspace-planner owner.
- The module can enter its only One-Shot Bufferize invocation.

## 4. Stage B: One-Shot Bufferize

### `one-shot-bufferize`

This is the MVP's only tensor-to-memref bufferization. Conceptual options are:

```text
bufferize-function-boundaries
function-boundary-type-conversion=identity-layout-map
allow-unknown-ops=false
```

Exact command-line spelling follows the LLVM/MLIR 22.1.8 pass registry. The
pipeline builder sets equivalent `BufferizationOptions` explicitly.

Legacy `tensor-bufferize` or `linalg-bufferize` passes must not run before this
stage, and no second global bufferization pass may run afterward.

The pipeline then runs:

```text
convert-bufferization-to-memref
drop-equivalent-buffer-results  # when a result aliases the Y destination
canonicalize
```

If One-Shot Bufferize leaves `bufferization.alloc_tensor` or materializes
`tensor.pad`, a static allocation lowering converts it to `memref.alloc`,
`memref.fill` or copy, and required `memref.subview` operations. This is
post-bufferization legalization, not another One-Shot Bufferize invocation.

## 5. Stage C: MemRef, Loop, and Schedule

### C1. `deepforge-workspace-plan`

Before ISA-specific modules are created, scan materialized static
`memref.alloc` operations:

1. Assign each temporary a 64-byte-aligned offset.
2. Rewrite each allocation as a view into a flat i8 workspace.
3. Remove the corresponding owned deallocation.
4. Fail if any allocation is dynamic or cannot be planned statically.

After workspace planning, clone three variants from the same module. Variant
lowering may create scalar or vector SSA values but must not change the number,
size, or lifetime of memref allocations.

### C2. Direct Conv Lowering

Consume the bufferized `linalg.conv_2d_nhwc_fhwc` directly. The logical loop
order is `n, oh, ow, k, r, s, c`. A SIMD schedule may strip-mine K by `KU` and
carry `KU` independent accumulators so one X load serves adjacent outputs; a
step-one loop handles `K mod KU`. This is a controlled C++ lowering function
rather than an arbitrarily composable command-line pass. It accepts only the
named Conv2D operation already validated against the support matrix.

### C3. Auxiliary Linalg-to-Loops

After the main Conv is removed, `convert-linalg-to-loops` handles only remaining
`linalg.fill`, copy, or padding materialization and emits SCF loops. The main
static packed Conv2D body does not depend on generic-pass pattern recognition.

### C4. Canonicalization

Run canonicalization after auxiliary loop lowering. The current MLIR pipeline
explicitly unrolls independent K outputs selected by the cost model, but does
not unroll R/S and makes no promise about backend unrolling. Any additional
unrolling must preserve the numeric contract and demonstrate a benchmark
benefit.

### C5. `deepforge-vectorize-conv-reduction`

Generate `vector<VFxf32>` for the C reduction:

```text
VF=16: AVX-512 variant
VF=8 : AVX2 variant
VF=1 : scalar variant
```

The required computation is:

```text
x = load X[...,c_vec]
for u in [0,KU):
  vacc[u] = fma(x, load W[k_base+u,...,c_vec], vacc[u])
  Y[...,k_base+u] = vector.reduction(add, vacc[u]) + scalar_tail[u]
```

K is not a vector lane: `vacc[u]` is a separate C-lane vector. A true
`load W[..., k_vec]` remains forbidden unless a weight-pack pass and new ABI and
ownership contract are introduced. The scalar variant uses a dedicated scalar
reduction path.

## 6. Stage D: Complete LLVM Conversion

The order verified with LLVM/MLIR 22.1.8 is:

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

Notes:

- `lower-affine` removes Affine.
- `convert-scf-to-cf` removes remaining SCF.
- `convert-vector-to-llvm` converts Vector to LLVM-compatible vector
  operations.
- Index, Arith, MemRef, Func, and CF each require their own conversion.
- `mlir-translate --mlir-to-llvmir` translates the LLVM Dialect and does not
  perform these conversions.
- Any remaining Tensor, Linalg, Affine, SCF, MemRef, Arith, Index, Func, CF, or
  `unrealized_conversion_cast` is a pipeline error.

The exact relative order of `convert-vector-to-llvm` and `lower-affine` is
validated against LLVM 22.1.8 conversion patterns and fixed by legality tests,
not by a shell command that happened to work once.

## 7. Stage E: LLVM IR and Objects

After target-independent workspace planning, the MVP copies the same validated
IR into three target-specific modules. Each receives its own vectorization and
lowering and emits an object. This prevents module-level target features from
leaking into the scalar variant. Target attributes are:

```text
baseline       -> x86-64 baseline
avx2           -> +avx2,+fma
avx512         -> +avx512f,+fma
```

Runtime metadata records required features for each function. Public
`Executable` exposes only the Frontend-shaped handle, variant-pack, workspace,
and execute operation. Raw function arguments are not exported.

## 8. Pass Dependencies and Analysis Invalidation

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

Before rewriting, the workspace planner computes static lifetimes in current
operation order and freezes them into a `WorkspacePlan`. Later lowering adds no
allocation. Future tiling, loop reordering, or additional allocation must
invalidate the affected analysis and rerun planning; stale results cannot be
reused.

## 9. Failure Strategy

Compile-time errors include schema or version mismatch, dynamic behavior
outside the declared pointwise, MATMUL, RESHAPE, and SDPA-forward override policies,
invalid layout or data type,
unknown nodes, unsupported access, an unplannable workspace, illegal
vectorization, and final dialect residue.

Runtime fallback applies only to CPU features: AVX-512 to AVX2 to scalar. It
does not hide input-contract failures. Only a graph compiled with a declared C6
override policy may supply validated runtime dimensions and strides.

## 10. Command-Line Form

```bash
deepforge-compile graph.json \
  --input-format=auto \
  --target=x86-64 \
  --emit=object \
  -o conv2d.dfo
```

Debug output:

```bash
deepforge-compile graph.json --dump-ir=imported:imported.mlir \
  --dump-ir=bufferized:bufferized.mlir \
  --dump-ir=llvm:llvm.mlir -o graph.dfo
deepforge-compile graph.json --emit=llvm-ir -o conv2d.ll
```

Each dump must pass the verifier for its stage. Final `.ll` output is not a
substitute for intermediate-stage verification.

## 11. Test Matrix

| Layer | Assertions |
|---|---|
| Import | JSON/UBJSON, UIDs, schema, and support diagnostics |
| Tensor/Linalg | Layout, padding, P/Q, and indexing maps |
| Bufferization | Exactly one invocation, no unknown tensor operations, workspace offsets |
| Schedule | C-lane correspondence, C tail, exact N/OH/OW/K bounds, reduction correctness |
| LLVM legality | No source-dialect residue and no unreconciled casts |
| Runtime | UID map, null/alignment/overlap, workspace contract, and CPUID |
| End to end | Scalar/AVX2/AVX-512 satisfy tolerance against the f64 reference |
| Regression | Invalid inputs fail with stable error codes |
