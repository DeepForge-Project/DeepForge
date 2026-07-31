# cuDNN Graph Operation Coverage Plan

[中文](cudnn-graph-coverage-plan.md)

## 1. Status

This is the approved post-MVP roadmap. It expands the current support contract
only as each corresponding stage is implemented and validated. The existing
static, packed, f32
`CONV_FPROP` path remains the regression baseline throughout the work.

The plan is based on the pinned cuDNN Frontend `v1.24.0` source and LLVM/MLIR
`llvmorg-22.1.8`. Normal builds, tests, and execution remain CPU-only and do
not require CUDA or a cuDNN backend library.

## 2. Meaning of "All Operations"

The proposed compatibility boundary is the cuDNN Frontend serialized Graph
protocol, not every descriptor in the lower-level cuDNN Backend API.

The pinned serializer emits 39 distinct top-level node tags:

```text
ADA_LAYER_NORM          ADA_LAYER_NORM_BPROP    BATCHNORM
BATCHNORM_INFERENCE     BLOCK_SCALE_DEQUANTIZE  BLOCK_SCALE_QUANTIZE
BN_FINALIZE             CONCATENATE             CONV_DGRAD
CONV_FPROP              CONV_WGRAD              DBN
DBN_WEIGHT              GENSTATS                INSTANCE_NORM
INSTANCE_NORM_BPROP     LAYER_NORM              LAYER_NORM_BPROP
MATMUL                  MATMUL_FP8              MOE_GROUPED_MATMUL
MOE_GROUPED_MATMUL_BWD  POINTWISE               REDUCTION
RESAMPLE                RESHAPE                 RMS_NORM
RMS_NORM_BPROP          RNG                     ROPE
ROPE_BWD                SDPA                    SDPA_BWD
SDPA_FP8_BWD            SDPA_FP8_FWD            SDPA_MXFP8_BWD
SDPA_MXFP8_FWD          SLICE                   TRANSPOSE
```

Some public Graph APIs, including softmax and mask helpers, are composite and
do not have a distinct serialized tag. They are covered when every primitive
node emitted by their serialization is supported.

`Graph::deserialize` in v1.24.0 explicitly reconstructs only 12 of the 39
tags: `CONV_FPROP`, `CONV_DGRAD`, `CONV_WGRAD`, `POINTWISE`, `REDUCTION`,
`MATMUL`, `RESAMPLE`, `SLICE`, `TRANSPOSE`, `SDPA`, `SDPA_BWD`, and
`MOE_GROUPED_MATMUL`. DeepForge will therefore track three separate states:

1. **Recognized:** the pinned serialized schema is parsed without dropping a
   node or attribute.
2. **Executable:** DeepForge can verify, lower, compile, and run the graph on
   CPU.
3. **Validated:** positive, negative, artifact, sanitizer, and numeric tests
   have passed for the declared shape, layout, and data-type subset.

"All operations supported" may be claimed only when the capability matrix
contains no unrecognized tag and every in-scope row reaches the declared
validated level. A recognized but unimplemented node must return a stable
`unsupported operation` diagnostic rather than being ignored.

## 3. Compatibility Contract

The expansion preserves these boundaries:

- input protocol: cuDNN Frontend `v1.24.0`, `json_version == "1.0"`, JSON and
  UBJSON carriers;
- execution target: Linux x86-64 CPU;
- public execution shape: opaque handle, UID-to-host-pointer variant pack,
  caller-owned workspace, and DeepForge status;
- compiler IR: upstream Tensor, Linalg, Arith, Math, SCF/Affine, Vector,
  MemRef, and LLVM dialects, without a public or serialized `cudnn.*` dialect;
- memory ownership: generated kernels do not allocate, and temporary storage
  is planned into caller-provided workspace;
- dependency discovery: CMake and the build script locate dependencies from
  caller-selected prefixes; no local absolute path is part of the contract.

The external execute API remains source compatible. The internal generated
kernel ABI and `.dfo` payload must become generic and are not public ABI.

## 4. Required Architecture Changes

### 4.1 Generic canonical Graph

Replace the one-Conv model with a protocol-independent graph representation:

- variable-rank dimensions and strides;
- the complete v1.24.0 data-type catalog and checked storage-size metadata;
- ordered input and output ports for tagged node attribute variants;
- arbitrary multi-node DAGs with stable node order and UID references;
- graph inputs, outputs, virtual tensors, pass-by-value scalars, and constants;
- explicit dynamic-dimension, ragged, reorder, and workspace metadata even
  while their execution is deferred;
- per-operation verification and shape inference, separate from JSON parsing.

A table-driven capability registry owns the recognized schema version,
supported attributes, legal data types/layouts, lowering state, and diagnostic
for each operation and mode. This avoids scattering support claims across
importer switch statements.

### 4.2 Standard MLIR lowering

Protocol nodes are normalized before bufferization:

- views and data movement map to Tensor/Linalg operations;
- pointwise and broadcast operations map to Linalg, Arith, and Math;
- reductions, matmul, convolution, and pooling use named or generic Linalg
  where their indexing semantics can be verified;
- normalization, RoPE, softmax, and attention first decompose into tested
  primitives;
- One-Shot Bufferize still runs exactly once after graph-level transforms.

A custom high-level dialect is not planned initially. It may be reconsidered
only if standard IR decomposition causes a measured compile-time or
optimization problem that cannot be addressed by a canonical C++ graph model.

### 4.3 Generic runtime and artifact

Replace the fixed X/W/Y metadata and three rank-4 f32 descriptors with:

- an ordered kernel argument table keyed by external tensor UID;
- element type, rank, dimensions, strides, byte range, alignment, and access
  mode for every external tensor;
- generated adapters for arbitrary graph signatures and workspace views;
- overlap checks based on read/write access rather than Conv-specific names;
- `.dfo` format version 2 containing a generic tensor/argument table and
  per-variant symbols.

The recommended compatibility rule is to keep reading existing format-v1
Conv2D artifacts while writing format v2 for newly compiled graphs. Unknown
artifact versions remain hard errors.

### 4.4 Cost model ownership

The cost model stays in the schedule and target-specialization layer. It does
not decide whether a serialized operation is accepted. Each operation family
provides legal schedule candidates and features; the cost model selects among
those candidates after semantic lowering and before final vector/LLVM codegen.
Scalar correctness remains available when no optimized candidate is eligible.

## 5. Staged Delivery

### C0. Generic foundation

**Status:** completed and validated on 2026-07-31.

**Work:** introduce generic tensors, nodes, multi-node DAG validation,
capability registry, generic compile metadata, runtime argument tables, and
`.dfo` v2. Migrate the existing Conv path onto these contracts without changing
its public behavior.

**Exit gate:** all current tests pass unchanged; format-v1 artifacts still
load; a multi-node graph can be parsed and rejected at a precise unsupported
node; runtime UID, byte-range, alignment, and overlap checks are table-driven.

### C1. Complete protocol recognition

**Status:** completed and validated on 2026-07-31.

**Work:** parse and structurally validate all 39 tags, all 50 non-sentinel
pointwise modes, all 9 reduction modes, every node port, and every serialized
attribute in v1.24.0. Add a generated or reviewed schema inventory and the
public capability matrix.

**Exit gate:** JSON and UBJSON produce equivalent canonical graphs for every
tag; malformed attributes fail deterministically; no known tag is reported as
unknown; unlowered tags report recognized-but-not-executable.

### C2. Foundational executable operations

**Work:** implement static-shape f32 execution for `RESHAPE`, `TRANSPOSE`,
`SLICE`, `CONCATENATE`, all applicable `POINTWISE` modes, all applicable
`REDUCTION` modes, `MATMUL`, and `RESAMPLE`. Implement NumPy-style broadcasting
where the Frontend contract permits it.

**Exit gate:** single-node and fused multi-node graphs pass scalar reference,
artifact round-trip, workspace, runtime validation, ASan, and UBSan tests.

### C3. Convolution and training families

**Work:** generalize `CONV_FPROP`, then add `CONV_DGRAD` and `CONV_WGRAD`;
add `GENSTATS`, `BN_FINALIZE`, `DBN`, `DBN_WEIGHT`, batch normalization,
instance normalization, layer normalization, RMS normalization, adaptive layer
normalization, and their serialized backward variants.

**Exit gate:** forward/backward shape inference and gradients are independently
checked; finite-difference gradient tests cover representative cases; mixed
graphs compose with C2 operations.

### C4. Sequence and attention families

**Work:** implement `ROPE`, `ROPE_BWD`, `RNG`, `SDPA`, and `SDPA_BWD` using
primitive decompositions first. Add masks, causal behavior, bias, dropout,
seed/offset handling, and sequence-length metadata as explicit substages.

**Exit gate:** deterministic RNG tests, attention reference tests, mask and
edge-shape tests, backward gradient checks, and bounded workspace tests pass.

### C5. Data types and specialized operations

**Work:** expand execution from f32 to f64, f16, bf16, integer, and boolean
where legal, then add FP8/FP4/INT4 storage and conversion semantics. Implement
block-scale quantize/dequantize, `MATMUL_FP8`, FP8/MXFP8 SDPA variants, and MoE
grouped matmul forward/backward.

The v1.24.0 enum has 20 non-sentinel data-type values. Support is declared per
operation/data-type pair; enum recognition alone is not execution support.

**Exit gate:** conversion edge cases, saturation/rounding, special floating
values, packed storage, accumulator type, and per-type numeric tolerances are
tested. Unsupported host ISA paths fall back correctly or fail explicitly.

### C6. Dynamic metadata, optimization, and release qualification

**Work:** add dynamic shapes, shape override, ragged tensors, reorder formats,
paged/cache-related composite metadata where it appears in serialized graphs,
then add fusion, threading, vector schedules, and family-specific cost models.

**Exit gate:** every in-scope capability-matrix row is validated; all sanitizer
and compatibility suites pass; scalar and optimized variants agree within
per-operation tolerances; performance baselines and user documentation are
published in English and Chinese.

## 6. Stage Execution Rules

Each stage is developed and reviewed separately:

1. update the capability matrix and normative contract before enabling a row;
2. add parser and verifier tests before lowering;
3. land scalar CPU correctness before vector or fusion work;
4. run importer-only, full MLIR, artifact, end-to-end, ASan, and UBSan suites;
5. update English and Chinese user/design documentation together;
6. commit a stage only after its exit gate passes, then begin the next stage.

Partial support is expressed as an exact combination of operation, mode,
attributes, shape class, layout, and data type. No stage uses a broad
"operation supported" label for a narrower implementation.

## 7. Test and Fixture Strategy

Normal CPU CI uses checked-in JSON fixtures and derives UBJSON through the
pinned nlohmann/json implementation. Fixtures cover every tag and attribute
branch, including malformed and unsupported combinations. Reference execution
uses independent scalar implementations, normally with higher-precision
accumulation for floating-point comparisons.

Numerical tolerances are operation- and data-type-specific. Reductions and
attention use scale-aware bounds; exact integer/boolean operations require
exact equality; NaN, infinity, signed zero, saturation, and RNG reproducibility
receive explicit tests.

An optional GPU job may generate producer fixtures with cuDNN Frontend v1.24.0
and compare supported graphs against real cuDNN. That job is a compatibility
and release-validation tool only. CUDA/cuDNN remains absent from CPU build,
test, install, and runtime dependencies.

## 8. Principal Risks

- **Scope ambiguity:** Graph serializer tags, composite Graph APIs, and Backend
  descriptors are different sets. Section 10 must be resolved first.
- **Schema asymmetry:** v1.24.0 serializes more tags than its public
  `Graph::deserialize` switch reconstructs. Fixtures and attribute inventories
  must be reviewed directly against node serializers.
- **Combinatorial coverage:** modes, attributes, layouts, and data types grow
  faster than the tag count. The capability registry and generated test matrix
  are required to keep claims auditable.
- **Low-precision CPU semantics:** FP8/FP4/INT4 may need software conversion and
  packed storage before optimization is possible.
- **Attention expansion:** primitive decomposition can increase compile time
  and workspace. C4 records baselines before considering a higher-level IR.
- **Artifact migration:** generic signatures require format v2; v1 compatibility
  must be tested rather than inferred.

## 9. Immediate Next Work After Approval

The first implementation increment is C0 only:

1. add the capability registry with current `CONV_FPROP` marked validated;
2. replace fixed-rank tensor metadata with checked variable-rank metadata;
3. introduce a generic node container while retaining Conv attributes;
4. add a generic external argument table and migrate runtime validation;
5. specify and test `.dfo` v2 while keeping a v1 reader;
6. rerun all existing correctness, artifact, sanitizer, and CI gates.

No new operation lowering begins until this compatibility-preserving migration
is complete.

## 10. Confirmed Project Decisions

The project owner accepted these defaults on 2026-07-31:

1. **Scope:** target all 39 v1.24.0 serialized tags and composite Graph APIs,
   but not unrelated lower-level Backend descriptors.
2. **Type order:** make all operations correct for static f32 first; expand the
   legal data-type matrix in C5.
3. **Shape order:** support arbitrary-rank static shapes first; defer dynamic,
   ragged, reorder, and paged metadata to C6.
4. **Artifact compatibility:** allow `.dfo` v2, retain a v1 reader, and write v2
   for new compilations.
5. **External validation:** keep CPU CI self-contained and eventually use an
   optional GPU runner for producer/differential release checks.
