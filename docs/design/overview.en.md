# DeepForge Architecture Overview

[中文](overview.md)

## 1. Project Scope

DeepForge compiles a serialized Graph from a pinned cuDNN Frontend version into
CPU object code. The MVP implements only a static, packed, f32 Conv2D forward
operation, while directly adopting the cuDNN Frontend Graph serialization and
UID variant-pack concepts for its input protocol and execution interface.
The current post-MVP C5 implementation executes declared static subsets for
all 39 serialized tags: three convolution, eight foundational, 14
normalization/statistics, five sequence/attention, and nine low-precision
specialized operations. Generic data remains f32, with INT32/INT64 on
documented sequence ports. The specialized path adds operation-scoped
f16/bf16/FP8/FP4/INT4 storage and conversion. It preserves the public runtime
interface and uses only upstream MemRef, SCF, Arith, Math, and LLVM dialects.

See [contracts.en.md](contracts.en.md) and the
[schema capability inventory](../cudnn-graph-schema-inventory.en.md) for the
normative support boundaries.

## 2. Design Principles

1. **Correctness before expansion:** the scalar kernel is the semantic baseline
   for every optimized path.
2. **Protocol compatibility is separate from operation coverage:** parse the
   complete relevant cuDNN schema and explicitly reject unimplemented nodes.
3. **Prefer upstream dialects:** reuse MLIR Tensor, Linalg, Affine/SCF, Vector,
   and LLVM.
4. **Bufferize once:** run One-Shot Bufferize once after Tensor/Linalg
   transforms are complete.
5. **Explicit legality boundaries:** each stage has an allowed-dialect set and
   runs a verifier.
6. **Target information does not change semantics:** the MVP fixes ISA
   features; a future cache cost model may influence only schedule and tiling
   choices and cannot model cache as addressable memory.
7. **Runtime isolates the internal ABI:** the public surface is a
   Frontend-shaped handle, UID variant-pack, and workspace. Internal kernel
   signatures may evolve.
8. **Recompute analyses on demand:** structural transforms do not retain stale
   dependence or alias results.

## 3. Components and IR Boundaries

```text
+----------------------------------------------------------------+
| cudnn-frontend v1.24.0 serialized Graph                         |
| JSON graph object or single-document UBJSON graph + metadata    |
+-------------------------------+--------------------------------+
                                | structured parse + validation
                                v
+----------------------------------------------------------------+
| DeepForge Importer                                              |
| logical cuDNN dims/strides -> physical MLIR tensor types         |
| CONV_FPROP -> optional tensor.pad + fill + linalg.conv_2d_*      |
+-------------------------------+--------------------------------+
                                | no cudnn.* op leaves importer
                                v
+----------------------------------------------------------------+
| Tensor + Linalg Dialects                                        |
| destination-passing tensor IR, named Conv kept for MVP          |
+-------------------------------+--------------------------------+
                                | one-shot-bufferize exactly once
                                v
+----------------------------------------------------------------+
| MemRef + Affine/SCF + Vector                                    |
| direct-conv loop schedule, C-reduction SIMD, workspace views    |
+-------------------------------+--------------------------------+
                                | complete dialect conversions
                                v
+----------------------------------------------------------------+
| LLVM Dialect -> LLVM IR -> x86-64 object variants               |
+-------------------------------+--------------------------------+
                                | hidden internal entry points
                                v
+----------------------------------------------------------------+
| Runtime Executable                                              |
| UID binding, pointer validation, workspace, CPUID dispatch      |
+----------------------------------------------------------------+
```

The diagram shows the original optimized Conv path. After canonical import, a
generic C2-C5 graph takes a parallel standard-MLIR path directly through
static MemRef/SCF/Arith/Math IR, planned virtual-tensor workspace views, and
the same LLVM object pipeline. This path includes grouped convolution,
convolution gradients, normalization, sequence transforms, and attention
forward/backward plus C5 software low-precision conversion, block scaling,
FP8 matmul, and MoE. It does not run the MVP Conv Tensor/Linalg bufferization
or direct-conv schedule.

### 3.1 Importer

The importer is the boundary from files and the object model into MLIR; it is
not an MLIR pass. It:

- detects JSON or UBJSON;
- parses a complete document strictly with nlohmann/json 3.11.3 and rejects
  trailing UBJSON bytes;
- validates `json_version`, the Frontend version, and required fields;
- ignores in-document GPU-only plan metadata while rejecting nonempty fields
  that carry unsupported execution semantics;
- resolves tensor and node references plus stable UIDs;
- validates the applicable capability subset;
- builds standard Tensor/Linalg IR for the optimized MVP Conv or standard
  MemRef/SCF/Math IR for a generic C2-C5 graph.

There is no temporary `cudnn.conv_fwd` operation. One-Shot Bufferize therefore
never encounters an unknown custom operation without a
`BufferizableOpInterface`.

### 3.2 Compiler Pipeline

The main pipeline consists of independently testable stages:

```text
import-cudnn-graph
  -> named tensor/linalg verification
  -> one-shot-bufferize
  -> workspace planning/rewrite
  -> clone target variants + direct-conv SCF schedule
  -> affine/scf/vector/memref/func/cf/arith/index to LLVM
```

See [pass-pipeline.en.md](pass-pipeline.en.md) for the detailed pass order.

### 3.3 Runtime

The runtime neither interprets the Graph nor repeats shape inference. A
compiled result records:

- X/W/Y UIDs and static byte counts;
- workspace size and alignment;
- three internal kernel entry points;
- CPU features required by each variant;
- numeric and alias contract versions.

`execute` ignores the compatibility handle and performs UID binding, pointer
metadata and computable overlap checks, CPUID dispatch including OS SIMD state,
and the internal call. The caller remains responsible for providing buffers
with the capacities required by the contract.

## 4. Core Conv2D Representation

cuDNN logical dimensions differ from MLIR physical shapes:

```text
cuDNN X: dim[N,C,H,W], stride[HWC,1,WC,C]
MLIR X:  tensor<NxHxWxCxf32>

cuDNN W: dim[K,C,R,S], stride[RSC,1,SC,C]
MLIR W:  tensor<KxRxSxCxf32>

cuDNN Y: dim[N,K,P,Q], stride[PQK,1,QK,K]
MLIR Y:  tensor<NxPxQxKxf32>
```

DeepForge therefore uses `linalg.conv_2d_nhwc_fhwc`; the filter F dimension is
K. The main path keeps the cuDNN KRSC packed buffer and does not require the
caller to repack weights outside the ABI.

The MVP vectorizes the C reduction. Input and filter values for one K are
contiguous in C. SIMD FMA lanes must be combined with
`vector.reduction <add>` before writing one scalar Y value. C lanes must never
be stored as K lanes.

## 5. Resolved Design Issues

| Original issue | MVP decision |
|---|---|
| C lanes incorrectly stored as K | Horizontally reduce C-vector accumulators; cover the path in the end-to-end example |
| Multiple dialects remain after Vector lowering | Define a complete LLVM conversion order and final legality check |
| cuDNN Graph, C++, and MLIR entry points conflated | Import only official JSON/UBJSON serialization |
| Repeated or premature bufferization | Run One-Shot Bufferize once after Tensor/Linalg transforms |
| ABI, strides, padding, and alignment undefined | Define them in `contracts.en.md`; validate computable runtime preconditions |
| L1/L2/L3 modeled as address spaces | Use cache information only as input to a future schedule cost model |
| AMX lifetime and tile configuration unresolved | Remove AMX from the MVP and prefer upstream X86 dialect support later |
| Dynamic shape and type-conversion overcommitment | Accept static f32 only; do not use `tensor.cast` for element conversion |
| Analysis results survive invalidating transforms | Query analyses per transform and let them be invalidated |
| Missing tails, features, tests, and versioning | Scalar cleanup, three-way CPUID dispatch, a test matrix, and exact pins |

## 6. Target and Cost Models

The current implementation has no unused `TargetProfile` abstraction. Object
code generation directly uses three `llvm::TargetMachine` configurations:

```text
host x86-64 triple + x86-64 baseline
host x86-64 triple + avx2,fma
host x86-64 triple + avx512f,fma
```

Their vector widths are 1, 8, and 16. A `.dfo` records the exact target triple
and required features, and its loader rejects native artifacts for a different
host triple.

There is no active performance cost model in the MVP. A future cost model is
owned by the Loop/Schedule layer because it chooses tile sizes, loop order, and
unroll factors. Target configuration supplies hardware facts, and benchmarks
validate decisions. Runtime CPUID dispatch is a safety capability check, not a
cost model. Cache profiles, software prefetch, and outer tiling remain in the
benchmark-driven Optimize phase.

## 7. Ownership and Aliasing

- Every external tensor carries read, write, or read-write access metadata.
- External buffers and workspace must not overlap.
- The runtime performs computable range-overlap checks before private kernels
  receive readonly/noalias attributes. Because the interface has no allocation
  lengths, it is not a portable bounds checker.
- All temporary buffers come from workspace. A kernel owns no external address
  and never calls `free`.
- Compiler AliasAnalysis results serve only the current pass and are recomputed
  after any reorder or fusion.

## 8. Deferred Scope

The following remain outside the frozen MVP baseline: Machine Dialect, AMX/bf16,
cache address spaces, OpenMP, multi-operation fusion, dynamic shapes,
non-packed strides, physical NCHW layout, grouped or depthwise convolution, and
a GPU backend. Post-MVP C2-C5 independently added arbitrary positive strides,
grouped convolution, bf16 on specialized ports, and the capability subsets in
the schema inventory. C6 has added `F8_128x4` physical scale decoding on its
documented block/MXFP8 ports and runtime override for one exact-shape external
f32 `POINTWISE` node. Standard f32 SDPA now supports ragged forward data/row
outputs and backward data/gradients, independently paged K/V with compact page
tables, forward block masks, and forward/backward sink tokens. Broader dynamic
behavior, paged backward, other physical reorder metadata, threading, and broad
fusion remain.

Deferral does not discard a direction. Each capability may be introduced
independently after its semantics, upstream support, correctness tests, and
performance baseline are defined.

## 9. Directory Layout

```text
include/DeepForge/
  Import/          serialized Graph parser and validator
  Compiler/        pipeline and target configuration
  Runtime/         Executable and variant-pack API
lib/
  Import/
  Transforms/      direct Conv schedule and workspace planning
  Compiler/
  Runtime/
tools/
  deepforge-compile/
test/
  Import/
  Transforms/
  E2E/
```

The MVP does not create `include/DeepForge/Dialect/Machine` or corresponding
TableGen directories.
