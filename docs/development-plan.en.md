# DeepForge Development Plan

[中文](development-plan.md)

This is the completed historical P0-P6 MVP plan. Current post-MVP work follows
the bilingual [C0-C6 operation coverage plan](cudnn-graph-coverage-plan.en.md);
the historical Conv constraints below remain the regression baseline rather
than the complete current capability claim.

## 1. Goal and Baseline

This plan turns the design documents into an executable development sequence
for the MVP without expanding its support scope. The
[MVP contract](design/contracts.en.md) is normative. This document defines what
to implement first, how to accept each stage, and when work may advance; it
does not replace the layer-specific designs.

Current baseline:

- P0-P6 are complete. The repository contains a canonical Graph model, strict
  JSON/UBJSON importer, standard Tensor/Linalg module builder, exactly one
  One-Shot Bufferize run, static workspace planning, three x86-64 objects, a
  Frontend-shaped runtime, reloadable `.dfo` artifacts, CLI tools, benchmarks,
  CTest, and sanitizer/CI configuration.
- LLVM/MLIR is pinned to `llvmorg-22.1.8`; neither `main` nor an LLVM 23 release
  candidate is supported.
- cuDNN Frontend `v1.24.0` is the serializer schema and fixture reference, with
  `json_version == "1.0"`. Other producer versions are rejected unless a
  separate compatibility review expands the boundary.
- The MVP supports Linux x86-64 and exactly one static, packed, f32
  `CONV_FPROP` with unit stride and dilation. Grouped/depthwise convolution,
  fusion, dynamic shapes, and physical NCHW layout are unsupported.
- Machine Dialect, AMX, bf16, multithreading, and a GPU backend are deferred.
- The public interface is a Frontend-shaped CPU-only ABI: opaque handle, UID
  variant pack, workspace, and DeepForge status. CUDA/cuDNN backend types and
  raw generated-kernel or memref-descriptor signatures are not public.
- The P1 importer is an independent CPU-only target that can be configured and
  tested with `DEEPFORGE_ENABLE_MLIR=OFF`. Compiler targets from P2 onward
  require MLIR.

## 2. Development Principles

1. **Complete the scalar vertical path first:** Graph importer to MLIR to
   scalar object to runtime execution precedes AVX optimization.
2. **Every stage has executable exit criteria:** a stage must pass its verifier,
   negative tests, and golden outputs before the next stage begins.
3. **Separate protocol compatibility from operation coverage:** parse official
   JSON/UBJSON, diagnose unsupported nodes, never skip nodes, and do not define
   a private `cudnn.*` MLIR operation.
4. **Bufferize once:** run One-Shot Bufferize only after Tensor/Linalg
   structural transforms; afterward perform only bufferization-dialect
   legalization and workspace planning.
5. **Plan resources before target specialization:** freeze workspace before
   ISA-specific vectorization. A vectorizer cannot change allocation count,
   size, or lifetime.
6. **Prefer standard dialects:** use Tensor, Linalg, MemRef, Affine/SCF, Vector,
   and LLVM rather than duplicating register allocation or cache address spaces.
7. **Correctness precedes performance:** scalar is the baseline. Every variant
   uses the same f64-accumulating reference and reports contract tolerances.

## 3. Work Packages and Dependencies

| ID | Work package | Main directories | Prerequisite | Deliverable |
|---|---|---|---|---|
| P0 | Toolchain and dependency pinning | `CMakeLists.txt`, `cmake/` | None | Reproducible MLIR 22.1.8 environment and version manifest |
| P1 | cuDNN serialization importer | `lib/Import`, `include/DeepForge/Import`, `test/Import` | P0 | Canonical Graph model and JSON/UBJSON parser |
| P2 | Tensor/Linalg IR import | `lib/MLIR`, `include/DeepForge/Compiler`, `test/MLIR` | P1 | Standard MLIR module, compile metadata, layout/padding verifier |
| P3 | Bufferization and workspace | `lib/Transforms`, `lib/Runtime`, `test/Transforms` | P2 | One bufferization, static workspace plan, internal adapter |
| P4 | Scalar lowering and runtime | `lib/Compiler`, `lib/Runtime`, `tools` | P3 | Scalar LLVM/object and UID-based execute |
| P5 | AVX2 and AVX-512 variants | `lib/Transforms`, `lib/Compiler`, `lib/Runtime`, `test/E2E` | P4 | C-reduction SIMD and CPUID/OS-state dispatch |
| P6 | CLI, packaging, and quality gates | `tools`, `test`, `docs` | P4; may finish alongside P5 | `.dfo` artifact, CI, benchmark, and release checks |

Critical dependency chain:

```text
P0 -> P1 -> P2 -> P3 -> P4 -> P5 -> P6
             |          |      |
             +-- IR tests +-- runtime tests
```

P1 parser tests and P0 toolchain installation may be prepared in parallel, but
all MLIR lowering work must target the pinned P0 version.

## 4. Implementation Stages

### P0. Pin Toolchain and Dependencies

**Goal:** make development, tests, and CI use one LLVM/MLIR toolchain instead
of tracking rolling revisions during implementation.

**Work:**

- build and install `llvmorg-22.1.8` with MLIR and the X86 target by following
  the README;
- require `LLVM_PACKAGE_VERSION == 22.1.8` during configuration;
- pin Ninja, a C++20 compiler, nlohmann/json `3.11.3`, and cuDNN Frontend
  `v1.24.0` serializer/fixture source without building a CUDA/cuDNN backend;
- record versions of `mlir-opt`, `mlir-translate`, and `llc` plus X86 target
  availability;
- parse, verify, translate, and compile a minimal module as a smoke test;
- use the 22.1.8 registry and headers as the source of truth for pass names and
  C++ APIs.

**Deliverables:**

- a caller-selected local install prefix discovered through `CMAKE_PREFIX_PATH`;
- successful CMake configuration logs and a version manifest;
- the [external toolchain table](../README.md#external-toolchain-versions),
  recording LLVM tag/commit, C++ compiler, parser, and fixture versions.

**Exit criteria:**

```text
mlir-opt --version       -> 22.1.8
mlir-translate --version -> 22.1.8
llc --version            -> includes X86 backend
cmake -S . -B build ...  -> configure succeeds
```

Before P0 completes, a CMake failure caused by missing `MLIRConfig.cmake` is
expected and lowering development does not begin.

**Current result:** P0 is complete. LLVM/MLIR 22.1.8 is installed; MLIR
parse/verify, `mlir-translate --mlir-to-llvmir`, X86 `llc`, and DeepForge CMake
configuration smoke tests passed. Actual external tool versions are recorded
in the English README table linked above.

### P1. cuDNN Frontend Serialization Importer

**Goal:** convert official Graph JSON/UBJSON to a carrier-independent canonical
model and validate every MVP boundary once.

**Work:**

- implement structured JSON/UBJSON reading; parse the real `v1.24.0` vector
  serialization as one strict UBJSON document and reject truncation or trailing
  bytes;
- validate `json_version == "1.0"`,
  `cudnn_frontend_version == 12400`, required fields, and integer ranges;
- accept and ignore embedded `cudnn_backend_data` and `behavior_notes`, validate
  `variant_pack_uids`, and reject nonempty `pass_by_values`,
  `workspace_modifications`, `variant_pack_replacements`, or nonzero
  `fe_workspace_size`;
- parse context, tensors, nodes, UIDs, dimensions, strides, data types,
  `is_virtual`, node port references, and Conv attributes;
- assign stable `DFE_UNSUPPORTED_*` diagnostics to unknown nodes and never skip
  an unrecognized node;
- define MLIR-independent `SerializedGraph`, `TensorDesc`, and `ConvFpropDesc`
  models for isolated tests;
- validate exact packed strides, f32, rank four, positive static dimensions,
  and output shape for X/W/Y;
- require JSON and UBJSON to decode to field-for-field equivalent canonical
  models independent of object key order or carrier encoding;
- do not link the CUDA/cuDNN execution path or reconstruct a GPU plan.

**Tests:**

- check in a minimal JSON fixture validated against the pinned serializer
  shape, and use vendored nlohmann/json 3.11.3 to encode it as UBJSON;
- reserve a producer-generated cross-check fixture for a separate CUDA/cuDNN
  environment without making it a CPU build, CI, or runtime prerequisite;
- test JSON/UBJSON canonical-model equivalence;
- cover missing or duplicate UIDs, truncated or trailing UBJSON, invalid schema,
  unknown nodes, non-f32 data, dynamic shapes, non-packed strides, non-unit
  stride/dilation, bad output shapes, and multiplication overflow;
- include asymmetric padding and C/K values not divisible by a vector width.

**Exit criteria:** with `DEEPFORGE_ENABLE_MLIR=OFF`, P1 configures and tests
without MLIR. Every unsupported input fails in the importer rather than later
in lowering.

### P2. Tensor/Linalg IR Import and Verification

**Goal:** convert the canonical model into destination-passing IR accepted by
upstream MLIR verification and One-Shot Bufferize.

**Work:**

- map cuDNN logical dimensions and strides to X `[N,H,W,C]`, W `[K,R,S,C]`,
  and Y `[N,P,Q,K]`;
- emit standard `tensor.pad` for nonzero padding, followed by `linalg.fill` and
  `linalg.conv_2d_nhwc_fhwc`; pass X directly when padding is zero and define no
  `cudnn.conv_fwd` operation;
- pass Y as a function-boundary destination and fill it with zero to avoid
  uninitialized output and ambiguous returned-buffer ownership;
- preserve static shape, identity layout, and f32 element type;
- independently verify indexing maps, padding, and P/Q;
- retain the named Conv and disable generalization until a generic-op
  indexing-map verifier and lowering pattern exist;
- emit import-stage IR and compile metadata.

**Tests:** standard parser/verifier golden tests; correct `[K,R,S,C]` versus
invalid `[R,S,C,K]`; symmetric and asymmetric padding; C/K tails; minimum and
boundary sizes; absence of custom cuDNN operations, dynamic dimensions, or
implicit element conversion.

**Exit criteria:** every positive case round-trips through the MLIR 22.1.8
parser/verifier and directly enters One-Shot Bufferize; every negative case has
a stable error code.

**Current result:** P2 is complete. `DeepForge::MLIRImport::import_conv2d`
builds a single-function static-f32 module containing optional `tensor.pad`,
`linalg.fill`, and `linalg.conv_2d_nhwc_fhwc`.
`verify_conv2d_module` independently checks physical shapes, padding,
destination semantics, and unit stride/dilation. UID, physical shape, padding,
stride/dilation, and function name are returned in
`Conv2DCompileMetadata`, not custom MLIR operations or attributes. Tests cover
goldens, parser/verifier round trips, padding, minimum dimensions, C/K tails,
invalid layout or shape, dynamic dimensions, HWCF rejection, and a
One-Shot Bufferize smoke test.

### P3. One-Shot Bufferize, Allocation Materialization, and Workspace

**Goal:** freeze memory ownership and the internal adapter so kernels never
call `malloc` or `free`.

**Work:**

- invoke One-Shot Bufferize once after Tensor/Linalg transforms;
- set identity function-boundary layout and `allow-unknown-ops=false`;
- legalize remaining `bufferization.alloc_tensor` and padding materialization
  to static `memref.alloc`, fill/copy, and subview operations without a second
  bufferization;
- remove an equivalent Y result when needed;
- implement checked static workspace planning for allocation lifetime,
  64-byte-aligned offsets, padding, and total size;
- rewrite temporary allocations as workspace views and remove owned deallocs;
- create internal memref descriptors while keeping only opaque handle, UID map,
  and workspace pointer public;
- validate UIDs, null, alignment, computable overlap, and overflow while making
  caller-provided capacity an explicit precondition;
- freeze workspace layout before target-specific vectorization.

**Tests:** dialect legality before and after bufferization; an assertion that it
runs exactly once; planner ownership for both `alloc_tensor` and padding;
offset alignment, lifetime reuse, overflow, and no owned allocations; overlap,
null, alignment, and missing-UID failures; concurrent calls with distinct
workspaces.

**Exit criteria:** fixed input produces deterministic workspace sizes and
offsets. Final internal IR contains no Tensor or Bufferization allocation
residue and generated kernels allocate no memory.

**Current result:** P3 is complete. A single entry runs and counts One-Shot
Bufferize. A checked-size lifetime planner rewrites padding allocations as
64-byte-aligned `memref.view` operations. A verifier rejects Tensor or
Bufferization residue, owned allocation, and inconsistent plans. Runtime checks
UIDs, null, alignment, computable aliasing, and workspace before entering a
hidden kernel. Concurrent tests use distinct workspaces.

### P4. Scalar Lowering, LLVM Conversion, and Runtime

**Goal:** produce an independently verifiable baseline object before SIMD.

**Work:**

- have `deepforge-lower-direct-conv` consume only the named Conv and emit
  `n, oh, ow, k, r, s, c` loops with one scalar output accumulator;
- use standard Linalg-to-Affine/SCF conversion for remaining fill, copy, and
  padding helpers;
- implement a `VF=1` scalar schedule with explicit C tails and exact OH/OW/K
  bounds;
- completely convert Vector, SCF, Affine, Index, Arith, MemRef, Func, and CF to
  LLVM using the verified 22.1.8 order;
- translate to LLVM IR and emit a baseline X86 object;
- close the loop through `compile(serialized_graph)`, immutable executable
  metadata, and `execute(handle, uid_map, workspace)`;
- select scalar only before adding AVX feature paths.

**Tests:** LLVM legality and `llc` smoke; scalar reference tests for random,
zero, unit-weight, negative, NaN, and Inf inputs; exhaustive small indexing and
padding boundaries; writing into the caller-provided Y without replacing its
address; actionable importer and runtime statuses.

**Exit criteria:** on x86-64 baseline, a source-validated JSON fixture compiles
and executes as a scalar object within numeric tolerance, with no Tensor,
Linalg, Affine, SCF, or MemRef residue.

**Current result:** P4 is complete. Named Conv lowers directly to static scalar
SCF loops, upstream conversion removes auxiliary Linalg, and the full path
reaches LLVM Dialect, LLVM IR, and object code. Development-time execution uses
MLIR ExecutionEngine. CTest covers legality, random and boundary numerics, and
runtime failures.

### P5. AVX2, AVX-512, and Feature Dispatch

**Goal:** add SIMD variants without changing Graph, ABI, workspace, or numeric
contracts.

**Work:**

- copy three target-specific modules from the same target-independent planned
  IR;
- use `vector<8xf32>` for AVX2 and `vector<16xf32>` for AVX-512;
- load contiguous X/W values only along C, apply vector FMA, and horizontally
  reduce to one Y scalar; prohibit K vectorization without explicit packing;
- generate scalar cleanup for `C % VF` without relying on masked loads;
- apply `+avx2,+fma` and `+avx512f,+fma`, and ensure scalar has no higher ISA;
- select the highest safe variant with CPUID plus OSXSAVE/XGETBV and fall back
  to scalar;
- inspect assembly and target attributes rather than fixed intrinsic text;
- run the same f64 reference and error report for every variant.

**Tests:** C-tail matrix `1, 7, 8, 15, 16, 17, 31, 32, 33`; K/OH/OW/N values
that are not tile multiples; baseline, AVX2, AVX2+FMA, AVX-512F, and missing OS
SIMD-state capability; element-wise variant comparison and maximum errors;
disassembly proving scalar contains no AVX instructions.

**Exit criteria:** hardware capabilities select the correct variant, no CPU
executes an unsupported ISA, and all variants share numeric and workspace
contracts.

**Current result:** P5 is complete. AVX2 and AVX-512 use C reductions with
`vector<8xf32>` and `vector<16xf32>`, with scalar cleanup for `C % VF`. CPUID,
FMA, OSXSAVE, and XGETBV jointly determine dispatch. Tests cover C/K values
`1,7,8,15,16,17,31,32,33`, N=2, 29x30 spatial boundaries, NaN/Inf and extremes.
Disassembly confirms scalar has no VEX/EVEX, AVX2 has no ZMM, and AVX-512 uses
ZMM.

### P6. CLI, Artifact, CI, and Performance Baseline

**Goal:** turn the compiler from a test program into a reproducible development
artifact.

**Work:**

- complete `deepforge-compile` JSON/UBJSON input, `--input-format`, `--target`,
  `--emit`, IR dumps, and diagnostics;
- define `.dfo` metadata, UID/shape/stride, workspace, numeric contract,
  variant symbols/features, and object sections;
- freeze public status/error codes and hide internal kernel symbols;
- add CTest/end-to-end tests and LLVM 22.1.8 CI;
- record small, medium, and large correctness and performance baselines for all
  variants;
- treat padding copies, tiling, and vector width as measurable optimizations,
  without requiring cache address spaces or Machine Dialect.

**Exit criteria:** one serialized fixture validated against pinned serializer
source reproduces artifact, IR dumps, execution, and test reports. Invalid
inputs have stable diagnostics, and toolchain version drift fails at configure
time.

**Current result:** P6 is complete. `deepforge-compile` supports JSON, UBJSON,
auto detection, three-stage MLIR dumps, LLVM IR, and `.dfo`. The loader accepts
files or memory, passes three objects to ORC, and reuses runtime validation and
dispatch. Artifact version, lengths, trailing data, and checksums are strict;
JSON and canonical UBJSON produce byte-identical artifacts; internal object
symbols are `GLOBAL HIDDEN`. CI, ASan/UBSan jobs, small/medium/large benchmarks,
and the native-artifact trust boundary are present.

## 5. Test Gates and Acceptance Matrix

Every stage has at least four test categories rather than relying only on the
final end-to-end path:

| Category | Focus |
|---|---|
| Unit | Parser, UIDs, shape/stride, checked arithmetic, and error codes |
| IR golden | Import IR, bufferized IR, workspace rewrite, and final legality |
| Negative | Schema, node, data type, layout, shape, alias, feature, and ABI preconditions |
| End to end | Scalar/AVX2/AVX-512, padding and C/K/spatial boundaries, numeric tolerance |

Minimum positive matrix:

- `N=1` and `N>1`;
- `R/S=1` and `3`, including asymmetric padding;
- C/K below, equal to, and above 8 and 16;
- non-square H/W and spatial loops spanning multiple vector widths;
- zero, random finite, extreme, NaN, and Inf values;
- both official JSON and UBJSON carriers.

The reference uses the same f32 inputs and f64 accumulation. Default acceptance
is:

```text
abs(actual - reference) <= 1e-4 + 1e-3 * abs(reference)
```

A relaxed tolerance must appear as a test parameter and report field; global
rules cannot change implicitly.

## 6. Public Interface Decision

The public execute operation is frozen to the Frontend-shaped overload with
arguments in this order: handle,
`std::unordered_map<int64_t, void*>&`, workspace. There is no handle-free
overload and the CPU runtime never dereferences the handle. The MVP uses an
opaque `void *` handle and DeepForge status, remains CPU-only, and requires no
CUDA/cuDNN headers or backend.

Exact official `cudnnHandle_t` and `cudnn_frontend::error_t` types, GPU
execution, and Frontend samples are future scope and cannot become CPU MVP
build prerequisites retroactively.

## 7. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| A new developer or CI host has no MLIR | Build and install the pinned P0 tag; verify every pass API against that installation |
| Pass names or APIs differ in 22.1.8 | Use a C++ pipeline builder and legality tests, not ad hoc command concatenation |
| UBJSON is treated as private framing or trailing data is accepted | Follow `to_ubjson` and strict `from_ubjson`; test truncation and trailing bytes |
| `alloc_tensor` escapes workspace planning | Inspect every static allocation after materialization and reject unknown allocations |
| AVX features leak into scalar | Separate target modules, disassembly checks, CPUID plus XGETBV |
| Vector reduction is indexed incorrectly | C-lane invariant, scalar reference, and dedicated tail tests |
| ABI cannot prove pointer capacity | State caller capacity as a precondition; do not claim portable bounds checking |
| Padding copy is slow | Preserve a correct baseline and let benchmarks decide fusion or boundary optimization |
| Machine/AMX enters too early | Forbid related MVP directories, operations, and passes; require a separate review gate |

## 8. Definition of Done

The MVP is complete only when all of the following hold:

- LLVM/MLIR is exactly `llvmorg-22.1.8`, with reproducible builds and CI;
- JSON/UBJSON fixtures validated against Frontend `v1.24.0` serializer source
  import successfully, and unknown or unsupported inputs fail stably;
- layout, padding, UIDs, workspace, alignment, aliasing, and numeric contracts
  are documented and tested;
- One-Shot Bufferize runs once and final LLVM conversion has no illegal
  dialect residue;
- scalar, AVX2, and AVX-512 share the public contract and dispatch safely;
- C/OH/OW/K tails, NaN/Inf classification, and default tolerance have
  end-to-end results;
- artifacts and runtime expose neither memref descriptors nor raw kernel ABI;
- Machine Dialect, AMX, bf16, fusion, and dynamic shapes remain outside MVP;
- source-level shape compatibility of the CPU-only Frontend-shaped API is
  tested, and CUDA/cuDNN backend is not a build dependency.

## 9. Result and Next Steps

P0-P6 completed in this order:

1. Installed and verified LLVM/MLIR `llvmorg-22.1.8` and recorded P0 tools.
2. Fixed the Conv2D JSON fixture against Frontend `v1.24.0` serializer source
   and generated equivalent UBJSON with the vendored parser. An official
   producer cross-check remains a release gate, not a CPU-only development
   blocker.
3. Implemented the P1 canonical model, strict JSON/UBJSON importer, and 59 test
   checks.
4. Implemented P2 destination-passing MLIR import, verification, and metadata.
5. Implemented P3 One-Shot Bufferize, allocation materialization, and workspace
   planning.
6. Completed P4 scalar, P5 SIMD dispatch, and P6 CLI, artifact, and quality
   gates.
7. Completed post-MVP C0-C3: generic graph execution, complete 39-tag schema
   recognition, eight foundational tags, rank-3 through rank-5 convolution
   forward/backward, and 14 normalization/statistics tags.

The next functional stage is C4 sequence and attention execution, followed by
C5 data types and specialized operations and C6 dynamic metadata/release
qualification. Benchmark-driven optimization remains owned by the
Loop/Schedule layer: establish repeatable pinned-core measurements first, then
evaluate outer tiling, padding fusion, and multithreading independently.
Optimizations must not change the frozen serialization, ABI, workspace
ownership, numeric, or artifact contracts.
