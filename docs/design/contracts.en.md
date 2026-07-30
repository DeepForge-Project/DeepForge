# MVP Compatibility and Runtime Contract

[中文](contracts.md)

This document defines the normative boundary of the DeepForge MVP. If another
design document conflicts with this one, this document takes precedence.

## 1. Version Policy

| Component | Pinned version | Policy |
|---|---|---|
| LLVM/MLIR | `llvmorg-22.1.8` | CMake requires an exact match |
| cuDNN Frontend | `v1.24.0` | Importer schema and fixtures are fixed against serializer source at this tag; other producer versions are rejected by the MVP |
| nlohmann/json | `3.11.3` | Matches the Frontend-vendored header; both JSON and UBJSON use this parser |
| CUDA/cuDNN backend | None | The CPU-only MVP does not build, link, or run a GPU backend |
| cuDNN Graph JSON | `json_version == "1.0"` | A mismatch is rejected with a version diagnostic |
| Platform | Linux x86-64 | The only MVP execution platform |

Dependency upgrades require a separate change and regeneration of serialized
fixtures, IR golden tests, and end-to-end numeric results. Unknown Graph schemas
must never be accepted silently.

## 2. cuDNN Frontend Input Contract

### 2.1 Accepted Carriers

`deepforge-compile` accepts the two native carriers produced by the open source
cuDNN Frontend:

1. Graph JSON produced by `Graph::serialize(nlohmann::json&)`.
2. A UBJSON blob produced by `Graph::serialize(std::vector<uint8_t>&)`.

In pinned `v1.24.0` source, the vector overload first performs Graph JSON
serialization, appends plan and runtime metadata, and encodes the complete JSON
object as **one UBJSON document** through `nlohmann::json::to_ubjson`. There is
no extra envelope and no GPU-plan byte stream after the UBJSON document;
`cudnn_backend_data` is a JSON value inside the document. The importer parses
the entire input in strict mode and compares it with the canonical bytes
produced by the same parser's default `to_ubjson`. It rejects truncation,
invalid tokens, trailing no-op markers, any other trailing bytes, and
non-default UBJSON encodings.

Both carriers require `json_version == "1.0"` and numeric
`cudnn_frontend_version == 12400`. `cudnn_backend_version` only records the
producer environment and does not affect CPU semantics. UBJSON may contain
additional fields:

| Field | MVP handling |
|---|---|
| `cudnn_backend_data`, `behavior_notes` | Accept and ignore; never interpret as a CPU execution plan |
| `variant_pack_uids` | If present, it must equal the set of non-virtual X/W/Y UIDs for the single Conv |
| `pass_by_values`, `workspace_modifications`, `variant_pack_replacements` | Must be absent or empty; an empty object or array is accepted for `variant_pack_replacements`, while a nonempty value represents unsupported execution semantics |
| `fe_workspace_size` | If present, must be zero; DeepForge replans CPU workspace |
| `tensors_to_dump` | Debug metadata; accept and ignore |

The importer uses a structured JSON/UBJSON parser. It does not parse with
string rules and does not define a private DeepForge JSON format. Input format
may be selected with `--input-format=json|ubjson|auto`. `auto` first tries
strict UTF-8 JSON and then strict UBJSON; it detects only the carrier and never
relaxes schema validation. One serialized Graph is limited to 16 MiB.
`parse_file` enforces the same limit while reading to avoid unbounded allocation
for oversized files.

### 2.2 MVP Serialized Schema

The pinned serializer subset requires an object root with these fields:

| Level | Required form | MVP constraint |
|---|---|---|
| root | `json_version`, `cudnn_frontend_version`, `graph_uid`, `context`, `nodes`, `tensors` | Versions are string `"1.0"` and integer `12400`; `nodes` has exactly one element |
| node | `tag`, `inputs`, `outputs`, `compute_data_type`, `pre_padding`, `post_padding`, `stride`, `dilation`, `math_mode` | `tag == "CONV_FPROP"`; ports are exactly `X/W` and `Y`; data type is `FLOAT`; spatial arrays have length two |
| tensor map | Object whose keys are decimal strings for explicit UIDs | Key, `uid`, and node reference agree; the MVP keeps exactly three non-virtual X/W/Y tensors |
| tensor | `name`, `data_type`, `dim`, `stride`, `is_virtual`, `is_pass_by_value`, `reordering_type`, `uid`, `uid_assigned` | `FLOAT`, rank four, `uid_assigned == true`, `is_virtual == false`, `is_pass_by_value == false`, and reordering `NONE` |

A node reference must be a JSON or UBJSON integer UID. A fallback reference by
tensor name is outside the MVP. JSON and UBJSON differ only as carriers and
must produce the same canonical model after parsing. Root execution and debug
metadata follows the rules in section 2.1 and cannot change that model's
semantics. The importer performs all validation in a temporary object; failure
must not partially modify the caller-provided output model.

### 2.3 Protocol Integration Versus Operation Coverage

Complete protocol integration means accurately reading the pinned cuDNN
Frontend serialization protocol, including tensor UIDs, dimensions, strides,
data types, `is_virtual`, node input and output port references, and attributes.
In this version, `set_output(true)` serializes as `is_virtual == false`; there
is no separate output field. Y is identified by `CONV_FPROP.outputs.Y`.
Complete integration does not mean that the MVP implements every cuDNN
operation.

The parser must traverse every valid node and return a structured diagnostic
for an unsupported node, for example:

```text
DFE_UNSUPPORTED_NODE: node[2] tag=POINTWISE is not supported by MVP
```

It must not skip an unknown node and continue compilation.

### 2.4 UID Constraints

Input, weight, and output tensors require explicit, unique, stable UIDs. The
runtime variant pack binds addresses by UID. A graph that has only tensor names
or depends on UID auto-assignment during deserialization is not reproducible
and is rejected by the importer.

## 3. Conv2D MVP Support Matrix

| Item | MVP support |
|---|---|
| Graph | Exactly one `CONV_FPROP` node |
| Tensor rank | Static rank-four X, W, and Y |
| Data type | f32 X, W, Y, and compute |
| Conv mode | Cross-correlation |
| X logical dimensions | cuDNN `[N, C, H, W]` |
| W logical dimensions | cuDNN `[K, C, R, S]` |
| Y logical dimensions | cuDNN `[N, K, P, Q]` |
| Physical layout | Packed NHWC input/output and packed KRSC filter |
| Spatial stride | `[1, 1]` |
| Dilation | `[1, 1]` |
| Padding | Static, nonnegative, and optionally asymmetric pre/post |
| Grouping | Grouped and depthwise convolution are unsupported |
| Fusion | Bias, activation, and other fused nodes are unsupported |
| Shapes | Every dimension is positive; C and K have no divisibility requirement |

cuDNN combines logical NCHW/KCRS dimensions with strides to describe physical
NHWC/KRSC buffers. The importer validates exact packed strides and converts to
MLIR physical dimension order:

| Tensor | cuDNN dimensions | cuDNN packed stride | MLIR tensor shape |
|---|---|---|---|
| X | `[N,C,H,W]` | `[H*W*C,1,W*C,C]` | `[N,H,W,C]` |
| W | `[K,C,R,S]` | `[R*S*C,1,S*C,C]` | `[K,R,S,C]` |
| Y | `[N,K,P,Q]` | `[P*Q*K,1,Q*K,K]` | `[N,P,Q,K]` |

W therefore maps to Linalg `FHWC`, where F is output feature K:
`[F,H,W,C] == [K,R,S,C]`. `[R,S,C,K]` is `HWCF` and cannot be passed to
`linalg.conv_2d_nhwc_fhwc`.

Output spatial dimensions must satisfy:

```text
P = H + pre_h + post_h - R + 1
Q = W + pre_w + post_w - S + 1
```

Compilation fails if serialized Y dimensions disagree with these values.

Every dimension product, packed stride, element byte count, and workspace
offset uses checked arithmetic. Any input outside the representable range of
`int64_t` or `size_t` returns `DFE_DIMENSION_OVERFLOW`; it must not silently
wrap in a later MLIR index or pointer computation.

## 4. Public Execution Interface

The MVP exposes only the call shape of the cuDNN Frontend `v1.24.0` Graph
execute operation. It does not provide a second handle-free API, descriptor,
or raw kernel ABI. The CPU-only form is:

```cpp
namespace deepforge::runtime {

using FrontendHandle = void *;
using VariantPack = std::unordered_map<int64_t, void *>;

class Executable {
public:
  int64_t get_workspace_size() const;
  import::Status execute(FrontendHandle handle,
                         VariantPack &uid_to_host_ptr,
                         void *workspace) const;
};

} // namespace deepforge::runtime
```

The contract is:

- Argument order and map type match the Frontend UID-map execute overload. The
  map remains a non-const reference for call compatibility, but DeepForge does
  not modify its keys or addresses.
- `handle` exists only for call-shape compatibility and may be null. The CPU
  runtime never dereferences, queries, or forwards it. The CPU-only form uses
  opaque `void *` and requires no CUDA Toolkit or cuDNN backend.
- The variant pack contains the serialized X, W, and Y UIDs. Extra UIDs are
  ignored.
- Pointers are CPU-addressable host pointers, not CUDA device pointers.
- X/W/Y are aligned to at least `alignof(float)`. Generated code must not
  unconditionally claim 64-byte alignment.
- The caller supplies workspace aligned to 64 bytes and allocates the byte
  count returned by `get_workspace_size()`. Workspace may be null when that
  count is zero.
- Valid X, W, Y, and workspace byte ranges do not overlap. The runtime can
  validate UIDs, null pointers, alignment, integer overflow, and computable
  range overlap. The C++ interface carries no allocation lengths and therefore
  cannot portably prove that a pointer refers to a sufficiently large object;
  the caller remains responsible for capacity. Only after these checks may an
  internal kernel rely on `noalias`.
- The caller allocates Y; DeepForge does not replace its address in the variant
  pack.
- Executable compile metadata is immutable. Concurrent calls are safe with
  distinct workspaces; one workspace cannot be shared by concurrent
  executions.
- `DeepForge/Runtime/Executable.h` can be included without MLIR, CUDA, or cuDNN
  include paths. MLIR ExecutionEngine, ORC, and compiler metadata appear only
  in internal factory headers.

CPU-only `deepforge::import::Status` provides `is_good/is_bad`, stable error
codes, and a message. It preserves the Frontend-shaped error-checking style but
is not binary-compatible with `cudnn_frontend::error_t`. Runtime failures map
to stable semantics such as `DFE_INVALID_VARIANT_PACK`,
`DFE_UNSUPPORTED_CPU_FEATURE`, and `DFE_GRAPH_EXECUTION_FAILED`. Private
generated-kernel arguments and memref descriptors exist only inside the
runtime. Both raw object functions and `_mlir_ciface_<symbol>` loader wrappers
are ELF `GLOBAL HIDDEN`.

This guarantees the same C++ call shape. It does not promise binary
interchangeability between a DeepForge `Executable` and an NVIDIA `Graph`;
`std::unordered_map`, `std::string`, and compiler ABI remain subject to the
pinned host toolchain in the README.

## 5. Workspace and Buffer Ownership

Every intermediate buffer comes from caller-provided workspace. MVP kernels
must not call `malloc` or `free`:

1. Run One-Shot Bufferize exactly once after Tensor/Linalg transforms.
2. Materialize `memref.alloc` and any remaining
   `bufferization.alloc_tensor` as static memref allocations. This is
   post-bufferization lowering, not a second bufferization.
3. `deepforge-workspace-plan` assigns aligned offsets to static temporaries.
4. Rewrite temporary `memref.alloc` operations as workspace
   `memref.view`/subview operations.
5. No owned allocation remains in final IR, so the kernel needs no deallocation.

The explicitly padded tensor is the primary MVP workspace consumer. A future
boundary-condition implementation or pad/conv fusion may remove it without
changing the public interface.

## 6. Numeric Semantics

The MVP uses f32 inputs, f32 weights, and f32 accumulation. Within the
reduction for one output element, it permits:

- FMA;
- addition reassociation across SIMD lanes;
- R/S/C loop unrolling or tiling.

It does not permit unrelated fast-math assumptions such as `nnan`, `ninf`,
`nsz`, or approximate reciprocal. The default validation rule is:

```text
abs(actual - reference) <= 1e-4 + 1e-3 * abs(reference)
```

The reference uses the same f32 inputs and accumulates in f64. Tests may
override tolerance through explicit parameters, but every relaxation must be
visible in the test report. Dedicated NaN and Inf tests compare classification
instead of applying the finite-value tolerance above.

## 7. CPU Feature Dispatch and Tails

Compiled output contains three internal variants:

| Variant | Vector width | Preconditions |
|---|---:|---|
| scalar | 1 | x86-64 baseline |
| AVX2 | 8 x f32 | AVX2 + FMA |
| AVX-512 | 16 x f32 | AVX-512F + FMA |

The runtime uses CPUID to select the highest supported variant. Each SIMD
variant uses a scalar cleanup loop for `C % VF`, so neither C nor K must be
divisible by a vector width. A function containing a higher ISA must never run
on an unsupported CPU.

This dispatch is a capability and safety decision, not a schedule performance
cost model.

## 8. Failure Strategy

The following are compile-time errors with no silent fallback: schema or
version mismatch, dynamic shapes, non-packed strides, non-f32 data, unknown
nodes, inconsistent output shapes, non-unit stride or dilation, and dimension
or byte-count overflow.

Insufficient CPU features are not an error; the runtime falls back to a lower
variant. AVX selection must check both CPUID and operating-system support for
XMM/YMM/ZMM state, for example through `OSXSAVE` and `XGETBV`. It cannot rely
only on hardware feature bits. Numeric tolerance failures, missing UIDs, null
pointers, and overlapping buffers are execution or test errors with actionable
error codes. Workspace pointer and size follow the
`get_workspace_size()` query contract. The Frontend-shaped MVP execute call has
no additional size parameter.

### 8.1 Stable Importer Diagnostics

P1 importer `Status::code()` and message prefixes use these stable identifiers:

| Diagnostic | Meaning |
|---|---|
| `DFE_INVALID_ARGUMENT`, `DFE_IO_ERROR` | Empty input, invalid arguments, or file-read failure |
| `DFE_PARSE_ERROR` | JSON/UBJSON syntax, truncation, trailing bytes, or canonical UBJSON failure |
| `DFE_SCHEMA_VERSION_MISMATCH`, `DFE_FRONTEND_VERSION_MISMATCH` | Graph schema or Frontend producer version mismatch |
| `DFE_MISSING_FIELD`, `DFE_INVALID_FIELD_TYPE`, `DFE_INVALID_VALUE` | Missing fixed-schema field, wrong type, or invalid value |
| `DFE_UNSUPPORTED_NODE`, `DFE_UNSUPPORTED_DATA_TYPE`, `DFE_UNSUPPORTED_EXECUTION_METADATA` | Valid input protocol with semantics outside the MVP |
| `DFE_DUPLICATE_UID`, `DFE_MISSING_UID` | Non-unique or missing UID, name fallback, or unresolved reference |
| `DFE_INVALID_LAYOUT`, `DFE_INVALID_SHAPE` | Invalid packed layout, rank, static dimension, or Conv output shape |
| `DFE_DIMENSION_OVERFLOW` | Integer, dimension product, packed stride, or byte count is out of range |

P2 reuses these stable codes. Physical layout, padding, and shape failures in
the canonical model return `DFE_INVALID_LAYOUT`, `DFE_INVALID_SHAPE`, or
`DFE_DIMENSION_OVERFLOW`. A standard MLIR verifier failure or invalid generated
module returns `DFE_INVALID_VALUE` with a message beginning
`DFE_INVALID_VALUE: mlir:`. P2 introduces no new dialect or custom operation;
UID, shape, and padding data travels in separate `Conv2DCompileMetadata`.

Runtime and artifact loading add fixed numeric assignments that must not be
reordered:

| Value | Diagnostic | Meaning |
|---:|---|---|
| 17 | `DFE_INVALID_VARIANT_PACK` | UID, null, alignment, computable alias, or workspace precondition failed |
| 18 | `DFE_UNSUPPORTED_CPU_FEATURE` | A forced object variant is unsupported by current CPUID or OS XSTATE |
| 19 | `DFE_GRAPH_EXECUTION_FAILED` | JIT or internal entry invocation failed |

See [DFO Artifact Format](../artifact-format.en.md) for `.dfo` layout, version
validation, ORC loading, and the native-code trust boundary. The FNV checksum
detects accidental corruption only; it is neither a signature nor an execution
sandbox. The runtime loads artifacts only from trusted sources.

## 9. Compatibility Decisions

- The producer is pinned to cuDNN Frontend `v1.24.0`; the MVP rejects other
  producer versions.
- Both Graph JSON and single-document UBJSON are accepted. GPU plan metadata is
  neither retained nor re-exported.
- Public execute is fixed to the handle, UID map, and workspace call shape. No
  handle-free overload is published, and the CPU runtime always ignores the
  handle.
- The public ABI is a Frontend-shaped CPU-only form: opaque `void *` handle,
  UID variant pack, workspace, and DeepForge status. CUDA/cuDNN headers are not
  required.
- Exact official `cudnnHandle_t` and `cudnn_frontend::error_t` types, GPU
  execution, and Frontend samples are outside the MVP. Introducing them later
  requires a separate ABI and dependency review.
