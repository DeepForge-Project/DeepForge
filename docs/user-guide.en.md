# DeepForge User Guide

[中文](user-guide.zh-CN.md)

This guide is for users compiling and executing cuDNN Frontend serialized
Graphs on the DeepForge CPU runtime. See the
[design overview](design/overview.en.md) for implementation details.

## 1. Supported Scope

DeepForge `0.1.0` currently supports:

| Area | Supported scope |
|---|---|
| Platform | Linux x86-64 |
| Input | Graph JSON or canonical UBJSON produced by cuDNN Frontend `v1.24.0` |
| Graph schema | `json_version == "1.0"`, `cudnn_frontend_version == 12400` |
| Operations | Validated CPU subsets for all 39 serialized v1.24.0 tags; exact-shape f32 `POINTWISE` DAGs, one standard-f32 `MATMUL`, one standard-f32 LOGICAL `RESHAPE`, one standard-f32 `REDUCTION`, one standard-f32 `TRANSPOSE`, one standard-f32 `CONCATENATE`, and one dense standard-f32 `SDPA` forward support shape override arrays, while MATMUL also has separate extent-override tensor ports |
| Generic tensors | Rank 1-64 f32 data with explicit UID; allocations are otherwise static; documented metadata may be INT32/INT64; virtual intermediates are supported |
| Generic layout | Positive, non-overlapping arbitrary strides; documented standard f32 SDPA tensors may use ragged batch-prefix storage; `F8_128x4` is enabled only for the scale ports below |
| C5 specialized storage | FLOAT16, BFLOAT16, FP8 E4M3/E5M2/E8M0, packed FP4 E2M1 and INT4, plus FLOAT controls on documented ports |
| Convolution | Rank 3-5 FPROP/DGRAD/WGRAD, grouped channels, positive stride/dilation, non-negative asymmetric padding, both math modes |
| CPU code | Scalar, AVX2+FMA, and AVX-512F+FMA with runtime dispatch |
| Output | LLVM IR or a `.dfo` artifact containing three native objects |

The foundational operation subset is:

| Tag | Current constraints |
|---|---|
| `RESHAPE` | `LOGICAL` only; equal element count; one standard-f32 operation supports bounded X/Y descriptor overrides |
| `TRANSPOSE` | Complete static permutation; one standard-f32 operation supports bounded X/Y descriptor overrides |
| `SLICE` | In-range half-open bounds and positive integer strides |
| `CONCATENATE` | Numbered inputs, non-negative axis, no in-place mode; one standard-f32 operation supports bounded input/Y descriptor overrides |
| `POINTWISE` | All 50 v1.24.0 modes with trailing-dimension NumPy broadcasting |
| `REDUCTION` | All 9 modes; same input/output rank and reduced extents equal to one; one standard-f32 operation supports bounded X/Y descriptor overrides |
| `MATMUL` | Equal rank >= 2, broadcast batch dimensions, optional per-batch INT32 M/N/K overrides, and finite f32 padding value |
| `RESAMPLE` | Three pooling modes plus integer `NEAREST`; three padding modes and no index output. `BILINEAR` is rejected because v1.24.0 drops fraction denominators during serialization |

The C3 operation families are:

| Tags | Current constraints |
|---|---|
| `CONV_FPROP`, `CONV_DGRAD`, `CONV_WGRAD` | Logical `[N,C,spatial...]` tensors of rank 3-5; group count is `X.C / W.C`; output shape must match padding, stride, and dilation |
| `BATCHNORM`, `BATCHNORM_INFERENCE`, `DBN`, `DBN_WEIGHT` | Per-channel parameters/statistics; training running-stat ports are all present or all absent; `peer_stats` must be empty |
| `GENSTATS`, `BN_FINALIZE` | Per-channel sum/square sum and training-stat finalization |
| `INSTANCE_NORM`, `INSTANCE_NORM_BPROP` | Per-channel parameters and per-instance/channel saved statistics |
| `LAYER_NORM`, `LAYER_NORM_BPROP` | Normalized axes derive from the full-rank broadcast scale shape |
| `RMS_NORM`, `RMS_NORM_BPROP` | RMS statistics derive from scale shape; bias and bias gradient are optional where serialized |
| `ADA_LAYER_NORM`, `ADA_LAYER_NORM_BPROP` | Layer normalization with batch-preserving statistics and adaptive full-rank parameters |

C4 sequence and attention support is:

| Tags | Current constraints |
|---|---|
| `RNG` | Bernoulli f32 output; fixed seed or scalar INT64 `Seed`/`Offset`; deterministic DeepForge CPU stream |
| `ROPE`, `ROPE_BWD` | f32 BHSD split-half rotation of the full or final even-width subspace; `[S,1,1,R]` frequencies and output scaling |
| `SDPA`, `SDPA_BWD` | f32 BHSD, GQA, scalar scale, broadcast bias, ALiBi causal mask, INT32 sequence lengths, top-left/bottom-right windows, custom/probability dropout, forward row outputs, backward Q/K/V/bias gradients, and the ragged/block-mask/sink subset below |

C5 specialized operation support is:

| Tags | Current constraints |
|---|---|
| `BLOCK_SCALE_QUANTIZE`, `BLOCK_SCALE_DEQUANTIZE` | Static divisible blocks; FLOAT compute; f32/f16/bf16 values and FP8/FP4/INT4 storage on the declared ports; FP4 uses packed low/high nibbles; E4M3/E8M0 scale output/input may use `F8_128x4` |
| `MATMUL_FP8` | FP8 E4M3/E5M2 A/B, scalar FLOAT descales/output scale, rank >= 2 batch broadcasting, FP8/f32/f16/bf16 C, scalar FLOAT `Amax_C`, and optional per-batch INT32 M/N/K overrides |
| `MOE_GROUPED_MATMUL`, `MOE_GROUPED_MATMUL_BWD` | `mode=NONE`, `top_k` 0 or 1, Token `[1,T,K]`, Weight `[E,K,N]`, INT32 offsets `[E,1,1]`, and one shared f32/f16/bf16 data type |
| `SDPA_FP8_FWD`, `SDPA_FP8_BWD` | Static FP8 E4M3/E5M2 BHSD with GQA, scalar FLOAT scales/descales, top-left or bottom-right windows, Stats and amax outputs; no padding, dropout, or ALiBi |
| `SDPA_MXFP8_FWD`, `SDPA_MXFP8_BWD` | Static BHSD/GQA, 32-element E8M0 block descales, f16/bf16/f32 output or gradients, transpose-oriented backward inputs, Stats and amax outputs; descale tensors accept `NONE` or Frontend `F8_128x4`, and backward dS uses the documented f32 CPU reference approximation |

Static f32 SDPA supports external ragged Q/K/V/O and forward row outputs;
backward also accepts ragged Q/K/V/O/dO/Stats and writes ragged dQ/dK/dV.
Forward K/V caches may be independently paged, and each INT32 page table may be
plain or compacted with its own prefix. Forward additionally accepts the
serialized UINT8 block mask, while forward/backward accept `SINK_TOKEN` and
backward may write `DSINK_TOKEN`. Ragged or paged storage requires
`padding_mask=true`, INT32 `SEQ_LEN_Q`/`SEQ_LEN_KV`, and the exact descriptors
documented in the runtime section. The pinned v1.24.0 schema exposes no paged
backward page-table ports, so that form is outside the input contract. C5 FP8
attention still defers padding, dropout, ALiBi, and optional ports. C6 also
implements producer-emitted `F8_128x4` scale reordering for the documented
block-scale and MXFP8 ports and the seven override subsets below.
The v1.24.0 standard-SDPA bottom-right causal path does not combine with bias,
ALiBi, or dropout. The CPU RNG is reproducible across DeepForge variants, but
it is not claimed to match cuDNN GPU Philox bits.

Comparison, logical, and generated-index pointwise outputs still use f32 `0`/`1`
or f32 index values. C2-C6 tags can be mixed when connected tensor types are
supported by both operations. Dynamic execution outside the pointwise, MATMUL,
RESHAPE, REDUCTION, TRANSPOSE, CONCATENATE, and SDPA-forward override subsets,
explicit aliasing, unsupported pass-by-value
descriptors, ragged/reordered tensors
outside the documented subset, distributed peer statistics, GPU execution,
CUDA device pointers, AMX, and internal multithreading are not supported. The maximum input file size is
16 MiB. The exact per-tag matrix is in the
[schema inventory](cudnn-graph-schema-inventory.en.md#5-capability-meaning).

The CUDA Toolkit and cuDNN backend are not dependencies. The project uses only
the serialization protocol and vendored nlohmann/json header from the open
source `cudnn-frontend` checkout.

## 2. Prerequisites

The pinned and verified dependencies are:

| Dependency | Version or requirement |
|---|---|
| LLVM/MLIR | Exactly `llvmorg-22.1.8` |
| cuDNN Frontend source | Exactly `v1.24.0` |
| nlohmann/json | Frontend-vendored `3.11.3` |
| CMake | `3.27` or newer |
| C++ compiler | C++20; GCC `13.4.0` is verified |
| Ninja | `1.10.1` is verified |

Select dependency locations for the current environment. DeepForge does not
provide machine-specific defaults:

```bash
: "${LLVM_INSTALL_PREFIX:?set LLVM_INSTALL_PREFIX}"
: "${CUDNN_FRONTEND_SOURCE_DIR:?set CUDNN_FRONTEND_SOURCE_DIR}"
```

In a new environment, only the cuDNN Frontend source checkout is needed; it
does not need to be built:

```bash
git clone --branch v1.24.0 --depth 1 \
  https://github.com/NVIDIA/cudnn-frontend.git \
  "$CUDNN_FRONTEND_SOURCE_DIR"
```

See the [README build section](../README.md#build) for the complete LLVM/MLIR
build commands. CMake finds both dependencies through `CMAKE_PREFIX_PATH`. It
rejects MLIR versions other than `22.1.8` and mismatched Frontend or JSON
headers.

## 3. Build and Install

Build the complete compiler, tools, benchmark, and tests from the repository
root:

```bash
./scripts/build.sh
```

Optional installation:

```bash
./scripts/build.sh --install install
export PATH="$PWD/install/bin:$PATH"
```

Sanitizer builds use the same dependency discovery and test path:

```bash
./scripts/build.sh --sanitizer address --build-dir build-asan
./scripts/build.sh --sanitizer undefined --build-dir build-ubsan
```

Run the fresh importer-only, Release, ASan, UBSan, and schedule A/B
qualification matrix with:

```bash
./scripts/release-check.sh --jobs 2
```

LeakSanitizer requires ptrace support. In a restricted local environment, use
`ASAN_OPTIONS=detect_leaks=0`; CI keeps leak detection enabled.

The current install rules publish only `deepforge-compile` and
`deepforge-benchmark`. Library headers, static libraries, and a CMake package
are not yet installed as a stable SDK. Use the source-tree CMake targets for
embedding.

## 4. Prepare an Input

Input should be produced by one of these APIs from the pinned Frontend version:

- `Graph::serialize(nlohmann::json&)` for JSON.
- `Graph::serialize(std::vector<uint8_t>&)` for UBJSON.

DeepForge does not define a private JSON schema. Use a repository fixture to
verify an installation:

```bash
cp test/fixtures/conv2d_f32_c17.json /tmp/graph.json
```

For the original optimized `CONV_FPROP` path, logical dimensions and packed
strides must be:

| Tensor | Logical dimensions | Packed stride |
|---|---|---|
| X | `[N,C,H,W]` | `[H*W*C,1,W*C,C]` |
| W | `[K,C,R,S]` | `[R*S*C,1,S*C,C]` |
| Y | `[N,K,P,Q]` | `[P*Q*K,1,Q*K,K]` |

The output spatial shape is:

```text
P = H + pre_h + post_h - R + 1
Q = W + pre_w + post_w - S + 1
```

The generic C3 convolution path accepts rank 3-5 tensors and positive
non-overlapping strides. For each spatial axis its output extent is:

```text
effective_filter = dilation * (filter_extent - 1) + 1
output_extent = 1 + (input_extent + pre + post - effective_filter) / stride
```

`W` is logically `[K,C_per_group,filter...]`; the group count is inferred as
`X.C / C_per_group`, and `Y.K` must be divisible by it.

X, W, and Y must have explicit, distinct UIDs. For every JSON or UBJSON graph,
nonempty root `workspace_modifications` or `variant_pack_replacements` are
rejected. Root `pass_by_values` is accepted only for the embedded scalar form
described below and otherwise fails strict consistency validation.

For a generic C2-C6 graph, every non-virtual tensor that is read or written,
except a graph-owned embedded scalar, must be present in the execute-time UID
map. Virtual tensors are omitted from the map and allocated in the queried
workspace. Writable buffers must not
overlap another argument or workspace. `VIEW_ONLY`, `in_place_index`, and a
node that reuses one UID as both input and output are rejected until alias
semantics are implemented. Tensor names do not replace explicit UIDs.
Normalization scalar inputs such as epsilon, momentum, and accumulation count
are f32 tensors with an all-ones shape matching the operation rank. They may be
ordinary tensor inputs or runtime pass-by-value inputs.

A runtime pass-by-value input has `is_pass_by_value=true`, a null
`pass_by_value` payload, an explicit UID, and dimensions that are all one. It
must be external, input-only, non-ragged, non-reordered, and use `INT64`,
`INT32`, `HALF`, `FLOAT`, `DOUBLE`, or `BFLOAT16`; the operation port may impose
a narrower type. The execute-time UID map supplies a host pointer to the
scalar, exactly as for another one-element read argument. This representation
survives artifact serialization without a format change. Pass-by-value outputs,
virtual tensors, and shape-override arrays remain unsupported.

An embedded pass-by-value scalar has the same tensor constraints, but
`pass_by_value` contains the v1.24.0 scalar variant object and root
`pass_by_values` contains an identical object under the canonical decimal UID.
The supported variant mapping is:

| `index` | Tensor type | `value` representation |
|---:|---|---|
| 0 | `INT64` | JSON integer |
| 1 | `INT32` | in-range JSON integer |
| 2 | `HALF` | JSON number |
| 3 | `FLOAT` | exactly eight hexadecimal digits containing the f32 bits |
| 4 | `DOUBLE` | JSON number |
| 5 | `BFLOAT16` | JSON number |

The tensor payload and root entry must be present one-to-one with equal type
and value. Missing, orphaned, malformed, or conflicting entries are rejected.
The compiler stores the scalar in a private read-only global, excludes its UID
from public argument metadata, and carries it inside every artifact object. The
execute-time UID map must omit the UID; an extra entry cannot override the
graph-owned value. The separate `has_compile_time_constant` fields present in
newer Frontend source are not emitted by the pinned tensor serializer and are
not part of this input contract.

An `F8_128x4` scale descriptor has two trailing logical matrix axes, M and K,
in either order. M is padded to a multiple of 128, K to a multiple of 4, the K
axis has stride 1, the M axis has stride K, and leading axes are packed. For a
flattened leading coordinate `l`, the byte offset is:

```text
((((l * (M / 128) + m / 128) * (K / 4) + k / 4) * 512)
 + (m % 32) * 16 + ((m / 32) % 4) * 4 + k % 4)
```

The UID map must point to the full padded physical byte span. Block-scale
quantize writes logical scale coordinates and initializes every padding slot
to numeric one: E4M3 `0x38` or E8M0 `0x7f`. `F8_128x4` on data tensors or
undocumented ports remains unsupported. `INT8x32` and `F16x16` are recognized
but non-executable: pinned v1.24.0 has no modern serialized-Graph producer path
that defines both a reachable operation port and its physical address mapping.
Support requires that producer contract and a generated fixture rather than an
inferred layout.

SDPA uses rank-4 BHSD tensors. `SEQ_LEN_Q` and `SEQ_LEN_KV` are INT32
`[B,1,1,1]`; probability dropout and dynamic RNG use one-element INT64 `Seed`
and `Offset` tensors. These metadata buffers are ordinary host pointers in the
same UID map. Forward `Stats` stores row log-sum-exp and is required by
`SDPA_BWD`; `Max`, `Sum_exp`, and `RNG_DUMP` are optional serialized outputs.

## 5. Compile an Artifact

Run from the build tree:

```bash
DEEPFORGE_COMPILE=build/tools/deepforge-compile

"$DEEPFORGE_COMPILE" /tmp/graph.json \
  --input-format=auto \
  --target=x86-64 \
  --emit=object \
  -o /tmp/conv2d.dfo
```

`--emit=object` emits a `.dfo` container holding scalar, AVX2, and AVX-512
objects. Without `-o`, the input extension is replaced with `.dfo`.

Compile UBJSON with:

```bash
"$DEEPFORGE_COMPILE" graph.ubjson \
  --input-format=ubjson \
  -o conv2d.dfo
```

Exit code `0` means success, `1` means a compilation, I/O, or artifact error,
and `2` means invalid command-line arguments. Output is published through a
unique same-directory temporary file and rename. Concurrent writes cannot
publish a partial file, but the last successful publication determines the
final content.

## 6. Inspect IR

Emit LLVM IR for one CPU variant:

```bash
"$DEEPFORGE_COMPILE" /tmp/graph.json \
  --emit=llvm-ir \
  --variant=avx2 \
  -o /tmp/conv2d-avx2.ll
```

Capture each MLIR stage while compiling an artifact:

```bash
"$DEEPFORGE_COMPILE" /tmp/graph.json \
  --variant=avx512 \
  --dump-ir=imported:/tmp/imported.mlir \
  --dump-ir=bufferized:/tmp/bufferized.mlir \
  --dump-ir=llvm:/tmp/llvm.mlir \
  -o /tmp/conv2d.dfo
```

`--variant` selects only the LLVM IR output or `llvm` dump. It does not remove
any of the three CPU variants from a `.dfo` artifact.

## 7. Inspect an Artifact

```bash
"$DEEPFORGE_COMPILE" --inspect /tmp/conv2d.dfo
```

The output reports the format and producer versions, target triple, function,
UIDs, workspace size, and each object's symbol, feature requirements, and byte
size. The loader requires an exact target-triple match with the current host.

A `.dfo` contains native code that executes with the current process's
permissions. Its FNV checksum detects accidental corruption but does not prove
provenance. Load artifacts only from trusted builds or distribution channels.

## 8. C++ API

Within a DeepForge source build, link `DeepForge::Compiler`:

```cmake
add_subdirectory(/path/to/DeepForge DeepForge-build)
target_link_libraries(my_app PRIVATE DeepForge::Compiler)
```

Minimal Graph compilation and execution example:

```cpp
#include "DeepForge/Compiler/Codegen.h"

#include <cstdlib>
#include <iostream>
#include <vector>

static std::size_t elements(std::array<std::int64_t, 4> const& shape) {
    std::size_t count = 1;
    for (auto value : shape) count *= static_cast<std::size_t>(value);
    return count;
}

int main() {
    deepforge::compiler::CompileOptions options;
    deepforge::compiler::CompilationResult result;
    auto status = deepforge::compiler::compile_file("graph.json", options,
                                                     result);
    if (status.is_bad()) {
        std::cerr << status.message() << '\n';
        return 1;
    }

    std::vector<float> x(elements(result.metadata.x_shape), 1.0F);
    std::vector<float> w(elements(result.metadata.w_shape), 1.0F);
    std::vector<float> y(elements(result.metadata.y_shape));
    deepforge::runtime::VariantPack pack{
        {result.metadata.x_uid, x.data()},
        {result.metadata.w_uid, w.data()},
        {result.metadata.y_uid, y.data()},
    };

    auto workspace_size = result.executable->get_workspace_size();
    if (workspace_size < 0) return 1;
    void* workspace = workspace_size == 0
                          ? nullptr
                          : std::aligned_alloc(
                                64, static_cast<std::size_t>(workspace_size));
    if (workspace_size != 0 && workspace == nullptr) return 1;

    status = result.executable->execute(nullptr, pack, workspace);
    std::free(workspace);
    if (status.is_bad()) {
        std::cerr << status.message() << '\n';
        return 1;
    }
}
```

Restore an executable from a file with:

```cpp
deepforge::compiler::ArtifactInfo info;
std::unique_ptr<deepforge::runtime::Executable> executable;
auto status = deepforge::compiler::load_artifact_executable(
    "conv2d.dfo", executable, &info);
```

With `is_override_shape_enabled=true`, the exact-pointwise policy accepts an
ordered DAG containing only `POINTWISE` nodes. Every used tensor is plain FLOAT
with the same compiled maximum shape; broadcasting, pass-by-value, ragged
offsets, and reordering are excluded. Inputs and the one external output use
the public UID map. Intermediate values may be virtual and use compiler-planned
workspace. Supply the same Frontend v1.24.0 override arrays to workspace query
and execution:

```cpp
deepforge::runtime::OverrideUids override_uids{a_uid, b_uid, y_uid};
deepforge::runtime::OverrideShapes override_shapes(
    3, std::vector<std::int64_t>{runtime_m, runtime_n});
deepforge::runtime::OverrideStrides override_strides(
    3, std::vector<std::int64_t>{runtime_ld, 1});

std::int64_t workspace_size = 0;
status = result.executable->get_workspace_size(
    nullptr, workspace_size, override_uids, override_shapes,
    override_strides);
if (status.is_good()) {
    status = result.executable->execute(
        nullptr, pack, workspace, override_uids, override_shapes,
        override_strides);
}
```

The three override arrays must have equal counts and unique external UIDs.
Each supplied shape preserves rank, has positive dimensions no larger than its
serialized maxima, and pairs with positive, supported non-overlapping strides
whose storage span fits the compiled byte bound. An empty list executes the
compiled maximum descriptors. A partial list is accepted only when all final
descriptors still satisfy the selected operation policy. Workspace query runs
the same checks as execution and returns the static maximum workspace bound.

For pointwise, all final external shapes must remain equal, so shrinking
normally overrides every external argument. Virtual UIDs never appear in the
arrays; their runtime shape is propagated from the equal external shape and
their packed views remain inside workspace allocated for serialized maxima.

One standard-f32 `MATMUL` can instead override its three external plain A, B,
and C descriptors. It cannot be composed with another node and cannot use
virtual, pass-by-value, ragged, reordered, or M/N/K extent-override tensors.
For rank three, a typical call is:

```cpp
deepforge::runtime::OverrideUids override_uids{a_uid, b_uid, c_uid};
deepforge::runtime::OverrideShapes override_shapes{
    {a_batch, runtime_m, runtime_k},
    {b_batch, runtime_k, runtime_n},
    {std::max(a_batch, b_batch), runtime_m, runtime_n}};
deepforge::runtime::OverrideStrides override_strides{
    a_runtime_strides, b_runtime_strides, c_runtime_strides};
```

All three tensors have equal rank at least two. The final descriptors must obey
`A[-2] == C[-2]`, `A[-1] == B[-2]`, and `B[-1] == C[-1]`. For every batch
axis, A and B are equal or one and C equals their maximum. Partial overrides
are legal when these relations remain true. DeepForge requires dimensions and
byte spans to fit the serialized maxima even though the upstream sample can
grow beyond a fake cache shape: the UID-map ABI has pointers but no allocation
lengths with which to validate a larger descriptor.

One dense standard-f32 `SDPA` forward can override external plain rank-4 Q, K,
V, O and optional Stats/Max/Sum_exp descriptors. It may be unmasked or use
top-left causal `right_bound=0`; bias, padding and sequence lengths, ALiBi,
sliding or bottom-right windows, dropout, paging/ragged storage, sink/block
masks, virtual tensors, and composed graphs are excluded. A typical complete
call uses:

```cpp
deepforge::runtime::OverrideUids override_uids{
    q_uid, k_uid, v_uid, o_uid, stats_uid, max_uid, sum_exp_uid};
deepforge::runtime::OverrideShapes override_shapes{
    {b, hq, sq, dqk}, {b, hk, skv, dqk}, {b, hv, skv, dv},
    {b, hq, sq, dv}, {b, hq, sq, 1}, {b, hq, sq, 1},
    {b, hq, sq, 1}};
deepforge::runtime::OverrideStrides override_strides{
    q_strides, k_strides, v_strides, o_strides,
    stats_strides, max_strides, sum_exp_strides};
```

Only B, Sq, and Skv may differ from the compiled descriptors; all head and
embedding dimensions remain fixed. Q/K/V/O batches are equal, Q/O share Hq and
Sq, K/V share Skv, Q/K share Dqk, O/V share Dv, and Hq must be divisible by Hk
and Hv. Optional row outputs are `[B,Hq,Sq,1]`. Partial overrides are accepted
only when the final descriptors preserve every relation.

One standard-f32 `RESHAPE` with `reshape_mode=LOGICAL` can override its external
plain X and Y descriptors. X is read-only and Y is write-only; virtual,
pass-by-value, ragged, reordered, additional tensors, and composed graphs are
excluded. X and Y may have different compiled ranks, but each runtime descriptor
must keep its own rank. For serialized maxima X `[2,3,4]` and Y `[4,6]`, one
valid non-contiguous call is:

```cpp
deepforge::runtime::OverrideUids override_uids{x_uid, y_uid};
deepforge::runtime::OverrideShapes override_shapes{
    {1, 3, 4},
    {3, 4}};
deepforge::runtime::OverrideStrides override_strides{
    {20, 6, 1},
    {7, 1}};
```

The resolved positive X and Y element counts must be equal, including after a
partial override. Execution preserves LOGICAL reshape order: lexicographic X
elements are written in lexicographic Y order. Dimensions, strides, and storage
spans remain subject to the common serialized bounds above.

One standard-f32 `REDUCTION` in any of the nine supported modes can override
external plain X and Y descriptors. X is read-only, Y is write-only, and both
keep the same compiled rank; virtual, pass-by-value, ragged, reordered,
additional tensors, and composed graphs are excluded. Reduction axes are frozen
from the serialized maxima: an axis is reduced when compiled X and Y differ,
with compiled Y equal to one. At runtime Y remains one on a reduced axis while X
may shrink. On every retained axis, final X and Y extents must be equal. For
serialized X `[2,3,4]` and Y `[2,1,4]`, a valid call is:

```cpp
deepforge::runtime::OverrideUids override_uids{x_uid, y_uid};
deepforge::runtime::OverrideShapes override_shapes{
    {1, 2, 3},
    {1, 1, 3}};
deepforge::runtime::OverrideStrides override_strides{
    {20, 5, 1},
    {7, 7, 1}};
```

Partial overrides must preserve the same axis relations. AVG divides by the
resolved runtime reduction count, not the serialized maximum count. All common
dimension, stride, and storage-span bounds still apply.

One standard-f32 `TRANSPOSE` can override external plain X and Y descriptors.
X is read-only, Y is write-only, and both retain the same compiled rank from
one through 64. Virtual, pass-by-value, ragged, reordered, additional tensors,
and composed graphs are excluded. The complete serialized permutation is fixed;
for permutation `[2,0,1]`, serialized X `[2,3,4]`, and Y `[4,2,3]`, one valid
non-contiguous call is:

```cpp
deepforge::runtime::OverrideUids override_uids{x_uid, y_uid};
deepforge::runtime::OverrideShapes override_shapes{
    {1, 2, 3},
    {3, 1, 2}};
deepforge::runtime::OverrideStrides override_strides{
    {20, 5, 1},
    {7, 4, 1}};
```

Every final descriptor pair, including after a partial override, must satisfy
`Y[i] == X[permutation[i]]`. X and Y strides may differ as long as each layout
is positive, non-overlapping, and within its serialized byte bound. The
permutation itself cannot be overridden.

One standard-f32 `CONCATENATE` can override one through 63 external plain input
descriptors plus external plain Y. Inputs are read-only, Y is write-only, every
role retains the same compiled rank from one through 64, and the serialized
non-negative axis is fixed. Role UIDs are distinct and input ports must be
contiguous `0..N-1`. In-place,
virtual, pass-by-value, ragged, reordered, additional tensors, and composed
graphs are excluded. For three inputs concatenated on axis 1, a valid
non-contiguous call is:

```cpp
deepforge::runtime::OverrideUids override_uids{
    x0_uid, x1_uid, x2_uid, y_uid};
deepforge::runtime::OverrideShapes override_shapes{
    {1, 1, 2}, {1, 2, 2}, {1, 1, 2}, {1, 4, 2}};
deepforge::runtime::OverrideStrides override_strides{
    {12, 4, 1}, {15, 4, 1}, {8, 4, 1}, {25, 4, 1}};
```

On every non-concatenation axis, each final input extent must equal Y. On the
fixed concatenation axis, the checked sum of all final input extents must equal
Y. These relations also apply after partial overrides. Each role may use an
independent positive non-overlapping stride within its serialized byte bound;
the input count, port order, rank, and axis cannot be overridden.

`is_dynamic_shape_enabled=true` without the override flag is persisted in the
plan and `.dfo` metadata but leaves execution descriptors static. Conversely,
the supported descriptor-override policies require only the override flag.

MATMUL extent overrides are a separate mechanism. `M_override`, `N_override`,
and `K_override` are optional external plain INT32 serialized node inputs passed
through the normal UID variant pack. Each has the same rank as C, trailing
dimensions `[1,1]`, and batch dimensions that are either one or equal to C.
Serialized A/B/C descriptors remain static allocation maxima. For each
broadcasted batch entry, valid values satisfy `0 <= M <= C[-2]`,
`0 <= N <= C[-1]`, and `0 <= K <= A[-1] == B[-2]`; an absent port selects its
full static extent. M/N select the valid output rectangle and K selects the
reduction extent. Standard f32 MATMUL writes its finite f32 `padding_value`
outside M/N, defaulting to zero. MATMUL_FP8 writes zero outside M/N, and its
`Amax_C` includes those zero values. Extent tensors do not change descriptors or
the execute ABI and cannot be combined with MATMUL descriptor override arrays.

For standard f32 SDPA, each supported ragged argument is an external plain
rank-4 logical tensor and names a separate external INT32 or INT64 offset tensor
with dimensions `[B+1,1,1,1]`. Forward supports Q/K/V/O/Stats/Max/Sum_exp;
backward supports Q/K/V/O/dO/Stats/dQ/dK/dV. Offsets are element indices, must
start at zero, be nondecreasing, and stay within the compiled maximum storage
span. At execution, each segment must hold the corresponding runtime sequence
length under the tensor's inner strides. The final prefix endpoint is the actual
data-buffer span used by alias checks, so a compact allocation is valid.
Backward `max_total_seq_len_q`/`max_total_seq_len_kv` are accepted only with
ragged storage, must be positive and no greater than `B*Sq`/`B*Skv`, and are
validation hints rather than numerical inputs. Runtime shape overrides cannot
be combined with ragged storage.

Paged SDPA forward accepts K and V independently. A paged container has
dimensions `[num_blocks,H,block_size,D]`; its INT32 page table has logical
dimensions `[B,1,page_slots,1]`. A table may be external plain storage or use an
independent INT32/INT64 `[B+1,1,1,1]` element prefix. For a compact table, batch
`b` must contain at least `ceil(SEQ_LEN_KV[b]/block_size)` page IDs; artifact v5
records the block-size divisor used by this conversion. `max_seq_len_kv`
supplies the logical K/V extent when present; otherwise Frontend order infers it
from an unpaged peer, Bias, `RNG_DUMP`, or finally the available page capacity.
Every page ID must be in `[0,num_blocks)`. Generated code guards an invalid ID
from becoming a memory address, but such a table violates the caller contract.
Ragged Q/O can be combined with paged K/V, while one K or V tensor cannot itself
be both paged and ragged. Runtime sequence values are caller preconditions:
`SEQ_LEN_Q` is in `[0,Sq]` and `SEQ_LEN_KV` is in `[0,logical_Skv]`; ragged
arguments additionally enforce their corresponding bound before dispatch.

The standard f32 forward block mask is an external plain UINT8 tensor with
dimensions `[B,Hq,ceil(Sq/128),ceil(ceil(Skv/128)/8)]`. Each bit enables one
128-by-128 query/key tile; key-tile bits are packed least-significant-bit first
within each byte. It combines with the other score masks.

`SINK_TOKEN` is an external plain f32 `[1,Hq,1,1]` tensor. Its per-head value is
an extra softmax logit in every valid row: it contributes to normalization but
has no V value. Standard f32 forward and backward support it; backward may emit
an external plain `DSINK_TOKEN` with the same shape, reduced over valid batch
and query rows.

Runtime contract:

- The variant pack must provide host pointers for every non-virtual argument
  UID in the metadata. Extra UIDs are ignored.
- Each argument must meet its recorded alignment and have capacity for its
  compiled span, except that a validated ragged prefix defines that call's
  compact span; an override call needs at least its supplied runtime span. The
  API carries no buffer lengths, so the runtime cannot prove actual sizes.
- Writable argument and workspace address ranges must not overlap.
- Allocate the size returned by `get_workspace_size()` with 64-byte alignment
  when it is non-zero.
- `FrontendHandle` is an opaque `void*` retained for the Frontend call shape.
  The CPU runtime does not dereference it, so null is valid.
- Concurrent calls on one `Executable` are safe, but the implementation
  currently serializes kernel invocation internally. Each call still requires
  separate output and workspace storage.

## 9. Benchmark

```bash
DEEPFORGE_BENCHMARK=build/tools/deepforge-benchmark
"$DEEPFORGE_BENCHMARK" \
  --profile=all --iterations=3 --schedule=both
```

Profiles are `small`, `medium`, `large`, or `all`; iterations must be in
`[1,1000]`. Schedule policy is `auto` (default), `baseline`, or `both`. CSV
output includes the policy and selected `direct-c-vf<VF>-ku<KU>` name,
compilation time, per-execution time, GFLOP/s, and maximum absolute and relative
differences from the scalar result. The cost model applies only to the original
optimized static packed f32 Conv path; generic graphs report
`generic-reference`. The benchmark is a regression baseline, not a performance
guarantee across hosts.

## 10. Troubleshooting

| Diagnostic | Action |
|---|---|
| CMake reports an LLVM version mismatch | Put the `22.1.8` prefix first in `CMAKE_PREFIX_PATH`, or set `MLIR_DIR` explicitly |
| Frontend or JSON header is missing | Add the `v1.24.0` checkout to `CMAKE_PREFIX_PATH`, or set `DEEPFORGE_CUDNN_FRONTEND_INCLUDE_DIR` explicitly |
| `DFE_SCHEMA_VERSION_MISMATCH` | Use Graph JSON schema `1.0` |
| `DFE_FRONTEND_VERSION_MISMATCH` | Re-serialize with cuDNN Frontend `v1.24.0` |
| `DFE_UNSUPPORTED_NODE` | Use a tag listed in the current capability matrix |
| `DFE_UNSUPPORTED_OPERATION` | Remove deferred attributes, unsupported peer statistics, or a configuration outside the tag's declared CPU subset |
| `DFE_INVALID_LAYOUT` | Check packed Conv strides or positive non-overlapping foundational strides |
| `DFE_INVALID_SHAPE` | Check static dimensions, operation shape rules, the Conv output formula, and override maxima/ranks/spans |
| `DFE_INVALID_VARIANT_PACK` | Check UIDs, host pointers, alignment, aliasing, and workspace |
| `DFE_UNSUPPORTED_CPU_FEATURE` | Do not force an unsupported variant; use automatic `execute` |
| Artifact target mismatch | Recompile on the target host or an identical target triple |

See the [MVP compatibility and runtime contract](design/contracts.en.md) and
[schema inventory](cudnn-graph-schema-inventory.en.md) for the normative
subsets, and [DFO Artifact Format](artifact-format.en.md) for the binary layout.
