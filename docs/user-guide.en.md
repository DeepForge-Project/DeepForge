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
| Operations | Validated CPU subsets for all 39 serialized v1.24.0 tags; one exact-shape f32 `POINTWISE` subset supports runtime override |
| Generic tensors | Rank 1-64 f32 data with explicit UID; shapes are otherwise static; documented C4 metadata may be INT32/INT64; virtual intermediates are supported |
| Generic layout | Positive, non-overlapping arbitrary strides; no ragged metadata; `F8_128x4` is enabled only for the scale ports below |
| C5 specialized storage | FLOAT16, BFLOAT16, FP8 E4M3/E5M2/E8M0, packed FP4 E2M1 and INT4, plus FLOAT controls on documented ports |
| Convolution | Rank 3-5 FPROP/DGRAD/WGRAD, grouped channels, positive stride/dilation, non-negative asymmetric padding, both math modes |
| CPU code | Scalar, AVX2+FMA, and AVX-512F+FMA with runtime dispatch |
| Output | LLVM IR or a `.dfo` artifact containing three native objects |

The foundational operation subset is:

| Tag | Current constraints |
|---|---|
| `RESHAPE` | `LOGICAL` only; equal element count |
| `TRANSPOSE` | Complete static permutation |
| `SLICE` | In-range half-open bounds and positive integer strides |
| `CONCATENATE` | Numbered inputs, non-negative axis, no in-place mode |
| `POINTWISE` | All 50 v1.24.0 modes with trailing-dimension NumPy broadcasting |
| `REDUCTION` | All 9 modes; same input/output rank and reduced extents equal to one |
| `MATMUL` | Equal rank >= 2, broadcast batch dimensions, no M/N/K override, zero padding value |
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
| `SDPA`, `SDPA_BWD` | f32 BHSD, GQA, scalar scale, broadcast bias, ALiBi causal mask, INT32 sequence lengths, top-left/bottom-right windows, custom/probability dropout, forward row outputs, and backward Q/K/V/bias gradients |

C5 specialized operation support is:

| Tags | Current constraints |
|---|---|
| `BLOCK_SCALE_QUANTIZE`, `BLOCK_SCALE_DEQUANTIZE` | Static divisible blocks; FLOAT compute; f32/f16/bf16 values and FP8/FP4/INT4 storage on the declared ports; FP4 uses packed low/high nibbles; E4M3/E8M0 scale output/input may use `F8_128x4` |
| `MATMUL_FP8` | FP8 E4M3/E5M2 A/B, scalar FLOAT descales/output scale, rank >= 2 batch broadcasting, FP8/f32/f16/bf16 C, scalar FLOAT `Amax_C`; no M/N/K override |
| `MOE_GROUPED_MATMUL`, `MOE_GROUPED_MATMUL_BWD` | `mode=NONE`, `top_k` 0 or 1, Token `[1,T,K]`, Weight `[E,K,N]`, INT32 offsets `[E,1,1]`, and one shared f32/f16/bf16 data type |
| `SDPA_FP8_FWD`, `SDPA_FP8_BWD` | Static FP8 E4M3/E5M2 BHSD with GQA, scalar FLOAT scales/descales, top-left or bottom-right windows, Stats and amax outputs; no padding, dropout, or ALiBi |
| `SDPA_MXFP8_FWD`, `SDPA_MXFP8_BWD` | Static BHSD/GQA, 32-element E8M0 block descales, f16/bf16/f32 output or gradients, transpose-oriented backward inputs, Stats and amax outputs; descale tensors accept `NONE` or Frontend `F8_128x4`, and backward dS uses the documented f32 CPU reference approximation |

Paged/cache attention, block masks, sink tokens, and packed/ragged attention
are deferred. C5 FP8 attention also defers padding, dropout, ALiBi, and optional
ports. C6 implements producer-emitted `F8_128x4` scale reordering for the
documented block-scale and MXFP8 ports and the pointwise override subset below.
The v1.24.0 standard-SDPA bottom-right causal path does not combine with bias,
ALiBi, or dropout. The CPU RNG is reproducible across DeepForge variants, but
it is not claimed to match cuDNN GPU Philox bits.

Comparison, logical, and generated-index pointwise outputs still use f32 `0`/`1`
or f32 index values. C2-C6 tags can be mixed when connected tensor types are
supported by both operations. Dynamic execution outside the pointwise override
subset, explicit aliasing, scalar pass-by-value, ragged/reordered tensors
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

X, W, and Y must have explicit, distinct UIDs. Non-empty UBJSON
`pass_by_values`, `workspace_modifications`, or `variant_pack_replacements`
are rejected because they carry execution semantics not implemented by the
CPU MVP.

For a generic C2-C5 graph, every non-virtual tensor that is read or written
must be present in the execute-time UID map. Virtual tensors are omitted from
the map and allocated in the queried workspace. Writable buffers must not
overlap another argument or workspace. `VIEW_ONLY`, `in_place_index`, and a
node that reuses one UID as both input and output are rejected until alias
semantics are implemented. Tensor names do not replace explicit UIDs.
Normalization scalar inputs such as epsilon, momentum, and accumulation count
are explicit f32 tensors with an all-ones shape matching the operation rank;
pass-by-value scalar serialization is deferred.

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
to numeric one: E4M3 `0x38` or E8M0 `0x7f`. `F8_128x4` on data tensors or undocumented ports, and
the `INT8x32` and `F16x16` reorder formats, remain unsupported.

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

For a graph serialized with `is_override_shape_enabled=true`, the currently
executable dynamic subset is exactly one `POINTWISE` node whose inputs and
output are external, plain FLOAT tensors with identical compiled shapes. It
does not allow broadcasting, virtual tensors, ragged offsets, or reordering.
Compiled dimensions and byte spans are maxima. Supply the same Frontend
v1.24.0 override arrays to workspace query and execution:

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
Each shape must preserve rank, use positive dimensions no larger than the
serialized maxima, and pair with positive, supported non-overlapping strides
whose storage span fits the compiled byte bound. All final pointwise argument
shapes must remain equal; a shrinking call therefore normally overrides every
argument. An empty list executes the compiled maximum shape. The workspace
query validates the same rules and is statically bounded for this subset.
`is_dynamic_shape_enabled=true` without the override flag is persisted in the
plan and `.dfo` metadata but leaves execution descriptors static.

Runtime contract:

- The variant pack must provide host pointers for every non-virtual argument
  UID in the metadata. Extra UIDs are ignored.
- Each argument must meet its recorded alignment and have capacity for its
  compiled span, or at least the supplied runtime span for an override call.
  The API carries no buffer lengths, so the runtime cannot prove actual sizes.
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
"$DEEPFORGE_BENCHMARK" --profile=all --iterations=3
```

Profiles are `small`, `medium`, `large`, or `all`; iterations must be in
`[1,1000]`. CSV output includes compilation time, per-execution time, GFLOP/s,
and maximum absolute and relative differences from the scalar result. The
benchmark is a regression baseline, not a performance guarantee across hosts.

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
