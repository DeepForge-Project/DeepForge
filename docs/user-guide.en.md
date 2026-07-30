# DeepForge User Guide

[中文](user-guide.zh-CN.md)

This guide is for users compiling and executing cuDNN Frontend serialized
Graphs with the DeepForge CPU MVP. See the [design overview](design/overview.en.md)
for implementation details.

## 1. Supported Scope

DeepForge `0.1.0` currently supports:

| Area | Supported scope |
|---|---|
| Platform | Linux x86-64 |
| Input | Graph JSON or canonical UBJSON produced by cuDNN Frontend `v1.24.0` |
| Graph schema | `json_version == "1.0"`, `cudnn_frontend_version == 12400` |
| Operation | Exactly one `CONV_FPROP` |
| Tensors | Static rank-4, f32, non-virtual, not pass-by-value |
| Layout | Packed NHWC X/Y and packed KRSC W |
| Convolution | Cross-correlation, stride 1, dilation 1, static non-negative padding |
| CPU code | Scalar, AVX2+FMA, and AVX-512F+FMA with runtime dispatch |
| Output | LLVM IR or a `.dfo` artifact containing three native objects |

Dynamic shapes, grouped/depthwise convolution, bias or activation fusion,
other operations, GPU execution, CUDA device pointers, bf16, AMX, and internal
multithreading are not supported. The maximum input file size is 16 MiB.

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

Logical dimensions and packed strides must be:

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

X, W, and Y must have explicit, distinct UIDs. Non-empty UBJSON
`pass_by_values`, `workspace_modifications`, or `variant_pack_replacements`
are rejected because they carry execution semantics not implemented by the
CPU MVP.

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

Runtime contract:

- The variant pack must provide host pointers for the X, W, and Y UIDs in the
  metadata. Extra UIDs are ignored.
- X, W, and Y must be aligned to at least `alignof(float)` and have full tensor
  capacity. The API carries no buffer lengths, so the runtime cannot prove the
  actual allocation sizes.
- The effective X, W, Y, and workspace address ranges must not overlap.
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
| `DFE_UNSUPPORTED_NODE` | Reduce the Graph to one `CONV_FPROP` |
| `DFE_INVALID_LAYOUT` | Check packed NHWC X/Y and packed KRSC W strides |
| `DFE_INVALID_SHAPE` | Check positive static dimensions and the Conv output formula |
| `DFE_INVALID_VARIANT_PACK` | Check UIDs, host pointers, alignment, aliasing, and workspace |
| `DFE_UNSUPPORTED_CPU_FEATURE` | Do not force an unsupported variant; use automatic `execute` |
| Artifact target mismatch | Recompile on the target host or an identical target triple |

See the [MVP compatibility and runtime contract](design/contracts.en.md) for the
normative specification and [DFO Artifact Format](artifact-format.en.md) for the
binary layout.
