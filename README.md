# DeepForge

[中文](README.zh-CN.md) | [English User Guide](docs/user-guide.en.md) | [中文用户指南](docs/user-guide.zh-CN.md)

DeepForge is an MLIR-based CPU compiler. It reads the serialized Graph format
defined by the open source `cudnn-frontend` project, lowers the supported graph
subset to LLVM IR and x86-64 machine code, and executes it through the cuDNN
Frontend-shaped UID variant-pack call interface.

**Current status:** CPU MVP phases P0-P6, post-MVP coverage phases C0-C5, and
the first four C6 increments are implemented. The end-to-end path
includes a strict JSON/UBJSON importer, standard Tensor/Linalg IR, exactly one
One-Shot Bufferize run, static workspace planning, scalar/AVX2/AVX-512 object
generation, CPUID dispatch, a Frontend-shaped runtime, reloadable `.dfo`
artifacts, CLI tools, benchmarks, and sanitizer coverage. All 39 serialized
v1.24.0 tags have declared and tested CPU execution subsets. These subsets are
strictly narrower than the complete set of legal cuDNN backend configurations.

**MVP scope:** one static, contiguous, f32 Conv2D forward operation targeting
x86-64, with scalar, AVX2, and AVX-512 code variants. Dynamic shapes, grouped
or depthwise convolution, fused operations, multithreading, Machine Dialect,
AMX, and bf16 are outside the MVP.

DeepForge does not build, link, or call the CUDA/cuDNN backend. The CPU runtime
uses the open source Frontend repository only as the pinned serialization and
call-interface reference.

## Pinned Versions

| Dependency | Pinned version/commit | Status |
|---|---|---|
| LLVM/MLIR | `llvmorg-22.1.8` (`ca7933e47d3a`) | Installed and verified |
| cuDNN Frontend | `v1.24.0` (`c4a97621eca5`) | Downloaded as the serialization reference |
| nlohmann/json | `3.11.3` | Vendored by the Frontend source tree |
| C++ language mode | `C++20` | Enforced by CMake |

LLVM `main`, rolling MLIR revisions, and LLVM 23 release candidates are not
supported. See the [MVP compatibility and runtime contract](docs/design/contracts.en.md)
for the normative compatibility boundary.

## External Toolchain Versions

The table below records the Linux x86-64 toolchain baseline used to build and
test the current implementation. A **pinned** dependency is part of the project
compatibility contract. A **verified** tool version is the tested workspace
baseline and should be preferred in CI. The minimum CMake version is enforced
by `cmake_minimum_required` in `CMakeLists.txt`.

| Tool/dependency | Version or constraint | Status and purpose |
|---|---|---|
| OS/architecture | Linux x86-64; glibc `2.35` | Verified host baseline |
| Git | `2.34.1` | Verified; fetches pinned tags and is not part of the runtime ABI |
| CMake | `4.4.0`; minimum `3.27` | Verified; configures LLVM/MLIR and DeepForge |
| Ninja | `1.10.1` | Verified build generator |
| GNU C/C++ | `13.4.0`; C++ `20` | Verified host compiler |
| GNU binutils/`ld` | `2.38` | Verified host linker |
| Python | `3.10.12`; LLVM requires at least `3.8` | Verified LLVM/MLIR configuration tool |
| zlib development package | `1.2.11` | Verified LLVM/MLIR compression dependency |
| LLVM/MLIR | `22.1.8`; X86 target | Pinned; the only supported MLIR/LLVM toolchain |
| cuDNN Frontend | `v1.24.0` (`c4a97621eca5`) | Pinned serialization schema and API reference |
| nlohmann/json | `3.11.3` | Pinned Frontend-vendored header; no system package required |
| CUDA Toolkit/cuDNN backend | Not an MVP dependency; not installed | Not needed by the CPU importer or runtime |

A verified version does not imply that every newer version is compatible.
Changes to CMake, the host compiler, binutils, LLVM/MLIR, or the parser require
the configuration smoke tests and the full end-to-end test suite to be rerun.
The LLVM/MLIR version check fails hard rather than silently accepting another
version.

CMake resolves build tools from `PATH`; DeepForge does not encode executable
locations. Probe a new environment before configuring the project:

```bash
command -v cmake ninja c++ ld python3 git
cmake --version | head -1
ninja --version
c++ --version | head -1
ld --version | head -1
python3 --version
git --version
```

If CMake is older than `3.27`, update `PATH` or install a supported CMake
before configuring DeepForge.

## Dependency Discovery

DeepForge does not encode dependency locations. CMake discovers MLIR with
`find_package(MLIR CONFIG)` and the cuDNN Frontend headers with `find_path`.
Add the LLVM/MLIR install prefix and cuDNN Frontend checkout to
`CMAKE_PREFIX_PATH`. For nonstandard layouts, `MLIR_DIR` and
`DEEPFORGE_CUDNN_FRONTEND_INCLUDE_DIR` are explicit overrides.

For the CPU MVP, downloading the cuDNN Frontend source is sufficient. DeepForge
reads the serializer implementation and reuses its vendored JSON header; it
does not run the Frontend CMake build or link the GPU backend. Frontend samples
and aggregate headers may require the CUDA Toolkit and `cudnn.h`, but those are
not needed to build or run DeepForge.

CUDA discovery is intentionally absent. Exact `cudnnHandle_t` and
`cudnn_frontend::error_t` types, GPU execution, and binary interchangeability
with a Frontend `Graph` object are outside this CPU-only build.

## Supported Input

The importer accepts the two carriers defined by cuDNN Frontend `v1.24.0`:

- Graph JSON produced through `Graph::serialize(nlohmann::json&)`.
- Canonical UBJSON produced through `Graph::serialize(std::vector<uint8_t>&)`.

The schema must have `json_version == "1.0"` and
`cudnn_frontend_version == 12400`. A serialized input is limited to 16 MiB.
The current executable forms are:

- the original optimized single-node packed f32 rank-4 `CONV_FPROP` path;
- a static ordered DAG using the three convolution tags, the eight C2
  foundational tags, the 14 C3 normalization/statistics tags, and the five C4
  sequence/attention tags;
- the nine C5 specialized tags for block-scale conversion, FP8 matmul,
  FP8/MXFP8 attention, and MoE grouped matmul, within the constraints declared
  in
  the [schema capability matrix](docs/cudnn-graph-schema-inventory.en.md#5-capability-meaning).

The generic path supports rank-3 through rank-5 grouped convolution with
stride, dilation, asymmetric padding, FPROP/DGRAD/WGRAD, and mixed C2-C5
graphs. It also supports normalization forward/backward, batch statistics,
running-stat updates, deterministic Bernoulli RNG, RoPE forward/backward, and
f32 SDPA forward/backward within the matrix constraints. C5 adds software
FLOAT16/BFLOAT16/FP8/FP4/INT4 conversion and packed storage where required by
the nine specialized tags. C4 sequence metadata uses INT32 lengths and scalar
INT64 seed/offset tensors. Virtual workspace intermediates and positive
non-overlapping strided layouts are supported. C6 additionally decodes the
Frontend `F8_128x4` physical layout on documented FP8 block-scale and E8M0
MXFP8 scale ports. Runtime shape override is executable for one external,
non-broadcasting, plain f32 `POINTWISE` node: serialized dimensions are maxima,
and the Frontend-shaped override arrays supply positive runtime dimensions and
strides within the compiled storage bounds. The standalone dynamic-shape
context flag is preserved as plan metadata but does not by itself make another
operation dynamic. Static f32 SDPA supports external ragged forward data and
row outputs, and ragged backward data and gradients, using validated
element-prefix offsets. Forward also supports independently paged K/V caches,
including compact INT32 page tables with independent prefixes, compressed
UINT8 block masks, and per-query-head sink logits; backward can return the sink
gradient. These storage forms require padding and explicit sequence lengths.
Explicit aliasing, scalar pass-by-value, paged backward attention, other tensor
reordering, distributed peer statistics, and optional features on the
specialized FP8/MXFP8 attention paths are not executable yet.

DeepForge does not define a private graph JSON format. Unsupported schema,
nodes, layouts, execution metadata, or shapes are rejected with stable
diagnostics rather than silently ignored or lowered through a fallback path.

## Architecture

```text
cuDNN Frontend serialized Graph (JSON or UBJSON)
        |
        v
DeepForge importer + support validation
        |
        v
MVP Conv: Tensor + Linalg    Generic C2-C6: MemRef + SCF + Math
        |  one-shot-bufferize once       |
        +----------------------+----------+
                               v
MemRef + Affine/SCF + Vector Dialect
        |  direct Conv2D schedule, C-reduction vectorization
        v
LLVM Dialect
        |  translate to LLVM IR, LLVM target code generation
        v
scalar / AVX2 / AVX-512 object code
        |
        v
DeepForge Executable::execute(handle, uid_to_host_ptr, workspace)
```

DeepForge-specific logic is limited to import/support validation, semantic
lowering, workspace planning, and runtime dispatch. The main IR pipeline uses
upstream MLIR dialects, without an intermediate `cudnn.*` dialect or a custom
Machine Dialect.

## Documentation

| Document | Contents |
|---|---|
| [English User Guide](docs/user-guide.en.md) | Build, CLI, runtime API, benchmark, and troubleshooting |
| [中文用户指南](docs/user-guide.zh-CN.md) | Build, CLI, runtime API, benchmark, and troubleshooting in Chinese |
| [MVP Contract](docs/design/contracts.en.md) | Versions, input schema, support matrix, Frontend-shaped API, and numeric rules |
| [Architecture Overview](docs/design/overview.en.md) | Components, IR stages, and major design decisions |
| [Tensor Layer](docs/design/tensor-layer.en.md) | Serialized Graph import, shape conversion, and padding |
| [Linalg Layer](docs/design/linalg-layer.en.md) | `linalg.conv_2d_nhwc_fhwc` semantics and layouts |
| [Loop and Schedule Layer](docs/design/loop-schedule-layer.en.md) | Direct Conv loops, reduction vectorization, cost-model ownership, and deferred tiling |
| [Machine Dialect](docs/design/machine-dialect.en.md) | Deferral rationale and criteria for reconsideration |
| [Vector and LLVM Layer](docs/design/vector-llvm-layer.en.md) | Complete LLVM lowering and CPU variants |
| [Pass Pipeline](docs/design/pass-pipeline.en.md) | MVP pass order, preconditions, and verification |
| [Conv2D Example](docs/design/example-conv2d.en.md) | End-to-end IR example |
| [DFO Artifact Format](docs/artifact-format.en.md) | Binary layout, ORC loading, and trust boundary |
| [Benchmark Baseline](docs/benchmark-baseline.en.md) | Small, medium, and large profiles and reproduction steps |
| [Development Plan](docs/development-plan.en.md) | Phase order, exit criteria, and test gates |
| [cuDNN Graph Operation Coverage Plan](docs/cudnn-graph-coverage-plan.en.md) | Approved post-MVP architecture, operation phases, and acceptance gates |
| [cuDNN Graph Schema Inventory](docs/cudnn-graph-schema-inventory.en.md) | v1.24.0 root fields, tensors, 39 operation tags, ports, attributes, and modes |

## Build

Once the pinned dependencies are available to CMake, build and test DeepForge
from the repository root:

```bash
./scripts/build.sh
```

Build only the CPU importer without finding MLIR or any CUDA/cuDNN backend:

```bash
./scripts/build.sh --importer-only
```

The script derives the source directory from its own location, preserves the
environment's `CMAKE_PREFIX_PATH`, and supplies `LLVM_INSTALL_PREFIX` and
`CUDNN_FRONTEND_SOURCE_DIR` as additional search prefixes when they are set. It
does not assume dependency locations. Common options:

```bash
./scripts/build.sh --build-type Debug --build-dir build-debug
./scripts/build.sh --jobs 8 --no-tests
./scripts/build.sh --help
```

For first-time dependency setup, choose writable source, build, and install
directories. The variables below intentionally have no project-defined
defaults:

```bash
# Set these variables to locations selected for your environment.
: "${LLVM_SOURCE_DIR:?set LLVM_SOURCE_DIR}"
: "${LLVM_BUILD_DIR:?set LLVM_BUILD_DIR}"
: "${LLVM_INSTALL_PREFIX:?set LLVM_INSTALL_PREFIX}"
: "${CUDNN_FRONTEND_SOURCE_DIR:?set CUDNN_FRONTEND_SOURCE_DIR}"

# Serialization reference source only; no CUDA installation or build step.
git clone --branch v1.24.0 --depth 1 \
  https://github.com/NVIDIA/cudnn-frontend.git \
  "$CUDNN_FRONTEND_SOURCE_DIR"

# Build and install LLVM/MLIR once.
git clone --branch llvmorg-22.1.8 --depth 1 \
  https://github.com/llvm/llvm-project.git \
  "$LLVM_SOURCE_DIR"
cmake -S "$LLVM_SOURCE_DIR/llvm" -B "$LLVM_BUILD_DIR" -G Ninja \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD=X86 \
  -DLLVM_BUILD_TESTS=OFF \
  -DMLIR_INCLUDE_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$LLVM_INSTALL_PREFIX"
cmake --build "$LLVM_BUILD_DIR" --target install -j
```

Verify the pinned LLVM/MLIR tools:

```bash
export PATH="${LLVM_INSTALL_PREFIX}/bin:${PATH}"
llvm-config --version
mlir-opt --version
mlir-translate --version
llc --version
```

CMake validates LLVM/MLIR `22.1.8`, cuDNN Frontend `1.24.0`, and the vendored
nlohmann/json `3.11.3` header. A mismatch is a configuration error.

Optional CLI installation:

```bash
./scripts/build.sh --install install
```

The current install rules publish `deepforge-compile` and
`deepforge-benchmark`. Library headers and a CMake package are not yet installed
as a stable SDK.

## Compile and Execute

Generate a reloadable three-variant artifact and inspect its metadata:

```bash
build/tools/deepforge-compile test/fixtures/conv2d_f32_c17.json \
  --input-format=auto \
  --target=x86-64 \
  --emit=object \
  --dump-ir=imported:build/imported.mlir \
  --dump-ir=bufferized:build/bufferized.mlir \
  --dump-ir=llvm:build/llvm.mlir \
  -o build/conv2d.dfo

build/tools/deepforge-compile --inspect build/conv2d.dfo
build/tools/deepforge-benchmark --profile=all --iterations=3
```

Use `--emit=llvm-ir --variant=scalar|avx2|avx512` to emit LLVM IR for one
variant. `load_artifact_executable` restores an ORC executable from a `.dfo`
file or an in-memory byte span. Artifacts contain native object code and are not
a security sandbox; load only artifacts from trusted sources.

The source-tree library API uses the following flow:

```cpp
deepforge::compiler::CompileOptions options;
deepforge::compiler::CompilationResult compilation;
auto status = deepforge::compiler::compile_file("graph.json", options,
                                                compilation);
if (status.is_bad()) {
    // status.code() and status.message()
}

deepforge::runtime::VariantPack variant_pack = {
    {x_uid, x_host_ptr},
    {w_uid, w_host_ptr},
    {y_uid, y_host_ptr},
};

auto workspace = allocate_aligned(
    compilation.executable->get_workspace_size(), 64);
deepforge::runtime::FrontendHandle handle = nullptr;
status = compilation.executable->execute(handle, variant_pack,
                                         workspace.get());

std::unique_ptr<deepforge::runtime::Executable> loaded;
status = deepforge::compiler::load_artifact_executable("conv2d.dfo", loaded);
```

The public call shapes are pinned to the cuDNN Frontend `v1.24.0` UID
overloads:

```cpp
execute(handle, std::unordered_map<int64_t, void*>&, workspace);
execute(handle, uid_map, workspace,
        override_uids, override_shapes, override_strides);
get_workspace_size(handle, workspace_size,
                   override_uids, override_shapes, override_strides);
```

The CPU runtime does not inspect the opaque handle. It accepts host pointers,
not CUDA device pointers. DeepForge uses its own status type and executable
class, so this is source-level call-shape compatibility rather than binary
interchangeability with a Frontend `Graph`. Public headers expose neither MLIR
memref descriptors nor raw generated-kernel signatures.

## Roadmap

| Phase | Status |
|---|---|
| P0-P2 | Complete: toolchain, strict importer, and Tensor/Linalg IR |
| P3 | Complete: One-Shot Bufferize and static workspace planning |
| P4 | Complete: scalar LLVM/object generation, JIT, and runtime |
| P5 | Complete: AVX2/AVX-512, tails, and CPUID/XGETBV dispatch |
| P6 | Complete: CLI, reloadable artifacts, CI, benchmark, and quality gates |
| C0-C5 | Complete: generic graph/runtime foundation and validated subsets for all 39 serialized tags |
| C6 | In progress: `F8_128x4`, exact-pointwise runtime shape override, and standard f32 SDPA ragged/packed/block-mask/sink metadata complete; broader dynamic/reorder behavior, optimization, and release qualification remain |
| Optimize | Pending benchmark-driven outer-loop tiling, padding fusion, and parallelism |
| Re-evaluate | Reconsider Machine Dialect only after two backends need a shared abstraction |

## References

- [MLIR](https://mlir.llvm.org/)
- [cuDNN Frontend](https://github.com/NVIDIA/cudnn-frontend)
- [LLVM releases](https://github.com/llvm/llvm-project/releases)

## License

Apache 2.0. See [LICENSE](LICENSE).
