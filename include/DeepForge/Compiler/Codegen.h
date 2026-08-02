#pragma once

#include "DeepForge/Compiler/Bufferization.h"
#include "DeepForge/Compiler/Schedule.h"
#include "DeepForge/Import/SerializedGraphImporter.h"
#include "DeepForge/Runtime/Executable.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace deepforge::compiler {

struct CompileOptions {
    Conv2DImportOptions importer;
    std::string foundational_function_name = "deepforge_graph";
    import::InputFormat input_format = import::InputFormat::kAuto;
    bool build_avx_variants = true;
    bool emit_object = true;
    bool emit_llvm_ir = true;
    bool capture_mlir = false;
    Conv2DSchedulePolicy schedule_policy = Conv2DSchedulePolicy::kAuto;
};

struct VariantCode {
    runtime::CpuVariant variant = runtime::CpuVariant::kScalar;
    std::string symbol;
    std::string required_features;
    std::string schedule;
    std::string mlir;
    std::string llvm_ir;
    std::vector<std::uint8_t> object;
};

struct CompilationResult {
    std::unique_ptr<runtime::Executable> executable;
    InvocationAdapterKind adapter_kind =
        InvocationAdapterKind::kConv2DRankedMemref;
    Conv2DCompileMetadata metadata;
    WorkspacePlan workspace;
    std::string target_triple;
    std::string imported_mlir;
    std::string bufferized_mlir;
    std::array<VariantCode, 3> variants;
};

// Compile an already validated serialized graph into scalar and, when enabled,
// AVX2/AVX-512 CPU variants.
[[nodiscard]] import::Status compile_graph(
    import::SerializedGraph const& graph,
    CompileOptions const& options,
    CompilationResult& output);

[[nodiscard]] import::Status compile_file(
    std::filesystem::path const& path,
    CompileOptions const& options,
    CompilationResult& output);

}  // namespace deepforge::compiler
