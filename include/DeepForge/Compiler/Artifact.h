#pragma once

#include "DeepForge/Compiler/Codegen.h"

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace deepforge::compiler {

inline constexpr std::uint32_t kLegacyArtifactFormatVersion = 1;
inline constexpr std::uint32_t kStaticMetadataArtifactFormatVersion = 2;
inline constexpr std::uint32_t kShapeOverrideArtifactFormatVersion = 3;
inline constexpr std::uint32_t kRaggedArtifactFormatVersion = 4;
inline constexpr std::uint32_t kArtifactFormatVersion = 5;
inline constexpr double kDefaultAbsoluteTolerance = 1.0e-4;
inline constexpr double kDefaultRelativeTolerance = 1.0e-3;

using ArtifactAdapterKind = InvocationAdapterKind;

struct ArtifactInfo {
    std::uint32_t format_version = 0;
    ArtifactAdapterKind adapter_kind =
        ArtifactAdapterKind::kConv2DRankedMemref;
    std::string deepforge_version;
    std::string llvm_version;
    std::string frontend_version;
    std::string target_triple;
    Conv2DCompileMetadata metadata;
    WorkspacePlan workspace;
    double absolute_tolerance = kDefaultAbsoluteTolerance;
    double relative_tolerance = kDefaultRelativeTolerance;
    std::array<VariantCode, 3> variants;
};

[[nodiscard]] import::Status serialize_artifact(
    CompilationResult const& compilation,
    std::vector<std::uint8_t>& output);

[[nodiscard]] import::Status parse_artifact(
    std::span<std::uint8_t const> input,
    ArtifactInfo& output);

[[nodiscard]] import::Status write_artifact(
    std::filesystem::path const& path,
    CompilationResult const& compilation);

[[nodiscard]] import::Status read_artifact(
    std::filesystem::path const& path,
    ArtifactInfo& output);

[[nodiscard]] import::Status load_artifact_executable(
    std::span<std::uint8_t const> input,
    std::unique_ptr<runtime::Executable>& output,
    ArtifactInfo* info = nullptr);

[[nodiscard]] import::Status load_artifact_executable(
    std::filesystem::path const& path,
    std::unique_ptr<runtime::Executable>& output,
    ArtifactInfo* info = nullptr);

}  // namespace deepforge::compiler
