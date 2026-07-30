#pragma once

#include "DeepForge/Compiler/Conv2DImporter.h"
#include "DeepForge/Import/Status.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace deepforge::compiler {

inline constexpr std::uint64_t kWorkspaceAlignment = 64;

struct WorkspaceRequest {
    std::string name;
    std::uint64_t size_bytes = 0;
    std::uint64_t alignment = kWorkspaceAlignment;
    std::uint64_t live_start = 0;
    std::uint64_t live_end = 0;
};

struct WorkspaceAllocation {
    std::string name;
    std::uint64_t offset = 0;
    std::uint64_t size_bytes = 0;
    std::uint64_t alignment = kWorkspaceAlignment;
    std::uint64_t live_start = 0;
    std::uint64_t live_end = 0;

    bool operator==(WorkspaceAllocation const&) const = default;
};

struct WorkspacePlan {
    std::uint64_t size_bytes = 0;
    std::uint64_t alignment = kWorkspaceAlignment;
    std::vector<WorkspaceAllocation> allocations;

    bool operator==(WorkspacePlan const&) const = default;
};

struct BufferizationResult {
    ::mlir::OwningOpRef<::mlir::ModuleOp> module;
    WorkspacePlan workspace;
    std::uint32_t one_shot_bufferize_runs = 0;
};

// Plan statically-sized temporary buffers. Requests whose lifetimes do not
// overlap may share the same byte range.
[[nodiscard]] import::Status plan_workspace(
    std::span<WorkspaceRequest const> requests,
    WorkspacePlan& output);

// Clone, bufferize and workspace-plan a validated P2 Conv2D module. Output is
// updated only after the complete transform and independent verification pass.
[[nodiscard]] import::Status bufferize_and_plan_conv2d(
    ::mlir::ModuleOp imported_module,
    Conv2DCompileMetadata const& metadata,
    BufferizationResult& output);

// Verify the P3 module and workspace metadata without trusting the transform.
[[nodiscard]] import::Status verify_bufferized_conv2d(
    ::mlir::ModuleOp module,
    Conv2DCompileMetadata const& metadata,
    WorkspacePlan const& workspace);

}  // namespace deepforge::compiler
