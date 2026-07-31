#pragma once

#include "DeepForge/Compiler/Bufferization.h"
#include "DeepForge/Compiler/Conv2DImporter.h"
#include "DeepForge/Runtime/Executable.h"

#include "mlir/ExecutionEngine/ExecutionEngine.h"

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace llvm::orc {
class LLJIT;
}

namespace deepforge::runtime {

struct ExecutableFactory {
    [[nodiscard]] static import::Status create(
        compiler::Conv2DCompileMetadata const& metadata,
        compiler::WorkspacePlan const& workspace,
        std::vector<std::unique_ptr<::mlir::ExecutionEngine>> engines,
        std::array<std::string, 3> symbols,
        std::unique_ptr<Executable>& output);

    [[nodiscard]] static import::Status create_object(
        compiler::InvocationAdapterKind adapter_kind,
        compiler::Conv2DCompileMetadata const& metadata,
        compiler::WorkspacePlan const& workspace,
        std::array<std::unique_ptr<llvm::orc::LLJIT>, 3> object_jits,
        std::array<void*, 3> entry_points,
        std::array<std::string, 3> symbols,
        std::unique_ptr<Executable>& output);
};

[[nodiscard]] import::Status create_executable(
    compiler::Conv2DCompileMetadata const& metadata,
    compiler::WorkspacePlan const& workspace,
    std::vector<std::unique_ptr<::mlir::ExecutionEngine>> engines,
    std::array<std::string, 3> symbols,
    std::unique_ptr<Executable>& output);

[[nodiscard]] import::Status create_object_executable(
    compiler::InvocationAdapterKind adapter_kind,
    compiler::Conv2DCompileMetadata const& metadata,
    compiler::WorkspacePlan const& workspace,
    std::array<std::unique_ptr<llvm::orc::LLJIT>, 3> object_jits,
    std::array<void*, 3> entry_points,
    std::array<std::string, 3> symbols,
    std::unique_ptr<Executable>& output);

[[nodiscard]] import::Status load_object_executable(
    compiler::InvocationAdapterKind adapter_kind,
    compiler::Conv2DCompileMetadata const& metadata,
    compiler::WorkspacePlan const& workspace,
    std::array<std::span<std::uint8_t const>, 3> objects,
    std::array<std::string, 3> symbols,
    std::unique_ptr<Executable>& output);

}  // namespace deepforge::runtime
