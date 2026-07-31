#pragma once

#include "DeepForge/Compiler/GraphMetadata.h"
#include "DeepForge/Import/SerializedGraph.h"
#include "DeepForge/Import/Status.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

#include <array>
#include <cstdint>
#include <string>
#include <string_view>

namespace deepforge::compiler {

struct Conv2DImportOptions {
    std::string function_name = "deepforge_conv2d";
};

// Metadata consumed by later bufferization, workspace planning and runtime
// stages. It intentionally remains independent of MLIR ownership and types.
struct Conv2DCompileMetadata : GraphCompileMetadata {
    std::int64_t x_uid = 0;
    std::int64_t w_uid = 0;
    std::int64_t y_uid = 0;
    std::array<std::int64_t, 4> x_shape{};
    std::array<std::int64_t, 4> w_shape{};
    std::array<std::int64_t, 4> y_shape{};
    std::array<std::int64_t, 4> padded_x_shape{};
    std::array<std::int64_t, 2> pre_padding{};
    std::array<std::int64_t, 2> post_padding{};
    std::array<std::int64_t, 2> stride{};
    std::array<std::int64_t, 2> dilation{};

    bool operator==(Conv2DCompileMetadata const&) const = default;
};

// Build standard Tensor/Linalg IR for the validated f32 Conv2D subset. Output
// and optional metadata are updated only after the complete import succeeds.
[[nodiscard]] import::Status import_conv2d(
    ::mlir::MLIRContext& context,
    import::SerializedGraph const& graph,
    ::mlir::OwningOpRef<::mlir::ModuleOp>& output,
    Conv2DImportOptions const& options = {},
    Conv2DCompileMetadata* metadata = nullptr);

// Independently verify the generated module against the canonical model.
[[nodiscard]] import::Status verify_conv2d_module(
    ::mlir::ModuleOp module,
    import::SerializedGraph const& graph,
    std::string_view function_name = "deepforge_conv2d");

}  // namespace deepforge::compiler
