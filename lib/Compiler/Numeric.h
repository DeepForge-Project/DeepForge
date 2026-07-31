#pragma once

#include "DeepForge/Import/SerializedGraph.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/SmallVector.h"

namespace deepforge::compiler::numeric {

[[nodiscard]] bool is_cpu_storage_type(import::DataType type) noexcept;
[[nodiscard]] bool is_packed_type(import::DataType type) noexcept;

[[nodiscard]] ::mlir::Type storage_element_type(
    ::mlir::MLIRContext& context,
    import::DataType type);

[[nodiscard]] ::mlir::Value load_as_f32(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    ::mlir::Value buffer,
    import::TensorDesc const& tensor,
    llvm::SmallVector<::mlir::Value> const& indices);

void store_from_f32(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    ::mlir::Value value,
    ::mlir::Value buffer,
    import::TensorDesc const& tensor,
    llvm::SmallVector<::mlir::Value> const& indices);

[[nodiscard]] ::mlir::Value decode_low_precision(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    ::mlir::Value code,
    import::DataType type);

[[nodiscard]] ::mlir::Value encode_low_precision(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    ::mlir::Value value,
    import::DataType type);

[[nodiscard]] ::mlir::Value quantize_f32(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    ::mlir::Value value,
    import::DataType type);

}  // namespace deepforge::compiler::numeric
