#pragma once

#include "DeepForge/Import/SerializedGraph.h"
#include "DeepForge/Import/Status.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/ArrayRef.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string_view>

namespace deepforge::compiler {

struct MatmulOverride {
    std::optional<std::int64_t> uid;
    import::TensorDesc const* tensor = nullptr;
};

struct MatmulOverrides {
    MatmulOverride m;
    MatmulOverride n;
    MatmulOverride k;
};

[[nodiscard]] bool is_matmul_override_input(
    import::OperationTag tag,
    std::string_view port,
    import::DataType data_type) noexcept;

[[nodiscard]] import::Status decode_matmul_overrides(
    import::GenericOperationDesc const& operation,
    import::SerializedGraph const& graph,
    import::TensorDesc const& output,
    std::string_view path,
    MatmulOverrides& result);

[[nodiscard]] ::mlir::Value matmul_override_index_is_active(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    MatmulOverride const& override,
    ::mlir::Value logical_index,
    llvm::ArrayRef<::mlir::Value> output_indices,
    std::map<std::int64_t, ::mlir::Value> const& values);

}  // namespace deepforge::compiler
