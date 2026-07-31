#pragma once

#include "DeepForge/Import/SerializedGraph.h"
#include "DeepForge/Import/Status.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"

#include <cstddef>
#include <cstdint>
#include <map>

namespace deepforge::compiler {

[[nodiscard]] bool is_training_operation(import::OperationTag tag) noexcept;

[[nodiscard]] import::Status validate_training_operation(
    import::OperationTag tag,
    import::GenericOperationDesc const& operation,
    import::SerializedGraph const& graph,
    std::size_t node_index);

[[nodiscard]] import::Status emit_training_operation(
    import::OperationTag tag,
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    import::GenericOperationDesc const& operation,
    import::SerializedGraph const& graph,
    std::map<std::int64_t, ::mlir::Value> const& values);

}  // namespace deepforge::compiler
