#pragma once

#include "DeepForge/Compiler/Codegen.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

#include <string_view>

namespace deepforge::compiler {

[[nodiscard]] import::Status build_foundational_graph(
    ::mlir::MLIRContext& context,
    import::SerializedGraph const& graph,
    std::string_view function_name,
    ::mlir::OwningOpRef<::mlir::ModuleOp>& output,
    Conv2DCompileMetadata& metadata,
    WorkspacePlan& workspace);

}  // namespace deepforge::compiler
