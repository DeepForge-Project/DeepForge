#pragma once

#include "DeepForge/Compiler/Conv2DImporter.h"
#include "DeepForge/Compiler/Schedule.h"
#include "DeepForge/Import/Status.h"
#include "DeepForge/Runtime/Executable.h"

#include "mlir/IR/BuiltinOps.h"

namespace deepforge::compiler {

// Lower the named Conv2D and all remaining standard Linalg helpers to the
// target-independent loop/vector form selected by `variant`.
[[nodiscard]] import::Status lower_conv2d_variant(
    ::mlir::ModuleOp module,
    Conv2DCompileMetadata const& metadata,
    runtime::CpuVariant variant,
    Conv2DSchedule const& schedule);

// Convert the loop/vector form to LLVM dialect and verify that no source
// dialect remains.
[[nodiscard]] import::Status lower_to_llvm(
    ::mlir::ModuleOp module,
    Conv2DCompileMetadata const& metadata,
    runtime::CpuVariant variant);

}  // namespace deepforge::compiler
