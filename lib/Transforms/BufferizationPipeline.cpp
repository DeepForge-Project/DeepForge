#include "DeepForge/Compiler/Bufferization.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace deepforge::compiler {
namespace {

using deepforge::import::ErrorCode;
using deepforge::import::Status;

struct AllocRecord {
    ::mlir::memref::AllocOp op;
    std::string name;
    std::uint64_t size_bytes = 0;
    std::uint64_t live_start = 0;
    std::uint64_t live_end = 0;
};

Status fail(ErrorCode code, std::string_view subject, std::string detail) {
    std::string message(deepforge::import::error_code_name(code));
    if (!subject.empty()) {
        message += ": ";
        message += subject;
    }
    if (!detail.empty()) {
        message += ": ";
        message += std::move(detail);
    }
    return Status::failure(code, std::move(message));
}

Status invalid_ir(std::string detail) {
    return fail(ErrorCode::kInvalidValue, "bufferization", std::move(detail));
}

bool checked_mul(std::uint64_t lhs,
                 std::uint64_t rhs,
                 std::uint64_t& result) {
    if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

bool is_static_f32_memref(::mlir::Type type,
                          std::array<std::int64_t, 4> const& shape) {
    auto memref = llvm::dyn_cast<::mlir::MemRefType>(type);
    if (!memref || !memref.hasStaticShape() || memref.getRank() != 4 ||
        memref.getElementType() !=
            ::mlir::Float32Type::get(type.getContext()) ||
        !memref.getLayout().isIdentity()) {
        return false;
    }
    return std::equal(memref.getShape().begin(), memref.getShape().end(),
                      shape.begin(), shape.end());
}

Status collect_allocations(::mlir::func::FuncOp function,
                           std::vector<AllocRecord>& records) {
    llvm::DenseMap<::mlir::Operation*, std::uint64_t> ordinal;
    std::uint64_t next_ordinal = 0;
    function.walk([&](::mlir::Operation* operation) {
        ordinal.try_emplace(operation, next_ordinal++);
    });

    std::uint64_t allocation_index = 0;
    auto walk_result = function.walk(
        [&](::mlir::memref::AllocOp allocation) -> ::mlir::WalkResult {
            auto type = allocation.getType();
            if (!type.hasStaticShape() || !type.getLayout().isIdentity() ||
                type.getMemorySpaceAsInt() != 0 ||
                !llvm::isa<::mlir::Float32Type>(type.getElementType())) {
                return ::mlir::WalkResult::interrupt();
            }
            std::uint64_t elements = 1;
            for (auto dimension : type.getShape()) {
                if (dimension <= 0 ||
                    !checked_mul(elements,
                                 static_cast<std::uint64_t>(dimension),
                                 elements)) {
                    return ::mlir::WalkResult::interrupt();
                }
            }
            std::uint64_t size_bytes = 0;
            if (!checked_mul(elements, sizeof(float), size_bytes) ||
                size_bytes > static_cast<std::uint64_t>(
                                 std::numeric_limits<std::int64_t>::max())) {
                return ::mlir::WalkResult::interrupt();
            }

            auto start_it = ordinal.find(allocation.getOperation());
            if (start_it == ordinal.end()) {
                return ::mlir::WalkResult::interrupt();
            }
            std::uint64_t last_use = start_it->second;
            for (auto& use : allocation.getResult().getUses()) {
                auto use_it = ordinal.find(use.getOwner());
                if (use_it == ordinal.end()) {
                    return ::mlir::WalkResult::interrupt();
                }
                last_use = std::max(last_use, use_it->second);
            }
            records.push_back(AllocRecord{allocation,
                                          "temporary_" +
                                              std::to_string(allocation_index++),
                                          size_bytes,
                                          start_it->second,
                                          last_use});
            return ::mlir::WalkResult::advance();
        });
    if (walk_result.wasInterrupted()) {
        return fail(ErrorCode::kDimensionOverflow, "workspace",
                    "encountered a dynamic, unsupported or overflowing allocation");
    }
    return Status::ok();
}

Status strip_equivalent_result(::mlir::func::FuncOp function) {
    if (!function.getBody().hasOneBlock()) {
        return invalid_ir("bufferized function must have one block");
    }
    auto returns = function.getOps<::mlir::func::ReturnOp>();
    if (std::distance(returns.begin(), returns.end()) != 1) {
        return invalid_ir("bufferized function must have one return");
    }
    auto return_op = *returns.begin();
    if (return_op.getNumOperands() == 1) {
        if (function.getNumArguments() < 3 ||
            return_op.getOperand(0) != function.getArgument(2)) {
            return invalid_ir("function result is not equivalent to Y destination");
        }
        return_op->setOperands({});
    } else if (return_op.getNumOperands() != 0) {
        return invalid_ir("bufferized function has an unexpected result count");
    }

    auto function_type = function.getFunctionType();
    function.setFunctionType(::mlir::FunctionType::get(
        function.getContext(), function_type.getInputs(), {}));
    return Status::ok();
}

Status rewrite_allocations(::mlir::func::FuncOp function,
                           std::vector<AllocRecord>& records,
                           WorkspacePlan const& plan) {
    if (records.size() != plan.allocations.size()) {
        return invalid_ir("workspace plan does not cover every allocation");
    }
    if (plan.size_bytes == 0) {
        if (!records.empty()) {
            return invalid_ir("non-empty allocation set has zero workspace size");
        }
        return Status::ok();
    }

    if (plan.size_bytes > static_cast<std::uint64_t>(
                                  std::numeric_limits<std::int64_t>::max())) {
        return fail(ErrorCode::kDimensionOverflow, "workspace",
                    "workspace type dimension does not fit int64");
    }

    auto* context = function.getContext();
    auto location = function.getLoc();
    auto workspace_type = ::mlir::MemRefType::get(
        {static_cast<std::int64_t>(plan.size_bytes)},
        ::mlir::IntegerType::get(context, 8));
    auto workspace = function.getBody().front().addArgument(workspace_type,
                                                             location);

    llvm::SmallVector<::mlir::Type> inputs(
        function.getFunctionType().getInputs());
    inputs.push_back(workspace_type);
    function.setFunctionType(
        ::mlir::FunctionType::get(context, inputs, {}));

    std::map<std::string, WorkspaceAllocation const*> planned_by_name;
    for (auto const& allocation : plan.allocations) {
        planned_by_name.try_emplace(allocation.name, &allocation);
    }

    for (auto& record : records) {
        auto plan_it = planned_by_name.find(record.name);
        if (plan_it == planned_by_name.end()) {
            return invalid_ir("workspace allocation name is missing from plan");
        }
        auto const& allocation = *plan_it->second;
        ::mlir::OpBuilder builder(record.op);
        auto byte_shift = ::mlir::arith::ConstantIndexOp::create(
            builder, record.op.getLoc(),
            static_cast<std::int64_t>(allocation.offset));
        auto view = ::mlir::memref::ViewOp::create(
            builder, record.op.getLoc(), record.op.getType(), workspace,
            byte_shift, ::mlir::ValueRange{});
        record.op.getResult().replaceAllUsesWith(view.getResult());
        record.op.erase();
    }

    llvm::SmallVector<::mlir::memref::DeallocOp> deallocations;
    function.walk([&](::mlir::memref::DeallocOp dealloc) {
        deallocations.push_back(dealloc);
    });
    for (auto dealloc : deallocations) {
        dealloc.erase();
    }
    return Status::ok();
}

Status run_one_shot_bufferize(::mlir::ModuleOp module) {
    ::mlir::DialectRegistry registry;
    ::mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
    ::mlir::bufferization::func_ext::
        registerBufferizableOpInterfaceExternalModels(registry);
    ::mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
    ::mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
    module.getContext()->appendDialectRegistry(registry);

    ::mlir::bufferization::OneShotBufferizePassOptions options;
    options.allowUnknownOps = false;
    options.bufferizeFunctionBoundaries = true;
    options.functionBoundaryTypeConversion =
        ::mlir::bufferization::LayoutMapOption::IdentityLayoutMap;
    options.bufferAlignment = kWorkspaceAlignment;

    ::mlir::PassManager manager(module.getContext());
    manager.enableVerifier(true);
    manager.addPass(
        ::mlir::bufferization::createOneShotBufferizePass(options));
    manager.addPass(
        ::mlir::bufferization::createDropEquivalentBufferResultsPass());
    manager.addPass(::mlir::createConvertBufferizationToMemRefPass());
    manager.addPass(::mlir::createCanonicalizerPass());
    if (llvm::failed(manager.run(module))) {
        return invalid_ir("One-Shot Bufferize pipeline failed");
    }
    return Status::ok();
}

Status verify_workspace_plan(WorkspacePlan const& workspace) {
    std::vector<WorkspaceRequest> requests;
    requests.reserve(workspace.allocations.size());
    for (auto const& allocation : workspace.allocations) {
        requests.push_back(WorkspaceRequest{allocation.name,
                                            allocation.size_bytes,
                                            allocation.alignment,
                                            allocation.live_start,
                                            allocation.live_end});
    }
    WorkspacePlan expected;
    auto status = plan_workspace(requests, expected);
    if (status.is_bad()) {
        return status;
    }
    if (expected != workspace) {
        return invalid_ir(
            "workspace metadata is not the canonical checked plan");
    }
    return Status::ok();
}

}  // namespace

Status bufferize_and_plan_conv2d(::mlir::ModuleOp imported_module,
                                 Conv2DCompileMetadata const& metadata,
                                 BufferizationResult& output) {
    if (!imported_module) {
        return fail(ErrorCode::kInvalidArgument, "module", "module is null");
    }
    if (metadata.function_name.empty()) {
        return fail(ErrorCode::kInvalidArgument, "metadata.function_name",
                    "function name must not be empty");
    }

    auto cloned_operation = imported_module->clone();
    auto cloned_module = llvm::dyn_cast<::mlir::ModuleOp>(cloned_operation);
    if (!cloned_module) {
        cloned_operation->erase();
        return invalid_ir("failed to clone builtin.module");
    }
    ::mlir::OwningOpRef<::mlir::ModuleOp> working(cloned_module);

    auto status = run_one_shot_bufferize(*working);
    if (status.is_bad()) {
        return status;
    }

    auto function = working->lookupSymbol<::mlir::func::FuncOp>(
        metadata.function_name);
    if (!function) {
        return invalid_ir("expected Conv2D function is absent after bufferization");
    }
    status = strip_equivalent_result(function);
    if (status.is_bad()) {
        return status;
    }

    std::vector<AllocRecord> records;
    status = collect_allocations(function, records);
    if (status.is_bad()) {
        return status;
    }
    std::vector<WorkspaceRequest> requests;
    requests.reserve(records.size());
    for (auto const& record : records) {
        requests.push_back(WorkspaceRequest{record.name,
                                            record.size_bytes,
                                            kWorkspaceAlignment,
                                            record.live_start,
                                            record.live_end});
    }
    WorkspacePlan plan;
    status = plan_workspace(requests, plan);
    if (status.is_bad()) {
        return status;
    }
    status = rewrite_allocations(function, records, plan);
    if (status.is_bad()) {
        return status;
    }
    if (llvm::failed(::mlir::verify(*working))) {
        return invalid_ir("workspace-rewritten module failed MLIR verification");
    }
    status = verify_bufferized_conv2d(*working, metadata, plan);
    if (status.is_bad()) {
        return status;
    }

    BufferizationResult result;
    result.module = std::move(working);
    result.workspace = std::move(plan);
    result.one_shot_bufferize_runs = 1;
    output = std::move(result);
    return Status::ok();
}

Status verify_bufferized_conv2d(::mlir::ModuleOp module,
                                Conv2DCompileMetadata const& metadata,
                                WorkspacePlan const& workspace) {
    if (!module) {
        return fail(ErrorCode::kInvalidArgument, "module", "module is null");
    }
    if (llvm::failed(::mlir::verify(module))) {
        return invalid_ir("module failed the standard MLIR verifier");
    }
    auto workspace_status = verify_workspace_plan(workspace);
    if (workspace_status.is_bad()) {
        return workspace_status;
    }
    auto function = module.lookupSymbol<::mlir::func::FuncOp>(
        metadata.function_name);
    if (!function || function.isExternal() ||
        !function.getBody().hasOneBlock()) {
        return invalid_ir("expected one defined Conv2D function");
    }
    auto type = function.getFunctionType();
    auto expected_inputs = workspace.size_bytes == 0 ? 3U : 4U;
    if (type.getNumInputs() != expected_inputs || type.getNumResults() != 0 ||
        !is_static_f32_memref(type.getInput(0), metadata.x_shape) ||
        !is_static_f32_memref(type.getInput(1), metadata.w_shape) ||
        !is_static_f32_memref(type.getInput(2), metadata.y_shape)) {
        return fail(ErrorCode::kInvalidShape, "bufferization.function_type",
                    "bufferized boundary does not match compile metadata");
    }
    if (workspace.size_bytes != 0) {
        auto workspace_type = llvm::dyn_cast<::mlir::MemRefType>(
            type.getInput(3));
        if (!workspace_type || workspace_type.getRank() != 1 ||
            !workspace_type.hasStaticShape() ||
            workspace_type.getDimSize(0) !=
                static_cast<std::int64_t>(workspace.size_bytes) ||
            !workspace_type.getLayout().isIdentity() ||
            !workspace_type.getElementType().isInteger(8)) {
            return fail(ErrorCode::kInvalidShape, "bufferization.workspace",
                        "workspace argument is not the planned flat i8 memref");
        }
    }

    bool has_tensor = false;
    bool has_bufferization = false;
    bool has_owned_allocation = false;
    std::vector<::mlir::memref::ViewOp> views;
    std::size_t conv_count = 0;
    function.walk([&](::mlir::Operation* operation) {
        if (operation->getName().getDialectNamespace() == "tensor") {
            has_tensor = true;
        }
        if (operation->getName().getDialectNamespace() == "bufferization") {
            has_bufferization = true;
        }
        if (llvm::isa<::mlir::memref::AllocOp,
                      ::mlir::memref::AllocaOp,
                      ::mlir::memref::DeallocOp>(operation)) {
            has_owned_allocation = true;
        }
        if (auto view = llvm::dyn_cast<::mlir::memref::ViewOp>(operation)) {
            views.push_back(view);
        }
        if (llvm::isa<::mlir::linalg::Conv2DNhwcFhwcOp>(operation)) {
            ++conv_count;
        }
    });
    if (has_tensor || has_bufferization || has_owned_allocation) {
        return invalid_ir("Tensor/Bufferization or owned allocation residue remains");
    }
    if (views.size() != workspace.allocations.size() || conv_count != 1) {
        return invalid_ir("workspace views or named Conv2D count is incorrect");
    }
    if (!views.empty()) {
        auto workspace_argument = function.getArgument(3);
        std::vector<bool> matched(workspace.allocations.size(), false);
        for (auto view : views) {
            auto view_type = view.getType();
            auto byte_shift =
                view.getByteShift().getDefiningOp<::mlir::arith::ConstantIndexOp>();
            if (view.getSource() != workspace_argument ||
                !view.getSizes().empty() || !view_type.hasStaticShape() ||
                !view_type.getLayout().isIdentity() ||
                view_type.getMemorySpaceAsInt() != 0 ||
                !llvm::isa<::mlir::Float32Type>(view_type.getElementType()) ||
                !byte_shift || byte_shift.value() < 0) {
                return invalid_ir(
                    "workspace view source, type or byte shift is invalid");
            }

            std::uint64_t view_size = sizeof(float);
            for (auto dimension : view_type.getShape()) {
                if (dimension <= 0 ||
                    !checked_mul(view_size,
                                 static_cast<std::uint64_t>(dimension),
                                 view_size)) {
                    return fail(ErrorCode::kDimensionOverflow,
                                "bufferization.workspace_view",
                                "workspace view byte size overflows");
                }
            }
            auto const offset =
                static_cast<std::uint64_t>(byte_shift.value());
            bool found = false;
            for (std::size_t index = 0;
                 index < workspace.allocations.size(); ++index) {
                auto const& allocation = workspace.allocations[index];
                if (!matched[index] && allocation.offset == offset &&
                    allocation.size_bytes == view_size) {
                    matched[index] = true;
                    found = true;
                    break;
                }
            }
            if (!found) {
                return invalid_ir(
                    "workspace view does not match a planned allocation");
            }
        }
    }

    auto return_op = llvm::dyn_cast<::mlir::func::ReturnOp>(
        &function.getBody().front().back());
    if (!return_op || return_op.getNumOperands() != 0) {
        return invalid_ir("bufferized function must return void");
    }
    return Status::ok();
}

}  // namespace deepforge::compiler
