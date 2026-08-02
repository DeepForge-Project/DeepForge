#include "MatmulGraph.h"

#include "DeepForge/Import/Capability.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "llvm/ADT/SmallVector.h"

#include <cstddef>
#include <string>
#include <utility>
#include <variant>

namespace deepforge::compiler {
namespace {

using import::DataType;
using import::ErrorCode;
using import::GenericOperationDesc;
using import::OperationTag;
using import::SerializedGraph;
using import::Status;
using import::TensorDesc;

Status fail(ErrorCode code, std::string subject, std::string detail) {
    std::string message(import::error_code_name(code));
    if (!subject.empty()) {
        message += ": ";
        message += std::move(subject);
    }
    if (!detail.empty()) {
        message += ": ";
        message += std::move(detail);
    }
    return Status::failure(code, std::move(message));
}

Status decode_override(GenericOperationDesc const& operation,
                       SerializedGraph const& graph,
                       TensorDesc const& output,
                       std::string_view path,
                       std::string_view port,
                       MatmulOverride& result) {
    auto const input = operation.inputs.find(std::string(port));
    if (input == operation.inputs.end()) return Status::ok();

    auto const* uid = std::get_if<std::int64_t>(&input->second);
    auto const subject = std::string(path) + ".inputs." + std::string(port);
    if (uid == nullptr) {
        return fail(ErrorCode::kUnsupportedOperation, subject,
                    "CPU execution requires an explicitly assigned UID");
    }
    auto const tensor = graph.tensors.find(*uid);
    if (tensor == graph.tensors.end()) {
        return fail(ErrorCode::kMissingUid, subject,
                    "override tensor reference is unresolved");
    }
    auto const& descriptor = tensor->second;
    if (descriptor.is_virtual || descriptor.is_pass_by_value ||
        descriptor.pass_by_value || descriptor.ragged_offset_uid ||
        descriptor.ragged_offset_name || descriptor.reordering_type != "NONE") {
        return fail(ErrorCode::kUnsupportedOperation, subject,
                    "MATMUL overrides require external plain tensors");
    }
    if (descriptor.data_type != DataType::kInt32) {
        return fail(ErrorCode::kUnsupportedDataType, subject,
                    "MATMUL overrides require INT32 elements");
    }
    if (descriptor.dim.size() != output.dim.size() ||
        descriptor.dim.size() < 2 ||
        descriptor.dim[descriptor.dim.size() - 2] != 1 ||
        descriptor.dim.back() != 1) {
        return fail(ErrorCode::kInvalidShape, subject,
                    "override rank must match C and its matrix dimensions must be 1");
    }
    for (std::size_t axis = 0; axis + 2 < output.dim.size(); ++axis) {
        if (descriptor.dim[axis] != 1 &&
            descriptor.dim[axis] != output.dim[axis]) {
            return fail(ErrorCode::kInvalidShape, subject,
                        "override batch dimensions must broadcast to C");
        }
    }
    result.uid = *uid;
    result.tensor = &descriptor;
    return Status::ok();
}

::mlir::Value index_constant(::mlir::OpBuilder& builder,
                             ::mlir::Location location,
                             std::int64_t value) {
    return ::mlir::arith::ConstantIndexOp::create(builder, location, value);
}

}  // namespace

bool is_matmul_override_input(OperationTag tag,
                              std::string_view port,
                              DataType data_type) noexcept {
    return (tag == OperationTag::kMatmul ||
            tag == OperationTag::kMatmulFp8) &&
           (port == "M_override" || port == "N_override" ||
            port == "K_override") &&
           data_type == DataType::kInt32;
}

Status decode_matmul_overrides(GenericOperationDesc const& operation,
                               SerializedGraph const& graph,
                               TensorDesc const& output,
                               std::string_view path,
                               MatmulOverrides& result) {
    MatmulOverrides candidate;
    auto status = decode_override(operation, graph, output, path,
                                  "M_override", candidate.m);
    if (status.is_bad()) return status;
    status = decode_override(operation, graph, output, path, "N_override",
                             candidate.n);
    if (status.is_bad()) return status;
    status = decode_override(operation, graph, output, path, "K_override",
                             candidate.k);
    if (status.is_bad()) return status;
    result = candidate;
    return Status::ok();
}

::mlir::Value matmul_override_index_is_active(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    MatmulOverride const& override,
    ::mlir::Value logical_index,
    llvm::ArrayRef<::mlir::Value> output_indices,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    if (!override.uid) {
        return ::mlir::arith::ConstantIntOp::create(builder, location, 1, 1);
    }

    llvm::SmallVector<::mlir::Value> indices;
    indices.reserve(override.tensor->dim.size());
    for (std::size_t axis = 0; axis + 2 < override.tensor->dim.size();
         ++axis) {
        indices.push_back(
            override.tensor->dim[axis] == 1
                ? index_constant(builder, location, 0)
                : output_indices[axis]);
    }
    indices.push_back(index_constant(builder, location, 0));
    indices.push_back(index_constant(builder, location, 0));
    auto extent_i32 = ::mlir::memref::LoadOp::create(
        builder, location, values.at(*override.uid), indices);
    auto extent = ::mlir::arith::IndexCastOp::create(
        builder, location, builder.getIndexType(), extent_i32);
    auto nonnegative = ::mlir::arith::CmpIOp::create(
        builder, location, ::mlir::arith::CmpIPredicate::sge, extent,
        index_constant(builder, location, 0));
    auto in_range = ::mlir::arith::CmpIOp::create(
        builder, location, ::mlir::arith::CmpIPredicate::slt,
        logical_index, extent);
    return ::mlir::arith::AndIOp::create(builder, location, nonnegative,
                                         in_range);
}

}  // namespace deepforge::compiler
