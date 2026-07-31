#include "SpecializedGraph.h"

#include "Numeric.h"
#include "SpecializedAttention.h"

#include "DeepForge/Import/Capability.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace deepforge::compiler {
namespace {

using import::DataType;
using import::ErrorCode;
using import::GenericOperationDesc;
using import::OperationTag;
using import::SerializedGraph;
using import::SerializedValue;
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

Status unsupported(std::string subject, std::string detail) {
    return fail(ErrorCode::kUnsupportedOperation, std::move(subject),
                std::move(detail));
}

SerializedValue const* attribute(GenericOperationDesc const& operation,
                                 std::string_view name) {
    auto const it = operation.attributes.find(std::string(name));
    return it == operation.attributes.end() ? nullptr : &it->second;
}

bool read_integer(SerializedValue const& value, std::int64_t& output) {
    if (auto const* integer = std::get_if<std::int64_t>(&value.value)) {
        output = *integer;
        return true;
    }
    if (auto const* integer = std::get_if<std::uint64_t>(&value.value);
        integer != nullptr &&
        *integer <= static_cast<std::uint64_t>(
                        std::numeric_limits<std::int64_t>::max())) {
        output = static_cast<std::int64_t>(*integer);
        return true;
    }
    return false;
}

bool read_string_attribute(GenericOperationDesc const& operation,
                           std::string_view name,
                           std::string_view& output) {
    auto const* value = attribute(operation, name);
    if (value == nullptr) return false;
    auto const* text = std::get_if<std::string>(&value->value);
    if (text == nullptr) return false;
    output = *text;
    return true;
}

bool read_bool_attribute(GenericOperationDesc const& operation,
                         std::string_view name,
                         bool& output) {
    auto const* value = attribute(operation, name);
    if (value == nullptr) return false;
    auto const* boolean = std::get_if<bool>(&value->value);
    if (boolean == nullptr) return false;
    output = *boolean;
    return true;
}

bool read_integer_attribute(GenericOperationDesc const& operation,
                            std::string_view name,
                            std::int64_t& output) {
    auto const* value = attribute(operation, name);
    return value != nullptr && read_integer(*value, output);
}

bool read_optional_integer_attribute(
    GenericOperationDesc const& operation,
    std::string_view name,
    std::optional<std::int64_t>& output) {
    auto const* value = attribute(operation, name);
    if (value == nullptr) return false;
    if (std::holds_alternative<std::nullptr_t>(value->value)) {
        output.reset();
        return true;
    }
    std::int64_t integer = 0;
    if (!read_integer(*value, integer)) return false;
    output = integer;
    return true;
}

bool read_integer_array(GenericOperationDesc const& operation,
                        std::string_view name,
                        std::vector<std::int64_t>& output) {
    auto const* value = attribute(operation, name);
    if (value == nullptr) return false;
    auto const* array = std::get_if<SerializedValue::Array>(&value->value);
    if (array == nullptr) return false;
    output.clear();
    output.reserve(array->size());
    for (auto const& item : *array) {
        std::int64_t integer = 0;
        if (!read_integer(item, integer)) return false;
        output.push_back(integer);
    }
    return true;
}

bool tensor_uid(GenericOperationDesc const& operation,
                bool input,
                std::string_view port,
                std::int64_t& uid) {
    auto const& ports = input ? operation.inputs : operation.outputs;
    auto const it = ports.find(std::string(port));
    if (it == ports.end()) return false;
    auto const* integer = std::get_if<std::int64_t>(&it->second);
    if (integer == nullptr) return false;
    uid = *integer;
    return true;
}

Status require_tensor(GenericOperationDesc const& operation,
                      SerializedGraph const& graph,
                      bool input,
                      std::string_view port,
                      std::string const& path,
                      std::int64_t& uid,
                      TensorDesc const*& tensor) {
    if (!tensor_uid(operation, input, port, uid)) {
        return fail(ErrorCode::kInvalidValue, path,
                    std::string(input ? "input " : "output ") +
                        std::string(port) +
                        " is required with an assigned UID");
    }
    auto const it = graph.tensors.find(uid);
    if (it == graph.tensors.end()) {
        return fail(ErrorCode::kMissingUid, path,
                    std::string(port) + " tensor is unresolved");
    }
    tensor = &it->second;
    return Status::ok();
}

bool has_type(DataType type, std::initializer_list<DataType> allowed) {
    return std::find(allowed.begin(), allowed.end(), type) != allowed.end();
}

Status require_type(TensorDesc const& tensor,
                    std::initializer_list<DataType> allowed,
                    std::string const& path) {
    if (has_type(tensor.data_type, allowed)) return Status::ok();
    return fail(ErrorCode::kUnsupportedDataType, path,
                std::string(import::data_type_name(tensor.data_type)) +
                    " is not supported on this port");
}

bool checked_element_count(std::vector<std::int64_t> const& dimensions,
                           std::uint64_t& output) {
    std::uint64_t count = 1;
    for (auto dimension : dimensions) {
        if (dimension <= 0 ||
            count > std::numeric_limits<std::uint64_t>::max() /
                        static_cast<std::uint64_t>(dimension)) {
            return false;
        }
        count *= static_cast<std::uint64_t>(dimension);
    }
    output = count;
    return true;
}

bool is_scalar(TensorDesc const& tensor) {
    std::uint64_t count = 0;
    return checked_element_count(tensor.dim, count) && count == 1;
}

Status require_float_compute(GenericOperationDesc const& operation,
                             std::string const& path) {
    std::string_view compute;
    if (!read_string_attribute(operation, "compute_data_type", compute) ||
        compute != "FLOAT") {
        return unsupported(path + ".compute_data_type",
                           "CPU specialized execution requires FLOAT");
    }
    return Status::ok();
}

Status validate_block_dequantize(GenericOperationDesc const& operation,
                                 SerializedGraph const& graph,
                                 std::string const& path) {
    if (operation.inputs.size() != 2 || operation.outputs.size() != 1) {
        return fail(ErrorCode::kInvalidValue, path,
                    "BLOCK_SCALE_DEQUANTIZE requires X, scale, and Y");
    }
    auto status = require_float_compute(operation, path);
    if (status.is_bad()) return status;
    std::int64_t x_uid = 0;
    std::int64_t scale_uid = 0;
    std::int64_t y_uid = 0;
    TensorDesc const* x = nullptr;
    TensorDesc const* scale = nullptr;
    TensorDesc const* y = nullptr;
    status = require_tensor(operation, graph, true, "X", path, x_uid, x);
    if (status.is_bad()) return status;
    status = require_tensor(operation, graph, true, "scale", path, scale_uid,
                            scale);
    if (status.is_bad()) return status;
    status = require_tensor(operation, graph, false, "Y", path, y_uid, y);
    if (status.is_bad()) return status;
    status = require_type(*x,
                          {DataType::kFp4E2M1, DataType::kFp8E4M3,
                           DataType::kFp8E5M2, DataType::kInt4},
                          path + ".X");
    if (status.is_bad()) return status;
    status = require_type(*scale,
                          {DataType::kFloat32, DataType::kFloat16,
                           DataType::kBFloat16, DataType::kFp8E4M3,
                           DataType::kFp8E8M0},
                          path + ".scale");
    if (status.is_bad()) return status;
    status = require_type(*y,
                          {DataType::kFloat32, DataType::kFloat16,
                           DataType::kBFloat16},
                          path + ".Y");
    if (status.is_bad()) return status;
    std::vector<std::int64_t> blocks;
    bool negative_scale = false;
    if (!read_integer_array(operation, "block_size", blocks) ||
        !read_bool_attribute(operation, "is_negative_scale",
                             negative_scale) ||
        blocks.empty() || blocks.size() > x->dim.size() ||
        std::any_of(blocks.begin(), blocks.end(),
                    [](std::int64_t value) { return value <= 0; })) {
        return fail(ErrorCode::kInvalidValue, path,
                    "block_size and is_negative_scale are malformed");
    }
    if (x->dim != y->dim || scale->dim.size() != x->dim.size()) {
        return fail(ErrorCode::kInvalidShape, path,
                    "Y must match X and scale must have the same rank");
    }
    auto expected = x->dim;
    auto const first = expected.size() - blocks.size();
    for (std::size_t index = 0; index < blocks.size(); ++index) {
        auto const axis = first + index;
        if (expected[axis] % blocks[index] != 0) {
            return fail(ErrorCode::kInvalidShape, path + ".block_size",
                        "blocked dimensions must be divisible by block_size");
        }
        expected[axis] /= blocks[index];
    }
    if (scale->dim != expected) {
        return fail(ErrorCode::kInvalidShape, path + ".scale",
                    "scale dimensions do not match the blocked X shape");
    }
    (void)negative_scale;
    return Status::ok();
}

Status validate_block_quantize(GenericOperationDesc const& operation,
                               SerializedGraph const& graph,
                               std::string const& path) {
    if (operation.inputs.size() != 1 || operation.outputs.size() != 2) {
        return fail(ErrorCode::kInvalidValue, path,
                    "BLOCK_SCALE_QUANTIZE requires X, Y, and scale");
    }
    auto status = require_float_compute(operation, path);
    if (status.is_bad()) return status;
    std::int64_t x_uid = 0;
    std::int64_t y_uid = 0;
    std::int64_t scale_uid = 0;
    TensorDesc const* x = nullptr;
    TensorDesc const* y = nullptr;
    TensorDesc const* scale = nullptr;
    status = require_tensor(operation, graph, true, "X", path, x_uid, x);
    if (status.is_bad()) return status;
    status = require_tensor(operation, graph, false, "Y", path, y_uid, y);
    if (status.is_bad()) return status;
    status = require_tensor(operation, graph, false, "scale", path, scale_uid,
                            scale);
    if (status.is_bad()) return status;
    status = require_type(*x,
                          {DataType::kFloat32, DataType::kFloat16,
                           DataType::kBFloat16},
                          path + ".X");
    if (status.is_bad()) return status;
    status = require_type(*y,
                          {DataType::kFp4E2M1, DataType::kFp8E4M3,
                           DataType::kFp8E5M2},
                          path + ".Y");
    if (status.is_bad()) return status;
    auto const expected_scale = y->data_type == DataType::kFp4E2M1
                                    ? DataType::kFp8E4M3
                                    : DataType::kFp8E8M0;
    if (scale->data_type != expected_scale) {
        return fail(ErrorCode::kUnsupportedDataType, path + ".scale",
                    "scale must be FP8_E4M3 for FP4 output and FP8_E8M0 "
                    "for FP8 output");
    }
    std::int64_t block = 0;
    std::optional<std::int64_t> axis;
    if (!read_integer_attribute(operation, "block_size", block) ||
        !read_optional_integer_attribute(operation, "axis", axis) ||
        block <= 0) {
        return fail(ErrorCode::kInvalidValue, path,
                    "block_size must be positive and axis must be integer/null");
    }
    auto const selected_axis = axis.value_or(
        static_cast<std::int64_t>(x->dim.size()) - 1);
    if (selected_axis < 0 ||
        selected_axis >= static_cast<std::int64_t>(x->dim.size()) ||
        x->dim[static_cast<std::size_t>(selected_axis)] % block != 0) {
        return fail(ErrorCode::kInvalidShape, path,
                    "axis is invalid or its dimension is not block divisible");
    }
    if (y->dim != x->dim || scale->dim.size() != x->dim.size()) {
        return fail(ErrorCode::kInvalidShape, path,
                    "Y must match X and scale must have the same rank");
    }
    auto expected = x->dim;
    expected[static_cast<std::size_t>(selected_axis)] /= block;
    if (scale->dim != expected) {
        return fail(ErrorCode::kInvalidShape, path + ".scale",
                    "scale dimensions do not match X/block_size");
    }
    return Status::ok();
}

Status validate_matmul_fp8(GenericOperationDesc const& operation,
                           SerializedGraph const& graph,
                           std::string const& path) {
    auto status = require_float_compute(operation, path);
    if (status.is_bad()) return status;
    for (auto port : {"M_override", "N_override", "K_override"}) {
        if (operation.inputs.contains(port)) {
            return unsupported(path + ".inputs." + port,
                               "runtime matmul overrides are implemented in C6");
        }
    }
    if (operation.inputs.size() != 5 || operation.outputs.size() != 2) {
        return fail(ErrorCode::kInvalidValue, path,
                    "MATMUL_FP8 requires A, B, two descales, Scale_C, C, "
                    "and Amax_C");
    }
    std::int64_t a_uid = 0;
    std::int64_t b_uid = 0;
    std::int64_t c_uid = 0;
    std::int64_t amax_uid = 0;
    TensorDesc const* a = nullptr;
    TensorDesc const* b = nullptr;
    TensorDesc const* c = nullptr;
    TensorDesc const* amax = nullptr;
    status = require_tensor(operation, graph, true, "A", path, a_uid, a);
    if (status.is_bad()) return status;
    status = require_tensor(operation, graph, true, "B", path, b_uid, b);
    if (status.is_bad()) return status;
    status = require_tensor(operation, graph, false, "C", path, c_uid, c);
    if (status.is_bad()) return status;
    status = require_tensor(operation, graph, false, "Amax_C", path,
                            amax_uid, amax);
    if (status.is_bad()) return status;
    status = require_type(*a, {DataType::kFp8E4M3, DataType::kFp8E5M2},
                          path + ".A");
    if (status.is_bad()) return status;
    status = require_type(*b, {DataType::kFp8E4M3, DataType::kFp8E5M2},
                          path + ".B");
    if (status.is_bad()) return status;
    status = require_type(*c,
                          {DataType::kFp8E4M3, DataType::kFp8E5M2,
                           DataType::kFloat32, DataType::kFloat16,
                           DataType::kBFloat16},
                          path + ".C");
    if (status.is_bad()) return status;
    status = require_type(*amax, {DataType::kFloat32}, path + ".Amax_C");
    if (status.is_bad()) return status;
    if (!is_scalar(*amax)) {
        return fail(ErrorCode::kInvalidShape, path + ".Amax_C",
                    "Amax_C must contain one FLOAT element");
    }
    for (auto port : {"Descale_A", "Descale_B", "Scale_C"}) {
        std::int64_t uid = 0;
        TensorDesc const* tensor = nullptr;
        status = require_tensor(operation, graph, true, port, path, uid,
                                tensor);
        if (status.is_bad()) return status;
        status = require_type(*tensor, {DataType::kFloat32},
                              path + "." + port);
        if (status.is_bad()) return status;
        if (!is_scalar(*tensor)) {
            return fail(ErrorCode::kInvalidShape, path + "." + port,
                        "scale tensors must contain one FLOAT element");
        }
    }
    if (a->dim.size() < 2 || a->dim.size() != b->dim.size() ||
        c->dim.size() != a->dim.size()) {
        return fail(ErrorCode::kInvalidShape, path,
                    "A, B, and C must have equal rank of at least two");
    }
    auto const rank = a->dim.size();
    if (a->dim[rank - 1] != b->dim[rank - 2] ||
        c->dim[rank - 2] != a->dim[rank - 2] ||
        c->dim[rank - 1] != b->dim[rank - 1]) {
        return fail(ErrorCode::kInvalidShape, path,
                    "matrix M, N, and K dimensions are inconsistent");
    }
    for (std::size_t axis_index = 0; axis_index + 2 < rank; ++axis_index) {
        if ((a->dim[axis_index] != 1 &&
             a->dim[axis_index] != c->dim[axis_index]) ||
            (b->dim[axis_index] != 1 &&
             b->dim[axis_index] != c->dim[axis_index])) {
            return fail(ErrorCode::kInvalidShape, path,
                        "batch dimensions are not broadcast-compatible");
        }
    }
    return Status::ok();
}

Status validate_moe_forward(GenericOperationDesc const& operation,
                            SerializedGraph const& graph,
                            std::string const& path) {
    std::string_view mode;
    std::int64_t top_k = 0;
    if (!read_string_attribute(operation, "mode", mode) || mode != "NONE" ||
        !read_integer_attribute(operation, "top_k", top_k) || top_k < 0 ||
        top_k > 1) {
        return unsupported(path,
                           "CPU MoE currently supports mode=NONE and top_k "
                           "0 or 1");
    }
    if (operation.inputs.size() != 3 || operation.outputs.size() != 1) {
        return fail(ErrorCode::kInvalidValue, path,
                    "mode=NONE requires Token, Weight, FirstTokenOffset, "
                    "and Output");
    }
    std::int64_t token_uid = 0;
    std::int64_t weight_uid = 0;
    std::int64_t offset_uid = 0;
    std::int64_t output_uid = 0;
    TensorDesc const* token = nullptr;
    TensorDesc const* weight = nullptr;
    TensorDesc const* offset = nullptr;
    TensorDesc const* output = nullptr;
    auto status = require_tensor(operation, graph, true, "Token", path,
                                 token_uid, token);
    if (status.is_bad()) return status;
    status = require_tensor(operation, graph, true, "Weight", path,
                            weight_uid, weight);
    if (status.is_bad()) return status;
    status = require_tensor(operation, graph, true, "FirstTokenOffset", path,
                            offset_uid, offset);
    if (status.is_bad()) return status;
    status = require_tensor(operation, graph, false, "Output", path,
                            output_uid, output);
    if (status.is_bad()) return status;
    for (auto const* tensor : {token, weight, output}) {
        status = require_type(*tensor,
                              {DataType::kFloat32, DataType::kFloat16,
                               DataType::kBFloat16},
                              path);
        if (status.is_bad()) return status;
    }
    status = require_type(*offset, {DataType::kInt32},
                          path + ".FirstTokenOffset");
    if (status.is_bad()) return status;
    if (token->data_type != weight->data_type ||
        output->data_type != token->data_type || token->dim.size() != 3 ||
        weight->dim.size() != 3 || output->dim.size() != 3 ||
        token->dim[0] != 1 || output->dim[0] != 1 ||
        output->dim[1] != token->dim[1] ||
        weight->dim[1] != token->dim[2] ||
        output->dim[2] != weight->dim[2] ||
        offset->dim != std::vector<std::int64_t>{weight->dim[0], 1, 1}) {
        return fail(ErrorCode::kInvalidShape, path,
                    "expected Token[1,T,K], Weight[E,K,N], offsets[E,1,1], "
                    "and Output[1,T,N] with one shared floating type");
    }
    return Status::ok();
}

Status validate_moe_backward(GenericOperationDesc const& operation,
                             SerializedGraph const& graph,
                             std::string const& path) {
    if (operation.inputs.size() != 3 || operation.outputs.size() != 1) {
        return fail(ErrorCode::kInvalidValue, path,
                    "MOE_GROUPED_MATMUL_BWD requires DOutput, Token, "
                    "FirstTokenOffset, and DWeight");
    }
    std::int64_t do_uid = 0;
    std::int64_t token_uid = 0;
    std::int64_t offset_uid = 0;
    std::int64_t dw_uid = 0;
    TensorDesc const* d_output = nullptr;
    TensorDesc const* token = nullptr;
    TensorDesc const* offset = nullptr;
    TensorDesc const* d_weight = nullptr;
    auto status = require_tensor(operation, graph, true, "DOutput", path,
                                 do_uid, d_output);
    if (status.is_bad()) return status;
    status = require_tensor(operation, graph, true, "Token", path, token_uid,
                            token);
    if (status.is_bad()) return status;
    status = require_tensor(operation, graph, true, "FirstTokenOffset", path,
                            offset_uid, offset);
    if (status.is_bad()) return status;
    status = require_tensor(operation, graph, false, "DWeight", path, dw_uid,
                            d_weight);
    if (status.is_bad()) return status;
    for (auto const* tensor : {d_output, token, d_weight}) {
        status = require_type(*tensor,
                              {DataType::kFloat32, DataType::kFloat16,
                               DataType::kBFloat16},
                              path);
        if (status.is_bad()) return status;
    }
    status = require_type(*offset, {DataType::kInt32},
                          path + ".FirstTokenOffset");
    if (status.is_bad()) return status;
    if (d_output->data_type != token->data_type ||
        d_weight->data_type != token->data_type ||
        d_output->dim.size() != 3 || token->dim.size() != 3 ||
        d_weight->dim.size() != 3 || d_output->dim[0] != 1 ||
        token->dim[0] != 1 || d_output->dim[1] != token->dim[1] ||
        d_weight->dim[1] != token->dim[2] ||
        d_weight->dim[2] != d_output->dim[2] ||
        offset->dim != std::vector<std::int64_t>{d_weight->dim[0], 1, 1}) {
        return fail(ErrorCode::kInvalidShape, path,
                    "expected dO[1,T,N], Token[1,T,K], offsets[E,1,1], "
                    "and dWeight[E,K,N] with one shared floating type");
    }
    return Status::ok();
}

::mlir::Value index_constant(::mlir::OpBuilder& builder,
                             ::mlir::Location location,
                             std::int64_t value) {
    return ::mlir::arith::ConstantIndexOp::create(builder, location, value);
}

::mlir::Value float_constant(::mlir::OpBuilder& builder,
                             ::mlir::Location location,
                             float value) {
    return ::mlir::arith::ConstantFloatOp::create(
        builder, location, ::mlir::Float32Type::get(builder.getContext()),
        llvm::APFloat(value));
}

llvm::SmallVector<::mlir::Value> logical_indices(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    ::mlir::Value linear,
    std::vector<std::int64_t> const& dimensions) {
    llvm::SmallVector<::mlir::Value> indices;
    indices.reserve(dimensions.size());
    std::vector<std::int64_t> divisors(dimensions.size(), 1);
    std::int64_t divisor = 1;
    for (std::size_t index = dimensions.size(); index > 1; --index) {
        divisor *= dimensions[index - 1];
        divisors[index - 2] = divisor;
    }
    for (std::size_t index = 0; index < dimensions.size(); ++index) {
        auto quotient = ::mlir::arith::DivUIOp::create(
            builder, location, linear,
            index_constant(builder, location, divisors[index]));
        indices.push_back(::mlir::arith::RemUIOp::create(
            builder, location, quotient,
            index_constant(builder, location, dimensions[index])));
    }
    return indices;
}

template <typename Body>
Status emit_flat_loop(::mlir::OpBuilder& builder,
                      ::mlir::Location location,
                      std::vector<std::int64_t> const& dimensions,
                      std::string_view name,
                      Body&& body) {
    std::uint64_t count = 0;
    if (!checked_element_count(dimensions, count) ||
        count > static_cast<std::uint64_t>(
                    std::numeric_limits<std::int64_t>::max())) {
        return fail(ErrorCode::kDimensionOverflow, std::string(name),
                    "element count does not fit an MLIR index");
    }
    auto loop = ::mlir::scf::buildLoopNest(
        builder, location,
        ::mlir::ValueRange{index_constant(builder, location, 0)},
        ::mlir::ValueRange{index_constant(
            builder, location, static_cast<std::int64_t>(count))},
        ::mlir::ValueRange{index_constant(builder, location, 1)},
        [&](::mlir::OpBuilder& body_builder, ::mlir::Location body_location,
            ::mlir::ValueRange induction_variables) {
            body(body_builder, body_location,
                 logical_indices(body_builder, body_location,
                                 induction_variables.front(), dimensions));
        });
    builder.setInsertionPointAfter(loop.loops.front());
    return Status::ok();
}

template <typename Body>
::mlir::Value reduce_extent(::mlir::OpBuilder& builder,
                            ::mlir::Location location,
                            std::int64_t extent,
                            ::mlir::Value initial,
                            Body&& body) {
    auto reduction = ::mlir::scf::buildLoopNest(
        builder, location,
        ::mlir::ValueRange{index_constant(builder, location, 0)},
        ::mlir::ValueRange{index_constant(builder, location, extent)},
        ::mlir::ValueRange{index_constant(builder, location, 1)},
        ::mlir::ValueRange{initial},
        [&](::mlir::OpBuilder& reduction_builder,
            ::mlir::Location reduction_location,
            ::mlir::ValueRange indices,
            ::mlir::ValueRange iter_args) -> ::mlir::scf::ValueVector {
            return {body(reduction_builder, reduction_location,
                         indices.front(), iter_args.front())};
        });
    builder.setInsertionPointAfter(reduction.loops.front());
    return reduction.results.front();
}

llvm::SmallVector<::mlir::Value> zero_indices(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    TensorDesc const& tensor) {
    return llvm::SmallVector<::mlir::Value>(
        tensor.dim.size(), index_constant(builder, location, 0));
}

::mlir::Value load_scalar(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    std::int64_t uid,
    TensorDesc const& tensor,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    return numeric::load_as_f32(builder, location, values.at(uid), tensor,
                                zero_indices(builder, location, tensor));
}

Status emit_block_dequantize(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    GenericOperationDesc const& operation,
    SerializedGraph const& graph,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    auto const x_uid = std::get<std::int64_t>(operation.inputs.at("X"));
    auto const scale_uid =
        std::get<std::int64_t>(operation.inputs.at("scale"));
    auto const y_uid = std::get<std::int64_t>(operation.outputs.at("Y"));
    auto const& x = graph.tensors.at(x_uid);
    auto const& scale = graph.tensors.at(scale_uid);
    auto const& y = graph.tensors.at(y_uid);
    std::vector<std::int64_t> blocks;
    (void)read_integer_array(operation, "block_size", blocks);
    return emit_flat_loop(
        builder, location, y.dim, "BLOCK_SCALE_DEQUANTIZE",
        [&](::mlir::OpBuilder& body_builder,
            ::mlir::Location body_location,
            llvm::SmallVector<::mlir::Value> const& indices) {
            auto scale_indices = indices;
            auto const first = indices.size() - blocks.size();
            for (std::size_t block_index = 0;
                 block_index < blocks.size(); ++block_index) {
                auto const axis = first + block_index;
                scale_indices[axis] = ::mlir::arith::DivUIOp::create(
                    body_builder, body_location, indices[axis],
                    index_constant(body_builder, body_location,
                                   blocks[block_index]));
            }
            auto x_value = numeric::load_as_f32(
                body_builder, body_location, values.at(x_uid), x, indices);
            auto scale_value = numeric::load_as_f32(
                body_builder, body_location, values.at(scale_uid), scale,
                scale_indices);
            auto result = ::mlir::arith::MulFOp::create(
                body_builder, body_location, x_value, scale_value);
            numeric::store_from_f32(body_builder, body_location, result,
                                    values.at(y_uid), y, indices);
        });
}

float quantized_maximum(DataType type) {
    switch (type) {
        case DataType::kFp4E2M1:
            return 6.0F;
        case DataType::kFp8E4M3:
            return 448.0F;
        case DataType::kFp8E5M2:
            return 57344.0F;
        default:
            return 1.0F;
    }
}

::mlir::Value block_scale_value(::mlir::OpBuilder& builder,
                                ::mlir::Location location,
                                ::mlir::Value amax,
                                DataType output_type,
                                DataType scale_type) {
    auto desired = ::mlir::arith::DivFOp::create(
        builder, location, amax,
        float_constant(builder, location, quantized_maximum(output_type)));
    auto positive = ::mlir::arith::CmpFOp::create(
        builder, location, ::mlir::arith::CmpFPredicate::OGT, desired,
        float_constant(builder, location, 0.0F));
    ::mlir::Value guarded = ::mlir::arith::SelectOp::create(
        builder, location, positive, desired,
        float_constant(builder, location, 1.0F));
    auto is_nan = ::mlir::arith::CmpFOp::create(
        builder, location, ::mlir::arith::CmpFPredicate::UNO, desired,
        desired);
    guarded = ::mlir::arith::SelectOp::create(builder, location, is_nan,
                                               desired, guarded);
    if (scale_type == DataType::kFp8E8M0) {
        auto exponent = ::mlir::math::CeilOp::create(
            builder, location,
            ::mlir::math::Log2Op::create(builder, location, guarded));
        guarded = ::mlir::math::PowFOp::create(
            builder, location, float_constant(builder, location, 2.0F),
            exponent);
    }
    return guarded;
}

Status emit_block_quantize(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    GenericOperationDesc const& operation,
    SerializedGraph const& graph,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    auto const x_uid = std::get<std::int64_t>(operation.inputs.at("X"));
    auto const y_uid = std::get<std::int64_t>(operation.outputs.at("Y"));
    auto const scale_uid =
        std::get<std::int64_t>(operation.outputs.at("scale"));
    auto const& x = graph.tensors.at(x_uid);
    auto const& y = graph.tensors.at(y_uid);
    auto const& scale = graph.tensors.at(scale_uid);
    std::int64_t block = 0;
    std::optional<std::int64_t> axis;
    (void)read_integer_attribute(operation, "block_size", block);
    (void)read_optional_integer_attribute(operation, "axis", axis);
    auto const selected_axis = static_cast<std::size_t>(
        axis.value_or(static_cast<std::int64_t>(x.dim.size()) - 1));

    auto status = emit_flat_loop(
        builder, location, scale.dim, "BLOCK_SCALE_QUANTIZE.scale",
        [&](::mlir::OpBuilder& body_builder,
            ::mlir::Location body_location,
            llvm::SmallVector<::mlir::Value> const& scale_indices) {
            auto amax = reduce_extent(
                body_builder, body_location, block,
                float_constant(body_builder, body_location, 0.0F),
                [&](::mlir::OpBuilder& reduction_builder,
                    ::mlir::Location reduction_location,
                    ::mlir::Value within_block,
                    ::mlir::Value accumulator) {
                    auto input_indices = scale_indices;
                    input_indices[selected_axis] =
                        ::mlir::arith::AddIOp::create(
                            reduction_builder, reduction_location,
                            ::mlir::arith::MulIOp::create(
                                reduction_builder, reduction_location,
                                scale_indices[selected_axis],
                                index_constant(reduction_builder,
                                               reduction_location, block)),
                            within_block);
                    auto value = numeric::load_as_f32(
                        reduction_builder, reduction_location,
                        values.at(x_uid), x, input_indices);
                    auto absolute = ::mlir::math::AbsFOp::create(
                        reduction_builder, reduction_location, value);
                    return ::mlir::arith::MaximumFOp::create(
                        reduction_builder, reduction_location, accumulator,
                        absolute);
                });
            auto scale_value = block_scale_value(
                body_builder, body_location, amax, y.data_type,
                scale.data_type);
            numeric::store_from_f32(body_builder, body_location, scale_value,
                                    values.at(scale_uid), scale,
                                    scale_indices);
        });
    if (status.is_bad()) return status;

    return emit_flat_loop(
        builder, location, y.dim, "BLOCK_SCALE_QUANTIZE.Y",
        [&](::mlir::OpBuilder& body_builder,
            ::mlir::Location body_location,
            llvm::SmallVector<::mlir::Value> const& indices) {
            auto scale_indices = indices;
            scale_indices[selected_axis] = ::mlir::arith::DivUIOp::create(
                body_builder, body_location, indices[selected_axis],
                index_constant(body_builder, body_location, block));
            auto input = numeric::load_as_f32(
                body_builder, body_location, values.at(x_uid), x, indices);
            auto scale_value = numeric::load_as_f32(
                body_builder, body_location, values.at(scale_uid), scale,
                scale_indices);
            auto output = ::mlir::arith::DivFOp::create(
                body_builder, body_location, input, scale_value);
            numeric::store_from_f32(body_builder, body_location, output,
                                    values.at(y_uid), y, indices);
        });
}

struct MatmulFp8Values {
    std::int64_t a_uid = 0;
    std::int64_t b_uid = 0;
    std::int64_t c_uid = 0;
    std::int64_t amax_uid = 0;
    TensorDesc const* a = nullptr;
    TensorDesc const* b = nullptr;
    TensorDesc const* c = nullptr;
    TensorDesc const* amax = nullptr;
    ::mlir::Value descale_a;
    ::mlir::Value descale_b;
    ::mlir::Value scale_c;
};

MatmulFp8Values prepare_matmul_fp8(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    GenericOperationDesc const& operation,
    SerializedGraph const& graph,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    MatmulFp8Values result;
    result.a_uid = std::get<std::int64_t>(operation.inputs.at("A"));
    result.b_uid = std::get<std::int64_t>(operation.inputs.at("B"));
    result.c_uid = std::get<std::int64_t>(operation.outputs.at("C"));
    result.amax_uid =
        std::get<std::int64_t>(operation.outputs.at("Amax_C"));
    result.a = &graph.tensors.at(result.a_uid);
    result.b = &graph.tensors.at(result.b_uid);
    result.c = &graph.tensors.at(result.c_uid);
    result.amax = &graph.tensors.at(result.amax_uid);
    auto scalar = [&](std::string_view port) {
        auto const uid =
            std::get<std::int64_t>(operation.inputs.at(std::string(port)));
        return load_scalar(builder, location, uid, graph.tensors.at(uid),
                           values);
    };
    result.descale_a = scalar("Descale_A");
    result.descale_b = scalar("Descale_B");
    result.scale_c = scalar("Scale_C");
    return result;
}

::mlir::Value compute_matmul_fp8_element(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    MatmulFp8Values const& matmul,
    llvm::SmallVector<::mlir::Value> const& output_indices,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    auto const rank = matmul.a->dim.size();
    auto result = reduce_extent(
        builder, location, matmul.a->dim.back(),
        float_constant(builder, location, 0.0F),
        [&](::mlir::OpBuilder& reduction_builder,
            ::mlir::Location reduction_location,
            ::mlir::Value k,
            ::mlir::Value accumulator) {
            auto a_indices = output_indices;
            auto b_indices = output_indices;
            for (std::size_t axis = 0; axis + 2 < rank; ++axis) {
                if (matmul.a->dim[axis] == 1) {
                    a_indices[axis] = index_constant(
                        reduction_builder, reduction_location, 0);
                }
                if (matmul.b->dim[axis] == 1) {
                    b_indices[axis] = index_constant(
                        reduction_builder, reduction_location, 0);
                }
            }
            a_indices[rank - 1] = k;
            b_indices[rank - 2] = k;
            auto a = numeric::load_as_f32(
                reduction_builder, reduction_location,
                values.at(matmul.a_uid), *matmul.a, a_indices);
            auto b = numeric::load_as_f32(
                reduction_builder, reduction_location,
                values.at(matmul.b_uid), *matmul.b, b_indices);
            return ::mlir::arith::AddFOp::create(
                reduction_builder, reduction_location, accumulator,
                ::mlir::arith::MulFOp::create(
                    reduction_builder, reduction_location, a, b));
        });
    result = ::mlir::arith::MulFOp::create(builder, location, result,
                                            matmul.descale_a);
    return ::mlir::arith::MulFOp::create(builder, location, result,
                                          matmul.descale_b);
}

Status emit_matmul_fp8(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    GenericOperationDesc const& operation,
    SerializedGraph const& graph,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    auto matmul =
        prepare_matmul_fp8(builder, location, operation, graph, values);
    std::uint64_t count = 0;
    (void)checked_element_count(matmul.c->dim, count);
    auto amax = reduce_extent(
        builder, location, static_cast<std::int64_t>(count),
        float_constant(builder, location, 0.0F),
        [&](::mlir::OpBuilder& reduction_builder,
            ::mlir::Location reduction_location,
            ::mlir::Value linear,
            ::mlir::Value accumulator) {
            auto indices = logical_indices(reduction_builder,
                                           reduction_location, linear,
                                           matmul.c->dim);
            auto value = compute_matmul_fp8_element(
                reduction_builder, reduction_location, matmul, indices,
                values);
            return ::mlir::arith::MaximumFOp::create(
                reduction_builder, reduction_location, accumulator,
                ::mlir::math::AbsFOp::create(
                    reduction_builder, reduction_location, value));
        });
    numeric::store_from_f32(builder, location, amax,
                            values.at(matmul.amax_uid), *matmul.amax,
                            zero_indices(builder, location, *matmul.amax));
    return emit_flat_loop(
        builder, location, matmul.c->dim, "MATMUL_FP8.C",
        [&](::mlir::OpBuilder& body_builder,
            ::mlir::Location body_location,
            llvm::SmallVector<::mlir::Value> const& indices) {
            auto result = compute_matmul_fp8_element(
                body_builder, body_location, matmul, indices, values);
            result = ::mlir::arith::MulFOp::create(
                body_builder, body_location, result, matmul.scale_c);
            numeric::store_from_f32(body_builder, body_location, result,
                                    values.at(matmul.c_uid), *matmul.c,
                                    indices);
        });
}

::mlir::Value expert_for_token(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    ::mlir::Value token,
    std::int64_t expert_count,
    ::mlir::Value offsets) {
    auto result = reduce_extent(
        builder, location, expert_count,
        index_constant(builder, location, 0),
        [&](::mlir::OpBuilder& reduction_builder,
            ::mlir::Location reduction_location,
            ::mlir::Value expert,
            ::mlir::Value selected) {
            llvm::SmallVector<::mlir::Value> indices{
                expert, index_constant(reduction_builder, reduction_location,
                                       0),
                index_constant(reduction_builder, reduction_location, 0)};
            auto start_i32 = ::mlir::memref::LoadOp::create(
                reduction_builder, reduction_location, offsets, indices);
            auto start = ::mlir::arith::IndexCastOp::create(
                reduction_builder, reduction_location,
                reduction_builder.getIndexType(), start_i32);
            auto applies = ::mlir::arith::CmpIOp::create(
                reduction_builder, reduction_location,
                ::mlir::arith::CmpIPredicate::uge, token, start);
            return ::mlir::arith::SelectOp::create(
                reduction_builder, reduction_location, applies, expert,
                selected);
        });
    return result;
}

Status emit_moe_forward(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    GenericOperationDesc const& operation,
    SerializedGraph const& graph,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    auto const token_uid =
        std::get<std::int64_t>(operation.inputs.at("Token"));
    auto const weight_uid =
        std::get<std::int64_t>(operation.inputs.at("Weight"));
    auto const offset_uid =
        std::get<std::int64_t>(operation.inputs.at("FirstTokenOffset"));
    auto const output_uid =
        std::get<std::int64_t>(operation.outputs.at("Output"));
    auto const& token = graph.tensors.at(token_uid);
    auto const& weight = graph.tensors.at(weight_uid);
    auto const& output = graph.tensors.at(output_uid);
    return emit_flat_loop(
        builder, location, output.dim, "MOE_GROUPED_MATMUL.Output",
        [&](::mlir::OpBuilder& body_builder,
            ::mlir::Location body_location,
            llvm::SmallVector<::mlir::Value> const& output_indices) {
            auto expert = expert_for_token(
                body_builder, body_location, output_indices[1],
                weight.dim[0], values.at(offset_uid));
            auto result = reduce_extent(
                body_builder, body_location, token.dim[2],
                float_constant(body_builder, body_location, 0.0F),
                [&](::mlir::OpBuilder& reduction_builder,
                    ::mlir::Location reduction_location,
                    ::mlir::Value k,
                    ::mlir::Value accumulator) {
                    llvm::SmallVector<::mlir::Value> token_indices{
                        index_constant(reduction_builder, reduction_location,
                                       0),
                        output_indices[1], k};
                    llvm::SmallVector<::mlir::Value> weight_indices{
                        expert, k, output_indices[2]};
                    auto token_value = numeric::load_as_f32(
                        reduction_builder, reduction_location,
                        values.at(token_uid), token, token_indices);
                    auto weight_value = numeric::load_as_f32(
                        reduction_builder, reduction_location,
                        values.at(weight_uid), weight, weight_indices);
                    return ::mlir::arith::AddFOp::create(
                        reduction_builder, reduction_location, accumulator,
                        ::mlir::arith::MulFOp::create(
                            reduction_builder, reduction_location,
                            token_value, weight_value));
                });
            numeric::store_from_f32(body_builder, body_location, result,
                                    values.at(output_uid), output,
                                    output_indices);
        });
}

Status emit_moe_backward(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    GenericOperationDesc const& operation,
    SerializedGraph const& graph,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    auto const do_uid =
        std::get<std::int64_t>(operation.inputs.at("DOutput"));
    auto const token_uid =
        std::get<std::int64_t>(operation.inputs.at("Token"));
    auto const offset_uid =
        std::get<std::int64_t>(operation.inputs.at("FirstTokenOffset"));
    auto const dw_uid =
        std::get<std::int64_t>(operation.outputs.at("DWeight"));
    auto const& d_output = graph.tensors.at(do_uid);
    auto const& token = graph.tensors.at(token_uid);
    auto const& d_weight = graph.tensors.at(dw_uid);
    return emit_flat_loop(
        builder, location, d_weight.dim, "MOE_GROUPED_MATMUL_BWD.DWeight",
        [&](::mlir::OpBuilder& body_builder,
            ::mlir::Location body_location,
            llvm::SmallVector<::mlir::Value> const& output_indices) {
            auto result = reduce_extent(
                body_builder, body_location, token.dim[1],
                float_constant(body_builder, body_location, 0.0F),
                [&](::mlir::OpBuilder& reduction_builder,
                    ::mlir::Location reduction_location,
                    ::mlir::Value token_index,
                    ::mlir::Value accumulator) {
                    auto token_expert = expert_for_token(
                        reduction_builder, reduction_location, token_index,
                        d_weight.dim[0], values.at(offset_uid));
                    auto selected = ::mlir::arith::CmpIOp::create(
                        reduction_builder, reduction_location,
                        ::mlir::arith::CmpIPredicate::eq, token_expert,
                        output_indices[0]);
                    llvm::SmallVector<::mlir::Value> token_indices{
                        index_constant(reduction_builder, reduction_location,
                                       0),
                        token_index, output_indices[1]};
                    llvm::SmallVector<::mlir::Value> do_indices{
                        index_constant(reduction_builder, reduction_location,
                                       0),
                        token_index, output_indices[2]};
                    auto token_value = numeric::load_as_f32(
                        reduction_builder, reduction_location,
                        values.at(token_uid), token, token_indices);
                    auto gradient = numeric::load_as_f32(
                        reduction_builder, reduction_location,
                        values.at(do_uid), d_output, do_indices);
                    ::mlir::Value contribution =
                        ::mlir::arith::MulFOp::create(
                        reduction_builder, reduction_location, token_value,
                        gradient);
                    contribution = ::mlir::arith::SelectOp::create(
                        reduction_builder, reduction_location, selected,
                        contribution,
                        float_constant(reduction_builder, reduction_location,
                                       0.0F));
                    return ::mlir::arith::AddFOp::create(
                        reduction_builder, reduction_location, accumulator,
                        contribution);
                });
            numeric::store_from_f32(body_builder, body_location, result,
                                    values.at(dw_uid), d_weight,
                                    output_indices);
        });
}

}  // namespace

bool is_specialized_operation(OperationTag tag) noexcept {
    switch (tag) {
        case OperationTag::kBlockScaleDequantize:
        case OperationTag::kBlockScaleQuantize:
        case OperationTag::kMatmulFp8:
        case OperationTag::kMoeGroupedMatmul:
        case OperationTag::kMoeGroupedMatmulBwd:
        case OperationTag::kSdpaFp8Bwd:
        case OperationTag::kSdpaFp8Fwd:
        case OperationTag::kSdpaMxfp8Bwd:
        case OperationTag::kSdpaMxfp8Fwd:
            return true;
        default:
            return false;
    }
}

Status validate_specialized_operation(OperationTag tag,
                                      GenericOperationDesc const& operation,
                                      SerializedGraph const& graph,
                                      std::size_t node_index) {
    auto const path = "nodes[" + std::to_string(node_index) + "]";
    switch (tag) {
        case OperationTag::kBlockScaleDequantize:
            return validate_block_dequantize(operation, graph, path);
        case OperationTag::kBlockScaleQuantize:
            return validate_block_quantize(operation, graph, path);
        case OperationTag::kMatmulFp8:
            return validate_matmul_fp8(operation, graph, path);
        case OperationTag::kMoeGroupedMatmul:
            return validate_moe_forward(operation, graph, path);
        case OperationTag::kMoeGroupedMatmulBwd:
            return validate_moe_backward(operation, graph, path);
        case OperationTag::kSdpaFp8Bwd:
        case OperationTag::kSdpaFp8Fwd:
        case OperationTag::kSdpaMxfp8Bwd:
        case OperationTag::kSdpaMxfp8Fwd:
            return validate_specialized_attention(tag, operation, graph,
                                                  node_index);
        default:
            return fail(ErrorCode::kInvalidValue, path,
                        "operation is not a C5 specialized operation");
    }
}

Status emit_specialized_operation(
    OperationTag tag,
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    GenericOperationDesc const& operation,
    SerializedGraph const& graph,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    switch (tag) {
        case OperationTag::kBlockScaleDequantize:
            return emit_block_dequantize(builder, location, operation, graph,
                                         values);
        case OperationTag::kBlockScaleQuantize:
            return emit_block_quantize(builder, location, operation, graph,
                                       values);
        case OperationTag::kMatmulFp8:
            return emit_matmul_fp8(builder, location, operation, graph,
                                  values);
        case OperationTag::kMoeGroupedMatmul:
            return emit_moe_forward(builder, location, operation, graph,
                                    values);
        case OperationTag::kMoeGroupedMatmulBwd:
            return emit_moe_backward(builder, location, operation, graph,
                                     values);
        case OperationTag::kSdpaFp8Bwd:
        case OperationTag::kSdpaFp8Fwd:
        case OperationTag::kSdpaMxfp8Bwd:
        case OperationTag::kSdpaMxfp8Fwd:
            return emit_specialized_attention(tag, builder, location,
                                              operation, graph, values);
        default:
            return fail(ErrorCode::kInvalidValue, "specialized",
                        "validated specialized operation has no emitter");
    }
}

}  // namespace deepforge::compiler
