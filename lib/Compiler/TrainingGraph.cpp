#include "TrainingGraph.h"
#include "NormalizationGraph.h"

#include "DeepForge/Import/Capability.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace deepforge::compiler {
namespace {

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

SerializedValue const* attribute(GenericOperationDesc const& operation,
                                 std::string_view name) {
    auto const it = operation.attributes.find(std::string(name));
    return it == operation.attributes.end() ? nullptr : &it->second;
}

bool read_string_attribute(GenericOperationDesc const& operation,
                           std::string_view name,
                           std::string_view& output) {
    auto const* value = attribute(operation, name);
    if (value == nullptr) {
        return false;
    }
    auto const* text = std::get_if<std::string>(&value->value);
    if (text == nullptr) {
        return false;
    }
    output = *text;
    return true;
}

bool read_integer(SerializedValue const& value, std::int64_t& output) {
    if (auto const* integer = std::get_if<std::int64_t>(&value.value)) {
        output = *integer;
        return true;
    }
    if (auto const* integer = std::get_if<std::uint64_t>(&value.value);
        integer != nullptr &&
        *integer <=
            static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
        output = static_cast<std::int64_t>(*integer);
        return true;
    }
    return false;
}

bool read_integer_array(GenericOperationDesc const& operation,
                        std::string_view name,
                        std::vector<std::int64_t>& output) {
    auto const* value = attribute(operation, name);
    if (value == nullptr) {
        return false;
    }
    auto const* array = std::get_if<SerializedValue::Array>(&value->value);
    if (array == nullptr) {
        return false;
    }
    output.clear();
    output.reserve(array->size());
    for (auto const& item : *array) {
        std::int64_t integer = 0;
        if (!read_integer(item, integer)) {
            return false;
        }
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
    if (it == ports.end()) {
        return false;
    }
    auto const* value = std::get_if<std::int64_t>(&it->second);
    if (value == nullptr) {
        return false;
    }
    uid = *value;
    return true;
}

struct ConvolutionDescription {
    std::int64_t x_uid = 0;
    std::int64_t w_uid = 0;
    std::int64_t y_uid = 0;
    TensorDesc const* x = nullptr;
    TensorDesc const* w = nullptr;
    TensorDesc const* y = nullptr;
    std::vector<std::int64_t> pre_padding;
    std::vector<std::int64_t> post_padding;
    std::vector<std::int64_t> stride;
    std::vector<std::int64_t> dilation;
    bool reverse_filter = false;
    std::int64_t groups = 0;
    std::int64_t output_channels_per_group = 0;
};

bool checked_add(std::int64_t lhs,
                 std::int64_t rhs,
                 std::int64_t& output) {
    if (rhs > 0 && lhs > std::numeric_limits<std::int64_t>::max() - rhs) {
        return false;
    }
    if (rhs < 0 && lhs < std::numeric_limits<std::int64_t>::min() - rhs) {
        return false;
    }
    output = lhs + rhs;
    return true;
}

bool checked_mul(std::int64_t lhs,
                 std::int64_t rhs,
                 std::int64_t& output) {
    if (lhs < 0 || rhs < 0) {
        return false;
    }
    if (lhs != 0 &&
        rhs > std::numeric_limits<std::int64_t>::max() / lhs) {
        return false;
    }
    output = lhs * rhs;
    return true;
}

Status decode_convolution(OperationTag tag,
                          GenericOperationDesc const& operation,
                          SerializedGraph const& graph,
                          std::string const& path,
                          ConvolutionDescription& output) {
    std::string_view x_port;
    std::string_view w_port;
    std::string_view y_port;
    bool x_is_input = true;
    bool w_is_input = true;
    bool y_is_input = false;
    switch (tag) {
        case OperationTag::kConvFprop:
            x_port = "X";
            w_port = "W";
            y_port = "Y";
            break;
        case OperationTag::kConvDgrad:
            x_port = "DX";
            w_port = "W";
            y_port = "DY";
            x_is_input = false;
            y_is_input = true;
            break;
        case OperationTag::kConvWgrad:
            x_port = "X";
            w_port = "DW";
            y_port = "DY";
            w_is_input = false;
            y_is_input = true;
            break;
        default:
            return fail(ErrorCode::kInvalidValue, path,
                        "operation is not a convolution tag");
    }

    if (operation.inputs.size() != 2 ||
        operation.outputs.size() != 1 || !operation.input_lists.empty() ||
        !tensor_uid(operation, x_is_input, x_port, output.x_uid) ||
        !tensor_uid(operation, w_is_input, w_port, output.w_uid) ||
        !tensor_uid(operation, y_is_input, y_port, output.y_uid)) {
        return fail(ErrorCode::kInvalidValue, path,
                    "convolution ports are incomplete or contain non-UID references");
    }
    auto const x_it = graph.tensors.find(output.x_uid);
    auto const w_it = graph.tensors.find(output.w_uid);
    auto const y_it = graph.tensors.find(output.y_uid);
    if (x_it == graph.tensors.end() || w_it == graph.tensors.end() ||
        y_it == graph.tensors.end()) {
        return fail(ErrorCode::kMissingUid, path,
                    "convolution tensor reference is unresolved");
    }
    output.x = &x_it->second;
    output.w = &w_it->second;
    output.y = &y_it->second;

    std::string_view compute_type;
    std::string_view math_mode;
    if (!read_string_attribute(operation, "compute_data_type", compute_type) ||
        compute_type != "FLOAT" ||
        !read_string_attribute(operation, "math_mode", math_mode) ||
        (math_mode != "CROSS_CORRELATION" &&
         math_mode != "CONVOLUTION") ||
        !read_integer_array(operation, "pre_padding", output.pre_padding) ||
        !read_integer_array(operation, "post_padding", output.post_padding) ||
        !read_integer_array(operation, "stride", output.stride) ||
        !read_integer_array(operation, "dilation", output.dilation)) {
        return fail(ErrorCode::kInvalidValue, path,
                    "convolution attributes are not executable f32 semantics");
    }
    output.reverse_filter = math_mode == "CONVOLUTION";

    auto const rank = output.x->dim.size();
    auto const spatial_rank = rank >= 2 ? rank - 2 : 0;
    if (rank < 3 || rank > 5 || output.w->dim.size() != rank ||
        output.y->dim.size() != rank ||
        output.pre_padding.size() != spatial_rank ||
        output.post_padding.size() != spatial_rank ||
        output.stride.size() != spatial_rank ||
        output.dilation.size() != spatial_rank) {
        return fail(ErrorCode::kInvalidShape, path,
                    "convolution requires matching rank 3-5 tensors and spatial attributes");
    }
    if (output.x->dim[0] != output.y->dim[0] ||
        output.w->dim[0] != output.y->dim[1] ||
        output.w->dim[1] <= 0 ||
        output.x->dim[1] % output.w->dim[1] != 0) {
        return fail(ErrorCode::kInvalidShape, path,
                    "convolution N/K/C dimensions are inconsistent");
    }
    output.groups = output.x->dim[1] / output.w->dim[1];
    if (output.groups <= 0 || output.y->dim[1] % output.groups != 0) {
        return fail(ErrorCode::kInvalidShape, path,
                    "convolution output channels are not divisible by groups");
    }
    output.output_channels_per_group = output.y->dim[1] / output.groups;

    for (std::size_t axis = 0; axis < spatial_rank; ++axis) {
        if (output.pre_padding[axis] < 0 || output.post_padding[axis] < 0 ||
            output.stride[axis] <= 0 || output.dilation[axis] <= 0) {
            return fail(ErrorCode::kInvalidValue, path,
                        "convolution padding must be non-negative and stride/dilation positive");
        }
        std::int64_t dilated_filter = 0;
        if (!checked_mul(output.dilation[axis],
                         output.w->dim[axis + 2] - 1,
                         dilated_filter) ||
            !checked_add(dilated_filter, 1, dilated_filter)) {
            return fail(ErrorCode::kDimensionOverflow, path,
                        "convolution effective filter extent overflows int64");
        }
        std::int64_t padded_input = 0;
        if (!checked_add(output.x->dim[axis + 2],
                         output.pre_padding[axis], padded_input) ||
            !checked_add(padded_input, output.post_padding[axis],
                         padded_input)) {
            return fail(ErrorCode::kDimensionOverflow, path,
                        "convolution padded input extent overflows int64");
        }
        if (padded_input < dilated_filter) {
            return fail(ErrorCode::kInvalidShape, path,
                        "convolution effective filter exceeds padded input");
        }
        auto const expected =
            1 + (padded_input - dilated_filter) / output.stride[axis];
        if (output.y->dim[axis + 2] != expected) {
            return fail(ErrorCode::kInvalidShape, path,
                        "convolution output spatial dimensions do not match attributes");
        }
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
    std::int64_t divisor = 1;
    std::vector<std::int64_t> divisors(dimensions.size(), 1);
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
                      Body&& body) {
    std::uint64_t count = 1;
    for (auto dimension : dimensions) {
        if (dimension <= 0 ||
            count > static_cast<std::uint64_t>(
                        std::numeric_limits<std::int64_t>::max()) /
                        static_cast<std::uint64_t>(dimension)) {
            return fail(ErrorCode::kDimensionOverflow, "convolution",
                        "output element count does not fit index");
        }
        count *= static_cast<std::uint64_t>(dimension);
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

::mlir::Value all_true(::mlir::OpBuilder& builder,
                       ::mlir::Location location) {
    return ::mlir::arith::ConstantIntOp::create(builder, location, 1, 1);
}

::mlir::Value and_condition(::mlir::OpBuilder& builder,
                            ::mlir::Location location,
                            ::mlir::Value lhs,
                            ::mlir::Value rhs) {
    return ::mlir::arith::AndIOp::create(builder, location, lhs, rhs);
}

::mlir::Value guarded_product(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    ::mlir::Value condition,
    ::mlir::Value lhs,
    llvm::SmallVector<::mlir::Value> const& lhs_indices,
    ::mlir::Value rhs,
    llvm::SmallVector<::mlir::Value> const& rhs_indices) {
    auto if_op = ::mlir::scf::IfOp::create(
        builder, location,
        ::mlir::TypeRange{::mlir::Float32Type::get(builder.getContext())},
        condition, true);
    builder.setInsertionPointToStart(if_op.thenBlock());
    auto lhs_value = ::mlir::memref::LoadOp::create(
        builder, location, lhs, lhs_indices);
    auto rhs_value = ::mlir::memref::LoadOp::create(
        builder, location, rhs, rhs_indices);
    ::mlir::scf::YieldOp::create(
        builder, location,
        ::mlir::ValueRange{::mlir::arith::MulFOp::create(
            builder, location, lhs_value, rhs_value)});
    builder.setInsertionPointToStart(if_op.elseBlock());
    ::mlir::scf::YieldOp::create(
        builder, location,
        ::mlir::ValueRange{float_constant(builder, location, 0.0F)});
    builder.setInsertionPointAfter(if_op);
    return if_op.getResult(0);
}

llvm::SmallVector<::mlir::Value> loop_lowers(::mlir::OpBuilder& builder,
                                              ::mlir::Location location,
                                              std::size_t count) {
    return llvm::SmallVector<::mlir::Value>(
        count, index_constant(builder, location, 0));
}

llvm::SmallVector<::mlir::Value> loop_steps(::mlir::OpBuilder& builder,
                                             ::mlir::Location location,
                                             std::size_t count) {
    return llvm::SmallVector<::mlir::Value>(
        count, index_constant(builder, location, 1));
}

::mlir::Value filter_coordinate(::mlir::OpBuilder& builder,
                                ::mlir::Location location,
                                ::mlir::Value index,
                                std::int64_t extent,
                                bool reverse) {
    if (!reverse) {
        return index;
    }
    return ::mlir::arith::SubIOp::create(
        builder, location, index_constant(builder, location, extent - 1),
        index);
}

Status emit_fprop(::mlir::OpBuilder& builder,
                  ::mlir::Location location,
                  ConvolutionDescription const& convolution,
                  std::map<std::int64_t, ::mlir::Value> const& values) {
    auto const spatial_rank = convolution.x->dim.size() - 2;
    return emit_flat_loop(
        builder, location, convolution.y->dim,
        [&](::mlir::OpBuilder& body_builder,
            ::mlir::Location body_location,
            llvm::SmallVector<::mlir::Value> const& y_indices) {
            auto lowers = loop_lowers(body_builder, body_location,
                                      spatial_rank + 1);
            llvm::SmallVector<::mlir::Value> uppers;
            uppers.push_back(index_constant(body_builder, body_location,
                                             convolution.w->dim[1]));
            for (std::size_t axis = 0; axis < spatial_rank; ++axis) {
                uppers.push_back(index_constant(
                    body_builder, body_location,
                    convolution.w->dim[axis + 2]));
            }
            auto steps = loop_steps(body_builder, body_location,
                                    spatial_rank + 1);
            auto reduction = ::mlir::scf::buildLoopNest(
                body_builder, body_location, lowers, uppers, steps,
                ::mlir::ValueRange{
                    float_constant(body_builder, body_location, 0.0F)},
                [&](::mlir::OpBuilder& reduction_builder,
                    ::mlir::Location reduction_location,
                    ::mlir::ValueRange reduction_indices,
                    ::mlir::ValueRange iter_args)
                    -> ::mlir::scf::ValueVector {
                    auto group = ::mlir::arith::DivUIOp::create(
                        reduction_builder, reduction_location, y_indices[1],
                        index_constant(reduction_builder, reduction_location,
                                       convolution.output_channels_per_group));
                    auto x_channel = ::mlir::arith::AddIOp::create(
                        reduction_builder, reduction_location,
                        ::mlir::arith::MulIOp::create(
                            reduction_builder, reduction_location, group,
                            index_constant(reduction_builder,
                                           reduction_location,
                                           convolution.w->dim[1])),
                        reduction_indices[0]);
                    llvm::SmallVector<::mlir::Value> x_indices{y_indices[0],
                                                               x_channel};
                    llvm::SmallVector<::mlir::Value> w_indices{y_indices[1],
                                                               reduction_indices[0]};
                    auto valid = all_true(reduction_builder,
                                          reduction_location);
                    for (std::size_t axis = 0; axis < spatial_rank; ++axis) {
                        auto actual_filter = reduction_indices[axis + 1];
                        auto effective_filter = filter_coordinate(
                            reduction_builder, reduction_location,
                            actual_filter,
                            convolution.w->dim[axis + 2],
                            convolution.reverse_filter);
                        auto coordinate = ::mlir::arith::SubIOp::create(
                            reduction_builder, reduction_location,
                            ::mlir::arith::AddIOp::create(
                                reduction_builder, reduction_location,
                                ::mlir::arith::MulIOp::create(
                                    reduction_builder, reduction_location,
                                    y_indices[axis + 2],
                                    index_constant(reduction_builder,
                                                   reduction_location,
                                                   convolution.stride[axis])),
                                ::mlir::arith::MulIOp::create(
                                    reduction_builder, reduction_location,
                                    effective_filter,
                                    index_constant(reduction_builder,
                                                   reduction_location,
                                                   convolution.dilation[axis]))),
                            index_constant(reduction_builder,
                                           reduction_location,
                                           convolution.pre_padding[axis]));
                        auto lower_valid = ::mlir::arith::CmpIOp::create(
                            reduction_builder, reduction_location,
                            ::mlir::arith::CmpIPredicate::sge, coordinate,
                            index_constant(reduction_builder,
                                           reduction_location, 0));
                        auto upper_valid = ::mlir::arith::CmpIOp::create(
                            reduction_builder, reduction_location,
                            ::mlir::arith::CmpIPredicate::slt, coordinate,
                            index_constant(reduction_builder,
                                           reduction_location,
                                           convolution.x->dim[axis + 2]));
                        valid = and_condition(
                            reduction_builder, reduction_location, valid,
                            and_condition(reduction_builder,
                                          reduction_location, lower_valid,
                                          upper_valid));
                        x_indices.push_back(coordinate);
                        w_indices.push_back(actual_filter);
                    }
                    auto product = guarded_product(
                        reduction_builder, reduction_location, valid,
                        values.at(convolution.x_uid), x_indices,
                        values.at(convolution.w_uid), w_indices);
                    return {::mlir::arith::AddFOp::create(
                        reduction_builder, reduction_location,
                        iter_args.front(), product)};
                });
            ::mlir::memref::StoreOp::create(
                body_builder, body_location, reduction.results.front(),
                values.at(convolution.y_uid), y_indices);
        });
}

Status emit_dgrad(::mlir::OpBuilder& builder,
                  ::mlir::Location location,
                  ConvolutionDescription const& convolution,
                  std::map<std::int64_t, ::mlir::Value> const& values) {
    auto const spatial_rank = convolution.x->dim.size() - 2;
    return emit_flat_loop(
        builder, location, convolution.x->dim,
        [&](::mlir::OpBuilder& body_builder,
            ::mlir::Location body_location,
            llvm::SmallVector<::mlir::Value> const& x_indices) {
            auto lowers = loop_lowers(body_builder, body_location,
                                      spatial_rank + 1);
            llvm::SmallVector<::mlir::Value> uppers;
            uppers.push_back(index_constant(
                body_builder, body_location,
                convolution.output_channels_per_group));
            for (std::size_t axis = 0; axis < spatial_rank; ++axis) {
                uppers.push_back(index_constant(
                    body_builder, body_location,
                    convolution.w->dim[axis + 2]));
            }
            auto steps = loop_steps(body_builder, body_location,
                                    spatial_rank + 1);
            auto reduction = ::mlir::scf::buildLoopNest(
                body_builder, body_location, lowers, uppers, steps,
                ::mlir::ValueRange{
                    float_constant(body_builder, body_location, 0.0F)},
                [&](::mlir::OpBuilder& reduction_builder,
                    ::mlir::Location reduction_location,
                    ::mlir::ValueRange reduction_indices,
                    ::mlir::ValueRange iter_args)
                    -> ::mlir::scf::ValueVector {
                    auto group = ::mlir::arith::DivUIOp::create(
                        reduction_builder, reduction_location, x_indices[1],
                        index_constant(reduction_builder, reduction_location,
                                       convolution.w->dim[1]));
                    auto local_channel = ::mlir::arith::RemUIOp::create(
                        reduction_builder, reduction_location, x_indices[1],
                        index_constant(reduction_builder, reduction_location,
                                       convolution.w->dim[1]));
                    auto y_channel = ::mlir::arith::AddIOp::create(
                        reduction_builder, reduction_location,
                        ::mlir::arith::MulIOp::create(
                            reduction_builder, reduction_location, group,
                            index_constant(reduction_builder,
                                           reduction_location,
                                           convolution.output_channels_per_group)),
                        reduction_indices[0]);
                    llvm::SmallVector<::mlir::Value> y_indices{x_indices[0],
                                                               y_channel};
                    llvm::SmallVector<::mlir::Value> w_indices{y_channel,
                                                               local_channel};
                    auto valid = all_true(reduction_builder,
                                          reduction_location);
                    for (std::size_t axis = 0; axis < spatial_rank; ++axis) {
                        auto actual_filter = reduction_indices[axis + 1];
                        auto effective_filter = filter_coordinate(
                            reduction_builder, reduction_location,
                            actual_filter,
                            convolution.w->dim[axis + 2],
                            convolution.reverse_filter);
                        auto numerator = ::mlir::arith::SubIOp::create(
                            reduction_builder, reduction_location,
                            ::mlir::arith::AddIOp::create(
                                reduction_builder, reduction_location,
                                x_indices[axis + 2],
                                index_constant(reduction_builder,
                                               reduction_location,
                                               convolution.pre_padding[axis])),
                            ::mlir::arith::MulIOp::create(
                                reduction_builder, reduction_location,
                                effective_filter,
                                index_constant(reduction_builder,
                                               reduction_location,
                                               convolution.dilation[axis])));
                        auto stride = index_constant(
                            reduction_builder, reduction_location,
                            convolution.stride[axis]);
                        auto coordinate = ::mlir::arith::DivSIOp::create(
                            reduction_builder, reduction_location, numerator,
                            stride);
                        auto nonnegative = ::mlir::arith::CmpIOp::create(
                            reduction_builder, reduction_location,
                            ::mlir::arith::CmpIPredicate::sge, numerator,
                            index_constant(reduction_builder,
                                           reduction_location, 0));
                        auto divisible = ::mlir::arith::CmpIOp::create(
                            reduction_builder, reduction_location,
                            ::mlir::arith::CmpIPredicate::eq,
                            ::mlir::arith::RemSIOp::create(
                                reduction_builder, reduction_location,
                                numerator, stride),
                            index_constant(reduction_builder,
                                           reduction_location, 0));
                        auto below_upper = ::mlir::arith::CmpIOp::create(
                            reduction_builder, reduction_location,
                            ::mlir::arith::CmpIPredicate::slt, coordinate,
                            index_constant(reduction_builder,
                                           reduction_location,
                                           convolution.y->dim[axis + 2]));
                        valid = and_condition(
                            reduction_builder, reduction_location, valid,
                            and_condition(
                                reduction_builder, reduction_location,
                                nonnegative,
                                and_condition(reduction_builder,
                                              reduction_location, divisible,
                                              below_upper)));
                        y_indices.push_back(coordinate);
                        w_indices.push_back(actual_filter);
                    }
                    auto product = guarded_product(
                        reduction_builder, reduction_location, valid,
                        values.at(convolution.y_uid), y_indices,
                        values.at(convolution.w_uid), w_indices);
                    return {::mlir::arith::AddFOp::create(
                        reduction_builder, reduction_location,
                        iter_args.front(), product)};
                });
            ::mlir::memref::StoreOp::create(
                body_builder, body_location, reduction.results.front(),
                values.at(convolution.x_uid), x_indices);
        });
}

Status emit_wgrad(::mlir::OpBuilder& builder,
                  ::mlir::Location location,
                  ConvolutionDescription const& convolution,
                  std::map<std::int64_t, ::mlir::Value> const& values) {
    auto const spatial_rank = convolution.x->dim.size() - 2;
    return emit_flat_loop(
        builder, location, convolution.w->dim,
        [&](::mlir::OpBuilder& body_builder,
            ::mlir::Location body_location,
            llvm::SmallVector<::mlir::Value> const& w_indices) {
            auto lowers = loop_lowers(body_builder, body_location,
                                      spatial_rank + 1);
            llvm::SmallVector<::mlir::Value> uppers;
            uppers.push_back(index_constant(body_builder, body_location,
                                             convolution.x->dim[0]));
            for (std::size_t axis = 0; axis < spatial_rank; ++axis) {
                uppers.push_back(index_constant(
                    body_builder, body_location,
                    convolution.y->dim[axis + 2]));
            }
            auto steps = loop_steps(body_builder, body_location,
                                    spatial_rank + 1);
            auto reduction = ::mlir::scf::buildLoopNest(
                body_builder, body_location, lowers, uppers, steps,
                ::mlir::ValueRange{
                    float_constant(body_builder, body_location, 0.0F)},
                [&](::mlir::OpBuilder& reduction_builder,
                    ::mlir::Location reduction_location,
                    ::mlir::ValueRange reduction_indices,
                    ::mlir::ValueRange iter_args)
                    -> ::mlir::scf::ValueVector {
                    auto group = ::mlir::arith::DivUIOp::create(
                        reduction_builder, reduction_location, w_indices[0],
                        index_constant(reduction_builder, reduction_location,
                                       convolution.output_channels_per_group));
                    auto x_channel = ::mlir::arith::AddIOp::create(
                        reduction_builder, reduction_location,
                        ::mlir::arith::MulIOp::create(
                            reduction_builder, reduction_location, group,
                            index_constant(reduction_builder,
                                           reduction_location,
                                           convolution.w->dim[1])),
                        w_indices[1]);
                    llvm::SmallVector<::mlir::Value> x_indices{
                        reduction_indices[0], x_channel};
                    llvm::SmallVector<::mlir::Value> y_indices{
                        reduction_indices[0], w_indices[0]};
                    auto valid = all_true(reduction_builder,
                                          reduction_location);
                    for (std::size_t axis = 0; axis < spatial_rank; ++axis) {
                        auto effective_filter = filter_coordinate(
                            reduction_builder, reduction_location,
                            w_indices[axis + 2],
                            convolution.w->dim[axis + 2],
                            convolution.reverse_filter);
                        auto coordinate = ::mlir::arith::SubIOp::create(
                            reduction_builder, reduction_location,
                            ::mlir::arith::AddIOp::create(
                                reduction_builder, reduction_location,
                                ::mlir::arith::MulIOp::create(
                                    reduction_builder, reduction_location,
                                    reduction_indices[axis + 1],
                                    index_constant(reduction_builder,
                                                   reduction_location,
                                                   convolution.stride[axis])),
                                ::mlir::arith::MulIOp::create(
                                    reduction_builder, reduction_location,
                                    effective_filter,
                                    index_constant(reduction_builder,
                                                   reduction_location,
                                                   convolution.dilation[axis]))),
                            index_constant(reduction_builder,
                                           reduction_location,
                                           convolution.pre_padding[axis]));
                        auto lower_valid = ::mlir::arith::CmpIOp::create(
                            reduction_builder, reduction_location,
                            ::mlir::arith::CmpIPredicate::sge, coordinate,
                            index_constant(reduction_builder,
                                           reduction_location, 0));
                        auto upper_valid = ::mlir::arith::CmpIOp::create(
                            reduction_builder, reduction_location,
                            ::mlir::arith::CmpIPredicate::slt, coordinate,
                            index_constant(reduction_builder,
                                           reduction_location,
                                           convolution.x->dim[axis + 2]));
                        valid = and_condition(
                            reduction_builder, reduction_location, valid,
                            and_condition(reduction_builder,
                                          reduction_location, lower_valid,
                                          upper_valid));
                        x_indices.push_back(coordinate);
                        y_indices.push_back(reduction_indices[axis + 1]);
                    }
                    auto product = guarded_product(
                        reduction_builder, reduction_location, valid,
                        values.at(convolution.x_uid), x_indices,
                        values.at(convolution.y_uid), y_indices);
                    return {::mlir::arith::AddFOp::create(
                        reduction_builder, reduction_location,
                        iter_args.front(), product)};
                });
            ::mlir::memref::StoreOp::create(
                body_builder, body_location, reduction.results.front(),
                values.at(convolution.w_uid), w_indices);
        });
}

}  // namespace

bool is_training_operation(OperationTag tag) noexcept {
    if (is_normalization_operation(tag)) {
        return true;
    }
    switch (tag) {
        case OperationTag::kConvFprop:
        case OperationTag::kConvDgrad:
        case OperationTag::kConvWgrad:
            return true;
        default:
            return false;
    }
}

Status validate_training_operation(OperationTag tag,
                                   GenericOperationDesc const& operation,
                                   SerializedGraph const& graph,
                                   std::size_t node_index) {
    if (is_normalization_operation(tag)) {
        return validate_normalization_operation(tag, operation, graph,
                                                node_index);
    }
    auto const path = "nodes[" + std::to_string(node_index) + "]";
    ConvolutionDescription convolution;
    return decode_convolution(tag, operation, graph, path, convolution);
}

Status emit_training_operation(
    OperationTag tag,
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    GenericOperationDesc const& operation,
    SerializedGraph const& graph,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    if (is_normalization_operation(tag)) {
        return emit_normalization_operation(tag, builder, location, operation,
                                            graph, values);
    }
    ConvolutionDescription convolution;
    auto status = decode_convolution(tag, operation, graph, "convolution",
                                     convolution);
    if (status.is_bad()) {
        return status;
    }
    switch (tag) {
        case OperationTag::kConvFprop:
            return emit_fprop(builder, location, convolution, values);
        case OperationTag::kConvDgrad:
            return emit_dgrad(builder, location, convolution, values);
        case OperationTag::kConvWgrad:
            return emit_wgrad(builder, location, convolution, values);
        default:
            return fail(ErrorCode::kInvalidValue, "convolution",
                        "validated training operation has no emitter");
    }
}

}  // namespace deepforge::compiler
