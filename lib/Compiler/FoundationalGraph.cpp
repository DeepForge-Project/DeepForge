#include "FoundationalGraph.h"

#include "DeepForge/Import/Capability.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
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
using import::TensorReference;

struct Usage {
    bool read = false;
    bool write = false;
    std::optional<std::uint64_t> producer;
    std::uint64_t last_consumer = 0;
};

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

Status reference_uid(TensorReference const& reference,
                     std::string const& path,
                     std::int64_t& uid) {
    auto const* value = std::get_if<std::int64_t>(&reference);
    if (value == nullptr) {
        return unsupported(path,
                           "CPU execution requires explicitly assigned UIDs");
    }
    uid = *value;
    return Status::ok();
}

std::string virtual_name(std::int64_t uid) {
    return "virtual_uid_" + std::to_string(uid);
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
    for (auto const& element : *array) {
        auto const* integer = std::get_if<std::int64_t>(&element.value);
        if (integer == nullptr) {
            return false;
        }
        output.push_back(*integer);
    }
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

bool read_integer_attribute(GenericOperationDesc const& operation,
                            std::string_view name,
                            std::int64_t& output) {
    auto const* value = attribute(operation, name);
    return value != nullptr && read_integer(*value, output);
}

bool read_optional_integer_attribute(GenericOperationDesc const& operation,
                                     std::string_view name,
                                     std::optional<std::int64_t>& output) {
    auto const* value = attribute(operation, name);
    if (value == nullptr) {
        return false;
    }
    if (std::holds_alternative<std::nullptr_t>(value->value)) {
        output.reset();
        return true;
    }
    std::int64_t integer = 0;
    if (!read_integer(*value, integer)) {
        return false;
    }
    output = integer;
    return true;
}

bool read_slice_array(
    GenericOperationDesc const& operation,
    std::string_view name,
    std::vector<std::array<std::int64_t, 2>>& output) {
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
    for (auto const& element : *array) {
        auto const* pair = std::get_if<SerializedValue::Array>(&element.value);
        if (pair == nullptr || pair->size() != 2) {
            return false;
        }
        std::array<std::int64_t, 2> bounds{};
        if (!read_integer((*pair)[0], bounds[0]) ||
            !read_integer((*pair)[1], bounds[1])) {
            return false;
        }
        output.push_back(bounds);
    }
    return true;
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

Status validate_tensor(TensorDesc const& tensor, std::string const& path) {
    if (tensor.data_type != import::DataType::kFloat32) {
        return fail(ErrorCode::kUnsupportedDataType, path,
                    "C2 execution requires FLOAT tensors");
    }
    if (!tensor.uid_assigned) {
        return unsupported(path, "CPU execution requires an assigned UID");
    }
    if (tensor.is_pass_by_value || tensor.pass_by_value) {
        return unsupported(path, "pass-by-value tensors are not in C2");
    }
    if (tensor.reordering_type != "NONE" || tensor.ragged_offset_uid ||
        tensor.ragged_offset_name) {
        return unsupported(path,
                           "reordered and ragged tensors are deferred to C6");
    }
    if (tensor.dim.empty() || tensor.dim.size() > 64 ||
        tensor.dim.size() != tensor.stride.size()) {
        return fail(ErrorCode::kInvalidShape, path,
                    "rank must be between 1 and 64");
    }
    return Status::ok();
}

Status validate_reshape(GenericOperationDesc const& operation,
                        SerializedGraph const& graph,
                        std::size_t node_index) {
    auto const path = "nodes[" + std::to_string(node_index) + "]";
    if (operation.inputs.size() != 1 || !operation.inputs.contains("X") ||
        operation.outputs.size() != 1 || !operation.outputs.contains("Y")) {
        return fail(ErrorCode::kInvalidValue, path,
                    "RESHAPE requires exactly X and Y");
    }
    auto const* x = graph.find_tensor(operation.inputs.at("X"));
    auto const* y = graph.find_tensor(operation.outputs.at("Y"));
    if (x == nullptr || y == nullptr) {
        return fail(ErrorCode::kMissingUid, path,
                    "RESHAPE tensor reference is unresolved");
    }
    std::string_view compute_type;
    std::string_view mode;
    std::vector<std::int64_t> dimensions;
    std::vector<std::int64_t> strides;
    if (!read_string_attribute(operation, "compute_data_type", compute_type) ||
        compute_type != "FLOAT" ||
        !read_string_attribute(operation, "reshape_mode", mode) ||
        (mode != "VIEW_ONLY" && mode != "LOGICAL") ||
        !read_integer_array(operation, "dim", dimensions) ||
        !read_integer_array(operation, "stride", strides)) {
        return fail(ErrorCode::kInvalidValue, path,
                    "RESHAPE attributes are not executable f32 semantics");
    }
    if (dimensions != y->dim || strides != y->stride) {
        return fail(ErrorCode::kInvalidShape, path,
                    "RESHAPE attributes must match the Y descriptor");
    }
    std::uint64_t x_elements = 0;
    std::uint64_t y_elements = 0;
    if (!checked_element_count(x->dim, x_elements) ||
        !checked_element_count(y->dim, y_elements) ||
        x_elements != y_elements ||
        x_elements >
            static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
        return fail(ErrorCode::kInvalidShape, path,
                    "RESHAPE X and Y element counts must match and fit int64");
    }
    return Status::ok();
}

Status validate_transpose(GenericOperationDesc const& operation,
                          SerializedGraph const& graph,
                          std::size_t node_index) {
    auto const path = "nodes[" + std::to_string(node_index) + "]";
    if (operation.inputs.size() != 1 || !operation.inputs.contains("X") ||
        operation.outputs.size() != 1 || !operation.outputs.contains("Y")) {
        return fail(ErrorCode::kInvalidValue, path,
                    "TRANSPOSE requires exactly X and Y");
    }
    auto const* x = graph.find_tensor(operation.inputs.at("X"));
    auto const* y = graph.find_tensor(operation.outputs.at("Y"));
    if (x == nullptr || y == nullptr) {
        return fail(ErrorCode::kMissingUid, path,
                    "TRANSPOSE tensor reference is unresolved");
    }
    std::string_view compute_type;
    std::vector<std::int64_t> permutation;
    if (!read_string_attribute(operation, "compute_data_type", compute_type) ||
        compute_type != "FLOAT" ||
        !read_integer_array(operation, "permutation", permutation)) {
        return fail(ErrorCode::kInvalidValue, path,
                    "TRANSPOSE attributes are not executable f32 semantics");
    }
    if (permutation.size() != x->dim.size() || y->dim.size() != x->dim.size()) {
        return fail(ErrorCode::kInvalidShape, path,
                    "TRANSPOSE ranks and permutation size must match");
    }
    std::vector<bool> seen(permutation.size(), false);
    for (std::size_t index = 0; index < permutation.size(); ++index) {
        auto source = permutation[index];
        if (source < 0 ||
            source >= static_cast<std::int64_t>(permutation.size()) ||
            seen[static_cast<std::size_t>(source)]) {
            return fail(ErrorCode::kInvalidShape, path,
                        "TRANSPOSE permutation must contain each axis once");
        }
        seen[static_cast<std::size_t>(source)] = true;
        if (y->dim[index] != x->dim[static_cast<std::size_t>(source)]) {
            return fail(ErrorCode::kInvalidShape, path,
                        "TRANSPOSE Y dimensions do not match permutation");
        }
    }
    return Status::ok();
}

Status validate_slice(GenericOperationDesc const& operation,
                      SerializedGraph const& graph,
                      std::size_t node_index) {
    auto const path = "nodes[" + std::to_string(node_index) + "]";
    if (operation.inputs.size() != 1 || !operation.inputs.contains("X") ||
        operation.outputs.size() != 1 || !operation.outputs.contains("Y")) {
        return fail(ErrorCode::kInvalidValue, path,
                    "SLICE requires exactly X and Y");
    }
    auto const* x = graph.find_tensor(operation.inputs.at("X"));
    auto const* y = graph.find_tensor(operation.outputs.at("Y"));
    if (x == nullptr || y == nullptr) {
        return fail(ErrorCode::kMissingUid, path,
                    "SLICE tensor reference is unresolved");
    }
    std::string_view compute_type;
    std::vector<std::array<std::int64_t, 2>> slices;
    std::vector<std::int64_t> strides;
    if (!read_string_attribute(operation, "compute_data_type", compute_type) ||
        compute_type != "FLOAT" ||
        !read_slice_array(operation, "slices", slices) ||
        !read_integer_array(operation, "slice_strides", strides)) {
        return fail(ErrorCode::kInvalidValue, path,
                    "SLICE attributes are not executable f32 semantics");
    }
    if (slices.size() != x->dim.size() || y->dim.size() != x->dim.size() ||
        strides.empty() || strides.size() > x->dim.size()) {
        return fail(ErrorCode::kInvalidShape, path,
                    "SLICE bounds and ranks must match X");
    }
    strides.resize(x->dim.size(), 1);
    for (std::size_t index = 0; index < x->dim.size(); ++index) {
        auto const start = slices[index][0];
        auto const limit = slices[index][1];
        auto const stride = strides[index];
        if (start < 0 || start >= limit || limit > x->dim[index] ||
            stride <= 0) {
            return fail(ErrorCode::kInvalidShape, path,
                        "SLICE bounds must be ordered, in range, and use positive strides");
        }
        auto const extent = 1 + (limit - start - 1) / stride;
        if (y->dim[index] != extent) {
            return fail(ErrorCode::kInvalidShape, path,
                        "SLICE Y dimensions do not match bounds and strides");
        }
    }
    return Status::ok();
}

Status validate_concatenate(GenericOperationDesc const& operation,
                            SerializedGraph const& graph,
                            std::size_t node_index) {
    auto const path = "nodes[" + std::to_string(node_index) + "]";
    if (operation.inputs.empty() || operation.outputs.size() != 1 ||
        !operation.outputs.contains("Y")) {
        return fail(ErrorCode::kInvalidValue, path,
                    "CONCATENATE requires indexed inputs and Y");
    }
    auto const* y = graph.find_tensor(operation.outputs.at("Y"));
    if (y == nullptr) {
        return fail(ErrorCode::kMissingUid, path,
                    "CONCATENATE Y reference is unresolved");
    }
    std::int64_t axis = 0;
    std::optional<std::int64_t> in_place_index;
    if (!read_integer_attribute(operation, "axis", axis) ||
        !read_optional_integer_attribute(operation, "in_place_index",
                                         in_place_index) ||
        axis < 0 || axis >= static_cast<std::int64_t>(y->dim.size())) {
        return fail(ErrorCode::kInvalidValue, path,
                    "CONCATENATE axis or in_place_index is invalid");
    }
    if (in_place_index &&
        (*in_place_index < 0 ||
         *in_place_index >= static_cast<std::int64_t>(operation.inputs.size()))) {
        return fail(ErrorCode::kInvalidValue, path,
                    "CONCATENATE in_place_index is outside the input list");
    }
    std::int64_t axis_extent = 0;
    for (std::size_t input_index = 0;
         input_index < operation.inputs.size(); ++input_index) {
        auto const port = std::to_string(input_index);
        auto const port_it = operation.inputs.find(port);
        auto const* input = port_it == operation.inputs.end()
                                ? nullptr
                                : graph.find_tensor(port_it->second);
        if (input == nullptr) {
            return fail(ErrorCode::kMissingUid, path,
                        "CONCATENATE indexed input is unresolved");
        }
        if (input->dim.size() != y->dim.size()) {
            return fail(ErrorCode::kInvalidShape, path,
                        "CONCATENATE input ranks must match Y");
        }
        for (std::size_t dimension = 0; dimension < y->dim.size();
             ++dimension) {
            if (dimension != static_cast<std::size_t>(axis) &&
                input->dim[dimension] != y->dim[dimension]) {
                return fail(ErrorCode::kInvalidShape, path,
                            "CONCATENATE non-axis dimensions must match");
            }
        }
        if (input->dim[static_cast<std::size_t>(axis)] >
            std::numeric_limits<std::int64_t>::max() - axis_extent) {
            return fail(ErrorCode::kDimensionOverflow, path,
                        "CONCATENATE axis extent overflows int64");
        }
        axis_extent += input->dim[static_cast<std::size_t>(axis)];
    }
    if (axis_extent != y->dim[static_cast<std::size_t>(axis)]) {
        return fail(ErrorCode::kInvalidShape, path,
                    "CONCATENATE Y axis extent is not the input sum");
    }
    return Status::ok();
}

Status validate_operation(OperationTag tag,
                          GenericOperationDesc const& operation,
                          SerializedGraph const& graph,
                          std::size_t node_index) {
    switch (tag) {
        case OperationTag::kReshape:
            return validate_reshape(operation, graph, node_index);
        case OperationTag::kTranspose:
            return validate_transpose(operation, graph, node_index);
        case OperationTag::kSlice:
            return validate_slice(operation, graph, node_index);
        case OperationTag::kConcatenate:
            return validate_concatenate(operation, graph, node_index);
        default:
            return unsupported(
                "nodes[" + std::to_string(node_index) + "]",
                std::string(import::operation_tag_name(tag)) +
                    " lowering is not implemented yet");
    }
}

Status analyze_graph(SerializedGraph const& graph,
                     std::string_view function_name,
                     Conv2DCompileMetadata& metadata,
                     WorkspacePlan& workspace) {
    if (function_name.empty() || function_name.find('\0') != std::string_view::npos) {
        return fail(ErrorCode::kInvalidArgument, "function_name",
                    "function name is empty or contains NUL");
    }
    if (graph.nodes.empty()) {
        return fail(ErrorCode::kInvalidValue, "nodes", "graph is empty");
    }
    if (!graph.named_tensors.empty()) {
        return unsupported("tensors",
                           "name-keyed execution is deferred until its public ABI is defined");
    }
    if ((graph.context.is_dynamic_shape_enabled &&
         *graph.context.is_dynamic_shape_enabled) ||
        (graph.context.is_override_shape_enabled &&
         *graph.context.is_override_shape_enabled)) {
        return unsupported("context", "C2 execution requires static shapes");
    }

    for (auto const& [uid, tensor] : graph.tensors) {
        auto status = validate_tensor(tensor, "tensors." + std::to_string(uid));
        if (status.is_bad()) {
            return status;
        }
    }

    std::map<std::int64_t, Usage> usage;
    for (std::size_t node_index = 0; node_index < graph.nodes.size();
         ++node_index) {
        auto const& node = graph.nodes[node_index];
        auto const* operation =
            std::get_if<GenericOperationDesc>(&node.attributes);
        if (operation == nullptr) {
            return unsupported("nodes[" + std::to_string(node_index) + "]",
                               "mixed Conv and foundational graphs are deferred to C3");
        }
        auto status = validate_operation(node.tag, *operation, graph,
                                         node_index);
        if (status.is_bad()) {
            return status;
        }
        for (auto const& [port, reference] : operation->inputs) {
            std::int64_t uid = 0;
            status = reference_uid(reference,
                                   "nodes[" + std::to_string(node_index) +
                                       "].inputs." + port,
                                   uid);
            if (status.is_bad()) {
                return status;
            }
            auto& info = usage[uid];
            info.read = true;
            info.last_consumer = std::max(
                info.last_consumer, static_cast<std::uint64_t>(node_index));
        }
        for (auto const& [port, reference] : operation->outputs) {
            std::int64_t uid = 0;
            status = reference_uid(reference,
                                   "nodes[" + std::to_string(node_index) +
                                       "].outputs." + port,
                                   uid);
            if (status.is_bad()) {
                return status;
            }
            auto& info = usage[uid];
            info.write = true;
            info.producer = static_cast<std::uint64_t>(node_index);
        }
    }

    Conv2DCompileMetadata candidate;
    candidate.function_name = function_name;
    for (auto const& [uid, info] : usage) {
        auto const tensor_it = graph.tensors.find(uid);
        if (tensor_it == graph.tensors.end()) {
            return fail(ErrorCode::kMissingUid, "metadata",
                        "used tensor is absent");
        }
        auto const& tensor = tensor_it->second;
        if (tensor.is_virtual) {
            continue;
        }
        TensorArgumentMetadata argument;
        argument.uid = uid;
        argument.name = tensor.name;
        argument.data_type = tensor.data_type;
        argument.dimensions = tensor.dim;
        argument.strides = tensor.stride;
        argument.alignment = alignof(float);
        argument.access = info.read && info.write
                              ? TensorAccess::kReadWrite
                              : (info.write ? TensorAccess::kWrite
                                            : TensorAccess::kRead);
        if (!import::tensor_storage_bytes(
                argument.data_type, argument.dimensions, argument.strides,
                argument.size_bytes)) {
            return fail(ErrorCode::kDimensionOverflow, tensor.name,
                        "external tensor byte range overflows uint64");
        }
        candidate.arguments.push_back(std::move(argument));
    }
    auto metadata_status = validate_graph_compile_metadata(candidate);
    if (metadata_status.is_bad()) {
        return metadata_status;
    }

    std::vector<WorkspaceRequest> requests;
    for (auto const& [uid, tensor] : graph.tensors) {
        if (!tensor.is_virtual) {
            continue;
        }
        auto const use = usage.find(uid);
        if (use == usage.end() || !use->second.producer) {
            return fail(ErrorCode::kInvalidValue, tensor.name,
                        "virtual tensor has no producer");
        }
        std::uint64_t size_bytes = 0;
        if (!import::tensor_storage_bytes(tensor.data_type, tensor.dim,
                                          tensor.stride, size_bytes)) {
            return fail(ErrorCode::kDimensionOverflow, tensor.name,
                        "virtual tensor byte range overflows uint64");
        }
        requests.push_back(WorkspaceRequest{
            virtual_name(uid), size_bytes, alignof(float),
            *use->second.producer,
            std::max(*use->second.producer, use->second.last_consumer)});
    }
    WorkspacePlan planned_workspace;
    auto status = plan_workspace(requests, planned_workspace);
    if (status.is_bad()) {
        return status;
    }
    metadata = std::move(candidate);
    workspace = std::move(planned_workspace);
    return Status::ok();
}

::mlir::MemRefType tensor_type(::mlir::MLIRContext& context,
                               TensorDesc const& tensor) {
    auto layout = ::mlir::StridedLayoutAttr::get(&context, 0, tensor.stride);
    return ::mlir::MemRefType::get(
        tensor.dim, ::mlir::Float32Type::get(&context), layout);
}

::mlir::Value index_constant(::mlir::OpBuilder& builder,
                             ::mlir::Location location,
                             std::int64_t value) {
    return ::mlir::arith::ConstantIndexOp::create(builder, location, value);
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
                      std::string_view operation_name,
                      Body&& body) {
    std::uint64_t element_count = 0;
    if (!checked_element_count(dimensions, element_count) ||
        element_count >
            static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
        return fail(ErrorCode::kDimensionOverflow,
                    std::string(operation_name),
                    "element count does not fit index");
    }
    auto zero = index_constant(builder, location, 0);
    auto upper = index_constant(builder, location,
                                static_cast<std::int64_t>(element_count));
    auto one = index_constant(builder, location, 1);
    auto loop_nest = ::mlir::scf::buildLoopNest(
        builder, location, ::mlir::ValueRange{zero},
        ::mlir::ValueRange{upper}, ::mlir::ValueRange{one},
        [&](::mlir::OpBuilder& body_builder, ::mlir::Location body_location,
            ::mlir::ValueRange induction_variables) {
            body(body_builder, body_location,
                 logical_indices(body_builder, body_location,
                                 induction_variables.front(), dimensions));
        });
    builder.setInsertionPointAfter(loop_nest.loops.front());
    return Status::ok();
}

Status emit_reshape(::mlir::OpBuilder& builder,
                    ::mlir::Location location,
                    GenericOperationDesc const& operation,
                    SerializedGraph const& graph,
                    std::map<std::int64_t, ::mlir::Value> const& values) {
    auto const x_uid = std::get<std::int64_t>(operation.inputs.at("X"));
    auto const y_uid = std::get<std::int64_t>(operation.outputs.at("Y"));
    auto const& x = graph.tensors.at(x_uid);
    auto const& y = graph.tensors.at(y_uid);
    return emit_flat_loop(
        builder, location, y.dim, "RESHAPE",
        [&](::mlir::OpBuilder& body_builder,
            ::mlir::Location body_location,
            llvm::SmallVector<::mlir::Value> const& y_indices) {
            auto linear = index_constant(body_builder, body_location, 0);
            for (std::size_t index = 0; index < y.dim.size(); ++index) {
                linear = ::mlir::arith::AddIOp::create(
                    body_builder, body_location,
                    ::mlir::arith::MulIOp::create(
                        body_builder, body_location, linear,
                        index_constant(body_builder, body_location,
                                       y.dim[index])),
                    y_indices[index]);
            }
            auto x_indices = logical_indices(body_builder, body_location,
                                             linear, x.dim);
            auto value = ::mlir::memref::LoadOp::create(
                body_builder, body_location, values.at(x_uid), x_indices);
            ::mlir::memref::StoreOp::create(body_builder, body_location, value,
                                            values.at(y_uid), y_indices);
        });
}

Status emit_transpose(::mlir::OpBuilder& builder,
                      ::mlir::Location location,
                      GenericOperationDesc const& operation,
                      SerializedGraph const& graph,
                      std::map<std::int64_t, ::mlir::Value> const& values) {
    auto const x_uid = std::get<std::int64_t>(operation.inputs.at("X"));
    auto const y_uid = std::get<std::int64_t>(operation.outputs.at("Y"));
    auto const& y = graph.tensors.at(y_uid);
    std::vector<std::int64_t> permutation;
    (void)read_integer_array(operation, "permutation", permutation);
    return emit_flat_loop(
        builder, location, y.dim, "TRANSPOSE",
        [&](::mlir::OpBuilder& body_builder,
            ::mlir::Location body_location,
            llvm::SmallVector<::mlir::Value> const& y_indices) {
            llvm::SmallVector<::mlir::Value> x_indices(y_indices.size());
            for (std::size_t index = 0; index < permutation.size(); ++index) {
                x_indices[static_cast<std::size_t>(permutation[index])] =
                    y_indices[index];
            }
            auto value = ::mlir::memref::LoadOp::create(
                body_builder, body_location, values.at(x_uid), x_indices);
            ::mlir::memref::StoreOp::create(body_builder, body_location, value,
                                            values.at(y_uid), y_indices);
        });
}

Status emit_slice(::mlir::OpBuilder& builder,
                  ::mlir::Location location,
                  GenericOperationDesc const& operation,
                  SerializedGraph const& graph,
                  std::map<std::int64_t, ::mlir::Value> const& values) {
    auto const x_uid = std::get<std::int64_t>(operation.inputs.at("X"));
    auto const y_uid = std::get<std::int64_t>(operation.outputs.at("Y"));
    auto const& y = graph.tensors.at(y_uid);
    std::vector<std::array<std::int64_t, 2>> slices;
    std::vector<std::int64_t> strides;
    (void)read_slice_array(operation, "slices", slices);
    (void)read_integer_array(operation, "slice_strides", strides);
    strides.resize(slices.size(), 1);
    return emit_flat_loop(
        builder, location, y.dim, "SLICE",
        [&](::mlir::OpBuilder& body_builder,
            ::mlir::Location body_location,
            llvm::SmallVector<::mlir::Value> const& y_indices) {
            llvm::SmallVector<::mlir::Value> x_indices;
            x_indices.reserve(y_indices.size());
            for (std::size_t index = 0; index < y_indices.size(); ++index) {
                auto scaled = ::mlir::arith::MulIOp::create(
                    body_builder, body_location, y_indices[index],
                    index_constant(body_builder, body_location,
                                   strides[index]));
                x_indices.push_back(::mlir::arith::AddIOp::create(
                    body_builder, body_location, scaled,
                    index_constant(body_builder, body_location,
                                   slices[index][0])));
            }
            auto value = ::mlir::memref::LoadOp::create(
                body_builder, body_location, values.at(x_uid), x_indices);
            ::mlir::memref::StoreOp::create(body_builder, body_location, value,
                                            values.at(y_uid), y_indices);
        });
}

Status emit_concatenate(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    GenericOperationDesc const& operation,
    SerializedGraph const& graph,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    auto const y_uid = std::get<std::int64_t>(operation.outputs.at("Y"));
    std::int64_t axis = 0;
    (void)read_integer_attribute(operation, "axis", axis);
    std::int64_t axis_offset = 0;
    for (std::size_t input_index = 0;
         input_index < operation.inputs.size(); ++input_index) {
        auto const uid = std::get<std::int64_t>(
            operation.inputs.at(std::to_string(input_index)));
        auto const& input = graph.tensors.at(uid);
        auto status = emit_flat_loop(
            builder, location, input.dim, "CONCATENATE",
            [&](::mlir::OpBuilder& body_builder,
                ::mlir::Location body_location,
                llvm::SmallVector<::mlir::Value> const& input_indices) {
                auto output_indices = input_indices;
                auto const axis_index = static_cast<std::size_t>(axis);
                output_indices[axis_index] = ::mlir::arith::AddIOp::create(
                    body_builder, body_location, input_indices[axis_index],
                    index_constant(body_builder, body_location, axis_offset));
                auto value = ::mlir::memref::LoadOp::create(
                    body_builder, body_location, values.at(uid), input_indices);
                ::mlir::memref::StoreOp::create(
                    body_builder, body_location, value, values.at(y_uid),
                    output_indices);
            });
        if (status.is_bad()) {
            return status;
        }
        axis_offset += input.dim[static_cast<std::size_t>(axis)];
    }
    return Status::ok();
}

Status emit_operation(
    OperationTag tag,
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    GenericOperationDesc const& operation,
    SerializedGraph const& graph,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    switch (tag) {
        case OperationTag::kReshape:
            return emit_reshape(builder, location, operation, graph, values);
        case OperationTag::kTranspose:
            return emit_transpose(builder, location, operation, graph, values);
        case OperationTag::kSlice:
            return emit_slice(builder, location, operation, graph, values);
        case OperationTag::kConcatenate:
            return emit_concatenate(builder, location, operation, graph,
                                    values);
        default:
            return fail(ErrorCode::kInvalidValue, "mlir",
                        "validated operation has no emitter");
    }
}

Status build_module(::mlir::MLIRContext& context,
                    SerializedGraph const& graph,
                    Conv2DCompileMetadata const& metadata,
                    WorkspacePlan const& workspace,
                    ::mlir::OwningOpRef<::mlir::ModuleOp>& output) {
    auto location = ::mlir::UnknownLoc::get(&context);
    auto module = ::mlir::ModuleOp::create(location);
    ::mlir::OpBuilder builder(&context);
    builder.setInsertionPointToStart(module.getBody());

    llvm::SmallVector<::mlir::Type> argument_types;
    argument_types.reserve(metadata.arguments.size() + 1);
    for (auto const& argument : metadata.arguments) {
        argument_types.push_back(tensor_type(context,
                                             graph.tensors.at(argument.uid)));
    }
    auto workspace_type = ::mlir::MemRefType::get(
        {::mlir::ShapedType::kDynamic}, ::mlir::IntegerType::get(&context, 8));
    argument_types.push_back(workspace_type);

    auto function_type = builder.getFunctionType(argument_types, {});
    auto function = ::mlir::func::FuncOp::create(
        builder, location, metadata.function_name, function_type);
    auto* entry = function.addEntryBlock();
    builder.setInsertionPointToStart(entry);

    std::map<std::int64_t, ::mlir::Value> values;
    for (std::size_t index = 0; index < metadata.arguments.size(); ++index) {
        values.emplace(metadata.arguments[index].uid, entry->getArgument(index));
    }
    auto workspace_value = entry->getArgument(metadata.arguments.size());
    for (auto const& [uid, tensor] : graph.tensors) {
        if (!tensor.is_virtual) {
            continue;
        }
        auto const allocation_name = virtual_name(uid);
        auto const allocation = std::find_if(
            workspace.allocations.begin(), workspace.allocations.end(),
            [&](WorkspaceAllocation const& item) {
                return item.name == allocation_name;
            });
        if (allocation == workspace.allocations.end() ||
            allocation->size_bytes % sizeof(float) != 0) {
            return fail(ErrorCode::kInvalidValue, allocation_name,
                        "workspace allocation is absent or misaligned");
        }
        auto storage_elements = static_cast<std::int64_t>(
            allocation->size_bytes / sizeof(float));
        auto storage_type = ::mlir::MemRefType::get(
            {storage_elements}, ::mlir::Float32Type::get(&context));
        auto byte_shift = index_constant(
            builder, location, static_cast<std::int64_t>(allocation->offset));
        auto storage = ::mlir::memref::ViewOp::create(
            builder, location, storage_type, workspace_value, byte_shift,
            ::mlir::ValueRange{});
        auto view = ::mlir::memref::ReinterpretCastOp::create(
            builder, location, tensor_type(context, tensor), storage, 0,
            tensor.dim, tensor.stride);
        values.emplace(uid, view);
    }

    for (auto const& node : graph.nodes) {
        auto const& operation = std::get<GenericOperationDesc>(node.attributes);
        auto status = emit_operation(node.tag, builder, location, operation,
                                     graph, values);
        if (status.is_bad()) {
            return status;
        }
    }
    ::mlir::func::ReturnOp::create(builder, location);
    if (::mlir::failed(::mlir::verify(module))) {
        return fail(ErrorCode::kInvalidValue, "mlir",
                    "foundational graph module failed verification");
    }
    output = ::mlir::OwningOpRef<::mlir::ModuleOp>(module);
    return Status::ok();
}

}  // namespace

Status build_foundational_graph(
    ::mlir::MLIRContext& context,
    import::SerializedGraph const& graph,
    std::string_view function_name,
    ::mlir::OwningOpRef<::mlir::ModuleOp>& output,
    Conv2DCompileMetadata& metadata,
    WorkspacePlan& workspace) {
    Conv2DCompileMetadata metadata_candidate;
    WorkspacePlan workspace_candidate;
    auto status = analyze_graph(graph, function_name, metadata_candidate,
                                workspace_candidate);
    if (status.is_bad()) {
        return status;
    }
    ::mlir::OwningOpRef<::mlir::ModuleOp> module;
    status = build_module(context, graph, metadata_candidate,
                          workspace_candidate, module);
    if (status.is_bad()) {
        return status;
    }
    output = std::move(module);
    metadata = std::move(metadata_candidate);
    workspace = std::move(workspace_candidate);
    return Status::ok();
}

}  // namespace deepforge::compiler
