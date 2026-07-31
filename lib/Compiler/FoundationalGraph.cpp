#include "FoundationalGraph.h"

#include "DeepForge/Import/Capability.h"
#include "DeepForge/Import/Schema.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <array>
#include <cmath>
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

bool read_optional_number_attribute(GenericOperationDesc const& operation,
                                    std::string_view name,
                                    std::optional<double>& output) {
    auto const* value = attribute(operation, name);
    if (value == nullptr) {
        return false;
    }
    if (std::holds_alternative<std::nullptr_t>(value->value)) {
        output.reset();
        return true;
    }
    if (auto const* number = std::get_if<double>(&value->value)) {
        output = *number;
        return true;
    }
    std::int64_t integer = 0;
    if (read_integer(*value, integer)) {
        output = static_cast<double>(integer);
        return true;
    }
    return false;
}

bool read_bool_attribute(GenericOperationDesc const& operation,
                         std::string_view name,
                         bool& output) {
    auto const* value = attribute(operation, name);
    if (value == nullptr) {
        return false;
    }
    auto const* boolean = std::get_if<bool>(&value->value);
    if (boolean == nullptr) {
        return false;
    }
    output = *boolean;
    return true;
}

bool read_optional_bool_attribute(GenericOperationDesc const& operation,
                                  std::string_view name,
                                  std::optional<bool>& output) {
    auto const* value = attribute(operation, name);
    if (value == nullptr) {
        return false;
    }
    if (std::holds_alternative<std::nullptr_t>(value->value)) {
        output.reset();
        return true;
    }
    auto const* boolean = std::get_if<bool>(&value->value);
    if (boolean == nullptr) {
        return false;
    }
    output = *boolean;
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

bool has_non_overlapping_layout(TensorDesc const& tensor) {
    std::vector<std::pair<std::uint64_t, std::uint64_t>> axes;
    axes.reserve(tensor.dim.size());
    for (std::size_t index = 0; index < tensor.dim.size(); ++index) {
        if (tensor.dim[index] <= 0 || tensor.stride[index] <= 0) {
            return false;
        }
        if (tensor.dim[index] > 1) {
            axes.emplace_back(static_cast<std::uint64_t>(tensor.stride[index]),
                              static_cast<std::uint64_t>(tensor.dim[index]));
        }
    }
    std::sort(axes.begin(), axes.end());
    std::uint64_t occupied_span = 1;
    for (auto const& [stride, extent] : axes) {
        if (stride < occupied_span ||
            extent - 1 >
                (std::numeric_limits<std::uint64_t>::max() - occupied_span) /
                    stride) {
            return false;
        }
        occupied_span += (extent - 1) * stride;
    }
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
    if (!has_non_overlapping_layout(tensor)) {
        return fail(ErrorCode::kInvalidLayout, path,
                    "C2 execution requires a non-overlapping strided layout");
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
        mode != "LOGICAL" ||
        !read_integer_array(operation, "dim", dimensions) ||
        !read_integer_array(operation, "stride", strides)) {
        if (mode == "VIEW_ONLY") {
            return unsupported(path,
                               "RESHAPE VIEW_ONLY aliasing is deferred to C6");
        }
        return fail(ErrorCode::kInvalidValue, path,
                    "RESHAPE attributes are not executable f32 LOGICAL semantics");
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
        strides.size() > x->dim.size()) {
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
    if (in_place_index) {
        return unsupported(path,
                           "CONCATENATE in-place aliasing is deferred to C6");
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

Status validate_pointwise(GenericOperationDesc const& operation,
                          SerializedGraph const& graph,
                          std::size_t node_index) {
    auto const path = "nodes[" + std::to_string(node_index) + "]";
    if (operation.inputs.empty() || operation.outputs.size() != 1 ||
        !operation.outputs.contains("OUT_0")) {
        return fail(ErrorCode::kInvalidValue, path,
                    "POINTWISE ports are incomplete");
    }
    auto const* output = graph.find_tensor(operation.outputs.at("OUT_0"));
    if (output == nullptr) {
        return fail(ErrorCode::kMissingUid, path,
                    "POINTWISE output reference is unresolved");
    }
    std::string_view compute_type;
    std::string_view mode;
    std::optional<std::int64_t> axis;
    if (!read_string_attribute(operation, "compute_data_type", compute_type) ||
        compute_type != "FLOAT" ||
        !read_string_attribute(operation, "mode", mode) ||
        !import::is_pointwise_mode(mode) ||
        !read_optional_integer_attribute(operation, "axis", axis)) {
        return fail(ErrorCode::kInvalidValue, path,
                    "POINTWISE attributes are not executable f32 semantics");
    }
    if (mode == "GEN_INDEX" &&
        (!axis || *axis < 0 ||
         *axis >= static_cast<std::int64_t>(output->dim.size()))) {
        return fail(ErrorCode::kInvalidValue, path,
                    "GEN_INDEX requires an axis in the output rank");
    }
    std::map<std::string_view, std::optional<double>> parameters;
    for (auto name : {"relu_lower_clip", "relu_upper_clip",
                      "relu_lower_clip_slope", "swish_beta", "elu_alpha",
                      "softplus_beta"}) {
        std::optional<double> value;
        if (!read_optional_number_attribute(operation, name, value) ||
            (value &&
             (!std::isfinite(*value) ||
              std::fabs(*value) >
                  static_cast<double>(std::numeric_limits<float>::max())))) {
            return fail(ErrorCode::kInvalidValue, path,
                        "POINTWISE numeric attributes must fit finite f32 or "
                        "be null");
        }
        parameters.emplace(name, value);
    }
    auto const lower = parameters.at("relu_lower_clip");
    auto const upper = parameters.at("relu_upper_clip");
    if (lower && upper && *lower > *upper) {
        return fail(ErrorCode::kInvalidValue, path,
                    "POINTWISE ReLU lower clip exceeds upper clip");
    }
    auto const softplus_beta = parameters.at("softplus_beta");
    if (softplus_beta && *softplus_beta <= 0.0) {
        return fail(ErrorCode::kInvalidValue, path,
                    "POINTWISE softplus beta must be positive");
    }

    std::vector<std::int64_t> broadcast_shape(output->dim.size(), 1);
    for (std::size_t input_index = 0;
         input_index < operation.inputs.size(); ++input_index) {
        auto const port = "IN_" + std::to_string(input_index);
        auto const port_it = operation.inputs.find(port);
        auto const* input = port_it == operation.inputs.end()
                                ? nullptr
                                : graph.find_tensor(port_it->second);
        if (input == nullptr) {
            return fail(ErrorCode::kMissingUid, path,
                        "POINTWISE input reference is unresolved");
        }
        if (input->dim.size() > output->dim.size()) {
            return fail(ErrorCode::kInvalidShape, path,
                        "POINTWISE input rank exceeds output rank");
        }
        auto const rank_offset = output->dim.size() - input->dim.size();
        for (std::size_t dimension = 0; dimension < input->dim.size();
             ++dimension) {
            auto const output_dimension = rank_offset + dimension;
            auto const extent = input->dim[dimension];
            auto& inferred = broadcast_shape[output_dimension];
            if (inferred != 1 && extent != 1 && inferred != extent) {
                return fail(ErrorCode::kInvalidShape, path,
                            "POINTWISE inputs are not broadcast compatible");
            }
            inferred = std::max(inferred, extent);
        }
    }
    if (broadcast_shape != output->dim) {
        return fail(ErrorCode::kInvalidShape, path,
                    "POINTWISE output is not the inferred broadcast shape");
    }
    return Status::ok();
}

Status validate_reduction(GenericOperationDesc const& operation,
                          SerializedGraph const& graph,
                          std::size_t node_index) {
    auto const path = "nodes[" + std::to_string(node_index) + "]";
    if (operation.inputs.size() != 1 || !operation.inputs.contains("X") ||
        operation.outputs.size() != 1 || !operation.outputs.contains("Y")) {
        return fail(ErrorCode::kInvalidValue, path,
                    "REDUCTION requires exactly X and Y");
    }
    auto const* x = graph.find_tensor(operation.inputs.at("X"));
    auto const* y = graph.find_tensor(operation.outputs.at("Y"));
    if (x == nullptr || y == nullptr) {
        return fail(ErrorCode::kMissingUid, path,
                    "REDUCTION tensor reference is unresolved");
    }
    std::string_view compute_type;
    std::string_view mode;
    bool deterministic = false;
    if (!read_string_attribute(operation, "compute_data_type", compute_type) ||
        compute_type != "FLOAT" ||
        !read_string_attribute(operation, "mode", mode) ||
        !import::is_reduction_mode(mode) ||
        !read_bool_attribute(operation, "is_deterministic", deterministic)) {
        return fail(ErrorCode::kInvalidValue, path,
                    "REDUCTION attributes are not executable f32 semantics");
    }
    (void)deterministic;
    if (x->dim.size() != y->dim.size()) {
        return fail(ErrorCode::kInvalidShape, path,
                    "REDUCTION X and Y ranks must match");
    }
    for (std::size_t dimension = 0; dimension < x->dim.size(); ++dimension) {
        if (y->dim[dimension] != 1 &&
            y->dim[dimension] != x->dim[dimension]) {
            return fail(ErrorCode::kInvalidShape, path,
                        "REDUCTION Y extents must retain or reduce each X axis");
        }
    }
    std::uint64_t input_elements = 0;
    if (!checked_element_count(x->dim, input_elements) ||
        input_elements >
            static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
        return fail(ErrorCode::kDimensionOverflow, path,
                    "REDUCTION logical element count does not fit int64");
    }
    return Status::ok();
}

Status validate_matmul(GenericOperationDesc const& operation,
                       SerializedGraph const& graph,
                       std::size_t node_index) {
    auto const path = "nodes[" + std::to_string(node_index) + "]";
    if (!operation.inputs.contains("A") || !operation.inputs.contains("B") ||
        operation.outputs.size() != 1 || !operation.outputs.contains("C")) {
        return fail(ErrorCode::kInvalidValue, path,
                    "MATMUL requires A, B, and C");
    }
    for (auto override_name : {"M_override", "N_override", "K_override"}) {
        if (operation.inputs.contains(override_name)) {
            return unsupported(path,
                               "MATMUL dimension overrides are deferred to C6");
        }
    }
    auto const* a = graph.find_tensor(operation.inputs.at("A"));
    auto const* b = graph.find_tensor(operation.inputs.at("B"));
    auto const* c = graph.find_tensor(operation.outputs.at("C"));
    if (a == nullptr || b == nullptr || c == nullptr) {
        return fail(ErrorCode::kMissingUid, path,
                    "MATMUL tensor reference is unresolved");
    }
    std::string_view compute_type;
    std::optional<double> padding_value;
    if (!read_string_attribute(operation, "compute_data_type", compute_type) ||
        compute_type != "FLOAT" ||
        !read_optional_number_attribute(operation, "padding_value",
                                        padding_value)) {
        return fail(ErrorCode::kInvalidValue, path,
                    "MATMUL attributes are not executable f32 semantics");
    }
    if (padding_value && !std::isfinite(*padding_value)) {
        return fail(ErrorCode::kInvalidValue, path,
                    "MATMUL padding value must be finite");
    }
    if (padding_value && *padding_value != 0.0) {
        return unsupported(
            path,
            "nonzero MATMUL padding is deferred with dimension overrides to C6");
    }
    if (a->dim.size() < 2 || a->dim.size() != b->dim.size() ||
        a->dim.size() != c->dim.size()) {
        return fail(ErrorCode::kInvalidShape, path,
                    "MATMUL A, B, and C require the same rank of at least two");
    }
    auto const rank = a->dim.size();
    if (a->dim[rank - 1] != b->dim[rank - 2] ||
        c->dim[rank - 2] != a->dim[rank - 2] ||
        c->dim[rank - 1] != b->dim[rank - 1]) {
        return fail(ErrorCode::kInvalidShape, path,
                    "MATMUL M, N, and K dimensions are inconsistent");
    }
    for (std::size_t dimension = 0; dimension + 2 < rank; ++dimension) {
        auto const a_extent = a->dim[dimension];
        auto const b_extent = b->dim[dimension];
        if (a_extent != b_extent && a_extent != 1 && b_extent != 1) {
            return fail(ErrorCode::kInvalidShape, path,
                        "MATMUL batch dimensions are not broadcast compatible");
        }
        if (c->dim[dimension] != std::max(a_extent, b_extent)) {
            return fail(ErrorCode::kInvalidShape, path,
                        "MATMUL C batch dimensions do not match broadcast");
        }
    }
    return Status::ok();
}

Status validate_resample(GenericOperationDesc const& operation,
                         SerializedGraph const& graph,
                         std::size_t node_index) {
    auto const path = "nodes[" + std::to_string(node_index) + "]";
    if (operation.inputs.size() != 1 || !operation.inputs.contains("X") ||
        !operation.outputs.contains("Y")) {
        return fail(ErrorCode::kInvalidValue, path,
                    "RESAMPLE requires X and Y");
    }
    if (operation.outputs.size() != 1 || operation.outputs.contains("Index")) {
        return unsupported(path,
                           "RESAMPLE index output data types are deferred to C5");
    }
    auto const* x = graph.find_tensor(operation.inputs.at("X"));
    auto const* y = graph.find_tensor(operation.outputs.at("Y"));
    if (x == nullptr || y == nullptr) {
        return fail(ErrorCode::kMissingUid, path,
                    "RESAMPLE tensor reference is unresolved");
    }
    std::optional<bool> generate_index;
    std::string_view mode;
    std::string_view padding_mode;
    std::vector<std::int64_t> pre_padding;
    std::vector<std::int64_t> post_padding;
    std::vector<std::int64_t> strides;
    std::vector<std::int64_t> windows;
    if (!read_optional_bool_attribute(operation, "generate_index",
                                      generate_index) ||
        !generate_index ||
        !read_string_attribute(operation, "resample_mode", mode) ||
        !read_string_attribute(operation, "padding_mode", padding_mode) ||
        !read_integer_array(operation, "pre_padding", pre_padding) ||
        !read_integer_array(operation, "post_padding", post_padding) ||
        !read_integer_array(operation, "stride", strides) ||
        !read_integer_array(operation, "window", windows)) {
        return fail(ErrorCode::kInvalidValue, path,
                    "RESAMPLE attributes are not executable integer semantics");
    }
    if (mode == "MAXPOOL" && *generate_index) {
        return unsupported(path,
                           "MAXPOOL index generation is deferred to C5");
    }
    if (mode != "AVGPOOL_EXCLUDE_PADDING" &&
        mode != "AVGPOOL_INCLUDE_PADDING" && mode != "BILINEAR" &&
        mode != "NEAREST" && mode != "MAXPOOL") {
        return fail(ErrorCode::kInvalidValue, path,
                    "RESAMPLE mode is unknown");
    }
    if (mode == "BILINEAR") {
        return unsupported(
            path,
            "cuDNN Frontend v1.24.0 serialization omits fraction denominators "
            "required to reconstruct BILINEAR semantics");
    }
    if (padding_mode != "EDGE_VAL_PAD" &&
        padding_mode != "NEG_INF_PAD" && padding_mode != "ZERO_PAD") {
        return fail(ErrorCode::kInvalidValue, path,
                    "RESAMPLE padding mode is unknown");
    }
    if (x->dim.size() < 3 || x->dim.size() != y->dim.size()) {
        return fail(ErrorCode::kInvalidShape, path,
                    "RESAMPLE X and Y require the same rank of at least three");
    }
    auto const spatial_rank = x->dim.size() - 2;
    if (pre_padding.size() != spatial_rank ||
        post_padding.size() != spatial_rank ||
        strides.size() != spatial_rank || windows.size() != spatial_rank ||
        x->dim[0] != y->dim[0] || x->dim[1] != y->dim[1]) {
        return fail(ErrorCode::kInvalidShape, path,
                    "RESAMPLE spatial parameters and N/C dimensions mismatch");
    }
    for (std::size_t dimension = 0; dimension < spatial_rank; ++dimension) {
        if (pre_padding[dimension] < 0 || post_padding[dimension] < 0 ||
            strides[dimension] <= 0 || windows[dimension] <= 0) {
            return fail(ErrorCode::kInvalidShape, path,
                        "RESAMPLE padding, stride, and window are invalid");
        }
        auto const input_extent = x->dim[dimension + 2];
        if (pre_padding[dimension] >
                std::numeric_limits<std::int64_t>::max() - input_extent ||
            post_padding[dimension] >
                std::numeric_limits<std::int64_t>::max() - input_extent -
                    pre_padding[dimension]) {
            return fail(ErrorCode::kDimensionOverflow, path,
                        "RESAMPLE padded extent overflows int64");
        }
        auto const padded_extent = input_extent + pre_padding[dimension] +
                                   post_padding[dimension];
        if (padded_extent < windows[dimension]) {
            return fail(ErrorCode::kInvalidShape, path,
                        "RESAMPLE window exceeds padded input");
        }
        auto const expected =
            1 + (padded_extent - windows[dimension]) / strides[dimension];
        if (y->dim[dimension + 2] != expected) {
            return fail(ErrorCode::kInvalidShape, path,
                        "RESAMPLE Y spatial shape does not match attributes");
        }
        if (mode == "NEAREST" && windows[dimension] != 1) {
            return unsupported(
                path,
                "integer NEAREST execution requires unit windows");
        }
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
        case OperationTag::kPointwise:
            return validate_pointwise(operation, graph, node_index);
        case OperationTag::kReduction:
            return validate_reduction(operation, graph, node_index);
        case OperationTag::kMatmul:
            return validate_matmul(operation, graph, node_index);
        case OperationTag::kResample:
            return validate_resample(operation, graph, node_index);
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
        for (auto const& [output_port, output] : operation->outputs) {
            for (auto const& [input_port, input] : operation->inputs) {
                if (output == input) {
                    return unsupported(
                        "nodes[" + std::to_string(node_index) + "]",
                        "in-place tensor UID reuse is deferred to C6 (" +
                            input_port + " -> " + output_port + ")");
                }
            }
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

::mlir::Value float_constant(::mlir::OpBuilder& builder,
                             ::mlir::Location location,
                             float value) {
    return ::mlir::arith::ConstantFloatOp::create(
        builder, location, ::mlir::Float32Type::get(builder.getContext()),
        llvm::APFloat(value));
}

::mlir::Value bool_as_float(::mlir::OpBuilder& builder,
                            ::mlir::Location location,
                            ::mlir::Value condition) {
    return ::mlir::arith::SelectOp::create(
        builder, location, condition, float_constant(builder, location, 1.0F),
        float_constant(builder, location, 0.0F));
}

llvm::SmallVector<::mlir::Value> broadcast_indices(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    llvm::SmallVector<::mlir::Value> const& output_indices,
    TensorDesc const& input,
    TensorDesc const& output) {
    auto const offset = output.dim.size() - input.dim.size();
    llvm::SmallVector<::mlir::Value> indices;
    indices.reserve(input.dim.size());
    for (std::size_t dimension = 0; dimension < input.dim.size(); ++dimension) {
        indices.push_back(input.dim[dimension] == 1
                              ? index_constant(builder, location, 0)
                              : output_indices[offset + dimension]);
    }
    return indices;
}

::mlir::Value sigmoid(::mlir::OpBuilder& builder,
                      ::mlir::Location location,
                      ::mlir::Value value) {
    auto one = float_constant(builder, location, 1.0F);
    auto negative = ::mlir::arith::NegFOp::create(builder, location, value);
    auto exponential =
        ::mlir::math::ExpOp::create(builder, location, negative);
    auto denominator =
        ::mlir::arith::AddFOp::create(builder, location, one, exponential);
    return ::mlir::arith::DivFOp::create(builder, location, one, denominator);
}

std::optional<::mlir::Value> pointwise_result(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    std::string_view mode,
    llvm::SmallVector<::mlir::Value> const& inputs,
    llvm::SmallVector<::mlir::Value> const& output_indices,
    GenericOperationDesc const& operation) {
    auto zero = float_constant(builder, location, 0.0F);
    auto one = float_constant(builder, location, 1.0F);
    auto half = float_constant(builder, location, 0.5F);
    auto x = inputs[0];
    auto y = inputs.size() > 1 ? inputs[1] : zero;
    auto optional_number = [&](std::string_view name, double fallback) {
        std::optional<double> value;
        (void)read_optional_number_attribute(operation, name, value);
        return static_cast<float>(value.value_or(fallback));
    };

    if (mode == "ADD") {
        return ::mlir::arith::AddFOp::create(builder, location, x, y);
    }
    if (mode == "MUL") {
        return ::mlir::arith::MulFOp::create(builder, location, x, y);
    }
    if (mode == "DIV") {
        return ::mlir::arith::DivFOp::create(builder, location, x, y);
    }
    if (mode == "SUB") {
        return ::mlir::arith::SubFOp::create(builder, location, x, y);
    }
    if (mode == "ADD_SQUARE") {
        auto square = ::mlir::arith::MulFOp::create(builder, location, y, y);
        return ::mlir::arith::AddFOp::create(builder, location, x, square);
    }
    if (mode == "MIN") {
        return ::mlir::arith::MinimumFOp::create(builder, location, x, y);
    }
    if (mode == "MAX") {
        return ::mlir::arith::MaximumFOp::create(builder, location, x, y);
    }
    if (mode == "MOD") {
        return ::mlir::arith::RemFOp::create(builder, location, x, y);
    }
    if (mode == "POW") {
        return ::mlir::math::PowFOp::create(builder, location, x, y);
    }
    if (mode == "SQRT") {
        return ::mlir::math::SqrtOp::create(builder, location, x);
    }
    if (mode == "RSQRT") {
        auto root = ::mlir::math::SqrtOp::create(builder, location, x);
        return ::mlir::arith::DivFOp::create(builder, location, one, root);
    }
    if (mode == "EXP") {
        return ::mlir::math::ExpOp::create(builder, location, x);
    }
    if (mode == "LOG") {
        return ::mlir::math::LogOp::create(builder, location, x);
    }
    if (mode == "NEG") {
        return ::mlir::arith::NegFOp::create(builder, location, x);
    }
    if (mode == "ABS") {
        return ::mlir::math::AbsFOp::create(builder, location, x);
    }
    if (mode == "CEIL") {
        return ::mlir::math::CeilOp::create(builder, location, x);
    }
    if (mode == "FLOOR") {
        return ::mlir::math::FloorOp::create(builder, location, x);
    }
    if (mode == "COS") {
        return ::mlir::math::CosOp::create(builder, location, x);
    }
    if (mode == "SIN") {
        return ::mlir::math::SinOp::create(builder, location, x);
    }
    if (mode == "TAN") {
        return ::mlir::math::TanOp::create(builder, location, x);
    }
    if (mode == "ERF") {
        return ::mlir::math::ErfOp::create(builder, location, x);
    }
    if (mode == "RECIPROCAL") {
        return ::mlir::arith::DivFOp::create(builder, location, one, x);
    }
    if (mode == "IDENTITY") {
        return x;
    }
    if (mode == "GEN_INDEX") {
        std::optional<std::int64_t> axis;
        (void)read_optional_integer_attribute(operation, "axis", axis);
        auto integer = ::mlir::arith::IndexCastUIOp::create(
            builder, location, ::mlir::IntegerType::get(builder.getContext(), 64),
            output_indices[static_cast<std::size_t>(*axis)]);
        return ::mlir::arith::UIToFPOp::create(
            builder, location, ::mlir::Float32Type::get(builder.getContext()),
            integer);
    }

    auto nonzero = [&](::mlir::Value value) {
        return ::mlir::arith::CmpFOp::create(
            builder, location, ::mlir::arith::CmpFPredicate::UNE, value, zero);
    };
    if (mode == "LOGICAL_NOT") {
        auto condition = ::mlir::arith::CmpFOp::create(
            builder, location, ::mlir::arith::CmpFPredicate::OEQ, x, zero);
        return bool_as_float(builder, location, condition);
    }
    if (mode == "LOGICAL_AND" || mode == "LOGICAL_OR") {
        auto lhs = nonzero(x);
        auto rhs = nonzero(y);
        auto condition = mode == "LOGICAL_AND"
                             ? static_cast<::mlir::Value>(
                                   ::mlir::arith::AndIOp::create(
                                       builder, location, lhs, rhs))
                             : static_cast<::mlir::Value>(
                                   ::mlir::arith::OrIOp::create(
                                       builder, location, lhs, rhs));
        return bool_as_float(builder, location, condition);
    }
    if (mode == "BINARY_SELECT") {
        return ::mlir::arith::SelectOp::create(
            builder, location, nonzero(inputs[2]), y, x);
    }

    auto comparison = [&](::mlir::arith::CmpFPredicate predicate) {
        return bool_as_float(
            builder, location,
            ::mlir::arith::CmpFOp::create(builder, location, predicate, x, y));
    };
    if (mode == "CMP_EQ") {
        return comparison(::mlir::arith::CmpFPredicate::OEQ);
    }
    if (mode == "CMP_NEQ") {
        return comparison(::mlir::arith::CmpFPredicate::UNE);
    }
    if (mode == "CMP_GT") {
        return comparison(::mlir::arith::CmpFPredicate::OGT);
    }
    if (mode == "CMP_GE") {
        return comparison(::mlir::arith::CmpFPredicate::OGE);
    }
    if (mode == "CMP_LT") {
        return comparison(::mlir::arith::CmpFPredicate::OLT);
    }
    if (mode == "CMP_LE") {
        return comparison(::mlir::arith::CmpFPredicate::OLE);
    }

    if (mode == "RELU_FWD" || mode == "RELU_BWD") {
        auto lower_value = optional_number("relu_lower_clip", 0.0);
        auto upper_value = optional_number(
            "relu_upper_clip", std::numeric_limits<float>::max());
        auto slope_value = optional_number("relu_lower_clip_slope", 0.0);
        auto lower = float_constant(builder, location, lower_value);
        auto upper = float_constant(builder, location, upper_value);
        auto slope = float_constant(builder, location, slope_value);
        auto below = ::mlir::arith::CmpFOp::create(
            builder, location, ::mlir::arith::CmpFPredicate::OLT,
            mode == "RELU_BWD" ? y : x, lower);
        auto above = ::mlir::arith::CmpFOp::create(
            builder, location, ::mlir::arith::CmpFPredicate::OGT,
            mode == "RELU_BWD" ? y : x, upper);
        if (mode == "RELU_FWD") {
            auto clipped = ::mlir::arith::MinimumFOp::create(
                builder, location,
                ::mlir::arith::MaximumFOp::create(builder, location, x, lower),
                upper);
            auto lower_branch = ::mlir::arith::AddFOp::create(
                builder, location, lower,
                ::mlir::arith::MulFOp::create(
                    builder, location, slope,
                    ::mlir::arith::SubFOp::create(builder, location, x,
                                                  lower)));
            return ::mlir::arith::SelectOp::create(builder, location, below,
                                                   lower_branch, clipped);
        }
        auto derivative = ::mlir::arith::SelectOp::create(
            builder, location, below, slope,
            ::mlir::arith::SelectOp::create(builder, location, above, zero,
                                            one));
        return ::mlir::arith::MulFOp::create(builder, location, x, derivative);
    }

    auto activation_input = mode.ends_with("_BWD") ? y : x;
    auto dy = x;
    if (mode == "TANH_FWD" || mode == "TANH_BWD") {
        auto tanh =
            ::mlir::math::TanhOp::create(builder, location, activation_input);
        if (mode == "TANH_FWD") {
            return tanh;
        }
        auto square = ::mlir::arith::MulFOp::create(builder, location, tanh,
                                                    tanh);
        auto derivative =
            ::mlir::arith::SubFOp::create(builder, location, one, square);
        return ::mlir::arith::MulFOp::create(builder, location, dy,
                                             derivative);
    }
    if (mode == "SIGMOID_FWD" || mode == "SIGMOID_BWD") {
        auto value = sigmoid(builder, location, activation_input);
        if (mode == "SIGMOID_FWD") {
            return value;
        }
        auto derivative = ::mlir::arith::MulFOp::create(
            builder, location, value,
            ::mlir::arith::SubFOp::create(builder, location, one, value));
        return ::mlir::arith::MulFOp::create(builder, location, dy,
                                             derivative);
    }
    if (mode == "ELU_FWD" || mode == "ELU_BWD") {
        auto alpha = float_constant(
            builder, location, optional_number("elu_alpha", 1.0));
        auto positive = ::mlir::arith::CmpFOp::create(
            builder, location, ::mlir::arith::CmpFPredicate::OGE,
            activation_input, zero);
        auto exponential =
            ::mlir::math::ExpOp::create(builder, location, activation_input);
        if (mode == "ELU_FWD") {
            auto negative = ::mlir::arith::MulFOp::create(
                builder, location, alpha,
                ::mlir::arith::SubFOp::create(builder, location, exponential,
                                              one));
            return ::mlir::arith::SelectOp::create(
                builder, location, positive, activation_input, negative);
        }
        auto derivative = ::mlir::arith::SelectOp::create(
            builder, location, positive, one,
            ::mlir::arith::MulFOp::create(builder, location, alpha,
                                          exponential));
        return ::mlir::arith::MulFOp::create(builder, location, dy,
                                             derivative);
    }
    if (mode == "SOFTPLUS_FWD" || mode == "SOFTPLUS_BWD") {
        auto beta = float_constant(
            builder, location, optional_number("softplus_beta", 1.0));
        auto scaled = ::mlir::arith::MulFOp::create(
            builder, location, beta, activation_input);
        if (mode == "SOFTPLUS_FWD") {
            auto absolute =
                ::mlir::math::AbsFOp::create(builder, location, scaled);
            auto logarithm = ::mlir::math::LogOp::create(
                builder, location,
                ::mlir::arith::AddFOp::create(
                    builder, location, one,
                    ::mlir::math::ExpOp::create(
                        builder, location,
                        ::mlir::arith::NegFOp::create(builder, location,
                                                     absolute))));
            auto stable = ::mlir::arith::AddFOp::create(
                builder, location,
                ::mlir::arith::MaximumFOp::create(builder, location, scaled,
                                                  zero),
                logarithm);
            return ::mlir::arith::DivFOp::create(builder, location, stable,
                                                 beta);
        }
        auto derivative = sigmoid(builder, location, scaled);
        return ::mlir::arith::MulFOp::create(builder, location, dy,
                                             derivative);
    }
    if (mode == "SWISH_FWD" || mode == "SWISH_BWD") {
        auto beta = float_constant(
            builder, location, optional_number("swish_beta", 1.0));
        auto scaled = ::mlir::arith::MulFOp::create(
            builder, location, beta, activation_input);
        auto s = sigmoid(builder, location, scaled);
        if (mode == "SWISH_FWD") {
            return ::mlir::arith::MulFOp::create(builder, location,
                                                 activation_input, s);
        }
        auto derivative = ::mlir::arith::AddFOp::create(
            builder, location, s,
            ::mlir::arith::MulFOp::create(
                builder, location, scaled,
                ::mlir::arith::MulFOp::create(
                    builder, location, s,
                    ::mlir::arith::SubFOp::create(builder, location, one, s))));
        return ::mlir::arith::MulFOp::create(builder, location, dy,
                                             derivative);
    }

    auto gelu_exact = [&](bool backward) -> ::mlir::Value {
        auto inverse_sqrt_two =
            float_constant(builder, location, 0.7071067811865475F);
        auto inverse_sqrt_two_pi =
            float_constant(builder, location, 0.3989422804014327F);
        auto scaled = ::mlir::arith::MulFOp::create(
            builder, location, activation_input, inverse_sqrt_two);
        auto erf = ::mlir::math::ErfOp::create(builder, location, scaled);
        auto cdf = ::mlir::arith::MulFOp::create(
            builder, location, half,
            ::mlir::arith::AddFOp::create(builder, location, one, erf));
        if (!backward) {
            return ::mlir::arith::MulFOp::create(builder, location,
                                                 activation_input, cdf);
        }
        auto square = ::mlir::arith::MulFOp::create(
            builder, location, activation_input, activation_input);
        auto density = ::mlir::arith::MulFOp::create(
            builder, location, inverse_sqrt_two_pi,
            ::mlir::math::ExpOp::create(
                builder, location,
                ::mlir::arith::MulFOp::create(
                    builder, location,
                    float_constant(builder, location, -0.5F), square)));
        auto derivative = ::mlir::arith::AddFOp::create(
            builder, location, cdf,
            ::mlir::arith::MulFOp::create(builder, location,
                                          activation_input, density));
        return ::mlir::arith::MulFOp::create(builder, location, dy,
                                             derivative);
    };
    if (mode == "GELU_FWD") {
        return gelu_exact(false);
    }
    if (mode == "GELU_BWD") {
        return gelu_exact(true);
    }

    auto gelu_approx = [&](bool backward) -> ::mlir::Value {
        auto coefficient =
            float_constant(builder, location, 0.7978845608028654F);
        auto cubic_coefficient =
            float_constant(builder, location, 0.044715F);
        auto square = ::mlir::arith::MulFOp::create(
            builder, location, activation_input, activation_input);
        auto cubic = ::mlir::arith::MulFOp::create(
            builder, location, square, activation_input);
        auto inner = ::mlir::arith::MulFOp::create(
            builder, location, coefficient,
            ::mlir::arith::AddFOp::create(
                builder, location, activation_input,
                ::mlir::arith::MulFOp::create(
                    builder, location, cubic_coefficient, cubic)));
        auto t = ::mlir::math::TanhOp::create(builder, location, inner);
        auto cdf = ::mlir::arith::MulFOp::create(
            builder, location, half,
            ::mlir::arith::AddFOp::create(builder, location, one, t));
        if (!backward) {
            return ::mlir::arith::MulFOp::create(builder, location,
                                                 activation_input, cdf);
        }
        auto inner_derivative = ::mlir::arith::MulFOp::create(
            builder, location, coefficient,
            ::mlir::arith::AddFOp::create(
                builder, location, one,
                ::mlir::arith::MulFOp::create(
                    builder, location,
                    float_constant(builder, location, 0.134145F), square)));
        auto tanh_derivative = ::mlir::arith::SubFOp::create(
            builder, location, one,
            ::mlir::arith::MulFOp::create(builder, location, t, t));
        auto derivative = ::mlir::arith::AddFOp::create(
            builder, location, cdf,
            ::mlir::arith::MulFOp::create(
                builder, location, half,
                ::mlir::arith::MulFOp::create(
                    builder, location, activation_input,
                    ::mlir::arith::MulFOp::create(
                        builder, location, tanh_derivative,
                        inner_derivative))));
        return ::mlir::arith::MulFOp::create(builder, location, dy,
                                             derivative);
    };
    if (mode == "GELU_APPROX_TANH_FWD") {
        return gelu_approx(false);
    }
    if (mode == "GELU_APPROX_TANH_BWD") {
        return gelu_approx(true);
    }

    return std::nullopt;
}

Status emit_pointwise(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    GenericOperationDesc const& operation,
    SerializedGraph const& graph,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    auto const output_uid =
        std::get<std::int64_t>(operation.outputs.at("OUT_0"));
    auto const& output = graph.tensors.at(output_uid);
    std::string_view mode;
    (void)read_string_attribute(operation, "mode", mode);
    auto emission_status = Status::ok();
    auto status = emit_flat_loop(
        builder, location, output.dim, "POINTWISE",
        [&](::mlir::OpBuilder& body_builder,
            ::mlir::Location body_location,
            llvm::SmallVector<::mlir::Value> const& output_indices) {
            llvm::SmallVector<::mlir::Value> loaded;
            loaded.reserve(operation.inputs.size());
            for (std::size_t input_index = 0;
                 input_index < operation.inputs.size(); ++input_index) {
                auto const uid = std::get<std::int64_t>(
                    operation.inputs.at("IN_" + std::to_string(input_index)));
                auto const& input = graph.tensors.at(uid);
                loaded.push_back(::mlir::memref::LoadOp::create(
                    body_builder, body_location, values.at(uid),
                    broadcast_indices(body_builder, body_location,
                                      output_indices, input, output)));
            }
            auto result = pointwise_result(body_builder, body_location, mode,
                                           loaded, output_indices, operation);
            if (!result) {
                emission_status =
                    fail(ErrorCode::kInvalidValue, "POINTWISE",
                         "validated mode has no scalar emitter: " +
                             std::string(mode));
                return;
            }
            ::mlir::memref::StoreOp::create(
                body_builder, body_location, *result, values.at(output_uid),
                output_indices);
        });
    return status.is_bad() ? status : emission_status;
}

Status emit_reduction(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    GenericOperationDesc const& operation,
    SerializedGraph const& graph,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    auto const x_uid = std::get<std::int64_t>(operation.inputs.at("X"));
    auto const y_uid = std::get<std::int64_t>(operation.outputs.at("Y"));
    auto const& x = graph.tensors.at(x_uid);
    auto const& y = graph.tensors.at(y_uid);
    std::string_view mode;
    (void)read_string_attribute(operation, "mode", mode);
    std::vector<std::size_t> reduced_axes;
    std::uint64_t reduction_count = 1;
    for (std::size_t dimension = 0; dimension < x.dim.size(); ++dimension) {
        if (y.dim[dimension] == 1 && x.dim[dimension] != 1) {
            reduced_axes.push_back(dimension);
            reduction_count *= static_cast<std::uint64_t>(x.dim[dimension]);
        }
    }
    return emit_flat_loop(
        builder, location, y.dim, "REDUCTION",
        [&](::mlir::OpBuilder& body_builder,
            ::mlir::Location body_location,
            llvm::SmallVector<::mlir::Value> const& y_indices) {
            if (reduced_axes.empty()) {
                auto value = ::mlir::memref::LoadOp::create(
                    body_builder, body_location, values.at(x_uid), y_indices);
                ::mlir::memref::StoreOp::create(
                    body_builder, body_location, value, values.at(y_uid),
                    y_indices);
                return;
            }
            float initial_value = 0.0F;
            if (mode == "MUL" || mode == "MUL_NO_ZEROS") {
                initial_value = 1.0F;
            } else if (mode == "MIN") {
                initial_value = std::numeric_limits<float>::infinity();
            } else if (mode == "MAX") {
                initial_value = -std::numeric_limits<float>::infinity();
            }
            auto initial =
                float_constant(body_builder, body_location, initial_value);
            llvm::SmallVector<::mlir::Value> lowers;
            llvm::SmallVector<::mlir::Value> uppers;
            llvm::SmallVector<::mlir::Value> steps;
            for (auto axis : reduced_axes) {
                lowers.push_back(index_constant(body_builder, body_location, 0));
                uppers.push_back(index_constant(body_builder, body_location,
                                                x.dim[axis]));
                steps.push_back(index_constant(body_builder, body_location, 1));
            }
            auto reduction = ::mlir::scf::buildLoopNest(
                body_builder, body_location, lowers, uppers, steps,
                ::mlir::ValueRange{initial},
                [&](::mlir::OpBuilder& reduction_builder,
                    ::mlir::Location reduction_location,
                    ::mlir::ValueRange reduction_indices,
                    ::mlir::ValueRange iter_args) -> ::mlir::scf::ValueVector {
                    auto x_indices = y_indices;
                    for (std::size_t index = 0; index < reduced_axes.size();
                         ++index) {
                        x_indices[reduced_axes[index]] = reduction_indices[index];
                    }
                    auto value = ::mlir::memref::LoadOp::create(
                        reduction_builder, reduction_location, values.at(x_uid),
                        x_indices);
                    auto accumulator = iter_args.front();
                    ::mlir::Value next;
                    if (mode == "ADD" || mode == "AVG") {
                        next = ::mlir::arith::AddFOp::create(
                            reduction_builder, reduction_location, accumulator,
                            value);
                    } else if (mode == "MUL") {
                        next = ::mlir::arith::MulFOp::create(
                            reduction_builder, reduction_location, accumulator,
                            value);
                    } else if (mode == "MIN") {
                        next = ::mlir::arith::MinimumFOp::create(
                            reduction_builder, reduction_location, accumulator,
                            value);
                    } else if (mode == "MAX") {
                        next = ::mlir::arith::MaximumFOp::create(
                            reduction_builder, reduction_location, accumulator,
                            value);
                    } else if (mode == "AMAX") {
                        next = ::mlir::arith::MaximumFOp::create(
                            reduction_builder, reduction_location, accumulator,
                            ::mlir::math::AbsFOp::create(
                                reduction_builder, reduction_location, value));
                    } else if (mode == "NORM1") {
                        next = ::mlir::arith::AddFOp::create(
                            reduction_builder, reduction_location, accumulator,
                            ::mlir::math::AbsFOp::create(
                                reduction_builder, reduction_location, value));
                    } else if (mode == "NORM2") {
                        next = ::mlir::arith::AddFOp::create(
                            reduction_builder, reduction_location, accumulator,
                            ::mlir::arith::MulFOp::create(
                                reduction_builder, reduction_location, value,
                                value));
                    } else {
                        auto nonzero = ::mlir::arith::CmpFOp::create(
                            reduction_builder, reduction_location,
                            ::mlir::arith::CmpFPredicate::UNE, value,
                            float_constant(reduction_builder,
                                           reduction_location, 0.0F));
                        next = ::mlir::arith::SelectOp::create(
                            reduction_builder, reduction_location, nonzero,
                            ::mlir::arith::MulFOp::create(
                                reduction_builder, reduction_location,
                                accumulator, value),
                            accumulator);
                    }
                    return {next};
                });
            auto result = reduction.results.front();
            if (mode == "AVG") {
                result = ::mlir::arith::DivFOp::create(
                    body_builder, body_location, result,
                    float_constant(body_builder, body_location,
                                   static_cast<float>(reduction_count)));
            } else if (mode == "NORM2") {
                result = ::mlir::math::SqrtOp::create(body_builder,
                                                      body_location, result);
            }
            ::mlir::memref::StoreOp::create(
                body_builder, body_location, result, values.at(y_uid),
                y_indices);
        });
}

Status emit_matmul(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    GenericOperationDesc const& operation,
    SerializedGraph const& graph,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    auto const a_uid = std::get<std::int64_t>(operation.inputs.at("A"));
    auto const b_uid = std::get<std::int64_t>(operation.inputs.at("B"));
    auto const c_uid = std::get<std::int64_t>(operation.outputs.at("C"));
    auto const& a = graph.tensors.at(a_uid);
    auto const& b = graph.tensors.at(b_uid);
    auto const& c = graph.tensors.at(c_uid);
    auto const rank = c.dim.size();
    return emit_flat_loop(
        builder, location, c.dim, "MATMUL",
        [&](::mlir::OpBuilder& body_builder,
            ::mlir::Location body_location,
            llvm::SmallVector<::mlir::Value> const& c_indices) {
            auto a_indices = c_indices;
            auto b_indices = c_indices;
            for (std::size_t dimension = 0; dimension + 2 < rank;
                 ++dimension) {
                if (a.dim[dimension] == 1) {
                    a_indices[dimension] =
                        index_constant(body_builder, body_location, 0);
                }
                if (b.dim[dimension] == 1) {
                    b_indices[dimension] =
                        index_constant(body_builder, body_location, 0);
                }
            }
            a_indices[rank - 2] = c_indices[rank - 2];
            b_indices[rank - 1] = c_indices[rank - 1];
            auto zero_index = index_constant(body_builder, body_location, 0);
            auto upper = index_constant(body_builder, body_location,
                                        a.dim[rank - 1]);
            auto one_index = index_constant(body_builder, body_location, 1);
            auto sum = ::mlir::scf::buildLoopNest(
                body_builder, body_location,
                ::mlir::ValueRange{zero_index}, ::mlir::ValueRange{upper},
                ::mlir::ValueRange{one_index},
                ::mlir::ValueRange{
                    float_constant(body_builder, body_location, 0.0F)},
                [&](::mlir::OpBuilder& reduction_builder,
                    ::mlir::Location reduction_location,
                    ::mlir::ValueRange reduction_indices,
                    ::mlir::ValueRange iter_args) -> ::mlir::scf::ValueVector {
                    a_indices[rank - 1] = reduction_indices.front();
                    b_indices[rank - 2] = reduction_indices.front();
                    auto a_value = ::mlir::memref::LoadOp::create(
                        reduction_builder, reduction_location, values.at(a_uid),
                        a_indices);
                    auto b_value = ::mlir::memref::LoadOp::create(
                        reduction_builder, reduction_location, values.at(b_uid),
                        b_indices);
                    auto product = ::mlir::arith::MulFOp::create(
                        reduction_builder, reduction_location, a_value,
                        b_value);
                    return {::mlir::arith::AddFOp::create(
                        reduction_builder, reduction_location,
                        iter_args.front(), product)};
                });
            ::mlir::memref::StoreOp::create(
                body_builder, body_location, sum.results.front(),
                values.at(c_uid), c_indices);
        });
}

struct ResampleCoordinates {
    llvm::SmallVector<::mlir::Value> indices;
    ::mlir::Value valid;
};

ResampleCoordinates resample_coordinates(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    llvm::SmallVector<::mlir::Value> const& output_indices,
    ::mlir::ValueRange window_indices,
    TensorDesc const& input,
    std::vector<std::int64_t> const& pre_padding,
    std::vector<std::int64_t> const& strides,
    bool clamp_to_edge) {
    ResampleCoordinates result;
    result.indices.reserve(input.dim.size());
    result.indices.push_back(output_indices[0]);
    result.indices.push_back(output_indices[1]);
    result.valid =
        ::mlir::arith::ConstantIntOp::create(builder, location, 1, 1);
    auto zero = index_constant(builder, location, 0);
    for (std::size_t spatial = 0; spatial < pre_padding.size(); ++spatial) {
        auto coordinate = ::mlir::arith::SubIOp::create(
            builder, location,
            ::mlir::arith::AddIOp::create(
                builder, location,
                ::mlir::arith::MulIOp::create(
                    builder, location, output_indices[spatial + 2],
                    index_constant(builder, location, strides[spatial])),
                window_indices.empty()
                    ? zero
                    : window_indices[spatial]),
            index_constant(builder, location, pre_padding[spatial]));
        auto upper = index_constant(builder, location, input.dim[spatial + 2]);
        auto lower_valid = ::mlir::arith::CmpIOp::create(
            builder, location, ::mlir::arith::CmpIPredicate::sge, coordinate,
            zero);
        auto upper_valid = ::mlir::arith::CmpIOp::create(
            builder, location, ::mlir::arith::CmpIPredicate::slt, coordinate,
            upper);
        result.valid = ::mlir::arith::AndIOp::create(
            builder, location, result.valid,
            ::mlir::arith::AndIOp::create(builder, location, lower_valid,
                                          upper_valid));
        if (clamp_to_edge) {
            auto nonnegative = ::mlir::arith::SelectOp::create(
                builder, location, lower_valid, coordinate, zero);
            auto below_upper = ::mlir::arith::CmpIOp::create(
                builder, location, ::mlir::arith::CmpIPredicate::slt,
                nonnegative, upper);
            result.indices.push_back(::mlir::arith::SelectOp::create(
                builder, location, below_upper, nonnegative,
                index_constant(builder, location,
                               input.dim[spatial + 2] - 1)));
        } else {
            result.indices.push_back(coordinate);
        }
    }
    return result;
}

::mlir::Value resample_load(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    ::mlir::Value input,
    ResampleCoordinates const& coordinates,
    bool clamp_to_edge,
    float padding_value) {
    if (clamp_to_edge) {
        return ::mlir::memref::LoadOp::create(builder, location, input,
                                              coordinates.indices);
    }
    auto if_op = ::mlir::scf::IfOp::create(
        builder, location,
        ::mlir::TypeRange{
            ::mlir::Float32Type::get(builder.getContext())},
        coordinates.valid, true);
    builder.setInsertionPointToStart(if_op.thenBlock());
    ::mlir::scf::YieldOp::create(
        builder, location, ::mlir::ValueRange{
                               ::mlir::memref::LoadOp::create(
                                   builder, location, input,
                                   coordinates.indices)});
    builder.setInsertionPointToStart(if_op.elseBlock());
    ::mlir::scf::YieldOp::create(
        builder, location,
        ::mlir::ValueRange{
            float_constant(builder, location, padding_value)});
    builder.setInsertionPointAfter(if_op);
    return if_op.getResult(0);
}

Status emit_resample(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    GenericOperationDesc const& operation,
    SerializedGraph const& graph,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    auto const x_uid = std::get<std::int64_t>(operation.inputs.at("X"));
    auto const y_uid = std::get<std::int64_t>(operation.outputs.at("Y"));
    auto const& x = graph.tensors.at(x_uid);
    auto const& y = graph.tensors.at(y_uid);
    std::string_view mode;
    std::string_view padding_mode;
    std::vector<std::int64_t> pre_padding;
    std::vector<std::int64_t> strides;
    std::vector<std::int64_t> windows;
    (void)read_string_attribute(operation, "resample_mode", mode);
    (void)read_string_attribute(operation, "padding_mode", padding_mode);
    (void)read_integer_array(operation, "pre_padding", pre_padding);
    (void)read_integer_array(operation, "stride", strides);
    (void)read_integer_array(operation, "window", windows);
    auto const clamp_to_edge = padding_mode == "EDGE_VAL_PAD";
    auto const padding_value = padding_mode == "NEG_INF_PAD"
                                   ? -std::numeric_limits<float>::infinity()
                                   : 0.0F;
    auto const is_pooling = mode == "AVGPOOL_EXCLUDE_PADDING" ||
                            mode == "AVGPOOL_INCLUDE_PADDING" ||
                            mode == "MAXPOOL";

    return emit_flat_loop(
        builder, location, y.dim, "RESAMPLE",
        [&](::mlir::OpBuilder& body_builder,
            ::mlir::Location body_location,
            llvm::SmallVector<::mlir::Value> const& output_indices) {
            if (!is_pooling) {
                auto coordinates = resample_coordinates(
                    body_builder, body_location, output_indices,
                    ::mlir::ValueRange{}, x, pre_padding, strides,
                    clamp_to_edge);
                auto sampled = resample_load(
                    body_builder, body_location, values.at(x_uid), coordinates,
                    clamp_to_edge, padding_value);
                ::mlir::memref::StoreOp::create(
                    body_builder, body_location, sampled, values.at(y_uid),
                    output_indices);
                return;
            }

            llvm::SmallVector<::mlir::Value> lowers;
            llvm::SmallVector<::mlir::Value> uppers;
            llvm::SmallVector<::mlir::Value> steps;
            for (auto window : windows) {
                lowers.push_back(index_constant(body_builder, body_location, 0));
                uppers.push_back(
                    index_constant(body_builder, body_location, window));
                steps.push_back(index_constant(body_builder, body_location, 1));
            }
            auto initial_value = mode == "MAXPOOL"
                                     ? -std::numeric_limits<float>::infinity()
                                     : 0.0F;
            auto reduction = ::mlir::scf::buildLoopNest(
                body_builder, body_location, lowers, uppers, steps,
                ::mlir::ValueRange{
                    float_constant(body_builder, body_location, initial_value),
                    float_constant(body_builder, body_location, 0.0F)},
                [&](::mlir::OpBuilder& reduction_builder,
                    ::mlir::Location reduction_location,
                    ::mlir::ValueRange window_indices,
                    ::mlir::ValueRange iter_args) -> ::mlir::scf::ValueVector {
                    auto coordinates = resample_coordinates(
                        reduction_builder, reduction_location, output_indices,
                        window_indices, x, pre_padding, strides,
                        clamp_to_edge);
                    auto sampled = resample_load(
                        reduction_builder, reduction_location, values.at(x_uid),
                        coordinates, clamp_to_edge,
                        mode == "AVGPOOL_EXCLUDE_PADDING" ? 0.0F
                                                           : padding_value);
                    ::mlir::Value next_value;
                    if (mode == "MAXPOOL") {
                        next_value = ::mlir::arith::MaximumFOp::create(
                            reduction_builder, reduction_location,
                            iter_args[0], sampled);
                    } else {
                        next_value = ::mlir::arith::AddFOp::create(
                            reduction_builder, reduction_location,
                            iter_args[0], sampled);
                    }
                    auto count_increment =
                        mode == "AVGPOOL_INCLUDE_PADDING" || clamp_to_edge
                            ? float_constant(reduction_builder,
                                             reduction_location, 1.0F)
                            : bool_as_float(reduction_builder,
                                            reduction_location,
                                            coordinates.valid);
                    auto next_count = ::mlir::arith::AddFOp::create(
                        reduction_builder, reduction_location, iter_args[1],
                        count_increment);
                    return {next_value, next_count};
                });
            auto result = reduction.results[0];
            if (mode != "MAXPOOL") {
                auto has_elements = ::mlir::arith::CmpFOp::create(
                    body_builder, body_location,
                    ::mlir::arith::CmpFPredicate::OGT, reduction.results[1],
                    float_constant(body_builder, body_location, 0.0F));
                result = ::mlir::arith::SelectOp::create(
                    body_builder, body_location, has_elements,
                    ::mlir::arith::DivFOp::create(
                        body_builder, body_location, reduction.results[0],
                        reduction.results[1]),
                    float_constant(body_builder, body_location, 0.0F));
            }
            ::mlir::memref::StoreOp::create(
                body_builder, body_location, result, values.at(y_uid),
                output_indices);
        });
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
        case OperationTag::kPointwise:
            return emit_pointwise(builder, location, operation, graph, values);
        case OperationTag::kReduction:
            return emit_reduction(builder, location, operation, graph, values);
        case OperationTag::kMatmul:
            return emit_matmul(builder, location, operation, graph, values);
        case OperationTag::kResample:
            return emit_resample(builder, location, operation, graph, values);
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
