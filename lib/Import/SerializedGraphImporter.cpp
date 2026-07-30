#include "DeepForge/Import/SerializedGraphImporter.h"

#include <cudnn_frontend/thirdparty/nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace deepforge::import {

std::string_view error_code_name(ErrorCode code) noexcept {
    switch (code) {
        case ErrorCode::kOk:
            return "DFE_OK";
        case ErrorCode::kInvalidArgument:
            return "DFE_INVALID_ARGUMENT";
        case ErrorCode::kIoError:
            return "DFE_IO_ERROR";
        case ErrorCode::kParseError:
            return "DFE_PARSE_ERROR";
        case ErrorCode::kSchemaVersionMismatch:
            return "DFE_SCHEMA_VERSION_MISMATCH";
        case ErrorCode::kFrontendVersionMismatch:
            return "DFE_FRONTEND_VERSION_MISMATCH";
        case ErrorCode::kMissingField:
            return "DFE_MISSING_FIELD";
        case ErrorCode::kInvalidFieldType:
            return "DFE_INVALID_FIELD_TYPE";
        case ErrorCode::kInvalidValue:
            return "DFE_INVALID_VALUE";
        case ErrorCode::kUnsupportedNode:
            return "DFE_UNSUPPORTED_NODE";
        case ErrorCode::kUnsupportedDataType:
            return "DFE_UNSUPPORTED_DATA_TYPE";
        case ErrorCode::kUnsupportedExecutionMetadata:
            return "DFE_UNSUPPORTED_EXECUTION_METADATA";
        case ErrorCode::kDuplicateUid:
            return "DFE_DUPLICATE_UID";
        case ErrorCode::kMissingUid:
            return "DFE_MISSING_UID";
        case ErrorCode::kInvalidLayout:
            return "DFE_INVALID_LAYOUT";
        case ErrorCode::kInvalidShape:
            return "DFE_INVALID_SHAPE";
        case ErrorCode::kDimensionOverflow:
            return "DFE_DIMENSION_OVERFLOW";
        case ErrorCode::kInvalidVariantPack:
            return "DFE_INVALID_VARIANT_PACK";
        case ErrorCode::kUnsupportedCpuFeature:
            return "DFE_UNSUPPORTED_CPU_FEATURE";
        case ErrorCode::kGraphExecutionFailed:
            return "DFE_GRAPH_EXECUTION_FAILED";
    }
    return "DFE_INTERNAL_ERROR";
}

namespace {

using Json = nlohmann::json;

static_assert(NLOHMANN_JSON_VERSION_MAJOR == 3 &&
              NLOHMANN_JSON_VERSION_MINOR == 11 &&
              NLOHMANN_JSON_VERSION_PATCH == 3);

Status fail(ErrorCode code, std::string const& path, std::string detail) {
    std::string message(error_code_name(code));
    if (!path.empty()) {
        message += ": ";
        message += path;
    }
    if (!detail.empty()) {
        message += ": ";
        message += std::move(detail);
    }
    return Status::failure(code, std::move(message));
}

std::string child_path(std::string const& path, std::string_view child) {
    if (path.empty()) {
        return std::string(child);
    }
    return path + "." + std::string(child);
}

std::string index_path(std::string const& path, std::size_t index) {
    return path + "[" + std::to_string(index) + "]";
}

Status require_field(Json const& object,
                     std::string const& path,
                     char const* name,
                     Json const*& value) {
    if (!object.is_object()) {
        return fail(ErrorCode::kInvalidFieldType, path, "expected object");
    }
    auto const it = object.find(name);
    if (it == object.end()) {
        return fail(ErrorCode::kMissingField, child_path(path, name), "required field is missing");
    }
    value = &it.value();
    return Status::ok();
}

Status read_string(Json const& value, std::string const& path, std::string& output) {
    if (!value.is_string()) {
        return fail(ErrorCode::kInvalidFieldType, path, "expected string");
    }
    output = value.get<std::string>();
    return Status::ok();
}

Status read_bool(Json const& value, std::string const& path, bool& output) {
    if (!value.is_boolean()) {
        return fail(ErrorCode::kInvalidFieldType, path, "expected boolean");
    }
    output = value.get<bool>();
    return Status::ok();
}

Status read_int64(Json const& value, std::string const& path, std::int64_t& output) {
    if (value.is_number_unsigned()) {
        auto const unsigned_value = value.get<std::uint64_t>();
        if (unsigned_value > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
            return fail(ErrorCode::kDimensionOverflow, path, "unsigned integer does not fit in int64");
        }
        output = static_cast<std::int64_t>(unsigned_value);
        return Status::ok();
    }
    if (value.is_number_integer()) {
        output = value.get<std::int64_t>();
        return Status::ok();
    }
    return fail(ErrorCode::kInvalidFieldType, path, "expected integer");
}

Status read_uint64(Json const& value, std::string const& path, std::uint64_t& output) {
    if (value.is_number_unsigned()) {
        output = value.get<std::uint64_t>();
        return Status::ok();
    }
    if (value.is_number_integer()) {
        auto const signed_value = value.get<std::int64_t>();
        if (signed_value < 0) {
            return fail(ErrorCode::kInvalidValue, path, "expected a non-negative integer");
        }
        output = static_cast<std::uint64_t>(signed_value);
        return Status::ok();
    }
    return fail(ErrorCode::kInvalidFieldType, path, "expected integer");
}

template <std::size_t N>
Status read_int_array(Json const& value,
                      std::string const& path,
                      std::array<std::int64_t, N>& output) {
    if (!value.is_array()) {
        return fail(ErrorCode::kInvalidFieldType, path, "expected integer array");
    }
    if (value.size() != N) {
        return fail(ErrorCode::kInvalidValue,
                    path,
                    "expected " + std::to_string(N) + " entries");
    }
    for (std::size_t index = 0; index < N; ++index) {
        auto status = read_int64(value[index], index_path(path, index), output[index]);
        if (status.is_bad()) {
            return status;
        }
    }
    return Status::ok();
}

Status read_float_type(Json const& value, std::string const& path, DataType& output) {
    std::string type;
    auto status = read_string(value, path, type);
    if (status.is_bad()) {
        return status;
    }
    if (type != "FLOAT") {
        return fail(ErrorCode::kUnsupportedDataType, path, "only FLOAT is supported");
    }
    output = DataType::kFloat32;
    return Status::ok();
}

Status read_optional_float_type(Json const& object,
                                std::string const& path,
                                char const* name,
                                std::optional<DataType>& output) {
    auto const it = object.find(name);
    if (it == object.end() || it->is_null()) {
        output.reset();
        return Status::ok();
    }
    DataType type = DataType::kFloat32;
    auto status = read_float_type(it.value(), child_path(path, name), type);
    if (status.is_bad()) {
        return status;
    }
    output = type;
    return Status::ok();
}

Status read_optional_string(Json const& object,
                            std::string const& path,
                            char const* name,
                            std::optional<std::string>& output) {
    auto const it = object.find(name);
    if (it == object.end() || it->is_null()) {
        output.reset();
        return Status::ok();
    }
    std::string value;
    auto status = read_string(it.value(), child_path(path, name), value);
    if (status.is_bad()) {
        return status;
    }
    output = std::move(value);
    return Status::ok();
}

Status read_optional_int64(Json const& object,
                           std::string const& path,
                           char const* name,
                           std::optional<std::int64_t>& output) {
    auto const it = object.find(name);
    if (it == object.end() || it->is_null()) {
        output.reset();
        return Status::ok();
    }
    std::int64_t value = 0;
    auto status = read_int64(it.value(), child_path(path, name), value);
    if (status.is_bad()) {
        return status;
    }
    output = value;
    return Status::ok();
}

Status read_optional_bool(Json const& object,
                          std::string const& path,
                          char const* name,
                          std::optional<bool>& output) {
    auto const it = object.find(name);
    if (it == object.end() || it->is_null()) {
        output.reset();
        return Status::ok();
    }
    bool value = false;
    auto status = read_bool(it.value(), child_path(path, name), value);
    if (status.is_bad()) {
        return status;
    }
    output = value;
    return Status::ok();
}

bool checked_add_nonnegative(std::int64_t lhs, std::int64_t rhs, std::int64_t& result) {
    if (lhs < 0 || rhs < 0 || lhs > std::numeric_limits<std::int64_t>::max() - rhs) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

bool checked_mul_nonnegative(std::int64_t lhs, std::int64_t rhs, std::int64_t& result) {
    if (lhs < 0 || rhs < 0) {
        return false;
    }
    if (lhs != 0 && rhs > std::numeric_limits<std::int64_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

Status check_tensor_element_size(TensorDesc const& tensor, std::string const& path) {
    std::int64_t element_count = 1;
    for (std::size_t index = 0; index < tensor.dim.size(); ++index) {
        if (!checked_mul_nonnegative(element_count, tensor.dim[index], element_count)) {
            return fail(ErrorCode::kDimensionOverflow, path, "tensor element count overflows int64");
        }
    }

    auto const max_elements = std::numeric_limits<std::size_t>::max() / sizeof(float);
    if (static_cast<std::uintmax_t>(element_count) > static_cast<std::uintmax_t>(max_elements)) {
        return fail(ErrorCode::kDimensionOverflow, path, "tensor byte size overflows size_t");
    }
    return Status::ok();
}

Status parse_tensor(Json const& value, std::string const& path, TensorDesc& output) {
    if (!value.is_object()) {
        return fail(ErrorCode::kInvalidFieldType, path, "expected tensor object");
    }

    Json const* field = nullptr;
    auto status = require_field(value, path, "name", field);
    if (status.is_bad()) {
        return status;
    }
    status = read_string(*field, child_path(path, "name"), output.name);
    if (status.is_bad()) {
        return status;
    }

    status = require_field(value, path, "data_type", field);
    if (status.is_bad()) {
        return status;
    }
    status = read_float_type(*field, child_path(path, "data_type"), output.data_type);
    if (status.is_bad()) {
        return status;
    }

    status = require_field(value, path, "dim", field);
    if (status.is_bad()) {
        return status;
    }
    status = read_int_array(*field, child_path(path, "dim"), output.dim);
    if (status.is_bad()) {
        return status;
    }

    status = require_field(value, path, "stride", field);
    if (status.is_bad()) {
        return status;
    }
    status = read_int_array(*field, child_path(path, "stride"), output.stride);
    if (status.is_bad()) {
        return status;
    }

    status = require_field(value, path, "is_virtual", field);
    if (status.is_bad()) {
        return status;
    }
    status = read_bool(*field, child_path(path, "is_virtual"), output.is_virtual);
    if (status.is_bad()) {
        return status;
    }
    if (output.is_virtual) {
        return fail(ErrorCode::kInvalidValue, child_path(path, "is_virtual"), "virtual tensors are not supported");
    }

    status = require_field(value, path, "is_pass_by_value", field);
    if (status.is_bad()) {
        return status;
    }
    status = read_bool(*field, child_path(path, "is_pass_by_value"), output.is_pass_by_value);
    if (status.is_bad()) {
        return status;
    }
    if (output.is_pass_by_value) {
        return fail(ErrorCode::kUnsupportedExecutionMetadata,
                    child_path(path, "is_pass_by_value"),
                    "pass-by-value tensors are not supported");
    }

    status = require_field(value, path, "reordering_type", field);
    if (status.is_bad()) {
        return status;
    }
    std::string reordering_type;
    status = read_string(*field, child_path(path, "reordering_type"), reordering_type);
    if (status.is_bad()) {
        return status;
    }
    if (reordering_type != "NONE") {
        return fail(ErrorCode::kInvalidLayout,
                    child_path(path, "reordering_type"),
                    "only NONE is supported for packed f32 tensors");
    }

    status = require_field(value, path, "uid_assigned", field);
    if (status.is_bad()) {
        return fail(ErrorCode::kMissingUid, child_path(path, "uid_assigned"), "explicit UID marker is required");
    }
    bool uid_assigned = false;
    status = read_bool(*field, child_path(path, "uid_assigned"), uid_assigned);
    if (status.is_bad()) {
        return status;
    }
    if (!uid_assigned) {
        return fail(ErrorCode::kMissingUid, path, "uid_assigned must be true");
    }

    status = require_field(value, path, "uid", field);
    if (status.is_bad()) {
        return fail(ErrorCode::kMissingUid, child_path(path, "uid"), "explicit UID is required");
    }
    status = read_int64(*field, child_path(path, "uid"), output.uid);
    if (status.is_bad()) {
        return status;
    }

    auto const pass_by_value = value.find("pass_by_value");
    if (pass_by_value != value.end() && !pass_by_value->is_null()) {
        return fail(ErrorCode::kUnsupportedExecutionMetadata,
                    child_path(path, "pass_by_value"),
                    "pass-by-value payload is not supported");
    }
    if (value.contains("ragged_offset_uid") || value.contains("ragged_offset_name")) {
        return fail(ErrorCode::kInvalidLayout, path, "ragged tensors are not supported");
    }

    for (std::size_t index = 0; index < output.dim.size(); ++index) {
        if (output.dim[index] <= 0) {
            return fail(ErrorCode::kInvalidShape,
                        index_path(child_path(path, "dim"), index),
                        "dimensions must be positive");
        }
        if (output.stride[index] <= 0) {
            return fail(ErrorCode::kInvalidLayout,
                        index_path(child_path(path, "stride"), index),
                        "strides must be positive");
        }
    }
    return check_tensor_element_size(output, path);
}

Status parse_context(Json const& root, GraphContext& output) {
    auto const it = root.find("context");
    if (it == root.end()) {
        return fail(ErrorCode::kMissingField, "context", "required field is missing");
    }
    if (!it->is_object()) {
        return fail(ErrorCode::kInvalidFieldType, "context", "expected object");
    }

    auto status = read_optional_string(it.value(), "context", "name", output.name);
    if (status.is_bad()) {
        return status;
    }
    status = read_optional_float_type(it.value(), "context", "compute_data_type", output.compute_data_type);
    if (status.is_bad()) {
        return status;
    }
    status = read_optional_float_type(it.value(), "context", "intermediate_data_type", output.intermediate_data_type);
    if (status.is_bad()) {
        return status;
    }
    status = read_optional_float_type(it.value(), "context", "io_data_type", output.io_data_type);
    if (status.is_bad()) {
        return status;
    }
    status = read_optional_int64(it.value(), "context", "sm_count", output.sm_count);
    if (status.is_bad()) {
        return status;
    }
    if (output.sm_count &&
        (*output.sm_count < std::numeric_limits<std::int32_t>::min() ||
         *output.sm_count > std::numeric_limits<std::int32_t>::max())) {
        return fail(ErrorCode::kInvalidValue, "context.sm_count", "must fit in the Frontend int32 field");
    }
    status = read_optional_bool(it.value(),
                                "context",
                                "is_dynamic_shape_enabled",
                                output.is_dynamic_shape_enabled);
    if (status.is_bad()) {
        return status;
    }
    if (output.is_dynamic_shape_enabled && *output.is_dynamic_shape_enabled) {
        return fail(ErrorCode::kInvalidShape, "context.is_dynamic_shape_enabled", "dynamic shapes are not supported");
    }
    status = read_optional_bool(it.value(),
                                "context",
                                "is_override_shape_enabled",
                                output.is_override_shape_enabled);
    if (status.is_bad()) {
        return status;
    }
    if (output.is_override_shape_enabled && *output.is_override_shape_enabled) {
        return fail(ErrorCode::kInvalidShape,
                    "context.is_override_shape_enabled",
                    "shape override is not supported by the static MVP");
    }
    return Status::ok();
}

Status parse_uid_key(std::string const& key, std::string const& path, std::int64_t& output) {
    if (key.empty()) {
        return fail(ErrorCode::kInvalidValue, path, "tensor map key must be a decimal UID");
    }
    auto const result = std::from_chars(key.data(), key.data() + key.size(), output);
    if (result.ec != std::errc{} || result.ptr != key.data() + key.size() ||
        std::to_string(output) != key) {
        return fail(ErrorCode::kInvalidValue, path, "tensor map key must be a decimal int64 UID");
    }
    return Status::ok();
}

Status parse_tensors(Json const& root, std::map<std::int64_t, TensorDesc>& output) {
    Json const* tensors = nullptr;
    auto status = require_field(root, "", "tensors", tensors);
    if (status.is_bad()) {
        return status;
    }
    if (!tensors->is_object()) {
        return fail(ErrorCode::kInvalidFieldType, "tensors", "expected object");
    }

    std::set<std::int64_t> seen_uids;
    for (auto it = tensors->begin(); it != tensors->end(); ++it) {
        auto const tensor_path = child_path("tensors", it.key());
        std::int64_t key_uid = 0;
        status = parse_uid_key(it.key(), tensor_path, key_uid);
        if (status.is_bad()) {
            return status;
        }

        TensorDesc tensor;
        status = parse_tensor(it.value(), tensor_path, tensor);
        if (status.is_bad()) {
            return status;
        }
        if (!seen_uids.insert(tensor.uid).second) {
            return fail(ErrorCode::kDuplicateUid, tensor_path, "tensor UID is used more than once");
        }
        if (tensor.uid != key_uid) {
            return fail(ErrorCode::kInvalidValue,
                        tensor_path,
                        "tensor map key and tensor.uid must identify the same UID");
        }
        output.emplace(key_uid, std::move(tensor));
    }

    if (output.size() != 3) {
        return fail(ErrorCode::kInvalidValue, "tensors", "MVP requires exactly X, W and Y tensors");
    }
    return Status::ok();
}

Status read_uid_reference(Json const& value, std::string const& path, std::int64_t& output) {
    if (value.is_string()) {
        return fail(ErrorCode::kMissingUid, path, "name references are not accepted; use an integer UID");
    }
    return read_int64(value, path, output);
}

Status parse_node(Json const& node, ConvFpropDesc& output) {
    if (!node.is_object()) {
        return fail(ErrorCode::kInvalidFieldType, "nodes[0]", "expected object");
    }
    Json const* field = nullptr;
    auto status = require_field(node, "nodes[0]", "tag", field);
    if (status.is_bad()) {
        return status;
    }
    std::string tag;
    status = read_string(*field, "nodes[0].tag", tag);
    if (status.is_bad()) {
        return status;
    }
    if (tag != "CONV_FPROP") {
        return fail(ErrorCode::kUnsupportedNode, "nodes[0].tag", "only CONV_FPROP is supported");
    }

    status = require_field(node, "nodes[0]", "name", field);
    if (status.is_bad()) {
        return status;
    }
    status = read_string(*field, "nodes[0].name", output.name);
    if (status.is_bad()) {
        return status;
    }

    status = require_field(node, "nodes[0]", "compute_data_type", field);
    if (status.is_bad()) {
        return status;
    }
    status = read_float_type(*field, "nodes[0].compute_data_type", output.compute_data_type);
    if (status.is_bad()) {
        return status;
    }

    status = require_field(node, "nodes[0]", "inputs", field);
    if (status.is_bad()) {
        return status;
    }
    if (!field->is_object() || field->size() != 2 || !field->contains("X") || !field->contains("W")) {
        return fail(ErrorCode::kInvalidValue, "nodes[0].inputs", "ports must be exactly X and W");
    }
    status = read_uid_reference(field->at("X"), "nodes[0].inputs.X", output.x_uid);
    if (status.is_bad()) {
        return status;
    }
    status = read_uid_reference(field->at("W"), "nodes[0].inputs.W", output.w_uid);
    if (status.is_bad()) {
        return status;
    }

    status = require_field(node, "nodes[0]", "outputs", field);
    if (status.is_bad()) {
        return status;
    }
    if (!field->is_object() || field->size() != 1 || !field->contains("Y")) {
        return fail(ErrorCode::kInvalidValue, "nodes[0].outputs", "ports must be exactly Y");
    }
    status = read_uid_reference(field->at("Y"), "nodes[0].outputs.Y", output.y_uid);
    if (status.is_bad()) {
        return status;
    }

    status = require_field(node, "nodes[0]", "pre_padding", field);
    if (status.is_bad()) {
        return status;
    }
    status = read_int_array(*field, "nodes[0].pre_padding", output.pre_padding);
    if (status.is_bad()) {
        return status;
    }
    status = require_field(node, "nodes[0]", "post_padding", field);
    if (status.is_bad()) {
        return status;
    }
    status = read_int_array(*field, "nodes[0].post_padding", output.post_padding);
    if (status.is_bad()) {
        return status;
    }
    status = require_field(node, "nodes[0]", "stride", field);
    if (status.is_bad()) {
        return status;
    }
    status = read_int_array(*field, "nodes[0].stride", output.stride);
    if (status.is_bad()) {
        return status;
    }
    status = require_field(node, "nodes[0]", "dilation", field);
    if (status.is_bad()) {
        return status;
    }
    status = read_int_array(*field, "nodes[0].dilation", output.dilation);
    if (status.is_bad()) {
        return status;
    }
    status = require_field(node, "nodes[0]", "math_mode", field);
    if (status.is_bad()) {
        return status;
    }
    std::string math_mode;
    status = read_string(*field, "nodes[0].math_mode", math_mode);
    if (status.is_bad()) {
        return status;
    }
    if (math_mode != "CROSS_CORRELATION") {
        return fail(ErrorCode::kInvalidValue,
                    "nodes[0].math_mode",
                    "only CROSS_CORRELATION is supported");
    }

    for (std::size_t index = 0; index < output.pre_padding.size(); ++index) {
        if (output.pre_padding[index] < 0 || output.post_padding[index] < 0) {
            return fail(ErrorCode::kInvalidShape, "nodes[0]", "padding must be non-negative");
        }
        if (output.stride[index] != 1 || output.dilation[index] != 1) {
            return fail(ErrorCode::kInvalidValue,
                        "nodes[0]",
                        "MVP requires unit stride and unit dilation");
        }
    }
    if (output.x_uid == output.w_uid || output.x_uid == output.y_uid || output.w_uid == output.y_uid) {
        return fail(ErrorCode::kInvalidValue, "nodes[0]", "X, W and Y must have distinct UIDs");
    }
    return Status::ok();
}

Status checked_output_extent(std::int64_t input,
                             std::int64_t pre_padding,
                             std::int64_t post_padding,
                             std::int64_t filter,
                             std::string const& path,
                             std::int64_t& output) {
    std::int64_t total = 0;
    if (!checked_add_nonnegative(input, pre_padding, total) ||
        !checked_add_nonnegative(total, post_padding, total)) {
        return fail(ErrorCode::kDimensionOverflow, path, "output extent arithmetic overflows int64");
    }
    if (total < filter) {
        return fail(ErrorCode::kInvalidShape, path, "filter and padding produce a non-positive output extent");
    }
    auto const numerator = total - filter;
    if (numerator == std::numeric_limits<std::int64_t>::max()) {
        return fail(ErrorCode::kDimensionOverflow, path, "output extent arithmetic overflows int64");
    }
    output = numerator + 1;
    return Status::ok();
}

Status expected_packed_stride(TensorDesc const& tensor,
                              std::string const& path,
                              std::array<std::int64_t, 4>& expected,
                              bool filter) {
    std::int64_t product = 0;
    if (filter) {
        if (!checked_mul_nonnegative(tensor.dim[3], tensor.dim[1], product) ||
            !checked_mul_nonnegative(tensor.dim[2], product, expected[0])) {
            return fail(ErrorCode::kDimensionOverflow, path, "packed stride arithmetic overflows int64");
        }
        expected[1] = 1;
        expected[2] = product;
        expected[3] = tensor.dim[1];
    } else {
        if (!checked_mul_nonnegative(tensor.dim[2], tensor.dim[3], product) ||
            !checked_mul_nonnegative(tensor.dim[1], product, expected[0])) {
            return fail(ErrorCode::kDimensionOverflow, path, "packed stride arithmetic overflows int64");
        }
        expected[1] = 1;
        if (!checked_mul_nonnegative(tensor.dim[3], tensor.dim[1], expected[2])) {
            return fail(ErrorCode::kDimensionOverflow, path, "packed stride arithmetic overflows int64");
        }
        expected[3] = tensor.dim[1];
    }
    return Status::ok();
}

Status validate_shapes(SerializedGraph const& graph) {
    auto const x_it = graph.tensors.find(graph.conv.x_uid);
    auto const w_it = graph.tensors.find(graph.conv.w_uid);
    auto const y_it = graph.tensors.find(graph.conv.y_uid);
    if (x_it == graph.tensors.end() || w_it == graph.tensors.end() || y_it == graph.tensors.end()) {
        return fail(ErrorCode::kMissingUid, "nodes[0]", "node references a tensor absent from tensors");
    }
    auto const& x = x_it->second;
    auto const& w = w_it->second;
    auto const& y = y_it->second;

    if (x.dim[1] != w.dim[1]) {
        return fail(ErrorCode::kInvalidShape, "tensors", "X.C must equal W.C");
    }
    if (y.dim[0] != x.dim[0] || y.dim[1] != w.dim[0]) {
        return fail(ErrorCode::kInvalidShape, "tensors", "Y.N/Y.K do not match X.N/W.K");
    }

    std::int64_t expected_p = 0;
    std::int64_t expected_q = 0;
    auto status = checked_output_extent(x.dim[2],
                                        graph.conv.pre_padding[0],
                                        graph.conv.post_padding[0],
                                        w.dim[2],
                                        "nodes[0].pre_padding[0]",
                                        expected_p);
    if (status.is_bad()) {
        return status;
    }
    status = checked_output_extent(x.dim[3],
                                   graph.conv.pre_padding[1],
                                   graph.conv.post_padding[1],
                                   w.dim[3],
                                   "nodes[0].pre_padding[1]",
                                   expected_q);
    if (status.is_bad()) {
        return status;
    }
    if (y.dim[2] != expected_p || y.dim[3] != expected_q) {
        return fail(ErrorCode::kInvalidShape, "tensors", "serialized Y shape does not match Conv2D inference");
    }

    std::array<std::int64_t, 4> expected{};
    status = expected_packed_stride(x, "tensors." + std::to_string(x.uid) + ".stride", expected, false);
    if (status.is_bad()) {
        return status;
    }
    if (x.stride != expected) {
        return fail(ErrorCode::kInvalidLayout, "tensors." + std::to_string(x.uid) + ".stride", "X is not packed NHWC");
    }
    status = expected_packed_stride(w, "tensors." + std::to_string(w.uid) + ".stride", expected, true);
    if (status.is_bad()) {
        return status;
    }
    if (w.stride != expected) {
        return fail(ErrorCode::kInvalidLayout, "tensors." + std::to_string(w.uid) + ".stride", "W is not packed KRSC");
    }
    status = expected_packed_stride(y, "tensors." + std::to_string(y.uid) + ".stride", expected, false);
    if (status.is_bad()) {
        return status;
    }
    if (y.stride != expected) {
        return fail(ErrorCode::kInvalidLayout, "tensors." + std::to_string(y.uid) + ".stride", "Y is not packed NHWC");
    }
    return Status::ok();
}

Status require_empty_object_if_present(Json const& root, char const* name) {
    auto const it = root.find(name);
    if (it == root.end()) {
        return Status::ok();
    }
    if (!it->is_object()) {
        return fail(ErrorCode::kInvalidFieldType, name, "expected an object");
    }
    if (!it->empty()) {
        return fail(ErrorCode::kUnsupportedExecutionMetadata, name, "non-empty metadata is not supported");
    }
    return Status::ok();
}

Status require_empty_object_or_array_if_present(Json const& root, char const* name) {
    auto const it = root.find(name);
    if (it == root.end()) {
        return Status::ok();
    }
    if (!it->is_object() && !it->is_array()) {
        return fail(ErrorCode::kInvalidFieldType, name, "expected an object or array");
    }
    if (!it->empty()) {
        return fail(ErrorCode::kUnsupportedExecutionMetadata, name, "non-empty metadata is not supported");
    }
    return Status::ok();
}

Status validate_variant_pack_uids(Json const& root, ConvFpropDesc const& conv) {
    auto const it = root.find("variant_pack_uids");
    if (it == root.end()) {
        return Status::ok();
    }
    if (!it->is_array()) {
        return fail(ErrorCode::kInvalidFieldType, "variant_pack_uids", "expected integer array");
    }

    std::set<std::int64_t> actual;
    for (std::size_t index = 0; index < it->size(); ++index) {
        std::int64_t uid = 0;
        auto status = read_int64((*it)[index], index_path("variant_pack_uids", index), uid);
        if (status.is_bad()) {
            return status;
        }
        if (!actual.insert(uid).second) {
            return fail(ErrorCode::kDuplicateUid, index_path("variant_pack_uids", index), "UID is repeated");
        }
    }

    std::set<std::int64_t> expected{conv.x_uid, conv.w_uid, conv.y_uid};
    if (actual != expected) {
        return fail(ErrorCode::kInvalidValue,
                    "variant_pack_uids",
                    "UID set must equal the non-virtual X/W/Y set");
    }
    return Status::ok();
}

Status validate_metadata(Json const& root, ConvFpropDesc const& conv) {
    auto status = require_empty_object_if_present(root, "pass_by_values");
    if (status.is_bad()) {
        return status;
    }
    status = require_empty_object_if_present(root, "workspace_modifications");
    if (status.is_bad()) {
        return status;
    }
    // nlohmann/json encodes the Frontend's integer-keyed unordered_map as an
    // array of pairs; hand-authored Graph JSON may use an object.
    status = require_empty_object_or_array_if_present(root, "variant_pack_replacements");
    if (status.is_bad()) {
        return status;
    }

    auto const workspace_size = root.find("fe_workspace_size");
    if (workspace_size != root.end()) {
        std::int64_t value = 0;
        status = read_int64(workspace_size.value(), "fe_workspace_size", value);
        if (status.is_bad()) {
            return status;
        }
        if (value != 0) {
            return fail(ErrorCode::kUnsupportedExecutionMetadata,
                        "fe_workspace_size",
                        "CPU MVP requires zero Frontend workspace");
        }
    }
    return validate_variant_pack_uids(root, conv);
}

Status read_document(Json const& root, SerializedGraph& output) {
    if (!root.is_object()) {
        return fail(ErrorCode::kInvalidFieldType, "root", "serialized Graph must be an object");
    }

    SerializedGraph graph;
    Json const* field = nullptr;
    auto status = require_field(root, "", "json_version", field);
    if (status.is_bad()) {
        return status;
    }
    status = read_string(*field, "json_version", graph.json_version);
    if (status.is_bad()) {
        return status;
    }
    if (graph.json_version != "1.0") {
        return fail(ErrorCode::kSchemaVersionMismatch, "json_version", "MVP requires version 1.0");
    }

    status = require_field(root, "", "cudnn_frontend_version", field);
    if (status.is_bad()) {
        return status;
    }
    status = read_int64(*field, "cudnn_frontend_version", graph.cudnn_frontend_version);
    if (status.is_bad()) {
        return status;
    }
    if (graph.cudnn_frontend_version != 12400) {
        return fail(ErrorCode::kFrontendVersionMismatch,
                    "cudnn_frontend_version",
                    "MVP requires cuDNN Frontend v1.24.0 (12400)");
    }

    status = require_field(root, "", "graph_uid", field);
    if (status.is_bad()) {
        return status;
    }
    status = read_uint64(*field, "graph_uid", graph.graph_uid);
    if (status.is_bad()) {
        return status;
    }

    status = parse_context(root, graph.context);
    if (status.is_bad()) {
        return status;
    }

    status = require_field(root, "", "nodes", field);
    if (status.is_bad()) {
        return status;
    }
    if (!field->is_array()) {
        return fail(ErrorCode::kInvalidFieldType, "nodes", "expected array");
    }
    if (field->size() != 1) {
        return fail(ErrorCode::kInvalidValue, "nodes", "MVP requires exactly one node");
    }
    status = parse_node((*field)[0], graph.conv);
    if (status.is_bad()) {
        return status;
    }

    status = parse_tensors(root, graph.tensors);
    if (status.is_bad()) {
        return status;
    }

    if (graph.tensors.find(graph.conv.x_uid) == graph.tensors.end() ||
        graph.tensors.find(graph.conv.w_uid) == graph.tensors.end() ||
        graph.tensors.find(graph.conv.y_uid) == graph.tensors.end()) {
        return fail(ErrorCode::kMissingUid, "nodes[0]", "node references an unknown tensor UID");
    }
    status = validate_shapes(graph);
    if (status.is_bad()) {
        return status;
    }
    status = validate_metadata(root, graph.conv);
    if (status.is_bad()) {
        return status;
    }

    output = std::move(graph);
    return Status::ok();
}

Status decode_json(std::span<std::uint8_t const> input, Json& output) {
    try {
        auto const* first = input.empty()
                                ? ""
                                : reinterpret_cast<char const*>(input.data());
        output = Json::parse(first, first + input.size(), nullptr, true, false);
        return Status::ok();
    } catch (std::exception const& exception) {
        return fail(ErrorCode::kParseError, "json", exception.what());
    }
}

Status decode_ubjson(std::span<std::uint8_t const> input, Json& output) {
    try {
        std::vector<std::uint8_t> bytes(input.begin(), input.end());
        output = Json::from_ubjson(bytes.begin(), bytes.end(), true, true);
        auto const canonical = Json::to_ubjson(output);
        if (canonical.size() != bytes.size() ||
            !std::equal(canonical.begin(), canonical.end(), bytes.begin())) {
            return fail(ErrorCode::kParseError,
                        "ubjson",
                        "input is not the canonical document produced by nlohmann/json 3.11.3");
        }
        return Status::ok();
    } catch (std::exception const& exception) {
        return fail(ErrorCode::kParseError, "ubjson", exception.what());
    }
}

Status validate_decoded(Json const& document, SerializedGraph& output) {
    try {
        return read_document(document, output);
    } catch (std::exception const& exception) {
        return fail(ErrorCode::kParseError, "document", exception.what());
    }
}

}  // namespace

Status SerializedGraphImporter::parse(std::span<std::uint8_t const> input,
                                      InputFormat format,
                                      SerializedGraph& output) const {
    if (input.empty()) {
        return fail(ErrorCode::kInvalidArgument, "input", "serialized Graph is empty");
    }
    if (input.size() > kMaximumSerializedGraphBytes) {
        return fail(ErrorCode::kInvalidArgument, "input",
                    "serialized Graph exceeds the 16 MiB input limit");
    }

    Status json_decode_status = Status::ok();
    if (format == InputFormat::kJson || format == InputFormat::kAuto) {
        Json document;
        json_decode_status = decode_json(input, document);
        if (json_decode_status.is_good()) {
            SerializedGraph candidate;
            auto validation_status = validate_decoded(document, candidate);
            if (validation_status.is_good()) {
                output = std::move(candidate);
            }
            return validation_status;
        }
        if (format == InputFormat::kJson) {
            return json_decode_status;
        }
    }

    if (format == InputFormat::kUbjson || format == InputFormat::kAuto) {
        Json document;
        auto decode_status = decode_ubjson(input, document);
        if (decode_status.is_good()) {
            SerializedGraph candidate;
            auto validation_status = validate_decoded(document, candidate);
            if (validation_status.is_good()) {
                output = std::move(candidate);
            }
            return validation_status;
        }
        if (format == InputFormat::kUbjson) {
            return decode_status;
        }
        return fail(ErrorCode::kParseError,
                    "auto",
                    "neither JSON nor UBJSON decoded (JSON: " + json_decode_status.message() +
                        "; UBJSON: " + decode_status.message() + ")");
    }

    return fail(ErrorCode::kInvalidArgument, "format", "unknown input format");
}

Status SerializedGraphImporter::parse_file(std::filesystem::path const& path,
                                           InputFormat format,
                                           SerializedGraph& output) const {
    if (path.empty()) {
        return fail(ErrorCode::kInvalidArgument, "path", "input path is empty");
    }
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return fail(ErrorCode::kIoError, path.string(), "cannot open input file");
    }

    std::error_code size_error;
    auto const file_size = std::filesystem::file_size(path, size_error);
    if (!size_error && file_size > kMaximumSerializedGraphBytes) {
        return fail(ErrorCode::kInvalidArgument, path.string(),
                    "serialized Graph exceeds the 16 MiB input limit");
    }

    std::vector<std::uint8_t> bytes;
    if (!size_error) {
        bytes.reserve(static_cast<std::size_t>(file_size));
    }
    std::array<char, 8192> buffer{};
    while (file) {
        file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        auto const count = file.gcount();
        if (count <= 0) {
            continue;
        }
        auto const byte_count = static_cast<std::size_t>(count);
        if (byte_count > kMaximumSerializedGraphBytes - bytes.size()) {
            return fail(ErrorCode::kInvalidArgument, path.string(),
                        "serialized Graph exceeds the 16 MiB input limit");
        }
        bytes.insert(bytes.end(), buffer.begin(), buffer.begin() + count);
    }
    if (!file.eof()) {
        return fail(ErrorCode::kIoError, path.string(), "cannot read input file");
    }
    return parse(std::span<std::uint8_t const>(bytes), format, output);
}

}  // namespace deepforge::import
