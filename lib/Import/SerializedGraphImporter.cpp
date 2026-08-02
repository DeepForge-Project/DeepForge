#include "DeepForge/Import/SerializedGraphImporter.h"

#include "DeepForge/Import/Capability.h"
#include "DeepForge/Import/Schema.h"

#include <cudnn_frontend/thirdparty/nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
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
        case ErrorCode::kUnsupportedOperation:
            return "DFE_UNSUPPORTED_OPERATION";
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

Status parse_serialized_value(Json const& value,
                              std::string const& path,
                              SerializedValue& output,
                              std::size_t depth = 0) {
    if (depth > 64) {
        return fail(ErrorCode::kInvalidValue, path,
                    "serialized attribute nesting exceeds 64 levels");
    }
    if (value.is_null()) {
        output.value = nullptr;
        return Status::ok();
    }
    if (value.is_boolean()) {
        output.value = value.get<bool>();
        return Status::ok();
    }
    if (value.is_number_unsigned()) {
        auto number = value.get<std::uint64_t>();
        if (number <= static_cast<std::uint64_t>(
                          std::numeric_limits<std::int64_t>::max())) {
            output.value = static_cast<std::int64_t>(number);
        } else {
            output.value = number;
        }
        return Status::ok();
    }
    if (value.is_number_integer()) {
        output.value = value.get<std::int64_t>();
        return Status::ok();
    }
    if (value.is_number_float()) {
        auto number = value.get<double>();
        if (!std::isfinite(number)) {
            return fail(ErrorCode::kInvalidValue, path,
                        "floating attribute must be finite");
        }
        output.value = number;
        return Status::ok();
    }
    if (value.is_string()) {
        auto text = value.get<std::string>();
        if (text.find('\0') != std::string::npos) {
            return fail(ErrorCode::kInvalidValue, path,
                        "serialized string contains NUL");
        }
        output.value = std::move(text);
        return Status::ok();
    }
    if (value.is_array()) {
        if (value.size() > 1U << 20) {
            return fail(ErrorCode::kInvalidValue, path,
                        "serialized array is too large");
        }
        SerializedValue::Array array;
        array.reserve(value.size());
        for (std::size_t index = 0; index < value.size(); ++index) {
            SerializedValue element;
            auto status = parse_serialized_value(
                value[index], index_path(path, index), element, depth + 1);
            if (status.is_bad()) {
                return status;
            }
            array.push_back(std::move(element));
        }
        output.value = std::move(array);
        return Status::ok();
    }
    if (value.is_object()) {
        if (value.size() > 1U << 20) {
            return fail(ErrorCode::kInvalidValue, path,
                        "serialized object is too large");
        }
        SerializedValue::Object object;
        for (auto it = value.begin(); it != value.end(); ++it) {
            if (it.key().find('\0') != std::string::npos) {
                return fail(ErrorCode::kInvalidValue, path,
                            "serialized object key contains NUL");
            }
            SerializedValue element;
            auto status = parse_serialized_value(
                it.value(), child_path(path, it.key()), element, depth + 1);
            if (status.is_bad()) {
                return status;
            }
            object.emplace(it.key(), std::move(element));
        }
        output.value = std::move(object);
        return Status::ok();
    }
    return fail(ErrorCode::kInvalidFieldType, path,
                "unsupported serialized JSON value");
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

Status read_int_array(Json const& value,
                      std::string const& path,
                      std::vector<std::int64_t>& output) {
    if (!value.is_array()) {
        return fail(ErrorCode::kInvalidFieldType, path, "expected integer array");
    }
    if (value.empty() || value.size() > 64) {
        return fail(ErrorCode::kInvalidValue, path,
                    "tensor rank must be between 1 and 64");
    }
    std::vector<std::int64_t> values(value.size());
    for (std::size_t index = 0; index < value.size(); ++index) {
        auto status = read_int64(value[index], index_path(path, index),
                                 values[index]);
        if (status.is_bad()) {
            return status;
        }
    }
    output = std::move(values);
    return Status::ok();
}

Status read_data_type(Json const& value, std::string const& path,
                      DataType& output) {
    std::string type;
    auto status = read_string(value, path, type);
    if (status.is_bad()) {
        return status;
    }
    auto parsed = data_type_from_name(type);
    if (!parsed) {
        return fail(ErrorCode::kUnsupportedDataType, path,
                    "unknown cuDNN Frontend data type: " + type);
    }
    output = *parsed;
    return Status::ok();
}

Status read_optional_data_type(Json const& object,
                               std::string const& path,
                               char const* name,
                               std::optional<DataType>& output) {
    auto const it = object.find(name);
    if (it == object.end() || it->is_null()) {
        output.reset();
        return Status::ok();
    }
    DataType type = DataType::kFloat32;
    auto status = read_data_type(it.value(), child_path(path, name), type);
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

    auto const storage_bits = data_type_storage_bits(tensor.data_type);
    if (storage_bits == 0) {
        return fail(ErrorCode::kUnsupportedDataType, path,
                    "data type has no storage representation");
    }
    auto const count = static_cast<std::uintmax_t>(element_count);
    auto const bits = static_cast<std::uintmax_t>(storage_bits);
    if (count > (std::numeric_limits<std::uintmax_t>::max() - 7U) / bits) {
        return fail(ErrorCode::kDimensionOverflow, path, "tensor byte size overflows size_t");
    }
    auto const storage_bytes = (count * bits + 7U) / 8U;
    if (storage_bytes > std::numeric_limits<std::size_t>::max()) {
        return fail(ErrorCode::kDimensionOverflow, path, "tensor byte size overflows size_t");
    }
    return Status::ok();
}

Status parse_tensor(Json const& value, std::string const& path, TensorDesc& output) {
    if (!value.is_object()) {
        return fail(ErrorCode::kInvalidFieldType, path, "expected tensor object");
    }
    constexpr std::array<std::string_view, 12> kTensorFields{{
        "name",          "data_type",       "dim",
        "stride",        "is_virtual",      "pass_by_value",
        "is_pass_by_value", "reordering_type", "uid",
        "uid_assigned",  "ragged_offset_uid", "ragged_offset_name",
    }};
    for (auto it = value.begin(); it != value.end(); ++it) {
        if (std::find(kTensorFields.begin(), kTensorFields.end(), it.key()) ==
            kTensorFields.end()) {
            return fail(ErrorCode::kInvalidValue,
                        child_path(path, it.key()),
                        "field is not emitted by the pinned tensor serializer");
        }
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
    if (output.name.empty() ||
        output.name.find('\0') != std::string::npos) {
        return fail(ErrorCode::kInvalidValue, child_path(path, "name"),
                    "tensor name is empty or contains NUL");
    }

    status = require_field(value, path, "data_type", field);
    if (status.is_bad()) {
        return status;
    }
    status = read_data_type(*field, child_path(path, "data_type"), output.data_type);
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
    if (output.stride.size() != output.dim.size()) {
        return fail(ErrorCode::kInvalidValue, child_path(path, "stride"),
                    "dimension and stride ranks must match");
    }

    status = require_field(value, path, "is_virtual", field);
    if (status.is_bad()) {
        return status;
    }
    status = read_bool(*field, child_path(path, "is_virtual"), output.is_virtual);
    if (status.is_bad()) {
        return status;
    }

    status = require_field(value, path, "is_pass_by_value", field);
    if (status.is_bad()) {
        return status;
    }
    status = read_bool(*field, child_path(path, "is_pass_by_value"), output.is_pass_by_value);
    if (status.is_bad()) {
        return status;
    }

    status = require_field(value, path, "reordering_type", field);
    if (status.is_bad()) {
        return status;
    }
    status = read_string(*field, child_path(path, "reordering_type"),
                         output.reordering_type);
    if (status.is_bad()) {
        return status;
    }
    if (output.reordering_type != "NONE" &&
        output.reordering_type != "INT8x32" &&
        output.reordering_type != "F16x16" &&
        output.reordering_type != "F8_128x4") {
        return fail(ErrorCode::kInvalidValue,
                    child_path(path, "reordering_type"),
                    "unknown TensorReordering_t value");
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
    output.uid_assigned = uid_assigned;

    status = require_field(value, path, "uid", field);
    if (status.is_bad()) {
        return fail(ErrorCode::kMissingUid, child_path(path, "uid"), "explicit UID is required");
    }
    status = read_int64(*field, child_path(path, "uid"), output.uid);
    if (status.is_bad()) {
        return status;
    }

    status = require_field(value, path, "pass_by_value", field);
    if (status.is_bad()) {
        return status;
    }
    if (!field->is_null()) {
        SerializedValue payload;
        status = parse_serialized_value(
            *field, child_path(path, "pass_by_value"), payload);
        if (status.is_bad()) {
            return status;
        }
        if (!std::holds_alternative<std::int64_t>(payload.value) &&
            !std::holds_alternative<std::uint64_t>(payload.value) &&
            !std::holds_alternative<double>(payload.value)) {
            return fail(ErrorCode::kInvalidFieldType,
                        child_path(path, "pass_by_value"),
                        "Frontend scalar payload must be numeric");
        }
        output.pass_by_value = std::move(payload);
    }
    if (output.pass_by_value && !output.is_pass_by_value) {
        return fail(ErrorCode::kInvalidValue,
                    child_path(path, "is_pass_by_value"),
                    "scalar payload requires is_pass_by_value=true");
    }
    auto const ragged_uid = value.find("ragged_offset_uid");
    if (ragged_uid != value.end() && !ragged_uid->is_null()) {
        std::int64_t uid = 0;
        status = read_int64(*ragged_uid, child_path(path, "ragged_offset_uid"),
                            uid);
        if (status.is_bad()) {
            return status;
        }
        output.ragged_offset_uid = uid;
    }
    auto const ragged_name = value.find("ragged_offset_name");
    if (ragged_name != value.end() && !ragged_name->is_null()) {
        std::string name;
        status = read_string(*ragged_name,
                             child_path(path, "ragged_offset_name"), name);
        if (status.is_bad()) {
            return status;
        }
        output.ragged_offset_name = std::move(name);
    }
    if (output.ragged_offset_uid.has_value() !=
        output.ragged_offset_name.has_value()) {
        return fail(ErrorCode::kInvalidValue, path,
                    "ragged offset UID and name must appear together");
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
    constexpr std::array<std::string_view, 7> kContextFields{{
        "name", "compute_data_type", "intermediate_data_type", "io_data_type",
        "sm_count", "is_dynamic_shape_enabled", "is_override_shape_enabled",
    }};
    for (auto name : kContextFields) {
        if (!it->contains(std::string(name))) {
            return fail(ErrorCode::kMissingField, child_path("context", name),
                        "serialized context field is missing");
        }
    }
    for (auto field = it->begin(); field != it->end(); ++field) {
        if (std::find(kContextFields.begin(), kContextFields.end(),
                      field.key()) == kContextFields.end()) {
            return fail(ErrorCode::kInvalidValue,
                        child_path("context", field.key()),
                        "field is not emitted by the pinned context serializer");
        }
    }

    auto status = read_optional_string(it.value(), "context", "name", output.name);
    if (status.is_bad()) {
        return status;
    }
    if (output.name && output.name->find('\0') != std::string::npos) {
        return fail(ErrorCode::kInvalidValue, "context.name",
                    "context name contains NUL");
    }
    status = read_optional_data_type(it.value(), "context", "compute_data_type", output.compute_data_type);
    if (status.is_bad()) {
        return status;
    }
    status = read_optional_data_type(it.value(), "context", "intermediate_data_type", output.intermediate_data_type);
    if (status.is_bad()) {
        return status;
    }
    status = read_optional_data_type(it.value(), "context", "io_data_type", output.io_data_type);
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
    status = read_optional_bool(it.value(),
                                "context",
                                "is_override_shape_enabled",
                                output.is_override_shape_enabled);
    if (status.is_bad()) {
        return status;
    }
    return Status::ok();
}

bool parse_uid_key(std::string const& key, std::int64_t& output) {
    if (key.empty()) {
        return false;
    }
    auto const result = std::from_chars(key.data(), key.data() + key.size(), output);
    return result.ec == std::errc{} && result.ptr == key.data() + key.size() &&
           std::to_string(output) == key;
}

Status parse_tensors(Json const& root,
                     std::map<std::int64_t, TensorDesc>& uid_tensors,
                     std::map<std::string, TensorDesc>& named_tensors) {
    Json const* tensors = nullptr;
    auto status = require_field(root, "", "tensors", tensors);
    if (status.is_bad()) {
        return status;
    }
    if (!tensors->is_object()) {
        return fail(ErrorCode::kInvalidFieldType, "tensors", "expected object");
    }

    for (auto it = tensors->begin(); it != tensors->end(); ++it) {
        auto const tensor_path = child_path("tensors", it.key());
        std::int64_t key_uid = 0;
        auto const key_is_uid = parse_uid_key(it.key(), key_uid);

        TensorDesc tensor;
        status = parse_tensor(it.value(), tensor_path, tensor);
        if (status.is_bad()) {
            return status;
        }
        if (tensor.uid_assigned) {
            if (!key_is_uid || tensor.uid != key_uid) {
                return fail(
                    ErrorCode::kInvalidValue, tensor_path,
                    "UID-assigned tensor requires a matching decimal map key");
            }
            if (!uid_tensors.emplace(key_uid, std::move(tensor)).second) {
                return fail(ErrorCode::kDuplicateUid, tensor_path,
                            "tensor UID is used more than once");
            }
        } else {
            if (it.key().empty() ||
                it.key().find('\0') != std::string::npos ||
                it.key() != tensor.name) {
                return fail(
                    ErrorCode::kInvalidValue, tensor_path,
                    "name-keyed tensor requires a map key matching tensor.name");
            }
            if (!named_tensors.emplace(it.key(), std::move(tensor)).second) {
                return fail(ErrorCode::kInvalidValue, tensor_path,
                            "tensor name is used more than once");
            }
        }
    }

    return Status::ok();
}

Status read_uid_reference(Json const& value, std::string const& path, std::int64_t& output) {
    if (value.is_string()) {
        return fail(ErrorCode::kMissingUid, path, "name references are not accepted; use an integer UID");
    }
    return read_int64(value, path, output);
}

bool contains_name(std::vector<std::string_view> const& names,
                   std::string_view name) {
    return std::find(names.begin(), names.end(), name) != names.end();
}

bool is_indexed_port(std::string_view port) {
    if (port.empty()) {
        return false;
    }
    std::uint64_t index = 0;
    auto const result =
        std::from_chars(port.data(), port.data() + port.size(), index);
    return result.ec == std::errc{} && result.ptr == port.data() + port.size() &&
           std::to_string(index) == port;
}

Status validate_node_fields(Json const& node,
                            std::string const& path,
                            OperationSchema const& schema) {
    for (auto const* common : {"tag", "name", "inputs", "outputs"}) {
        if (!node.contains(common)) {
            return fail(ErrorCode::kMissingField, child_path(path, common),
                        "required field is missing");
        }
    }
    for (auto name : schema.required_attributes) {
        if (!node.contains(std::string(name))) {
            return fail(ErrorCode::kMissingField,
                        child_path(path, name),
                        "serialized attribute is missing");
        }
    }
    for (auto it = node.begin(); it != node.end(); ++it) {
        auto const& name = it.key();
        if (name == "tag" || name == "name" || name == "inputs" ||
            name == "outputs" ||
            contains_name(schema.required_attributes, name) ||
            contains_name(schema.optional_attributes, name)) {
            continue;
        }
        return fail(ErrorCode::kInvalidValue, child_path(path, name),
                    "field is not emitted by the pinned serializer for " +
                        std::string(schema.serialized_tag));
    }
    return Status::ok();
}

Status read_tensor_reference(Json const& value,
                             std::string const& path,
                             TensorReference& output) {
    if (value.is_string()) {
        std::string name;
        auto status = read_string(value, path, name);
        if (status.is_bad()) {
            return status;
        }
        if (name.empty() || name.find('\0') != std::string::npos) {
            return fail(ErrorCode::kInvalidValue, path,
                        "tensor name reference is empty or contains NUL");
        }
        output = std::move(name);
        return Status::ok();
    }
    std::int64_t uid = 0;
    auto status = read_int64(value, path, uid);
    if (status.is_bad()) {
        return status;
    }
    output = uid;
    return Status::ok();
}

Status parse_ports(Json const& value,
                   std::string const& path,
                   std::vector<std::string_view> const& allowed,
                   bool allows_indexed,
                   std::map<std::string, TensorReference>& output) {
    if (!value.is_object()) {
        return fail(ErrorCode::kInvalidFieldType, path,
                    "expected tensor-reference object");
    }
    for (auto it = value.begin(); it != value.end(); ++it) {
        if ((!allows_indexed && !contains_name(allowed, it.key())) ||
            (allows_indexed && !is_indexed_port(it.key()))) {
            return fail(ErrorCode::kInvalidValue,
                        child_path(path, it.key()),
                        "port is not part of the pinned operation schema");
        }
        TensorReference reference;
        auto status = read_tensor_reference(
            it.value(), child_path(path, it.key()), reference);
        if (status.is_bad()) {
            return status;
        }
        output.emplace(it.key(), std::move(reference));
    }
    return Status::ok();
}

bool enum_value(Json const& value,
                std::initializer_list<std::string_view> allowed,
                bool nullable = true) {
    if (value.is_null()) {
        return nullable;
    }
    if (!value.is_string()) {
        return false;
    }
    auto const text = value.get_ref<std::string const&>();
    return std::find(allowed.begin(), allowed.end(), text) != allowed.end();
}

bool is_integer(Json const& value) {
    return value.is_number_integer() || value.is_number_unsigned();
}

bool integer_array(Json const& value) {
    return value.is_array() &&
           std::all_of(value.begin(), value.end(), [](Json const& item) {
               return is_integer(item);
           });
}

bool slice_array(Json const& value) {
    return value.is_array() &&
           std::all_of(value.begin(), value.end(), [](Json const& slice) {
               return slice.is_array() && slice.size() == 2 &&
                      std::all_of(slice.begin(), slice.end(),
                                  [](Json const& item) {
                                      return is_integer(item);
                                  });
           });
}

bool integer_fits_int32(Json const& value) {
    if (value.is_number_unsigned()) {
        return value.get<std::uint64_t>() <=
               static_cast<std::uint64_t>(
                   std::numeric_limits<std::int32_t>::max());
    }
    if (!value.is_number_integer()) {
        return false;
    }
    auto const integer = value.get<std::int64_t>();
    return integer >= std::numeric_limits<std::int32_t>::min() &&
           integer <= std::numeric_limits<std::int32_t>::max();
}

bool int32_array(Json const& value) {
    return value.is_array() &&
           std::all_of(value.begin(), value.end(), [](Json const& item) {
               return integer_fits_int32(item);
    });
}

bool nullable_integer(Json const& value) {
    return value.is_null() || value.is_number_integer() ||
           value.is_number_unsigned();
}

bool nullable_number(Json const& value) {
    return value.is_null() || value.is_number();
}

Status validate_attribute(Json const& value,
                          std::string const& path,
                          OperationTag tag,
                          std::string_view name) {
    bool valid = false;
    if (name == "compute_data_type" || name == "mma_core_mode") {
        if (value.is_null()) {
            valid = true;
        } else if (value.is_string()) {
            valid = data_type_from_name(value.get_ref<std::string const&>())
                        .has_value();
        }
    } else if (name == "mode") {
        if (tag == OperationTag::kPointwise) {
            valid = value.is_string() &&
                    is_pointwise_mode(value.get_ref<std::string const&>());
        } else if (tag == OperationTag::kReduction) {
            valid = value.is_string() &&
                    is_reduction_mode(value.get_ref<std::string const&>());
        } else {
            valid = enum_value(value, {"NONE", "GATHER", "SCATTER"},
                               false);
        }
    } else if (name == "math_mode") {
        valid = enum_value(value, {"CONVOLUTION", "CROSS_CORRELATION"},
                           false);
    } else if (name == "forward_phase") {
        valid = enum_value(value, {"INFERENCE", "TRAINING"});
    } else if (name == "distribution") {
        valid = enum_value(value, {"BERNOULLI", "UNIFORM", "NORMAL"});
    } else if (name == "resample_mode") {
        valid = enum_value(value,
                           {"AVGPOOL_EXCLUDE_PADDING",
                            "AVGPOOL_INCLUDE_PADDING", "BILINEAR", "NEAREST",
                            "MAXPOOL"});
    } else if (name == "padding_mode") {
        valid = enum_value(value,
                           {"EDGE_VAL_PAD", "NEG_INF_PAD", "ZERO_PAD"});
    } else if (name == "reshape_mode") {
        valid = enum_value(value, {"VIEW_ONLY", "LOGICAL"});
    } else if (name == "diagonal_alignment") {
        valid = enum_value(value, {"TOP_LEFT", "BOTTOM_RIGHT"}, false);
    } else if (name == "implementation") {
        valid = enum_value(value, {"AUTO", "COMPOSITE", "UNIFIED"}, false);
    } else if (name == "slices") {
        valid = slice_array(value);
    } else if (name == "pre_padding" || name == "post_padding" ||
               name == "stride" || name == "dilation" || name == "dim" ||
               name == "window" || name == "permutation" ||
               name == "slice_strides") {
        valid = integer_array(value);
    } else if (name == "block_size") {
        valid = tag == OperationTag::kBlockScaleDequantize
                    ? int32_array(value)
                    : value.is_null() || integer_fits_int32(value);
    } else if (name == "peer_stats") {
        valid = value.is_array();
    } else if (name == "is_negative_scale" ||
               name == "is_deterministic" || name == "alibi_mask" ||
               name == "padding_mask" ||
               name == "is_deterministic_algorithm" || name == "is_mxfp8" ||
               name == "unfuse_fma") {
        valid = value.is_boolean();
    } else if (name == "generate_index" || name == "generate_stats") {
        valid = value.is_null() || value.is_boolean();
    } else if (name == "top_k") {
        valid = integer_fits_int32(value);
    } else if (name == "rope_dim") {
        valid = is_integer(value);
    } else if (name == "max_seq_len_kv") {
        valid = value.is_null() || integer_fits_int32(value);
    } else if (name == "axis" || name == "in_place_index" ||
               name == "seed" ||
               name == "left_bound" ||
               name == "right_bound" || name == "max_total_seq_len_q" ||
               name == "max_total_seq_len_kv") {
        valid = nullable_integer(value);
    } else if (name == "padding_value" || name == "relu_lower_clip" ||
               name == "relu_upper_clip" ||
               name == "relu_lower_clip_slope" || name == "swish_beta" ||
               name == "elu_alpha" || name == "softplus_beta" ||
               name == "bernoulli_probability" ||
               name == "dropout_probability" ||
               name == "attn_scale_value" ||
               name == "rescale_threshold") {
        valid = nullable_number(value);
    } else if (name == "output_scale") {
        valid = value.is_number();
    }

    if (!valid) {
        return fail(ErrorCode::kInvalidFieldType, path,
                    "attribute has a value outside the pinned schema");
    }
    return Status::ok();
}

Status merge_embedded_tensor(
    TensorDesc tensor,
    std::string const& path,
    std::map<std::int64_t, TensorDesc>& uid_tensors,
    std::map<std::string, TensorDesc>& named_tensors,
    TensorReference& reference) {
    if (tensor.uid_assigned) {
        reference = tensor.uid;
        auto const existing = uid_tensors.find(tensor.uid);
        if (existing == uid_tensors.end()) {
            uid_tensors.emplace(tensor.uid, std::move(tensor));
            return Status::ok();
        }
        if (existing->second != tensor) {
            return fail(ErrorCode::kDuplicateUid, path,
                        "embedded tensor conflicts with the existing UID");
        }
        return Status::ok();
    }

    reference = tensor.name;
    auto const existing = named_tensors.find(tensor.name);
    if (existing == named_tensors.end()) {
        named_tensors.emplace(tensor.name, std::move(tensor));
        return Status::ok();
    }
    if (existing->second != tensor) {
        return fail(ErrorCode::kInvalidValue, path,
                    "embedded tensor conflicts with the existing name");
    }
    return Status::ok();
}

Status parse_embedded_tensor_list(
    Json const& value,
    std::string const& path,
    std::map<std::int64_t, TensorDesc>& uid_tensors,
    std::map<std::string, TensorDesc>& named_tensors,
    std::vector<TensorReference>& references) {
    references.reserve(value.size());
    for (std::size_t index = 0; index < value.size(); ++index) {
        auto const tensor_path = index_path(path, index);
        TensorDesc tensor;
        auto status = parse_tensor(value[index], tensor_path, tensor);
        if (status.is_bad()) {
            return status;
        }
        TensorReference reference;
        status = merge_embedded_tensor(std::move(tensor), tensor_path,
                                       uid_tensors, named_tensors, reference);
        if (status.is_bad()) {
            return status;
        }
        references.push_back(std::move(reference));
    }
    return Status::ok();
}

Status parse_generic_node(Json const& node,
                          std::string const& path,
                          OperationSchema const& schema,
                          std::map<std::int64_t, TensorDesc>& uid_tensors,
                          std::map<std::string, TensorDesc>& named_tensors,
                          NodeDesc& output) {
    GenericOperationDesc operation;
    auto status = parse_ports(node.at("inputs"), child_path(path, "inputs"),
                              schema.input_ports, schema.allows_indexed_inputs,
                              operation.inputs);
    if (status.is_bad()) {
        return status;
    }
    status = parse_ports(node.at("outputs"), child_path(path, "outputs"),
                         schema.output_ports, false, operation.outputs);
    if (status.is_bad()) {
        return status;
    }
    for (auto name : schema.required_attributes) {
        auto const& value = node.at(std::string(name));
        status = validate_attribute(value, child_path(path, name), schema.tag,
                                    name);
        if (status.is_bad()) {
            return status;
        }
        SerializedValue canonical;
        status = parse_serialized_value(value, child_path(path, name),
                                        canonical);
        if (status.is_bad()) {
            return status;
        }
        operation.attributes.emplace(std::string(name),
                                     std::move(canonical));
        if (name == "peer_stats") {
            status = parse_embedded_tensor_list(
                value, child_path(path, name), uid_tensors, named_tensors,
                operation.input_lists[std::string(name)]);
            if (status.is_bad()) {
                return status;
            }
        }
    }
    for (auto name : schema.optional_attributes) {
        auto const it = node.find(std::string(name));
        if (it == node.end()) {
            continue;
        }
        status = validate_attribute(it.value(), child_path(path, name),
                                    schema.tag, name);
        if (status.is_bad()) {
            return status;
        }
        SerializedValue canonical;
        status = parse_serialized_value(it.value(), child_path(path, name),
                                        canonical);
        if (status.is_bad()) {
            return status;
        }
        operation.attributes.emplace(std::string(name),
                                     std::move(canonical));
        if (name == "peer_stats") {
            status = parse_embedded_tensor_list(
                it.value(), child_path(path, name), uid_tensors,
                named_tensors, operation.input_lists[std::string(name)]);
            if (status.is_bad()) {
                return status;
            }
        }
    }
    if (operation.outputs.empty()) {
        return fail(ErrorCode::kInvalidValue, child_path(path, "outputs"),
                    "operation must expose at least one serialized output");
    }
    if (schema.tag == OperationTag::kPointwise) {
        auto const& mode = node.at("mode").get_ref<std::string const&>();
        auto const expected_inputs = pointwise_input_count(mode);
        if (!expected_inputs || operation.inputs.size() != *expected_inputs ||
            operation.outputs.size() != 1 ||
            operation.outputs.find("OUT_0") == operation.outputs.end()) {
            return fail(ErrorCode::kInvalidValue, path,
                        "pointwise ports do not match mode arity");
        }
        for (std::size_t index = 0; index < *expected_inputs; ++index) {
            if (operation.inputs.find("IN_" + std::to_string(index)) ==
                operation.inputs.end()) {
                return fail(ErrorCode::kInvalidValue,
                            child_path(path, "inputs"),
                            "pointwise inputs must be contiguous from IN_0");
            }
        }
    } else if (schema.tag == OperationTag::kReduction) {
        if (operation.inputs.size() != 1 ||
            operation.inputs.find("X") == operation.inputs.end() ||
            operation.outputs.size() != 1 ||
            operation.outputs.find("Y") == operation.outputs.end()) {
            return fail(ErrorCode::kInvalidValue, path,
                        "reduction ports must be exactly X and Y");
        }
    } else if (schema.allows_indexed_inputs) {
        if (operation.inputs.empty()) {
            return fail(ErrorCode::kInvalidValue,
                        child_path(path, "inputs"),
                        "concatenate requires at least one indexed input");
        }
        for (std::size_t index = 0; index < operation.inputs.size(); ++index) {
            if (operation.inputs.find(std::to_string(index)) ==
                operation.inputs.end()) {
                return fail(ErrorCode::kInvalidValue,
                            child_path(path, "inputs"),
                            "indexed inputs must be contiguous from zero");
            }
        }
    }
    output.attributes = std::move(operation);
    return Status::ok();
}

Status resolve_node_schema(Json const& node,
                           std::string const& path,
                           OperationSchema const*& schema) {
    if (!node.is_object()) {
        return fail(ErrorCode::kInvalidFieldType, path, "expected object");
    }
    Json const* field = nullptr;
    auto status = require_field(node, path, "tag", field);
    if (status.is_bad()) {
        return status;
    }
    std::string tag;
    status = read_string(*field, child_path(path, "tag"), tag);
    if (status.is_bad()) {
        return status;
    }
    auto const* capability = find_operation_capability(tag);
    if (capability == nullptr) {
        return fail(ErrorCode::kUnsupportedNode, child_path(path, "tag"),
                    "tag is not part of the pinned v1.24.0 schema: " + tag);
    }
    schema = find_operation_schema(capability->tag);
    if (schema == nullptr || schema->serialized_tag != tag) {
        return fail(ErrorCode::kUnsupportedNode, child_path(path, "tag"),
                    "internal schema inventory is incomplete for: " + tag);
    }
    return Status::ok();
}

Status parse_node(Json const& node,
                  std::size_t index,
                  std::map<std::int64_t, TensorDesc>& uid_tensors,
                  std::map<std::string, TensorDesc>& named_tensors,
                  NodeDesc& output) {
    auto const path = index_path("nodes", index);
    OperationSchema const* schema = nullptr;
    auto status = resolve_node_schema(node, path, schema);
    if (status.is_bad()) {
        return status;
    }
    status = validate_node_fields(node, path, *schema);
    if (status.is_bad()) {
        return status;
    }
    output.tag = schema->tag;

    Json const* field = nullptr;
    status = require_field(node, path, "name", field);
    if (status.is_bad()) {
        return status;
    }
    status = read_string(*field, child_path(path, "name"), output.name);
    if (status.is_bad()) {
        return status;
    }
    if (output.name.empty() || output.name.find('\0') != std::string::npos) {
        return fail(ErrorCode::kInvalidValue, child_path(path, "name"),
                    "node name is empty or contains NUL");
    }
    if (schema->tag != OperationTag::kConvFprop) {
        return parse_generic_node(node, path, *schema, uid_tensors,
                                  named_tensors, output);
    }

    status = parse_generic_node(node, path, *schema, uid_tensors,
                                named_tensors, output);
    if (status.is_bad()) {
        return status;
    }

    auto const& compute_type = node.at("compute_data_type");
    auto const& inputs = node.at("inputs");
    auto const& outputs = node.at("outputs");
    auto const is_pair = [](Json const& value) {
        return value.is_array() && value.size() == 2 &&
               std::all_of(value.begin(), value.end(), [](Json const& item) {
                   return item.is_number_integer() ||
                          item.is_number_unsigned();
               });
    };
    if (!compute_type.is_string() ||
        compute_type.get_ref<std::string const&>() != "FLOAT" ||
        !inputs.is_object() || inputs.size() != 2 ||
        !inputs.contains("X") || !inputs.contains("W") ||
        !outputs.is_object() || outputs.size() != 1 ||
        !outputs.contains("Y") || inputs.at("X").is_string() ||
        inputs.at("W").is_string() || outputs.at("Y").is_string() ||
        !is_pair(node.at("pre_padding")) ||
        !is_pair(node.at("post_padding")) ||
        !is_pair(node.at("stride")) || !is_pair(node.at("dilation")) ||
        node.at("math_mode") != "CROSS_CORRELATION") {
        return Status::ok();
    }

    ConvFpropDesc conv;
    conv.name = output.name;

    status = require_field(node, path, "compute_data_type", field);
    if (status.is_bad()) {
        return status;
    }
    status = read_data_type(*field, child_path(path, "compute_data_type"),
                            conv.compute_data_type);
    if (status.is_bad()) {
        return status;
    }
    if (conv.compute_data_type != DataType::kFloat32) {
        return fail(ErrorCode::kUnsupportedDataType,
                    child_path(path, "compute_data_type"),
                    "CONV_FPROP execution currently requires FLOAT");
    }

    status = require_field(node, path, "inputs", field);
    if (status.is_bad()) {
        return status;
    }
    if (!field->is_object() || field->size() != 2 || !field->contains("X") || !field->contains("W")) {
        return fail(ErrorCode::kInvalidValue, child_path(path, "inputs"),
                    "ports must be exactly X and W");
    }
    status = read_uid_reference(field->at("X"),
                                child_path(child_path(path, "inputs"), "X"),
                                conv.x_uid);
    if (status.is_bad()) {
        return status;
    }
    status = read_uid_reference(field->at("W"),
                                child_path(child_path(path, "inputs"), "W"),
                                conv.w_uid);
    if (status.is_bad()) {
        return status;
    }

    status = require_field(node, path, "outputs", field);
    if (status.is_bad()) {
        return status;
    }
    if (!field->is_object() || field->size() != 1 || !field->contains("Y")) {
        return fail(ErrorCode::kInvalidValue, child_path(path, "outputs"),
                    "ports must be exactly Y");
    }
    status = read_uid_reference(field->at("Y"),
                                child_path(child_path(path, "outputs"), "Y"),
                                conv.y_uid);
    if (status.is_bad()) {
        return status;
    }

    status = require_field(node, path, "pre_padding", field);
    if (status.is_bad()) {
        return status;
    }
    status = read_int_array(*field, child_path(path, "pre_padding"),
                            conv.pre_padding);
    if (status.is_bad()) {
        return status;
    }
    status = require_field(node, path, "post_padding", field);
    if (status.is_bad()) {
        return status;
    }
    status = read_int_array(*field, child_path(path, "post_padding"),
                            conv.post_padding);
    if (status.is_bad()) {
        return status;
    }
    status = require_field(node, path, "stride", field);
    if (status.is_bad()) {
        return status;
    }
    status = read_int_array(*field, child_path(path, "stride"), conv.stride);
    if (status.is_bad()) {
        return status;
    }
    status = require_field(node, path, "dilation", field);
    if (status.is_bad()) {
        return status;
    }
    status = read_int_array(*field, child_path(path, "dilation"),
                            conv.dilation);
    if (status.is_bad()) {
        return status;
    }
    status = require_field(node, path, "math_mode", field);
    if (status.is_bad()) {
        return status;
    }
    std::string math_mode;
    status = read_string(*field, child_path(path, "math_mode"), math_mode);
    if (status.is_bad()) {
        return status;
    }
    if (math_mode != "CROSS_CORRELATION") {
        return fail(ErrorCode::kInvalidValue,
                    child_path(path, "math_mode"),
                    "only CROSS_CORRELATION is supported");
    }

    for (std::size_t axis = 0; axis < conv.pre_padding.size(); ++axis) {
        if (conv.pre_padding[axis] < 0 || conv.post_padding[axis] < 0) {
            return Status::ok();
        }
        if (conv.stride[axis] != 1 || conv.dilation[axis] != 1) {
            return Status::ok();
        }
    }
    if (conv.x_uid == conv.w_uid || conv.x_uid == conv.y_uid ||
        conv.w_uid == conv.y_uid) {
        return Status::ok();
    }
    output.attributes = std::move(conv);
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
                              std::vector<std::int64_t>& expected,
                              bool filter) {
    expected.assign(4, 0);
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

Status validate_shapes(SerializedGraph const& graph,
                       ConvFpropDesc const& conv) {
    if (graph.tensor_count() != 3) {
        return fail(ErrorCode::kInvalidValue, "tensors",
                    "validated CONV_FPROP requires exactly X, W and Y tensors");
    }
    auto const x_it = graph.tensors.find(conv.x_uid);
    auto const w_it = graph.tensors.find(conv.w_uid);
    auto const y_it = graph.tensors.find(conv.y_uid);
    if (x_it == graph.tensors.end() || w_it == graph.tensors.end() || y_it == graph.tensors.end()) {
        return fail(ErrorCode::kMissingUid, "nodes[0]", "node references a tensor absent from tensors");
    }
    auto const& x = x_it->second;
    auto const& w = w_it->second;
    auto const& y = y_it->second;

    for (auto const* tensor : {&x, &w, &y}) {
        if (tensor->data_type != DataType::kFloat32) {
            return fail(ErrorCode::kUnsupportedDataType,
                        "tensors." + std::to_string(tensor->uid) +
                            ".data_type",
                        "validated CONV_FPROP requires FLOAT tensors");
        }
        if (tensor->dim.size() != 4 || tensor->stride.size() != 4) {
            return fail(ErrorCode::kInvalidShape,
                        "tensors." + std::to_string(tensor->uid),
                        "validated CONV_FPROP requires rank-4 tensors");
        }
        if (tensor->is_virtual) {
            return fail(ErrorCode::kInvalidValue,
                        "tensors." + std::to_string(tensor->uid) +
                            ".is_virtual",
                        "validated CONV_FPROP does not accept virtual tensors");
        }
        if (tensor->is_pass_by_value) {
            return fail(ErrorCode::kUnsupportedExecutionMetadata,
                        "tensors." + std::to_string(tensor->uid) +
                            ".is_pass_by_value",
                        "validated CONV_FPROP does not accept pass-by-value tensors");
        }
        if (tensor->reordering_type != "NONE") {
            return fail(ErrorCode::kInvalidLayout,
                        "tensors." + std::to_string(tensor->uid) +
                            ".reordering_type",
                        "validated CONV_FPROP requires NONE reordering");
        }
        if (tensor->ragged_offset_uid || tensor->ragged_offset_name) {
            return fail(ErrorCode::kInvalidLayout,
                        "tensors." + std::to_string(tensor->uid),
                        "validated CONV_FPROP does not accept ragged tensors");
        }
    }

    if (x.dim[1] != w.dim[1]) {
        return fail(ErrorCode::kInvalidShape, "tensors", "X.C must equal W.C");
    }
    if (y.dim[0] != x.dim[0] || y.dim[1] != w.dim[0]) {
        return fail(ErrorCode::kInvalidShape, "tensors", "Y.N/Y.K do not match X.N/W.K");
    }

    std::int64_t expected_p = 0;
    std::int64_t expected_q = 0;
    auto status = checked_output_extent(x.dim[2],
                                        conv.pre_padding[0],
                                        conv.post_padding[0],
                                        w.dim[2],
                                        "nodes[0].pre_padding[0]",
                                        expected_p);
    if (status.is_bad()) {
        return status;
    }
    status = checked_output_extent(x.dim[3],
                                   conv.pre_padding[1],
                                   conv.post_padding[1],
                                   w.dim[3],
                                   "nodes[0].pre_padding[1]",
                                   expected_q);
    if (status.is_bad()) {
        return status;
    }
    if (y.dim[2] != expected_p || y.dim[3] != expected_q) {
        return fail(ErrorCode::kInvalidShape, "tensors", "serialized Y shape does not match Conv2D inference");
    }

    std::vector<std::int64_t> expected;
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

Status validate_graph_references(SerializedGraph const& graph) {
    std::map<TensorReference, std::size_t> producers;
    for (std::size_t node_index = 0; node_index < graph.nodes.size();
         ++node_index) {
        auto const& node = graph.nodes[node_index];
        auto const node_path = index_path("nodes", node_index);
        auto check_reference = [&](TensorReference const& reference,
                                   std::string const& path) -> Status {
            if (graph.find_tensor(reference) == nullptr) {
                return fail(ErrorCode::kMissingUid, path,
                            "tensor reference is absent from tensors");
            }
            return Status::ok();
        };
        auto add_producer = [&](TensorReference const& reference,
                                std::string const& path) -> Status {
            auto status = check_reference(reference, path);
            if (status.is_bad()) {
                return status;
            }
            if (!producers.emplace(reference, node_index).second) {
                return fail(ErrorCode::kInvalidValue, path,
                            "tensor has more than one producing node");
            }
            return Status::ok();
        };
        auto check_input = [&](TensorReference const& reference,
                               std::string const& path) -> Status {
            auto status = check_reference(reference, path);
            if (status.is_bad()) {
                return status;
            }
            auto const* tensor = graph.find_tensor(reference);
            if (tensor->is_virtual &&
                producers.find(reference) == producers.end()) {
                return fail(ErrorCode::kInvalidValue, path,
                            "virtual tensor must be produced by an earlier node");
            }
            return Status::ok();
        };

        if (auto const* conv = std::get_if<ConvFpropDesc>(&node.attributes)) {
            std::array<std::pair<std::string_view, std::int64_t>, 2>
                input_references{{{"X", conv->x_uid}, {"W", conv->w_uid}}};
            for (auto const& [port, uid] : input_references) {
                auto status = check_input(
                    TensorReference{uid},
                    child_path(child_path(node_path, "inputs"), port));
                if (status.is_bad()) {
                    return status;
                }
            }
            auto status = add_producer(
                TensorReference{conv->y_uid},
                child_path(child_path(node_path, "outputs"), "Y"));
            if (status.is_bad()) {
                return status;
            }
            continue;
        }

        auto const* operation =
            std::get_if<GenericOperationDesc>(&node.attributes);
        if (operation == nullptr) {
            return fail(ErrorCode::kInvalidValue, node_path,
                        "node has no canonical attributes");
        }
        for (auto const& [port, reference] : operation->inputs) {
            auto status = check_input(
                reference, child_path(child_path(node_path, "inputs"), port));
            if (status.is_bad()) {
                return status;
            }
        }
        for (auto const& [name, references] : operation->input_lists) {
            for (std::size_t index = 0; index < references.size(); ++index) {
                auto status = check_input(
                    references[index], index_path(child_path(node_path, name),
                                                  index));
                if (status.is_bad()) {
                    return status;
                }
            }
        }
        for (auto const& [port, reference] : operation->outputs) {
            auto status = add_producer(
                reference, child_path(child_path(node_path, "outputs"), port));
            if (status.is_bad()) {
                return status;
            }
        }
    }

    auto validate_virtual = [&](TensorReference const& reference,
                                TensorDesc const& tensor) -> Status {
        if (tensor.is_virtual && producers.find(reference) == producers.end()) {
            return fail(ErrorCode::kInvalidValue, "tensors." + tensor.name,
                        "virtual tensor has no producing node");
        }
        return Status::ok();
    };
    for (auto const& [uid, tensor] : graph.tensors) {
        auto status = validate_virtual(TensorReference{uid}, tensor);
        if (status.is_bad()) {
            return status;
        }
    }
    for (auto const& [name, tensor] : graph.named_tensors) {
        auto status = validate_virtual(TensorReference{name}, tensor);
        if (status.is_bad()) {
            return status;
        }
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

Status validate_execution_metadata(Json const& root) {
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

    return Status::ok();
}

Status validate_conv_metadata(Json const& root, ConvFpropDesc const& conv) {
    auto status = Status::ok();

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

    status = require_field(root, "", "cudnn_backend_version", field);
    if (status.is_bad()) {
        return status;
    }
    status = read_string(*field, "cudnn_backend_version",
                         graph.cudnn_backend_version);
    if (status.is_bad()) {
        return status;
    }
    if (graph.cudnn_backend_version.find('\0') != std::string::npos) {
        return fail(ErrorCode::kInvalidValue, "cudnn_backend_version",
                    "backend version contains NUL");
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

    status = validate_execution_metadata(root);
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
    if (field->empty()) {
        return fail(ErrorCode::kInvalidValue, "nodes",
                    "serialized Graph must contain at least one node");
    }
    for (std::size_t index = 0; index < field->size(); ++index) {
        OperationSchema const* schema = nullptr;
        status = resolve_node_schema((*field)[index], index_path("nodes", index),
                                     schema);
        if (status.is_bad()) {
            return status;
        }
    }

    status = parse_tensors(root, graph.tensors, graph.named_tensors);
    if (status.is_bad()) {
        return status;
    }

    graph.nodes.reserve(field->size());
    for (std::size_t index = 0; index < field->size(); ++index) {
        NodeDesc node;
        status = parse_node((*field)[index], index, graph.tensors,
                            graph.named_tensors, node);
        if (status.is_bad()) {
            return status;
        }
        graph.nodes.push_back(std::move(node));
    }
    status = validate_graph_references(graph);
    if (status.is_bad()) {
        return status;
    }

    auto const* conv = graph.single_conv_fprop();
    if (conv != nullptr) {
        if ((graph.context.is_dynamic_shape_enabled &&
             *graph.context.is_dynamic_shape_enabled) ||
            (graph.context.is_override_shape_enabled &&
             *graph.context.is_override_shape_enabled)) {
            return fail(ErrorCode::kUnsupportedExecutionMetadata, "context",
                        "validated CONV_FPROP requires static shapes");
        }
        for (auto const* type : {&graph.context.compute_data_type,
                                 &graph.context.intermediate_data_type,
                                 &graph.context.io_data_type}) {
            if (*type && **type != DataType::kFloat32) {
                return fail(ErrorCode::kUnsupportedDataType, "context",
                            "validated CONV_FPROP requires FLOAT context types");
            }
        }
        status = validate_shapes(graph, *conv);
        if (status.is_bad()) {
            return status;
        }
        status = validate_conv_metadata(root, *conv);
        if (status.is_bad()) {
            return status;
        }
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
