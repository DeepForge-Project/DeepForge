#include "SpecializedAttention.h"

#include "Numeric.h"

#include "DeepForge/Import/Capability.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <map>
#include <optional>
#include <set>
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

bool read_number(SerializedValue const& value, double& output) {
    if (auto const* number = std::get_if<double>(&value.value)) {
        output = *number;
        return true;
    }
    std::int64_t integer = 0;
    if (!read_integer(value, integer)) return false;
    output = static_cast<double>(integer);
    return true;
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

bool read_optional_number_attribute(GenericOperationDesc const& operation,
                                    std::string_view name,
                                    std::optional<double>& output) {
    auto const* value = attribute(operation, name);
    if (value == nullptr) return false;
    if (std::holds_alternative<std::nullptr_t>(value->value)) {
        output.reset();
        return true;
    }
    double number = 0.0;
    if (!read_number(*value, number)) return false;
    output = number;
    return true;
}

struct Port {
    std::int64_t uid = 0;
    TensorDesc const* tensor = nullptr;
};

struct AttentionDescription {
    bool backward = false;
    bool mxfp8 = false;
    bool generate_stats = false;
    bool bottom_right = false;
    std::optional<std::int64_t> left_bound;
    std::optional<std::int64_t> right_bound;
    std::optional<double> attention_scale_value;
    double rescale_threshold = 0.0;
    std::map<std::string, Port> inputs;
    std::map<std::string, Port> outputs;

    Port const& input(std::string_view name) const {
        return inputs.at(std::string(name));
    }
    Port const& output(std::string_view name) const {
        return outputs.at(std::string(name));
    }
};

bool is_forward(OperationTag tag) {
    return tag == OperationTag::kSdpaFp8Fwd ||
           tag == OperationTag::kSdpaMxfp8Fwd;
}

bool is_mxfp8(OperationTag tag) {
    return tag == OperationTag::kSdpaMxfp8Fwd ||
           tag == OperationTag::kSdpaMxfp8Bwd;
}

bool has_type(DataType type, std::initializer_list<DataType> allowed) {
    return std::find(allowed.begin(), allowed.end(), type) != allowed.end();
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

Status require_type(Port const& port,
                    std::initializer_list<DataType> allowed,
                    std::string const& path) {
    if (has_type(port.tensor->data_type, allowed)) return Status::ok();
    return fail(ErrorCode::kUnsupportedDataType, path,
                std::string(import::data_type_name(port.tensor->data_type)) +
                    " is not supported on this attention port");
}

Status resolve_ports(GenericOperationDesc const& operation,
                     SerializedGraph const& graph,
                     std::string const& path,
                     AttentionDescription& result) {
    auto resolve = [&](auto const& source, auto& destination,
                       std::string_view kind) -> Status {
        for (auto const& [name, reference] : source) {
            auto const* uid = std::get_if<std::int64_t>(&reference);
            if (uid == nullptr) {
                return unsupported(path + "." + std::string(kind) + "." +
                                       name,
                                   "CPU execution requires assigned UIDs");
            }
            auto const it = graph.tensors.find(*uid);
            if (it == graph.tensors.end()) {
                return fail(ErrorCode::kMissingUid,
                            path + "." + std::string(kind) + "." + name,
                            "tensor is unresolved");
            }
            destination.emplace(name, Port{*uid, &it->second});
        }
        return Status::ok();
    };
    auto status = resolve(operation.inputs, result.inputs, "inputs");
    if (status.is_bad()) return status;
    return resolve(operation.outputs, result.outputs, "outputs");
}

Status require_ports(AttentionDescription const& description,
                     std::initializer_list<std::string_view> inputs,
                     std::initializer_list<std::string_view> outputs,
                     std::string const& path) {
    for (auto name : inputs) {
        if (!description.inputs.contains(std::string(name))) {
            return fail(ErrorCode::kInvalidValue, path + ".inputs",
                        std::string(name) + " is required");
        }
    }
    for (auto name : outputs) {
        if (!description.outputs.contains(std::string(name))) {
            return fail(ErrorCode::kInvalidValue, path + ".outputs",
                        std::string(name) + " is required");
        }
    }
    return Status::ok();
}

Status reject_extra_ports(
    AttentionDescription const& description,
    std::initializer_list<std::string_view> allowed_inputs,
    std::initializer_list<std::string_view> allowed_outputs,
    std::string const& path) {
    std::set<std::string_view> input_set(allowed_inputs);
    std::set<std::string_view> output_set(allowed_outputs);
    for (auto const& [name, unused] : description.inputs) {
        if (!input_set.contains(name)) {
            return unsupported(path + ".inputs." + name,
                               "this optional attention feature is deferred "
                               "to C6");
        }
        (void)unused;
    }
    for (auto const& [name, unused] : description.outputs) {
        if (!output_set.contains(name)) {
            return unsupported(path + ".outputs." + name,
                               "this optional attention feature is deferred "
                               "to C6");
        }
        (void)unused;
    }
    return Status::ok();
}

Status validate_scalar_float(Port const& port, std::string const& path) {
    auto status = require_type(port, {DataType::kFloat32}, path);
    if (status.is_bad()) return status;
    if (!is_scalar(*port.tensor)) {
        return fail(ErrorCode::kInvalidShape, path,
                    "tensor must contain exactly one FLOAT element");
    }
    return Status::ok();
}

Status validate_common_attributes(OperationTag tag,
                                  GenericOperationDesc const& operation,
                                  std::string const& path,
                                  AttentionDescription& result) {
    result.backward = !is_forward(tag);
    result.mxfp8 = is_mxfp8(tag);
    bool padding = false;
    bool serialized_mxfp8 = false;
    std::optional<double> dropout;
    std::string_view alignment;
    if (!read_bool_attribute(operation, "padding_mask", padding) ||
        !read_optional_number_attribute(operation, "dropout_probability",
                                        dropout) ||
        !read_optional_integer_attribute(operation, "left_bound",
                                         result.left_bound) ||
        !read_optional_integer_attribute(operation, "right_bound",
                                         result.right_bound) ||
        !read_string_attribute(operation, "diagonal_alignment", alignment) ||
        !read_optional_number_attribute(operation, "attn_scale_value",
                                        result.attention_scale_value) ||
        !read_bool_attribute(operation, "is_mxfp8", serialized_mxfp8)) {
        return fail(ErrorCode::kInvalidValue, path,
                    "serialized attention attributes are malformed");
    }
    if (padding || dropout) {
        return unsupported(path,
                           "C5 attention supports no padding mask or dropout");
    }
    if (serialized_mxfp8 != result.mxfp8) {
        return fail(ErrorCode::kInvalidValue, path + ".is_mxfp8",
                    "attribute does not match the serialized operation tag");
    }
    if (alignment != "TOP_LEFT" && alignment != "BOTTOM_RIGHT") {
        return fail(ErrorCode::kInvalidValue, path + ".diagonal_alignment",
                    "expected TOP_LEFT or BOTTOM_RIGHT");
    }
    result.bottom_right = alignment == "BOTTOM_RIGHT";
    if ((result.left_bound && *result.left_bound <= 0) ||
        (result.right_bound && *result.right_bound < 0)) {
        return fail(ErrorCode::kInvalidValue, path,
                    "left_bound must be positive and right_bound nonnegative");
    }
    if (result.attention_scale_value &&
        !std::isfinite(*result.attention_scale_value)) {
        return fail(ErrorCode::kInvalidValue, path + ".attn_scale_value",
                    "attention scale must be finite");
    }
    if (auto const* threshold = attribute(operation, "rescale_threshold")) {
        if (std::holds_alternative<std::nullptr_t>(threshold->value)) {
            result.rescale_threshold = 0.0;
        } else if (!read_number(*threshold, result.rescale_threshold) ||
                   !std::isfinite(result.rescale_threshold) ||
                   result.rescale_threshold < 0.0) {
            return fail(ErrorCode::kInvalidValue,
                        path + ".rescale_threshold",
                        "threshold must be a finite nonnegative number/null");
        }
    }
    return Status::ok();
}

Status validate_qkv_shapes(AttentionDescription const& description,
                           std::string const& path) {
    auto const& q = *description.input("Q").tensor;
    auto const& k = *description.input("K").tensor;
    auto const& v = *description.input("V").tensor;
    for (auto const* tensor : {&q, &k, &v}) {
        if (tensor->dim.size() != 4 || tensor->stride.back() != 1) {
            return fail(ErrorCode::kInvalidLayout, path,
                        "Q, K, and V must be rank-4 BHSD tensors with unit "
                        "embedding stride");
        }
    }
    if (k.dim[0] != q.dim[0] || v.dim[0] != q.dim[0] ||
        k.dim[3] != q.dim[3] || v.dim[2] != k.dim[2] ||
        q.dim[1] % k.dim[1] != 0 || q.dim[1] % v.dim[1] != 0) {
        return fail(ErrorCode::kInvalidShape, path,
                    "Q/K/V batch, head, sequence, or embedding dimensions "
                    "are inconsistent");
    }
    if (q.dim[2] > k.dim[2] &&
        (description.left_bound ||
         (description.bottom_right && description.right_bound))) {
        return unsupported(path,
                           "windowed attention with Sq > Skv can create "
                           "fully masked rows in the static C5 subset");
    }
    return Status::ok();
}

bool scale_shape_covers(TensorDesc const& scale,
                        std::int64_t batches,
                        std::int64_t heads,
                        std::int64_t rows,
                        std::int64_t columns) {
    return scale.dim.size() == 4 && scale.dim[0] == batches &&
           scale.dim[1] == heads && scale.dim[2] >= rows &&
           scale.dim[3] >= columns;
}

Status validate_mx_scale(Port const& scale,
                         std::int64_t batches,
                         std::int64_t heads,
                         std::int64_t rows,
                         std::int64_t columns,
                         std::string const& path) {
    auto status = require_type(scale, {DataType::kFp8E8M0}, path);
    if (status.is_bad()) return status;
    if (!scale_shape_covers(*scale.tensor, batches, heads, rows, columns)) {
        return fail(ErrorCode::kInvalidShape, path,
                    "MXFP8 scale tensor does not cover its logical block "
                    "coordinates");
    }
    return Status::ok();
}

Status validate_forward(AttentionDescription& description,
                        GenericOperationDesc const& operation,
                        std::string const& path) {
    auto status = require_ports(description, {"Q", "K", "V"}, {"O"},
                                path);
    if (status.is_bad()) return status;
    if (description.mxfp8) {
        status = require_ports(description,
                               {"Descale_Q", "Descale_K", "Descale_V"},
                               {"Stats", "Amax_O"}, path);
        if (status.is_bad()) return status;
        status = reject_extra_ports(
            description,
            {"Q", "K", "V", "Descale_Q", "Descale_K", "Descale_V",
             "Attn_scale"},
            {"O", "Stats", "Amax_O"}, path);
    } else {
        status = require_ports(
            description,
            {"Descale_Q", "Descale_K", "Descale_V", "Descale_S",
             "Scale_S", "Scale_O"},
            {"Amax_S", "Amax_O"}, path);
        if (status.is_bad()) return status;
        status = reject_extra_ports(
            description,
            {"Q", "K", "V", "Descale_Q", "Descale_K", "Descale_V",
             "Descale_S", "Scale_S", "Scale_O", "Attn_scale"},
            {"O", "Stats", "Max", "Sum_exp", "Amax_S", "Amax_O"},
            path);
    }
    if (status.is_bad()) return status;

    bool alibi = false;
    bool unfuse_fma = false;
    if (!read_bool_attribute(operation, "generate_stats",
                             description.generate_stats) ||
        !read_bool_attribute(operation, "alibi_mask", alibi) ||
        !read_bool_attribute(operation, "unfuse_fma", unfuse_fma)) {
        return fail(ErrorCode::kInvalidValue, path,
                    "forward attention booleans are malformed");
    }
    if (alibi) {
        return unsupported(path + ".alibi_mask",
                           "ALiBi is not part of the C5 FP8 subset");
    }
    auto const has_stats = description.outputs.contains("Stats");
    if (description.generate_stats != has_stats ||
        (description.mxfp8 && !has_stats)) {
        return fail(ErrorCode::kInvalidValue, path,
                    "Stats output presence must match generate_stats; MXFP8 "
                    "requires statistics");
    }
    (void)unfuse_fma;

    status = validate_qkv_shapes(description, path);
    if (status.is_bad()) return status;
    for (auto name : {"Q", "K", "V"}) {
        status = require_type(description.input(name),
                              {DataType::kFp8E4M3, DataType::kFp8E5M2},
                              path + "." + name);
        if (status.is_bad()) return status;
    }
    auto const& q = *description.input("Q").tensor;
    auto const& k = *description.input("K").tensor;
    auto const& v = *description.input("V").tensor;
    auto const& o = description.output("O");
    std::vector<std::int64_t> const output_dimensions{
        q.dim[0], q.dim[1], q.dim[2], v.dim[3]};
    if (o.tensor->dim != output_dimensions || o.tensor->stride.back() != 1) {
        return fail(ErrorCode::kInvalidShape, path + ".O",
                    "O must have BHSD dimensions [B,Hq,Sq,Dv]");
    }
    status = description.mxfp8
                 ? require_type(o,
                                {DataType::kFloat16, DataType::kBFloat16,
                                 DataType::kFloat32},
                                path + ".O")
                 : require_type(o,
                                {DataType::kFp8E4M3, DataType::kFp8E5M2},
                                path + ".O");
    if (status.is_bad()) return status;

    if (description.inputs.contains("Attn_scale")) {
        if (description.attention_scale_value) {
            return fail(ErrorCode::kInvalidValue, path,
                        "Attn_scale and attn_scale_value are mutually exclusive");
        }
        status = validate_scalar_float(description.input("Attn_scale"),
                                       path + ".Attn_scale");
        if (status.is_bad()) return status;
    }
    if (description.mxfp8) {
        status = validate_mx_scale(description.input("Descale_Q"), q.dim[0],
                                   q.dim[1], q.dim[2], (q.dim[3] + 31) / 32,
                                   path + ".Descale_Q");
        if (status.is_bad()) return status;
        status = validate_mx_scale(description.input("Descale_K"), k.dim[0],
                                   k.dim[1], k.dim[2], (k.dim[3] + 31) / 32,
                                   path + ".Descale_K");
        if (status.is_bad()) return status;
        status = validate_mx_scale(description.input("Descale_V"), v.dim[0],
                                   v.dim[1], (v.dim[2] + 31) / 32, v.dim[3],
                                   path + ".Descale_V");
        if (status.is_bad()) return status;
    } else {
        for (auto name : {"Descale_Q", "Descale_K", "Descale_V",
                          "Descale_S", "Scale_S", "Scale_O"}) {
            status = validate_scalar_float(description.input(name),
                                           path + "." + name);
            if (status.is_bad()) return status;
        }
    }
    for (auto name : {"Amax_O", "Amax_S"}) {
        if (!description.outputs.contains(name)) continue;
        status = validate_scalar_float(description.output(name),
                                       path + "." + name);
        if (status.is_bad()) return status;
    }
    std::vector<std::int64_t> const row_dimensions{
        q.dim[0], q.dim[1], q.dim[2], 1};
    for (auto name : {"Stats", "Max", "Sum_exp"}) {
        if (!description.outputs.contains(name)) continue;
        auto const& port = description.output(name);
        status = require_type(port, {DataType::kFloat32}, path + "." + name);
        if (status.is_bad()) return status;
        if (port.tensor->dim != row_dimensions) {
            return fail(ErrorCode::kInvalidShape, path + "." + name,
                        "row output must have dimensions [B,Hq,Sq,1]");
        }
    }
    return Status::ok();
}

Status validate_backward(AttentionDescription const& description,
                         GenericOperationDesc const& operation,
                         std::string const& path) {
    std::string_view compute;
    bool deterministic = false;
    if (!read_string_attribute(operation, "compute_data_type", compute) ||
        compute != "FLOAT" ||
        !read_bool_attribute(operation, "is_deterministic_algorithm",
                             deterministic)) {
        return unsupported(path,
                           "CPU FP8 backward requires FLOAT compute metadata");
    }
    (void)deterministic;
    auto status = require_ports(
        description, {"Q", "K", "V", "O", "dO", "Stats"},
        {"dQ", "dK", "dV", "Amax_dQ", "Amax_dK", "Amax_dV"}, path);
    if (status.is_bad()) return status;
    if (description.mxfp8) {
        status = require_ports(
            description,
            {"Q_T", "K_T", "dO_T", "dO_f16", "Descale_Q",
             "Descale_Q_T", "Descale_K", "Descale_K_T", "Descale_V",
             "Descale_dO", "Descale_dO_T"},
            {}, path);
        if (status.is_bad()) return status;
        status = reject_extra_ports(
            description,
            {"Q", "Q_T", "K", "K_T", "V", "O", "dO", "dO_T",
             "dO_f16", "Stats", "Descale_Q", "Descale_Q_T",
             "Descale_K", "Descale_K_T", "Descale_V", "Descale_dO",
             "Descale_dO_T", "Attn_scale"},
            {"dQ", "dK", "dV", "Amax_dQ", "Amax_dK", "Amax_dV"},
            path);
    } else {
        status = require_ports(
            description,
            {"Descale_Q", "Descale_K", "Descale_V", "Descale_O",
             "Descale_dO", "Descale_S", "Descale_dP", "Scale_S",
             "Scale_dP", "Scale_dQ", "Scale_dK", "Scale_dV"},
            {"Amax_dP"}, path);
        if (status.is_bad()) return status;
        status = reject_extra_ports(
            description,
            {"Q", "K", "V", "O", "dO", "Stats", "Descale_Q",
             "Descale_K", "Descale_V", "Descale_O", "Descale_dO",
             "Descale_S", "Descale_dP", "Scale_S", "Scale_dP",
             "Scale_dQ", "Scale_dK", "Scale_dV", "Attn_scale"},
            {"dQ", "dK", "dV", "Amax_dQ", "Amax_dK", "Amax_dV",
             "Amax_dP"},
            path);
    }
    if (status.is_bad()) return status;
    status = validate_qkv_shapes(description, path);
    if (status.is_bad()) return status;
    for (auto name : {"Q", "K", "V", "dO"}) {
        status = require_type(description.input(name),
                              {DataType::kFp8E4M3, DataType::kFp8E5M2},
                              path + "." + name);
        if (status.is_bad()) return status;
    }
    auto const& q = *description.input("Q").tensor;
    auto const& k = *description.input("K").tensor;
    auto const& v = *description.input("V").tensor;
    std::vector<std::int64_t> const o_dimensions{
        q.dim[0], q.dim[1], q.dim[2], v.dim[3]};
    if (description.input("O").tensor->dim != o_dimensions ||
        description.input("dO").tensor->dim != o_dimensions ||
        description.output("dQ").tensor->dim != q.dim ||
        description.output("dK").tensor->dim != k.dim ||
        description.output("dV").tensor->dim != v.dim) {
        return fail(ErrorCode::kInvalidShape, path,
                    "O/dO and gradient dimensions must match forward shapes");
    }
    auto const row_dimensions =
        std::vector<std::int64_t>{q.dim[0], q.dim[1], q.dim[2], 1};
    if (description.input("Stats").tensor->dim != row_dimensions ||
        description.input("Stats").tensor->data_type != DataType::kFloat32) {
        return fail(ErrorCode::kInvalidShape, path + ".Stats",
                    "Stats must be FLOAT [B,Hq,Sq,1]");
    }
    if (description.inputs.contains("Attn_scale")) {
        if (description.attention_scale_value) {
            return fail(ErrorCode::kInvalidValue, path,
                        "Attn_scale and attn_scale_value are mutually exclusive");
        }
        status = validate_scalar_float(description.input("Attn_scale"),
                                       path + ".Attn_scale");
        if (status.is_bad()) return status;
    }

    if (description.mxfp8) {
        for (auto name : {"Q_T", "K_T", "dO_T"}) {
            status = require_type(description.input(name),
                                  {DataType::kFp8E4M3, DataType::kFp8E5M2},
                                  path + "." + name);
            if (status.is_bad()) return status;
        }
        for (auto name : {"O", "dO_f16"}) {
            status = require_type(description.input(name),
                                  {DataType::kFloat16, DataType::kBFloat16,
                                   DataType::kFloat32},
                                  path + "." + name);
            if (status.is_bad()) return status;
        }
        if (description.input("Q_T").tensor->dim != q.dim ||
            description.input("K_T").tensor->dim != k.dim ||
            description.input("dO_T").tensor->dim != o_dimensions ||
            description.input("dO_f16").tensor->dim != o_dimensions) {
            return fail(ErrorCode::kInvalidShape, path,
                        "Q_T, K_T, dO_T, and dO_f16 must match their logical "
                        "Q, K, and dO dimensions");
        }
        for (auto name : {"dQ", "dK", "dV"}) {
            status = require_type(description.output(name),
                                  {DataType::kFloat16, DataType::kBFloat16,
                                   DataType::kFloat32},
                                  path + "." + name);
            if (status.is_bad()) return status;
        }
        struct ScaleRequirement {
            char const* name;
            std::int64_t b;
            std::int64_t h;
            std::int64_t r;
            std::int64_t c;
        };
        std::vector<ScaleRequirement> requirements{
            {"Descale_Q", q.dim[0], q.dim[1], q.dim[2],
             (q.dim[3] + 31) / 32},
            {"Descale_K", k.dim[0], k.dim[1], k.dim[2],
             (k.dim[3] + 31) / 32},
            {"Descale_V", v.dim[0], v.dim[1], v.dim[2],
             (v.dim[3] + 31) / 32},
            {"Descale_dO", q.dim[0], q.dim[1], q.dim[2],
             (v.dim[3] + 31) / 32},
            {"Descale_Q_T", q.dim[0], q.dim[1], (q.dim[2] + 31) / 32,
             q.dim[3]},
            {"Descale_K_T", k.dim[0], k.dim[1], (k.dim[2] + 31) / 32,
             k.dim[3]},
            {"Descale_dO_T", q.dim[0], q.dim[1],
             (q.dim[2] + 31) / 32, v.dim[3]}};
        for (auto const& requirement : requirements) {
            status = validate_mx_scale(
                description.input(requirement.name), requirement.b,
                requirement.h, requirement.r, requirement.c,
                path + "." + requirement.name);
            if (status.is_bad()) return status;
        }
    } else {
        status = require_type(description.input("O"),
                              {DataType::kFp8E4M3, DataType::kFp8E5M2},
                              path + ".O");
        if (status.is_bad()) return status;
        for (auto name : {"dQ", "dK", "dV"}) {
            status = require_type(description.output(name),
                                  {DataType::kFp8E4M3, DataType::kFp8E5M2},
                                  path + "." + name);
            if (status.is_bad()) return status;
        }
        for (auto name : {"Descale_Q", "Descale_K", "Descale_V",
                          "Descale_O", "Descale_dO", "Descale_S",
                          "Descale_dP", "Scale_S", "Scale_dP",
                          "Scale_dQ", "Scale_dK", "Scale_dV"}) {
            status = validate_scalar_float(description.input(name),
                                           path + "." + name);
            if (status.is_bad()) return status;
        }
    }
    for (auto name : {"Amax_dQ", "Amax_dK", "Amax_dV", "Amax_dP"}) {
        if (!description.outputs.contains(name)) continue;
        status = validate_scalar_float(description.output(name),
                                       path + "." + name);
        if (status.is_bad()) return status;
    }
    return Status::ok();
}

Status decode_attention(OperationTag tag,
                        GenericOperationDesc const& operation,
                        SerializedGraph const& graph,
                        std::string const& path,
                        AttentionDescription& output) {
    AttentionDescription result;
    auto status = validate_common_attributes(tag, operation, path, result);
    if (status.is_bad()) return status;
    status = resolve_ports(operation, graph, path, result);
    if (status.is_bad()) return status;
    status = is_forward(tag) ? validate_forward(result, operation, path)
                             : validate_backward(result, operation, path);
    if (status.is_bad()) return status;
    output = std::move(result);
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

template <typename Body>
::mlir::Value reduce_extents(::mlir::OpBuilder& builder,
                             ::mlir::Location location,
                             std::vector<std::int64_t> const& extents,
                             ::mlir::Value initial,
                             Body&& body) {
    llvm::SmallVector<::mlir::Value> lower;
    llvm::SmallVector<::mlir::Value> upper;
    llvm::SmallVector<::mlir::Value> step;
    for (auto extent : extents) {
        lower.push_back(index_constant(builder, location, 0));
        upper.push_back(index_constant(builder, location, extent));
        step.push_back(index_constant(builder, location, 1));
    }
    auto reduction = ::mlir::scf::buildLoopNest(
        builder, location, lower, upper, step, ::mlir::ValueRange{initial},
        [&](::mlir::OpBuilder& reduction_builder,
            ::mlir::Location reduction_location,
            ::mlir::ValueRange indices,
            ::mlir::ValueRange iter_args) -> ::mlir::scf::ValueVector {
            return {body(reduction_builder, reduction_location, indices,
                         iter_args.front())};
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

struct RuntimeScales {
    ::mlir::Value attention;
    std::map<std::string, ::mlir::Value> scalar;
};

::mlir::Value load_port(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    Port const& port,
    llvm::SmallVector<::mlir::Value> const& indices,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    return numeric::load_as_f32(builder, location, values.at(port.uid),
                                *port.tensor, indices);
}

::mlir::Value load_scalar(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    Port const& port,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    return load_port(builder, location, port,
                     zero_indices(builder, location, *port.tensor), values);
}

RuntimeScales prepare_scales(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    AttentionDescription const& description,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    RuntimeScales result;
    if (description.attention_scale_value) {
        result.attention = float_constant(
            builder, location,
            static_cast<float>(*description.attention_scale_value));
    } else if (description.inputs.contains("Attn_scale")) {
        result.attention = load_scalar(
            builder, location, description.input("Attn_scale"), values);
    } else {
        result.attention = float_constant(builder, location, 1.0F);
    }
    if (!description.mxfp8) {
        for (auto const& [name, port] : description.inputs) {
            if (name.starts_with("Descale_") || name.starts_with("Scale_")) {
                result.scalar.emplace(
                    name, load_scalar(builder, location, port, values));
            }
        }
    }
    return result;
}

enum class MxScaleAxis { kEmbedding, kSequence };

::mlir::Value load_mx_value(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    Port const& data,
    Port const& scale,
    MxScaleAxis axis,
    llvm::SmallVector<::mlir::Value> const& indices,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    auto scale_indices = indices;
    auto thirty_two = index_constant(builder, location, 32);
    if (axis == MxScaleAxis::kEmbedding) {
        scale_indices[3] = ::mlir::arith::DivUIOp::create(
            builder, location, indices[3], thirty_two);
    } else {
        scale_indices[2] = ::mlir::arith::DivUIOp::create(
            builder, location, indices[2], thirty_two);
    }
    auto decoded = load_port(builder, location, data, indices, values);
    auto factor = load_port(builder, location, scale, scale_indices, values);
    return ::mlir::arith::MulFOp::create(builder, location, decoded, factor);
}

::mlir::Value load_attention_value(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    AttentionDescription const& description,
    RuntimeScales const& scales,
    std::string_view data_name,
    std::string_view scale_name,
    MxScaleAxis mx_axis,
    llvm::SmallVector<::mlir::Value> const& indices,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    auto const& data = description.input(data_name);
    if (description.mxfp8) {
        return load_mx_value(builder, location, data,
                             description.input(scale_name), mx_axis, indices,
                             values);
    }
    auto decoded = load_port(builder, location, data, indices, values);
    return ::mlir::arith::MulFOp::create(
        builder, location, decoded, scales.scalar.at(std::string(scale_name)));
}

::mlir::Value source_head(::mlir::OpBuilder& builder,
                          ::mlir::Location location,
                          ::mlir::Value query_head,
                          std::int64_t query_heads,
                          std::int64_t source_heads) {
    return ::mlir::arith::DivUIOp::create(
        builder, location, query_head,
        index_constant(builder, location, query_heads / source_heads));
}

::mlir::Value score_is_visible(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    AttentionDescription const& description,
    ::mlir::Value query,
    ::mlir::Value key) {
    ::mlir::Value visible = ::mlir::arith::ConstantIntOp::create(
        builder, location, 1, 1);
    auto const& q = *description.input("Q").tensor;
    auto const& k = *description.input("K").tensor;
    auto diagonal = ::mlir::arith::AddIOp::create(
        builder, location, query,
        index_constant(builder, location,
                       description.bottom_right ? k.dim[2] - q.dim[2] : 0));
    if (description.right_bound) {
        auto right = ::mlir::arith::AddIOp::create(
            builder, location, diagonal,
            index_constant(builder, location, *description.right_bound));
        auto within = ::mlir::arith::CmpIOp::create(
            builder, location, ::mlir::arith::CmpIPredicate::sle, key, right);
        visible = ::mlir::arith::AndIOp::create(builder, location, visible,
                                                within);
    }
    if (description.left_bound) {
        auto left = ::mlir::arith::SubIOp::create(
            builder, location, diagonal,
            index_constant(builder, location, *description.left_bound));
        auto within = ::mlir::arith::CmpIOp::create(
            builder, location, ::mlir::arith::CmpIPredicate::sgt, key, left);
        visible = ::mlir::arith::AndIOp::create(builder, location, visible,
                                                within);
    }
    return visible;
}

::mlir::Value attention_score(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    AttentionDescription const& description,
    RuntimeScales const& scales,
    llvm::SmallVector<::mlir::Value> const& score_indices,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    auto const& q = *description.input("Q").tensor;
    auto const& k = *description.input("K").tensor;
    auto key_head = source_head(builder, location, score_indices[1], q.dim[1],
                                k.dim[1]);
    auto dot = reduce_extent(
        builder, location, q.dim[3],
        float_constant(builder, location, 0.0F),
        [&](::mlir::OpBuilder& reduction_builder,
            ::mlir::Location reduction_location,
            ::mlir::Value embedding,
            ::mlir::Value accumulator) {
            llvm::SmallVector<::mlir::Value> q_indices{
                score_indices[0], score_indices[1], score_indices[2],
                embedding};
            llvm::SmallVector<::mlir::Value> k_indices{
                score_indices[0], key_head, score_indices[3], embedding};
            auto q_value = load_attention_value(
                reduction_builder, reduction_location, description, scales,
                "Q", "Descale_Q", MxScaleAxis::kEmbedding, q_indices,
                values);
            auto k_value = load_attention_value(
                reduction_builder, reduction_location, description, scales,
                "K", "Descale_K", MxScaleAxis::kEmbedding, k_indices,
                values);
            return ::mlir::arith::AddFOp::create(
                reduction_builder, reduction_location, accumulator,
                ::mlir::arith::MulFOp::create(
                    reduction_builder, reduction_location, q_value,
                    k_value));
        });
    dot = ::mlir::arith::MulFOp::create(builder, location, dot,
                                         scales.attention);
    auto visible = score_is_visible(builder, location, description,
                                    score_indices[2], score_indices[3]);
    return ::mlir::arith::SelectOp::create(
        builder, location, visible, dot,
        float_constant(builder, location,
                       -std::numeric_limits<float>::infinity()));
}

struct SoftmaxRow {
    ::mlir::Value maximum;
    ::mlir::Value sum_exp;
};

SoftmaxRow softmax_row(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    AttentionDescription const& description,
    RuntimeScales const& scales,
    llvm::SmallVector<::mlir::Value> const& row_indices,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    auto const key_count = description.input("K").tensor->dim[2];
    auto maximum = reduce_extent(
        builder, location, key_count,
        float_constant(builder, location,
                       -std::numeric_limits<float>::infinity()),
        [&](::mlir::OpBuilder& reduction_builder,
            ::mlir::Location reduction_location,
            ::mlir::Value key,
            ::mlir::Value accumulator) {
            llvm::SmallVector<::mlir::Value> score_indices{
                row_indices[0], row_indices[1], row_indices[2], key};
            return ::mlir::arith::MaximumFOp::create(
                reduction_builder, reduction_location, accumulator,
                attention_score(reduction_builder, reduction_location,
                                description, scales, score_indices, values));
        });
    auto sum = reduce_extent(
        builder, location, key_count,
        float_constant(builder, location, 0.0F),
        [&](::mlir::OpBuilder& reduction_builder,
            ::mlir::Location reduction_location,
            ::mlir::Value key,
            ::mlir::Value accumulator) {
            llvm::SmallVector<::mlir::Value> score_indices{
                row_indices[0], row_indices[1], row_indices[2], key};
            auto score = attention_score(
                reduction_builder, reduction_location, description, scales,
                score_indices, values);
            auto exponential = ::mlir::math::ExpOp::create(
                reduction_builder, reduction_location,
                ::mlir::arith::SubFOp::create(
                    reduction_builder, reduction_location, score, maximum));
            return ::mlir::arith::AddFOp::create(
                reduction_builder, reduction_location, accumulator,
                exponential);
        });
    return {maximum, sum};
}

::mlir::Value probability(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    AttentionDescription const& description,
    RuntimeScales const& scales,
    llvm::SmallVector<::mlir::Value> const& score_indices,
    SoftmaxRow const& row,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    auto score = attention_score(builder, location, description, scales,
                                 score_indices, values);
    auto numerator = ::mlir::math::ExpOp::create(
        builder, location,
        ::mlir::arith::SubFOp::create(builder, location, score, row.maximum));
    return ::mlir::arith::DivFOp::create(builder, location, numerator,
                                         row.sum_exp);
}

::mlir::Value backward_probability(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    AttentionDescription const& description,
    RuntimeScales const& scales,
    llvm::SmallVector<::mlir::Value> const& score_indices,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    auto score = attention_score(builder, location, description, scales,
                                 score_indices, values);
    llvm::SmallVector<::mlir::Value> stats_indices{
        score_indices[0], score_indices[1], score_indices[2],
        index_constant(builder, location, 0)};
    auto stats = load_port(builder, location, description.input("Stats"),
                           stats_indices, values);
    auto result = ::mlir::math::ExpOp::create(
        builder, location,
        ::mlir::arith::SubFOp::create(builder, location, score, stats));
    auto visible = score_is_visible(builder, location, description,
                                    score_indices[2], score_indices[3]);
    return ::mlir::arith::SelectOp::create(
        builder, location, visible, result,
        float_constant(builder, location, 0.0F));
}

::mlir::Value quantized_probability(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    AttentionDescription const& description,
    RuntimeScales const& scales,
    ::mlir::Value value) {
    auto const type = description.input("Q").tensor->data_type;
    if (description.mxfp8) {
        auto scaled = ::mlir::arith::MulFOp::create(
            builder, location, value, float_constant(builder, location, 16.0F));
        auto quantized = numeric::quantize_f32(builder, location, scaled, type);
        return ::mlir::arith::MulFOp::create(
            builder, location, quantized,
            float_constant(builder, location, 1.0F / 16.0F));
    }
    auto scale = scales.scalar.at("Scale_S");
    auto descale = scales.scalar.at("Descale_S");
    if (description.rescale_threshold != 0.0) {
        auto const correction = static_cast<float>(
            std::exp2(-description.rescale_threshold));
        scale = ::mlir::arith::MulFOp::create(
            builder, location, scale,
            float_constant(builder, location, correction));
        descale = ::mlir::arith::MulFOp::create(
            builder, location, descale,
            float_constant(builder, location, 1.0F / correction));
    }
    auto scaled = ::mlir::arith::MulFOp::create(builder, location, value,
                                                 scale);
    auto quantized = numeric::quantize_f32(builder, location, scaled, type);
    return ::mlir::arith::MulFOp::create(builder, location, quantized,
                                         descale);
}

::mlir::Value forward_output_element(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    AttentionDescription const& description,
    RuntimeScales const& scales,
    llvm::SmallVector<::mlir::Value> const& output_indices,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    auto const& q = *description.input("Q").tensor;
    auto const& k = *description.input("K").tensor;
    auto const& v = *description.input("V").tensor;
    llvm::SmallVector<::mlir::Value> row_indices{
        output_indices[0], output_indices[1], output_indices[2]};
    auto row = softmax_row(builder, location, description, scales, row_indices,
                           values);
    auto value_head = source_head(builder, location, output_indices[1],
                                  q.dim[1], v.dim[1]);
    return reduce_extent(
        builder, location, k.dim[2],
        float_constant(builder, location, 0.0F),
        [&](::mlir::OpBuilder& reduction_builder,
            ::mlir::Location reduction_location,
            ::mlir::Value key,
            ::mlir::Value accumulator) {
            llvm::SmallVector<::mlir::Value> score_indices{
                output_indices[0], output_indices[1], output_indices[2], key};
            auto p = probability(reduction_builder, reduction_location,
                                 description, scales, score_indices, row,
                                 values);
            p = quantized_probability(reduction_builder, reduction_location,
                                      description, scales, p);
            llvm::SmallVector<::mlir::Value> v_indices{
                output_indices[0], value_head, key, output_indices[3]};
            auto v_value = load_attention_value(
                reduction_builder, reduction_location, description, scales,
                "V", "Descale_V", MxScaleAxis::kSequence, v_indices,
                values);
            return ::mlir::arith::AddFOp::create(
                reduction_builder, reduction_location, accumulator,
                ::mlir::arith::MulFOp::create(
                    reduction_builder, reduction_location, p, v_value));
        });
}

void store_scalar_output(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    Port const& port,
    ::mlir::Value value,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    numeric::store_from_f32(builder, location, value, values.at(port.uid),
                            *port.tensor,
                            zero_indices(builder, location, *port.tensor));
}

Status emit_forward(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    AttentionDescription const& description,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    auto scales = prepare_scales(builder, location, description, values);
    auto const& q = *description.input("Q").tensor;
    auto const& k = *description.input("K").tensor;
    auto const& o_port = description.output("O");
    auto const& o = *o_port.tensor;
    std::vector<std::int64_t> const row_dimensions{q.dim[0], q.dim[1],
                                                   q.dim[2]};
    if (description.outputs.contains("Stats") ||
        description.outputs.contains("Max") ||
        description.outputs.contains("Sum_exp")) {
        auto status = emit_flat_loop(
            builder, location, row_dimensions, "FP8_SDPA.softmax_rows",
            [&](::mlir::OpBuilder& body_builder,
                ::mlir::Location body_location,
                llvm::SmallVector<::mlir::Value> const& row_indices) {
                auto row = softmax_row(body_builder, body_location,
                                       description, scales, row_indices,
                                       values);
                auto output_indices = row_indices;
                output_indices.push_back(
                    index_constant(body_builder, body_location, 0));
                if (description.outputs.contains("Stats")) {
                    auto stats = ::mlir::arith::AddFOp::create(
                        body_builder, body_location, row.maximum,
                        ::mlir::math::LogOp::create(
                            body_builder, body_location, row.sum_exp));
                    auto const& port = description.output("Stats");
                    numeric::store_from_f32(
                        body_builder, body_location, stats,
                        values.at(port.uid), *port.tensor, output_indices);
                }
                if (description.outputs.contains("Max")) {
                    auto const& port = description.output("Max");
                    numeric::store_from_f32(
                        body_builder, body_location, row.maximum,
                        values.at(port.uid), *port.tensor, output_indices);
                }
                if (description.outputs.contains("Sum_exp")) {
                    auto const& port = description.output("Sum_exp");
                    numeric::store_from_f32(
                        body_builder, body_location, row.sum_exp,
                        values.at(port.uid), *port.tensor, output_indices);
                }
            });
        if (status.is_bad()) return status;
    }

    if (description.outputs.contains("Amax_S")) {
        auto const score_dimensions = std::vector<std::int64_t>{
            q.dim[0], q.dim[1], q.dim[2], k.dim[2]};
        std::uint64_t count = 0;
        (void)checked_element_count(score_dimensions, count);
        auto amax_s = reduce_extent(
            builder, location, static_cast<std::int64_t>(count),
            float_constant(builder, location, 0.0F),
            [&](::mlir::OpBuilder& reduction_builder,
                ::mlir::Location reduction_location,
                ::mlir::Value linear,
                ::mlir::Value accumulator) {
                auto score_indices = logical_indices(
                    reduction_builder, reduction_location, linear,
                    score_dimensions);
                llvm::SmallVector<::mlir::Value> row_indices{
                    score_indices[0], score_indices[1], score_indices[2]};
                auto row = softmax_row(reduction_builder, reduction_location,
                                       description, scales, row_indices,
                                       values);
                auto p = probability(reduction_builder, reduction_location,
                                     description, scales, score_indices, row,
                                     values);
                return ::mlir::arith::MaximumFOp::create(
                    reduction_builder, reduction_location, accumulator,
                    ::mlir::math::AbsFOp::create(
                        reduction_builder, reduction_location, p));
            });
        store_scalar_output(builder, location, description.output("Amax_S"),
                            amax_s, values);
    }

    std::uint64_t output_count = 0;
    (void)checked_element_count(o.dim, output_count);
    auto amax_o = reduce_extent(
        builder, location, static_cast<std::int64_t>(output_count),
        float_constant(builder, location, 0.0F),
        [&](::mlir::OpBuilder& reduction_builder,
            ::mlir::Location reduction_location,
            ::mlir::Value linear,
            ::mlir::Value accumulator) {
            auto indices = logical_indices(reduction_builder,
                                           reduction_location, linear, o.dim);
            auto output = forward_output_element(
                reduction_builder, reduction_location, description, scales,
                indices, values);
            return ::mlir::arith::MaximumFOp::create(
                reduction_builder, reduction_location, accumulator,
                ::mlir::math::AbsFOp::create(
                    reduction_builder, reduction_location, output));
        });
    store_scalar_output(builder, location, description.output("Amax_O"),
                        amax_o, values);

    return emit_flat_loop(
        builder, location, o.dim, "FP8_SDPA.O",
        [&](::mlir::OpBuilder& body_builder,
            ::mlir::Location body_location,
            llvm::SmallVector<::mlir::Value> const& indices) {
            ::mlir::Value output = forward_output_element(
                body_builder, body_location, description, scales, indices,
                values);
            if (!description.mxfp8) {
                output = ::mlir::arith::MulFOp::create(
                    body_builder, body_location, output,
                    scales.scalar.at("Scale_O"));
            }
            numeric::store_from_f32(body_builder, body_location, output,
                                    values.at(o_port.uid), o, indices);
        });
}

::mlir::Value backward_output_dot(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    AttentionDescription const& description,
    RuntimeScales const& scales,
    llvm::SmallVector<::mlir::Value> const& row_indices,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    auto const embedding = description.input("V").tensor->dim[3];
    return reduce_extent(
        builder, location, embedding,
        float_constant(builder, location, 0.0F),
        [&](::mlir::OpBuilder& reduction_builder,
            ::mlir::Location reduction_location,
            ::mlir::Value d,
            ::mlir::Value accumulator) {
            llvm::SmallVector<::mlir::Value> indices{
                row_indices[0], row_indices[1], row_indices[2], d};
            ::mlir::Value o;
            ::mlir::Value d_o;
            if (description.mxfp8) {
                o = load_port(reduction_builder, reduction_location,
                              description.input("O"), indices, values);
                d_o = load_port(reduction_builder, reduction_location,
                                description.input("dO_f16"), indices,
                                values);
            } else {
                o = load_attention_value(
                    reduction_builder, reduction_location, description,
                    scales, "O", "Descale_O", MxScaleAxis::kEmbedding,
                    indices, values);
                d_o = load_attention_value(
                    reduction_builder, reduction_location, description,
                    scales, "dO", "Descale_dO", MxScaleAxis::kEmbedding,
                    indices, values);
            }
            return ::mlir::arith::AddFOp::create(
                reduction_builder, reduction_location, accumulator,
                ::mlir::arith::MulFOp::create(
                    reduction_builder, reduction_location, o, d_o));
        });
}

::mlir::Value backward_d_p(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    AttentionDescription const& description,
    RuntimeScales const& scales,
    llvm::SmallVector<::mlir::Value> const& score_indices,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    auto const& q = *description.input("Q").tensor;
    auto const& v = *description.input("V").tensor;
    auto value_head = source_head(builder, location, score_indices[1],
                                  q.dim[1], v.dim[1]);
    return reduce_extent(
        builder, location, v.dim[3],
        float_constant(builder, location, 0.0F),
        [&](::mlir::OpBuilder& reduction_builder,
            ::mlir::Location reduction_location,
            ::mlir::Value d,
            ::mlir::Value accumulator) {
            llvm::SmallVector<::mlir::Value> do_indices{
                score_indices[0], score_indices[1], score_indices[2], d};
            llvm::SmallVector<::mlir::Value> v_indices{
                score_indices[0], value_head, score_indices[3], d};
            auto d_o = load_attention_value(
                reduction_builder, reduction_location, description, scales,
                "dO", "Descale_dO", MxScaleAxis::kEmbedding, do_indices,
                values);
            auto v_value = load_attention_value(
                reduction_builder, reduction_location, description, scales,
                "V", "Descale_V", MxScaleAxis::kEmbedding, v_indices,
                values);
            return ::mlir::arith::AddFOp::create(
                reduction_builder, reduction_location, accumulator,
                ::mlir::arith::MulFOp::create(
                    reduction_builder, reduction_location, d_o, v_value));
        });
}

::mlir::Value backward_d_s(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    AttentionDescription const& description,
    RuntimeScales const& scales,
    llvm::SmallVector<::mlir::Value> const& score_indices,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    llvm::SmallVector<::mlir::Value> row_indices{
        score_indices[0], score_indices[1], score_indices[2]};
    auto p = backward_probability(builder, location, description, scales,
                                  score_indices, values);
    auto centered = ::mlir::arith::SubFOp::create(
        builder, location,
        backward_d_p(builder, location, description, scales, score_indices,
                     values),
        backward_output_dot(builder, location, description, scales,
                            row_indices, values));
    return ::mlir::arith::MulFOp::create(
        builder, location,
        ::mlir::arith::MulFOp::create(builder, location, p, centered),
        scales.attention);
}

::mlir::Value quantized_d_s(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    AttentionDescription const& description,
    RuntimeScales const& scales,
    llvm::SmallVector<::mlir::Value> const& score_indices,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    auto d_s = backward_d_s(builder, location, description, scales,
                            score_indices, values);
    if (description.mxfp8) {
        // The MXFP8 backend requantizes dS per 32-element block. Keeping dS
        // in f32 is the CPU reference approximation allowed by the C5 policy.
        return d_s;
    }
    auto scaled = ::mlir::arith::MulFOp::create(
        builder, location, d_s, scales.scalar.at("Scale_dP"));
    auto quantized = numeric::quantize_f32(
        builder, location, scaled, description.input("Q").tensor->data_type);
    return ::mlir::arith::MulFOp::create(
        builder, location, quantized, scales.scalar.at("Descale_dP"));
}

::mlir::Value backward_quantized_probability(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    AttentionDescription const& description,
    RuntimeScales const& scales,
    llvm::SmallVector<::mlir::Value> const& score_indices,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    auto p = backward_probability(builder, location, description, scales,
                                  score_indices, values);
    if (!description.mxfp8) {
        return quantized_probability(builder, location, description, scales,
                                     p);
    }
    auto scaled = ::mlir::arith::MulFOp::create(
        builder, location, p, float_constant(builder, location, 256.0F));
    auto quantized = numeric::quantize_f32(
        builder, location, scaled, description.input("Q").tensor->data_type);
    return ::mlir::arith::MulFOp::create(
        builder, location, quantized,
        float_constant(builder, location, 1.0F / 256.0F));
}

::mlir::Value d_q_element(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    AttentionDescription const& description,
    RuntimeScales const& scales,
    llvm::SmallVector<::mlir::Value> const& output_indices,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    auto const& q = *description.input("Q").tensor;
    auto const& k = *description.input("K").tensor;
    auto key_head = source_head(builder, location, output_indices[1], q.dim[1],
                                k.dim[1]);
    return reduce_extent(
        builder, location, k.dim[2],
        float_constant(builder, location, 0.0F),
        [&](::mlir::OpBuilder& reduction_builder,
            ::mlir::Location reduction_location,
            ::mlir::Value key,
            ::mlir::Value accumulator) {
            llvm::SmallVector<::mlir::Value> score_indices{
                output_indices[0], output_indices[1], output_indices[2], key};
            auto d_s = quantized_d_s(reduction_builder, reduction_location,
                                     description, scales, score_indices,
                                     values);
            llvm::SmallVector<::mlir::Value> k_indices{
                output_indices[0], key_head, key, output_indices[3]};
            ::mlir::Value k_value;
            if (description.mxfp8) {
                k_value = load_mx_value(
                    reduction_builder, reduction_location,
                    description.input("K_T"),
                    description.input("Descale_K_T"),
                    MxScaleAxis::kSequence, k_indices, values);
            } else {
                k_value = load_attention_value(
                    reduction_builder, reduction_location, description,
                    scales, "K", "Descale_K", MxScaleAxis::kEmbedding,
                    k_indices, values);
            }
            return ::mlir::arith::AddFOp::create(
                reduction_builder, reduction_location, accumulator,
                ::mlir::arith::MulFOp::create(
                    reduction_builder, reduction_location, d_s, k_value));
        });
}

::mlir::Value d_k_element(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    AttentionDescription const& description,
    RuntimeScales const& scales,
    llvm::SmallVector<::mlir::Value> const& output_indices,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    auto const& q = *description.input("Q").tensor;
    auto const& k = *description.input("K").tensor;
    auto const group = q.dim[1] / k.dim[1];
    return reduce_extents(
        builder, location, {group, q.dim[2]},
        float_constant(builder, location, 0.0F),
        [&](::mlir::OpBuilder& reduction_builder,
            ::mlir::Location reduction_location,
            ::mlir::ValueRange reduction_indices,
            ::mlir::Value accumulator) {
            auto query_head = ::mlir::arith::AddIOp::create(
                reduction_builder, reduction_location,
                ::mlir::arith::MulIOp::create(
                    reduction_builder, reduction_location, output_indices[1],
                    index_constant(reduction_builder, reduction_location,
                                   group)),
                reduction_indices[0]);
            llvm::SmallVector<::mlir::Value> score_indices{
                output_indices[0], query_head, reduction_indices[1],
                output_indices[2]};
            auto d_s = quantized_d_s(reduction_builder, reduction_location,
                                     description, scales, score_indices,
                                     values);
            llvm::SmallVector<::mlir::Value> q_indices{
                output_indices[0], query_head, reduction_indices[1],
                output_indices[3]};
            ::mlir::Value q_value;
            if (description.mxfp8) {
                q_value = load_mx_value(
                    reduction_builder, reduction_location,
                    description.input("Q_T"),
                    description.input("Descale_Q_T"),
                    MxScaleAxis::kSequence, q_indices, values);
            } else {
                q_value = load_attention_value(
                    reduction_builder, reduction_location, description,
                    scales, "Q", "Descale_Q", MxScaleAxis::kEmbedding,
                    q_indices, values);
            }
            return ::mlir::arith::AddFOp::create(
                reduction_builder, reduction_location, accumulator,
                ::mlir::arith::MulFOp::create(
                    reduction_builder, reduction_location, d_s, q_value));
        });
}

::mlir::Value d_v_element(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    AttentionDescription const& description,
    RuntimeScales const& scales,
    llvm::SmallVector<::mlir::Value> const& output_indices,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    auto const& q = *description.input("Q").tensor;
    auto const& v = *description.input("V").tensor;
    auto const group = q.dim[1] / v.dim[1];
    return reduce_extents(
        builder, location, {group, q.dim[2]},
        float_constant(builder, location, 0.0F),
        [&](::mlir::OpBuilder& reduction_builder,
            ::mlir::Location reduction_location,
            ::mlir::ValueRange reduction_indices,
            ::mlir::Value accumulator) {
            auto query_head = ::mlir::arith::AddIOp::create(
                reduction_builder, reduction_location,
                ::mlir::arith::MulIOp::create(
                    reduction_builder, reduction_location, output_indices[1],
                    index_constant(reduction_builder, reduction_location,
                                   group)),
                reduction_indices[0]);
            llvm::SmallVector<::mlir::Value> score_indices{
                output_indices[0], query_head, reduction_indices[1],
                output_indices[2]};
            auto p = backward_quantized_probability(
                reduction_builder, reduction_location, description, scales,
                score_indices, values);
            llvm::SmallVector<::mlir::Value> do_indices{
                output_indices[0], query_head, reduction_indices[1],
                output_indices[3]};
            ::mlir::Value d_o;
            if (description.mxfp8) {
                d_o = load_mx_value(
                    reduction_builder, reduction_location,
                    description.input("dO_T"),
                    description.input("Descale_dO_T"),
                    MxScaleAxis::kSequence, do_indices, values);
            } else {
                d_o = load_attention_value(
                    reduction_builder, reduction_location, description,
                    scales, "dO", "Descale_dO",
                    MxScaleAxis::kEmbedding, do_indices, values);
            }
            return ::mlir::arith::AddFOp::create(
                reduction_builder, reduction_location, accumulator,
                ::mlir::arith::MulFOp::create(
                    reduction_builder, reduction_location, p, d_o));
        });
}

template <typename Body>
Status emit_gradient(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    AttentionDescription const& description,
    RuntimeScales const& scales,
    std::string_view output_name,
    std::string_view amax_name,
    std::string_view scale_name,
    std::map<std::int64_t, ::mlir::Value> const& values,
    Body&& body) {
    auto const& output = description.output(output_name);
    std::uint64_t count = 0;
    (void)checked_element_count(output.tensor->dim, count);
    auto amax = reduce_extent(
        builder, location, static_cast<std::int64_t>(count),
        float_constant(builder, location, 0.0F),
        [&](::mlir::OpBuilder& reduction_builder,
            ::mlir::Location reduction_location,
            ::mlir::Value linear,
            ::mlir::Value accumulator) {
            auto indices = logical_indices(
                reduction_builder, reduction_location, linear,
                output.tensor->dim);
            auto raw = body(reduction_builder, reduction_location, indices);
            return ::mlir::arith::MaximumFOp::create(
                reduction_builder, reduction_location, accumulator,
                ::mlir::math::AbsFOp::create(
                    reduction_builder, reduction_location, raw));
        });
    store_scalar_output(builder, location, description.output(amax_name), amax,
                        values);
    return emit_flat_loop(
        builder, location, output.tensor->dim,
        "FP8_SDPA_BWD." + std::string(output_name),
        [&](::mlir::OpBuilder& body_builder,
            ::mlir::Location body_location,
            llvm::SmallVector<::mlir::Value> const& indices) {
            ::mlir::Value raw = body(body_builder, body_location, indices);
            if (!description.mxfp8) {
                raw = ::mlir::arith::MulFOp::create(
                    body_builder, body_location, raw,
                    scales.scalar.at(std::string(scale_name)));
            }
            numeric::store_from_f32(body_builder, body_location, raw,
                                    values.at(output.uid), *output.tensor,
                                    indices);
        });
}

Status emit_backward(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    AttentionDescription const& description,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    auto scales = prepare_scales(builder, location, description, values);
    if (description.outputs.contains("Amax_dP")) {
        auto const& q = *description.input("Q").tensor;
        auto const& k = *description.input("K").tensor;
        auto dimensions =
            std::vector<std::int64_t>{q.dim[0], q.dim[1], q.dim[2], k.dim[2]};
        std::uint64_t count = 0;
        (void)checked_element_count(dimensions, count);
        auto amax = reduce_extent(
            builder, location, static_cast<std::int64_t>(count),
            float_constant(builder, location, 0.0F),
            [&](::mlir::OpBuilder& reduction_builder,
                ::mlir::Location reduction_location,
                ::mlir::Value linear,
                ::mlir::Value accumulator) {
                auto indices = logical_indices(
                    reduction_builder, reduction_location, linear,
                    dimensions);
                auto d_p = backward_d_p(reduction_builder, reduction_location,
                                        description, scales, indices, values);
                return ::mlir::arith::MaximumFOp::create(
                    reduction_builder, reduction_location, accumulator,
                    ::mlir::math::AbsFOp::create(
                        reduction_builder, reduction_location, d_p));
            });
        store_scalar_output(builder, location,
                            description.output("Amax_dP"), amax, values);
    }

    auto status = emit_gradient(
        builder, location, description, scales, "dQ", "Amax_dQ",
        "Scale_dQ", values,
        [&](::mlir::OpBuilder& body_builder,
            ::mlir::Location body_location,
            llvm::SmallVector<::mlir::Value> const& indices) {
            return d_q_element(body_builder, body_location, description,
                               scales, indices, values);
        });
    if (status.is_bad()) return status;
    status = emit_gradient(
        builder, location, description, scales, "dK", "Amax_dK",
        "Scale_dK", values,
        [&](::mlir::OpBuilder& body_builder,
            ::mlir::Location body_location,
            llvm::SmallVector<::mlir::Value> const& indices) {
            return d_k_element(body_builder, body_location, description,
                               scales, indices, values);
        });
    if (status.is_bad()) return status;
    return emit_gradient(
        builder, location, description, scales, "dV", "Amax_dV",
        "Scale_dV", values,
        [&](::mlir::OpBuilder& body_builder,
            ::mlir::Location body_location,
            llvm::SmallVector<::mlir::Value> const& indices) {
            return d_v_element(body_builder, body_location, description,
                               scales, indices, values);
        });
}

}  // namespace

Status validate_specialized_attention(OperationTag tag,
                                      GenericOperationDesc const& operation,
                                      SerializedGraph const& graph,
                                      std::size_t node_index) {
    AttentionDescription description;
    return decode_attention(tag, operation, graph,
                            "nodes[" + std::to_string(node_index) + "]",
                            description);
}

Status emit_specialized_attention(
    OperationTag tag,
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    GenericOperationDesc const& operation,
    SerializedGraph const& graph,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    AttentionDescription description;
    auto status = decode_attention(tag, operation, graph, "attention",
                                   description);
    if (status.is_bad()) return status;
    return is_forward(tag)
               ? emit_forward(builder, location, description, values)
               : emit_backward(builder, location, description, values);
}

}  // namespace deepforge::compiler
