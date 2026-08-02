#include "SequenceGraph.h"

#include "DeepForge/Import/Capability.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
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
        *integer <=
            static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
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
    if (!read_integer(value, integer)) {
        return false;
    }
    output = static_cast<double>(integer);
    return true;
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

bool read_number_attribute(GenericOperationDesc const& operation,
                           std::string_view name,
                           double& output) {
    auto const* value = attribute(operation, name);
    return value != nullptr && read_number(*value, output);
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
    double number = 0.0;
    if (!read_number(*value, number)) {
        return false;
    }
    output = number;
    return true;
}

bool read_optional_string_attribute(GenericOperationDesc const& operation,
                                    std::string_view name,
                                    std::optional<std::string_view>& output) {
    auto const* value = attribute(operation, name);
    if (value == nullptr) {
        return false;
    }
    if (std::holds_alternative<std::nullptr_t>(value->value)) {
        output.reset();
        return true;
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
    auto const* integer = std::get_if<std::int64_t>(&it->second);
    if (integer == nullptr) {
        return false;
    }
    uid = *integer;
    return true;
}

bool has_input(GenericOperationDesc const& operation, std::string_view port) {
    return operation.inputs.contains(std::string(port));
}

bool has_output(GenericOperationDesc const& operation, std::string_view port) {
    return operation.outputs.contains(std::string(port));
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

bool checked_ragged_storage_elements(TensorDesc const& tensor,
                                     std::int64_t& output) {
    if (tensor.dim.empty() || tensor.dim.front() <= 0) return false;
    std::uint64_t inner_span = 1;
    for (std::size_t axis = 1; axis < tensor.dim.size(); ++axis) {
        if (tensor.dim[axis] <= 0 || tensor.stride[axis] <= 0) return false;
        auto const extent = static_cast<std::uint64_t>(tensor.dim[axis] - 1);
        auto const stride = static_cast<std::uint64_t>(tensor.stride[axis]);
        if (extent >
            (std::numeric_limits<std::uint64_t>::max() - inner_span) /
                stride) {
            return false;
        }
        inner_span += extent * stride;
    }
    auto const batches = static_cast<std::uint64_t>(tensor.dim.front());
    if (batches > static_cast<std::uint64_t>(
                      std::numeric_limits<std::int64_t>::max()) /
                      inner_span) {
        return false;
    }
    output = static_cast<std::int64_t>(batches * inner_span);
    return true;
}

bool is_scalar(TensorDesc const& tensor) {
    std::uint64_t count = 0;
    return checked_element_count(tensor.dim, count) && count == 1;
}

bool broadcast_compatible(TensorDesc const& input,
                          std::vector<std::int64_t> const& output_dimensions) {
    if (input.dim.size() > output_dimensions.size()) {
        return false;
    }
    auto const offset = output_dimensions.size() - input.dim.size();
    for (std::size_t index = 0; index < input.dim.size(); ++index) {
        if (input.dim[index] != 1 &&
            input.dim[index] != output_dimensions[offset + index]) {
            return false;
        }
    }
    return true;
}

Status require_tensor(GenericOperationDesc const& operation,
                      SerializedGraph const& graph,
                      bool input,
                      std::string_view port,
                      DataType data_type,
                      std::string const& path,
                      std::int64_t& uid,
                      TensorDesc const*& tensor) {
    if (!tensor_uid(operation, input, port, uid)) {
        return fail(ErrorCode::kInvalidValue, path,
                    std::string(input ? "input " : "output ") +
                        std::string(port) + " is required with an assigned UID");
    }
    auto const tensor_it = graph.tensors.find(uid);
    if (tensor_it == graph.tensors.end()) {
        return fail(ErrorCode::kMissingUid, path,
                    std::string(port) + " tensor is unresolved");
    }
    tensor = &tensor_it->second;
    if (tensor->data_type != data_type) {
        return fail(ErrorCode::kUnsupportedDataType, path + "." +
                                                      std::string(port),
                    "expected " +
                        std::string(import::data_type_name(data_type)));
    }
    return Status::ok();
}

enum class DropoutKind { kNone, kInternal, kCustom };

struct DropoutDescription {
    DropoutKind kind = DropoutKind::kNone;
    double probability = 0.0;
    std::int64_t seed_uid = 0;
    std::int64_t offset_uid = 0;
    std::int64_t mask_uid = 0;
    std::int64_t scale_uid = 0;
    std::int64_t scale_inv_uid = 0;
    TensorDesc const* seed = nullptr;
    TensorDesc const* offset = nullptr;
    TensorDesc const* mask = nullptr;
    TensorDesc const* scale = nullptr;
    TensorDesc const* scale_inv = nullptr;
};

struct AttentionDescription {
    bool backward = false;
    std::int64_t q_uid = 0;
    std::int64_t k_uid = 0;
    std::int64_t v_uid = 0;
    std::int64_t o_uid = 0;
    std::int64_t do_uid = 0;
    std::int64_t stats_uid = 0;
    std::int64_t dq_uid = 0;
    std::int64_t dk_uid = 0;
    std::int64_t dv_uid = 0;
    std::int64_t bias_uid = 0;
    std::int64_t dbias_uid = 0;
    std::int64_t attn_scale_uid = 0;
    std::int64_t seq_q_uid = 0;
    std::int64_t seq_kv_uid = 0;
    std::int64_t page_k_uid = 0;
    std::int64_t page_v_uid = 0;
    std::int64_t block_mask_uid = 0;
    std::int64_t sink_uid = 0;
    std::int64_t dsink_uid = 0;
    std::int64_t rng_dump_uid = 0;
    TensorDesc const* q = nullptr;
    TensorDesc const* k = nullptr;
    TensorDesc const* v = nullptr;
    TensorDesc const* o = nullptr;
    TensorDesc const* d_o = nullptr;
    TensorDesc const* stats = nullptr;
    TensorDesc const* d_q = nullptr;
    TensorDesc const* d_k = nullptr;
    TensorDesc const* d_v = nullptr;
    TensorDesc const* bias = nullptr;
    TensorDesc const* d_bias = nullptr;
    TensorDesc const* attn_scale = nullptr;
    TensorDesc const* seq_q = nullptr;
    TensorDesc const* seq_kv = nullptr;
    TensorDesc const* page_k = nullptr;
    TensorDesc const* page_v = nullptr;
    TensorDesc const* block_mask = nullptr;
    TensorDesc const* sink = nullptr;
    TensorDesc const* d_sink = nullptr;
    TensorDesc const* rng_dump = nullptr;
    std::vector<std::pair<std::int64_t, TensorDesc const*>> row_outputs;
    bool generate_stats = false;
    bool alibi = false;
    bool padding = false;
    bool bottom_right = false;
    std::optional<std::int64_t> left_bound;
    std::optional<std::int64_t> right_bound;
    std::optional<double> attn_scale_value;
    std::int64_t s_kv = 0;
    DropoutDescription dropout;
};

Status validate_scalar_port(GenericOperationDesc const& operation,
                            SerializedGraph const& graph,
                            bool input,
                            std::string_view port,
                            DataType data_type,
                            std::string const& path,
                            std::int64_t& uid,
                            TensorDesc const*& tensor) {
    auto status = require_tensor(operation, graph, input, port, data_type, path,
                                 uid, tensor);
    if (status.is_bad()) {
        return status;
    }
    if (!is_scalar(*tensor)) {
        return fail(ErrorCode::kInvalidShape, path + "." + std::string(port),
                    "tensor must contain exactly one element");
    }
    return Status::ok();
}

Status decode_dropout(GenericOperationDesc const& operation,
                      SerializedGraph const& graph,
                      bool backward,
                      std::vector<std::int64_t> const& score_dimensions,
                      std::string const& path,
                      DropoutDescription& output) {
    std::optional<double> probability;
    if (!read_optional_number_attribute(operation, "dropout_probability",
                                        probability)) {
        return fail(ErrorCode::kInvalidValue, path + ".dropout_probability",
                    "expected a number or null");
    }
    if (probability &&
        (!std::isfinite(*probability) || *probability < 0.0 ||
         *probability >= 1.0)) {
        return fail(ErrorCode::kInvalidValue, path + ".dropout_probability",
                    "probability must be finite and in [0, 1)");
    }

    auto const has_custom = has_input(operation, "Dropout_mask");
    if (probability && has_custom) {
        return fail(ErrorCode::kInvalidValue, path,
                    "internal and custom dropout cannot be combined");
    }
    DropoutDescription result;
    if (probability) {
        result.kind = DropoutKind::kInternal;
        result.probability = *probability;
        auto status = validate_scalar_port(operation, graph, true, "Seed",
                                           DataType::kInt64, path,
                                           result.seed_uid, result.seed);
        if (status.is_bad()) return status;
        status = validate_scalar_port(operation, graph, true, "Offset",
                                      DataType::kInt64, path,
                                      result.offset_uid, result.offset);
        if (status.is_bad()) return status;
        for (auto port : {"Dropout_scale", "Dropout_scale_inv"}) {
            if (has_input(operation, port)) {
                return fail(ErrorCode::kInvalidValue, path,
                            std::string(port) +
                                " is implicit for probability dropout");
            }
        }
    } else if (has_custom) {
        result.kind = DropoutKind::kCustom;
        auto status = require_tensor(operation, graph, true, "Dropout_mask",
                                     DataType::kFloat32, path, result.mask_uid,
                                     result.mask);
        if (status.is_bad()) return status;
        if (!broadcast_compatible(*result.mask, score_dimensions)) {
            return fail(ErrorCode::kInvalidShape, path + ".Dropout_mask",
                        "mask is not broadcast-compatible with attention scores");
        }
        status = validate_scalar_port(operation, graph, true, "Dropout_scale",
                                      DataType::kFloat32, path,
                                      result.scale_uid, result.scale);
        if (status.is_bad()) return status;
        if (backward) {
            status = validate_scalar_port(
                operation, graph, true, "Dropout_scale_inv",
                DataType::kFloat32, path, result.scale_inv_uid,
                result.scale_inv);
            if (status.is_bad()) return status;
        } else if (has_input(operation, "Dropout_scale_inv")) {
            return fail(ErrorCode::kInvalidValue, path,
                        "forward custom dropout does not accept scale_inv");
        }
        if (has_input(operation, "Seed") || has_input(operation, "Offset")) {
            return fail(ErrorCode::kInvalidValue, path,
                        "custom dropout does not accept Seed or Offset");
        }
    } else {
        for (auto port : {"Seed", "Offset", "Dropout_scale",
                          "Dropout_scale_inv"}) {
            if (has_input(operation, port)) {
                return fail(ErrorCode::kInvalidValue, path,
                            std::string(port) +
                                " requires a configured dropout mode");
            }
        }
    }
    output = result;
    return Status::ok();
}

Status decode_attention(OperationTag tag,
                        GenericOperationDesc const& operation,
                        SerializedGraph const& graph,
                        std::string const& path,
                        AttentionDescription& output) {
    AttentionDescription result;
    result.backward = tag == OperationTag::kSdpaBwd;

    auto require = [&](bool input, std::string_view port, std::int64_t& uid,
                       TensorDesc const*& tensor) {
        return require_tensor(operation, graph, input, port,
                              DataType::kFloat32, path, uid, tensor);
    };
    auto status = require(true, "Q", result.q_uid, result.q);
    if (status.is_bad()) return status;
    status = require(true, "K", result.k_uid, result.k);
    if (status.is_bad()) return status;
    status = require(true, "V", result.v_uid, result.v);
    if (status.is_bad()) return status;

    if (result.backward) {
        status = require(true, "O", result.o_uid, result.o);
        if (status.is_bad()) return status;
        status = require(true, "dO", result.do_uid, result.d_o);
        if (status.is_bad()) return status;
        status = require(true, "Stats", result.stats_uid, result.stats);
        if (status.is_bad()) return status;
        status = require(false, "dQ", result.dq_uid, result.d_q);
        if (status.is_bad()) return status;
        status = require(false, "dK", result.dk_uid, result.d_k);
        if (status.is_bad()) return status;
        status = require(false, "dV", result.dv_uid, result.d_v);
        if (status.is_bad()) return status;
    } else {
        status = require(false, "O", result.o_uid, result.o);
        if (status.is_bad()) return status;
    }

    for (auto const* tensor : {result.q, result.k, result.v}) {
        if (tensor->dim.size() != 4 || tensor->stride.back() != 1) {
            return fail(ErrorCode::kInvalidLayout, path,
                        "Q, K, and V must be rank-4 tensors with unit "
                        "embedding stride");
        }
    }
    auto const b = result.q->dim[0];
    auto const h_q = result.q->dim[1];
    auto const s_q = result.q->dim[2];
    auto const d_qk = result.q->dim[3];
    auto const h_k = result.k->dim[1];
    auto const h_v = result.v->dim[1];
    auto const d_v = result.v->dim[3];

    if (has_input(operation, "Bias")) {
        status = require(true, "Bias", result.bias_uid, result.bias);
        if (status.is_bad()) return status;
    }
    if (has_input(operation, "SINK_TOKEN")) {
        status = require(true, "SINK_TOKEN", result.sink_uid, result.sink);
        if (status.is_bad()) return status;
    }
    if (result.sink != nullptr) {
        std::vector<std::int64_t> const expected{1, h_q, 1, 1};
        if (result.sink->dim != expected) {
            return fail(ErrorCode::kInvalidShape, path + ".SINK_TOKEN",
                        "sink token must have dimensions [1,Hq,1,1]");
        }
        if (result.sink->is_virtual || result.sink->is_pass_by_value ||
            result.sink->pass_by_value || result.sink->ragged_offset_uid ||
            result.sink->reordering_type != "NONE") {
            return unsupported(path + ".SINK_TOKEN",
                               "sink token must be an external plain FLOAT "
                               "tensor");
        }
    }
    if (result.backward && has_output(operation, "DSINK_TOKEN")) {
        if (result.sink == nullptr) {
            return fail(ErrorCode::kInvalidValue, path + ".DSINK_TOKEN",
                        "sink gradient requires a SINK_TOKEN input");
        }
        status = require(false, "DSINK_TOKEN", result.dsink_uid,
                         result.d_sink);
        if (status.is_bad()) return status;
        if (result.d_sink->dim != result.sink->dim) {
            return fail(ErrorCode::kInvalidShape, path + ".DSINK_TOKEN",
                        "sink gradient must have dimensions [1,Hq,1,1]");
        }
        if (result.d_sink->is_virtual || result.d_sink->is_pass_by_value ||
            result.d_sink->pass_by_value ||
            result.d_sink->ragged_offset_uid ||
            result.d_sink->reordering_type != "NONE") {
            return unsupported(path + ".DSINK_TOKEN",
                               "sink gradient must be an external plain "
                               "FLOAT tensor");
        }
    }
    if (!result.backward && has_output(operation, "RNG_DUMP")) {
        status = require(false, "RNG_DUMP", result.rng_dump_uid,
                         result.rng_dump);
        if (status.is_bad()) return status;
    }

    std::optional<std::int64_t> max_seq_len;
    if (result.backward) {
        if (has_input(operation, "Page_table_K") ||
            has_input(operation, "Page_table_V")) {
            return unsupported(path,
                               "paged SDPA backward is deferred to a later "
                               "C6 attention increment");
        }
    } else {
        if (!read_optional_integer_attribute(operation, "max_seq_len_kv",
                                             max_seq_len)) {
            return fail(ErrorCode::kInvalidValue, path + ".max_seq_len_kv",
                        "expected an integer or null");
        }
        if (max_seq_len && *max_seq_len <= 0) {
            return fail(ErrorCode::kInvalidValue, path + ".max_seq_len_kv",
                        "paged maximum sequence length must be positive");
        }
        if (has_input(operation, "Page_table_K")) {
            status = require_tensor(operation, graph, true, "Page_table_K",
                                    DataType::kInt32, path,
                                    result.page_k_uid, result.page_k);
            if (status.is_bad()) return status;
        }
        if (has_input(operation, "Page_table_V")) {
            status = require_tensor(operation, graph, true, "Page_table_V",
                                    DataType::kInt32, path,
                                    result.page_v_uid, result.page_v);
            if (status.is_bad()) return status;
        }
    }

    auto validate_page_table = [&](TensorDesc const* table,
                                   std::string_view port) -> Status {
        if (table == nullptr) return Status::ok();
        std::vector<std::int64_t> const expected_prefix{b, 1};
        if (table->dim.size() != 4 || table->dim[0] != expected_prefix[0] ||
            table->dim[1] != expected_prefix[1] || table->dim[3] != 1) {
            return fail(ErrorCode::kInvalidShape,
                        path + "." + std::string(port),
                        "page table must have dimensions [B,1,P,1]");
        }
        if (table->is_virtual || table->reordering_type != "NONE") {
            return unsupported(path + "." + std::string(port),
                               "page table must be an external plain INT32 "
                               "tensor");
        }
        return Status::ok();
    };
    status = validate_page_table(result.page_k, "Page_table_K");
    if (status.is_bad()) return status;
    status = validate_page_table(result.page_v, "Page_table_V");
    if (status.is_bad()) return status;

    auto page_capacity = [&](TensorDesc const* table,
                             TensorDesc const& container,
                             std::int64_t& capacity) -> Status {
        if (table->dim[2] >
            std::numeric_limits<std::int64_t>::max() / container.dim[2]) {
            return fail(ErrorCode::kDimensionOverflow, path,
                        "paged sequence capacity overflows int64");
        }
        capacity = table->dim[2] * container.dim[2];
        return Status::ok();
    };
    std::int64_t k_capacity = result.k->dim[2];
    std::int64_t v_capacity = result.v->dim[2];
    if (result.page_k != nullptr) {
        status = page_capacity(result.page_k, *result.k, k_capacity);
        if (status.is_bad()) return status;
    }
    if (result.page_v != nullptr) {
        status = page_capacity(result.page_v, *result.v, v_capacity);
        if (status.is_bad()) return status;
    }
    auto const is_paged = result.page_k != nullptr || result.page_v != nullptr;
    if (max_seq_len && !is_paged) {
        return unsupported(path + ".max_seq_len_kv",
                           "explicit maximum sequence length is only valid "
                           "for paged attention");
    }
    if (max_seq_len) {
        result.s_kv = *max_seq_len;
    } else if (result.page_k == nullptr) {
        result.s_kv = result.k->dim[2];
    } else if (result.page_v == nullptr) {
        result.s_kv = result.v->dim[2];
    } else if (result.bias != nullptr) {
        if (result.bias->dim.size() != 4) {
            return fail(ErrorCode::kInvalidShape, path + ".Bias",
                        "both-paged sequence inference requires rank-4 Bias");
        }
        result.s_kv = result.bias->dim[3];
    } else if (result.rng_dump != nullptr) {
        if (result.rng_dump->dim.size() != 4) {
            return fail(
                ErrorCode::kInvalidShape, path + ".RNG_DUMP",
                "both-paged sequence inference requires rank-4 RNG_DUMP");
        }
        result.s_kv = result.rng_dump->dim[3];
    } else {
        result.s_kv = std::min(k_capacity, v_capacity);
    }

    if (result.k->dim[3] != d_qk || h_q % h_k != 0 || h_q % h_v != 0 ||
        (result.page_k == nullptr &&
         (result.k->dim[0] != b || result.k->dim[2] != result.s_kv)) ||
        (result.page_v == nullptr &&
         (result.v->dim[0] != b || result.v->dim[2] != result.s_kv)) ||
        (result.page_k != nullptr && k_capacity < result.s_kv) ||
        (result.page_v != nullptr && v_capacity < result.s_kv)) {
        return fail(ErrorCode::kInvalidShape, path,
                    "Q/K/V batch, logical sequence, embedding, paged "
                    "capacity, or grouped-head dimensions are inconsistent");
    }
    if ((result.page_k != nullptr && result.k->ragged_offset_uid) ||
        (result.page_v != nullptr && result.v->ragged_offset_uid)) {
        return unsupported(path,
                           "a paged K/V container cannot also use ragged "
                           "addressing in this C6 increment");
    }
    std::vector<std::int64_t> const o_dimensions{b, h_q, s_q, d_v};
    if (result.o->dim != o_dimensions || result.o->stride.back() != 1) {
        return fail(ErrorCode::kInvalidShape, path + ".O",
                    "O must have BHSD dimensions [B,Hq,Sq,Dv]");
    }
    if (result.backward &&
        (result.d_o->dim != o_dimensions || result.d_o->stride.back() != 1 ||
         result.d_q->dim != result.q->dim || result.d_k->dim != result.k->dim ||
         result.d_v->dim != result.v->dim ||
         result.d_q->stride.back() != 1 || result.d_k->stride.back() != 1 ||
         result.d_v->stride.back() != 1)) {
        return fail(ErrorCode::kInvalidShape, path,
                    "dO/dQ/dK/dV must match their forward tensor dimensions");
    }

    auto validate_ragged = [&](TensorDesc const& tensor,
                               std::string_view port) -> Status {
        if (!tensor.ragged_offset_uid) return Status::ok();
        if (tensor.is_virtual || tensor.reordering_type != "NONE") {
            return unsupported(path + "." + std::string(port),
                               "ragged attention data must be an external "
                               "plain tensor");
        }
        if (b == std::numeric_limits<std::int64_t>::max()) {
            return fail(ErrorCode::kDimensionOverflow,
                        path + "." + std::string(port),
                        "ragged batch extent overflows int64");
        }
        auto const offset_it = graph.tensors.find(*tensor.ragged_offset_uid);
        if (offset_it == graph.tensors.end()) {
            return fail(ErrorCode::kMissingUid,
                        path + "." + std::string(port),
                        "ragged offset tensor is unresolved");
        }
        std::vector<std::int64_t> const expected{b + 1, 1, 1, 1};
        if (offset_it->second.dim != expected) {
            return fail(ErrorCode::kInvalidShape,
                        path + "." + std::string(port),
                        "ragged offset must have dimensions [B+1,1,1,1]");
        }
        std::int64_t storage_elements = 0;
        if (!checked_ragged_storage_elements(tensor, storage_elements)) {
            return fail(ErrorCode::kDimensionOverflow,
                        path + "." + std::string(port),
                        "maximum ragged storage span does not fit an MLIR "
                        "index");
        }
        return Status::ok();
    };
    for (auto const& [tensor, port] :
         std::array<std::pair<TensorDesc const*, std::string_view>, 4>{
             std::pair{result.q, std::string_view{"Q"}},
             std::pair{result.k, std::string_view{"K"}},
             std::pair{result.v, std::string_view{"V"}},
             std::pair{result.o, std::string_view{"O"}}}) {
        status = validate_ragged(*tensor, port);
        if (status.is_bad()) return status;
    }
    if (result.backward) {
        for (auto const& [tensor, port] :
             std::array<std::pair<TensorDesc const*, std::string_view>, 5>{
                 std::pair{result.d_o, std::string_view{"dO"}},
                 std::pair{result.stats, std::string_view{"Stats"}},
                 std::pair{result.d_q, std::string_view{"dQ"}},
                 std::pair{result.d_k, std::string_view{"dK"}},
                 std::pair{result.d_v, std::string_view{"dV"}}}) {
            status = validate_ragged(*tensor, port);
            if (status.is_bad()) return status;
        }
    } else {
        for (auto const& [tensor, port] :
             std::array<std::pair<TensorDesc const*, std::string_view>, 2>{
                 std::pair{result.page_k, std::string_view{"Page_table_K"}},
                 std::pair{result.page_v,
                           std::string_view{"Page_table_V"}}}) {
            if (tensor == nullptr) continue;
            status = validate_ragged(*tensor, port);
            if (status.is_bad()) return status;
        }
    }

    if (!read_bool_attribute(operation, "alibi_mask", result.alibi) ||
        !read_bool_attribute(operation, "padding_mask", result.padding) ||
        !read_optional_integer_attribute(operation, "left_bound",
                                         result.left_bound) ||
        !read_optional_integer_attribute(operation, "right_bound",
                                         result.right_bound) ||
        !read_optional_number_attribute(operation, "attn_scale_value",
                                        result.attn_scale_value)) {
        return fail(ErrorCode::kInvalidValue, path,
                    "attention attributes have invalid serialized types");
    }
    if (result.attn_scale_value &&
        !std::isfinite(*result.attn_scale_value)) {
        return fail(ErrorCode::kInvalidValue, path + ".attn_scale_value",
                    "attention scale must be finite");
    }
    if (result.left_bound && *result.left_bound <= 0) {
        return fail(ErrorCode::kInvalidValue, path + ".left_bound",
                    "left bound must be greater than zero");
    }
    if (result.right_bound && *result.right_bound < 0) {
        return fail(ErrorCode::kInvalidValue, path + ".right_bound",
                    "right bound must be nonnegative");
    }
    std::string_view alignment;
    if (!read_string_attribute(operation, "diagonal_alignment", alignment) ||
        (alignment != "TOP_LEFT" && alignment != "BOTTOM_RIGHT")) {
        return fail(ErrorCode::kInvalidValue, path + ".diagonal_alignment",
                    "expected TOP_LEFT or BOTTOM_RIGHT");
    }
    result.bottom_right = alignment == "BOTTOM_RIGHT";

    if (has_input(operation, "Attn_scale")) {
        if (result.attn_scale_value) {
            return fail(ErrorCode::kInvalidValue, path,
                        "Attn_scale tensor and attn_scale_value are mutually "
                        "exclusive");
        }
        status = validate_scalar_port(operation, graph, true, "Attn_scale",
                                      DataType::kFloat32, path,
                                      result.attn_scale_uid,
                                      result.attn_scale);
        if (status.is_bad()) return status;
    }

    std::vector<std::int64_t> const score_dimensions{
        b, h_q, s_q, result.s_kv};
    if (!result.backward && has_input(operation, "Block_mask")) {
        status = require_tensor(operation, graph, true, "Block_mask",
                                DataType::kUInt8, path,
                                result.block_mask_uid, result.block_mask);
        if (status.is_bad()) return status;
        auto const query_tiles = (s_q - 1) / 128 + 1;
        auto const key_tiles = (result.s_kv - 1) / 128 + 1;
        std::vector<std::int64_t> const expected{
            b, h_q, query_tiles, (key_tiles + 7) / 8};
        if (result.block_mask->dim != expected) {
            return fail(
                ErrorCode::kInvalidShape, path + ".Block_mask",
                "block mask must have dimensions "
                "[B,Hq,ceil(Sq/128),ceil(ceil(Skv/128)/8)]");
        }
        if (result.block_mask->is_virtual ||
            result.block_mask->is_pass_by_value ||
            result.block_mask->pass_by_value ||
            result.block_mask->ragged_offset_uid ||
            result.block_mask->reordering_type != "NONE") {
            return unsupported(path + ".Block_mask",
                               "block mask must be an external plain UINT8 "
                               "tensor");
        }
    }
    if (result.bias != nullptr) {
        if (!broadcast_compatible(*result.bias, score_dimensions)) {
            return fail(ErrorCode::kInvalidShape, path + ".Bias",
                        "bias is not broadcast-compatible with attention scores");
        }
    }

    if (result.padding) {
        status = require_tensor(operation, graph, true, "SEQ_LEN_Q",
                                DataType::kInt32, path, result.seq_q_uid,
                                result.seq_q);
        if (status.is_bad()) return status;
        status = require_tensor(operation, graph, true, "SEQ_LEN_KV",
                                DataType::kInt32, path, result.seq_kv_uid,
                                result.seq_kv);
        if (status.is_bad()) return status;
        std::vector<std::int64_t> const expected{b, 1, 1, 1};
        if (result.seq_q->dim != expected || result.seq_kv->dim != expected) {
            return fail(ErrorCode::kInvalidShape, path,
                        "sequence lengths must have dimensions [B,1,1,1]");
        }
    } else if (has_input(operation, "SEQ_LEN_Q") ||
               has_input(operation, "SEQ_LEN_KV")) {
        return fail(ErrorCode::kInvalidValue, path,
                    "sequence lengths require padding_mask=true");
    }
    auto const has_ragged = result.q->ragged_offset_uid ||
                            result.k->ragged_offset_uid ||
                            result.v->ragged_offset_uid ||
                            result.o->ragged_offset_uid ||
                            (result.d_o != nullptr &&
                             result.d_o->ragged_offset_uid) ||
                            (result.stats != nullptr &&
                             result.stats->ragged_offset_uid) ||
                            (result.d_q != nullptr &&
                             result.d_q->ragged_offset_uid) ||
                            (result.d_k != nullptr &&
                             result.d_k->ragged_offset_uid) ||
                            (result.d_v != nullptr &&
                             result.d_v->ragged_offset_uid) ||
                            (result.page_k != nullptr &&
                             result.page_k->ragged_offset_uid) ||
                            (result.page_v != nullptr &&
                             result.page_v->ragged_offset_uid);
    if ((is_paged || has_ragged) && !result.padding) {
        return unsupported(path,
                           "paged and ragged SDPA require padding_mask=true "
                           "with both sequence-length tensors");
    }

    if (result.alibi &&
        (!result.right_bound || *result.right_bound != 0)) {
        return unsupported(path,
                           "ALiBi requires diagonal right_bound=0 in "
                           "cuDNN Frontend v1.24.0");
    }
    if (result.bottom_right && result.right_bound &&
        (!result.padding && s_q > result.s_kv)) {
        return unsupported(path,
                           "bottom-right masking without padding requires "
                           "Sq <= Skv");
    }
    if (result.left_bound && !result.padding && s_q > result.s_kv) {
        return unsupported(path,
                           "sliding-window attention without padding requires "
                           "Sq <= Skv");
    }

    status = decode_dropout(operation, graph, result.backward,
                            score_dimensions, path, result.dropout);
    if (status.is_bad()) return status;
    if (result.bottom_right && result.right_bound &&
        (result.bias != nullptr || result.alibi ||
         result.dropout.kind != DropoutKind::kNone)) {
        return unsupported(path,
                           "cuDNN bottom-right causal masking does not combine "
                           "with bias, ALiBi, or dropout");
    }

    if (result.rng_dump != nullptr) {
        if (result.dropout.kind != DropoutKind::kInternal) {
            return fail(ErrorCode::kInvalidValue, path + ".RNG_DUMP",
                        "RNG_DUMP requires probability dropout");
        }
        if (result.rng_dump->dim != score_dimensions) {
            return fail(ErrorCode::kInvalidShape, path + ".RNG_DUMP",
                        "RNG_DUMP must match [B,Hq,Sq,Skv]");
        }
    }

    std::vector<std::int64_t> const row_dimensions{b, h_q, s_q, 1};
    if (result.backward) {
        if (result.stats->dim != row_dimensions) {
            return fail(ErrorCode::kInvalidShape, path + ".Stats",
                        "Stats must have dimensions [B,Hq,Sq,1]");
        }
        if (has_output(operation, "dBias")) {
            if (result.bias == nullptr) {
                return fail(ErrorCode::kInvalidValue, path + ".dBias",
                            "dBias requires a Bias input");
            }
            status = require(false, "dBias", result.dbias_uid,
                             result.d_bias);
            if (status.is_bad()) return status;
            if (result.d_bias->dim.size() != 4 ||
                !broadcast_compatible(*result.d_bias, score_dimensions)) {
                return fail(ErrorCode::kInvalidShape, path + ".dBias",
                            "dBias must be a rank-4 reduction shape for scores");
            }
        }
        std::optional<std::int64_t> max_q;
        std::optional<std::int64_t> max_kv;
        bool deterministic = false;
        if (!read_optional_integer_attribute(operation, "max_total_seq_len_q",
                                             max_q) ||
            !read_optional_integer_attribute(operation, "max_total_seq_len_kv",
                                             max_kv) ||
            !read_bool_attribute(operation, "is_deterministic_algorithm",
                                 deterministic)) {
            return fail(ErrorCode::kInvalidValue, path,
                        "backward metadata attributes are malformed");
        }
        auto validate_max_total = [&](std::optional<std::int64_t> value,
                                      std::int64_t sequence,
                                      std::string_view name) -> Status {
            if (!value) return Status::ok();
            if (!has_ragged) {
                return unsupported(
                    path + "." + std::string(name),
                    "max_total_seq_len metadata requires packed/ragged "
                    "attention storage");
            }
            if (*value <= 0 ||
                b > std::numeric_limits<std::int64_t>::max() / sequence ||
                *value > b * sequence) {
                return fail(ErrorCode::kInvalidValue,
                            path + "." + std::string(name),
                            "maximum total sequence length must be positive "
                            "and within the nominal aggregate bound");
            }
            return Status::ok();
        };
        status = validate_max_total(max_q, s_q, "max_total_seq_len_q");
        if (status.is_bad()) return status;
        status = validate_max_total(max_kv, result.s_kv,
                                    "max_total_seq_len_kv");
        if (status.is_bad()) return status;
    } else {
        if (!read_bool_attribute(operation, "generate_stats",
                                 result.generate_stats)) {
            return fail(ErrorCode::kInvalidValue, path + ".generate_stats",
                        "expected a boolean");
        }
        auto add_row_output = [&](std::string_view port) -> Status {
            if (!has_output(operation, port)) return Status::ok();
            std::int64_t uid = 0;
            TensorDesc const* tensor = nullptr;
            auto item_status = require(false, port, uid, tensor);
            if (item_status.is_bad()) return item_status;
            if (tensor->dim != row_dimensions) {
                return fail(ErrorCode::kInvalidShape,
                            path + "." + std::string(port),
                            "softmax row output must be [B,Hq,Sq,1]");
            }
            item_status = validate_ragged(*tensor, port);
            if (item_status.is_bad()) return item_status;
            result.row_outputs.emplace_back(uid, tensor);
            if (port == "Stats") {
                result.stats_uid = uid;
                result.stats = tensor;
            }
            return Status::ok();
        };
        for (auto port : {"Stats", "Max", "Sum_exp"}) {
            status = add_row_output(port);
            if (status.is_bad()) return status;
        }
        if (result.generate_stats != (result.stats != nullptr)) {
            return fail(ErrorCode::kInvalidValue, path,
                        "Stats output presence must equal generate_stats");
        }
        std::optional<std::string_view> mma_mode;
        std::optional<std::string_view> implementation;
        bool is_mxfp8 = false;
        bool unfuse_fma = false;
        if (!read_optional_string_attribute(operation, "mma_core_mode",
                                            mma_mode) ||
            !read_optional_string_attribute(operation, "implementation",
                                            implementation) ||
            !read_bool_attribute(operation, "is_mxfp8", is_mxfp8) ||
            !read_bool_attribute(operation, "unfuse_fma", unfuse_fma)) {
            return fail(ErrorCode::kInvalidValue, path,
                        "forward implementation attributes are malformed");
        }
        static constexpr std::string_view kMmaModes[] = {
            "NOT_SET", "FLOAT", "HALF", "BFLOAT16"};
        static constexpr std::string_view kImplementations[] = {
            "AUTO", "COMPOSITE", "UNIFIED"};
        if ((mma_mode &&
             std::find(std::begin(kMmaModes), std::end(kMmaModes), *mma_mode) ==
                 std::end(kMmaModes)) ||
            (implementation &&
             std::find(std::begin(kImplementations),
                       std::end(kImplementations), *implementation) ==
                 std::end(kImplementations))) {
            return fail(ErrorCode::kInvalidValue, path,
                        "unknown SDPA implementation enum");
        }
        if (is_mxfp8) {
            return unsupported(path, "MXFP8 attention uses the specialized C5 path");
        }
        for (auto port : {"Descale_Q", "Descale_K", "Descale_V",
                          "Descale_S", "Scale_S", "Scale_O"}) {
            if (has_input(operation, port)) {
                return unsupported(path,
                                   std::string(port) +
                                       " attention is deferred to C5/C6");
            }
        }
        for (auto port : {"Amax_S", "Amax_O"}) {
            if (has_output(operation, port)) {
                return unsupported(path,
                                   std::string(port) +
                                       " is an FP8 attention output");
            }
        }
        if (attribute(operation, "rescale_threshold") != nullptr) {
            return unsupported(path,
                               "rescale_threshold is an FP8 implementation "
                               "control");
        }
    }

    output = std::move(result);
    return Status::ok();
}

Status validate_rng(GenericOperationDesc const& operation,
                    SerializedGraph const& graph,
                    std::string const& path) {
    std::int64_t y_uid = 0;
    TensorDesc const* y = nullptr;
    auto status = require_tensor(operation, graph, false, "Y",
                                 DataType::kFloat32, path, y_uid, y);
    if (status.is_bad()) return status;
    if (operation.outputs.size() != 1) {
        return fail(ErrorCode::kInvalidValue, path,
                    "RNG requires exactly one Y output");
    }
    std::string_view distribution;
    std::vector<std::int64_t> dimensions;
    std::vector<std::int64_t> strides;
    std::optional<std::int64_t> seed;
    std::optional<double> probability;
    if (!read_string_attribute(operation, "distribution", distribution) ||
        distribution != "BERNOULLI" ||
        !read_integer_array(operation, "dim", dimensions) ||
        !read_integer_array(operation, "stride", strides) ||
        !read_optional_integer_attribute(operation, "seed", seed) ||
        !read_optional_number_attribute(operation, "bernoulli_probability",
                                        probability) ||
        !probability || !std::isfinite(*probability) || *probability < 0.0 ||
        *probability > 1.0) {
        return fail(ErrorCode::kInvalidValue, path,
                    "RNG requires a valid BERNOULLI configuration");
    }
    if (dimensions != y->dim || strides != y->stride) {
        return fail(ErrorCode::kInvalidShape, path,
                    "RNG dim/stride attributes must match Y");
    }
    if (seed) {
        if (!operation.inputs.empty()) {
            return fail(ErrorCode::kInvalidValue, path,
                        "fixed-seed RNG does not accept Seed/Offset tensors");
        }
    } else {
        std::int64_t uid = 0;
        TensorDesc const* tensor = nullptr;
        status = validate_scalar_port(operation, graph, true, "Seed",
                                      DataType::kInt64, path, uid, tensor);
        if (status.is_bad()) return status;
        status = validate_scalar_port(operation, graph, true, "Offset",
                                      DataType::kInt64, path, uid, tensor);
        if (status.is_bad()) return status;
        if (operation.inputs.size() != 2) {
            return fail(ErrorCode::kInvalidValue, path,
                        "dynamic RNG requires exactly Seed and Offset");
        }
    }
    return Status::ok();
}

Status validate_rope(OperationTag tag,
                     GenericOperationDesc const& operation,
                     SerializedGraph const& graph,
                     std::string const& path) {
    auto const input_port = tag == OperationTag::kRope ? "INPUT" : "DY";
    auto const output_port = tag == OperationTag::kRope ? "OUTPUT" : "DX";
    std::int64_t input_uid = 0;
    std::int64_t freqs_uid = 0;
    std::int64_t output_uid = 0;
    TensorDesc const* input = nullptr;
    TensorDesc const* freqs = nullptr;
    TensorDesc const* output = nullptr;
    auto status = require_tensor(operation, graph, true, input_port,
                                 DataType::kFloat32, path, input_uid, input);
    if (status.is_bad()) return status;
    status = require_tensor(operation, graph, true, "FREQS",
                            DataType::kFloat32, path, freqs_uid, freqs);
    if (status.is_bad()) return status;
    status = require_tensor(operation, graph, false, output_port,
                            DataType::kFloat32, path, output_uid, output);
    if (status.is_bad()) return status;
    if (operation.inputs.size() != 2 || operation.outputs.size() != 1) {
        return fail(ErrorCode::kInvalidValue, path,
                    "RoPE requires two inputs and one output");
    }
    std::string_view compute_type;
    double output_scale = 0.0;
    std::int64_t rope_dim = 0;
    if (!read_string_attribute(operation, "compute_data_type", compute_type) ||
        compute_type != "FLOAT" ||
        !read_number_attribute(operation, "output_scale", output_scale) ||
        !std::isfinite(output_scale) ||
        !read_integer_attribute(operation, "rope_dim", rope_dim) ||
        rope_dim < 0) {
        return fail(ErrorCode::kInvalidValue, path,
                    "RoPE attributes are not executable f32 semantics");
    }
    if (input->dim.size() != 4 || output->dim != input->dim ||
        freqs->dim.size() != 4 || freqs->dim[0] < input->dim[2] ||
        freqs->dim[1] != 1 || freqs->dim[2] != 1) {
        return fail(ErrorCode::kInvalidShape, path,
                    "RoPE requires BHSD input/output and [S,1,1,R] freqs");
    }
    auto const effective_dim = rope_dim == 0 ? input->dim[3] : rope_dim;
    if (effective_dim <= 0 || effective_dim > input->dim[3] ||
        effective_dim % 2 != 0 || freqs->dim[3] != effective_dim) {
        return fail(ErrorCode::kInvalidShape, path,
                    "effective rope_dim must be positive, even, no larger "
                    "than D, and equal FREQS[-1]");
    }
    return Status::ok();
}

::mlir::Value index_constant(::mlir::OpBuilder& builder,
                             ::mlir::Location location,
                             std::int64_t value) {
    return ::mlir::arith::ConstantIndexOp::create(builder, location, value);
}

::mlir::Value integer_constant(::mlir::OpBuilder& builder,
                               ::mlir::Location location,
                               std::int64_t value,
                               unsigned width = 64) {
    return ::mlir::arith::ConstantIntOp::create(builder, location, value, width);
}

::mlir::Value float_constant(::mlir::OpBuilder& builder,
                             ::mlir::Location location,
                             float value) {
    return ::mlir::arith::ConstantFloatOp::create(
        builder, location, ::mlir::Float32Type::get(builder.getContext()),
        llvm::APFloat(value));
}

::mlir::Value true_value(::mlir::OpBuilder& builder,
                         ::mlir::Location location) {
    return integer_constant(builder, location, 1, 1);
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
    std::uint64_t count = 0;
    if (!checked_element_count(dimensions, count) ||
        count > static_cast<std::uint64_t>(
                    std::numeric_limits<std::int64_t>::max())) {
        return fail(ErrorCode::kDimensionOverflow,
                    std::string(operation_name),
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

::mlir::Value linear_index(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    llvm::SmallVector<::mlir::Value> const& indices,
    std::vector<std::int64_t> const& dimensions) {
    auto linear = index_constant(builder, location, 0);
    for (std::size_t axis = 0; axis < dimensions.size(); ++axis) {
        linear = ::mlir::arith::AddIOp::create(
            builder, location,
            ::mlir::arith::MulIOp::create(
                builder, location, linear,
                index_constant(builder, location, dimensions[axis])),
            indices[axis]);
    }
    return linear;
}

llvm::SmallVector<::mlir::Value> zero_indices(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    TensorDesc const& tensor) {
    return llvm::SmallVector<::mlir::Value>(
        tensor.dim.size(), index_constant(builder, location, 0));
}

llvm::SmallVector<::mlir::Value> broadcast_indices(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    llvm::SmallVector<::mlir::Value> const& output_indices,
    TensorDesc const& input) {
    auto const offset = output_indices.size() - input.dim.size();
    llvm::SmallVector<::mlir::Value> indices;
    indices.reserve(input.dim.size());
    for (std::size_t axis = 0; axis < input.dim.size(); ++axis) {
        indices.push_back(input.dim[axis] == 1
                              ? index_constant(builder, location, 0)
                              : output_indices[offset + axis]);
    }
    return indices;
}

::mlir::Value load_scalar(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    std::int64_t uid,
    TensorDesc const& tensor,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    return ::mlir::memref::LoadOp::create(
        builder, location, values.at(uid),
        zero_indices(builder, location, tensor));
}

::mlir::Value bool_as_float(::mlir::OpBuilder& builder,
                            ::mlir::Location location,
                            ::mlir::Value condition) {
    return ::mlir::arith::SelectOp::create(
        builder, location, condition,
        float_constant(builder, location, 1.0F),
        float_constant(builder, location, 0.0F));
}

::mlir::Value index_as_i64(::mlir::OpBuilder& builder,
                           ::mlir::Location location,
                           ::mlir::Value index) {
    return ::mlir::arith::IndexCastUIOp::create(
        builder, location,
        ::mlir::IntegerType::get(builder.getContext(), 64), index);
}

::mlir::Value extend_i32(::mlir::OpBuilder& builder,
                         ::mlir::Location location,
                         ::mlir::Value value) {
    return ::mlir::arith::ExtSIOp::create(
        builder, location,
        ::mlir::IntegerType::get(builder.getContext(), 64), value);
}

::mlir::Value stable_bernoulli_condition(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    ::mlir::Value seed,
    ::mlir::Value offset,
    ::mlir::Value linear,
    double probability) {
    ::mlir::Value z = ::mlir::arith::AddIOp::create(
        builder, location,
        ::mlir::arith::AddIOp::create(
            builder, location,
            ::mlir::arith::AddIOp::create(
                builder, location, seed, offset),
            index_as_i64(builder, location, linear)),
        integer_constant(builder, location, -7046029254386353131LL));
    z = ::mlir::arith::MulIOp::create(
        builder, location,
        ::mlir::arith::XOrIOp::create(
            builder, location, z,
            ::mlir::arith::ShRUIOp::create(
                builder, location, z,
                integer_constant(builder, location, 30))),
        integer_constant(builder, location, -4658895280553007687LL));
    z = ::mlir::arith::MulIOp::create(
        builder, location,
        ::mlir::arith::XOrIOp::create(
            builder, location, z,
            ::mlir::arith::ShRUIOp::create(
                builder, location, z,
                integer_constant(builder, location, 27))),
        integer_constant(builder, location, -7723592293110705685LL));
    z = ::mlir::arith::XOrIOp::create(
        builder, location, z,
        ::mlir::arith::ShRUIOp::create(
            builder, location, z,
            integer_constant(builder, location, 31)));
    auto sample = ::mlir::arith::ShRUIOp::create(
        builder, location, z, integer_constant(builder, location, 40));
    auto const threshold = static_cast<std::int64_t>(
        std::floor(probability * static_cast<double>(1U << 24)));
    return ::mlir::arith::CmpIOp::create(
        builder, location, ::mlir::arith::CmpIPredicate::ult, sample,
        integer_constant(builder, location, threshold));
}

Status emit_rng(::mlir::OpBuilder& builder,
                ::mlir::Location location,
                GenericOperationDesc const& operation,
                SerializedGraph const& graph,
                std::map<std::int64_t, ::mlir::Value> const& values) {
    std::int64_t y_uid = 0;
    (void)tensor_uid(operation, false, "Y", y_uid);
    auto const& y = graph.tensors.at(y_uid);
    std::optional<std::int64_t> fixed_seed;
    std::optional<double> probability;
    (void)read_optional_integer_attribute(operation, "seed", fixed_seed);
    (void)read_optional_number_attribute(operation, "bernoulli_probability",
                                         probability);
    ::mlir::Value seed;
    ::mlir::Value offset;
    if (fixed_seed) {
        seed = integer_constant(builder, location, *fixed_seed);
        offset = integer_constant(builder, location, 0);
    } else {
        std::int64_t seed_uid = 0;
        std::int64_t offset_uid = 0;
        (void)tensor_uid(operation, true, "Seed", seed_uid);
        (void)tensor_uid(operation, true, "Offset", offset_uid);
        seed = load_scalar(builder, location, seed_uid,
                           graph.tensors.at(seed_uid), values);
        offset = load_scalar(builder, location, offset_uid,
                             graph.tensors.at(offset_uid), values);
    }
    return emit_flat_loop(
        builder, location, y.dim, "RNG",
        [&](::mlir::OpBuilder& body_builder,
            ::mlir::Location body_location,
            llvm::SmallVector<::mlir::Value> const& indices) {
            auto linear = linear_index(body_builder, body_location, indices,
                                       y.dim);
            auto keep = stable_bernoulli_condition(
                body_builder, body_location, seed, offset, linear,
                *probability);
            ::mlir::memref::StoreOp::create(
                body_builder, body_location,
                bool_as_float(body_builder, body_location, keep),
                values.at(y_uid), indices);
        });
}

Status emit_rope(OperationTag tag,
                 ::mlir::OpBuilder& builder,
                 ::mlir::Location location,
                 GenericOperationDesc const& operation,
                 SerializedGraph const& graph,
                 std::map<std::int64_t, ::mlir::Value> const& values) {
    bool const forward = tag == OperationTag::kRope;
    auto const input_port = forward ? "INPUT" : "DY";
    auto const output_port = forward ? "OUTPUT" : "DX";
    std::int64_t input_uid = 0;
    std::int64_t freqs_uid = 0;
    std::int64_t output_uid = 0;
    (void)tensor_uid(operation, true, input_port, input_uid);
    (void)tensor_uid(operation, true, "FREQS", freqs_uid);
    (void)tensor_uid(operation, false, output_port, output_uid);
    auto const& input = graph.tensors.at(input_uid);
    auto const& output = graph.tensors.at(output_uid);
    std::int64_t rope_dim = 0;
    double output_scale = 1.0;
    (void)read_integer_attribute(operation, "rope_dim", rope_dim);
    (void)read_number_attribute(operation, "output_scale", output_scale);
    auto const effective_dim = rope_dim == 0 ? input.dim[3] : rope_dim;
    auto const rotation_start = input.dim[3] - effective_dim;
    auto const half = effective_dim / 2;

    return emit_flat_loop(
        builder, location, output.dim, forward ? "ROPE" : "ROPE_BWD",
        [&](::mlir::OpBuilder& body_builder,
            ::mlir::Location body_location,
            llvm::SmallVector<::mlir::Value> const& indices) {
            auto const d = indices[3];
            auto is_passthrough = ::mlir::arith::CmpIOp::create(
                body_builder, body_location,
                ::mlir::arith::CmpIPredicate::ult, d,
                index_constant(body_builder, body_location, rotation_start));
            auto is_low = ::mlir::arith::CmpIOp::create(
                body_builder, body_location,
                ::mlir::arith::CmpIPredicate::ult, d,
                index_constant(body_builder, body_location,
                               rotation_start + half));
            auto low_angle = ::mlir::arith::SubIOp::create(
                body_builder, body_location, d,
                index_constant(body_builder, body_location, rotation_start));
            auto high_angle = ::mlir::arith::SubIOp::create(
                body_builder, body_location, d,
                index_constant(body_builder, body_location,
                               rotation_start + half));
            auto angle_index = ::mlir::arith::SelectOp::create(
                body_builder, body_location, is_passthrough,
                index_constant(body_builder, body_location, 0),
                ::mlir::arith::SelectOp::create(
                    body_builder, body_location, is_low, low_angle,
                    high_angle));
            auto low_pair = ::mlir::arith::AddIOp::create(
                body_builder, body_location, d,
                index_constant(body_builder, body_location, half));
            auto high_pair = ::mlir::arith::SubIOp::create(
                body_builder, body_location, d,
                index_constant(body_builder, body_location, half));
            auto pair_index = ::mlir::arith::SelectOp::create(
                body_builder, body_location, is_passthrough, d,
                ::mlir::arith::SelectOp::create(
                    body_builder, body_location, is_low, low_pair,
                    high_pair));

            auto input_value = ::mlir::memref::LoadOp::create(
                body_builder, body_location, values.at(input_uid), indices);
            auto pair_indices = indices;
            pair_indices[3] = pair_index;
            auto pair_value = ::mlir::memref::LoadOp::create(
                body_builder, body_location, values.at(input_uid),
                pair_indices);
            llvm::SmallVector<::mlir::Value> frequency_indices{
                indices[2], index_constant(body_builder, body_location, 0),
                index_constant(body_builder, body_location, 0), angle_index};
            auto angle = ::mlir::memref::LoadOp::create(
                body_builder, body_location, values.at(freqs_uid),
                frequency_indices);
            auto cosine = ::mlir::math::CosOp::create(
                body_builder, body_location, angle);
            auto sine = ::mlir::math::SinOp::create(
                body_builder, body_location, angle);
            auto cosine_term = ::mlir::arith::MulFOp::create(
                body_builder, body_location, input_value, cosine);
            auto sine_term = ::mlir::arith::MulFOp::create(
                body_builder, body_location, pair_value, sine);
            ::mlir::Value rotated;
            if (forward) {
                auto low_result = ::mlir::arith::SubFOp::create(
                    body_builder, body_location, cosine_term, sine_term);
                auto high_result = ::mlir::arith::AddFOp::create(
                    body_builder, body_location, cosine_term, sine_term);
                rotated = ::mlir::arith::SelectOp::create(
                    body_builder, body_location, is_low, low_result,
                    high_result);
            } else {
                auto low_result = ::mlir::arith::AddFOp::create(
                    body_builder, body_location, cosine_term, sine_term);
                auto high_result = ::mlir::arith::SubFOp::create(
                    body_builder, body_location, cosine_term, sine_term);
                rotated = ::mlir::arith::SelectOp::create(
                    body_builder, body_location, is_low, low_result,
                    high_result);
            }
            ::mlir::Value result = ::mlir::arith::SelectOp::create(
                body_builder, body_location, is_passthrough, input_value,
                rotated);
            result = ::mlir::arith::MulFOp::create(
                body_builder, body_location, result,
                float_constant(body_builder, body_location,
                               static_cast<float>(output_scale)));
            ::mlir::memref::StoreOp::create(
                body_builder, body_location, result, values.at(output_uid),
                indices);
        });
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
    llvm::SmallVector<::mlir::Value> lowers;
    llvm::SmallVector<::mlir::Value> uppers;
    llvm::SmallVector<::mlir::Value> steps;
    lowers.reserve(extents.size());
    uppers.reserve(extents.size());
    steps.reserve(extents.size());
    for (auto extent : extents) {
        lowers.push_back(index_constant(builder, location, 0));
        uppers.push_back(index_constant(builder, location, extent));
        steps.push_back(index_constant(builder, location, 1));
    }
    auto reduction = ::mlir::scf::buildLoopNest(
        builder, location, lowers, uppers, steps,
        ::mlir::ValueRange{initial},
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

struct AttentionEmissionValues {
    ::mlir::Value attention_scale;
    ::mlir::Value dropout_scale;
    ::mlir::Value dropout_scale_inv;
    ::mlir::Value seed;
    ::mlir::Value offset;
};

AttentionEmissionValues prepare_attention_values(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    AttentionDescription const& attention,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    AttentionEmissionValues result;
    if (attention.attn_scale_value) {
        result.attention_scale = float_constant(
            builder, location,
            static_cast<float>(*attention.attn_scale_value));
    } else if (attention.attn_scale != nullptr) {
        result.attention_scale = load_scalar(
            builder, location, attention.attn_scale_uid,
            *attention.attn_scale, values);
    } else {
        result.attention_scale = float_constant(builder, location, 1.0F);
    }

    result.dropout_scale = float_constant(builder, location, 1.0F);
    result.dropout_scale_inv = float_constant(builder, location, 1.0F);
    if (attention.dropout.kind == DropoutKind::kInternal) {
        result.seed = load_scalar(builder, location,
                                  attention.dropout.seed_uid,
                                  *attention.dropout.seed, values);
        result.offset = load_scalar(builder, location,
                                    attention.dropout.offset_uid,
                                    *attention.dropout.offset, values);
        auto const keep = 1.0 - attention.dropout.probability;
        result.dropout_scale = float_constant(
            builder, location, static_cast<float>(1.0 / keep));
        result.dropout_scale_inv = float_constant(
            builder, location, static_cast<float>(keep));
    } else if (attention.dropout.kind == DropoutKind::kCustom) {
        result.dropout_scale = load_scalar(
            builder, location, attention.dropout.scale_uid,
            *attention.dropout.scale, values);
        if (attention.backward) {
            result.dropout_scale_inv = load_scalar(
                builder, location, attention.dropout.scale_inv_uid,
                *attention.dropout.scale_inv, values);
        }
    }
    return result;
}

::mlir::Value query_to_source_head(::mlir::OpBuilder& builder,
                                   ::mlir::Location location,
                                   ::mlir::Value query_head,
                                   std::int64_t query_heads,
                                   std::int64_t source_heads) {
    return ::mlir::arith::DivUIOp::create(
        builder, location, query_head,
        index_constant(builder, location, query_heads / source_heads));
}

::mlir::Value alibi_slope(::mlir::OpBuilder& builder,
                          ::mlir::Location location,
                          ::mlir::Value head,
                          std::int64_t head_count) {
    auto const power = static_cast<int>(
        std::floor(std::log2(static_cast<double>(head_count))));
    auto const n = std::int64_t{1} << power;
    std::vector<float> slopes;
    slopes.reserve(static_cast<std::size_t>(head_count));
    for (std::int64_t index = 0; index < n; ++index) {
        slopes.push_back(std::pow(
            2.0F, -8.0F * static_cast<float>(index + 1) /
                      static_cast<float>(n)));
    }
    for (std::int64_t index = 0; index < 2 * (head_count - n); index += 2) {
        slopes.push_back(std::pow(
            2.0F, -8.0F * (static_cast<float>(index + 1) * 0.5F) /
                      static_cast<float>(n)));
    }
    auto result = float_constant(builder, location, slopes.front());
    for (std::int64_t index = 1; index < head_count; ++index) {
        auto selected = ::mlir::arith::CmpIOp::create(
            builder, location, ::mlir::arith::CmpIPredicate::eq, head,
            index_constant(builder, location, index));
        result = ::mlir::arith::SelectOp::create(
            builder, location, selected,
            float_constant(builder, location,
                           slopes[static_cast<std::size_t>(index)]),
            result);
    }
    return result;
}

::mlir::Value sequence_length(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    ::mlir::Value batch,
    std::int64_t uid,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    llvm::SmallVector<::mlir::Value> indices{
        batch, index_constant(builder, location, 0),
        index_constant(builder, location, 0),
        index_constant(builder, location, 0)};
    return extend_i32(builder, location,
                      ::mlir::memref::LoadOp::create(
                          builder, location, values.at(uid), indices));
}

::mlir::Value flatten_ragged_buffer(::mlir::OpBuilder& builder,
                                    ::mlir::Location location,
                                    ::mlir::Value buffer,
                                    TensorDesc const& tensor) {
    std::int64_t elements = 0;
    (void)checked_ragged_storage_elements(tensor, elements);
    auto source_type = llvm::cast<::mlir::MemRefType>(buffer.getType());
    auto type = ::mlir::MemRefType::get({elements},
                                        source_type.getElementType());
    return ::mlir::memref::ReinterpretCastOp::create(
        builder, location, type, buffer, 0,
        llvm::ArrayRef<std::int64_t>{elements},
        llvm::ArrayRef<std::int64_t>{1});
}

::mlir::Value ragged_element_index(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    TensorDesc const& tensor,
    llvm::SmallVector<::mlir::Value> const& indices,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    llvm::SmallVector<::mlir::Value> offset_indices{
        indices[0], index_constant(builder, location, 0),
        index_constant(builder, location, 0),
        index_constant(builder, location, 0)};
    ::mlir::Value base = ::mlir::memref::LoadOp::create(
        builder, location, values.at(*tensor.ragged_offset_uid),
        offset_indices);
    if (base.getType().isInteger(32)) {
        base = extend_i32(builder, location, base);
    }
    ::mlir::Value physical = ::mlir::arith::IndexCastOp::create(
        builder, location, builder.getIndexType(), base);
    for (std::size_t axis = 1; axis < indices.size(); ++axis) {
        physical = ::mlir::arith::AddIOp::create(
            builder, location, physical,
            ::mlir::arith::MulIOp::create(
                builder, location, indices[axis],
                index_constant(builder, location, tensor.stride[axis])));
    }
    return physical;
}

::mlir::Value load_paged_attention_tensor(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    std::int64_t tensor_uid,
    TensorDesc const& tensor,
    std::int64_t page_uid,
    TensorDesc const& page_table,
    llvm::SmallVector<::mlir::Value> const& logical_indices,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    auto const block_size = tensor.dim[2];
    auto page_slot = ::mlir::arith::DivUIOp::create(
        builder, location, logical_indices[2],
        index_constant(builder, location, block_size));
    auto within_block = ::mlir::arith::RemUIOp::create(
        builder, location, logical_indices[2],
        index_constant(builder, location, block_size));
    llvm::SmallVector<::mlir::Value> table_indices{
        logical_indices[0], index_constant(builder, location, 0), page_slot,
        index_constant(builder, location, 0)};
    ::mlir::Value page_i32;
    if (page_table.ragged_offset_uid) {
        page_i32 = ::mlir::memref::LoadOp::create(
            builder, location, values.at(page_uid),
            ::mlir::ValueRange{ragged_element_index(
                builder, location, page_table, table_indices, values)});
    } else {
        page_i32 = ::mlir::memref::LoadOp::create(
            builder, location, values.at(page_uid), table_indices);
    }
    auto page_i64 = extend_i32(builder, location, page_i32);
    auto nonnegative = ::mlir::arith::CmpIOp::create(
        builder, location, ::mlir::arith::CmpIPredicate::sge, page_i64,
        integer_constant(builder, location, 0));
    auto in_range = ::mlir::arith::CmpIOp::create(
        builder, location, ::mlir::arith::CmpIPredicate::slt, page_i64,
        integer_constant(builder, location, tensor.dim[0]));
    auto valid = ::mlir::arith::AndIOp::create(builder, location, nonnegative,
                                               in_range);
    auto safe_page_i64 = ::mlir::arith::SelectOp::create(
        builder, location, valid, page_i64,
        integer_constant(builder, location, 0));
    auto safe_page = ::mlir::arith::IndexCastOp::create(
        builder, location, builder.getIndexType(), safe_page_i64);
    llvm::SmallVector<::mlir::Value> physical_indices{
        safe_page, logical_indices[1], within_block, logical_indices[3]};
    auto loaded = ::mlir::memref::LoadOp::create(
        builder, location, values.at(tensor_uid), physical_indices);
    return ::mlir::arith::SelectOp::create(
        builder, location, valid, loaded,
        float_constant(builder, location, 0.0F));
}

::mlir::Value load_attention_tensor(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    std::int64_t tensor_uid,
    TensorDesc const& tensor,
    TensorDesc const* page_table,
    std::int64_t page_uid,
    llvm::SmallVector<::mlir::Value> const& indices,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    if (page_table != nullptr) {
        return load_paged_attention_tensor(builder, location, tensor_uid,
                                           tensor, page_uid, *page_table,
                                           indices, values);
    }
    if (tensor.ragged_offset_uid) {
        return ::mlir::memref::LoadOp::create(
            builder, location, values.at(tensor_uid),
            ::mlir::ValueRange{ragged_element_index(
                builder, location, tensor, indices, values)});
    }
    return ::mlir::memref::LoadOp::create(builder, location,
                                          values.at(tensor_uid), indices);
}

::mlir::Value guarded_attention_load(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    ::mlir::Value condition,
    std::int64_t tensor_uid,
    TensorDesc const& tensor,
    TensorDesc const* page_table,
    std::int64_t page_uid,
    llvm::SmallVector<::mlir::Value> const& indices,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    if (!tensor.ragged_offset_uid) {
        return load_attention_tensor(builder, location, tensor_uid, tensor,
                                     page_table, page_uid, indices, values);
    }
    auto if_op = ::mlir::scf::IfOp::create(
        builder, location,
        ::mlir::TypeRange{::mlir::Float32Type::get(builder.getContext())},
        condition, true);
    builder.setInsertionPointToStart(if_op.thenBlock());
    ::mlir::scf::YieldOp::create(
        builder, location,
        ::mlir::ValueRange{load_attention_tensor(
            builder, location, tensor_uid, tensor, page_table, page_uid,
            indices, values)});
    builder.setInsertionPointToStart(if_op.elseBlock());
    ::mlir::scf::YieldOp::create(
        builder, location,
        ::mlir::ValueRange{float_constant(builder, location, 0.0F)});
    builder.setInsertionPointAfter(if_op);
    return if_op.getResult(0);
}

void store_attention_tensor(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    ::mlir::Value value,
    std::int64_t tensor_uid,
    TensorDesc const& tensor,
    llvm::SmallVector<::mlir::Value> const& indices,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    if (tensor.ragged_offset_uid) {
        ::mlir::memref::StoreOp::create(
            builder, location, value, values.at(tensor_uid),
            ::mlir::ValueRange{ragged_element_index(
                builder, location, tensor, indices, values)});
        return;
    }
    ::mlir::memref::StoreOp::create(builder, location, value,
                                    values.at(tensor_uid), indices);
}

void guarded_attention_store(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    ::mlir::Value condition,
    ::mlir::Value value,
    std::int64_t tensor_uid,
    TensorDesc const& tensor,
    llvm::SmallVector<::mlir::Value> const& indices,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    if (!tensor.ragged_offset_uid) {
        store_attention_tensor(builder, location, value, tensor_uid, tensor,
                               indices, values);
        return;
    }
    auto if_op = ::mlir::scf::IfOp::create(
        builder, location, ::mlir::TypeRange{}, condition, false);
    builder.setInsertionPointToStart(if_op.thenBlock());
    store_attention_tensor(builder, location, value, tensor_uid, tensor,
                           indices, values);
    builder.setInsertionPointAfter(if_op);
}

::mlir::Value score_is_valid(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    AttentionDescription const& attention,
    llvm::SmallVector<::mlir::Value> const& score_indices,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    auto valid = true_value(builder, location);
    auto const q_index = index_as_i64(builder, location, score_indices[2]);
    auto const kv_index = index_as_i64(builder, location, score_indices[3]);
    ::mlir::Value seq_q;
    ::mlir::Value seq_kv;
    if (attention.padding) {
        seq_q = sequence_length(builder, location, score_indices[0],
                                attention.seq_q_uid, values);
        seq_kv = sequence_length(builder, location, score_indices[0],
                                 attention.seq_kv_uid, values);
        auto q_valid = ::mlir::arith::CmpIOp::create(
            builder, location, ::mlir::arith::CmpIPredicate::slt, q_index,
            seq_q);
        auto kv_valid = ::mlir::arith::CmpIOp::create(
            builder, location, ::mlir::arith::CmpIPredicate::slt, kv_index,
            seq_kv);
        auto q_nonnegative = ::mlir::arith::CmpIOp::create(
            builder, location, ::mlir::arith::CmpIPredicate::sge, seq_q,
            integer_constant(builder, location, 0));
        auto kv_nonnegative = ::mlir::arith::CmpIOp::create(
            builder, location, ::mlir::arith::CmpIPredicate::sge, seq_kv,
            integer_constant(builder, location, 0));
        valid = ::mlir::arith::AndIOp::create(
            builder, location, valid,
            ::mlir::arith::AndIOp::create(
                builder, location,
                ::mlir::arith::AndIOp::create(builder, location, q_valid,
                                              kv_valid),
                ::mlir::arith::AndIOp::create(builder, location,
                                              q_nonnegative,
                                              kv_nonnegative)));
    }
    auto difference = ::mlir::arith::SubIOp::create(
        builder, location, kv_index, q_index);
    ::mlir::Value shift = integer_constant(builder, location, 0);
    if (attention.bottom_right) {
        if (attention.padding) {
            shift = ::mlir::arith::SubIOp::create(builder, location, seq_kv,
                                                  seq_q);
        } else {
            shift = integer_constant(builder, location,
                                     attention.s_kv -
                                         attention.q->dim[2]);
        }
    }
    if (attention.right_bound) {
        auto upper = ::mlir::arith::AddIOp::create(
            builder, location, shift,
            integer_constant(builder, location, *attention.right_bound));
        valid = ::mlir::arith::AndIOp::create(
            builder, location, valid,
            ::mlir::arith::CmpIOp::create(
                builder, location, ::mlir::arith::CmpIPredicate::sle,
                difference, upper));
    }
    if (attention.left_bound) {
        auto lower = ::mlir::arith::SubIOp::create(
            builder, location, shift,
            integer_constant(builder, location, *attention.left_bound));
        valid = ::mlir::arith::AndIOp::create(
            builder, location, valid,
            ::mlir::arith::CmpIOp::create(
                builder, location, ::mlir::arith::CmpIPredicate::sgt,
                difference, lower));
    }
    if (attention.block_mask != nullptr) {
        auto const tile = index_constant(builder, location, 128);
        auto query_tile = ::mlir::arith::DivUIOp::create(
            builder, location, score_indices[2], tile);
        auto key_tile = ::mlir::arith::DivUIOp::create(
            builder, location, score_indices[3], tile);
        auto byte_index = ::mlir::arith::DivUIOp::create(
            builder, location, key_tile,
            index_constant(builder, location, 8));
        llvm::SmallVector<::mlir::Value> mask_indices{
            score_indices[0], score_indices[1], query_tile, byte_index};
        auto byte = ::mlir::memref::LoadOp::create(
            builder, location, values.at(attention.block_mask_uid),
            mask_indices);
        auto wide_byte = ::mlir::arith::ExtUIOp::create(
            builder, location,
            ::mlir::IntegerType::get(builder.getContext(), 64), byte);
        auto bit_index = ::mlir::arith::RemUIOp::create(
            builder, location, index_as_i64(builder, location, key_tile),
            integer_constant(builder, location, 8));
        auto selected_bit = ::mlir::arith::AndIOp::create(
            builder, location,
            ::mlir::arith::ShRUIOp::create(builder, location, wide_byte,
                                            bit_index),
            integer_constant(builder, location, 1));
        auto enabled = ::mlir::arith::CmpIOp::create(
            builder, location, ::mlir::arith::CmpIPredicate::ne,
            selected_bit, integer_constant(builder, location, 0));
        valid = ::mlir::arith::AndIOp::create(builder, location, valid,
                                             enabled);
    }
    return valid;
}

::mlir::Value query_position_is_valid(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    AttentionDescription const& attention,
    llvm::SmallVector<::mlir::Value> const& indices,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    if (!attention.padding) {
        return true_value(builder, location);
    }
    auto seq_q = sequence_length(builder, location, indices[0],
                                 attention.seq_q_uid, values);
    auto nonnegative = ::mlir::arith::CmpIOp::create(
        builder, location, ::mlir::arith::CmpIPredicate::sge, seq_q,
        integer_constant(builder, location, 0));
    auto in_range = ::mlir::arith::CmpIOp::create(
        builder, location, ::mlir::arith::CmpIPredicate::slt,
        index_as_i64(builder, location, indices[2]), seq_q);
    return ::mlir::arith::AndIOp::create(builder, location, nonnegative,
                                         in_range);
}

::mlir::Value key_position_is_valid(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    AttentionDescription const& attention,
    llvm::SmallVector<::mlir::Value> const& indices,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    if (!attention.padding) {
        return true_value(builder, location);
    }
    auto seq_kv = sequence_length(builder, location, indices[0],
                                  attention.seq_kv_uid, values);
    auto nonnegative = ::mlir::arith::CmpIOp::create(
        builder, location, ::mlir::arith::CmpIPredicate::sge, seq_kv,
        integer_constant(builder, location, 0));
    auto in_range = ::mlir::arith::CmpIOp::create(
        builder, location, ::mlir::arith::CmpIPredicate::slt,
        index_as_i64(builder, location, indices[2]), seq_kv);
    return ::mlir::arith::AndIOp::create(builder, location, nonnegative,
                                         in_range);
}

::mlir::Value attention_score(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    AttentionDescription const& attention,
    AttentionEmissionValues const& emission_values,
    llvm::SmallVector<::mlir::Value> const& score_indices,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    auto const key_head = query_to_source_head(
        builder, location, score_indices[1], attention.q->dim[1],
        attention.k->dim[1]);
    auto dot = reduce_extent(
        builder, location, attention.q->dim[3],
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
            auto q_value = load_attention_tensor(
                reduction_builder, reduction_location, attention.q_uid,
                *attention.q, nullptr, 0, q_indices, values);
            auto k_value = load_attention_tensor(
                reduction_builder, reduction_location, attention.k_uid,
                *attention.k, attention.page_k, attention.page_k_uid,
                k_indices, values);
            return ::mlir::arith::AddFOp::create(
                reduction_builder, reduction_location, accumulator,
                ::mlir::arith::MulFOp::create(
                    reduction_builder, reduction_location, q_value, k_value));
        });
    ::mlir::Value result = ::mlir::arith::MulFOp::create(
        builder, location, dot, emission_values.attention_scale);
    if (attention.bias != nullptr) {
        auto bias_value = ::mlir::memref::LoadOp::create(
            builder, location, values.at(attention.bias_uid),
            broadcast_indices(builder, location, score_indices,
                              *attention.bias));
        result = ::mlir::arith::AddFOp::create(builder, location, result,
                                               bias_value);
    }
    if (attention.alibi) {
        auto difference = ::mlir::arith::SubIOp::create(
            builder, location,
            index_as_i64(builder, location, score_indices[3]),
            index_as_i64(builder, location, score_indices[2]));
        auto distance = ::mlir::arith::SIToFPOp::create(
            builder, location,
            ::mlir::Float32Type::get(builder.getContext()), difference);
        result = ::mlir::arith::AddFOp::create(
            builder, location, result,
            ::mlir::arith::MulFOp::create(
                builder, location, distance,
                alibi_slope(builder, location, score_indices[1],
                            attention.q->dim[1])));
    }
    return result;
}

struct MaskedScore {
    ::mlir::Value score;
    ::mlir::Value valid;
};

MaskedScore masked_attention_score(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    AttentionDescription const& attention,
    AttentionEmissionValues const& emission_values,
    llvm::SmallVector<::mlir::Value> const& score_indices,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    auto valid = score_is_valid(builder, location, attention, score_indices,
                                values);
    ::mlir::Value score;
    if (attention.q->ragged_offset_uid || attention.k->ragged_offset_uid) {
        auto if_op = ::mlir::scf::IfOp::create(
            builder, location,
            ::mlir::TypeRange{
                ::mlir::Float32Type::get(builder.getContext())},
            valid, true);
        builder.setInsertionPointToStart(if_op.thenBlock());
        ::mlir::scf::YieldOp::create(
            builder, location,
            ::mlir::ValueRange{attention_score(
                builder, location, attention, emission_values, score_indices,
                values)});
        builder.setInsertionPointToStart(if_op.elseBlock());
        ::mlir::scf::YieldOp::create(
            builder, location,
            ::mlir::ValueRange{float_constant(
                builder, location,
                -std::numeric_limits<float>::infinity())});
        builder.setInsertionPointAfter(if_op);
        score = if_op.getResult(0);
    } else {
        score = attention_score(builder, location, attention, emission_values,
                                score_indices, values);
        score = ::mlir::arith::SelectOp::create(
            builder, location, valid, score,
            float_constant(builder, location,
                           -std::numeric_limits<float>::infinity()));
    }
    return {score, valid};
}

struct SoftmaxRow {
    ::mlir::Value maximum;
    ::mlir::Value sum_exp;
};

SoftmaxRow compute_softmax_row(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    AttentionDescription const& attention,
    AttentionEmissionValues const& emission_values,
    llvm::SmallVector<::mlir::Value> const& row_indices,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    ::mlir::Value sink;
    ::mlir::Value initial_maximum = float_constant(
        builder, location, -std::numeric_limits<float>::infinity());
    if (attention.sink != nullptr) {
        llvm::SmallVector<::mlir::Value> sink_indices{
            index_constant(builder, location, 0), row_indices[1],
            index_constant(builder, location, 0),
            index_constant(builder, location, 0)};
        sink = ::mlir::memref::LoadOp::create(
            builder, location, values.at(attention.sink_uid), sink_indices);
        initial_maximum = sink;
    }
    auto maximum = reduce_extent(
        builder, location, attention.s_kv,
        initial_maximum,
        [&](::mlir::OpBuilder& reduction_builder,
            ::mlir::Location reduction_location, ::mlir::Value kv,
            ::mlir::Value accumulator) {
            llvm::SmallVector<::mlir::Value> score_indices{
                row_indices[0], row_indices[1], row_indices[2], kv};
            auto item = masked_attention_score(
                reduction_builder, reduction_location, attention,
                emission_values, score_indices, values);
            return ::mlir::arith::MaximumFOp::create(
                reduction_builder, reduction_location, accumulator,
                item.score);
        });
    ::mlir::Value initial_sum = float_constant(builder, location, 0.0F);
    if (attention.sink != nullptr) {
        initial_sum = ::mlir::math::ExpOp::create(
            builder, location,
            ::mlir::arith::SubFOp::create(builder, location, sink, maximum));
    }
    auto sum = reduce_extent(
        builder, location, attention.s_kv,
        initial_sum,
        [&](::mlir::OpBuilder& reduction_builder,
            ::mlir::Location reduction_location, ::mlir::Value kv,
            ::mlir::Value accumulator) {
            llvm::SmallVector<::mlir::Value> score_indices{
                row_indices[0], row_indices[1], row_indices[2], kv};
            auto item = masked_attention_score(
                reduction_builder, reduction_location, attention,
                emission_values, score_indices, values);
            ::mlir::Value exponential = ::mlir::math::ExpOp::create(
                reduction_builder, reduction_location,
                ::mlir::arith::SubFOp::create(
                    reduction_builder, reduction_location, item.score,
                    maximum));
            exponential = ::mlir::arith::SelectOp::create(
                reduction_builder, reduction_location, item.valid,
                exponential,
                float_constant(reduction_builder, reduction_location, 0.0F));
            return ::mlir::arith::AddFOp::create(
                reduction_builder, reduction_location, accumulator,
                exponential);
        });
    return {maximum, sum};
}

::mlir::Value forward_probability(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    AttentionDescription const& attention,
    AttentionEmissionValues const& emission_values,
    llvm::SmallVector<::mlir::Value> const& score_indices,
    SoftmaxRow const& row,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    auto item = masked_attention_score(builder, location, attention,
                                       emission_values, score_indices, values);
    auto positive_sum = ::mlir::arith::CmpFOp::create(
        builder, location, ::mlir::arith::CmpFPredicate::OGT, row.sum_exp,
        float_constant(builder, location, 0.0F));
    auto valid = ::mlir::arith::AndIOp::create(builder, location, item.valid,
                                               positive_sum);
    auto probability = ::mlir::arith::DivFOp::create(
        builder, location,
        ::mlir::math::ExpOp::create(
            builder, location,
            ::mlir::arith::SubFOp::create(builder, location, item.score,
                                          row.maximum)),
        row.sum_exp);
    return ::mlir::arith::SelectOp::create(
        builder, location, valid, probability,
        float_constant(builder, location, 0.0F));
}

::mlir::Value backward_probability(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    AttentionDescription const& attention,
    AttentionEmissionValues const& emission_values,
    llvm::SmallVector<::mlir::Value> const& score_indices,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    auto item = masked_attention_score(builder, location, attention,
                                       emission_values, score_indices, values);
    llvm::SmallVector<::mlir::Value> stats_indices{
        score_indices[0], score_indices[1], score_indices[2],
        index_constant(builder, location, 0)};
    auto stats = guarded_attention_load(
        builder, location, item.valid, attention.stats_uid,
        *attention.stats, nullptr, 0, stats_indices, values);
    auto probability = ::mlir::math::ExpOp::create(
        builder, location,
        ::mlir::arith::SubFOp::create(builder, location, item.score, stats));
    return ::mlir::arith::SelectOp::create(
        builder, location, item.valid, probability,
        float_constant(builder, location, 0.0F));
}

::mlir::Value dropout_mask_value(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    AttentionDescription const& attention,
    AttentionEmissionValues const& emission_values,
    llvm::SmallVector<::mlir::Value> const& score_indices,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    if (attention.dropout.kind == DropoutKind::kNone) {
        return float_constant(builder, location, 1.0F);
    }
    if (attention.dropout.kind == DropoutKind::kCustom) {
        return ::mlir::memref::LoadOp::create(
            builder, location, values.at(attention.dropout.mask_uid),
            broadcast_indices(builder, location, score_indices,
                              *attention.dropout.mask));
    }
    std::vector<std::int64_t> const score_dimensions{
        attention.q->dim[0], attention.q->dim[1], attention.q->dim[2],
        attention.s_kv};
    auto linear = linear_index(builder, location, score_indices,
                               score_dimensions);
    auto keep = stable_bernoulli_condition(
        builder, location, emission_values.seed, emission_values.offset,
        linear, 1.0 - attention.dropout.probability);
    return bool_as_float(builder, location, keep);
}

Status emit_rng_dump(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    AttentionDescription const& attention,
    AttentionEmissionValues const& emission_values,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    if (attention.rng_dump == nullptr) {
        return Status::ok();
    }
    return emit_flat_loop(
        builder, location, attention.rng_dump->dim, "SDPA.RNG_DUMP",
        [&](::mlir::OpBuilder& body_builder,
            ::mlir::Location body_location,
            llvm::SmallVector<::mlir::Value> const& indices) {
            auto mask = dropout_mask_value(body_builder, body_location,
                                           attention, emission_values,
                                           indices, values);
            ::mlir::memref::StoreOp::create(
                body_builder, body_location, mask,
                values.at(attention.rng_dump_uid), indices);
        });
}

Status emit_sdpa_forward(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    GenericOperationDesc const& operation,
    AttentionDescription const& attention,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    auto execution_values = values;
    for (auto const& [uid, tensor] :
         std::array<std::pair<std::int64_t, TensorDesc const*>, 4>{
             std::pair{attention.q_uid, attention.q},
             std::pair{attention.k_uid, attention.k},
             std::pair{attention.v_uid, attention.v},
             std::pair{attention.o_uid, attention.o}}) {
        if (tensor->ragged_offset_uid) {
            execution_values[uid] = flatten_ragged_buffer(
                builder, location, values.at(uid), *tensor);
        }
    }
    for (auto const& [uid, tensor] : attention.row_outputs) {
        if (tensor->ragged_offset_uid) {
            execution_values[uid] = flatten_ragged_buffer(
                builder, location, values.at(uid), *tensor);
        }
    }
    for (auto const& [uid, tensor] :
         std::array<std::pair<std::int64_t, TensorDesc const*>, 2>{
             std::pair{attention.page_k_uid, attention.page_k},
             std::pair{attention.page_v_uid, attention.page_v}}) {
        if (tensor != nullptr && tensor->ragged_offset_uid) {
            execution_values[uid] = flatten_ragged_buffer(
                builder, location, values.at(uid), *tensor);
        }
    }
    auto emission_values =
        prepare_attention_values(builder, location, attention,
                                 execution_values);
    auto status = emit_rng_dump(builder, location, attention, emission_values,
                                execution_values);
    if (status.is_bad()) return status;

    std::int64_t max_uid = 0;
    std::int64_t sum_uid = 0;
    bool const has_max = tensor_uid(operation, false, "Max", max_uid);
    bool const has_sum = tensor_uid(operation, false, "Sum_exp", sum_uid);
    auto const row_tensor = [&](std::int64_t uid) -> TensorDesc const* {
        auto const item = std::find_if(
            attention.row_outputs.begin(), attention.row_outputs.end(),
            [&](auto const& candidate) { return candidate.first == uid; });
        return item == attention.row_outputs.end() ? nullptr : item->second;
    };
    auto const* max_tensor = has_max ? row_tensor(max_uid) : nullptr;
    auto const* sum_tensor = has_sum ? row_tensor(sum_uid) : nullptr;
    if (attention.stats != nullptr || has_max || has_sum) {
        std::vector<std::int64_t> const row_dimensions{
            attention.q->dim[0], attention.q->dim[1], attention.q->dim[2]};
        status = emit_flat_loop(
            builder, location, row_dimensions, "SDPA.softmax_rows",
            [&](::mlir::OpBuilder& body_builder,
                ::mlir::Location body_location,
                llvm::SmallVector<::mlir::Value> const& row_indices) {
                auto row = compute_softmax_row(
                    body_builder, body_location, attention, emission_values,
                    row_indices, execution_values);
                llvm::SmallVector<::mlir::Value> output_indices{
                    row_indices[0], row_indices[1], row_indices[2],
                    index_constant(body_builder, body_location, 0)};
                auto valid = query_position_is_valid(
                    body_builder, body_location, attention, output_indices,
                    execution_values);
                if (attention.stats != nullptr) {
                    auto stats = ::mlir::arith::AddFOp::create(
                        body_builder, body_location, row.maximum,
                        ::mlir::math::LogOp::create(body_builder,
                                                    body_location,
                                                    row.sum_exp));
                    guarded_attention_store(
                        body_builder, body_location, valid, stats,
                        attention.stats_uid, *attention.stats, output_indices,
                        execution_values);
                }
                if (has_max) {
                    guarded_attention_store(
                        body_builder, body_location, valid, row.maximum,
                        max_uid, *max_tensor, output_indices,
                        execution_values);
                }
                if (has_sum) {
                    guarded_attention_store(
                        body_builder, body_location, valid, row.sum_exp,
                        sum_uid, *sum_tensor, output_indices,
                        execution_values);
                }
            });
        if (status.is_bad()) return status;
    }

    return emit_flat_loop(
        builder, location, attention.o->dim, "SDPA.O",
        [&](::mlir::OpBuilder& body_builder,
            ::mlir::Location body_location,
            llvm::SmallVector<::mlir::Value> const& output_indices) {
            llvm::SmallVector<::mlir::Value> row_indices{
                output_indices[0], output_indices[1], output_indices[2]};
            auto row = compute_softmax_row(
                body_builder, body_location, attention, emission_values,
                row_indices, execution_values);
            auto const value_head = query_to_source_head(
                body_builder, body_location, output_indices[1],
                attention.q->dim[1], attention.v->dim[1]);
            auto result = reduce_extent(
                body_builder, body_location, attention.s_kv,
                float_constant(body_builder, body_location, 0.0F),
                [&](::mlir::OpBuilder& reduction_builder,
                    ::mlir::Location reduction_location, ::mlir::Value kv,
                    ::mlir::Value accumulator) {
                    llvm::SmallVector<::mlir::Value> score_indices{
                        output_indices[0], output_indices[1],
                        output_indices[2], kv};
                    auto probability = forward_probability(
                        reduction_builder, reduction_location, attention,
                        emission_values, score_indices, row,
                        execution_values);
                    auto dropout_mask = dropout_mask_value(
                        reduction_builder, reduction_location, attention,
                        emission_values, score_indices, execution_values);
                    auto weighted_probability = ::mlir::arith::MulFOp::create(
                        reduction_builder, reduction_location,
                        ::mlir::arith::MulFOp::create(
                            reduction_builder, reduction_location,
                            probability, dropout_mask),
                        emission_values.dropout_scale);
                    llvm::SmallVector<::mlir::Value> v_indices{
                        output_indices[0], value_head, kv, output_indices[3]};
                    auto source_valid = score_is_valid(
                        reduction_builder, reduction_location, attention,
                        score_indices, execution_values);
                    auto v_value = guarded_attention_load(
                        reduction_builder, reduction_location, source_valid,
                        attention.v_uid, *attention.v, attention.page_v,
                        attention.page_v_uid, v_indices, execution_values);
                    return ::mlir::arith::AddFOp::create(
                        reduction_builder, reduction_location, accumulator,
                        ::mlir::arith::MulFOp::create(
                            reduction_builder, reduction_location,
                            weighted_probability, v_value));
                });
            auto valid = query_position_is_valid(
                body_builder, body_location, attention, output_indices,
                execution_values);
            guarded_attention_store(
                body_builder, body_location, valid, result, attention.o_uid,
                *attention.o, output_indices, execution_values);
        });
}

::mlir::Value output_gradient_dot(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    AttentionDescription const& attention,
    llvm::SmallVector<::mlir::Value> const& row_indices,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    auto row_valid = query_position_is_valid(builder, location, attention,
                                              row_indices, values);
    return reduce_extent(
        builder, location, attention.o->dim[3],
        float_constant(builder, location, 0.0F),
        [&](::mlir::OpBuilder& reduction_builder,
            ::mlir::Location reduction_location, ::mlir::Value embedding,
            ::mlir::Value accumulator) {
            llvm::SmallVector<::mlir::Value> indices{
                row_indices[0], row_indices[1], row_indices[2], embedding};
            auto o = guarded_attention_load(
                reduction_builder, reduction_location, row_valid,
                attention.o_uid, *attention.o, nullptr, 0, indices, values);
            auto d_o = guarded_attention_load(
                reduction_builder, reduction_location, row_valid,
                attention.do_uid, *attention.d_o, nullptr, 0, indices,
                values);
            return ::mlir::arith::AddFOp::create(
                reduction_builder, reduction_location, accumulator,
                ::mlir::arith::MulFOp::create(
                    reduction_builder, reduction_location, o, d_o));
        });
}

::mlir::Value unscaled_score_gradient(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    AttentionDescription const& attention,
    AttentionEmissionValues const& emission_values,
    llvm::SmallVector<::mlir::Value> const& score_indices,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    llvm::SmallVector<::mlir::Value> row_indices{
        score_indices[0], score_indices[1], score_indices[2]};
    auto const value_head = query_to_source_head(
        builder, location, score_indices[1], attention.q->dim[1],
        attention.v->dim[1]);
    auto source_valid = score_is_valid(builder, location, attention,
                                       score_indices, values);
    auto d_p = reduce_extent(
        builder, location, attention.v->dim[3],
        float_constant(builder, location, 0.0F),
        [&](::mlir::OpBuilder& reduction_builder,
            ::mlir::Location reduction_location, ::mlir::Value embedding,
            ::mlir::Value accumulator) {
            llvm::SmallVector<::mlir::Value> do_indices{
                score_indices[0], score_indices[1], score_indices[2],
                embedding};
            llvm::SmallVector<::mlir::Value> v_indices{
                score_indices[0], value_head, score_indices[3], embedding};
            auto d_o = guarded_attention_load(
                reduction_builder, reduction_location, source_valid,
                attention.do_uid, *attention.d_o, nullptr, 0, do_indices,
                values);
            auto v = guarded_attention_load(
                reduction_builder, reduction_location, source_valid,
                attention.v_uid, *attention.v, nullptr, 0, v_indices,
                values);
            return ::mlir::arith::AddFOp::create(
                reduction_builder, reduction_location, accumulator,
                ::mlir::arith::MulFOp::create(
                    reduction_builder, reduction_location, d_o, v));
        });
    auto mask = dropout_mask_value(builder, location, attention,
                                   emission_values, score_indices, values);
    auto masked_d_p = ::mlir::arith::MulFOp::create(builder, location, d_p,
                                                    mask);
    auto softmax_sum = ::mlir::arith::MulFOp::create(
        builder, location,
        output_gradient_dot(builder, location, attention, row_indices, values),
        emission_values.dropout_scale_inv);
    auto centered = ::mlir::arith::SubFOp::create(
        builder, location, masked_d_p, softmax_sum);
    auto probability = backward_probability(
        builder, location, attention, emission_values, score_indices, values);
    return ::mlir::arith::MulFOp::create(
        builder, location,
        ::mlir::arith::MulFOp::create(builder, location, centered,
                                      probability),
        emission_values.dropout_scale);
}

Status emit_sdpa_backward(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    AttentionDescription const& attention,
    std::map<std::int64_t, ::mlir::Value> const& input_values) {
    auto values = input_values;
    for (auto const& [uid, tensor] :
         std::array<std::pair<std::int64_t, TensorDesc const*>, 9>{
             std::pair{attention.q_uid, attention.q},
             std::pair{attention.k_uid, attention.k},
             std::pair{attention.v_uid, attention.v},
             std::pair{attention.o_uid, attention.o},
             std::pair{attention.do_uid, attention.d_o},
             std::pair{attention.stats_uid, attention.stats},
             std::pair{attention.dq_uid, attention.d_q},
             std::pair{attention.dk_uid, attention.d_k},
             std::pair{attention.dv_uid, attention.d_v}}) {
        if (tensor->ragged_offset_uid) {
            values[uid] = flatten_ragged_buffer(
                builder, location, input_values.at(uid), *tensor);
        }
    }
    auto emission_values =
        prepare_attention_values(builder, location, attention, values);
    auto status = emit_rng_dump(builder, location, attention, emission_values,
                                values);
    if (status.is_bad()) return status;

    status = emit_flat_loop(
        builder, location, attention.d_q->dim, "SDPA_BWD.dQ",
        [&](::mlir::OpBuilder& body_builder,
            ::mlir::Location body_location,
            llvm::SmallVector<::mlir::Value> const& output_indices) {
            auto const key_head = query_to_source_head(
                body_builder, body_location, output_indices[1],
                attention.q->dim[1], attention.k->dim[1]);
            auto result = reduce_extent(
                body_builder, body_location, attention.s_kv,
                float_constant(body_builder, body_location, 0.0F),
                [&](::mlir::OpBuilder& reduction_builder,
                    ::mlir::Location reduction_location, ::mlir::Value kv,
                    ::mlir::Value accumulator) {
                    llvm::SmallVector<::mlir::Value> score_indices{
                        output_indices[0], output_indices[1],
                        output_indices[2], kv};
                    auto gradient = unscaled_score_gradient(
                        reduction_builder, reduction_location, attention,
                        emission_values, score_indices, values);
                    gradient = ::mlir::arith::MulFOp::create(
                        reduction_builder, reduction_location, gradient,
                        emission_values.attention_scale);
                    llvm::SmallVector<::mlir::Value> k_indices{
                        output_indices[0], key_head, kv, output_indices[3]};
                    auto source_valid = score_is_valid(
                        reduction_builder, reduction_location, attention,
                        score_indices, values);
                    auto k = guarded_attention_load(
                        reduction_builder, reduction_location, source_valid,
                        attention.k_uid, *attention.k, nullptr, 0, k_indices,
                        values);
                    return ::mlir::arith::AddFOp::create(
                        reduction_builder, reduction_location, accumulator,
                        ::mlir::arith::MulFOp::create(
                            reduction_builder, reduction_location, gradient,
                            k));
                });
            auto valid = query_position_is_valid(
                body_builder, body_location, attention, output_indices,
                values);
            guarded_attention_store(
                body_builder, body_location, valid, result,
                attention.dq_uid, *attention.d_q, output_indices, values);
        });
    if (status.is_bad()) return status;

    status = emit_flat_loop(
        builder, location, attention.d_k->dim, "SDPA_BWD.dK",
        [&](::mlir::OpBuilder& body_builder,
            ::mlir::Location body_location,
            llvm::SmallVector<::mlir::Value> const& output_indices) {
            auto const group_size =
                attention.q->dim[1] / attention.k->dim[1];
            auto result = reduce_extents(
                body_builder, body_location,
                {group_size, attention.q->dim[2]},
                float_constant(body_builder, body_location, 0.0F),
                [&](::mlir::OpBuilder& reduction_builder,
                    ::mlir::Location reduction_location,
                    ::mlir::ValueRange reduction_indices,
                    ::mlir::Value accumulator) {
                    auto query_head = ::mlir::arith::AddIOp::create(
                        reduction_builder, reduction_location,
                        ::mlir::arith::MulIOp::create(
                            reduction_builder, reduction_location,
                            output_indices[1],
                            index_constant(reduction_builder,
                                           reduction_location, group_size)),
                        reduction_indices[0]);
                    llvm::SmallVector<::mlir::Value> score_indices{
                        output_indices[0], query_head, reduction_indices[1],
                        output_indices[2]};
                    auto gradient = unscaled_score_gradient(
                        reduction_builder, reduction_location, attention,
                        emission_values, score_indices, values);
                    gradient = ::mlir::arith::MulFOp::create(
                        reduction_builder, reduction_location, gradient,
                        emission_values.attention_scale);
                    llvm::SmallVector<::mlir::Value> q_indices{
                        output_indices[0], query_head, reduction_indices[1],
                        output_indices[3]};
                    auto source_valid = score_is_valid(
                        reduction_builder, reduction_location, attention,
                        score_indices, values);
                    auto q = guarded_attention_load(
                        reduction_builder, reduction_location, source_valid,
                        attention.q_uid, *attention.q, nullptr, 0, q_indices,
                        values);
                    return ::mlir::arith::AddFOp::create(
                        reduction_builder, reduction_location, accumulator,
                        ::mlir::arith::MulFOp::create(
                            reduction_builder, reduction_location, gradient,
                            q));
                });
            auto valid = key_position_is_valid(
                body_builder, body_location, attention, output_indices,
                values);
            guarded_attention_store(
                body_builder, body_location, valid, result,
                attention.dk_uid, *attention.d_k, output_indices, values);
        });
    if (status.is_bad()) return status;

    status = emit_flat_loop(
        builder, location, attention.d_v->dim, "SDPA_BWD.dV",
        [&](::mlir::OpBuilder& body_builder,
            ::mlir::Location body_location,
            llvm::SmallVector<::mlir::Value> const& output_indices) {
            auto const group_size =
                attention.q->dim[1] / attention.v->dim[1];
            auto result = reduce_extents(
                body_builder, body_location,
                {group_size, attention.q->dim[2]},
                float_constant(body_builder, body_location, 0.0F),
                [&](::mlir::OpBuilder& reduction_builder,
                    ::mlir::Location reduction_location,
                    ::mlir::ValueRange reduction_indices,
                    ::mlir::Value accumulator) {
                    auto query_head = ::mlir::arith::AddIOp::create(
                        reduction_builder, reduction_location,
                        ::mlir::arith::MulIOp::create(
                            reduction_builder, reduction_location,
                            output_indices[1],
                            index_constant(reduction_builder,
                                           reduction_location, group_size)),
                        reduction_indices[0]);
                    llvm::SmallVector<::mlir::Value> score_indices{
                        output_indices[0], query_head, reduction_indices[1],
                        output_indices[2]};
                    auto probability = backward_probability(
                        reduction_builder, reduction_location, attention,
                        emission_values, score_indices, values);
                    auto mask = dropout_mask_value(
                        reduction_builder, reduction_location, attention,
                        emission_values, score_indices, values);
                    auto weighted_probability = ::mlir::arith::MulFOp::create(
                        reduction_builder, reduction_location,
                        ::mlir::arith::MulFOp::create(
                            reduction_builder, reduction_location,
                            probability, mask),
                        emission_values.dropout_scale);
                    llvm::SmallVector<::mlir::Value> do_indices{
                        output_indices[0], query_head, reduction_indices[1],
                        output_indices[3]};
                    auto source_valid = score_is_valid(
                        reduction_builder, reduction_location, attention,
                        score_indices, values);
                    auto d_o = guarded_attention_load(
                        reduction_builder, reduction_location, source_valid,
                        attention.do_uid, *attention.d_o, nullptr, 0,
                        do_indices, values);
                    return ::mlir::arith::AddFOp::create(
                        reduction_builder, reduction_location, accumulator,
                        ::mlir::arith::MulFOp::create(
                            reduction_builder, reduction_location,
                            weighted_probability, d_o));
                });
            auto valid = key_position_is_valid(
                body_builder, body_location, attention, output_indices,
                values);
            guarded_attention_store(
                body_builder, body_location, valid, result,
                attention.dv_uid, *attention.d_v, output_indices, values);
        });
    if (status.is_bad()) return status;

    if (attention.d_bias != nullptr) {
        status = emit_flat_loop(
            builder, location, attention.d_bias->dim, "SDPA_BWD.dBias",
            [&](::mlir::OpBuilder& body_builder,
                ::mlir::Location body_location,
                llvm::SmallVector<::mlir::Value> const& output_indices) {
                std::vector<std::int64_t> reduction_extents;
                std::vector<std::size_t> reduction_axes;
                std::vector<std::int64_t> const score_dimensions{
                    attention.q->dim[0], attention.q->dim[1],
                    attention.q->dim[2], attention.s_kv};
                for (std::size_t axis = 0; axis < 4; ++axis) {
                    if (attention.d_bias->dim[axis] == 1 &&
                        score_dimensions[axis] != 1) {
                        reduction_axes.push_back(axis);
                        reduction_extents.push_back(score_dimensions[axis]);
                    }
                }
                auto contribution = [&](::mlir::OpBuilder& item_builder,
                                        ::mlir::Location item_location,
                                        ::mlir::ValueRange reduction_indices) {
                    llvm::SmallVector<::mlir::Value> score_indices =
                        output_indices;
                    for (std::size_t index = 0;
                         index < reduction_axes.size(); ++index) {
                        score_indices[reduction_axes[index]] =
                            reduction_indices[index];
                    }
                    return unscaled_score_gradient(
                        item_builder, item_location, attention,
                        emission_values, score_indices, values);
                };
                ::mlir::Value result;
                if (reduction_extents.empty()) {
                    result = contribution(body_builder, body_location, {});
                } else {
                    result = reduce_extents(
                        body_builder, body_location, reduction_extents,
                        float_constant(body_builder, body_location, 0.0F),
                        [&](::mlir::OpBuilder& reduction_builder,
                            ::mlir::Location reduction_location,
                            ::mlir::ValueRange reduction_indices,
                            ::mlir::Value accumulator) {
                            return ::mlir::arith::AddFOp::create(
                                reduction_builder, reduction_location,
                                accumulator,
                                contribution(reduction_builder,
                                             reduction_location,
                                             reduction_indices));
                        });
                }
                ::mlir::memref::StoreOp::create(
                    body_builder, body_location, result,
                    values.at(attention.dbias_uid), output_indices);
            });
    }
    if (status.is_bad()) return status;

    if (attention.d_sink != nullptr) {
        status = emit_flat_loop(
            builder, location, attention.d_sink->dim, "SDPA_BWD.dSink",
            [&](::mlir::OpBuilder& body_builder,
                ::mlir::Location body_location,
                llvm::SmallVector<::mlir::Value> const& output_indices) {
                auto result = reduce_extents(
                    body_builder, body_location,
                    {attention.q->dim[0], attention.q->dim[2]},
                    float_constant(body_builder, body_location, 0.0F),
                    [&](::mlir::OpBuilder& reduction_builder,
                        ::mlir::Location reduction_location,
                        ::mlir::ValueRange reduction_indices,
                        ::mlir::Value accumulator) {
                        llvm::SmallVector<::mlir::Value> row_indices{
                            reduction_indices[0], output_indices[1],
                            reduction_indices[1]};
                        llvm::SmallVector<::mlir::Value> stats_indices{
                            row_indices[0], row_indices[1], row_indices[2],
                            index_constant(reduction_builder,
                                           reduction_location, 0)};
                        auto valid = query_position_is_valid(
                            reduction_builder, reduction_location, attention,
                            stats_indices, values);
                        auto stats = guarded_attention_load(
                            reduction_builder, reduction_location, valid,
                            attention.stats_uid, *attention.stats, nullptr, 0,
                            stats_indices, values);
                        auto sink = ::mlir::memref::LoadOp::create(
                            reduction_builder, reduction_location,
                            values.at(attention.sink_uid), output_indices);
                        auto probability = ::mlir::math::ExpOp::create(
                            reduction_builder, reduction_location,
                            ::mlir::arith::SubFOp::create(
                                reduction_builder, reduction_location, sink,
                                stats));
                        ::mlir::Value contribution =
                            ::mlir::arith::NegFOp::create(
                                reduction_builder, reduction_location,
                                ::mlir::arith::MulFOp::create(
                                    reduction_builder, reduction_location,
                                    probability,
                                    output_gradient_dot(
                                        reduction_builder,
                                        reduction_location, attention,
                                        row_indices, values)));
                        contribution = ::mlir::arith::SelectOp::create(
                            reduction_builder, reduction_location, valid,
                            contribution,
                            float_constant(reduction_builder,
                                           reduction_location, 0.0F));
                        return ::mlir::arith::AddFOp::create(
                            reduction_builder, reduction_location,
                            accumulator, contribution);
                    });
                ::mlir::memref::StoreOp::create(
                    body_builder, body_location, result,
                    values.at(attention.dsink_uid), output_indices);
            });
    }
    return status;
}

Status emit_attention(
    OperationTag tag,
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    GenericOperationDesc const& operation,
    SerializedGraph const& graph,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    AttentionDescription attention;
    auto status = decode_attention(tag, operation, graph, "attention",
                                   attention);
    if (status.is_bad()) return status;
    if (tag == OperationTag::kSdpa) {
        return emit_sdpa_forward(builder, location, operation, attention,
                                 values);
    }
    return emit_sdpa_backward(builder, location, attention, values);
}

}  // namespace

bool is_sequence_operation(OperationTag tag) noexcept {
    switch (tag) {
        case OperationTag::kRng:
        case OperationTag::kRope:
        case OperationTag::kRopeBwd:
        case OperationTag::kSdpa:
        case OperationTag::kSdpaBwd:
            return true;
        default:
            return false;
    }
}

bool is_sequence_metadata_input(OperationTag tag,
                                std::string_view port,
                                DataType data_type) noexcept {
    if (tag == OperationTag::kRng) {
        return (port == "Seed" || port == "Offset") &&
               data_type == DataType::kInt64;
    }
    if (tag != OperationTag::kSdpa && tag != OperationTag::kSdpaBwd) {
        return false;
    }
    if (port == "SEQ_LEN_Q" || port == "SEQ_LEN_KV") {
        return data_type == DataType::kInt32;
    }
    if (tag == OperationTag::kSdpa &&
        (port == "Page_table_K" || port == "Page_table_V")) {
        return data_type == DataType::kInt32;
    }
    if (tag == OperationTag::kSdpa && port == "Block_mask") {
        return data_type == DataType::kUInt8;
    }
    if (port == "Seed" || port == "Offset") {
        return data_type == DataType::kInt64;
    }
    return false;
}

Status validate_sequence_operation(OperationTag tag,
                                   GenericOperationDesc const& operation,
                                   SerializedGraph const& graph,
                                   std::size_t node_index) {
    auto const path = "nodes[" + std::to_string(node_index) + "]";
    switch (tag) {
        case OperationTag::kRng:
            return validate_rng(operation, graph, path);
        case OperationTag::kRope:
        case OperationTag::kRopeBwd:
            return validate_rope(tag, operation, graph, path);
        case OperationTag::kSdpa:
        case OperationTag::kSdpaBwd: {
            AttentionDescription attention;
            return decode_attention(tag, operation, graph, path, attention);
        }
        default:
            return fail(ErrorCode::kInvalidValue, path,
                        "operation is not a C4 sequence operation");
    }
}

Status emit_sequence_operation(
    OperationTag tag,
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    GenericOperationDesc const& operation,
    SerializedGraph const& graph,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    switch (tag) {
        case OperationTag::kRng:
            return emit_rng(builder, location, operation, graph, values);
        case OperationTag::kRope:
        case OperationTag::kRopeBwd:
            return emit_rope(tag, builder, location, operation, graph, values);
        case OperationTag::kSdpa:
        case OperationTag::kSdpaBwd:
            return emit_attention(tag, builder, location, operation, graph,
                                  values);
        default:
            return fail(ErrorCode::kInvalidValue, "sequence",
                        "validated sequence operation has no emitter");
    }
}

}  // namespace deepforge::compiler
