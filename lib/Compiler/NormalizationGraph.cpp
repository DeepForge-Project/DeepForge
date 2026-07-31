#include "NormalizationGraph.h"

#include "DeepForge/Import/Capability.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/SmallVector.h"

#include <algorithm>
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

bool has_nonempty_input_list(GenericOperationDesc const& operation) {
    return std::any_of(
        operation.input_lists.begin(), operation.input_lists.end(),
        [](auto const& item) { return !item.second.empty(); });
}

template <typename Map>
bool ports_match(Map const& ports,
                 std::initializer_list<std::string_view> required,
                 std::initializer_list<std::string_view> optional = {}) {
    for (auto name : required) {
        if (!ports.contains(std::string(name))) {
            return false;
        }
    }
    for (auto const& [name, reference] : ports) {
        (void)reference;
        auto allowed = std::find(required.begin(), required.end(), name) !=
                           required.end() ||
                       std::find(optional.begin(), optional.end(), name) !=
                           optional.end();
        if (!allowed) {
            return false;
        }
    }
    return true;
}

struct Binding {
    std::int64_t uid = 0;
    TensorDesc const* tensor = nullptr;
};

Status bind_port(GenericOperationDesc const& operation,
                 SerializedGraph const& graph,
                 bool input,
                 std::string_view port,
                 std::string const& path,
                 Binding& output) {
    auto const& ports = input ? operation.inputs : operation.outputs;
    auto const it = ports.find(std::string(port));
    if (it == ports.end()) {
        return fail(ErrorCode::kInvalidValue, path,
                    "required tensor port is absent: " + std::string(port));
    }
    auto const* uid = std::get_if<std::int64_t>(&it->second);
    if (uid == nullptr) {
        return fail(ErrorCode::kUnsupportedOperation, path,
                    "CPU normalization requires UID tensor references");
    }
    auto const tensor_it = graph.tensors.find(*uid);
    if (tensor_it == graph.tensors.end()) {
        return fail(ErrorCode::kMissingUid, path,
                    "tensor reference is unresolved: " + std::string(port));
    }
    output = Binding{*uid, &tensor_it->second};
    return Status::ok();
}

Status bind_optional_port(GenericOperationDesc const& operation,
                          SerializedGraph const& graph,
                          bool input,
                          std::string_view port,
                          std::string const& path,
                          std::optional<Binding>& output) {
    auto const& ports = input ? operation.inputs : operation.outputs;
    if (!ports.contains(std::string(port))) {
        output.reset();
        return Status::ok();
    }
    Binding binding;
    auto status = bind_port(operation, graph, input, port, path, binding);
    if (status.is_bad()) {
        return status;
    }
    output = binding;
    return Status::ok();
}

bool same_shape(Binding const& lhs, Binding const& rhs) {
    return lhs.tensor->dim == rhs.tensor->dim;
}

bool scalar_shape(TensorDesc const& tensor, std::size_t rank) {
    return tensor.dim.size() == rank &&
           std::all_of(tensor.dim.begin(), tensor.dim.end(),
                       [](std::int64_t dimension) { return dimension == 1; });
}

bool broadcast_shape(TensorDesc const& tensor, TensorDesc const& target) {
    if (tensor.dim.size() != target.dim.size()) {
        return false;
    }
    for (std::size_t axis = 0; axis < target.dim.size(); ++axis) {
        if (tensor.dim[axis] != 1 &&
            tensor.dim[axis] != target.dim[axis]) {
            return false;
        }
    }
    return true;
}

std::vector<std::int64_t> per_channel_shape(TensorDesc const& x) {
    std::vector<std::int64_t> shape(x.dim.size(), 1);
    if (shape.size() >= 2) {
        shape[1] = x.dim[1];
    }
    return shape;
}

std::vector<std::int64_t> instance_stats_shape(TensorDesc const& x) {
    auto shape = x.dim;
    for (std::size_t axis = 2; axis < shape.size(); ++axis) {
        shape[axis] = 1;
    }
    return shape;
}

std::vector<std::int64_t> parameter_stats_shape(
    TensorDesc const& x,
    TensorDesc const& scale,
    bool preserve_batch) {
    auto shape = x.dim;
    for (std::size_t axis = 0; axis < shape.size(); ++axis) {
        if (preserve_batch && axis == 0) {
            continue;
        }
        if (scale.dim[axis] != 1) {
            shape[axis] = 1;
        }
    }
    return shape;
}

Status validate_reduction_extent(TensorDesc const& x,
                                 std::vector<std::int64_t> const& stats_shape,
                                 std::string const& path) {
    std::uint64_t count = 1;
    for (std::size_t axis = 0; axis < x.dim.size(); ++axis) {
        if (stats_shape[axis] == 1 && x.dim[axis] != 1) {
            if (count >
                static_cast<std::uint64_t>(
                    std::numeric_limits<std::int64_t>::max()) /
                    static_cast<std::uint64_t>(x.dim[axis])) {
                return fail(ErrorCode::kDimensionOverflow, path,
                            "normalization reduction extent overflows int64");
            }
            count *= static_cast<std::uint64_t>(x.dim[axis]);
        }
    }
    return Status::ok();
}

Status validate_compute_type(GenericOperationDesc const& operation,
                             std::string const& path) {
    std::string_view compute_type;
    if (!read_string_attribute(operation, "compute_data_type", compute_type) ||
        compute_type != "FLOAT") {
        return fail(ErrorCode::kUnsupportedDataType, path,
                    "C3 normalization requires FLOAT compute data type");
    }
    return Status::ok();
}

struct ForwardNorm {
    OperationTag tag = OperationTag::kLayerNorm;
    Binding x;
    Binding scale;
    std::optional<Binding> bias;
    Binding epsilon;
    Binding y;
    std::optional<Binding> mean;
    std::optional<Binding> inv_variance;
    std::optional<Binding> previous_mean;
    std::optional<Binding> previous_variance;
    std::optional<Binding> momentum;
    std::optional<Binding> next_mean;
    std::optional<Binding> next_variance;
    std::vector<std::int64_t> stats_shape;
    bool training = true;
    bool rms = false;
    bool batch = false;
};

Status decode_forward_norm(OperationTag tag,
                           GenericOperationDesc const& operation,
                           SerializedGraph const& graph,
                           std::string const& path,
                           ForwardNorm& output) {
    auto status = validate_compute_type(operation, path);
    if (status.is_bad()) {
        return status;
    }
    if (has_nonempty_input_list(operation)) {
        return fail(ErrorCode::kUnsupportedOperation, path,
                    "distributed peer statistics are deferred");
    }
    output.tag = tag;

    if (tag == OperationTag::kBatchNorm) {
        if (!ports_match(operation.inputs,
                         {"X", "SCALE", "BIAS", "EPSILON"},
                         {"PREV_RUNNING_MEAN", "PREV_RUNNING_VAR",
                          "MOMENTUM"}) ||
            !ports_match(operation.outputs,
                         {"Y", "MEAN", "INV_VARIANCE"},
                         {"NEXT_RUNNING_MEAN", "NEXT_RUNNING_VAR"})) {
            return fail(ErrorCode::kInvalidValue, path,
                        "BATCHNORM ports are incomplete");
        }
        output.batch = true;
        status = bind_port(operation, graph, true, "X", path, output.x);
        if (status.is_bad()) return status;
        status = bind_port(operation, graph, true, "SCALE", path,
                           output.scale);
        if (status.is_bad()) return status;
        Binding bias;
        status = bind_port(operation, graph, true, "BIAS", path, bias);
        if (status.is_bad()) return status;
        output.bias = bias;
        status = bind_port(operation, graph, true, "EPSILON", path,
                           output.epsilon);
        if (status.is_bad()) return status;
        status = bind_port(operation, graph, false, "Y", path, output.y);
        if (status.is_bad()) return status;
        Binding mean;
        status = bind_port(operation, graph, false, "MEAN", path, mean);
        if (status.is_bad()) return status;
        output.mean = mean;
        Binding inv;
        status = bind_port(operation, graph, false, "INV_VARIANCE", path,
                           inv);
        if (status.is_bad()) return status;
        output.inv_variance = inv;
        status = bind_optional_port(operation, graph, true,
                                    "PREV_RUNNING_MEAN", path,
                                    output.previous_mean);
        if (status.is_bad()) return status;
        status = bind_optional_port(operation, graph, true,
                                    "PREV_RUNNING_VAR", path,
                                    output.previous_variance);
        if (status.is_bad()) return status;
        status = bind_optional_port(operation, graph, true, "MOMENTUM", path,
                                    output.momentum);
        if (status.is_bad()) return status;
        status = bind_optional_port(operation, graph, false,
                                    "NEXT_RUNNING_MEAN", path,
                                    output.next_mean);
        if (status.is_bad()) return status;
        status = bind_optional_port(operation, graph, false,
                                    "NEXT_RUNNING_VAR", path,
                                    output.next_variance);
        if (status.is_bad()) return status;
        auto const running_inputs = output.previous_mean.has_value() +
                                    output.previous_variance.has_value() +
                                    output.momentum.has_value();
        auto const running_outputs = output.next_mean.has_value() +
                                     output.next_variance.has_value();
        if ((running_inputs != 0 && running_inputs != 3) ||
            (running_outputs != 0 && running_outputs != 2) ||
            (running_inputs == 0) != (running_outputs == 0)) {
            return fail(ErrorCode::kInvalidValue, path,
                        "BATCHNORM running-stat ports must be all present or all absent");
        }
        if (output.x.tensor->dim.size() < 2) {
            return fail(ErrorCode::kInvalidShape, path,
                        "BATCHNORM requires rank at least two");
        }
        output.stats_shape = per_channel_shape(*output.x.tensor);
    } else {
        output.rms = tag == OperationTag::kRmsNorm;
        auto const bias_optional = output.rms;
        if (!ports_match(operation.inputs,
                         bias_optional
                             ? std::initializer_list<std::string_view>{
                                   "X", "SCALE", "EPSILON"}
                             : std::initializer_list<std::string_view>{
                                   "X", "SCALE", "BIAS", "EPSILON"},
                         bias_optional
                             ? std::initializer_list<std::string_view>{"BIAS"}
                             : std::initializer_list<std::string_view>{})) {
            return fail(ErrorCode::kInvalidValue, path,
                        "normalization forward input ports are incomplete");
        }
        std::string_view phase;
        if (!read_string_attribute(operation, "forward_phase", phase) ||
            (phase != "TRAINING" && phase != "INFERENCE")) {
            return fail(ErrorCode::kInvalidValue, path,
                        "normalization forward phase is invalid");
        }
        output.training = phase == "TRAINING";
        auto outputs_valid = output.training
                                 ? (output.rms
                                        ? ports_match(operation.outputs,
                                                      {"Y", "INV_VARIANCE"})
                                        : ports_match(operation.outputs,
                                                      {"Y", "MEAN",
                                                       "INV_VARIANCE"}))
                                 : ports_match(operation.outputs, {"Y"});
        if (!outputs_valid) {
            return fail(ErrorCode::kInvalidValue, path,
                        "normalization outputs do not match forward phase");
        }
        status = bind_port(operation, graph, true, "X", path, output.x);
        if (status.is_bad()) return status;
        status = bind_port(operation, graph, true, "SCALE", path,
                           output.scale);
        if (status.is_bad()) return status;
        status = bind_optional_port(operation, graph, true, "BIAS", path,
                                    output.bias);
        if (status.is_bad()) return status;
        status = bind_port(operation, graph, true, "EPSILON", path,
                           output.epsilon);
        if (status.is_bad()) return status;
        status = bind_port(operation, graph, false, "Y", path, output.y);
        if (status.is_bad()) return status;
        if (output.training && !output.rms) {
            Binding mean;
            status = bind_port(operation, graph, false, "MEAN", path, mean);
            if (status.is_bad()) return status;
            output.mean = mean;
        }
        if (output.training) {
            Binding inv;
            status = bind_port(operation, graph, false, "INV_VARIANCE", path,
                               inv);
            if (status.is_bad()) return status;
            output.inv_variance = inv;
        }
        if (tag == OperationTag::kInstanceNorm) {
            if (output.x.tensor->dim.size() < 3) {
                return fail(ErrorCode::kInvalidShape, path,
                            "INSTANCE_NORM requires rank at least three");
            }
            output.stats_shape = instance_stats_shape(*output.x.tensor);
            if (output.scale.tensor->dim !=
                per_channel_shape(*output.x.tensor)) {
                return fail(ErrorCode::kInvalidShape, path,
                            "INSTANCE_NORM parameters require per-channel shape");
            }
        } else {
            output.stats_shape = parameter_stats_shape(
                *output.x.tensor, *output.scale.tensor,
                tag == OperationTag::kAdaLayerNorm);
        }
    }

    if (!same_shape(output.x, output.y) ||
        !broadcast_shape(*output.scale.tensor, *output.x.tensor) ||
        (output.bias && output.bias->tensor->dim != output.scale.tensor->dim) ||
        !scalar_shape(*output.epsilon.tensor, output.x.tensor->dim.size())) {
        return fail(ErrorCode::kInvalidShape, path,
                    "normalization X/Y, parameter, or epsilon shapes are inconsistent");
    }
    if (output.batch && output.scale.tensor->dim != output.stats_shape) {
        return fail(ErrorCode::kInvalidShape, path,
                    "BATCHNORM parameters must have per-channel shape");
    }
    if (output.mean && output.mean->tensor->dim != output.stats_shape) {
        return fail(ErrorCode::kInvalidShape, path,
                    "normalization MEAN shape is inconsistent");
    }
    if (output.inv_variance &&
        output.inv_variance->tensor->dim != output.stats_shape) {
        return fail(ErrorCode::kInvalidShape, path,
                    "normalization INV_VARIANCE shape is inconsistent");
    }
    if (output.previous_mean) {
        if (output.previous_mean->tensor->dim != output.stats_shape ||
            output.previous_variance->tensor->dim != output.stats_shape ||
            output.next_mean->tensor->dim != output.stats_shape ||
            output.next_variance->tensor->dim != output.stats_shape ||
            !scalar_shape(*output.momentum->tensor,
                          output.x.tensor->dim.size())) {
            return fail(ErrorCode::kInvalidShape, path,
                        "BATCHNORM running-stat shapes are inconsistent");
        }
    }
    return validate_reduction_extent(*output.x.tensor, output.stats_shape,
                                     path);
}

struct InferenceNorm {
    Binding x;
    Binding scale;
    Binding bias;
    Binding mean;
    Binding inv_variance;
    Binding y;
};

Status decode_batchnorm_inference(GenericOperationDesc const& operation,
                                  SerializedGraph const& graph,
                                  std::string const& path,
                                  InferenceNorm& output) {
    auto status = validate_compute_type(operation, path);
    if (status.is_bad()) return status;
    if (!ports_match(operation.inputs,
                     {"X", "SCALE", "BIAS", "MEAN", "INV_VARIANCE"}) ||
        !ports_match(operation.outputs, {"Y"}) ||
        has_nonempty_input_list(operation)) {
        return fail(ErrorCode::kInvalidValue, path,
                    "BATCHNORM_INFERENCE ports are incomplete");
    }
    status = bind_port(operation, graph, true, "X", path, output.x);
    if (status.is_bad()) return status;
    status = bind_port(operation, graph, true, "SCALE", path, output.scale);
    if (status.is_bad()) return status;
    status = bind_port(operation, graph, true, "BIAS", path, output.bias);
    if (status.is_bad()) return status;
    status = bind_port(operation, graph, true, "MEAN", path, output.mean);
    if (status.is_bad()) return status;
    status = bind_port(operation, graph, true, "INV_VARIANCE", path,
                       output.inv_variance);
    if (status.is_bad()) return status;
    status = bind_port(operation, graph, false, "Y", path, output.y);
    if (status.is_bad()) return status;
    if (output.x.tensor->dim.size() < 2 || !same_shape(output.x, output.y)) {
        return fail(ErrorCode::kInvalidShape, path,
                    "BATCHNORM_INFERENCE X/Y shapes are inconsistent");
    }
    auto expected = per_channel_shape(*output.x.tensor);
    if (output.scale.tensor->dim != expected ||
        output.bias.tensor->dim != expected ||
        output.mean.tensor->dim != expected ||
        output.inv_variance.tensor->dim != expected) {
        return fail(ErrorCode::kInvalidShape, path,
                    "BATCHNORM_INFERENCE parameters require per-channel shape");
    }
    return Status::ok();
}

struct BackwardNorm {
    OperationTag tag = OperationTag::kLayerNormBprop;
    Binding dy;
    Binding x;
    Binding scale;
    std::optional<Binding> mean;
    Binding inv_variance;
    std::optional<Binding> epsilon;
    Binding dx;
    Binding dscale;
    std::optional<Binding> dbias;
    std::vector<std::int64_t> stats_shape;
    bool rms = false;
};

Status decode_backward_norm(OperationTag tag,
                            GenericOperationDesc const& operation,
                            SerializedGraph const& graph,
                            std::string const& path,
                            BackwardNorm& output) {
    auto status = validate_compute_type(operation, path);
    if (status.is_bad()) return status;
    if (has_nonempty_input_list(operation)) {
        return fail(ErrorCode::kUnsupportedOperation, path,
                    "distributed peer statistics are deferred");
    }
    output.tag = tag;
    output.rms = tag == OperationTag::kRmsNormBprop;
    auto const has_epsilon = tag == OperationTag::kLayerNormBprop ||
                             tag == OperationTag::kAdaLayerNormBprop;
    auto input_ports_valid =
        output.rms
            ? ports_match(operation.inputs,
                          {"DY", "X", "SCALE", "INV_VARIANCE"})
            : (has_epsilon
                   ? ports_match(operation.inputs,
                                 {"DY", "X", "SCALE", "MEAN",
                                  "INV_VARIANCE", "EPSILON"})
                   : ports_match(operation.inputs,
                                 {"DY", "X", "SCALE", "MEAN",
                                  "INV_VARIANCE"}));
    auto output_ports_valid =
        (output.rms || tag == OperationTag::kAdaLayerNormBprop)
            ? ports_match(operation.outputs, {"DX", "DSCALE"}, {"DBIAS"})
            : ports_match(operation.outputs, {"DX", "DSCALE", "DBIAS"});
    if (!input_ports_valid || !output_ports_valid) {
        return fail(ErrorCode::kInvalidValue, path,
                    "normalization backward ports are incomplete");
    }
    status = bind_port(operation, graph, true, "DY", path, output.dy);
    if (status.is_bad()) return status;
    status = bind_port(operation, graph, true, "X", path, output.x);
    if (status.is_bad()) return status;
    status = bind_port(operation, graph, true, "SCALE", path, output.scale);
    if (status.is_bad()) return status;
    if (!output.rms) {
        Binding mean;
        status = bind_port(operation, graph, true, "MEAN", path, mean);
        if (status.is_bad()) return status;
        output.mean = mean;
    }
    status = bind_port(operation, graph, true, "INV_VARIANCE", path,
                       output.inv_variance);
    if (status.is_bad()) return status;
    if (has_epsilon) {
        Binding epsilon;
        status = bind_port(operation, graph, true, "EPSILON", path, epsilon);
        if (status.is_bad()) return status;
        output.epsilon = epsilon;
    }
    status = bind_port(operation, graph, false, "DX", path, output.dx);
    if (status.is_bad()) return status;
    status = bind_port(operation, graph, false, "DSCALE", path,
                       output.dscale);
    if (status.is_bad()) return status;
    status = bind_optional_port(operation, graph, false, "DBIAS", path,
                                output.dbias);
    if (status.is_bad()) return status;

    if (!same_shape(output.x, output.dy) || !same_shape(output.x, output.dx) ||
        !broadcast_shape(*output.scale.tensor, *output.x.tensor) ||
        output.dscale.tensor->dim != output.scale.tensor->dim ||
        (output.dbias &&
         output.dbias->tensor->dim != output.scale.tensor->dim) ||
        (output.epsilon &&
         !scalar_shape(*output.epsilon->tensor,
                       output.x.tensor->dim.size()))) {
        return fail(ErrorCode::kInvalidShape, path,
                    "normalization backward data or parameter shapes are inconsistent");
    }
    if (tag == OperationTag::kDbn) {
        if (output.x.tensor->dim.size() < 2) {
            return fail(ErrorCode::kInvalidShape, path,
                        "DBN requires rank at least two");
        }
        output.stats_shape = per_channel_shape(*output.x.tensor);
        if (output.scale.tensor->dim != output.stats_shape) {
            return fail(ErrorCode::kInvalidShape, path,
                        "DBN scale requires per-channel shape");
        }
    } else if (tag == OperationTag::kInstanceNormBprop) {
        if (output.x.tensor->dim.size() < 3) {
            return fail(ErrorCode::kInvalidShape, path,
                        "INSTANCE_NORM_BPROP requires rank at least three");
        }
        output.stats_shape = instance_stats_shape(*output.x.tensor);
        if (output.scale.tensor->dim !=
            per_channel_shape(*output.x.tensor)) {
            return fail(ErrorCode::kInvalidShape, path,
                        "INSTANCE_NORM_BPROP parameters require per-channel shape");
        }
    } else {
        output.stats_shape = parameter_stats_shape(
            *output.x.tensor, *output.scale.tensor,
            tag == OperationTag::kAdaLayerNormBprop);
    }
    if ((!output.rms && output.mean->tensor->dim != output.stats_shape) ||
        output.inv_variance.tensor->dim != output.stats_shape) {
        return fail(ErrorCode::kInvalidShape, path,
                    "normalization backward statistic shapes are inconsistent");
    }
    return validate_reduction_extent(*output.x.tensor, output.stats_shape,
                                     path);
}

Status validate_genstats(GenericOperationDesc const& operation,
                         SerializedGraph const& graph,
                         std::string const& path) {
    auto status = validate_compute_type(operation, path);
    if (status.is_bad()) return status;
    if (!ports_match(operation.inputs, {"X"}) ||
        !ports_match(operation.outputs, {"SUM", "SQ_SUM"}) ||
        has_nonempty_input_list(operation)) {
        return fail(ErrorCode::kInvalidValue, path,
                    "GENSTATS requires X, SUM, and SQ_SUM");
    }
    Binding x;
    Binding sum;
    Binding square_sum;
    status = bind_port(operation, graph, true, "X", path, x);
    if (status.is_bad()) return status;
    status = bind_port(operation, graph, false, "SUM", path, sum);
    if (status.is_bad()) return status;
    status = bind_port(operation, graph, false, "SQ_SUM", path, square_sum);
    if (status.is_bad()) return status;
    auto expected = per_channel_shape(*x.tensor);
    if (x.tensor->dim.size() < 2 || sum.tensor->dim != expected ||
        square_sum.tensor->dim != expected) {
        return fail(ErrorCode::kInvalidShape, path,
                    "GENSTATS outputs require per-channel shape");
    }
    return validate_reduction_extent(*x.tensor, expected, path);
}

Status validate_bn_finalize(GenericOperationDesc const& operation,
                            SerializedGraph const& graph,
                            std::string const& path) {
    auto status = validate_compute_type(operation, path);
    if (status.is_bad()) return status;
    if (!ports_match(operation.inputs,
                     {"SUM", "SQ_SUM", "SCALE", "BIAS", "EPSILON",
                      "ACCUM_COUNT", "PREV_RUNNING_MEAN",
                      "PREV_RUNNING_VAR", "MOMENTUM"}) ||
        !ports_match(operation.outputs,
                     {"EQ_SCALE", "EQ_BIAS", "MEAN", "INV_VARIANCE",
                      "NEXT_RUNNING_MEAN", "NEXT_RUNNING_VAR"}) ||
        has_nonempty_input_list(operation)) {
        return fail(ErrorCode::kInvalidValue, path,
                    "BN_FINALIZE ports are incomplete");
    }
    Binding sum;
    status = bind_port(operation, graph, true, "SUM", path, sum);
    if (status.is_bad()) return status;
    for (auto port : {"SQ_SUM", "SCALE", "BIAS", "PREV_RUNNING_MEAN",
                      "PREV_RUNNING_VAR"}) {
        Binding item;
        status = bind_port(operation, graph, true, port, path, item);
        if (status.is_bad()) return status;
        if (!same_shape(sum, item)) {
            return fail(ErrorCode::kInvalidShape, path,
                        "BN_FINALIZE channel tensors require identical shapes");
        }
    }
    for (auto port : {"EQ_SCALE", "EQ_BIAS", "MEAN", "INV_VARIANCE",
                      "NEXT_RUNNING_MEAN", "NEXT_RUNNING_VAR"}) {
        Binding item;
        status = bind_port(operation, graph, false, port, path, item);
        if (status.is_bad()) return status;
        if (!same_shape(sum, item)) {
            return fail(ErrorCode::kInvalidShape, path,
                        "BN_FINALIZE outputs require SUM shape");
        }
    }
    for (auto port : {"EPSILON", "ACCUM_COUNT", "MOMENTUM"}) {
        Binding scalar;
        status = bind_port(operation, graph, true, port, path, scalar);
        if (status.is_bad()) return status;
        if (!scalar_shape(*scalar.tensor, sum.tensor->dim.size())) {
            return fail(ErrorCode::kInvalidShape, path,
                        "BN_FINALIZE scalar tensors must be all-ones shapes");
        }
    }
    return Status::ok();
}

Status validate_dbn_weight(GenericOperationDesc const& operation,
                           SerializedGraph const& graph,
                           std::string const& path) {
    auto status = validate_compute_type(operation, path);
    if (status.is_bad()) return status;
    if (!ports_match(operation.inputs,
                     {"DY", "X", "SCALE", "MEAN", "INV_VARIANCE"}) ||
        !ports_match(operation.outputs,
                     {"DSCALE", "DBIAS", "EQ_BIAS", "EQ_SCALE_DY",
                      "EQ_SCALE_X"}) ||
        has_nonempty_input_list(operation)) {
        return fail(ErrorCode::kInvalidValue, path,
                    "DBN_WEIGHT ports are incomplete");
    }
    Binding x;
    Binding dy;
    status = bind_port(operation, graph, true, "X", path, x);
    if (status.is_bad()) return status;
    status = bind_port(operation, graph, true, "DY", path, dy);
    if (status.is_bad()) return status;
    if (!same_shape(x, dy) || x.tensor->dim.size() < 2) {
        return fail(ErrorCode::kInvalidShape, path,
                    "DBN_WEIGHT X and DY shapes must match");
    }
    auto expected = per_channel_shape(*x.tensor);
    for (auto port : {"SCALE", "MEAN", "INV_VARIANCE"}) {
        Binding item;
        status = bind_port(operation, graph, true, port, path, item);
        if (status.is_bad()) return status;
        if (item.tensor->dim != expected) {
            return fail(ErrorCode::kInvalidShape, path,
                        "DBN_WEIGHT inputs require per-channel shape");
        }
    }
    for (auto port : {"DSCALE", "DBIAS", "EQ_BIAS", "EQ_SCALE_DY",
                      "EQ_SCALE_X"}) {
        Binding item;
        status = bind_port(operation, graph, false, port, path, item);
        if (status.is_bad()) return status;
        if (item.tensor->dim != expected) {
            return fail(ErrorCode::kInvalidShape, path,
                        "DBN_WEIGHT outputs require per-channel shape");
        }
    }
    return validate_reduction_extent(*x.tensor, expected, path);
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
                      std::string_view operation_name,
                      Body&& body) {
    std::uint64_t count = 1;
    for (auto dimension : dimensions) {
        if (dimension <= 0 ||
            count > static_cast<std::uint64_t>(
                        std::numeric_limits<std::int64_t>::max()) /
                        static_cast<std::uint64_t>(dimension)) {
            return fail(ErrorCode::kDimensionOverflow,
                        std::string(operation_name),
                        "element count does not fit index");
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

llvm::SmallVector<::mlir::Value> broadcast_indices(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    llvm::SmallVector<::mlir::Value> const& x_indices,
    std::vector<std::int64_t> const& dimensions) {
    llvm::SmallVector<::mlir::Value> result;
    result.reserve(dimensions.size());
    for (std::size_t axis = 0; axis < dimensions.size(); ++axis) {
        result.push_back(dimensions[axis] == 1
                             ? index_constant(builder, location, 0)
                             : x_indices[axis]);
    }
    return result;
}

llvm::SmallVector<::mlir::Value> broadcast_indices(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    llvm::SmallVector<::mlir::Value> const& x_indices,
    TensorDesc const& tensor) {
    return broadcast_indices(builder, location, x_indices, tensor.dim);
}

::mlir::Value load_binding(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    Binding const& binding,
    llvm::SmallVector<::mlir::Value> const& indices,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    return ::mlir::memref::LoadOp::create(builder, location,
                                          values.at(binding.uid), indices);
}

::mlir::Value load_scalar(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    Binding const& binding,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    llvm::SmallVector<::mlir::Value> indices(
        binding.tensor->dim.size(), index_constant(builder, location, 0));
    return load_binding(builder, location, binding, indices, values);
}

std::vector<std::size_t> reduction_axes(
    TensorDesc const& x,
    std::vector<std::int64_t> const& output_shape) {
    std::vector<std::size_t> axes;
    for (std::size_t axis = 0; axis < x.dim.size(); ++axis) {
        if (output_shape[axis] == 1 && x.dim[axis] != 1) {
            axes.push_back(axis);
        }
    }
    return axes;
}

std::int64_t reduction_count(
    TensorDesc const& x,
    std::vector<std::int64_t> const& output_shape) {
    std::int64_t count = 1;
    for (auto axis : reduction_axes(x, output_shape)) {
        count *= x.dim[axis];
    }
    return count;
}

struct Sums {
    ::mlir::Value sum;
    ::mlir::Value square_sum;
};

Sums emit_sum_and_square(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    Binding const& x,
    std::vector<std::int64_t> const& output_shape,
    llvm::SmallVector<::mlir::Value> const& base_indices,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    auto axes = reduction_axes(*x.tensor, output_shape);
    llvm::SmallVector<::mlir::Value> x_indices(base_indices.begin(),
                                                base_indices.end());
    if (axes.empty()) {
        auto value = load_binding(builder, location, x, x_indices, values);
        return {value,
                ::mlir::arith::MulFOp::create(builder, location, value,
                                              value)};
    }
    llvm::SmallVector<::mlir::Value> lowers;
    llvm::SmallVector<::mlir::Value> uppers;
    llvm::SmallVector<::mlir::Value> steps;
    for (auto axis : axes) {
        lowers.push_back(index_constant(builder, location, 0));
        uppers.push_back(
            index_constant(builder, location, x.tensor->dim[axis]));
        steps.push_back(index_constant(builder, location, 1));
    }
    auto reduction = ::mlir::scf::buildLoopNest(
        builder, location, lowers, uppers, steps,
        ::mlir::ValueRange{float_constant(builder, location, 0.0F),
                           float_constant(builder, location, 0.0F)},
        [&](::mlir::OpBuilder& reduction_builder,
            ::mlir::Location reduction_location,
            ::mlir::ValueRange reduction_indices,
            ::mlir::ValueRange iter_args) -> ::mlir::scf::ValueVector {
            for (std::size_t index = 0; index < axes.size(); ++index) {
                x_indices[axes[index]] = reduction_indices[index];
            }
            auto value = load_binding(reduction_builder, reduction_location, x,
                                      x_indices, values);
            auto square = ::mlir::arith::MulFOp::create(
                reduction_builder, reduction_location, value, value);
            return {
                ::mlir::arith::AddFOp::create(
                    reduction_builder, reduction_location, iter_args[0],
                    value),
                ::mlir::arith::AddFOp::create(
                    reduction_builder, reduction_location, iter_args[1],
                    square)};
        });
    builder.setInsertionPointAfter(reduction.loops.front());
    return {reduction.results[0], reduction.results[1]};
}

struct Statistics {
    ::mlir::Value mean;
    ::mlir::Value variance;
    ::mlir::Value inv_variance;
};

Statistics make_statistics(::mlir::OpBuilder& builder,
                           ::mlir::Location location,
                           Sums const& sums,
                           std::int64_t count,
                           ::mlir::Value epsilon,
                           bool rms) {
    auto divisor = float_constant(builder, location,
                                  static_cast<float>(count));
    auto mean = ::mlir::arith::DivFOp::create(builder, location, sums.sum,
                                              divisor);
    auto mean_square = ::mlir::arith::DivFOp::create(
        builder, location, sums.square_sum, divisor);
    auto variance = rms
                        ? static_cast<::mlir::Value>(mean_square)
                        : static_cast<::mlir::Value>(
                              ::mlir::arith::MaximumFOp::create(
                                  builder, location,
                                  ::mlir::arith::SubFOp::create(
                                      builder, location, mean_square,
                                      ::mlir::arith::MulFOp::create(
                                          builder, location, mean, mean)),
                                  float_constant(builder, location, 0.0F)));
    auto root = ::mlir::math::SqrtOp::create(
        builder, location,
        ::mlir::arith::AddFOp::create(builder, location, variance, epsilon));
    auto inverse = ::mlir::arith::DivFOp::create(
        builder, location, float_constant(builder, location, 1.0F), root);
    return {rms ? float_constant(builder, location, 0.0F) : mean, variance,
            inverse};
}

Statistics compute_statistics(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    ForwardNorm const& norm,
    llvm::SmallVector<::mlir::Value> const& base_indices,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    auto sums = emit_sum_and_square(builder, location, norm.x,
                                    norm.stats_shape, base_indices, values);
    return make_statistics(
        builder, location, sums,
        reduction_count(*norm.x.tensor, norm.stats_shape),
        load_scalar(builder, location, norm.epsilon, values), norm.rms);
}

Status emit_forward_norm(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    ForwardNorm const& norm,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    if (norm.training) {
        auto status = emit_flat_loop(
            builder, location, norm.stats_shape, "normalization statistics",
            [&](::mlir::OpBuilder& body_builder,
                ::mlir::Location body_location,
                llvm::SmallVector<::mlir::Value> const& stats_indices) {
                auto statistics = compute_statistics(
                    body_builder, body_location, norm, stats_indices, values);
                if (norm.mean) {
                    ::mlir::memref::StoreOp::create(
                        body_builder, body_location, statistics.mean,
                        values.at(norm.mean->uid), stats_indices);
                }
                ::mlir::memref::StoreOp::create(
                    body_builder, body_location, statistics.inv_variance,
                    values.at(norm.inv_variance->uid), stats_indices);

                if (norm.previous_mean) {
                    auto previous_mean = load_binding(
                        body_builder, body_location, *norm.previous_mean,
                        stats_indices, values);
                    auto previous_variance = load_binding(
                        body_builder, body_location, *norm.previous_variance,
                        stats_indices, values);
                    auto momentum = load_scalar(body_builder, body_location,
                                                *norm.momentum, values);
                    auto one_minus = ::mlir::arith::SubFOp::create(
                        body_builder, body_location,
                        float_constant(body_builder, body_location, 1.0F),
                        momentum);
                    auto next_mean = ::mlir::arith::AddFOp::create(
                        body_builder, body_location,
                        ::mlir::arith::MulFOp::create(
                            body_builder, body_location, previous_mean,
                            one_minus),
                        ::mlir::arith::MulFOp::create(
                            body_builder, body_location, statistics.mean,
                            momentum));
                    auto count = reduction_count(*norm.x.tensor,
                                                 norm.stats_shape);
                    auto sample_variance = statistics.variance;
                    if (count > 1) {
                        sample_variance = ::mlir::arith::MulFOp::create(
                            body_builder, body_location, statistics.variance,
                            float_constant(body_builder, body_location,
                                           static_cast<float>(count) /
                                               static_cast<float>(count - 1)));
                    }
                    auto next_variance = ::mlir::arith::AddFOp::create(
                        body_builder, body_location,
                        ::mlir::arith::MulFOp::create(
                            body_builder, body_location, previous_variance,
                            one_minus),
                        ::mlir::arith::MulFOp::create(
                            body_builder, body_location, sample_variance,
                            momentum));
                    ::mlir::memref::StoreOp::create(
                        body_builder, body_location, next_mean,
                        values.at(norm.next_mean->uid), stats_indices);
                    ::mlir::memref::StoreOp::create(
                        body_builder, body_location, next_variance,
                        values.at(norm.next_variance->uid), stats_indices);
                }
            });
        if (status.is_bad()) return status;
    }

    return emit_flat_loop(
        builder, location, norm.x.tensor->dim, "normalization forward",
        [&](::mlir::OpBuilder& body_builder,
            ::mlir::Location body_location,
            llvm::SmallVector<::mlir::Value> const& x_indices) {
            ::mlir::Value mean;
            ::mlir::Value inverse;
            auto stats_indices = broadcast_indices(
                body_builder, body_location, x_indices, norm.stats_shape);
            if (norm.training) {
                mean = norm.rms
                           ? float_constant(body_builder, body_location, 0.0F)
                           : load_binding(body_builder, body_location,
                                          *norm.mean, stats_indices, values);
                inverse = load_binding(body_builder, body_location,
                                       *norm.inv_variance, stats_indices,
                                       values);
            } else {
                auto statistics = compute_statistics(
                    body_builder, body_location, norm, x_indices, values);
                mean = statistics.mean;
                inverse = statistics.inv_variance;
            }
            auto x_value = load_binding(body_builder, body_location, norm.x,
                                        x_indices, values);
            auto normalized = norm.rms
                                  ? static_cast<::mlir::Value>(
                                        ::mlir::arith::MulFOp::create(
                                            body_builder, body_location,
                                            x_value, inverse))
                                  : static_cast<::mlir::Value>(
                                        ::mlir::arith::MulFOp::create(
                                            body_builder, body_location,
                                            ::mlir::arith::SubFOp::create(
                                                body_builder, body_location,
                                                x_value, mean),
                                            inverse));
            auto scale_indices = broadcast_indices(
                body_builder, body_location, x_indices, *norm.scale.tensor);
            auto scaled = ::mlir::arith::MulFOp::create(
                body_builder, body_location, normalized,
                load_binding(body_builder, body_location, norm.scale,
                             scale_indices, values));
            ::mlir::Value result = scaled;
            if (norm.bias) {
                result = ::mlir::arith::AddFOp::create(
                    body_builder, body_location, scaled,
                    load_binding(body_builder, body_location, *norm.bias,
                                 scale_indices, values));
            }
            ::mlir::memref::StoreOp::create(
                body_builder, body_location, result, values.at(norm.y.uid),
                x_indices);
        });
}

Status emit_batchnorm_inference(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    InferenceNorm const& norm,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    return emit_flat_loop(
        builder, location, norm.x.tensor->dim, "batchnorm inference",
        [&](::mlir::OpBuilder& body_builder,
            ::mlir::Location body_location,
            llvm::SmallVector<::mlir::Value> const& x_indices) {
            auto parameter_indices = broadcast_indices(
                body_builder, body_location, x_indices, *norm.scale.tensor);
            auto centered = ::mlir::arith::SubFOp::create(
                body_builder, body_location,
                load_binding(body_builder, body_location, norm.x, x_indices,
                             values),
                load_binding(body_builder, body_location, norm.mean,
                             parameter_indices, values));
            auto normalized = ::mlir::arith::MulFOp::create(
                body_builder, body_location, centered,
                load_binding(body_builder, body_location, norm.inv_variance,
                             parameter_indices, values));
            auto result = ::mlir::arith::AddFOp::create(
                body_builder, body_location,
                ::mlir::arith::MulFOp::create(
                    body_builder, body_location, normalized,
                    load_binding(body_builder, body_location, norm.scale,
                                 parameter_indices, values)),
                load_binding(body_builder, body_location, norm.bias,
                             parameter_indices, values));
            ::mlir::memref::StoreOp::create(
                body_builder, body_location, result, values.at(norm.y.uid),
                x_indices);
        });
}

struct GradientSums {
    ::mlir::Value dscale;
    ::mlir::Value dbias;
};

GradientSums gradient_contribution(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    BackwardNorm const& norm,
    llvm::SmallVector<::mlir::Value> const& x_indices,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    auto stats_indices = broadcast_indices(
        builder, location, x_indices, *norm.inv_variance.tensor);
    auto inverse = load_binding(builder, location, norm.inv_variance,
                                stats_indices, values);
    auto x_value = load_binding(builder, location, norm.x, x_indices, values);
    ::mlir::Value normalized;
    if (norm.rms) {
        normalized = ::mlir::arith::MulFOp::create(
            builder, location, x_value, inverse);
    } else {
        normalized = ::mlir::arith::MulFOp::create(
            builder, location,
            ::mlir::arith::SubFOp::create(
                builder, location, x_value,
                load_binding(builder, location, *norm.mean, stats_indices,
                             values)),
            inverse);
    }
    auto dy = load_binding(builder, location, norm.dy, x_indices, values);
    return {::mlir::arith::MulFOp::create(builder, location, dy, normalized),
            dy};
}

Status emit_parameter_gradients(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    BackwardNorm const& norm,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    return emit_flat_loop(
        builder, location, norm.scale.tensor->dim,
        "normalization parameter gradients",
        [&](::mlir::OpBuilder& body_builder,
            ::mlir::Location body_location,
            llvm::SmallVector<::mlir::Value> const& parameter_indices) {
            auto axes = reduction_axes(*norm.x.tensor,
                                       norm.scale.tensor->dim);
            llvm::SmallVector<::mlir::Value> x_indices(
                parameter_indices.begin(), parameter_indices.end());
            GradientSums sums;
            if (axes.empty()) {
                sums = gradient_contribution(body_builder, body_location,
                                             norm, x_indices, values);
            } else {
                llvm::SmallVector<::mlir::Value> lowers;
                llvm::SmallVector<::mlir::Value> uppers;
                llvm::SmallVector<::mlir::Value> steps;
                for (auto axis : axes) {
                    lowers.push_back(
                        index_constant(body_builder, body_location, 0));
                    uppers.push_back(index_constant(
                        body_builder, body_location, norm.x.tensor->dim[axis]));
                    steps.push_back(
                        index_constant(body_builder, body_location, 1));
                }
                auto reduction = ::mlir::scf::buildLoopNest(
                    body_builder, body_location, lowers, uppers, steps,
                    ::mlir::ValueRange{
                        float_constant(body_builder, body_location, 0.0F),
                        float_constant(body_builder, body_location, 0.0F)},
                    [&](::mlir::OpBuilder& reduction_builder,
                        ::mlir::Location reduction_location,
                        ::mlir::ValueRange reduction_indices,
                        ::mlir::ValueRange iter_args)
                        -> ::mlir::scf::ValueVector {
                        for (std::size_t index = 0; index < axes.size();
                             ++index) {
                            x_indices[axes[index]] = reduction_indices[index];
                        }
                        auto contribution = gradient_contribution(
                            reduction_builder, reduction_location, norm,
                            x_indices, values);
                        return {
                            ::mlir::arith::AddFOp::create(
                                reduction_builder, reduction_location,
                                iter_args[0], contribution.dscale),
                            ::mlir::arith::AddFOp::create(
                                reduction_builder, reduction_location,
                                iter_args[1], contribution.dbias)};
                    });
                body_builder.setInsertionPointAfter(reduction.loops.front());
                sums = {reduction.results[0], reduction.results[1]};
            }
            ::mlir::memref::StoreOp::create(
                body_builder, body_location, sums.dscale,
                values.at(norm.dscale.uid), parameter_indices);
            if (norm.dbias) {
                ::mlir::memref::StoreOp::create(
                    body_builder, body_location, sums.dbias,
                    values.at(norm.dbias->uid), parameter_indices);
            }
        });
}

struct BackwardGroupSums {
    ::mlir::Value g;
    ::mlir::Value g_product;
};

BackwardGroupSums emit_backward_group_sums(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    BackwardNorm const& norm,
    llvm::SmallVector<::mlir::Value> const& base_indices,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    auto axes = reduction_axes(*norm.x.tensor, norm.stats_shape);
    auto accumulate = [&](::mlir::OpBuilder& current_builder,
                          ::mlir::Location current_location,
                          llvm::SmallVector<::mlir::Value> const& x_indices) {
        auto scale_indices = broadcast_indices(
            current_builder, current_location, x_indices,
            *norm.scale.tensor);
        auto g = ::mlir::arith::MulFOp::create(
            current_builder, current_location,
            load_binding(current_builder, current_location, norm.dy,
                         x_indices, values),
            load_binding(current_builder, current_location, norm.scale,
                         scale_indices, values));
        auto x_value = load_binding(current_builder, current_location, norm.x,
                                    x_indices, values);
        ::mlir::Value product_value = x_value;
        if (!norm.rms) {
            auto stats_indices = broadcast_indices(
                current_builder, current_location, x_indices,
                *norm.inv_variance.tensor);
            product_value = ::mlir::arith::MulFOp::create(
                current_builder, current_location,
                ::mlir::arith::SubFOp::create(
                    current_builder, current_location, x_value,
                    load_binding(current_builder, current_location,
                                 *norm.mean, stats_indices, values)),
                load_binding(current_builder, current_location,
                             norm.inv_variance, stats_indices, values));
        }
        return BackwardGroupSums{
            g, ::mlir::arith::MulFOp::create(
                   current_builder, current_location, g, product_value)};
    };

    llvm::SmallVector<::mlir::Value> x_indices(base_indices.begin(),
                                                base_indices.end());
    if (axes.empty()) {
        return accumulate(builder, location, x_indices);
    }
    llvm::SmallVector<::mlir::Value> lowers;
    llvm::SmallVector<::mlir::Value> uppers;
    llvm::SmallVector<::mlir::Value> steps;
    for (auto axis : axes) {
        lowers.push_back(index_constant(builder, location, 0));
        uppers.push_back(
            index_constant(builder, location, norm.x.tensor->dim[axis]));
        steps.push_back(index_constant(builder, location, 1));
    }
    auto reduction = ::mlir::scf::buildLoopNest(
        builder, location, lowers, uppers, steps,
        ::mlir::ValueRange{float_constant(builder, location, 0.0F),
                           float_constant(builder, location, 0.0F)},
        [&](::mlir::OpBuilder& reduction_builder,
            ::mlir::Location reduction_location,
            ::mlir::ValueRange reduction_indices,
            ::mlir::ValueRange iter_args) -> ::mlir::scf::ValueVector {
            for (std::size_t index = 0; index < axes.size(); ++index) {
                x_indices[axes[index]] = reduction_indices[index];
            }
            auto contribution =
                accumulate(reduction_builder, reduction_location, x_indices);
            return {
                ::mlir::arith::AddFOp::create(
                    reduction_builder, reduction_location, iter_args[0],
                    contribution.g),
                ::mlir::arith::AddFOp::create(
                    reduction_builder, reduction_location, iter_args[1],
                    contribution.g_product)};
        });
    builder.setInsertionPointAfter(reduction.loops.front());
    return {reduction.results[0], reduction.results[1]};
}

Status emit_data_gradient(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    BackwardNorm const& norm,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    auto const count = reduction_count(*norm.x.tensor, norm.stats_shape);
    return emit_flat_loop(
        builder, location, norm.x.tensor->dim, "normalization data gradient",
        [&](::mlir::OpBuilder& body_builder,
            ::mlir::Location body_location,
            llvm::SmallVector<::mlir::Value> const& x_indices) {
            auto stats_indices = broadcast_indices(
                body_builder, body_location, x_indices,
                *norm.inv_variance.tensor);
            auto scale_indices = broadcast_indices(
                body_builder, body_location, x_indices, *norm.scale.tensor);
            auto inverse = load_binding(
                body_builder, body_location, norm.inv_variance, stats_indices,
                values);
            auto x_value = load_binding(body_builder, body_location, norm.x,
                                        x_indices, values);
            auto g = ::mlir::arith::MulFOp::create(
                body_builder, body_location,
                load_binding(body_builder, body_location, norm.dy, x_indices,
                             values),
                load_binding(body_builder, body_location, norm.scale,
                             scale_indices, values));
            auto sums = emit_backward_group_sums(
                body_builder, body_location, norm, x_indices, values);
            auto divisor = float_constant(body_builder, body_location,
                                          static_cast<float>(count));
            ::mlir::Value dx;
            if (norm.rms) {
                auto inverse_squared = ::mlir::arith::MulFOp::create(
                    body_builder, body_location, inverse, inverse);
                auto correction = ::mlir::arith::MulFOp::create(
                    body_builder, body_location, x_value,
                    ::mlir::arith::MulFOp::create(
                        body_builder, body_location, inverse_squared,
                        ::mlir::arith::DivFOp::create(
                            body_builder, body_location, sums.g_product,
                            divisor)));
                dx = ::mlir::arith::MulFOp::create(
                    body_builder, body_location, inverse,
                    ::mlir::arith::SubFOp::create(
                        body_builder, body_location, g, correction));
            } else {
                auto mean = load_binding(body_builder, body_location,
                                         *norm.mean, stats_indices, values);
                auto xhat = ::mlir::arith::MulFOp::create(
                    body_builder, body_location,
                    ::mlir::arith::SubFOp::create(
                        body_builder, body_location, x_value, mean),
                    inverse);
                auto centered_gradient = ::mlir::arith::SubFOp::create(
                    body_builder, body_location,
                    ::mlir::arith::SubFOp::create(
                        body_builder, body_location, g,
                        ::mlir::arith::DivFOp::create(
                            body_builder, body_location, sums.g, divisor)),
                    ::mlir::arith::MulFOp::create(
                        body_builder, body_location, xhat,
                        ::mlir::arith::DivFOp::create(
                            body_builder, body_location, sums.g_product,
                            divisor)));
                dx = ::mlir::arith::MulFOp::create(
                    body_builder, body_location, inverse,
                    centered_gradient);
            }
            ::mlir::memref::StoreOp::create(
                body_builder, body_location, dx, values.at(norm.dx.uid),
                x_indices);
        });
}

Status emit_backward_norm(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    BackwardNorm const& norm,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    auto status = emit_parameter_gradients(builder, location, norm, values);
    if (status.is_bad()) return status;
    return emit_data_gradient(builder, location, norm, values);
}

Status emit_genstats(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    GenericOperationDesc const& operation,
    SerializedGraph const& graph,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    Binding x;
    Binding sum;
    Binding square_sum;
    auto status = bind_port(operation, graph, true, "X", "GENSTATS", x);
    if (status.is_bad()) return status;
    status = bind_port(operation, graph, false, "SUM", "GENSTATS", sum);
    if (status.is_bad()) return status;
    status = bind_port(operation, graph, false, "SQ_SUM", "GENSTATS",
                       square_sum);
    if (status.is_bad()) return status;
    return emit_flat_loop(
        builder, location, sum.tensor->dim, "GENSTATS",
        [&](::mlir::OpBuilder& body_builder,
            ::mlir::Location body_location,
            llvm::SmallVector<::mlir::Value> const& output_indices) {
            auto sums = emit_sum_and_square(body_builder, body_location, x,
                                            sum.tensor->dim, output_indices,
                                            values);
            ::mlir::memref::StoreOp::create(
                body_builder, body_location, sums.sum, values.at(sum.uid),
                output_indices);
            ::mlir::memref::StoreOp::create(
                body_builder, body_location, sums.square_sum,
                values.at(square_sum.uid), output_indices);
        });
}

Status emit_bn_finalize(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    GenericOperationDesc const& operation,
    SerializedGraph const& graph,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    auto bind_input = [&](std::string_view port, Binding& binding) {
        return bind_port(operation, graph, true, port, "BN_FINALIZE",
                         binding);
    };
    auto bind_output = [&](std::string_view port, Binding& binding) {
        return bind_port(operation, graph, false, port, "BN_FINALIZE",
                         binding);
    };
    Binding sum;
    Binding square_sum;
    Binding scale;
    Binding bias;
    Binding epsilon;
    Binding count;
    Binding previous_mean;
    Binding previous_variance;
    Binding momentum;
    Binding equivalent_scale;
    Binding equivalent_bias;
    Binding mean;
    Binding inverse;
    Binding next_mean;
    Binding next_variance;
    for (auto item : {
             std::pair<std::string_view, Binding*>{"SUM", &sum},
             {"SQ_SUM", &square_sum}, {"SCALE", &scale}, {"BIAS", &bias},
             {"EPSILON", &epsilon}, {"ACCUM_COUNT", &count},
             {"PREV_RUNNING_MEAN", &previous_mean},
             {"PREV_RUNNING_VAR", &previous_variance},
             {"MOMENTUM", &momentum}}) {
        auto status = bind_input(item.first, *item.second);
        if (status.is_bad()) return status;
    }
    for (auto item : {
             std::pair<std::string_view, Binding*>{"EQ_SCALE",
                                                   &equivalent_scale},
             {"EQ_BIAS", &equivalent_bias}, {"MEAN", &mean},
             {"INV_VARIANCE", &inverse},
             {"NEXT_RUNNING_MEAN", &next_mean},
             {"NEXT_RUNNING_VAR", &next_variance}}) {
        auto status = bind_output(item.first, *item.second);
        if (status.is_bad()) return status;
    }
    return emit_flat_loop(
        builder, location, sum.tensor->dim, "BN_FINALIZE",
        [&](::mlir::OpBuilder& body_builder,
            ::mlir::Location body_location,
            llvm::SmallVector<::mlir::Value> const& indices) {
            auto count_value = load_scalar(body_builder, body_location, count,
                                           values);
            auto sum_value = load_binding(body_builder, body_location, sum,
                                          indices, values);
            auto square_sum_value = load_binding(
                body_builder, body_location, square_sum, indices, values);
            auto mean_value = ::mlir::arith::DivFOp::create(
                body_builder, body_location, sum_value, count_value);
            auto population_variance = ::mlir::arith::MaximumFOp::create(
                body_builder, body_location,
                ::mlir::arith::SubFOp::create(
                    body_builder, body_location,
                    ::mlir::arith::DivFOp::create(
                        body_builder, body_location, square_sum_value,
                        count_value),
                    ::mlir::arith::MulFOp::create(
                        body_builder, body_location, mean_value, mean_value)),
                float_constant(body_builder, body_location, 0.0F));
            auto inverse_value = ::mlir::arith::DivFOp::create(
                body_builder, body_location,
                float_constant(body_builder, body_location, 1.0F),
                ::mlir::math::SqrtOp::create(
                    body_builder, body_location,
                    ::mlir::arith::AddFOp::create(
                        body_builder, body_location, population_variance,
                        load_scalar(body_builder, body_location, epsilon,
                                    values))));
            auto equivalent_scale_value = ::mlir::arith::MulFOp::create(
                body_builder, body_location,
                load_binding(body_builder, body_location, scale, indices,
                             values),
                inverse_value);
            auto equivalent_bias_value = ::mlir::arith::SubFOp::create(
                body_builder, body_location,
                load_binding(body_builder, body_location, bias, indices,
                             values),
                ::mlir::arith::MulFOp::create(
                    body_builder, body_location, mean_value,
                    equivalent_scale_value));
            auto one = float_constant(body_builder, body_location, 1.0F);
            auto momentum_value = load_scalar(
                body_builder, body_location, momentum, values);
            auto old_factor = ::mlir::arith::SubFOp::create(
                body_builder, body_location, one, momentum_value);
            auto next_mean_value = ::mlir::arith::AddFOp::create(
                body_builder, body_location,
                ::mlir::arith::MulFOp::create(
                    body_builder, body_location,
                    load_binding(body_builder, body_location, previous_mean,
                                 indices, values),
                    old_factor),
                ::mlir::arith::MulFOp::create(
                    body_builder, body_location, mean_value,
                    momentum_value));
            auto count_exceeds_one = ::mlir::arith::CmpFOp::create(
                body_builder, body_location,
                ::mlir::arith::CmpFPredicate::OGT, count_value, one);
            auto sample_variance = ::mlir::arith::SelectOp::create(
                body_builder, body_location, count_exceeds_one,
                ::mlir::arith::MulFOp::create(
                    body_builder, body_location, population_variance,
                    ::mlir::arith::DivFOp::create(
                        body_builder, body_location, count_value,
                        ::mlir::arith::SubFOp::create(
                            body_builder, body_location, count_value, one))),
                population_variance);
            auto next_variance_value = ::mlir::arith::AddFOp::create(
                body_builder, body_location,
                ::mlir::arith::MulFOp::create(
                    body_builder, body_location,
                    load_binding(body_builder, body_location,
                                 previous_variance, indices, values),
                    old_factor),
                ::mlir::arith::MulFOp::create(
                    body_builder, body_location, sample_variance,
                    momentum_value));
            for (auto item : {
                     std::pair<Binding const*, ::mlir::Value>{
                         &equivalent_scale, equivalent_scale_value},
                     {&equivalent_bias, equivalent_bias_value},
                     {&mean, mean_value}, {&inverse, inverse_value},
                     {&next_mean, next_mean_value},
                     {&next_variance, next_variance_value}}) {
                ::mlir::memref::StoreOp::create(
                    body_builder, body_location, item.second,
                    values.at(item.first->uid), indices);
            }
        });
}

struct DbnWeightBindings {
    Binding dy;
    Binding x;
    Binding scale;
    Binding mean;
    Binding inverse;
    Binding dscale;
    Binding dbias;
    Binding equivalent_bias;
    Binding equivalent_scale_dy;
    Binding equivalent_scale_x;
};

Status emit_dbn_weight(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    GenericOperationDesc const& operation,
    SerializedGraph const& graph,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    DbnWeightBindings bindings;
    for (auto item : {
             std::pair<std::string_view, Binding*>{"DY", &bindings.dy},
             {"X", &bindings.x}, {"SCALE", &bindings.scale},
             {"MEAN", &bindings.mean},
             {"INV_VARIANCE", &bindings.inverse}}) {
        auto status = bind_port(operation, graph, true, item.first,
                                "DBN_WEIGHT", *item.second);
        if (status.is_bad()) return status;
    }
    for (auto item : {
             std::pair<std::string_view, Binding*>{"DSCALE",
                                                   &bindings.dscale},
             {"DBIAS", &bindings.dbias},
             {"EQ_BIAS", &bindings.equivalent_bias},
             {"EQ_SCALE_DY", &bindings.equivalent_scale_dy},
             {"EQ_SCALE_X", &bindings.equivalent_scale_x}}) {
        auto status = bind_port(operation, graph, false, item.first,
                                "DBN_WEIGHT", *item.second);
        if (status.is_bad()) return status;
    }
    auto const count = reduction_count(*bindings.x.tensor,
                                       bindings.scale.tensor->dim);
    return emit_flat_loop(
        builder, location, bindings.scale.tensor->dim, "DBN_WEIGHT",
        [&](::mlir::OpBuilder& body_builder,
            ::mlir::Location body_location,
            llvm::SmallVector<::mlir::Value> const& parameter_indices) {
            auto axes = reduction_axes(*bindings.x.tensor,
                                       bindings.scale.tensor->dim);
            llvm::SmallVector<::mlir::Value> x_indices(
                parameter_indices.begin(), parameter_indices.end());
            llvm::SmallVector<::mlir::Value> lowers;
            llvm::SmallVector<::mlir::Value> uppers;
            llvm::SmallVector<::mlir::Value> steps;
            for (auto axis : axes) {
                lowers.push_back(
                    index_constant(body_builder, body_location, 0));
                uppers.push_back(index_constant(
                    body_builder, body_location,
                    bindings.x.tensor->dim[axis]));
                steps.push_back(
                    index_constant(body_builder, body_location, 1));
            }
            auto inverse_value = load_binding(
                body_builder, body_location, bindings.inverse,
                parameter_indices, values);
            auto mean_value = load_binding(body_builder, body_location,
                                           bindings.mean, parameter_indices,
                                           values);
            auto contribution = [&](::mlir::OpBuilder& current_builder,
                                    ::mlir::Location current_location) {
                auto dy = load_binding(current_builder, current_location,
                                       bindings.dy, x_indices, values);
                auto x = load_binding(current_builder, current_location,
                                      bindings.x, x_indices, values);
                auto xhat = ::mlir::arith::MulFOp::create(
                    current_builder, current_location,
                    ::mlir::arith::SubFOp::create(
                        current_builder, current_location, x, mean_value),
                    inverse_value);
                return GradientSums{
                    ::mlir::arith::MulFOp::create(
                        current_builder, current_location, dy, xhat),
                    dy};
            };
            GradientSums sums;
            if (axes.empty()) {
                sums = contribution(body_builder, body_location);
            } else {
                auto reduction = ::mlir::scf::buildLoopNest(
                    body_builder, body_location, lowers, uppers, steps,
                    ::mlir::ValueRange{
                        float_constant(body_builder, body_location, 0.0F),
                        float_constant(body_builder, body_location, 0.0F)},
                    [&](::mlir::OpBuilder& reduction_builder,
                        ::mlir::Location reduction_location,
                        ::mlir::ValueRange reduction_indices,
                        ::mlir::ValueRange iter_args)
                        -> ::mlir::scf::ValueVector {
                        for (std::size_t index = 0; index < axes.size();
                             ++index) {
                            x_indices[axes[index]] = reduction_indices[index];
                        }
                        auto item =
                            contribution(reduction_builder,
                                         reduction_location);
                        return {
                            ::mlir::arith::AddFOp::create(
                                reduction_builder, reduction_location,
                                iter_args[0], item.dscale),
                            ::mlir::arith::AddFOp::create(
                                reduction_builder, reduction_location,
                                iter_args[1], item.dbias)};
                    });
                body_builder.setInsertionPointAfter(reduction.loops.front());
                sums = {reduction.results[0], reduction.results[1]};
            }
            auto scale_value = load_binding(
                body_builder, body_location, bindings.scale,
                parameter_indices, values);
            auto equivalent_dy = ::mlir::arith::MulFOp::create(
                body_builder, body_location, scale_value, inverse_value);
            auto equivalent_x = ::mlir::arith::NegFOp::create(
                body_builder, body_location,
                ::mlir::arith::DivFOp::create(
                    body_builder, body_location,
                    ::mlir::arith::MulFOp::create(
                        body_builder, body_location,
                        ::mlir::arith::MulFOp::create(
                            body_builder, body_location, scale_value,
                            inverse_value),
                        ::mlir::arith::MulFOp::create(
                            body_builder, body_location, inverse_value,
                            sums.dscale)),
                    float_constant(body_builder, body_location,
                                   static_cast<float>(count))));
            auto equivalent_bias = ::mlir::arith::SubFOp::create(
                body_builder, body_location,
                ::mlir::arith::NegFOp::create(
                    body_builder, body_location,
                    ::mlir::arith::DivFOp::create(
                        body_builder, body_location,
                        ::mlir::arith::MulFOp::create(
                            body_builder, body_location, equivalent_dy,
                            sums.dbias),
                        float_constant(body_builder, body_location,
                                       static_cast<float>(count)))),
                ::mlir::arith::MulFOp::create(
                    body_builder, body_location, mean_value, equivalent_x));
            for (auto item : {
                     std::pair<Binding const*, ::mlir::Value>{
                         &bindings.dscale, sums.dscale},
                     {&bindings.dbias, sums.dbias},
                     {&bindings.equivalent_scale_dy, equivalent_dy},
                     {&bindings.equivalent_scale_x, equivalent_x},
                     {&bindings.equivalent_bias, equivalent_bias}}) {
                ::mlir::memref::StoreOp::create(
                    body_builder, body_location, item.second,
                    values.at(item.first->uid), parameter_indices);
            }
        });
}

}  // namespace

bool is_normalization_operation(OperationTag tag) noexcept {
    switch (tag) {
        case OperationTag::kAdaLayerNorm:
        case OperationTag::kAdaLayerNormBprop:
        case OperationTag::kBatchNorm:
        case OperationTag::kBatchNormInference:
        case OperationTag::kBnFinalize:
        case OperationTag::kDbn:
        case OperationTag::kDbnWeight:
        case OperationTag::kGenStats:
        case OperationTag::kInstanceNorm:
        case OperationTag::kInstanceNormBprop:
        case OperationTag::kLayerNorm:
        case OperationTag::kLayerNormBprop:
        case OperationTag::kRmsNorm:
        case OperationTag::kRmsNormBprop:
            return true;
        default:
            return false;
    }
}

Status validate_normalization_operation(OperationTag tag,
                                        GenericOperationDesc const& operation,
                                        SerializedGraph const& graph,
                                        std::size_t node_index) {
    auto const path = "nodes[" + std::to_string(node_index) + "]";
    switch (tag) {
        case OperationTag::kAdaLayerNorm:
        case OperationTag::kBatchNorm:
        case OperationTag::kInstanceNorm:
        case OperationTag::kLayerNorm:
        case OperationTag::kRmsNorm: {
            ForwardNorm norm;
            return decode_forward_norm(tag, operation, graph, path, norm);
        }
        case OperationTag::kBatchNormInference: {
            InferenceNorm norm;
            return decode_batchnorm_inference(operation, graph, path, norm);
        }
        case OperationTag::kAdaLayerNormBprop:
        case OperationTag::kDbn:
        case OperationTag::kInstanceNormBprop:
        case OperationTag::kLayerNormBprop:
        case OperationTag::kRmsNormBprop: {
            BackwardNorm norm;
            return decode_backward_norm(tag, operation, graph, path, norm);
        }
        case OperationTag::kGenStats:
            return validate_genstats(operation, graph, path);
        case OperationTag::kBnFinalize:
            return validate_bn_finalize(operation, graph, path);
        case OperationTag::kDbnWeight:
            return validate_dbn_weight(operation, graph, path);
        default:
            return fail(ErrorCode::kInvalidValue, path,
                        "operation is not a normalization tag");
    }
}

Status emit_normalization_operation(
    OperationTag tag,
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    GenericOperationDesc const& operation,
    SerializedGraph const& graph,
    std::map<std::int64_t, ::mlir::Value> const& values) {
    switch (tag) {
        case OperationTag::kAdaLayerNorm:
        case OperationTag::kBatchNorm:
        case OperationTag::kInstanceNorm:
        case OperationTag::kLayerNorm:
        case OperationTag::kRmsNorm: {
            ForwardNorm norm;
            auto status =
                decode_forward_norm(tag, operation, graph, "normalization",
                                    norm);
            if (status.is_bad()) return status;
            return emit_forward_norm(builder, location, norm, values);
        }
        case OperationTag::kBatchNormInference: {
            InferenceNorm norm;
            auto status = decode_batchnorm_inference(
                operation, graph, "BATCHNORM_INFERENCE", norm);
            if (status.is_bad()) return status;
            return emit_batchnorm_inference(builder, location, norm, values);
        }
        case OperationTag::kAdaLayerNormBprop:
        case OperationTag::kDbn:
        case OperationTag::kInstanceNormBprop:
        case OperationTag::kLayerNormBprop:
        case OperationTag::kRmsNormBprop: {
            BackwardNorm norm;
            auto status = decode_backward_norm(tag, operation, graph,
                                               "normalization backward",
                                               norm);
            if (status.is_bad()) return status;
            return emit_backward_norm(builder, location, norm, values);
        }
        case OperationTag::kGenStats:
            return emit_genstats(builder, location, operation, graph, values);
        case OperationTag::kBnFinalize:
            return emit_bn_finalize(builder, location, operation, graph,
                                    values);
        case OperationTag::kDbnWeight:
            return emit_dbn_weight(builder, location, operation, graph,
                                   values);
        default:
            return fail(ErrorCode::kInvalidValue, "normalization",
                        "validated normalization operation has no emitter");
    }
}

}  // namespace deepforge::compiler
