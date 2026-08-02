#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace deepforge::import {

enum class DataType : std::uint8_t {
    kFloat32,
    kFloat64,
    kFloat16,
    kInt8,
    kInt32,
    kInt8x4,
    kUInt8,
    kUInt8x4,
    kInt8x32,
    kBFloat16,
    kInt64,
    kBoolean,
    kFp8E4M3,
    kFp8E5M2,
    kFastFloatForFp8,
    kFp8E8M0,
    kFp4E2M1,
    kInt4,
    kComplexFloat32,
    kComplexFloat64,
};

enum class OperationTag : std::uint8_t {
    kAdaLayerNorm,
    kAdaLayerNormBprop,
    kBatchNorm,
    kBatchNormInference,
    kBlockScaleDequantize,
    kBlockScaleQuantize,
    kBnFinalize,
    kConcatenate,
    kConvDgrad,
    kConvFprop,
    kConvWgrad,
    kDbn,
    kDbnWeight,
    kGenStats,
    kInstanceNorm,
    kInstanceNormBprop,
    kLayerNorm,
    kLayerNormBprop,
    kMatmul,
    kMatmulFp8,
    kMoeGroupedMatmul,
    kMoeGroupedMatmulBwd,
    kPointwise,
    kReduction,
    kResample,
    kReshape,
    kRmsNorm,
    kRmsNormBprop,
    kRng,
    kRope,
    kRopeBwd,
    kSdpa,
    kSdpaBwd,
    kSdpaFp8Bwd,
    kSdpaFp8Fwd,
    kSdpaMxfp8Bwd,
    kSdpaMxfp8Fwd,
    kSlice,
    kTranspose,
};

struct GraphContext {
    std::optional<std::string> name;
    std::optional<DataType> compute_data_type;
    std::optional<DataType> intermediate_data_type;
    std::optional<DataType> io_data_type;
    std::optional<std::int64_t> sm_count;
    std::optional<bool> is_dynamic_shape_enabled;
    std::optional<bool> is_override_shape_enabled;

    bool operator==(GraphContext const&) const = default;
};

struct SerializedValue {
    using Array = std::vector<SerializedValue>;
    using Object = std::map<std::string, SerializedValue>;
    using Storage =
        std::variant<std::nullptr_t, bool, std::int64_t, std::uint64_t, double,
                     std::string, Array, Object>;

    Storage value = nullptr;

    bool operator==(SerializedValue const&) const = default;
};

enum class PassByValueKind : std::uint8_t {
    kInt64 = 0,
    kInt32 = 1,
    kFloat16 = 2,
    kFloat32 = 3,
    kFloat64 = 4,
    kBFloat16 = 5,
};

struct PassByValueScalar {
    // FLOAT uses uint32 bits; HALF/DOUBLE/BFLOAT16 retain the JSON number's
    // double bits in uint64 until MLIR converts to the destination semantics.
    using Storage =
        std::variant<std::int64_t, std::int32_t, std::uint32_t, std::uint64_t>;

    PassByValueKind kind = PassByValueKind::kFloat32;
    Storage value = std::uint32_t{0};

    bool operator==(PassByValueScalar const&) const = default;
};

using TensorReference = std::variant<std::int64_t, std::string>;

struct TensorDesc {
    std::string name;
    DataType data_type = DataType::kFloat32;
    std::vector<std::int64_t> dim;
    std::vector<std::int64_t> stride;
    bool is_virtual = false;
    bool is_pass_by_value = false;
    std::optional<PassByValueScalar> pass_by_value;
    std::string reordering_type = "NONE";
    std::optional<std::int64_t> ragged_offset_uid;
    std::optional<std::string> ragged_offset_name;
    std::int64_t uid = 0;
    bool uid_assigned = false;

    bool operator==(TensorDesc const&) const = default;
};

struct ConvFpropDesc {
    std::string name;
    DataType compute_data_type = DataType::kFloat32;
    std::int64_t x_uid = 0;
    std::int64_t w_uid = 0;
    std::int64_t y_uid = 0;
    std::array<std::int64_t, 2> pre_padding{};
    std::array<std::int64_t, 2> post_padding{};
    std::array<std::int64_t, 2> stride{};
    std::array<std::int64_t, 2> dilation{};

    bool operator==(ConvFpropDesc const&) const = default;
};

struct GenericOperationDesc {
    std::map<std::string, TensorReference> inputs;
    std::map<std::string, std::vector<TensorReference>> input_lists;
    std::map<std::string, TensorReference> outputs;
    SerializedValue::Object attributes;

    bool operator==(GenericOperationDesc const&) const = default;
};

using OperationAttributes =
    std::variant<std::monostate, ConvFpropDesc, GenericOperationDesc>;

struct NodeDesc {
    OperationTag tag = OperationTag::kConvFprop;
    std::string name;
    OperationAttributes attributes;

    bool operator==(NodeDesc const&) const = default;
};

struct SerializedGraph {
    std::string json_version;
    std::string cudnn_backend_version;
    std::int64_t cudnn_frontend_version = 0;
    std::uint64_t graph_uid = 0;
    GraphContext context;
    std::map<std::int64_t, TensorDesc> tensors;
    std::map<std::string, TensorDesc> named_tensors;
    std::vector<NodeDesc> nodes;

    [[nodiscard]] std::size_t tensor_count() const noexcept {
        return tensors.size() + named_tensors.size();
    }

    [[nodiscard]] TensorDesc const* find_tensor(
        TensorReference const& reference) const noexcept {
        if (auto const* uid = std::get_if<std::int64_t>(&reference)) {
            auto const it = tensors.find(*uid);
            return it == tensors.end() ? nullptr : &it->second;
        }
        auto const& name = std::get<std::string>(reference);
        auto const it = named_tensors.find(name);
        return it == named_tensors.end() ? nullptr : &it->second;
    }

    [[nodiscard]] ConvFpropDesc const* single_conv_fprop() const noexcept {
        if (nodes.size() != 1 || nodes.front().tag != OperationTag::kConvFprop) {
            return nullptr;
        }
        return std::get_if<ConvFpropDesc>(&nodes.front().attributes);
    }

    [[nodiscard]] ConvFpropDesc* single_conv_fprop() noexcept {
        if (nodes.size() != 1 || nodes.front().tag != OperationTag::kConvFprop) {
            return nullptr;
        }
        return std::get_if<ConvFpropDesc>(&nodes.front().attributes);
    }

    ConvFpropDesc& emplace_conv_fprop() {
        nodes.push_back(NodeDesc{OperationTag::kConvFprop, {},
                                 ConvFpropDesc{}});
        return std::get<ConvFpropDesc>(nodes.back().attributes);
    }

    bool operator==(SerializedGraph const&) const = default;
};

}  // namespace deepforge::import
