#include "DeepForge/Import/Capability.h"

#include <array>
#include <limits>

namespace deepforge::import {
namespace {

constexpr std::array<OperationCapability, 39> kCapabilities{{
    {OperationTag::kAdaLayerNorm, "ADA_LAYER_NORM", CapabilityLevel::kValidated},
    {OperationTag::kAdaLayerNormBprop, "ADA_LAYER_NORM_BPROP", CapabilityLevel::kValidated},
    {OperationTag::kBatchNorm, "BATCHNORM", CapabilityLevel::kValidated},
    {OperationTag::kBatchNormInference, "BATCHNORM_INFERENCE", CapabilityLevel::kValidated},
    {OperationTag::kBlockScaleDequantize, "BLOCK_SCALE_DEQUANTIZE", CapabilityLevel::kValidated},
    {OperationTag::kBlockScaleQuantize, "BLOCK_SCALE_QUANTIZE", CapabilityLevel::kValidated},
    {OperationTag::kBnFinalize, "BN_FINALIZE", CapabilityLevel::kValidated},
    {OperationTag::kConcatenate, "CONCATENATE", CapabilityLevel::kValidated},
    {OperationTag::kConvDgrad, "CONV_DGRAD", CapabilityLevel::kValidated},
    {OperationTag::kConvFprop, "CONV_FPROP", CapabilityLevel::kValidated},
    {OperationTag::kConvWgrad, "CONV_WGRAD", CapabilityLevel::kValidated},
    {OperationTag::kDbn, "DBN", CapabilityLevel::kValidated},
    {OperationTag::kDbnWeight, "DBN_WEIGHT", CapabilityLevel::kValidated},
    {OperationTag::kGenStats, "GENSTATS", CapabilityLevel::kValidated},
    {OperationTag::kInstanceNorm, "INSTANCE_NORM", CapabilityLevel::kValidated},
    {OperationTag::kInstanceNormBprop, "INSTANCE_NORM_BPROP", CapabilityLevel::kValidated},
    {OperationTag::kLayerNorm, "LAYER_NORM", CapabilityLevel::kValidated},
    {OperationTag::kLayerNormBprop, "LAYER_NORM_BPROP", CapabilityLevel::kValidated},
    {OperationTag::kMatmul, "MATMUL", CapabilityLevel::kValidated},
    {OperationTag::kMatmulFp8, "MATMUL_FP8", CapabilityLevel::kValidated},
    {OperationTag::kMoeGroupedMatmul, "MOE_GROUPED_MATMUL", CapabilityLevel::kValidated},
    {OperationTag::kMoeGroupedMatmulBwd, "MOE_GROUPED_MATMUL_BWD", CapabilityLevel::kValidated},
    {OperationTag::kPointwise, "POINTWISE", CapabilityLevel::kValidated},
    {OperationTag::kReduction, "REDUCTION", CapabilityLevel::kValidated},
    {OperationTag::kResample, "RESAMPLE", CapabilityLevel::kValidated},
    {OperationTag::kReshape, "RESHAPE", CapabilityLevel::kValidated},
    {OperationTag::kRmsNorm, "RMS_NORM", CapabilityLevel::kValidated},
    {OperationTag::kRmsNormBprop, "RMS_NORM_BPROP", CapabilityLevel::kValidated},
    {OperationTag::kRng, "RNG", CapabilityLevel::kValidated},
    {OperationTag::kRope, "ROPE", CapabilityLevel::kValidated},
    {OperationTag::kRopeBwd, "ROPE_BWD", CapabilityLevel::kValidated},
    {OperationTag::kSdpa, "SDPA", CapabilityLevel::kValidated},
    {OperationTag::kSdpaBwd, "SDPA_BWD", CapabilityLevel::kValidated},
    {OperationTag::kSdpaFp8Bwd, "SDPA_FP8_BWD", CapabilityLevel::kValidated},
    {OperationTag::kSdpaFp8Fwd, "SDPA_FP8_FWD", CapabilityLevel::kValidated},
    {OperationTag::kSdpaMxfp8Bwd, "SDPA_MXFP8_BWD", CapabilityLevel::kValidated},
    {OperationTag::kSdpaMxfp8Fwd, "SDPA_MXFP8_FWD", CapabilityLevel::kValidated},
    {OperationTag::kSlice, "SLICE", CapabilityLevel::kValidated},
    {OperationTag::kTranspose, "TRANSPOSE", CapabilityLevel::kValidated},
}};

struct DataTypeName {
    DataType type;
    std::string_view name;
    std::uint16_t storage_bits;
};

constexpr std::array<DataTypeName, 20> kDataTypes{{
    {DataType::kFloat32, "FLOAT", 32},
    {DataType::kFloat64, "DOUBLE", 64},
    {DataType::kFloat16, "HALF", 16},
    {DataType::kInt8, "INT8", 8},
    {DataType::kInt32, "INT32", 32},
    {DataType::kInt8x4, "INT8x4", 32},
    {DataType::kUInt8, "UINT8", 8},
    {DataType::kUInt8x4, "UINT8x4", 32},
    {DataType::kInt8x32, "INT8x32", 256},
    {DataType::kBFloat16, "BFLOAT16", 16},
    {DataType::kInt64, "INT64", 64},
    {DataType::kBoolean, "BOOLEAN", 8},
    {DataType::kFp8E4M3, "FP8_E4M3", 8},
    {DataType::kFp8E5M2, "FP8_E5M2", 8},
    {DataType::kFastFloatForFp8, "FAST_FLOAT_FOR_FP8", 32},
    {DataType::kFp8E8M0, "FP8_E8M0", 8},
    {DataType::kFp4E2M1, "FP4_E2M1", 4},
    {DataType::kInt4, "INT4", 4},
    {DataType::kComplexFloat32, "COMPLEX_FP32", 64},
    {DataType::kComplexFloat64, "COMPLEX_FP64", 128},
}};

}  // namespace

std::span<OperationCapability const> operation_capabilities() noexcept {
    return kCapabilities;
}

OperationCapability const*
find_operation_capability(std::string_view serialized_tag) noexcept {
    for (auto const& capability : kCapabilities) {
        if (capability.serialized_tag == serialized_tag) {
            return &capability;
        }
    }
    return nullptr;
}

std::string_view operation_tag_name(OperationTag tag) noexcept {
    for (auto const& capability : kCapabilities) {
        if (capability.tag == tag) {
            return capability.serialized_tag;
        }
    }
    return "UNKNOWN";
}

std::optional<DataType> data_type_from_name(std::string_view name) noexcept {
    for (auto const& data_type : kDataTypes) {
        if (data_type.name == name) {
            return data_type.type;
        }
    }
    return std::nullopt;
}

std::string_view data_type_name(DataType type) noexcept {
    for (auto const& data_type : kDataTypes) {
        if (data_type.type == type) {
            return data_type.name;
        }
    }
    return "UNKNOWN";
}

std::uint16_t data_type_storage_bits(DataType type) noexcept {
    for (auto const& data_type : kDataTypes) {
        if (data_type.type == type) {
            return data_type.storage_bits;
        }
    }
    return 0;
}

bool tensor_storage_bytes(DataType type,
                          std::span<std::int64_t const> dimensions,
                          std::span<std::int64_t const> strides,
                          std::uint64_t& output) noexcept {
    if (dimensions.empty() || dimensions.size() != strides.size()) {
        return false;
    }
    std::uint64_t maximum_element_offset = 0;
    for (std::size_t index = 0; index < dimensions.size(); ++index) {
        if (dimensions[index] <= 0 || strides[index] <= 0) {
            return false;
        }
        auto extent = static_cast<std::uint64_t>(dimensions[index] - 1);
        auto stride = static_cast<std::uint64_t>(strides[index]);
        if (extent != 0 &&
            stride > std::numeric_limits<std::uint64_t>::max() / extent) {
            return false;
        }
        auto contribution = extent * stride;
        if (contribution > std::numeric_limits<std::uint64_t>::max() -
                               maximum_element_offset) {
            return false;
        }
        maximum_element_offset += contribution;
    }
    if (maximum_element_offset == std::numeric_limits<std::uint64_t>::max()) {
        return false;
    }
    auto element_slots = maximum_element_offset + 1;
    auto bits = static_cast<std::uint64_t>(data_type_storage_bits(type));
    if (bits == 0 ||
        element_slots > std::numeric_limits<std::uint64_t>::max() / bits) {
        return false;
    }
    auto storage_bits = element_slots * bits;
    if (storage_bits > std::numeric_limits<std::uint64_t>::max() - 7) {
        return false;
    }
    output = (storage_bits + 7) / 8;
    return output != 0;
}

}  // namespace deepforge::import
