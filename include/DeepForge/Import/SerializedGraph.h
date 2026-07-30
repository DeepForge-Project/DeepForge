#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <string>

namespace deepforge::import {

enum class DataType : std::uint8_t {
    kFloat32,
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

struct TensorDesc {
    std::string name;
    DataType data_type = DataType::kFloat32;
    std::array<std::int64_t, 4> dim{};
    std::array<std::int64_t, 4> stride{};
    bool is_virtual = false;
    bool is_pass_by_value = false;
    std::int64_t uid = 0;

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

struct SerializedGraph {
    std::string json_version;
    std::int64_t cudnn_frontend_version = 0;
    std::uint64_t graph_uid = 0;
    GraphContext context;
    std::map<std::int64_t, TensorDesc> tensors;
    ConvFpropDesc conv;

    bool operator==(SerializedGraph const&) const = default;
};

}  // namespace deepforge::import
