#pragma once

#include "DeepForge/Import/SerializedGraph.h"

#include <cstddef>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace deepforge::import {

struct OperationSchema {
    OperationTag tag;
    std::string_view serialized_tag;
    std::vector<std::string_view> required_attributes;
    std::vector<std::string_view> optional_attributes;
    std::vector<std::string_view> input_ports;
    std::vector<std::string_view> output_ports;
    bool allows_indexed_inputs = false;
};

[[nodiscard]] std::span<OperationSchema const> operation_schemas() noexcept;

[[nodiscard]] OperationSchema const*
find_operation_schema(OperationTag tag) noexcept;

[[nodiscard]] std::span<std::string_view const> pointwise_modes() noexcept;

[[nodiscard]] std::span<std::string_view const> reduction_modes() noexcept;

[[nodiscard]] bool is_pointwise_mode(std::string_view mode) noexcept;

[[nodiscard]] bool is_reduction_mode(std::string_view mode) noexcept;

[[nodiscard]] std::optional<std::size_t>
pointwise_input_count(std::string_view mode) noexcept;

}  // namespace deepforge::import
