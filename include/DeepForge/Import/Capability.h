#pragma once

#include "DeepForge/Import/SerializedGraph.h"

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

namespace deepforge::import {

enum class CapabilityLevel : std::uint8_t {
    kSchemaKnown,
    kParsed,
    kExecutable,
    kValidated,
};

struct OperationCapability {
    OperationTag tag;
    std::string_view serialized_tag;
    CapabilityLevel level;

    bool operator==(OperationCapability const&) const = default;
};

[[nodiscard]] std::span<OperationCapability const>
operation_capabilities() noexcept;

[[nodiscard]] OperationCapability const*
find_operation_capability(std::string_view serialized_tag) noexcept;

[[nodiscard]] std::string_view operation_tag_name(OperationTag tag) noexcept;

[[nodiscard]] std::optional<DataType>
data_type_from_name(std::string_view name) noexcept;

[[nodiscard]] std::string_view data_type_name(DataType type) noexcept;

[[nodiscard]] std::uint16_t data_type_storage_bits(DataType type) noexcept;

[[nodiscard]] bool tensor_storage_bytes(
    DataType type,
    std::span<std::int64_t const> dimensions,
    std::span<std::int64_t const> strides,
    std::uint64_t& output) noexcept;

}  // namespace deepforge::import
