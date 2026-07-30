#pragma once

#include "DeepForge/Import/SerializedGraph.h"
#include "DeepForge/Import/Status.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>

namespace deepforge::import {

inline constexpr std::size_t kMaximumSerializedGraphBytes =
    16U * 1024U * 1024U;

enum class InputFormat : std::uint8_t {
    kJson,
    kUbjson,
    kAuto,
};

class SerializedGraphImporter final {
public:
    [[nodiscard]] Status parse(std::span<std::uint8_t const> input,
                                InputFormat format,
                                SerializedGraph& output) const;

    [[nodiscard]] Status parse(std::span<std::uint8_t const> input,
                                SerializedGraph& output) const {
        return parse(input, InputFormat::kAuto, output);
    }

    [[nodiscard]] Status parse_file(std::filesystem::path const& path,
                                    InputFormat format,
                                    SerializedGraph& output) const;

    [[nodiscard]] Status parse_file(std::filesystem::path const& path,
                                    SerializedGraph& output) const {
        return parse_file(path, InputFormat::kAuto, output);
    }
};

}  // namespace deepforge::import
