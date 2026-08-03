#include "DeepForge/Compiler/Artifact.h"

#include <cudnn_frontend/thirdparty/nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

using Json = nlohmann::json;

class TestRunner {
public:
    void check(bool condition, std::string const& name) {
        ++checks_;
        if (!condition) {
            ++failures_;
            std::cerr << "FAIL: " << name << '\n';
        }
    }

    void good(deepforge::import::Status const& status,
              std::string const& name) {
        ++checks_;
        if (status.is_bad()) {
            ++failures_;
            std::cerr << "FAIL: " << name << ": " << status.message() << '\n';
        }
    }

    void parse_error(deepforge::import::Status const& status,
                     std::string const& name) {
        check(status.code() == deepforge::import::ErrorCode::kParseError, name);
    }

    int finish() const {
        if (failures_ == 0) {
            std::cout << "deepforge-artifact: " << checks_
                      << " checks passed\n";
            return 0;
        }
        std::cerr << "deepforge-artifact: " << failures_ << " of " << checks_
                  << " checks failed\n";
        return 1;
    }

private:
    int checks_ = 0;
    int failures_ = 0;
};

std::string read_text(std::filesystem::path const& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("cannot open fixture: " + path.string());
    }
    return std::string((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
}

bool write_bytes(std::filesystem::path const& path,
                 std::span<std::uint8_t const> bytes) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(reinterpret_cast<char const*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
    return static_cast<bool>(file);
}

std::uint32_t read_u32(std::span<std::uint8_t const> bytes,
                       std::size_t& offset) {
    if (offset + 4 > bytes.size()) {
        throw std::runtime_error("artifact test offset is out of range");
    }
    std::uint32_t value = 0;
    for (unsigned index = 0; index < 4; ++index) {
        value |= static_cast<std::uint32_t>(bytes[offset++]) << (index * 8);
    }
    return value;
}

std::uint64_t read_u64(std::span<std::uint8_t const> bytes,
                       std::size_t& offset) {
    if (offset + 8 > bytes.size()) {
        throw std::runtime_error("artifact test offset is out of range");
    }
    std::uint64_t value = 0;
    for (unsigned index = 0; index < 8; ++index) {
        value |= static_cast<std::uint64_t>(bytes[offset++]) << (index * 8);
    }
    return value;
}

std::size_t skip_string(std::span<std::uint8_t const> bytes,
                        std::size_t offset,
                        std::string_view context) {
    auto size = read_u32(bytes, offset);
    if (size > bytes.size() - offset) {
        throw std::runtime_error(
            "artifact test skipped string is truncated at " +
            std::to_string(offset - 4) + " with size " +
            std::to_string(size) + " while reading " +
            std::string(context));
    }
    return offset + size;
}

void write_u64(std::vector<std::uint8_t>& bytes, std::size_t offset,
               std::uint64_t value) {
    if (offset + 8 > bytes.size()) {
        throw std::runtime_error("artifact test offset is out of range");
    }
    for (unsigned index = 0; index < 8; ++index) {
        bytes[offset + index] =
            static_cast<std::uint8_t>(value >> (index * 8));
    }
}

void write_u32(std::vector<std::uint8_t>& bytes, std::size_t offset,
               std::uint32_t value) {
    if (offset + 4 > bytes.size()) {
        throw std::runtime_error("artifact test offset is out of range");
    }
    for (unsigned index = 0; index < 4; ++index) {
        bytes[offset + index] =
            static_cast<std::uint8_t>(value >> (index * 8));
    }
}

std::uint64_t fnv1a(std::span<std::uint8_t const> bytes) {
    std::uint64_t hash = 14695981039346656037ULL;
    for (auto byte : bytes) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    return hash;
}

void refresh_checksum(std::vector<std::uint8_t>& bytes) {
    if (bytes.size() < 8) {
        throw std::runtime_error("artifact test input is truncated");
    }
    write_u64(bytes, bytes.size() - 8,
              fnv1a(std::span(bytes).first(bytes.size() - 8)));
}

std::size_t metadata_numbers_offset(
    std::span<std::uint8_t const> bytes) {
    std::size_t offset = 8 + 4 + 4;
    for (int index = 0; index < 5; ++index) {
        auto size = read_u32(bytes, offset);
        if (size > bytes.size() - offset) {
            throw std::runtime_error("artifact test header string is truncated");
        }
        offset += size;
    }
    return offset;
}

std::size_t argument_table_offset(std::span<std::uint8_t const> bytes) {
    std::size_t version_offset = 8;
    auto const version = read_u32(bytes, version_offset);
    auto offset = metadata_numbers_offset(bytes);
    if (version == deepforge::compiler::kShapeOverrideArtifactFormatVersion ||
        version == deepforge::compiler::kRaggedArtifactFormatVersion ||
        version ==
            deepforge::compiler::kRaggedSequenceArtifactFormatVersion ||
        version ==
            deepforge::compiler::kMatmulOverrideArtifactFormatVersion ||
        version ==
            deepforge::compiler::kSdpaOverrideArtifactFormatVersion ||
        version ==
            deepforge::compiler::kReshapeOverrideArtifactFormatVersion ||
        version ==
            deepforge::compiler::kReductionOverrideArtifactFormatVersion ||
        version ==
            deepforge::compiler::kTransposeOverrideArtifactFormatVersion ||
        version == deepforge::compiler::
                       kConcatenateOverrideArtifactFormatVersion ||
        version == deepforge::compiler::kArtifactFormatVersion) {
        offset += 3 * 4;
    }
    if (version ==
            deepforge::compiler::kMatmulOverrideArtifactFormatVersion ||
        version ==
            deepforge::compiler::kSdpaOverrideArtifactFormatVersion ||
        version ==
            deepforge::compiler::kReshapeOverrideArtifactFormatVersion ||
        version ==
            deepforge::compiler::kReductionOverrideArtifactFormatVersion ||
        version ==
            deepforge::compiler::kTransposeOverrideArtifactFormatVersion ||
        version == deepforge::compiler::
                       kConcatenateOverrideArtifactFormatVersion ||
        version == deepforge::compiler::kArtifactFormatVersion) {
        auto const role_count = read_u32(bytes, offset);
        offset += static_cast<std::size_t>(role_count) * 8;
    }
    if (version ==
            deepforge::compiler::kTransposeOverrideArtifactFormatVersion ||
        version == deepforge::compiler::
                       kConcatenateOverrideArtifactFormatVersion ||
        version == deepforge::compiler::kArtifactFormatVersion) {
        auto const axis_count = read_u32(bytes, offset);
        offset += static_cast<std::size_t>(axis_count) * 8;
    }
    return offset;
}

std::size_t override_axis_count_offset(
    std::span<std::uint8_t const> bytes) {
    auto offset = metadata_numbers_offset(bytes) + 3 * 4;
    auto const role_count = read_u32(bytes, offset);
    if (role_count > 64 ||
        static_cast<std::uint64_t>(role_count) * 8 >
            bytes.size() - offset) {
        throw std::runtime_error(
            "artifact test override role metadata is invalid");
    }
    return offset + static_cast<std::size_t>(role_count) * 8;
}

std::size_t adapter_kind_offset(std::span<std::uint8_t const> bytes) {
    auto offset = argument_table_offset(bytes);
    auto count = read_u32(bytes, offset);
    std::size_t version_offset = 8;
    auto const version = read_u32(bytes, version_offset);
    for (std::uint32_t index = 0; index < count; ++index) {
        offset += 8;
        offset = skip_string(bytes, offset, "adapter argument");
        offset += 4 + 4;
        if (version == deepforge::compiler::kRaggedArtifactFormatVersion ||
            version ==
                deepforge::compiler::kRaggedSequenceArtifactFormatVersion ||
            version ==
                deepforge::compiler::kMatmulOverrideArtifactFormatVersion ||
            version ==
                deepforge::compiler::kSdpaOverrideArtifactFormatVersion ||
            version ==
                deepforge::compiler::kReshapeOverrideArtifactFormatVersion ||
            version == deepforge::compiler::
                           kReductionOverrideArtifactFormatVersion ||
            version == deepforge::compiler::
                           kTransposeOverrideArtifactFormatVersion ||
            version == deepforge::compiler::
                           kConcatenateOverrideArtifactFormatVersion ||
            version == deepforge::compiler::kArtifactFormatVersion) {
            offset += 4 + 8 + 8;
        }
        if (version ==
                deepforge::compiler::kRaggedSequenceArtifactFormatVersion ||
            version ==
                deepforge::compiler::kMatmulOverrideArtifactFormatVersion ||
            version ==
                deepforge::compiler::kSdpaOverrideArtifactFormatVersion ||
            version ==
                deepforge::compiler::kReshapeOverrideArtifactFormatVersion ||
            version == deepforge::compiler::
                           kReductionOverrideArtifactFormatVersion ||
            version == deepforge::compiler::
                           kTransposeOverrideArtifactFormatVersion ||
            version == deepforge::compiler::
                           kConcatenateOverrideArtifactFormatVersion ||
            version == deepforge::compiler::kArtifactFormatVersion) {
            offset += 8;
        }
        offset += 8 + 8;
        auto rank = read_u32(bytes, offset);
        if (rank == 0 || rank > 64 ||
            static_cast<std::uint64_t>(rank) * 16 > bytes.size() - offset) {
            throw std::runtime_error("artifact test argument rank is invalid");
        }
        offset += static_cast<std::size_t>(rank) * 16;
    }
    return offset;
}

std::size_t adapter_payload_offset(std::span<std::uint8_t const> bytes) {
    auto offset = adapter_kind_offset(bytes) + 4;
    auto size = read_u64(bytes, offset);
    if (size > bytes.size() - offset) {
        throw std::runtime_error("artifact test adapter payload is truncated");
    }
    return offset;
}

std::size_t workspace_size_offset(std::span<std::uint8_t const> bytes) {
    auto payload = adapter_payload_offset(bytes);
    auto size_offset = adapter_kind_offset(bytes) + 4;
    auto size = read_u64(bytes, size_offset);
    return payload + static_cast<std::size_t>(size);
}

std::size_t workspace_alignment_offset(
    std::span<std::uint8_t const> bytes) {
    return workspace_size_offset(bytes) + 8;
}

std::size_t first_argument_storage_policy_offset(
    std::span<std::uint8_t const> bytes) {
    auto offset = argument_table_offset(bytes);
    auto count = read_u32(bytes, offset);
    if (count == 0) {
        throw std::runtime_error("artifact test argument table is empty");
    }
    offset += 8;
    offset = skip_string(bytes, offset, "first argument");
    return offset + 4 + 4;
}

std::size_t first_argument_alignment_offset(
    std::span<std::uint8_t const> bytes) {
    return first_argument_storage_policy_offset(bytes) + 4 + 8 + 8 + 8;
}

std::size_t first_argument_ragged_sequence_divisor_offset(
    std::span<std::uint8_t const> bytes) {
    return first_argument_storage_policy_offset(bytes) + 4 + 8 + 8;
}

std::vector<std::uint8_t> make_ragged_sequence_v5(
    std::span<std::uint8_t const> current) {
    auto const roles_begin = metadata_numbers_offset(current) + 3 * 4;
    auto const table = argument_table_offset(current);
    std::vector<std::uint8_t> version_five(
        current.begin(),
        current.begin() + static_cast<std::ptrdiff_t>(roles_begin));
    version_five.insert(
        version_five.end(),
        current.begin() + static_cast<std::ptrdiff_t>(table),
        current.end() - 8);
    version_five.resize(version_five.size() + 8);
    write_u32(version_five, 8,
              deepforge::compiler::kRaggedSequenceArtifactFormatVersion);
    refresh_checksum(version_five);
    return version_five;
}

std::vector<std::uint8_t> make_role_override_legacy(
    std::span<std::uint8_t const> current,
    std::uint32_t target_version) {
    auto axis_begin = metadata_numbers_offset(current) + 3 * 4;
    auto const role_count = read_u32(current, axis_begin);
    axis_begin += static_cast<std::size_t>(role_count) * 8;
    auto const table = argument_table_offset(current);
    std::vector<std::uint8_t> legacy(
        current.begin(),
        current.begin() + static_cast<std::ptrdiff_t>(axis_begin));
    legacy.insert(legacy.end(),
                  current.begin() + static_cast<std::ptrdiff_t>(table),
                  current.end() - 8);
    legacy.resize(legacy.size() + 8);
    write_u32(legacy, 8, target_version);
    refresh_checksum(legacy);
    return legacy;
}

std::vector<std::uint8_t> make_matmul_override_v6(
    std::span<std::uint8_t const> current) {
    return make_role_override_legacy(
        current, deepforge::compiler::kMatmulOverrideArtifactFormatVersion);
}

std::vector<std::uint8_t> make_sdpa_override_v7(
    std::span<std::uint8_t const> current) {
    return make_role_override_legacy(
        current, deepforge::compiler::kSdpaOverrideArtifactFormatVersion);
}

std::vector<std::uint8_t> make_reshape_override_v8(
    std::span<std::uint8_t const> current) {
    return make_role_override_legacy(
        current, deepforge::compiler::kReshapeOverrideArtifactFormatVersion);
}

std::vector<std::uint8_t> make_reduction_override_v9(
    std::span<std::uint8_t const> current) {
    return make_role_override_legacy(
        current,
        deepforge::compiler::kReductionOverrideArtifactFormatVersion);
}

std::vector<std::uint8_t> make_transpose_override_v10(
    std::span<std::uint8_t const> current) {
    std::vector<std::uint8_t> version_ten(current.begin(), current.end());
    write_u32(version_ten, 8,
              deepforge::compiler::kTransposeOverrideArtifactFormatVersion);
    refresh_checksum(version_ten);
    return version_ten;
}

std::vector<std::uint8_t> make_ragged_v4(
    std::span<std::uint8_t const> current) {
    auto version_five = make_ragged_sequence_v5(current);
    auto const table = argument_table_offset(version_five);
    auto count_offset = table;
    auto const count = read_u32(version_five, count_offset);
    std::vector<std::uint8_t> version_four(
        version_five.begin(),
        version_five.begin() + static_cast<std::ptrdiff_t>(table + 4));
    auto offset = count_offset;
    for (std::uint32_t index = 0; index < count; ++index) {
        auto const entry_begin = offset;
        offset += 8;
        offset = skip_string(version_five, offset,
                             "v4 conversion argument " +
                                 std::to_string(index));
        offset += 4 + 4 + 4 + 8 + 8;
        version_four.insert(
            version_four.end(),
            version_five.begin() + static_cast<std::ptrdiff_t>(entry_begin),
            version_five.begin() + static_cast<std::ptrdiff_t>(offset));
        offset += 8;
        auto const tail_begin = offset;
        offset += 8 + 8;
        auto const rank = read_u32(version_five, offset);
        offset += static_cast<std::size_t>(rank) * 16;
        version_four.insert(
            version_four.end(),
            version_five.begin() + static_cast<std::ptrdiff_t>(tail_begin),
            version_five.begin() + static_cast<std::ptrdiff_t>(offset));
    }
    version_four.insert(
        version_four.end(),
        version_five.begin() + static_cast<std::ptrdiff_t>(offset),
        version_five.end() - 8);
    version_four.resize(version_four.size() + 8);
    write_u32(version_four, 8,
              deepforge::compiler::kRaggedArtifactFormatVersion);
    refresh_checksum(version_four);
    return version_four;
}

std::vector<std::uint8_t> make_shape_override_v3(
    std::span<std::uint8_t const> current) {
    auto version_four = make_ragged_v4(current);
    auto const table = argument_table_offset(version_four);
    auto count_offset = table;
    auto const count = read_u32(version_four, count_offset);
    std::vector<std::uint8_t> version_three(
        version_four.begin(),
        version_four.begin() + static_cast<std::ptrdiff_t>(table + 4));
    auto offset = count_offset;
    for (std::uint32_t index = 0; index < count; ++index) {
        auto const entry_begin = offset;
        offset += 8;
        offset = skip_string(version_four, offset,
                             "v3 conversion argument " +
                                 std::to_string(index));
        offset += 4 + 4;
        version_three.insert(
            version_three.end(),
            version_four.begin() + static_cast<std::ptrdiff_t>(entry_begin),
            version_four.begin() + static_cast<std::ptrdiff_t>(offset));
        offset += 4 + 8 + 8;
        auto const old_tail_begin = offset;
        offset += 8 + 8;
        auto const rank = read_u32(version_four, offset);
        offset += static_cast<std::size_t>(rank) * 16;
        version_three.insert(
            version_three.end(),
            version_four.begin() +
                static_cast<std::ptrdiff_t>(old_tail_begin),
            version_four.begin() + static_cast<std::ptrdiff_t>(offset));
    }
    version_three.insert(
        version_three.end(),
        version_four.begin() + static_cast<std::ptrdiff_t>(offset),
        version_four.end() - 8);
    version_three.resize(version_three.size() + 8);
    write_u32(version_three, 8,
              deepforge::compiler::kShapeOverrideArtifactFormatVersion);
    refresh_checksum(version_three);
    return version_three;
}

std::vector<std::uint8_t> make_legacy_v1(
    std::span<std::uint8_t const> current) {
    auto common_metadata_end = metadata_numbers_offset(current);
    auto adapter_begin = adapter_payload_offset(current);
    auto workspace_begin = workspace_size_offset(current);
    std::vector<std::uint8_t> legacy(
        current.begin(),
        current.begin() +
            static_cast<std::ptrdiff_t>(common_metadata_end));
    legacy.insert(legacy.end(),
                  current.begin() +
                      static_cast<std::ptrdiff_t>(adapter_begin),
                  current.begin() +
                      static_cast<std::ptrdiff_t>(workspace_begin));
    legacy.insert(legacy.end(),
                  current.begin() +
                      static_cast<std::ptrdiff_t>(workspace_begin),
                  current.end() - 8);
    legacy.resize(legacy.size() + 8);
    write_u32(legacy, 8,
              deepforge::compiler::kLegacyArtifactFormatVersion);
    refresh_checksum(legacy);
    return legacy;
}

std::vector<std::uint8_t> make_static_metadata_v2(
    std::span<std::uint8_t const> current) {
    auto version_three = make_shape_override_v3(current);
    auto const flags_begin = metadata_numbers_offset(current);
    auto const arguments_begin = argument_table_offset(version_three);
    std::vector<std::uint8_t> version_two(
        version_three.begin(),
        version_three.begin() + static_cast<std::ptrdiff_t>(flags_begin));
    version_two.insert(
        version_two.end(),
        version_three.begin() + static_cast<std::ptrdiff_t>(arguments_begin),
        version_three.end() - 8);
    version_two.resize(version_two.size() + 8);
    write_u32(version_two, 8,
              deepforge::compiler::kStaticMetadataArtifactFormatVersion);
    refresh_checksum(version_two);
    return version_two;
}

struct AlignedBytes {
    explicit AlignedBytes(std::size_t size) : size(size) {
        if (size != 0) {
            pointer = std::aligned_alloc(64, size);
        }
    }
    ~AlignedBytes() { std::free(pointer); }
    void* pointer = nullptr;
    std::size_t size = 0;
};

std::size_t element_count(std::array<std::int64_t, 4> const& shape) {
    std::size_t count = 1;
    for (auto dimension : shape) {
        count *= static_cast<std::size_t>(dimension);
    }
    return count;
}

bool close(std::vector<float> const& lhs, std::vector<float> const& rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (std::size_t index = 0; index < lhs.size(); ++index) {
        auto tolerance = 1.0e-4F + 1.0e-3F * std::fabs(lhs[index]);
        if (!std::isfinite(lhs[index]) || !std::isfinite(rhs[index]) ||
            std::fabs(lhs[index] - rhs[index]) > tolerance) {
            return false;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: deepforge_artifact_test <fixture.json> <scratch-dir>\n";
        return 2;
    }

    TestRunner tests;
    try {
        std::filesystem::path fixture(argv[1]);
        std::filesystem::path scratch(argv[2]);
        std::filesystem::create_directories(scratch);

        deepforge::compiler::CompileOptions options;
        options.capture_mlir = true;
        deepforge::compiler::CompilationResult compilation;
        auto status = deepforge::compiler::compile_file(fixture, options,
                                                        compilation);
        tests.good(status, "compile JSON for artifact");
        if (status.is_bad()) {
            return tests.finish();
        }
        tests.check(!compilation.imported_mlir.empty(),
                    "imported MLIR is captured");
        tests.check(!compilation.bufferized_mlir.empty(),
                    "bufferized MLIR is captured");
        tests.check(!compilation.variants[0].mlir.empty(),
                    "LLVM dialect MLIR is captured");
        if (compilation.metadata.pre_padding ==
                std::array<std::int64_t, 2>{0, 0} &&
            compilation.metadata.post_padding ==
                std::array<std::int64_t, 2>{0, 0}) {
            tests.check(compilation.workspace.size_bytes == 0 &&
                            compilation.workspace.allocations.empty(),
                        "zero padding does not allocate workspace");
        }
        tests.check(compilation.variants[0].symbol ==
                        compilation.metadata.function_name + "_scalar" &&
                        compilation.variants[1].symbol ==
                            compilation.metadata.function_name + "_avx2" &&
                        compilation.variants[2].symbol ==
                            compilation.metadata.function_name + "_avx512",
                    "variant symbols are unique and stable");

        std::vector<std::uint8_t> first;
        std::vector<std::uint8_t> second;
        tests.good(deepforge::compiler::serialize_artifact(compilation, first),
                   "serialize artifact");
        tests.good(deepforge::compiler::serialize_artifact(compilation, second),
                   "repeat artifact serialization");
        tests.check(first == second, "artifact serialization is deterministic");
        tests.check(first.size() > 8 && first[0] == 'D' && first[1] == 'F' &&
                        first[2] == 'O',
                    "artifact has DFOBJ magic");

        deepforge::compiler::ArtifactInfo parsed;
        tests.good(deepforge::compiler::parse_artifact(first, parsed),
                   "parse serialized artifact");
        tests.check(parsed.format_version ==
                        deepforge::compiler::kArtifactFormatVersion,
                    "writer emits artifact format v12");
        tests.check(parsed.adapter_kind ==
                        deepforge::compiler::ArtifactAdapterKind::
                            kConv2DRankedMemref,
                    "artifact records the transitional Conv adapter kind");
        tests.check(parsed.metadata == compilation.metadata,
                    "artifact metadata round-trips");
        tests.check(parsed.workspace == compilation.workspace,
                    "artifact workspace plan round-trips");
        tests.check(parsed.target_triple == compilation.target_triple,
                    "artifact target triple round-trips");
        tests.check(parsed.absolute_tolerance ==
                            deepforge::compiler::kDefaultAbsoluteTolerance &&
                        parsed.relative_tolerance ==
                            deepforge::compiler::kDefaultRelativeTolerance,
                    "artifact numeric contract round-trips");
        for (std::size_t index = 0; index < parsed.variants.size(); ++index) {
            tests.check(parsed.variants[index].symbol ==
                                compilation.variants[index].symbol &&
                            parsed.variants[index].required_features ==
                                compilation.variants[index].required_features &&
                            parsed.variants[index].object ==
                                compilation.variants[index].object,
                        "artifact variant section round-trips");
        }

        auto const saved_matmul_adapter = compilation.adapter_kind;
        auto const saved_matmul_metadata = compilation.metadata;
        compilation.adapter_kind =
            deepforge::compiler::ArtifactAdapterKind::
                kGenericRankedMemrefPointerTable;
        compilation.metadata.dynamic_shape_enabled = false;
        compilation.metadata.override_shape_enabled = true;
        compilation.metadata.override_policy =
            deepforge::compiler::ShapeOverridePolicy::kMatmul;
        compilation.metadata.override_role_uids = {9001, 9002, 9003};
        auto set_argument =
            [&](std::size_t index, std::int64_t uid, std::string name,
                std::vector<std::int64_t> dimensions,
                std::vector<std::int64_t> strides, std::uint64_t size_bytes,
                deepforge::compiler::TensorAccess access) {
                auto& argument = compilation.metadata.arguments[index];
                argument.uid = uid;
                argument.name = std::move(name);
                argument.data_type = deepforge::import::DataType::kFloat32;
                argument.dimensions = std::move(dimensions);
                argument.strides = std::move(strides);
                argument.size_bytes = size_bytes;
                argument.alignment = alignof(float);
                argument.access = access;
                argument.storage_policy =
                    deepforge::compiler::TensorStoragePolicy::kStrided;
                argument.ragged_offset_uid = 0;
                argument.ragged_sequence_uid = 0;
                argument.ragged_sequence_divisor = 1;
            };
        set_argument(0, 9001, "A", {2, 3}, {3, 1}, 24,
                     deepforge::compiler::TensorAccess::kRead);
        set_argument(1, 9002, "B", {3, 4}, {4, 1}, 48,
                     deepforge::compiler::TensorAccess::kRead);
        set_argument(2, 9003, "C", {2, 4}, {4, 1}, 32,
                     deepforge::compiler::TensorAccess::kWrite);
        std::vector<std::uint8_t> matmul_v12;
        tests.good(deepforge::compiler::serialize_artifact(compilation,
                                                           matmul_v12),
                   "serialize role-based MATMUL artifact for compatibility");
        compilation.adapter_kind = saved_matmul_adapter;
        compilation.metadata = saved_matmul_metadata;
        auto matmul_v6 = make_matmul_override_v6(matmul_v12);
        deepforge::compiler::ArtifactInfo matmul_v6_info;
        tests.good(deepforge::compiler::parse_artifact(matmul_v6,
                                                       matmul_v6_info),
                   "parse MATMUL override format-v6 artifact");
        tests.check(
            matmul_v6_info.format_version ==
                    deepforge::compiler::
                        kMatmulOverrideArtifactFormatVersion &&
                matmul_v6_info.metadata.override_policy ==
                    deepforge::compiler::ShapeOverridePolicy::kMatmul &&
                matmul_v6_info.metadata.override_role_uids ==
                    std::vector<std::int64_t>({9001, 9002, 9003}),
            "format-v6 reader preserves MATMUL role metadata");
        auto v6_sdpa_policy = matmul_v6;
        auto const v6_flags = metadata_numbers_offset(v6_sdpa_policy);
        write_u32(v6_sdpa_policy, v6_flags + 8, 3);
        refresh_checksum(v6_sdpa_policy);
        status = deepforge::compiler::parse_artifact(v6_sdpa_policy, parsed);
        tests.parse_error(status,
                          "format-v6 rejects the format-v7 SDPA policy");

        compilation.adapter_kind =
            deepforge::compiler::ArtifactAdapterKind::
                kGenericRankedMemrefPointerTable;
        compilation.metadata.dynamic_shape_enabled = false;
        compilation.metadata.override_shape_enabled = true;
        compilation.metadata.override_policy =
            deepforge::compiler::ShapeOverridePolicy::kSdpaForward;
        compilation.metadata.override_role_uids = {9101, 9102, 9103, 9104};
        compilation.metadata.arguments.resize(4);
        set_argument(0, 9101, "Q", {1, 2, 3, 2}, {12, 6, 2, 1}, 48,
                     deepforge::compiler::TensorAccess::kRead);
        set_argument(1, 9102, "K", {1, 1, 4, 2}, {8, 8, 2, 1}, 32,
                     deepforge::compiler::TensorAccess::kRead);
        set_argument(2, 9103, "V", {1, 1, 4, 3}, {12, 12, 3, 1}, 48,
                     deepforge::compiler::TensorAccess::kRead);
        set_argument(3, 9104, "O", {1, 2, 3, 3}, {18, 9, 3, 1}, 72,
                     deepforge::compiler::TensorAccess::kWrite);
        std::vector<std::uint8_t> sdpa_v12;
        tests.good(deepforge::compiler::serialize_artifact(compilation,
                                                           sdpa_v12),
                   "serialize role-based SDPA artifact for compatibility");
        compilation.adapter_kind = saved_matmul_adapter;
        compilation.metadata = saved_matmul_metadata;
        auto sdpa_v7 = make_sdpa_override_v7(sdpa_v12);
        deepforge::compiler::ArtifactInfo sdpa_v7_info;
        tests.good(deepforge::compiler::parse_artifact(sdpa_v7,
                                                       sdpa_v7_info),
                   "parse SDPA override format-v7 artifact");
        tests.check(
            sdpa_v7_info.format_version ==
                    deepforge::compiler::kSdpaOverrideArtifactFormatVersion &&
                sdpa_v7_info.metadata.override_policy ==
                    deepforge::compiler::ShapeOverridePolicy::kSdpaForward &&
                sdpa_v7_info.metadata.override_role_uids ==
                    std::vector<std::int64_t>({9101, 9102, 9103, 9104}),
            "format-v7 reader preserves SDPA role metadata");
        auto v7_reshape_policy = sdpa_v7;
        auto const v7_flags = metadata_numbers_offset(v7_reshape_policy);
        write_u32(v7_reshape_policy, v7_flags + 8, 4);
        refresh_checksum(v7_reshape_policy);
        status = deepforge::compiler::parse_artifact(v7_reshape_policy,
                                                      parsed);
        tests.parse_error(status,
                          "format-v7 rejects the format-v8 RESHAPE policy");

        compilation.adapter_kind =
            deepforge::compiler::ArtifactAdapterKind::
                kGenericRankedMemrefPointerTable;
        compilation.metadata.dynamic_shape_enabled = false;
        compilation.metadata.override_shape_enabled = true;
        compilation.metadata.override_policy =
            deepforge::compiler::ShapeOverridePolicy::kReshape;
        compilation.metadata.override_role_uids = {9201, 9202};
        compilation.metadata.arguments.resize(2);
        set_argument(0, 9201, "X", {2, 3, 4}, {12, 4, 1}, 96,
                     deepforge::compiler::TensorAccess::kRead);
        set_argument(1, 9202, "Y", {4, 6}, {6, 1}, 96,
                     deepforge::compiler::TensorAccess::kWrite);
        std::vector<std::uint8_t> reshape_v12;
        tests.good(deepforge::compiler::serialize_artifact(compilation,
                                                           reshape_v12),
                   "serialize role-based RESHAPE artifact for compatibility");
        compilation.adapter_kind = saved_matmul_adapter;
        compilation.metadata = saved_matmul_metadata;
        auto reshape_v8 = make_reshape_override_v8(reshape_v12);
        deepforge::compiler::ArtifactInfo reshape_v8_info;
        tests.good(deepforge::compiler::parse_artifact(reshape_v8,
                                                       reshape_v8_info),
                   "parse RESHAPE override format-v8 artifact");
        tests.check(
            reshape_v8_info.format_version ==
                    deepforge::compiler::
                        kReshapeOverrideArtifactFormatVersion &&
                reshape_v8_info.metadata.override_policy ==
                    deepforge::compiler::ShapeOverridePolicy::kReshape &&
                reshape_v8_info.metadata.override_role_uids ==
                    std::vector<std::int64_t>({9201, 9202}),
            "format-v8 reader preserves RESHAPE role metadata");
        auto malformed_v12_reduction = reshape_v12;
        auto const v12_flags =
            metadata_numbers_offset(malformed_v12_reduction);
        write_u32(malformed_v12_reduction, v12_flags + 8, 5);
        refresh_checksum(malformed_v12_reduction);
        status = deepforge::compiler::parse_artifact(
            malformed_v12_reduction, parsed);
        tests.parse_error(
            status,
            "format-v12 rejects REDUCTION roles with incompatible maximum shapes");
        auto v8_reduction_policy = reshape_v8;
        auto const v8_flags = metadata_numbers_offset(v8_reduction_policy);
        write_u32(v8_reduction_policy, v8_flags + 8, 5);
        refresh_checksum(v8_reduction_policy);
        status = deepforge::compiler::parse_artifact(v8_reduction_policy,
                                                      parsed);
        tests.parse_error(
            status,
            "format-v8 rejects the format-v9 REDUCTION policy");

        compilation.adapter_kind =
            deepforge::compiler::ArtifactAdapterKind::
                kGenericRankedMemrefPointerTable;
        compilation.metadata.dynamic_shape_enabled = false;
        compilation.metadata.override_shape_enabled = true;
        compilation.metadata.override_policy =
            deepforge::compiler::ShapeOverridePolicy::kReduction;
        compilation.metadata.override_role_uids = {9301, 9302};
        compilation.metadata.override_axis_map.clear();
        compilation.metadata.arguments.resize(2);
        set_argument(0, 9301, "X", {2, 3, 4}, {12, 4, 1}, 96,
                     deepforge::compiler::TensorAccess::kRead);
        set_argument(1, 9302, "Y", {2, 1, 4}, {4, 4, 1}, 32,
                     deepforge::compiler::TensorAccess::kWrite);
        std::vector<std::uint8_t> reduction_v12;
        tests.good(deepforge::compiler::serialize_artifact(compilation,
                                                           reduction_v12),
                   "serialize REDUCTION artifact for v9 compatibility");
        compilation.adapter_kind = saved_matmul_adapter;
        compilation.metadata = saved_matmul_metadata;
        auto reduction_v9 = make_reduction_override_v9(reduction_v12);
        deepforge::compiler::ArtifactInfo reduction_v9_info;
        tests.good(deepforge::compiler::parse_artifact(reduction_v9,
                                                       reduction_v9_info),
                   "parse REDUCTION override format-v9 artifact");
        tests.check(
            reduction_v9_info.format_version ==
                    deepforge::compiler::
                        kReductionOverrideArtifactFormatVersion &&
                reduction_v9_info.metadata.override_policy ==
                    deepforge::compiler::ShapeOverridePolicy::kReduction &&
                reduction_v9_info.metadata.override_role_uids ==
                    std::vector<std::int64_t>({9301, 9302}) &&
                reduction_v9_info.metadata.override_axis_map.empty(),
            "format-v9 reader preserves REDUCTION roles and defaults the axis map");

        compilation.adapter_kind =
            deepforge::compiler::ArtifactAdapterKind::
                kGenericRankedMemrefPointerTable;
        compilation.metadata.dynamic_shape_enabled = false;
        compilation.metadata.override_shape_enabled = true;
        compilation.metadata.override_policy =
            deepforge::compiler::ShapeOverridePolicy::kTranspose;
        compilation.metadata.override_role_uids = {9401, 9402};
        compilation.metadata.override_axis_map = {2, 0, 1};
        compilation.metadata.arguments.resize(2);
        set_argument(0, 9401, "X", {2, 3, 4}, {12, 4, 1}, 96,
                     deepforge::compiler::TensorAccess::kRead);
        set_argument(1, 9402, "Y", {4, 2, 3}, {6, 3, 1}, 96,
                     deepforge::compiler::TensorAccess::kWrite);
        std::vector<std::uint8_t> transpose_v12;
        tests.good(deepforge::compiler::serialize_artifact(compilation,
                                                           transpose_v12),
                   "serialize current TRANSPOSE override artifact");
        deepforge::compiler::ArtifactInfo transpose_v12_info;
        tests.good(deepforge::compiler::parse_artifact(transpose_v12,
                                                       transpose_v12_info),
                   "parse current TRANSPOSE override artifact");
        tests.check(
            transpose_v12_info.format_version ==
                    deepforge::compiler::kArtifactFormatVersion &&
                transpose_v12_info.metadata.override_policy ==
                    deepforge::compiler::ShapeOverridePolicy::kTranspose &&
                transpose_v12_info.metadata.override_role_uids ==
                    std::vector<std::int64_t>({9401, 9402}) &&
                transpose_v12_info.metadata.override_axis_map ==
                    std::vector<std::int64_t>({2, 0, 1}),
            "current format preserves TRANSPOSE roles and permutation");
        compilation.adapter_kind = saved_matmul_adapter;
        compilation.metadata = saved_matmul_metadata;
        auto transpose_v10 = make_transpose_override_v10(transpose_v12);
        deepforge::compiler::ArtifactInfo transpose_v10_info;
        tests.good(deepforge::compiler::parse_artifact(transpose_v10,
                                                       transpose_v10_info),
                   "parse TRANSPOSE override format-v10 artifact");
        tests.check(
            transpose_v10_info.format_version ==
                    deepforge::compiler::
                        kTransposeOverrideArtifactFormatVersion &&
                transpose_v10_info.metadata.override_policy ==
                    deepforge::compiler::ShapeOverridePolicy::kTranspose &&
                transpose_v10_info.metadata.override_role_uids ==
                    std::vector<std::int64_t>({9401, 9402}) &&
                transpose_v10_info.metadata.override_axis_map ==
                    std::vector<std::int64_t>({2, 0, 1}),
            "format-v10 reader preserves TRANSPOSE roles and permutation");
        auto v9_transpose_policy =
            make_reduction_override_v9(transpose_v12);
        status = deepforge::compiler::parse_artifact(v9_transpose_policy,
                                                      parsed);
        tests.parse_error(status,
                          "format-v9 rejects the format-v10 TRANSPOSE policy");
        auto duplicate_transpose_axis = transpose_v12;
        auto const transpose_axis_count =
            override_axis_count_offset(duplicate_transpose_axis);
        write_u64(duplicate_transpose_axis, transpose_axis_count + 4 + 8, 2);
        refresh_checksum(duplicate_transpose_axis);
        status = deepforge::compiler::parse_artifact(duplicate_transpose_axis,
                                                      parsed);
        tests.parse_error(status,
                          "current format rejects duplicate TRANSPOSE axes");
        auto non_axis_policy_map = transpose_v12;
        auto const transpose_flags =
            metadata_numbers_offset(non_axis_policy_map);
        write_u32(non_axis_policy_map, transpose_flags + 8, 4);
        refresh_checksum(non_axis_policy_map);
        status = deepforge::compiler::parse_artifact(non_axis_policy_map,
                                                      parsed);
        tests.parse_error(
            status,
            "current format rejects an axis map on a non-axis policy");

        compilation.adapter_kind =
            deepforge::compiler::ArtifactAdapterKind::
                kGenericRankedMemrefPointerTable;
        compilation.metadata.dynamic_shape_enabled = false;
        compilation.metadata.override_shape_enabled = true;
        compilation.metadata.override_policy =
            deepforge::compiler::ShapeOverridePolicy::kConcatenate;
        compilation.metadata.override_role_uids = {9501, 9502, 9503};
        compilation.metadata.override_axis_map = {1};
        compilation.metadata.arguments.resize(3);
        set_argument(0, 9501, "X0", {2, 1}, {1, 1}, 8,
                     deepforge::compiler::TensorAccess::kRead);
        set_argument(1, 9502, "X1", {2, 2}, {2, 1}, 16,
                     deepforge::compiler::TensorAccess::kRead);
        set_argument(2, 9503, "Y", {2, 3}, {3, 1}, 24,
                     deepforge::compiler::TensorAccess::kWrite);
        std::vector<std::uint8_t> concatenate_v12;
        tests.good(deepforge::compiler::serialize_artifact(compilation,
                                                           concatenate_v12),
                   "serialize current CONCATENATE override artifact");
        auto concatenate_v11 = concatenate_v12;
        write_u32(concatenate_v11, 8,
                  deepforge::compiler::
                      kConcatenateOverrideArtifactFormatVersion);
        refresh_checksum(concatenate_v11);
        deepforge::compiler::ArtifactInfo concatenate_v11_info;
        tests.good(deepforge::compiler::parse_artifact(concatenate_v11,
                                                       concatenate_v11_info),
                   "parse CONCATENATE override format-v11 artifact");
        tests.check(
            concatenate_v11_info.format_version ==
                    deepforge::compiler::
                        kConcatenateOverrideArtifactFormatVersion &&
                concatenate_v11_info.metadata.override_policy ==
                    deepforge::compiler::ShapeOverridePolicy::kConcatenate &&
                concatenate_v11_info.metadata.override_role_uids ==
                    std::vector<std::int64_t>({9501, 9502, 9503}) &&
                concatenate_v11_info.metadata.override_axis_map ==
                    std::vector<std::int64_t>({1}),
            "format-v11 preserves CONCATENATE roles and axis");
        auto v11_block_scale_policy = concatenate_v11;
        auto const concatenate_flags =
            metadata_numbers_offset(v11_block_scale_policy);
        write_u32(
            v11_block_scale_policy, concatenate_flags + 8,
            static_cast<std::uint32_t>(
                deepforge::compiler::ShapeOverridePolicy::kBlockScaleMatmul));
        refresh_checksum(v11_block_scale_policy);
        status = deepforge::compiler::parse_artifact(v11_block_scale_policy,
                                                      parsed);
        tests.parse_error(
            status,
            "format-v11 rejects the format-v12 block-scale MATMUL policy");
        auto malformed_v12_block_scale = concatenate_v12;
        auto const current_flags =
            metadata_numbers_offset(malformed_v12_block_scale);
        write_u32(
            malformed_v12_block_scale, current_flags + 8,
            static_cast<std::uint32_t>(
                deepforge::compiler::ShapeOverridePolicy::kBlockScaleMatmul));
        refresh_checksum(malformed_v12_block_scale);
        status = deepforge::compiler::parse_artifact(
            malformed_v12_block_scale, parsed);
        tests.parse_error(
            status,
            "format-v12 rejects malformed block-scale MATMUL role metadata");
        compilation.adapter_kind = saved_matmul_adapter;
        compilation.metadata = saved_matmul_metadata;
        auto v10_concatenate_policy = concatenate_v11;
        write_u32(v10_concatenate_policy, 8,
                  deepforge::compiler::
                      kTransposeOverrideArtifactFormatVersion);
        refresh_checksum(v10_concatenate_policy);
        status = deepforge::compiler::parse_artifact(v10_concatenate_policy,
                                                      parsed);
        tests.parse_error(
            status,
            "format-v10 rejects the format-v11 CONCATENATE policy");

        auto ragged_sequence_v5 = make_ragged_sequence_v5(first);
        deepforge::compiler::ArtifactInfo ragged_sequence_v5_info;
        tests.good(deepforge::compiler::parse_artifact(
                       ragged_sequence_v5, ragged_sequence_v5_info),
                   "parse ragged-sequence format-v5 artifact");
        tests.check(
            ragged_sequence_v5_info.format_version ==
                    deepforge::compiler::
                        kRaggedSequenceArtifactFormatVersion &&
                ragged_sequence_v5_info.metadata == compilation.metadata,
            "format-v5 reader defaults override role UIDs to empty");
        auto v5_matmul_policy = ragged_sequence_v5;
        auto const v5_flags = metadata_numbers_offset(v5_matmul_policy);
        write_u32(v5_matmul_policy, v5_flags + 4, 1);
        write_u32(v5_matmul_policy, v5_flags + 8, 2);
        refresh_checksum(v5_matmul_policy);
        status = deepforge::compiler::parse_artifact(v5_matmul_policy, parsed);
        tests.parse_error(status,
                          "format-v5 rejects the format-v6 MATMUL policy");

        auto ragged_v4 = make_ragged_v4(first);
        deepforge::compiler::ArtifactInfo ragged_v4_info;
        tests.good(deepforge::compiler::parse_artifact(ragged_v4,
                                                       ragged_v4_info),
                   "parse ragged format-v4 artifact");
        tests.check(
            ragged_v4_info.format_version ==
                    deepforge::compiler::kRaggedArtifactFormatVersion &&
                ragged_v4_info.metadata == compilation.metadata,
            "format-v4 reader defaults ragged sequence divisor to one");

        auto shape_v3 = make_shape_override_v3(first);
        deepforge::compiler::ArtifactInfo shape_v3_info;
        tests.good(deepforge::compiler::parse_artifact(shape_v3,
                                                       shape_v3_info),
                   "parse shape-override format-v3 artifact");
        tests.check(
            shape_v3_info.format_version ==
                    deepforge::compiler::kShapeOverrideArtifactFormatVersion &&
                shape_v3_info.metadata == compilation.metadata,
            "format-v3 reader defaults tensor storage policy to strided");

        auto static_v2 = make_static_metadata_v2(first);
        deepforge::compiler::ArtifactInfo static_v2_info;
        tests.good(deepforge::compiler::parse_artifact(static_v2,
                                                       static_v2_info),
                   "parse static-metadata format-v2 artifact");
        tests.check(
            static_v2_info.format_version ==
                    deepforge::compiler::kStaticMetadataArtifactFormatVersion &&
                static_v2_info.metadata == compilation.metadata,
            "format-v2 reader defaults dynamic metadata to disabled");

        auto legacy_v1 = make_legacy_v1(first);
        deepforge::compiler::ArtifactInfo legacy_info;
        tests.good(deepforge::compiler::parse_artifact(legacy_v1, legacy_info),
                   "parse legacy format-v1 artifact");
        tests.check(legacy_info.format_version ==
                            deepforge::compiler::kLegacyArtifactFormatVersion &&
                        legacy_info.adapter_kind ==
                            deepforge::compiler::ArtifactAdapterKind::
                                kConv2DRankedMemref &&
                        legacy_info.metadata == compilation.metadata,
                    "legacy reader reconstructs the generic argument table");
        std::unique_ptr<deepforge::runtime::Executable> legacy_executable;
        tests.good(deepforge::compiler::load_artifact_executable(
                       legacy_v1, legacy_executable),
                   "load legacy format-v1 artifact");
        tests.check(legacy_executable != nullptr,
                    "legacy artifact produces an executable");
        tests.check(write_bytes(scratch / "scalar.o",
                                compilation.variants[0].object),
                    "write scalar object for ISA inspection");
        tests.check(write_bytes(scratch / "avx2.o",
                                compilation.variants[1].object),
                    "write AVX2 object for ISA inspection");
        tests.check(write_bytes(scratch / "avx512.o",
                                compilation.variants[2].object),
                    "write AVX-512 object for ISA inspection");

        auto artifact_path = scratch / "conv2d.dfo";
        tests.good(deepforge::compiler::write_artifact(artifact_path,
                                                       compilation),
                   "atomically write artifact");
        deepforge::compiler::ArtifactInfo from_file;
        tests.good(deepforge::compiler::read_artifact(artifact_path, from_file),
                   "read artifact file");
        tests.check(from_file.metadata == compilation.metadata,
                    "artifact file metadata matches");

        auto concurrent_path = scratch / "concurrent.dfo";
        deepforge::import::Status concurrent_status_a;
        deepforge::import::Status concurrent_status_b;
        std::thread writer_a([&] {
            concurrent_status_a = deepforge::compiler::write_artifact(
                concurrent_path, compilation);
        });
        std::thread writer_b([&] {
            concurrent_status_b = deepforge::compiler::write_artifact(
                concurrent_path, compilation);
        });
        writer_a.join();
        writer_b.join();
        tests.good(concurrent_status_a,
                   "first concurrent artifact publication");
        tests.good(concurrent_status_b,
                   "second concurrent artifact publication");
        deepforge::compiler::ArtifactInfo concurrent_artifact;
        tests.good(deepforge::compiler::read_artifact(
                       concurrent_path, concurrent_artifact),
                   "concurrently published artifact remains readable");
        tests.check(concurrent_artifact.metadata == compilation.metadata,
                    "concurrent artifact preserves metadata");

        std::unique_ptr<deepforge::runtime::Executable> loaded_executable;
        deepforge::compiler::ArtifactInfo loaded_info;
        tests.good(deepforge::compiler::load_artifact_executable(
                       artifact_path, loaded_executable, &loaded_info),
                   "load artifact objects into ORC runtime");
        tests.check(loaded_executable != nullptr &&
                        loaded_info.metadata == compilation.metadata &&
                        loaded_executable->get_workspace_size() ==
                            compilation.executable->get_workspace_size(),
                    "loaded executable preserves runtime metadata");
        if (loaded_executable) {
            std::vector<float> x(element_count(compilation.metadata.x_shape));
            std::vector<float> w(element_count(compilation.metadata.w_shape));
            std::vector<float> compiled_y(
                element_count(compilation.metadata.y_shape));
            std::vector<float> loaded_y(compiled_y.size());
            for (std::size_t index = 0; index < x.size(); ++index) {
                x[index] = static_cast<float>(
                               (static_cast<int>(index * 7) % 19) - 9) /
                           7.0F;
            }
            for (std::size_t index = 0; index < w.size(); ++index) {
                w[index] = static_cast<float>(
                               (static_cast<int>(index * 5) % 13) - 6) /
                           5.0F;
            }
            auto workspace_size = static_cast<std::size_t>(
                loaded_executable->get_workspace_size());
            AlignedBytes compiled_workspace(workspace_size);
            AlignedBytes loaded_workspace(workspace_size);
            deepforge::runtime::VariantPack compiled_pack{
                {compilation.metadata.x_uid, x.data()},
                {compilation.metadata.w_uid, w.data()},
                {compilation.metadata.y_uid, compiled_y.data()}};
            deepforge::runtime::VariantPack loaded_pack{
                {loaded_info.metadata.x_uid, x.data()},
                {loaded_info.metadata.w_uid, w.data()},
                {loaded_info.metadata.y_uid, loaded_y.data()}};
            std::array<deepforge::runtime::CpuVariant, 3> variants{
                deepforge::runtime::CpuVariant::kScalar,
                deepforge::runtime::CpuVariant::kAvx2,
                deepforge::runtime::CpuVariant::kAvx512};
            for (auto variant : variants) {
                if (!loaded_executable->supports_variant(variant)) {
                    continue;
                }
                std::fill(compiled_y.begin(), compiled_y.end(), -1.0F);
                std::fill(loaded_y.begin(), loaded_y.end(), -2.0F);
                tests.good(compilation.executable->execute_variant(
                               variant, nullptr, compiled_pack,
                               compiled_workspace.pointer),
                           "execute in-process compiled variant");
                tests.good(loaded_executable->execute_variant(
                               variant, nullptr, loaded_pack,
                               loaded_workspace.pointer),
                           "execute ORC-loaded artifact variant");
                tests.check(close(compiled_y, loaded_y),
                            "loaded artifact output matches compiled output");
            }
            std::fill(loaded_y.begin(), loaded_y.end(), -2.0F);
            tests.good(loaded_executable->execute(nullptr, loaded_pack,
                                                  loaded_workspace.pointer),
                       "loaded artifact automatic dispatch executes");
            tests.check(close(compiled_y, loaded_y),
                        "loaded artifact automatic dispatch is correct");
        }

        std::unique_ptr<deepforge::runtime::Executable> memory_executable;
        tests.good(deepforge::compiler::load_artifact_executable(
                       first, memory_executable),
                   "load artifact directly from memory");
        tests.check(memory_executable != nullptr,
                    "memory artifact produces an executable");
        if (memory_executable) {
            std::vector<float> x(element_count(compilation.metadata.x_shape),
                                 0.0F);
            std::vector<float> w(element_count(compilation.metadata.w_shape),
                                 1.0F);
            std::vector<float> y(element_count(compilation.metadata.y_shape),
                                 -1.0F);
            AlignedBytes workspace(static_cast<std::size_t>(
                memory_executable->get_workspace_size()));
            deepforge::runtime::VariantPack pack{
                {compilation.metadata.x_uid, x.data()},
                {compilation.metadata.w_uid, w.data()},
                {compilation.metadata.y_uid, y.data()}};
            tests.good(memory_executable->execute(nullptr, pack,
                                                  workspace.pointer),
                       "execute artifact loaded directly from memory");
            tests.check(std::all_of(y.begin(), y.end(),
                                    [](float value) { return value == 0.0F; }),
                        "memory-loaded artifact writes expected output");
        }

        auto corrupted = first;
        corrupted[20] ^= 0x40;
        status = deepforge::compiler::parse_artifact(corrupted, parsed);
        tests.parse_error(status, "artifact corruption is rejected");
        auto malformed_dynamic_flag = first;
        write_u32(malformed_dynamic_flag,
                  metadata_numbers_offset(malformed_dynamic_flag), 2);
        refresh_checksum(malformed_dynamic_flag);
        status = deepforge::compiler::parse_artifact(malformed_dynamic_flag,
                                                      parsed);
        tests.parse_error(status,
                          "checksummed non-boolean dynamic flag is rejected");
        auto inconsistent_override_policy = first;
        write_u32(inconsistent_override_policy,
                  metadata_numbers_offset(inconsistent_override_policy) + 4,
                  1);
        refresh_checksum(inconsistent_override_policy);
        status = deepforge::compiler::parse_artifact(
            inconsistent_override_policy, parsed);
        tests.parse_error(status,
                          "checksummed inconsistent override policy is rejected");
        auto unknown_override_policy = first;
        write_u32(unknown_override_policy,
                  metadata_numbers_offset(unknown_override_policy) + 8,
                  std::numeric_limits<std::uint32_t>::max());
        refresh_checksum(unknown_override_policy);
        status = deepforge::compiler::parse_artifact(unknown_override_policy,
                                                      parsed);
        tests.parse_error(status,
                          "checksummed unknown override policy is rejected");
        auto excessive_override_roles = first;
        write_u32(excessive_override_roles,
                  metadata_numbers_offset(excessive_override_roles) + 12,
                  std::numeric_limits<std::uint32_t>::max());
        refresh_checksum(excessive_override_roles);
        status = deepforge::compiler::parse_artifact(excessive_override_roles,
                                                      parsed);
        tests.parse_error(status,
                          "checksummed excessive override role count is rejected");
        auto excessive_override_axes = first;
        write_u32(excessive_override_axes,
                  override_axis_count_offset(excessive_override_axes),
                  std::numeric_limits<std::uint32_t>::max());
        refresh_checksum(excessive_override_axes);
        status = deepforge::compiler::parse_artifact(excessive_override_axes,
                                                      parsed);
        tests.parse_error(status,
                          "checksummed excessive override axis count is rejected");
        auto dynamic_conv_adapter = first;
        auto const dynamic_flags = metadata_numbers_offset(dynamic_conv_adapter);
        write_u32(dynamic_conv_adapter, dynamic_flags, 1);
        write_u32(dynamic_conv_adapter, dynamic_flags + 4, 1);
        write_u32(dynamic_conv_adapter, dynamic_flags + 8, 1);
        refresh_checksum(dynamic_conv_adapter);
        status = deepforge::compiler::parse_artifact(dynamic_conv_adapter,
                                                      parsed);
        tests.parse_error(status,
                          "Conv adapter rejects dynamic override metadata");
        auto invalid_alignment = first;
        write_u64(invalid_alignment,
                  workspace_alignment_offset(invalid_alignment), 0);
        refresh_checksum(invalid_alignment);
        status = deepforge::compiler::parse_artifact(invalid_alignment, parsed);
        tests.parse_error(status,
                          "checksummed zero workspace alignment is rejected");
        auto invalid_argument_alignment = first;
        write_u64(invalid_argument_alignment,
                  first_argument_alignment_offset(invalid_argument_alignment),
                  0);
        refresh_checksum(invalid_argument_alignment);
        status = deepforge::compiler::parse_artifact(
            invalid_argument_alignment, parsed);
        tests.parse_error(status,
                          "checksummed zero argument alignment is rejected");
        auto invalid_storage_policy = first;
        write_u32(invalid_storage_policy,
                  first_argument_storage_policy_offset(
                      invalid_storage_policy),
                  std::numeric_limits<std::uint32_t>::max());
        refresh_checksum(invalid_storage_policy);
        status = deepforge::compiler::parse_artifact(invalid_storage_policy,
                                                      parsed);
        tests.parse_error(status,
                          "checksummed unknown tensor storage policy is rejected");
        auto invalid_ragged_sequence_divisor = first;
        write_u64(invalid_ragged_sequence_divisor,
                  first_argument_ragged_sequence_divisor_offset(
                      invalid_ragged_sequence_divisor),
                  0);
        refresh_checksum(invalid_ragged_sequence_divisor);
        status = deepforge::compiler::parse_artifact(
            invalid_ragged_sequence_divisor, parsed);
        tests.parse_error(
            status,
            "checksummed zero ragged sequence divisor is rejected");
        auto inconsistent_strided_storage = first;
        write_u64(inconsistent_strided_storage,
                  first_argument_storage_policy_offset(
                      inconsistent_strided_storage) +
                      4,
                  999);
        refresh_checksum(inconsistent_strided_storage);
        status = deepforge::compiler::parse_artifact(
            inconsistent_strided_storage, parsed);
        tests.parse_error(status,
                          "strided argument rejects ragged UID references");
        auto invalid_adapter_kind = first;
        write_u32(invalid_adapter_kind,
                  adapter_kind_offset(invalid_adapter_kind),
                  std::numeric_limits<std::uint32_t>::max());
        refresh_checksum(invalid_adapter_kind);
        status = deepforge::compiler::parse_artifact(invalid_adapter_kind,
                                                      parsed);
        tests.parse_error(status,
                          "checksummed unknown adapter kind is rejected");
        auto invalid_shape = first;
        auto const metadata_offset = adapter_payload_offset(invalid_shape);
        auto const maximum_dimension = static_cast<std::uint64_t>(
            std::numeric_limits<std::int64_t>::max());
        write_u64(invalid_shape, metadata_offset + 3 * 8,
                  maximum_dimension);
        write_u64(invalid_shape, metadata_offset + 11 * 8,
                  maximum_dimension);
        write_u64(invalid_shape, metadata_offset + 15 * 8,
                  maximum_dimension);
        refresh_checksum(invalid_shape);
        status = deepforge::compiler::parse_artifact(invalid_shape, parsed);
        tests.parse_error(status,
                          "checksummed tensor element overflow is rejected");
        auto invalid_allocation_count = first;
        write_u32(invalid_allocation_count,
                  workspace_alignment_offset(invalid_allocation_count) + 8,
                  std::numeric_limits<std::uint32_t>::max());
        refresh_checksum(invalid_allocation_count);
        status = deepforge::compiler::parse_artifact(invalid_allocation_count,
                                                      parsed);
        tests.parse_error(status,
                          "checksummed excessive allocation count is rejected");
        auto invalid_feature = first;
        constexpr std::string_view baseline = "baseline";
        auto feature = std::search(invalid_feature.begin(),
                                   invalid_feature.end() - 8,
                                   baseline.begin(), baseline.end());
        tests.check(feature != invalid_feature.end() - 8,
                    "locate scalar feature metadata");
        if (feature != invalid_feature.end() - 8) {
            *feature = 'x';
            refresh_checksum(invalid_feature);
            auto previous_target = parsed.target_triple;
            status = deepforge::compiler::parse_artifact(invalid_feature,
                                                          parsed);
            tests.parse_error(status,
                              "checksummed wrong feature metadata is rejected");
            tests.check(parsed.target_triple == previous_target,
                        "failed parse leaves output unchanged");
        }
        auto saved_alignment = compilation.workspace.alignment;
        compilation.workspace.alignment = 0;
        std::vector<std::uint8_t> invalid_output;
        status = deepforge::compiler::serialize_artifact(compilation,
                                                         invalid_output);
        tests.check(status.code() ==
                        deepforge::import::ErrorCode::kInvalidValue,
                    "writer rejects invalid workspace contract");
        compilation.workspace.alignment = saved_alignment;
        auto saved_arguments = compilation.metadata.arguments;
        compilation.metadata.arguments.clear();
        invalid_output = {0x56, 0x78};
        status = deepforge::compiler::serialize_artifact(compilation,
                                                         invalid_output);
        tests.check(status.code() ==
                        deepforge::import::ErrorCode::kInvalidValue,
                    "writer rejects an empty generic argument table");
        tests.check(invalid_output == std::vector<std::uint8_t>({0x56, 0x78}),
                    "failed argument serialization leaves output unchanged");
        compilation.metadata.arguments = std::move(saved_arguments);
        auto const saved_override_flag =
            compilation.metadata.override_shape_enabled;
        compilation.metadata.override_shape_enabled = true;
        status = deepforge::compiler::serialize_artifact(compilation,
                                                         invalid_output);
        tests.check(status.code() ==
                        deepforge::import::ErrorCode::kInvalidValue,
                    "writer rejects inconsistent override metadata");
        compilation.metadata.override_shape_enabled = saved_override_flag;
        auto const saved_override_policy = compilation.metadata.override_policy;
        auto const saved_override_roles =
            compilation.metadata.override_role_uids;
        compilation.metadata.override_shape_enabled = true;
        compilation.metadata.override_policy =
            deepforge::compiler::ShapeOverridePolicy::kMatmul;
        compilation.metadata.override_role_uids = {
            compilation.metadata.arguments[0].uid,
            compilation.metadata.arguments[0].uid,
            compilation.metadata.arguments[2].uid};
        status = deepforge::compiler::serialize_artifact(compilation,
                                                         invalid_output);
        tests.check(status.code() ==
                        deepforge::import::ErrorCode::kInvalidValue,
                    "writer rejects duplicate MATMUL override role UIDs");
        compilation.metadata.override_shape_enabled = saved_override_flag;
        compilation.metadata.override_policy = saved_override_policy;
        compilation.metadata.override_role_uids = saved_override_roles;
        auto const saved_metadata = compilation.metadata;
        compilation.metadata.x_shape[0] =
            std::numeric_limits<std::int64_t>::max();
        compilation.metadata.y_shape[0] =
            std::numeric_limits<std::int64_t>::max();
        compilation.metadata.padded_x_shape[0] =
            std::numeric_limits<std::int64_t>::max();
        invalid_output = {0x12, 0x34};
        status = deepforge::compiler::serialize_artifact(compilation,
                                                         invalid_output);
        tests.check(status.code() ==
                        deepforge::import::ErrorCode::kInvalidShape,
                    "writer rejects tensor element overflow");
        tests.check(invalid_output == std::vector<std::uint8_t>({0x12, 0x34}),
                    "failed artifact serialization leaves output unchanged");
        compilation.metadata = saved_metadata;
        auto truncated = first;
        truncated.pop_back();
        status = deepforge::compiler::parse_artifact(truncated, parsed);
        tests.parse_error(status, "truncated artifact is rejected");
        for (auto payload_size : std::array<std::size_t, 3>{
                 16, workspace_alignment_offset(first) + 4,
                 first.size() / 2}) {
            auto checksummed_truncation = std::vector<std::uint8_t>(
                first.begin(), first.begin() +
                                   static_cast<std::ptrdiff_t>(payload_size));
            checksummed_truncation.resize(checksummed_truncation.size() + 8);
            refresh_checksum(checksummed_truncation);
            status = deepforge::compiler::parse_artifact(
                checksummed_truncation, parsed);
            tests.parse_error(status,
                              "checksummed internal truncation is rejected");
        }
        auto trailing = first;
        trailing.push_back(0);
        status = deepforge::compiler::parse_artifact(trailing, parsed);
        tests.parse_error(status, "artifact trailing byte is rejected");

        auto document = Json::parse(read_text(fixture));
        auto ubjson = Json::to_ubjson(document);
        auto ubjson_path = scratch / "conv2d.ubjson";
        tests.check(write_bytes(ubjson_path, ubjson), "write UBJSON fixture");
        deepforge::compiler::CompileOptions ubjson_options;
        ubjson_options.input_format = deepforge::import::InputFormat::kUbjson;
        deepforge::compiler::CompilationResult ubjson_compilation;
        status = deepforge::compiler::compile_file(
            ubjson_path, ubjson_options, ubjson_compilation);
        tests.good(status, "compile strict UBJSON end to end");
        if (status.is_good()) {
            tests.check(ubjson_compilation.metadata == compilation.metadata &&
                            ubjson_compilation.workspace == compilation.workspace,
                        "JSON and UBJSON compilation metadata is identical");
            std::vector<std::uint8_t> ubjson_artifact;
            tests.good(deepforge::compiler::serialize_artifact(
                           ubjson_compilation, ubjson_artifact),
                       "serialize independently compiled UBJSON artifact");
            tests.check(ubjson_artifact == first,
                        "independent JSON and UBJSON compilation is reproducible");
            std::vector<float> x(element_count(ubjson_compilation.metadata.x_shape),
                                 0.0F);
            std::vector<float> w(element_count(ubjson_compilation.metadata.w_shape),
                                 1.0F);
            std::vector<float> y(element_count(ubjson_compilation.metadata.y_shape),
                                 -1.0F);
            AlignedBytes workspace(
                static_cast<std::size_t>(
                    ubjson_compilation.executable->get_workspace_size()));
            deepforge::runtime::VariantPack pack{
                {ubjson_compilation.metadata.x_uid, x.data()},
                {ubjson_compilation.metadata.w_uid, w.data()},
                {ubjson_compilation.metadata.y_uid, y.data()}};
            status = ubjson_compilation.executable->execute(
                nullptr, pack, workspace.pointer);
            tests.good(status, "execute UBJSON-compiled graph");
            tests.check(std::all_of(y.begin(), y.end(),
                                    [](float value) { return value == 0.0F; }),
                        "UBJSON-compiled graph writes expected output");
        }
    } catch (std::exception const& exception) {
        std::cerr << "FAIL: unexpected exception: " << exception.what() << '\n';
        return 1;
    }
    return tests.finish();
}
