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
            throw std::runtime_error("artifact test string is truncated");
        }
        offset += size;
    }
    return offset;
}

std::size_t workspace_alignment_offset(
    std::span<std::uint8_t const> bytes) {
    auto offset = metadata_numbers_offset(bytes);
    offset += (3 + 4 * 4 + 4 * 2) * 8;
    return offset + 8;
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
        auto invalid_alignment = first;
        write_u64(invalid_alignment,
                  workspace_alignment_offset(invalid_alignment), 0);
        refresh_checksum(invalid_alignment);
        status = deepforge::compiler::parse_artifact(invalid_alignment, parsed);
        tests.parse_error(status,
                          "checksummed zero workspace alignment is rejected");
        auto invalid_shape = first;
        auto const metadata_offset = metadata_numbers_offset(invalid_shape);
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
