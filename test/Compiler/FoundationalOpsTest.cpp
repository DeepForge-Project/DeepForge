#include "DeepForge/Compiler/Artifact.h"
#include "DeepForge/Compiler/Codegen.h"
#include "DeepForge/Import/SerializedGraphImporter.h"

#include <cudnn_frontend/thirdparty/nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <span>
#include <string>
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

    int finish() const {
        if (failures_ == 0) {
            std::cout << "deepforge-foundational-ops: " << checks_
                      << " checks passed\n";
            return 0;
        }
        std::cerr << "deepforge-foundational-ops: " << failures_ << " of "
                  << checks_ << " checks failed\n";
        return 1;
    }

private:
    int checks_ = 0;
    int failures_ = 0;
};

struct AlignedBytes {
    explicit AlignedBytes(std::size_t size) : size(size) {
        if (size != 0) {
            pointer = std::aligned_alloc(64, size);
        }
    }
    ~AlignedBytes() { std::free(pointer); }
    AlignedBytes(AlignedBytes const&) = delete;
    AlignedBytes& operator=(AlignedBytes const&) = delete;

    void* pointer = nullptr;
    std::size_t size = 0;
};

Json tensor(std::string name,
            std::int64_t uid,
            std::vector<std::int64_t> dimensions,
            std::vector<std::int64_t> strides,
            bool is_virtual) {
    return Json{{"name", std::move(name)},
                {"data_type", "FLOAT"},
                {"dim", std::move(dimensions)},
                {"stride", std::move(strides)},
                {"is_virtual", is_virtual},
                {"pass_by_value", nullptr},
                {"is_pass_by_value", false},
                {"reordering_type", "NONE"},
                {"uid", uid},
                {"uid_assigned", true}};
}

Json reshape_node(std::string name,
                  std::int64_t x,
                  std::int64_t y,
                  std::vector<std::int64_t> dimensions,
                  std::vector<std::int64_t> strides) {
    return Json{{"tag", "RESHAPE"},
                {"name", std::move(name)},
                {"inputs", Json::object({{"X", x}})},
                {"outputs", Json::object({{"Y", y}})},
                {"compute_data_type", "FLOAT"},
                {"dim", std::move(dimensions)},
                {"stride", std::move(strides)},
                {"reshape_mode", "LOGICAL"}};
}

Json reshape_graph() {
    return Json{
        {"context",
         Json{{"name", "reshape_chain"},
              {"compute_data_type", "FLOAT"},
              {"intermediate_data_type", "FLOAT"},
              {"io_data_type", "FLOAT"},
              {"sm_count", -1},
              {"is_dynamic_shape_enabled", false},
              {"is_override_shape_enabled", false}}},
        {"graph_uid", 2001},
        {"json_version", "1.0"},
        {"cudnn_backend_version", "cpu-test"},
        {"cudnn_frontend_version", 12400},
        {"nodes",
         Json::array({reshape_node("reshape_to_virtual", 1, 2, {1, 6},
                                   {6, 1}),
                      reshape_node("reshape_to_output", 2, 3, {3, 2},
                                   {3, 1})})},
        {"tensors",
         Json::object(
             {{"1", tensor("X", 1, {2, 3}, {4, 1}, false)},
              {"2", tensor("V", 2, {1, 6}, {6, 1}, true)},
              {"3", tensor("Y", 3, {3, 2}, {3, 1}, false)}})}};
}

Json transform_graph() {
    auto transpose = Json{{"tag", "TRANSPOSE"},
                          {"name", "transpose"},
                          {"inputs", Json::object({{"X", 11}})},
                          {"outputs", Json::object({{"Y", 12}})},
                          {"compute_data_type", "FLOAT"},
                          {"permutation", Json::array({1, 0})}};
    auto slice = Json{{"tag", "SLICE"},
                      {"name", "slice"},
                      {"inputs", Json::object({{"X", 12}})},
                      {"outputs", Json::object({{"Y", 13}})},
                      {"compute_data_type", "FLOAT"},
                      {"slices",
                       Json::array({Json::array({1, 3}),
                                    Json::array({0, 2})})},
                      {"slice_strides", Json::array({1, 1})}};
    auto concatenate = Json{
        {"tag", "CONCATENATE"},
        {"name", "concatenate"},
        {"inputs", Json::object({{"0", 13}, {"1", 14}})},
        {"outputs", Json::object({{"Y", 15}})},
        {"axis", 1},
        {"in_place_index", nullptr}};
    return Json{
        {"context",
         Json{{"name", "transform_chain"},
              {"compute_data_type", "FLOAT"},
              {"intermediate_data_type", "FLOAT"},
              {"io_data_type", "FLOAT"},
              {"sm_count", -1},
              {"is_dynamic_shape_enabled", false},
              {"is_override_shape_enabled", false}}},
        {"graph_uid", 2002},
        {"json_version", "1.0"},
        {"cudnn_backend_version", "cpu-test"},
        {"cudnn_frontend_version", 12400},
        {"nodes", Json::array({transpose, slice, concatenate})},
        {"tensors",
         Json::object(
             {{"11", tensor("TX", 11, {2, 3}, {4, 1}, false)},
              {"12", tensor("TT", 12, {3, 2}, {2, 1}, true)},
              {"13", tensor("TS", 13, {2, 2}, {2, 1}, true)},
              {"14", tensor("TC", 14, {2, 1}, {2, 1}, false)},
              {"15", tensor("TY", 15, {2, 3}, {4, 1}, false)}})}};
}

deepforge::import::Status parse_graph(
    Json const& document,
    deepforge::import::SerializedGraph& graph) {
    auto text = document.dump();
    std::vector<std::uint8_t> bytes(text.begin(), text.end());
    deepforge::import::SerializedGraphImporter importer;
    return importer.parse(std::span<std::uint8_t const>(bytes),
                          deepforge::import::InputFormat::kJson, graph);
}

bool output_matches(std::vector<float> const& output) {
    constexpr std::array<std::size_t, 6> offsets{0, 1, 3, 4, 6, 7};
    for (std::size_t index = 0; index < offsets.size(); ++index) {
        if (std::fabs(output[offsets[index]] -
                      static_cast<float>(index + 1)) > 1.0e-6F) {
            return false;
        }
    }
    return output[2] == -99.0F && output[5] == -99.0F;
}

void initialize_input(std::vector<float>& input) {
    std::fill(input.begin(), input.end(), -77.0F);
    constexpr std::array<std::size_t, 6> offsets{0, 1, 2, 4, 5, 6};
    for (std::size_t index = 0; index < offsets.size(); ++index) {
        input[offsets[index]] = static_cast<float>(index + 1);
    }
}

bool transform_output_matches(std::vector<float> const& output) {
    constexpr std::array<float, 6> expected{2.0F, 5.0F, 10.0F,
                                             3.0F, 6.0F, 20.0F};
    constexpr std::array<std::size_t, 6> offsets{0, 1, 2, 4, 5, 6};
    for (std::size_t index = 0; index < offsets.size(); ++index) {
        if (std::fabs(output[offsets[index]] - expected[index]) > 1.0e-6F) {
            return false;
        }
    }
    return output[3] == -99.0F;
}

}  // namespace

int main() {
    TestRunner tests;
    deepforge::import::SerializedGraph graph;
    auto status = parse_graph(reshape_graph(), graph);
    tests.good(status, "parse two-node reshape graph");
    if (status.is_bad()) {
        return tests.finish();
    }

    deepforge::compiler::CompileOptions options;
    options.capture_mlir = true;
    deepforge::compiler::CompilationResult compilation;
    status = deepforge::compiler::compile_graph(graph, options, compilation);
    tests.good(status, "compile two-node reshape graph");
    if (status.is_bad() || !compilation.executable) {
        return tests.finish();
    }
    tests.check(
        compilation.adapter_kind ==
                deepforge::compiler::InvocationAdapterKind::
                    kGenericRankedMemrefPointerTable &&
            compilation.metadata.arguments.size() == 2 &&
            compilation.metadata.arguments[0].uid == 1 &&
            compilation.metadata.arguments[1].uid == 3 &&
            compilation.workspace.allocations.size() == 1 &&
            compilation.workspace.size_bytes == 64,
        "generic metadata and virtual workspace are planned");
    tests.check(compilation.imported_mlir.find("memref.view") !=
                        std::string::npos &&
                    compilation.imported_mlir.find("scf.for") !=
                        std::string::npos &&
                    compilation.variants[0].llvm_ir.find(
                        "deepforge_graph_scalar") != std::string::npos,
                "standard MLIR and pointer-table adapter are emitted");

    std::vector<float> input(7);
    std::vector<float> output(8, -99.0F);
    initialize_input(input);
    AlignedBytes workspace(
        static_cast<std::size_t>(compilation.executable->get_workspace_size()));
    tests.check(workspace.pointer != nullptr, "workspace allocation");
    deepforge::runtime::VariantPack pack{{1, input.data()}, {3, output.data()}};
    status = compilation.executable->execute_variant(
        deepforge::runtime::CpuVariant::kScalar, nullptr, pack,
        workspace.pointer);
    tests.good(status, "execute scalar reshape chain");
    tests.check(output_matches(output),
                "reshape chain honors non-contiguous external strides");

    std::vector<std::uint8_t> artifact;
    status = deepforge::compiler::serialize_artifact(compilation, artifact);
    tests.good(status, "serialize generic artifact");
    deepforge::compiler::ArtifactInfo info;
    status = deepforge::compiler::parse_artifact(artifact, info);
    tests.good(status, "parse generic artifact");
    tests.check(
        info.adapter_kind ==
                deepforge::compiler::ArtifactAdapterKind::
                    kGenericRankedMemrefPointerTable &&
            info.metadata.arguments == compilation.metadata.arguments &&
            info.workspace == compilation.workspace,
        "generic artifact preserves metadata and workspace");

    std::unique_ptr<deepforge::runtime::Executable> loaded;
    status = deepforge::compiler::load_artifact_executable(artifact, loaded);
    tests.good(status, "load generic artifact executable");
    std::fill(output.begin(), output.end(), -99.0F);
    if (loaded) {
        status = loaded->execute(nullptr, pack, workspace.pointer);
        tests.good(status, "execute loaded generic artifact");
        tests.check(output_matches(output),
                    "loaded artifact output matches reshape reference");
    }

    auto missing = pack;
    missing.erase(1);
    status = compilation.executable->execute_variant(
        deepforge::runtime::CpuVariant::kScalar, nullptr, missing,
        workspace.pointer);
    tests.check(status.code() ==
                    deepforge::import::ErrorCode::kInvalidVariantPack,
                "generic runtime rejects a missing UID");
    status = compilation.executable->execute_variant(
        deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr);
    tests.check(status.code() ==
                    deepforge::import::ErrorCode::kInvalidVariantPack,
                "generic runtime rejects a missing workspace");
    auto aliased = pack;
    aliased[3] = input.data();
    status = compilation.executable->execute_variant(
        deepforge::runtime::CpuVariant::kScalar, nullptr, aliased,
        workspace.pointer);
    tests.check(status.code() ==
                    deepforge::import::ErrorCode::kInvalidVariantPack,
                "generic runtime rejects read/write aliasing");

    deepforge::import::SerializedGraph transforms;
    status = parse_graph(transform_graph(), transforms);
    tests.good(status, "parse transpose-slice-concatenate graph");
    deepforge::compiler::CompilationResult transform_compilation;
    status = deepforge::compiler::compile_graph(
        transforms, options, transform_compilation);
    tests.good(status, "compile transpose-slice-concatenate graph");
    if (status.is_good() && transform_compilation.executable) {
        std::vector<float> transform_input(7, -77.0F);
        initialize_input(transform_input);
        std::vector<float> concatenate_input{10.0F, -88.0F, 20.0F};
        std::vector<float> transform_output(7, -99.0F);
        AlignedBytes transform_workspace(static_cast<std::size_t>(
            transform_compilation.executable->get_workspace_size()));
        deepforge::runtime::VariantPack transform_pack{
            {11, transform_input.data()},
            {14, concatenate_input.data()},
            {15, transform_output.data()}};
        status = transform_compilation.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr, transform_pack,
            transform_workspace.pointer);
        tests.good(status, "execute transpose-slice-concatenate graph");
        tests.check(transform_output_matches(transform_output),
                    "transform chain matches scalar reference");
        tests.check(transform_compilation.workspace.allocations.size() == 2 &&
                        transform_compilation.workspace.size_bytes == 64,
                    "transform virtual tensors use planned workspace");
    }

    auto invalid_transpose = transform_graph();
    invalid_transpose["nodes"][0]["permutation"] = Json::array({0, 0});
    deepforge::import::SerializedGraph invalid_transpose_graph;
    status = parse_graph(invalid_transpose, invalid_transpose_graph);
    tests.good(status, "structurally valid duplicate permutation parses");
    deepforge::compiler::CompilationResult invalid_transform_result;
    status = deepforge::compiler::compile_graph(
        invalid_transpose_graph, options, invalid_transform_result);
    tests.check(status.code() == deepforge::import::ErrorCode::kInvalidShape,
                "duplicate transpose axis is rejected before codegen");

    auto invalid_document = reshape_graph();
    invalid_document["nodes"][1]["dim"] = Json::array({2, 3});
    deepforge::import::SerializedGraph invalid_graph;
    status = parse_graph(invalid_document, invalid_graph);
    tests.good(status, "structurally valid mismatched reshape parses");
    deepforge::compiler::CompilationResult rejected;
    status = deepforge::compiler::compile_graph(invalid_graph, options, rejected);
    tests.check(status.code() == deepforge::import::ErrorCode::kInvalidShape,
                "reshape descriptor mismatch is rejected before codegen");

    return tests.finish();
}
