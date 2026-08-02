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
                      {"slice_strides", Json::array({1})}};
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

Json dynamic_pointwise_graph() {
    Json node{{"tag", "POINTWISE"},
              {"name", "dynamic_add"},
              {"inputs", Json::object({{"IN_0", 101}, {"IN_1", 102}})},
              {"outputs", Json::object({{"OUT_0", 103}})},
              {"compute_data_type", "FLOAT"},
              {"mode", "ADD"},
              {"axis", nullptr},
              {"relu_lower_clip", nullptr},
              {"relu_upper_clip", nullptr},
              {"relu_lower_clip_slope", nullptr},
              {"swish_beta", nullptr},
              {"elu_alpha", nullptr},
              {"softplus_beta", nullptr}};
    auto document = Json{
        {"context",
         Json{{"name", "dynamic_pointwise"},
              {"compute_data_type", "FLOAT"},
              {"intermediate_data_type", "FLOAT"},
              {"io_data_type", "FLOAT"},
              {"sm_count", -1},
              {"is_dynamic_shape_enabled", true},
              {"is_override_shape_enabled", true}}},
        {"graph_uid", 2003},
        {"json_version", "1.0"},
        {"cudnn_backend_version", "cpu-test"},
        {"cudnn_frontend_version", 12400},
        {"nodes", Json::array({node})},
        {"tensors",
         Json::object(
             {{"101", tensor("A", 101, {4, 5}, {5, 1}, false)},
              {"102", tensor("B", 102, {4, 5}, {5, 1}, false)},
              {"103", tensor("Y", 103, {4, 5}, {5, 1}, false)}})}};
    return document;
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

deepforge::import::Status compile_document(Json const& document) {
    deepforge::import::SerializedGraph graph;
    auto status = parse_graph(document, graph);
    if (status.is_bad()) {
        return status;
    }
    deepforge::compiler::CompilationResult compilation;
    return deepforge::compiler::compile_graph(
        graph, deepforge::compiler::CompileOptions{}, compilation);
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
    status = compilation.executable->execute_variant(
        deepforge::runtime::CpuVariant::kScalar, nullptr, pack,
        workspace.pointer, {1}, {{2, 3}}, {{4, 1}});
    tests.check(
        status.code() ==
            deepforge::import::ErrorCode::kUnsupportedExecutionMetadata,
        "static artifact rejects runtime shape overrides");
    std::int64_t static_workspace_size = -7;
    status = compilation.executable->get_workspace_size(
        nullptr, static_workspace_size, {1}, {{2, 3}}, {{4, 1}});
    tests.check(
        status.code() ==
                deepforge::import::ErrorCode::kUnsupportedExecutionMetadata &&
            static_workspace_size == -7,
        "static workspace query rejects overrides without changing output");

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

    deepforge::import::SerializedGraph dynamic_graph;
    status = parse_graph(dynamic_pointwise_graph(), dynamic_graph);
    tests.good(status, "parse dynamic pointwise graph");
    deepforge::compiler::CompilationResult dynamic_compilation;
    status = deepforge::compiler::compile_graph(dynamic_graph, options,
                                                dynamic_compilation);
    tests.good(status, "compile dynamic pointwise graph");
    if (status.is_good() && dynamic_compilation.executable) {
        tests.check(
            dynamic_compilation.metadata.dynamic_shape_enabled &&
                dynamic_compilation.metadata.override_shape_enabled &&
                dynamic_compilation.metadata.override_policy ==
                    deepforge::compiler::ShapeOverridePolicy::kPointwiseExact &&
                dynamic_compilation.workspace.size_bytes == 0 &&
                dynamic_compilation.imported_mlir.find("memref.dim") !=
                    std::string::npos &&
                dynamic_compilation.imported_mlir.find("?x?xf32") !=
                    std::string::npos,
            "dynamic pointwise emits runtime dimensions and records policy");

        std::vector<float> a(20, -77.0F);
        std::vector<float> b(20, -88.0F);
        std::vector<float> y(20, -99.0F);
        for (std::size_t row = 0; row < 2; ++row) {
            for (std::size_t column = 0; column < 3; ++column) {
                auto const offset = row * 4 + column;
                a[offset] = static_cast<float>(10 * row + column);
                b[offset] = static_cast<float>(100 + 10 * row + column);
            }
        }
        deepforge::runtime::VariantPack dynamic_pack{
            {101, a.data()}, {102, b.data()}, {103, y.data()}};
        deepforge::runtime::OverrideUids override_uids{101, 102, 103};
        deepforge::runtime::OverrideShapes override_shapes(
            3, std::vector<std::int64_t>{2, 3});
        deepforge::runtime::OverrideStrides override_strides(
            3, std::vector<std::int64_t>{4, 1});
        std::int64_t dynamic_workspace_size = -1;
        status = dynamic_compilation.executable->get_workspace_size(
            nullptr, dynamic_workspace_size, override_uids, override_shapes,
            override_strides);
        tests.good(status, "query workspace with Frontend shape overrides");
        tests.check(dynamic_workspace_size == 0,
                    "dynamic pointwise workspace remains statically bounded");
        tests.check(dynamic_compilation.executable->get_workspace_size(
                        nullptr, override_uids, override_shapes,
                        override_strides) == 0,
                    "Frontend workspace convenience overload accepts overrides");
        status = dynamic_compilation.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr, dynamic_pack,
            nullptr, override_uids, override_shapes, override_strides);
        tests.good(status, "execute pointwise with Frontend shape overrides");
        bool dynamic_matches = true;
        for (std::size_t row = 0; row < 2; ++row) {
            for (std::size_t column = 0; column < 3; ++column) {
                auto const offset = row * 4 + column;
                dynamic_matches =
                    dynamic_matches && y[offset] == a[offset] + b[offset];
            }
        }
        dynamic_matches = dynamic_matches && y[3] == -99.0F &&
                          y[7] == -99.0F && y[8] == -99.0F;
        tests.check(dynamic_matches,
                    "dynamic loop bounds and overridden strides are honored");

        std::vector<std::uint8_t> dynamic_artifact;
        status = deepforge::compiler::serialize_artifact(dynamic_compilation,
                                                         dynamic_artifact);
        tests.good(status, "serialize dynamic artifact");
        deepforge::compiler::ArtifactInfo dynamic_info;
        status = deepforge::compiler::parse_artifact(dynamic_artifact,
                                                     dynamic_info);
        tests.good(status, "parse dynamic artifact");
        tests.check(dynamic_info.metadata.dynamic_shape_enabled &&
                        dynamic_info.metadata.override_shape_enabled &&
                        dynamic_info.metadata.override_policy ==
                            deepforge::compiler::ShapeOverridePolicy::
                                kPointwiseExact,
                    "artifact v4 preserves dynamic override metadata");
        std::unique_ptr<deepforge::runtime::Executable> dynamic_loaded;
        status = deepforge::compiler::load_artifact_executable(
            dynamic_artifact, dynamic_loaded);
        tests.good(status, "load dynamic artifact executable");
        if (dynamic_loaded) {
            std::fill(y.begin(), y.end(), -99.0F);
            status = dynamic_loaded->execute(
                nullptr, dynamic_pack, nullptr, override_uids,
                override_shapes, override_strides);
            tests.good(status, "execute loaded dynamic artifact");
            tests.check(y[0] == a[0] + b[0] && y[6] == a[6] + b[6] &&
                            y[7] == -99.0F,
                        "loaded dynamic artifact uses runtime descriptors");
        }

        for (std::size_t index = 0; index < a.size(); ++index) {
            a[index] = static_cast<float>(index);
            b[index] = static_cast<float>(100 + index);
        }
        std::fill(y.begin(), y.end(), -99.0F);
        status = dynamic_compilation.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr, dynamic_pack,
            nullptr);
        tests.good(status, "execute dynamic artifact without override arrays");
        bool maximum_shape_matches = true;
        for (std::size_t index = 0; index < y.size(); ++index) {
            maximum_shape_matches =
                maximum_shape_matches && y[index] == a[index] + b[index];
        }
        tests.check(maximum_shape_matches,
                    "empty override arrays use the compiled maximum shape");

        std::vector<float> runtime_spans(21, -99.0F);
        for (std::size_t row = 0; row < 2; ++row) {
            for (std::size_t column = 0; column < 3; ++column) {
                auto const offset = row * 4 + column;
                runtime_spans[offset] = static_cast<float>(offset);
                runtime_spans[7 + offset] =
                    static_cast<float>(100 + offset);
            }
        }
        deepforge::runtime::VariantPack runtime_span_pack{
            {101, runtime_spans.data()},
            {102, runtime_spans.data() + 7},
            {103, runtime_spans.data() + 14}};
        status = dynamic_compilation.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr,
            runtime_span_pack, nullptr, override_uids, override_shapes,
            override_strides);
        tests.good(status, "execute with adjacent runtime storage spans");
        bool runtime_spans_match = true;
        for (std::size_t row = 0; row < 2; ++row) {
            for (std::size_t column = 0; column < 3; ++column) {
                auto const offset = row * 4 + column;
                runtime_spans_match =
                    runtime_spans_match &&
                    runtime_spans[14 + offset] ==
                        runtime_spans[offset] + runtime_spans[7 + offset];
            }
        }
        tests.check(runtime_spans_match,
                    "alias validation uses resolved runtime byte spans");

        status = dynamic_compilation.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr, dynamic_pack,
            nullptr, override_uids,
            deepforge::runtime::OverrideShapes(2, {2, 3}), override_strides);
        tests.check(status.code() == deepforge::import::ErrorCode::kInvalidValue,
                    "override arrays require equal counts");
        dynamic_workspace_size = 37;
        status = dynamic_compilation.executable->get_workspace_size(
            nullptr, dynamic_workspace_size, override_uids,
            deepforge::runtime::OverrideShapes(2, {2, 3}), override_strides);
        tests.check(
            status.code() == deepforge::import::ErrorCode::kInvalidValue &&
                dynamic_workspace_size == 37,
            "failed override workspace query leaves output unchanged");
        status = dynamic_compilation.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr, dynamic_pack,
            nullptr, {101, 101}, {{2, 3}, {2, 3}}, {{4, 1}, {4, 1}});
        tests.check(status.code() == deepforge::import::ErrorCode::kInvalidValue,
                    "duplicate override UIDs are rejected");
        status = dynamic_compilation.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr, dynamic_pack,
            nullptr, {999}, {{2, 3}}, {{4, 1}});
        tests.check(
            status.code() == deepforge::import::ErrorCode::kInvalidVariantPack,
            "unknown override UIDs are rejected");
        status = dynamic_compilation.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr, dynamic_pack,
            nullptr, {101}, {{2}}, {{1}});
        tests.check(status.code() == deepforge::import::ErrorCode::kInvalidShape,
                    "override rank must match the compiled tensor rank");
        status = dynamic_compilation.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr, dynamic_pack,
            nullptr, {103}, {{2, 3}}, {{4, 1}});
        tests.check(status.code() == deepforge::import::ErrorCode::kInvalidShape,
                    "partial shape changes reject inconsistent pointwise tensors");
        status = dynamic_compilation.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr, dynamic_pack,
            nullptr, override_uids,
            deepforge::runtime::OverrideShapes(3, {5, 3}),
            deepforge::runtime::OverrideStrides(3, {3, 1}));
        tests.check(status.code() == deepforge::import::ErrorCode::kInvalidShape,
                    "override dimensions cannot exceed compiled maxima");
        status = dynamic_compilation.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr, dynamic_pack,
            nullptr, override_uids, override_shapes,
            deepforge::runtime::OverrideStrides(3, {1, 1}));
        tests.check(status.code() == deepforge::import::ErrorCode::kInvalidLayout,
                    "overlapping override strides are rejected");
        status = dynamic_compilation.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr, dynamic_pack,
            nullptr, override_uids, override_shapes,
            deepforge::runtime::OverrideStrides(3, {19, 1}));
        tests.check(status.code() == deepforge::import::ErrorCode::kInvalidShape,
                    "override storage cannot exceed the compiled byte bound");
    }

    auto broadcast_override = dynamic_pointwise_graph();
    broadcast_override["tensors"]["102"]["dim"] = Json::array({1, 5});
    broadcast_override["tensors"]["102"]["stride"] = Json::array({5, 1});
    status = compile_document(broadcast_override);
    tests.check(status.code() ==
                    deepforge::import::ErrorCode::kUnsupportedOperation,
                "dynamic pointwise broadcasting is rejected explicitly");

    auto dynamic_only_document = reshape_graph();
    dynamic_only_document["context"]["is_dynamic_shape_enabled"] = true;
    deepforge::import::SerializedGraph dynamic_only_graph;
    status = parse_graph(dynamic_only_document, dynamic_only_graph);
    tests.good(status, "parse dynamic-plan metadata without overrides");
    deepforge::compiler::CompilationResult dynamic_only_compilation;
    status = deepforge::compiler::compile_graph(
        dynamic_only_graph, options, dynamic_only_compilation);
    tests.good(status, "compile dynamic-plan metadata without overrides");
    tests.check(status.is_good() &&
                    dynamic_only_compilation.metadata.dynamic_shape_enabled &&
                    !dynamic_only_compilation.metadata.override_shape_enabled,
                "dynamic-plan flag does not change static descriptor semantics");

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

    auto view_only = reshape_graph();
    view_only["nodes"][0]["reshape_mode"] = "VIEW_ONLY";
    status = compile_document(view_only);
    tests.check(
        status.code() == deepforge::import::ErrorCode::kUnsupportedOperation,
        "VIEW_ONLY reshape is rejected until alias semantics are supported");

    auto in_place_concatenate = transform_graph();
    in_place_concatenate["nodes"][2]["in_place_index"] = 0;
    status = compile_document(in_place_concatenate);
    tests.check(
        status.code() == deepforge::import::ErrorCode::kUnsupportedOperation,
        "in-place concatenate is rejected until alias semantics are supported");

    auto reused_uid = reshape_graph();
    reused_uid["nodes"] = Json::array({reused_uid["nodes"][0]});
    reused_uid["nodes"][0]["outputs"]["Y"] = 1;
    reused_uid["nodes"][0]["dim"] = Json::array({2, 3});
    reused_uid["nodes"][0]["stride"] = Json::array({4, 1});
    reused_uid["tensors"].erase("2");
    reused_uid["tensors"].erase("3");
    status = compile_document(reused_uid);
    tests.check(
        status.code() == deepforge::import::ErrorCode::kUnsupportedOperation,
        "same-UID input and output are rejected before codegen");

    auto overlapping_layout = reshape_graph();
    overlapping_layout["tensors"]["3"]["stride"] = Json::array({1, 1});
    overlapping_layout["nodes"][1]["stride"] = Json::array({1, 1});
    status = compile_document(overlapping_layout);
    tests.check(status.code() == deepforge::import::ErrorCode::kInvalidLayout,
                "overlapping logical tensor layouts are rejected");

    auto empty_slice_strides = transform_graph();
    empty_slice_strides["nodes"][1]["slice_strides"] = Json::array();
    status = compile_document(empty_slice_strides);
    tests.check(status.is_good(),
                "empty slice strides use the Frontend unit-stride default");

    return tests.finish();
}
