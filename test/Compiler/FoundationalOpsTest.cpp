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

Json transpose_node(std::string name,
                    std::int64_t x,
                    std::int64_t y,
                    std::vector<std::int64_t> permutation) {
    return Json{{"tag", "TRANSPOSE"},
                {"name", std::move(name)},
                {"inputs", Json::object({{"X", x}})},
                {"outputs", Json::object({{"Y", y}})},
                {"compute_data_type", "FLOAT"},
                {"permutation", std::move(permutation)}};
}

Json concatenate_node(std::string name,
                      std::vector<std::int64_t> inputs,
                      std::int64_t y,
                      std::int64_t axis,
                      Json in_place_index = nullptr) {
    Json indexed_inputs = Json::object();
    for (std::size_t index = 0; index < inputs.size(); ++index) {
        indexed_inputs[std::to_string(index)] = inputs[index];
    }
    return Json{{"tag", "CONCATENATE"},
                {"name", std::move(name)},
                {"inputs", std::move(indexed_inputs)},
                {"outputs", Json::object({{"Y", y}})},
                {"axis", axis},
                {"in_place_index", std::move(in_place_index)}};
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

Json dynamic_reshape_graph() {
    return Json{
        {"context",
         Json{{"name", "dynamic_reshape"},
              {"compute_data_type", "FLOAT"},
              {"intermediate_data_type", "FLOAT"},
              {"io_data_type", "FLOAT"},
              {"sm_count", -1},
              {"is_dynamic_shape_enabled", false},
              {"is_override_shape_enabled", true}}},
        {"graph_uid", 2005},
        {"json_version", "1.0"},
        {"cudnn_backend_version", "cpu-test"},
        {"cudnn_frontend_version", 12400},
        {"nodes",
         Json::array({reshape_node("dynamic_reshape", 301, 302, {4, 6},
                                   {8, 1})})},
        {"tensors",
         Json::object(
             {{"301", tensor("X", 301, {2, 3, 4}, {20, 6, 1}, false)},
              {"302", tensor("Y", 302, {4, 6}, {8, 1}, false)}})}};
}

Json dynamic_transpose_graph() {
    return Json{
        {"context",
         Json{{"name", "dynamic_transpose"},
              {"compute_data_type", "FLOAT"},
              {"intermediate_data_type", "FLOAT"},
              {"io_data_type", "FLOAT"},
              {"sm_count", -1},
              {"is_dynamic_shape_enabled", false},
              {"is_override_shape_enabled", true}}},
        {"graph_uid", 2006},
        {"json_version", "1.0"},
        {"cudnn_backend_version", "cpu-test"},
        {"cudnn_frontend_version", 12400},
        {"nodes",
         Json::array(
             {transpose_node("dynamic_transpose", 311, 312, {2, 0, 1})})},
        {"tensors",
         Json::object(
             {{"311", tensor("X", 311, {2, 3, 4}, {20, 6, 1}, false)},
              {"312", tensor("Y", 312, {4, 2, 3}, {9, 4, 1}, false)}})}};
}

Json dynamic_concatenate_graph() {
    return Json{
        {"context",
         Json{{"name", "dynamic_concatenate"},
              {"compute_data_type", "FLOAT"},
              {"intermediate_data_type", "FLOAT"},
              {"io_data_type", "FLOAT"},
              {"sm_count", -1},
              {"is_dynamic_shape_enabled", false},
              {"is_override_shape_enabled", true}}},
        {"graph_uid", 2007},
        {"json_version", "1.0"},
        {"cudnn_backend_version", "cpu-test"},
        {"cudnn_frontend_version", 12400},
        {"nodes",
         Json::array({concatenate_node("dynamic_concatenate",
                                       {321, 322, 323}, 324, 1)})},
        {"tensors",
         Json::object(
             {{"321", tensor("X0", 321, {2, 2, 3}, {12, 4, 1}, false)},
              {"322", tensor("X1", 322, {2, 3, 3}, {15, 4, 1}, false)},
              {"323", tensor("X2", 323, {2, 1, 3}, {8, 4, 1}, false)},
              {"324", tensor("Y", 324, {2, 6, 3}, {25, 4, 1}, false)}})}};
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

Json dynamic_pointwise_chain_graph() {
    auto pointwise = [](std::string name, std::string mode, Json inputs,
                        std::int64_t output) {
        return Json{{"tag", "POINTWISE"},
                    {"name", std::move(name)},
                    {"inputs", std::move(inputs)},
                    {"outputs", Json::object({{"OUT_0", output}})},
                    {"compute_data_type", "FLOAT"},
                    {"mode", std::move(mode)},
                    {"axis", nullptr},
                    {"relu_lower_clip", 0.0},
                    {"relu_upper_clip", 1000.0},
                    {"relu_lower_clip_slope", 0.0},
                    {"swish_beta", nullptr},
                    {"elu_alpha", nullptr},
                    {"softplus_beta", nullptr}};
    };
    auto add = pointwise(
        "dynamic_add_to_virtual", "ADD",
        Json::object({{"IN_0", 201}, {"IN_1", 202}}), 204);
    auto relu = pointwise("dynamic_relu_to_virtual", "RELU_FWD",
                          Json::object({{"IN_0", 204}}), 205);
    auto multiply = pointwise(
        "dynamic_multiply_to_output", "MUL",
        Json::object({{"IN_0", 205}, {"IN_1", 203}}), 206);
    return Json{
        {"context",
         Json{{"name", "dynamic_pointwise_chain"},
              {"compute_data_type", "FLOAT"},
              {"intermediate_data_type", "FLOAT"},
              {"io_data_type", "FLOAT"},
              {"sm_count", -1},
              {"is_dynamic_shape_enabled", true},
              {"is_override_shape_enabled", true}}},
        {"graph_uid", 2004},
        {"json_version", "1.0"},
        {"cudnn_backend_version", "cpu-test"},
        {"cudnn_frontend_version", 12400},
        {"nodes", Json::array({add, relu, multiply})},
        {"tensors",
         Json::object(
             {{"201", tensor("A", 201, {4, 5}, {5, 1}, false)},
              {"202", tensor("B", 202, {4, 5}, {5, 1}, false)},
              {"203", tensor("C", 203, {4, 5}, {5, 1}, false)},
              {"204", tensor("Add", 204, {4, 5}, {5, 1}, true)},
              {"205", tensor("Relu", 205, {4, 5}, {5, 1}, true)},
              {"206", tensor("Y", 206, {4, 5}, {5, 1}, false)}})}};
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

void run_dynamic_reshape_tests(TestRunner& tests) {
    deepforge::import::SerializedGraph graph;
    auto status = parse_graph(dynamic_reshape_graph(), graph);
    tests.good(status, "parse runtime shape-override RESHAPE");
    deepforge::compiler::CompileOptions options;
    options.capture_mlir = true;
    deepforge::compiler::CompilationResult compilation;
    if (status.is_good()) {
        status = deepforge::compiler::compile_graph(graph, options, compilation);
    }
    tests.good(status, "compile runtime shape-override RESHAPE");
    if (status.is_bad() || !compilation.executable) return;

    tests.check(
        !compilation.metadata.dynamic_shape_enabled &&
            compilation.metadata.override_shape_enabled &&
            compilation.metadata.override_policy ==
                deepforge::compiler::ShapeOverridePolicy::kReshape &&
            compilation.metadata.override_role_uids ==
                std::vector<std::int64_t>({301, 302}) &&
            compilation.workspace.size_bytes == 0 &&
            compilation.imported_mlir.find("memref.dim") !=
                std::string::npos &&
            compilation.imported_mlir.find("?x?x?xf32") !=
                std::string::npos &&
            compilation.imported_mlir.find("?x?xf32") != std::string::npos,
        "RESHAPE override records X/Y roles and emits runtime descriptors");

    std::vector<float> x(36, -77.0F);
    std::vector<float> y(30, -99.0F);
    for (std::int64_t row = 0; row < 3; ++row) {
        for (std::int64_t column = 0; column < 4; ++column) {
            x[static_cast<std::size_t>(row * 6 + column)] =
                static_cast<float>(row * 4 + column + 1);
        }
    }
    deepforge::runtime::VariantPack pack{{301, x.data()}, {302, y.data()}};
    deepforge::runtime::OverrideUids const override_uids{301, 302};
    deepforge::runtime::OverrideShapes const override_shapes{{1, 3, 4},
                                                              {3, 4}};
    deepforge::runtime::OverrideStrides const override_strides{{20, 6, 1},
                                                                {7, 1}};
    status = compilation.executable->execute_variant(
        deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr,
        override_uids, override_shapes, override_strides);
    tests.good(status, "execute runtime shape-override RESHAPE");
    auto runtime_output_matches = [&]() {
        std::vector<bool> occupied(y.size(), false);
        bool matches = true;
        for (std::int64_t row = 0; row < 3; ++row) {
            for (std::int64_t column = 0; column < 4; ++column) {
                auto const offset =
                    static_cast<std::size_t>(row * 7 + column);
                occupied[offset] = true;
                matches = matches &&
                          y[offset] ==
                              static_cast<float>(row * 4 + column + 1);
            }
        }
        for (std::size_t index = 0; index < y.size(); ++index) {
            matches = matches && (occupied[index] || y[index] == -99.0F);
        }
        return matches;
    };
    tests.check(runtime_output_matches(),
                "dynamic RESHAPE preserves lexicographic order across ranks and strides");

    std::vector<std::uint8_t> artifact;
    status = deepforge::compiler::serialize_artifact(compilation, artifact);
    tests.good(status, "serialize runtime shape-override RESHAPE artifact");
    deepforge::compiler::ArtifactInfo artifact_info;
    std::unique_ptr<deepforge::runtime::Executable> loaded;
    if (status.is_good()) {
        status = deepforge::compiler::load_artifact_executable(
            artifact, loaded, &artifact_info);
    }
    tests.good(status, "load runtime shape-override RESHAPE artifact");
    tests.check(
        artifact_info.format_version ==
                deepforge::compiler::kArtifactFormatVersion &&
            artifact_info.metadata.override_policy ==
                deepforge::compiler::ShapeOverridePolicy::kReshape &&
            artifact_info.metadata.override_role_uids == override_uids,
        "artifact v12 preserves ordered RESHAPE override roles");
    if (loaded) {
        std::fill(y.begin(), y.end(), -99.0F);
        status = loaded->execute(nullptr, pack, nullptr, override_uids,
                                 override_shapes, override_strides);
        tests.good(status, "execute loaded shape-override RESHAPE artifact");
        tests.check(runtime_output_matches(),
                    "loaded RESHAPE artifact uses runtime dimensions and strides");
    }

    std::fill(x.begin(), x.end(), -77.0F);
    std::fill(y.begin(), y.end(), -99.0F);
    for (std::int64_t first = 0; first < 2; ++first) {
        for (std::int64_t second = 0; second < 3; ++second) {
            for (std::int64_t third = 0; third < 4; ++third) {
                auto const linear = (first * 3 + second) * 4 + third;
                x[static_cast<std::size_t>(first * 20 + second * 6 + third)] =
                    static_cast<float>(linear + 1);
            }
        }
    }
    status = compilation.executable->execute(nullptr, pack, nullptr);
    tests.good(status,
               "execute shape-override RESHAPE at compiled maximum descriptors");
    bool maximum_matches = true;
    for (std::int64_t row = 0; row < 4; ++row) {
        for (std::int64_t column = 0; column < 6; ++column) {
            maximum_matches =
                maximum_matches &&
                y[static_cast<std::size_t>(row * 8 + column)] ==
                    static_cast<float>(row * 6 + column + 1);
        }
    }
    tests.check(maximum_matches,
                "empty RESHAPE override arrays use compiled maximum descriptors");

    std::int64_t workspace_size = -1;
    status = compilation.executable->get_workspace_size(
        nullptr, workspace_size, {301}, {{2, 3, 4}}, {{18, 5, 1}});
    tests.good(status, "partial RESHAPE override may change only X strides");
    tests.check(workspace_size == 0,
                "partial RESHAPE override preserves the static workspace bound");
    status = compilation.executable->get_workspace_size(
        nullptr, workspace_size, {301}, {{1, 3, 4}}, {{20, 6, 1}});
    tests.check(status.code() == deepforge::import::ErrorCode::kInvalidShape,
                "partial RESHAPE shape override must preserve element count");
    status = compilation.executable->execute_variant(
        deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr,
        override_uids, {{1, 3, 4}, {2, 4}}, override_strides);
    tests.check(status.code() == deepforge::import::ErrorCode::kInvalidShape,
                "RESHAPE rejects unequal runtime element counts");
    status = compilation.executable->execute_variant(
        deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr, {301},
        {{3, 4}}, {{4, 1}});
    tests.check(status.code() == deepforge::import::ErrorCode::kInvalidShape,
                "RESHAPE override rank must match each compiled role");
    status = compilation.executable->execute_variant(
        deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr,
        override_uids, override_shapes, {{20, 6, 1}, {1, 1}});
    tests.check(status.code() == deepforge::import::ErrorCode::kInvalidLayout,
                "RESHAPE rejects overlapping runtime strides");
    status = compilation.executable->execute_variant(
        deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr,
        override_uids, override_shapes, {{20, 6, 1}, {14, 1}});
    tests.check(status.code() == deepforge::import::ErrorCode::kInvalidShape,
                "RESHAPE runtime span cannot exceed the compiled byte bound");

    auto virtual_role = dynamic_reshape_graph();
    virtual_role["tensors"]["302"]["is_virtual"] = true;
    status = compile_document(virtual_role);
    tests.check(status.code() ==
                    deepforge::import::ErrorCode::kUnsupportedOperation,
                "RESHAPE overrides reject virtual role tensors");
    auto view_only = dynamic_reshape_graph();
    view_only["nodes"][0]["reshape_mode"] = "VIEW_ONLY";
    status = compile_document(view_only);
    tests.check(status.code() ==
                    deepforge::import::ErrorCode::kUnsupportedOperation,
                "RESHAPE overrides reject VIEW_ONLY alias semantics");
    auto composed = dynamic_reshape_graph();
    composed["tensors"]["303"] =
        tensor("Z", 303, {4, 6}, {8, 1}, false);
    composed["nodes"].push_back(
        reshape_node("second_reshape", 302, 303, {4, 6}, {8, 1}));
    status = compile_document(composed);
    tests.check(status.code() ==
                    deepforge::import::ErrorCode::kUnsupportedOperation,
                "RESHAPE overrides reject composed graphs");
}

void run_dynamic_transpose_tests(TestRunner& tests) {
    deepforge::import::SerializedGraph graph;
    auto status = parse_graph(dynamic_transpose_graph(), graph);
    tests.good(status, "parse runtime shape-override TRANSPOSE");
    deepforge::compiler::CompileOptions options;
    options.capture_mlir = true;
    deepforge::compiler::CompilationResult compilation;
    if (status.is_good()) {
        status = deepforge::compiler::compile_graph(graph, options, compilation);
    }
    tests.good(status, "compile runtime shape-override TRANSPOSE");
    if (status.is_bad() || !compilation.executable) return;

    tests.check(
        !compilation.metadata.dynamic_shape_enabled &&
            compilation.metadata.override_shape_enabled &&
            compilation.metadata.override_policy ==
                deepforge::compiler::ShapeOverridePolicy::kTranspose &&
            compilation.metadata.override_role_uids ==
                std::vector<std::int64_t>({311, 312}) &&
            compilation.metadata.override_axis_map ==
                std::vector<std::int64_t>({2, 0, 1}) &&
            compilation.workspace.size_bytes == 0 &&
            compilation.imported_mlir.find("memref.dim") !=
                std::string::npos &&
            compilation.imported_mlir.find("?x?x?xf32") !=
                std::string::npos,
        "TRANSPOSE override records roles and permutation and emits runtime descriptors");

    std::vector<float> x(36, -77.0F);
    std::vector<float> y(34, -99.0F);
    for (std::int64_t b = 0; b < 2; ++b) {
        for (std::int64_t c = 0; c < 3; ++c) {
            x[static_cast<std::size_t>(b * 5 + c)] =
                static_cast<float>(10 * b + c);
        }
    }
    deepforge::runtime::VariantPack pack{{311, x.data()}, {312, y.data()}};
    deepforge::runtime::OverrideUids const override_uids{311, 312};
    deepforge::runtime::OverrideShapes const override_shapes{{1, 2, 3},
                                                              {3, 1, 2}};
    deepforge::runtime::OverrideStrides const override_strides{{20, 5, 1},
                                                                {7, 4, 1}};
    status = compilation.executable->execute_variant(
        deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr,
        override_uids, override_shapes, override_strides);
    tests.good(status, "execute runtime shape-override TRANSPOSE");
    auto runtime_output_matches = [&]() {
        std::vector<bool> occupied(y.size(), false);
        bool matches = true;
        for (std::int64_t c = 0; c < 3; ++c) {
            for (std::int64_t b = 0; b < 2; ++b) {
                auto const offset = static_cast<std::size_t>(c * 7 + b);
                occupied[offset] = true;
                matches = matches &&
                          y[offset] == static_cast<float>(10 * b + c);
            }
        }
        for (std::size_t index = 0; index < y.size(); ++index) {
            matches = matches && (occupied[index] || y[index] == -99.0F);
        }
        return matches;
    };
    tests.check(runtime_output_matches(),
                "dynamic TRANSPOSE honors permutation and non-contiguous strides");

    std::vector<std::uint8_t> artifact;
    status = deepforge::compiler::serialize_artifact(compilation, artifact);
    tests.good(status, "serialize runtime shape-override TRANSPOSE artifact");
    deepforge::compiler::ArtifactInfo artifact_info;
    std::unique_ptr<deepforge::runtime::Executable> loaded;
    if (status.is_good()) {
        status = deepforge::compiler::load_artifact_executable(
            artifact, loaded, &artifact_info);
    }
    tests.good(status, "load runtime shape-override TRANSPOSE artifact");
    tests.check(
        artifact_info.format_version ==
                deepforge::compiler::kArtifactFormatVersion &&
            artifact_info.metadata.override_policy ==
                deepforge::compiler::ShapeOverridePolicy::kTranspose &&
            artifact_info.metadata.override_role_uids == override_uids &&
            artifact_info.metadata.override_axis_map ==
                std::vector<std::int64_t>({2, 0, 1}),
        "artifact v12 preserves ordered TRANSPOSE roles and permutation");
    if (loaded) {
        std::fill(y.begin(), y.end(), -99.0F);
        status = loaded->execute(nullptr, pack, nullptr, override_uids,
                                 override_shapes, override_strides);
        tests.good(status, "execute loaded shape-override TRANSPOSE artifact");
        tests.check(runtime_output_matches(),
                    "loaded TRANSPOSE artifact uses runtime descriptors");
    }

    std::fill(x.begin(), x.end(), -77.0F);
    std::fill(y.begin(), y.end(), -99.0F);
    for (std::int64_t a = 0; a < 2; ++a) {
        for (std::int64_t b = 0; b < 3; ++b) {
            for (std::int64_t c = 0; c < 4; ++c) {
                x[static_cast<std::size_t>(a * 20 + b * 6 + c)] =
                    static_cast<float>(100 * a + 10 * b + c);
            }
        }
    }
    status = compilation.executable->execute(nullptr, pack, nullptr);
    tests.good(status,
               "execute shape-override TRANSPOSE at compiled maximum descriptors");
    bool maximum_matches = true;
    for (std::int64_t c = 0; c < 4; ++c) {
        for (std::int64_t a = 0; a < 2; ++a) {
            for (std::int64_t b = 0; b < 3; ++b) {
                maximum_matches =
                    maximum_matches &&
                    y[static_cast<std::size_t>(c * 9 + a * 4 + b)] ==
                        static_cast<float>(100 * a + 10 * b + c);
            }
        }
    }
    tests.check(maximum_matches,
                "empty TRANSPOSE override arrays use compiled maximum descriptors");

    std::int64_t workspace_size = -1;
    status = compilation.executable->get_workspace_size(
        nullptr, workspace_size, {311}, {{2, 3, 4}}, {{18, 5, 1}});
    tests.good(status,
               "partial TRANSPOSE override may change only X strides");
    tests.check(workspace_size == 0,
                "partial TRANSPOSE override preserves the static workspace bound");
    status = compilation.executable->get_workspace_size(
        nullptr, workspace_size, {311}, {{1, 3, 4}}, {{20, 6, 1}});
    tests.check(status.code() == deepforge::import::ErrorCode::kInvalidShape,
                "partial TRANSPOSE shapes must preserve the final permutation relation");
    status = compilation.executable->execute_variant(
        deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr,
        override_uids, {{1, 2, 3}, {3, 1, 1}}, override_strides);
    tests.check(status.code() == deepforge::import::ErrorCode::kInvalidShape,
                "TRANSPOSE rejects runtime dimensions that violate the permutation");
    status = compilation.executable->execute_variant(
        deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr, {311},
        {{2, 3}}, {{3, 1}});
    tests.check(status.code() == deepforge::import::ErrorCode::kInvalidShape,
                "TRANSPOSE override rank must match the compiled role");
    status = compilation.executable->execute_variant(
        deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr,
        override_uids, override_shapes, {{20, 5, 1}, {1, 1, 1}});
    tests.check(status.code() == deepforge::import::ErrorCode::kInvalidLayout,
                "TRANSPOSE rejects overlapping runtime strides");
    status = compilation.executable->execute_variant(
        deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr,
        override_uids, override_shapes, {{20, 5, 1}, {17, 4, 1}});
    tests.check(status.code() == deepforge::import::ErrorCode::kInvalidShape,
                "TRANSPOSE runtime span cannot exceed the compiled byte bound");

    auto virtual_role = dynamic_transpose_graph();
    virtual_role["tensors"]["312"]["is_virtual"] = true;
    status = compile_document(virtual_role);
    tests.check(status.code() ==
                    deepforge::import::ErrorCode::kUnsupportedOperation,
                "TRANSPOSE overrides reject virtual role tensors");
    auto pass_by_value = dynamic_transpose_graph();
    pass_by_value["tensors"]["311"]["is_pass_by_value"] = true;
    status = compile_document(pass_by_value);
    tests.check(status.code() ==
                    deepforge::import::ErrorCode::kUnsupportedOperation,
                "TRANSPOSE overrides reject pass-by-value role tensors");
    auto malformed_permutation = dynamic_transpose_graph();
    malformed_permutation["nodes"][0]["permutation"] =
        Json::array({2, 2, 0});
    status = compile_document(malformed_permutation);
    tests.check(status.code() == deepforge::import::ErrorCode::kInvalidShape,
                "TRANSPOSE overrides reject malformed permutations");
    auto composed = transform_graph();
    composed["context"]["is_override_shape_enabled"] = true;
    status = compile_document(composed);
    tests.check(status.code() ==
                    deepforge::import::ErrorCode::kUnsupportedOperation,
                "TRANSPOSE overrides reject composed graphs");
}

void run_dynamic_concatenate_tests(TestRunner& tests) {
    deepforge::import::SerializedGraph graph;
    auto status = parse_graph(dynamic_concatenate_graph(), graph);
    tests.good(status, "parse runtime shape-override CONCATENATE");
    deepforge::compiler::CompileOptions options;
    options.capture_mlir = true;
    deepforge::compiler::CompilationResult compilation;
    if (status.is_good()) {
        status = deepforge::compiler::compile_graph(graph, options, compilation);
    }
    tests.good(status, "compile runtime shape-override CONCATENATE");
    if (status.is_bad() || !compilation.executable) return;

    tests.check(
        !compilation.metadata.dynamic_shape_enabled &&
            compilation.metadata.override_shape_enabled &&
            compilation.metadata.override_policy ==
                deepforge::compiler::ShapeOverridePolicy::kConcatenate &&
            compilation.metadata.override_role_uids ==
                std::vector<std::int64_t>({321, 322, 323, 324}) &&
            compilation.metadata.override_axis_map ==
                std::vector<std::int64_t>({1}) &&
            compilation.workspace.size_bytes == 0 &&
            compilation.imported_mlir.find("memref.dim") !=
                std::string::npos &&
            compilation.imported_mlir.find("?x?x?xf32") !=
                std::string::npos,
        "CONCATENATE override records ordered roles and axis and emits runtime descriptors");

    std::vector<float> x0(19, -77.0F);
    std::vector<float> x1(26, -78.0F);
    std::vector<float> x2(11, -79.0F);
    std::vector<float> y(48, -99.0F);
    for (std::int64_t width = 0; width < 2; ++width) {
        x0[static_cast<std::size_t>(width)] =
            static_cast<float>(10 + width);
        x1[static_cast<std::size_t>(width)] =
            static_cast<float>(20 + width);
        x1[static_cast<std::size_t>(4 + width)] =
            static_cast<float>(30 + width);
        x2[static_cast<std::size_t>(width)] =
            static_cast<float>(40 + width);
    }
    deepforge::runtime::VariantPack pack{{321, x0.data()},
                                         {322, x1.data()},
                                         {323, x2.data()},
                                         {324, y.data()}};
    deepforge::runtime::OverrideUids const override_uids{321, 322, 323, 324};
    deepforge::runtime::OverrideShapes const override_shapes{
        {1, 1, 2}, {1, 2, 2}, {1, 1, 2}, {1, 4, 2}};
    deepforge::runtime::OverrideStrides const override_strides{
        {12, 4, 1}, {15, 4, 1}, {8, 4, 1}, {25, 4, 1}};
    status = compilation.executable->execute_variant(
        deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr,
        override_uids, override_shapes, override_strides);
    tests.good(status, "execute runtime shape-override CONCATENATE");
    auto runtime_output_matches = [&]() {
        std::vector<bool> occupied(y.size(), false);
        bool matches = true;
        for (std::int64_t axis = 0; axis < 4; ++axis) {
            for (std::int64_t width = 0; width < 2; ++width) {
                auto const offset =
                    static_cast<std::size_t>(axis * 4 + width);
                auto const expected =
                    static_cast<float>(10 * (axis + 1) + width);
                occupied[offset] = true;
                matches = matches && y[offset] == expected;
            }
        }
        for (std::size_t index = 0; index < y.size(); ++index) {
            matches = matches && (occupied[index] || y[index] == -99.0F);
        }
        return matches;
    };
    tests.check(runtime_output_matches(),
                "dynamic CONCATENATE honors runtime extents, offsets, and strides");

    std::vector<std::uint8_t> artifact;
    status = deepforge::compiler::serialize_artifact(compilation, artifact);
    tests.good(status,
               "serialize runtime shape-override CONCATENATE artifact");
    deepforge::compiler::ArtifactInfo artifact_info;
    std::unique_ptr<deepforge::runtime::Executable> loaded;
    if (status.is_good()) {
        status = deepforge::compiler::load_artifact_executable(
            artifact, loaded, &artifact_info);
    }
    tests.good(status, "load runtime shape-override CONCATENATE artifact");
    tests.check(
        artifact_info.format_version ==
                deepforge::compiler::kArtifactFormatVersion &&
            artifact_info.metadata.override_policy ==
                deepforge::compiler::ShapeOverridePolicy::kConcatenate &&
            artifact_info.metadata.override_role_uids == override_uids &&
            artifact_info.metadata.override_axis_map ==
                std::vector<std::int64_t>({1}),
        "artifact v12 preserves ordered CONCATENATE roles and axis");
    if (loaded) {
        std::fill(y.begin(), y.end(), -99.0F);
        status = loaded->execute(nullptr, pack, nullptr, override_uids,
                                 override_shapes, override_strides);
        tests.good(status,
                   "execute loaded shape-override CONCATENATE artifact");
        tests.check(runtime_output_matches(),
                    "loaded CONCATENATE artifact uses runtime descriptors");
    }

    std::fill(x0.begin(), x0.end(), -77.0F);
    std::fill(x1.begin(), x1.end(), -78.0F);
    std::fill(x2.begin(), x2.end(), -79.0F);
    std::fill(y.begin(), y.end(), -99.0F);
    auto fill_input = [](std::vector<float>& input, std::int64_t batch_stride,
                         std::int64_t axis_extent, float base) {
        for (std::int64_t batch = 0; batch < 2; ++batch) {
            for (std::int64_t axis = 0; axis < axis_extent; ++axis) {
                for (std::int64_t width = 0; width < 3; ++width) {
                    input[static_cast<std::size_t>(batch * batch_stride +
                                                   axis * 4 + width)] =
                        base + static_cast<float>(100 * batch + 10 * axis +
                                                  width);
                }
            }
        }
    };
    fill_input(x0, 12, 2, 0.0F);
    fill_input(x1, 15, 3, 1000.0F);
    fill_input(x2, 8, 1, 2000.0F);
    status = compilation.executable->execute(nullptr, pack, nullptr);
    tests.good(status,
               "execute shape-override CONCATENATE at compiled maximum descriptors");
    bool maximum_matches = true;
    for (std::int64_t batch = 0; batch < 2; ++batch) {
        for (std::int64_t axis = 0; axis < 6; ++axis) {
            for (std::int64_t width = 0; width < 3; ++width) {
                float expected = 0.0F;
                if (axis < 2) {
                    expected = static_cast<float>(100 * batch + 10 * axis +
                                                  width);
                } else if (axis < 5) {
                    expected = 1000.0F +
                               static_cast<float>(100 * batch +
                                                  10 * (axis - 2) + width);
                } else {
                    expected = 2000.0F +
                               static_cast<float>(100 * batch + width);
                }
                maximum_matches =
                    maximum_matches &&
                    y[static_cast<std::size_t>(batch * 25 + axis * 4 +
                                               width)] == expected;
            }
        }
    }
    tests.check(maximum_matches,
                "empty CONCATENATE override arrays use compiled maximum descriptors");

    std::int64_t workspace_size = -1;
    status = compilation.executable->get_workspace_size(
        nullptr, workspace_size, {321, 324}, {{2, 1, 3}, {2, 5, 3}},
        {{12, 4, 1}, {25, 4, 1}});
    tests.good(status,
               "partial CONCATENATE override may shrink one input and Y axis");
    tests.check(workspace_size == 0,
                "partial CONCATENATE override preserves the static workspace bound");
    status = compilation.executable->get_workspace_size(
        nullptr, workspace_size, {321}, {{2, 1, 3}}, {{12, 4, 1}});
    tests.check(status.code() == deepforge::import::ErrorCode::kInvalidShape,
                "partial CONCATENATE override must preserve the final axis sum");
    status = compilation.executable->execute_variant(
        deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr,
        override_uids, {{1, 1, 2}, {1, 2, 2}, {1, 1, 2}, {1, 3, 2}},
        override_strides);
    tests.check(status.code() == deepforge::import::ErrorCode::kInvalidShape,
                "CONCATENATE rejects an inconsistent runtime axis sum");
    status = compilation.executable->get_workspace_size(
        nullptr, workspace_size, {322}, {{2, 3, 2}}, {{15, 4, 1}});
    tests.check(status.code() == deepforge::import::ErrorCode::kInvalidShape,
                "CONCATENATE rejects inconsistent non-axis dimensions");
    status = compilation.executable->execute_variant(
        deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr, {321},
        {{2, 2}}, {{2, 1}});
    tests.check(status.code() == deepforge::import::ErrorCode::kInvalidShape,
                "CONCATENATE override rank must match the compiled role");
    status = compilation.executable->execute_variant(
        deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr,
        override_uids, override_shapes,
        {{12, 4, 1}, {15, 4, 1}, {8, 4, 1}, {1, 1, 1}});
    tests.check(status.code() == deepforge::import::ErrorCode::kInvalidLayout,
                "CONCATENATE rejects overlapping runtime strides");
    status = compilation.executable->execute_variant(
        deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr,
        override_uids, override_shapes,
        {{12, 4, 1}, {15, 4, 1}, {8, 4, 1}, {25, 16, 1}});
    tests.check(status.code() == deepforge::import::ErrorCode::kInvalidShape,
                "CONCATENATE runtime span cannot exceed the compiled byte bound");

    auto virtual_role = dynamic_concatenate_graph();
    virtual_role["tensors"]["324"]["is_virtual"] = true;
    status = compile_document(virtual_role);
    tests.check(status.code() ==
                    deepforge::import::ErrorCode::kUnsupportedOperation,
                "CONCATENATE overrides reject virtual role tensors");
    auto pass_by_value = dynamic_concatenate_graph();
    pass_by_value["tensors"]["321"]["is_pass_by_value"] = true;
    status = compile_document(pass_by_value);
    tests.check(status.code() ==
                    deepforge::import::ErrorCode::kUnsupportedOperation,
                "CONCATENATE overrides reject pass-by-value role tensors");
    auto in_place = dynamic_concatenate_graph();
    in_place["nodes"][0]["in_place_index"] = 0;
    status = compile_document(in_place);
    tests.check(status.code() ==
                    deepforge::import::ErrorCode::kUnsupportedOperation,
                "CONCATENATE overrides reject in-place aliasing");
    auto malformed_ports = dynamic_concatenate_graph();
    malformed_ports["nodes"][0]["inputs"]["3"] =
        malformed_ports["nodes"][0]["inputs"]["1"];
    malformed_ports["nodes"][0]["inputs"].erase("1");
    status = compile_document(malformed_ports);
    tests.check(status.code() == deepforge::import::ErrorCode::kInvalidValue,
                "CONCATENATE overrides require contiguous indexed inputs");
    auto composed = dynamic_concatenate_graph();
    composed["tensors"]["325"] =
        tensor("Z", 325, {2, 6, 3}, {25, 4, 1}, false);
    composed["nodes"].push_back(
        concatenate_node("second_concatenate", {324}, 325, 1));
    status = compile_document(composed);
    tests.check(status.code() ==
                    deepforge::import::ErrorCode::kUnsupportedOperation,
                "CONCATENATE overrides reject composed graphs");
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
    tests.check(
        std::all_of(
            compilation.variants.begin(), compilation.variants.end(),
            [](deepforge::compiler::VariantCode const& code) {
                return code.schedule == "generic-reference";
            }),
        "generic graph variants retain the reference schedule");

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

    run_dynamic_reshape_tests(tests);
    run_dynamic_transpose_tests(tests);
    run_dynamic_concatenate_tests(tests);

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
                    "artifact v5 preserves dynamic override metadata");
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

    deepforge::import::SerializedGraph dynamic_chain;
    status = parse_graph(dynamic_pointwise_chain_graph(), dynamic_chain);
    tests.good(status, "parse dynamic multi-node pointwise graph");
    deepforge::compiler::CompilationResult dynamic_chain_compilation;
    status = deepforge::compiler::compile_graph(
        dynamic_chain, options, dynamic_chain_compilation);
    tests.good(status, "compile dynamic multi-node pointwise graph");
    if (status.is_good() && dynamic_chain_compilation.executable) {
        auto const workspace_size =
            dynamic_chain_compilation.executable->get_workspace_size();
        tests.check(
            dynamic_chain_compilation.metadata.arguments.size() == 4 &&
                dynamic_chain_compilation.workspace.allocations.size() == 2 &&
                workspace_size >= 160 &&
                dynamic_chain_compilation.imported_mlir.find(
                    "memref.reinterpret_cast") != std::string::npos,
            "dynamic virtual tensors use bounded workspace views");

        std::vector<float> chain_a(20, -77.0F);
        std::vector<float> chain_b(20, -88.0F);
        std::vector<float> chain_c(20, -66.0F);
        std::vector<float> chain_y(20, -99.0F);
        for (std::size_t row = 0; row < 2; ++row) {
            for (std::size_t column = 0; column < 3; ++column) {
                auto const offset = row * 4 + column;
                chain_a[offset] = static_cast<float>(3 * row + column) - 4.0F;
                chain_b[offset] = static_cast<float>(row + column);
                chain_c[offset] = static_cast<float>(column + 1);
            }
        }
        deepforge::runtime::VariantPack chain_pack{
            {201, chain_a.data()},
            {202, chain_b.data()},
            {203, chain_c.data()},
            {206, chain_y.data()}};
        deepforge::runtime::OverrideUids chain_override_uids{201, 202, 203,
                                                             206};
        deepforge::runtime::OverrideShapes chain_override_shapes(
            4, std::vector<std::int64_t>{2, 3});
        deepforge::runtime::OverrideStrides chain_override_strides(
            4, std::vector<std::int64_t>{4, 1});
        AlignedBytes chain_workspace(static_cast<std::size_t>(workspace_size));
        std::int64_t queried_workspace_size = -1;
        status = dynamic_chain_compilation.executable->get_workspace_size(
            nullptr, queried_workspace_size, chain_override_uids,
            chain_override_shapes, chain_override_strides);
        tests.check(status.is_good() &&
                        queried_workspace_size == workspace_size,
                    "dynamic pointwise chain keeps a maximum workspace bound");
        status = dynamic_chain_compilation.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr, chain_pack,
            chain_workspace.pointer, chain_override_uids,
            chain_override_shapes, chain_override_strides);
        tests.good(status, "execute dynamic multi-node pointwise graph");
        bool chain_matches = true;
        for (std::size_t row = 0; row < 2; ++row) {
            for (std::size_t column = 0; column < 3; ++column) {
                auto const offset = row * 4 + column;
                auto const expected =
                    std::max(chain_a[offset] + chain_b[offset], 0.0F) *
                    chain_c[offset];
                chain_matches = chain_matches &&
                                std::fabs(chain_y[offset] - expected) < 1.0e-6F;
            }
        }
        chain_matches = chain_matches && chain_y[3] == -99.0F &&
                        chain_y[7] == -99.0F && chain_y[8] == -99.0F;
        tests.check(chain_matches,
                    "dynamic virtual shapes propagate through the pointwise DAG");

        std::vector<std::uint8_t> chain_artifact;
        status = deepforge::compiler::serialize_artifact(
            dynamic_chain_compilation, chain_artifact);
        tests.good(status, "serialize dynamic pointwise-chain artifact");
        std::unique_ptr<deepforge::runtime::Executable> chain_loaded;
        status = deepforge::compiler::load_artifact_executable(
            chain_artifact, chain_loaded);
        tests.good(status, "load dynamic pointwise-chain artifact");
        if (chain_loaded) {
            std::fill(chain_y.begin(), chain_y.end(), -99.0F);
            status = chain_loaded->execute(
                nullptr, chain_pack, chain_workspace.pointer,
                chain_override_uids, chain_override_shapes,
                chain_override_strides);
            tests.good(status, "execute loaded dynamic pointwise-chain artifact");
            tests.check(chain_y[0] == 0.0F && chain_y[6] == 12.0F &&
                            chain_y[7] == -99.0F,
                        "loaded pointwise-chain artifact propagates descriptors");
        }
        for (std::size_t index = 0; index < chain_a.size(); ++index) {
            chain_a[index] = static_cast<float>(index) - 8.0F;
            chain_b[index] = 1.0F;
            chain_c[index] = 2.0F;
        }
        std::fill(chain_y.begin(), chain_y.end(), -99.0F);
        status = dynamic_chain_compilation.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr, chain_pack,
            chain_workspace.pointer);
        tests.good(status,
                   "execute dynamic pointwise chain at compiled maximum shape");
        bool maximum_chain_matches = true;
        for (std::size_t index = 0; index < chain_y.size(); ++index) {
            auto const expected =
                std::max(chain_a[index] + chain_b[index], 0.0F) *
                chain_c[index];
            maximum_chain_matches = maximum_chain_matches &&
                                    chain_y[index] == expected;
        }
        tests.check(maximum_chain_matches,
                    "empty overrides propagate maximum virtual dimensions");
        status = dynamic_chain_compilation.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr, chain_pack,
            chain_workspace.pointer, {204}, {{2, 3}}, {{3, 1}});
        tests.check(
            status.code() == deepforge::import::ErrorCode::kInvalidVariantPack,
            "callers cannot override an internal virtual tensor UID");
    }

    auto mismatched_virtual = dynamic_pointwise_chain_graph();
    mismatched_virtual["tensors"]["204"]["dim"] = Json::array({4, 4});
    mismatched_virtual["tensors"]["204"]["stride"] = Json::array({4, 1});
    status = compile_document(mismatched_virtual);
    tests.check(status.code() ==
                    deepforge::import::ErrorCode::kUnsupportedOperation,
                "dynamic virtual tensors must share the external maximum shape");

    auto mixed_override = dynamic_pointwise_chain_graph();
    mixed_override["nodes"][2] =
        reshape_node("override_reshape", 205, 206, {4, 5}, {5, 1});
    status = compile_document(mixed_override);
    tests.check(status.code() ==
                    deepforge::import::ErrorCode::kUnsupportedOperation,
                "shape override rejects mixed pointwise and transform graphs");

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
