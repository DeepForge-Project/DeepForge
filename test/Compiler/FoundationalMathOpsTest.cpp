#include "DeepForge/Compiler/Artifact.h"
#include "DeepForge/Compiler/Codegen.h"
#include "DeepForge/Import/Schema.h"
#include "DeepForge/Import/SerializedGraphImporter.h"

#include <cudnn_frontend/thirdparty/nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

using Json = nlohmann::json;

constexpr double kReluLower = 0.25;
constexpr double kReluUpper = 1.0;
constexpr double kReluSlope = 0.2;
constexpr double kSwishBeta = 0.5;
constexpr double kEluAlpha = 2.0;
constexpr double kSoftplusBeta = 2.0;

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
            std::cout << "deepforge-foundational-math: " << checks_
                      << " checks passed\n";
            return 0;
        }
        std::cerr << "deepforge-foundational-math: " << failures_ << " of "
                  << checks_ << " checks failed\n";
        return 1;
    }

private:
    int checks_ = 0;
    int failures_ = 0;
};

class AlignedBytes {
public:
    explicit AlignedBytes(std::size_t size) {
        if (size != 0) {
            pointer_ = std::aligned_alloc(64, size);
        }
    }
    ~AlignedBytes() { std::free(pointer_); }
    AlignedBytes(AlignedBytes const&) = delete;
    AlignedBytes& operator=(AlignedBytes const&) = delete;

    [[nodiscard]] void* get() const noexcept { return pointer_; }

private:
    void* pointer_ = nullptr;
};

Json tensor(std::string name,
            std::int64_t uid,
            std::vector<std::int64_t> dimensions,
            std::vector<std::int64_t> strides,
            bool is_virtual = false,
            std::string data_type = "FLOAT") {
    return Json{{"name", std::move(name)},
                {"data_type", std::move(data_type)},
                {"dim", std::move(dimensions)},
                {"stride", std::move(strides)},
                {"is_virtual", is_virtual},
                {"pass_by_value", nullptr},
                {"is_pass_by_value", false},
                {"reordering_type", "NONE"},
                {"uid", uid},
                {"uid_assigned", true}};
}

Json context(std::string name) {
    return Json{{"name", std::move(name)},
                {"compute_data_type", "FLOAT"},
                {"intermediate_data_type", "FLOAT"},
                {"io_data_type", "FLOAT"},
                {"sm_count", -1},
                {"is_dynamic_shape_enabled", false},
                {"is_override_shape_enabled", false}};
}

Json graph_document(std::uint64_t uid,
                    std::string name,
                    Json nodes,
                    Json tensors) {
    return Json{{"context", context(std::move(name))},
                {"graph_uid", uid},
                {"json_version", "1.0"},
                {"cudnn_backend_version", "cpu-test"},
                {"cudnn_frontend_version", 12400},
                {"nodes", std::move(nodes)},
                {"tensors", std::move(tensors)}};
}

Json pointwise_graph() {
    Json nodes = Json::array();
    Json tensors = Json::object();
    tensors["101"] = tensor("A", 101, {2, 3}, {3, 1});
    tensors["102"] = tensor("B", 102, {3}, {1});
    tensors["103"] = tensor("T", 103, {2, 1}, {1, 1});
    auto modes = deepforge::import::pointwise_modes();
    for (std::size_t index = 0; index < modes.size(); ++index) {
        auto const mode = modes[index];
        auto const input_count =
            deepforge::import::pointwise_input_count(mode).value();
        Json inputs = Json::object();
        if (input_count >= 1) {
            inputs["IN_0"] = 101;
        }
        if (input_count >= 2) {
            inputs["IN_1"] = 102;
        }
        if (input_count >= 3) {
            inputs["IN_2"] = 103;
        }
        auto const output_uid = static_cast<std::int64_t>(1000 + index);
        nodes.push_back(
            Json{{"tag", "POINTWISE"},
                 {"name", "pointwise_" + std::string(mode)},
                 {"inputs", std::move(inputs)},
                 {"outputs", Json::object({{"OUT_0", output_uid}})},
                 {"compute_data_type", "FLOAT"},
                 {"mode", mode},
                 {"axis", 1},
                 {"relu_lower_clip", kReluLower},
                 {"relu_upper_clip", kReluUpper},
                 {"relu_lower_clip_slope", kReluSlope},
                 {"swish_beta", kSwishBeta},
                 {"elu_alpha", kEluAlpha},
                 {"softplus_beta", kSoftplusBeta}});
        tensors[std::to_string(output_uid)] =
            tensor("O_" + std::string(mode), output_uid, {2, 3}, {3, 1});
    }
    return graph_document(2101, "pointwise_all_modes", std::move(nodes),
                          std::move(tensors));
}

Json single_softplus_graph(double beta = 1.0) {
    Json tensors = Json::object();
    tensors["501"] = tensor("SX", 501, {2}, {1});
    tensors["502"] = tensor("SY", 502, {2}, {1});
    Json node = Json{{"tag", "POINTWISE"},
                     {"name", "stable_softplus"},
                     {"inputs", Json::object({{"IN_0", 501}})},
                     {"outputs", Json::object({{"OUT_0", 502}})},
                     {"compute_data_type", "FLOAT"},
                     {"mode", "SOFTPLUS_FWD"},
                     {"axis", nullptr},
                     {"relu_lower_clip", nullptr},
                     {"relu_upper_clip", nullptr},
                     {"relu_lower_clip_slope", nullptr},
                     {"swish_beta", nullptr},
                     {"elu_alpha", nullptr},
                     {"softplus_beta", beta}};
    return graph_document(2105, "single_softplus", Json::array({node}),
                          std::move(tensors));
}

Json runtime_scalar_pointwise_graph() {
    Json tensors = Json::object();
    tensors["601"] = tensor("X", 601, {2, 3}, {3, 1});
    tensors["602"] = tensor("ALPHA", 602, {1, 1}, {1, 1});
    tensors["602"]["is_pass_by_value"] = true;
    tensors["603"] = tensor("Y", 603, {2, 3}, {3, 1});
    Json node = Json{{"tag", "POINTWISE"},
                     {"name", "runtime_scalar_add"},
                     {"inputs", Json::object({{"IN_0", 601}, {"IN_1", 602}})},
                     {"outputs", Json::object({{"OUT_0", 603}})},
                     {"compute_data_type", "FLOAT"},
                     {"mode", "ADD"},
                     {"axis", nullptr},
                     {"relu_lower_clip", nullptr},
                     {"relu_upper_clip", nullptr},
                     {"relu_lower_clip_slope", nullptr},
                     {"swish_beta", nullptr},
                     {"elu_alpha", nullptr},
                     {"softplus_beta", nullptr}};
    return graph_document(2106, "runtime-scalar-pointwise",
                          Json::array({node}), std::move(tensors));
}

Json embedded_scalar_pointwise_graph() {
    auto graph = runtime_scalar_pointwise_graph();
    graph["graph_uid"] = 2107;
    graph["context"]["name"] = "embedded-scalar-pointwise";
    auto const payload = Json{{"index", 3}, {"value", "40200000"}};
    graph["tensors"]["602"]["pass_by_value"] = payload;
    graph["pass_by_values"] = Json::object({{"602", payload}});
    return graph;
}

Json all_embedded_scalar_types_graph() {
    auto graph = embedded_scalar_pointwise_graph();
    graph["graph_uid"] = 2108;
    graph["context"]["name"] = "all-embedded-scalar-types";
    struct ScalarCase {
        char const* uid;
        char const* data_type;
        Json payload;
    };
    std::array<ScalarCase, 5> const scalars{{
        {"604", "INT64", Json{{"index", 0}, {"value", -7}}},
        {"605", "INT32", Json{{"index", 1}, {"value", 7}}},
        {"606", "HALF", Json{{"index", 2}, {"value", 1.5}}},
        {"607", "DOUBLE", Json{{"index", 4}, {"value", 1.25}}},
        {"608", "BFLOAT16", Json{{"index", 5}, {"value", -2.0}}},
    }};
    for (auto const& scalar : scalars) {
        auto const uid = std::stoll(scalar.uid);
        graph["tensors"][scalar.uid] =
            tensor("UNUSED_" + std::string(scalar.data_type), uid, {1}, {1},
                   false, scalar.data_type);
        graph["tensors"][scalar.uid]["is_pass_by_value"] = true;
        graph["tensors"][scalar.uid]["pass_by_value"] = scalar.payload;
        graph["pass_by_values"][scalar.uid] = scalar.payload;
    }
    return graph;
}

Json reduction_graph() {
    Json nodes = Json::array();
    Json tensors = Json::object();
    tensors["201"] = tensor("RX", 201, {2, 3}, {3, 1});
    auto modes = deepforge::import::reduction_modes();
    for (std::size_t index = 0; index < modes.size(); ++index) {
        auto const output_uid = static_cast<std::int64_t>(2000 + index);
        nodes.push_back(
            Json{{"tag", "REDUCTION"},
                 {"name", "reduction_" + std::string(modes[index])},
                 {"inputs", Json::object({{"X", 201}})},
                 {"outputs", Json::object({{"Y", output_uid}})},
                 {"compute_data_type", "FLOAT"},
                 {"mode", modes[index]},
                 {"is_deterministic", true}});
        tensors[std::to_string(output_uid)] =
            tensor("R_" + std::string(modes[index]), output_uid, {2, 1},
                   {1, 1});
    }
    return graph_document(2102, "reduction_all_modes", std::move(nodes),
                          std::move(tensors));
}

Json matmul_graph() {
    Json tensors = Json::object();
    tensors["301"] = tensor("MA", 301, {2, 2, 3}, {6, 3, 1});
    tensors["302"] = tensor("MB", 302, {1, 3, 2}, {6, 2, 1});
    tensors["303"] = tensor("Bias", 303, {2}, {1});
    tensors["304"] = tensor("MC", 304, {2, 2, 2}, {4, 2, 1}, true);
    tensors["305"] = tensor("MY", 305, {2, 2, 2}, {4, 2, 1});
    Json matmul = Json{{"tag", "MATMUL"},
                       {"name", "batched_matmul"},
                       {"inputs", Json::object({{"A", 301}, {"B", 302}})},
                       {"outputs", Json::object({{"C", 304}})},
                       {"compute_data_type", "FLOAT"},
                       {"padding_value", 0.0}};
    Json add = Json{{"tag", "POINTWISE"},
                    {"name", "bias_add"},
                    {"inputs", Json::object({{"IN_0", 304}, {"IN_1", 303}})},
                    {"outputs", Json::object({{"OUT_0", 305}})},
                    {"compute_data_type", "FLOAT"},
                    {"mode", "ADD"},
                    {"axis", nullptr},
                    {"relu_lower_clip", nullptr},
                    {"relu_upper_clip", nullptr},
                    {"relu_lower_clip_slope", nullptr},
                    {"swish_beta", nullptr},
                    {"elu_alpha", nullptr},
                    {"softplus_beta", nullptr}};
    return graph_document(2103, "matmul_bias", Json::array({matmul, add}),
                          std::move(tensors));
}

Json matmul_override_graph() {
    Json tensors = Json::object();
    tensors["306"] = tensor("OverrideA", 306, {2, 3, 4}, {12, 4, 1});
    tensors["307"] = tensor("OverrideB", 307, {1, 4, 3}, {12, 3, 1});
    tensors["308"] =
        tensor("M_override", 308, {2, 1, 1}, {1, 1, 1}, false, "INT32");
    tensors["309"] =
        tensor("N_override", 309, {1, 1, 1}, {1, 1, 1}, false, "INT32");
    tensors["310"] =
        tensor("K_override", 310, {2, 1, 1}, {1, 1, 1}, false, "INT32");
    tensors["311"] = tensor("OverrideC", 311, {2, 3, 3}, {9, 3, 1});
    Json matmul =
        Json{{"tag", "MATMUL"},
             {"name", "batched_override_matmul"},
             {"inputs", Json::object({{"A", 306},
                                      {"B", 307},
                                      {"M_override", 308},
                                      {"N_override", 309},
                                      {"K_override", 310}})},
             {"outputs", Json::object({{"C", 311}})},
             {"compute_data_type", "FLOAT"},
             {"padding_value", -7.0}};
    return graph_document(2106, "matmul_overrides", Json::array({matmul}),
                          std::move(tensors));
}

Json runtime_shape_override_matmul_graph() {
    Json tensors = Json::object();
    tensors["312"] =
        tensor("DynamicA", 312, {2, 4, 5}, {20, 5, 1});
    tensors["313"] =
        tensor("DynamicB", 313, {2, 5, 3}, {15, 1, 5});
    tensors["314"] =
        tensor("DynamicC", 314, {2, 4, 3}, {12, 3, 1});
    Json matmul =
        Json{{"tag", "MATMUL"},
             {"name", "runtime_shape_override_matmul"},
             {"inputs", Json::object({{"A", 312}, {"B", 313}})},
             {"outputs", Json::object({{"C", 314}})},
             {"compute_data_type", "FLOAT"},
             {"padding_value", 0.0}};
    auto graph = graph_document(2109, "runtime_shape_override_matmul",
                                Json::array({matmul}), std::move(tensors));
    graph["context"]["is_override_shape_enabled"] = true;
    return graph;
}

Json resample_node(std::string mode,
                   std::string padding,
                   std::int64_t output_uid,
                   std::vector<std::int64_t> pre_padding,
                   std::vector<std::int64_t> post_padding,
                   std::vector<std::int64_t> strides,
                   std::vector<std::int64_t> windows) {
    return Json{{"tag", "RESAMPLE"},
                {"name", "resample_" + mode},
                {"inputs", Json::object({{"X", 401}})},
                {"outputs", Json::object({{"Y", output_uid}})},
                {"generate_index", false},
                {"resample_mode", std::move(mode)},
                {"padding_mode", std::move(padding)},
                {"pre_padding", std::move(pre_padding)},
                {"post_padding", std::move(post_padding)},
                {"stride", std::move(strides)},
                {"window", std::move(windows)}};
}

Json resample_graph() {
    Json tensors = Json::object();
    tensors["401"] = tensor("PX", 401, {1, 1, 3, 3}, {9, 9, 3, 1});
    Json nodes = Json::array();
    constexpr std::array<std::string_view, 3> pooling_modes{
        "AVGPOOL_EXCLUDE_PADDING", "AVGPOOL_INCLUDE_PADDING", "MAXPOOL"};
    for (std::size_t index = 0; index < pooling_modes.size(); ++index) {
        auto const uid = static_cast<std::int64_t>(410 + index);
        nodes.push_back(resample_node(
            std::string(pooling_modes[index]),
            pooling_modes[index] == "AVGPOOL_INCLUDE_PADDING"
                ? "ZERO_PAD"
                : "NEG_INF_PAD",
            uid, {1, 1}, {0, 0}, {2, 2}, {2, 2}));
        tensors[std::to_string(uid)] =
            tensor("Pool_" + std::string(pooling_modes[index]), uid,
                   {1, 1, 2, 2}, {4, 4, 2, 1});
    }
    nodes.push_back(resample_node("NEAREST", "EDGE_VAL_PAD", 420,
                                  {1, 1}, {0, 0}, {1, 1}, {1, 1}));
    tensors["420"] = tensor("Nearest", 420, {1, 1, 4, 4}, {16, 16, 4, 1});
    return graph_document(2104, "resample_modes", std::move(nodes),
                          std::move(tensors));
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

double sigmoid(double value) {
    return 1.0 / (1.0 + std::exp(-value));
}

double pointwise_reference(std::string_view mode,
                           double a,
                           double b,
                           double c,
                           std::size_t linear_index) {
    if (mode == "ADD") return a + b;
    if (mode == "MUL") return a * b;
    if (mode == "SQRT") return std::sqrt(a);
    if (mode == "MAX") return std::max(a, b);
    if (mode == "MIN") return std::min(a, b);
    if (mode == "RELU_FWD") {
        if (a < kReluLower) {
            return kReluLower + kReluSlope * (a - kReluLower);
        }
        return a > kReluUpper ? kReluUpper : a;
    }
    if (mode == "TANH_FWD") return std::tanh(a);
    if (mode == "SIGMOID_FWD") return sigmoid(a);
    if (mode == "ELU_FWD") {
        return a >= 0.0 ? a : kEluAlpha * (std::exp(a) - 1.0);
    }
    if (mode == "GELU_FWD") {
        return 0.5 * a * (1.0 + std::erf(a / std::sqrt(2.0)));
    }
    if (mode == "SOFTPLUS_FWD") {
        return std::log(1.0 + std::exp(kSoftplusBeta * a)) /
               kSoftplusBeta;
    }
    if (mode == "SWISH_FWD") return a * sigmoid(kSwishBeta * a);
    if (mode == "RELU_BWD") {
        auto derivative = b < kReluLower
                              ? kReluSlope
                              : (b > kReluUpper ? 0.0 : 1.0);
        return a * derivative;
    }
    if (mode == "TANH_BWD") {
        auto t = std::tanh(b);
        return a * (1.0 - t * t);
    }
    if (mode == "SIGMOID_BWD") {
        auto s = sigmoid(b);
        return a * s * (1.0 - s);
    }
    if (mode == "ELU_BWD") {
        return a * (b >= 0.0 ? 1.0 : kEluAlpha * std::exp(b));
    }
    if (mode == "GELU_BWD") {
        auto cdf = 0.5 * (1.0 + std::erf(b / std::sqrt(2.0)));
        auto density = std::exp(-0.5 * b * b) / std::sqrt(2.0 * std::acos(-1.0));
        return a * (cdf + b * density);
    }
    if (mode == "SOFTPLUS_BWD") {
        return a * sigmoid(kSoftplusBeta * b);
    }
    if (mode == "SWISH_BWD") {
        auto s = sigmoid(kSwishBeta * b);
        return a * (s + kSwishBeta * b * s * (1.0 - s));
    }
    if (mode == "ERF") return std::erf(a);
    if (mode == "IDENTITY") return a;
    if (mode == "GELU_APPROX_TANH_FWD" ||
        mode == "GELU_APPROX_TANH_BWD") {
        auto inner = std::sqrt(2.0 / std::acos(-1.0)) *
                     (b + 0.044715 * b * b * b);
        if (mode == "GELU_APPROX_TANH_FWD") {
            inner = std::sqrt(2.0 / std::acos(-1.0)) *
                    (a + 0.044715 * a * a * a);
            return 0.5 * a * (1.0 + std::tanh(inner));
        }
        auto t = std::tanh(inner);
        auto derivative = 0.5 * (1.0 + t) +
                          0.5 * b * (1.0 - t * t) *
                              std::sqrt(2.0 / std::acos(-1.0)) *
                              (1.0 + 3.0 * 0.044715 * b * b);
        return a * derivative;
    }
    if (mode == "GEN_INDEX") return static_cast<double>(linear_index % 3);
    if (mode == "BINARY_SELECT") return c != 0.0 ? b : a;
    if (mode == "EXP") return std::exp(a);
    if (mode == "LOG") return std::log(a);
    if (mode == "NEG") return -a;
    if (mode == "MOD") return std::fmod(a, b);
    if (mode == "POW") return std::pow(a, b);
    if (mode == "ABS") return std::fabs(a);
    if (mode == "CEIL") return std::ceil(a);
    if (mode == "COS") return std::cos(a);
    if (mode == "FLOOR") return std::floor(a);
    if (mode == "RSQRT") return 1.0 / std::sqrt(a);
    if (mode == "SIN") return std::sin(a);
    if (mode == "LOGICAL_NOT") return a == 0.0 ? 1.0 : 0.0;
    if (mode == "TAN") return std::tan(a);
    if (mode == "SUB") return a - b;
    if (mode == "ADD_SQUARE") return a + b * b;
    if (mode == "DIV") return a / b;
    if (mode == "CMP_EQ") return a == b ? 1.0 : 0.0;
    if (mode == "CMP_NEQ") return a != b ? 1.0 : 0.0;
    if (mode == "CMP_GT") return a > b ? 1.0 : 0.0;
    if (mode == "CMP_GE") return a >= b ? 1.0 : 0.0;
    if (mode == "CMP_LT") return a < b ? 1.0 : 0.0;
    if (mode == "CMP_LE") return a <= b ? 1.0 : 0.0;
    if (mode == "LOGICAL_AND") return a != 0.0 && b != 0.0 ? 1.0 : 0.0;
    if (mode == "LOGICAL_OR") return a != 0.0 || b != 0.0 ? 1.0 : 0.0;
    if (mode == "RECIPROCAL") return 1.0 / a;
    return std::numeric_limits<double>::quiet_NaN();
}

bool close(float actual, double expected) {
    if (std::isnan(expected)) {
        return std::isnan(actual);
    }
    if (std::isinf(expected)) {
        return static_cast<double>(actual) == expected;
    }
    auto const difference = std::fabs(static_cast<double>(actual) - expected);
    return difference <= 3.0e-4 + 3.0e-4 * std::fabs(expected);
}

bool pointwise_outputs_match(
    std::vector<std::vector<float>> const& outputs,
    std::vector<float> const& a,
    std::vector<float> const& b,
    std::vector<float> const& c) {
    auto modes = deepforge::import::pointwise_modes();
    for (std::size_t mode_index = 0; mode_index < modes.size(); ++mode_index) {
        for (std::size_t index = 0; index < a.size(); ++index) {
            auto expected = pointwise_reference(
                modes[mode_index], a[index], b[index % 3], c[index / 3], index);
            if (!close(outputs[mode_index][index], expected)) {
                std::cerr << "mode " << modes[mode_index] << " index " << index
                          << " expected " << expected << " got "
                          << outputs[mode_index][index] << '\n';
                return false;
            }
        }
    }
    return true;
}

double reduction_reference(std::string_view mode,
                           std::span<float const> row) {
    if (mode == "ADD" || mode == "AVG") {
        double value = 0.0;
        for (auto item : row) value += item;
        return mode == "AVG" ? value / static_cast<double>(row.size()) : value;
    }
    if (mode == "MUL" || mode == "MUL_NO_ZEROS") {
        double value = 1.0;
        for (auto item : row) {
            if (mode != "MUL_NO_ZEROS" || item != 0.0F) value *= item;
        }
        return value;
    }
    if (mode == "MIN") return *std::min_element(row.begin(), row.end());
    if (mode == "MAX") return *std::max_element(row.begin(), row.end());
    double value = 0.0;
    for (auto item : row) {
        if (mode == "AMAX") {
            value = std::max(value, std::fabs(static_cast<double>(item)));
        }
        if (mode == "NORM1") value += std::fabs(item);
        if (mode == "NORM2") value += item * item;
    }
    return mode == "NORM2" ? std::sqrt(value) : value;
}

}  // namespace

int main() {
    TestRunner tests;
    deepforge::compiler::CompileOptions options;

    deepforge::import::SerializedGraph pointwise;
    auto status = parse_graph(pointwise_graph(), pointwise);
    tests.good(status, "parse all pointwise modes");
    deepforge::compiler::CompilationResult pointwise_compilation;
    if (status.is_good()) {
        status = deepforge::compiler::compile_graph(
            pointwise, options, pointwise_compilation);
    }
    tests.good(status, "compile all pointwise modes");
    std::vector<float> a{0.0F, 0.25F, 0.5F, 0.75F, 1.0F, 1.5F};
    std::vector<float> b{0.0F, 0.25F, -2.0F};
    std::vector<float> c{0.0F, 9.0F};
    std::vector<std::vector<float>> pointwise_outputs(
        deepforge::import::pointwise_modes().size(),
        std::vector<float>(a.size(), -99.0F));
    deepforge::runtime::VariantPack pointwise_pack{
        {101, a.data()}, {102, b.data()}, {103, c.data()}};
    for (std::size_t index = 0; index < pointwise_outputs.size(); ++index) {
        pointwise_pack.emplace(static_cast<std::int64_t>(1000 + index),
                               pointwise_outputs[index].data());
    }
    if (pointwise_compilation.executable) {
        status = pointwise_compilation.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr, pointwise_pack,
            nullptr);
        tests.good(status, "execute all pointwise modes");
        tests.check(pointwise_outputs_match(pointwise_outputs, a, b, c),
                    "all pointwise modes match scalar references");
    }

    std::vector<std::uint8_t> artifact;
    status = deepforge::compiler::serialize_artifact(pointwise_compilation,
                                                      artifact);
    tests.good(status, "serialize pointwise artifact");
    std::unique_ptr<deepforge::runtime::Executable> loaded;
    if (status.is_good()) {
        status = deepforge::compiler::load_artifact_executable(artifact,
                                                               loaded);
    }
    tests.good(status, "load pointwise artifact with math symbols");
    if (loaded) {
        for (auto& output : pointwise_outputs) {
            std::fill(output.begin(), output.end(), -99.0F);
        }
        status = loaded->execute(nullptr, pointwise_pack, nullptr);
        tests.good(status, "execute loaded pointwise artifact");
        tests.check(pointwise_outputs_match(pointwise_outputs, a, b, c),
                    "loaded pointwise artifact matches references");
    }

    deepforge::import::SerializedGraph runtime_scalar;
    status = parse_graph(runtime_scalar_pointwise_graph(), runtime_scalar);
    tests.good(status, "parse runtime pass-by-value scalar");
    deepforge::compiler::CompilationResult runtime_scalar_compilation;
    if (status.is_good()) {
        status = deepforge::compiler::compile_graph(
            runtime_scalar, options, runtime_scalar_compilation);
    }
    tests.good(status, "compile runtime pass-by-value scalar");
    auto const scalar_argument = std::find_if(
        runtime_scalar_compilation.metadata.arguments.begin(),
        runtime_scalar_compilation.metadata.arguments.end(),
        [](auto const& argument) { return argument.uid == 602; });
    tests.check(
        scalar_argument != runtime_scalar_compilation.metadata.arguments.end() &&
            scalar_argument->access ==
                deepforge::compiler::TensorAccess::kRead &&
            scalar_argument->dimensions ==
                std::vector<std::int64_t>({1, 1}) &&
            scalar_argument->size_bytes == sizeof(float),
        "runtime scalar remains an ordinary one-element read argument");
    std::vector<float> scalar_input{1.0F, 2.0F, 3.0F,
                                    4.0F, 5.0F, 6.0F};
    float alpha = 2.5F;
    std::vector<float> scalar_output(6, -99.0F);
    deepforge::runtime::VariantPack scalar_pack{{601, scalar_input.data()},
                                                {602, &alpha},
                                                {603, scalar_output.data()}};
    if (runtime_scalar_compilation.executable) {
        status = runtime_scalar_compilation.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr, scalar_pack,
            nullptr);
        tests.good(status, "execute runtime pass-by-value scalar");
        tests.check(scalar_output ==
                        std::vector<float>({3.5F, 4.5F, 5.5F,
                                            6.5F, 7.5F, 8.5F}),
                    "runtime scalar broadcasts from the UID-map pointer");
    }

    std::vector<std::uint8_t> scalar_artifact;
    status = deepforge::compiler::serialize_artifact(
        runtime_scalar_compilation, scalar_artifact);
    tests.good(status, "serialize runtime scalar artifact");
    std::unique_ptr<deepforge::runtime::Executable> scalar_loaded;
    deepforge::compiler::ArtifactInfo scalar_info;
    if (status.is_good()) {
        status = deepforge::compiler::load_artifact_executable(
            scalar_artifact, scalar_loaded, &scalar_info);
    }
    tests.good(status, "load runtime scalar artifact");
    tests.check(scalar_info.metadata == runtime_scalar_compilation.metadata,
                "runtime scalar artifact preserves argument metadata");
    if (scalar_loaded) {
        std::fill(scalar_output.begin(), scalar_output.end(), -99.0F);
        status = scalar_loaded->execute(nullptr, scalar_pack, nullptr);
        tests.good(status, "execute loaded runtime scalar artifact");
        tests.check(scalar_output ==
                        std::vector<float>({3.5F, 4.5F, 5.5F,
                                            6.5F, 7.5F, 8.5F}),
                    "loaded artifact consumes the runtime scalar pointer");
    }

    deepforge::import::SerializedGraph embedded_scalar;
    status = parse_graph(embedded_scalar_pointwise_graph(), embedded_scalar);
    tests.good(status, "parse embedded pass-by-value scalar");
    auto embedded_options = options;
    embedded_options.capture_mlir = true;
    deepforge::compiler::CompilationResult embedded_compilation;
    if (status.is_good()) {
        status = deepforge::compiler::compile_graph(
            embedded_scalar, embedded_options, embedded_compilation);
    }
    tests.good(status, "compile embedded pass-by-value scalar");
    auto const embedded_argument = std::find_if(
        embedded_compilation.metadata.arguments.begin(),
        embedded_compilation.metadata.arguments.end(),
        [](auto const& argument) { return argument.uid == 602; });
    tests.check(
        embedded_argument == embedded_compilation.metadata.arguments.end() &&
            embedded_compilation.metadata.arguments.size() == 2,
        "embedded scalar is graph-owned and absent from public arguments");
    tests.check(
        embedded_compilation.imported_mlir.find("memref.global") !=
                std::string::npos &&
            embedded_compilation.imported_mlir.find("__deepforge_pbv_") !=
                std::string::npos,
        "embedded scalar lowers to a private MLIR global");
    std::vector<float> embedded_output(6, -99.0F);
    deepforge::runtime::VariantPack embedded_pack{
        {601, scalar_input.data()}, {603, embedded_output.data()}};
    if (embedded_compilation.executable) {
        status = embedded_compilation.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr, embedded_pack,
            nullptr);
        tests.good(status, "execute without embedded scalar UID");
        tests.check(embedded_output ==
                        std::vector<float>({3.5F, 4.5F, 5.5F,
                                            6.5F, 7.5F, 8.5F}),
                    "embedded scalar broadcasts from graph-owned storage");

        float attempted_override = 100.0F;
        embedded_pack.emplace(602, &attempted_override);
        std::fill(embedded_output.begin(), embedded_output.end(), -99.0F);
        status = embedded_compilation.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr, embedded_pack,
            nullptr);
        tests.good(status, "execute with ignored embedded scalar override");
        tests.check(embedded_output ==
                        std::vector<float>({3.5F, 4.5F, 5.5F,
                                            6.5F, 7.5F, 8.5F}),
                    "caller cannot override a graph-owned scalar");
        embedded_pack.erase(602);
    }

    std::vector<std::uint8_t> embedded_artifact;
    status = deepforge::compiler::serialize_artifact(
        embedded_compilation, embedded_artifact);
    tests.good(status, "serialize embedded scalar artifact");
    std::unique_ptr<deepforge::runtime::Executable> embedded_loaded;
    deepforge::compiler::ArtifactInfo embedded_info;
    if (status.is_good()) {
        status = deepforge::compiler::load_artifact_executable(
            embedded_artifact, embedded_loaded, &embedded_info);
    }
    tests.good(status, "load embedded scalar artifact");
    tests.check(embedded_info.metadata == embedded_compilation.metadata,
                "artifact preserves embedded scalar argument exclusion");
    if (embedded_loaded) {
        std::fill(embedded_output.begin(), embedded_output.end(), -99.0F);
        status = embedded_loaded->execute(nullptr, embedded_pack, nullptr);
        tests.good(status, "execute loaded embedded scalar artifact");
        tests.check(embedded_output ==
                        std::vector<float>({3.5F, 4.5F, 5.5F,
                                            6.5F, 7.5F, 8.5F}),
                    "artifact carries graph-owned scalar data");
    }

    deepforge::import::SerializedGraph all_embedded_scalars;
    status = parse_graph(all_embedded_scalar_types_graph(),
                         all_embedded_scalars);
    tests.good(status, "parse all embedded scalar storage types");
    deepforge::compiler::CompilationResult all_embedded_compilation;
    if (status.is_good()) {
        status = deepforge::compiler::compile_graph(
            all_embedded_scalars, embedded_options,
            all_embedded_compilation);
    }
    tests.good(status, "compile all embedded scalar storage types");
    std::size_t global_count = 0;
    std::size_t global_position = 0;
    while ((global_position = all_embedded_compilation.imported_mlir.find(
                "memref.global", global_position)) != std::string::npos) {
        ++global_count;
        global_position += std::string_view("memref.global").size();
    }
    tests.check(
        global_count == 6 &&
            std::none_of(all_embedded_compilation.metadata.arguments.begin(),
                         all_embedded_compilation.metadata.arguments.end(),
                         [](auto const& argument) {
                             return argument.uid == 602 ||
                                    (argument.uid >= 604 && argument.uid <= 608);
                         }),
        "all six Frontend scalar variants build graph-owned globals");

    deepforge::import::SerializedGraph softplus;
    status = parse_graph(single_softplus_graph(), softplus);
    tests.good(status, "parse single-node stable softplus");
    deepforge::compiler::CompilationResult softplus_compilation;
    if (status.is_good()) {
        status = deepforge::compiler::compile_graph(
            softplus, options, softplus_compilation);
    }
    tests.good(status, "compile single-node stable softplus");
    if (softplus_compilation.executable) {
        std::vector<float> input{100.0F, -100.0F};
        std::vector<float> output(2, -99.0F);
        deepforge::runtime::VariantPack pack{{501, input.data()},
                                             {502, output.data()}};
        status = softplus_compilation.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr);
        tests.good(status, "execute single-node stable softplus");
        tests.check(close(output[0], 100.0) && output[1] >= 0.0F &&
                        output[1] < 1.0e-40F,
                    "softplus remains finite at large magnitudes");
    }

    deepforge::import::SerializedGraph invalid_softplus;
    status = parse_graph(single_softplus_graph(0.0), invalid_softplus);
    tests.good(status, "parse structurally valid zero-beta softplus");
    deepforge::compiler::CompilationResult invalid_softplus_compilation;
    status = deepforge::compiler::compile_graph(
        invalid_softplus, options, invalid_softplus_compilation);
    tests.check(status.code() == deepforge::import::ErrorCode::kInvalidValue,
                "zero-beta softplus is rejected before codegen");

    deepforge::import::SerializedGraph overflowing_softplus;
    status = parse_graph(single_softplus_graph(1.0e300),
                         overflowing_softplus);
    tests.good(status, "parse structurally valid oversized softplus beta");
    deepforge::compiler::CompilationResult overflowing_softplus_compilation;
    status = deepforge::compiler::compile_graph(
        overflowing_softplus, options, overflowing_softplus_compilation);
    tests.check(status.code() == deepforge::import::ErrorCode::kInvalidValue,
                "numeric pointwise attributes must fit finite f32");

    deepforge::import::SerializedGraph reduction;
    status = parse_graph(reduction_graph(), reduction);
    tests.good(status, "parse all reduction modes");
    deepforge::compiler::CompilationResult reduction_compilation;
    if (status.is_good()) {
        status = deepforge::compiler::compile_graph(
            reduction, options, reduction_compilation);
    }
    tests.good(status, "compile all reduction modes");
    std::vector<float> reduction_input{1.0F, -2.0F, 0.0F,
                                       4.0F, -5.0F, 6.0F};
    std::vector<std::vector<float>> reduction_outputs(
        deepforge::import::reduction_modes().size(),
        std::vector<float>(2, -99.0F));
    deepforge::runtime::VariantPack reduction_pack{{201,
                                                    reduction_input.data()}};
    for (std::size_t index = 0; index < reduction_outputs.size(); ++index) {
        reduction_pack.emplace(static_cast<std::int64_t>(2000 + index),
                               reduction_outputs[index].data());
    }
    if (reduction_compilation.executable) {
        status = reduction_compilation.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr, reduction_pack,
            nullptr);
        tests.good(status, "execute all reduction modes");
        bool reductions_match = true;
        auto modes = deepforge::import::reduction_modes();
        for (std::size_t mode_index = 0; mode_index < modes.size();
             ++mode_index) {
            for (std::size_t row = 0; row < 2; ++row) {
                auto expected = reduction_reference(
                    modes[mode_index],
                    std::span<float const>(reduction_input).subspan(row * 3, 3));
                reductions_match &=
                    close(reduction_outputs[mode_index][row], expected);
            }
        }
        tests.check(reductions_match,
                    "all reduction modes match scalar references");
    }

    auto identity_reduction_document = reduction_graph();
    identity_reduction_document["nodes"] =
        Json::array({identity_reduction_document["nodes"][0]});
    identity_reduction_document["tensors"]["2000"]["dim"] =
        Json::array({2, 3});
    identity_reduction_document["tensors"]["2000"]["stride"] =
        Json::array({3, 1});
    deepforge::import::SerializedGraph identity_reduction;
    status = parse_graph(identity_reduction_document, identity_reduction);
    tests.good(status, "parse reduction with no inferred axes");
    deepforge::compiler::CompilationResult identity_reduction_compilation;
    if (status.is_good()) {
        status = deepforge::compiler::compile_graph(
            identity_reduction, options, identity_reduction_compilation);
    }
    tests.good(status, "compile reduction with no inferred axes");
    if (identity_reduction_compilation.executable) {
        std::vector<float> identity_output(6, -99.0F);
        deepforge::runtime::VariantPack pack{
            {201, reduction_input.data()}, {2000, identity_output.data()}};
        status = identity_reduction_compilation.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr);
        tests.good(status, "execute reduction with no inferred axes");
        tests.check(identity_output == reduction_input,
                    "reduction with no inferred axes is an identity");
    }

    deepforge::import::SerializedGraph matmul;
    status = parse_graph(matmul_graph(), matmul);
    tests.good(status, "parse batched matmul with broadcast bias");
    deepforge::compiler::CompilationResult matmul_compilation;
    if (status.is_good()) {
        status = deepforge::compiler::compile_graph(matmul, options,
                                                    matmul_compilation);
    }
    tests.good(status, "compile batched matmul with broadcast bias");
    if (matmul_compilation.executable) {
        std::vector<float> matmul_a{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F,
                                    -1.0F, 0.0F, 1.0F, 2.0F, -2.0F, 1.0F};
        std::vector<float> matmul_b{1.0F, 2.0F, 0.0F,
                                    1.0F, 1.0F, 0.0F};
        std::vector<float> bias{0.5F, -1.0F};
        std::vector<float> output(8, -99.0F);
        AlignedBytes workspace(static_cast<std::size_t>(
            matmul_compilation.executable->get_workspace_size()));
        deepforge::runtime::VariantPack pack{{301, matmul_a.data()},
                                             {302, matmul_b.data()},
                                             {303, bias.data()},
                                             {305, output.data()}};
        status = matmul_compilation.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr, pack,
            workspace.get());
        tests.good(status, "execute batched matmul with broadcast bias");
        constexpr float expected[]{4.5F, 3.0F, 10.5F, 12.0F,
                                   0.5F, -3.0F, 3.5F, 1.0F};
        bool matches = true;
        for (std::size_t index = 0; index < output.size(); ++index) {
            matches &= close(output[index], expected[index]);
        }
        tests.check(matches, "batched matmul composition matches reference");
        tests.check(matmul_compilation.workspace.allocations.size() == 1 &&
                        matmul_compilation.workspace.size_bytes == 64,
                    "matmul virtual output uses workspace");
    }

    deepforge::import::SerializedGraph override_matmul;
    status = parse_graph(matmul_override_graph(), override_matmul);
    tests.good(status, "parse batched matmul dimension overrides");
    deepforge::compiler::CompilationResult override_matmul_compilation;
    if (status.is_good()) {
        status = deepforge::compiler::compile_graph(
            override_matmul, options, override_matmul_compilation);
    }
    tests.good(status, "compile batched matmul dimension overrides");
    auto const int32_override_count = std::count_if(
        override_matmul_compilation.metadata.arguments.begin(),
        override_matmul_compilation.metadata.arguments.end(),
        [](auto const& argument) {
            return argument.uid >= 308 && argument.uid <= 310 &&
                   argument.data_type ==
                       deepforge::import::DataType::kInt32 &&
                   argument.access ==
                       deepforge::compiler::TensorAccess::kRead;
        });
    tests.check(int32_override_count == 3,
                "matmul override parameters retain INT32 artifact metadata");
    if (override_matmul_compilation.executable) {
        std::vector<float> override_a{
            1.0F,  2.0F,  3.0F,  4.0F, 5.0F, 6.0F,
            7.0F,  8.0F,  9.0F, 10.0F, 11.0F, 12.0F,
            1.0F, -1.0F,  2.0F,  0.0F, 0.0F, 3.0F,
           -2.0F,  1.0F,  4.0F,  1.0F,  0.0F, -1.0F};
        std::vector<float> override_b{1.0F,  2.0F,  3.0F, 4.0F,
                                      5.0F,  6.0F,  7.0F, 8.0F,
                                      9.0F, 10.0F, 11.0F, 12.0F};
        std::vector<std::int32_t> m_override{2, 3};
        std::vector<std::int32_t> n_override{1};
        std::vector<std::int32_t> k_override{2, 4};
        std::vector<float> output(18, -99.0F);
        deepforge::runtime::VariantPack pack{{306, override_a.data()},
                                             {307, override_b.data()},
                                             {308, m_override.data()},
                                             {309, n_override.data()},
                                             {310, k_override.data()},
                                             {311, output.data()}};
        status = override_matmul_compilation.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr);
        tests.good(status, "execute batched matmul dimension overrides");
        constexpr std::array<float, 18> expected{
            9.0F, -7.0F, -7.0F, 29.0F, -7.0F, -7.0F,
           -7.0F, -7.0F, -7.0F, 11.0F, -7.0F, -7.0F,
            8.0F, -7.0F, -7.0F, -2.0F, -7.0F, -7.0F};
        tests.check(std::equal(output.begin(), output.end(), expected.begin(),
                               [](float lhs, float rhs) {
                                   return close(lhs, rhs);
                               }),
                    "M/N/K overrides and nonzero padding match reference");

        std::vector<std::uint8_t> override_artifact;
        status = deepforge::compiler::serialize_artifact(
            override_matmul_compilation, override_artifact);
        tests.good(status, "serialize matmul override artifact");
        std::unique_ptr<deepforge::runtime::Executable> override_loaded;
        deepforge::compiler::ArtifactInfo override_info;
        if (status.is_good()) {
            status = deepforge::compiler::load_artifact_executable(
                override_artifact, override_loaded, &override_info);
        }
        tests.good(status, "load matmul override artifact");
        tests.check(override_info.metadata ==
                        override_matmul_compilation.metadata,
                    "matmul override artifact preserves parameter metadata");
        if (override_loaded) {
            std::fill(output.begin(), output.end(), -99.0F);
            status = override_loaded->execute(nullptr, pack, nullptr);
            tests.good(status, "execute loaded matmul override artifact");
            tests.check(std::equal(
                            output.begin(), output.end(), expected.begin(),
                            [](float lhs, float rhs) {
                                return close(lhs, rhs);
                            }),
                        "loaded matmul override artifact matches reference");
        }
    }

    deepforge::import::SerializedGraph dynamic_matmul;
    status = parse_graph(runtime_shape_override_matmul_graph(), dynamic_matmul);
    tests.good(status, "parse runtime shape-override MATMUL");
    deepforge::compiler::CompilationResult dynamic_matmul_compilation;
    if (status.is_good()) {
        auto dynamic_matmul_options = options;
        dynamic_matmul_options.capture_mlir = true;
        status = deepforge::compiler::compile_graph(
            dynamic_matmul, dynamic_matmul_options,
            dynamic_matmul_compilation);
    }
    tests.good(status, "compile runtime shape-override MATMUL");
    if (dynamic_matmul_compilation.executable) {
        tests.check(dynamic_matmul_compilation.metadata.override_policy ==
                        deepforge::compiler::ShapeOverridePolicy::kMatmul,
                    "dynamic MATMUL records its override policy");
        tests.check(
            !dynamic_matmul_compilation.metadata.dynamic_shape_enabled &&
                dynamic_matmul_compilation.metadata.override_shape_enabled,
            "MATMUL override execution does not require the independent dynamic flag");
        tests.check(dynamic_matmul_compilation.metadata.override_role_uids ==
                        std::vector<std::int64_t>({312, 313, 314}),
                    "dynamic MATMUL records ordered A/B/C role UIDs");
        tests.check(dynamic_matmul_compilation.imported_mlir.find(
                        "memref.dim") != std::string::npos,
                    "dynamic MATMUL emits runtime extents");

        std::vector<float> dynamic_a(40, -77.0F);
        std::vector<float> dynamic_b(30, -88.0F);
        std::vector<float> dynamic_c(24, -99.0F);
        for (std::size_t batch = 0; batch < 2; ++batch) {
            for (std::size_t row = 0; row < 2; ++row) {
                for (std::size_t k = 0; k < 4; ++k) {
                    dynamic_a[batch * 10 + row * 4 + k] =
                        static_cast<float>(1 + batch * 8 + row * 4 + k);
                }
            }
        }
        for (std::size_t k = 0; k < 4; ++k) {
            for (std::size_t column = 0; column < 2; ++column) {
                dynamic_b[k + column * 4] =
                    static_cast<float>(1 + k + column);
            }
        }
        deepforge::runtime::VariantPack dynamic_matmul_pack{
            {312, dynamic_a.data()},
            {313, dynamic_b.data()},
            {314, dynamic_c.data()}};
        deepforge::runtime::OverrideUids dynamic_matmul_uids{312, 313, 314};
        deepforge::runtime::OverrideShapes dynamic_matmul_shapes{
            {2, 2, 4}, {1, 4, 2}, {2, 2, 2}};
        deepforge::runtime::OverrideStrides dynamic_matmul_strides{
            {10, 4, 1}, {8, 1, 4}, {6, 3, 1}};
        status = dynamic_matmul_compilation.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr,
            dynamic_matmul_pack, nullptr, dynamic_matmul_uids,
            dynamic_matmul_shapes, dynamic_matmul_strides);
        tests.good(status, "execute runtime shape-override MATMUL");
        bool dynamic_matmul_matches = true;
        for (std::size_t batch = 0; batch < 2; ++batch) {
            for (std::size_t row = 0; row < 2; ++row) {
                for (std::size_t column = 0; column < 2; ++column) {
                    float expected = 0.0F;
                    for (std::size_t k = 0; k < 4; ++k) {
                        expected += dynamic_a[batch * 10 + row * 4 + k] *
                                    dynamic_b[k + column * 4];
                    }
                    dynamic_matmul_matches =
                        dynamic_matmul_matches &&
                        close(dynamic_c[batch * 6 + row * 3 + column],
                              expected);
                }
            }
        }
        dynamic_matmul_matches = dynamic_matmul_matches &&
                                 dynamic_c[2] == -99.0F &&
                                 dynamic_c[5] == -99.0F &&
                                 dynamic_c[8] == -99.0F;
        tests.check(
            dynamic_matmul_matches,
            "dynamic MATMUL honors runtime K, batch broadcast, and strides");

        std::vector<std::uint8_t> dynamic_matmul_artifact;
        status = deepforge::compiler::serialize_artifact(
            dynamic_matmul_compilation, dynamic_matmul_artifact);
        tests.good(status, "serialize runtime shape-override MATMUL artifact");
        std::unique_ptr<deepforge::runtime::Executable> dynamic_matmul_loaded;
        deepforge::compiler::ArtifactInfo dynamic_matmul_info;
        if (status.is_good()) {
            status = deepforge::compiler::load_artifact_executable(
                dynamic_matmul_artifact, dynamic_matmul_loaded,
                &dynamic_matmul_info);
        }
        tests.good(status, "load runtime shape-override MATMUL artifact");
        tests.check(
            dynamic_matmul_info.format_version ==
                    deepforge::compiler::kArtifactFormatVersion &&
                dynamic_matmul_info.metadata.override_role_uids ==
                    std::vector<std::int64_t>({312, 313, 314}),
            "artifact v7 preserves MATMUL role UIDs");
        if (dynamic_matmul_loaded) {
            std::fill(dynamic_c.begin(), dynamic_c.end(), -99.0F);
            status = dynamic_matmul_loaded->execute(
                nullptr, dynamic_matmul_pack, nullptr, dynamic_matmul_uids,
                dynamic_matmul_shapes, dynamic_matmul_strides);
            tests.good(status,
                       "execute loaded runtime shape-override MATMUL artifact");
            tests.check(dynamic_c[0] == 30.0F && dynamic_c[10] == 208.0F &&
                            dynamic_c[11] == -99.0F,
                        "loaded dynamic MATMUL preserves role semantics");
        }

        auto invalid_k_shapes = dynamic_matmul_shapes;
        invalid_k_shapes[1] = {1, 3, 2};
        status = dynamic_matmul_compilation.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr,
            dynamic_matmul_pack, nullptr, dynamic_matmul_uids,
            invalid_k_shapes, dynamic_matmul_strides);
        tests.check(status.code() == deepforge::import::ErrorCode::kInvalidShape,
                    "dynamic MATMUL rejects inconsistent K extents");
        auto invalid_batch_shapes = dynamic_matmul_shapes;
        invalid_batch_shapes[2][0] = 1;
        status = dynamic_matmul_compilation.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr,
            dynamic_matmul_pack, nullptr, dynamic_matmul_uids,
            invalid_batch_shapes, dynamic_matmul_strides);
        tests.check(status.code() == deepforge::import::ErrorCode::kInvalidShape,
                    "dynamic MATMUL rejects an incorrect output batch extent");
        status = dynamic_matmul_compilation.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr,
            dynamic_matmul_pack, nullptr, {312}, {{2, 2, 4}}, {{10, 4, 1}});
        tests.check(status.code() == deepforge::import::ErrorCode::kInvalidShape,
                    "partial MATMUL overrides must preserve all shape relations");
        std::int64_t partial_workspace_size = -1;
        status = dynamic_matmul_compilation.executable->get_workspace_size(
            nullptr, partial_workspace_size, {313}, {{1, 5, 3}},
            {{15, 1, 5}});
        tests.good(status,
                   "partial MATMUL override may introduce a valid batch broadcast");
        tests.check(partial_workspace_size == 0,
                    "partial MATMUL override preserves the static workspace bound");
    }

    auto dynamic_matmul_virtual = runtime_shape_override_matmul_graph();
    dynamic_matmul_virtual["tensors"]["314"]["is_virtual"] = true;
    status = compile_document(dynamic_matmul_virtual);
    tests.check(status.code() ==
                    deepforge::import::ErrorCode::kUnsupportedOperation,
                "runtime MATMUL shape override rejects virtual arguments");
    auto mixed_matmul_override = matmul_override_graph();
    mixed_matmul_override["context"]["is_override_shape_enabled"] = true;
    status = compile_document(mixed_matmul_override);
    tests.check(
        status.code() == deepforge::import::ErrorCode::kUnsupportedOperation,
        "shape arrays cannot mix with MATMUL extent-override tensors");
    auto composed_dynamic_matmul = matmul_graph();
    composed_dynamic_matmul["context"]["is_override_shape_enabled"] = true;
    status = compile_document(composed_dynamic_matmul);
    tests.check(status.code() ==
                    deepforge::import::ErrorCode::kUnsupportedOperation,
                "runtime MATMUL override rejects composed graphs in C6.10");

    deepforge::import::SerializedGraph resample;
    status = parse_graph(resample_graph(), resample);
    tests.good(status, "parse supported integer resample modes");
    deepforge::compiler::CompilationResult resample_compilation;
    if (status.is_good()) {
        status = deepforge::compiler::compile_graph(resample, options,
                                                    resample_compilation);
    }
    tests.good(status, "compile supported integer resample modes");
    if (resample_compilation.executable) {
        std::vector<float> input{1.0F, 2.0F, 3.0F, 4.0F, 5.0F,
                                 6.0F, 7.0F, 8.0F, 9.0F};
        std::vector<float> avg_exclude(4, -99.0F);
        std::vector<float> avg_include(4, -99.0F);
        std::vector<float> max_pool(4, -99.0F);
        std::vector<float> nearest(16, -99.0F);
        deepforge::runtime::VariantPack pack{{401, input.data()},
                                             {410, avg_exclude.data()},
                                             {411, avg_include.data()},
                                             {412, max_pool.data()},
                                             {420, nearest.data()}};
        status = resample_compilation.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr);
        tests.good(status, "execute supported integer resample modes");
        constexpr std::array<float, 4> expected_exclude{1.0F, 2.5F, 5.5F,
                                                        7.0F};
        constexpr std::array<float, 4> expected_include{0.25F, 1.25F, 2.75F,
                                                        7.0F};
        constexpr std::array<float, 4> expected_max{1.0F, 3.0F, 7.0F, 9.0F};
        constexpr std::array<float, 16> expected_edge{
            1.0F, 1.0F, 2.0F, 3.0F, 1.0F, 1.0F, 2.0F, 3.0F,
            4.0F, 4.0F, 5.0F, 6.0F, 7.0F, 7.0F, 8.0F, 9.0F};
        auto vector_matches = [](auto const& actual, auto const& expected) {
            return std::equal(actual.begin(), actual.end(), expected.begin(),
                              [](float lhs, float rhs) {
                                  return std::fabs(lhs - rhs) <= 1.0e-6F;
                              });
        };
        tests.check(vector_matches(avg_exclude, expected_exclude) &&
                        vector_matches(avg_include, expected_include) &&
                        vector_matches(max_pool, expected_max),
                    "pooling modes match padded scalar references");
        tests.check(vector_matches(nearest, expected_edge),
                    "integer nearest mode matches edge reference");
    }

    auto bilinear = resample_graph();
    bilinear["nodes"][3]["resample_mode"] = "BILINEAR";
    bilinear["nodes"][3]["pre_padding"] = Json::array({1, 1});
    bilinear["nodes"][3]["post_padding"] = Json::array({1, 1});
    bilinear["nodes"][3]["stride"] = Json::array({1, 1});
    bilinear["nodes"][3]["window"] = Json::array({2, 2});
    bilinear["tensors"]["420"]["dim"] = Json::array({1, 1, 6, 6});
    bilinear["tensors"]["420"]["stride"] = Json::array({36, 36, 6, 1});
    status = compile_document(bilinear);
    tests.check(
        status.code() == deepforge::import::ErrorCode::kUnsupportedOperation,
        "bilinear resample is rejected when serialized fractions are lossy");

    auto invalid_broadcast = single_softplus_graph();
    invalid_broadcast["tensors"]["502"]["dim"] = Json::array({3});
    status = compile_document(invalid_broadcast);
    tests.check(status.code() == deepforge::import::ErrorCode::kInvalidShape,
                "incompatible pointwise output shape is rejected");

    auto nonscalar_pass_by_value = runtime_scalar_pointwise_graph();
    nonscalar_pass_by_value["tensors"]["602"]["dim"] =
        Json::array({1, 2});
    nonscalar_pass_by_value["tensors"]["602"]["stride"] =
        Json::array({2, 1});
    status = compile_document(nonscalar_pass_by_value);
    tests.check(
        status.code() == deepforge::import::ErrorCode::kUnsupportedOperation,
        "runtime pass-by-value rejects non-scalar tensors");

    auto output_pass_by_value = runtime_scalar_pointwise_graph();
    for (auto uid : {"601", "603"}) {
        output_pass_by_value["tensors"][uid]["dim"] = Json::array({1, 1});
        output_pass_by_value["tensors"][uid]["stride"] = Json::array({1, 1});
    }
    output_pass_by_value["tensors"]["603"]["is_pass_by_value"] = true;
    status = compile_document(output_pass_by_value);
    tests.check(
        status.code() == deepforge::import::ErrorCode::kUnsupportedOperation,
        "runtime pass-by-value rejects output tensors");

    auto fused_scalar = runtime_scalar_pointwise_graph();
    fused_scalar["tensors"]["602"]["pass_by_value"] = 2.5;
    status = compile_document(fused_scalar);
    tests.check(status.code() ==
                    deepforge::import::ErrorCode::kInvalidFieldType,
                "raw-number scalar payload is not producer-compatible");

    auto nonscalar_embedded = embedded_scalar_pointwise_graph();
    nonscalar_embedded["tensors"]["602"]["dim"] = Json::array({1, 2});
    nonscalar_embedded["tensors"]["602"]["stride"] = Json::array({2, 1});
    status = compile_document(nonscalar_embedded);
    tests.check(
        status.code() == deepforge::import::ErrorCode::kUnsupportedOperation,
        "embedded pass-by-value rejects non-scalar tensors");

    auto virtual_embedded = embedded_scalar_pointwise_graph();
    virtual_embedded["tensors"]["602"]["is_virtual"] = true;
    status = compile_document(virtual_embedded);
    tests.check(
        status.code() == deepforge::import::ErrorCode::kInvalidValue,
        "embedded pass-by-value rejects virtual tensors");

    auto virtual_scalar = runtime_scalar_pointwise_graph();
    virtual_scalar["tensors"]["604"] =
        tensor("SCALAR_X", 604, {1, 1}, {1, 1});
    virtual_scalar["tensors"]["605"] =
        tensor("SCALAR_Y", 605, {1, 1}, {1, 1});
    virtual_scalar["tensors"]["602"]["is_virtual"] = true;
    auto scalar_consumer = virtual_scalar["nodes"][0];
    auto scalar_producer = scalar_consumer;
    scalar_producer["name"] = "produce_virtual_scalar";
    scalar_producer["inputs"] =
        Json::object({{"IN_0", 604}, {"IN_1", 605}});
    scalar_producer["outputs"] = Json::object({{"OUT_0", 602}});
    virtual_scalar["nodes"] =
        Json::array({std::move(scalar_producer), std::move(scalar_consumer)});
    status = compile_document(virtual_scalar);
    tests.check(
        status.code() == deepforge::import::ErrorCode::kUnsupportedOperation,
        "runtime pass-by-value rejects virtual tensors");

    auto dynamic_scalar = runtime_scalar_pointwise_graph();
    dynamic_scalar["context"]["is_dynamic_shape_enabled"] = true;
    dynamic_scalar["context"]["is_override_shape_enabled"] = true;
    status = compile_document(dynamic_scalar);
    tests.check(
        status.code() == deepforge::import::ErrorCode::kUnsupportedOperation,
        "runtime pass-by-value is not a pointwise shape-override array");

    auto nonzero_padding = matmul_graph();
    nonzero_padding["nodes"][0]["padding_value"] = 1.0;
    status = compile_document(nonzero_padding);
    tests.good(status, "nonzero matmul padding is accepted");

    auto oversized_padding = matmul_graph();
    oversized_padding["nodes"][0]["padding_value"] = 1.0e300;
    status = compile_document(oversized_padding);
    tests.check(status.code() == deepforge::import::ErrorCode::kInvalidValue,
                "matmul padding must fit finite f32");

    auto dimension_override = matmul_graph();
    dimension_override["tensors"]["306"] =
        tensor("M_override", 306, {1}, {1});
    dimension_override["nodes"][0]["inputs"]["M_override"] = 306;
    status = compile_document(dimension_override);
    tests.check(
        status.code() == deepforge::import::ErrorCode::kUnsupportedDataType,
        "matmul dimension override rejects non-INT32 elements");

    auto invalid_override_shape = matmul_override_graph();
    invalid_override_shape["tensors"]["308"]["dim"] =
        Json::array({2, 2, 1});
    invalid_override_shape["tensors"]["308"]["stride"] =
        Json::array({2, 1, 1});
    status = compile_document(invalid_override_shape);
    tests.check(status.code() == deepforge::import::ErrorCode::kInvalidShape,
                "matmul override matrix dimensions must be singleton");

    auto index_output = resample_graph();
    index_output["tensors"]["422"] =
        tensor("Index", 422, {1, 1, 2, 2}, {4, 4, 2, 1});
    index_output["nodes"][0]["outputs"]["Index"] = 422;
    status = compile_document(index_output);
    tests.check(
        status.code() == deepforge::import::ErrorCode::kUnsupportedOperation,
        "resample index output is rejected until index data types are supported");

    return tests.finish();
}
