#include "DeepForge/Compiler/Codegen.h"
#include "DeepForge/Import/SerializedGraphImporter.h"

#include <cudnn_frontend/thirdparty/nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <string>
#include <utility>
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
            std::cerr << "FAIL: " << name << ": " << status.message()
                      << '\n';
        }
    }

    int finish() const {
        if (failures_ == 0) {
            std::cout << "deepforge-training-conv: " << checks_
                      << " checks passed\n";
            return 0;
        }
        std::cerr << "deepforge-training-conv: " << failures_ << " of "
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
            bool is_virtual = false) {
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

Json convolution_node(std::string tag,
                      Json inputs,
                      Json outputs,
                      std::vector<std::int64_t> pre_padding,
                      std::vector<std::int64_t> post_padding,
                      std::vector<std::int64_t> stride,
                      std::vector<std::int64_t> dilation,
                      std::string math_mode = "CROSS_CORRELATION") {
    return Json{{"tag", tag},
                {"name", "test_" + tag},
                {"inputs", std::move(inputs)},
                {"outputs", std::move(outputs)},
                {"compute_data_type", "FLOAT"},
                {"pre_padding", std::move(pre_padding)},
                {"post_padding", std::move(post_padding)},
                {"stride", std::move(stride)},
                {"dilation", std::move(dilation)},
                {"math_mode", std::move(math_mode)}};
}

Json pointwise_relu(std::int64_t input, std::int64_t output) {
    return Json{{"tag", "POINTWISE"},
                {"name", "relu_after_conv"},
                {"inputs", Json::object({{"IN_0", input}})},
                {"outputs", Json::object({{"OUT_0", output}})},
                {"compute_data_type", "FLOAT"},
                {"mode", "RELU_FWD"},
                {"axis", nullptr},
                {"relu_lower_clip", 0.0},
                {"relu_upper_clip", nullptr},
                {"relu_lower_clip_slope", 0.0},
                {"swish_beta", nullptr},
                {"elu_alpha", nullptr},
                {"softplus_beta", nullptr}};
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

deepforge::import::Status compile_document(
    Json const& document,
    deepforge::compiler::CompilationResult& compilation,
    deepforge::import::SerializedGraph* parsed = nullptr) {
    deepforge::import::SerializedGraph graph;
    auto status = parse_graph(document, graph);
    if (status.is_bad()) {
        return status;
    }
    status = deepforge::compiler::compile_graph(
        graph, deepforge::compiler::CompileOptions{}, compilation);
    if (parsed != nullptr) {
        *parsed = std::move(graph);
    }
    return status;
}

std::size_t offset4(std::vector<std::int64_t> const& dimensions,
                    std::int64_t n,
                    std::int64_t c,
                    std::int64_t h,
                    std::int64_t w) {
    return static_cast<std::size_t>(
        ((n * dimensions[1] + c) * dimensions[2] + h) * dimensions[3] + w);
}

std::vector<float> grouped_fprop_reference(
    std::vector<float> const& x,
    std::vector<float> const& weight) {
    std::vector<std::int64_t> x_dim{1, 4, 5, 5};
    std::vector<std::int64_t> w_dim{6, 2, 2, 2};
    std::vector<std::int64_t> y_dim{1, 6, 2, 3};
    std::vector<float> y(36, 0.0F);
    for (std::int64_t k = 0; k < 6; ++k) {
        auto const group = k / 3;
        for (std::int64_t p = 0; p < 2; ++p) {
            for (std::int64_t q = 0; q < 3; ++q) {
                double sum = 0.0;
                for (std::int64_t c = 0; c < 2; ++c) {
                    for (std::int64_t r = 0; r < 2; ++r) {
                        for (std::int64_t s = 0; s < 2; ++s) {
                            auto const h = p * 2 - 1 + r * 2;
                            auto const w = q * 2 + s;
                            if (h < 0 || h >= 5 || w < 0 || w >= 5) {
                                continue;
                            }
                            auto const x_index =
                                offset4(x_dim, 0, group * 2 + c, h, w);
                            auto const w_index =
                                offset4(w_dim, k, c, r, s);
                            sum += static_cast<double>(x[x_index]) *
                                   weight[w_index];
                        }
                    }
                }
                y[offset4(y_dim, 0, k, p, q)] = static_cast<float>(sum);
            }
        }
    }
    return y;
}

bool close_vectors(std::vector<float> const& actual,
                   std::vector<float> const& expected,
                   float tolerance = 2.0e-5F) {
    if (actual.size() != expected.size()) {
        return false;
    }
    for (std::size_t index = 0; index < actual.size(); ++index) {
        auto const bound = tolerance * (1.0F + std::fabs(expected[index]));
        if (std::fabs(actual[index] - expected[index]) > bound) {
            std::cerr << "mismatch at " << index << ": expected "
                      << expected[index] << " got " << actual[index] << '\n';
            return false;
        }
    }
    return true;
}

Json grouped_fprop_graph() {
    Json tensors = Json::object();
    tensors["1"] = tensor("X", 1, {1, 4, 5, 5}, {100, 25, 5, 1});
    tensors["2"] = tensor("W", 2, {6, 2, 2, 2}, {8, 4, 2, 1});
    tensors["3"] = tensor("Y", 3, {1, 6, 2, 3}, {36, 6, 3, 1});
    auto node = convolution_node(
        "CONV_FPROP", Json::object({{"X", 1}, {"W", 2}}),
        Json::object({{"Y", 3}}), {1, 0}, {0, 1}, {2, 2}, {2, 1});
    return graph_document(3001, "grouped_fprop", Json::array({node}),
                          std::move(tensors));
}

Json conv1d_graph(std::string const& tag) {
    Json tensors = Json::object();
    Json node;
    if (tag == "CONV_FPROP") {
        tensors["11"] = tensor("X", 11, {1, 2, 4}, {8, 4, 1});
        tensors["12"] = tensor("W", 12, {3, 2, 3}, {6, 3, 1});
        tensors["13"] = tensor("Y", 13, {1, 3, 4}, {12, 4, 1});
        node = convolution_node(
            tag, Json::object({{"X", 11}, {"W", 12}}),
            Json::object({{"Y", 13}}), {1}, {1}, {1}, {1});
    } else if (tag == "CONV_DGRAD") {
        tensors["12"] = tensor("W", 12, {3, 2, 3}, {6, 3, 1});
        tensors["14"] = tensor("DY", 14, {1, 3, 4}, {12, 4, 1});
        tensors["15"] = tensor("DX", 15, {1, 2, 4}, {8, 4, 1});
        node = convolution_node(
            tag, Json::object({{"W", 12}, {"DY", 14}}),
            Json::object({{"DX", 15}}), {1}, {1}, {1}, {1});
    } else {
        tensors["11"] = tensor("X", 11, {1, 2, 4}, {8, 4, 1});
        tensors["14"] = tensor("DY", 14, {1, 3, 4}, {12, 4, 1});
        tensors["16"] = tensor("DW", 16, {3, 2, 3}, {6, 3, 1});
        node = convolution_node(
            tag, Json::object({{"X", 11}, {"DY", 14}}),
            Json::object({{"DW", 16}}), {1}, {1}, {1}, {1});
    }
    return graph_document(3002, tag, Json::array({node}),
                          std::move(tensors));
}

std::vector<double> fprop1d(std::vector<float> const& x,
                            std::vector<float> const& weight) {
    std::vector<double> y(12, 0.0);
    for (std::int64_t k = 0; k < 3; ++k) {
        for (std::int64_t p = 0; p < 4; ++p) {
            for (std::int64_t c = 0; c < 2; ++c) {
                for (std::int64_t r = 0; r < 3; ++r) {
                    auto const coordinate = p - 1 + r;
                    if (coordinate >= 0 && coordinate < 4) {
                        y[static_cast<std::size_t>(k * 4 + p)] +=
                            static_cast<double>(
                                x[static_cast<std::size_t>(c * 4 +
                                                           coordinate)]) *
                            weight[static_cast<std::size_t>((k * 2 + c) * 3 +
                                                            r)];
                    }
                }
            }
        }
    }
    return y;
}

double loss1d(std::vector<float> const& x,
              std::vector<float> const& weight,
              std::vector<float> const& dy) {
    auto y = fprop1d(x, weight);
    double loss = 0.0;
    for (std::size_t index = 0; index < y.size(); ++index) {
        loss += y[index] * dy[index];
    }
    return loss;
}

std::vector<float> finite_difference(
    std::vector<float> values,
    std::vector<float> const& other,
    std::vector<float> const& dy,
    bool differentiate_x) {
    constexpr float epsilon = 1.0e-3F;
    std::vector<float> gradient(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        values[index] += epsilon;
        auto positive = differentiate_x ? loss1d(values, other, dy)
                                        : loss1d(other, values, dy);
        values[index] -= 2.0F * epsilon;
        auto negative = differentiate_x ? loss1d(values, other, dy)
                                        : loss1d(other, values, dy);
        gradient[index] = static_cast<float>(
            (positive - negative) / (2.0 * static_cast<double>(epsilon)));
        values[index] += epsilon;
    }
    return gradient;
}

Json mixed_conv_pointwise_graph() {
    Json tensors = Json::object();
    tensors["21"] = tensor("X", 21, {1, 1, 3, 3}, {9, 1, 3, 1});
    tensors["22"] = tensor("W", 22, {1, 1, 2, 2}, {4, 1, 2, 1});
    tensors["23"] = tensor("V", 23, {1, 1, 2, 2}, {4, 1, 2, 1}, true);
    tensors["24"] = tensor("Y", 24, {1, 1, 2, 2}, {4, 1, 2, 1});
    auto conv = convolution_node(
        "CONV_FPROP", Json::object({{"X", 21}, {"W", 22}}),
        Json::object({{"Y", 23}}), {0, 0}, {0, 0}, {1, 1}, {1, 1});
    return graph_document(3003, "mixed_conv_pointwise",
                          Json::array({conv, pointwise_relu(23, 24)}),
                          std::move(tensors));
}

}  // namespace

int main() {
    TestRunner tests;

    deepforge::compiler::CompilationResult grouped_compilation;
    auto status = compile_document(grouped_fprop_graph(), grouped_compilation);
    tests.good(status, "compile grouped strided dilated fprop");
    if (status.is_good() && grouped_compilation.executable) {
        std::vector<float> x(100);
        std::vector<float> weight(48);
        for (std::size_t index = 0; index < x.size(); ++index) {
            x[index] = static_cast<float>(static_cast<int>(index % 11) - 5) /
                       7.0F;
        }
        for (std::size_t index = 0; index < weight.size(); ++index) {
            weight[index] =
                static_cast<float>(static_cast<int>(index % 7) - 3) / 5.0F;
        }
        std::vector<float> y(36, -99.0F);
        deepforge::runtime::VariantPack pack{{1, x.data()},
                                             {2, weight.data()},
                                             {3, y.data()}};
        status = grouped_compilation.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr);
        tests.good(status, "execute grouped strided dilated fprop");
        tests.check(close_vectors(y, grouped_fprop_reference(x, weight)),
                    "generalized fprop matches grouped scalar reference");
    }

    std::vector<float> x{0.2F, -0.4F, 0.7F, 1.1F,
                         -0.3F, 0.5F, -0.8F, 0.9F};
    std::vector<float> weight{
        0.1F, -0.2F, 0.3F, 0.4F, -0.5F, 0.6F,
        -0.7F, 0.8F, 0.2F, -0.1F, 0.9F, -0.4F,
        0.3F, 0.2F, -0.6F, 0.5F, -0.8F, 0.7F};
    std::vector<float> dy{0.2F, -0.3F, 0.4F, -0.5F,
                          0.6F, 0.1F, -0.2F, 0.3F,
                          -0.4F, 0.5F, 0.7F, -0.1F};

    deepforge::compiler::CompilationResult fprop_compilation;
    status = compile_document(conv1d_graph("CONV_FPROP"), fprop_compilation);
    tests.good(status, "compile rank-3 fprop");
    if (status.is_good() && fprop_compilation.executable) {
        std::vector<float> y(12, -99.0F);
        deepforge::runtime::VariantPack pack{{11, x.data()},
                                             {12, weight.data()},
                                             {13, y.data()}};
        status = fprop_compilation.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr);
        tests.good(status, "execute rank-3 fprop");
        auto expected_double = fprop1d(x, weight);
        std::vector<float> expected(expected_double.begin(),
                                    expected_double.end());
        tests.check(close_vectors(y, expected),
                    "rank-3 fprop matches independent reference");
    }

    deepforge::compiler::CompilationResult dgrad_compilation;
    status = compile_document(conv1d_graph("CONV_DGRAD"), dgrad_compilation);
    tests.good(status, "compile rank-3 dgrad");
    if (status.is_good() && dgrad_compilation.executable) {
        std::vector<float> dx(8, -99.0F);
        deepforge::runtime::VariantPack pack{{12, weight.data()},
                                             {14, dy.data()},
                                             {15, dx.data()}};
        status = dgrad_compilation.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr);
        tests.good(status, "execute rank-3 dgrad");
        tests.check(close_vectors(dx, finite_difference(x, weight, dy, true),
                                  2.0e-3F),
                    "dgrad matches finite-difference gradient");
    }

    deepforge::compiler::CompilationResult wgrad_compilation;
    status = compile_document(conv1d_graph("CONV_WGRAD"), wgrad_compilation);
    tests.good(status, "compile rank-3 wgrad");
    if (status.is_good() && wgrad_compilation.executable) {
        std::vector<float> dw(18, -99.0F);
        deepforge::runtime::VariantPack pack{{11, x.data()},
                                             {14, dy.data()},
                                             {16, dw.data()}};
        status = wgrad_compilation.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr);
        tests.good(status, "execute rank-3 wgrad");
        tests.check(close_vectors(dw, finite_difference(weight, x, dy, false),
                                  2.0e-3F),
                    "wgrad matches finite-difference gradient");
    }

    deepforge::compiler::CompilationResult mixed_compilation;
    deepforge::import::SerializedGraph mixed_graph;
    status = compile_document(mixed_conv_pointwise_graph(), mixed_compilation,
                              &mixed_graph);
    tests.good(status, "compile mixed legacy-conv and pointwise graph");
    tests.check(
        !mixed_graph.nodes.empty() &&
            std::holds_alternative<deepforge::import::ConvFpropDesc>(
                mixed_graph.nodes.front().attributes),
        "mixed graph exercises legacy ConvFpropDesc normalization");
    if (status.is_good() && mixed_compilation.executable) {
        std::vector<float> mixed_x{1.0F, 2.0F, 3.0F,
                                   4.0F, 5.0F, 6.0F,
                                   7.0F, 8.0F, 9.0F};
        std::vector<float> mixed_w{1.0F, -2.0F, 0.5F, 1.0F};
        std::vector<float> mixed_y(4, -99.0F);
        AlignedBytes workspace(static_cast<std::size_t>(
            mixed_compilation.executable->get_workspace_size()));
        deepforge::runtime::VariantPack pack{{21, mixed_x.data()},
                                             {22, mixed_w.data()},
                                             {24, mixed_y.data()}};
        status = mixed_compilation.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr, pack,
            workspace.get());
        tests.good(status, "execute mixed conv-pointwise graph");
        std::vector<float> expected(4);
        for (std::int64_t p = 0; p < 2; ++p) {
            for (std::int64_t q = 0; q < 2; ++q) {
                float sum = 0.0F;
                for (std::int64_t r = 0; r < 2; ++r) {
                    for (std::int64_t s = 0; s < 2; ++s) {
                        sum += mixed_x[static_cast<std::size_t>(
                                   (p + r) * 3 + q + s)] *
                               mixed_w[static_cast<std::size_t>(r * 2 + s)];
                    }
                }
                expected[static_cast<std::size_t>(p * 2 + q)] =
                    std::max(sum, 0.0F);
            }
        }
        tests.check(close_vectors(mixed_y, expected),
                    "mixed graph composes convolution with C2 pointwise");
    }

    auto invalid_group = grouped_fprop_graph();
    invalid_group["tensors"]["2"]["dim"][1] = 3;
    invalid_group["tensors"]["2"]["stride"] =
        Json::array({12, 4, 2, 1});
    deepforge::compiler::CompilationResult rejected;
    status = compile_document(invalid_group, rejected);
    tests.check(status.code() == deepforge::import::ErrorCode::kInvalidShape,
                "invalid inferred convolution group count is rejected");

    return tests.finish();
}
