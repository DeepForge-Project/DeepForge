#include "DeepForge/Compiler/Artifact.h"
#include "DeepForge/Compiler/Codegen.h"
#include "DeepForge/Import/SerializedGraphImporter.h"

#include <cudnn_frontend/thirdparty/nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <numeric>
#include <span>
#include <string>
#include <string_view>
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
            std::cout << "deepforge-training-norm: " << checks_
                      << " checks passed\n";
            return 0;
        }
        std::cerr << "deepforge-training-norm: " << failures_ << " of "
                  << checks_ << " checks failed\n";
        return 1;
    }

private:
    int checks_ = 0;
    int failures_ = 0;
};

std::size_t element_count(std::vector<std::int64_t> const& dimensions) {
    return std::accumulate(
        dimensions.begin(), dimensions.end(), std::size_t{1},
        [](std::size_t lhs, std::int64_t rhs) {
            return lhs * static_cast<std::size_t>(rhs);
        });
}

std::vector<std::int64_t> row_major_strides(
    std::vector<std::int64_t> const& dimensions) {
    std::vector<std::int64_t> strides(dimensions.size(), 1);
    for (std::size_t axis = dimensions.size(); axis > 1; --axis) {
        strides[axis - 2] = strides[axis - 1] * dimensions[axis - 1];
    }
    return strides;
}

Json tensor(std::string name,
            std::int64_t uid,
            std::vector<std::int64_t> dimensions,
            bool is_virtual = false) {
    auto strides = row_major_strides(dimensions);
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
    deepforge::compiler::CompilationResult& compilation) {
    deepforge::import::SerializedGraph graph;
    auto status = parse_graph(document, graph);
    if (status.is_bad()) return status;
    return deepforge::compiler::compile_graph(
        graph, deepforge::compiler::CompileOptions{}, compilation);
}

std::vector<std::int64_t> coordinates(
    std::size_t linear,
    std::vector<std::int64_t> const& dimensions) {
    std::vector<std::int64_t> result(dimensions.size());
    for (std::size_t axis = dimensions.size(); axis > 0; --axis) {
        auto const dimension = static_cast<std::size_t>(dimensions[axis - 1]);
        result[axis - 1] = static_cast<std::int64_t>(linear % dimension);
        linear /= dimension;
    }
    return result;
}

std::size_t mapped_index(
    std::vector<std::int64_t> const& source_coordinates,
    std::vector<std::int64_t> const& target_dimensions) {
    std::size_t result = 0;
    for (std::size_t axis = 0; axis < target_dimensions.size(); ++axis) {
        result *= static_cast<std::size_t>(target_dimensions[axis]);
        if (target_dimensions[axis] != 1) {
            result += static_cast<std::size_t>(source_coordinates[axis]);
        }
    }
    return result;
}

struct NormCase {
    std::string forward_tag;
    std::string backward_tag;
    std::vector<std::int64_t> x_dim;
    std::vector<std::int64_t> scale_dim;
    std::vector<std::int64_t> stats_dim;
    bool rms = false;
    bool backward_epsilon = false;
};

struct NormReference {
    std::vector<float> y;
    std::vector<float> mean;
    std::vector<float> inverse;
    std::vector<float> dx;
    std::vector<float> dscale;
    std::vector<float> dbias;
};

NormReference norm_reference(NormCase const& configuration,
                             std::vector<float> const& x,
                             std::vector<float> const& scale,
                             std::vector<float> const& bias,
                             std::vector<float> const& dy,
                             float epsilon) {
    NormReference result;
    auto const x_count = element_count(configuration.x_dim);
    auto const stats_count = element_count(configuration.stats_dim);
    auto const scale_count = element_count(configuration.scale_dim);
    result.y.resize(x_count);
    result.mean.assign(stats_count, 0.0F);
    result.inverse.assign(stats_count, 0.0F);
    result.dx.resize(x_count);
    result.dscale.assign(scale_count, 0.0F);
    result.dbias.assign(scale_count, 0.0F);
    std::vector<double> sum(stats_count, 0.0);
    std::vector<double> square_sum(stats_count, 0.0);
    std::vector<std::size_t> count(stats_count, 0);
    for (std::size_t index = 0; index < x_count; ++index) {
        auto coordinate = coordinates(index, configuration.x_dim);
        auto const stats_index =
            mapped_index(coordinate, configuration.stats_dim);
        sum[stats_index] += x[index];
        square_sum[stats_index] +=
            static_cast<double>(x[index]) * x[index];
        ++count[stats_index];
    }
    for (std::size_t index = 0; index < stats_count; ++index) {
        auto const mean = sum[index] / static_cast<double>(count[index]);
        auto const mean_square =
            square_sum[index] / static_cast<double>(count[index]);
        auto const variance = configuration.rms
                                  ? mean_square
                                  : std::max(0.0, mean_square - mean * mean);
        result.mean[index] =
            configuration.rms ? 0.0F : static_cast<float>(mean);
        result.inverse[index] =
            static_cast<float>(1.0 / std::sqrt(variance + epsilon));
    }
    std::vector<double> group_g(stats_count, 0.0);
    std::vector<double> group_product(stats_count, 0.0);
    for (std::size_t index = 0; index < x_count; ++index) {
        auto coordinate = coordinates(index, configuration.x_dim);
        auto const stats_index =
            mapped_index(coordinate, configuration.stats_dim);
        auto const scale_index =
            mapped_index(coordinate, configuration.scale_dim);
        auto const normalized =
            (x[index] - result.mean[stats_index]) *
            result.inverse[stats_index];
        result.y[index] = normalized * scale[scale_index] + bias[scale_index];
        result.dscale[scale_index] += dy[index] * normalized;
        result.dbias[scale_index] += dy[index];
        auto const g = static_cast<double>(dy[index]) * scale[scale_index];
        group_g[stats_index] += g;
        group_product[stats_index] +=
            g * (configuration.rms ? x[index] : normalized);
    }
    for (std::size_t index = 0; index < x_count; ++index) {
        auto coordinate = coordinates(index, configuration.x_dim);
        auto const stats_index =
            mapped_index(coordinate, configuration.stats_dim);
        auto const scale_index =
            mapped_index(coordinate, configuration.scale_dim);
        auto const g = static_cast<double>(dy[index]) * scale[scale_index];
        auto const inverse = static_cast<double>(result.inverse[stats_index]);
        auto const group_count = static_cast<double>(count[stats_index]);
        if (configuration.rms) {
            result.dx[index] = static_cast<float>(
                inverse *
                (g - static_cast<double>(x[index]) * inverse * inverse *
                         group_product[stats_index] / group_count));
        } else {
            auto const xhat =
                (x[index] - result.mean[stats_index]) * inverse;
            result.dx[index] = static_cast<float>(
                inverse *
                (g - group_g[stats_index] / group_count -
                 xhat * group_product[stats_index] / group_count));
        }
    }
    return result;
}

bool close_vectors(std::vector<float> const& actual,
                   std::vector<float> const& expected,
                   float tolerance = 4.0e-5F) {
    if (actual.size() != expected.size()) return false;
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

Json norm_forward_graph(NormCase const& configuration) {
    Json tensors = Json::object();
    tensors["101"] = tensor("X", 101, configuration.x_dim);
    tensors["102"] = tensor("SCALE", 102, configuration.scale_dim);
    tensors["103"] = tensor("BIAS", 103, configuration.scale_dim);
    tensors["104"] = tensor("EPSILON", 104,
                             std::vector<std::int64_t>(
                                 configuration.x_dim.size(), 1));
    tensors["105"] = tensor("Y", 105, configuration.x_dim);
    Json outputs = Json::object({{"Y", 105}});
    if (!configuration.rms) {
        tensors["106"] = tensor("MEAN", 106, configuration.stats_dim);
        outputs["MEAN"] = 106;
    }
    tensors["107"] = tensor("INV", 107, configuration.stats_dim);
    outputs["INV_VARIANCE"] = 107;
    Json node{{"tag", configuration.forward_tag},
              {"name", "forward_" + configuration.forward_tag},
              {"inputs",
               Json::object({{"X", 101},
                             {"SCALE", 102},
                             {"BIAS", 103},
                             {"EPSILON", 104}})},
              {"outputs", std::move(outputs)},
              {"compute_data_type", "FLOAT"},
              {"forward_phase", "TRAINING"}};
    return graph_document(4001, configuration.forward_tag,
                          Json::array({node}), std::move(tensors));
}

Json norm_backward_graph(NormCase const& configuration) {
    Json tensors = Json::object();
    tensors["201"] = tensor("DY", 201, configuration.x_dim);
    tensors["202"] = tensor("X", 202, configuration.x_dim);
    tensors["203"] = tensor("SCALE", 203, configuration.scale_dim);
    Json inputs = Json::object(
        {{"DY", 201}, {"X", 202}, {"SCALE", 203}});
    if (!configuration.rms) {
        tensors["204"] = tensor("MEAN", 204, configuration.stats_dim);
        inputs["MEAN"] = 204;
    }
    tensors["205"] = tensor("INV", 205, configuration.stats_dim);
    inputs["INV_VARIANCE"] = 205;
    if (configuration.backward_epsilon) {
        tensors["206"] = tensor("EPSILON", 206,
                                 std::vector<std::int64_t>(
                                     configuration.x_dim.size(), 1));
        inputs["EPSILON"] = 206;
    }
    tensors["207"] = tensor("DX", 207, configuration.x_dim);
    tensors["208"] = tensor("DSCALE", 208, configuration.scale_dim);
    tensors["209"] = tensor("DBIAS", 209, configuration.scale_dim);
    Json node{{"tag", configuration.backward_tag},
              {"name", "backward_" + configuration.backward_tag},
              {"inputs", std::move(inputs)},
              {"outputs",
               Json::object({{"DX", 207},
                             {"DSCALE", 208},
                             {"DBIAS", 209}})},
              {"compute_data_type", "FLOAT"}};
    if (configuration.backward_tag == "DBN") {
        node["peer_stats"] = Json::array();
    }
    return graph_document(4002, configuration.backward_tag,
                          Json::array({node}), std::move(tensors));
}

double norm_loss(NormCase const& configuration,
                 std::vector<float> const& x,
                 std::vector<float> const& scale,
                 std::vector<float> const& bias,
                 std::vector<float> const& dy,
                 float epsilon) {
    auto reference =
        norm_reference(configuration, x, scale, bias, dy, epsilon);
    double result = 0.0;
    for (std::size_t index = 0; index < dy.size(); ++index) {
        result += static_cast<double>(reference.y[index]) * dy[index];
    }
    return result;
}

std::vector<float> finite_difference_norm(
    NormCase const& configuration,
    std::vector<float> values,
    std::vector<float> const& other,
    std::vector<float> const& bias,
    std::vector<float> const& dy,
    float epsilon,
    bool differentiate_x) {
    constexpr float step = 2.0e-3F;
    std::vector<float> gradient(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        values[index] += step;
        auto positive = differentiate_x
                            ? norm_loss(configuration, values, other, bias, dy,
                                        epsilon)
                            : norm_loss(configuration, other, values, bias, dy,
                                        epsilon);
        values[index] -= 2.0F * step;
        auto negative = differentiate_x
                            ? norm_loss(configuration, values, other, bias, dy,
                                        epsilon)
                            : norm_loss(configuration, other, values, bias, dy,
                                        epsilon);
        gradient[index] = static_cast<float>(
            (positive - negative) / (2.0 * static_cast<double>(step)));
        values[index] += step;
    }
    return gradient;
}

void run_norm_case(TestRunner& tests,
                   NormCase const& configuration,
                   bool finite_difference_check) {
    std::vector<float> x{0.2F, -0.5F, 0.7F, 1.1F,
                         -0.3F, 0.4F, -0.8F, 0.9F};
    x.resize(element_count(configuration.x_dim));
    std::vector<float> scale(element_count(configuration.scale_dim));
    std::vector<float> bias(scale.size());
    for (std::size_t index = 0; index < scale.size(); ++index) {
        scale[index] = 0.8F + 0.1F * static_cast<float>(index);
        bias[index] = -0.2F + 0.05F * static_cast<float>(index);
    }
    std::vector<float> dy{0.3F, -0.2F, 0.5F, -0.7F,
                          0.4F, 0.1F, -0.6F, 0.8F};
    dy.resize(x.size());
    constexpr float epsilon = 1.0e-4F;
    std::vector<float> epsilon_buffer{epsilon};
    auto reference =
        norm_reference(configuration, x, scale, bias, dy, epsilon);

    deepforge::compiler::CompilationResult forward;
    auto status = compile_document(norm_forward_graph(configuration), forward);
    tests.good(status, "compile " + configuration.forward_tag);
    if (status.is_good() && forward.executable) {
        std::vector<float> y(x.size(), -99.0F);
        std::vector<float> mean(reference.mean.size(), -99.0F);
        std::vector<float> inverse(reference.inverse.size(), -99.0F);
        deepforge::runtime::VariantPack pack{{101, x.data()},
                                             {102, scale.data()},
                                             {103, bias.data()},
                                             {104, epsilon_buffer.data()},
                                             {105, y.data()},
                                             {107, inverse.data()}};
        if (!configuration.rms) pack.emplace(106, mean.data());
        status = forward.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr);
        tests.good(status, "execute " + configuration.forward_tag);
        tests.check(close_vectors(y, reference.y) &&
                        close_vectors(inverse, reference.inverse) &&
                        (configuration.rms ||
                         close_vectors(mean, reference.mean)),
                    configuration.forward_tag +
                        " matches independent forward reference");
    }

    deepforge::compiler::CompilationResult backward;
    status = compile_document(norm_backward_graph(configuration), backward);
    tests.good(status, "compile " + configuration.backward_tag);
    if (status.is_good() && backward.executable) {
        std::vector<float> dx(x.size(), -99.0F);
        std::vector<float> dscale(scale.size(), -99.0F);
        std::vector<float> dbias(scale.size(), -99.0F);
        deepforge::runtime::VariantPack pack{{201, dy.data()},
                                             {202, x.data()},
                                             {203, scale.data()},
                                             {205, reference.inverse.data()},
                                             {207, dx.data()},
                                             {208, dscale.data()},
                                             {209, dbias.data()}};
        if (!configuration.rms) pack.emplace(204, reference.mean.data());
        if (configuration.backward_epsilon) {
            pack.emplace(206, epsilon_buffer.data());
        }
        status = backward.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr);
        tests.good(status, "execute " + configuration.backward_tag);
        tests.check(close_vectors(dx, reference.dx, 8.0e-5F) &&
                        close_vectors(dscale, reference.dscale, 8.0e-5F) &&
                        close_vectors(dbias, reference.dbias, 8.0e-5F),
                    configuration.backward_tag +
                        " matches independent backward reference");
        if (finite_difference_check) {
            tests.check(
                close_vectors(
                    dx,
                    finite_difference_norm(configuration, x, scale, bias, dy,
                                           epsilon, true),
                    3.0e-3F) &&
                    close_vectors(
                        dscale,
                        finite_difference_norm(configuration, scale, x, bias,
                                               dy, epsilon, false),
                        3.0e-3F),
                configuration.backward_tag +
                    " matches finite-difference gradients");
        }
    }
}

Json batchnorm_graph(std::string tag) {
    std::vector<std::int64_t> x_dim{2, 2, 2};
    std::vector<std::int64_t> channel_dim{1, 2, 1};
    std::vector<std::int64_t> scalar_dim{1, 1, 1};
    Json tensors = Json::object();
    Json node;
    if (tag == "BATCHNORM") {
        for (auto item : {
                 std::pair<std::string, Json>{"301", tensor("X", 301, x_dim)},
                 {"302", tensor("SCALE", 302, channel_dim)},
                 {"303", tensor("BIAS", 303, channel_dim)},
                 {"304", tensor("EPSILON", 304, scalar_dim)},
                 {"305", tensor("PREV_MEAN", 305, channel_dim)},
                 {"306", tensor("PREV_VAR", 306, channel_dim)},
                 {"307", tensor("MOMENTUM", 307, scalar_dim)},
                 {"308", tensor("Y", 308, x_dim)},
                 {"309", tensor("MEAN", 309, channel_dim)},
                 {"310", tensor("INV", 310, channel_dim)},
                 {"311", tensor("NEXT_MEAN", 311, channel_dim)},
                 {"312", tensor("NEXT_VAR", 312, channel_dim)}}) {
            tensors[item.first] = item.second;
        }
        node = Json{{"tag", tag},
                    {"name", "batchnorm_training"},
                    {"inputs",
                     Json::object({{"X", 301},
                                   {"SCALE", 302},
                                   {"BIAS", 303},
                                   {"EPSILON", 304},
                                   {"PREV_RUNNING_MEAN", 305},
                                   {"PREV_RUNNING_VAR", 306},
                                   {"MOMENTUM", 307}})},
                    {"outputs",
                     Json::object({{"Y", 308},
                                   {"MEAN", 309},
                                   {"INV_VARIANCE", 310},
                                   {"NEXT_RUNNING_MEAN", 311},
                                   {"NEXT_RUNNING_VAR", 312}})},
                    {"compute_data_type", "FLOAT"},
                    {"peer_stats", Json::array()}};
    } else if (tag == "BATCHNORM_INFERENCE") {
        tensors["301"] = tensor("X", 301, x_dim);
        tensors["302"] = tensor("SCALE", 302, channel_dim);
        tensors["303"] = tensor("BIAS", 303, channel_dim);
        tensors["309"] = tensor("MEAN", 309, channel_dim);
        tensors["310"] = tensor("INV", 310, channel_dim);
        tensors["308"] = tensor("Y", 308, x_dim);
        node = Json{{"tag", tag},
                    {"name", "batchnorm_inference"},
                    {"inputs",
                     Json::object({{"X", 301},
                                   {"SCALE", 302},
                                   {"BIAS", 303},
                                   {"MEAN", 309},
                                   {"INV_VARIANCE", 310}})},
                    {"outputs", Json::object({{"Y", 308}})},
                    {"compute_data_type", "FLOAT"}};
    } else if (tag == "DBN") {
        tensors["301"] = tensor("X", 301, x_dim);
        tensors["302"] = tensor("SCALE", 302, channel_dim);
        tensors["309"] = tensor("MEAN", 309, channel_dim);
        tensors["310"] = tensor("INV", 310, channel_dim);
        tensors["313"] = tensor("DY", 313, x_dim);
        tensors["314"] = tensor("DX", 314, x_dim);
        tensors["315"] = tensor("DSCALE", 315, channel_dim);
        tensors["316"] = tensor("DBIAS", 316, channel_dim);
        node = Json{{"tag", tag},
                    {"name", "dbn"},
                    {"inputs",
                     Json::object({{"DY", 313},
                                   {"X", 301},
                                   {"SCALE", 302},
                                   {"MEAN", 309},
                                   {"INV_VARIANCE", 310}})},
                    {"outputs",
                     Json::object({{"DX", 314},
                                   {"DSCALE", 315},
                                   {"DBIAS", 316}})},
                    {"compute_data_type", "FLOAT"},
                    {"peer_stats", Json::array()}};
    } else {
        tensors["301"] = tensor("X", 301, x_dim);
        tensors["302"] = tensor("SCALE", 302, channel_dim);
        tensors["309"] = tensor("MEAN", 309, channel_dim);
        tensors["310"] = tensor("INV", 310, channel_dim);
        tensors["313"] = tensor("DY", 313, x_dim);
        for (std::int64_t uid = 315; uid <= 319; ++uid) {
            tensors[std::to_string(uid)] =
                tensor("O" + std::to_string(uid), uid, channel_dim);
        }
        node = Json{{"tag", "DBN_WEIGHT"},
                    {"name", "dbn_weight"},
                    {"inputs",
                     Json::object({{"DY", 313},
                                   {"X", 301},
                                   {"SCALE", 302},
                                   {"MEAN", 309},
                                   {"INV_VARIANCE", 310}})},
                    {"outputs",
                     Json::object({{"DSCALE", 315},
                                   {"DBIAS", 316},
                                   {"EQ_BIAS", 317},
                                   {"EQ_SCALE_DY", 318},
                                   {"EQ_SCALE_X", 319}})},
                    {"compute_data_type", "FLOAT"}};
    }
    return graph_document(4003, tag, Json::array({node}),
                          std::move(tensors));
}

Json mixed_layernorm_pointwise_graph() {
    std::vector<std::int64_t> x_dim{2, 2, 2};
    std::vector<std::int64_t> parameter_dim{1, 2, 2};
    std::vector<std::int64_t> scalar_dim{1, 1, 1};
    Json tensors = Json::object();
    tensors["501"] = tensor("X", 501, x_dim);
    tensors["502"] = tensor("SCALE", 502, parameter_dim);
    tensors["503"] = tensor("BIAS", 503, parameter_dim);
    tensors["504"] = tensor("EPSILON", 504, scalar_dim);
    tensors["505"] = tensor("NORMALIZED", 505, x_dim, true);
    tensors["506"] = tensor("Y", 506, x_dim);
    Json layernorm{{"tag", "LAYER_NORM"},
                   {"name", "layernorm_inference"},
                   {"inputs",
                    Json::object({{"X", 501},
                                  {"SCALE", 502},
                                  {"BIAS", 503},
                                  {"EPSILON", 504}})},
                   {"outputs", Json::object({{"Y", 505}})},
                   {"compute_data_type", "FLOAT"},
                   {"forward_phase", "INFERENCE"}};
    Json relu{{"tag", "POINTWISE"},
              {"name", "relu_after_layernorm"},
              {"inputs", Json::object({{"IN_0", 505}})},
              {"outputs", Json::object({{"OUT_0", 506}})},
              {"compute_data_type", "FLOAT"},
              {"mode", "RELU_FWD"},
              {"axis", nullptr},
              {"relu_lower_clip", nullptr},
              {"relu_upper_clip", nullptr},
              {"relu_lower_clip_slope", nullptr},
              {"swish_beta", nullptr},
              {"elu_alpha", nullptr},
              {"softplus_beta", nullptr}};
    return graph_document(4004, "mixed_layernorm_pointwise",
                          Json::array({layernorm, relu}), std::move(tensors));
}

void run_mixed_graph_and_artifact(TestRunner& tests) {
    NormCase configuration{"LAYER_NORM", "LAYER_NORM_BPROP", {2, 2, 2},
                           {1, 2, 2}, {2, 1, 1}, false, true};
    std::vector<float> x{0.2F, -0.5F, 0.7F, 1.1F,
                         -0.3F, 0.4F, -0.8F, 0.9F};
    std::vector<float> scale{0.8F, 0.9F, 1.0F, 1.1F};
    std::vector<float> bias{-0.2F, -0.1F, 0.0F, 0.1F};
    std::vector<float> dy(x.size(), 0.0F);
    std::vector<float> epsilon{1.0e-4F};
    auto reference = norm_reference(configuration, x, scale, bias, dy,
                                    epsilon.front());
    for (auto& value : reference.y) value = std::max(0.0F, value);

    deepforge::compiler::CompilationResult compilation;
    auto status =
        compile_document(mixed_layernorm_pointwise_graph(), compilation);
    tests.good(status, "compile mixed LAYER_NORM and POINTWISE graph");
    if (status.is_bad() || !compilation.executable) return;

    std::unique_ptr<void, decltype(&std::free)> workspace(
        std::aligned_alloc(
            static_cast<std::size_t>(compilation.workspace.alignment),
            static_cast<std::size_t>(compilation.workspace.size_bytes)),
        &std::free);
    tests.check(compilation.workspace.size_bytes != 0 && workspace != nullptr,
                "mixed graph plans aligned virtual workspace");
    if (!workspace) return;

    std::vector<float> output(x.size(), -99.0F);
    deepforge::runtime::VariantPack pack{{501, x.data()},
                                         {502, scale.data()},
                                         {503, bias.data()},
                                         {504, epsilon.data()},
                                         {506, output.data()}};
    status = compilation.executable->execute_variant(
        deepforge::runtime::CpuVariant::kScalar, nullptr, pack,
        workspace.get());
    tests.good(status, "execute mixed LAYER_NORM and POINTWISE graph");
    tests.check(close_vectors(output, reference.y),
                "mixed graph matches independent reference");

    std::vector<std::uint8_t> artifact;
    status = deepforge::compiler::serialize_artifact(compilation, artifact);
    tests.good(status, "serialize C3 mixed graph artifact");
    std::unique_ptr<deepforge::runtime::Executable> loaded;
    if (status.is_good()) {
        status =
            deepforge::compiler::load_artifact_executable(artifact, loaded);
    }
    tests.good(status, "load C3 mixed graph artifact");
    if (loaded) {
        std::fill(output.begin(), output.end(), -99.0F);
        status = loaded->execute(nullptr, pack, workspace.get());
        tests.good(status, "execute reloaded C3 mixed graph artifact");
        tests.check(close_vectors(output, reference.y),
                    "reloaded C3 artifact matches independent reference");
    }
}

void run_batchnorm_family(TestRunner& tests) {
    NormCase configuration{"BATCHNORM", "DBN", {2, 2, 2}, {1, 2, 1},
                           {1, 2, 1}, false, false};
    std::vector<float> x{0.2F, -0.5F, 0.7F, 1.1F,
                         -0.3F, 0.4F, -0.8F, 0.9F};
    std::vector<float> scale{0.8F, 1.2F};
    std::vector<float> bias{-0.1F, 0.2F};
    std::vector<float> dy{0.3F, -0.2F, 0.5F, -0.7F,
                          0.4F, 0.1F, -0.6F, 0.8F};
    constexpr float epsilon = 1.0e-4F;
    constexpr float momentum = 0.25F;
    std::vector<float> scalar_epsilon{epsilon};
    std::vector<float> scalar_momentum{momentum};
    std::vector<float> previous_mean{0.4F, -0.3F};
    std::vector<float> previous_variance{1.5F, 0.8F};
    auto reference = norm_reference(configuration, x, scale, bias, dy,
                                    epsilon);
    std::vector<float> expected_next_mean(2);
    std::vector<float> expected_next_variance(2);
    for (std::size_t channel = 0; channel < 2; ++channel) {
        auto const population_variance =
            1.0F / (reference.inverse[channel] * reference.inverse[channel]) -
            epsilon;
        auto const sample_variance = population_variance * 4.0F / 3.0F;
        expected_next_mean[channel] =
            (1.0F - momentum) * previous_mean[channel] +
            momentum * reference.mean[channel];
        expected_next_variance[channel] =
            (1.0F - momentum) * previous_variance[channel] +
            momentum * sample_variance;
    }

    deepforge::compiler::CompilationResult batchnorm;
    auto status = compile_document(batchnorm_graph("BATCHNORM"), batchnorm);
    tests.good(status, "compile BATCHNORM");
    if (status.is_good() && batchnorm.executable) {
        std::vector<float> y(8, -99.0F);
        std::vector<float> mean(2, -99.0F);
        std::vector<float> inverse(2, -99.0F);
        std::vector<float> next_mean(2, -99.0F);
        std::vector<float> next_variance(2, -99.0F);
        deepforge::runtime::VariantPack pack{
            {301, x.data()},          {302, scale.data()},
            {303, bias.data()},       {304, scalar_epsilon.data()},
            {305, previous_mean.data()},
            {306, previous_variance.data()},
            {307, scalar_momentum.data()},
            {308, y.data()},          {309, mean.data()},
            {310, inverse.data()},    {311, next_mean.data()},
            {312, next_variance.data()}};
        status = batchnorm.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr);
        tests.good(status, "execute BATCHNORM");
        tests.check(close_vectors(y, reference.y) &&
                        close_vectors(mean, reference.mean) &&
                        close_vectors(inverse, reference.inverse) &&
                        close_vectors(next_mean, expected_next_mean) &&
                        close_vectors(next_variance,
                                      expected_next_variance, 8.0e-5F),
                    "BATCHNORM outputs and running stats match reference");
    }

    deepforge::compiler::CompilationResult inference;
    status = compile_document(batchnorm_graph("BATCHNORM_INFERENCE"),
                              inference);
    tests.good(status, "compile BATCHNORM_INFERENCE");
    if (status.is_good() && inference.executable) {
        std::vector<float> y(8, -99.0F);
        deepforge::runtime::VariantPack pack{{301, x.data()},
                                             {302, scale.data()},
                                             {303, bias.data()},
                                             {309, reference.mean.data()},
                                             {310, reference.inverse.data()},
                                             {308, y.data()}};
        status = inference.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr);
        tests.good(status, "execute BATCHNORM_INFERENCE");
        tests.check(close_vectors(y, reference.y),
                    "BATCHNORM_INFERENCE matches saved-stat reference");
    }

    deepforge::compiler::CompilationResult dbn;
    status = compile_document(batchnorm_graph("DBN"), dbn);
    tests.good(status, "compile DBN");
    if (status.is_good() && dbn.executable) {
        std::vector<float> dx(8, -99.0F);
        std::vector<float> dscale(2, -99.0F);
        std::vector<float> dbias(2, -99.0F);
        deepforge::runtime::VariantPack pack{{301, x.data()},
                                             {302, scale.data()},
                                             {309, reference.mean.data()},
                                             {310, reference.inverse.data()},
                                             {313, dy.data()},
                                             {314, dx.data()},
                                             {315, dscale.data()},
                                             {316, dbias.data()}};
        status = dbn.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr);
        tests.good(status, "execute DBN");
        tests.check(close_vectors(dx, reference.dx, 8.0e-5F) &&
                        close_vectors(dscale, reference.dscale) &&
                        close_vectors(dbias, reference.dbias),
                    "DBN matches independent backward reference");
    }

    deepforge::compiler::CompilationResult dbn_weight;
    status = compile_document(batchnorm_graph("DBN_WEIGHT"), dbn_weight);
    tests.good(status, "compile DBN_WEIGHT");
    if (status.is_good() && dbn_weight.executable) {
        std::vector<float> dscale(2, -99.0F);
        std::vector<float> dbias(2, -99.0F);
        std::vector<float> equivalent_bias(2, -99.0F);
        std::vector<float> equivalent_dy(2, -99.0F);
        std::vector<float> equivalent_x(2, -99.0F);
        deepforge::runtime::VariantPack pack{
            {301, x.data()},       {302, scale.data()},
            {309, reference.mean.data()},
            {310, reference.inverse.data()},
            {313, dy.data()},      {315, dscale.data()},
            {316, dbias.data()},   {317, equivalent_bias.data()},
            {318, equivalent_dy.data()},
            {319, equivalent_x.data()}};
        status = dbn_weight.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr);
        tests.good(status, "execute DBN_WEIGHT");
        std::vector<float> reconstructed_dx(8);
        for (std::size_t index = 0; index < x.size(); ++index) {
            auto coordinate = coordinates(index, configuration.x_dim);
            auto const channel = mapped_index(coordinate,
                                              configuration.scale_dim);
            reconstructed_dx[index] = equivalent_dy[channel] * dy[index] +
                                      equivalent_x[channel] * x[index] +
                                      equivalent_bias[channel];
        }
        tests.check(close_vectors(dscale, reference.dscale) &&
                        close_vectors(dbias, reference.dbias) &&
                        close_vectors(reconstructed_dx, reference.dx,
                                      9.0e-5F),
                    "DBN_WEIGHT gradients and equivalent coefficients match DBN");
    }
}

Json genstats_graph() {
    Json tensors = Json::object();
    tensors["401"] = tensor("X", 401, {2, 2, 2});
    tensors["402"] = tensor("SUM", 402, {1, 2, 1});
    tensors["403"] = tensor("SQ_SUM", 403, {1, 2, 1});
    Json node{{"tag", "GENSTATS"},
              {"name", "genstats"},
              {"inputs", Json::object({{"X", 401}})},
              {"outputs", Json::object({{"SUM", 402}, {"SQ_SUM", 403}})},
              {"compute_data_type", "FLOAT"}};
    return graph_document(4004, "genstats", Json::array({node}),
                          std::move(tensors));
}

Json bn_finalize_graph() {
    Json tensors = Json::object();
    std::vector<std::int64_t> channel_dim{1, 2, 1};
    std::vector<std::int64_t> scalar_dim{1, 1, 1};
    for (auto item : {
             std::pair<std::string, Json>{"402", tensor("SUM", 402, channel_dim)},
             {"403", tensor("SQ_SUM", 403, channel_dim)},
             {"404", tensor("SCALE", 404, channel_dim)},
             {"405", tensor("BIAS", 405, channel_dim)},
             {"406", tensor("EPSILON", 406, scalar_dim)},
             {"407", tensor("COUNT", 407, scalar_dim)},
             {"408", tensor("PREV_MEAN", 408, channel_dim)},
             {"409", tensor("PREV_VAR", 409, channel_dim)},
             {"410", tensor("MOMENTUM", 410, scalar_dim)}}) {
        tensors[item.first] = item.second;
    }
    for (std::int64_t uid = 411; uid <= 416; ++uid) {
        tensors[std::to_string(uid)] =
            tensor("O" + std::to_string(uid), uid, channel_dim);
    }
    Json node{{"tag", "BN_FINALIZE"},
              {"name", "bn_finalize"},
              {"inputs",
               Json::object({{"SUM", 402},
                             {"SQ_SUM", 403},
                             {"SCALE", 404},
                             {"BIAS", 405},
                             {"EPSILON", 406},
                             {"ACCUM_COUNT", 407},
                             {"PREV_RUNNING_MEAN", 408},
                             {"PREV_RUNNING_VAR", 409},
                             {"MOMENTUM", 410}})},
              {"outputs",
               Json::object({{"EQ_SCALE", 411},
                             {"EQ_BIAS", 412},
                             {"MEAN", 413},
                             {"INV_VARIANCE", 414},
                             {"NEXT_RUNNING_MEAN", 415},
                             {"NEXT_RUNNING_VAR", 416}})},
              {"compute_data_type", "FLOAT"}};
    return graph_document(4005, "bn_finalize", Json::array({node}),
                          std::move(tensors));
}

void run_statistics_family(TestRunner& tests) {
    std::vector<float> x{0.2F, -0.5F, 0.7F, 1.1F,
                         -0.3F, 0.4F, -0.8F, 0.9F};
    std::vector<float> expected_sum(2, 0.0F);
    std::vector<float> expected_square_sum(2, 0.0F);
    for (std::size_t index = 0; index < x.size(); ++index) {
        auto const channel = (index / 2) % 2;
        expected_sum[channel] += x[index];
        expected_square_sum[channel] += x[index] * x[index];
    }
    deepforge::compiler::CompilationResult genstats;
    auto status = compile_document(genstats_graph(), genstats);
    tests.good(status, "compile GENSTATS");
    std::vector<float> sum(2, -99.0F);
    std::vector<float> square_sum(2, -99.0F);
    if (status.is_good() && genstats.executable) {
        deepforge::runtime::VariantPack pack{{401, x.data()},
                                             {402, sum.data()},
                                             {403, square_sum.data()}};
        status = genstats.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr);
        tests.good(status, "execute GENSTATS");
        tests.check(close_vectors(sum, expected_sum) &&
                        close_vectors(square_sum, expected_square_sum),
                    "GENSTATS computes per-channel sum and square sum");
    }

    std::vector<float> scale{0.8F, 1.2F};
    std::vector<float> bias{-0.1F, 0.2F};
    std::vector<float> epsilon{1.0e-4F};
    std::vector<float> count{4.0F};
    std::vector<float> previous_mean{0.4F, -0.3F};
    std::vector<float> previous_variance{1.5F, 0.8F};
    std::vector<float> momentum{0.25F};
    std::vector<float> equivalent_scale(2, -99.0F);
    std::vector<float> equivalent_bias(2, -99.0F);
    std::vector<float> mean(2, -99.0F);
    std::vector<float> inverse(2, -99.0F);
    std::vector<float> next_mean(2, -99.0F);
    std::vector<float> next_variance(2, -99.0F);
    deepforge::compiler::CompilationResult finalize;
    status = compile_document(bn_finalize_graph(), finalize);
    tests.good(status, "compile BN_FINALIZE");
    if (status.is_good() && finalize.executable) {
        deepforge::runtime::VariantPack pack{
            {402, expected_sum.data()},
            {403, expected_square_sum.data()},
            {404, scale.data()},
            {405, bias.data()},
            {406, epsilon.data()},
            {407, count.data()},
            {408, previous_mean.data()},
            {409, previous_variance.data()},
            {410, momentum.data()},
            {411, equivalent_scale.data()},
            {412, equivalent_bias.data()},
            {413, mean.data()},
            {414, inverse.data()},
            {415, next_mean.data()},
            {416, next_variance.data()}};
        status = finalize.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr);
        tests.good(status, "execute BN_FINALIZE");
        bool matches = true;
        for (std::size_t channel = 0; channel < 2; ++channel) {
            auto const expected_mean = expected_sum[channel] / 4.0F;
            auto const variance =
                expected_square_sum[channel] / 4.0F -
                expected_mean * expected_mean;
            auto const expected_inverse =
                1.0F / std::sqrt(variance + epsilon[0]);
            auto const expected_scale = scale[channel] * expected_inverse;
            auto const expected_bias =
                bias[channel] - expected_mean * expected_scale;
            auto const expected_running_mean =
                0.75F * previous_mean[channel] + 0.25F * expected_mean;
            auto const expected_running_variance =
                0.75F * previous_variance[channel] +
                0.25F * variance * 4.0F / 3.0F;
            matches &= std::fabs(mean[channel] - expected_mean) < 1.0e-5F;
            matches &=
                std::fabs(inverse[channel] - expected_inverse) < 1.0e-5F;
            matches &= std::fabs(equivalent_scale[channel] - expected_scale) <
                       1.0e-5F;
            matches &= std::fabs(equivalent_bias[channel] - expected_bias) <
                       1.0e-5F;
            matches &= std::fabs(next_mean[channel] - expected_running_mean) <
                       1.0e-5F;
            matches &= std::fabs(next_variance[channel] -
                                 expected_running_variance) < 1.0e-5F;
        }
        tests.check(matches,
                    "BN_FINALIZE statistics and equivalent affine outputs match reference");
    }
}

}  // namespace

int main() {
    TestRunner tests;
    run_norm_case(tests,
                  NormCase{"INSTANCE_NORM", "INSTANCE_NORM_BPROP",
                           {2, 2, 2}, {1, 2, 1}, {2, 2, 1}, false, false},
                  false);
    run_norm_case(tests,
                  NormCase{"LAYER_NORM", "LAYER_NORM_BPROP", {2, 2, 2},
                           {1, 2, 2}, {2, 1, 1}, false, true},
                  true);
    run_norm_case(tests,
                  NormCase{"RMS_NORM", "RMS_NORM_BPROP", {2, 2, 2},
                           {1, 2, 2}, {2, 1, 1}, true, false},
                  false);
    run_norm_case(tests,
                  NormCase{"ADA_LAYER_NORM", "ADA_LAYER_NORM_BPROP",
                           {2, 2, 2}, {2, 2, 2}, {2, 1, 1}, false, true},
                  false);
    run_batchnorm_family(tests);
    run_statistics_family(tests);
    run_mixed_graph_and_artifact(tests);

    auto invalid = norm_forward_graph(
        NormCase{"LAYER_NORM", "LAYER_NORM_BPROP", {2, 2, 2}, {1, 2, 2},
                 {2, 1, 1}, false, true});
    invalid["tensors"]["107"]["dim"] = Json::array({1, 2, 1});
    invalid["tensors"]["107"]["stride"] = Json::array({2, 1, 1});
    deepforge::compiler::CompilationResult rejected;
    auto status = compile_document(invalid, rejected);
    tests.check(status.code() == deepforge::import::ErrorCode::kInvalidShape,
                "mismatched normalization statistic shape is rejected");

    auto invalid_instance = norm_forward_graph(
        NormCase{"INSTANCE_NORM", "INSTANCE_NORM_BPROP", {2, 2, 2},
                 {1, 2, 1}, {2, 2, 1}, false, false});
    for (auto uid : {"102", "103"}) {
        invalid_instance["tensors"][uid]["dim"] = Json::array({2, 2, 1});
        invalid_instance["tensors"][uid]["stride"] =
            Json::array({2, 1, 1});
    }
    status = compile_document(invalid_instance, rejected);
    tests.check(status.code() == deepforge::import::ErrorCode::kInvalidShape,
                "INSTANCE_NORM rejects non-channel parameter shapes");

    auto invalid_context = norm_forward_graph(
        NormCase{"LAYER_NORM", "LAYER_NORM_BPROP", {2, 2, 2}, {1, 2, 2},
                 {2, 1, 1}, false, true});
    invalid_context["context"]["io_data_type"] = "HALF";
    status = compile_document(invalid_context, rejected);
    tests.check(
        status.code() == deepforge::import::ErrorCode::kUnsupportedDataType,
        "CPU graph execution rejects non-FLOAT context data types");

    return tests.finish();
}
