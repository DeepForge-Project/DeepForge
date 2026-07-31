#include "DeepForge/Compiler/Artifact.h"
#include "DeepForge/Compiler/Codegen.h"
#include "DeepForge/Import/SerializedGraphImporter.h"

#include <cudnn_frontend/thirdparty/nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
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
            std::cout << "deepforge-sequence: " << checks_
                      << " checks passed\n";
            return 0;
        }
        std::cerr << "deepforge-sequence: " << failures_ << " of "
                  << checks_ << " checks failed\n";
        return 1;
    }

private:
    int checks_ = 0;
    int failures_ = 0;
};

std::vector<std::int64_t> contiguous_strides(
    std::vector<std::int64_t> const& dimensions) {
    std::vector<std::int64_t> strides(dimensions.size(), 1);
    for (std::size_t index = dimensions.size(); index > 1; --index) {
        strides[index - 2] = strides[index - 1] * dimensions[index - 1];
    }
    return strides;
}

Json tensor(std::string name,
            std::int64_t uid,
            std::vector<std::int64_t> dimensions,
            std::string data_type = "FLOAT",
            bool is_virtual = false) {
    auto strides = contiguous_strides(dimensions);
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

bool close_vectors(std::vector<float> const& actual,
                   std::vector<float> const& expected,
                   float tolerance = 8.0e-5F) {
    if (actual.size() != expected.size()) return false;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (std::isinf(expected[index])) {
            if (actual[index] != expected[index]) {
                std::cerr << "infinity mismatch at " << index << '\n';
                return false;
            }
            continue;
        }
        auto const bound = tolerance * (1.0F + std::fabs(expected[index]));
        if (std::fabs(actual[index] - expected[index]) > bound) {
            std::cerr << "mismatch at " << index << ": expected "
                      << expected[index] << " got " << actual[index] << '\n';
            return false;
        }
    }
    return true;
}

std::size_t offset4(std::array<std::int64_t, 4> const& dimensions,
                    std::int64_t a,
                    std::int64_t b,
                    std::int64_t c,
                    std::int64_t d) {
    return static_cast<std::size_t>(
        ((a * dimensions[1] + b) * dimensions[2] + c) * dimensions[3] + d);
}

std::uint64_t splitmix(std::uint64_t seed,
                       std::uint64_t offset,
                       std::uint64_t linear) {
    auto z = seed + offset + linear + 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

float bernoulli(std::int64_t seed,
                std::int64_t offset,
                std::uint64_t linear,
                double probability) {
    auto const sample = splitmix(static_cast<std::uint64_t>(seed),
                                 static_cast<std::uint64_t>(offset), linear) >>
                        40;
    auto const threshold = static_cast<std::uint64_t>(
        std::floor(probability * static_cast<double>(1U << 24)));
    return sample < threshold ? 1.0F : 0.0F;
}

Json rng_graph(bool fixed_seed = false) {
    Json tensors = Json::object();
    tensors["1"] = tensor("Seed", 1, {1, 1, 1, 1}, "INT64");
    tensors["2"] = tensor("Offset", 2, {1, 1, 1, 1}, "INT64");
    tensors["3"] = tensor("Y", 3, {2, 3});
    Json inputs = fixed_seed ? Json::object()
                             : Json::object({{"Seed", 1}, {"Offset", 2}});
    Json node{{"tag", "RNG"},
              {"name", "bernoulli"},
              {"inputs", std::move(inputs)},
              {"outputs", Json::object({{"Y", 3}})},
              {"distribution", "BERNOULLI"},
              {"dim", Json::array({2, 3})},
              {"stride", Json::array({3, 1})},
              {"seed", fixed_seed ? Json(17) : Json(nullptr)},
              {"bernoulli_probability", 0.375}};
    return graph_document(4001, "rng", Json::array({node}),
                          std::move(tensors));
}

Json rope_graph() {
    Json tensors = Json::object();
    tensors["11"] = tensor("X", 11, {1, 2, 3, 6});
    tensors["12"] = tensor("FREQS", 12, {4, 1, 1, 4});
    tensors["13"] = tensor("Y", 13, {1, 2, 3, 6});
    tensors["14"] = tensor("DY", 14, {1, 2, 3, 6});
    tensors["15"] = tensor("DX", 15, {1, 2, 3, 6});
    Json common{{"compute_data_type", "FLOAT"},
                {"output_scale", 1.25},
                {"rope_dim", 4}};
    Json forward = common;
    forward.update(Json{{"tag", "ROPE"},
                        {"name", "partial_rope"},
                        {"inputs", Json::object({{"INPUT", 11},
                                                  {"FREQS", 12}})},
                        {"outputs", Json::object({{"OUTPUT", 13}})}});
    Json backward = common;
    backward.update(Json{{"tag", "ROPE_BWD"},
                         {"name", "partial_rope_bwd"},
                         {"inputs", Json::object({{"DY", 14},
                                                   {"FREQS", 12}})},
                         {"outputs", Json::object({{"DX", 15}})}});
    return graph_document(4002, "rope", Json::array({forward, backward}),
                          std::move(tensors));
}

std::vector<float> rope_reference(std::vector<float> const& input,
                                  std::vector<float> const& frequencies,
                                  bool backward) {
    std::array<std::int64_t, 4> const x_dim{1, 2, 3, 6};
    std::array<std::int64_t, 4> const f_dim{4, 1, 1, 4};
    std::vector<float> output(input.size());
    for (std::int64_t h = 0; h < 2; ++h) {
        for (std::int64_t s = 0; s < 3; ++s) {
            for (std::int64_t d = 0; d < 6; ++d) {
                auto const index = offset4(x_dim, 0, h, s, d);
                if (d < 2) {
                    output[index] = input[index] * 1.25F;
                    continue;
                }
                auto const low = d < 4;
                auto const angle_index = low ? d - 2 : d - 4;
                auto const pair = low ? d + 2 : d - 2;
                auto const angle =
                    frequencies[offset4(f_dim, s, 0, 0, angle_index)];
                auto const cosine = std::cos(angle);
                auto const sine = std::sin(angle);
                auto const pair_value =
                    input[offset4(x_dim, 0, h, s, pair)];
                float result = input[index] * cosine;
                if (!backward) {
                    result += (low ? -1.0F : 1.0F) * pair_value * sine;
                } else {
                    result += (low ? 1.0F : -1.0F) * pair_value * sine;
                }
                output[index] = result * 1.25F;
            }
        }
    }
    return output;
}

struct AttentionCase {
    std::array<std::int64_t, 4> q_dim{1, 2, 2, 2};
    std::array<std::int64_t, 4> k_dim{1, 1, 3, 2};
    std::array<std::int64_t, 4> v_dim{1, 1, 3, 2};
    std::array<std::int64_t, 4> o_dim{1, 2, 2, 2};
    std::vector<float> q;
    std::vector<float> k;
    std::vector<float> v;
    std::vector<float> bias;
    std::vector<float> dropout_mask;
    std::vector<std::int32_t> seq_q{2};
    std::vector<std::int32_t> seq_kv{2};
    float attention_scale = 0.7F;
    float dropout_scale = 1.25F;
    float dropout_scale_inv = 0.8F;
    bool alibi = false;
    bool padding = true;
    bool bottom_right = false;
    std::optional<std::int64_t> left_bound{2};
    std::optional<std::int64_t> right_bound{1};
};

AttentionCase feature_case() {
    AttentionCase value;
    value.q = {0.2F, -0.3F, 0.7F, 0.1F,
               -0.5F, 0.4F, 0.3F, -0.8F};
    value.k = {0.6F, -0.2F, -0.4F, 0.9F, 0.1F, 0.5F};
    value.v = {0.3F, -0.7F, 0.8F, 0.2F, -0.6F, 0.4F};
    value.bias = {0.1F, -0.2F, 0.05F, -0.15F, 0.25F, -0.05F};
    value.dropout_mask = {1.0F, 0.0F, 1.0F, 1.0F, 1.0F, 0.0F,
                          0.0F, 1.0F, 1.0F, 1.0F, 0.0F, 1.0F};
    return value;
}

float alibi_slope(std::int64_t head, std::int64_t heads) {
    auto const n = std::int64_t{1} << static_cast<int>(
                       std::floor(std::log2(static_cast<double>(heads))));
    if (head < n) {
        return std::pow(2.0F, -8.0F * static_cast<float>(head + 1) /
                                  static_cast<float>(n));
    }
    auto const extra = 2 * (head - n) + 1;
    return std::pow(2.0F,
                    -8.0F * (static_cast<float>(extra) * 0.5F) /
                        static_cast<float>(n));
}

bool score_valid(AttentionCase const& configuration,
                 std::int64_t batch,
                 std::int64_t query,
                 std::int64_t key) {
    if (configuration.padding &&
        (query >= configuration.seq_q[static_cast<std::size_t>(batch)] ||
         key >= configuration.seq_kv[static_cast<std::size_t>(batch)])) {
        return false;
    }
    std::int64_t shift = 0;
    if (configuration.bottom_right) {
        shift = configuration.padding
                    ? configuration.seq_kv[static_cast<std::size_t>(batch)] -
                          configuration.seq_q[static_cast<std::size_t>(batch)]
                    : configuration.k_dim[2] - configuration.q_dim[2];
    }
    auto const difference = key - query;
    if (configuration.right_bound &&
        difference > shift + *configuration.right_bound) {
        return false;
    }
    if (configuration.left_bound &&
        difference <= shift - *configuration.left_bound) {
        return false;
    }
    return true;
}

struct AttentionReference {
    std::vector<float> o;
    std::vector<float> stats;
    std::vector<float> maximum;
    std::vector<float> sum_exp;
    std::vector<float> probability;
};

AttentionReference attention_forward(AttentionCase const& configuration) {
    auto const [b, h_q, s_q, d_qk] = configuration.q_dim;
    auto const s_kv = configuration.k_dim[2];
    auto const d_v = configuration.v_dim[3];
    std::array<std::int64_t, 4> const score_dim{b, h_q, s_q, s_kv};
    std::array<std::int64_t, 4> const bias_dim{1, h_q, 1, s_kv};
    AttentionReference result;
    result.o.assign(static_cast<std::size_t>(b * h_q * s_q * d_v), 0.0F);
    result.stats.assign(static_cast<std::size_t>(b * h_q * s_q), 0.0F);
    result.maximum.resize(result.stats.size());
    result.sum_exp.resize(result.stats.size());
    result.probability.assign(static_cast<std::size_t>(b * h_q * s_q * s_kv),
                              0.0F);
    for (std::int64_t n = 0; n < b; ++n) {
        for (std::int64_t h = 0; h < h_q; ++h) {
            auto const kh = h / (h_q / configuration.k_dim[1]);
            auto const vh = h / (h_q / configuration.v_dim[1]);
            for (std::int64_t q = 0; q < s_q; ++q) {
                auto maximum = -std::numeric_limits<float>::infinity();
                std::vector<float> scores(static_cast<std::size_t>(s_kv),
                                          maximum);
                for (std::int64_t k = 0; k < s_kv; ++k) {
                    if (!score_valid(configuration, n, q, k)) continue;
                    double score = 0.0;
                    for (std::int64_t d = 0; d < d_qk; ++d) {
                        score += configuration.q[offset4(
                                     configuration.q_dim, n, h, q, d)] *
                                 configuration.k[offset4(
                                     configuration.k_dim, n, kh, k, d)];
                    }
                    score *= configuration.attention_scale;
                    if (!configuration.bias.empty()) {
                        score += configuration.bias[offset4(bias_dim, 0, h, 0,
                                                            k)];
                    }
                    if (configuration.alibi) {
                        score += static_cast<double>(k - q) *
                                 alibi_slope(h, h_q);
                    }
                    scores[static_cast<std::size_t>(k)] =
                        static_cast<float>(score);
                    maximum = std::max(maximum,
                                       scores[static_cast<std::size_t>(k)]);
                }
                double sum = 0.0;
                for (std::int64_t k = 0; k < s_kv; ++k) {
                    if (!score_valid(configuration, n, q, k)) continue;
                    sum += std::exp(scores[static_cast<std::size_t>(k)] -
                                    maximum);
                }
                auto const row = static_cast<std::size_t>(
                    (n * h_q + h) * s_q + q);
                result.maximum[row] = maximum;
                result.sum_exp[row] = static_cast<float>(sum);
                result.stats[row] =
                    maximum + static_cast<float>(std::log(sum));
                for (std::int64_t k = 0; k < s_kv; ++k) {
                    auto const score_index = offset4(score_dim, n, h, q, k);
                    if (score_valid(configuration, n, q, k) && sum > 0.0) {
                        result.probability[score_index] = static_cast<float>(
                            std::exp(scores[static_cast<std::size_t>(k)] -
                                     maximum) /
                            sum);
                    }
                    auto mask = configuration.dropout_mask.empty()
                                    ? 1.0F
                                    : configuration.dropout_mask[score_index];
                    auto const weighted = result.probability[score_index] *
                                          mask *
                                          configuration.dropout_scale;
                    for (std::int64_t d = 0; d < d_v; ++d) {
                        result.o[offset4(configuration.o_dim, n, h, q, d)] +=
                            weighted * configuration.v[offset4(
                                           configuration.v_dim, n, vh, k, d)];
                    }
                }
            }
        }
    }
    return result;
}

Json forward_attention_node(Json inputs,
                            Json outputs,
                            bool generate_stats,
                            bool alibi,
                            bool padding,
                            Json dropout_probability,
                            Json left_bound,
                            Json right_bound,
                            std::string alignment = "TOP_LEFT",
                            float attention_scale = 0.7F) {
    return Json{{"tag", "SDPA"},
                {"name", "sdpa_forward"},
                {"inputs", std::move(inputs)},
                {"outputs", std::move(outputs)},
                {"generate_stats", generate_stats},
                {"alibi_mask", alibi},
                {"padding_mask", padding},
                {"dropout_probability", std::move(dropout_probability)},
                {"attn_scale_value", attention_scale},
                {"max_seq_len_kv", nullptr},
                {"mma_core_mode", "FLOAT"},
                {"left_bound", std::move(left_bound)},
                {"right_bound", std::move(right_bound)},
                {"diagonal_alignment", std::move(alignment)},
                {"implementation", "AUTO"},
                {"is_mxfp8", false},
                {"unfuse_fma", false}};
}

Json feature_sdpa_graph() {
    Json tensors = Json::object();
    tensors["101"] = tensor("Q", 101, {1, 2, 2, 2});
    tensors["102"] = tensor("K", 102, {1, 1, 3, 2});
    tensors["103"] = tensor("V", 103, {1, 1, 3, 2});
    tensors["104"] = tensor("Bias", 104, {1, 2, 1, 3});
    tensors["105"] = tensor("SEQ_LEN_Q", 105, {1, 1, 1, 1}, "INT32");
    tensors["106"] = tensor("SEQ_LEN_KV", 106, {1, 1, 1, 1}, "INT32");
    tensors["107"] = tensor("Dropout_mask", 107, {1, 2, 2, 3});
    tensors["108"] = tensor("Dropout_scale", 108, {1, 1, 1, 1});
    tensors["109"] = tensor("O", 109, {1, 2, 2, 2});
    tensors["110"] = tensor("Stats", 110, {1, 2, 2, 1});
    tensors["111"] = tensor("Max", 111, {1, 2, 2, 1});
    tensors["112"] = tensor("Sum_exp", 112, {1, 2, 2, 1});
    auto node = forward_attention_node(
        Json::object({{"Q", 101},
                      {"K", 102},
                      {"V", 103},
                      {"Bias", 104},
                      {"SEQ_LEN_Q", 105},
                      {"SEQ_LEN_KV", 106},
                      {"Dropout_mask", 107},
                      {"Dropout_scale", 108}}),
        Json::object({{"O", 109},
                      {"Stats", 110},
                      {"Max", 111},
                      {"Sum_exp", 112}}),
        true, false, true, nullptr, 2, 1);
    return graph_document(4003, "feature_sdpa", Json::array({node}),
                          std::move(tensors));
}

Json feature_sdpa_backward_graph() {
    Json tensors = Json::object();
    tensors["201"] = tensor("Q", 201, {1, 2, 2, 2});
    tensors["202"] = tensor("K", 202, {1, 1, 3, 2});
    tensors["203"] = tensor("V", 203, {1, 1, 3, 2});
    tensors["204"] = tensor("O", 204, {1, 2, 2, 2});
    tensors["205"] = tensor("dO", 205, {1, 2, 2, 2});
    tensors["206"] = tensor("Stats", 206, {1, 2, 2, 1});
    tensors["207"] = tensor("Bias", 207, {1, 2, 1, 3});
    tensors["208"] = tensor("SEQ_LEN_Q", 208, {1, 1, 1, 1}, "INT32");
    tensors["209"] = tensor("SEQ_LEN_KV", 209, {1, 1, 1, 1}, "INT32");
    tensors["210"] = tensor("Dropout_mask", 210, {1, 2, 2, 3});
    tensors["211"] = tensor("Dropout_scale", 211, {1, 1, 1, 1});
    tensors["212"] = tensor("Dropout_scale_inv", 212, {1, 1, 1, 1});
    tensors["213"] = tensor("dQ", 213, {1, 2, 2, 2});
    tensors["214"] = tensor("dK", 214, {1, 1, 3, 2});
    tensors["215"] = tensor("dV", 215, {1, 1, 3, 2});
    tensors["216"] = tensor("dBias", 216, {1, 2, 1, 3});
    Json node{{"tag", "SDPA_BWD"},
              {"name", "sdpa_backward"},
              {"inputs",
               Json::object({{"Q", 201},
                             {"K", 202},
                             {"V", 203},
                             {"O", 204},
                             {"dO", 205},
                             {"Stats", 206},
                             {"Bias", 207},
                             {"SEQ_LEN_Q", 208},
                             {"SEQ_LEN_KV", 209},
                             {"Dropout_mask", 210},
                             {"Dropout_scale", 211},
                             {"Dropout_scale_inv", 212}})},
              {"outputs",
               Json::object({{"dQ", 213},
                             {"dK", 214},
                             {"dV", 215},
                             {"dBias", 216}})},
              {"alibi_mask", false},
              {"padding_mask", true},
              {"dropout_probability", nullptr},
              {"attn_scale_value", 0.7},
              {"left_bound", 2},
              {"right_bound", 1},
              {"diagonal_alignment", "TOP_LEFT"},
              {"max_total_seq_len_q", nullptr},
              {"max_total_seq_len_kv", nullptr},
              {"is_deterministic_algorithm", true}};
    return graph_document(4004, "feature_sdpa_backward",
                          Json::array({node}), std::move(tensors));
}

Json internal_dropout_graph() {
    Json tensors = Json::object();
    tensors["301"] = tensor("Q", 301, {1, 1, 2, 1});
    tensors["302"] = tensor("K", 302, {1, 1, 3, 1});
    tensors["303"] = tensor("V", 303, {1, 1, 3, 1});
    tensors["304"] = tensor("Seed", 304, {1, 1, 1, 1}, "INT64");
    tensors["305"] = tensor("Offset", 305, {1, 1, 1, 1}, "INT64");
    tensors["306"] = tensor("O", 306, {1, 1, 2, 1});
    tensors["307"] = tensor("RNG_DUMP", 307, {1, 1, 2, 3});
    auto node = forward_attention_node(
        Json::object({{"Q", 301},
                      {"K", 302},
                      {"V", 303},
                      {"Seed", 304},
                      {"Offset", 305}}),
        Json::object({{"O", 306}, {"RNG_DUMP", 307}}), false, false,
        false, 0.4, nullptr, nullptr, "TOP_LEFT", 0.5F);
    return graph_document(4005, "internal_dropout", Json::array({node}),
                          std::move(tensors));
}

Json bottom_right_sdpa_graph() {
    Json tensors = Json::object();
    tensors["501"] = tensor("Q", 501, {1, 1, 2, 1});
    tensors["502"] = tensor("K", 502, {1, 1, 4, 1});
    tensors["503"] = tensor("V", 503, {1, 1, 4, 1});
    tensors["504"] = tensor("SEQ_LEN_Q", 504, {1, 1, 1, 1}, "INT32");
    tensors["505"] = tensor("SEQ_LEN_KV", 505, {1, 1, 1, 1}, "INT32");
    tensors["506"] = tensor("O", 506, {1, 1, 2, 1});
    tensors["507"] = tensor("Stats", 507, {1, 1, 2, 1});
    auto node = forward_attention_node(
        Json::object({{"Q", 501},
                      {"K", 502},
                      {"V", 503},
                      {"SEQ_LEN_Q", 504},
                      {"SEQ_LEN_KV", 505}}),
        Json::object({{"O", 506}, {"Stats", 507}}), true, false, true,
        nullptr, 2, 0, "BOTTOM_RIGHT", 0.8F);
    return graph_document(4007, "bottom_right_sdpa", Json::array({node}),
                          std::move(tensors));
}

Json alibi_sdpa_graph() {
    Json tensors = Json::object();
    tensors["401"] = tensor("Q", 401, {1, 3, 2, 2});
    tensors["402"] = tensor("K", 402, {1, 1, 3, 2});
    tensors["403"] = tensor("V", 403, {1, 1, 3, 2});
    tensors["404"] = tensor("O", 404, {1, 3, 2, 2});
    tensors["405"] = tensor("Stats", 405, {1, 3, 2, 1});
    auto node = forward_attention_node(
        Json::object({{"Q", 401}, {"K", 402}, {"V", 403}}),
        Json::object({{"O", 404}, {"Stats", 405}}), true, true, false,
        nullptr, nullptr, 0, "TOP_LEFT", 0.6F);
    return graph_document(4006, "alibi_sdpa", Json::array({node}),
                          std::move(tensors));
}

struct AttentionGradients {
    std::vector<float> d_q;
    std::vector<float> d_k;
    std::vector<float> d_v;
    std::vector<float> d_bias;
};

AttentionGradients attention_backward(
    AttentionCase const& configuration,
    AttentionReference const& forward,
    std::vector<float> const& d_o) {
    auto const [b, h_q, s_q, d_qk] = configuration.q_dim;
    auto const s_kv = configuration.k_dim[2];
    auto const d_v = configuration.v_dim[3];
    std::array<std::int64_t, 4> const score_dim{b, h_q, s_q, s_kv};
    std::array<std::int64_t, 4> const dbias_dim{1, h_q, 1, s_kv};
    AttentionGradients result;
    result.d_q.assign(configuration.q.size(), 0.0F);
    result.d_k.assign(configuration.k.size(), 0.0F);
    result.d_v.assign(configuration.v.size(), 0.0F);
    result.d_bias.assign(configuration.bias.size(), 0.0F);
    for (std::int64_t n = 0; n < b; ++n) {
        for (std::int64_t h = 0; h < h_q; ++h) {
            auto const kh = h / (h_q / configuration.k_dim[1]);
            auto const vh = h / (h_q / configuration.v_dim[1]);
            for (std::int64_t q = 0; q < s_q; ++q) {
                double output_dot = 0.0;
                for (std::int64_t d = 0; d < d_v; ++d) {
                    auto const index = offset4(configuration.o_dim, n, h, q, d);
                    output_dot += forward.o[index] * d_o[index];
                }
                for (std::int64_t k = 0; k < s_kv; ++k) {
                    auto const score_index = offset4(score_dim, n, h, q, k);
                    double d_p = 0.0;
                    for (std::int64_t d = 0; d < d_v; ++d) {
                        d_p += d_o[offset4(configuration.o_dim, n, h, q, d)] *
                               configuration.v[offset4(configuration.v_dim, n,
                                                        vh, k, d)];
                    }
                    auto const mask = configuration.dropout_mask.empty()
                                          ? 1.0F
                                          : configuration.dropout_mask[score_index];
                    auto const base = forward.probability[score_index] *
                                      configuration.dropout_scale *
                                      (static_cast<float>(d_p) * mask -
                                       static_cast<float>(output_dot) *
                                           configuration.dropout_scale_inv);
                    result.d_bias[offset4(dbias_dim, 0, h, 0, k)] += base;
                    auto const d_score = base * configuration.attention_scale;
                    for (std::int64_t d = 0; d < d_qk; ++d) {
                        result.d_q[offset4(configuration.q_dim, n, h, q, d)] +=
                            d_score * configuration.k[offset4(
                                          configuration.k_dim, n, kh, k, d)];
                        result.d_k[offset4(configuration.k_dim, n, kh, k, d)] +=
                            d_score * configuration.q[offset4(
                                          configuration.q_dim, n, h, q, d)];
                    }
                    auto const dropped_probability =
                        forward.probability[score_index] * mask *
                        configuration.dropout_scale;
                    for (std::int64_t d = 0; d < d_v; ++d) {
                        result.d_v[offset4(configuration.v_dim, n, vh, k, d)] +=
                            dropped_probability *
                            d_o[offset4(configuration.o_dim, n, h, q, d)];
                    }
                }
            }
        }
    }
    return result;
}

double attention_loss(AttentionCase configuration,
                      std::vector<float> const& d_o) {
    auto forward = attention_forward(configuration);
    double loss = 0.0;
    for (std::size_t index = 0; index < forward.o.size(); ++index) {
        loss += static_cast<double>(forward.o[index]) * d_o[index];
    }
    return loss;
}

std::vector<float> finite_difference_q(AttentionCase configuration,
                                       std::vector<float> const& d_o) {
    constexpr float epsilon = 1.0e-3F;
    std::vector<float> gradient(configuration.q.size());
    for (std::size_t index = 0; index < configuration.q.size(); ++index) {
        configuration.q[index] += epsilon;
        auto const positive = attention_loss(configuration, d_o);
        configuration.q[index] -= 2.0F * epsilon;
        auto const negative = attention_loss(configuration, d_o);
        configuration.q[index] += epsilon;
        gradient[index] = static_cast<float>((positive - negative) /
                                             (2.0 * epsilon));
    }
    return gradient;
}

void run_rng_tests(TestRunner& tests) {
    deepforge::compiler::CompilationResult compilation;
    auto status = compile_document(rng_graph(), compilation);
    tests.good(status, "compile dynamic-seed RNG");
    if (status.is_bad() || !compilation.executable) return;

    std::int64_t seed = -31;
    std::int64_t offset = 19;
    std::vector<float> expected(6);
    for (std::size_t index = 0; index < expected.size(); ++index) {
        expected[index] = bernoulli(seed, offset, index, 0.375);
    }
    std::vector<float> scalar(6, -1.0F);
    deepforge::runtime::VariantPack pack{{1, &seed},
                                         {2, &offset},
                                         {3, scalar.data()}};
    status = compilation.executable->execute_variant(
        deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr);
    tests.good(status, "execute dynamic-seed RNG");
    tests.check(scalar == expected,
                "RNG stream matches the stable CPU Bernoulli contract");

    for (auto variant : {deepforge::runtime::CpuVariant::kAvx2,
                         deepforge::runtime::CpuVariant::kAvx512}) {
        if (!compilation.executable->supports_variant(variant)) continue;
        std::vector<float> output(6, -1.0F);
        pack[3] = output.data();
        status = compilation.executable->execute_variant(
            variant, nullptr, pack, nullptr);
        tests.good(status, "execute RNG target variant");
        tests.check(output == scalar,
                    "RNG is bit-identical across CPU variants");
    }

    std::vector<float> changed_expected(expected.size());
    do {
        offset += 1;
        for (std::size_t index = 0; index < changed_expected.size(); ++index) {
            changed_expected[index] = bernoulli(seed, offset, index, 0.375);
        }
    } while (changed_expected == expected);
    std::vector<float> changed(6, -1.0F);
    pack[3] = changed.data();
    status = compilation.executable->execute_variant(
        deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr);
    tests.good(status, "execute RNG with changed offset");
    tests.check(changed == changed_expected,
                "changed RNG offset matches the stable CPU contract");
    tests.check(changed != scalar, "RNG offset changes the generated stream");

    deepforge::compiler::CompilationResult fixed;
    status = compile_document(rng_graph(true), fixed);
    tests.good(status, "compile fixed-seed RNG");
    if (status.is_good() && fixed.executable) {
        std::vector<float> output(6, -1.0F);
        deepforge::runtime::VariantPack fixed_pack{{3, output.data()}};
        status = fixed.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr, fixed_pack,
            nullptr);
        tests.good(status, "execute fixed-seed RNG");
        std::vector<float> fixed_expected(6);
        for (std::size_t index = 0; index < fixed_expected.size(); ++index) {
            fixed_expected[index] = bernoulli(17, 0, index, 0.375);
        }
        tests.check(output == fixed_expected,
                    "fixed seed uses an implicit zero offset");
    }
}

void run_rope_tests(TestRunner& tests) {
    deepforge::compiler::CompilationResult compilation;
    auto status = compile_document(rope_graph(), compilation);
    tests.good(status, "compile partial RoPE forward/backward");
    if (status.is_bad() || !compilation.executable) return;

    std::vector<float> x(36);
    std::vector<float> dy(36);
    for (std::size_t index = 0; index < x.size(); ++index) {
        x[index] = static_cast<float>(static_cast<int>(index % 13) - 6) / 7.0F;
        dy[index] = static_cast<float>(static_cast<int>(index % 9) - 4) / 6.0F;
    }
    std::vector<float> frequencies(16, 0.0F);
    for (std::int64_t s = 0; s < 4; ++s) {
        frequencies[static_cast<std::size_t>(s * 4)] = 0.13F * s;
        frequencies[static_cast<std::size_t>(s * 4 + 1)] = 0.31F * s;
    }
    std::vector<float> y(36, -99.0F);
    std::vector<float> dx(36, -99.0F);
    deepforge::runtime::VariantPack pack{{11, x.data()},
                                         {12, frequencies.data()},
                                         {13, y.data()},
                                         {14, dy.data()},
                                         {15, dx.data()}};
    status = compilation.executable->execute_variant(
        deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr);
    tests.good(status, "execute partial RoPE forward/backward");
    tests.check(close_vectors(y, rope_reference(x, frequencies, false)) &&
                    close_vectors(dx, rope_reference(dy, frequencies, true)),
                "RoPE rotation, partial pass-through, scale, and adjoint match");

    auto const expected_y = rope_reference(x, frequencies, false);
    double lhs = 0.0;
    double rhs = 0.0;
    for (std::size_t index = 0; index < x.size(); ++index) {
        lhs += static_cast<double>(expected_y[index]) * dy[index];
        rhs += static_cast<double>(x[index]) * dx[index];
    }
    tests.check(std::fabs(lhs - rhs) < 2.0e-5,
                "RoPE backward is the forward linear adjoint");
}

void run_feature_attention_tests(TestRunner& tests) {
    auto configuration = feature_case();
    auto reference = attention_forward(configuration);
    deepforge::compiler::CompilationResult compilation;
    auto status = compile_document(feature_sdpa_graph(), compilation);
    tests.good(status, "compile feature-rich SDPA forward");
    if (status.is_bad() || !compilation.executable) return;

    std::vector<float> scale{configuration.dropout_scale};
    std::vector<float> o(reference.o.size(), -99.0F);
    std::vector<float> stats(reference.stats.size(), -99.0F);
    std::vector<float> maximum(reference.maximum.size(), -99.0F);
    std::vector<float> sum(reference.sum_exp.size(), -99.0F);
    deepforge::runtime::VariantPack pack{
        {101, configuration.q.data()},
        {102, configuration.k.data()},
        {103, configuration.v.data()},
        {104, configuration.bias.data()},
        {105, configuration.seq_q.data()},
        {106, configuration.seq_kv.data()},
        {107, configuration.dropout_mask.data()},
        {108, scale.data()},
        {109, o.data()},
        {110, stats.data()},
        {111, maximum.data()},
        {112, sum.data()}};
    status = compilation.executable->execute_variant(
        deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr);
    tests.good(status, "execute feature-rich SDPA forward");
    tests.check(close_vectors(o, reference.o) &&
                    close_vectors(stats, reference.stats) &&
                    close_vectors(maximum, reference.maximum) &&
                    close_vectors(sum, reference.sum_exp),
                "GQA, bias, padding, window, softmax stats, and custom dropout match reference");

    bool found_i32_metadata = false;
    for (auto const& argument : compilation.metadata.arguments) {
        if ((argument.uid == 105 || argument.uid == 106) &&
            argument.data_type == deepforge::import::DataType::kInt32 &&
            argument.alignment == alignof(std::int32_t)) {
            found_i32_metadata = true;
        }
    }
    tests.check(found_i32_metadata,
                "sequence metadata preserves INT32 type and alignment");

    std::vector<std::uint8_t> artifact;
    status = deepforge::compiler::serialize_artifact(compilation, artifact);
    tests.good(status, "serialize C4 attention artifact");
    std::unique_ptr<deepforge::runtime::Executable> loaded;
    if (status.is_good()) {
        status = deepforge::compiler::load_artifact_executable(artifact, loaded);
    }
    tests.good(status, "load C4 attention artifact");
    if (loaded) {
        std::fill(o.begin(), o.end(), -99.0F);
        std::fill(stats.begin(), stats.end(), -99.0F);
        status = loaded->execute(nullptr, pack, nullptr);
        tests.good(status, "execute reloaded C4 attention artifact");
        tests.check(close_vectors(o, reference.o) &&
                        close_vectors(stats, reference.stats),
                    "reloaded attention artifact preserves integer metadata ABI");
    }

    deepforge::compiler::CompilationResult backward_compilation;
    status = compile_document(feature_sdpa_backward_graph(),
                              backward_compilation);
    tests.good(status, "compile feature-rich SDPA backward");
    if (status.is_bad() || !backward_compilation.executable) return;
    std::vector<float> d_o{0.2F, -0.4F, 0.7F, 0.1F,
                           -0.3F, 0.6F, -0.5F, 0.8F};
    auto gradients = attention_backward(configuration, reference, d_o);
    std::vector<float> scale_inv{configuration.dropout_scale_inv};
    std::vector<float> d_q(gradients.d_q.size(), -99.0F);
    std::vector<float> d_k(gradients.d_k.size(), -99.0F);
    std::vector<float> d_v(gradients.d_v.size(), -99.0F);
    std::vector<float> d_bias(gradients.d_bias.size(), -99.0F);
    deepforge::runtime::VariantPack backward_pack{
        {201, configuration.q.data()},
        {202, configuration.k.data()},
        {203, configuration.v.data()},
        {204, reference.o.data()},
        {205, d_o.data()},
        {206, reference.stats.data()},
        {207, configuration.bias.data()},
        {208, configuration.seq_q.data()},
        {209, configuration.seq_kv.data()},
        {210, configuration.dropout_mask.data()},
        {211, scale.data()},
        {212, scale_inv.data()},
        {213, d_q.data()},
        {214, d_k.data()},
        {215, d_v.data()},
        {216, d_bias.data()}};
    status = backward_compilation.executable->execute_variant(
        deepforge::runtime::CpuVariant::kScalar, nullptr, backward_pack,
        nullptr);
    tests.good(status, "execute feature-rich SDPA backward");
    tests.check(close_vectors(d_q, gradients.d_q, 2.5e-4F) &&
                    close_vectors(d_k, gradients.d_k, 2.5e-4F) &&
                    close_vectors(d_v, gradients.d_v, 2.5e-4F) &&
                    close_vectors(d_bias, gradients.d_bias, 2.5e-4F),
                "SDPA backward Q/K/V/bias gradients match independent reference");
    tests.check(close_vectors(d_q, finite_difference_q(configuration, d_o),
                              2.5e-3F),
                "SDPA dQ matches finite differences through masked dropout forward");
}

void run_internal_dropout_tests(TestRunner& tests) {
    deepforge::compiler::CompilationResult compilation;
    auto status = compile_document(internal_dropout_graph(), compilation);
    tests.good(status, "compile probability-dropout SDPA");
    if (status.is_bad() || !compilation.executable) return;
    std::vector<float> q{0.2F, -0.4F};
    std::vector<float> k{0.3F, -0.7F, 0.9F};
    std::vector<float> v{0.5F, -0.2F, 0.8F};
    std::int64_t seed = 123;
    std::int64_t offset = 9;
    std::vector<float> o(2, -99.0F);
    std::vector<float> mask(6, -1.0F);
    deepforge::runtime::VariantPack pack{{301, q.data()},
                                         {302, k.data()},
                                         {303, v.data()},
                                         {304, &seed},
                                         {305, &offset},
                                         {306, o.data()},
                                         {307, mask.data()}};
    status = compilation.executable->execute_variant(
        deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr);
    tests.good(status, "execute probability-dropout SDPA");
    std::vector<float> expected_mask(6);
    for (std::size_t index = 0; index < expected_mask.size(); ++index) {
        expected_mask[index] = bernoulli(seed, offset, index, 0.6);
    }
    tests.check(mask == expected_mask,
                "SDPA RNG_DUMP uses the shared deterministic stream");

    AttentionCase configuration;
    configuration.q_dim = {1, 1, 2, 1};
    configuration.k_dim = {1, 1, 3, 1};
    configuration.v_dim = {1, 1, 3, 1};
    configuration.o_dim = {1, 1, 2, 1};
    configuration.q = q;
    configuration.k = k;
    configuration.v = v;
    configuration.bias.clear();
    configuration.dropout_mask = mask;
    configuration.padding = false;
    configuration.left_bound.reset();
    configuration.right_bound.reset();
    configuration.attention_scale = 0.5F;
    configuration.dropout_scale = 1.0F / 0.6F;
    configuration.dropout_scale_inv = 0.6F;
    tests.check(close_vectors(o, attention_forward(configuration).o),
                "probability dropout output matches RNG_DUMP reference");

    auto first_o = o;
    auto first_mask = mask;
    std::fill(o.begin(), o.end(), -99.0F);
    std::fill(mask.begin(), mask.end(), -1.0F);
    status = compilation.executable->execute_variant(
        deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr);
    tests.good(status, "repeat probability-dropout SDPA");
    tests.check(o == first_o && mask == first_mask,
                "attention dropout is reproducible for seed and offset");
}

void run_alibi_tests(TestRunner& tests) {
    AttentionCase configuration;
    configuration.q_dim = {1, 3, 2, 2};
    configuration.k_dim = {1, 1, 3, 2};
    configuration.v_dim = {1, 1, 3, 2};
    configuration.o_dim = {1, 3, 2, 2};
    configuration.q.resize(12);
    configuration.k.resize(6);
    configuration.v.resize(6);
    for (std::size_t index = 0; index < configuration.q.size(); ++index) {
        configuration.q[index] =
            static_cast<float>(static_cast<int>(index % 7) - 3) / 5.0F;
    }
    for (std::size_t index = 0; index < configuration.k.size(); ++index) {
        configuration.k[index] =
            static_cast<float>(static_cast<int>(index % 5) - 2) / 4.0F;
        configuration.v[index] =
            static_cast<float>(static_cast<int>(index % 6) - 2) / 3.0F;
    }
    configuration.bias.clear();
    configuration.dropout_mask.clear();
    configuration.dropout_scale = 1.0F;
    configuration.dropout_scale_inv = 1.0F;
    configuration.padding = false;
    configuration.alibi = true;
    configuration.left_bound.reset();
    configuration.right_bound = 0;
    configuration.attention_scale = 0.6F;
    auto reference = attention_forward(configuration);

    deepforge::compiler::CompilationResult compilation;
    auto status = compile_document(alibi_sdpa_graph(), compilation);
    tests.good(status, "compile ALiBi causal SDPA");
    if (status.is_bad() || !compilation.executable) return;
    std::vector<float> o(reference.o.size(), -99.0F);
    std::vector<float> stats(reference.stats.size(), -99.0F);
    deepforge::runtime::VariantPack pack{{401, configuration.q.data()},
                                         {402, configuration.k.data()},
                                         {403, configuration.v.data()},
                                         {404, o.data()},
                                         {405, stats.data()}};
    status = compilation.executable->execute_variant(
        deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr);
    tests.good(status, "execute ALiBi causal SDPA");
    tests.check(close_vectors(o, reference.o) &&
                    close_vectors(stats, reference.stats),
                "non-power-of-two ALiBi slopes and causal mask match reference");
}

void run_bottom_right_tests(TestRunner& tests) {
    AttentionCase configuration;
    configuration.q_dim = {1, 1, 2, 1};
    configuration.k_dim = {1, 1, 4, 1};
    configuration.v_dim = {1, 1, 4, 1};
    configuration.o_dim = {1, 1, 2, 1};
    configuration.q = {0.4F, -0.7F};
    configuration.k = {0.2F, -0.5F, 0.9F, 0.3F};
    configuration.v = {-0.6F, 0.8F, 0.1F, 0.5F};
    configuration.bias.clear();
    configuration.dropout_mask.clear();
    configuration.dropout_scale = 1.0F;
    configuration.dropout_scale_inv = 1.0F;
    configuration.padding = true;
    configuration.bottom_right = true;
    configuration.seq_q = {1};
    configuration.seq_kv = {3};
    configuration.left_bound = 2;
    configuration.right_bound = 0;
    configuration.attention_scale = 0.8F;
    auto reference = attention_forward(configuration);

    deepforge::compiler::CompilationResult compilation;
    auto status = compile_document(bottom_right_sdpa_graph(), compilation);
    tests.good(status, "compile bottom-right variable-length SDPA");
    if (status.is_bad() || !compilation.executable) return;
    std::vector<float> o(reference.o.size(), -99.0F);
    std::vector<float> stats(reference.stats.size(), -99.0F);
    deepforge::runtime::VariantPack pack{{501, configuration.q.data()},
                                         {502, configuration.k.data()},
                                         {503, configuration.v.data()},
                                         {504, configuration.seq_q.data()},
                                         {505, configuration.seq_kv.data()},
                                         {506, o.data()},
                                         {507, stats.data()}};
    status = compilation.executable->execute_variant(
        deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr);
    tests.good(status, "execute bottom-right variable-length SDPA");
    tests.check(close_vectors(o, reference.o) &&
                    close_vectors(stats, reference.stats),
                "bottom-right shifted window and fully masked rows match reference");
}

void run_rejection_tests(TestRunner& tests) {
    deepforge::compiler::CompilationResult rejected;
    auto invalid_rng = rng_graph();
    invalid_rng["tensors"]["1"]["data_type"] = "FLOAT";
    auto status = compile_document(invalid_rng, rejected);
    tests.check(status.code() ==
                    deepforge::import::ErrorCode::kUnsupportedDataType,
                "RNG rejects non-INT64 seed metadata");

    auto invalid_rope = rope_graph();
    invalid_rope["nodes"][0]["rope_dim"] = 3;
    status = compile_document(invalid_rope, rejected);
    tests.check(status.code() == deepforge::import::ErrorCode::kInvalidShape,
                "RoPE rejects odd rotation widths");

    auto invalid_stats = feature_sdpa_graph();
    invalid_stats["nodes"][0]["generate_stats"] = false;
    status = compile_document(invalid_stats, rejected);
    tests.check(status.code() == deepforge::import::ErrorCode::kInvalidValue,
                "SDPA rejects Stats inconsistent with generate_stats");

    auto unsupported_block = feature_sdpa_graph();
    unsupported_block["tensors"]["120"] =
        tensor("Block_mask", 120, {1, 1, 1, 1});
    unsupported_block["nodes"][0]["inputs"]["Block_mask"] = 120;
    status = compile_document(unsupported_block, rejected);
    tests.check(status.code() ==
                    deepforge::import::ErrorCode::kUnsupportedOperation,
                "SDPA reports block-mask deferral explicitly");

    auto invalid_bottom_right = feature_sdpa_graph();
    invalid_bottom_right["nodes"][0]["diagonal_alignment"] = "BOTTOM_RIGHT";
    invalid_bottom_right["nodes"][0]["right_bound"] = 0;
    status = compile_document(invalid_bottom_right, rejected);
    tests.check(status.code() ==
                    deepforge::import::ErrorCode::kUnsupportedOperation,
                "bottom-right attention rejects unsupported bias/dropout mix");
}

}  // namespace

int main() {
    TestRunner tests;
    run_rng_tests(tests);
    run_rope_tests(tests);
    run_feature_attention_tests(tests);
    run_internal_dropout_tests(tests);
    run_alibi_tests(tests);
    run_bottom_right_tests(tests);
    run_rejection_tests(tests);
    return tests.finish();
}
