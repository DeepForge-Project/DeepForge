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

Json strided_tensor(std::string name,
                    std::int64_t uid,
                    std::vector<std::int64_t> dimensions,
                    std::vector<std::int64_t> strides,
                    std::string data_type = "FLOAT",
                    bool is_virtual = false) {
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

Json tensor(std::string name,
            std::int64_t uid,
            std::vector<std::int64_t> dimensions,
            std::string data_type = "FLOAT",
            bool is_virtual = false) {
    auto strides = contiguous_strides(dimensions);
    return strided_tensor(std::move(name), uid, std::move(dimensions),
                          std::move(strides), std::move(data_type),
                          is_virtual);
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

std::size_t strided_offset4(std::vector<std::int64_t> const& strides,
                            std::int64_t a,
                            std::int64_t b,
                            std::int64_t c,
                            std::int64_t d) {
    return static_cast<std::size_t>(a * strides[0] + b * strides[1] +
                                    c * strides[2] + d * strides[3]);
}

std::vector<float> pack_bshd(
    std::vector<float> const& dense,
    std::array<std::int64_t, 4> const& dimensions,
    std::vector<std::int32_t> const& lengths) {
    std::vector<float> packed;
    for (std::int64_t batch = 0; batch < dimensions[0]; ++batch) {
        for (std::int64_t sequence = 0;
             sequence < lengths[static_cast<std::size_t>(batch)];
             ++sequence) {
            for (std::int64_t head = 0; head < dimensions[1]; ++head) {
                for (std::int64_t embedding = 0;
                     embedding < dimensions[3]; ++embedding) {
                    packed.push_back(dense[offset4(
                        dimensions, batch, head, sequence, embedding)]);
                }
            }
        }
    }
    return packed;
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
    if (!fixed_seed) {
        tensors["1"]["is_pass_by_value"] = true;
        tensors["2"]["is_pass_by_value"] = true;
    }
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

Json embedded_rng_graph() {
    auto graph = rng_graph();
    graph["graph_uid"] = 4011;
    graph["context"]["name"] = "embedded-rng";
    auto const seed = Json{{"index", 0}, {"value", -31}};
    auto const offset = Json{{"index", 0}, {"value", 19}};
    graph["tensors"]["1"]["pass_by_value"] = seed;
    graph["tensors"]["2"]["pass_by_value"] = offset;
    graph["pass_by_values"] =
        Json::object({{"1", seed}, {"2", offset}});
    return graph;
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
    std::vector<std::uint8_t> block_mask;
    std::vector<float> sink_token;
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
                 std::int64_t head,
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
    if (!configuration.block_mask.empty()) {
        auto const query_tiles = (configuration.q_dim[2] - 1) / 128 + 1;
        auto const key_tiles = (configuration.k_dim[2] - 1) / 128 + 1;
        auto const key_bytes = (key_tiles + 7) / 8;
        std::array<std::int64_t, 4> const mask_dim{
            configuration.q_dim[0], configuration.q_dim[1], query_tiles,
            key_bytes};
        auto const key_tile = key / 128;
        auto const byte = configuration.block_mask[offset4(
            mask_dim, batch, head, query / 128, key_tile / 8)];
        auto const bit = static_cast<std::uint8_t>(1U << (key_tile % 8));
        if ((byte & bit) == 0) return false;
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
                auto maximum = configuration.sink_token.empty()
                                   ? -std::numeric_limits<float>::infinity()
                                   : configuration.sink_token[
                                         static_cast<std::size_t>(h)];
                std::vector<float> scores(static_cast<std::size_t>(s_kv),
                                          maximum);
                for (std::int64_t k = 0; k < s_kv; ++k) {
                    if (!score_valid(configuration, n, h, q, k)) continue;
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
                double sum = configuration.sink_token.empty()
                                 ? 0.0
                                 : std::exp(
                                       configuration.sink_token[
                                           static_cast<std::size_t>(h)] -
                                       maximum);
                for (std::int64_t k = 0; k < s_kv; ++k) {
                    if (!score_valid(configuration, n, h, q, k)) continue;
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
                    if (score_valid(configuration, n, h, q, k) && sum > 0.0) {
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

Json dynamic_sdpa_graph() {
    Json tensors = Json::object();
    tensors["1301"] = tensor("Q", 1301, {2, 2, 3, 2});
    tensors["1302"] = tensor("K", 1302, {2, 1, 4, 2});
    tensors["1303"] = tensor("V", 1303, {2, 1, 4, 3});
    tensors["1304"] = tensor("O", 1304, {2, 2, 3, 3});
    tensors["1305"] = tensor("Stats", 1305, {2, 2, 3, 1});
    tensors["1306"] = tensor("Max", 1306, {2, 2, 3, 1});
    tensors["1307"] = tensor("Sum_exp", 1307, {2, 2, 3, 1});
    auto node = forward_attention_node(
        Json::object({{"Q", 1301}, {"K", 1302}, {"V", 1303}}),
        Json::object({{"O", 1304},
                      {"Stats", 1305},
                      {"Max", 1306},
                      {"Sum_exp", 1307}}),
        true, false, false, nullptr, nullptr, 0, "TOP_LEFT", 0.65F);
    auto graph = graph_document(4020, "dynamic_sdpa", Json::array({node}),
                                std::move(tensors));
    graph["context"]["is_override_shape_enabled"] = true;
    return graph;
}

Json ragged_sdpa_graph() {
    Json tensors = Json::object();
    auto q = strided_tensor("Q", 601, {2, 1, 3, 2}, {2, 2, 2, 1});
    q["ragged_offset_uid"] = 605;
    q["ragged_offset_name"] = "QO_offsets";
    tensors["601"] = std::move(q);
    tensors["602"] = tensor("K", 602, {2, 1, 3, 2});
    tensors["603"] = tensor("V", 603, {2, 1, 3, 2});
    tensors["604"] = tensor("SEQ_LEN_Q", 604, {2, 1, 1, 1}, "INT32");
    tensors["605"] = tensor("QO_offsets", 605, {3, 1, 1, 1}, "INT64");
    tensors["606"] = tensor("SEQ_LEN_KV", 606, {2, 1, 1, 1}, "INT32");
    auto o = strided_tensor("O", 607, {2, 1, 3, 2}, {2, 2, 2, 1});
    o["ragged_offset_uid"] = 605;
    o["ragged_offset_name"] = "QO_offsets";
    tensors["607"] = std::move(o);
    auto node = forward_attention_node(
        Json::object({{"Q", 601},
                      {"K", 602},
                      {"V", 603},
                      {"SEQ_LEN_Q", 604},
                      {"SEQ_LEN_KV", 606}}),
        Json::object({{"O", 607}}), false, false, true, nullptr, nullptr,
        nullptr, "TOP_LEFT", 0.65F);
    return graph_document(4008, "ragged_sdpa", Json::array({node}),
                          std::move(tensors));
}

Json fully_ragged_sdpa_graph() {
    Json tensors = Json::object();
    auto q = strided_tensor("Q", 801, {2, 2, 3, 2}, {12, 2, 4, 1});
    q["ragged_offset_uid"] = 811;
    q["ragged_offset_name"] = "Q_offsets";
    tensors["801"] = std::move(q);
    auto k = strided_tensor("K", 802, {2, 1, 3, 2}, {6, 2, 2, 1});
    k["ragged_offset_uid"] = 812;
    k["ragged_offset_name"] = "K_offsets";
    tensors["802"] = std::move(k);
    auto v = strided_tensor("V", 803, {2, 1, 3, 2}, {6, 2, 2, 1});
    v["ragged_offset_uid"] = 813;
    v["ragged_offset_name"] = "V_offsets";
    tensors["803"] = std::move(v);
    tensors["804"] = tensor("SEQ_LEN_Q", 804, {2, 1, 1, 1}, "INT32");
    tensors["805"] = tensor("SEQ_LEN_KV", 805, {2, 1, 1, 1}, "INT32");
    auto o = strided_tensor("O", 806, {2, 2, 3, 2}, {12, 2, 4, 1});
    o["ragged_offset_uid"] = 814;
    o["ragged_offset_name"] = "O_offsets";
    tensors["806"] = std::move(o);
    tensors["811"] = tensor("Q_offsets", 811, {3, 1, 1, 1}, "INT64");
    tensors["812"] = tensor("K_offsets", 812, {3, 1, 1, 1}, "INT32");
    tensors["813"] = tensor("V_offsets", 813, {3, 1, 1, 1}, "INT32");
    tensors["814"] = tensor("O_offsets", 814, {3, 1, 1, 1}, "INT64");
    auto node = forward_attention_node(
        Json::object({{"Q", 801},
                      {"K", 802},
                      {"V", 803},
                      {"SEQ_LEN_Q", 804},
                      {"SEQ_LEN_KV", 805}}),
        Json::object({{"O", 806}}), false, false, true, nullptr, nullptr,
        nullptr, "TOP_LEFT", 0.55F);
    return graph_document(4010, "fully_ragged_sdpa", Json::array({node}),
                          std::move(tensors));
}

Json ragged_row_sdpa_graph() {
    auto ragged = [](Json value, std::int64_t offset_uid,
                     std::string offset_name) {
        value["ragged_offset_uid"] = offset_uid;
        value["ragged_offset_name"] = std::move(offset_name);
        return value;
    };
    Json tensors = Json::object();
    tensors["821"] = ragged(
        strided_tensor("Q", 821, {2, 2, 3, 2}, {12, 2, 4, 1}), 831,
        "Q_offsets");
    tensors["822"] = ragged(
        strided_tensor("K", 822, {2, 1, 3, 2}, {6, 2, 2, 1}), 832,
        "K_offsets");
    tensors["823"] = ragged(
        strided_tensor("V", 823, {2, 1, 3, 2}, {6, 2, 2, 1}), 833,
        "V_offsets");
    tensors["824"] = tensor("SEQ_LEN_Q", 824, {2, 1, 1, 1}, "INT32");
    tensors["825"] = tensor("SEQ_LEN_KV", 825, {2, 1, 1, 1}, "INT32");
    tensors["826"] = ragged(
        strided_tensor("O", 826, {2, 2, 3, 2}, {12, 2, 4, 1}), 834,
        "O_offsets");
    for (auto const& [uid, name] :
         std::array<std::pair<std::int64_t, char const*>, 3>{
             std::pair{827, "Stats"}, std::pair{828, "Max"},
             std::pair{829, "Sum_exp"}}) {
        tensors[std::to_string(uid)] = ragged(
            strided_tensor(name, uid, {2, 2, 3, 1}, {6, 1, 2, 1}),
            835, "Row_offsets");
    }
    tensors["831"] = tensor("Q_offsets", 831, {3, 1, 1, 1}, "INT64");
    tensors["832"] = tensor("K_offsets", 832, {3, 1, 1, 1}, "INT32");
    tensors["833"] = tensor("V_offsets", 833, {3, 1, 1, 1}, "INT64");
    tensors["834"] = tensor("O_offsets", 834, {3, 1, 1, 1}, "INT32");
    tensors["835"] = tensor("Row_offsets", 835, {3, 1, 1, 1}, "INT64");
    auto node = forward_attention_node(
        Json::object({{"Q", 821},
                      {"K", 822},
                      {"V", 823},
                      {"SEQ_LEN_Q", 824},
                      {"SEQ_LEN_KV", 825}}),
        Json::object({{"O", 826},
                      {"Stats", 827},
                      {"Max", 828},
                      {"Sum_exp", 829}}),
        true, false, true, nullptr, nullptr, nullptr, "TOP_LEFT", 0.55F);
    return graph_document(4011, "ragged_row_sdpa", Json::array({node}),
                          std::move(tensors));
}

Json ragged_sdpa_backward_graph() {
    auto ragged = [](Json value, std::int64_t offset_uid,
                     std::string offset_name) {
        value["ragged_offset_uid"] = offset_uid;
        value["ragged_offset_name"] = std::move(offset_name);
        return value;
    };
    Json tensors = Json::object();
    auto qo = [&](std::string name, std::int64_t uid,
                  std::int64_t offset_uid, std::string offset_name) {
        return ragged(strided_tensor(std::move(name), uid, {2, 2, 3, 2},
                                     {12, 2, 4, 1}),
                      offset_uid, std::move(offset_name));
    };
    auto kv = [&](std::string name, std::int64_t uid,
                  std::int64_t offset_uid, std::string offset_name) {
        return ragged(strided_tensor(std::move(name), uid, {2, 1, 3, 2},
                                     {6, 2, 2, 1}),
                      offset_uid, std::move(offset_name));
    };
    tensors["901"] = qo("Q", 901, 921, "Q_offsets");
    tensors["902"] = kv("K", 902, 922, "K_offsets");
    tensors["903"] = kv("V", 903, 923, "V_offsets");
    tensors["904"] = qo("O", 904, 924, "O_offsets");
    tensors["905"] = qo("dO", 905, 924, "O_offsets");
    tensors["906"] = ragged(
        strided_tensor("Stats", 906, {2, 2, 3, 1}, {6, 1, 2, 1}),
        925, "Stats_offsets");
    tensors["907"] = tensor("SEQ_LEN_Q", 907, {2, 1, 1, 1}, "INT32");
    tensors["908"] = tensor("SEQ_LEN_KV", 908, {2, 1, 1, 1}, "INT32");
    tensors["909"] = qo("dQ", 909, 921, "Q_offsets");
    tensors["910"] = kv("dK", 910, 922, "K_offsets");
    tensors["911"] = kv("dV", 911, 923, "V_offsets");
    tensors["921"] = tensor("Q_offsets", 921, {3, 1, 1, 1}, "INT64");
    tensors["922"] = tensor("K_offsets", 922, {3, 1, 1, 1}, "INT32");
    tensors["923"] = tensor("V_offsets", 923, {3, 1, 1, 1}, "INT64");
    tensors["924"] = tensor("O_offsets", 924, {3, 1, 1, 1}, "INT32");
    tensors["925"] = tensor("Stats_offsets", 925, {3, 1, 1, 1}, "INT64");
    Json node{{"tag", "SDPA_BWD"},
              {"name", "ragged_sdpa_backward"},
              {"inputs",
               Json::object({{"Q", 901},
                             {"K", 902},
                             {"V", 903},
                             {"O", 904},
                             {"dO", 905},
                             {"Stats", 906},
                             {"SEQ_LEN_Q", 907},
                             {"SEQ_LEN_KV", 908}})},
              {"outputs",
               Json::object({{"dQ", 909}, {"dK", 910}, {"dV", 911}})},
              {"alibi_mask", false},
              {"padding_mask", true},
              {"dropout_probability", nullptr},
              {"attn_scale_value", 0.55},
              {"left_bound", nullptr},
              {"right_bound", nullptr},
              {"diagonal_alignment", "TOP_LEFT"},
              {"max_total_seq_len_q", 3},
              {"max_total_seq_len_kv", 5},
              {"is_deterministic_algorithm", false}};
    return graph_document(4012, "ragged_sdpa_backward", Json::array({node}),
                          std::move(tensors));
}

Json paged_sdpa_graph() {
    Json tensors = Json::object();
    tensors["701"] = tensor("Q", 701, {2, 1, 1, 2});
    tensors["702"] = tensor("K_container", 702, {4, 1, 2, 2});
    tensors["703"] = tensor("V_container", 703, {4, 1, 2, 2});
    tensors["704"] = tensor("Page_table_K", 704, {2, 1, 2, 1}, "INT32");
    tensors["705"] = tensor("Page_table_V", 705, {2, 1, 2, 1}, "INT32");
    tensors["706"] = tensor("SEQ_LEN_Q", 706, {2, 1, 1, 1}, "INT32");
    tensors["707"] = tensor("SEQ_LEN_KV", 707, {2, 1, 1, 1}, "INT32");
    tensors["708"] = tensor("O", 708, {2, 1, 1, 2});
    auto node = forward_attention_node(
        Json::object({{"Q", 701},
                      {"K", 702},
                      {"V", 703},
                      {"Page_table_K", 704},
                      {"Page_table_V", 705},
                      {"SEQ_LEN_Q", 706},
                      {"SEQ_LEN_KV", 707}}),
        Json::object({{"O", 708}}), false, false, true, nullptr, nullptr,
        nullptr, "TOP_LEFT", 0.5F);
    node["max_seq_len_kv"] = 3;
    return graph_document(4009, "paged_sdpa", Json::array({node}),
                          std::move(tensors));
}

Json block_mask_sdpa_graph() {
    Json tensors = Json::object();
    tensors["1101"] = tensor("Q", 1101, {1, 1, 129, 1});
    tensors["1102"] = tensor("K", 1102, {1, 1, 1025, 1});
    tensors["1103"] = tensor("V", 1103, {1, 1, 1025, 1});
    tensors["1104"] = strided_tensor("Block_mask", 1104, {1, 1, 2, 2},
                                      {8, 8, 4, 1}, "UINT8");
    tensors["1105"] = tensor("O", 1105, {1, 1, 129, 1});
    tensors["1106"] = tensor("Stats", 1106, {1, 1, 129, 1});
    tensors["1107"] = tensor("Max", 1107, {1, 1, 129, 1});
    tensors["1108"] = tensor("Sum_exp", 1108, {1, 1, 129, 1});
    auto node = forward_attention_node(
        Json::object({{"Q", 1101},
                      {"K", 1102},
                      {"V", 1103},
                      {"Block_mask", 1104}}),
        Json::object({{"O", 1105},
                      {"Stats", 1106},
                      {"Max", 1107},
                      {"Sum_exp", 1108}}),
        true, false, false, nullptr, nullptr, nullptr, "TOP_LEFT", 0.75F);
    return graph_document(4013, "block_mask_sdpa", Json::array({node}),
                          std::move(tensors));
}

Json sink_sdpa_graph() {
    Json tensors = Json::object();
    tensors["1201"] = tensor("Q", 1201, {2, 2, 2, 2});
    tensors["1202"] = tensor("K", 1202, {2, 1, 3, 2});
    tensors["1203"] = tensor("V", 1203, {2, 1, 3, 2});
    tensors["1204"] = tensor("SINK_TOKEN", 1204, {1, 2, 1, 1});
    tensors["1205"] = tensor("SEQ_LEN_Q", 1205, {2, 1, 1, 1}, "INT32");
    tensors["1206"] = tensor("SEQ_LEN_KV", 1206, {2, 1, 1, 1}, "INT32");
    tensors["1207"] = tensor("O", 1207, {2, 2, 2, 2});
    tensors["1208"] = tensor("Stats", 1208, {2, 2, 2, 1});
    tensors["1209"] = tensor("Max", 1209, {2, 2, 2, 1});
    tensors["1210"] = tensor("Sum_exp", 1210, {2, 2, 2, 1});
    tensors["1211"] = tensor("Dropout_mask", 1211, {2, 2, 2, 3});
    tensors["1212"] = tensor("Dropout_scale", 1212, {1, 1, 1, 1});
    auto node = forward_attention_node(
        Json::object({{"Q", 1201},
                      {"K", 1202},
                      {"V", 1203},
                      {"SINK_TOKEN", 1204},
                      {"SEQ_LEN_Q", 1205},
                      {"SEQ_LEN_KV", 1206},
                      {"Dropout_mask", 1211},
                      {"Dropout_scale", 1212}}),
        Json::object({{"O", 1207},
                      {"Stats", 1208},
                      {"Max", 1209},
                      {"Sum_exp", 1210}}),
        true, false, true, nullptr, nullptr, nullptr, "TOP_LEFT", 0.6F);
    return graph_document(4014, "sink_sdpa", Json::array({node}),
                          std::move(tensors));
}

Json sink_sdpa_backward_graph() {
    Json tensors = Json::object();
    tensors["1221"] = tensor("Q", 1221, {2, 2, 2, 2});
    tensors["1222"] = tensor("K", 1222, {2, 1, 3, 2});
    tensors["1223"] = tensor("V", 1223, {2, 1, 3, 2});
    tensors["1224"] = tensor("O", 1224, {2, 2, 2, 2});
    tensors["1225"] = tensor("dO", 1225, {2, 2, 2, 2});
    tensors["1226"] = tensor("Stats", 1226, {2, 2, 2, 1});
    tensors["1227"] = tensor("SINK_TOKEN", 1227, {1, 2, 1, 1});
    tensors["1228"] = tensor("SEQ_LEN_Q", 1228, {2, 1, 1, 1}, "INT32");
    tensors["1229"] = tensor("SEQ_LEN_KV", 1229, {2, 1, 1, 1}, "INT32");
    tensors["1230"] = tensor("dQ", 1230, {2, 2, 2, 2});
    tensors["1231"] = tensor("dK", 1231, {2, 1, 3, 2});
    tensors["1232"] = tensor("dV", 1232, {2, 1, 3, 2});
    tensors["1233"] = tensor("DSINK_TOKEN", 1233, {1, 2, 1, 1});
    tensors["1234"] = tensor("Dropout_mask", 1234, {2, 2, 2, 3});
    tensors["1235"] = tensor("Dropout_scale", 1235, {1, 1, 1, 1});
    tensors["1236"] = tensor("Dropout_scale_inv", 1236, {1, 1, 1, 1});
    Json node{{"tag", "SDPA_BWD"},
              {"name", "sink_sdpa_backward"},
              {"inputs",
               Json::object({{"Q", 1221},
                             {"K", 1222},
                             {"V", 1223},
                             {"O", 1224},
                             {"dO", 1225},
                             {"Stats", 1226},
                             {"SINK_TOKEN", 1227},
                             {"SEQ_LEN_Q", 1228},
                             {"SEQ_LEN_KV", 1229},
                             {"Dropout_mask", 1234},
                             {"Dropout_scale", 1235},
                             {"Dropout_scale_inv", 1236}})},
              {"outputs",
               Json::object({{"dQ", 1230},
                             {"dK", 1231},
                             {"dV", 1232},
                             {"DSINK_TOKEN", 1233}})},
              {"alibi_mask", false},
              {"padding_mask", true},
              {"dropout_probability", nullptr},
              {"attn_scale_value", 0.6},
              {"left_bound", nullptr},
              {"right_bound", nullptr},
              {"diagonal_alignment", "TOP_LEFT"},
              {"max_total_seq_len_q", nullptr},
              {"max_total_seq_len_kv", nullptr},
              {"is_deterministic_algorithm", false}};
    return graph_document(4015, "sink_sdpa_backward", Json::array({node}),
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
    std::vector<float> d_sink;
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
    result.d_sink.assign(configuration.sink_token.size(), 0.0F);
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
                if (!result.d_sink.empty() &&
                    (!configuration.padding ||
                     q < configuration.seq_q[static_cast<std::size_t>(n)])) {
                    auto const row = static_cast<std::size_t>(
                        (n * h_q + h) * s_q + q);
                    result.d_sink[static_cast<std::size_t>(h)] -=
                        std::exp(configuration.sink_token[
                                     static_cast<std::size_t>(h)] -
                                 forward.stats[row]) *
                        static_cast<float>(output_dot);
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
                    if (!result.d_bias.empty()) {
                        result.d_bias[offset4(dbias_dim, 0, h, 0, k)] += base;
                    }
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

std::vector<float> finite_difference_sink(AttentionCase configuration,
                                          std::vector<float> const& d_o) {
    constexpr float epsilon = 1.0e-3F;
    std::vector<float> gradient(configuration.sink_token.size());
    for (std::size_t index = 0; index < configuration.sink_token.size();
         ++index) {
        configuration.sink_token[index] += epsilon;
        auto const positive = attention_loss(configuration, d_o);
        configuration.sink_token[index] -= 2.0F * epsilon;
        auto const negative = attention_loss(configuration, d_o);
        configuration.sink_token[index] += epsilon;
        gradient[index] = static_cast<float>((positive - negative) /
                                             (2.0 * epsilon));
    }
    return gradient;
}

void run_rng_tests(TestRunner& tests) {
    deepforge::compiler::CompilationResult compilation;
    auto status = compile_document(rng_graph(), compilation);
    tests.good(status, "compile runtime-PBV-seed RNG");
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
    tests.good(status, "execute runtime-PBV-seed RNG");
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

    deepforge::compiler::CompilationResult embedded;
    status = compile_document(embedded_rng_graph(), embedded);
    tests.good(status, "compile RNG with embedded PBV seed and offset");
    tests.check(
        std::none_of(embedded.metadata.arguments.begin(),
                     embedded.metadata.arguments.end(), [](auto const& argument) {
                         return argument.uid == 1 || argument.uid == 2;
                     }),
        "embedded RNG scalars are absent from public arguments");
    if (status.is_good() && embedded.executable) {
        std::vector<float> output(6, -1.0F);
        deepforge::runtime::VariantPack embedded_pack{{3, output.data()}};
        status = embedded.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr, embedded_pack,
            nullptr);
        tests.good(status, "execute RNG without embedded scalar UIDs");
        tests.check(output == expected,
                    "embedded INT64 RNG scalars preserve stream semantics");
    }

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

void run_dynamic_attention_tests(TestRunner& tests) {
    AttentionCase configuration;
    configuration.q_dim = {1, 2, 2, 2};
    configuration.k_dim = {1, 1, 3, 2};
    configuration.v_dim = {1, 1, 3, 3};
    configuration.o_dim = {1, 2, 2, 3};
    configuration.q = {0.2F, -0.3F, 0.7F, 0.1F,
                       -0.5F, 0.4F, 0.3F, -0.8F};
    configuration.k = {0.6F, -0.2F, -0.4F, 0.9F, 0.1F, 0.5F};
    configuration.v = {0.3F, -0.7F, 0.2F, 0.8F, 0.2F,
                       -0.4F, -0.6F, 0.4F, 0.9F};
    configuration.padding = false;
    configuration.left_bound.reset();
    configuration.right_bound = 0;
    configuration.attention_scale = 0.65F;
    configuration.dropout_scale = 1.0F;
    auto const reference = attention_forward(configuration);

    deepforge::import::SerializedGraph graph;
    auto status = parse_graph(dynamic_sdpa_graph(), graph);
    tests.good(status, "parse runtime shape-override SDPA");
    deepforge::compiler::CompilationResult compilation;
    if (status.is_good()) {
        deepforge::compiler::CompileOptions options;
        options.capture_mlir = true;
        status = deepforge::compiler::compile_graph(graph, options,
                                                    compilation);
    }
    tests.good(status, "compile runtime shape-override SDPA");
    if (status.is_bad() || !compilation.executable) return;
    tests.check(
        !compilation.metadata.dynamic_shape_enabled &&
            compilation.metadata.override_shape_enabled &&
            compilation.metadata.override_policy ==
                deepforge::compiler::ShapeOverridePolicy::kSdpaForward &&
            compilation.metadata.override_role_uids ==
                std::vector<std::int64_t>(
                    {1301, 1302, 1303, 1304, 1305, 1306, 1307}) &&
            compilation.imported_mlir.find("memref.dim") !=
                std::string::npos &&
            compilation.imported_mlir.find("?x?x?x?xf32") !=
                std::string::npos,
        "SDPA override metadata and MLIR use runtime descriptors independently of the dynamic flag");

    deepforge::runtime::OverrideUids const override_uids{
        1301, 1302, 1303, 1304, 1305, 1306, 1307};
    deepforge::runtime::OverrideShapes const override_shapes{
        {1, 2, 2, 2}, {1, 1, 3, 2}, {1, 1, 3, 3}, {1, 2, 2, 3},
        {1, 2, 2, 1}, {1, 2, 2, 1}, {1, 2, 2, 1}};
    deepforge::runtime::OverrideStrides const override_strides{
        {20, 8, 3, 1}, {15, 12, 3, 1}, {22, 20, 4, 1},
        {30, 12, 4, 1}, {11, 5, 2, 1}, {11, 5, 2, 1},
        {11, 5, 2, 1}};

    std::vector<float> q(24, -77.0F);
    std::vector<float> k(16, -77.0F);
    std::vector<float> v(24, -77.0F);
    std::vector<float> o(36, -99.0F);
    std::vector<float> stats(12, -99.0F);
    std::vector<float> maximum(12, -99.0F);
    std::vector<float> sum(12, -99.0F);
    auto scatter = [](std::vector<float> const& dense,
                      std::array<std::int64_t, 4> const& dimensions,
                      std::vector<std::int64_t> const& strides,
                      std::vector<float>& strided) {
        for (std::int64_t a = 0; a < dimensions[0]; ++a) {
            for (std::int64_t b = 0; b < dimensions[1]; ++b) {
                for (std::int64_t c = 0; c < dimensions[2]; ++c) {
                    for (std::int64_t d = 0; d < dimensions[3]; ++d) {
                        strided[strided_offset4(strides, a, b, c, d)] =
                            dense[offset4(dimensions, a, b, c, d)];
                    }
                }
            }
        }
    };
    scatter(configuration.q, configuration.q_dim, override_strides[0], q);
    scatter(configuration.k, configuration.k_dim, override_strides[1], k);
    scatter(configuration.v, configuration.v_dim, override_strides[2], v);

    deepforge::runtime::VariantPack pack{{1301, q.data()},
                                          {1302, k.data()},
                                          {1303, v.data()},
                                          {1304, o.data()},
                                          {1305, stats.data()},
                                          {1306, maximum.data()},
                                          {1307, sum.data()}};
    status = compilation.executable->execute_variant(
        deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr,
        override_uids, override_shapes, override_strides);
    tests.good(status, "execute runtime shape-override SDPA");

    auto matches_strided = [](std::vector<float> const& actual,
                              std::array<std::int64_t, 4> const& dimensions,
                              std::vector<std::int64_t> const& strides,
                              std::vector<float> const& expected) {
        std::vector<float> gathered;
        std::vector<bool> occupied(actual.size(), false);
        gathered.reserve(expected.size());
        for (std::int64_t a = 0; a < dimensions[0]; ++a) {
            for (std::int64_t b = 0; b < dimensions[1]; ++b) {
                for (std::int64_t c = 0; c < dimensions[2]; ++c) {
                    for (std::int64_t d = 0; d < dimensions[3]; ++d) {
                        auto const index =
                            strided_offset4(strides, a, b, c, d);
                        occupied[index] = true;
                        gathered.push_back(actual[index]);
                    }
                }
            }
        }
        bool gaps_untouched = true;
        for (std::size_t index = 0; index < actual.size(); ++index) {
            gaps_untouched = gaps_untouched &&
                             (occupied[index] || actual[index] == -99.0F);
        }
        return gaps_untouched && close_vectors(gathered, expected);
    };
    std::array<std::int64_t, 4> const row_dimensions{1, 2, 2, 1};
    tests.check(
        matches_strided(o, configuration.o_dim, override_strides[3],
                        reference.o) &&
            matches_strided(stats, row_dimensions, override_strides[4],
                            reference.stats) &&
            matches_strided(maximum, row_dimensions, override_strides[5],
                            reference.maximum) &&
            matches_strided(sum, row_dimensions, override_strides[6],
                            reference.sum_exp),
        "dynamic SDPA honors B/Sq/Skv, GQA, causal masking, row outputs, and non-contiguous strides");

    std::vector<std::uint8_t> artifact;
    status = deepforge::compiler::serialize_artifact(compilation, artifact);
    tests.good(status, "serialize runtime shape-override SDPA artifact");
    std::unique_ptr<deepforge::runtime::Executable> loaded;
    deepforge::compiler::ArtifactInfo artifact_info;
    if (status.is_good()) {
        status = deepforge::compiler::load_artifact_executable(
            artifact, loaded, &artifact_info);
    }
    tests.good(status, "load runtime shape-override SDPA artifact");
    tests.check(
        artifact_info.format_version ==
                deepforge::compiler::kArtifactFormatVersion &&
            artifact_info.metadata.override_policy ==
                deepforge::compiler::ShapeOverridePolicy::kSdpaForward &&
            artifact_info.metadata.override_role_uids == override_uids,
        "artifact v9 preserves ordered SDPA override roles");
    if (loaded) {
        std::fill(o.begin(), o.end(), -99.0F);
        std::fill(stats.begin(), stats.end(), -99.0F);
        std::fill(maximum.begin(), maximum.end(), -99.0F);
        std::fill(sum.begin(), sum.end(), -99.0F);
        status = loaded->execute(nullptr, pack, nullptr, override_uids,
                                 override_shapes, override_strides);
        tests.good(status, "execute loaded shape-override SDPA artifact");
        tests.check(matches_strided(o, configuration.o_dim,
                                    override_strides[3], reference.o) &&
                        matches_strided(stats, row_dimensions,
                                        override_strides[4], reference.stats),
                    "loaded dynamic SDPA preserves runtime descriptor semantics");
    }

    std::int64_t workspace_size = -1;
    status = compilation.executable->get_workspace_size(
        nullptr, workspace_size, {1302, 1303},
        {{2, 1, 3, 2}, {2, 1, 3, 3}},
        {{8, 8, 2, 1}, {12, 12, 3, 1}});
    tests.good(status,
               "partial SDPA override may shrink K and V sequence together");
    tests.check(workspace_size == 0,
                "partial SDPA override preserves the static workspace bound");
    status = compilation.executable->get_workspace_size(
        nullptr, workspace_size, {1301}, {{1, 2, 2, 2}},
        {{12, 6, 2, 1}});
    tests.check(status.code() == deepforge::import::ErrorCode::kInvalidShape,
                "partial SDPA override must preserve Q/K/V/O relations");

    auto invalid_shapes = override_shapes;
    invalid_shapes[3] = {1, 2, 1, 3};
    status = compilation.executable->execute_variant(
        deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr,
        override_uids, invalid_shapes, override_strides);
    tests.check(status.code() == deepforge::import::ErrorCode::kInvalidShape,
                "dynamic SDPA rejects an O sequence inconsistent with Q");
    invalid_shapes = override_shapes;
    invalid_shapes[2] = {1, 1, 2, 3};
    status = compilation.executable->execute_variant(
        deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr,
        override_uids, invalid_shapes, override_strides);
    tests.check(status.code() == deepforge::import::ErrorCode::kInvalidShape,
                "dynamic SDPA rejects inconsistent K/V sequence lengths");
    invalid_shapes = override_shapes;
    invalid_shapes[0] = {1, 1, 2, 2};
    status = compilation.executable->execute_variant(
        deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr,
        override_uids, invalid_shapes, override_strides);
    tests.check(status.code() == deepforge::import::ErrorCode::kInvalidShape,
                "dynamic SDPA rejects runtime head-count changes");
    invalid_shapes = override_shapes;
    invalid_shapes[4] = {1, 2, 1, 1};
    status = compilation.executable->execute_variant(
        deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr,
        override_uids, invalid_shapes, override_strides);
    tests.check(status.code() == deepforge::import::ErrorCode::kInvalidShape,
                "dynamic SDPA rejects a row output inconsistent with Q");

    deepforge::compiler::CompilationResult rejected;
    auto minimal = dynamic_sdpa_graph();
    minimal["nodes"][0]["outputs"].erase("Stats");
    minimal["nodes"][0]["outputs"].erase("Max");
    minimal["nodes"][0]["outputs"].erase("Sum_exp");
    minimal["nodes"][0]["generate_stats"] = false;
    minimal["nodes"][0]["right_bound"] = nullptr;
    status = compile_document(minimal, rejected);
    tests.good(status, "compile minimal unmasked four-role SDPA override");
    tests.check(
        rejected.metadata.override_role_uids ==
            std::vector<std::int64_t>({1301, 1302, 1303, 1304}),
        "minimal SDPA override records only Q/K/V/O roles");
    auto padded = dynamic_sdpa_graph();
    padded["nodes"][0]["padding_mask"] = true;
    status = compile_document(padded, rejected);
    tests.check(status.code() ==
                    deepforge::import::ErrorCode::kUnsupportedOperation,
                "SDPA overrides reject padding and sequence metadata");
    auto bottom_right = dynamic_sdpa_graph();
    bottom_right["nodes"][0]["diagonal_alignment"] = "BOTTOM_RIGHT";
    status = compile_document(bottom_right, rejected);
    tests.check(status.code() ==
                    deepforge::import::ErrorCode::kUnsupportedOperation,
                "SDPA overrides reject bottom-right masking");
    auto dropout = dynamic_sdpa_graph();
    dropout["nodes"][0]["dropout_probability"] = 0.1;
    status = compile_document(dropout, rejected);
    tests.check(status.code() ==
                    deepforge::import::ErrorCode::kUnsupportedOperation,
                "SDPA overrides reject probability dropout");
    auto virtual_q = dynamic_sdpa_graph();
    virtual_q["tensors"]["1305"]["is_virtual"] = true;
    status = compile_document(virtual_q, rejected);
    tests.check(status.code() ==
                    deepforge::import::ErrorCode::kUnsupportedOperation,
                "SDPA overrides reject virtual role tensors");
    auto composed = dynamic_sdpa_graph();
    composed["tensors"]["1308"] = tensor("postprocessed", 1308,
                                           {2, 2, 3, 3});
    composed["nodes"].push_back(
        Json{{"tag", "POINTWISE"},
             {"name", "postprocess"},
             {"inputs", Json::object({{"IN_0", 1304}})},
             {"outputs", Json::object({{"OUT_0", 1308}})},
             {"compute_data_type", "FLOAT"},
             {"mode", "IDENTITY"},
             {"axis", nullptr},
             {"relu_lower_clip", nullptr},
             {"relu_upper_clip", nullptr},
             {"relu_lower_clip_slope", nullptr},
             {"swish_beta", nullptr},
             {"elu_alpha", nullptr},
             {"softplus_beta", nullptr}});
    status = compile_document(composed, rejected);
    tests.check(status.code() ==
                    deepforge::import::ErrorCode::kUnsupportedOperation,
                "SDPA overrides reject composed graphs");
}

void run_ragged_attention_tests(TestRunner& tests) {
    AttentionCase configuration;
    configuration.q_dim = {2, 1, 3, 2};
    configuration.k_dim = {2, 1, 3, 2};
    configuration.v_dim = {2, 1, 3, 2};
    configuration.o_dim = {2, 1, 3, 2};
    configuration.q = {0.2F, -0.3F, 0.7F, 0.1F, -9.0F, -9.0F,
                       -0.5F, 0.4F, -9.0F, -9.0F, -9.0F, -9.0F};
    configuration.k = {0.6F, -0.2F, -0.4F, 0.9F, 0.1F, 0.5F,
                       -0.3F, 0.8F, 0.7F, -0.6F, 0.2F, 0.4F};
    configuration.v = {0.3F, -0.7F, 0.8F, 0.2F, -0.6F, 0.4F,
                       0.5F, -0.1F, -0.2F, 0.9F, 0.6F, -0.8F};
    configuration.bias.clear();
    configuration.dropout_mask.clear();
    configuration.dropout_scale = 1.0F;
    configuration.dropout_scale_inv = 1.0F;
    configuration.padding = true;
    configuration.seq_q = {2, 1};
    configuration.seq_kv = {3, 2};
    configuration.left_bound.reset();
    configuration.right_bound.reset();
    configuration.attention_scale = 0.65F;
    auto reference = attention_forward(configuration);

    std::vector<float> compact_q;
    std::vector<float> expected_o;
    for (std::int64_t batch = 0; batch < configuration.q_dim[0]; ++batch) {
        for (std::int64_t sequence = 0;
             sequence < configuration.seq_q[static_cast<std::size_t>(batch)];
             ++sequence) {
            for (std::int64_t embedding = 0;
                 embedding < configuration.q_dim[3]; ++embedding) {
                compact_q.push_back(configuration.q[offset4(
                    configuration.q_dim, batch, 0, sequence, embedding)]);
                expected_o.push_back(reference.o[offset4(
                    configuration.o_dim, batch, 0, sequence, embedding)]);
            }
        }
    }
    std::vector<std::int64_t> offsets{0, 4, 6};
    std::vector<float> output(expected_o.size(), -99.0F);

    deepforge::compiler::CompilationResult compilation;
    auto status = compile_document(ragged_sdpa_graph(), compilation);
    tests.good(status, "compile compact ragged SDPA forward");
    if (status.is_bad() || !compilation.executable) return;
    auto const q_metadata = std::find_if(
        compilation.metadata.arguments.begin(),
        compilation.metadata.arguments.end(), [](auto const& argument) {
            return argument.uid == 601;
        });
    tests.check(
        q_metadata != compilation.metadata.arguments.end() &&
            q_metadata->storage_policy ==
                deepforge::compiler::TensorStoragePolicy::kRaggedBatchPrefix &&
            q_metadata->ragged_offset_uid == 605 &&
            q_metadata->ragged_sequence_uid == 604,
        "ragged storage policy records offset and sequence UIDs");

    deepforge::runtime::VariantPack pack{{601, compact_q.data()},
                                         {602, configuration.k.data()},
                                         {603, configuration.v.data()},
                                         {604, configuration.seq_q.data()},
                                         {605, offsets.data()},
                                         {606, configuration.seq_kv.data()},
                                         {607, output.data()}};
    status = compilation.executable->execute_variant(
        deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr);
    tests.good(status, "execute exact-allocation ragged SDPA forward");
    tests.check(close_vectors(output, expected_o),
                "ragged Q/O element offsets match dense SDPA reference");

    std::vector<std::uint8_t> artifact;
    status = deepforge::compiler::serialize_artifact(compilation, artifact);
    tests.good(status, "serialize ragged artifact v5");
    std::unique_ptr<deepforge::runtime::Executable> loaded;
    if (status.is_good()) {
        status = deepforge::compiler::load_artifact_executable(artifact,
                                                               loaded);
    }
    tests.good(status, "load ragged artifact v5");
    if (loaded) {
        std::fill(output.begin(), output.end(), -99.0F);
        status = loaded->execute(nullptr, pack, nullptr);
        tests.good(status, "execute reloaded ragged artifact");
        tests.check(close_vectors(output, expected_o),
                    "artifact preserves ragged runtime validation and addressing");
    }

    offsets = {0, 3, 6};
    status = compilation.executable->execute_variant(
        deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr);
    tests.check(status.code() ==
                    deepforge::import::ErrorCode::kInvalidVariantPack,
                "runtime rejects ragged segments shorter than sequence metadata");

    AttentionCase packed_configuration;
    packed_configuration.q_dim = {2, 2, 3, 2};
    packed_configuration.k_dim = {2, 1, 3, 2};
    packed_configuration.v_dim = {2, 1, 3, 2};
    packed_configuration.o_dim = {2, 2, 3, 2};
    packed_configuration.q.resize(24);
    packed_configuration.k.resize(12);
    packed_configuration.v.resize(12);
    for (std::size_t index = 0; index < packed_configuration.q.size();
         ++index) {
        packed_configuration.q[index] =
            static_cast<float>(static_cast<int>(index % 13) - 6) / 7.0F;
    }
    for (std::size_t index = 0; index < packed_configuration.k.size();
         ++index) {
        packed_configuration.k[index] =
            static_cast<float>(static_cast<int>(index % 9) - 4) / 6.0F;
        packed_configuration.v[index] =
            static_cast<float>(static_cast<int>((index * 3) % 11) - 5) /
            8.0F;
    }
    packed_configuration.bias.clear();
    packed_configuration.dropout_mask.clear();
    packed_configuration.dropout_scale = 1.0F;
    packed_configuration.dropout_scale_inv = 1.0F;
    packed_configuration.padding = true;
    packed_configuration.seq_q = {2, 1};
    packed_configuration.seq_kv = {3, 2};
    packed_configuration.left_bound.reset();
    packed_configuration.right_bound.reset();
    packed_configuration.attention_scale = 0.55F;
    auto packed_reference = attention_forward(packed_configuration);
    auto packed_q = pack_bshd(packed_configuration.q,
                              packed_configuration.q_dim,
                              packed_configuration.seq_q);
    auto packed_k = pack_bshd(packed_configuration.k,
                              packed_configuration.k_dim,
                              packed_configuration.seq_kv);
    auto packed_v = pack_bshd(packed_configuration.v,
                              packed_configuration.v_dim,
                              packed_configuration.seq_kv);
    auto packed_expected_o = pack_bshd(
        packed_reference.o, packed_configuration.o_dim,
        packed_configuration.seq_q);
    std::vector<float> packed_o(packed_expected_o.size(), -99.0F);
    std::vector<std::int64_t> q_offsets{0, 8, 12};
    std::vector<std::int32_t> k_offsets{0, 6, 10};
    std::vector<std::int32_t> v_offsets{0, 6, 10};
    std::vector<std::int64_t> o_offsets{0, 8, 12};

    deepforge::compiler::CompilationResult packed_compilation;
    status = compile_document(fully_ragged_sdpa_graph(), packed_compilation);
    tests.good(status, "compile producer-stride ragged Q/K/V/O SDPA");
    if (status.is_good() && packed_compilation.executable) {
        deepforge::runtime::VariantPack packed_pack{
            {801, packed_q.data()},
            {802, packed_k.data()},
            {803, packed_v.data()},
            {804, packed_configuration.seq_q.data()},
            {805, packed_configuration.seq_kv.data()},
            {806, packed_o.data()},
            {811, q_offsets.data()},
            {812, k_offsets.data()},
            {813, v_offsets.data()},
            {814, o_offsets.data()}};
        status = packed_compilation.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr, packed_pack,
            nullptr);
        tests.good(status, "execute producer-stride ragged Q/K/V/O SDPA");
        tests.check(close_vectors(packed_o, packed_expected_o),
                    "BSHD-packed inner strides and independent prefixes match reference");
    }

    std::array<std::int64_t, 4> const row_dimensions{2, 2, 3, 1};
    auto packed_stats = pack_bshd(packed_reference.stats, row_dimensions,
                                  packed_configuration.seq_q);
    auto packed_maximum = pack_bshd(packed_reference.maximum, row_dimensions,
                                    packed_configuration.seq_q);
    auto packed_sum = pack_bshd(packed_reference.sum_exp, row_dimensions,
                                packed_configuration.seq_q);
    std::vector<float> row_o(packed_expected_o.size(), -99.0F);
    std::vector<float> row_stats(packed_stats.size(), -99.0F);
    std::vector<float> row_maximum(packed_maximum.size(), -99.0F);
    std::vector<float> row_sum(packed_sum.size(), -99.0F);
    std::vector<std::int64_t> row_v_offsets{0, 6, 10};
    std::vector<std::int32_t> row_o_offsets{0, 8, 12};
    std::vector<std::int64_t> row_offsets{0, 4, 6};
    deepforge::compiler::CompilationResult row_compilation;
    status = compile_document(ragged_row_sdpa_graph(), row_compilation);
    tests.good(status, "compile ragged SDPA row outputs");
    if (status.is_good() && row_compilation.executable) {
        deepforge::runtime::VariantPack row_pack{
            {821, packed_q.data()},
            {822, packed_k.data()},
            {823, packed_v.data()},
            {824, packed_configuration.seq_q.data()},
            {825, packed_configuration.seq_kv.data()},
            {826, row_o.data()},
            {827, row_stats.data()},
            {828, row_maximum.data()},
            {829, row_sum.data()},
            {831, q_offsets.data()},
            {832, k_offsets.data()},
            {833, row_v_offsets.data()},
            {834, row_o_offsets.data()},
            {835, row_offsets.data()}};
        status = row_compilation.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr, row_pack,
            nullptr);
        tests.good(status, "execute exact-allocation ragged row outputs");
        tests.check(
            close_vectors(row_o, packed_expected_o) &&
                close_vectors(row_stats, packed_stats) &&
                close_vectors(row_maximum, packed_maximum) &&
                close_vectors(row_sum, packed_sum),
            "packed O/Stats/Max/Sum_exp match dense attention reference");
    }

    std::vector<float> dense_d_o(packed_reference.o.size());
    for (std::size_t index = 0; index < dense_d_o.size(); ++index) {
        dense_d_o[index] =
            static_cast<float>(static_cast<int>((index * 5) % 17) - 8) /
            9.0F;
    }
    auto packed_gradients = attention_backward(
        packed_configuration, packed_reference, dense_d_o);
    auto packed_d_o = pack_bshd(dense_d_o, packed_configuration.o_dim,
                                packed_configuration.seq_q);
    auto expected_d_q = pack_bshd(
        packed_gradients.d_q, packed_configuration.q_dim,
        packed_configuration.seq_q);
    auto expected_d_k = pack_bshd(
        packed_gradients.d_k, packed_configuration.k_dim,
        packed_configuration.seq_kv);
    auto expected_d_v = pack_bshd(
        packed_gradients.d_v, packed_configuration.v_dim,
        packed_configuration.seq_kv);
    std::vector<float> packed_d_q(expected_d_q.size(), -99.0F);
    std::vector<float> packed_d_k(expected_d_k.size(), -99.0F);
    std::vector<float> packed_d_v(expected_d_v.size(), -99.0F);
    std::vector<std::int32_t> backward_o_offsets{0, 8, 12};
    deepforge::compiler::CompilationResult backward_compilation;
    status = compile_document(ragged_sdpa_backward_graph(),
                              backward_compilation);
    tests.good(status, "compile producer-stride ragged SDPA backward");
    if (status.is_good() && backward_compilation.executable) {
        deepforge::runtime::VariantPack backward_pack{
            {901, packed_q.data()},
            {902, packed_k.data()},
            {903, packed_v.data()},
            {904, packed_expected_o.data()},
            {905, packed_d_o.data()},
            {906, packed_stats.data()},
            {907, packed_configuration.seq_q.data()},
            {908, packed_configuration.seq_kv.data()},
            {909, packed_d_q.data()},
            {910, packed_d_k.data()},
            {911, packed_d_v.data()},
            {921, q_offsets.data()},
            {922, k_offsets.data()},
            {923, row_v_offsets.data()},
            {924, backward_o_offsets.data()},
            {925, row_offsets.data()}};
        status = backward_compilation.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr,
            backward_pack, nullptr);
        tests.good(status, "execute exact-allocation ragged SDPA backward");
        tests.check(close_vectors(packed_d_q, expected_d_q, 3.0e-4F) &&
                        close_vectors(packed_d_k, expected_d_k, 3.0e-4F) &&
                        close_vectors(packed_d_v, expected_d_v, 3.0e-4F),
                    "packed backward Q/K/V gradients match dense reference");

        std::vector<std::uint8_t> backward_artifact;
        status = deepforge::compiler::serialize_artifact(
            backward_compilation, backward_artifact);
        tests.good(status, "serialize ragged backward artifact");
        std::unique_ptr<deepforge::runtime::Executable> loaded_backward;
        if (status.is_good()) {
            status = deepforge::compiler::load_artifact_executable(
                backward_artifact, loaded_backward);
        }
        tests.good(status, "load ragged backward artifact");
        if (loaded_backward) {
            std::fill(packed_d_q.begin(), packed_d_q.end(), -99.0F);
            std::fill(packed_d_k.begin(), packed_d_k.end(), -99.0F);
            std::fill(packed_d_v.begin(), packed_d_v.end(), -99.0F);
            status = loaded_backward->execute(nullptr, backward_pack,
                                              nullptr);
            tests.good(status, "execute reloaded ragged SDPA backward");
            tests.check(close_vectors(packed_d_q, expected_d_q, 3.0e-4F) &&
                            close_vectors(packed_d_k, expected_d_k,
                                          3.0e-4F) &&
                            close_vectors(packed_d_v, expected_d_v,
                                          3.0e-4F),
                        "artifact preserves packed backward addressing");
        }
    }
}

void run_paged_attention_tests(TestRunner& tests) {
    AttentionCase configuration;
    configuration.q_dim = {2, 1, 1, 2};
    configuration.k_dim = {2, 1, 3, 2};
    configuration.v_dim = {2, 1, 3, 2};
    configuration.o_dim = {2, 1, 1, 2};
    configuration.q = {0.2F, -0.3F, -0.5F, 0.4F};
    configuration.k = {0.6F, -0.2F, -0.4F, 0.9F, 0.1F, 0.5F,
                       -0.3F, 0.8F, 0.7F, -0.6F, 0.2F, 0.4F};
    configuration.v = {0.3F, -0.7F, 0.8F, 0.2F, -0.6F, 0.4F,
                       0.5F, -0.1F, -0.2F, 0.9F, 0.6F, -0.8F};
    configuration.bias.clear();
    configuration.dropout_mask.clear();
    configuration.dropout_scale = 1.0F;
    configuration.dropout_scale_inv = 1.0F;
    configuration.padding = true;
    configuration.seq_q = {1, 1};
    configuration.seq_kv = {3, 3};
    configuration.left_bound.reset();
    configuration.right_bound.reset();
    configuration.attention_scale = 0.5F;
    auto reference = attention_forward(configuration);

    std::vector<std::int32_t> page_k{2, 0, 3, 1};
    std::vector<std::int32_t> page_v{1, 3, 0, 2};
    std::array<std::int64_t, 4> const container_dim{4, 1, 2, 2};
    auto make_container = [&](std::vector<float> const& logical,
                              std::vector<std::int32_t> const& table) {
        std::vector<float> container(16, -17.0F);
        for (std::int64_t batch = 0; batch < 2; ++batch) {
            for (std::int64_t sequence = 0; sequence < 3; ++sequence) {
                auto const page = table[static_cast<std::size_t>(
                    batch * 2 + sequence / 2)];
                for (std::int64_t embedding = 0; embedding < 2;
                     ++embedding) {
                    container[offset4(container_dim, page, 0, sequence % 2,
                                      embedding)] =
                        logical[offset4(configuration.k_dim, batch, 0,
                                        sequence, embedding)];
                }
            }
        }
        return container;
    };
    auto k_container = make_container(configuration.k, page_k);
    auto v_container = make_container(configuration.v, page_v);
    std::vector<float> output(reference.o.size(), -99.0F);

    deepforge::compiler::CompilationResult compilation;
    auto status = compile_document(paged_sdpa_graph(), compilation);
    tests.good(status, "compile independently paged K/V SDPA");
    if (status.is_bad() || !compilation.executable) return;
    deepforge::runtime::VariantPack pack{{701, configuration.q.data()},
                                         {702, k_container.data()},
                                         {703, v_container.data()},
                                         {704, page_k.data()},
                                         {705, page_v.data()},
                                         {706, configuration.seq_q.data()},
                                         {707, configuration.seq_kv.data()},
                                         {708, output.data()}};
    status = compilation.executable->execute_variant(
        deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr);
    tests.good(status, "execute independently paged K/V SDPA");
    tests.check(close_vectors(output, reference.o),
                "page-table block mapping matches contiguous SDPA reference");

    auto inferred_graph = paged_sdpa_graph();
    inferred_graph["nodes"][0]["max_seq_len_kv"] = nullptr;
    inferred_graph["tensors"]["709"] = tensor("Bias", 709, {1, 1, 1, 3});
    inferred_graph["nodes"][0]["inputs"]["Bias"] = 709;
    deepforge::compiler::CompilationResult inferred_compilation;
    status = compile_document(inferred_graph, inferred_compilation);
    tests.good(status, "infer both-paged logical sequence from Bias");
    if (status.is_good() && inferred_compilation.executable) {
        std::vector<float> zero_bias(3, 0.0F);
        deepforge::runtime::VariantPack inferred_pack = pack;
        inferred_pack[709] = zero_bias.data();
        std::fill(output.begin(), output.end(), -99.0F);
        status = inferred_compilation.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr, inferred_pack,
            nullptr);
        tests.good(status, "execute Bias-inferred both-paged SDPA");
        tests.check(close_vectors(output, reference.o),
                    "both-paged inference follows Frontend Bias precedence");
    }

    auto packed_graph = paged_sdpa_graph();
    packed_graph["tensors"]["704"]["ragged_offset_uid"] = 709;
    packed_graph["tensors"]["704"]["ragged_offset_name"] =
        "Page_table_K_offsets";
    packed_graph["tensors"]["705"]["ragged_offset_uid"] = 710;
    packed_graph["tensors"]["705"]["ragged_offset_name"] =
        "Page_table_V_offsets";
    packed_graph["tensors"]["709"] =
        tensor("Page_table_K_offsets", 709, {3, 1, 1, 1}, "INT64");
    packed_graph["tensors"]["710"] =
        tensor("Page_table_V_offsets", 710, {3, 1, 1, 1}, "INT32");
    deepforge::compiler::CompilationResult packed_compilation;
    status = compile_document(packed_graph, packed_compilation);
    tests.good(status, "compile packed independently paged K/V SDPA");
    if (status.is_good() && packed_compilation.executable) {
        AttentionCase packed_configuration = configuration;
        packed_configuration.seq_kv = {3, 1};
        auto packed_reference = attention_forward(packed_configuration);
        std::vector<std::int32_t> packed_page_k{2, 0, 1};
        std::vector<std::int32_t> packed_page_v{1, 3, 2};
        std::vector<std::int64_t> packed_page_k_offsets{0, 2, 3};
        std::vector<std::int32_t> packed_page_v_offsets{0, 2, 3};
        auto make_packed_container =
            [&](std::vector<float> const& logical,
                std::vector<std::int32_t> const& table,
                auto const& offsets) {
                std::vector<float> container(16, -17.0F);
                for (std::int64_t batch = 0; batch < 2; ++batch) {
                    auto const sequence_length =
                        packed_configuration.seq_kv[
                            static_cast<std::size_t>(batch)];
                    for (std::int64_t sequence = 0;
                         sequence < sequence_length; ++sequence) {
                        auto const table_index =
                            offsets[static_cast<std::size_t>(batch)] +
                            sequence / 2;
                        auto const page = table[static_cast<std::size_t>(
                            table_index)];
                        for (std::int64_t embedding = 0; embedding < 2;
                             ++embedding) {
                            container[offset4(container_dim, page, 0,
                                              sequence % 2, embedding)] =
                                logical[offset4(packed_configuration.k_dim,
                                                batch, 0, sequence,
                                                embedding)];
                        }
                    }
                }
                return container;
            };
        auto packed_k_container = make_packed_container(
            packed_configuration.k, packed_page_k, packed_page_k_offsets);
        auto packed_v_container = make_packed_container(
            packed_configuration.v, packed_page_v, packed_page_v_offsets);
        std::vector<float> packed_output(packed_reference.o.size(), -99.0F);
        auto const page_metadata_is_packed = [&](std::int64_t uid,
                                                 std::int64_t offset_uid) {
            auto const argument = std::find_if(
                packed_compilation.metadata.arguments.begin(),
                packed_compilation.metadata.arguments.end(),
                [&](auto const& candidate) { return candidate.uid == uid; });
            return argument != packed_compilation.metadata.arguments.end() &&
                   argument->storage_policy ==
                       deepforge::compiler::TensorStoragePolicy::
                           kRaggedBatchPrefix &&
                   argument->ragged_offset_uid == offset_uid &&
                   argument->ragged_sequence_uid == 707 &&
                   argument->ragged_sequence_divisor == 2;
        };
        tests.check(page_metadata_is_packed(704, 709) &&
                        page_metadata_is_packed(705, 710),
                    "packed page metadata records prefix UIDs and block divisor");
        deepforge::runtime::VariantPack packed_pack{
            {701, packed_configuration.q.data()},
            {702, packed_k_container.data()},
            {703, packed_v_container.data()},
            {704, packed_page_k.data()},
            {705, packed_page_v.data()},
            {706, packed_configuration.seq_q.data()},
            {707, packed_configuration.seq_kv.data()},
            {708, packed_output.data()},
            {709, packed_page_k_offsets.data()},
            {710, packed_page_v_offsets.data()}};
        status = packed_compilation.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr, packed_pack,
            nullptr);
        tests.good(status, "execute exact-allocation packed page tables");
        tests.check(close_vectors(packed_output, packed_reference.o),
                    "packed page tables match contiguous SDPA reference");

        std::vector<std::uint8_t> packed_artifact;
        status = deepforge::compiler::serialize_artifact(packed_compilation,
                                                         packed_artifact);
        tests.good(status, "serialize packed-page artifact v5");
        std::unique_ptr<deepforge::runtime::Executable> loaded_packed;
        if (status.is_good()) {
            status = deepforge::compiler::load_artifact_executable(
                packed_artifact, loaded_packed);
        }
        tests.good(status, "load packed-page artifact v5");
        if (loaded_packed) {
            std::fill(packed_output.begin(), packed_output.end(), -99.0F);
            status = loaded_packed->execute(nullptr, packed_pack, nullptr);
            tests.good(status, "execute reloaded packed page tables");
            tests.check(close_vectors(packed_output, packed_reference.o),
                        "artifact preserves packed page-table addressing");
        }

        packed_page_k_offsets = {0, 1, 3};
        status = packed_compilation.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr, packed_pack,
            nullptr);
        tests.check(
            status.code() ==
                deepforge::import::ErrorCode::kInvalidVariantPack,
            "runtime rejects packed page segments shorter than block demand");
    }

    page_k[0] = -1;
    page_k[1] = 99;
    page_v[0] = -1;
    page_v[1] = 99;
    std::fill(output.begin(), output.end(), -99.0F);
    status = compilation.executable->execute_variant(
        deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr);
    tests.good(status, "execute guarded invalid page IDs without an OOB access");
    tests.check(output[0] == 0.0F && output[1] == 0.0F,
                "invalid page IDs produce guarded zero K/V loads");
}

void run_block_mask_tests(TestRunner& tests) {
    AttentionCase configuration;
    configuration.q_dim = {1, 1, 129, 1};
    configuration.k_dim = {1, 1, 1025, 1};
    configuration.v_dim = {1, 1, 1025, 1};
    configuration.o_dim = {1, 1, 129, 1};
    configuration.q.resize(129);
    configuration.k.resize(1025);
    configuration.v.resize(1025);
    for (std::size_t index = 0; index < configuration.q.size(); ++index) {
        configuration.q[index] =
            static_cast<float>(static_cast<int>(index % 7) - 3) / 5.0F;
    }
    for (std::size_t index = 0; index < configuration.k.size(); ++index) {
        configuration.k[index] =
            static_cast<float>(static_cast<int>(index % 11) - 5) / 7.0F;
        configuration.v[index] =
            static_cast<float>(static_cast<int>(index % 17) - 8) / 6.0F;
    }
    configuration.block_mask = {0x00, 0x01, 0x01, 0x00};
    configuration.bias.clear();
    configuration.dropout_mask.clear();
    configuration.dropout_scale = 1.0F;
    configuration.dropout_scale_inv = 1.0F;
    configuration.padding = false;
    configuration.left_bound.reset();
    configuration.right_bound.reset();
    configuration.attention_scale = 0.75F;
    auto reference = attention_forward(configuration);

    deepforge::compiler::CompilationResult compilation;
    auto status = compile_document(block_mask_sdpa_graph(), compilation);
    tests.good(status, "compile UINT8 block-mask SDPA");
    if (status.is_bad() || !compilation.executable) return;
    std::vector<float> output(reference.o.size(), -99.0F);
    std::vector<float> stats(reference.stats.size(), -99.0F);
    std::vector<float> maximum(reference.maximum.size(), -99.0F);
    std::vector<float> sum_exp(reference.sum_exp.size(), -99.0F);
    std::vector<std::uint8_t> strided_block_mask{
        0x00, 0x01, 0xff, 0xff, 0x01, 0x00};
    deepforge::runtime::VariantPack pack{{1101, configuration.q.data()},
                                         {1102, configuration.k.data()},
                                         {1103, configuration.v.data()},
                                         {1104, strided_block_mask.data()},
                                         {1105, output.data()},
                                         {1106, stats.data()},
                                         {1107, maximum.data()},
                                         {1108, sum_exp.data()}};
    status = compilation.executable->execute_variant(
        deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr);
    tests.good(status, "execute UINT8 block-mask SDPA");
    tests.check(close_vectors(output, reference.o) &&
                    close_vectors(stats, reference.stats) &&
                    close_vectors(maximum, reference.maximum) &&
                    close_vectors(sum_exp, reference.sum_exp),
                "LSB-first block mask matches independent tiled reference");
    tests.check(output.front() == configuration.v[1024] &&
                    output[127] == configuration.v[1024] &&
                    output.back() != configuration.v[1024],
                "block-mask bytes select independent query and key tiles");
}

void run_sink_token_tests(TestRunner& tests) {
    AttentionCase configuration;
    configuration.q_dim = {2, 2, 2, 2};
    configuration.k_dim = {2, 1, 3, 2};
    configuration.v_dim = {2, 1, 3, 2};
    configuration.o_dim = {2, 2, 2, 2};
    configuration.q.resize(16);
    configuration.k.resize(12);
    configuration.v.resize(12);
    for (std::size_t index = 0; index < configuration.q.size(); ++index) {
        configuration.q[index] =
            static_cast<float>(static_cast<int>(index % 9) - 4) / 6.0F;
    }
    for (std::size_t index = 0; index < configuration.k.size(); ++index) {
        configuration.k[index] =
            static_cast<float>(static_cast<int>((index * 3) % 13) - 6) /
            8.0F;
        configuration.v[index] =
            static_cast<float>(static_cast<int>((index * 5) % 11) - 5) /
            7.0F;
    }
    configuration.bias.clear();
    configuration.dropout_mask.resize(24);
    for (std::size_t index = 0; index < configuration.dropout_mask.size();
         ++index) {
        configuration.dropout_mask[index] =
            (index % 4 == 1 || index % 7 == 0) ? 0.0F : 1.0F;
    }
    configuration.dropout_scale = 1.25F;
    configuration.dropout_scale_inv = 0.8F;
    configuration.sink_token = {0.35F, -0.4F};
    configuration.padding = true;
    configuration.seq_q = {2, 1};
    configuration.seq_kv = {3, 2};
    configuration.left_bound.reset();
    configuration.right_bound.reset();
    configuration.attention_scale = 0.6F;
    auto reference = attention_forward(configuration);

    deepforge::compiler::CompilationResult forward_compilation;
    auto status = compile_document(sink_sdpa_graph(), forward_compilation);
    tests.good(status, "compile sink-token SDPA forward");
    if (status.is_bad() || !forward_compilation.executable) return;
    std::vector<float> output(reference.o.size(), -99.0F);
    std::vector<float> stats(reference.stats.size(), -99.0F);
    std::vector<float> maximum(reference.maximum.size(), -99.0F);
    std::vector<float> sum_exp(reference.sum_exp.size(), -99.0F);
    std::array<float, 1> dropout_scale{configuration.dropout_scale};
    std::array<float, 1> dropout_scale_inv{
        configuration.dropout_scale_inv};
    deepforge::runtime::VariantPack forward_pack{
        {1201, configuration.q.data()},
        {1202, configuration.k.data()},
        {1203, configuration.v.data()},
        {1204, configuration.sink_token.data()},
        {1205, configuration.seq_q.data()},
        {1206, configuration.seq_kv.data()},
        {1211, configuration.dropout_mask.data()},
        {1212, dropout_scale.data()},
        {1207, output.data()},
        {1208, stats.data()},
        {1209, maximum.data()},
        {1210, sum_exp.data()}};
    status = forward_compilation.executable->execute_variant(
        deepforge::runtime::CpuVariant::kScalar, nullptr, forward_pack,
        nullptr);
    tests.good(status, "execute sink-token SDPA forward");
    tests.check(close_vectors(output, reference.o) &&
                    close_vectors(stats, reference.stats) &&
                    close_vectors(maximum, reference.maximum) &&
                    close_vectors(sum_exp, reference.sum_exp),
                "sink token participates in normalization and row metadata");

    std::vector<float> d_o(reference.o.size());
    for (std::size_t index = 0; index < d_o.size(); ++index) {
        d_o[index] =
            static_cast<float>(static_cast<int>((index * 7) % 19) - 9) /
            10.0F;
    }
    auto gradients = attention_backward(configuration, reference, d_o);
    deepforge::compiler::CompilationResult backward_compilation;
    status = compile_document(sink_sdpa_backward_graph(),
                              backward_compilation);
    tests.good(status, "compile sink-token SDPA backward");
    if (status.is_bad() || !backward_compilation.executable) return;
    std::vector<float> d_q(gradients.d_q.size(), -99.0F);
    std::vector<float> d_k(gradients.d_k.size(), -99.0F);
    std::vector<float> d_v(gradients.d_v.size(), -99.0F);
    std::vector<float> d_sink(gradients.d_sink.size(), -99.0F);
    deepforge::runtime::VariantPack backward_pack{
        {1221, configuration.q.data()},
        {1222, configuration.k.data()},
        {1223, configuration.v.data()},
        {1224, reference.o.data()},
        {1225, d_o.data()},
        {1226, reference.stats.data()},
        {1227, configuration.sink_token.data()},
        {1228, configuration.seq_q.data()},
        {1229, configuration.seq_kv.data()},
        {1234, configuration.dropout_mask.data()},
        {1235, dropout_scale.data()},
        {1236, dropout_scale_inv.data()},
        {1230, d_q.data()},
        {1231, d_k.data()},
        {1232, d_v.data()},
        {1233, d_sink.data()}};
    status = backward_compilation.executable->execute_variant(
        deepforge::runtime::CpuVariant::kScalar, nullptr, backward_pack,
        nullptr);
    tests.good(status, "execute sink-token SDPA backward");
    tests.check(close_vectors(d_q, gradients.d_q, 3.0e-4F) &&
                    close_vectors(d_k, gradients.d_k, 3.0e-4F) &&
                    close_vectors(d_v, gradients.d_v, 3.0e-4F) &&
                    close_vectors(d_sink, gradients.d_sink, 3.0e-4F),
                "sink backward gradients match independent reference");
    tests.check(close_vectors(d_sink,
                              finite_difference_sink(configuration, d_o),
                              2.0e-3F),
                "dSink matches finite differences without attention scaling");
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

    auto invalid_block_shape = block_mask_sdpa_graph();
    invalid_block_shape["tensors"]["1104"]["dim"] = {1, 1, 2, 1};
    status = compile_document(invalid_block_shape, rejected);
    tests.check(status.code() == deepforge::import::ErrorCode::kInvalidShape,
                "SDPA rejects an incorrectly compressed block mask");

    auto missing_sink = sink_sdpa_backward_graph();
    missing_sink["nodes"][0]["inputs"].erase("SINK_TOKEN");
    status = compile_document(missing_sink, rejected);
    tests.check(status.code() == deepforge::import::ErrorCode::kInvalidValue,
                "DSINK_TOKEN requires the corresponding sink input");

    auto invalid_sink_shape = sink_sdpa_graph();
    invalid_sink_shape["tensors"]["1204"]["dim"] = {1, 1, 1, 1};
    status = compile_document(invalid_sink_shape, rejected);
    tests.check(status.code() == deepforge::import::ErrorCode::kInvalidShape,
                "sink token must provide one value per query head");

    auto invalid_ragged_shape = ragged_sdpa_graph();
    invalid_ragged_shape["tensors"]["605"]["dim"] = {2, 1, 1, 1};
    status = compile_document(invalid_ragged_shape, rejected);
    tests.check(status.code() == deepforge::import::ErrorCode::kInvalidShape,
                "ragged SDPA requires B+1 prefix offsets");

    auto invalid_ragged_identity = ragged_sdpa_graph();
    invalid_ragged_identity["tensors"]["601"]["ragged_offset_name"] =
        "different";
    status = compile_document(invalid_ragged_identity, rejected);
    tests.check(status.code() == deepforge::import::ErrorCode::kInvalidValue,
                "ragged offset UID and serialized name must identify one tensor");

    auto invalid_ragged_batch = ragged_sdpa_graph();
    invalid_ragged_batch["tensors"]["601"]["dim"][0] = 0;
    status = compile_document(invalid_ragged_batch, rejected);
    tests.check(status.code() == deepforge::import::ErrorCode::kInvalidShape,
                "ragged SDPA rejects a non-positive batch extent");

    auto missing_ragged_sequence = ragged_sdpa_graph();
    missing_ragged_sequence["nodes"][0]["inputs"].erase("SEQ_LEN_Q");
    status = compile_document(missing_ragged_sequence, rejected);
    tests.check(status.code() == deepforge::import::ErrorCode::kInvalidValue,
                "ragged SDPA requires explicit sequence metadata");

    auto invalid_paged_capacity = paged_sdpa_graph();
    invalid_paged_capacity["nodes"][0]["max_seq_len_kv"] = 5;
    status = compile_document(invalid_paged_capacity, rejected);
    tests.check(status.code() == deepforge::import::ErrorCode::kInvalidShape,
                "paged SDPA rejects logical sequence beyond table capacity");

    auto unmasked_paged = paged_sdpa_graph();
    unmasked_paged["nodes"][0]["padding_mask"] = false;
    unmasked_paged["nodes"][0]["inputs"].erase("SEQ_LEN_Q");
    unmasked_paged["nodes"][0]["inputs"].erase("SEQ_LEN_KV");
    status = compile_document(unmasked_paged, rejected);
    tests.check(status.code() ==
                    deepforge::import::ErrorCode::kUnsupportedOperation,
                "paged SDPA requires padding and sequence metadata");

    auto dense_max_total = feature_sdpa_backward_graph();
    dense_max_total["nodes"][0]["max_total_seq_len_q"] = 2;
    status = compile_document(dense_max_total, rejected);
    tests.check(status.code() ==
                    deepforge::import::ErrorCode::kUnsupportedOperation,
                "max-total sequence hints require ragged backward storage");

    auto invalid_max_total = ragged_sdpa_backward_graph();
    invalid_max_total["nodes"][0]["max_total_seq_len_kv"] = 7;
    status = compile_document(invalid_max_total, rejected);
    tests.check(status.code() == deepforge::import::ErrorCode::kInvalidValue,
                "ragged backward rejects max-total values beyond B*Skv");

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
    run_dynamic_attention_tests(tests);
    run_ragged_attention_tests(tests);
    run_paged_attention_tests(tests);
    run_block_mask_tests(tests);
    run_sink_token_tests(tests);
    run_internal_dropout_tests(tests);
    run_alibi_tests(tests);
    run_bottom_right_tests(tests);
    run_rejection_tests(tests);
    return tests.finish();
}
