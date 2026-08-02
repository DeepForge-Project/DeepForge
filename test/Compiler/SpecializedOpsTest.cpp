#include "DeepForge/Compiler/Codegen.h"
#include "DeepForge/Import/SerializedGraphImporter.h"

#include <cudnn_frontend/thirdparty/nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <limits>
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
            std::cout << "deepforge-specialized: " << checks_
                      << " checks passed\n";
            return 0;
        }
        std::cerr << "deepforge-specialized: " << failures_ << " of "
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
            std::string data_type,
            bool is_virtual = false,
            std::vector<std::int64_t> strides = {}) {
    if (strides.empty()) strides = contiguous_strides(dimensions);
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

Json f8_reordered_scale(std::string name,
                        std::int64_t uid,
                        bool transposed = false) {
    auto result = transposed
                      ? tensor(std::move(name), uid, {1, 1, 4, 128},
                               "FP8_E8M0", false, {512, 512, 1, 4})
                      : tensor(std::move(name), uid, {1, 1, 128, 4},
                               "FP8_E8M0", false, {512, 512, 4, 1});
    result["reordering_type"] = "F8_128x4";
    return result;
}

std::size_t f8_128x4_offset(std::size_t m,
                            std::size_t k,
                            std::size_t leading,
                            std::size_t m_extent,
                            std::size_t k_extent) {
    auto const m_blocks = m_extent / 128;
    auto const k_blocks = k_extent / 4;
    return (((leading * m_blocks + m / 128) * k_blocks + k / 4) * 512) +
           (m % 32) * 16 + ((m / 32) % 4) * 4 + k % 4;
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

deepforge::import::Status compile_document(
    Json const& document,
    deepforge::compiler::CompilationResult& compilation) {
    auto text = document.dump();
    std::vector<std::uint8_t> bytes(text.begin(), text.end());
    deepforge::import::SerializedGraph graph;
    deepforge::import::SerializedGraphImporter importer;
    auto status = importer.parse(std::span<std::uint8_t const>(bytes),
                                 deepforge::import::InputFormat::kJson, graph);
    if (status.is_bad()) return status;
    return deepforge::compiler::compile_graph(
        graph, deepforge::compiler::CompileOptions{}, compilation);
}

bool close_vectors(std::vector<float> const& actual,
                   std::vector<float> const& expected,
                   float tolerance = 1.0e-5F) {
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

Json block_dequantize_graph() {
    Json tensors = Json::object();
    tensors["1"] = tensor("X", 1, {1, 4}, "FP8_E4M3");
    tensors["2"] = tensor("scale", 2, {1, 2}, "FP8_E8M0");
    tensors["3"] = tensor("Y", 3, {1, 4}, "FLOAT");
    Json node{{"tag", "BLOCK_SCALE_DEQUANTIZE"},
              {"name", "dequantize"},
              {"inputs", Json::object({{"X", 1}, {"scale", 2}})},
              {"outputs", Json::object({{"Y", 3}})},
              {"compute_data_type", "FLOAT"},
              {"block_size", Json::array({2})},
              {"is_negative_scale", true}};
    return graph_document(5001, "block-dequantize", Json::array({node}),
                          std::move(tensors));
}

Json block_quantize_graph() {
    Json tensors = Json::object();
    tensors["11"] = tensor("X", 11, {1, 4}, "FLOAT");
    tensors["12"] = tensor("Y", 12, {1, 4}, "FP8_E4M3");
    tensors["13"] = tensor("scale", 13, {1, 2}, "FP8_E8M0");
    Json node{{"tag", "BLOCK_SCALE_QUANTIZE"},
              {"name", "quantize"},
              {"inputs", Json::object({{"X", 11}})},
              {"outputs", Json::object({{"Y", 12}, {"scale", 13}})},
              {"compute_data_type", "FLOAT"},
              {"block_size", 2},
              {"axis", 1}};
    return graph_document(5002, "block-quantize", Json::array({node}),
                          std::move(tensors));
}

Json reordered_block_dequantize_graph() {
    Json tensors = Json::object();
    tensors["51"] = tensor("X", 51, {128, 32}, "FP8_E4M3");
    tensors["52"] = tensor("scale", 52, {128, 4}, "FP8_E8M0", false,
                            {4, 1});
    tensors["52"]["reordering_type"] = "F8_128x4";
    tensors["53"] = tensor("Y", 53, {128, 32}, "FLOAT");
    Json node{{"tag", "BLOCK_SCALE_DEQUANTIZE"},
              {"name", "reordered_dequantize"},
              {"inputs", Json::object({{"X", 51}, {"scale", 52}})},
              {"outputs", Json::object({{"Y", 53}})},
              {"compute_data_type", "FLOAT"},
              {"block_size", Json::array({32})},
              {"is_negative_scale", false}};
    return graph_document(5014, "reordered-block-dequantize",
                          Json::array({node}), std::move(tensors));
}

Json reordered_block_quantize_graph() {
    Json tensors = Json::object();
    tensors["61"] = tensor("X", 61, {128, 32}, "FLOAT");
    tensors["62"] = tensor("Y", 62, {128, 32}, "FP8_E4M3");
    tensors["63"] = tensor("scale", 63, {128, 4}, "FP8_E8M0", false,
                            {4, 1});
    tensors["63"]["reordering_type"] = "F8_128x4";
    Json node{{"tag", "BLOCK_SCALE_QUANTIZE"},
              {"name", "reordered_quantize"},
              {"inputs", Json::object({{"X", 61}})},
              {"outputs", Json::object({{"Y", 62}, {"scale", 63}})},
              {"compute_data_type", "FLOAT"},
              {"block_size", 32},
              {"axis", 1}};
    return graph_document(5015, "reordered-block-quantize",
                          Json::array({node}), std::move(tensors));
}

Json transposed_reordered_block_dequantize_graph() {
    Json tensors = Json::object();
    tensors["71"] = tensor("X", 71, {32, 128}, "FP8_E4M3");
    tensors["72"] = tensor("scale", 72, {4, 128}, "FP8_E8M0", false,
                            {1, 4});
    tensors["72"]["reordering_type"] = "F8_128x4";
    tensors["73"] = tensor("Y", 73, {32, 128}, "FLOAT");
    Json node{{"tag", "BLOCK_SCALE_DEQUANTIZE"},
              {"name", "transposed_reordered_dequantize"},
              {"inputs", Json::object({{"X", 71}, {"scale", 72}})},
              {"outputs", Json::object({{"Y", 73}})},
              {"compute_data_type", "FLOAT"},
              {"block_size", Json::array({32, 1})},
              {"is_negative_scale", false}};
    return graph_document(5017, "transposed-reordered-block-dequantize",
                          Json::array({node}), std::move(tensors));
}

Json fp4_roundtrip_graph() {
    Json tensors = Json::object();
    tensors["14"] = tensor("X", 14, {1, 4}, "FLOAT");
    tensors["15"] = tensor("Packed", 15, {1, 4}, "FP4_E2M1", true);
    tensors["16"] = tensor("BlockScale", 16, {1, 2}, "FP8_E4M3", true);
    tensors["17"] = tensor("Y", 17, {1, 4}, "FLOAT");
    Json quantize{{"tag", "BLOCK_SCALE_QUANTIZE"},
                  {"name", "quantize_fp4"},
                  {"inputs", Json::object({{"X", 14}})},
                  {"outputs",
                   Json::object({{"Y", 15}, {"scale", 16}})},
                  {"compute_data_type", "FLOAT"},
                  {"block_size", 2},
                  {"axis", 1}};
    Json dequantize{{"tag", "BLOCK_SCALE_DEQUANTIZE"},
                    {"name", "dequantize_fp4"},
                    {"inputs",
                     Json::object({{"X", 15}, {"scale", 16}})},
                    {"outputs", Json::object({{"Y", 17}})},
                    {"compute_data_type", "FLOAT"},
                    {"block_size", Json::array({2})},
                    {"is_negative_scale", false}};
    return graph_document(5009, "fp4-roundtrip",
                          Json::array({quantize, dequantize}),
                          std::move(tensors));
}

Json bf16_fp8_f16_roundtrip_graph() {
    Json tensors = Json::object();
    tensors["18"] = tensor("X", 18, {1, 4}, "BFLOAT16");
    tensors["19"] = tensor("Packed", 19, {1, 4}, "FP8_E4M3", true);
    tensors["20"] = tensor("BlockScale", 20, {1, 2}, "FP8_E8M0", true);
    tensors["21"] = tensor("Y", 21, {1, 4}, "HALF");
    Json quantize{{"tag", "BLOCK_SCALE_QUANTIZE"},
                  {"name", "quantize_bf16"},
                  {"inputs", Json::object({{"X", 18}})},
                  {"outputs",
                   Json::object({{"Y", 19}, {"scale", 20}})},
                  {"compute_data_type", "FLOAT"},
                  {"block_size", 2},
                  {"axis", 1}};
    Json dequantize{{"tag", "BLOCK_SCALE_DEQUANTIZE"},
                    {"name", "dequantize_f16"},
                    {"inputs",
                     Json::object({{"X", 19}, {"scale", 20}})},
                    {"outputs", Json::object({{"Y", 21}})},
                    {"compute_data_type", "FLOAT"},
                    {"block_size", Json::array({2})},
                    {"is_negative_scale", false}};
    return graph_document(5010, "bf16-fp8-f16-roundtrip",
                          Json::array({quantize, dequantize}),
                          std::move(tensors));
}

Json matmul_fp8_graph() {
    Json tensors = Json::object();
    tensors["21"] = tensor("A", 21, {1, 2, 2}, "FP8_E4M3");
    tensors["22"] = tensor("B", 22, {1, 2, 2}, "FP8_E4M3");
    tensors["23"] = tensor("Descale_A", 23, {1}, "FLOAT");
    tensors["24"] = tensor("Descale_B", 24, {1}, "FLOAT");
    tensors["25"] = tensor("Scale_C", 25, {1}, "FLOAT");
    tensors["26"] = tensor("C", 26, {1, 2, 2}, "FLOAT");
    tensors["27"] = tensor("Amax_C", 27, {1}, "FLOAT");
    Json node{{"tag", "MATMUL_FP8"},
              {"name", "matmul_fp8"},
              {"inputs", Json::object({{"A", 21},
                                         {"B", 22},
                                         {"Descale_A", 23},
                                         {"Descale_B", 24},
                                         {"Scale_C", 25}})},
              {"outputs", Json::object({{"C", 26}, {"Amax_C", 27}})},
              {"compute_data_type", "FLOAT"}};
    return graph_document(5003, "matmul-fp8", Json::array({node}),
                          std::move(tensors));
}

Json matmul_fp8_pass_by_value_graph() {
    auto graph = matmul_fp8_graph();
    for (auto uid : {"23", "24", "25"}) {
        graph["tensors"][uid]["is_pass_by_value"] = true;
    }
    return graph;
}

Json matmul_fp8_embedded_graph() {
    auto graph = matmul_fp8_pass_by_value_graph();
    graph["graph_uid"] = 5019;
    graph["context"]["name"] = "matmul-fp8-embedded";
    auto const descale_a = Json{{"index", 3}, {"value", "3F000000"}};
    auto const descale_b = Json{{"index", 3}, {"value", "40000000"}};
    auto const scale_c = Json{{"index", 3}, {"value", "3F800000"}};
    graph["tensors"]["23"]["pass_by_value"] = descale_a;
    graph["tensors"]["24"]["pass_by_value"] = descale_b;
    graph["tensors"]["25"]["pass_by_value"] = scale_c;
    graph["pass_by_values"] = Json::object(
        {{"23", descale_a}, {"24", descale_b}, {"25", scale_c}});
    return graph;
}

Json matmul_fp8_override_graph() {
    auto graph = matmul_fp8_graph();
    graph["graph_uid"] = 5018;
    graph["context"]["name"] = "matmul-fp8-overrides";
    graph["tensors"]["28"] =
        tensor("M_override", 28, {1, 1, 1}, "INT32");
    graph["tensors"]["29"] =
        tensor("N_override", 29, {1, 1, 1}, "INT32");
    graph["tensors"]["30"] =
        tensor("K_override", 30, {1, 1, 1}, "INT32");
    graph["nodes"][0]["inputs"]["M_override"] = 28;
    graph["nodes"][0]["inputs"]["N_override"] = 29;
    graph["nodes"][0]["inputs"]["K_override"] = 30;
    return graph;
}

Json moe_graph() {
    Json tensors = Json::object();
    tensors["31"] = tensor("Token", 31, {1, 4, 2}, "FLOAT");
    tensors["32"] = tensor("Weight", 32, {2, 2, 2}, "FLOAT");
    tensors["33"] = tensor("FirstTokenOffset", 33, {2, 1, 1}, "INT32");
    tensors["34"] = tensor("Output", 34, {1, 4, 2}, "FLOAT");
    tensors["35"] = tensor("DOutput", 35, {1, 4, 2}, "FLOAT");
    tensors["36"] = tensor("DWeight", 36, {2, 2, 2}, "FLOAT");
    Json forward{{"tag", "MOE_GROUPED_MATMUL"},
                 {"name", "moe"},
                 {"inputs", Json::object({{"Token", 31},
                                            {"Weight", 32},
                                            {"FirstTokenOffset", 33}})},
                 {"outputs", Json::object({{"Output", 34}})},
                 {"mode", "NONE"},
                 {"top_k", 1}};
    Json backward{{"tag", "MOE_GROUPED_MATMUL_BWD"},
                  {"name", "moe_bwd"},
                  {"inputs", Json::object({{"DOutput", 35},
                                             {"Token", 31},
                                             {"FirstTokenOffset", 33}})},
                  {"outputs", Json::object({{"DWeight", 36}})}};
    return graph_document(5004, "moe", Json::array({forward, backward}),
                          std::move(tensors));
}

Json fp8_forward_attributes(bool mxfp8) {
    return Json{{"generate_stats", true},
                {"alibi_mask", false},
                {"padding_mask", false},
                {"dropout_probability", nullptr},
                {"attn_scale_value", 1.0},
                {"max_seq_len_kv", nullptr},
                {"mma_core_mode", "FP8_E4M3"},
                {"left_bound", nullptr},
                {"right_bound", nullptr},
                {"diagonal_alignment", "TOP_LEFT"},
                {"implementation", "AUTO"},
                {"is_mxfp8", mxfp8},
                {"unfuse_fma", false},
                {"rescale_threshold", 0.0}};
}

Json fp8_backward_attributes(bool mxfp8) {
    return Json{{"compute_data_type", "FLOAT"},
                {"padding_mask", false},
                {"dropout_probability", nullptr},
                {"left_bound", nullptr},
                {"right_bound", nullptr},
                {"diagonal_alignment", "TOP_LEFT"},
                {"attn_scale_value", 1.0},
                {"is_deterministic_algorithm", true},
                {"is_mxfp8", mxfp8},
                {"rescale_threshold", 0.0}};
}

Json sdpa_fp8_forward_graph(bool mxfp8) {
    auto const base = mxfp8 ? 200 : 100;
    Json tensors = Json::object();
    tensors[std::to_string(base + 1)] =
        tensor("Q", base + 1, {1, 1, 2, 2}, "FP8_E4M3");
    tensors[std::to_string(base + 2)] =
        tensor("K", base + 2, {1, 1, 2, 2}, "FP8_E4M3");
    tensors[std::to_string(base + 3)] =
        tensor("V", base + 3, {1, 1, 2, 2}, "FP8_E4M3");
    Json inputs = Json::object(
        {{"Q", base + 1}, {"K", base + 2}, {"V", base + 3}});
    if (mxfp8) {
        tensors[std::to_string(base + 4)] =
            f8_reordered_scale("Descale_Q", base + 4);
        tensors[std::to_string(base + 5)] =
            f8_reordered_scale("Descale_K", base + 5);
        tensors[std::to_string(base + 6)] =
            f8_reordered_scale("Descale_V", base + 6, true);
        inputs.update(Json{{"Descale_Q", base + 4},
                           {"Descale_K", base + 5},
                           {"Descale_V", base + 6}});
        tensors[std::to_string(base + 10)] =
            tensor("O", base + 10, {1, 1, 2, 2}, "FLOAT");
    } else {
        for (int index = 4; index <= 9; ++index) {
            auto const names = std::vector<std::string>{
                "Descale_Q", "Descale_K", "Descale_V", "Descale_S",
                "Scale_S", "Scale_O"};
            tensors[std::to_string(base + index)] =
                tensor(names[static_cast<std::size_t>(index - 4)],
                       base + index, {1}, "FLOAT");
            inputs[names[static_cast<std::size_t>(index - 4)]] = base + index;
        }
        tensors[std::to_string(base + 10)] =
            tensor("O", base + 10, {1, 1, 2, 2}, "FP8_E4M3");
    }
    tensors[std::to_string(base + 11)] =
        tensor("Stats", base + 11, {1, 1, 2, 1}, "FLOAT");
    tensors[std::to_string(base + 12)] =
        tensor("Amax_O", base + 12, {1}, "FLOAT");
    Json outputs{{"O", base + 10},
                 {"Stats", base + 11},
                 {"Amax_O", base + 12}};
    if (!mxfp8) {
        tensors[std::to_string(base + 13)] =
            tensor("Amax_S", base + 13, {1}, "FLOAT");
        outputs["Amax_S"] = base + 13;
    }
    Json node = fp8_forward_attributes(mxfp8);
    node.update(Json{{"tag", mxfp8 ? "SDPA_MXFP8_FWD" : "SDPA_FP8_FWD"},
                     {"name", "fp8_attention"},
                     {"inputs", std::move(inputs)},
                     {"outputs", std::move(outputs)}});
    return graph_document(mxfp8 ? 5006 : 5005,
                          mxfp8 ? "mxfp8-fwd" : "fp8-fwd",
                          Json::array({node}), std::move(tensors));
}

Json sdpa_fp8_backward_graph(bool mxfp8) {
    auto const base = mxfp8 ? 400 : 300;
    Json tensors = Json::object();
    for (auto const& [offset, name] :
         std::vector<std::pair<int, std::string>>{{1, "Q"},
                                                   {2, "K"},
                                                   {3, "V"},
                                                   {5, "dO"}}) {
        tensors[std::to_string(base + offset)] =
            tensor(name, base + offset, {1, 1, 2, 2}, "FP8_E4M3");
    }
    tensors[std::to_string(base + 4)] = tensor(
        "O", base + 4, {1, 1, 2, 2}, mxfp8 ? "FLOAT" : "FP8_E4M3");
    tensors[std::to_string(base + 6)] =
        tensor("Stats", base + 6, {1, 1, 2, 1}, "FLOAT");
    Json inputs{{"Q", base + 1},
                {"K", base + 2},
                {"V", base + 3},
                {"O", base + 4},
                {"dO", base + 5},
                {"Stats", base + 6}};
    int next = 7;
    if (mxfp8) {
        for (auto const& name : {"Q_T", "K_T", "dO_T"}) {
            tensors[std::to_string(base + next)] =
                tensor(name, base + next, {1, 1, 2, 2}, "FP8_E4M3");
            inputs[name] = base + next++;
        }
        tensors[std::to_string(base + next)] =
            tensor("dO_f16", base + next, {1, 1, 2, 2}, "FLOAT");
        inputs["dO_f16"] = base + next++;
        for (auto const& name : {"Descale_Q", "Descale_K", "Descale_V",
                                 "Descale_dO"}) {
            tensors[std::to_string(base + next)] =
                f8_reordered_scale(name, base + next);
            inputs[name] = base + next++;
        }
        for (auto const& name : {"Descale_Q_T", "Descale_K_T",
                                 "Descale_dO_T"}) {
            tensors[std::to_string(base + next)] =
                f8_reordered_scale(name, base + next, true);
            inputs[name] = base + next++;
        }
    } else {
        for (auto const& name : {"Descale_Q", "Descale_K", "Descale_V",
                                 "Descale_O", "Descale_dO", "Descale_S",
                                 "Descale_dP", "Scale_S", "Scale_dP",
                                 "Scale_dQ", "Scale_dK", "Scale_dV"}) {
            tensors[std::to_string(base + next)] =
                tensor(name, base + next, {1}, "FLOAT");
            inputs[name] = base + next++;
        }
    }
    auto const output_type = mxfp8 ? "FLOAT" : "FP8_E4M3";
    Json outputs = Json::object();
    for (auto const& name : {"dQ", "dK", "dV"}) {
        tensors[std::to_string(base + next)] =
            tensor(name, base + next, {1, 1, 2, 2}, output_type);
        outputs[name] = base + next++;
    }
    for (auto const& name : {"Amax_dQ", "Amax_dK", "Amax_dV"}) {
        tensors[std::to_string(base + next)] =
            tensor(name, base + next, {1}, "FLOAT");
        outputs[name] = base + next++;
    }
    if (!mxfp8) {
        tensors[std::to_string(base + next)] =
            tensor("Amax_dP", base + next, {1}, "FLOAT");
        outputs["Amax_dP"] = base + next++;
    }
    Json node = fp8_backward_attributes(mxfp8);
    node.update(Json{{"tag", mxfp8 ? "SDPA_MXFP8_BWD" : "SDPA_FP8_BWD"},
                     {"name", "fp8_attention_bwd"},
                     {"inputs", std::move(inputs)},
                     {"outputs", std::move(outputs)}});
    return graph_document(mxfp8 ? 5008 : 5007,
                          mxfp8 ? "mxfp8-bwd" : "fp8-bwd",
                          Json::array({node}), std::move(tensors));
}

void run_block_tests(TestRunner& tests) {
    deepforge::compiler::CompilationResult dequantize;
    auto status = compile_document(block_dequantize_graph(), dequantize);
    tests.good(status, "compile block-scale dequantize");
    if (status.is_good() && dequantize.executable) {
        std::vector<std::uint8_t> x{0x38, 0x40, 0xb8, 0x30};
        std::vector<std::uint8_t> scale{127, 128};
        std::vector<float> y(4, -99.0F);
        deepforge::runtime::VariantPack pack{
            {1, x.data()}, {2, scale.data()}, {3, y.data()}};
        status = dequantize.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr);
        tests.good(status, "execute block-scale dequantize");
        tests.check(close_vectors(y, {1.0F, 2.0F, -2.0F, 1.0F}),
                    "FP8 values multiply by per-block E8M0 scales");
    }

    auto e5_document = block_dequantize_graph();
    e5_document["graph_uid"] = 5011;
    e5_document["tensors"]["1"]["data_type"] = "FP8_E5M2";
    deepforge::compiler::CompilationResult e5_dequantize;
    status = compile_document(e5_document, e5_dequantize);
    tests.good(status, "compile FP8 E5M2 block-scale dequantize");
    if (status.is_good() && e5_dequantize.executable) {
        std::vector<std::uint8_t> x{0x3c, 0x40, 0xc0, 0x38};
        std::vector<std::uint8_t> scale{127, 128};
        std::vector<float> y(4, -99.0F);
        deepforge::runtime::VariantPack pack{
            {1, x.data()}, {2, scale.data()}, {3, y.data()}};
        status = e5_dequantize.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr);
        tests.good(status, "execute FP8 E5M2 block-scale dequantize");
        tests.check(close_vectors(y, {1.0F, 2.0F, -4.0F, 1.0F}),
                    "FP8 E5M2 bytes decode with E8M0 block scales");
    }

    auto int4_document = block_dequantize_graph();
    int4_document["graph_uid"] = 5012;
    int4_document["tensors"]["1"]["data_type"] = "INT4";
    int4_document["tensors"]["2"]["data_type"] = "HALF";
    int4_document["tensors"]["3"]["data_type"] = "BFLOAT16";
    deepforge::compiler::CompilationResult int4_dequantize;
    status = compile_document(int4_document, int4_dequantize);
    tests.good(status, "compile INT4/F16/BF16 block-scale dequantize");
    if (status.is_good() && int4_dequantize.executable) {
        std::vector<std::uint8_t> x{0xf8, 0x70};
        std::vector<std::uint16_t> scale{0x3800, 0x4000};
        std::vector<std::uint16_t> y(4, 0xffff);
        deepforge::runtime::VariantPack pack{
            {1, x.data()}, {2, scale.data()}, {3, y.data()}};
        status = int4_dequantize.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr);
        tests.good(status, "execute INT4/F16/BF16 block-scale dequantize");
        tests.check(y == std::vector<std::uint16_t>(
                             {0xc080, 0xbf00, 0x0000, 0x4160}),
                    "signed INT4 nibbles, F16 load, and BF16 store match bits");
    }

    deepforge::compiler::CompilationResult quantize;
    status = compile_document(block_quantize_graph(), quantize);
    tests.good(status, "compile block-scale quantize");
    if (status.is_good() && quantize.executable) {
        std::vector<float> x{1.0F, 2.0F, -4.0F, 8.0F};
        std::vector<std::uint8_t> y(4, 0);
        std::vector<std::uint8_t> scale(2, 0);
        deepforge::runtime::VariantPack pack{
            {11, x.data()}, {12, y.data()}, {13, scale.data()}};
        status = quantize.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr);
        tests.good(status, "execute block-scale quantize");
        tests.check(scale == std::vector<std::uint8_t>({120, 122}),
                    "E8M0 scale is the ceiling power-of-two block ratio");
        tests.check(y == std::vector<std::uint8_t>({0x70, 0x78, 0xf0, 0x78}),
                    "quantized FP8 bytes use the emitted block scales");

        x = {std::numeric_limits<float>::infinity(), 0.0F,
             std::numeric_limits<float>::quiet_NaN(), 0.0F};
        std::fill(y.begin(), y.end(), 0);
        std::fill(scale.begin(), scale.end(), 0);
        status = quantize.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr);
        tests.good(status, "execute exceptional block-scale quantize");
        if (scale != std::vector<std::uint8_t>({0xfe, 0xff})) {
            std::cerr << "exceptional E8M0 codes: "
                      << static_cast<unsigned>(scale[0]) << ", "
                      << static_cast<unsigned>(scale[1]) << '\n';
        }
        tests.check(scale == std::vector<std::uint8_t>({0xfe, 0xff}),
                    "E8M0 saturates positive infinity and canonicalizes NaN");
    }

    auto e5_quantize_document = block_quantize_graph();
    e5_quantize_document["graph_uid"] = 5013;
    e5_quantize_document["tensors"]["12"]["data_type"] = "FP8_E5M2";
    deepforge::compiler::CompilationResult e5_quantize;
    status = compile_document(e5_quantize_document, e5_quantize);
    tests.good(status, "compile FP8 E5M2 block-scale quantize");
    if (status.is_good() && e5_quantize.executable) {
        std::vector<float> x{1.0F, 2.0F, -4.0F, 8.0F};
        std::vector<std::uint8_t> y(4, 0);
        std::vector<std::uint8_t> scale(2, 0);
        deepforge::runtime::VariantPack pack{
            {11, x.data()}, {12, y.data()}, {13, scale.data()}};
        status = e5_quantize.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr);
        tests.good(status, "execute FP8 E5M2 block-scale quantize");
        tests.check(scale == std::vector<std::uint8_t>({113, 115}) &&
                        y == std::vector<std::uint8_t>(
                                 {0x74, 0x78, 0xf4, 0x78}),
                    "FP8 E5M2 encoding rounds and saturates with E8M0 scales");
    }

    deepforge::compiler::CompilationResult fp4_roundtrip;
    status = compile_document(fp4_roundtrip_graph(), fp4_roundtrip);
    tests.good(status, "compile packed FP4 quantize/dequantize roundtrip");
    if (status.is_good() && fp4_roundtrip.executable) {
        std::vector<float> x{1.0F, 2.0F, -3.0F, 6.0F};
        std::vector<float> y(4, -99.0F);
        deepforge::runtime::VariantPack pack{{14, x.data()}, {17, y.data()}};
        auto const workspace_size = static_cast<std::size_t>(
            fp4_roundtrip.executable->get_workspace_size());
        auto* workspace = std::aligned_alloc(64, workspace_size);
        status = fp4_roundtrip.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr, pack, workspace);
        std::free(workspace);
        tests.good(status, "execute packed FP4 virtual workspace roundtrip");
        tests.check(close_vectors(y, {1.03125F, 2.0625F, -3.0F, 6.0F},
                                  1.0e-5F),
                    "packed FP4 nibbles and FP8_E4M3 block scales roundtrip");
    }

    deepforge::compiler::CompilationResult mixed_roundtrip;
    status = compile_document(bf16_fp8_f16_roundtrip_graph(), mixed_roundtrip);
    tests.good(status, "compile BF16-to-FP8-to-F16 roundtrip");
    if (status.is_good() && mixed_roundtrip.executable) {
        std::vector<std::uint16_t> x{0x3f80, 0x4000, 0xc080, 0x4100};
        std::vector<std::uint16_t> y(4, 0xffff);
        deepforge::runtime::VariantPack pack{{18, x.data()}, {21, y.data()}};
        auto const workspace_size = static_cast<std::size_t>(
            mixed_roundtrip.executable->get_workspace_size());
        auto* workspace = std::aligned_alloc(64, workspace_size);
        status = mixed_roundtrip.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr, pack, workspace);
        std::free(workspace);
        tests.good(status, "execute BF16-to-FP8-to-F16 roundtrip");
        tests.check(y == std::vector<std::uint16_t>(
                             {0x3c00, 0x4000, 0xc400, 0x4800}),
                    "BF16 load and F16 store preserve exact block-scaled values");
    }

    deepforge::compiler::CompilationResult reordered_dequantize;
    status = compile_document(reordered_block_dequantize_graph(),
                              reordered_dequantize);
    tests.good(status, "compile F8_128x4 block-scale dequantize");
    if (status.is_good() && reordered_dequantize.executable) {
        std::vector<std::uint8_t> x(128 * 32, 0x38);
        std::vector<std::uint8_t> scale(128 * 4, 127);
        scale[f8_128x4_offset(32, 0, 0, 128, 4)] = 128;
        scale[f8_128x4_offset(127, 0, 0, 128, 4)] = 126;
        std::vector<float> y(128 * 32, -99.0F);
        deepforge::runtime::VariantPack pack{
            {51, x.data()}, {52, scale.data()}, {53, y.data()}};
        status = reordered_dequantize.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr);
        tests.good(status, "execute F8_128x4 block-scale dequantize");
        std::vector<float> expected(128 * 32, 1.0F);
        std::fill_n(expected.begin() + 32 * 32, 32, 2.0F);
        std::fill_n(expected.begin() + 127 * 32, 32, 0.5F);
        tests.check(close_vectors(y, expected) &&
                        f8_128x4_offset(32, 0, 0, 128, 4) == 4 &&
                        f8_128x4_offset(127, 0, 0, 128, 4) == 508,
                    "F8_128x4 scale bytes use the CUTLASS 32x4xrest mapping");
    }

    deepforge::compiler::CompilationResult transposed_dequantize;
    status = compile_document(transposed_reordered_block_dequantize_graph(),
                              transposed_dequantize);
    tests.good(status, "compile transposed F8_128x4 block-scale dequantize");
    if (status.is_good() && transposed_dequantize.executable) {
        std::vector<std::uint8_t> x(32 * 128, 0x38);
        std::vector<std::uint8_t> scale(4 * 128, 127);
        scale[f8_128x4_offset(32, 0, 0, 128, 4)] = 128;
        scale[f8_128x4_offset(127, 0, 0, 128, 4)] = 126;
        std::vector<float> y(32 * 128, -99.0F);
        deepforge::runtime::VariantPack pack{
            {71, x.data()}, {72, scale.data()}, {73, y.data()}};
        status = transposed_dequantize.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr);
        tests.good(status, "execute transposed F8_128x4 dequantize");
        std::vector<float> expected(32 * 128, 1.0F);
        for (std::size_t row = 0; row < 32; ++row) {
            expected[row * 128 + 32] = 2.0F;
            expected[row * 128 + 127] = 0.5F;
        }
        tests.check(close_vectors(y, expected),
                    "transposed KxM descriptor uses the same physical tile");
    }

    deepforge::compiler::CompilationResult reordered_quantize;
    status = compile_document(reordered_block_quantize_graph(),
                              reordered_quantize);
    tests.good(status, "compile F8_128x4 block-scale quantize");
    if (status.is_good() && reordered_quantize.executable) {
        std::vector<float> x(128 * 32, 448.0F);
        std::fill_n(x.begin() + 32 * 32, 32, 224.0F);
        std::fill_n(x.begin() + 127 * 32, 32, 112.0F);
        std::vector<std::uint8_t> y(128 * 32, 0);
        std::vector<std::uint8_t> scale(128 * 4, 0);
        deepforge::runtime::VariantPack pack{
            {61, x.data()}, {62, y.data()}, {63, scale.data()}};
        status = reordered_quantize.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr);
        tests.good(status, "execute F8_128x4 block-scale quantize");
        bool scale_matches = true;
        for (std::size_t m = 0; m < 128; ++m) {
            for (std::size_t k = 0; k < 4; ++k) {
                auto expected = static_cast<std::uint8_t>(127);
                if (k == 0 && m == 32) expected = 126;
                if (k == 0 && m == 127) expected = 125;
                scale_matches = scale_matches &&
                                scale[f8_128x4_offset(m, k, 0, 128, 4)] ==
                                    expected;
            }
        }
        tests.check(y == std::vector<std::uint8_t>(128 * 32, 0x7e) &&
                        scale_matches,
                    "reordered quantize writes logical scales and initializes "
                    "all padded scale slots");
    }

    auto reordered_fp4_document = reordered_block_quantize_graph();
    reordered_fp4_document["graph_uid"] = 5016;
    reordered_fp4_document["tensors"]["62"]["data_type"] = "FP4_E2M1";
    reordered_fp4_document["tensors"]["63"]["data_type"] = "FP8_E4M3";
    deepforge::compiler::CompilationResult reordered_fp4_quantize;
    status = compile_document(reordered_fp4_document,
                              reordered_fp4_quantize);
    tests.good(status, "compile E4M3 F8_128x4 scale quantize");
    if (status.is_good() && reordered_fp4_quantize.executable) {
        std::vector<float> x(128 * 32, 6.0F);
        std::fill_n(x.begin() + 32 * 32, 32, 3.0F);
        std::vector<std::uint8_t> y(128 * 32 / 2, 0);
        std::vector<std::uint8_t> scale(128 * 4, 0);
        deepforge::runtime::VariantPack pack{
            {61, x.data()}, {62, y.data()}, {63, scale.data()}};
        status = reordered_fp4_quantize.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr);
        tests.good(status, "execute E4M3 F8_128x4 scale quantize");
        bool scale_matches = true;
        for (std::size_t m = 0; m < 128; ++m) {
            for (std::size_t k = 0; k < 4; ++k) {
                auto const expected = static_cast<std::uint8_t>(
                    k == 0 && m == 32 ? 0x30 : 0x38);
                scale_matches = scale_matches &&
                                scale[f8_128x4_offset(m, k, 0, 128, 4)] ==
                                    expected;
            }
        }
        tests.check(y == std::vector<std::uint8_t>(128 * 32 / 2, 0x77) &&
                        scale_matches,
                    "E4M3 reordered scale supports Frontend FP4 producers");
    }
}

void run_matmul_test(TestRunner& tests) {
    deepforge::compiler::CompilationResult compilation;
    auto status =
        compile_document(matmul_fp8_pass_by_value_graph(), compilation);
    tests.good(status, "compile FP8 matmul with runtime PBV scalars");
    if (status.is_bad() || !compilation.executable) return;
    std::vector<std::uint8_t> a{0x38, 0x40, 0x44, 0x48};
    std::vector<std::uint8_t> b{0x38, 0xb8, 0x40, 0x38};
    std::vector<float> descale_a{0.5F};
    std::vector<float> descale_b{2.0F};
    std::vector<float> scale_c{1.0F};
    std::vector<float> c(4, -99.0F);
    std::vector<float> amax(1, -99.0F);
    deepforge::runtime::VariantPack pack{{21, a.data()},
                                         {22, b.data()},
                                         {23, descale_a.data()},
                                         {24, descale_b.data()},
                                         {25, scale_c.data()},
                                         {26, c.data()},
                                         {27, amax.data()}};
    status = compilation.executable->execute_variant(
        deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr);
    tests.good(status, "execute FP8 matmul with runtime PBV scalars");
    tests.check(close_vectors(c, {5.0F, 1.0F, 11.0F, 1.0F}) &&
                    close_vectors(amax, {11.0F}),
                "FP8 matmul descales inputs and reports pre-output-scale amax");

    deepforge::compiler::CompilationResult embedded;
    status = compile_document(matmul_fp8_embedded_graph(), embedded);
    tests.good(status, "compile FP8 matmul with embedded PBV scalars");
    tests.check(
        std::none_of(embedded.metadata.arguments.begin(),
                     embedded.metadata.arguments.end(), [](auto const& argument) {
                         return argument.uid >= 23 && argument.uid <= 25;
                     }),
        "embedded FP8 scales are absent from public arguments");
    if (status.is_good() && embedded.executable) {
        std::fill(c.begin(), c.end(), -99.0F);
        std::fill(amax.begin(), amax.end(), -99.0F);
        deepforge::runtime::VariantPack embedded_pack{{21, a.data()},
                                                       {22, b.data()},
                                                       {26, c.data()},
                                                       {27, amax.data()}};
        status = embedded.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr, embedded_pack,
            nullptr);
        tests.good(status, "execute FP8 matmul without embedded scalar UIDs");
        tests.check(close_vectors(c, {5.0F, 1.0F, 11.0F, 1.0F}) &&
                        close_vectors(amax, {11.0F}),
                    "embedded FP8 scales match runtime scalar semantics");
    }

    deepforge::compiler::CompilationResult override_compilation;
    status = compile_document(matmul_fp8_override_graph(),
                              override_compilation);
    tests.good(status, "compile FP8 matmul dimension overrides");
    if (status.is_bad() || !override_compilation.executable) return;
    std::vector<std::int32_t> m_override{1};
    std::vector<std::int32_t> n_override{1};
    std::vector<std::int32_t> k_override{1};
    std::fill(c.begin(), c.end(), -99.0F);
    std::fill(amax.begin(), amax.end(), -99.0F);
    pack.emplace(28, m_override.data());
    pack.emplace(29, n_override.data());
    pack.emplace(30, k_override.data());
    status = override_compilation.executable->execute_variant(
        deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr);
    tests.good(status, "execute FP8 matmul dimension overrides");
    tests.check(close_vectors(c, {1.0F, 0.0F, 0.0F, 0.0F}) &&
                    close_vectors(amax, {1.0F}),
                "FP8 M/N/K overrides zero inactive outputs and products");
}

void run_moe_test(TestRunner& tests) {
    deepforge::compiler::CompilationResult compilation;
    auto status = compile_document(moe_graph(), compilation);
    tests.good(status, "compile MoE grouped matmul forward/backward");
    if (status.is_bad() || !compilation.executable) return;
    std::vector<float> token{1.0F, 2.0F, 3.0F, 4.0F,
                             5.0F, 6.0F, 7.0F, 8.0F};
    std::vector<float> weight{1.0F, 0.0F, 0.0F, 1.0F,
                              0.0F, 1.0F, 1.0F, 0.0F};
    std::vector<std::int32_t> offsets{0, 2};
    std::vector<float> output(8, -99.0F);
    std::vector<float> d_output(8, 1.0F);
    std::vector<float> d_weight(8, -99.0F);
    deepforge::runtime::VariantPack pack{{31, token.data()},
                                         {32, weight.data()},
                                         {33, offsets.data()},
                                         {34, output.data()},
                                         {35, d_output.data()},
                                         {36, d_weight.data()}};
    status = compilation.executable->execute_variant(
        deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr);
    tests.good(status, "execute MoE grouped matmul forward/backward");
    tests.check(close_vectors(output,
                              {1.0F, 2.0F, 3.0F, 4.0F,
                               6.0F, 5.0F, 8.0F, 7.0F}),
                "FirstTokenOffset selects the per-token expert");
    tests.check(close_vectors(d_weight,
                              {4.0F, 4.0F, 6.0F, 6.0F,
                               12.0F, 12.0F, 14.0F, 14.0F}),
                "MoE backward computes Token^T times DOutput per expert");
}

void run_attention_forward_test(TestRunner& tests, bool mxfp8) {
    auto const base = mxfp8 ? 200 : 100;
    deepforge::compiler::CompilationResult compilation;
    auto status = compile_document(sdpa_fp8_forward_graph(mxfp8), compilation);
    tests.good(status, mxfp8 ? "compile MXFP8 SDPA forward"
                             : "compile FP8 SDPA forward");
    if (status.is_bad() || !compilation.executable) return;
    std::vector<std::uint8_t> q(4, 0);
    std::vector<std::uint8_t> k(4, 0);
    std::vector<std::uint8_t> v{0x38, 0x44, 0x44, 0x4a};
    std::vector<float> stats(2, -99.0F);
    std::vector<float> amax_o(1, -99.0F);
    deepforge::runtime::VariantPack pack{{base + 1, q.data()},
                                         {base + 2, k.data()},
                                         {base + 3, v.data()},
                                         {base + 11, stats.data()},
                                         {base + 12, amax_o.data()}};
    std::vector<float> output_float(4, -99.0F);
    std::vector<std::uint8_t> output_fp8(4, 0);
    std::vector<float> amax_s(1, -99.0F);
    std::vector<float> scalar_scales(6, 1.0F);
    std::vector<std::uint8_t> mx_q_scale(512, 127);
    std::vector<std::uint8_t> mx_k_scale(512, 127);
    std::vector<std::uint8_t> mx_v_scale(512, 127);
    if (mxfp8) {
        pack.insert_or_assign(base + 4, mx_q_scale.data());
        pack.insert_or_assign(base + 5, mx_k_scale.data());
        pack.insert_or_assign(base + 6, mx_v_scale.data());
        pack.insert_or_assign(base + 10, output_float.data());
    } else {
        for (int index = 0; index < 6; ++index) {
            pack.insert_or_assign(base + 4 + index,
                                  &scalar_scales[static_cast<std::size_t>(index)]);
        }
        pack.insert_or_assign(base + 10, output_fp8.data());
        pack.insert_or_assign(base + 13, amax_s.data());
    }
    status = compilation.executable->execute_variant(
        deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr);
    tests.good(status, mxfp8 ? "execute MXFP8 SDPA forward"
                             : "execute FP8 SDPA forward");
    auto const expected_stats = std::vector<float>(2, std::log(2.0F));
    bool output_matches = false;
    if (mxfp8) {
        output_matches = close_vectors(output_float,
                                       {2.0F, 4.0F, 2.0F, 4.0F});
    } else {
        output_matches =
            output_fp8 == std::vector<std::uint8_t>({0x40, 0x48, 0x40, 0x48}) &&
            close_vectors(amax_s, {0.5F});
    }
    tests.check(output_matches && close_vectors(stats, expected_stats) &&
                    close_vectors(amax_o, {4.0F}),
                mxfp8 ? "MXFP8 block descales and softmax match reference"
                       : "FP8 S/O scaling, stats, and amax match reference");
}

void run_attention_backward_test(TestRunner& tests, bool mxfp8) {
    auto const base = mxfp8 ? 400 : 300;
    auto document = sdpa_fp8_backward_graph(mxfp8);
    deepforge::compiler::CompilationResult compilation;
    auto status = compile_document(document, compilation);
    tests.good(status, mxfp8 ? "compile MXFP8 SDPA backward"
                             : "compile FP8 SDPA backward");
    if (status.is_bad() || !compilation.executable) return;

    std::vector<std::uint8_t> q(4, 0);
    std::vector<std::uint8_t> k(4, 0);
    std::vector<std::uint8_t> v{0x38, 0x44, 0x44, 0x4a};
    std::vector<std::uint8_t> d_o(4, 0x38);
    std::vector<float> stats(2, std::log(4.0F));
    deepforge::runtime::VariantPack pack{{base + 1, q.data()},
                                         {base + 2, k.data()},
                                         {base + 3, v.data()},
                                         {base + 5, d_o.data()},
                                         {base + 6, stats.data()}};
    int next = 7;
    std::vector<float> o_float{2.0F, 4.0F, 2.0F, 4.0F};
    std::vector<std::uint8_t> o_fp8{0x40, 0x48, 0x40, 0x48};
    pack.insert_or_assign(base + 4, mxfp8 ? static_cast<void*>(o_float.data())
                                          : static_cast<void*>(o_fp8.data()));
    std::vector<float> scalar_scales(12, 1.0F);
    std::vector<std::uint8_t> mx_ones(512, 127);
    std::vector<std::uint8_t> mx_q_t(4, 0);
    std::vector<std::uint8_t> mx_k_t(4, 0);
    std::vector<std::uint8_t> mx_do_t(4, 0x38);
    std::vector<float> d_o_float(4, 1.0F);
    if (mxfp8) {
        pack.insert_or_assign(base + next++, mx_q_t.data());
        pack.insert_or_assign(base + next++, mx_k_t.data());
        pack.insert_or_assign(base + next++, mx_do_t.data());
        pack.insert_or_assign(base + next++, d_o_float.data());
        for (int index = 0; index < 7; ++index) {
            pack.insert_or_assign(base + next++, mx_ones.data());
        }
    } else {
        for (int index = 0; index < 12; ++index) {
            pack.insert_or_assign(base + next++,
                                  &scalar_scales[static_cast<std::size_t>(index)]);
        }
    }
    auto const dq_uid = base + next++;
    auto const dk_uid = base + next++;
    auto const dv_uid = base + next++;
    std::vector<float> dq_float(4, -99.0F);
    std::vector<float> dk_float(4, -99.0F);
    std::vector<float> dv_float(4, -99.0F);
    std::vector<std::uint8_t> dq_fp8(4, 0xff);
    std::vector<std::uint8_t> dk_fp8(4, 0xff);
    std::vector<std::uint8_t> dv_fp8(4, 0xff);
    pack.insert_or_assign(dq_uid, mxfp8 ? static_cast<void*>(dq_float.data())
                                        : static_cast<void*>(dq_fp8.data()));
    pack.insert_or_assign(dk_uid, mxfp8 ? static_cast<void*>(dk_float.data())
                                        : static_cast<void*>(dk_fp8.data()));
    pack.insert_or_assign(dv_uid, mxfp8 ? static_cast<void*>(dv_float.data())
                                        : static_cast<void*>(dv_fp8.data()));
    std::vector<float> amax_q(1, -99.0F);
    std::vector<float> amax_k(1, -99.0F);
    std::vector<float> amax_v(1, -99.0F);
    std::vector<float> amax_p(1, -99.0F);
    pack.insert_or_assign(base + next++, amax_q.data());
    pack.insert_or_assign(base + next++, amax_k.data());
    pack.insert_or_assign(base + next++, amax_v.data());
    if (!mxfp8) pack.insert_or_assign(base + next++, amax_p.data());

    status = compilation.executable->execute_variant(
        deepforge::runtime::CpuVariant::kScalar, nullptr, pack, nullptr);
    tests.good(status, mxfp8 ? "execute MXFP8 SDPA backward"
                             : "execute FP8 SDPA backward");
    bool gradients_match = false;
    if (mxfp8) {
        gradients_match = close_vectors(dq_float, {0, 0, 0, 0}) &&
                          close_vectors(dk_float, {0, 0, 0, 0}) &&
                          close_vectors(dv_float, {0.5F, 0.5F, 0.5F, 0.5F});
    } else {
        gradients_match = dq_fp8 == std::vector<std::uint8_t>(4, 0) &&
                          dk_fp8 == std::vector<std::uint8_t>(4, 0) &&
                          dv_fp8 == std::vector<std::uint8_t>(4, 0x30) &&
                          close_vectors(amax_p, {8.0F});
    }
    tests.check(gradients_match && close_vectors(amax_q, {0.0F}) &&
                    close_vectors(amax_k, {0.0F}) &&
                    close_vectors(amax_v, {0.5F}),
                mxfp8 ? "MXFP8 backward consumes Stats and matches reference"
                       : "FP8 backward consumes Stats and matches reference");
}

void run_validation_tests(TestRunner& tests) {
    auto document = sdpa_fp8_backward_graph(true);
    document["tensors"]["407"]["dim"] = {1, 1, 1, 2};
    document["tensors"]["407"]["stride"] = {2, 2, 2, 1};
    deepforge::compiler::CompilationResult compilation;
    auto status = compile_document(document, compilation);
    tests.check(status.code() == deepforge::import::ErrorCode::kInvalidShape,
                "MXFP8 backward rejects mismatched Q_T dimensions");

    document = reordered_block_dequantize_graph();
    document["tensors"]["52"]["stride"] = {5, 1};
    status = compile_document(document, compilation);
    tests.check(
        status.code() == deepforge::import::ErrorCode::kUnsupportedOperation,
        "F8_128x4 rejects descriptors that do not encode a packed MxK tile");

    document = sdpa_fp8_forward_graph(true);
    document["tensors"]["201"]["dim"] = {1, 1, 128, 4};
    document["tensors"]["201"]["stride"] = {512, 512, 4, 1};
    document["tensors"]["201"]["reordering_type"] = "F8_128x4";
    status = compile_document(document, compilation);
    tests.check(
        status.code() == deepforge::import::ErrorCode::kUnsupportedOperation,
        "F8_128x4 rejects non-scale MXFP8 ports");
}

}  // namespace

int main() {
    TestRunner tests;
    run_block_tests(tests);
    run_matmul_test(tests);
    run_moe_test(tests);
    run_attention_forward_test(tests, false);
    run_attention_backward_test(tests, false);
    run_attention_forward_test(tests, true);
    run_attention_backward_test(tests, true);
    run_validation_tests(tests);
    return tests.finish();
}
