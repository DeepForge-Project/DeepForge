#include "DeepForge/Import/SerializedGraphImporter.h"
#include "DeepForge/Import/Capability.h"
#include "DeepForge/Import/Schema.h"

#include <cudnn_frontend/thirdparty/nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

using deepforge::import::ErrorCode;
using deepforge::import::InputFormat;
using deepforge::import::SerializedGraph;
using deepforge::import::SerializedGraphImporter;
using deepforge::import::Status;
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

    void expect_good(Status const& status, std::string const& name) {
        ++checks_;
        if (status.is_bad()) {
            ++failures_;
            std::cerr << "FAIL: " << name << ": " << status.message() << '\n';
        }
    }

    void expect_code(Status const& status, ErrorCode expected, std::string const& name) {
        ++checks_;
        if (status.code() != expected) {
            ++failures_;
            std::cerr << "FAIL: " << name << ": expected "
                      << deepforge::import::error_code_name(expected) << ", got "
                      << deepforge::import::error_code_name(status.code()) << " ("
                      << status.message() << ")\n";
        }
    }

    [[nodiscard]] int finish() const {
        if (failures_ == 0) {
            std::cout << "deepforge-importer: " << checks_ << " checks passed\n";
            return 0;
        }
        std::cerr << "deepforge-importer: " << failures_ << " of " << checks_
                  << " checks failed\n";
        return 1;
    }

private:
    int checks_ = 0;
    int failures_ = 0;
};

std::string read_file(std::filesystem::path const& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("cannot open fixture: " + path.string());
    }
    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

std::vector<std::uint8_t> json_bytes(Json const& document) {
    auto text = document.dump();
    return std::vector<std::uint8_t>(text.begin(), text.end());
}

Status parse_json(SerializedGraphImporter const& importer,
                  Json const& document,
                  SerializedGraph& output) {
    auto bytes = json_bytes(document);
    return importer.parse(std::span<std::uint8_t const>(bytes), InputFormat::kJson, output);
}

Json schema_attribute(deepforge::import::OperationTag tag,
                      std::string_view name) {
    using OT = deepforge::import::OperationTag;
    if (name == "compute_data_type") return "FLOAT";
    if (name == "mode") {
        if (tag == OT::kPointwise || tag == OT::kReduction) return "ADD";
        return "NONE";
    }
    if (name == "math_mode") return "CROSS_CORRELATION";
    if (name == "forward_phase") return "INFERENCE";
    if (name == "distribution") return "UNIFORM";
    if (name == "resample_mode") return "AVGPOOL_EXCLUDE_PADDING";
    if (name == "padding_mode") return "ZERO_PAD";
    if (name == "reshape_mode") return "VIEW_ONLY";
    if (name == "diagonal_alignment") return "TOP_LEFT";
    if (name == "implementation") return "COMPOSITE";
    if (name == "mma_core_mode") return "HALF";
    if (name == "peer_stats") return Json::array();
    if (name == "slices") return Json::array({Json::array({0, 1})});
    if (name == "pre_padding" || name == "post_padding") {
        return Json::array({0});
    }
    if (name == "stride" || name == "dilation" || name == "window" ||
        name == "dim" || name == "block_size" ||
        name == "slice_strides") {
        if (name == "block_size" && tag == OT::kBlockScaleQuantize) return 1;
        return Json::array({1});
    }
    if (name == "permutation") return Json::array({0});
    if (name == "is_negative_scale" || name == "is_deterministic" ||
        name == "alibi_mask" || name == "padding_mask" ||
        name == "is_deterministic_algorithm" || name == "unfuse_fma") {
        return false;
    }
    if (name == "is_mxfp8") {
        return tag == OT::kSdpaMxfp8Fwd || tag == OT::kSdpaMxfp8Bwd;
    }
    if (name == "generate_index" || name == "generate_stats" ||
        name == "axis" || name == "in_place_index" || name == "seed" ||
        name == "max_seq_len_kv" || name == "left_bound" ||
        name == "right_bound" || name == "max_total_seq_len_q" ||
        name == "max_total_seq_len_kv" || name == "padding_value" ||
        name == "relu_lower_clip" || name == "relu_upper_clip" ||
        name == "relu_lower_clip_slope" || name == "swish_beta" ||
        name == "elu_alpha" || name == "softplus_beta" ||
        name == "bernoulli_probability" ||
        name == "dropout_probability" || name == "attn_scale_value") {
        return nullptr;
    }
    if (name == "rope_dim") return 0;
    if (name == "top_k") return 1;
    if (name == "output_scale") return 1.0;
    if (name == "rescale_threshold") return 0.5;
    throw std::runtime_error("test has no schema value for " +
                             std::string(name));
}

Json make_schema_node(deepforge::import::OperationSchema const& schema) {
    Json node{{"tag", schema.serialized_tag},
              {"name", std::string(schema.serialized_tag) + "_test"},
              {"inputs", Json::object()},
              {"outputs", Json::object()}};
    if (schema.allows_indexed_inputs) {
        node["inputs"]["0"] = 1;
    } else if (!schema.input_ports.empty()) {
        node["inputs"][std::string(schema.input_ports.front())] = 1;
    }
    if (schema.tag == deepforge::import::OperationTag::kPointwise) {
        node["inputs"]["IN_1"] = 2;
    }
    if (!schema.output_ports.empty()) {
        node["outputs"][std::string(schema.output_ports.front())] = 3;
    }
    for (auto name : schema.required_attributes) {
        node[std::string(name)] = schema_attribute(schema.tag, name);
    }
    for (auto name : schema.optional_attributes) {
        node[std::string(name)] = schema_attribute(schema.tag, name);
    }
    return node;
}

Json graph_for_schema(Json const& fixture,
                      deepforge::import::OperationSchema const& schema) {
    Json graph = fixture;
    graph["nodes"] = Json::array({make_schema_node(schema)});
    return graph;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: deepforge_importer_test <fixture.json>\n";
        return 2;
    }

    TestRunner tests;
    try {
        std::filesystem::path const fixture_path(argv[1]);
        auto const fixture_text = read_file(fixture_path);
        auto const fixture = Json::parse(fixture_text);
        std::vector<std::uint8_t> fixture_bytes(fixture_text.begin(), fixture_text.end());

        SerializedGraphImporter importer;
        SerializedGraph from_json;
        auto status = importer.parse(std::span<std::uint8_t const>(fixture_bytes),
                                     InputFormat::kJson,
                                     from_json);
        tests.expect_good(status, "valid JSON fixture");
        tests.check(from_json.json_version == "1.0", "canonical schema version");
        tests.check(from_json.cudnn_frontend_version == 12400, "canonical Frontend version");
        tests.check(from_json.tensors.size() == 3, "canonical tensor count");
        tests.check(from_json.nodes.size() == 1 &&
                        from_json.nodes.front().tag ==
                            deepforge::import::OperationTag::kConvFprop,
                    "canonical node container");
        auto const* conv = from_json.single_conv_fprop();
        tests.check(conv != nullptr, "canonical Conv attributes");
        if (conv == nullptr) {
            return tests.finish();
        }
        tests.check(conv->x_uid == 1 && conv->w_uid == 2 && conv->y_uid == 3,
                    "canonical Conv ports");
        tests.check(conv->pre_padding == std::array<std::int64_t, 2>{1, 0} &&
                        conv->post_padding == std::array<std::int64_t, 2>{1, 2},
                    "asymmetric padding is preserved");
        tests.check(from_json.context.sm_count && *from_json.context.sm_count == -1,
                    "Frontend default sm_count is accepted");
        tests.check(from_json.tensors.at(1).dim ==
                            std::vector<std::int64_t>({1, 3, 5, 5}) &&
                        from_json.tensors.at(2).dim ==
                            std::vector<std::int64_t>({5, 3, 3, 3}) &&
                        from_json.tensors.at(3).dim ==
                            std::vector<std::int64_t>({1, 5, 5, 5}),
                    "non-vector-multiple C/K dimensions are preserved");

        auto capabilities = deepforge::import::operation_capabilities();
        tests.check(capabilities.size() == 39,
                    "capability registry covers every v1.24.0 serializer tag");
        std::set<std::string_view> capability_names;
        std::size_t validated_count = 0;
        std::size_t parsed_count = 0;
        for (auto const& capability : capabilities) {
            capability_names.insert(capability.serialized_tag);
            validated_count +=
                capability.level ==
                deepforge::import::CapabilityLevel::kValidated;
            parsed_count += capability.level ==
                            deepforge::import::CapabilityLevel::kParsed;
            tests.check(deepforge::import::operation_tag_name(capability.tag) ==
                            capability.serialized_tag,
                        "operation tag name round-trips");
        }
        tests.check(capability_names.size() == capabilities.size() &&
                        validated_count == 1 && parsed_count == 38,
                    "capability tags are unique and all schemas are parsed");
        auto schemas = deepforge::import::operation_schemas();
        std::set<std::string_view> schema_names;
        for (auto const& schema : schemas) {
            schema_names.insert(schema.serialized_tag);
            auto const* capability =
                deepforge::import::find_operation_capability(
                    schema.serialized_tag);
            tests.check(capability != nullptr && capability->tag == schema.tag,
                        "schema and capability registry agree");
        }
        tests.check(schemas.size() == 39 && schema_names.size() == 39,
                    "schema inventory covers 39 unique tags");
        tests.check(deepforge::import::pointwise_modes().size() == 50 &&
                        deepforge::import::reduction_modes().size() == 9,
                    "pointwise and reduction mode catalogs are complete");
        for (auto const name : {"FLOAT", "DOUBLE", "HALF", "INT8",
                                "INT32", "INT8x4", "UINT8", "UINT8x4",
                                "INT8x32", "BFLOAT16", "INT64", "BOOLEAN",
                                "FP8_E4M3", "FP8_E5M2",
                                "FAST_FLOAT_FOR_FP8", "FP8_E8M0",
                                "FP4_E2M1", "INT4", "COMPLEX_FP32",
                                "COMPLEX_FP64"}) {
            auto type = deepforge::import::data_type_from_name(name);
            tests.check(type && deepforge::import::data_type_name(*type) == name &&
                            deepforge::import::data_type_storage_bits(*type) != 0,
                        "data type catalog entry round-trips");
        }

        SerializedGraph from_file;
        status = importer.parse_file(fixture_path, InputFormat::kJson, from_file);
        tests.expect_good(status, "parse_file JSON");
        tests.check(from_file == from_json, "parse_file canonical equality");

        auto ubjson = Json::to_ubjson(fixture);
        SerializedGraph from_ubjson;
        status = importer.parse(std::span<std::uint8_t const>(ubjson),
                                InputFormat::kUbjson,
                                from_ubjson);
        tests.expect_good(status, "valid UBJSON fixture");
        tests.check(from_ubjson == from_json, "JSON and UBJSON canonical equality");

        SerializedGraph auto_json;
        status = importer.parse(std::span<std::uint8_t const>(fixture_bytes), auto_json);
        tests.expect_good(status, "auto-detect JSON");
        tests.check(auto_json == from_json, "auto-detected JSON canonical equality");

        SerializedGraph auto_ubjson;
        status = importer.parse(std::span<std::uint8_t const>(ubjson), auto_ubjson);
        tests.expect_good(status, "auto-detect UBJSON");
        tests.check(auto_ubjson == from_json, "auto-detected UBJSON canonical equality");

        Json graph_json = fixture;
        graph_json.erase("variant_pack_uids");
        graph_json.erase("pass_by_values");
        graph_json.erase("workspace_modifications");
        graph_json.erase("variant_pack_replacements");
        graph_json.erase("fe_workspace_size");
        graph_json.erase("behavior_notes");
        graph_json.erase("tensors_to_dump");
        SerializedGraph graph_json_model;
        status = parse_json(importer, graph_json, graph_json_model);
        tests.expect_good(status, "Graph JSON without vector metadata");
        tests.check(graph_json_model == from_json, "optional execution metadata is not canonical semantics");

        for (auto const& schema : schemas) {
            auto document = graph_for_schema(fixture, schema);
            SerializedGraph json_graph;
            status = parse_json(importer, document, json_graph);
            tests.expect_good(status,
                              "parse schema tag " +
                                  std::string(schema.serialized_tag));
            auto schema_ubjson = Json::to_ubjson(document);
            SerializedGraph ubjson_graph;
            status = importer.parse(
                std::span<std::uint8_t const>(schema_ubjson),
                InputFormat::kUbjson, ubjson_graph);
            tests.expect_good(status,
                              "parse UBJSON schema tag " +
                                  std::string(schema.serialized_tag));
            tests.check(json_graph == ubjson_graph &&
                            json_graph.nodes.size() == 1 &&
                            json_graph.nodes.front().tag == schema.tag,
                        "JSON/UBJSON canonical schema equality for " +
                            std::string(schema.serialized_tag));
        }

        auto const* pointwise_schema =
            deepforge::import::find_operation_schema(
                deepforge::import::OperationTag::kPointwise);
        auto const* reduction_schema =
            deepforge::import::find_operation_schema(
                deepforge::import::OperationTag::kReduction);
        tests.check(pointwise_schema != nullptr && reduction_schema != nullptr,
                    "mode schemas are available");
        if (pointwise_schema != nullptr && reduction_schema != nullptr) {
            for (auto mode : deepforge::import::pointwise_modes()) {
                auto document = graph_for_schema(fixture, *pointwise_schema);
                document["nodes"][0]["mode"] = std::string(mode);
                document["nodes"][0]["inputs"] = Json::object();
                auto input_count =
                    deepforge::import::pointwise_input_count(mode);
                tests.check(input_count.has_value(),
                            "pointwise mode has an arity");
                for (std::size_t index = 0; index < input_count.value_or(0);
                     ++index) {
                    document["nodes"][0]["inputs"]
                            ["IN_" + std::to_string(index)] =
                        static_cast<std::int64_t>(index % 2 + 1);
                }
                SerializedGraph graph;
                tests.expect_good(parse_json(importer, document, graph),
                                  "pointwise mode " + std::string(mode));
            }
            for (auto mode : deepforge::import::reduction_modes()) {
                auto document = graph_for_schema(fixture, *reduction_schema);
                document["nodes"][0]["mode"] = std::string(mode);
                SerializedGraph graph;
                tests.expect_good(parse_json(importer, document, graph),
                                  "reduction mode " + std::string(mode));
            }
        }

        Json generalized_conv = fixture;
        generalized_conv["nodes"][0]["stride"] = Json::array({2, 1});
        SerializedGraph generalized_conv_graph;
        status = parse_json(importer, generalized_conv,
                            generalized_conv_graph);
        tests.expect_good(status,
                          "non-unit Conv is recognized without specialization");
        tests.check(generalized_conv_graph.single_conv_fprop() == nullptr &&
                        std::holds_alternative<
                            deepforge::import::GenericOperationDesc>(
                            generalized_conv_graph.nodes.front().attributes),
                    "generalized Conv remains a generic canonical node");

        auto expect_invalid = [&](std::string const& name, ErrorCode code, auto mutate) {
            Json invalid = fixture;
            mutate(invalid);
            SerializedGraph result = from_json;
            auto const invalid_status = parse_json(importer, invalid, result);
            tests.expect_code(invalid_status, code, name);
            tests.check(result == from_json, name + " leaves output unchanged");
        };

        expect_invalid("unknown serialized attribute", ErrorCode::kInvalidValue,
                       [&](Json& value) {
                           value = graph_for_schema(value, *pointwise_schema);
                           value["nodes"][0]["not_serialized"] = 1;
                       });
        expect_invalid("missing serialized attribute", ErrorCode::kMissingField,
                       [&](Json& value) {
                           value = graph_for_schema(value, *pointwise_schema);
                           value["nodes"][0].erase("axis");
                       });
        expect_invalid("unknown pointwise mode",
                       ErrorCode::kInvalidFieldType, [&](Json& value) {
                           value = graph_for_schema(value, *pointwise_schema);
                           value["nodes"][0]["mode"] = "NOT_A_MODE";
                       });
        expect_invalid("unknown reduction mode",
                       ErrorCode::kInvalidFieldType, [&](Json& value) {
                           value = graph_for_schema(value, *reduction_schema);
                           value["nodes"][0]["mode"] = "NOT_A_MODE";
                       });
        expect_invalid("unknown operation port", ErrorCode::kInvalidValue,
                       [&](Json& value) {
                           value = graph_for_schema(value, *pointwise_schema);
                           value["nodes"][0]["inputs"]["BAD_PORT"] = 2;
                       });
        expect_invalid("pointwise input arity", ErrorCode::kInvalidValue,
                       [&](Json& value) {
                           value = graph_for_schema(value, *pointwise_schema);
                           value["nodes"][0]["inputs"].erase("IN_1");
                       });
        auto const* concatenate_schema =
            deepforge::import::find_operation_schema(
                deepforge::import::OperationTag::kConcatenate);
        tests.check(concatenate_schema != nullptr,
                    "concatenate schema is available");
        expect_invalid("concatenate input gap", ErrorCode::kInvalidValue,
                       [&](Json& value) {
                           value = graph_for_schema(value, *concatenate_schema);
                           value["nodes"][0]["inputs"].erase("0");
                           value["nodes"][0]["inputs"]["1"] = 1;
                       });
        auto const* moe_schema = deepforge::import::find_operation_schema(
            deepforge::import::OperationTag::kMoeGroupedMatmul);
        tests.check(moe_schema != nullptr, "MoE schema is available");
        expect_invalid("non-optional top_k", ErrorCode::kInvalidFieldType,
                       [&](Json& value) {
                           value = graph_for_schema(value, *moe_schema);
                           value["nodes"][0]["top_k"] = nullptr;
                       });
        expect_invalid("top_k exceeds int32", ErrorCode::kInvalidFieldType,
                       [&](Json& value) {
                           value = graph_for_schema(value, *moe_schema);
                           value["nodes"][0]["top_k"] =
                               static_cast<std::int64_t>(
                                   std::numeric_limits<std::int32_t>::max()) +
                               1;
                       });
        expect_invalid("MoE mode is non-null", ErrorCode::kInvalidFieldType,
                       [&](Json& value) {
                           value = graph_for_schema(value, *moe_schema);
                           value["nodes"][0]["mode"] = nullptr;
                       });

        expect_invalid("schema version mismatch", ErrorCode::kSchemaVersionMismatch, [](Json& value) {
            value["json_version"] = "2.0";
        });
        expect_invalid("missing backend version", ErrorCode::kMissingField,
                       [](Json& value) {
                           value.erase("cudnn_backend_version");
                       });
        expect_invalid("Frontend version mismatch", ErrorCode::kFrontendVersionMismatch, [](Json& value) {
            value["cudnn_frontend_version"] = 12300;
        });
        expect_invalid("unknown context field", ErrorCode::kInvalidValue,
                       [](Json& value) {
                           value["context"]["not_serialized"] = false;
                       });
        expect_invalid("unknown tensor field", ErrorCode::kInvalidValue,
                       [](Json& value) {
                           value["tensors"]["1"]["not_serialized"] = 0;
                       });
        expect_invalid("unknown node", ErrorCode::kUnsupportedNode, [](Json& value) {
            value["nodes"][0]["tag"] = "NOT_A_CUDNN_FRONTEND_TAG";
            value.erase("tensors");
        });
        expect_invalid("second known node is diagnosed precisely",
                       ErrorCode::kMissingField, [&](Json& value) {
            auto second = make_schema_node(*pointwise_schema);
            second.erase("mode");
            value["nodes"].push_back(std::move(second));
        });
        expect_invalid("missing explicit UID", ErrorCode::kMissingUid, [](Json& value) {
            value["tensors"]["1"].erase("uid_assigned");
        });
        expect_invalid("tensor UID/key mismatch", ErrorCode::kInvalidValue, [](Json& value) {
            value["tensors"]["2"]["uid"] = 1;
        });
        expect_invalid("name tensor reference", ErrorCode::kMissingUid, [](Json& value) {
            value["nodes"][0]["inputs"]["X"] = "X";
        });
        expect_invalid("non-f32 tensor", ErrorCode::kUnsupportedDataType, [](Json& value) {
            value["tensors"]["1"]["data_type"] = "HALF";
        });
        expect_invalid("dynamic shape context",
                       ErrorCode::kUnsupportedExecutionMetadata, [](Json& value) {
            value["context"]["is_dynamic_shape_enabled"] = true;
        });
        expect_invalid("sm_count outside int32", ErrorCode::kInvalidValue, [](Json& value) {
            value["context"]["sm_count"] = static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()) + 1;
        });
        expect_invalid("non-packed stride", ErrorCode::kInvalidLayout, [](Json& value) {
            value["tensors"]["1"]["stride"][0] = 74;
        });
        expect_invalid("wrong output shape", ErrorCode::kInvalidShape, [](Json& value) {
            value["tensors"]["3"]["dim"][2] = 4;
        });
        expect_invalid("non-empty pass-by-values",
                       ErrorCode::kUnsupportedExecutionMetadata,
                       [](Json& value) {
                           value["pass_by_values"]["4"] = 1.0;
                       });
        expect_invalid("non-empty variant replacements",
                       ErrorCode::kUnsupportedExecutionMetadata,
                       [](Json& value) {
                           value["variant_pack_replacements"] = Json::array({Json::array({1, Json::array({3, 0})})});
                       });
        expect_invalid("variant-pack UID mismatch", ErrorCode::kInvalidValue, [](Json& value) {
            value["variant_pack_uids"] = Json::array({1, 2});
        });
        expect_invalid("dimension multiplication overflow", ErrorCode::kDimensionOverflow, [](Json& value) {
            value["tensors"]["1"]["dim"][2] = std::numeric_limits<std::int64_t>::max();
        });
        expect_invalid("UID exceeds int64", ErrorCode::kDimensionOverflow, [](Json& value) {
            value["nodes"][0]["inputs"]["X"] = std::numeric_limits<std::uint64_t>::max();
        });

        Json named_graph = graph_for_schema(fixture, *pointwise_schema);
        auto named_x = fixture["tensors"]["1"];
        named_x["name"] = "named_x";
        named_x["uid"] = 0;
        named_x["uid_assigned"] = false;
        auto named_y = fixture["tensors"]["3"];
        named_y["name"] = "named_y";
        named_y["uid"] = 0;
        named_y["uid_assigned"] = false;
        named_graph["tensors"] =
            Json::object({{"named_x", named_x}, {"named_y", named_y}});
        named_graph["nodes"][0]["inputs"]["IN_0"] = "named_x";
        named_graph["nodes"][0]["inputs"]["IN_1"] = "named_x";
        named_graph["nodes"][0]["outputs"]["OUT_0"] = "named_y";
        SerializedGraph named_model;
        status = parse_json(importer, named_graph, named_model);
        tests.expect_good(status, "name-keyed tensors and references");
        tests.check(named_model.tensors.empty() &&
                        named_model.named_tensors.size() == 2,
                    "name-keyed tensors remain distinct canonical entries");
        auto named_ubjson = Json::to_ubjson(named_graph);
        SerializedGraph named_ubjson_model;
        status = importer.parse(std::span<std::uint8_t const>(named_ubjson),
                                InputFormat::kUbjson, named_ubjson_model);
        tests.expect_good(status, "name-keyed UBJSON graph");
        tests.check(named_model == named_ubjson_model,
                    "name-keyed JSON/UBJSON canonical equality");

        Json scalar_graph = graph_for_schema(fixture, *pointwise_schema);
        scalar_graph["tensors"]["1"]["is_pass_by_value"] = true;
        scalar_graph["tensors"]["1"]["pass_by_value"] = 2.5;
        SerializedGraph scalar_model;
        status = parse_json(importer, scalar_graph, scalar_model);
        tests.expect_good(status, "pass-by-value payload is recognized");
        tests.check(scalar_model.tensors.at(1).pass_by_value.has_value(),
                    "pass-by-value payload is preserved canonically");

        auto const* batchnorm_schema =
            deepforge::import::find_operation_schema(
                deepforge::import::OperationTag::kBatchNorm);
        tests.check(batchnorm_schema != nullptr,
                    "batchnorm schema is available");
        Json peer_stats_graph = graph_for_schema(fixture, *batchnorm_schema);
        auto peer_tensor = fixture["tensors"]["2"];
        peer_tensor["name"] = "peer_stat";
        peer_tensor["uid"] = 4;
        peer_stats_graph["nodes"][0]["peer_stats"] =
            Json::array({peer_tensor});
        SerializedGraph peer_stats_model;
        status = parse_json(importer, peer_stats_graph, peer_stats_model);
        tests.expect_good(status, "embedded peer_stats tensor");
        auto const* batchnorm = std::get_if<
            deepforge::import::GenericOperationDesc>(
            &peer_stats_model.nodes.front().attributes);
        bool peer_list_matches = false;
        if (batchnorm != nullptr) {
            auto const peer_list = batchnorm->input_lists.find("peer_stats");
            peer_list_matches =
                peer_list != batchnorm->input_lists.end() &&
                peer_list->second ==
                    std::vector<deepforge::import::TensorReference>{
                        deepforge::import::TensorReference{std::int64_t{4}}};
        }
        tests.check(
            peer_stats_model.tensor_count() == 4 &&
                peer_stats_model.tensors.contains(4) && batchnorm != nullptr &&
                peer_list_matches,
            "peer_stats is normalized into tensors and canonical inputs");
        auto peer_stats_ubjson = Json::to_ubjson(peer_stats_graph);
        SerializedGraph peer_stats_ubjson_model;
        status = importer.parse(
            std::span<std::uint8_t const>(peer_stats_ubjson),
            InputFormat::kUbjson, peer_stats_ubjson_model);
        tests.expect_good(status, "embedded peer_stats UBJSON tensor");
        tests.check(peer_stats_model == peer_stats_ubjson_model,
                    "peer_stats JSON/UBJSON canonical equality");

        Json duplicate_peer_stats = peer_stats_graph;
        duplicate_peer_stats["tensors"]["4"] = peer_tensor;
        SerializedGraph duplicate_peer_stats_model;
        status = parse_json(importer, duplicate_peer_stats,
                            duplicate_peer_stats_model);
        tests.expect_good(status, "identical root and peer_stats tensor");
        tests.check(duplicate_peer_stats_model.tensor_count() == 4,
                    "identical embedded tensor is coalesced");

        expect_invalid("conflicting peer_stats UID", ErrorCode::kDuplicateUid,
                       [&](Json& value) {
                           value = duplicate_peer_stats;
                           value["nodes"][0]["peer_stats"][0]["name"] =
                               "conflicting_peer";
                       });
        expect_invalid("malformed peer_stats tensor", ErrorCode::kMissingField,
                       [&](Json& value) {
                           value = peer_stats_graph;
                           value["nodes"][0]["peer_stats"][0].erase("stride");
                       });
        expect_invalid("virtual peer_stats without producer",
                       ErrorCode::kInvalidValue, [&](Json& value) {
                           value = peer_stats_graph;
                           value["nodes"][0]["peer_stats"][0]["is_virtual"] =
                               true;
                       });

        Json dag_graph = graph_for_schema(fixture, *pointwise_schema);
        auto virtual_tensor = fixture["tensors"]["3"];
        virtual_tensor["name"] = "virtual_mid";
        virtual_tensor["uid"] = 4;
        virtual_tensor["uid_assigned"] = true;
        virtual_tensor["is_virtual"] = true;
        dag_graph["tensors"]["4"] = virtual_tensor;
        auto first_node = make_schema_node(*pointwise_schema);
        first_node["name"] = "first";
        first_node["outputs"]["OUT_0"] = 4;
        auto second_node = make_schema_node(*pointwise_schema);
        second_node["name"] = "second";
        second_node["inputs"]["IN_0"] = 4;
        second_node["outputs"]["OUT_0"] = 3;
        dag_graph["nodes"] = Json::array({first_node, second_node});
        SerializedGraph dag_model;
        status = parse_json(importer, dag_graph, dag_model);
        tests.expect_good(status, "topologically ordered two-node DAG");
        tests.check(dag_model.nodes.size() == 2,
                    "multi-node DAG is preserved canonically");

        Json reversed_dag = dag_graph;
        reversed_dag["nodes"] = Json::array({second_node, first_node});
        SerializedGraph unchanged = from_json;
        status = parse_json(importer, reversed_dag, unchanged);
        tests.expect_code(status, ErrorCode::kInvalidValue,
                          "virtual input before producer");
        tests.check(unchanged == from_json,
                    "failed DAG parse leaves output unchanged");

        Json duplicate_producer = dag_graph;
        duplicate_producer["nodes"][1]["inputs"]["IN_0"] = 1;
        duplicate_producer["nodes"][1]["outputs"]["OUT_0"] = 4;
        unchanged = from_json;
        status = parse_json(importer, duplicate_producer, unchanged);
        tests.expect_code(status, ErrorCode::kInvalidValue,
                          "duplicate virtual tensor producer");
        tests.check(unchanged == from_json,
                    "duplicate producer leaves output unchanged");

        auto trailing_ubjson = ubjson;
        trailing_ubjson.push_back(0);
        SerializedGraph rejected;
        status = importer.parse(std::span<std::uint8_t const>(trailing_ubjson),
                                InputFormat::kUbjson,
                                rejected);
        tests.expect_code(status, ErrorCode::kParseError, "UBJSON trailing byte");

        auto trailing_noop_ubjson = ubjson;
        trailing_noop_ubjson.push_back('N');
        status = importer.parse(std::span<std::uint8_t const>(trailing_noop_ubjson),
                                InputFormat::kUbjson,
                                rejected);
        tests.expect_code(status, ErrorCode::kParseError, "UBJSON trailing no-op");

        auto optimized_ubjson = Json::to_ubjson(fixture, true, true);
        status = importer.parse(std::span<std::uint8_t const>(optimized_ubjson),
                                InputFormat::kUbjson,
                                rejected);
        tests.expect_code(status, ErrorCode::kParseError, "non-default UBJSON encoding");

        auto truncated_ubjson = ubjson;
        truncated_ubjson.pop_back();
        status = importer.parse(std::span<std::uint8_t const>(truncated_ubjson),
                                InputFormat::kUbjson,
                                rejected);
        tests.expect_code(status, ErrorCode::kParseError, "truncated UBJSON");

        auto trailing_json = fixture_bytes;
        trailing_json.push_back('x');
        status = importer.parse(std::span<std::uint8_t const>(trailing_json),
                                InputFormat::kJson,
                                rejected);
        tests.expect_code(status, ErrorCode::kParseError, "JSON trailing token");

        status = importer.parse(std::span<std::uint8_t const>(), InputFormat::kAuto, rejected);
        tests.expect_code(status, ErrorCode::kInvalidArgument, "empty input");

        status = importer.parse(std::span<std::uint8_t const>(fixture_bytes),
                                static_cast<InputFormat>(255), rejected);
        tests.expect_code(status, ErrorCode::kInvalidArgument,
                          "unknown input format");

        std::vector<std::uint8_t> oversized(
            deepforge::import::kMaximumSerializedGraphBytes + 1, ' ');
        status = importer.parse(std::span<std::uint8_t const>(oversized),
                                InputFormat::kJson, rejected);
        tests.expect_code(status, ErrorCode::kInvalidArgument,
                          "oversized serialized Graph");

        status = importer.parse_file(fixture_path.parent_path() / "missing.json",
                                     InputFormat::kJson,
                                     rejected);
        tests.expect_code(status, ErrorCode::kIoError, "missing input file");
    } catch (std::exception const& exception) {
        std::cerr << "FAIL: unhandled test exception: " << exception.what() << '\n';
        return 2;
    }

    return tests.finish();
}
