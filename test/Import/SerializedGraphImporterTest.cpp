#include "DeepForge/Import/SerializedGraphImporter.h"

#include <cudnn_frontend/thirdparty/nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
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
        tests.check(from_json.conv.x_uid == 1 && from_json.conv.w_uid == 2 &&
                        from_json.conv.y_uid == 3,
                    "canonical Conv ports");
        tests.check(from_json.conv.pre_padding == std::array<std::int64_t, 2>{1, 0} &&
                        from_json.conv.post_padding == std::array<std::int64_t, 2>{1, 2},
                    "asymmetric padding is preserved");
        tests.check(from_json.context.sm_count && *from_json.context.sm_count == -1,
                    "Frontend default sm_count is accepted");
        tests.check(from_json.tensors.at(1).dim == std::array<std::int64_t, 4>{1, 3, 5, 5} &&
                        from_json.tensors.at(2).dim == std::array<std::int64_t, 4>{5, 3, 3, 3} &&
                        from_json.tensors.at(3).dim == std::array<std::int64_t, 4>{1, 5, 5, 5},
                    "non-vector-multiple C/K dimensions are preserved");

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

        auto expect_invalid = [&](std::string const& name, ErrorCode code, auto mutate) {
            Json invalid = fixture;
            mutate(invalid);
            SerializedGraph result = from_json;
            auto const invalid_status = parse_json(importer, invalid, result);
            tests.expect_code(invalid_status, code, name);
            tests.check(result == from_json, name + " leaves output unchanged");
        };

        expect_invalid("schema version mismatch", ErrorCode::kSchemaVersionMismatch, [](Json& value) {
            value["json_version"] = "2.0";
        });
        expect_invalid("Frontend version mismatch", ErrorCode::kFrontendVersionMismatch, [](Json& value) {
            value["cudnn_frontend_version"] = 12300;
        });
        expect_invalid("unknown node", ErrorCode::kUnsupportedNode, [](Json& value) {
            value["nodes"][0]["tag"] = "POINTWISE";
            value.erase("tensors");
        });
        expect_invalid("missing explicit UID", ErrorCode::kMissingUid, [](Json& value) {
            value["tensors"]["1"].erase("uid_assigned");
        });
        expect_invalid("duplicate tensor UID", ErrorCode::kDuplicateUid, [](Json& value) {
            value["tensors"]["2"]["uid"] = 1;
        });
        expect_invalid("name tensor reference", ErrorCode::kMissingUid, [](Json& value) {
            value["nodes"][0]["inputs"]["X"] = "X";
        });
        expect_invalid("non-f32 tensor", ErrorCode::kUnsupportedDataType, [](Json& value) {
            value["tensors"]["1"]["data_type"] = "HALF";
        });
        expect_invalid("dynamic shape context", ErrorCode::kInvalidShape, [](Json& value) {
            value["context"]["is_dynamic_shape_enabled"] = true;
        });
        expect_invalid("sm_count outside int32", ErrorCode::kInvalidValue, [](Json& value) {
            value["context"]["sm_count"] = static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()) + 1;
        });
        expect_invalid("non-packed stride", ErrorCode::kInvalidLayout, [](Json& value) {
            value["tensors"]["1"]["stride"][0] = 74;
        });
        expect_invalid("non-unit Conv stride", ErrorCode::kInvalidValue, [](Json& value) {
            value["nodes"][0]["stride"] = Json::array({2, 1});
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
