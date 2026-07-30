#include "DeepForge/Compiler/Conv2DImporter.h"
#include "DeepForge/Import/SerializedGraphImporter.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace {

using deepforge::compiler::import_conv2d;
using deepforge::compiler::verify_conv2d_module;
using deepforge::import::ErrorCode;
using deepforge::import::InputFormat;
using deepforge::import::SerializedGraph;
using deepforge::import::SerializedGraphImporter;
using deepforge::import::Status;

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

    void expect_code(Status const& status,
                     ErrorCode expected,
                     std::string const& name) {
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
            std::cout << "deepforge-mlir-import: " << checks_ << " checks passed\n";
            return 0;
        }
        std::cerr << "deepforge-mlir-import: " << failures_ << " of " << checks_
                  << " checks failed\n";
        return 1;
    }

private:
    int checks_ = 0;
    int failures_ = 0;
};

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: deepforge_mlir_import_test <fixture.json>\n";
        return 2;
    }

    TestRunner tests;
    SerializedGraph graph;
    SerializedGraphImporter importer;
    auto status = importer.parse_file(std::filesystem::path(argv[1]),
                                      InputFormat::kJson, graph);
    tests.expect_good(status, "canonical model import");
    if (status.is_bad()) {
        return tests.finish();
    }

    mlir::DialectRegistry registry;
    registry.insert<mlir::arith::ArithDialect,
                    mlir::func::FuncDialect,
                    mlir::linalg::LinalgDialect,
                    mlir::tensor::TensorDialect>();
    mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
    context.loadAllAvailableDialects();

    mlir::OwningOpRef<mlir::ModuleOp> module;
    deepforge::compiler::Conv2DCompileMetadata metadata;
    status = import_conv2d(context, graph, module, {}, &metadata);
    tests.expect_good(status, "standard Tensor/Linalg import");
    tests.check(static_cast<bool>(module), "import returns a module");
    tests.check(metadata.x_uid == 1 && metadata.w_uid == 2 && metadata.y_uid == 3,
                "compile metadata preserves tensor UIDs");
    tests.check(metadata.x_shape == std::array<std::int64_t, 4>{1, 5, 5, 3} &&
                    metadata.w_shape == std::array<std::int64_t, 4>{5, 3, 3, 3} &&
                    metadata.y_shape == std::array<std::int64_t, 4>{1, 5, 5, 5},
                "compile metadata records physical tensor shapes");
    tests.check(metadata.padded_x_shape ==
                    std::array<std::int64_t, 4>{1, 7, 7, 3} &&
                    metadata.pre_padding == std::array<std::int64_t, 2>{1, 0} &&
                    metadata.post_padding == std::array<std::int64_t, 2>{1, 2},
                "compile metadata records materialized padding");
    if (!module) {
        return tests.finish();
    }

    tests.expect_good(verify_conv2d_module(*module, graph),
                      "independent generated IR verifier");
    tests.check(!llvm::failed(mlir::verify(*module)),
                "standard MLIR verifier accepts generated module");

    std::string printed;
    llvm::raw_string_ostream stream(printed);
    module->print(stream);
    stream.flush();
    tests.check(printed.find("linalg.conv_2d_nhwc_fhwc") != std::string::npos,
                "named Linalg Conv2D is present");
    tests.check(printed.find("tensor.pad") != std::string::npos,
                "standard tensor.pad is present");
    tests.check(printed.find("linalg.fill") != std::string::npos,
                "destination fill is present");
    tests.check(printed.find("cudnn.") == std::string::npos,
                "no custom cuDNN operation is present");
    tests.check(printed.find("tensor<?") == std::string::npos,
                "generated module has no dynamic tensor dimensions");
    tests.check(printed.find("tensor.cast") == std::string::npos,
                "generated module has no implicit layout cast");
    tests.check(printed.find("low[0, 1, 0, 0]") != std::string::npos &&
                    printed.find("high[0, 1, 2, 0]") != std::string::npos,
                "asymmetric logical padding maps to physical NHWC padding");
    tests.check(printed.find("tensor<5x3x3x3xf32>") != std::string::npos,
                "filter uses FHW C order");
    tests.check(printed.find("tensor<3x3x3x5xf32>") == std::string::npos,
                "filter is not incorrectly emitted as HWCF");

    mlir::ParserConfig parser_config(&context);
    auto round_trip = mlir::parseSourceString<mlir::ModuleOp>(printed,
                                                               parser_config,
                                                               "generated.mlir");
    tests.check(static_cast<bool>(round_trip), "printed module parses again");
    if (round_trip) {
        tests.check(!llvm::failed(mlir::verify(*round_trip)),
                    "round-tripped module passes standard verifier");
        tests.expect_good(verify_conv2d_module(*round_trip, graph),
                          "round-tripped module passes independent verifier");
    }

    std::filesystem::path golden_path =
        std::filesystem::path(argv[1]).parent_path() / "conv2d_f32.mlir";
    std::ifstream golden_file(golden_path, std::ios::binary);
    tests.check(static_cast<bool>(golden_file), "MLIR golden fixture is readable");
    if (golden_file) {
        std::string golden_text{std::istreambuf_iterator<char>(golden_file),
                                std::istreambuf_iterator<char>()};
        auto golden = mlir::parseSourceString<mlir::ModuleOp>(
            golden_text, parser_config, "conv2d_f32.mlir");
        tests.check(static_cast<bool>(golden),
                    "MLIR golden fixture parses with MLIR 22.1.8");
        if (golden) {
            tests.expect_good(verify_conv2d_module(*golden, graph),
                              "MLIR golden fixture passes independent verifier");
        }
    }

    auto symmetric = graph;
    symmetric.conv.pre_padding = {1, 1};
    symmetric.conv.post_padding = {1, 1};
    status = import_conv2d(context, symmetric, module);
    tests.expect_good(status, "symmetric padding import");
    if (module) {
        tests.expect_good(verify_conv2d_module(*module, symmetric),
                          "symmetric padding verifier");
    }

    auto minimal = graph;
    minimal.tensors.at(minimal.conv.x_uid).dim = {1, 1, 1, 1};
    minimal.tensors.at(minimal.conv.w_uid).dim = {1, 1, 1, 1};
    minimal.tensors.at(minimal.conv.y_uid).dim = {1, 1, 1, 1};
    minimal.tensors.at(minimal.conv.x_uid).stride = {1, 1, 1, 1};
    minimal.tensors.at(minimal.conv.w_uid).stride = {1, 1, 1, 1};
    minimal.tensors.at(minimal.conv.y_uid).stride = {1, 1, 1, 1};
    minimal.conv.pre_padding = {0, 0};
    minimal.conv.post_padding = {0, 0};
    status = import_conv2d(context, minimal, module);
    tests.expect_good(status, "minimal 1x1 Conv2D import");
    if (module) {
        tests.expect_good(verify_conv2d_module(*module, minimal),
                          "minimal 1x1 Conv2D verifier");
    }

    deepforge::compiler::Conv2DImportOptions custom_options;
    custom_options.function_name = "deepforge_conv2d_custom";
    status = import_conv2d(context, graph, module, custom_options);
    tests.expect_good(status, "custom function name import");
    if (module) {
        tests.expect_good(verify_conv2d_module(*module, graph,
                                                custom_options.function_name),
                          "custom function name verifier");
    }

    auto wrong_layout = graph;
    wrong_layout.tensors.at(graph.conv.w_uid).stride[0] = 1;
    auto const preserved_metadata = metadata;
    status = import_conv2d(context, wrong_layout, module, {}, &metadata);
    tests.expect_code(status, ErrorCode::kInvalidLayout,
                      "wrong KRSC layout is rejected before IR construction");
    tests.check(module && module->lookupSymbol<mlir::func::FuncOp>(
                              custom_options.function_name),
                "failed import leaves the previous module unchanged");
    tests.check(metadata == preserved_metadata,
                "failed import leaves previous metadata unchanged");

    auto wrong_shape = graph;
    wrong_shape.tensors.at(graph.conv.y_uid).dim[2] = 4;
    mlir::OwningOpRef<mlir::ModuleOp> rejected;
    status = import_conv2d(context, wrong_shape, rejected);
    tests.expect_code(status, ErrorCode::kInvalidShape,
                      "wrong serialized output shape is rejected");

    auto dynamic_model = graph;
    dynamic_model.context.is_dynamic_shape_enabled = true;
    status = import_conv2d(context, dynamic_model, rejected);
    tests.expect_code(status, ErrorCode::kInvalidShape,
                      "dynamic shape model is rejected");

    auto parsed_bad = mlir::parseSourceString<mlir::ModuleOp>(
        "module {\n"
        "  func.func @deepforge_conv2d(%x: tensor<?x3x5x5xf32>, "
        "%w: tensor<5x3x3x3xf32>, %y: tensor<1x5x5x5xf32>) -> "
        "tensor<1x5x5x5xf32> {\n"
        "    func.return %y : tensor<1x5x5x5xf32>\n"
        "  }\n"
        "}\n",
        parser_config,
        "dynamic-negative.mlir");
    tests.check(static_cast<bool>(parsed_bad),
                "negative dynamic module is syntactically parseable");
    if (parsed_bad) {
        tests.expect_code(verify_conv2d_module(*parsed_bad, graph),
                          ErrorCode::kInvalidShape,
                          "independent verifier rejects dynamic module");
    }

    std::string wrong_filter_text = printed;
    std::string const correct_filter = "tensor<5x3x3x3xf32>";
    std::string const wrong_filter = "tensor<3x3x3x5xf32>";
    std::size_t filter_position = 0;
    while ((filter_position = wrong_filter_text.find(correct_filter,
                                                     filter_position)) !=
           std::string::npos) {
        wrong_filter_text.replace(filter_position, correct_filter.size(),
                                  wrong_filter);
        filter_position += wrong_filter.size();
    }
    mlir::ParserConfig unverified_parser_config(&context, false);
    auto wrong_filter_module = mlir::parseSourceString<mlir::ModuleOp>(
        wrong_filter_text, unverified_parser_config, "wrong-filter-order.mlir");
    tests.check(static_cast<bool>(wrong_filter_module),
                "wrong HWCF filter module parses before verification");
    if (wrong_filter_module) {
        mlir::ScopedDiagnosticHandler suppress_diagnostic(
            &context, [](mlir::Diagnostic&) {});
        tests.expect_code(verify_conv2d_module(*wrong_filter_module, graph),
                          ErrorCode::kInvalidValue,
                          "wrong HWCF filter order is rejected");
    }

    return tests.finish();
}
