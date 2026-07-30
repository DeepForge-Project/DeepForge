#include "DeepForge/Compiler/Bufferization.h"
#include "DeepForge/Compiler/Conv2DImporter.h"
#include "DeepForge/Import/SerializedGraphImporter.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Verifier.h"

#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

using deepforge::compiler::BufferizationResult;
using deepforge::compiler::WorkspacePlan;
using deepforge::compiler::WorkspaceRequest;
using deepforge::compiler::bufferize_and_plan_conv2d;
using deepforge::compiler::import_conv2d;
using deepforge::compiler::plan_workspace;
using deepforge::compiler::verify_bufferized_conv2d;
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
            std::cout << "deepforge-bufferization: " << checks_
                      << " checks passed\n";
            return 0;
        }
        std::cerr << "deepforge-bufferization: " << failures_ << " of "
                  << checks_ << " checks failed\n";
        return 1;
    }

private:
    int checks_ = 0;
    int failures_ = 0;
};

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: deepforge_bufferization_test <fixture.json>\n";
        return 2;
    }
    TestRunner tests;

    std::vector<WorkspaceRequest> requests{
        {"a", 64, 64, 0, 3},
        {"b", 64, 64, 4, 7},
        {"c", 96, 32, 2, 5},
    };
    WorkspacePlan plan;
    auto status = plan_workspace(requests, plan);
    tests.expect_good(status, "lifetime-aware workspace plan");
    tests.check(plan.allocations.size() == 3,
                "workspace plan covers every request");
    if (plan.allocations.size() == 3) {
        tests.check(plan.allocations[0].offset == 0 &&
                        plan.allocations[1].offset == 64 &&
                        plan.allocations[2].offset == 0,
                    "non-overlapping lifetimes reuse offset zero");
    }
    tests.check(plan.size_bytes == 192 && plan.alignment == 64,
                "workspace high watermark is rounded to 64 bytes");
    auto const valid_plan = plan;

    auto invalid_alignment = requests;
    invalid_alignment[0].alignment = 3;
    tests.expect_code(plan_workspace(invalid_alignment, plan),
                      ErrorCode::kInvalidArgument,
                      "non-power-of-two alignment is rejected");
    tests.check(plan == valid_plan,
                "invalid alignment leaves workspace output unchanged");
    auto duplicate_name = requests;
    duplicate_name[1].name = "a";
    tests.expect_code(plan_workspace(duplicate_name, plan),
                      ErrorCode::kInvalidValue,
                      "duplicate allocation name is rejected");
    tests.check(plan == valid_plan,
                "duplicate name leaves workspace output unchanged");
    std::vector<WorkspaceRequest> overflow{
        {"overflow", static_cast<std::uint64_t>(INT64_MAX), 64, 0, 1},
    };
    tests.expect_code(plan_workspace(overflow, plan),
                      ErrorCode::kDimensionOverflow,
                      "workspace size overflow is rejected");
    tests.check(plan == valid_plan,
                "workspace overflow leaves output unchanged");

    SerializedGraph graph;
    SerializedGraphImporter serialized_importer;
    status = serialized_importer.parse_file(std::filesystem::path(argv[1]),
                                             InputFormat::kJson, graph);
    tests.expect_good(status, "canonical graph import");
    if (status.is_bad()) {
        return tests.finish();
    }

    mlir::DialectRegistry registry;
    registry.insert<mlir::arith::ArithDialect,
                    mlir::bufferization::BufferizationDialect,
                    mlir::func::FuncDialect,
                    mlir::linalg::LinalgDialect,
                    mlir::memref::MemRefDialect,
                    mlir::tensor::TensorDialect>();
    mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
    context.loadAllAvailableDialects();

    mlir::OwningOpRef<mlir::ModuleOp> imported;
    deepforge::compiler::Conv2DCompileMetadata metadata;
    status = import_conv2d(context, graph, imported, {}, &metadata);
    tests.expect_good(status, "P2 module import");
    if (!imported) {
        return tests.finish();
    }

    BufferizationResult bufferized;
    status = bufferize_and_plan_conv2d(*imported, metadata, bufferized);
    tests.expect_good(status, "P3 bufferization and workspace planning");
    tests.check(static_cast<bool>(bufferized.module),
                "P3 returns a bufferized module");
    tests.check(bufferized.one_shot_bufferize_runs == 1,
                "One-Shot Bufferize runs exactly once");
    tests.check(bufferized.workspace.allocations.size() == 1,
                "padded input is the only workspace allocation");
    tests.check(bufferized.workspace.size_bytes == 640,
                "padded 588-byte input rounds to 640-byte workspace");
    if (bufferized.workspace.allocations.size() == 1) {
        tests.check(bufferized.workspace.allocations.front().offset == 0 &&
                        bufferized.workspace.allocations.front().size_bytes ==
                            588,
                    "padded input allocation has checked size and offset");
    }
    if (bufferized.module) {
        tests.check(!llvm::failed(mlir::verify(*bufferized.module)),
                    "standard MLIR verifier accepts P3 module");
        tests.expect_good(verify_bufferized_conv2d(
                              *bufferized.module, metadata,
                              bufferized.workspace),
                          "independent P3 verifier");

        std::string text;
        llvm::raw_string_ostream stream(text);
        bufferized.module->print(stream);
        stream.flush();
        tests.check(text.find("memref<640xi8>") != std::string::npos &&
                        text.find("memref.view") != std::string::npos,
                    "workspace is a flat argument with a typed view");
        tests.check(text.find("memref.alloc") == std::string::npos &&
                        text.find("memref.dealloc") == std::string::npos &&
                        text.find("bufferization.") == std::string::npos &&
                        text.find("tensor.") == std::string::npos,
                    "P3 module has no owned or tensor allocation residue");
        tests.check(text.find("linalg.conv_2d_nhwc_fhwc") !=
                        std::string::npos,
                    "P3 preserves named Conv2D for direct lowering");

        auto malformed_module = bufferized.module->clone();
        tests.check(static_cast<bool>(malformed_module),
                    "bufferized module can be cloned for negative checks");
        if (malformed_module) {
            mlir::OwningOpRef<mlir::ModuleOp> malformed(malformed_module);
            bool changed_offset = false;
            malformed->walk([&](mlir::memref::ViewOp view) {
                auto byte_shift = view.getByteShift().getDefiningOp<
                    mlir::arith::ConstantIndexOp>();
                if (byte_shift && !changed_offset) {
                    byte_shift.setValueAttr(mlir::IntegerAttr::get(
                        mlir::IndexType::get(&context), 64));
                    changed_offset = true;
                }
            });
            tests.check(changed_offset,
                        "negative module contains a workspace byte shift");
            tests.expect_code(verify_bufferized_conv2d(
                                  *malformed, metadata,
                                  bufferized.workspace),
                              ErrorCode::kInvalidValue,
                              "verifier rejects a mismatched workspace view");
        }
    }

    BufferizationResult rejected = std::move(bufferized);
    auto const preserved_workspace = rejected.workspace;
    status = bufferize_and_plan_conv2d(nullptr, metadata, rejected);
    tests.expect_code(status, ErrorCode::kInvalidArgument,
                      "null module is rejected");
    tests.check(rejected.module &&
                    rejected.workspace == preserved_workspace &&
                    rejected.one_shot_bufferize_runs == 1,
                "failed P3 transform leaves previous output unchanged");

    return tests.finish();
}
