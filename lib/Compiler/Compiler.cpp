#include "DeepForge/Compiler/Codegen.h"

#include "DeepForge/Compiler/Lowering.h"
#include "DeepForge/Import/SerializedGraphImporter.h"
#include "../Runtime/RuntimeInternal.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Target/LLVM/ModuleToObject.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Error.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace deepforge::compiler {
namespace {

using deepforge::import::ErrorCode;
using deepforge::import::Status;

struct TargetSpec {
    char const* cpu = "x86-64";
    char const* features = "";
    char const* feature_name = "baseline";
};

TargetSpec target_spec(runtime::CpuVariant variant) {
    switch (variant) {
        case runtime::CpuVariant::kScalar:
            return {"x86-64", "", "baseline"};
        case runtime::CpuVariant::kAvx2:
            return {"x86-64", "+avx2,+fma", "avx2,fma"};
        case runtime::CpuVariant::kAvx512:
            return {"x86-64", "+avx512f,+fma", "avx512f,fma"};
    }
    return {};
}

Status fail(ErrorCode code, std::string_view subject, std::string detail) {
    std::string message(deepforge::import::error_code_name(code));
    if (!subject.empty()) {
        message += ": ";
        message += subject;
    }
    if (!detail.empty()) {
        message += ": ";
        message += std::move(detail);
    }
    return Status::failure(code, std::move(message));
}

Status compiler_error(std::string subject, std::string detail) {
    return fail(ErrorCode::kInvalidValue, subject, std::move(detail));
}

std::string print_module(mlir::ModuleOp module) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    module.print(stream);
    stream.flush();
    return text;
}

std::string variant_symbol(std::string_view base,
                           runtime::CpuVariant variant) {
    std::string symbol(base);
    symbol += '_';
    symbol += runtime::cpu_variant_name(variant);
    return symbol;
}

mlir::DialectRegistry make_registry() {
    mlir::DialectRegistry registry;
    registry.insert<mlir::affine::AffineDialect,
                    mlir::arith::ArithDialect,
                    mlir::bufferization::BufferizationDialect,
                    mlir::cf::ControlFlowDialect,
                    mlir::func::FuncDialect,
                    mlir::index::IndexDialect,
                    mlir::linalg::LinalgDialect,
                    mlir::LLVM::LLVMDialect,
                    mlir::memref::MemRefDialect,
                    mlir::scf::SCFDialect,
                    mlir::tensor::TensorDialect,
                    mlir::vector::VectorDialect>();
    mlir::registerBuiltinDialectTranslation(registry);
    mlir::registerLLVMDialectTranslation(registry);
    return registry;
}

llvm::Expected<std::unique_ptr<llvm::TargetMachine>> make_target_machine(
    runtime::CpuVariant variant) {
    auto triple_string = llvm::sys::getDefaultTargetTriple();
    llvm::Triple triple(triple_string);
    if (triple.getArch() != llvm::Triple::x86_64) {
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "DeepForge MVP requires an x86-64 target");
    }
    std::string error;
    auto* target = llvm::TargetRegistry::lookupTarget(triple, error);
    if (!target) {
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "target lookup failed: " + error);
    }
    auto spec = target_spec(variant);
    llvm::TargetOptions options;
    auto* machine = target->createTargetMachine(
        triple, spec.cpu, spec.features, options, llvm::Reloc::PIC_,
        llvm::CodeModel::Small, llvm::CodeGenOptLevel::Default, true);
    if (!machine) {
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "target machine creation failed");
    }
    return std::unique_ptr<llvm::TargetMachine>(machine);
}

Status translate_to_llvm(mlir::ModuleOp module,
                         runtime::CpuVariant variant,
                         std::string& llvm_ir,
                         std::vector<std::uint8_t>& object) {
    llvm::LLVMContext llvm_context;
    auto llvm_module = mlir::translateModuleToLLVMIR(module, llvm_context);
    if (!llvm_module) {
        return compiler_error("llvm.translation", "MLIR to LLVM IR failed");
    }

    auto target_machine_or_error = make_target_machine(variant);
    if (!target_machine_or_error) {
        return compiler_error("llvm.target",
                              llvm::toString(target_machine_or_error.takeError()));
    }
    auto target_machine = std::move(*target_machine_or_error);
    llvm_module->setTargetTriple(target_machine->getTargetTriple());
    llvm_module->setDataLayout(target_machine->createDataLayout());

    llvm::raw_string_ostream ir_stream(llvm_ir);
    ir_stream << *llvm_module;
    ir_stream.flush();

    llvm::SmallVector<char, 0> bytes;
    llvm::raw_svector_ostream object_stream(bytes);
    llvm::legacy::PassManager codegen;
    if (target_machine->addPassesToEmitFile(
            codegen, object_stream, nullptr, llvm::CodeGenFileType::ObjectFile)) {
        return compiler_error("llvm.object", "target cannot emit an object file");
    }
    codegen.run(*llvm_module);
    object.assign(reinterpret_cast<std::uint8_t const*>(bytes.data()),
                  reinterpret_cast<std::uint8_t const*>(bytes.data()) +
                      bytes.size());
    return Status::ok();
}

Status create_engine(mlir::ModuleOp module, runtime::CpuVariant variant,
                     std::unique_ptr<mlir::ExecutionEngine>& output) {
    auto target_machine_or_error = make_target_machine(variant);
    if (!target_machine_or_error) {
        return compiler_error("runtime.jit",
                              llvm::toString(target_machine_or_error.takeError()));
    }
    mlir::ExecutionEngineOptions engine_options;
    engine_options.enableObjectDump = true;
    auto engine_or_error = mlir::ExecutionEngine::create(
        module, engine_options, std::move(*target_machine_or_error));
    if (!engine_or_error) {
        return compiler_error("runtime.jit",
                              llvm::toString(engine_or_error.takeError()));
    }
    output = std::move(*engine_or_error);
    return Status::ok();
}

}  // namespace

Status compile_graph(import::SerializedGraph const& graph,
                     CompileOptions const& options, CompilationResult& output) {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    CompilationResult result;
    auto registry = make_registry();
    mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
    context.loadAllAvailableDialects();

    mlir::OwningOpRef<mlir::ModuleOp> imported;
    Conv2DCompileMetadata metadata;
    auto status = import_conv2d(context, graph, imported, options.importer,
                                 &metadata);
    if (status.is_bad()) {
        return status;
    }
    if (options.capture_mlir) {
        result.imported_mlir = print_module(*imported);
    }
    BufferizationResult bufferized;
    status = bufferize_and_plan_conv2d(*imported, metadata, bufferized);
    if (status.is_bad()) {
        return status;
    }
    if (options.capture_mlir) {
        result.bufferized_mlir = print_module(*bufferized.module);
    }

    std::vector<std::unique_ptr<mlir::ExecutionEngine>> engines(3);
    std::array<std::string, 3> symbols;
    std::array<runtime::CpuVariant, 3> variants{
        runtime::CpuVariant::kScalar, runtime::CpuVariant::kAvx2,
        runtime::CpuVariant::kAvx512};
    for (std::size_t index = 0; index < variants.size(); ++index) {
        auto variant = variants[index];
        if (index != 0 && !options.build_avx_variants) {
            continue;
        }
        auto cloned_module = bufferized.module->clone();
        if (!cloned_module) {
            return compiler_error("compiler.clone", "failed to clone planned module");
        }
        mlir::OwningOpRef<mlir::ModuleOp> variant_module(cloned_module);
        Conv2DCompileMetadata variant_metadata = metadata;
        variant_metadata.function_name =
            variant_symbol(metadata.function_name, variant);
        auto function = variant_module->lookupSymbol<mlir::func::FuncOp>(
            metadata.function_name);
        if (!function) {
            return compiler_error("compiler.symbol",
                                  "planned Conv2D function is absent");
        }
        function.setSymName(variant_metadata.function_name);
        status = lower_conv2d_variant(*variant_module, variant_metadata, variant);
        if (status.is_bad()) {
            return status;
        }
        status = lower_to_llvm(*variant_module, variant_metadata, variant);
        if (status.is_bad()) {
            return status;
        }

        auto& code = result.variants[index];
        code.variant = variant;
        code.symbol = variant_metadata.function_name;
        symbols[index] = code.symbol;
        code.required_features = target_spec(variant).feature_name;
        if (options.capture_mlir) {
            code.mlir = print_module(*variant_module);
        }
        if (options.emit_llvm_ir || options.emit_object) {
            status = translate_to_llvm(*variant_module, variant, code.llvm_ir,
                                       code.object);
            if (status.is_bad()) {
                return status;
            }
            if (!options.emit_llvm_ir) {
                code.llvm_ir.clear();
            }
            if (!options.emit_object) {
                code.object.clear();
            }
        }
        status = create_engine(*variant_module, variant, engines[index]);
        if (status.is_bad()) {
            return status;
        }
    }

    status = runtime::create_executable(metadata, bufferized.workspace,
                                        std::move(engines), std::move(symbols),
                                        result.executable);
    if (status.is_bad()) {
        return status;
    }
    result.metadata = metadata;
    result.workspace = bufferized.workspace;
    result.target_triple = llvm::sys::getDefaultTargetTriple();
    output = std::move(result);
    return Status::ok();
}

Status compile_file(std::filesystem::path const& path,
                    CompileOptions const& options, CompilationResult& output) {
    import::SerializedGraph graph;
    import::SerializedGraphImporter importer;
    auto status = importer.parse_file(path, options.input_format, graph);
    if (status.is_bad()) {
        return status;
    }
    return compile_graph(graph, options, output);
}

}  // namespace deepforge::compiler
