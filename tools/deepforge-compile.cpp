#include "DeepForge/Compiler/Artifact.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <unistd.h>

namespace {

enum class EmitKind {
    kArtifact,
    kLlvmIr,
};

enum class DumpStage {
    kImported,
    kBufferized,
    kLlvm,
};

struct DumpRequest {
    DumpStage stage = DumpStage::kImported;
    std::filesystem::path path;
};

struct Options {
    std::filesystem::path input;
    std::filesystem::path output;
    deepforge::import::InputFormat input_format =
        deepforge::import::InputFormat::kAuto;
    deepforge::runtime::CpuVariant variant =
        deepforge::runtime::CpuVariant::kScalar;
    EmitKind emit = EmitKind::kArtifact;
    std::vector<DumpRequest> dumps;
    std::optional<std::filesystem::path> inspect;
};

void print_usage(std::ostream& stream) {
    stream
        << "usage: deepforge-compile <graph.json|graph.ubjson> [options]\n"
        << "       deepforge-compile --inspect <artifact.dfo>\n\n"
        << "options:\n"
        << "  --input-format=auto|json|ubjson\n"
        << "  --target=x86-64\n"
        << "  --emit=object|llvm-ir\n"
        << "  --variant=scalar|avx2|avx512   LLVM IR output/dump variant\n"
        << "  --dump-ir=STAGE:PATH           STAGE is imported, bufferized, or llvm\n"
        << "  -o PATH, --output=PATH\n"
        << "  --help\n"
        << "  --version\n";
}

std::optional<std::string_view> inline_value(std::string_view argument,
                                             std::string_view option) {
    if (argument.size() > option.size() && argument.starts_with(option) &&
        argument[option.size()] == '=') {
        return argument.substr(option.size() + 1);
    }
    return std::nullopt;
}

bool parse_input_format(std::string_view value,
                        deepforge::import::InputFormat& output) {
    if (value == "auto") {
        output = deepforge::import::InputFormat::kAuto;
        return true;
    }
    if (value == "json") {
        output = deepforge::import::InputFormat::kJson;
        return true;
    }
    if (value == "ubjson") {
        output = deepforge::import::InputFormat::kUbjson;
        return true;
    }
    return false;
}

bool parse_variant(std::string_view value,
                   deepforge::runtime::CpuVariant& output) {
    if (value == "scalar") {
        output = deepforge::runtime::CpuVariant::kScalar;
        return true;
    }
    if (value == "avx2") {
        output = deepforge::runtime::CpuVariant::kAvx2;
        return true;
    }
    if (value == "avx512" || value == "avx-512") {
        output = deepforge::runtime::CpuVariant::kAvx512;
        return true;
    }
    return false;
}

bool infer_dump_stage(std::string_view value, DumpStage& stage,
                      std::filesystem::path& path) {
    auto separator = value.find(':');
    std::string_view stage_name;
    std::string_view path_name;
    if (separator != std::string_view::npos) {
        stage_name = value.substr(0, separator);
        path_name = value.substr(separator + 1);
    } else {
        path_name = value;
        auto filename = std::filesystem::path(value).filename().string();
        if (filename.find("imported") != std::string::npos) {
            stage_name = "imported";
        } else if (filename.find("bufferized") != std::string::npos) {
            stage_name = "bufferized";
        } else if (filename.find("llvm") != std::string::npos) {
            stage_name = "llvm";
        }
    }
    if (stage_name == "imported") {
        stage = DumpStage::kImported;
    } else if (stage_name == "bufferized") {
        stage = DumpStage::kBufferized;
    } else if (stage_name == "llvm") {
        stage = DumpStage::kLlvm;
    } else {
        return false;
    }
    path = std::filesystem::path(path_name);
    return !path.empty();
}

std::optional<std::string_view> next_value(int& index, int argc, char** argv,
                                           std::string_view option,
                                           std::string& error) {
    if (index + 1 >= argc) {
        error = std::string(option) + " requires a value";
        return std::nullopt;
    }
    ++index;
    return std::string_view(argv[index]);
}

bool parse_options(int argc, char** argv, Options& output,
                   std::string& error, bool& handled) {
    handled = false;
    bool compilation_option_seen = false;
    for (int index = 1; index < argc; ++index) {
        std::string_view argument(argv[index]);
        if (argument == "--help" || argument == "-h") {
            print_usage(std::cout);
            handled = true;
            return true;
        }
        if (argument == "--version") {
            std::cout << "DeepForge 0.1.0 (LLVM/MLIR 22.1.8, "
                         "cuDNN Frontend serialization 1.24.0)\n";
            handled = true;
            return true;
        }

        if (argument == "-o") {
            compilation_option_seen = true;
            auto value = next_value(index, argc, argv, argument, error);
            if (!value) {
                return false;
            }
            if (value->empty()) {
                error = "output path must not be empty";
                return false;
            }
            output.output = *value;
            continue;
        }
        if (argument == "--input-format") {
            compilation_option_seen = true;
            auto value = next_value(index, argc, argv, argument, error);
            if (!value || !parse_input_format(*value, output.input_format)) {
                error = "--input-format must be auto, json, or ubjson";
                return false;
            }
            continue;
        }
        if (argument == "--target") {
            compilation_option_seen = true;
            auto value = next_value(index, argc, argv, argument, error);
            if (!value || *value != "x86-64") {
                error = "--target must be x86-64 for the MVP";
                return false;
            }
            continue;
        }
        if (argument == "--emit") {
            compilation_option_seen = true;
            auto value = next_value(index, argc, argv, argument, error);
            if (!value) {
                return false;
            }
            if (*value == "object" || *value == "artifact") {
                output.emit = EmitKind::kArtifact;
            } else if (*value == "llvm-ir") {
                output.emit = EmitKind::kLlvmIr;
            } else {
                error = "--emit must be object or llvm-ir";
                return false;
            }
            continue;
        } else if (argument == "--variant") {
            compilation_option_seen = true;
            auto value = next_value(index, argc, argv, argument, error);
            if (!value || !parse_variant(*value, output.variant)) {
                error = "--variant must be scalar, avx2, or avx512";
                return false;
            }
            continue;
        } else if (argument == "--dump-ir") {
            compilation_option_seen = true;
            auto value = next_value(index, argc, argv, argument, error);
            if (!value) {
                return false;
            }
            DumpRequest request;
            if (!infer_dump_stage(*value, request.stage, request.path)) {
                error = "--dump-ir requires STAGE:PATH (imported, bufferized, llvm)";
                return false;
            }
            output.dumps.push_back(std::move(request));
            continue;
        } else if (argument == "--inspect") {
            auto value = next_value(index, argc, argv, argument, error);
            if (!value) {
                return false;
            }
            if (value->empty() || output.inspect) {
                error = "--inspect requires exactly one path";
                return false;
            }
            output.inspect = std::filesystem::path(*value);
            continue;
        }

        if (auto value = inline_value(argument, "--output")) {
            compilation_option_seen = true;
            if (value->empty()) {
                error = "output path must not be empty";
                return false;
            }
            output.output = *value;
        } else if (auto value = inline_value(argument, "--input-format")) {
            compilation_option_seen = true;
            if (!parse_input_format(*value, output.input_format)) {
                error = "--input-format must be auto, json, or ubjson";
                return false;
            }
        } else if (auto value = inline_value(argument, "--target")) {
            compilation_option_seen = true;
            if (*value != "x86-64") {
                error = "--target must be x86-64 for the MVP";
                return false;
            }
        } else if (auto value = inline_value(argument, "--emit")) {
            compilation_option_seen = true;
            if (*value == "object" || *value == "artifact") {
                output.emit = EmitKind::kArtifact;
            } else if (*value == "llvm-ir") {
                output.emit = EmitKind::kLlvmIr;
            } else {
                error = "--emit must be object or llvm-ir";
                return false;
            }
        } else if (auto value = inline_value(argument, "--variant")) {
            compilation_option_seen = true;
            if (!parse_variant(*value, output.variant)) {
                error = "--variant must be scalar, avx2, or avx512";
                return false;
            }
        } else if (auto value = inline_value(argument, "--dump-ir")) {
            compilation_option_seen = true;
            DumpRequest request;
            if (!infer_dump_stage(*value, request.stage, request.path)) {
                error = "--dump-ir requires STAGE:PATH (imported, bufferized, llvm)";
                return false;
            }
            output.dumps.push_back(std::move(request));
        } else if (auto value = inline_value(argument, "--inspect")) {
            if (value->empty() || output.inspect) {
                error = "--inspect requires exactly one path";
                return false;
            }
            output.inspect = std::filesystem::path(*value);
        } else if (argument.starts_with('-')) {
            error = "unknown option: " + std::string(argument);
            return false;
        } else if (output.input.empty()) {
            compilation_option_seen = true;
            output.input = argument;
        } else {
            error = "multiple input files are not supported";
            return false;
        }
    }

    if (output.inspect) {
        if (compilation_option_seen) {
            error = "--inspect cannot be combined with compilation options";
            return false;
        }
        return true;
    }
    if (output.input.empty()) {
        error = "an input graph is required";
        return false;
    }
    if (output.output.empty()) {
        output.output = output.input;
        output.output.replace_extension(output.emit == EmitKind::kArtifact
                                            ? ".dfo"
                                            : ".ll");
    }
    return true;
}

std::filesystem::path unique_temporary_path(
    std::filesystem::path const& path) {
    static std::atomic<std::uint64_t> sequence{0};
    auto temporary = path;
    temporary += ".tmp." +
                 std::to_string(static_cast<std::uint64_t>(::getpid())) +
                 "." +
                 std::to_string(sequence.fetch_add(1,
                                                   std::memory_order_relaxed));
    return temporary;
}

bool write_text(std::filesystem::path const& path, std::string_view text,
                std::string& error) {
    if (path.empty() || text.empty()) {
        error = "output path and content must be non-empty";
        return false;
    }
    auto const temporary = unique_temporary_path(path);
    {
        std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
        file.write(text.data(), static_cast<std::streamsize>(text.size()));
        file.flush();
        if (!file) {
            error = "cannot write " + temporary.string();
            std::error_code ignored;
            std::filesystem::remove(temporary, ignored);
            return false;
        }
    }
    std::error_code filesystem_error;
    std::filesystem::rename(temporary, path, filesystem_error);
    if (filesystem_error) {
        error = "cannot publish " + path.string() + ": " +
                filesystem_error.message();
        std::filesystem::remove(temporary, filesystem_error);
        return false;
    }
    return true;
}

std::string_view dump_text(deepforge::compiler::CompilationResult const& result,
                           DumpStage stage,
                           deepforge::runtime::CpuVariant variant) {
    switch (stage) {
        case DumpStage::kImported:
            return result.imported_mlir;
        case DumpStage::kBufferized:
            return result.bufferized_mlir;
        case DumpStage::kLlvm:
            return result.variants[static_cast<std::size_t>(variant)].mlir;
    }
    return {};
}

int inspect_artifact(std::filesystem::path const& path) {
    deepforge::compiler::ArtifactInfo artifact;
    auto status = deepforge::compiler::read_artifact(path, artifact);
    if (status.is_bad()) {
        std::cerr << status.message() << '\n';
        return 1;
    }
    std::cout << "format: DFOBJ/" << deepforge::compiler::kArtifactFormatVersion
              << "\n"
              << "deepforge: " << artifact.deepforge_version << "\n"
              << "llvm: " << artifact.llvm_version << "\n"
              << "frontend: " << artifact.frontend_version << "\n"
              << "target: " << artifact.target_triple << "\n"
              << "function: " << artifact.metadata.function_name << "\n"
              << "uids: " << artifact.metadata.x_uid << ','
              << artifact.metadata.w_uid << ',' << artifact.metadata.y_uid
              << "\n"
              << "workspace: " << artifact.workspace.size_bytes << " bytes\n";
    for (auto const& variant : artifact.variants) {
        std::cout << "variant: "
                  << deepforge::runtime::cpu_variant_name(variant.variant)
                  << " symbol=" << variant.symbol
                  << " features=" << variant.required_features
                  << " object_bytes=" << variant.object.size() << '\n';
    }
    return 0;
}

}  // namespace

int run(int argc, char** argv) {
    Options options;
    std::string error;
    bool handled = false;
    if (!parse_options(argc, argv, options, error, handled)) {
        std::cerr << "deepforge-compile: " << error << '\n';
        print_usage(std::cerr);
        return 2;
    }
    if (handled) {
        return 0;
    }
    if (options.inspect) {
        return inspect_artifact(*options.inspect);
    }

    deepforge::compiler::CompileOptions compile_options;
    compile_options.input_format = options.input_format;
    compile_options.capture_mlir = !options.dumps.empty();
    compile_options.emit_object = options.emit == EmitKind::kArtifact;
    compile_options.emit_llvm_ir = options.emit == EmitKind::kLlvmIr;
    deepforge::compiler::CompilationResult compilation;
    auto status = deepforge::compiler::compile_file(options.input,
                                                    compile_options,
                                                    compilation);
    if (status.is_bad()) {
        std::cerr << status.message() << '\n';
        return 1;
    }

    for (auto const& dump : options.dumps) {
        auto text = dump_text(compilation, dump.stage, options.variant);
        if (!write_text(dump.path, text, error)) {
            std::cerr << "deepforge-compile: " << error << '\n';
            return 1;
        }
    }

    if (options.emit == EmitKind::kArtifact) {
        status = deepforge::compiler::write_artifact(options.output,
                                                     compilation);
        if (status.is_bad()) {
            std::cerr << status.message() << '\n';
            return 1;
        }
    } else {
        auto const& llvm_ir =
            compilation.variants[static_cast<std::size_t>(options.variant)].llvm_ir;
        if (!write_text(options.output, llvm_ir, error)) {
            std::cerr << "deepforge-compile: " << error << '\n';
            return 1;
        }
    }

    std::cout << "output: " << options.output << '\n'
              << "target: " << compilation.target_triple << '\n'
              << "workspace: " << compilation.workspace.size_bytes
              << " bytes\n";
    return 0;
}

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (std::exception const& exception) {
        std::cerr << "deepforge-compile: " << exception.what() << '\n';
        return 1;
    }
}
