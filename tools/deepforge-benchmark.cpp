#include "DeepForge/Compiler/Codegen.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Profile {
    std::string_view name;
    std::int64_t n;
    std::int64_t h;
    std::int64_t w;
    std::int64_t c;
    std::int64_t k;
    std::int64_t r;
    std::int64_t s;
    std::int64_t pad_h;
    std::int64_t pad_w;
};

constexpr std::array<Profile, 3> kProfiles{{
    {"small", 1, 8, 8, 8, 8, 3, 3, 1, 1},
    {"medium", 1, 32, 32, 32, 32, 3, 3, 1, 1},
    {"large", 1, 56, 56, 64, 64, 3, 3, 1, 1},
}};

struct Options {
    std::string_view profile = "all";
    int iterations = 3;
};

struct AlignedBytes {
    explicit AlignedBytes(std::size_t size) : size(size) {
        if (size != 0) {
            pointer = std::aligned_alloc(64, size);
        }
    }
    ~AlignedBytes() { std::free(pointer); }
    void* pointer = nullptr;
    std::size_t size = 0;
};

void usage(std::ostream& stream) {
    stream << "usage: deepforge-benchmark "
              "[--profile=small|medium|large|all] [--iterations=N]\n";
}

bool parse_options(int argc, char** argv, Options& options) {
    for (int index = 1; index < argc; ++index) {
        std::string_view argument(argv[index]);
        if (argument.starts_with("--profile=")) {
            options.profile = argument.substr(std::string_view("--profile=").size());
            if (options.profile != "small" && options.profile != "medium" &&
                options.profile != "large" && options.profile != "all") {
                std::cerr << "invalid profile: " << options.profile << '\n';
                return false;
            }
            continue;
        }
        if (argument.starts_with("--iterations=")) {
            auto value = argument.substr(std::string_view("--iterations=").size());
            int parsed = 0;
            auto result = std::from_chars(value.data(), value.data() + value.size(),
                                          parsed);
            if (result.ec != std::errc{} || result.ptr != value.data() + value.size() ||
                parsed <= 0 || parsed > 1000) {
                std::cerr << "iterations must be in [1, 1000]\n";
                return false;
            }
            options.iterations = parsed;
            continue;
        }
        std::cerr << "unknown option: " << argument << '\n';
        return false;
    }
    return true;
}

std::int64_t output_extent(std::int64_t input, std::int64_t filter,
                           std::int64_t padding) {
    return input + 2 * padding - filter + 1;
}

deepforge::import::SerializedGraph make_graph(Profile profile) {
    auto output_h = output_extent(profile.h, profile.r, profile.pad_h);
    auto output_w = output_extent(profile.w, profile.s, profile.pad_w);
    deepforge::import::SerializedGraph graph;
    graph.json_version = "1.0";
    graph.cudnn_frontend_version = 12400;
    graph.graph_uid = static_cast<std::uint64_t>(profile.h * 1000000 +
                                                 profile.c * 1000 + profile.k);
    graph.context.name = std::string("benchmark_") + std::string(profile.name);
    graph.context.compute_data_type = deepforge::import::DataType::kFloat32;
    graph.context.intermediate_data_type =
        deepforge::import::DataType::kFloat32;
    graph.context.io_data_type = deepforge::import::DataType::kFloat32;
    graph.context.sm_count = -1;
    graph.context.is_dynamic_shape_enabled = false;
    graph.context.is_override_shape_enabled = false;

    deepforge::import::TensorDesc x;
    x.name = "X";
    x.uid = 201;
    x.dim = {profile.n, profile.c, profile.h, profile.w};
    x.stride = {profile.h * profile.w * profile.c, 1,
                profile.w * profile.c, profile.c};
    deepforge::import::TensorDesc weight;
    weight.name = "W";
    weight.uid = 202;
    weight.dim = {profile.k, profile.c, profile.r, profile.s};
    weight.stride = {profile.r * profile.s * profile.c, 1,
                     profile.s * profile.c, profile.c};
    deepforge::import::TensorDesc y;
    y.name = "Y";
    y.uid = 203;
    y.dim = {profile.n, profile.k, output_h, output_w};
    y.stride = {output_h * output_w * profile.k, 1,
                output_w * profile.k, profile.k};
    graph.tensors.emplace(x.uid, x);
    graph.tensors.emplace(weight.uid, weight);
    graph.tensors.emplace(y.uid, y);

    graph.conv.name = "conv_fprop";
    graph.conv.x_uid = x.uid;
    graph.conv.w_uid = weight.uid;
    graph.conv.y_uid = y.uid;
    graph.conv.pre_padding = {profile.pad_h, profile.pad_w};
    graph.conv.post_padding = {profile.pad_h, profile.pad_w};
    graph.conv.stride = {1, 1};
    graph.conv.dilation = {1, 1};
    return graph;
}

std::size_t elements(std::array<std::int64_t, 4> const& shape) {
    std::size_t count = 1;
    for (auto dimension : shape) {
        count *= static_cast<std::size_t>(dimension);
    }
    return count;
}

deepforge::import::Status run_variant(
    deepforge::compiler::CompilationResult const& compilation,
    deepforge::runtime::CpuVariant variant, std::vector<float>& x,
    std::vector<float>& weight, std::vector<float>& y, void* workspace) {
    deepforge::runtime::VariantPack pack{
        {compilation.metadata.x_uid, x.data()},
        {compilation.metadata.w_uid, weight.data()},
        {compilation.metadata.y_uid, y.data()}};
    return compilation.executable->execute_variant(variant, nullptr, pack,
                                                   workspace);
}

bool compare(std::vector<float> const& actual,
             std::vector<float> const& reference,
             double& max_absolute, double& max_relative) {
    max_absolute = 0.0;
    max_relative = 0.0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (!std::isfinite(actual[index]) || !std::isfinite(reference[index])) {
            return false;
        }
        auto absolute = std::fabs(static_cast<double>(actual[index]) -
                                  reference[index]);
        auto relative = absolute /
                        std::max(std::fabs(static_cast<double>(reference[index])),
                                 1.0e-30);
        max_absolute = std::max(max_absolute, absolute);
        max_relative = std::max(max_relative, relative);
        if (absolute > 1.0e-4 + 1.0e-3 * std::fabs(reference[index])) {
            return false;
        }
    }
    return true;
}

int benchmark(Profile profile, int iterations) {
    deepforge::compiler::CompileOptions options;
    options.emit_object = false;
    options.emit_llvm_ir = false;
    deepforge::compiler::CompilationResult compilation;
    auto compile_start = std::chrono::steady_clock::now();
    auto status = deepforge::compiler::compile_graph(make_graph(profile), options,
                                                     compilation);
    auto compile_end = std::chrono::steady_clock::now();
    if (status.is_bad()) {
        std::cerr << status.message() << '\n';
        return 1;
    }

    std::vector<float> x(elements(compilation.metadata.x_shape));
    std::vector<float> weight(elements(compilation.metadata.w_shape));
    std::vector<float> y(elements(compilation.metadata.y_shape));
    std::vector<float> scalar_reference(y.size());
    for (std::size_t index = 0; index < x.size(); ++index) {
        x[index] = static_cast<float>(
                       (static_cast<int>(index * 13) % 37) - 18) /
                   17.0F;
    }
    for (std::size_t index = 0; index < weight.size(); ++index) {
        weight[index] = static_cast<float>(
                            (static_cast<int>(index * 11) % 31) - 15) /
                        19.0F;
    }
    auto workspace_size = compilation.executable->get_workspace_size();
    if (workspace_size < 0 || workspace_size % 64 != 0) {
        std::cerr << "invalid workspace size\n";
        return 1;
    }
    AlignedBytes workspace(static_cast<std::size_t>(workspace_size));
    if (workspace.size != 0 && workspace.pointer == nullptr) {
        std::cerr << "workspace allocation failed\n";
        return 1;
    }

    status = run_variant(compilation, deepforge::runtime::CpuVariant::kScalar,
                         x, weight, scalar_reference, workspace.pointer);
    if (status.is_bad()) {
        std::cerr << status.message() << '\n';
        return 1;
    }

    auto compile_ms = std::chrono::duration<double, std::milli>(compile_end -
                                                                compile_start)
                          .count();
    auto output_h = output_extent(profile.h, profile.r, profile.pad_h);
    auto output_w = output_extent(profile.w, profile.s, profile.pad_w);
    double operations = 2.0 * profile.n * output_h * output_w * profile.k *
                        profile.r * profile.s * profile.c;
    std::array<deepforge::runtime::CpuVariant, 3> variants{
        deepforge::runtime::CpuVariant::kScalar,
        deepforge::runtime::CpuVariant::kAvx2,
        deepforge::runtime::CpuVariant::kAvx512};
    for (auto variant : variants) {
        if (!compilation.executable->supports_variant(variant)) {
            continue;
        }
        status = run_variant(compilation, variant, x, weight, y,
                             workspace.pointer);
        if (status.is_bad()) {
            std::cerr << status.message() << '\n';
            return 1;
        }
        double max_absolute = 0.0;
        double max_relative = 0.0;
        if (!compare(y, scalar_reference, max_absolute, max_relative)) {
            std::cerr << "numeric mismatch for "
                      << deepforge::runtime::cpu_variant_name(variant) << '\n';
            return 1;
        }

        auto start = std::chrono::steady_clock::now();
        for (int iteration = 0; iteration < iterations; ++iteration) {
            status = run_variant(compilation, variant, x, weight, y,
                                 workspace.pointer);
            if (status.is_bad()) {
                std::cerr << status.message() << '\n';
                return 1;
            }
        }
        auto end = std::chrono::steady_clock::now();
        auto milliseconds =
            std::chrono::duration<double, std::milli>(end - start).count() /
            iterations;
        auto gflops = operations / (milliseconds * 1.0e6);
        volatile float checksum =
            std::accumulate(y.begin(), y.end(), 0.0F);
        (void)checksum;
        std::cout << profile.name << ','
                  << deepforge::runtime::cpu_variant_name(variant) << ','
                  << profile.n << 'x' << profile.h << 'x' << profile.w << 'x'
                  << profile.c << "_k" << profile.k << "_r" << profile.r << 'x'
                  << profile.s << ',' << compilation.workspace.size_bytes << ','
                  << iterations << ',' << std::fixed << std::setprecision(3)
                  << compile_ms << ',' << milliseconds << ',' << gflops << ','
                  << std::scientific << max_absolute << ',' << max_relative
                  << '\n';
    }
    return 0;
}

}  // namespace

int run(int argc, char** argv) {
    if (argc == 2 && (std::string_view(argv[1]) == "--help" ||
                      std::string_view(argv[1]) == "-h")) {
        usage(std::cout);
        return 0;
    }
    Options options;
    if (!parse_options(argc, argv, options)) {
        usage(std::cerr);
        return 2;
    }
    std::cout << "profile,variant,shape,workspace_bytes,iterations,compile_ms,"
                 "execute_ms,gflops,max_abs,max_rel\n";
    for (auto profile : kProfiles) {
        if (options.profile == "all" || options.profile == profile.name) {
            if (benchmark(profile, options.iterations) != 0) {
                return 1;
            }
        }
    }
    return 0;
}

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (std::exception const& exception) {
        std::cerr << "deepforge-benchmark: " << exception.what() << '\n';
        return 1;
    }
}
