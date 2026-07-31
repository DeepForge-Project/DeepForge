#include "DeepForge/Compiler/Codegen.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

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
            std::cerr << "FAIL: " << name << ": " << status.message() << '\n';
        }
    }

    int finish() const {
        if (failures_ == 0) {
            std::cout << "deepforge-variant-matrix: " << checks_
                      << " checks passed\n";
            return 0;
        }
        std::cerr << "deepforge-variant-matrix: " << failures_ << " of "
                  << checks_ << " checks failed\n";
        return 1;
    }

private:
    int checks_ = 0;
    int failures_ = 0;
};

struct AlignedBytes {
    explicit AlignedBytes(std::size_t size) : size(size) {
        if (size != 0) {
            pointer = std::aligned_alloc(64, size);
        }
    }
    ~AlignedBytes() { std::free(pointer); }
    AlignedBytes(AlignedBytes const&) = delete;
    AlignedBytes& operator=(AlignedBytes const&) = delete;
    void* pointer = nullptr;
    std::size_t size = 0;
};

struct ShapeCase {
    std::int64_t channels = 1;
    std::int64_t output_channels = 1;
    std::int64_t height = 2;
    std::int64_t width = 3;
};

deepforge::import::SerializedGraph make_graph(ShapeCase shape) {
    constexpr std::int64_t n = 2;
    deepforge::import::SerializedGraph graph;
    graph.json_version = "1.0";
    graph.cudnn_frontend_version = 12400;
    graph.graph_uid = static_cast<std::uint64_t>(shape.channels * 1000 +
                                                 shape.output_channels);
    graph.context.name = "variant_matrix";
    graph.context.compute_data_type = deepforge::import::DataType::kFloat32;
    graph.context.intermediate_data_type =
        deepforge::import::DataType::kFloat32;
    graph.context.io_data_type = deepforge::import::DataType::kFloat32;
    graph.context.sm_count = -1;
    graph.context.is_dynamic_shape_enabled = false;
    graph.context.is_override_shape_enabled = false;

    deepforge::import::TensorDesc x;
    x.name = "X";
    x.uid = 11;
    x.dim = {n, shape.channels, shape.height, shape.width};
    x.stride = {shape.height * shape.width * shape.channels, 1,
                shape.width * shape.channels, shape.channels};

    deepforge::import::TensorDesc w;
    w.name = "W";
    w.uid = 12;
    w.dim = {shape.output_channels, shape.channels, 1, 1};
    w.stride = {shape.channels, 1, shape.channels, shape.channels};

    deepforge::import::TensorDesc y;
    y.name = "Y";
    y.uid = 13;
    y.dim = {n, shape.output_channels, shape.height, shape.width};
    y.stride = {shape.height * shape.width * shape.output_channels, 1,
                shape.width * shape.output_channels, shape.output_channels};
    graph.tensors.emplace(x.uid, x);
    graph.tensors.emplace(w.uid, w);
    graph.tensors.emplace(y.uid, y);

    auto& conv = graph.emplace_conv_fprop();
    conv.name = "conv_fprop";
    conv.x_uid = x.uid;
    conv.w_uid = w.uid;
    conv.y_uid = y.uid;
    conv.pre_padding = {0, 0};
    conv.post_padding = {0, 0};
    conv.stride = {1, 1};
    conv.dilation = {1, 1};
    graph.nodes.back().name = conv.name;
    return graph;
}

std::vector<float> reference(std::vector<float> const& x,
                             std::vector<float> const& w,
                             ShapeCase shape) {
    constexpr std::int64_t n_size = 2;
    std::vector<float> output(static_cast<std::size_t>(
        n_size * shape.height * shape.width * shape.output_channels));
    for (std::int64_t n = 0; n < n_size; ++n) {
        for (std::int64_t h = 0; h < shape.height; ++h) {
            for (std::int64_t column = 0; column < shape.width; ++column) {
                for (std::int64_t k = 0; k < shape.output_channels; ++k) {
                    double sum = 0.0;
                    for (std::int64_t c = 0; c < shape.channels; ++c) {
                        auto x_index =
                            ((n * shape.height + h) * shape.width + column) *
                                shape.channels +
                            c;
                        auto w_index = k * shape.channels + c;
                        sum += static_cast<double>(
                                   x[static_cast<std::size_t>(x_index)]) *
                               w[static_cast<std::size_t>(w_index)];
                    }
                    auto y_index =
                        ((n * shape.height + h) * shape.width + column) *
                            shape.output_channels +
                        k;
                    output[static_cast<std::size_t>(y_index)] =
                        static_cast<float>(sum);
                }
            }
        }
    }
    return output;
}

bool output_matches(std::vector<float> const& actual,
                    std::vector<float> const& expected) {
    for (std::size_t index = 0; index < actual.size(); ++index) {
        float tolerance = 1.0e-4F + 1.0e-3F * std::fabs(expected[index]);
        if (!std::isfinite(actual[index]) ||
            std::fabs(actual[index] - expected[index]) > tolerance) {
            return false;
        }
    }
    return true;
}

deepforge::import::Status execute(
    deepforge::compiler::CompilationResult const& compilation,
    deepforge::runtime::CpuVariant variant,
    std::vector<float>& x, std::vector<float>& w, std::vector<float>& y,
    void* workspace) {
    deepforge::runtime::VariantPack pack{
        {compilation.metadata.x_uid, x.data()},
        {compilation.metadata.w_uid, w.data()},
        {compilation.metadata.y_uid, y.data()}};
    return compilation.executable->execute_variant(variant, nullptr, pack,
                                                   workspace);
}

void test_feature_policy(TestRunner& tests) {
    using deepforge::runtime::CpuFeatures;
    using deepforge::runtime::CpuVariant;
    std::array<bool, 3> all{true, true, true};

    tests.check(deepforge::runtime::select_cpu_variant(CpuFeatures{}, all) ==
                    CpuVariant::kScalar,
                "baseline feature set selects scalar");
    CpuFeatures no_fma{true, false, true, true, true, true};
    tests.check(deepforge::runtime::select_cpu_variant(no_fma, all) ==
                    CpuVariant::kScalar,
                "AVX hardware without FMA selects scalar");
    CpuFeatures no_os_state{true, true, true, true, false, false};
    tests.check(deepforge::runtime::select_cpu_variant(no_os_state, all) ==
                    CpuVariant::kScalar,
                "missing OS SIMD state selects scalar");
    CpuFeatures avx2{true, true, true, false, true, false};
    tests.check(deepforge::runtime::select_cpu_variant(avx2, all) ==
                    CpuVariant::kAvx2,
                "AVX2 plus FMA and YMM state selects AVX2");
    CpuFeatures no_zmm{true, true, true, true, true, false};
    tests.check(deepforge::runtime::select_cpu_variant(no_zmm, all) ==
                    CpuVariant::kAvx2,
                "AVX-512 hardware without ZMM OS state falls back to AVX2");
    CpuFeatures avx512{true, true, true, true, true, true};
    tests.check(deepforge::runtime::select_cpu_variant(avx512, all) ==
                    CpuVariant::kAvx512,
                "complete AVX-512 feature set selects AVX-512");
    std::array<bool, 3> no_avx512_object{true, true, false};
    tests.check(deepforge::runtime::select_cpu_variant(
                    avx512, no_avx512_object) == CpuVariant::kAvx2,
                "missing AVX-512 object falls back to AVX2");
}

void test_special_values(TestRunner& tests,
                         deepforge::compiler::CompilationResult const& compilation,
                         ShapeCase shape, AlignedBytes& workspace) {
    std::vector<float> x(static_cast<std::size_t>(
        2 * shape.height * shape.width * shape.channels));
    std::vector<float> w(static_cast<std::size_t>(
        shape.output_channels * shape.channels));
    std::vector<float> y(static_cast<std::size_t>(
        2 * shape.height * shape.width * shape.output_channels));
    std::array<deepforge::runtime::CpuVariant, 3> variants{
        deepforge::runtime::CpuVariant::kScalar,
        deepforge::runtime::CpuVariant::kAvx2,
        deepforge::runtime::CpuVariant::kAvx512};

    for (auto variant : variants) {
        if (!compilation.executable->supports_variant(variant)) {
            continue;
        }
        std::fill(x.begin(), x.end(), 0.0F);
        std::fill(w.begin(), w.end(), 1.0F);
        auto status = execute(compilation, variant, x, w, y, workspace.pointer);
        tests.good(status, "zero classification execution");
        tests.check(std::all_of(y.begin(), y.end(),
                                [](float value) { return value == 0.0F; }),
                    "all-zero input remains zero");

        std::fill(x.begin(), x.end(),
                  std::numeric_limits<float>::quiet_NaN());
        std::fill(w.begin(), w.end(), 1.0F);
        status = execute(compilation, variant, x, w, y, workspace.pointer);
        tests.good(status, "NaN classification execution");
        tests.check(std::all_of(y.begin(), y.end(),
                                [](float value) { return std::isnan(value); }),
                    "NaN classification is preserved");

        std::fill(x.begin(), x.end(),
                  std::numeric_limits<float>::infinity());
        std::fill(w.begin(), w.end(), 1.0F);
        status = execute(compilation, variant, x, w, y, workspace.pointer);
        tests.good(status, "infinity classification execution");
        tests.check(std::all_of(y.begin(), y.end(), [](float value) {
                        return std::isinf(value) && value > 0.0F;
                    }),
                    "positive infinity classification is preserved");

        std::fill(w.begin(), w.end(), 0.0F);
        status = execute(compilation, variant, x, w, y, workspace.pointer);
        tests.good(status, "infinity times zero execution");
        tests.check(std::all_of(y.begin(), y.end(),
                                [](float value) { return std::isnan(value); }),
                    "infinity times zero produces NaN");

        std::fill(x.begin(), x.end(), 1.0e20F);
        std::fill(w.begin(), w.end(), 1.0e-20F);
        auto expected = reference(x, w, shape);
        status = execute(compilation, variant, x, w, y, workspace.pointer);
        tests.good(status, "extreme finite input execution");
        tests.check(output_matches(y, expected),
                    "extreme finite inputs satisfy numeric tolerance");
    }
}

}  // namespace

int main() {
    TestRunner tests;
    test_feature_policy(tests);

    constexpr std::array<std::int64_t, 9> tails{1, 7, 8, 15, 16,
                                                17, 31, 32, 33};
    std::vector<ShapeCase> cases;
    for (auto size : tails) {
        cases.push_back({size, size, 2, 3});
    }
    cases.push_back({17, 33, 29, 30});

    for (auto shape : cases) {
        deepforge::compiler::CompileOptions options;
        options.emit_object = false;
        options.emit_llvm_ir = false;
        deepforge::compiler::CompilationResult compilation;
        auto status = deepforge::compiler::compile_graph(
            make_graph(shape), options, compilation);
        tests.good(status, "compile C/K and spatial tail case");
        if (status.is_bad()) {
            continue;
        }
        tests.check(std::all_of(
                        compilation.variants.begin(),
                        compilation.variants.end(),
                        [](deepforge::compiler::VariantCode const& code) {
                            return code.object.empty() && code.llvm_ir.empty();
                        }),
                    "disabled code outputs remain absent after JIT loading");
        auto workspace_size = compilation.executable->get_workspace_size();
        tests.check(workspace_size >= 0 && workspace_size % 64 == 0,
                    "tail case workspace is valid and aligned");
        AlignedBytes workspace(static_cast<std::size_t>(workspace_size));
        tests.check(workspace.size == 0 || workspace.pointer != nullptr,
                    "tail case workspace allocation succeeds");
        if (workspace.size != 0 && workspace.pointer == nullptr) {
            continue;
        }

        std::vector<float> x(static_cast<std::size_t>(
            2 * shape.height * shape.width * shape.channels));
        std::vector<float> w(static_cast<std::size_t>(
            shape.output_channels * shape.channels));
        std::vector<float> y(static_cast<std::size_t>(
            2 * shape.height * shape.width * shape.output_channels));
        for (std::size_t index = 0; index < x.size(); ++index) {
            x[index] = static_cast<float>(
                           (static_cast<int>(index * 11) % 29) - 14) /
                       9.0F;
        }
        for (std::size_t index = 0; index < w.size(); ++index) {
            w[index] = static_cast<float>(
                           (static_cast<int>(index * 7) % 23) - 11) /
                       8.0F;
        }
        auto expected = reference(x, w, shape);
        std::array<deepforge::runtime::CpuVariant, 3> variants{
            deepforge::runtime::CpuVariant::kScalar,
            deepforge::runtime::CpuVariant::kAvx2,
            deepforge::runtime::CpuVariant::kAvx512};
        for (auto variant : variants) {
            std::fill(y.begin(), y.end(), -1.0F);
            status = execute(compilation, variant, x, w, y, workspace.pointer);
            if (!compilation.executable->supports_variant(variant)) {
                tests.check(status.code() == deepforge::import::ErrorCode::
                                                 kUnsupportedCpuFeature,
                            "unsupported tail variant is rejected safely");
                continue;
            }
            tests.good(status, "execute C/K and spatial tail variant");
            tests.check(output_matches(y, expected),
                        "tail variant matches f64 reference");
        }

        std::fill(y.begin(), y.end(), -1.0F);
        deepforge::runtime::VariantPack pack{
            {compilation.metadata.x_uid, x.data()},
            {compilation.metadata.w_uid, w.data()},
            {compilation.metadata.y_uid, y.data()}};
        status = compilation.executable->execute(nullptr, pack,
                                                 workspace.pointer);
        tests.good(status, "automatic dispatch executes tail case");
        tests.check(output_matches(y, expected),
                    "automatic dispatch tail output matches reference");

        if (shape.channels == 1 && shape.output_channels == 1) {
            test_special_values(tests, compilation, shape, workspace);
        }
    }
    return tests.finish();
}
