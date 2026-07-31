#include "DeepForge/Compiler/Codegen.h"

#include "DeepForge/Runtime/Executable.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>
#include <unordered_map>
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
            std::cout << "deepforge-compiler: " << checks_
                      << " checks passed\n";
            return 0;
        }
        std::cerr << "deepforge-compiler: " << failures_ << " of " << checks_
                  << " checks failed\n";
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

std::size_t element_count(std::array<std::int64_t, 4> const& shape) {
    std::size_t result = 1;
    for (auto dimension : shape) {
        result *= static_cast<std::size_t>(dimension);
    }
    return result;
}

float reference_at(std::vector<float> const& x, std::vector<float> const& w,
                   deepforge::compiler::Conv2DCompileMetadata const& metadata,
                   std::int64_t n, std::int64_t oh, std::int64_t ow,
                   std::int64_t k) {
    double sum = 0.0;
    auto const input_h_size = metadata.x_shape[1];
    auto const input_w_size = metadata.x_shape[2];
    auto const channels = metadata.x_shape[3];
    auto const filter_h_size = metadata.w_shape[1];
    auto const filter_w_size = metadata.w_shape[2];
    for (std::int64_t r = 0; r < filter_h_size; ++r) {
        for (std::int64_t s = 0; s < filter_w_size; ++s) {
            for (std::int64_t c = 0; c < channels; ++c) {
                auto input_h = oh * metadata.stride[0] +
                               r * metadata.dilation[0] -
                               metadata.pre_padding[0];
                auto input_w = ow * metadata.stride[1] +
                               s * metadata.dilation[1] -
                               metadata.pre_padding[1];
                float input = 0.0F;
                if (input_h >= 0 && input_h < input_h_size && input_w >= 0 &&
                    input_w < input_w_size) {
                    auto input_index =
                        ((n * input_h_size + input_h) * input_w_size + input_w) *
                            channels +
                        c;
                    input = x[static_cast<std::size_t>(input_index)];
                }
                auto weight_index =
                    ((k * filter_h_size + r) * filter_w_size + s) * channels +
                    c;
                float weight = w[static_cast<std::size_t>(weight_index)];
                sum += static_cast<double>(input) * weight;
            }
        }
    }
    return static_cast<float>(sum);
}

}  // namespace

int main(int argc, char** argv) {
    TestRunner tests;
    if (argc != 2) {
        std::cerr << "usage: deepforge_compiler_test <fixture.json>\n";
        return 2;
    }

    deepforge::compiler::CompileOptions options;
    options.build_avx_variants = true;
    deepforge::compiler::CompilationResult compilation;
    auto status = deepforge::compiler::compile_file(
        std::filesystem::path(argv[1]), options, compilation);
    tests.good(status, "compile official JSON fixture");
    if (status.is_bad() || !compilation.executable) {
        return tests.finish();
    }
    auto expected_workspace =
        element_count(compilation.metadata.padded_x_shape) * sizeof(float);
    expected_workspace = (expected_workspace + 63U) & ~std::size_t{63U};
    auto workspace_size = compilation.executable->get_workspace_size();
    tests.check(workspace_size >= 0 &&
                    static_cast<std::size_t>(workspace_size) ==
                        expected_workspace,
                "runtime exposes planned workspace size");
    tests.check(!compilation.variants[0].object.empty(),
                "scalar object is emitted");
    tests.check(compilation.variants[0].llvm_ir.find("define") !=
                    std::string::npos,
                "scalar LLVM IR is emitted");
    tests.check(!compilation.variants[1].object.empty(),
                "AVX2 object is emitted");
    tests.check(!compilation.variants[2].object.empty(),
                "AVX-512 object is emitted");
    tests.check(compilation.metadata.arguments.size() == 3 &&
                    compilation.metadata.arguments[0].uid ==
                        compilation.metadata.x_uid &&
                    compilation.metadata.arguments[1].uid ==
                        compilation.metadata.w_uid &&
                    compilation.metadata.arguments[2].uid ==
                        compilation.metadata.y_uid &&
                    compilation.metadata.arguments[0].access ==
                        deepforge::compiler::TensorAccess::kRead &&
                    compilation.metadata.arguments[1].access ==
                        deepforge::compiler::TensorAccess::kRead &&
                    compilation.metadata.arguments[2].access ==
                        deepforge::compiler::TensorAccess::kWrite,
                "compile metadata exposes an ordered generic argument table");
    tests.check(compilation.variants[0].symbol != compilation.variants[1].symbol &&
                    compilation.variants[1].symbol !=
                        compilation.variants[2].symbol,
                "variant entry symbols are distinct");
    tests.check(compilation.variants[0].llvm_ir.find("<8 x float>") ==
                        std::string::npos &&
                    compilation.variants[0].llvm_ir.find("<16 x float>") ==
                        std::string::npos,
                "scalar LLVM IR contains no SIMD reduction type");
    if (compilation.metadata.x_shape[3] >= 17) {
        tests.check(compilation.variants[1].llvm_ir.find("<8 x float>") !=
                            std::string::npos &&
                        compilation.variants[2].llvm_ir.find("<16 x float>") !=
                            std::string::npos,
                    "AVX LLVM IR contains full-width vector reductions");
    }

    std::vector<float> x(element_count(compilation.metadata.x_shape));
    std::vector<float> w(element_count(compilation.metadata.w_shape));
    std::vector<float> y(element_count(compilation.metadata.y_shape), -91.0F);
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = static_cast<float>((static_cast<int>(i * 7) % 19) - 9) / 7.0F;
    }
    for (std::size_t i = 0; i < w.size(); ++i) {
        w[i] = static_cast<float>((static_cast<int>(i * 5) % 13) - 6) / 5.0F;
    }
    AlignedBytes workspace(static_cast<std::size_t>(workspace_size));
    tests.check(workspace.size == 0 || workspace.pointer != nullptr,
                "aligned workspace allocation");
    if (workspace.size != 0 && !workspace.pointer) {
        return tests.finish();
    }

    deepforge::runtime::VariantPack pack{{compilation.metadata.x_uid, x.data()},
                                         {compilation.metadata.w_uid, w.data()},
                                         {compilation.metadata.y_uid, y.data()}};
    auto* y_address = y.data();
    std::array<deepforge::runtime::CpuVariant, 3> variants{
        deepforge::runtime::CpuVariant::kScalar,
        deepforge::runtime::CpuVariant::kAvx2,
        deepforge::runtime::CpuVariant::kAvx512};
    for (auto variant : variants) {
        std::fill(y.begin(), y.end(), -91.0F);
        status = compilation.executable->execute_variant(
            variant, nullptr, pack, workspace.pointer);
        if (!compilation.executable->supports_variant(variant)) {
            tests.check(
                status.code() ==
                    deepforge::import::ErrorCode::kUnsupportedCpuFeature,
                std::string("unsupported variant is rejected: ") +
                    std::string(deepforge::runtime::cpu_variant_name(variant)));
            continue;
        }
        tests.good(status, std::string("JIT execution: ") +
                              std::string(deepforge::runtime::cpu_variant_name(
                                  variant)));
        tests.check(pack.at(compilation.metadata.y_uid) == y_address,
                    "execute does not replace the caller Y pointer");
        for (std::int64_t n = 0; n < compilation.metadata.y_shape[0]; ++n) {
            for (std::int64_t oh = 0; oh < compilation.metadata.y_shape[1];
                 ++oh) {
                for (std::int64_t ow = 0; ow < compilation.metadata.y_shape[2];
                     ++ow) {
                    for (std::int64_t k = 0;
                         k < compilation.metadata.y_shape[3]; ++k) {
                    auto index =
                        ((n * compilation.metadata.y_shape[1] + oh) *
                             compilation.metadata.y_shape[2] +
                         ow) *
                            compilation.metadata.y_shape[3] +
                        k;
                    float expected = reference_at(
                        x, w, compilation.metadata, n, oh, ow, k);
                    float actual = y[static_cast<std::size_t>(index)];
                    float tolerance = 1.0e-4F + 1.0e-3F * std::fabs(expected);
                    tests.check(std::isfinite(actual) &&
                                    std::fabs(actual - expected) <= tolerance,
                                "variant output matches f64 reference");
                    }
                }
            }
        }
    }

    std::fill(y.begin(), y.end(), -91.0F);
    status = compilation.executable->execute(nullptr, pack, workspace.pointer);
    tests.good(status, "automatic CPU dispatch executes");
    tests.check(compilation.executable->supports_variant(
                    compilation.executable->selected_variant()),
                "automatic CPU dispatch selects a supported variant");

    auto missing = pack;
    missing.erase(compilation.metadata.w_uid);
    status = compilation.executable->execute_variant(
        deepforge::runtime::CpuVariant::kScalar, nullptr, missing,
        workspace.pointer);
    tests.check(status.code() ==
                    deepforge::import::ErrorCode::kInvalidVariantPack,
                "missing UID is rejected at runtime");

    auto null_tensor = pack;
    null_tensor[compilation.metadata.x_uid] = nullptr;
    status = compilation.executable->execute_variant(
        deepforge::runtime::CpuVariant::kScalar, nullptr, null_tensor,
        workspace.pointer);
    tests.check(status.code() ==
                    deepforge::import::ErrorCode::kInvalidVariantPack,
                "null tensor pointer is rejected at runtime");

    std::vector<std::uint8_t> unaligned_storage(x.size() * sizeof(float) + 1);
    auto unaligned = pack;
    unaligned[compilation.metadata.x_uid] = unaligned_storage.data() + 1;
    status = compilation.executable->execute_variant(
        deepforge::runtime::CpuVariant::kScalar, nullptr, unaligned,
        workspace.pointer);
    tests.check(status.code() ==
                    deepforge::import::ErrorCode::kInvalidVariantPack,
                "misaligned tensor pointer is rejected at runtime");

    auto tensor_alias = pack;
    tensor_alias[compilation.metadata.y_uid] = x.data();
    status = compilation.executable->execute_variant(
        deepforge::runtime::CpuVariant::kScalar, nullptr, tensor_alias,
        workspace.pointer);
    tests.check(status.code() ==
                    deepforge::import::ErrorCode::kInvalidVariantPack,
                "overlapping tensor ranges are rejected at runtime");

    auto shared_size = std::max(x.size(), w.size());
    std::vector<float> shared_read_storage(shared_size, 0.25F);
    std::vector<float> shared_read_y(y.size(), -1.0F);
    deepforge::runtime::VariantPack shared_read_pack{
        {compilation.metadata.x_uid, shared_read_storage.data()},
        {compilation.metadata.w_uid, shared_read_storage.data()},
        {compilation.metadata.y_uid, shared_read_y.data()}};
    status = compilation.executable->execute_variant(
        deepforge::runtime::CpuVariant::kScalar, nullptr, shared_read_pack,
        workspace.pointer);
    tests.good(status, "overlapping read-only tensor ranges are accepted");

    if (workspace.size != 0) {
        auto workspace_alias = pack;
        workspace_alias[compilation.metadata.y_uid] = workspace.pointer;
        status = compilation.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr, workspace_alias,
            workspace.pointer);
        tests.check(status.code() ==
                        deepforge::import::ErrorCode::kInvalidVariantPack,
                    "tensor and workspace overlap is rejected at runtime");

        status = compilation.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr, pack,
            static_cast<std::uint8_t*>(workspace.pointer) + 1);
        tests.check(
            status.code() ==
                deepforge::import::ErrorCode::kInvalidVariantPack,
            "misaligned workspace is rejected at runtime");
    }

    auto extra_uid = pack;
    extra_uid.emplace(999999, x.data());
    status = compilation.executable->execute_variant(
        deepforge::runtime::CpuVariant::kScalar,
        reinterpret_cast<void*>(std::uintptr_t{1}), extra_uid,
        workspace.pointer);
    tests.good(status, "extra UID and opaque non-null handle are accepted");

    if (workspace.size != 0) {
        auto null_workspace = pack;
        status = compilation.executable->execute_variant(
            deepforge::runtime::CpuVariant::kScalar, nullptr, null_workspace,
            nullptr);
        tests.check(
            status.code() ==
                deepforge::import::ErrorCode::kInvalidVariantPack,
                    "null workspace is rejected at runtime");
    }

    std::vector<float> concurrent_y_a(y.size(), -1.0F);
    std::vector<float> concurrent_y_b(y.size(), -2.0F);
    AlignedBytes concurrent_workspace_a(workspace.size);
    AlignedBytes concurrent_workspace_b(workspace.size);
    deepforge::runtime::VariantPack concurrent_pack_a{
        {compilation.metadata.x_uid, x.data()},
        {compilation.metadata.w_uid, w.data()},
        {compilation.metadata.y_uid, concurrent_y_a.data()}};
    deepforge::runtime::VariantPack concurrent_pack_b{
        {compilation.metadata.x_uid, x.data()},
        {compilation.metadata.w_uid, w.data()},
        {compilation.metadata.y_uid, concurrent_y_b.data()}};
    deepforge::import::Status concurrent_status_a;
    deepforge::import::Status concurrent_status_b;
    std::thread first([&] {
        concurrent_status_a = compilation.executable->execute(
            nullptr, concurrent_pack_a, concurrent_workspace_a.pointer);
    });
    std::thread second([&] {
        concurrent_status_b = compilation.executable->execute(
            nullptr, concurrent_pack_b, concurrent_workspace_b.pointer);
    });
    first.join();
    second.join();
    tests.good(concurrent_status_a, "first concurrent execution");
    tests.good(concurrent_status_b, "second concurrent execution");
    tests.check(concurrent_y_a == concurrent_y_b,
                "concurrent executions with separate workspace agree");
    return tests.finish();
}
