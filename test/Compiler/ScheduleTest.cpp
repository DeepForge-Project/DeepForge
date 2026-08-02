#include "DeepForge/Compiler/Schedule.h"

#include <cstdint>
#include <iostream>
#include <string>

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

    int finish() const {
        if (failures_ == 0) {
            std::cout << "deepforge-schedule: " << checks_
                      << " checks passed\n";
            return 0;
        }
        std::cerr << "deepforge-schedule: " << failures_ << " of " << checks_
                  << " checks failed\n";
        return 1;
    }

private:
    int checks_ = 0;
    int failures_ = 0;
};

deepforge::compiler::Conv2DCompileMetadata metadata(std::int64_t channels,
                                                     std::int64_t outputs,
                                                     std::int64_t filter) {
    deepforge::compiler::Conv2DCompileMetadata result;
    result.x_shape = {1, 32, 32, channels};
    result.w_shape = {outputs, filter, filter, channels};
    result.y_shape = {1, 32, 32, outputs};
    return result;
}

}  // namespace

int main() {
    using deepforge::compiler::Conv2DSchedulePolicy;
    using deepforge::runtime::CpuVariant;

    TestRunner tests;
    auto medium = metadata(32, 32, 3);
    auto scalar = deepforge::compiler::select_conv2d_schedule(
        medium, CpuVariant::kScalar, Conv2DSchedulePolicy::kAuto);
    tests.check(scalar.vector_width == 1 &&
                    scalar.output_channel_unroll == 1 &&
                    !scalar.cost_model_applied,
                "scalar keeps the fixed baseline schedule");

    auto avx2 = deepforge::compiler::select_conv2d_schedule(
        medium, CpuVariant::kAvx2, Conv2DSchedulePolicy::kAuto);
    tests.check(avx2.vector_width == 8 && avx2.output_channel_unroll == 4 &&
                    avx2.estimated_input_loads == 288 &&
                    avx2.estimated_weight_loads == 1152 &&
                    avx2.cost_model_applied,
                "AVX2 auto policy selects the legal four-output schedule");
    tests.check(deepforge::compiler::conv2d_schedule_name(avx2) ==
                    "direct-c-vf8-ku4",
                "AVX2 schedule has a stable inspectable name");

    auto avx512 = deepforge::compiler::select_conv2d_schedule(
        medium, CpuVariant::kAvx512, Conv2DSchedulePolicy::kAuto);
    tests.check(avx512.vector_width == 16 &&
                    avx512.output_channel_unroll == 8 &&
                    avx512.estimated_input_loads == 72 &&
                    avx512.estimated_weight_loads == 576 &&
                    avx512.cost_model_applied,
                "AVX-512 auto policy uses its larger register budget");

    auto baseline = deepforge::compiler::select_conv2d_schedule(
        medium, CpuVariant::kAvx512, Conv2DSchedulePolicy::kBaseline);
    tests.check(baseline.vector_width == 16 &&
                    baseline.output_channel_unroll == 1 &&
                    baseline.estimated_input_loads == 576 &&
                    baseline.estimated_weight_loads == 576 &&
                    !baseline.cost_model_applied,
                "baseline policy disables output-channel unrolling");

    auto tiny = deepforge::compiler::select_conv2d_schedule(
        metadata(1, 1, 1), CpuVariant::kAvx2,
        Conv2DSchedulePolicy::kAuto);
    tests.check(tiny.output_channel_unroll == 1 &&
                    tiny.cost_model_applied,
                "auto policy retains the baseline when no larger candidate is legal");

    auto tail = deepforge::compiler::select_conv2d_schedule(
        metadata(17, 33, 3), CpuVariant::kAvx512,
        Conv2DSchedulePolicy::kAuto);
    tests.check(tail.output_channel_unroll == 8 &&
                    tail.estimated_input_loads == 90 &&
                    tail.estimated_weight_loads == 594,
                "cost estimate accounts for C and K tails");
    return tests.finish();
}
