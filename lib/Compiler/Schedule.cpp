#include "DeepForge/Compiler/Schedule.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>

namespace deepforge::compiler {
namespace {

std::uint64_t saturating_add(std::uint64_t left, std::uint64_t right) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return left + right;
}

std::uint64_t saturating_multiply(std::uint64_t left,
                                  std::uint64_t right) {
    if (left != 0 &&
        right > std::numeric_limits<std::uint64_t>::max() / left) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return left * right;
}

std::uint64_t positive(std::int64_t value) {
    return value > 0 ? static_cast<std::uint64_t>(value) : 0;
}

struct TargetFacts {
    std::int64_t vector_width = 1;
    std::int64_t vector_registers = 0;
};

TargetFacts target_facts(runtime::CpuVariant variant) {
    switch (variant) {
        case runtime::CpuVariant::kScalar:
            return {1, 0};
        case runtime::CpuVariant::kAvx2:
            return {8, 16};
        case runtime::CpuVariant::kAvx512:
            return {16, 32};
    }
    return {1, 0};
}

std::uint64_t reduction_load_steps(Conv2DCompileMetadata const& metadata,
                                   std::int64_t vector_width) {
    auto const channels = positive(metadata.w_shape[3]);
    auto const vector_blocks =
        channels / static_cast<std::uint64_t>(vector_width);
    auto const scalar_tail =
        channels % static_cast<std::uint64_t>(vector_width);
    auto const channel_steps = saturating_add(vector_blocks, scalar_tail);
    return saturating_multiply(
        saturating_multiply(positive(metadata.w_shape[1]),
                            positive(metadata.w_shape[2])),
        channel_steps);
}

Conv2DSchedule candidate(Conv2DCompileMetadata const& metadata,
                         TargetFacts facts,
                         std::int64_t output_channel_unroll,
                         bool cost_model_applied) {
    auto const channels = positive(metadata.y_shape[3]);
    auto const full_blocks =
        channels / static_cast<std::uint64_t>(output_channel_unroll);
    auto const channel_tail =
        channels % static_cast<std::uint64_t>(output_channel_unroll);
    auto const input_groups = saturating_add(full_blocks, channel_tail);
    auto const reduction_steps =
        reduction_load_steps(metadata, facts.vector_width);
    Conv2DSchedule result;
    result.vector_width = facts.vector_width;
    result.output_channel_unroll = output_channel_unroll;
    result.estimated_input_loads =
        saturating_multiply(input_groups, reduction_steps);
    result.estimated_weight_loads =
        saturating_multiply(channels, reduction_steps);
    result.cost_model_applied = cost_model_applied;
    return result;
}

std::uint64_t score(Conv2DSchedule const& schedule) {
    auto loads = saturating_add(
        saturating_multiply(schedule.estimated_input_loads, 2),
        schedule.estimated_weight_loads);
    // Larger unroll factors create more IR and consume more registers. This
    // fixed penalty prevents tiny reductions from selecting a larger body.
    return saturating_add(
        loads,
        saturating_multiply(
            static_cast<std::uint64_t>(schedule.output_channel_unroll), 16));
}

}  // namespace

Conv2DSchedule select_conv2d_schedule(
    Conv2DCompileMetadata const& metadata,
    runtime::CpuVariant variant,
    Conv2DSchedulePolicy policy) {
    auto const facts = target_facts(variant);
    auto best = candidate(metadata, facts, 1, false);
    if (policy == Conv2DSchedulePolicy::kBaseline ||
        variant == runtime::CpuVariant::kScalar) {
        return best;
    }

    auto best_score = score(best);
    for (auto unroll : std::array<std::int64_t, 3>{2, 4, 8}) {
        if (unroll > metadata.y_shape[3] ||
            2 * unroll + 4 > facts.vector_registers) {
            continue;
        }
        auto current = candidate(metadata, facts, unroll, true);
        auto const current_score = score(current);
        if (current_score < best_score) {
            best = current;
            best_score = current_score;
        }
    }
    best.cost_model_applied = true;
    return best;
}

std::string conv2d_schedule_name(Conv2DSchedule const& schedule) {
    return "direct-c-vf" + std::to_string(schedule.vector_width) + "-ku" +
           std::to_string(schedule.output_channel_unroll);
}

}  // namespace deepforge::compiler
