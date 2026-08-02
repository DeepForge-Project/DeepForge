#pragma once

#include "DeepForge/Compiler/Conv2DImporter.h"
#include "DeepForge/Runtime/Executable.h"

#include <cstdint>
#include <string>

namespace deepforge::compiler {

enum class Conv2DSchedulePolicy : std::uint8_t {
    kBaseline = 0,
    kAuto = 1,
};

struct Conv2DSchedule {
    std::int64_t vector_width = 1;
    std::int64_t output_channel_unroll = 1;
    std::uint64_t estimated_input_loads = 0;
    std::uint64_t estimated_weight_loads = 0;
    bool cost_model_applied = false;

    bool operator==(Conv2DSchedule const&) const = default;
};

// Select a legal direct-Conv schedule. The auto policy minimizes a deterministic
// load/register-pressure estimate and always retains the baseline candidate.
[[nodiscard]] Conv2DSchedule select_conv2d_schedule(
    Conv2DCompileMetadata const& metadata,
    runtime::CpuVariant variant,
    Conv2DSchedulePolicy policy = Conv2DSchedulePolicy::kAuto);

[[nodiscard]] std::string conv2d_schedule_name(
    Conv2DSchedule const& schedule);

}  // namespace deepforge::compiler
