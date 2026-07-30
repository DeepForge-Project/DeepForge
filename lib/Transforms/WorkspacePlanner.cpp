#include "DeepForge/Compiler/Bufferization.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace deepforge::compiler {
namespace {

using deepforge::import::ErrorCode;
using deepforge::import::Status;

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

bool is_power_of_two(std::uint64_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

bool checked_add(std::uint64_t lhs,
                 std::uint64_t rhs,
                 std::uint64_t& result) {
    if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

bool checked_align_up(std::uint64_t value,
                      std::uint64_t alignment,
                      std::uint64_t& result) {
    std::uint64_t with_padding = 0;
    if (!checked_add(value, alignment - 1, with_padding)) {
        return false;
    }
    result = with_padding & ~(alignment - 1);
    return true;
}

bool lifetimes_overlap(WorkspaceAllocation const& placed,
                       WorkspaceRequest const& request) {
    return !(placed.live_end < request.live_start ||
             request.live_end < placed.live_start);
}

bool ranges_overlap(std::uint64_t lhs_begin,
                    std::uint64_t lhs_end,
                    std::uint64_t rhs_begin,
                    std::uint64_t rhs_end) {
    return lhs_begin < rhs_end && rhs_begin < lhs_end;
}

}  // namespace

Status plan_workspace(std::span<WorkspaceRequest const> requests,
                      WorkspacePlan& output) {
    WorkspacePlan plan;
    std::set<std::string> names;
    for (auto const& request : requests) {
        if (request.name.empty()) {
            return fail(ErrorCode::kInvalidArgument, "workspace.request",
                        "allocation name must not be empty");
        }
        if (!names.insert(request.name).second) {
            return fail(ErrorCode::kInvalidValue, "workspace.request",
                        "allocation names must be unique");
        }
        if (request.size_bytes == 0) {
            return fail(ErrorCode::kInvalidArgument, request.name,
                        "allocation size must be positive");
        }
        if (!is_power_of_two(request.alignment) ||
            request.alignment > kWorkspaceAlignment) {
            return fail(ErrorCode::kInvalidArgument, request.name,
                        "alignment must be a power of two no greater than 64");
        }
        if (request.live_start > request.live_end) {
            return fail(ErrorCode::kInvalidArgument, request.name,
                        "lifetime start must not exceed lifetime end");
        }
    }

    std::vector<WorkspaceRequest const*> ordered;
    ordered.reserve(requests.size());
    for (auto const& request : requests) {
        ordered.push_back(&request);
    }
    std::stable_sort(ordered.begin(), ordered.end(),
                     [](WorkspaceRequest const* lhs,
                        WorkspaceRequest const* rhs) {
                         if (lhs->live_start != rhs->live_start) {
                             return lhs->live_start < rhs->live_start;
                         }
                         if (lhs->live_end != rhs->live_end) {
                             return lhs->live_end < rhs->live_end;
                         }
                         return lhs->name < rhs->name;
                     });

    for (auto const* request : ordered) {
        std::uint64_t candidate = 0;
        while (true) {
            if (!checked_align_up(candidate, request->alignment, candidate)) {
                return fail(ErrorCode::kDimensionOverflow, request->name,
                            "workspace offset alignment overflows uint64");
            }
            std::uint64_t candidate_end = 0;
            if (!checked_add(candidate, request->size_bytes, candidate_end)) {
                return fail(ErrorCode::kDimensionOverflow, request->name,
                            "workspace allocation range overflows uint64");
            }

            bool conflict = false;
            std::uint64_t next_candidate = candidate;
            for (auto const& placed : plan.allocations) {
                if (!lifetimes_overlap(placed, *request)) {
                    continue;
                }
                std::uint64_t placed_end = 0;
                if (!checked_add(placed.offset, placed.size_bytes, placed_end)) {
                    return fail(ErrorCode::kDimensionOverflow, placed.name,
                                "workspace allocation range overflows uint64");
                }
                if (ranges_overlap(candidate, candidate_end,
                                   placed.offset, placed_end)) {
                    conflict = true;
                    next_candidate = std::max(next_candidate, placed_end);
                }
            }
            if (!conflict) {
                plan.allocations.push_back(
                    WorkspaceAllocation{request->name,
                                        candidate,
                                        request->size_bytes,
                                        request->alignment,
                                        request->live_start,
                                        request->live_end});
                break;
            }
            candidate = next_candidate;
        }
    }

    std::uint64_t high_watermark = 0;
    for (auto const& allocation : plan.allocations) {
        std::uint64_t end = 0;
        if (!checked_add(allocation.offset, allocation.size_bytes, end)) {
            return fail(ErrorCode::kDimensionOverflow, allocation.name,
                        "workspace allocation range overflows uint64");
        }
        high_watermark = std::max(high_watermark, end);
    }
    if (high_watermark != 0 &&
        !checked_align_up(high_watermark, kWorkspaceAlignment,
                          plan.size_bytes)) {
        return fail(ErrorCode::kDimensionOverflow, "workspace",
                    "total workspace size overflows uint64");
    }
    if (plan.size_bytes >
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
        return fail(ErrorCode::kDimensionOverflow, "workspace",
                    "total workspace size does not fit int64");
    }
    output = std::move(plan);
    return Status::ok();
}

}  // namespace deepforge::compiler
