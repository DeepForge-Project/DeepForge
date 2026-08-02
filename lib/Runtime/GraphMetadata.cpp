#include "DeepForge/Compiler/GraphMetadata.h"

#include "DeepForge/Import/Capability.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <set>
#include <string>
#include <utility>

namespace deepforge::compiler {
namespace {

import::Status fail(std::string detail) {
    return import::Status::failure(
        import::ErrorCode::kInvalidValue,
        std::string(import::error_code_name(import::ErrorCode::kInvalidValue)) +
            ": graph_metadata: " + std::move(detail));
}

bool is_power_of_two(std::uint64_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

bool valid_access(TensorAccess access) {
    switch (access) {
        case TensorAccess::kRead:
        case TensorAccess::kWrite:
        case TensorAccess::kReadWrite:
            return true;
    }
    return false;
}

bool valid_override_policy(ShapeOverridePolicy policy) {
    switch (policy) {
        case ShapeOverridePolicy::kNone:
        case ShapeOverridePolicy::kPointwiseExact:
            return true;
    }
    return false;
}

bool valid_storage_policy(TensorStoragePolicy policy) {
    switch (policy) {
        case TensorStoragePolicy::kStrided:
        case TensorStoragePolicy::kRaggedBatchPrefix:
            return true;
    }
    return false;
}

bool ragged_storage_bytes(TensorArgumentMetadata const& argument,
                          std::uint64_t& output) {
    if (argument.dimensions.empty()) return false;
    std::uint64_t inner_span = 1;
    for (std::size_t axis = 1; axis < argument.dimensions.size(); ++axis) {
        auto const extent =
            static_cast<std::uint64_t>(argument.dimensions[axis] - 1);
        auto const stride = static_cast<std::uint64_t>(argument.strides[axis]);
        if (argument.dimensions[axis] <= 0 || argument.strides[axis] <= 0 ||
            extent >
                (std::numeric_limits<std::uint64_t>::max() - inner_span) /
                    stride) {
            return false;
        }
        inner_span += extent * stride;
    }
    if (argument.dimensions.front() <= 0) return false;
    auto const batches =
        static_cast<std::uint64_t>(argument.dimensions.front());
    if (batches > std::numeric_limits<std::uint64_t>::max() / inner_span) {
        return false;
    }
    auto const elements = batches * inner_span;
    auto const bits = import::data_type_storage_bits(argument.data_type);
    if (bits == 0 ||
        elements > (std::numeric_limits<std::uint64_t>::max() - 7) / bits) {
        return false;
    }
    output = (elements * bits + 7) / 8;
    return true;
}

}  // namespace

import::Status validate_graph_compile_metadata(
    GraphCompileMetadata const& metadata) {
    if (metadata.function_name.empty() ||
        metadata.function_name.find('\0') != std::string::npos) {
        return fail("function name is empty or contains NUL");
    }
    if (metadata.arguments.empty() || metadata.arguments.size() > 4096) {
        return fail("argument count must be between 1 and 4096");
    }
    if (!valid_override_policy(metadata.override_policy) ||
        metadata.override_shape_enabled !=
            (metadata.override_policy != ShapeOverridePolicy::kNone)) {
        return fail("shape override flag and policy are inconsistent");
    }

    std::set<std::int64_t> uids;
    std::size_t override_write_count = 0;
    for (auto const& argument : metadata.arguments) {
        std::uint64_t expected_size = 0;
        if (!uids.insert(argument.uid).second) {
            return fail("argument UIDs are not unique");
        }
        if (argument.name.empty() ||
            argument.name.find('\0') != std::string::npos) {
            return fail("argument name is empty or contains NUL");
        }
        if (argument.dimensions.empty() || argument.dimensions.size() > 64 ||
            argument.dimensions.size() != argument.strides.size()) {
            return fail("argument rank or stride rank is invalid");
        }
        if (!is_power_of_two(argument.alignment) ||
            argument.alignment > 4096) {
            return fail("argument alignment is invalid");
        }
        if (!valid_access(argument.access)) {
            return fail("argument access mode is invalid");
        }
        if (!valid_storage_policy(argument.storage_policy)) {
            return fail("argument storage policy is invalid");
        }
        if (metadata.override_policy == ShapeOverridePolicy::kPointwiseExact) {
            if (argument.data_type != import::DataType::kFloat32 ||
                argument.dimensions != metadata.arguments.front().dimensions) {
                return fail(
                    "exact-pointwise override arguments must be equal-shape FLOAT tensors");
            }
            if (argument.access == TensorAccess::kWrite) {
                ++override_write_count;
            } else if (argument.access != TensorAccess::kRead) {
                return fail(
                    "exact-pointwise override arguments cannot be read-write");
            }
        }
        auto const valid_size =
            argument.storage_policy == TensorStoragePolicy::kRaggedBatchPrefix
                ? ragged_storage_bytes(argument, expected_size)
                : import::tensor_storage_bytes(
                      argument.data_type, argument.dimensions,
                      argument.strides, expected_size);
        if (!valid_size ||
            expected_size != argument.size_bytes ||
            argument.size_bytes >
                static_cast<std::uint64_t>(
                    std::numeric_limits<std::size_t>::max())) {
            return fail("argument storage range is invalid");
        }
        if (argument.storage_policy == TensorStoragePolicy::kStrided &&
            (argument.ragged_offset_uid != 0 ||
             argument.ragged_sequence_uid != 0)) {
            return fail("strided argument carries ragged references");
        }
    }
    if (metadata.override_policy == ShapeOverridePolicy::kPointwiseExact &&
        (metadata.arguments.size() < 2 || override_write_count != 1)) {
        return fail(
            "exact-pointwise override metadata requires inputs and one output");
    }
    for (auto const& argument : metadata.arguments) {
        if (argument.storage_policy !=
            TensorStoragePolicy::kRaggedBatchPrefix) {
            continue;
        }
        auto const find_uid = [&](std::int64_t uid) {
            return std::find_if(
                metadata.arguments.begin(), metadata.arguments.end(),
                [&](TensorArgumentMetadata const& candidate) {
                    return candidate.uid == uid;
                });
        };
        auto const offset = find_uid(argument.ragged_offset_uid);
        auto const sequence = find_uid(argument.ragged_sequence_uid);
        if (argument.data_type != import::DataType::kFloat32 ||
            argument.dimensions.size() != 4 ||
            offset == metadata.arguments.end() ||
            sequence == metadata.arguments.end() ||
            (offset->data_type != import::DataType::kInt32 &&
             offset->data_type != import::DataType::kInt64) ||
            sequence->data_type != import::DataType::kInt32 ||
            argument.dimensions[0] ==
                std::numeric_limits<std::int64_t>::max() ||
            offset->dimensions !=
                std::vector<std::int64_t>{argument.dimensions[0] + 1, 1, 1,
                                          1} ||
            sequence->dimensions !=
                std::vector<std::int64_t>{argument.dimensions[0], 1, 1, 1} ||
            offset->access != TensorAccess::kRead ||
            sequence->access != TensorAccess::kRead) {
            return fail("ragged argument references are inconsistent");
        }
    }
    return import::Status::ok();
}

}  // namespace deepforge::compiler
