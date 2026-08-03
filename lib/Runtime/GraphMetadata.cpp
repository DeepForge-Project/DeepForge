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
        case ShapeOverridePolicy::kMatmul:
        case ShapeOverridePolicy::kSdpaForward:
        case ShapeOverridePolicy::kReshape:
        case ShapeOverridePolicy::kReduction:
        case ShapeOverridePolicy::kTranspose:
        case ShapeOverridePolicy::kConcatenate:
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

bool valid_matmul_dimensions(TensorArgumentMetadata const& a,
                             TensorArgumentMetadata const& b,
                             TensorArgumentMetadata const& c) {
    if (a.dimensions.size() < 2 ||
        a.dimensions.size() != b.dimensions.size() ||
        a.dimensions.size() != c.dimensions.size()) {
        return false;
    }
    auto const rank = a.dimensions.size();
    if (a.dimensions[rank - 1] != b.dimensions[rank - 2] ||
        a.dimensions[rank - 2] != c.dimensions[rank - 2] ||
        b.dimensions[rank - 1] != c.dimensions[rank - 1]) {
        return false;
    }
    for (std::size_t axis = 0; axis + 2 < rank; ++axis) {
        auto const a_extent = a.dimensions[axis];
        auto const b_extent = b.dimensions[axis];
        if ((a_extent != b_extent && a_extent != 1 && b_extent != 1) ||
            c.dimensions[axis] != std::max(a_extent, b_extent)) {
            return false;
        }
    }
    return true;
}

bool valid_sdpa_dimensions(
    std::vector<TensorArgumentMetadata const*> const& roles) {
    if (roles.size() < 4 || roles.size() > 7 ||
        std::any_of(roles.begin(), roles.end(), [](auto const* argument) {
            return argument->dimensions.size() != 4;
        })) {
        return false;
    }
    auto const& q = roles[0]->dimensions;
    auto const& k = roles[1]->dimensions;
    auto const& v = roles[2]->dimensions;
    auto const& o = roles[3]->dimensions;
    if (q[1] <= 0 || k[1] <= 0 || v[1] <= 0 ||
        q[0] != k[0] || q[0] != v[0] || q[0] != o[0] ||
        q[1] != o[1] || q[2] != o[2] || q[3] != k[3] ||
        k[2] != v[2] || o[3] != v[3] || q[1] % k[1] != 0 ||
        q[1] % v[1] != 0) {
        return false;
    }
    std::vector<std::int64_t> const row_dimensions{q[0], q[1], q[2], 1};
    return std::all_of(roles.begin() + 4, roles.end(),
                       [&](auto const* argument) {
                           return argument->dimensions == row_dimensions;
                       });
}

bool checked_element_count(std::vector<std::int64_t> const& dimensions,
                           std::uint64_t& output) {
    std::uint64_t count = 1;
    for (auto dimension : dimensions) {
        if (dimension <= 0 ||
            static_cast<std::uint64_t>(dimension) >
                std::numeric_limits<std::uint64_t>::max() / count) {
            return false;
        }
        count *= static_cast<std::uint64_t>(dimension);
    }
    output = count;
    return true;
}

bool valid_reshape_dimensions(TensorArgumentMetadata const& x,
                              TensorArgumentMetadata const& y) {
    std::uint64_t x_elements = 0;
    std::uint64_t y_elements = 0;
    return checked_element_count(x.dimensions, x_elements) &&
           checked_element_count(y.dimensions, y_elements) &&
           x_elements == y_elements &&
           x_elements <=
               static_cast<std::uint64_t>(
                   std::numeric_limits<std::int64_t>::max());
}

bool valid_reduction_dimensions(TensorArgumentMetadata const& x,
                                TensorArgumentMetadata const& y) {
    if (x.dimensions.size() != y.dimensions.size()) return false;
    for (std::size_t axis = 0; axis < x.dimensions.size(); ++axis) {
        if (y.dimensions[axis] != 1 &&
            y.dimensions[axis] != x.dimensions[axis]) {
            return false;
        }
    }
    std::uint64_t elements = 0;
    return checked_element_count(x.dimensions, elements) &&
           elements <= static_cast<std::uint64_t>(
                           std::numeric_limits<std::int64_t>::max());
}

bool valid_transpose_dimensions(
    TensorArgumentMetadata const& x,
    TensorArgumentMetadata const& y,
    std::vector<std::int64_t> const& permutation) {
    auto const rank = x.dimensions.size();
    if (rank == 0 || y.dimensions.size() != rank ||
        permutation.size() != rank) {
        return false;
    }
    std::vector<bool> seen(rank, false);
    for (std::size_t axis = 0; axis < rank; ++axis) {
        auto const source_axis = permutation[axis];
        if (source_axis < 0 ||
            static_cast<std::size_t>(source_axis) >= rank ||
            seen[static_cast<std::size_t>(source_axis)] ||
            y.dimensions[axis] !=
                x.dimensions[static_cast<std::size_t>(source_axis)]) {
            return false;
        }
        seen[static_cast<std::size_t>(source_axis)] = true;
    }
    return true;
}

bool valid_concatenate_dimensions(
    std::vector<TensorArgumentMetadata const*> const& roles,
    std::int64_t axis) {
    if (roles.size() < 2 || roles.size() > 64) return false;
    auto const& output = roles.back()->dimensions;
    if (output.empty() || axis < 0 ||
        static_cast<std::size_t>(axis) >= output.size()) {
        return false;
    }
    std::int64_t axis_extent = 0;
    for (std::size_t role = 0; role + 1 < roles.size(); ++role) {
        auto const* input = roles[role];
        if (input->dimensions.size() != output.size()) return false;
        for (std::size_t dimension = 0; dimension < output.size();
             ++dimension) {
            if (dimension != static_cast<std::size_t>(axis) &&
                input->dimensions[dimension] != output[dimension]) {
                return false;
            }
        }
        auto const extent = input->dimensions[static_cast<std::size_t>(axis)];
        if (extent > std::numeric_limits<std::int64_t>::max() - axis_extent) {
            return false;
        }
        axis_extent += extent;
    }
    return axis_extent == output[static_cast<std::size_t>(axis)];
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
             argument.ragged_sequence_uid != 0 ||
             argument.ragged_sequence_divisor != 1)) {
            return fail("strided argument carries ragged references");
        }
    }
    if (metadata.override_policy == ShapeOverridePolicy::kPointwiseExact &&
        (metadata.arguments.size() < 2 || override_write_count != 1)) {
        return fail(
            "exact-pointwise override metadata requires inputs and one output");
    }
    if (metadata.override_policy != ShapeOverridePolicy::kMatmul &&
        metadata.override_policy != ShapeOverridePolicy::kSdpaForward &&
        metadata.override_policy != ShapeOverridePolicy::kReshape &&
        metadata.override_policy != ShapeOverridePolicy::kReduction &&
        metadata.override_policy != ShapeOverridePolicy::kTranspose &&
        metadata.override_policy != ShapeOverridePolicy::kConcatenate &&
        !metadata.override_role_uids.empty()) {
        return fail("override role UIDs require a role-based policy");
    }
    if (metadata.override_policy != ShapeOverridePolicy::kTranspose &&
        metadata.override_policy != ShapeOverridePolicy::kConcatenate &&
        !metadata.override_axis_map.empty()) {
        return fail(
            "override axis map requires the TRANSPOSE or CONCATENATE policy");
    }
    if (metadata.override_policy == ShapeOverridePolicy::kMatmul) {
        if (metadata.arguments.size() != 3 ||
            metadata.override_role_uids.size() != 3 ||
            std::set<std::int64_t>(metadata.override_role_uids.begin(),
                                   metadata.override_role_uids.end())
                    .size() != 3) {
            return fail(
                "MATMUL override metadata requires distinct A/B/C role UIDs");
        }
        auto find_role = [&](std::size_t role) {
            return std::find_if(
                metadata.arguments.begin(), metadata.arguments.end(),
                [&](TensorArgumentMetadata const& argument) {
                    return argument.uid == metadata.override_role_uids[role];
                });
        };
        auto const a = find_role(0);
        auto const b = find_role(1);
        auto const c = find_role(2);
        if (a == metadata.arguments.end() || b == metadata.arguments.end() ||
            c == metadata.arguments.end() ||
            a->data_type != import::DataType::kFloat32 ||
            b->data_type != import::DataType::kFloat32 ||
            c->data_type != import::DataType::kFloat32 ||
            a->access != TensorAccess::kRead ||
            b->access != TensorAccess::kRead ||
            c->access != TensorAccess::kWrite ||
            a->storage_policy != TensorStoragePolicy::kStrided ||
            b->storage_policy != TensorStoragePolicy::kStrided ||
            c->storage_policy != TensorStoragePolicy::kStrided ||
            !valid_matmul_dimensions(*a, *b, *c)) {
            return fail("MATMUL override roles or maximum shapes are invalid");
        }
    }
    if (metadata.override_policy == ShapeOverridePolicy::kSdpaForward) {
        auto const role_count = metadata.override_role_uids.size();
        if (role_count < 4 || role_count > 7 ||
            metadata.arguments.size() != role_count ||
            std::set<std::int64_t>(metadata.override_role_uids.begin(),
                                   metadata.override_role_uids.end())
                    .size() != role_count) {
            return fail(
                "SDPA override metadata requires distinct Q/K/V/O and optional row-output role UIDs");
        }
        std::vector<TensorArgumentMetadata const*> roles;
        roles.reserve(role_count);
        for (auto uid : metadata.override_role_uids) {
            auto const role = std::find_if(
                metadata.arguments.begin(), metadata.arguments.end(),
                [&](TensorArgumentMetadata const& argument) {
                    return argument.uid == uid;
                });
            if (role == metadata.arguments.end()) {
                return fail("SDPA override role UID is unresolved");
            }
            roles.push_back(&*role);
        }
        bool valid_roles = valid_sdpa_dimensions(roles);
        for (std::size_t role = 0; role < roles.size(); ++role) {
            auto const* argument = roles[role];
            valid_roles =
                valid_roles &&
                argument->data_type == import::DataType::kFloat32 &&
                argument->storage_policy == TensorStoragePolicy::kStrided &&
                argument->access ==
                    (role < 3 ? TensorAccess::kRead : TensorAccess::kWrite) &&
                (role >= 4 || argument->strides[3] == 1);
        }
        if (!valid_roles) {
            return fail("SDPA override roles or maximum shapes are invalid");
        }
    }
    if (metadata.override_policy == ShapeOverridePolicy::kReshape) {
        if (metadata.arguments.size() != 2 ||
            metadata.override_role_uids.size() != 2 ||
            metadata.override_role_uids[0] ==
                metadata.override_role_uids[1]) {
            return fail(
                "RESHAPE override metadata requires distinct X/Y role UIDs");
        }
        auto find_role = [&](std::size_t role) {
            return std::find_if(
                metadata.arguments.begin(), metadata.arguments.end(),
                [&](TensorArgumentMetadata const& argument) {
                    return argument.uid == metadata.override_role_uids[role];
                });
        };
        auto const x = find_role(0);
        auto const y = find_role(1);
        if (x == metadata.arguments.end() || y == metadata.arguments.end() ||
            x->data_type != import::DataType::kFloat32 ||
            y->data_type != import::DataType::kFloat32 ||
            x->access != TensorAccess::kRead ||
            y->access != TensorAccess::kWrite ||
            x->storage_policy != TensorStoragePolicy::kStrided ||
            y->storage_policy != TensorStoragePolicy::kStrided ||
            !valid_reshape_dimensions(*x, *y)) {
            return fail(
                "RESHAPE override roles or maximum shapes are invalid");
        }
    }
    if (metadata.override_policy == ShapeOverridePolicy::kReduction) {
        if (metadata.arguments.size() != 2 ||
            metadata.override_role_uids.size() != 2 ||
            metadata.override_role_uids[0] ==
                metadata.override_role_uids[1]) {
            return fail(
                "REDUCTION override metadata requires distinct X/Y role UIDs");
        }
        auto find_role = [&](std::size_t role) {
            return std::find_if(
                metadata.arguments.begin(), metadata.arguments.end(),
                [&](TensorArgumentMetadata const& argument) {
                    return argument.uid == metadata.override_role_uids[role];
                });
        };
        auto const x = find_role(0);
        auto const y = find_role(1);
        if (x == metadata.arguments.end() || y == metadata.arguments.end() ||
            x->data_type != import::DataType::kFloat32 ||
            y->data_type != import::DataType::kFloat32 ||
            x->access != TensorAccess::kRead ||
            y->access != TensorAccess::kWrite ||
            x->storage_policy != TensorStoragePolicy::kStrided ||
            y->storage_policy != TensorStoragePolicy::kStrided ||
            !valid_reduction_dimensions(*x, *y)) {
            return fail(
                "REDUCTION override roles or maximum shapes are invalid");
        }
    }
    if (metadata.override_policy == ShapeOverridePolicy::kTranspose) {
        if (metadata.arguments.size() != 2 ||
            metadata.override_role_uids.size() != 2 ||
            metadata.override_role_uids[0] ==
                metadata.override_role_uids[1]) {
            return fail(
                "TRANSPOSE override metadata requires distinct X/Y role UIDs");
        }
        auto find_role = [&](std::size_t role) {
            return std::find_if(
                metadata.arguments.begin(), metadata.arguments.end(),
                [&](TensorArgumentMetadata const& argument) {
                    return argument.uid == metadata.override_role_uids[role];
                });
        };
        auto const x = find_role(0);
        auto const y = find_role(1);
        if (x == metadata.arguments.end() || y == metadata.arguments.end() ||
            x->data_type != import::DataType::kFloat32 ||
            y->data_type != import::DataType::kFloat32 ||
            x->access != TensorAccess::kRead ||
            y->access != TensorAccess::kWrite ||
            x->storage_policy != TensorStoragePolicy::kStrided ||
            y->storage_policy != TensorStoragePolicy::kStrided ||
            !valid_transpose_dimensions(*x, *y,
                                        metadata.override_axis_map)) {
            return fail(
                "TRANSPOSE override roles, permutation, or maximum shapes are invalid");
        }
    }
    if (metadata.override_policy == ShapeOverridePolicy::kConcatenate) {
        auto const role_count = metadata.override_role_uids.size();
        if (role_count < 2 || role_count > 64 ||
            metadata.arguments.size() != role_count ||
            metadata.override_axis_map.size() != 1 ||
            std::set<std::int64_t>(metadata.override_role_uids.begin(),
                                   metadata.override_role_uids.end())
                    .size() != role_count) {
            return fail(
                "CONCATENATE override metadata requires ordered distinct input/Y roles and one axis");
        }
        std::vector<TensorArgumentMetadata const*> roles;
        roles.reserve(role_count);
        for (auto uid : metadata.override_role_uids) {
            auto const role = std::find_if(
                metadata.arguments.begin(), metadata.arguments.end(),
                [&](TensorArgumentMetadata const& argument) {
                    return argument.uid == uid;
                });
            if (role == metadata.arguments.end()) {
                return fail("CONCATENATE override role UID is unresolved");
            }
            roles.push_back(&*role);
        }
        bool valid_roles = valid_concatenate_dimensions(
            roles, metadata.override_axis_map.front());
        for (std::size_t role = 0; role < roles.size(); ++role) {
            valid_roles =
                valid_roles &&
                roles[role]->data_type == import::DataType::kFloat32 &&
                roles[role]->storage_policy == TensorStoragePolicy::kStrided &&
                roles[role]->access ==
                    (role + 1 == roles.size() ? TensorAccess::kWrite
                                              : TensorAccess::kRead);
        }
        if (!valid_roles) {
            return fail(
                "CONCATENATE override roles, axis, or maximum shapes are invalid");
        }
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
        if ((argument.data_type != import::DataType::kFloat32 &&
             argument.data_type != import::DataType::kInt32) ||
            argument.dimensions.size() != 4 ||
            argument.ragged_sequence_divisor <= 0 ||
            argument.dimensions[2] >
                std::numeric_limits<std::int64_t>::max() /
                    argument.ragged_sequence_divisor ||
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
