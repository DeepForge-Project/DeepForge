#include "DeepForge/Runtime/Executable.h"

#include "DeepForge/Import/Capability.h"
#include "RuntimeInternal.h"

#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"

#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"

#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace deepforge::runtime {
namespace {

using deepforge::import::ErrorCode;
using deepforge::import::Status;

Status fail(ErrorCode code, std::string subject, std::string detail) {
    std::string message(deepforge::import::error_code_name(code));
    if (!subject.empty()) {
        message += ": ";
        message += std::move(subject);
    }
    if (!detail.empty()) {
        message += ": ";
        message += std::move(detail);
    }
    return Status::failure(code, std::move(message));
}

bool checked_add(std::uintptr_t lhs, std::size_t rhs,
                 std::uintptr_t& result) {
    if (rhs > std::numeric_limits<std::uintptr_t>::max() - lhs) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

struct Interval {
    std::uintptr_t begin = 0;
    std::uintptr_t end = 0;
    std::string name;
    compiler::TensorAccess access = compiler::TensorAccess::kRead;
};

bool overlaps(Interval const& lhs, Interval const& rhs) {
    return lhs.begin < rhs.end && rhs.begin < lhs.end;
}

bool writes(compiler::TensorAccess access) {
    return access != compiler::TensorAccess::kRead;
}

bool has_non_overlapping_layout(
    std::span<std::int64_t const> dimensions,
    std::span<std::int64_t const> strides) {
    if (dimensions.size() != strides.size()) return false;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> axes;
    axes.reserve(dimensions.size());
    for (std::size_t index = 0; index < dimensions.size(); ++index) {
        if (dimensions[index] <= 0 || strides[index] <= 0) return false;
        if (dimensions[index] > 1) {
            axes.emplace_back(static_cast<std::uint64_t>(strides[index]),
                              static_cast<std::uint64_t>(dimensions[index]));
        }
    }
    std::sort(axes.begin(), axes.end());
    std::uint64_t occupied_span = 1;
    for (auto const& [stride, extent] : axes) {
        if (stride < occupied_span ||
            extent - 1 >
                (std::numeric_limits<std::uint64_t>::max() - occupied_span) /
                    stride) {
            return false;
        }
        occupied_span += (extent - 1) * stride;
    }
    return true;
}

bool checked_element_count(std::span<std::int64_t const> dimensions,
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

struct ResolvedArgument {
    std::vector<std::int64_t> dimensions;
    std::vector<std::int64_t> strides;
    std::uint64_t size_bytes = 0;
};

Status resolve_overrides(
    compiler::GraphCompileMetadata const& metadata,
    OverrideUids const& override_uids,
    OverrideShapes const& override_shapes,
    OverrideStrides const& override_strides,
    std::vector<ResolvedArgument>& output) {
    if (override_uids.size() != override_shapes.size() ||
        override_uids.size() != override_strides.size()) {
        return fail(ErrorCode::kInvalidValue, "runtime.override",
                    "override UIDs, shapes, and strides must have equal counts");
    }
    if (!override_uids.empty() && !metadata.override_shape_enabled) {
        return fail(ErrorCode::kUnsupportedExecutionMetadata,
                    "runtime.override",
                    "artifact was not compiled with shape overrides enabled");
    }

    std::vector<ResolvedArgument> resolved;
    resolved.reserve(metadata.arguments.size());
    for (auto const& argument : metadata.arguments) {
        resolved.push_back(
            {argument.dimensions, argument.strides, argument.size_bytes});
    }
    std::set<std::int64_t> seen_overrides;
    for (std::size_t override_index = 0;
         override_index < override_uids.size(); ++override_index) {
        auto const uid = override_uids[override_index];
        if (!seen_overrides.insert(uid).second) {
            return fail(ErrorCode::kInvalidValue, "runtime.override",
                        "override UIDs must be unique");
        }
        auto argument_it = std::find_if(
            metadata.arguments.begin(), metadata.arguments.end(),
            [&](compiler::TensorArgumentMetadata const& argument) {
                return argument.uid == uid;
            });
        if (argument_it == metadata.arguments.end()) {
            return fail(ErrorCode::kInvalidVariantPack, "runtime.override",
                        "override UID is not an external tensor argument");
        }
        auto const argument_index = static_cast<std::size_t>(
            argument_it - metadata.arguments.begin());
        auto const& dimensions = override_shapes[override_index];
        auto const& strides = override_strides[override_index];
        if (dimensions.size() != argument_it->dimensions.size() ||
            strides.size() != argument_it->strides.size()) {
            return fail(ErrorCode::kInvalidShape, argument_it->name,
                        "override rank does not match the compiled tensor rank");
        }
        for (std::size_t axis = 0; axis < dimensions.size(); ++axis) {
            if (dimensions[axis] <= 0 ||
                dimensions[axis] > argument_it->dimensions[axis]) {
                return fail(
                    ErrorCode::kInvalidShape, argument_it->name,
                    "override dimensions must be positive and within compiled maxima");
            }
        }
        if (!has_non_overlapping_layout(dimensions, strides)) {
            return fail(ErrorCode::kInvalidLayout, argument_it->name,
                        "override strides must define a positive non-overlapping layout");
        }
        std::uint64_t size_bytes = 0;
        if (!import::tensor_storage_bytes(argument_it->data_type, dimensions,
                                          strides, size_bytes) ||
            size_bytes > argument_it->size_bytes) {
            return fail(ErrorCode::kInvalidShape, argument_it->name,
                        "override storage span exceeds the compiled tensor bound");
        }
        resolved[argument_index] = {dimensions, strides, size_bytes};
    }
    if (metadata.override_policy ==
        compiler::ShapeOverridePolicy::kPointwiseExact) {
        for (std::size_t index = 1; index < resolved.size(); ++index) {
            if (resolved[index].dimensions != resolved.front().dimensions) {
                return fail(ErrorCode::kInvalidShape, "runtime.override",
                            "all pointwise tensor shapes must remain equal");
            }
        }
    } else if (metadata.override_policy ==
               compiler::ShapeOverridePolicy::kMatmul) {
        std::array<ResolvedArgument const*, 3> roles{};
        for (std::size_t role = 0; role < roles.size(); ++role) {
            auto const argument = std::find_if(
                metadata.arguments.begin(), metadata.arguments.end(),
                [&](compiler::TensorArgumentMetadata const& candidate) {
                    return candidate.uid == metadata.override_role_uids[role];
                });
            if (argument == metadata.arguments.end()) {
                return fail(ErrorCode::kUnsupportedExecutionMetadata,
                            "runtime.override",
                            "MATMUL override role UID is unresolved");
            }
            roles[role] = &resolved[static_cast<std::size_t>(
                argument - metadata.arguments.begin())];
        }
        auto const& a = roles[0]->dimensions;
        auto const& b = roles[1]->dimensions;
        auto const& c = roles[2]->dimensions;
        if (a.size() < 2 || a.size() != b.size() || a.size() != c.size()) {
            return fail(ErrorCode::kInvalidShape, "runtime.override",
                        "MATMUL override ranks must match and be at least two");
        }
        auto const rank = a.size();
        if (a[rank - 1] != b[rank - 2] ||
            a[rank - 2] != c[rank - 2] ||
            b[rank - 1] != c[rank - 1]) {
            return fail(ErrorCode::kInvalidShape, "runtime.override",
                        "MATMUL override M/N/K dimensions are inconsistent");
        }
        for (std::size_t axis = 0; axis + 2 < rank; ++axis) {
            if ((a[axis] != b[axis] && a[axis] != 1 && b[axis] != 1) ||
                c[axis] != std::max(a[axis], b[axis])) {
                return fail(
                    ErrorCode::kInvalidShape, "runtime.override",
                    "MATMUL override batch dimensions are not broadcast compatible");
            }
        }
    } else if (metadata.override_policy ==
               compiler::ShapeOverridePolicy::kSdpaForward) {
        auto const role_count = metadata.override_role_uids.size();
        if (role_count < 4 || role_count > 7) {
            return fail(ErrorCode::kUnsupportedExecutionMetadata,
                        "runtime.override",
                        "SDPA override role metadata is malformed");
        }
        std::vector<ResolvedArgument const*> roles;
        std::vector<compiler::TensorArgumentMetadata const*> compiled_roles;
        roles.reserve(role_count);
        compiled_roles.reserve(role_count);
        for (auto uid : metadata.override_role_uids) {
            auto const argument = std::find_if(
                metadata.arguments.begin(), metadata.arguments.end(),
                [&](compiler::TensorArgumentMetadata const& candidate) {
                    return candidate.uid == uid;
                });
            if (argument == metadata.arguments.end()) {
                return fail(ErrorCode::kUnsupportedExecutionMetadata,
                            "runtime.override",
                            "SDPA override role UID is unresolved");
            }
            auto const index = static_cast<std::size_t>(
                argument - metadata.arguments.begin());
            roles.push_back(&resolved[index]);
            compiled_roles.push_back(&*argument);
        }
        for (std::size_t role = 0; role < roles.size(); ++role) {
            auto const& dimensions = roles[role]->dimensions;
            auto const& compiled = compiled_roles[role]->dimensions;
            if (dimensions.size() != 4 || dimensions[1] != compiled[1] ||
                dimensions[3] != compiled[3]) {
                return fail(
                    ErrorCode::kInvalidShape, "runtime.override",
                    "SDPA overrides may change only batch and sequence dimensions");
            }
        }
        auto const& q = roles[0]->dimensions;
        auto const& k = roles[1]->dimensions;
        auto const& v = roles[2]->dimensions;
        auto const& o = roles[3]->dimensions;
        if (q[0] != k[0] || q[0] != v[0] || q[0] != o[0] ||
            q[1] != o[1] || q[2] != o[2] || q[3] != k[3] ||
            k[2] != v[2] || o[3] != v[3] || q[1] % k[1] != 0 ||
            q[1] % v[1] != 0) {
            return fail(ErrorCode::kInvalidShape, "runtime.override",
                        "SDPA Q/K/V/O runtime dimensions are inconsistent");
        }
        std::vector<std::int64_t> const row_dimensions{q[0], q[1], q[2], 1};
        for (std::size_t role = 4; role < roles.size(); ++role) {
            if (roles[role]->dimensions != row_dimensions) {
                return fail(
                    ErrorCode::kInvalidShape, "runtime.override",
                    "SDPA row outputs must have runtime dimensions [B,Hq,Sq,1]");
            }
        }
    } else if (metadata.override_policy ==
               compiler::ShapeOverridePolicy::kReshape) {
        if (metadata.override_role_uids.size() != 2) {
            return fail(ErrorCode::kUnsupportedExecutionMetadata,
                        "runtime.override",
                        "RESHAPE override role metadata is malformed");
        }
        std::array<ResolvedArgument const*, 2> roles{};
        for (std::size_t role = 0; role < roles.size(); ++role) {
            auto const argument = std::find_if(
                metadata.arguments.begin(), metadata.arguments.end(),
                [&](compiler::TensorArgumentMetadata const& candidate) {
                    return candidate.uid == metadata.override_role_uids[role];
                });
            if (argument == metadata.arguments.end()) {
                return fail(ErrorCode::kUnsupportedExecutionMetadata,
                            "runtime.override",
                            "RESHAPE override role UID is unresolved");
            }
            roles[role] = &resolved[static_cast<std::size_t>(
                argument - metadata.arguments.begin())];
        }
        std::uint64_t x_elements = 0;
        std::uint64_t y_elements = 0;
        if (!checked_element_count(roles[0]->dimensions, x_elements) ||
            !checked_element_count(roles[1]->dimensions, y_elements) ||
            x_elements != y_elements) {
            return fail(
                ErrorCode::kInvalidShape, "runtime.override",
                "RESHAPE X and Y runtime element counts must match");
        }
    }
    output = std::move(resolved);
    return Status::ok();
}

std::int64_t load_signed_element(void const* pointer,
                                 import::DataType data_type,
                                 std::uint64_t element) {
    if (data_type == import::DataType::kInt32) {
        std::int32_t value = 0;
        std::memcpy(&value,
                    static_cast<std::uint8_t const*>(pointer) +
                        element * sizeof(value),
                    sizeof(value));
        return value;
    }
    std::int64_t value = 0;
    std::memcpy(&value,
                static_cast<std::uint8_t const*>(pointer) +
                    element * sizeof(value),
                sizeof(value));
    return value;
}

Status resolve_ragged_size(
    compiler::GraphCompileMetadata const& metadata,
    compiler::TensorArgumentMetadata const& argument,
    VariantPack const& pointers,
    std::uint64_t& size_bytes) {
    auto const find_argument = [&](std::int64_t uid) {
        return std::find_if(
            metadata.arguments.begin(), metadata.arguments.end(),
            [&](compiler::TensorArgumentMetadata const& candidate) {
                return candidate.uid == uid;
            });
    };
    auto const offset = find_argument(argument.ragged_offset_uid);
    auto const sequence = find_argument(argument.ragged_sequence_uid);
    if (offset == metadata.arguments.end() ||
        sequence == metadata.arguments.end()) {
        return fail(ErrorCode::kInvalidValue, argument.name,
                    "ragged metadata references are unresolved");
    }
    auto const offset_pointer = pointers.at(offset->uid);
    auto const sequence_pointer = pointers.at(sequence->uid);
    auto const batches = argument.dimensions[0];
    auto const maximum_sequence =
        argument.dimensions[2] * argument.ragged_sequence_divisor;
    auto read_offset = [&](std::int64_t batch) {
        return load_signed_element(
            offset_pointer, offset->data_type,
            static_cast<std::uint64_t>(batch) *
                static_cast<std::uint64_t>(offset->strides[0]));
    };
    auto read_sequence = [&](std::int64_t batch) {
        return load_signed_element(
            sequence_pointer, sequence->data_type,
            static_cast<std::uint64_t>(batch) *
                static_cast<std::uint64_t>(sequence->strides[0]));
    };

    if (read_offset(0) != 0) {
        return fail(ErrorCode::kInvalidVariantPack, argument.name,
                    "ragged offsets must start at zero");
    }
    auto const element_bytes =
        (import::data_type_storage_bits(argument.data_type) + 7) / 8;
    auto const maximum_elements = argument.size_bytes / element_bytes;
    for (std::int64_t batch = 0; batch < batches; ++batch) {
        auto const start = read_offset(batch);
        auto const end = read_offset(batch + 1);
        auto const sequence_length = read_sequence(batch);
        if (start < 0 || end < start ||
            static_cast<std::uint64_t>(end) > maximum_elements) {
            return fail(ErrorCode::kInvalidVariantPack, argument.name,
                        "ragged offsets must be monotonic and within the "
                        "compiled storage bound");
        }
        if (sequence_length < 0 || sequence_length > maximum_sequence) {
            return fail(ErrorCode::kInvalidVariantPack, argument.name,
                        "runtime sequence length exceeds the ragged logical "
                        "bound");
        }
        auto const storage_sequence =
            sequence_length / argument.ragged_sequence_divisor +
            (sequence_length % argument.ragged_sequence_divisor != 0 ? 1
                                                                     : 0);
        std::uint64_t required_span = 0;
        if (storage_sequence != 0) {
            required_span = 1;
            for (std::size_t axis = 1; axis < argument.dimensions.size();
                 ++axis) {
                auto const extent = static_cast<std::uint64_t>(
                    (axis == 2 ? storage_sequence
                               : argument.dimensions[axis]) -
                    1);
                auto const stride =
                    static_cast<std::uint64_t>(argument.strides[axis]);
                if (extent >
                    (std::numeric_limits<std::uint64_t>::max() -
                     required_span) /
                        stride) {
                    return fail(ErrorCode::kDimensionOverflow, argument.name,
                                "runtime ragged span overflows uint64");
                }
                required_span += extent * stride;
            }
        }
        if (static_cast<std::uint64_t>(end - start) < required_span) {
            return fail(ErrorCode::kInvalidVariantPack, argument.name,
                        "ragged segment is shorter than its runtime sequence "
                        "length and inner layout require");
        }
    }
    auto const endpoint = read_offset(batches);
    if (static_cast<std::uint64_t>(endpoint) >
        std::numeric_limits<std::uint64_t>::max() / element_bytes) {
        return fail(ErrorCode::kDimensionOverflow, argument.name,
                    "runtime ragged byte span overflows uint64");
    }
    size_bytes = static_cast<std::uint64_t>(endpoint) * element_bytes;
    return Status::ok();
}

Status validate_argument_table(
    compiler::InvocationAdapterKind adapter_kind,
    compiler::Conv2DCompileMetadata const& metadata) {
    auto status = compiler::validate_graph_compile_metadata(metadata);
    if (status.is_bad()) {
        return status;
    }
    if (adapter_kind ==
        compiler::InvocationAdapterKind::kGenericRankedMemrefPointerTable) {
        return Status::ok();
    }
    if (adapter_kind !=
        compiler::InvocationAdapterKind::kConv2DRankedMemref) {
        return fail(ErrorCode::kInvalidValue, "metadata.adapter",
                    "invocation adapter kind is unknown");
    }
    if (metadata.arguments.size() != 3) {
        return fail(ErrorCode::kInvalidValue, "metadata.arguments",
                    "Conv adapter requires a function name and three arguments");
    }
    if (metadata.dynamic_shape_enabled || metadata.override_shape_enabled ||
        metadata.override_policy != compiler::ShapeOverridePolicy::kNone) {
        return fail(ErrorCode::kInvalidValue, "metadata.override",
                    "Conv adapter does not support dynamic or override metadata");
    }
    for (auto const& argument : metadata.arguments) {
        if (argument.data_type != import::DataType::kFloat32 ||
            argument.dimensions.size() != 4) {
            return fail(ErrorCode::kInvalidValue, "metadata.arguments",
                        "Conv adapter requires rank-4 FLOAT arguments");
        }
    }
    auto const& x = metadata.arguments[0];
    auto const& w = metadata.arguments[1];
    auto const& y = metadata.arguments[2];
    if (x.uid != metadata.x_uid || w.uid != metadata.w_uid ||
        y.uid != metadata.y_uid || x.name != "X" || w.name != "W" ||
        y.name != "Y" || x.access != compiler::TensorAccess::kRead ||
        w.access != compiler::TensorAccess::kRead ||
        y.access != compiler::TensorAccess::kWrite ||
        !std::equal(x.dimensions.begin(), x.dimensions.end(),
                    metadata.x_shape.begin()) ||
        !std::equal(w.dimensions.begin(), w.dimensions.end(),
                    metadata.w_shape.begin()) ||
        !std::equal(y.dimensions.begin(), y.dimensions.end(),
                    metadata.y_shape.begin())) {
        return fail(ErrorCode::kInvalidValue, "metadata.arguments",
                    "argument table does not match the Conv adapter");
    }
    return Status::ok();
}

#if defined(__x86_64__) || defined(__i386__)
std::uint64_t xgetbv_zero() noexcept {
    std::uint32_t eax = 0;
    std::uint32_t edx = 0;
    asm volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return (static_cast<std::uint64_t>(edx) << 32) | eax;
}

#endif

template <int Rank>
::StridedMemRefType<float, Rank> make_descriptor(
    float* pointer, std::array<std::int64_t, Rank> const& shape) {
    ::StridedMemRefType<float, Rank> descriptor{};
    descriptor.basePtr = pointer;
    descriptor.data = pointer;
    descriptor.offset = 0;
    std::int64_t stride = 1;
    for (int index = Rank - 1; index >= 0; --index) {
        descriptor.sizes[index] = shape[index];
        descriptor.strides[index] = stride;
        stride *= shape[index];
    }
    return descriptor;
}

::StridedMemRefType<std::int8_t, 1> make_workspace_descriptor(
    void* pointer, std::int64_t size) {
    ::StridedMemRefType<std::int8_t, 1> descriptor{};
    auto* bytes = static_cast<std::int8_t*>(pointer);
    descriptor.basePtr = bytes;
    descriptor.data = bytes;
    descriptor.offset = 0;
    descriptor.sizes[0] = size;
    descriptor.strides[0] = 1;
    return descriptor;
}

class DynamicRankedMemRefDescriptor {
public:
    DynamicRankedMemRefDescriptor(
        void* pointer,
        std::span<std::int64_t const> dimensions,
        std::span<std::int64_t const> strides)
        : words_(3 + 2 * dimensions.size()) {
        static_assert(sizeof(void*) <= sizeof(std::uint64_t));
        std::memcpy(&words_[0], &pointer, sizeof(pointer));
        std::memcpy(&words_[1], &pointer, sizeof(pointer));
        words_[2] = 0;
        for (std::size_t index = 0; index < dimensions.size(); ++index) {
            words_[3 + index] = static_cast<std::uint64_t>(dimensions[index]);
            words_[3 + dimensions.size() + index] =
                static_cast<std::uint64_t>(strides[index]);
        }
    }

    [[nodiscard]] void* data() noexcept { return words_.data(); }

private:
    std::vector<std::uint64_t> words_;
};

}  // namespace

struct Executable::Impl {
    compiler::InvocationAdapterKind adapter_kind =
        compiler::InvocationAdapterKind::kConv2DRankedMemref;
    compiler::Conv2DCompileMetadata metadata;
    compiler::WorkspacePlan workspace;
    std::array<std::unique_ptr<::mlir::ExecutionEngine>, 3> engines;
    std::array<std::unique_ptr<::llvm::orc::LLJIT>, 3> object_jits;
    std::array<void*, 3> entry_points{};
    std::array<std::string, 3> symbols;
    mutable std::mutex invoke_mutex;
};

std::string_view cpu_variant_name(CpuVariant variant) noexcept {
    switch (variant) {
        case CpuVariant::kScalar:
            return "scalar";
        case CpuVariant::kAvx2:
            return "avx2";
        case CpuVariant::kAvx512:
            return "avx512";
    }
    return "unknown";
}

CpuFeatures detect_cpu_features() noexcept {
    CpuFeatures features;
#if defined(__x86_64__) || defined(__i386__)
    unsigned max_leaf = __get_cpuid_max(0, nullptr);
    unsigned eax = 0;
    unsigned ebx = 0;
    unsigned ecx = 0;
    unsigned edx = 0;
    if (max_leaf < 1 || !__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        return features;
    }
    features.avx = (ecx & bit_AVX) != 0;
    features.fma = (ecx & bit_FMA) != 0;
    if ((ecx & bit_OSXSAVE) != 0) {
        auto xcr0 = xgetbv_zero();
        features.os_ymm_state = (xcr0 & 0x6) == 0x6;
        features.os_zmm_state = (xcr0 & 0xE6) == 0xE6;
    }
    if (max_leaf >= 7) {
        __cpuid_count(7, 0, eax, ebx, ecx, edx);
        features.avx2 = (ebx & bit_AVX2) != 0;
        features.avx512f = (ebx & bit_AVX512F) != 0;
    }
#endif
    return features;
}

bool cpu_supports_variant(CpuFeatures const& features,
                          CpuVariant variant) noexcept {
    switch (variant) {
        case CpuVariant::kScalar:
            return true;
        case CpuVariant::kAvx2:
            return features.avx && features.fma && features.avx2 &&
                   features.os_ymm_state;
        case CpuVariant::kAvx512:
            return features.avx && features.fma && features.avx512f &&
                   features.os_ymm_state && features.os_zmm_state;
    }
    return false;
}

CpuVariant select_cpu_variant(
    CpuFeatures const& features,
    std::array<bool, 3> const& compiled_variants) noexcept {
    if (compiled_variants[static_cast<std::size_t>(CpuVariant::kAvx512)] &&
        cpu_supports_variant(features, CpuVariant::kAvx512)) {
        return CpuVariant::kAvx512;
    }
    if (compiled_variants[static_cast<std::size_t>(CpuVariant::kAvx2)] &&
        cpu_supports_variant(features, CpuVariant::kAvx2)) {
        return CpuVariant::kAvx2;
    }
    return CpuVariant::kScalar;
}

Executable::Executable(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

Executable::~Executable() = default;
Executable::Executable(Executable&&) noexcept = default;
Executable& Executable::operator=(Executable&&) noexcept = default;

std::int64_t Executable::get_workspace_size() const noexcept {
    if (!impl_ || impl_->workspace.size_bytes >
                     static_cast<std::uint64_t>(
                         std::numeric_limits<std::int64_t>::max())) {
        return -1;
    }
    return static_cast<std::int64_t>(impl_->workspace.size_bytes);
}

Status Executable::get_workspace_size(std::int64_t& workspace_size) const {
    if (!impl_) {
        return fail(ErrorCode::kInvalidArgument, "executable",
                    "executable is empty");
    }
    auto const size = get_workspace_size();
    if (size < 0) {
        return fail(ErrorCode::kDimensionOverflow, "workspace",
                    "workspace size does not fit int64");
    }
    workspace_size = size;
    return Status::ok();
}

Status Executable::get_workspace_size(
    FrontendHandle handle,
    std::int64_t& workspace_size,
    OverrideUids const& override_uids,
    OverrideShapes const& override_shapes,
    OverrideStrides const& override_strides) const {
    (void)handle;
    if (!impl_) {
        return fail(ErrorCode::kInvalidArgument, "executable",
                    "executable is empty");
    }
    std::vector<ResolvedArgument> resolved;
    auto status = resolve_overrides(impl_->metadata, override_uids,
                                    override_shapes, override_strides,
                                    resolved);
    if (status.is_bad()) return status;
    return get_workspace_size(workspace_size);
}

std::int64_t Executable::get_workspace_size(
    FrontendHandle handle,
    OverrideUids const& override_uids,
    OverrideShapes const& override_shapes,
    OverrideStrides const& override_strides) const {
    std::int64_t workspace_size = 0;
    (void)get_workspace_size(handle, workspace_size, override_uids,
                             override_shapes, override_strides);
    return workspace_size;
}

CpuVariant Executable::selected_variant() const noexcept {
    if (!impl_) {
        return CpuVariant::kScalar;
    }
    std::array<bool, 3> compiled{};
    for (std::size_t index = 0; index < compiled.size(); ++index) {
        compiled[index] = static_cast<bool>(impl_->engines[index]) ||
                          impl_->entry_points[index] != nullptr;
    }
    return select_cpu_variant(detect_cpu_features(), compiled);
}

bool Executable::supports_variant(CpuVariant variant) const noexcept {
    if (!impl_) {
        return false;
    }
    auto variant_index = static_cast<std::size_t>(variant);
    if (variant_index >= impl_->engines.size() ||
        (!impl_->engines[variant_index] &&
         impl_->entry_points[variant_index] == nullptr)) {
        return false;
    }
    return cpu_supports_variant(detect_cpu_features(), variant);
}

Status Executable::execute(FrontendHandle handle, VariantPack& uid_to_host_ptr,
                           void* workspace) const {
    return execute(handle, uid_to_host_ptr, workspace, {}, {}, {});
}

Status Executable::execute(
    FrontendHandle handle,
    VariantPack& uid_to_host_ptr,
    void* workspace,
    OverrideUids const& override_uids,
    OverrideShapes const& override_shapes,
    OverrideStrides const& override_strides) const {
    return execute_variant(selected_variant(), handle, uid_to_host_ptr,
                           workspace, override_uids, override_shapes,
                           override_strides);
}

Status Executable::execute_variant(CpuVariant variant, FrontendHandle handle,
                                   VariantPack& uid_to_host_ptr,
                                   void* workspace) const {
    return execute_variant(variant, handle, uid_to_host_ptr, workspace, {},
                           {}, {});
}

Status Executable::execute_variant(
    CpuVariant variant,
    FrontendHandle handle,
    VariantPack& uid_to_host_ptr,
    void* workspace,
    OverrideUids const& override_uids,
    OverrideShapes const& override_shapes,
    OverrideStrides const& override_strides) const {
    (void)handle;
    if (!impl_) {
        return fail(ErrorCode::kInvalidArgument, "executable",
                    "executable is empty");
    }
    auto variant_index = static_cast<std::size_t>(variant);
    if (variant_index >= impl_->engines.size() ||
        (!impl_->engines[variant_index] &&
         impl_->entry_points[variant_index] == nullptr)) {
        return fail(ErrorCode::kInvalidArgument, "variant",
                    std::string("compiled variant is unavailable: ") +
                        std::string(cpu_variant_name(variant)));
    }
    if (!supports_variant(variant)) {
        return fail(ErrorCode::kUnsupportedCpuFeature, "variant",
                    std::string("host CPU does not support: ") +
                        std::string(cpu_variant_name(variant)));
    }

    std::vector<ResolvedArgument> resolved;
    auto override_status = resolve_overrides(
        impl_->metadata, override_uids, override_shapes, override_strides,
        resolved);
    if (override_status.is_bad()) return override_status;

    std::vector<void*> argument_pointers;
    argument_pointers.reserve(impl_->metadata.arguments.size());
    for (auto const& argument : impl_->metadata.arguments) {
        auto it = uid_to_host_ptr.find(argument.uid);
        if (it == uid_to_host_ptr.end()) {
            return fail(ErrorCode::kInvalidVariantPack, argument.name,
                        "UID is absent from variant pack");
        }
        if (it->second == nullptr) {
            return fail(ErrorCode::kInvalidVariantPack, argument.name,
                        "variant-pack pointer is null");
        }
        auto begin = reinterpret_cast<std::uintptr_t>(it->second);
        if (begin % argument.alignment != 0) {
            return fail(ErrorCode::kInvalidVariantPack, argument.name,
                        "tensor pointer does not meet required alignment");
        }
        argument_pointers.push_back(it->second);
    }

    std::vector<Interval> intervals;
    intervals.reserve(impl_->metadata.arguments.size());
    for (std::size_t argument_index = 0;
         argument_index < impl_->metadata.arguments.size();
         ++argument_index) {
        auto const& argument = impl_->metadata.arguments[argument_index];
        auto size_bytes = resolved[argument_index].size_bytes;
        if (argument.storage_policy ==
            compiler::TensorStoragePolicy::kRaggedBatchPrefix) {
            auto status = resolve_ragged_size(impl_->metadata, argument,
                                              uid_to_host_ptr, size_bytes);
            if (status.is_bad()) return status;
            resolved[argument_index].size_bytes = size_bytes;
        }
        if (size_bytes > std::numeric_limits<std::size_t>::max()) {
            return fail(ErrorCode::kDimensionOverflow, argument.name,
                        "tensor byte range overflows size_t");
        }
        auto const begin = reinterpret_cast<std::uintptr_t>(
            argument_pointers[argument_index]);
        std::uintptr_t end = 0;
        if (!checked_add(begin, static_cast<std::size_t>(size_bytes),
                         end)) {
            return fail(ErrorCode::kDimensionOverflow, argument.name,
                        "tensor pointer range overflows uintptr_t");
        }
        intervals.push_back(
            {begin, end, argument.name, argument.access});
    }
    for (std::size_t lhs = 0; lhs < intervals.size(); ++lhs) {
        for (std::size_t rhs = lhs + 1; rhs < intervals.size(); ++rhs) {
            if (overlaps(intervals[lhs], intervals[rhs]) &&
                (writes(intervals[lhs].access) ||
                 writes(intervals[rhs].access))) {
                return fail(ErrorCode::kInvalidVariantPack, "runtime.alias",
                            intervals[lhs].name + " overlaps " +
                                intervals[rhs].name);
            }
        }
    }

    if (impl_->workspace.size_bytes != 0) {
        if (impl_->workspace.alignment == 0 ||
            (impl_->workspace.alignment & (impl_->workspace.alignment - 1)) != 0) {
            return fail(ErrorCode::kInvalidValue, "workspace",
                        "compiled workspace alignment is invalid");
        }
        if (workspace == nullptr) {
            return fail(ErrorCode::kInvalidVariantPack, "workspace",
                        "workspace pointer is null");
        }
        auto workspace_address = reinterpret_cast<std::uintptr_t>(workspace);
        if (workspace_address % impl_->workspace.alignment != 0) {
            return fail(ErrorCode::kInvalidVariantPack, "workspace",
                        "workspace pointer does not meet required alignment");
        }
        if (impl_->workspace.size_bytes > std::numeric_limits<std::size_t>::max()) {
            return fail(ErrorCode::kDimensionOverflow, "workspace",
                        "workspace size overflows size_t");
        }
        std::uintptr_t workspace_end = 0;
        if (!checked_add(workspace_address,
                         static_cast<std::size_t>(impl_->workspace.size_bytes),
                         workspace_end)) {
            return fail(ErrorCode::kDimensionOverflow, "workspace",
                        "workspace pointer range overflows uintptr_t");
        }
        Interval workspace_interval{workspace_address, workspace_end,
                                    "workspace",
                                    compiler::TensorAccess::kReadWrite};
        for (auto const& interval : intervals) {
            if (overlaps(interval, workspace_interval)) {
                return fail(ErrorCode::kInvalidVariantPack, "runtime.alias",
                            interval.name + " overlaps workspace");
            }
        }
    }

    std::lock_guard lock(impl_->invoke_mutex);
    if (impl_->adapter_kind ==
        compiler::InvocationAdapterKind::kGenericRankedMemrefPointerTable) {
        if (impl_->entry_points[variant_index] == nullptr) {
            return fail(ErrorCode::kGraphExecutionFailed, "runtime.execute",
                        "generic adapter requires an object entry point");
        }
        std::vector<DynamicRankedMemRefDescriptor> descriptors;
        descriptors.reserve(impl_->metadata.arguments.size() + 1);
        for (std::size_t index = 0; index < impl_->metadata.arguments.size();
             ++index) {
            descriptors.emplace_back(argument_pointers[index],
                                     resolved[index].dimensions,
                                     resolved[index].strides);
        }
        std::array<std::int64_t, 1> workspace_dimensions{
            static_cast<std::int64_t>(impl_->workspace.size_bytes)};
        constexpr std::array<std::int64_t, 1> workspace_strides{1};
        descriptors.emplace_back(workspace, workspace_dimensions,
                                 workspace_strides);
        std::vector<void*> descriptor_table;
        descriptor_table.reserve(descriptors.size());
        for (auto& descriptor : descriptors) {
            descriptor_table.push_back(descriptor.data());
        }
        using EntryPoint = void (*)(void**);
        auto entry = reinterpret_cast<EntryPoint>(
            impl_->entry_points[variant_index]);
        entry(descriptor_table.data());
        return Status::ok();
    }

    auto x = make_descriptor<4>(static_cast<float*>(argument_pointers[0]),
                                impl_->metadata.x_shape);
    auto w = make_descriptor<4>(static_cast<float*>(argument_pointers[1]),
                                impl_->metadata.w_shape);
    auto y = make_descriptor<4>(static_cast<float*>(argument_pointers[2]),
                                impl_->metadata.y_shape);
    auto workspace_descriptor = make_workspace_descriptor(
        workspace, static_cast<std::int64_t>(impl_->workspace.size_bytes));

    if (impl_->entry_points[variant_index] != nullptr) {
        if (impl_->workspace.size_bytes == 0) {
            using EntryPoint = void (*)(void*, void*, void*);
            auto entry = reinterpret_cast<EntryPoint>(
                impl_->entry_points[variant_index]);
            entry(&x, &w, &y);
        } else {
            using EntryPoint = void (*)(void*, void*, void*, void*);
            auto entry = reinterpret_cast<EntryPoint>(
                impl_->entry_points[variant_index]);
            entry(&x, &w, &y, &workspace_descriptor);
        }
    } else {
        llvm::Error error = llvm::Error::success();
        if (impl_->workspace.size_bytes == 0) {
            error = impl_->engines[variant_index]->invoke(
                impl_->symbols[variant_index], &x, &w, &y);
        } else {
            error = impl_->engines[variant_index]->invoke(
                impl_->symbols[variant_index], &x, &w, &y,
                &workspace_descriptor);
        }
        if (error) {
            return fail(ErrorCode::kGraphExecutionFailed, "runtime.execute",
                        llvm::toString(std::move(error)));
        }
    }
    return Status::ok();
}

Status ExecutableFactory::create(
    compiler::Conv2DCompileMetadata const& metadata,
    compiler::WorkspacePlan const& workspace,
    std::vector<std::unique_ptr<::mlir::ExecutionEngine>> engines,
    std::array<std::string, 3> symbols,
    std::unique_ptr<Executable>& output) {
    auto metadata_status = validate_argument_table(
        compiler::InvocationAdapterKind::kConv2DRankedMemref, metadata);
    if (metadata_status.is_bad()) {
        return metadata_status;
    }
    if (engines.size() != 3) {
        return fail(ErrorCode::kInvalidArgument, "engines",
                    "expected scalar, AVX2 and AVX-512 slots");
    }
    auto impl = std::make_unique<Executable::Impl>();
    impl->adapter_kind =
        compiler::InvocationAdapterKind::kConv2DRankedMemref;
    impl->metadata = metadata;
    impl->workspace = workspace;
    impl->symbols = std::move(symbols);
    for (std::size_t index = 0; index < engines.size(); ++index) {
        impl->engines[index] = std::move(engines[index]);
    }
    if (!impl->engines[static_cast<std::size_t>(CpuVariant::kScalar)]) {
        return fail(ErrorCode::kInvalidValue, "engines",
                    "scalar engine is required");
    }
    if (impl->symbols[static_cast<std::size_t>(CpuVariant::kScalar)].empty()) {
        return fail(ErrorCode::kInvalidValue, "symbols",
                    "scalar entry symbol is required");
    }
    output = std::unique_ptr<Executable>(new Executable(std::move(impl)));
    return Status::ok();
}

Status ExecutableFactory::create_object(
    compiler::InvocationAdapterKind adapter_kind,
    compiler::Conv2DCompileMetadata const& metadata,
    compiler::WorkspacePlan const& workspace,
    std::array<std::unique_ptr<::llvm::orc::LLJIT>, 3> object_jits,
    std::array<void*, 3> entry_points,
    std::array<std::string, 3> symbols,
    std::unique_ptr<Executable>& output) {
    auto metadata_status = validate_argument_table(adapter_kind, metadata);
    if (metadata_status.is_bad()) {
        return metadata_status;
    }
    for (std::size_t index = 0; index < entry_points.size(); ++index) {
        if (!object_jits[index] || entry_points[index] == nullptr ||
            symbols[index].empty()) {
            return fail(ErrorCode::kInvalidValue, "artifact.variant",
                        "every object variant requires a JIT, entry and symbol");
        }
    }
    auto impl = std::make_unique<Executable::Impl>();
    impl->adapter_kind = adapter_kind;
    impl->metadata = metadata;
    impl->workspace = workspace;
    impl->object_jits = std::move(object_jits);
    impl->entry_points = entry_points;
    impl->symbols = std::move(symbols);
    output = std::unique_ptr<Executable>(new Executable(std::move(impl)));
    return Status::ok();
}

Status create_executable(
    compiler::Conv2DCompileMetadata const& metadata,
    compiler::WorkspacePlan const& workspace,
    std::vector<std::unique_ptr<::mlir::ExecutionEngine>> engines,
    std::array<std::string, 3> symbols,
    std::unique_ptr<Executable>& output) {
    return ExecutableFactory::create(metadata, workspace, std::move(engines),
                                     std::move(symbols), output);
}

Status create_object_executable(
    compiler::InvocationAdapterKind adapter_kind,
    compiler::Conv2DCompileMetadata const& metadata,
    compiler::WorkspacePlan const& workspace,
    std::array<std::unique_ptr<::llvm::orc::LLJIT>, 3> object_jits,
    std::array<void*, 3> entry_points,
    std::array<std::string, 3> symbols,
    std::unique_ptr<Executable>& output) {
    return ExecutableFactory::create_object(
        adapter_kind, metadata, workspace, std::move(object_jits), entry_points,
        std::move(symbols), output);
}

Status load_object_executable(
    compiler::InvocationAdapterKind adapter_kind,
    compiler::Conv2DCompileMetadata const& metadata,
    compiler::WorkspacePlan const& workspace,
    std::array<std::span<std::uint8_t const>, 3> objects,
    std::array<std::string, 3> symbols,
    std::unique_ptr<Executable>& output) {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    std::array<std::unique_ptr<llvm::orc::LLJIT>, 3> object_jits;
    std::array<void*, 3> entry_points{};
    for (std::size_t index = 0; index < objects.size(); ++index) {
        if (objects[index].empty() || symbols[index].empty()) {
            return fail(ErrorCode::kInvalidValue, "runtime.object",
                        "every CPU variant requires object bytes and a symbol");
        }
        auto jit_or_error = llvm::orc::LLJITBuilder().create();
        if (!jit_or_error) {
            return fail(ErrorCode::kGraphExecutionFailed, "runtime.jit",
                        llvm::toString(jit_or_error.takeError()));
        }
        auto jit = std::move(*jit_or_error);
        auto generator_or_error =
            llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
                jit->getDataLayout().getGlobalPrefix());
        if (!generator_or_error) {
            return fail(ErrorCode::kGraphExecutionFailed, "runtime.jit",
                        llvm::toString(generator_or_error.takeError()));
        }
        jit->getMainJITDylib().addGenerator(
            std::move(*generator_or_error));
        auto object = llvm::MemoryBuffer::getMemBufferCopy(
            llvm::StringRef(
                reinterpret_cast<char const*>(objects[index].data()),
                objects[index].size()),
            symbols[index] + ".o");
        if (auto error = jit->addObjectFile(std::move(object))) {
            return fail(ErrorCode::kParseError, "runtime.object",
                        llvm::toString(std::move(error)));
        }
        auto address_or_error =
            jit->lookup("_mlir_ciface_" + symbols[index]);
        if (!address_or_error) {
            return fail(ErrorCode::kParseError, "runtime.symbol",
                        llvm::toString(address_or_error.takeError()));
        }
        entry_points[index] = address_or_error->toPtr<void*>();
        object_jits[index] = std::move(jit);
    }
    return create_object_executable(
        adapter_kind, metadata, workspace, std::move(object_jits), entry_points,
        std::move(symbols), output);
}

}  // namespace deepforge::runtime
