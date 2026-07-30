#include "DeepForge/Runtime/Executable.h"

#include "RuntimeInternal.h"

#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"

#include "llvm/Support/Error.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"

#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <string>
#include <utility>

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

bool checked_product(std::array<std::int64_t, 4> const& shape,
                     std::size_t& bytes) {
    std::uint64_t elements = 1;
    for (auto dimension : shape) {
        if (dimension <= 0 ||
            static_cast<std::uint64_t>(dimension) >
                std::numeric_limits<std::uint64_t>::max() / elements) {
            return false;
        }
        elements *= static_cast<std::uint64_t>(dimension);
    }
    if (elements > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        return false;
    }
    bytes = static_cast<std::size_t>(elements * sizeof(float));
    return true;
}

struct Interval {
    std::uintptr_t begin = 0;
    std::uintptr_t end = 0;
    std::string name;
};

bool overlaps(Interval const& lhs, Interval const& rhs) {
    return lhs.begin < rhs.end && rhs.begin < lhs.end;
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

}  // namespace

struct Executable::Impl {
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
    (void)handle;
    return execute_variant(selected_variant(), handle, uid_to_host_ptr,
                           workspace);
}

Status Executable::execute_variant(CpuVariant variant, FrontendHandle handle,
                                   VariantPack& uid_to_host_ptr,
                                   void* workspace) const {
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

    auto find_pointer = [&](std::int64_t uid,
                            std::string const& name) -> std::pair<void*, Status> {
        auto it = uid_to_host_ptr.find(uid);
        if (it == uid_to_host_ptr.end()) {
            return {nullptr, fail(ErrorCode::kInvalidVariantPack, name,
                                  "UID is absent from variant pack")};
        }
        if (it->second == nullptr) {
            return {nullptr, fail(ErrorCode::kInvalidVariantPack, name,
                                  "variant-pack pointer is null")};
        }
        if (reinterpret_cast<std::uintptr_t>(it->second) % alignof(float) != 0) {
            return {nullptr, fail(ErrorCode::kInvalidVariantPack, name,
                                  "tensor pointer is not float-aligned")};
        }
        return {it->second, Status::ok()};
    };

    auto x_result = find_pointer(impl_->metadata.x_uid, "X");
    if (x_result.second.is_bad()) {
        return x_result.second;
    }
    auto w_result = find_pointer(impl_->metadata.w_uid, "W");
    if (w_result.second.is_bad()) {
        return w_result.second;
    }
    auto y_result = find_pointer(impl_->metadata.y_uid, "Y");
    if (y_result.second.is_bad()) {
        return y_result.second;
    }

    std::size_t x_bytes = 0;
    std::size_t w_bytes = 0;
    std::size_t y_bytes = 0;
    if (!checked_product(impl_->metadata.x_shape, x_bytes) ||
        !checked_product(impl_->metadata.w_shape, w_bytes) ||
        !checked_product(impl_->metadata.y_shape, y_bytes)) {
        return fail(ErrorCode::kDimensionOverflow, "runtime.tensor",
                    "tensor byte range overflows size_t");
    }

    std::vector<Interval> intervals;
    intervals.reserve(4);
    for (auto const& item : std::array<std::pair<char const*,
                                                 std::pair<void*, std::size_t>>,
                                         3>{{{"X", {x_result.first, x_bytes}},
                                             {"W", {w_result.first, w_bytes}},
                                             {"Y", {y_result.first, y_bytes}}}}) {
        auto begin = reinterpret_cast<std::uintptr_t>(item.second.first);
        std::uintptr_t end = 0;
        if (!checked_add(begin, item.second.second, end)) {
            return fail(ErrorCode::kDimensionOverflow, item.first,
                        "tensor pointer range overflows uintptr_t");
        }
        intervals.push_back({begin, end, item.first});
    }
    for (std::size_t lhs = 0; lhs < intervals.size(); ++lhs) {
        for (std::size_t rhs = lhs + 1; rhs < intervals.size(); ++rhs) {
            if (overlaps(intervals[lhs], intervals[rhs])) {
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
                                    "workspace"};
        for (auto const& interval : intervals) {
            if (overlaps(interval, workspace_interval)) {
                return fail(ErrorCode::kInvalidVariantPack, "runtime.alias",
                            interval.name + " overlaps workspace");
            }
        }
    }

    auto x = make_descriptor<4>(static_cast<float*>(x_result.first),
                                impl_->metadata.x_shape);
    auto w = make_descriptor<4>(static_cast<float*>(w_result.first),
                                impl_->metadata.w_shape);
    auto y = make_descriptor<4>(static_cast<float*>(y_result.first),
                                impl_->metadata.y_shape);
    auto workspace_descriptor = make_workspace_descriptor(
        workspace, static_cast<std::int64_t>(impl_->workspace.size_bytes));

    std::lock_guard lock(impl_->invoke_mutex);
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
    if (engines.size() != 3) {
        return fail(ErrorCode::kInvalidArgument, "engines",
                    "expected scalar, AVX2 and AVX-512 slots");
    }
    auto impl = std::make_unique<Executable::Impl>();
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
    compiler::Conv2DCompileMetadata const& metadata,
    compiler::WorkspacePlan const& workspace,
    std::array<std::unique_ptr<::llvm::orc::LLJIT>, 3> object_jits,
    std::array<void*, 3> entry_points,
    std::array<std::string, 3> symbols,
    std::unique_ptr<Executable>& output) {
    for (std::size_t index = 0; index < entry_points.size(); ++index) {
        if (!object_jits[index] || entry_points[index] == nullptr ||
            symbols[index].empty()) {
            return fail(ErrorCode::kInvalidValue, "artifact.variant",
                        "every object variant requires a JIT, entry and symbol");
        }
    }
    auto impl = std::make_unique<Executable::Impl>();
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
    compiler::Conv2DCompileMetadata const& metadata,
    compiler::WorkspacePlan const& workspace,
    std::array<std::unique_ptr<::llvm::orc::LLJIT>, 3> object_jits,
    std::array<void*, 3> entry_points,
    std::array<std::string, 3> symbols,
    std::unique_ptr<Executable>& output) {
    return ExecutableFactory::create_object(
        metadata, workspace, std::move(object_jits), entry_points,
        std::move(symbols), output);
}

}  // namespace deepforge::runtime
