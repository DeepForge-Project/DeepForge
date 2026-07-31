#include "DeepForge/Runtime/Executable.h"

#include "DeepForge/Import/Capability.h"
#include "RuntimeInternal.h"

#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"

#include "llvm/Support/Error.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
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

    std::vector<void*> argument_pointers;
    argument_pointers.reserve(impl_->metadata.arguments.size());
    std::vector<Interval> intervals;
    intervals.reserve(impl_->metadata.arguments.size());
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
        if (argument.size_bytes > std::numeric_limits<std::size_t>::max()) {
            return fail(ErrorCode::kDimensionOverflow, argument.name,
                        "tensor byte range overflows size_t");
        }
        std::uintptr_t end = 0;
        if (!checked_add(begin, static_cast<std::size_t>(argument.size_bytes),
                         end)) {
            return fail(ErrorCode::kDimensionOverflow, argument.name,
                        "tensor pointer range overflows uintptr_t");
        }
        argument_pointers.push_back(it->second);
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
            auto const& argument = impl_->metadata.arguments[index];
            descriptors.emplace_back(argument_pointers[index],
                                     argument.dimensions, argument.strides);
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
