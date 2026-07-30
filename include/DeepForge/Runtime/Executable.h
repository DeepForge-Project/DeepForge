#pragma once

#include "DeepForge/Import/Status.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string_view>
#include <unordered_map>

namespace deepforge::runtime {

struct ExecutableFactory;

using FrontendHandle = void*;
using VariantPack = std::unordered_map<std::int64_t, void*>;

enum class CpuVariant : std::uint8_t {
    kScalar = 0,
    kAvx2,
    kAvx512,
};

struct CpuFeatures {
    bool avx = false;
    bool fma = false;
    bool avx2 = false;
    bool avx512f = false;
    bool os_ymm_state = false;
    bool os_zmm_state = false;
};

[[nodiscard]] std::string_view cpu_variant_name(CpuVariant variant) noexcept;
[[nodiscard]] CpuFeatures detect_cpu_features() noexcept;
[[nodiscard]] bool cpu_supports_variant(CpuFeatures const& features,
                                        CpuVariant variant) noexcept;
[[nodiscard]] CpuVariant select_cpu_variant(
    CpuFeatures const& features,
    std::array<bool, 3> const& compiled_variants) noexcept;

class Executable final {
public:
    ~Executable();

    Executable(Executable&&) noexcept;
    Executable& operator=(Executable&&) noexcept;

    Executable(Executable const&) = delete;
    Executable& operator=(Executable const&) = delete;

    [[nodiscard]] std::int64_t get_workspace_size() const noexcept;

    // Select the highest safe CPU variant and execute it. The opaque handle is
    // accepted for Frontend-shaped source compatibility and is not inspected.
    [[nodiscard]] import::Status execute(FrontendHandle handle,
                                          VariantPack& uid_to_host_ptr,
                                          void* workspace) const;

    // Used by correctness tests and diagnostics to execute a specific compiled
    // variant without changing the public dispatch contract.
    [[nodiscard]] import::Status execute_variant(
        CpuVariant variant,
        FrontendHandle handle,
        VariantPack& uid_to_host_ptr,
        void* workspace) const;

    [[nodiscard]] bool supports_variant(CpuVariant variant) const noexcept;
    [[nodiscard]] CpuVariant selected_variant() const noexcept;

private:
    struct Impl;
    explicit Executable(std::unique_ptr<Impl> impl);

    friend struct ExecutableFactory;

    std::unique_ptr<Impl> impl_;
};

}  // namespace deepforge::runtime
