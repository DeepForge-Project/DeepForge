#include "DeepForge/Runtime/Executable.h"

#include <concepts>
#include <cstdint>

namespace {

using ExpectedExecute = deepforge::import::Status (
    deepforge::runtime::Executable::*)(
    deepforge::runtime::FrontendHandle,
    deepforge::runtime::VariantPack&,
    void*) const;

using ExpectedOverrideExecute = deepforge::import::Status (
    deepforge::runtime::Executable::*)(
    deepforge::runtime::FrontendHandle,
    deepforge::runtime::VariantPack&,
    void*,
    deepforge::runtime::OverrideUids const&,
    deepforge::runtime::OverrideShapes const&,
    deepforge::runtime::OverrideStrides const&) const;

using ExpectedWorkspaceSize = std::int64_t (
    deepforge::runtime::Executable::*)() const noexcept;

using ExpectedWorkspaceStatus = deepforge::import::Status (
    deepforge::runtime::Executable::*)(std::int64_t&) const;

using ExpectedOverrideWorkspaceSize = deepforge::import::Status (
    deepforge::runtime::Executable::*)(
    deepforge::runtime::FrontendHandle,
    std::int64_t&,
    deepforge::runtime::OverrideUids const&,
    deepforge::runtime::OverrideShapes const&,
    deepforge::runtime::OverrideStrides const&) const;

using ExpectedOverrideWorkspaceValue = std::int64_t (
    deepforge::runtime::Executable::*)(
    deepforge::runtime::FrontendHandle,
    deepforge::runtime::OverrideUids const&,
    deepforge::runtime::OverrideShapes const&,
    deepforge::runtime::OverrideStrides const&) const;

static_assert(std::same_as<
              decltype(static_cast<ExpectedExecute>(
                  &deepforge::runtime::Executable::execute)),
              ExpectedExecute>);
static_assert(std::same_as<
              decltype(static_cast<ExpectedOverrideExecute>(
                  &deepforge::runtime::Executable::execute)),
              ExpectedOverrideExecute>);
static_assert(std::same_as<
              decltype(static_cast<ExpectedWorkspaceSize>(
                  &deepforge::runtime::Executable::get_workspace_size)),
              ExpectedWorkspaceSize>);
static_assert(std::same_as<
              decltype(static_cast<ExpectedWorkspaceStatus>(
                  &deepforge::runtime::Executable::get_workspace_size)),
              ExpectedWorkspaceStatus>);
static_assert(std::same_as<
              decltype(static_cast<ExpectedOverrideWorkspaceSize>(
                  &deepforge::runtime::Executable::get_workspace_size)),
              ExpectedOverrideWorkspaceSize>);
static_assert(std::same_as<
              decltype(static_cast<ExpectedOverrideWorkspaceValue>(
                  &deepforge::runtime::Executable::get_workspace_size)),
              ExpectedOverrideWorkspaceValue>);

}  // namespace

int main() {
    return 0;
}
