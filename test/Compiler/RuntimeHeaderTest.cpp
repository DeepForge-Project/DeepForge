#include "DeepForge/Runtime/Executable.h"

#include <concepts>
#include <cstdint>

namespace {

using ExpectedExecute = deepforge::import::Status (
    deepforge::runtime::Executable::*)(
    deepforge::runtime::FrontendHandle,
    deepforge::runtime::VariantPack&,
    void*) const;

static_assert(std::same_as<decltype(&deepforge::runtime::Executable::execute),
                           ExpectedExecute>);
static_assert(std::same_as<
              decltype(&deepforge::runtime::Executable::get_workspace_size),
              std::int64_t (deepforge::runtime::Executable::*)() const noexcept>);

}  // namespace

int main() {
    return 0;
}
