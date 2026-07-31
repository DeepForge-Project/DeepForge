#include "DeepForge/Compiler/GraphMetadata.h"

#include "DeepForge/Import/Capability.h"

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

    std::set<std::int64_t> uids;
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
        if (!import::tensor_storage_bytes(argument.data_type,
                                          argument.dimensions,
                                          argument.strides, expected_size) ||
            expected_size != argument.size_bytes ||
            argument.size_bytes >
                static_cast<std::uint64_t>(
                    std::numeric_limits<std::size_t>::max())) {
            return fail("argument storage range is invalid");
        }
    }
    return import::Status::ok();
}

}  // namespace deepforge::compiler
