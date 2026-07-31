#pragma once

#include "DeepForge/Import/SerializedGraph.h"
#include "DeepForge/Import/Status.h"

#include <cstdint>
#include <string>
#include <vector>

namespace deepforge::compiler {

enum class TensorAccess : std::uint8_t {
    kRead,
    kWrite,
    kReadWrite,
};

struct TensorArgumentMetadata {
    std::int64_t uid = 0;
    std::string name;
    import::DataType data_type = import::DataType::kFloat32;
    std::vector<std::int64_t> dimensions;
    std::vector<std::int64_t> strides;
    std::uint64_t size_bytes = 0;
    std::uint64_t alignment = 1;
    TensorAccess access = TensorAccess::kRead;

    bool operator==(TensorArgumentMetadata const&) const = default;
};

struct GraphCompileMetadata {
    std::string function_name;
    std::vector<TensorArgumentMetadata> arguments;

    bool operator==(GraphCompileMetadata const&) const = default;
};

[[nodiscard]] import::Status validate_graph_compile_metadata(
    GraphCompileMetadata const& metadata);

}  // namespace deepforge::compiler
