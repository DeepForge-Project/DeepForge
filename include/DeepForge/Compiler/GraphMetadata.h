#pragma once

#include "DeepForge/Import/SerializedGraph.h"
#include "DeepForge/Import/Status.h"

#include <cstdint>
#include <string>
#include <vector>

namespace deepforge::compiler {

enum class InvocationAdapterKind : std::uint32_t {
    kConv2DRankedMemref = 0,
    kGenericRankedMemrefPointerTable = 1,
};

enum class TensorAccess : std::uint8_t {
    kRead,
    kWrite,
    kReadWrite,
};

enum class ShapeOverridePolicy : std::uint8_t {
    kNone = 0,
    kPointwiseExact,
};

enum class TensorStoragePolicy : std::uint8_t {
    kStrided = 0,
    kRaggedBatchPrefix,
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
    TensorStoragePolicy storage_policy = TensorStoragePolicy::kStrided;
    std::int64_t ragged_offset_uid = 0;
    std::int64_t ragged_sequence_uid = 0;

    bool operator==(TensorArgumentMetadata const&) const = default;
};

struct GraphCompileMetadata {
    std::string function_name;
    std::vector<TensorArgumentMetadata> arguments;
    bool dynamic_shape_enabled = false;
    bool override_shape_enabled = false;
    ShapeOverridePolicy override_policy = ShapeOverridePolicy::kNone;

    bool operator==(GraphCompileMetadata const&) const = default;
};

[[nodiscard]] import::Status validate_graph_compile_metadata(
    GraphCompileMetadata const& metadata);

}  // namespace deepforge::compiler
