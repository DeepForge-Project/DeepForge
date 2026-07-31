#include "DeepForge/Compiler/Artifact.h"
#include "DeepForge/Import/Capability.h"
#include "../Runtime/RuntimeInternal.h"

#include "llvm/TargetParser/Host.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <set>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

#include <unistd.h>

namespace deepforge::compiler {
namespace {

using import::ErrorCode;
using import::Status;

constexpr std::array<std::uint8_t, 8> kMagic{
    'D', 'F', 'O', 'B', 'J', '\r', '\n', 0x1A};
constexpr std::uint32_t kEndianMarker = 0x01020304U;
constexpr std::string_view kDeepForgeVersion = "0.1.0";
constexpr std::string_view kLlvmVersion = "22.1.8";
constexpr std::string_view kFrontendVersion = "1.24.0";
constexpr std::size_t kMaximumStringSize = 1U << 20;

Status fail(ErrorCode code, std::string_view subject, std::string detail) {
    std::string message(import::error_code_name(code));
    if (!subject.empty()) {
        message += ": ";
        message += subject;
    }
    if (!detail.empty()) {
        message += ": ";
        message += std::move(detail);
    }
    return Status::failure(code, std::move(message));
}

std::uint64_t checksum(std::span<std::uint8_t const> bytes) {
    std::uint64_t hash = 14695981039346656037ULL;
    for (auto byte : bytes) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    return hash;
}

void append_u32(std::vector<std::uint8_t>& output, std::uint32_t value) {
    for (unsigned shift = 0; shift < 32; shift += 8) {
        output.push_back(static_cast<std::uint8_t>(value >> shift));
    }
}

void append_u64(std::vector<std::uint8_t>& output, std::uint64_t value) {
    for (unsigned shift = 0; shift < 64; shift += 8) {
        output.push_back(static_cast<std::uint8_t>(value >> shift));
    }
}

void append_i64(std::vector<std::uint8_t>& output, std::int64_t value) {
    append_u64(output, std::bit_cast<std::uint64_t>(value));
}

bool append_string(std::vector<std::uint8_t>& output, std::string_view value) {
    if (value.size() > std::numeric_limits<std::uint32_t>::max()) {
        return false;
    }
    append_u32(output, static_cast<std::uint32_t>(value.size()));
    output.insert(output.end(), value.begin(), value.end());
    return true;
}

void append_blob(std::vector<std::uint8_t>& output,
                 std::span<std::uint8_t const> value) {
    append_u64(output, value.size());
    output.insert(output.end(), value.begin(), value.end());
}

template <std::size_t Size>
void append_i64_array(std::vector<std::uint8_t>& output,
                      std::array<std::int64_t, Size> const& values) {
    for (auto value : values) {
        append_i64(output, value);
    }
}

void append_i64_vector(std::vector<std::uint8_t>& output,
                       std::vector<std::int64_t> const& values) {
    for (auto value : values) {
        append_i64(output, value);
    }
}

class Reader {
public:
    explicit Reader(std::span<std::uint8_t const> input) : input_(input) {}

    bool read_u32(std::uint32_t& value) {
        if (!has(4)) {
            return false;
        }
        value = 0;
        for (unsigned index = 0; index < 4; ++index) {
            value |= static_cast<std::uint32_t>(input_[offset_++]) <<
                     (index * 8);
        }
        return true;
    }

    bool read_u64(std::uint64_t& value) {
        if (!has(8)) {
            return false;
        }
        value = 0;
        for (unsigned index = 0; index < 8; ++index) {
            value |= static_cast<std::uint64_t>(input_[offset_++]) <<
                     (index * 8);
        }
        return true;
    }

    bool read_i64(std::int64_t& value) {
        std::uint64_t bits = 0;
        if (!read_u64(bits)) {
            return false;
        }
        value = std::bit_cast<std::int64_t>(bits);
        return true;
    }

    bool read_string(std::string& value) {
        std::uint32_t size = 0;
        if (!read_u32(size) || size > kMaximumStringSize || !has(size)) {
            return false;
        }
        value.assign(reinterpret_cast<char const*>(input_.data() + offset_),
                     size);
        offset_ += size;
        return true;
    }

    bool read_blob(std::vector<std::uint8_t>& value) {
        std::uint64_t size = 0;
        if (!read_u64(size) || size > remaining()) {
            return false;
        }
        auto count = static_cast<std::size_t>(size);
        value.assign(input_.begin() + static_cast<std::ptrdiff_t>(offset_),
                     input_.begin() +
                         static_cast<std::ptrdiff_t>(offset_ + count));
        offset_ += count;
        return true;
    }

    template <std::size_t Size>
    bool read_i64_array(std::array<std::int64_t, Size>& values) {
        for (auto& value : values) {
            if (!read_i64(value)) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] std::size_t offset() const { return offset_; }
    [[nodiscard]] std::size_t remaining() const {
        return input_.size() - offset_;
    }

private:
    [[nodiscard]] bool has(std::uint64_t count) const {
        return count <= remaining();
    }

    std::span<std::uint8_t const> input_;
    std::size_t offset_ = 0;
};

void append_conv_adapter_metadata(
    std::vector<std::uint8_t>& output,
    Conv2DCompileMetadata const& metadata) {
    append_i64(output, metadata.x_uid);
    append_i64(output, metadata.w_uid);
    append_i64(output, metadata.y_uid);
    append_i64_array(output, metadata.x_shape);
    append_i64_array(output, metadata.w_shape);
    append_i64_array(output, metadata.y_shape);
    append_i64_array(output, metadata.padded_x_shape);
    append_i64_array(output, metadata.pre_padding);
    append_i64_array(output, metadata.post_padding);
    append_i64_array(output, metadata.stride);
    append_i64_array(output, metadata.dilation);
}

bool read_conv_adapter_metadata(Reader& reader,
                                Conv2DCompileMetadata& metadata) {
    return reader.read_i64(metadata.x_uid) &&
           reader.read_i64(metadata.w_uid) &&
           reader.read_i64(metadata.y_uid) &&
           reader.read_i64_array(metadata.x_shape) &&
           reader.read_i64_array(metadata.w_shape) &&
           reader.read_i64_array(metadata.y_shape) &&
           reader.read_i64_array(metadata.padded_x_shape) &&
           reader.read_i64_array(metadata.pre_padding) &&
           reader.read_i64_array(metadata.post_padding) &&
           reader.read_i64_array(metadata.stride) &&
           reader.read_i64_array(metadata.dilation);
}

bool valid_shape(std::array<std::int64_t, 4> const& shape) {
    std::uint64_t element_count = 1;
    constexpr auto kMaximumElements =
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
    for (auto dimension : shape) {
        if (dimension <= 0) {
            return false;
        }
        auto const extent = static_cast<std::uint64_t>(dimension);
        if (extent > kMaximumElements / element_count) {
            return false;
        }
        element_count *= extent;
    }
    return element_count <=
           static_cast<std::uintmax_t>(
               std::numeric_limits<std::size_t>::max() / sizeof(float));
}

std::vector<std::int64_t> contiguous_strides(
    std::array<std::int64_t, 4> const& shape) {
    std::vector<std::int64_t> strides(shape.size(), 1);
    for (std::size_t index = shape.size() - 1; index > 0; --index) {
        if (shape[index] <= 0 ||
            strides[index] > std::numeric_limits<std::int64_t>::max() /
                                 shape[index]) {
            return {};
        }
        strides[index - 1] = strides[index] * shape[index];
    }
    return strides;
}

bool append_legacy_argument(Conv2DCompileMetadata& metadata,
                            std::int64_t uid,
                            std::string name,
                            std::array<std::int64_t, 4> const& shape,
                            TensorAccess access) {
    TensorArgumentMetadata argument;
    argument.uid = uid;
    argument.name = std::move(name);
    argument.data_type = import::DataType::kFloat32;
    argument.dimensions.assign(shape.begin(), shape.end());
    argument.strides = contiguous_strides(shape);
    argument.alignment = alignof(float);
    argument.access = access;
    if (argument.strides.empty() ||
        !import::tensor_storage_bytes(argument.data_type, argument.dimensions,
                                      argument.strides,
                                      argument.size_bytes)) {
        return false;
    }
    metadata.arguments.push_back(std::move(argument));
    return true;
}

bool synthesize_legacy_arguments(Conv2DCompileMetadata& metadata) {
    metadata.arguments.clear();
    return append_legacy_argument(metadata, metadata.x_uid, "X",
                                  metadata.x_shape, TensorAccess::kRead) &&
           append_legacy_argument(metadata, metadata.w_uid, "W",
                                  metadata.w_shape, TensorAccess::kRead) &&
           append_legacy_argument(metadata, metadata.y_uid, "Y",
                                  metadata.y_shape, TensorAccess::kWrite);
}

bool validate_conv_arguments(Conv2DCompileMetadata const& metadata,
                             std::string& detail) {
    auto status = validate_graph_compile_metadata(metadata);
    if (status.is_bad()) {
        detail = status.message();
        return false;
    }
    if (metadata.arguments.size() != 3) {
        detail = "Conv adapter requires exactly three tensor arguments";
        return false;
    }
    auto const& x = metadata.arguments[0];
    auto const& w = metadata.arguments[1];
    auto const& y = metadata.arguments[2];
    auto shape_matches = [](TensorArgumentMetadata const& argument,
                            std::array<std::int64_t, 4> const& shape) {
        return argument.data_type == import::DataType::kFloat32 &&
               argument.dimensions.size() == shape.size() &&
               std::equal(argument.dimensions.begin(),
                          argument.dimensions.end(), shape.begin());
    };
    if (x.uid != metadata.x_uid || w.uid != metadata.w_uid ||
        y.uid != metadata.y_uid || x.name != "X" || w.name != "W" ||
        y.name != "Y" || x.access != TensorAccess::kRead ||
        w.access != TensorAccess::kRead || y.access != TensorAccess::kWrite ||
        !shape_matches(x, metadata.x_shape) ||
        !shape_matches(w, metadata.w_shape) ||
        !shape_matches(y, metadata.y_shape)) {
        detail = "generic argument table does not match Conv metadata";
        return false;
    }
    return true;
}

std::filesystem::path unique_temporary_path(
    std::filesystem::path const& path) {
    static std::atomic<std::uint64_t> sequence{0};
    auto temporary = path;
    temporary += ".tmp." +
                 std::to_string(static_cast<std::uint64_t>(::getpid())) +
                 "." +
                 std::to_string(sequence.fetch_add(1,
                                                   std::memory_order_relaxed));
    return temporary;
}

bool checked_add_i64(std::int64_t lhs, std::int64_t rhs,
                     std::int64_t& result) {
    if (lhs < 0 || rhs < 0 ||
        rhs > std::numeric_limits<std::int64_t>::max() - lhs) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

bool checked_add_u64(std::uint64_t lhs, std::uint64_t rhs,
                     std::uint64_t& result) {
    if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

bool is_power_of_two(std::uint64_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

bool validate_contract(
    Conv2DCompileMetadata const& metadata, WorkspacePlan const& workspace,
    std::array<VariantCode, 3> const& variants, std::string& detail) {
    if (!validate_conv_arguments(metadata, detail)) {
        return false;
    }
    if (metadata.function_name.empty() ||
        metadata.function_name.find('\0') != std::string::npos) {
        detail = "function name is empty or contains NUL";
        return false;
    }
    if (metadata.x_uid == metadata.w_uid ||
        metadata.x_uid == metadata.y_uid ||
        metadata.w_uid == metadata.y_uid) {
        detail = "X, W and Y UIDs must be distinct";
        return false;
    }
    if (!valid_shape(metadata.x_shape) || !valid_shape(metadata.w_shape) ||
        !valid_shape(metadata.y_shape) ||
        !valid_shape(metadata.padded_x_shape)) {
        detail = "all shapes must be static, positive and fit the address range";
        return false;
    }
    if (metadata.stride != std::array<std::int64_t, 2>{1, 1} ||
        metadata.dilation != std::array<std::int64_t, 2>{1, 1}) {
        detail = "stride and dilation must match the unit-stride MVP";
        return false;
    }
    for (std::size_t axis = 0; axis < 2; ++axis) {
        if (metadata.pre_padding[axis] < 0 ||
            metadata.post_padding[axis] < 0) {
            detail = "padding must be non-negative";
            return false;
        }
        std::int64_t padded = 0;
        auto input_extent = metadata.x_shape[axis + 1];
        if (!checked_add_i64(input_extent, metadata.pre_padding[axis], padded) ||
            !checked_add_i64(padded, metadata.post_padding[axis], padded) ||
            metadata.padded_x_shape[axis + 1] != padded) {
            detail = "padded input shape does not match padding metadata";
            return false;
        }
        auto filter_extent = metadata.w_shape[axis + 1];
        if (padded < filter_extent ||
            metadata.y_shape[axis + 1] != padded - filter_extent + 1) {
            detail = "output shape does not match Conv2D metadata";
            return false;
        }
    }
    if (metadata.padded_x_shape[0] != metadata.x_shape[0] ||
        metadata.padded_x_shape[3] != metadata.x_shape[3] ||
        metadata.w_shape[3] != metadata.x_shape[3] ||
        metadata.y_shape[0] != metadata.x_shape[0] ||
        metadata.y_shape[3] != metadata.w_shape[0]) {
        detail = "N/C/K dimensions are inconsistent";
        return false;
    }

    if (workspace.alignment != kWorkspaceAlignment ||
        workspace.size_bytes >
            static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
        workspace.size_bytes % kWorkspaceAlignment != 0) {
        detail = "workspace size or alignment is invalid";
        return false;
    }
    if ((workspace.size_bytes == 0) != workspace.allocations.empty()) {
        detail = "workspace size and allocation count disagree";
        return false;
    }
    std::set<std::string> allocation_names;
    std::uint64_t high_watermark = 0;
    for (auto const& allocation : workspace.allocations) {
        if (allocation.name.empty() ||
            allocation.name.find('\0') != std::string::npos ||
            !allocation_names.insert(allocation.name).second ||
            allocation.size_bytes == 0 ||
            !is_power_of_two(allocation.alignment) ||
            allocation.alignment > kWorkspaceAlignment ||
            allocation.offset % allocation.alignment != 0 ||
            allocation.live_start > allocation.live_end) {
            detail = "workspace allocation metadata is invalid";
            return false;
        }
        std::uint64_t end = 0;
        if (!checked_add_u64(allocation.offset, allocation.size_bytes, end) ||
            end > workspace.size_bytes) {
            detail = "workspace allocation is outside the planned range";
            return false;
        }
        high_watermark = std::max(high_watermark, end);
    }
    for (std::size_t lhs = 0; lhs < workspace.allocations.size(); ++lhs) {
        auto const& left = workspace.allocations[lhs];
        auto left_end = left.offset + left.size_bytes;
        for (std::size_t rhs = lhs + 1; rhs < workspace.allocations.size(); ++rhs) {
            auto const& right = workspace.allocations[rhs];
            bool lifetime_overlap = !(left.live_end < right.live_start ||
                                      right.live_end < left.live_start);
            auto right_end = right.offset + right.size_bytes;
            bool range_overlap = left.offset < right_end &&
                                 right.offset < left_end;
            if (lifetime_overlap && range_overlap) {
                detail = "simultaneously live workspace allocations overlap";
                return false;
            }
        }
    }
    std::uint64_t expected_size = 0;
    if (high_watermark != 0) {
        std::uint64_t with_padding = 0;
        if (!checked_add_u64(high_watermark, kWorkspaceAlignment - 1,
                             with_padding)) {
            detail = "workspace high watermark overflows";
            return false;
        }
        expected_size = with_padding & ~(kWorkspaceAlignment - 1);
    }
    if (workspace.size_bytes != expected_size) {
        detail = "workspace size is not the aligned high watermark";
        return false;
    }

    constexpr std::array<std::string_view, 3> kFeatures{
        "baseline", "avx2,fma", "avx512f,fma"};
    constexpr std::array<std::string_view, 3> kSuffixes{
        "_scalar", "_avx2", "_avx512"};
    for (std::size_t index = 0; index < variants.size(); ++index) {
        auto const& variant = variants[index];
        auto expected_symbol = metadata.function_name +
                               std::string(kSuffixes[index]);
        if (static_cast<std::size_t>(variant.variant) != index ||
            variant.symbol != expected_symbol ||
            variant.required_features != kFeatures[index] ||
            variant.object.empty()) {
            detail = "variant identity, symbol, feature set or object is invalid";
            return false;
        }
    }
    return true;
}

Status validate_compilation(CompilationResult const& compilation) {
    if (compilation.target_triple.empty() ||
        compilation.target_triple.find('\0') != std::string::npos ||
        compilation.metadata.function_name.empty()) {
        return fail(ErrorCode::kInvalidArgument, "artifact",
                    "target triple and function name are required");
    }
    if (!valid_shape(compilation.metadata.x_shape) ||
        !valid_shape(compilation.metadata.w_shape) ||
        !valid_shape(compilation.metadata.y_shape) ||
        !valid_shape(compilation.metadata.padded_x_shape)) {
        return fail(ErrorCode::kInvalidShape, "artifact",
                    "serialized shapes must be static, positive and fit the address range");
    }
    for (std::size_t index = 0; index < compilation.variants.size(); ++index) {
        auto const& variant = compilation.variants[index];
        if (static_cast<std::size_t>(variant.variant) != index ||
            variant.symbol.empty() || variant.required_features.empty() ||
            variant.object.empty()) {
            return fail(ErrorCode::kInvalidValue, "artifact.variant",
                        "scalar, AVX2 and AVX-512 objects are required");
        }
    }
    std::string detail;
    if (!validate_contract(compilation.metadata, compilation.workspace,
                           compilation.variants, detail)) {
        return fail(ErrorCode::kInvalidValue, "artifact.contract",
                    std::move(detail));
    }
    return Status::ok();
}

Status parse_failure(std::string detail) {
    return fail(ErrorCode::kParseError, "artifact", std::move(detail));
}

Status create_loaded_executable(ArtifactInfo const& artifact,
                                std::unique_ptr<runtime::Executable>& output) {
    auto host_triple = llvm::sys::getDefaultTargetTriple();
    if (artifact.target_triple != host_triple) {
        return fail(ErrorCode::kInvalidValue, "artifact.target",
                    "artifact target " + artifact.target_triple +
                        " does not match host " + host_triple);
    }

    std::array<std::span<std::uint8_t const>, 3> objects;
    std::array<std::string, 3> symbols;
    for (std::size_t index = 0; index < artifact.variants.size(); ++index) {
        auto const& variant = artifact.variants[index];
        objects[index] = variant.object;
        symbols[index] = variant.symbol;
    }
    return runtime::load_object_executable(
        artifact.metadata, artifact.workspace, objects, std::move(symbols),
        output);
}

}  // namespace

Status serialize_artifact(CompilationResult const& compilation,
                          std::vector<std::uint8_t>& output) {
    auto status = validate_compilation(compilation);
    if (status.is_bad()) {
        return status;
    }

    std::vector<std::uint8_t> bytes(kMagic.begin(), kMagic.end());
    append_u32(bytes, kArtifactFormatVersion);
    append_u32(bytes, kEndianMarker);
    if (!append_string(bytes, kDeepForgeVersion) ||
        !append_string(bytes, kLlvmVersion) ||
        !append_string(bytes, kFrontendVersion) ||
        !append_string(bytes, compilation.target_triple) ||
        !append_string(bytes, compilation.metadata.function_name)) {
        return fail(ErrorCode::kDimensionOverflow, "artifact.string",
                    "string is too large");
    }

    if (compilation.metadata.arguments.size() >
        std::numeric_limits<std::uint32_t>::max()) {
        return fail(ErrorCode::kDimensionOverflow, "artifact.arguments",
                    "too many tensor arguments");
    }
    append_u32(
        bytes,
        static_cast<std::uint32_t>(compilation.metadata.arguments.size()));
    for (auto const& argument : compilation.metadata.arguments) {
        append_i64(bytes, argument.uid);
        if (!append_string(bytes, argument.name)) {
            return fail(ErrorCode::kDimensionOverflow, "artifact.arguments",
                        "argument name is too large");
        }
        append_u32(bytes, static_cast<std::uint32_t>(argument.data_type));
        append_u32(bytes, static_cast<std::uint32_t>(argument.access));
        append_u64(bytes, argument.alignment);
        append_u64(bytes, argument.size_bytes);
        append_u32(bytes,
                   static_cast<std::uint32_t>(argument.dimensions.size()));
        append_i64_vector(bytes, argument.dimensions);
        append_i64_vector(bytes, argument.strides);
    }

    append_u32(
        bytes,
        static_cast<std::uint32_t>(ArtifactAdapterKind::kConv2DRankedMemref));
    std::vector<std::uint8_t> adapter_metadata;
    append_conv_adapter_metadata(adapter_metadata, compilation.metadata);
    append_blob(bytes, adapter_metadata);

    append_u64(bytes, compilation.workspace.size_bytes);
    append_u64(bytes, compilation.workspace.alignment);
    if (compilation.workspace.allocations.size() >
        std::numeric_limits<std::uint32_t>::max()) {
        return fail(ErrorCode::kDimensionOverflow, "artifact.workspace",
                    "too many workspace allocations");
    }
    append_u32(bytes,
               static_cast<std::uint32_t>(compilation.workspace.allocations.size()));
    for (auto const& allocation : compilation.workspace.allocations) {
        if (!append_string(bytes, allocation.name)) {
            return fail(ErrorCode::kDimensionOverflow, "artifact.workspace",
                        "allocation name is too large");
        }
        append_u64(bytes, allocation.offset);
        append_u64(bytes, allocation.size_bytes);
        append_u64(bytes, allocation.alignment);
        append_u64(bytes, allocation.live_start);
        append_u64(bytes, allocation.live_end);
    }

    append_u64(bytes, std::bit_cast<std::uint64_t>(kDefaultAbsoluteTolerance));
    append_u64(bytes, std::bit_cast<std::uint64_t>(kDefaultRelativeTolerance));
    append_u32(bytes, static_cast<std::uint32_t>(compilation.variants.size()));
    for (auto const& variant : compilation.variants) {
        append_u32(bytes, static_cast<std::uint32_t>(variant.variant));
        if (!append_string(bytes, variant.symbol) ||
            !append_string(bytes, variant.required_features)) {
            return fail(ErrorCode::kDimensionOverflow, "artifact.variant",
                        "variant metadata is too large");
        }
        append_blob(bytes, variant.object);
    }
    append_u64(bytes, checksum(bytes));
    output = std::move(bytes);
    return Status::ok();
}

Status parse_artifact(std::span<std::uint8_t const> input,
                      ArtifactInfo& output) {
    if (input.size() < kMagic.size() + 4 + 4 + 8) {
        return parse_failure("input is truncated");
    }
    if (!std::equal(kMagic.begin(), kMagic.end(), input.begin())) {
        return parse_failure("magic does not match DFOBJ");
    }
    auto payload = input.first(input.size() - 8);
    Reader checksum_reader(input.last(8));
    std::uint64_t stored_checksum = 0;
    if (!checksum_reader.read_u64(stored_checksum) ||
        stored_checksum != checksum(payload)) {
        return parse_failure("checksum mismatch");
    }

    Reader reader(payload.subspan(kMagic.size()));
    std::uint32_t version = 0;
    std::uint32_t endian = 0;
    if (!reader.read_u32(version) || !reader.read_u32(endian)) {
        return parse_failure("header is truncated");
    }
    if (version != kLegacyArtifactFormatVersion &&
        version != kArtifactFormatVersion) {
        return parse_failure("unsupported format version");
    }
    if (endian != kEndianMarker) {
        return parse_failure("endianness marker is invalid");
    }

    ArtifactInfo result;
    result.format_version = version;
    if (!reader.read_string(result.deepforge_version) ||
        !reader.read_string(result.llvm_version) ||
        !reader.read_string(result.frontend_version) ||
        !reader.read_string(result.target_triple) ||
        !reader.read_string(result.metadata.function_name)) {
        return parse_failure("metadata is truncated or malformed");
    }
    if (result.deepforge_version != kDeepForgeVersion ||
        result.llvm_version != kLlvmVersion ||
        result.frontend_version != kFrontendVersion) {
        return parse_failure("producer version is incompatible");
    }
    if (result.target_triple.empty() ||
        result.target_triple.find('\0') != std::string::npos) {
        return parse_failure("target triple is invalid");
    }
    if (version == kArtifactFormatVersion) {
        std::uint32_t argument_count = 0;
        if (!reader.read_u32(argument_count) || argument_count == 0 ||
            argument_count > 4096) {
            return parse_failure("argument table count is invalid");
        }
        result.metadata.arguments.reserve(argument_count);
        for (std::uint32_t index = 0; index < argument_count; ++index) {
            TensorArgumentMetadata argument;
            std::uint32_t data_type = 0;
            std::uint32_t access = 0;
            std::uint32_t rank = 0;
            if (!reader.read_i64(argument.uid) ||
                !reader.read_string(argument.name) ||
                !reader.read_u32(data_type) || !reader.read_u32(access) ||
                !reader.read_u64(argument.alignment) ||
                !reader.read_u64(argument.size_bytes) ||
                !reader.read_u32(rank) || rank == 0 || rank > 64 ||
                data_type > static_cast<std::uint32_t>(
                                import::DataType::kComplexFloat64) ||
                access >
                    static_cast<std::uint32_t>(TensorAccess::kReadWrite)) {
                return parse_failure("argument table entry is malformed");
            }
            argument.data_type = static_cast<import::DataType>(data_type);
            argument.access = static_cast<TensorAccess>(access);
            argument.dimensions.resize(rank);
            argument.strides.resize(rank);
            for (auto& dimension : argument.dimensions) {
                if (!reader.read_i64(dimension)) {
                    return parse_failure("argument dimensions are truncated");
                }
            }
            for (auto& stride : argument.strides) {
                if (!reader.read_i64(stride)) {
                    return parse_failure("argument strides are truncated");
                }
            }
            result.metadata.arguments.push_back(std::move(argument));
        }

        std::uint32_t adapter_kind = 0;
        std::vector<std::uint8_t> adapter_metadata;
        if (!reader.read_u32(adapter_kind) ||
            !reader.read_blob(adapter_metadata)) {
            return parse_failure("adapter metadata is truncated or malformed");
        }
        if (adapter_kind != static_cast<std::uint32_t>(
                                ArtifactAdapterKind::kConv2DRankedMemref)) {
            return parse_failure("adapter kind is unsupported");
        }
        result.adapter_kind = static_cast<ArtifactAdapterKind>(adapter_kind);
        Reader adapter_reader(adapter_metadata);
        if (!read_conv_adapter_metadata(adapter_reader, result.metadata) ||
            adapter_reader.remaining() != 0) {
            return parse_failure("Conv adapter metadata is malformed");
        }
    } else {
        result.adapter_kind = ArtifactAdapterKind::kConv2DRankedMemref;
        if (!read_conv_adapter_metadata(reader, result.metadata)) {
            return parse_failure("legacy Conv metadata is truncated");
        }
        if (!synthesize_legacy_arguments(result.metadata)) {
            return parse_failure(
                "legacy argument metadata cannot be reconstructed");
        }
    }

    if (!valid_shape(result.metadata.x_shape) ||
        !valid_shape(result.metadata.w_shape) ||
        !valid_shape(result.metadata.y_shape) ||
        !valid_shape(result.metadata.padded_x_shape)) {
        return parse_failure("serialized shape is invalid");
    }

    std::uint32_t allocation_count = 0;
    if (!reader.read_u64(result.workspace.size_bytes) ||
        !reader.read_u64(result.workspace.alignment) ||
        !reader.read_u32(allocation_count) ||
        allocation_count > reader.remaining() / (4 + 5 * 8)) {
        return parse_failure("workspace metadata is malformed");
    }
    result.workspace.allocations.reserve(allocation_count);
    for (std::uint32_t index = 0; index < allocation_count; ++index) {
        WorkspaceAllocation allocation;
        if (!reader.read_string(allocation.name) ||
            !reader.read_u64(allocation.offset) ||
            !reader.read_u64(allocation.size_bytes) ||
            !reader.read_u64(allocation.alignment) ||
            !reader.read_u64(allocation.live_start) ||
            !reader.read_u64(allocation.live_end)) {
            return parse_failure("workspace allocation is malformed");
        }
        result.workspace.allocations.push_back(std::move(allocation));
    }

    std::uint64_t absolute_bits = 0;
    std::uint64_t relative_bits = 0;
    std::uint32_t variant_count = 0;
    if (!reader.read_u64(absolute_bits) || !reader.read_u64(relative_bits) ||
        !reader.read_u32(variant_count) || variant_count != 3) {
        return parse_failure("numeric contract or variant count is invalid");
    }
    result.absolute_tolerance = std::bit_cast<double>(absolute_bits);
    result.relative_tolerance = std::bit_cast<double>(relative_bits);
    if (result.absolute_tolerance != kDefaultAbsoluteTolerance ||
        result.relative_tolerance != kDefaultRelativeTolerance) {
        return parse_failure("numeric contract is unsupported");
    }

    std::array<bool, 3> seen{};
    for (std::uint32_t index = 0; index < variant_count; ++index) {
        std::uint32_t id = 0;
        if (!reader.read_u32(id) || id >= result.variants.size() || seen[id]) {
            return parse_failure("variant identifier is invalid or duplicated");
        }
        seen[id] = true;
        auto& variant = result.variants[id];
        variant.variant = static_cast<runtime::CpuVariant>(id);
        if (!reader.read_string(variant.symbol) ||
            !reader.read_string(variant.required_features) ||
            !reader.read_blob(variant.object) || variant.symbol.empty() ||
            variant.required_features.empty() || variant.object.empty()) {
            return parse_failure("variant section is malformed");
        }
    }
    if (reader.remaining() != 0) {
        return parse_failure("artifact has trailing payload bytes");
    }
    std::string detail;
    if (!validate_contract(result.metadata, result.workspace, result.variants,
                           detail)) {
        return parse_failure("contract is invalid: " + detail);
    }
    output = std::move(result);
    return Status::ok();
}

Status write_artifact(std::filesystem::path const& path,
                      CompilationResult const& compilation) {
    std::vector<std::uint8_t> bytes;
    auto status = serialize_artifact(compilation, bytes);
    if (status.is_bad()) {
        return status;
    }
    if (path.empty()) {
        return fail(ErrorCode::kInvalidArgument, "artifact.path",
                    "output path is empty");
    }
    auto const temporary = unique_temporary_path(path);
    {
        std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
        if (!file) {
            return fail(ErrorCode::kIoError, temporary.string(),
                        "cannot open temporary output");
        }
        file.write(reinterpret_cast<char const*>(bytes.data()),
                   static_cast<std::streamsize>(bytes.size()));
        file.flush();
        if (!file) {
            std::error_code ignored;
            std::filesystem::remove(temporary, ignored);
            return fail(ErrorCode::kIoError, temporary.string(),
                        "failed to write artifact");
        }
    }
    std::error_code error;
    std::filesystem::rename(temporary, path, error);
    if (error) {
        std::filesystem::remove(temporary, error);
        return fail(ErrorCode::kIoError, path.string(),
                    "failed to publish artifact");
    }
    return Status::ok();
}

Status read_artifact(std::filesystem::path const& path, ArtifactInfo& output) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return fail(ErrorCode::kIoError, path.string(),
                    "cannot open artifact");
    }
    std::vector<char> raw_bytes((std::istreambuf_iterator<char>(file)),
                                std::istreambuf_iterator<char>());
    if (!file.eof() && file.fail()) {
        return fail(ErrorCode::kIoError, path.string(),
                    "failed to read artifact");
    }
    std::vector<std::uint8_t> bytes(raw_bytes.begin(), raw_bytes.end());
    return parse_artifact(bytes, output);
}

Status load_artifact_executable(
    std::span<std::uint8_t const> input,
    std::unique_ptr<runtime::Executable>& output,
    ArtifactInfo* info) {
    ArtifactInfo parsed;
    auto status = parse_artifact(input, parsed);
    if (status.is_bad()) {
        return status;
    }
    std::unique_ptr<runtime::Executable> executable;
    status = create_loaded_executable(parsed, executable);
    if (status.is_bad()) {
        return status;
    }
    output = std::move(executable);
    if (info != nullptr) {
        *info = std::move(parsed);
    }
    return Status::ok();
}

Status load_artifact_executable(
    std::filesystem::path const& path,
    std::unique_ptr<runtime::Executable>& output,
    ArtifactInfo* info) {
    ArtifactInfo parsed;
    auto status = read_artifact(path, parsed);
    if (status.is_bad()) {
        return status;
    }
    std::unique_ptr<runtime::Executable> executable;
    status = create_loaded_executable(parsed, executable);
    if (status.is_bad()) {
        return status;
    }
    output = std::move(executable);
    if (info != nullptr) {
        *info = std::move(parsed);
    }
    return Status::ok();
}

}  // namespace deepforge::compiler
