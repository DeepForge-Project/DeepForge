#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

namespace deepforge::import {

enum class ErrorCode : std::uint8_t {
    kOk = 0,
    kInvalidArgument = 1,
    kIoError = 2,
    kParseError = 3,
    kSchemaVersionMismatch = 4,
    kFrontendVersionMismatch = 5,
    kMissingField = 6,
    kInvalidFieldType = 7,
    kInvalidValue = 8,
    kUnsupportedNode = 9,
    kUnsupportedDataType = 10,
    kUnsupportedExecutionMetadata = 11,
    kDuplicateUid = 12,
    kMissingUid = 13,
    kInvalidLayout = 14,
    kInvalidShape = 15,
    kDimensionOverflow = 16,
    kInvalidVariantPack = 17,
    kUnsupportedCpuFeature = 18,
    kGraphExecutionFailed = 19,
};

std::string_view error_code_name(ErrorCode code) noexcept;

class Status {
public:
    Status() = default;

    static Status ok() {
        return {};
    }

    static Status failure(ErrorCode code, std::string message) {
        return Status(code, std::move(message));
    }

    [[nodiscard]] bool is_good() const noexcept {
        return code_ == ErrorCode::kOk;
    }

    [[nodiscard]] bool is_bad() const noexcept {
        return !is_good();
    }

    [[nodiscard]] ErrorCode code() const noexcept {
        return code_;
    }

    [[nodiscard]] std::string const& message() const noexcept {
        return message_;
    }

    explicit operator bool() const noexcept {
        return is_good();
    }

private:
    Status(ErrorCode code, std::string message) : code_(code), message_(std::move(message)) {}

    ErrorCode code_ = ErrorCode::kOk;
    std::string message_;
};

}  // namespace deepforge::import
