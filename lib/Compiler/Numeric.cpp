#include "Numeric.h"

#include "DeepForge/Import/Capability.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string_view>

namespace deepforge::compiler::numeric {
namespace {

using import::DataType;
using import::TensorDesc;

::mlir::Value index_constant(::mlir::OpBuilder& builder,
                             ::mlir::Location location,
                             std::int64_t value) {
    return ::mlir::arith::ConstantIndexOp::create(builder, location, value);
}

::mlir::Value integer_constant(::mlir::OpBuilder& builder,
                               ::mlir::Location location,
                               std::int64_t value,
                               unsigned width) {
    return ::mlir::arith::ConstantIntOp::create(builder, location, value,
                                                width);
}

::mlir::Value float_constant(::mlir::OpBuilder& builder,
                             ::mlir::Location location,
                             float value) {
    return ::mlir::arith::ConstantFloatOp::create(
        builder, location, ::mlir::Float32Type::get(builder.getContext()),
        llvm::APFloat(value));
}

float decode_e4m3(std::uint8_t code) {
    auto const sign = (code & 0x80U) != 0;
    auto const exponent = static_cast<unsigned>((code >> 3U) & 0x0fU);
    auto const mantissa = static_cast<unsigned>(code & 0x07U);
    float value = 0.0F;
    if (exponent == 0) {
        value = std::ldexp(static_cast<float>(mantissa), -9);
    } else if (exponent == 15 && mantissa == 7) {
        return std::numeric_limits<float>::quiet_NaN();
    } else {
        value = std::ldexp(1.0F + static_cast<float>(mantissa) / 8.0F,
                           static_cast<int>(exponent) - 7);
    }
    return sign ? -value : value;
}

float decode_e5m2(std::uint8_t code) {
    auto const sign = (code & 0x80U) != 0;
    auto const exponent = static_cast<unsigned>((code >> 2U) & 0x1fU);
    auto const mantissa = static_cast<unsigned>(code & 0x03U);
    float value = 0.0F;
    if (exponent == 0) {
        value = std::ldexp(static_cast<float>(mantissa), -16);
    } else if (exponent == 31) {
        if (mantissa == 0) {
            value = std::numeric_limits<float>::infinity();
        } else {
            return std::numeric_limits<float>::quiet_NaN();
        }
    } else {
        value = std::ldexp(1.0F + static_cast<float>(mantissa) / 4.0F,
                           static_cast<int>(exponent) - 15);
    }
    return sign ? -value : value;
}

float decode_e8m0(std::uint8_t code) {
    if (code == 0xffU) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    return std::ldexp(1.0F, static_cast<int>(code) - 127);
}

float decode_e2m1(std::uint8_t code) {
    code &= 0x0fU;
    auto const sign = (code & 0x08U) != 0;
    auto const exponent = static_cast<unsigned>((code >> 1U) & 0x03U);
    auto const mantissa = static_cast<unsigned>(code & 0x01U);
    float value = 0.0F;
    if (exponent == 0) {
        value = static_cast<float>(mantissa) * 0.5F;
    } else {
        value = std::ldexp(1.0F + static_cast<float>(mantissa) * 0.5F,
                           static_cast<int>(exponent) - 1);
    }
    return sign ? -value : value;
}

struct DecodeTable {
    std::string_view symbol;
    std::size_t size;
    float (*decode)(std::uint8_t);
};

DecodeTable decode_table(DataType type) {
    switch (type) {
        case DataType::kFp8E4M3:
            return {"__deepforge_decode_fp8_e4m3", 256, decode_e4m3};
        case DataType::kFp8E5M2:
            return {"__deepforge_decode_fp8_e5m2", 256, decode_e5m2};
        case DataType::kFp8E8M0:
            return {"__deepforge_decode_fp8_e8m0", 256, decode_e8m0};
        case DataType::kFp4E2M1:
            return {"__deepforge_decode_fp4_e2m1", 16, decode_e2m1};
        default:
            return {};
    }
}

::mlir::Value get_decode_table(::mlir::OpBuilder& builder,
                               ::mlir::Location location,
                               DataType type) {
    auto const table = decode_table(type);
    auto module = builder.getInsertionBlock()
                      ->getParentOp()
                      ->getParentOfType<::mlir::ModuleOp>();
    auto f32 = ::mlir::Float32Type::get(builder.getContext());
    auto memref_type = ::mlir::MemRefType::get(
        {static_cast<std::int64_t>(table.size)}, f32);
    if (module.lookupSymbol<::mlir::memref::GlobalOp>(table.symbol) == nullptr) {
        llvm::SmallVector<float> values;
        values.reserve(table.size);
        for (std::size_t index = 0; index < table.size; ++index) {
            values.push_back(table.decode(static_cast<std::uint8_t>(index)));
        }
        auto tensor_type = ::mlir::RankedTensorType::get(
            {static_cast<std::int64_t>(table.size)}, f32);
        auto initial = ::mlir::DenseFPElementsAttr::get(tensor_type, values);
        ::mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(module.getBody());
        ::mlir::memref::GlobalOp::create(
            builder, location, table.symbol, builder.getStringAttr("private"),
            memref_type, initial, true, ::mlir::IntegerAttr());
    }
    return ::mlir::memref::GetGlobalOp::create(builder, location, memref_type,
                                                table.symbol);
}

::mlir::Value logical_offset(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    TensorDesc const& tensor,
    llvm::SmallVector<::mlir::Value> const& indices) {
    auto result = index_constant(builder, location, 0);
    for (std::size_t axis = 0; axis < indices.size(); ++axis) {
        auto contribution = ::mlir::arith::MulIOp::create(
            builder, location, indices[axis],
            index_constant(builder, location, tensor.stride[axis]));
        result = ::mlir::arith::AddIOp::create(builder, location, result,
                                               contribution);
    }
    return result;
}

::mlir::Value packed_byte_view(::mlir::OpBuilder& builder,
                               ::mlir::Location location,
                               ::mlir::Value buffer,
                               TensorDesc const& tensor) {
    std::uint64_t bytes = 0;
    (void)import::tensor_storage_bytes(tensor.data_type, tensor.dim,
                                       tensor.stride, bytes);
    auto type = ::mlir::MemRefType::get(
        {static_cast<std::int64_t>(bytes)},
        ::mlir::IntegerType::get(builder.getContext(), 8));
    return ::mlir::memref::ReinterpretCastOp::create(
        builder, location, type, buffer, 0,
        llvm::ArrayRef<std::int64_t>{static_cast<std::int64_t>(bytes)},
        llvm::ArrayRef<std::int64_t>{1});
}

::mlir::Value load_packed_code(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    ::mlir::Value buffer,
    TensorDesc const& tensor,
    llvm::SmallVector<::mlir::Value> const& indices) {
    auto slot = logical_offset(builder, location, tensor, indices);
    auto two = index_constant(builder, location, 2);
    auto byte_index = ::mlir::arith::DivUIOp::create(builder, location, slot,
                                                     two);
    auto high = ::mlir::arith::CmpIOp::create(
        builder, location, ::mlir::arith::CmpIPredicate::ne,
        ::mlir::arith::RemUIOp::create(builder, location, slot, two),
        index_constant(builder, location, 0));
    auto byte = ::mlir::memref::LoadOp::create(
        builder, location, packed_byte_view(builder, location, buffer, tensor),
        ::mlir::ValueRange{byte_index});
    auto shifted = ::mlir::arith::ShRUIOp::create(
        builder, location, byte, integer_constant(builder, location, 4, 8));
    auto nibble = ::mlir::arith::SelectOp::create(builder, location, high,
                                                  shifted, byte);
    return ::mlir::arith::AndIOp::create(
        builder, location, nibble, integer_constant(builder, location, 15, 8));
}

void store_packed_code(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    ::mlir::Value code,
    ::mlir::Value buffer,
    TensorDesc const& tensor,
    llvm::SmallVector<::mlir::Value> const& indices) {
    auto slot = logical_offset(builder, location, tensor, indices);
    auto two = index_constant(builder, location, 2);
    auto byte_index = ::mlir::arith::DivUIOp::create(builder, location, slot,
                                                     two);
    auto high = ::mlir::arith::CmpIOp::create(
        builder, location, ::mlir::arith::CmpIPredicate::ne,
        ::mlir::arith::RemUIOp::create(builder, location, slot, two),
        index_constant(builder, location, 0));
    auto view = packed_byte_view(builder, location, buffer, tensor);
    auto old = ::mlir::memref::LoadOp::create(builder, location, view,
                                               ::mlir::ValueRange{byte_index});
    code = ::mlir::arith::AndIOp::create(
        builder, location, code, integer_constant(builder, location, 15, 8));
    auto low_result = ::mlir::arith::OrIOp::create(
        builder, location,
        ::mlir::arith::AndIOp::create(
            builder, location, old, integer_constant(builder, location, 0xf0, 8)),
        code);
    auto high_result = ::mlir::arith::OrIOp::create(
        builder, location,
        ::mlir::arith::AndIOp::create(
            builder, location, old, integer_constant(builder, location, 0x0f, 8)),
        ::mlir::arith::ShLIOp::create(
            builder, location, code, integer_constant(builder, location, 4, 8)));
    auto result = ::mlir::arith::SelectOp::create(builder, location, high,
                                                   high_result, low_result);
    ::mlir::memref::StoreOp::create(builder, location, result, view,
                                    ::mlir::ValueRange{byte_index});
}

struct EncodingProperties {
    std::int64_t candidate_count = 0;
    std::int64_t positive_max_code = 0;
    std::int64_t negative_zero_code = 0;
    std::int64_t canonical_nan_code = 0;
    float maximum = 0.0F;
};

EncodingProperties encoding_properties(DataType type) {
    switch (type) {
        case DataType::kFp8E4M3:
            return {256, 0x7e, 0x80, 0x7f, 448.0F};
        case DataType::kFp8E5M2:
            return {256, 0x7b, 0x80, 0x7f, 57344.0F};
        case DataType::kFp8E8M0:
            return {255, 0xfe, 0, 0xff,
                    std::ldexp(1.0F, 127)};
        case DataType::kFp4E2M1:
            return {16, 0x07, 0x08, 0x07, 6.0F};
        default:
            return {};
    }
}

::mlir::Value code_is_valid(::mlir::OpBuilder& builder,
                            ::mlir::Location location,
                            ::mlir::Value code_index,
                            DataType type) {
    if (type == DataType::kFp4E2M1 || type == DataType::kFp8E8M0) {
        return integer_constant(builder, location, 1, 1);
    }
    auto low = ::mlir::arith::RemUIOp::create(
        builder, location, code_index, index_constant(builder, location, 128));
    auto upper = type == DataType::kFp8E4M3 ? 127 : 124;
    return ::mlir::arith::CmpIOp::create(
        builder, location, ::mlir::arith::CmpIPredicate::ult, low,
        index_constant(builder, location, upper));
}

}  // namespace

bool is_cpu_storage_type(DataType type) noexcept {
    switch (type) {
        case DataType::kFloat32:
        case DataType::kFloat64:
        case DataType::kFloat16:
        case DataType::kInt8:
        case DataType::kInt32:
        case DataType::kUInt8:
        case DataType::kBFloat16:
        case DataType::kInt64:
        case DataType::kBoolean:
        case DataType::kFp8E4M3:
        case DataType::kFp8E5M2:
        case DataType::kFastFloatForFp8:
        case DataType::kFp8E8M0:
        case DataType::kFp4E2M1:
        case DataType::kInt4:
            return true;
        default:
            return false;
    }
}

bool is_packed_type(DataType type) noexcept {
    return type == DataType::kFp4E2M1 || type == DataType::kInt4;
}

::mlir::Type storage_element_type(::mlir::MLIRContext& context,
                                  DataType type) {
    switch (type) {
        case DataType::kFloat32:
        case DataType::kFastFloatForFp8:
            return ::mlir::Float32Type::get(&context);
        case DataType::kFloat64:
            return ::mlir::Float64Type::get(&context);
        case DataType::kFloat16:
            return ::mlir::Float16Type::get(&context);
        case DataType::kBFloat16:
            return ::mlir::BFloat16Type::get(&context);
        case DataType::kInt32:
            return ::mlir::IntegerType::get(&context, 32);
        case DataType::kInt64:
            return ::mlir::IntegerType::get(&context, 64);
        case DataType::kInt8:
        case DataType::kUInt8:
        case DataType::kBoolean:
        case DataType::kFp8E4M3:
        case DataType::kFp8E5M2:
        case DataType::kFp8E8M0:
        case DataType::kFp4E2M1:
        case DataType::kInt4:
            return ::mlir::IntegerType::get(&context, 8);
        default:
            return {};
    }
}

::mlir::Value decode_low_precision(::mlir::OpBuilder& builder,
                                   ::mlir::Location location,
                                   ::mlir::Value code,
                                   DataType type) {
    auto table = get_decode_table(builder, location, type);
    auto index = ::mlir::arith::IndexCastUIOp::create(
        builder, location, builder.getIndexType(), code);
    return ::mlir::memref::LoadOp::create(builder, location, table,
                                          ::mlir::ValueRange{index});
}

::mlir::Value encode_low_precision(::mlir::OpBuilder& builder,
                                   ::mlir::Location location,
                                   ::mlir::Value value,
                                   DataType type) {
    auto const properties = encoding_properties(type);
    auto zero_index = index_constant(builder, location, 0);
    auto one_index = index_constant(builder, location, 1);
    auto upper = index_constant(builder, location,
                                properties.candidate_count);
    auto zero_code = integer_constant(builder, location, 0, 8);
    auto best_difference = ::mlir::math::AbsFOp::create(
        builder, location,
        ::mlir::arith::SubFOp::create(
            builder, location, value,
            decode_low_precision(builder, location, zero_code, type)));
    auto loop = ::mlir::scf::ForOp::create(
        builder, location, zero_index, upper, one_index,
        ::mlir::ValueRange{zero_code, best_difference});
    builder.setInsertionPointToStart(loop.getBody());
    auto candidate_index = loop.getInductionVar();
    auto candidate_code = ::mlir::arith::IndexCastUIOp::create(
        builder, location, ::mlir::IntegerType::get(builder.getContext(), 8),
        candidate_index);
    auto candidate = decode_low_precision(builder, location, candidate_code,
                                           type);
    auto difference = ::mlir::math::AbsFOp::create(
        builder, location,
        ::mlir::arith::SubFOp::create(builder, location, value, candidate));
    auto better = ::mlir::arith::CmpFOp::create(
        builder, location, ::mlir::arith::CmpFPredicate::OLT, difference,
        loop.getRegionIterArgs()[1]);
    auto equal = ::mlir::arith::CmpFOp::create(
        builder, location, ::mlir::arith::CmpFPredicate::OEQ, difference,
        loop.getRegionIterArgs()[1]);
    auto candidate_even = ::mlir::arith::CmpIOp::create(
        builder, location, ::mlir::arith::CmpIPredicate::eq,
        ::mlir::arith::AndIOp::create(
            builder, location, candidate_code,
            integer_constant(builder, location, 1, 8)),
        zero_code);
    auto best_odd = ::mlir::arith::CmpIOp::create(
        builder, location, ::mlir::arith::CmpIPredicate::ne,
        ::mlir::arith::AndIOp::create(
            builder, location, loop.getRegionIterArgs()[0],
            integer_constant(builder, location, 1, 8)),
        zero_code);
    auto tie_to_even = ::mlir::arith::AndIOp::create(
        builder, location, equal,
        ::mlir::arith::AndIOp::create(builder, location, candidate_even,
                                      best_odd));
    auto select_candidate = ::mlir::arith::AndIOp::create(
        builder, location,
        code_is_valid(builder, location, candidate_index, type),
        ::mlir::arith::OrIOp::create(builder, location, better,
                                     tie_to_even));
    auto next_code = ::mlir::arith::SelectOp::create(
        builder, location, select_candidate, candidate_code,
        loop.getRegionIterArgs()[0]);
    auto next_difference = ::mlir::arith::SelectOp::create(
        builder, location, select_candidate, difference,
        loop.getRegionIterArgs()[1]);
    ::mlir::scf::YieldOp::create(
        builder, location, ::mlir::ValueRange{next_code, next_difference});
    builder.setInsertionPointAfter(loop);

    if (type == DataType::kFp8E8M0) {
        ::mlir::Value result = loop.getResult(0);
        auto overflow = ::mlir::arith::CmpFOp::create(
            builder, location, ::mlir::arith::CmpFPredicate::OGT, value,
            float_constant(builder, location, properties.maximum));
        result = ::mlir::arith::SelectOp::create(
            builder, location, overflow,
            integer_constant(builder, location,
                             properties.positive_max_code, 8),
            result);
        auto nonpositive = ::mlir::arith::CmpFOp::create(
            builder, location, ::mlir::arith::CmpFPredicate::OLE, value,
            float_constant(builder, location, 0.0F));
        result = ::mlir::arith::SelectOp::create(
            builder, location, nonpositive, zero_code, result);
        auto is_nan = ::mlir::arith::CmpFOp::create(
            builder, location, ::mlir::arith::CmpFPredicate::UNO, value,
            value);
        return ::mlir::arith::SelectOp::create(
            builder, location, is_nan,
            integer_constant(builder, location,
                             properties.canonical_nan_code, 8),
            result);
    }

    auto absolute = ::mlir::math::AbsFOp::create(builder, location, value);
    auto bits = ::mlir::arith::BitcastOp::create(
        builder, location, ::mlir::IntegerType::get(builder.getContext(), 32),
        value);
    auto negative = ::mlir::arith::CmpIOp::create(
        builder, location, ::mlir::arith::CmpIPredicate::ne,
        ::mlir::arith::AndIOp::create(
            builder, location, bits,
            integer_constant(builder, location, 0x80000000ULL, 32)),
        integer_constant(builder, location, 0, 32));
    auto sign_mask = integer_constant(
        builder, location, type == DataType::kFp4E2M1 ? 0x08 : 0x80, 8);
    auto signed_zero = ::mlir::arith::SelectOp::create(
        builder, location, negative,
        integer_constant(builder, location, properties.negative_zero_code, 8),
        zero_code);
    auto is_zero = ::mlir::arith::CmpFOp::create(
        builder, location, ::mlir::arith::CmpFPredicate::OEQ, absolute,
        float_constant(builder, location, 0.0F));
    auto result = ::mlir::arith::SelectOp::create(builder, location, is_zero,
                                                   signed_zero,
                                                   loop.getResult(0));
    auto saturated = ::mlir::arith::OrIOp::create(
        builder, location,
        integer_constant(builder, location, properties.positive_max_code, 8),
        ::mlir::arith::SelectOp::create(builder, location, negative, sign_mask,
                                        zero_code));
    auto overflow = ::mlir::arith::CmpFOp::create(
        builder, location, ::mlir::arith::CmpFPredicate::OGT, absolute,
        float_constant(builder, location, properties.maximum));
    result = ::mlir::arith::SelectOp::create(builder, location, overflow,
                                              saturated, result);
    auto is_nan = ::mlir::arith::CmpFOp::create(
        builder, location, ::mlir::arith::CmpFPredicate::UNO, value, value);
    return ::mlir::arith::SelectOp::create(
        builder, location, is_nan,
        integer_constant(builder, location, properties.canonical_nan_code, 8),
        result);
}

::mlir::Value quantize_f32(::mlir::OpBuilder& builder,
                           ::mlir::Location location,
                           ::mlir::Value value,
                           DataType type) {
    return decode_low_precision(
        builder, location,
        encode_low_precision(builder, location, value, type), type);
}

::mlir::Value load_as_f32(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    ::mlir::Value buffer,
    TensorDesc const& tensor,
    llvm::SmallVector<::mlir::Value> const& indices) {
    if (tensor.data_type == DataType::kFp4E2M1) {
        return decode_low_precision(
            builder, location,
            load_packed_code(builder, location, buffer, tensor, indices),
            tensor.data_type);
    }
    if (tensor.data_type == DataType::kInt4) {
        auto code = load_packed_code(builder, location, buffer, tensor, indices);
        auto wide = ::mlir::arith::ExtUIOp::create(
            builder, location,
            ::mlir::IntegerType::get(builder.getContext(), 32), code);
        auto negative = ::mlir::arith::CmpIOp::create(
            builder, location, ::mlir::arith::CmpIPredicate::sge, wide,
            integer_constant(builder, location, 8, 32));
        auto signed_value = ::mlir::arith::SelectOp::create(
            builder, location, negative,
            ::mlir::arith::SubIOp::create(
                builder, location, wide,
                integer_constant(builder, location, 16, 32)),
            wide);
        return ::mlir::arith::SIToFPOp::create(
            builder, location, ::mlir::Float32Type::get(builder.getContext()),
            signed_value);
    }
    ::mlir::Value value = ::mlir::memref::LoadOp::create(
        builder, location, buffer, indices);
    switch (tensor.data_type) {
        case DataType::kFloat32:
        case DataType::kFastFloatForFp8:
            return value;
        case DataType::kFloat64:
            return ::mlir::arith::TruncFOp::create(
                builder, location,
                ::mlir::Float32Type::get(builder.getContext()), value);
        case DataType::kFloat16:
        case DataType::kBFloat16:
            return ::mlir::arith::ExtFOp::create(
                builder, location,
                ::mlir::Float32Type::get(builder.getContext()), value);
        case DataType::kInt8:
        case DataType::kInt32:
        case DataType::kInt64:
            return ::mlir::arith::SIToFPOp::create(
                builder, location,
                ::mlir::Float32Type::get(builder.getContext()), value);
        case DataType::kUInt8:
            return ::mlir::arith::UIToFPOp::create(
                builder, location,
                ::mlir::Float32Type::get(builder.getContext()), value);
        case DataType::kBoolean: {
            auto nonzero = ::mlir::arith::CmpIOp::create(
                builder, location, ::mlir::arith::CmpIPredicate::ne, value,
                integer_constant(builder, location, 0, 8));
            return ::mlir::arith::SelectOp::create(
                builder, location, nonzero,
                float_constant(builder, location, 1.0F),
                float_constant(builder, location, 0.0F));
        }
        case DataType::kFp8E4M3:
        case DataType::kFp8E5M2:
        case DataType::kFp8E8M0:
            return decode_low_precision(builder, location, value,
                                        tensor.data_type);
        default:
            return {};
    }
}

void store_from_f32(
    ::mlir::OpBuilder& builder,
    ::mlir::Location location,
    ::mlir::Value value,
    ::mlir::Value buffer,
    TensorDesc const& tensor,
    llvm::SmallVector<::mlir::Value> const& indices) {
    if (tensor.data_type == DataType::kFp4E2M1) {
        store_packed_code(
            builder, location,
            encode_low_precision(builder, location, value, tensor.data_type),
            buffer, tensor, indices);
        return;
    }
    ::mlir::Value stored = value;
    switch (tensor.data_type) {
        case DataType::kFloat32:
        case DataType::kFastFloatForFp8:
            break;
        case DataType::kFloat64:
            stored = ::mlir::arith::ExtFOp::create(
                builder, location,
                ::mlir::Float64Type::get(builder.getContext()), value);
            break;
        case DataType::kFloat16:
            stored = ::mlir::arith::TruncFOp::create(
                builder, location,
                ::mlir::Float16Type::get(builder.getContext()), value);
            break;
        case DataType::kBFloat16:
            stored = ::mlir::arith::TruncFOp::create(
                builder, location,
                ::mlir::BFloat16Type::get(builder.getContext()), value);
            break;
        case DataType::kFp8E4M3:
        case DataType::kFp8E5M2:
        case DataType::kFp8E8M0:
            stored = encode_low_precision(builder, location, value,
                                          tensor.data_type);
            break;
        case DataType::kBoolean: {
            auto nonzero = ::mlir::arith::CmpFOp::create(
                builder, location, ::mlir::arith::CmpFPredicate::UNE, value,
                float_constant(builder, location, 0.0F));
            stored = ::mlir::arith::SelectOp::create(
                builder, location, nonzero,
                integer_constant(builder, location, 1, 8),
                integer_constant(builder, location, 0, 8));
            break;
        }
        case DataType::kInt8:
        case DataType::kInt32:
        case DataType::kInt64:
            stored = ::mlir::arith::FPToSIOp::create(
                builder, location, storage_element_type(*builder.getContext(),
                                                        tensor.data_type),
                value);
            break;
        case DataType::kUInt8:
            stored = ::mlir::arith::FPToUIOp::create(
                builder, location,
                ::mlir::IntegerType::get(builder.getContext(), 8), value);
            break;
        default:
            return;
    }
    ::mlir::memref::StoreOp::create(builder, location, stored, buffer,
                                    indices);
}

}  // namespace deepforge::compiler::numeric
