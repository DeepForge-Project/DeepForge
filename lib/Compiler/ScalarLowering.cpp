#include "DeepForge/Compiler/Lowering.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/LinalgToStandard/LinalgToStandard.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MathToLibm/MathToLibm.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace deepforge::compiler {
namespace {

using deepforge::import::ErrorCode;
using deepforge::import::Status;

Status fail(ErrorCode code, std::string_view subject, std::string detail) {
    std::string message(deepforge::import::error_code_name(code));
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

Status invalid_ir(std::string detail) {
    return fail(ErrorCode::kInvalidValue, "mlir", std::move(detail));
}

bool is_supported_variant(runtime::CpuVariant variant) {
    return variant == runtime::CpuVariant::kScalar ||
           variant == runtime::CpuVariant::kAvx2 ||
           variant == runtime::CpuVariant::kAvx512;
}

mlir::Value make_index(mlir::OpBuilder& builder, mlir::Location loc,
                       std::int64_t value) {
    return mlir::arith::ConstantIndexOp::create(builder, loc, value);
}

mlir::Value make_zero(mlir::OpBuilder& builder, mlir::Location loc,
                      mlir::FloatType type) {
    return mlir::arith::ConstantFloatOp::create(builder, loc, type,
                                                llvm::APFloat(0.0F));
}

Status replace_memref_copies(mlir::ModuleOp module) {
    llvm::SmallVector<mlir::memref::CopyOp> copies;
    module.walk([&](mlir::memref::CopyOp copy) { copies.push_back(copy); });
    for (auto copy : copies) {
        mlir::OpBuilder builder(copy);
        mlir::linalg::makeMemRefCopyOp(builder, copy.getLoc(), copy.getSource(),
                                       copy.getTarget());
        copy.erase();
    }
    return Status::ok();
}

Status lower_scalar_conv(mlir::func::FuncOp function,
                         Conv2DCompileMetadata const& metadata) {
    llvm::SmallVector<mlir::linalg::Conv2DNhwcFhwcOp> convs;
    function.walk([&](mlir::linalg::Conv2DNhwcFhwcOp conv) {
        convs.push_back(conv);
    });
    if (convs.size() != 1) {
        return invalid_ir("expected exactly one named Conv2D operation");
    }
    auto conv = convs.front();
    if (conv.getInputs().size() != 2 || conv.getOutputs().size() != 1) {
        return invalid_ir("Conv2D operand arity is not supported");
    }
    if (metadata.stride != std::array<std::int64_t, 2>{1, 1} ||
        metadata.dilation != std::array<std::int64_t, 2>{1, 1}) {
        return fail(ErrorCode::kInvalidValue, "mlir.conv2d",
                    "scalar lowering requires unit stride and dilation");
    }

    auto input = conv.getInputs()[0];
    auto weight = conv.getInputs()[1];
    auto output = conv.getOutputs()[0];
    auto input_type = llvm::dyn_cast<mlir::MemRefType>(input.getType());
    auto weight_type = llvm::dyn_cast<mlir::MemRefType>(weight.getType());
    auto output_type = llvm::dyn_cast<mlir::MemRefType>(output.getType());
    if (!input_type || !weight_type || !output_type ||
        !input_type.hasStaticShape() || !weight_type.hasStaticShape() ||
        !output_type.hasStaticShape() || input_type.getRank() != 4 ||
        weight_type.getRank() != 4 || output_type.getRank() != 4 ||
        input_type.getElementType() != output_type.getElementType() ||
        input_type.getElementType() != weight_type.getElementType() ||
        !input_type.getElementType().isF32()) {
        return fail(ErrorCode::kInvalidShape, "mlir.conv2d",
                    "expected static rank-4 f32 memrefs");
    }
    auto const& x_shape = input_type.getShape();
    auto const& w_shape = weight_type.getShape();
    auto const& y_shape = output_type.getShape();
    if (x_shape[0] != metadata.padded_x_shape[0] ||
        x_shape[1] != metadata.padded_x_shape[1] ||
        x_shape[2] != metadata.padded_x_shape[2] ||
        x_shape[3] != metadata.padded_x_shape[3] ||
        w_shape[0] != metadata.w_shape[0] ||
        w_shape[1] != metadata.w_shape[1] ||
        w_shape[2] != metadata.w_shape[2] ||
        w_shape[3] != metadata.w_shape[3] ||
        y_shape[0] != metadata.y_shape[0] ||
        y_shape[1] != metadata.y_shape[1] ||
        y_shape[2] != metadata.y_shape[2] ||
        y_shape[3] != metadata.y_shape[3]) {
        return fail(ErrorCode::kInvalidShape, "mlir.conv2d",
                    "Conv2D memref shapes disagree with compile metadata");
    }

    mlir::OpBuilder builder(conv);
    auto loc = conv.getLoc();
    auto f32 = mlir::Float32Type::get(function.getContext());
    auto zero = make_zero(builder, loc, f32);
    auto c0 = make_index(builder, loc, 0);
    auto c1 = make_index(builder, loc, 1);
    llvm::SmallVector<mlir::Value> outer_lbs(4, c0);
    llvm::SmallVector<mlir::Value> outer_ubs{
        make_index(builder, loc, metadata.y_shape[0]),
        make_index(builder, loc, metadata.y_shape[1]),
        make_index(builder, loc, metadata.y_shape[2]),
        make_index(builder, loc, metadata.y_shape[3]),
    };
    llvm::SmallVector<mlir::Value> steps(4, c1);

    mlir::scf::buildLoopNest(
        builder, loc, outer_lbs, outer_ubs, steps,
        [&](mlir::OpBuilder& body_builder, mlir::Location body_loc,
            mlir::ValueRange ivs) {
            auto n = ivs[0];
            auto oh = ivs[1];
            auto ow = ivs[2];
            auto k = ivs[3];

            llvm::SmallVector<mlir::Value> red_lbs(3, c0);
            llvm::SmallVector<mlir::Value> red_ubs{
                make_index(body_builder, body_loc, metadata.w_shape[1]),
                make_index(body_builder, body_loc, metadata.w_shape[2]),
                make_index(body_builder, body_loc, metadata.w_shape[3]),
            };
            llvm::SmallVector<mlir::Value> red_steps(3, c1);
            auto reduction = mlir::scf::buildLoopNest(
                body_builder, body_loc, red_lbs, red_ubs, red_steps,
                mlir::ValueRange{zero},
                [&](mlir::OpBuilder& reduction_builder,
                    mlir::Location reduction_loc, mlir::ValueRange red_ivs,
                    mlir::ValueRange args) -> mlir::scf::ValueVector {
                    auto r = red_ivs[0];
                    auto s = red_ivs[1];
                    auto c = red_ivs[2];
                    auto h = mlir::arith::AddIOp::create(
                        reduction_builder, reduction_loc, oh, r);
                    auto w = mlir::arith::AddIOp::create(
                        reduction_builder, reduction_loc, ow, s);
                    llvm::SmallVector<mlir::Value> x_indices{n, h, w, c};
                    llvm::SmallVector<mlir::Value> w_indices{k, r, s, c};
                    auto x_value = mlir::memref::LoadOp::create(
                        reduction_builder, reduction_loc, input, x_indices);
                    auto w_value = mlir::memref::LoadOp::create(
                        reduction_builder, reduction_loc, weight, w_indices);
                    auto product = mlir::arith::MulFOp::create(
                        reduction_builder, reduction_loc, x_value, w_value);
                    auto sum = mlir::arith::AddFOp::create(
                        reduction_builder, reduction_loc, args.front(), product);
                    return {sum};
                });
            mlir::memref::StoreOp::create(
                body_builder, body_loc, reduction.results.front(), output,
                llvm::SmallVector<mlir::Value>{n, oh, ow, k});
        });
    conv.erase();
    return Status::ok();
}

Status lower_vector_conv(mlir::func::FuncOp function,
                         Conv2DCompileMetadata const& metadata,
                         std::int64_t vector_width,
                         std::int64_t output_channel_unroll) {
    llvm::SmallVector<mlir::linalg::Conv2DNhwcFhwcOp> convs;
    function.walk([&](mlir::linalg::Conv2DNhwcFhwcOp conv) {
        convs.push_back(conv);
    });
    if (convs.size() != 1 || vector_width <= 1 ||
        output_channel_unroll <= 0 ||
        output_channel_unroll > metadata.y_shape[3]) {
        return invalid_ir("expected exactly one vectorizable Conv2D operation");
    }
    auto conv = convs.front();
    if (metadata.stride != std::array<std::int64_t, 2>{1, 1} ||
        metadata.dilation != std::array<std::int64_t, 2>{1, 1}) {
        return fail(ErrorCode::kInvalidValue, "mlir.conv2d",
                    "SIMD lowering requires unit stride and dilation");
    }
    auto input = conv.getInputs()[0];
    auto weight = conv.getInputs()[1];
    auto output = conv.getOutputs()[0];
    auto input_type = llvm::dyn_cast<mlir::MemRefType>(input.getType());
    auto weight_type = llvm::dyn_cast<mlir::MemRefType>(weight.getType());
    auto output_type = llvm::dyn_cast<mlir::MemRefType>(output.getType());
    if (!input_type || !weight_type || !output_type ||
        !input_type.hasStaticShape() || !weight_type.hasStaticShape() ||
        !output_type.hasStaticShape() || input_type.getRank() != 4 ||
        weight_type.getRank() != 4 || output_type.getRank() != 4 ||
        input_type.getElementType() != weight_type.getElementType() ||
        input_type.getElementType() != output_type.getElementType() ||
        !input_type.getElementType().isF32()) {
        return fail(ErrorCode::kInvalidShape, "mlir.conv2d",
                    "expected static rank-4 f32 memrefs");
    }
    auto const& x_shape = input_type.getShape();
    auto const& w_shape = weight_type.getShape();
    auto const& y_shape = output_type.getShape();
    if (x_shape[0] != metadata.padded_x_shape[0] ||
        x_shape[1] != metadata.padded_x_shape[1] ||
        x_shape[2] != metadata.padded_x_shape[2] ||
        x_shape[3] != metadata.padded_x_shape[3] ||
        w_shape[0] != metadata.w_shape[0] ||
        w_shape[1] != metadata.w_shape[1] ||
        w_shape[2] != metadata.w_shape[2] ||
        w_shape[3] != metadata.w_shape[3] ||
        y_shape[0] != metadata.y_shape[0] ||
        y_shape[1] != metadata.y_shape[1] ||
        y_shape[2] != metadata.y_shape[2] ||
        y_shape[3] != metadata.y_shape[3]) {
        return fail(ErrorCode::kInvalidShape, "mlir.conv2d",
                    "Conv2D memref shapes disagree with compile metadata");
    }

    mlir::OpBuilder builder(conv);
    auto loc = conv.getLoc();
    auto f32 = mlir::Float32Type::get(function.getContext());
    auto zero = make_zero(builder, loc, f32);
    auto c0 = make_index(builder, loc, 0);
    auto c1 = make_index(builder, loc, 1);
    auto vf = make_index(builder, loc, vector_width);
    auto vector_type = mlir::VectorType::get({vector_width}, f32);
    auto vector_zero = mlir::vector::BroadcastOp::create(
        builder, loc, vector_type, zero);

    auto emit_output_block =
        [&](mlir::OpBuilder& block_builder, mlir::Location block_loc,
            mlir::Value n, mlir::Value oh, mlir::Value ow,
            mlir::Value k_base, std::int64_t block_size) {
            llvm::SmallVector<mlir::Value> output_channels;
            output_channels.reserve(static_cast<std::size_t>(block_size));
            for (std::int64_t index = 0; index < block_size; ++index) {
                output_channels.push_back(
                    index == 0
                        ? k_base
                        : mlir::arith::AddIOp::create(
                              block_builder, block_loc, k_base,
                              make_index(block_builder, block_loc, index)));
            }

            llvm::SmallVector<mlir::Value> initial_values;
            initial_values.reserve(static_cast<std::size_t>(2 * block_size));
            for (std::int64_t index = 0; index < block_size; ++index) {
                initial_values.push_back(vector_zero);
            }
            for (std::int64_t index = 0; index < block_size; ++index) {
                initial_values.push_back(zero);
            }

            auto r_upper = make_index(block_builder, block_loc,
                                      metadata.w_shape[1]);
            auto s_upper = make_index(block_builder, block_loc,
                                      metadata.w_shape[2]);
            auto c_upper = make_index(block_builder, block_loc,
                                      metadata.w_shape[3]);
            auto c_vector_upper = make_index(
                block_builder, block_loc,
                (metadata.w_shape[3] / vector_width) * vector_width);

            auto r_loop = mlir::scf::ForOp::create(
                block_builder, block_loc, c0, r_upper, c1, initial_values);
            block_builder.setInsertionPointToStart(r_loop.getBody());
            auto s_loop = mlir::scf::ForOp::create(
                block_builder, block_loc, c0, s_upper, c1,
                r_loop.getRegionIterArgs());
            block_builder.setInsertionPointToStart(s_loop.getBody());

            auto r = r_loop.getInductionVar();
            auto s = s_loop.getInductionVar();
            auto h = mlir::arith::AddIOp::create(block_builder, block_loc, oh,
                                                r);
            auto w = mlir::arith::AddIOp::create(block_builder, block_loc, ow,
                                                s);

            llvm::SmallVector<mlir::Value> vector_initials;
            vector_initials.reserve(static_cast<std::size_t>(block_size));
            for (std::int64_t index = 0; index < block_size; ++index) {
                vector_initials.push_back(
                    s_loop.getRegionIterArgs()[static_cast<std::size_t>(index)]);
            }
            auto c_vector_loop = mlir::scf::ForOp::create(
                block_builder, block_loc, c0, c_vector_upper, vf,
                vector_initials);
            block_builder.setInsertionPointToStart(c_vector_loop.getBody());
            auto c_vector = c_vector_loop.getInductionVar();
            llvm::SmallVector<mlir::Value> x_indices{n, h, w, c_vector};
            auto x_vector = mlir::vector::LoadOp::create(
                block_builder, block_loc, vector_type, input, x_indices);
            llvm::SmallVector<mlir::Value> next_vectors;
            next_vectors.reserve(static_cast<std::size_t>(block_size));
            for (std::int64_t index = 0; index < block_size; ++index) {
                llvm::SmallVector<mlir::Value> w_indices{
                    output_channels[static_cast<std::size_t>(index)], r, s,
                    c_vector};
                auto w_vector = mlir::vector::LoadOp::create(
                    block_builder, block_loc, vector_type, weight, w_indices);
                next_vectors.push_back(mlir::vector::FMAOp::create(
                    block_builder, block_loc, x_vector, w_vector,
                    c_vector_loop.getRegionIterArgs()[
                        static_cast<std::size_t>(index)]));
            }
            mlir::scf::YieldOp::create(block_builder, block_loc, next_vectors);

            block_builder.setInsertionPointAfter(c_vector_loop);
            llvm::SmallVector<mlir::Value> scalar_initials;
            scalar_initials.reserve(static_cast<std::size_t>(block_size));
            for (std::int64_t index = 0; index < block_size; ++index) {
                scalar_initials.push_back(s_loop.getRegionIterArgs()[
                    static_cast<std::size_t>(block_size + index)]);
            }
            auto c_tail_loop = mlir::scf::ForOp::create(
                block_builder, block_loc, c_vector_upper, c_upper, c1,
                scalar_initials);
            block_builder.setInsertionPointToStart(c_tail_loop.getBody());
            auto c_tail = c_tail_loop.getInductionVar();
            llvm::SmallVector<mlir::Value> x_tail_indices{n, h, w, c_tail};
            auto x_scalar = mlir::memref::LoadOp::create(
                block_builder, block_loc, input, x_tail_indices);
            llvm::SmallVector<mlir::Value> next_scalars;
            next_scalars.reserve(static_cast<std::size_t>(block_size));
            for (std::int64_t index = 0; index < block_size; ++index) {
                llvm::SmallVector<mlir::Value> w_tail_indices{
                    output_channels[static_cast<std::size_t>(index)], r, s,
                    c_tail};
                auto w_scalar = mlir::memref::LoadOp::create(
                    block_builder, block_loc, weight, w_tail_indices);
                auto product = mlir::arith::MulFOp::create(
                    block_builder, block_loc, x_scalar, w_scalar);
                next_scalars.push_back(mlir::arith::AddFOp::create(
                    block_builder, block_loc,
                    c_tail_loop.getRegionIterArgs()[
                        static_cast<std::size_t>(index)],
                    product));
            }
            mlir::scf::YieldOp::create(block_builder, block_loc, next_scalars);

            block_builder.setInsertionPointToEnd(s_loop.getBody());
            llvm::SmallVector<mlir::Value> s_results;
            s_results.reserve(static_cast<std::size_t>(2 * block_size));
            s_results.append(c_vector_loop.getResults().begin(),
                             c_vector_loop.getResults().end());
            s_results.append(c_tail_loop.getResults().begin(),
                             c_tail_loop.getResults().end());
            mlir::scf::YieldOp::create(block_builder, block_loc, s_results);
            block_builder.setInsertionPointToEnd(r_loop.getBody());
            mlir::scf::YieldOp::create(block_builder, block_loc,
                                       s_loop.getResults());

            block_builder.setInsertionPointAfter(r_loop);
            for (std::int64_t index = 0; index < block_size; ++index) {
                auto reduced = mlir::vector::ReductionOp::create(
                    block_builder, block_loc,
                    mlir::vector::CombiningKind::ADD,
                    r_loop.getResults()[static_cast<std::size_t>(index)],
                    r_loop.getResults()[
                        static_cast<std::size_t>(block_size + index)]);
                mlir::memref::StoreOp::create(
                    block_builder, block_loc, reduced, output,
                    llvm::SmallVector<mlir::Value>{
                        n, oh, ow,
                        output_channels[static_cast<std::size_t>(index)]});
            }
        };

    llvm::SmallVector<mlir::Value> outer_lbs(3, c0);
    llvm::SmallVector<mlir::Value> outer_ubs{
        make_index(builder, loc, metadata.y_shape[0]),
        make_index(builder, loc, metadata.y_shape[1]),
        make_index(builder, loc, metadata.y_shape[2]),
    };
    llvm::SmallVector<mlir::Value> outer_steps(3, c1);
    auto outer = mlir::scf::buildLoopNest(
        builder, loc, outer_lbs, outer_ubs, outer_steps,
        [&](mlir::OpBuilder& body_builder, mlir::Location body_loc,
            mlir::ValueRange ivs) {
            auto n = ivs[0];
            auto oh = ivs[1];
            auto ow = ivs[2];
            auto k_upper = make_index(body_builder, body_loc,
                                      metadata.y_shape[3]);
            if (output_channel_unroll == 1) {
                auto k_loop = mlir::scf::ForOp::create(
                    body_builder, body_loc, c0, k_upper, c1);
                body_builder.setInsertionPointToStart(k_loop.getBody());
                emit_output_block(body_builder, body_loc, n, oh, ow,
                                  k_loop.getInductionVar(), 1);
                body_builder.setInsertionPointAfter(k_loop);
                return;
            }

            auto const full_channel_count =
                (metadata.y_shape[3] / output_channel_unroll) *
                output_channel_unroll;
            auto k_main_upper =
                make_index(body_builder, body_loc, full_channel_count);
            auto k_step = make_index(body_builder, body_loc,
                                     output_channel_unroll);
            auto k_main = mlir::scf::ForOp::create(
                body_builder, body_loc, c0, k_main_upper, k_step);
            body_builder.setInsertionPointToStart(k_main.getBody());
            emit_output_block(body_builder, body_loc, n, oh, ow,
                              k_main.getInductionVar(),
                              output_channel_unroll);
            body_builder.setInsertionPointAfter(k_main);

            auto k_tail = mlir::scf::ForOp::create(
                body_builder, body_loc, k_main_upper, k_upper, c1);
            body_builder.setInsertionPointToStart(k_tail.getBody());
            emit_output_block(body_builder, body_loc, n, oh, ow,
                              k_tail.getInductionVar(), 1);
            body_builder.setInsertionPointAfter(k_tail);
        });
    (void)outer;
    conv.erase();
    return Status::ok();
}

bool has_illegal_source_ops(mlir::ModuleOp module, std::string& name) {
    bool illegal = false;
    module.walk([&](mlir::Operation* operation) {
        if (illegal || operation->getName().getStringRef() == "builtin.module") {
            return;
        }
        auto dialect = operation->getName().getDialectNamespace();
        if (dialect != "llvm" && dialect != "builtin") {
            illegal = true;
            name = operation->getName().getStringRef().str();
        }
    });
    return illegal;
}

}  // namespace

Status lower_conv2d_variant(mlir::ModuleOp module,
                            Conv2DCompileMetadata const& metadata,
                            runtime::CpuVariant variant,
                            Conv2DSchedule const& schedule) {
    if (!module) {
        return fail(ErrorCode::kInvalidArgument, "module", "module is null");
    }
    if (!is_supported_variant(variant)) {
        return fail(ErrorCode::kInvalidArgument, "variant",
                    "unknown CPU variant");
    }
    auto const expected_vector_width =
        variant == runtime::CpuVariant::kScalar
            ? 1
            : (variant == runtime::CpuVariant::kAvx2 ? 8 : 16);
    if (schedule.vector_width != expected_vector_width ||
        schedule.output_channel_unroll <= 0 ||
        (variant == runtime::CpuVariant::kScalar &&
         schedule.output_channel_unroll != 1)) {
        return fail(ErrorCode::kInvalidArgument, "schedule",
                    "schedule is incompatible with the CPU variant");
    }
    auto function = module.lookupSymbol<mlir::func::FuncOp>(
        metadata.function_name);
    if (!function) {
        return invalid_ir("Conv2D function is absent");
    }
    auto status = replace_memref_copies(module);
    if (status.is_bad()) {
        return status;
    }
    if (variant == runtime::CpuVariant::kScalar) {
        status = lower_scalar_conv(function, metadata);
    } else {
        status = lower_vector_conv(
            function, metadata, schedule.vector_width,
            schedule.output_channel_unroll);
    }
    if (status.is_bad()) {
        return status;
    }

    mlir::PassManager passes(module.getContext());
    passes.addPass(mlir::createConvertLinalgToLoopsPass());
    passes.addPass(mlir::createCanonicalizerPass());
    if (mlir::failed(passes.run(module))) {
        return invalid_ir("standard Linalg-to-loop lowering failed");
    }
    return Status::ok();
}

Status lower_to_llvm(mlir::ModuleOp module,
                     Conv2DCompileMetadata const& metadata,
                     runtime::CpuVariant variant) {
    if (!module) {
        return fail(ErrorCode::kInvalidArgument, "module", "module is null");
    }
    auto function = module.lookupSymbol<mlir::func::FuncOp>(
        metadata.function_name);
    if (function) {
        function->setAttr("llvm.emit_c_interface",
                          mlir::UnitAttr::get(module.getContext()));
    }

    mlir::PassManager passes(module.getContext());
    if (variant != runtime::CpuVariant::kScalar) {
        passes.addPass(mlir::createCanonicalizerPass());
        passes.addPass(mlir::createConvertVectorToLLVMPass());
    }
    passes.addPass(mlir::createLowerAffinePass());
    passes.addPass(mlir::createSCFToControlFlowPass());
    passes.addPass(mlir::memref::createExpandStridedMetadataPass());
    passes.addPass(mlir::createConvertIndexToLLVMPass());
    passes.addPass(mlir::createConvertMathToLibmPass());
    passes.addPass(mlir::createConvertMathToLLVMPass());
    passes.addPass(mlir::createArithToLLVMConversionPass());
    passes.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
    passes.addPass(mlir::createConvertFuncToLLVMPass());
    passes.addPass(mlir::createConvertControlFlowToLLVMPass());
    passes.addPass(mlir::createReconcileUnrealizedCastsPass());
    if (mlir::failed(passes.run(module))) {
        return invalid_ir("standard dialect to LLVM conversion failed");
    }
    for (auto function : module.getOps<mlir::LLVM::LLVMFuncOp>()) {
        function.setVisibility_(mlir::LLVM::Visibility::Hidden);
    }
    std::string illegal_name;
    if (has_illegal_source_ops(module, illegal_name)) {
        return invalid_ir("LLVM module retains operation " + illegal_name);
    }
    if (mlir::failed(mlir::verify(module))) {
        return invalid_ir("LLVM dialect module failed verification");
    }
    return Status::ok();
}

}  // namespace deepforge::compiler
