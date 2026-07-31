#include "DeepForge/Compiler/Conv2DImporter.h"

#include "DeepForge/Import/Capability.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/LogicalResult.h"

#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <iterator>
#include <utility>

namespace deepforge::compiler {
namespace {

using deepforge::import::DataType;
using deepforge::import::ErrorCode;
using deepforge::import::SerializedGraph;
using deepforge::import::Status;
using deepforge::import::TensorDesc;

template <typename T>
bool checked_add(T lhs, T rhs, T& result) {
    static_assert(std::numeric_limits<T>::is_integer);
    if (rhs > 0 && lhs > std::numeric_limits<T>::max() - rhs) {
        return false;
    }
    if (rhs < 0 && lhs < std::numeric_limits<T>::min() - rhs) {
        return false;
    }
    result = static_cast<T>(lhs + rhs);
    return true;
}

bool checked_mul_nonnegative(std::int64_t lhs,
                             std::int64_t rhs,
                             std::int64_t& result) {
    if (lhs < 0 || rhs < 0) {
        return false;
    }
    if (lhs != 0 && rhs > std::numeric_limits<std::int64_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

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

bool is_f32(DataType type) {
    return type == DataType::kFloat32;
}

Status checked_output_extent(std::int64_t input,
                             std::int64_t pre,
                             std::int64_t post,
                             std::int64_t filter,
                             std::string_view subject,
                             std::int64_t& output) {
    std::int64_t total = 0;
    if (!checked_add(input, pre, total) || !checked_add(total, post, total)) {
        return fail(ErrorCode::kDimensionOverflow, subject,
                    "output extent arithmetic overflows int64");
    }
    if (total < filter) {
        return fail(ErrorCode::kInvalidShape, subject,
                    "filter and padding produce a non-positive output extent");
    }
    if (total == std::numeric_limits<std::int64_t>::max()) {
        return fail(ErrorCode::kDimensionOverflow, subject,
                    "output extent arithmetic overflows int64");
    }
    output = total - filter + 1;
    return Status::ok();
}

Status expected_packed_stride(TensorDesc const& tensor,
                              bool filter,
                              std::vector<std::int64_t>& expected) {
    expected.assign(4, 0);
    std::int64_t product = 0;
    if (filter) {
        if (!checked_mul_nonnegative(tensor.dim[3], tensor.dim[1], product) ||
            !checked_mul_nonnegative(tensor.dim[2], product, expected[0])) {
            return fail(ErrorCode::kDimensionOverflow, "mlir.layout",
                        "filter packed stride arithmetic overflows int64");
        }
        expected[1] = 1;
        expected[2] = product;
        expected[3] = tensor.dim[1];
        return Status::ok();
    }

    if (!checked_mul_nonnegative(tensor.dim[2], tensor.dim[3], product) ||
        !checked_mul_nonnegative(tensor.dim[1], product, expected[0]) ||
        !checked_mul_nonnegative(tensor.dim[3], tensor.dim[1], expected[2])) {
        return fail(ErrorCode::kDimensionOverflow, "mlir.layout",
                    "tensor packed stride arithmetic overflows int64");
    }
    expected[1] = 1;
    expected[3] = tensor.dim[1];
    return Status::ok();
}

Status validate_canonical_model(SerializedGraph const& graph) {
    auto const* conv_ptr = graph.single_conv_fprop();
    if (conv_ptr == nullptr) {
        return fail(ErrorCode::kUnsupportedOperation, "nodes",
                    "Tensor/Linalg Conv importer requires one CONV_FPROP node");
    }
    auto const& conv = *conv_ptr;
    if (!is_f32(conv.compute_data_type)) {
        return fail(ErrorCode::kUnsupportedDataType, "nodes[0].compute_data_type",
                    "P2 requires f32 computation");
    }
    if (graph.context.is_dynamic_shape_enabled.value_or(false)) {
        return fail(ErrorCode::kInvalidShape, "context.is_dynamic_shape_enabled",
                    "dynamic shapes are not supported by the Tensor/Linalg importer");
    }
    if (graph.tensor_count() != 3) {
        return fail(ErrorCode::kInvalidValue, "tensors",
                    "P2 requires exactly X, W and Y tensors");
    }
    if (conv.x_uid == conv.w_uid || conv.x_uid == conv.y_uid ||
        conv.w_uid == conv.y_uid) {
        return fail(ErrorCode::kInvalidValue, "nodes[0]",
                    "X, W and Y UIDs must be distinct");
    }

    auto const x_it = graph.tensors.find(conv.x_uid);
    auto const w_it = graph.tensors.find(conv.w_uid);
    auto const y_it = graph.tensors.find(conv.y_uid);
    if (x_it == graph.tensors.end() || w_it == graph.tensors.end() ||
        y_it == graph.tensors.end()) {
        return fail(ErrorCode::kMissingUid, "nodes[0]",
                    "node references a tensor absent from tensors");
    }
    auto const& x = x_it->second;
    auto const& w = w_it->second;
    auto const& y = y_it->second;

    for (auto const& [uid, tensor] : graph.tensors) {
        if (uid != tensor.uid) {
            return fail(ErrorCode::kInvalidValue, "tensors",
                        "tensor map key and descriptor UID differ");
        }
        if (!is_f32(tensor.data_type)) {
            return fail(ErrorCode::kUnsupportedDataType, "tensors",
                        "P2 requires f32 tensors");
        }
        if (tensor.is_virtual || tensor.is_pass_by_value) {
            return fail(ErrorCode::kInvalidValue, "tensors",
                        "virtual and pass-by-value tensors are not supported");
        }
        if (tensor.dim.size() != 4 || tensor.stride.size() != 4) {
            return fail(ErrorCode::kInvalidShape, "tensors",
                        "Conv2D lowering requires rank-4 tensors");
        }
        for (auto dim : tensor.dim) {
            if (dim <= 0) {
                return fail(ErrorCode::kInvalidShape, "tensors",
                            "all tensor dimensions must be positive");
            }
        }
    }

    if (x.dim[1] != w.dim[1]) {
        return fail(ErrorCode::kInvalidShape, "tensors",
                    "X.C must equal W.C");
    }
    if (y.dim[0] != x.dim[0] || y.dim[1] != w.dim[0]) {
        return fail(ErrorCode::kInvalidShape, "tensors",
                    "Y.N/Y.K do not match X.N/W.K");
    }
    for (std::size_t i = 0; i < conv.stride.size(); ++i) {
        if (conv.pre_padding[i] < 0 || conv.post_padding[i] < 0) {
            return fail(ErrorCode::kInvalidShape, "nodes[0]",
                        "padding must be non-negative");
        }
        if (conv.stride[i] != 1 || conv.dilation[i] != 1) {
            return fail(ErrorCode::kInvalidValue, "nodes[0]",
                        "P2 requires unit stride and unit dilation");
        }
    }

    std::int64_t expected_p = 0;
    std::int64_t expected_q = 0;
    auto status = checked_output_extent(x.dim[2], conv.pre_padding[0],
                                        conv.post_padding[0], w.dim[2],
                                        "nodes[0].pre_padding[0]", expected_p);
    if (status.is_bad()) {
        return status;
    }
    status = checked_output_extent(x.dim[3], conv.pre_padding[1],
                                   conv.post_padding[1], w.dim[3],
                                   "nodes[0].pre_padding[1]", expected_q);
    if (status.is_bad()) {
        return status;
    }
    if (y.dim[2] != expected_p || y.dim[3] != expected_q) {
        return fail(ErrorCode::kInvalidShape, "tensors",
                    "serialized Y shape does not match Conv2D inference");
    }

    std::vector<std::int64_t> expected;
    status = expected_packed_stride(x, false, expected);
    if (status.is_bad()) {
        return status;
    }
    if (x.stride != expected) {
        return fail(ErrorCode::kInvalidLayout, "tensors.X.stride",
                    "X is not packed NHWC");
    }
    status = expected_packed_stride(w, true, expected);
    if (status.is_bad()) {
        return status;
    }
    if (w.stride != expected) {
        return fail(ErrorCode::kInvalidLayout, "tensors.W.stride",
                    "W is not packed KRSC");
    }
    status = expected_packed_stride(y, false, expected);
    if (status.is_bad()) {
        return status;
    }
    if (y.stride != expected) {
        return fail(ErrorCode::kInvalidLayout, "tensors.Y.stride",
                    "Y is not packed NHWC");
    }
    return Status::ok();
}

llvm::SmallVector<std::int64_t, 4> physical_shape(TensorDesc const& tensor,
                                                   bool filter) {
    if (filter) {
        return {tensor.dim[0], tensor.dim[2], tensor.dim[3], tensor.dim[1]};
    }
    return {tensor.dim[0], tensor.dim[2], tensor.dim[3], tensor.dim[1]};
}

bool is_static_f32_tensor(::mlir::Type type,
                          llvm::ArrayRef<std::int64_t> expected_shape) {
    auto ranked = llvm::dyn_cast<::mlir::RankedTensorType>(type);
    if (!ranked || !ranked.hasStaticShape() ||
        ranked.getElementType() != ::mlir::Float32Type::get(type.getContext()) ||
        ranked.getRank() != static_cast<int64_t>(expected_shape.size())) {
        return false;
    }
    for (auto [actual, expected] : llvm::zip(ranked.getShape(), expected_shape)) {
        if (actual != expected) {
            return false;
        }
    }
    return true;
}

bool dense_i64_equals(::mlir::DenseIntElementsAttr attr,
                      llvm::ArrayRef<std::int64_t> expected) {
    if (!attr || attr.getType().getRank() != 1 ||
        attr.getType().getDimSize(0) != static_cast<int64_t>(expected.size())) {
        return false;
    }
    auto it = attr.begin();
    for (auto value : expected) {
        if (it == attr.end() || (*it++).getSExtValue() != value) {
            return false;
        }
    }
    return it == attr.end();
}

std::vector<std::int64_t> packed_strides(
    std::array<std::int64_t, 4> const& shape) {
    std::vector<std::int64_t> strides(shape.size(), 1);
    for (std::size_t index = shape.size() - 1; index > 0; --index) {
        strides[index - 1] = strides[index] * shape[index];
    }
    return strides;
}

Status append_argument(Conv2DCompileMetadata& metadata,
                       std::int64_t uid,
                       std::string name,
                       std::array<std::int64_t, 4> const& shape,
                       TensorAccess access) {
    TensorArgumentMetadata argument;
    argument.uid = uid;
    argument.name = std::move(name);
    argument.data_type = DataType::kFloat32;
    argument.dimensions.assign(shape.begin(), shape.end());
    argument.strides = packed_strides(shape);
    argument.alignment = alignof(float);
    argument.access = access;
    if (!deepforge::import::tensor_storage_bytes(
            argument.data_type, argument.dimensions, argument.strides,
            argument.size_bytes)) {
        return fail(ErrorCode::kDimensionOverflow, "metadata.arguments",
                    "tensor byte range overflows uint64");
    }
    metadata.arguments.push_back(std::move(argument));
    return Status::ok();
}

Status populate_metadata(SerializedGraph const& graph,
                         Conv2DImportOptions const& options,
                         Conv2DCompileMetadata& metadata) {
    auto const& conv = *graph.single_conv_fprop();
    auto const& x = graph.tensors.find(conv.x_uid)->second;
    auto const& w = graph.tensors.find(conv.w_uid)->second;
    auto const& y = graph.tensors.find(conv.y_uid)->second;
    metadata.function_name = options.function_name;
    metadata.x_uid = x.uid;
    metadata.w_uid = w.uid;
    metadata.y_uid = y.uid;
    metadata.x_shape =
        std::array<std::int64_t, 4>{x.dim[0], x.dim[2], x.dim[3], x.dim[1]};
    metadata.w_shape =
        std::array<std::int64_t, 4>{w.dim[0], w.dim[2], w.dim[3], w.dim[1]};
    metadata.y_shape =
        std::array<std::int64_t, 4>{y.dim[0], y.dim[2], y.dim[3], y.dim[1]};
    metadata.padded_x_shape = metadata.x_shape;
    metadata.padded_x_shape[1] +=
        conv.pre_padding[0] + conv.post_padding[0];
    metadata.padded_x_shape[2] +=
        conv.pre_padding[1] + conv.post_padding[1];
    metadata.pre_padding = conv.pre_padding;
    metadata.post_padding = conv.post_padding;
    metadata.stride = conv.stride;
    metadata.dilation = conv.dilation;
    auto status = append_argument(metadata, metadata.x_uid, "X",
                                  metadata.x_shape, TensorAccess::kRead);
    if (status.is_bad()) {
        return status;
    }
    status = append_argument(metadata, metadata.w_uid, "W", metadata.w_shape,
                             TensorAccess::kRead);
    if (status.is_bad()) {
        return status;
    }
    return append_argument(metadata, metadata.y_uid, "Y", metadata.y_shape,
                           TensorAccess::kWrite);
}

Status verify_pad_region(::mlir::tensor::PadOp pad, ::mlir::Value zero) {
    auto& region = pad.getRegion();
    if (!region.hasOneBlock()) {
        return invalid_ir("tensor.pad must have one region block");
    }
    auto& block = region.front();
    if (block.getNumArguments() != 4 || block.getOperations().size() != 1) {
        return invalid_ir("tensor.pad region must contain four indices and one yield");
    }
    auto yield = llvm::dyn_cast<::mlir::tensor::YieldOp>(&block.front());
    if (!yield || yield.getValue() != zero) {
        return invalid_ir("tensor.pad must yield the f32 zero constant");
    }
    return Status::ok();
}

Status verify_conv_structure(::mlir::ModuleOp module,
                             SerializedGraph const& graph,
                             std::string_view expected_function_name) {
    auto const& conv_desc = *graph.single_conv_fprop();
    if (module.getBody()->getOperations().size() != 1) {
        return invalid_ir("module must contain only the generated func.func");
    }
    auto funcs = module.getOps<::mlir::func::FuncOp>();
    if (std::distance(funcs.begin(), funcs.end()) != 1) {
        return invalid_ir("module must contain exactly one func.func");
    }
    auto func = *funcs.begin();
    if (func.getSymName() !=
            llvm::StringRef(expected_function_name.data(), expected_function_name.size())) {
        return invalid_ir("unexpected Conv2D function name");
    }
    if (func.isExternal() || !func.getBody().hasOneBlock()) {
        return invalid_ir("Conv2D function must have one entry block");
    }

    auto const x_it = graph.tensors.find(conv_desc.x_uid);
    auto const w_it = graph.tensors.find(conv_desc.w_uid);
    auto const y_it = graph.tensors.find(conv_desc.y_uid);
    if (x_it == graph.tensors.end() || w_it == graph.tensors.end() ||
        y_it == graph.tensors.end()) {
        return fail(ErrorCode::kMissingUid, "mlir", "canonical model has missing Conv UID");
    }
    auto const x_shape = physical_shape(x_it->second, false);
    auto const w_shape = physical_shape(w_it->second, true);
    auto const y_shape = physical_shape(y_it->second, false);
    llvm::SmallVector<std::int64_t, 4> padded_shape = x_shape;
    padded_shape[1] +=
        conv_desc.pre_padding[0] + conv_desc.post_padding[0];
    padded_shape[2] +=
        conv_desc.pre_padding[1] + conv_desc.post_padding[1];

    auto func_type = func.getFunctionType();
    if (func_type.getNumInputs() != 3 || func_type.getNumResults() != 1 ||
        !is_static_f32_tensor(func_type.getInput(0), x_shape) ||
        !is_static_f32_tensor(func_type.getInput(1), w_shape) ||
        !is_static_f32_tensor(func_type.getInput(2), y_shape) ||
        !is_static_f32_tensor(func_type.getResult(0), y_shape)) {
        return fail(ErrorCode::kInvalidShape, "mlir.function_type",
                    "function boundary does not match physical Conv2D tensors");
    }

    bool needs_padding = conv_desc.pre_padding !=
                             std::array<std::int64_t, 2>{0, 0} ||
                         conv_desc.post_padding !=
                             std::array<std::int64_t, 2>{0, 0};
    auto& entry = func.getBody().front();
    auto expected_operations = needs_padding ? 5U : 4U;
    if (entry.getNumArguments() != 3 ||
        entry.getOperations().size() != expected_operations) {
        return invalid_ir("entry block has an unexpected operation count");
    }
    auto it = entry.begin();
    auto constant = llvm::dyn_cast<::mlir::arith::ConstantOp>(&*it++);
    ::mlir::tensor::PadOp pad;
    if (needs_padding) {
        pad = llvm::dyn_cast<::mlir::tensor::PadOp>(&*it++);
    }
    auto fill = llvm::dyn_cast<::mlir::linalg::FillOp>(&*it++);
    auto conv = llvm::dyn_cast<::mlir::linalg::Conv2DNhwcFhwcOp>(&*it++);
    auto ret = llvm::dyn_cast<::mlir::func::ReturnOp>(&*it++);
    if (!constant || (needs_padding && !pad) || !fill || !conv || !ret ||
        it != entry.end()) {
        return invalid_ir("entry block contains an unexpected operation");
    }

    auto zero_attr = llvm::dyn_cast<::mlir::FloatAttr>(constant.getValue());
    if (!zero_attr || !zero_attr.getValue().isZero() ||
        constant.getResult().getType() !=
            ::mlir::Float32Type::get(module.getContext())) {
        return invalid_ir("constant must be f32 zero");
    }
    auto zero = constant.getResult();
    ::mlir::Value conv_input = entry.getArgument(0);
    if (needs_padding) {
        if (pad.getSource() != entry.getArgument(0) ||
            !pad.getLow().empty() || !pad.getHigh().empty() ||
            !is_static_f32_tensor(pad.getResult().getType(), padded_shape) ||
            !llvm::equal(
                pad.getStaticLow(),
                llvm::ArrayRef<std::int64_t>{0, conv_desc.pre_padding[0],
                                             conv_desc.pre_padding[1], 0}) ||
            !llvm::equal(
                pad.getStaticHigh(),
                llvm::ArrayRef<std::int64_t>{0, conv_desc.post_padding[0],
                                             conv_desc.post_padding[1], 0})) {
            return fail(ErrorCode::kInvalidLayout, "mlir.tensor.pad",
                        "padding or physical NHWC order is incorrect");
        }
        auto status = verify_pad_region(pad, zero);
        if (status.is_bad()) {
            return status;
        }
        conv_input = pad.getResult();
    }

    auto fill_inputs = fill.getInputs();
    auto fill_outputs = fill.getOutputs();
    auto fill_results = fill.getResultTensors();
    if (fill_inputs.size() != 1 || fill_outputs.size() != 1 ||
        fill_results.size() != 1 || fill_inputs.front() != zero ||
        fill_outputs.front() != entry.getArgument(2) ||
        fill_results.front().getType() != entry.getArgument(2).getType()) {
        return invalid_ir("linalg.fill must initialize the Y destination");
    }

    auto conv_inputs = conv.getInputs();
    auto conv_outputs = conv.getOutputs();
    auto conv_results = conv.getResultTensors();
    if (conv_inputs.size() != 2 || conv_outputs.size() != 1 ||
        conv_results.size() != 1 || conv_inputs[0] != conv_input ||
        conv_inputs[1] != entry.getArgument(1) ||
        conv_outputs.front() != fill_results.front() ||
        conv_results.front().getType() != entry.getArgument(2).getType() ||
        !dense_i64_equals(conv.getStridesAttr(), {1, 1}) ||
        !dense_i64_equals(conv.getDilationsAttr(), {1, 1})) {
        return fail(ErrorCode::kInvalidLayout, "mlir.linalg.conv_2d_nhwc_fhwc",
                    "inputs, destination or indexing attributes are incorrect");
    }
    if (ret.getNumOperands() != 1 || ret.getOperand(0) != conv_results.front()) {
        return invalid_ir("func.return must return the Conv2D result");
    }
    return Status::ok();
}

}  // namespace

Status import_conv2d(::mlir::MLIRContext& context,
                     SerializedGraph const& graph,
                     ::mlir::OwningOpRef<::mlir::ModuleOp>& output,
                     Conv2DImportOptions const& options,
                     Conv2DCompileMetadata* metadata) {
    if (options.function_name.empty()) {
        return fail(ErrorCode::kInvalidArgument, "function_name",
                    "function name must not be empty");
    }

    auto status = validate_canonical_model(graph);
    if (status.is_bad()) {
        return status;
    }

    Conv2DCompileMetadata metadata_candidate;
    status = populate_metadata(graph, options, metadata_candidate);
    if (status.is_bad()) {
        return status;
    }

    auto const& conv_desc = *graph.single_conv_fprop();
    auto const x = graph.tensors.find(conv_desc.x_uid)->second;
    auto const w = graph.tensors.find(conv_desc.w_uid)->second;
    auto const y = graph.tensors.find(conv_desc.y_uid)->second;
    auto const x_shape = physical_shape(x, false);
    auto const w_shape = physical_shape(w, true);
    auto const y_shape = physical_shape(y, false);
    llvm::SmallVector<std::int64_t, 4> padded_shape = x_shape;
    padded_shape[1] +=
        conv_desc.pre_padding[0] + conv_desc.post_padding[0];
    padded_shape[2] +=
        conv_desc.pre_padding[1] + conv_desc.post_padding[1];

    auto f32 = ::mlir::Float32Type::get(&context);
    auto x_type = ::mlir::RankedTensorType::get(x_shape, f32);
    auto w_type = ::mlir::RankedTensorType::get(w_shape, f32);
    auto y_type = ::mlir::RankedTensorType::get(y_shape, f32);
    auto padded_type = ::mlir::RankedTensorType::get(padded_shape, f32);

    auto loc = ::mlir::UnknownLoc::get(&context);
    auto module = ::mlir::ModuleOp::create(loc);
    ::mlir::OpBuilder builder(&context);
    builder.setInsertionPointToStart(module.getBody());
    auto function_type = builder.getFunctionType(
        {x_type, w_type, y_type}, {y_type});
    auto function = ::mlir::func::FuncOp::create(
        builder, loc, options.function_name, function_type);
    auto* entry = function.addEntryBlock();
    builder.setInsertionPointToStart(entry);

    auto zero_attr = builder.getFloatAttr(f32, 0.0);
    auto zero = ::mlir::arith::ConstantOp::create(builder, loc, zero_attr);

    llvm::SmallVector<::mlir::OpFoldResult, 4> low;
    llvm::SmallVector<::mlir::OpFoldResult, 4> high;
    low.push_back(builder.getIndexAttr(0));
    low.push_back(builder.getIndexAttr(conv_desc.pre_padding[0]));
    low.push_back(builder.getIndexAttr(conv_desc.pre_padding[1]));
    low.push_back(builder.getIndexAttr(0));
    high.push_back(builder.getIndexAttr(0));
    high.push_back(builder.getIndexAttr(conv_desc.post_padding[0]));
    high.push_back(builder.getIndexAttr(conv_desc.post_padding[1]));
    high.push_back(builder.getIndexAttr(0));
    bool needs_padding = conv_desc.pre_padding !=
                             std::array<std::int64_t, 2>{0, 0} ||
                         conv_desc.post_padding !=
                             std::array<std::int64_t, 2>{0, 0};
    ::mlir::Value conv_input = entry->getArgument(0);
    if (needs_padding) {
        auto pad = ::mlir::tensor::PadOp::create(
            builder, loc, padded_type, entry->getArgument(0), low, high,
            zero.getResult());
        conv_input = pad.getResult();
    }
    auto fill = ::mlir::linalg::FillOp::create(
        builder, loc, ::mlir::ValueRange{zero.getResult()},
        ::mlir::ValueRange{entry->getArgument(2)});

    auto index_type = ::mlir::RankedTensorType::get({2}, builder.getI64Type());
    std::array<std::int64_t, 2> const unit_attributes{1, 1};
    auto strides = ::mlir::DenseIntElementsAttr::get(index_type, unit_attributes);
    auto dilations = ::mlir::DenseIntElementsAttr::get(index_type, unit_attributes);
    auto conv = ::mlir::linalg::Conv2DNhwcFhwcOp::create(
        builder, loc, ::mlir::TypeRange{y_type},
        ::mlir::ValueRange{conv_input, entry->getArgument(1)},
        ::mlir::ValueRange{fill.getResultTensors().front()}, strides,
        dilations);
    ::mlir::func::ReturnOp::create(builder, loc,
                                   ::mlir::ValueRange{conv.getResultTensors().front()});

    if (llvm::failed(::mlir::verify(module))) {
        module->erase();
        return invalid_ir("generated standard Tensor/Linalg module failed MLIR verification");
    }
    status = verify_conv_structure(module, graph, options.function_name);
    if (status.is_bad()) {
        module->erase();
        return status;
    }
    output = ::mlir::OwningOpRef<::mlir::ModuleOp>(module);
    if (metadata) {
        *metadata = std::move(metadata_candidate);
    }
    return Status::ok();
}

Status verify_conv2d_module(::mlir::ModuleOp module,
                            SerializedGraph const& graph,
                            std::string_view function_name) {
    if (!module) {
        return fail(ErrorCode::kInvalidArgument, "module", "module is null");
    }
    auto status = validate_canonical_model(graph);
    if (status.is_bad()) {
        return status;
    }
    if (llvm::failed(::mlir::verify(module))) {
        return invalid_ir("module failed the standard MLIR verifier");
    }
    return verify_conv_structure(module, graph, function_name);
}

}  // namespace deepforge::compiler
