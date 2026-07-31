# cuDNN Frontend v1.24.0 Serialized Graph Schema Inventory

[中文](cudnn-graph-schema-inventory.md)

This document records the protocol surface recognized by DeepForge. It was
reviewed against the open-source cuDNN Frontend v1.24.0 definitions in
`graph_properties.h`, `utils/serialize.h`, `graph_interface.h`, and each node's
`serialize()` implementation. The executable subset is intentionally smaller
than the recognized protocol surface.

## 1. Graph Envelope

The importer requires these root fields:

| Field | Contract |
|---|---|
| `json_version` | String, exactly `1.0` |
| `cudnn_backend_version` | String, preserved as producer metadata |
| `cudnn_frontend_version` | Integer, exactly `12400` |
| `graph_uid` | Unsigned 64-bit integer |
| `context` | Context object described below |
| `nodes` | Non-empty ordered node array |
| `tensors` | Tensor object keyed by decimal UID or tensor name |

Every context serializer field is required: `name`, `compute_data_type`,
`intermediate_data_type`, `io_data_type`, `sm_count`,
`is_dynamic_shape_enabled`, and `is_override_shape_enabled`. Nullable context
fields remain nullable in the canonical model. `sm_count` must fit `int32`.

Plan-only root fields such as `variant_pack_uids`, `pass_by_values`,
`workspace_modifications`, `variant_pack_replacements`, `fe_workspace_size`,
`behavior_notes`, and `tensors_to_dump` are not node semantics. The validated
Conv2D adapter applies its existing strict execution-metadata checks; generic
nodes preserve protocol semantics without treating those plan fields as node
attributes.

## 2. Tensor And Graph Rules

Each tensor has the ten fields emitted by `Tensor_attributes`: `name`,
`data_type`, `dim`, `stride`, `is_virtual`, `pass_by_value`,
`is_pass_by_value`, `reordering_type`, `uid`, and `uid_assigned`.
`ragged_offset_uid` and `ragged_offset_name` are optional but must occur
together. A pass-by-value payload is null or numeric. The recognized reorder
values are `NONE`, `INT8x32`, `F16x16`, and `F8_128x4`.

The importer recognizes all 20 non-sentinel v1.24.0 data types. This is enum
recognition, not an execution claim. Tensor references are signed 64-bit UIDs
or names, matching the Frontend's compact graph serializer. References must
resolve, a tensor may have at most one producer, virtual inputs must be
produced earlier in node order, and every virtual tensor must have a producer.

`BATCHNORM` and `DBN` are the serializer's one special input shape:
`peer_stats` is a vector of complete tensor objects rather than entries in the
compact `inputs` object. The importer preserves that attribute and also
normalizes its tensors into the graph tensor table and a canonical input list.
An identical root/embedded descriptor is coalesced; a conflicting UID or name
is rejected.

## 3. Operation Inventory

`name`, `inputs`, and `outputs` are common to every row and are omitted from
the attribute column. `?` marks the only conditionally emitted attribute.
Listed ports are the complete serialized port catalog; individual Frontend
operations may use only the ports enabled by their options.

| Serialized tag | Serialized attributes | Input ports | Output ports |
|---|---|---|---|
| `ADA_LAYER_NORM` | `compute_data_type`, `forward_phase` | `X`, `SCALE`, `BIAS`, `EPSILON` | `Y`, `MEAN`, `INV_VARIANCE` |
| `ADA_LAYER_NORM_BPROP` | `compute_data_type` | `DY`, `X`, `SCALE`, `MEAN`, `INV_VARIANCE`, `EPSILON` | `DX`, `DSCALE`, `DBIAS` |
| `BATCHNORM` | `compute_data_type`, `peer_stats` | `X`, `SCALE`, `BIAS`, `EPSILON`, `PREV_RUNNING_MEAN`, `PREV_RUNNING_VAR`, `MOMENTUM` | `Y`, `MEAN`, `INV_VARIANCE`, `NEXT_RUNNING_MEAN`, `NEXT_RUNNING_VAR` |
| `BATCHNORM_INFERENCE` | `compute_data_type` | `X`, `SCALE`, `BIAS`, `MEAN`, `INV_VARIANCE` | `Y` |
| `BLOCK_SCALE_DEQUANTIZE` | `compute_data_type`, `block_size`, `is_negative_scale` | `X`, `scale` | `Y` |
| `BLOCK_SCALE_QUANTIZE` | `compute_data_type`, `block_size`, `axis` | `X` | `Y`, `scale` |
| `BN_FINALIZE` | `compute_data_type` | `SUM`, `SQ_SUM`, `SCALE`, `BIAS`, `EPSILON`, `ACCUM_COUNT`, `PREV_RUNNING_MEAN`, `PREV_RUNNING_VAR`, `MOMENTUM` | `EQ_SCALE`, `EQ_BIAS`, `MEAN`, `INV_VARIANCE`, `NEXT_RUNNING_MEAN`, `NEXT_RUNNING_VAR` |
| `CONCATENATE` | `axis`, `in_place_index` | Contiguous numeric ports `0..N-1` | `Y` |
| `CONV_DGRAD` | `compute_data_type`, `pre_padding`, `post_padding`, `stride`, `dilation`, `math_mode` | `W`, `DY` | `DX` |
| `CONV_FPROP` | `compute_data_type`, `pre_padding`, `post_padding`, `stride`, `dilation`, `math_mode` | `X`, `W` | `Y` |
| `CONV_WGRAD` | `compute_data_type`, `pre_padding`, `post_padding`, `stride`, `dilation`, `math_mode` | `DY`, `X` | `DW` |
| `DBN` | `compute_data_type`, `peer_stats` | `DY`, `X`, `SCALE`, `MEAN`, `INV_VARIANCE` | `DX`, `DSCALE`, `DBIAS` |
| `DBN_WEIGHT` | `compute_data_type` | `DY`, `X`, `SCALE`, `MEAN`, `INV_VARIANCE` | `DSCALE`, `DBIAS`, `EQ_BIAS`, `EQ_SCALE_DY`, `EQ_SCALE_X` |
| `GENSTATS` | `compute_data_type` | `X` | `SUM`, `SQ_SUM` |
| `INSTANCE_NORM` | `compute_data_type`, `forward_phase` | `X`, `SCALE`, `BIAS`, `EPSILON` | `Y`, `MEAN`, `INV_VARIANCE` |
| `INSTANCE_NORM_BPROP` | `compute_data_type` | `DY`, `X`, `SCALE`, `MEAN`, `INV_VARIANCE` | `DX`, `DSCALE`, `DBIAS` |
| `LAYER_NORM` | `compute_data_type`, `forward_phase` | `X`, `SCALE`, `BIAS`, `EPSILON` | `Y`, `MEAN`, `INV_VARIANCE` |
| `LAYER_NORM_BPROP` | `compute_data_type` | `DY`, `X`, `SCALE`, `MEAN`, `INV_VARIANCE`, `EPSILON` | `DX`, `DSCALE`, `DBIAS` |
| `MATMUL` | `compute_data_type`, `padding_value` | `A`, `B`, `M_override`, `N_override`, `K_override` | `C` |
| `MATMUL_FP8` | `compute_data_type` | `A`, `B`, `Descale_A`, `Descale_B`, `M_override`, `N_override`, `K_override`, `Scale_C` | `C`, `Amax_C` |
| `MOE_GROUPED_MATMUL` | `mode`, `top_k` | `Token`, `Weight`, `FirstTokenOffset`, `TokenIndex`, `TokenKs` | `Output` |
| `MOE_GROUPED_MATMUL_BWD` | None | `DOutput`, `Token`, `FirstTokenOffset` | `DWeight` |
| `POINTWISE` | `compute_data_type`, `mode`, `axis`, `relu_lower_clip`, `relu_upper_clip`, `relu_lower_clip_slope`, `swish_beta`, `elu_alpha`, `softplus_beta` | `IN_0`, `IN_1`, `IN_2`, constrained by mode arity | `OUT_0` |
| `REDUCTION` | `compute_data_type`, `mode`, `is_deterministic` | `X` | `Y` |
| `RESAMPLE` | `generate_index`, `resample_mode`, `padding_mode`, `pre_padding`, `post_padding`, `stride`, `window` | `X` | `Y`, `Index` |
| `RESHAPE` | `compute_data_type`, `dim`, `stride`, `reshape_mode` | `X` | `Y` |
| `RMS_NORM` | `compute_data_type`, `forward_phase` | `X`, `SCALE`, `BIAS`, `EPSILON` | `Y`, `INV_VARIANCE` |
| `RMS_NORM_BPROP` | `compute_data_type` | `DY`, `X`, `SCALE`, `INV_VARIANCE` | `DX`, `DSCALE`, `DBIAS` |
| `RNG` | `distribution`, `dim`, `stride`, `seed`, `bernoulli_probability` | `Seed`, `Offset` | `Y` |
| `ROPE` | `compute_data_type`, `output_scale`, `rope_dim` | `INPUT`, `FREQS` | `OUTPUT` |
| `ROPE_BWD` | `compute_data_type`, `output_scale`, `rope_dim` | `DY`, `FREQS` | `DX` |
| `SDPA` | Forward SDPA attributes below | Forward SDPA ports below | Forward SDPA outputs below |
| `SDPA_BWD` | Backward SDPA attributes below | Backward SDPA ports below | Backward SDPA outputs below |
| `SDPA_FP8_BWD` | FP8 backward attributes below | FP8 backward ports below | FP8 backward outputs below |
| `SDPA_FP8_FWD` | Forward SDPA attributes below | Forward SDPA ports below | Forward SDPA outputs below |
| `SDPA_MXFP8_BWD` | FP8 backward attributes below | FP8 backward ports below | FP8 backward outputs below |
| `SDPA_MXFP8_FWD` | Forward SDPA attributes below | Forward SDPA ports below | Forward SDPA outputs below |
| `SLICE` | `compute_data_type`, `slices`, `slice_strides` | `X` | `Y` |
| `TRANSPOSE` | `compute_data_type`, `permutation` | `X` | `Y` |

The shared forward SDPA attributes are `generate_stats`, `alibi_mask`,
`padding_mask`, `dropout_probability`, `attn_scale_value`, `max_seq_len_kv`,
`mma_core_mode`, `left_bound`, `right_bound`, `diagonal_alignment`,
`implementation`, `is_mxfp8`, `unfuse_fma`, and `rescale_threshold?`. Its
inputs are `Q`, `K`, `V`, `Attn_scale`, `Bias`, `SEQ_LEN_Q`, `SEQ_LEN_KV`,
`Seed`, `Offset`, `Dropout_mask`, `Dropout_scale`, `Page_table_K`,
`Page_table_V`, `Block_mask`, `Descale_Q`, `Descale_K`, `Descale_V`,
`Descale_S`, `Scale_S`, `Scale_O`, and `SINK_TOKEN`. Its outputs are `O`,
`Stats`, `Max`, `Sum_exp`, `RNG_DUMP`, `Amax_S`, and `Amax_O`.

The backward SDPA attributes are `alibi_mask`, `padding_mask`,
`dropout_probability`, `attn_scale_value`, `left_bound`, `right_bound`,
`diagonal_alignment`, `max_total_seq_len_q`, `max_total_seq_len_kv`, and
`is_deterministic_algorithm`. Inputs are `Q`, `K`, `V`, `O`, `dO`, `Stats`,
`Attn_scale`, `Bias`, `SEQ_LEN_Q`, `SEQ_LEN_KV`, `Seed`, `Offset`,
`Dropout_mask`, `Dropout_scale`, `Dropout_scale_inv`, and `SINK_TOKEN`.
Outputs are `dQ`, `dK`, `dV`, `dBias`, `RNG_DUMP`, and `DSINK_TOKEN`.

The FP8/MXFP8 backward attributes are `compute_data_type`, `padding_mask`,
`dropout_probability`, `left_bound`, `right_bound`, `diagonal_alignment`,
`attn_scale_value`, `is_deterministic_algorithm`, `is_mxfp8`, and
`rescale_threshold?`. The complete port lists are normative in
`lib/Import/Schema.cpp` and are matrix-tested because they contain 35 inputs
and 8 outputs.

## 4. Mode Inventories

Pointwise mode arity is validated exactly:

- One input: `SQRT`, `RELU_FWD`, `TANH_FWD`, `SIGMOID_FWD`, `ELU_FWD`,
  `GELU_FWD`, `SOFTPLUS_FWD`, `SWISH_FWD`, `ERF`, `IDENTITY`,
  `GELU_APPROX_TANH_FWD`, `GEN_INDEX`, `EXP`, `LOG`, `NEG`, `ABS`, `CEIL`,
  `COS`, `FLOOR`, `RSQRT`, `SIN`, `LOGICAL_NOT`, `TAN`, `RECIPROCAL`.
- Two inputs: `ADD`, `MUL`, `DIV`, `ADD_SQUARE`, `SUB`, `CMP_EQ`, `CMP_NEQ`,
  `CMP_GT`, `CMP_GE`, `CMP_LT`, `CMP_LE`, `LOGICAL_AND`, `LOGICAL_OR`, `MIN`,
  `MAX`, `MOD`, `RELU_BWD`, `TANH_BWD`, `SIGMOID_BWD`, `ELU_BWD`,
  `GELU_BWD`, `SOFTPLUS_BWD`, `SWISH_BWD`, `GELU_APPROX_TANH_BWD`, `POW`.
- Three inputs: `BINARY_SELECT`.

The nine reduction modes are `ADD`, `MUL`, `MIN`, `MAX`, `AMAX`, `AVG`,
`NORM1`, `NORM2`, and `MUL_NO_ZEROS`.

## 5. Capability Meaning

All 39 rows are `parsed`: JSON and canonical UBJSON are accepted and
normalized, known malformed fields are rejected, and references form an
ordered DAG. At C5 completion, all 39 rows have a declared `validated`
execution subset:

| Tag | Validated CPU subset |
|---|---|
| `CONV_FPROP` | Static f32 rank 3-5, grouped channels, positive stride/dilation, non-negative asymmetric padding, and `CROSS_CORRELATION` or `CONVOLUTION` |
| `CONV_DGRAD` | Gradient with respect to X for the same convolution subset |
| `CONV_WGRAD` | Gradient with respect to W for the same convolution subset |
| `RESHAPE` | `LOGICAL` mode, equal element counts; `VIEW_ONLY` aliasing is deferred |
| `TRANSPOSE` | Static permutation containing every axis exactly once |
| `SLICE` | Rank-preserving, in-range half-open bounds and positive integer strides |
| `CONCATENATE` | Static indexed inputs and non-negative axis; no `in_place_index` |
| `POINTWISE` | All 50 modes, f32 outputs, trailing-dimension NumPy broadcasting |
| `REDUCTION` | All 9 modes, rank-preserving output with reduced extents set to one |
| `MATMUL` | Equal rank of at least two, broadcast batch dimensions, no dimension override, zero padding value |
| `RESAMPLE` | Three pooling modes plus integer `NEAREST`, with three padding modes and no index output; `BILINEAR` is rejected because v1.24.0 serialization omits fraction denominators |
| `ADA_LAYER_NORM` | Training or inference, full-rank broadcast parameters, batch-preserving statistics |
| `ADA_LAYER_NORM_BPROP` | Saved mean/inverse standard deviation, explicit all-ones epsilon tensor, optional bias gradient |
| `BATCHNORM` | Training, per-channel parameters/statistics, optional all-or-none running-stat update, empty `peer_stats` |
| `BATCHNORM_INFERENCE` | Per-channel supplied mean and inverse standard deviation |
| `BN_FINALIZE` | Per-channel sum/square-sum finalization, equivalent affine values, saved statistics, and sample running variance |
| `DBN` | Batch-normalization data and parameter gradients with empty `peer_stats` |
| `DBN_WEIGHT` | Batch-normalization parameter gradients and equivalent data-gradient coefficients |
| `GENSTATS` | Per-channel sum and square sum |
| `INSTANCE_NORM` | Training or inference, per-channel parameters, per-instance/channel statistics |
| `INSTANCE_NORM_BPROP` | Data and per-channel parameter gradients from saved statistics |
| `LAYER_NORM` | Training or inference; normalized axes derive from the full-rank broadcast scale shape |
| `LAYER_NORM_BPROP` | Data and parameter gradients from saved statistics and an all-ones epsilon tensor |
| `RMS_NORM` | Training or inference, optional bias, RMS statistics derived from scale shape |
| `RMS_NORM_BPROP` | Data and scale gradients, with optional bias gradient |
| `RNG` | Static f32 Bernoulli output; either a serialized fixed seed or explicit scalar INT64 `Seed` and `Offset` inputs |
| `ROPE` | Static f32 BHSD split-half rotation, optional scaled pass-through prefix, and `[S,1,1,R]` frequencies |
| `ROPE_BWD` | Adjoint of the validated full or partial `ROPE` transform, including `output_scale` |
| `SDPA` | Static f32 BHSD attention with GQA, bias, scale, ALiBi, padding, top-left/bottom-right windows, custom or probability dropout, and serialized row/RNG outputs under the restrictions below |
| `SDPA_BWD` | Data and optional bias gradients for the same attention subset using serialized `O` and log-sum-exp `Stats` |
| `BLOCK_SCALE_DEQUANTIZE` | FLOAT compute; FP4 E2M1, FP8 E4M3/E5M2, or INT4 X; f32/f16/bf16/FP8 E4M3/E8M0 scale; trailing static block dimensions; f32/f16/bf16 Y |
| `BLOCK_SCALE_QUANTIZE` | FLOAT compute; f32/f16/bf16 X; one divisible static block axis; FP4 E2M1 or FP8 E4M3/E5M2 Y and the corresponding E4M3/E8M0 scale |
| `MATMUL_FP8` | FP8 E4M3/E5M2 A/B, FLOAT scalar controls and amax, static rank >= 2 batch broadcasting, FP8/f32/f16/bf16 C, no M/N/K override |
| `MOE_GROUPED_MATMUL` | `mode=NONE`, `top_k` 0 or 1, `[1,T,K]` tokens, `[E,K,N]` weights, INT32 `[E,1,1]` first-token offsets, shared f32/f16/bf16 data type |
| `MOE_GROUPED_MATMUL_BWD` | Per-expert weight gradient for the same static `mode=NONE` tensor and offset layout |
| `SDPA_FP8_FWD` | Static FP8 E4M3/E5M2 BHSD/GQA, scalar FLOAT scale controls, both diagonal alignments and bounded windows, optional row outputs, amax; no padding, dropout, or ALiBi |
| `SDPA_FP8_BWD` | Q/K/V gradients and amax for the same subset, consuming serialized `O`, `Stats`, FP8 dO, and scalar FLOAT scale controls |
| `SDPA_MXFP8_FWD` | Static BHSD/GQA with 32-element E8M0 descale blocks, f16/bf16/f32 O, required Stats and amax; logical `NONE` scale ordering in C5 |
| `SDPA_MXFP8_BWD` | Static transpose-oriented Q/K/dO inputs and E8M0 block descales, f16/bf16/f32 gradients and amax; f32 dS reference approximation and logical `NONE` scale ordering in C5 |

All validated generic rows require static, positive, explicitly UID-assigned
tensors and f32 graph context types, positive non-overlapping strides, `NONE`
reordering, and no pass-by-value or ragged metadata. Data tensors are f32;
C4 additionally permits INT32 sequence lengths and scalar INT64 RNG seed and
offset tensors on their documented ports. Virtual intermediates use planned
workspace. Convolution grouping is inferred from `X.C / W.C`; output channels
must be divisible by that group count.

The nine C5 specialized rows use FLOAT accumulation and software conversion on
the CPU. FP8 E4M3/E5M2 uses saturating round-to-nearest-even conversion; E8M0
is unsigned power-of-two scale storage with canonical NaN; FP4 E2M1 and INT4
use two logical values per byte. Only the operation/port combinations listed
above are enabled. Recognizing f64, integer, boolean, or another serializer
enum value does not make it legal on a generic operation.

For both validated MoE rows, runtime `FirstTokenOffset` values are a caller
precondition: element zero is 0, values are nondecreasing, and every value is
in `[0,T]`. The serialized graph fixes the offset tensor shape and type, but it
does not carry those runtime values for compile-time validation.

MXFP8 block size is 32. C5 consumes scale tensors in their logical strided
order and therefore rejects the `F8_128x4` reordering emitted by normal
Frontend MXFP8 producers; physical reorder decoding is a C6 requirement.
FP8/MXFP8 attention rejects padding, dropout, ALiBi, block masks, sink tokens,
and unlisted optional ports. Windowed forms that could produce fully masked
rows are rejected. Backward consumes the supplied log-sum-exp `Stats`.

Normalization scalar inputs such as epsilon, momentum, and accumulation count
are explicit f32 tensors whose dimensions are all one; scalar pass-by-value
metadata remains deferred. Distributed nonempty `peer_stats` is not
executable. Running-stat ports must be either all present or all absent, and
their runtime values are required to provide positive accumulation counts and
valid epsilon values.

`ROPE` rotates the final even `rope_dim` values, or the full final dimension
when `rope_dim == 0`; the preceding values are scaled pass-through. `FREQS`
has shape `[S,1,1,rope_dim]`, and the split-half transform consumes its first
half. `ROPE_BWD` is the linear adjoint with the same scale.

Validated SDPA uses rank-4 BHSD Q/K/V/O, unit embedding stride, and independent
integer GQA ratios for K and V heads. Bias and dropout masks use trailing-axis
broadcasting. Padding lengths are INT32 `[B,1,1,1]`. Both diagonal alignments
and left/right bounds are executable; the v1.24.0 bottom-right causal path does
not combine with bias, ALiBi, or dropout. ALiBi requires `right_bound == 0`.
Probability dropout takes scalar INT64 seed/offset and can expose `RNG_DUMP`;
custom dropout takes a mask and explicit scale, plus scale inverse in backward.
Paged/cache attention, block masks, sink tokens, packed/ragged metadata,
dynamic shapes, and the C5-specialized optional features above remain deferred.

The CPU Bernoulli stream is stable and bit-identical across DeepForge CPU
variants for the same seed and offset. It is an implementation-defined CPU
stream and is not claimed to reproduce cuDNN's GPU Philox bits.

Comparison, logical, and `GEN_INDEX` pointwise results remain f32 `0`/`1` or
f32 indices. C2-C5 operations can be mixed in one ordered DAG when every
connecting port accepts the tensor type. Explicit aliasing, dynamic shape
metadata, shape overrides, ragged tensors, and physical reorder handling are
deferred.

Passing schema recognition never implies that every configuration lowers or
executes on the CPU. An attribute combination outside a declared subset
returns `kUnsupportedOperation`, not `kUnsupportedNode`. `validated` always
refers to the subset above, not every legal cuDNN backend configuration for the
tag.
