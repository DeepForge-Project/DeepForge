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
ordered DAG. At C3 completion, 25 rows have a declared `validated` execution
subset:

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

All validated generic rows require static, positive, explicitly UID-assigned
f32 tensors and f32 graph context types, positive non-overlapping strides,
`NONE` reordering, and no pass-by-value or ragged metadata. Virtual
intermediates use planned workspace. Convolution grouping is inferred from
`X.C / W.C`; output channels must be divisible by that group count.

Normalization scalar inputs such as epsilon, momentum, and accumulation count
are explicit f32 tensors whose dimensions are all one; scalar pass-by-value
metadata remains deferred. Distributed nonempty `peer_stats` is not
executable. Running-stat ports must be either all present or all absent, and
their runtime values are required to provide positive accumulation counts and
valid epsilon values.

Comparison, logical, and `GEN_INDEX` pointwise results are represented as f32
`0`/`1` or f32 indices in C2; native boolean/integer outputs are deferred to
C5. C2 and C3 operations can be mixed in one ordered DAG. Explicit aliasing,
dynamic shape metadata, shape overrides, and non-f32 execution are deferred.

The remaining 14 rows stay `parsed`. Passing schema recognition never implies
lowering or CPU execution support. A recognized node without lowering returns
`kUnsupportedOperation`, not `kUnsupportedNode`. `validated` always refers to
the declared subset above, not every legal cuDNN backend configuration for the
tag.
