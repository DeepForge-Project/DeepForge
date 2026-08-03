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

Plan/runtime root fields such as `variant_pack_uids`, `pass_by_values`,
`workspace_modifications`, `variant_pack_replacements`, `fe_workspace_size`,
`behavior_notes`, and `tensors_to_dump` are not node semantics. The validated
Conv2D adapter applies its existing strict execution-metadata checks. Generic
nodes preserve protocol semantics without treating those fields as node
attributes; `pass_by_values` is additionally validated against embedded tensor
payloads before lowering.

## 2. Tensor And Graph Rules

Each tensor has the ten fields emitted by `Tensor_attributes`: `name`,
`data_type`, `dim`, `stride`, `is_virtual`, `pass_by_value`,
`is_pass_by_value`, `reordering_type`, `uid`, and `uid_assigned`.
`ragged_offset_uid` and `ragged_offset_name` are optional but must occur
together. A pass-by-value payload is null or the exact Frontend
`{index,value}` scalar variant object. The recognized reorder values are
`NONE`, `INT8x32`, `F16x16`, and `F8_128x4`.

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
| `MATMUL` | Equal rank of at least two, broadcast batch dimensions, optional per-batch INT32 M/N/K overrides, finite f32 padding value |
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
| `BLOCK_SCALE_DEQUANTIZE` | FLOAT compute; FP4 E2M1, FP8 E4M3/E5M2, or INT4 X; f32/f16/bf16/FP8 E4M3/E8M0 scale; trailing static block dimensions; f32/f16/bf16 Y; E4M3/E8M0 scale may use `F8_128x4` |
| `BLOCK_SCALE_QUANTIZE` | FLOAT compute; f32/f16/bf16 X; one divisible static block axis; FP4 E2M1 or FP8 E4M3/E5M2 Y and the corresponding E4M3/E8M0 scale; FP8 scale output may use `F8_128x4` |
| `MATMUL_FP8` | FP8 E4M3/E5M2 A/B, FLOAT scalar controls and amax, static rank >= 2 batch broadcasting, FP8/f32/f16/bf16 C, optional per-batch INT32 M/N/K overrides |
| `MOE_GROUPED_MATMUL` | `mode=NONE`, `top_k` 0 or 1, `[1,T,K]` tokens, `[E,K,N]` weights, INT32 `[E,1,1]` first-token offsets, shared f32/f16/bf16 data type |
| `MOE_GROUPED_MATMUL_BWD` | Per-expert weight gradient for the same static `mode=NONE` tensor and offset layout |
| `SDPA_FP8_FWD` | Static FP8 E4M3/E5M2 BHSD/GQA, scalar FLOAT scale controls, both diagonal alignments and bounded windows, optional row outputs, amax; no padding, dropout, or ALiBi |
| `SDPA_FP8_BWD` | Q/K/V gradients and amax for the same subset, consuming serialized `O`, `Stats`, FP8 dO, and scalar FLOAT scale controls |
| `SDPA_MXFP8_FWD` | Static BHSD/GQA with 32-element E8M0 descale blocks, f16/bf16/f32 O, required Stats and amax; descales use `NONE` or `F8_128x4` ordering |
| `SDPA_MXFP8_BWD` | Static transpose-oriented Q/K/dO inputs and E8M0 block descales, f16/bf16/f32 gradients and amax; f32 dS reference approximation; descales use `NONE` or `F8_128x4` ordering |

All validated generic rows require static, positive, explicitly UID-assigned
tensors and f32 graph context types, positive non-overlapping strides, `NONE`
reordering, and no pass-by-value metadata except the C6 runtime scalar subset
below. Ragged metadata is legal only for
the explicit standard-f32 SDPA subset below. Data tensors are f32. C4 permits
INT32 sequence lengths and scalar INT64 RNG seed/offset tensors, while MATMUL
and MATMUL_FP8 permit INT32 extent-override tensors on their documented ports.
Virtual intermediates use planned workspace. Convolution grouping is inferred
from `X.C / W.C`; output channels
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

MXFP8 block size is 32. C6 decodes the `F8_128x4` scale ordering emitted by
normal Frontend MXFP8 producers. The padded descriptor uses trailing M/K axes
in either order, with M divisible by 128, K divisible by 4, K stride 1, M
stride K, and packed leading axes. This layout is enabled only for E4M3/E8M0
scale input/output ports of block-scale conversion and for E8M0 MXFP8
`Descale_*` inputs.
Other ports and the `INT8x32`/`F16x16` reorder formats are rejected. Beyond
enum/string conversion plumbing in pinned v1.24.0, `F16x16` is exercised only
by attribute serialization round trips, while executable `INT8x32` sample and
reorder-helper usage is confined to the legacy backend/filter path. Neither has
a reachable modern serialized-Graph operation port with a defined physical
address mapping, so DeepForge does not infer one from the enum name.
FP8/MXFP8 attention rejects padding, dropout, ALiBi, block masks, sink tokens,
and unlisted optional ports. Windowed forms that could produce fully masked
rows are rejected. Backward consumes the supplied log-sum-exp `Stats`.

Normalization scalar inputs such as epsilon, momentum, and accumulation count
are f32 tensors whose dimensions are all one; they may be ordinary inputs,
runtime pass-by-value inputs, or embedded pass-by-value constants. Distributed
nonempty `peer_stats` is not
executable. Running-stat ports must be either all present or all absent, and
their runtime values are required to provide positive accumulation counts and
valid epsilon values.

C6 runtime pass-by-value accepts an input tensor only when
`is_pass_by_value=true`, `pass_by_value` is null, the UID is explicit, every
dimension is one, and the descriptor is external, non-ragged, and `NONE`
reordered. The scalar type is `INT64`, `INT32`, `HALF`, `FLOAT`, `DOUBLE`, or
`BFLOAT16`, subject to each operation port's narrower type rules. Its host
pointer is supplied under the ordinary UID and is persisted as a one-element
read argument in artifacts. Outputs, virtual tensors, and runtime
shape-override arrays are rejected.

C6 embedded pass-by-value uses the same descriptor constraints with a non-null
payload. Variant indices 0-5 map to `INT64`, `INT32`, `HALF`, `FLOAT`, `DOUBLE`,
and `BFLOAT16`; FLOAT stores an exact eight-digit hexadecimal bit pattern and
the other floating variants store JSON numbers. Root `pass_by_values` and
tensor payloads must form an exact one-to-one typed-value map. The compiler
binds each value from a private constant global, excludes its UID from public
arguments, and persists it in existing artifact objects. The distinct
Frontend compile-time-constant source fields are not serialized by v1.24.0 and
are outside this inventory.

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
Static f32 forward also accepts independent external ragged
Q/K/V/O/Stats/Max/Sum_exp storage with INT32/INT64 `[B+1,1,1,1]` element
prefixes. Backward accepts the same storage for Q/K/V/O/dO/Stats/dQ/dK/dV and
validated max-total-sequence hints. Forward independently paged K/V containers
have shape `[num_blocks,H,block_size,D]`; each INT32
`[B,1,page_slots,1]` table may be plain or compacted by an independent element
prefix, with runtime segment demand `ceil(SEQ_LEN_KV/block_size)`. These storage
forms require padding and both sequence lengths; page IDs must be valid
container block indices. Ragged Q/O may be combined with paged K/V. Forward
also supports the exact compressed UINT8 128-by-128 `Block_mask`. External f32
`SINK_TOKEN` `[1,Hq,1,1]` is supported by forward/backward and backward may
return matching `DSINK_TOKEN`. The pinned v1.24.0 backward serializer exposes
no K/V page-table ports, so paged backward is not representable by this input
schema. The C5-specialized optional features above remain deferred.

The CPU Bernoulli stream is stable and bit-identical across DeepForge CPU
variants for the same seed and offset. It is an implementation-defined CPU
stream and is not claimed to reproduce cuDNN's GPU Philox bits.

Comparison, logical, and `GEN_INDEX` pointwise results remain f32 `0`/`1` or
f32 indices. C2-C6 operations can be mixed in one ordered DAG when every
connecting port accepts the tensor type. C6 accepts both dynamic context flags.
Runtime shape override is executable for a non-broadcasting, `POINTWISE`-only
ordered DAG whose used tensors are plain f32 with equal compiled dimensions.
External arguments are read-only inputs plus one write-only output; virtual
intermediates receive the common runtime shape through statically bounded packed
workspace views. It is also executable for one standard-f32 `MATMUL` whose A/B/C
are external plain tensors of equal rank at least two. Its final descriptors
must preserve M/N/K relations and batch broadcasting; composed graphs, virtual
tensors, and simultaneous M/N/K extent ports are rejected. Shape override is
additionally executable for one dense
standard-f32 `SDPA` forward with external plain Q/K/V/O and optional
Stats/Max/Sum_exp. It is unmasked or top-left causal and may change only B, Sq,
and Skv; heads, embeddings, GQA, and all cross-tensor relations remain fixed.
Padding/sequence metadata, bias, ALiBi, other windows, dropout, paging/ragged
storage, sink/block masks, virtual tensors, and composed graphs are rejected.
Compiled dimensions are maxima; runtime dimensions must be positive and no
larger, runtime strides
must satisfy the supported positive non-overlap condition, and each external
storage span must fit its compiled bound. The dynamic flag alone is persisted
without changing static descriptor semantics. Independently,
MATMUL and MATMUL_FP8 accept optional external plain INT32 M/N/K inputs whose
rank matches C, whose trailing dimensions are `[1,1]`, and whose batch
dimensions broadcast to C. Values select the output and reduction extents
within static maxima; standard MATMUL uses a finite f32 padding value outside
M/N and MATMUL_FP8 uses zero. Explicit aliasing, other dynamic operations,
ragged tensors outside the documented standard-f32 SDPA subset, and physical
reorder contracts not emitted by the pinned modern Graph producer are deferred.
New artifacts use format v7 for ordered SDPA-forward override roles while
retaining v6 MATMUL roles, v4 ragged references, and v5 logical-sequence
divisors; v1-v6 remain readable, with v4 divisors defaulted to one and pre-v6
role lists empty.

Passing schema recognition never implies that every configuration lowers or
executes on the CPU. An attribute combination outside a declared subset
returns `kUnsupportedOperation`, not `kUnsupportedNode`. `validated` always
refers to the subset above, not every legal cuDNN backend configuration for the
tag.
