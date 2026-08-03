# MVP Compatibility and Runtime Contract

[中文](contracts.md)

This document defines the normative boundary of the original DeepForge Conv2D
MVP. Post-MVP operation extensions are normative jointly with the
[schema capability inventory](../cudnn-graph-schema-inventory.en.md) and the
[coverage plan](../cudnn-graph-coverage-plan.en.md). The stricter applicable
contract takes precedence if documents conflict.

## 1. Version Policy

| Component | Pinned version | Policy |
|---|---|---|
| LLVM/MLIR | `llvmorg-22.1.8` | CMake requires an exact match |
| cuDNN Frontend | `v1.24.0` | Importer schema and fixtures are fixed against serializer source at this tag; other producer versions are rejected by the MVP |
| nlohmann/json | `3.11.3` | Matches the Frontend-vendored header; both JSON and UBJSON use this parser |
| CUDA/cuDNN backend | None | The CPU-only MVP does not build, link, or run a GPU backend |
| cuDNN Graph JSON | `json_version == "1.0"` | A mismatch is rejected with a version diagnostic |
| Platform | Linux x86-64 | The only MVP execution platform |

Dependency upgrades require a separate change and regeneration of serialized
fixtures, IR golden tests, and end-to-end numeric results. Unknown Graph schemas
must never be accepted silently.

## 2. cuDNN Frontend Input Contract

### 2.1 Accepted Carriers

`deepforge-compile` accepts the two native carriers produced by the open source
cuDNN Frontend:

1. Graph JSON produced by `Graph::serialize(nlohmann::json&)`.
2. A UBJSON blob produced by `Graph::serialize(std::vector<uint8_t>&)`.

In pinned `v1.24.0` source, the vector overload first performs Graph JSON
serialization, appends plan and runtime metadata, and encodes the complete JSON
object as **one UBJSON document** through `nlohmann::json::to_ubjson`. There is
no extra envelope and no GPU-plan byte stream after the UBJSON document;
`cudnn_backend_data` is a JSON value inside the document. The importer parses
the entire input in strict mode and compares it with the canonical bytes
produced by the same parser's default `to_ubjson`. It rejects truncation,
invalid tokens, trailing no-op markers, any other trailing bytes, and
non-default UBJSON encodings.

Both carriers require `json_version == "1.0"` and numeric
`cudnn_frontend_version == 12400`. `cudnn_backend_version` only records the
producer environment and does not affect CPU semantics. UBJSON may contain
additional fields:

| Field | MVP handling |
|---|---|
| `cudnn_backend_data`, `behavior_notes` | Accept and ignore; never interpret as a CPU execution plan |
| `variant_pack_uids` | If present, it must equal the set of non-virtual X/W/Y UIDs for the single Conv |
| `pass_by_values` | Absent/empty when no embedded scalar exists; otherwise an object with exactly one typed scalar variant per matching tensor payload under its canonical decimal UID |
| `workspace_modifications`, `variant_pack_replacements` | Must be absent or empty; an empty object or array is accepted for `variant_pack_replacements`, while a nonempty value represents unsupported execution semantics |
| `fe_workspace_size` | If present, must be zero; DeepForge replans CPU workspace |
| `tensors_to_dump` | Debug metadata; accept and ignore |

The importer uses a structured JSON/UBJSON parser. It does not parse with
string rules and does not define a private DeepForge JSON format. Input format
may be selected with `--input-format=json|ubjson|auto`. `auto` first tries
strict UTF-8 JSON and then strict UBJSON; it detects only the carrier and never
relaxes schema validation. One serialized Graph is limited to 16 MiB.
`parse_file` enforces the same limit while reading to avoid unbounded allocation
for oversized files.

### 2.2 MVP Serialized Schema

The pinned serializer subset requires an object root with these fields:

| Level | Required form | MVP constraint |
|---|---|---|
| root | `json_version`, `cudnn_frontend_version`, `graph_uid`, `context`, `nodes`, `tensors` | Versions are string `"1.0"` and integer `12400`; `nodes` has exactly one element |
| node | `tag`, `inputs`, `outputs`, `compute_data_type`, `pre_padding`, `post_padding`, `stride`, `dilation`, `math_mode` | `tag == "CONV_FPROP"`; ports are exactly `X/W` and `Y`; data type is `FLOAT`; spatial arrays have length two |
| tensor map | Object whose keys are decimal strings for explicit UIDs | Key, `uid`, and node reference agree; the MVP keeps exactly three non-virtual X/W/Y tensors |
| tensor | `name`, `data_type`, `dim`, `stride`, `is_virtual`, `is_pass_by_value`, `reordering_type`, `uid`, `uid_assigned` | `FLOAT`, rank four, `uid_assigned == true`, `is_virtual == false`, `is_pass_by_value == false`, and reordering `NONE` |

A node reference must be a JSON or UBJSON integer UID. A fallback reference by
tensor name is outside the MVP. JSON and UBJSON differ only as carriers and
must produce the same canonical model after parsing. Root execution and debug
metadata follows the rules in section 2.1 and cannot change that model's
semantics. The importer performs all validation in a temporary object; failure
must not partially modify the caller-provided output model.

### 2.3 Protocol Integration Versus Operation Coverage

Complete protocol integration means accurately reading the pinned cuDNN
Frontend serialization protocol, including tensor UIDs, dimensions, strides,
data types, `is_virtual`, node input and output port references, and attributes.
In this version, `set_output(true)` serializes as `is_virtual == false`; there
is no separate output field. Y is identified by `CONV_FPROP.outputs.Y`.
Complete integration does not mean that the MVP implements every cuDNN
operation.

The parser must traverse every valid node and return a structured diagnostic
for an unsupported node, for example:

```text
DFE_UNSUPPORTED_NODE: node[2] tag=POINTWISE is not supported by MVP
```

It must not skip an unknown node and continue compilation.

### 2.4 UID Constraints

Input, weight, and output tensors require explicit, unique, stable UIDs. The
runtime variant pack binds addresses by UID. A graph that has only tensor names
or depends on UID auto-assignment during deserialization is not reproducible
and is rejected by the importer.

## 3. Conv2D MVP Support Matrix

| Item | MVP support |
|---|---|
| Graph | Exactly one `CONV_FPROP` node |
| Tensor rank | Static rank-four X, W, and Y |
| Data type | f32 X, W, Y, and compute |
| Conv mode | Cross-correlation |
| X logical dimensions | cuDNN `[N, C, H, W]` |
| W logical dimensions | cuDNN `[K, C, R, S]` |
| Y logical dimensions | cuDNN `[N, K, P, Q]` |
| Physical layout | Packed NHWC input/output and packed KRSC filter |
| Spatial stride | `[1, 1]` |
| Dilation | `[1, 1]` |
| Padding | Static, nonnegative, and optionally asymmetric pre/post |
| Grouping | Grouped and depthwise convolution are unsupported |
| Fusion | Bias, activation, and other fused nodes are unsupported |
| Shapes | Every dimension is positive; C and K have no divisibility requirement |

cuDNN combines logical NCHW/KCRS dimensions with strides to describe physical
NHWC/KRSC buffers. The importer validates exact packed strides and converts to
MLIR physical dimension order:

| Tensor | cuDNN dimensions | cuDNN packed stride | MLIR tensor shape |
|---|---|---|---|
| X | `[N,C,H,W]` | `[H*W*C,1,W*C,C]` | `[N,H,W,C]` |
| W | `[K,C,R,S]` | `[R*S*C,1,S*C,C]` | `[K,R,S,C]` |
| Y | `[N,K,P,Q]` | `[P*Q*K,1,Q*K,K]` | `[N,P,Q,K]` |

W therefore maps to Linalg `FHWC`, where F is output feature K:
`[F,H,W,C] == [K,R,S,C]`. `[R,S,C,K]` is `HWCF` and cannot be passed to
`linalg.conv_2d_nhwc_fhwc`.

Output spatial dimensions must satisfy:

```text
P = H + pre_h + post_h - R + 1
Q = W + pre_w + post_w - S + 1
```

Compilation fails if serialized Y dimensions disagree with these values.

Every dimension product, packed stride, element byte count, and workspace
offset uses checked arithmetic. Any input outside the representable range of
`int64_t` or `size_t` returns `DFE_DIMENSION_OVERFLOW`; it must not silently
wrap in a later MLIR index or pointer computation.

### 3.1 Post-MVP C2 Extension

C2 adds a second executable graph form without weakening the Conv contract: a
statically allocated f32 DAG composed only of `RESHAPE`, `TRANSPOSE`, `SLICE`,
`CONCATENATE`, `POINTWISE`, `REDUCTION`, `MATMUL`, and `RESAMPLE`. Its exact
mode, shape, layout, attribute, and alias restrictions are the validated rows
in the [schema capability inventory](../cudnn-graph-schema-inventory.en.md#5-capability-meaning).

All C2 data tensors have positive static allocation dimensions and positive
non-overlapping strides, explicit UIDs, `NONE` reordering, and no ragged
metadata. Pass-by-value is limited to the runtime scalar extension in section
3.8. Virtual tensors are assigned to static workspace.
`VIEW_ONLY` reshape, in-place concatenate, same-UID input/output, and RESAMPLE
index outputs remain unsupported. RESAMPLE `BILINEAR` is also rejected because the
v1.24.0 serialized fraction representation omits the denominator needed to
reconstruct fractional scale semantics. The public execution interface in
section 4 is unchanged; only the runtime's hidden invocation adapter differs.

Standard f32 `MATMUL` may have optional `M_override`, `N_override`, and
`K_override` inputs. Each is an external plain INT32 tensor with the same rank
as C, trailing dimensions `[1,1]`, and batch dimensions equal to one or the
corresponding C dimension. A broadcast batch entry is a caller precondition in
the ranges `0 <= M <= C[-2]`, `0 <= N <= C[-1]`, and
`0 <= K <= A[-1] == B[-2]`; an absent input selects the static maximum. M/N
control the valid output rectangle and K controls the reduction extent. The
remaining output is filled with a finite f32 `padding_value`, whose default is
zero. These metadata tensors use ordinary UID-map arguments and do not alter
the public execution ABI.

### 3.2 Post-MVP C3 Extension

C3 extends the same static f32 DAG with `CONV_FPROP`, `CONV_DGRAD`, and
`CONV_WGRAD` at rank 3-5. Logical tensors use `[N,C,spatial...]` and filters use
`[K,C_per_group,filter...]`. The group count is inferred as
`X.C / C_per_group`; it and `Y.K / groups` must be integral and positive.
Spatial stride and dilation are positive, pre/post padding is non-negative,
and both `CROSS_CORRELATION` and filter-reversing `CONVOLUTION` are supported.
Every serialized output extent must equal the checked convolution formula.

C3 also executes the 14 normalization/statistics rows declared in the schema
capability inventory. Batch parameters and statistics use
`[1,C,1,...]`; instance parameters use that same per-channel shape while saved
statistics use `[N,C,1,...]`. Layer/RMS reduction axes derive from dimensions
where the full-rank scale is non-unit. Adaptive layer normalization follows
the same rule but always preserves the batch axis.

Forward normalization uses population variance for normalization and
`1/sqrt(variance + epsilon)` as the serialized inverse-variance output.
Running variance updates use sample variance when the reduction count exceeds
one. A running-stat update is `(1 - momentum) * previous + momentum * current`.
Backward rows consume saved statistics and implement gradients for data,
scale, and serialized bias outputs. `GENSTATS`, `BN_FINALIZE`, and
`DBN_WEIGHT` expose their serialized sums, equivalent affine values, and
gradient coefficients without a private side channel.

Scalar-like inputs are f32 tensors with every dimension equal to one. They may
be ordinary inputs, runtime pass-by-value inputs under section 3.8, or embedded
pass-by-value constants under section 3.9. Nonempty distributed `peer_stats` is
rejected.
BATCHNORM running-stat ports must be all present or all absent. At execution,
epsilon must make the square-root operand positive and `ACCUM_COUNT` must be
positive; these data values are caller preconditions rather than compile-time
metadata. C2 and C3 nodes may be mixed, including through virtual workspace
tensors. Dynamic shape, aliasing, ragged/reordered storage, and other non-f32
execution remain deferred.

### 3.3 Post-MVP C4 Extension

C4 adds `RNG`, `ROPE`, `ROPE_BWD`, `SDPA`, and `SDPA_BWD` to the same ordered
DAG. Data tensors and graph context types remain f32. The only integer tensors
are INT32 SDPA sequence lengths and scalar INT64 RNG seed/offset inputs on the
serialized ports defined by v1.24.0.

`RNG` executes Bernoulli output with probability in `[0,1]`. It accepts either
the serialized fixed seed or exactly two one-element INT64 `Seed` and `Offset`
inputs. The stream is deterministic and bit-identical across DeepForge CPU
variants. Its algorithm is part of the DeepForge CPU artifact semantics; bit
identity with cuDNN's GPU Philox implementation is not promised.

`ROPE` and `ROPE_BWD` accept f32 BHSD data and `[S,1,1,R]` frequencies. The
effective rotation width is the final dimension when `rope_dim == 0`, otherwise
the final even `rope_dim` values. Rotation uses split halves; a preceding
subspace is scaled pass-through. Backward is the linear adjoint and uses the
same `output_scale`.

Validated SDPA uses rank-4 BHSD Q/K/V/O with unit embedding stride. K and V
heads may each divide query heads for grouped-query attention. The score is
`attn_scale * (Q @ K^T) + bias + alibi`, followed by padding and diagonal-band
masks, softmax, dropout, and multiplication by V. `Stats` is row log-sum-exp;
`Max`, `Sum_exp`, and Bernoulli-mask `RNG_DUMP` are optional forward outputs.
Bias and custom dropout masks use trailing-axis broadcasting. Padding lengths
are INT32 `[B,1,1,1]` and callers must keep each runtime value in the
corresponding static sequence extent.

Top-left and bottom-right diagonal alignment support optional positive left
and nonnegative right bounds. ALiBi requires `right_bound == 0`. In accordance
with the v1.24.0 Frontend composite restrictions, bottom-right causal masking
does not combine with bias, ALiBi, or dropout. Probability dropout uses scalar
INT64 seed/offset and implicit reciprocal keep scale. Custom dropout uses an
explicit mask and scale; backward additionally consumes scale inverse.
`SDPA_BWD` consumes caller-consistent Q/K/V/O/dO/Stats and produces dQ/dK/dV
plus optional reduced dBias.

Paged/cache attention, block masks, sink tokens, max-total packed metadata,
FP8/MXFP8 controls, ragged layouts, and dynamic shapes remain unsupported by
C4.
C2, C3, and C4 nodes may be mixed; the primitive scalar implementation may
recompute softmax rows rather than materialize a private attention workspace.

### 3.4 Post-MVP C5 Extension

C5 adds static subsets for the remaining nine serialized tags:
`BLOCK_SCALE_QUANTIZE`, `BLOCK_SCALE_DEQUANTIZE`, `MATMUL_FP8`,
`MOE_GROUPED_MATMUL`, `MOE_GROUPED_MATMUL_BWD`, `SDPA_FP8_FWD`,
`SDPA_FP8_BWD`, `SDPA_MXFP8_FWD`, and `SDPA_MXFP8_BWD`. Their exact port,
shape, type, and optional-feature constraints are normative in the
[schema capability inventory](../cudnn-graph-schema-inventory.en.md#5-capability-meaning).

The CPU numeric layer accumulates in f32. FP8 E4M3/E5M2 conversion saturates
and rounds to nearest with ties to even. E8M0 stores unsigned powers of two and
uses `0xff` as canonical NaN. FP4 E2M1 and signed INT4 pack the lower-indexed
logical value into the low nibble and the next value into the high nibble.
External pointers and virtual workspace therefore use byte counts rounded up
from the exact storage-bit span.

FP8/MXFP8 SDPA is static rank-4 BHSD with GQA and supplied log-sum-exp Stats in
backward. C5 excludes padding, dropout, ALiBi, block masks, sink tokens, ragged
metadata, and unlisted optional ports. MXFP8 block size is 32; C5 accepts
logical `NONE` scale ordering and uses an f32 dS reference approximation.
Frontend-produced `F8_128x4` physical scale ordering entered through the C6
extension below.

Validated MoE execution requires runtime `FirstTokenOffset` values to start at
zero, be nondecreasing, and remain in `[0,T]`. This is a caller precondition:
the graph compiler validates the INT32 `[E,1,1]` descriptor, while the public
execute ABI intentionally does not inspect tensor contents before dispatch.

C2-C5 nodes may be mixed when both ends support the connected tensor type.
The public UID variant-pack and workspace ABI is unchanged. Dynamic/override
shapes, aliasing, ragged storage, reorder formats, and paged/cache metadata are
unsupported except for the explicit C6 extensions below.

### 3.5 C6 F8_128x4 Physical Scale Extension

`F8_128x4` is executable only for an E4M3/E8M0 scale input to
`BLOCK_SCALE_DEQUANTIZE`, an E4M3/E8M0 scale output from
`BLOCK_SCALE_QUANTIZE`, and E8M0 `Descale_*` inputs to
`SDPA_MXFP8_FWD`/`SDPA_MXFP8_BWD`. It is rejected on all data ports. `INT8x32`
and `F16x16` remain recognized but non-executable.

The final two descriptor axes represent M and K in either order. M and K must
be padded to multiples of 128 and 4 respectively. K has stride 1, M has stride
K, and all leading axes are packed. Given flattened leading coordinate `l`,
the physical byte offset is

```text
((((l * (M / 128) + m / 128) * (K / 4) + k / 4) * 512)
 + (m % 32) * 16 + ((m / 32) % 4) * 4 + k % 4).
```

The runtime pointer covers the complete padded descriptor span. Quantize
initializes that complete span to numeric one, E4M3 `0x38` or E8M0 `0x7f`,
before overwriting logical scale coordinates. The public execute and workspace
ABI is unchanged.

### 3.6 C6 Runtime Shape Override Extension

`is_dynamic_shape_enabled` and `is_override_shape_enabled` are parsed and
persisted independently. The dynamic flag alone is plan metadata and does not
alter a static kernel descriptor, and override execution does not require that
flag to be set. Enabling override selects one of four explicit policies.

The exact-pointwise policy accepts a nonempty ordered DAG containing only
`POINTWISE` nodes. Every used tensor is non-ragged, non-reordered,
non-pass-by-value FLOAT with the same compiled dimensions, so broadcasting is
excluded. External arguments consist of read-only inputs and one write-only
output; intermediate tensors may be virtual.

The MATMUL policy accepts exactly one standard-f32 `MATMUL`. A, B, and C are
external plain strided tensors of equal rank at least two; A/B are read-only and
C is write-only. Virtual, pass-by-value, ragged, reordered, or additional
tensors are rejected, as are composed graphs and simultaneous serialized
`M_override`/`N_override`/`K_override` ports. Metadata records three distinct
role UIDs in A, B, C order.

The SDPA-forward policy accepts exactly one dense standard-f32 `SDPA`. Q, K,
V, O and optional Stats/Max/Sum_exp are external plain rank-4 tensors; inputs
are read-only and outputs are write-only. Metadata records Q, K, V, O followed
by the present row outputs in Stats, Max, Sum_exp order. The operation may be
unmasked or use top-left causal `right_bound=0`; padding, sequence metadata,
bias, ALiBi, sliding windows, bottom-right alignment, dropout, paging, ragged
storage, sink/block masks, virtual tensors, and composed graphs are rejected.

The RESHAPE policy accepts exactly one standard-f32 operation with
`reshape_mode=LOGICAL`. X and Y are external plain strided tensors whose ranks
are independently fixed between one and 64 and may differ. X is read-only and
Y is write-only. Virtual, pass-by-value, ragged, reordered, or additional
tensors are rejected, as are composed graphs. Metadata records two distinct
role UIDs in X, Y order.

The REDUCTION policy accepts exactly one standard-f32 operation in any of the
nine supported modes. X and Y are external plain strided tensors with the same
fixed rank; X is read-only and Y is write-only. Virtual, pass-by-value, ragged,
reordered, or additional tensors are rejected, as are composed graphs. Metadata
records two distinct role UIDs in X, Y order.

Compiled dimensions and `size_bytes` are maxima. At workspace query and
execution, the three Frontend override arrays must have equal counts and unique
external argument UIDs. Each supplied shape preserves rank, has positive
dimensions no larger than its compiled maxima, and uses positive strides that
satisfy the supported non-overlap condition. Its computed storage span must
not exceed the compiled byte bound. An empty list executes the compiled maximum
shape, and a partial list is legal only if the final descriptors still satisfy
the selected policy.

For pointwise, every final external shape remains exactly equal; a shrinking
call therefore normally overrides all external arguments. Virtual UIDs are not
public arguments and cannot be overridden. For MATMUL, final descriptors obey
`A[-2] == C[-2]`, `A[-1] == B[-2]`, and `B[-1] == C[-1]`. On every batch axis,
A and B extents are equal or one and C equals their maximum. This permits a
partial override such as changing one input batch extent to one while preserving
the relation. For SDPA, only B, Sq, and Skv may change. All roles have the same
B; Q/O share Hq and Sq, K/V share Skv, Q/K share Dqk, O/V share Dv, Hq is
divisible by both source-head counts, and every row output is `[B,Hq,Sq,1]`.
Heads and embedding dimensions must equal their compiled values. For RESHAPE,
each final X and Y descriptor keeps its own compiled rank, and their positive
element counts must be equal. Runtime dimensions may repartition that element
count across the fixed axes. This relation applies after complete or partial
overrides. For REDUCTION, compiled X/Y extents freeze the axis classification.
An axis with different compiled extents is reduced and has compiled Y extent
one; runtime Y remains one while runtime X may shrink. On all other axes,
runtime X and Y extents remain equal. Complete and partial overrides preserve
this classification.

The compiler emits dynamic memref dimensions and strides plus `memref.dim`
loop bounds under all five policies. Runtime descriptors carry the resolved values
to in-process and artifact-loaded objects. Pointwise virtual intermediates use
the common runtime dimensions and internal packed views over workspace sized
for serialized maxima. MATMUL loops use runtime C extents, runtime K, and
runtime singleton-batch selection. SDPA loops use runtime Q/O extents and
runtime K sequence length for softmax and V reduction. RESHAPE loops traverse
the runtime Y extent and map each lexicographic linear index back into the
runtime X descriptor. REDUCTION loops use runtime Y output extents and runtime X
reduction extents; AVG derives its divisor from the latter. Alias checks use
resolved external byte spans. Workspace
remains statically bounded, and workspace query performs the same validation
as execution.

Artifact formats `3` through `5` record both context flags and pointwise policy
`1`; v6 adds MATMUL policy `2` and its ordered role UIDs; v7 adds SDPA-forward
policy `3` using the same role-list field; v8 adds logical-RESHAPE policy `4`
with X/Y role UIDs; v9 adds REDUCTION policy `5` with X/Y role UIDs. The v1/v2
readers default override metadata to disabled, v1-v5 readers default role UIDs
to empty, and v6-v8 remain readable with their MATMUL/SDPA/RESHAPE roles. The
pinned Frontend MATMUL sample can execute shapes larger than its fake cache
shape, but DeepForge deliberately requires every runtime dimension and byte
span to fit the serialized maxima: the public UID-map ABI carries pointers, not
allocation lengths, so larger descriptors cannot be validated safely.

MATMUL descriptor overrides change A/B/C runtime descriptors. They are distinct
from the serialized INT32 MATMUL extent-override tensor ports in section 3.1,
which select active regions inside static descriptors. The two mechanisms cannot
be combined in one graph.

### 3.7 C6 Standard f32 SDPA Metadata Extension

This extension is limited to static standard-f32 `SDPA`/`SDPA_BWD`; it does not
apply to FP8/MXFP8 attention or runtime shape overrides. Ragged or paged storage
requires `padding_mask=true` and both INT32 `[B,1,1,1]` sequence-length inputs.
Runtime lengths are caller preconditions in `[0,Sq]` and
`[0,logical_Skv]`; ragged arguments additionally validate their corresponding
values before dispatch.

Forward Q/K/V/O/Stats/Max/Sum_exp and backward Q/K/V/O/dO/Stats/dQ/dK/dV may
independently use ragged storage. Each is an external plain rank-4 logical
tensor whose `ragged_offset_uid` and `ragged_offset_name` identify one separate
external plain INT32 or INT64 `[B+1,1,1,1]` tensor. Prefix values are element
offsets, start at zero, are nondecreasing, and do not exceed the compiled
maximum storage span. Each segment must hold the runtime sequence extent and
the data tensor's inner strided layout. Execution validates these rules before
dispatch and uses the final prefix endpoint, rather than the conservative
maximum, for alias checks. Backward `max_total_seq_len_q` and
`max_total_seq_len_kv` require ragged storage and values in `(0,B*Sq]` and
`(0,B*Skv]`; they are validation hints and do not alter the computation.

K and V may independently use paged storage. A container has dimensions
`[num_blocks,H,block_size,D]`, and its separate INT32 table has logical
dimensions `[B,1,page_slots,1]`. A page table may be external plain storage or
use its own external INT32/INT64 `[B+1,1,1,1]` element prefix. Each compact
batch segment must contain at least `ceil(SEQ_LEN_KV[b]/block_size)` entries.
The logical K/V sequence is the positive `max_seq_len_kv` attribute when
present; otherwise it is inferred from the unpaged peer, Bias, `RNG_DUMP`, or
the minimum available page capacity, in that Frontend order. Capacity must
cover the logical sequence. Page IDs are a caller precondition and must lie in
`[0,num_blocks)`. Generated code substitutes a safe address and zero value for
an invalid ID, so it cannot cause an out-of-bounds container access, but the
result is outside the semantic contract.

Ragged Q/O may be combined with paged K/V. An individual K or V container
cannot be both paged and ragged. The pinned cuDNN Frontend v1.24.0 backward
serializer exposes no K/V page-table ports, so paged backward cannot be
represented by the accepted input schema.

Forward `Block_mask` is an external plain UINT8 tensor with exact dimensions
`[B,Hq,ceil(Sq/128),ceil(ceil(Skv/128)/8)]`. Each bit enables one 128-by-128
query/key tile; key-tile bits are packed least-significant-bit first in each
byte. The block mask composes with the other score-validity predicates.

`SINK_TOKEN` is an external plain FLOAT `[1,Hq,1,1]` tensor supported by forward
and backward. Its per-head value is an additional logit in every valid softmax
row, contributing to row maximum and denominator but not to the V-weighted
output. Backward may emit external plain `DSINK_TOKEN` with the same shape; it
is the gradient reduced over valid batch/query rows and requires the matching
sink input.

Artifact format `5` introduced the ragged storage policy, offset/sequence UIDs,
and a positive logical-sequence divisor. Ordinary ragged arguments use divisor
one; compact page tables use the corresponding cache block size. Formats v6
through v9 retain those fields, and the reader accepts v1-v8 with v4 divisors
defaulted to one.

### 3.8 C6 Runtime Scalar Pass-by-Value Extension

The accepted Frontend runtime form has `is_pass_by_value=true` and a null
`pass_by_value` payload. The tensor must have an explicit UID, rank 1-64 with
every dimension equal to one, a positive non-overlapping layout, and `NONE`
reordering. It is external, input-only, non-ragged, and has type `INT64`,
`INT32`, `HALF`, `FLOAT`, `DOUBLE`, or `BFLOAT16`; the consuming operation port
may impose a narrower type.

The caller supplies a host pointer to the scalar in the ordinary UID map. The
compiler records the tensor as a one-element read argument, so the public ABI,
workspace rules, invocation adapter, and artifact format are unchanged. The
same behavior is exercised by pointwise broadcasting, normalization epsilon,
and FP8 MATMUL scale controls.

Pass-by-value outputs, virtual/ragged/reordered descriptors, and graph-level
runtime shape overrides are rejected. A non-null tensor payload selects the
embedded form in section 3.9 rather than this runtime-pointer form. Nonempty
root `workspace_modifications` and `variant_pack_replacements` are rejected for
every graph form, not only the optimized Conv specialization.

### 3.9 C6 Embedded Pass-by-Value Scalar Extension

The embedded form has the same input-only, external, all-one, plain-layout and
scalar-type constraints as section 3.8. Its tensor `pass_by_value` is exactly
the pinned Frontend variant object `{index,value}`: indices 0-5 denote `INT64`,
`INT32`, `HALF`, `FLOAT`, `DOUBLE`, and `BFLOAT16`, respectively. Integer values
are JSON integers, HALF/DOUBLE/BFLOAT16 values are JSON numbers, and FLOAT is
an eight-hex-digit string containing the exact f32 bit pattern.

Root `pass_by_values` must contain exactly the same typed value under the
tensor's canonical decimal UID. Every tensor payload requires one root entry,
and every root entry requires one UID-assigned tensor payload. Unknown UIDs,
missing counterparts, wrong indices, malformed values, out-of-range INT32,
and unequal values are rejected before lowering.

The value is graph-owned. The compiler emits a private constant `memref.global`
in each target object, binds its view internally, and omits the UID from public
argument metadata. Execution neither requires nor honors a caller pointer for
that UID. This feature needs no dedicated artifact section because all three
objects already carry the constant; reload reproduces the same behavior. The
distinct `has_compile_time_constant` source fields are not serialized by the
pinned v1.24.0 tensor serializer and remain outside this input contract.

## 4. Public Execution Interface

DeepForge exposes the UID-map call shapes of the cuDNN Frontend `v1.24.0`
Graph execute and workspace operations. It does not provide a second
handle-free execute API, descriptor, or raw kernel ABI. The CPU-only form is:

```cpp
namespace deepforge::runtime {

using FrontendHandle = void *;
using VariantPack = std::unordered_map<int64_t, void *>;
using OverrideUids = std::vector<int64_t>;
using OverrideShapes = std::vector<std::vector<int64_t>>;
using OverrideStrides = std::vector<std::vector<int64_t>>;

class Executable {
public:
  import::Status get_workspace_size(int64_t &workspace_size) const;
  import::Status get_workspace_size(
      FrontendHandle handle, int64_t &workspace_size,
      OverrideUids const &uids, OverrideShapes const &shapes,
      OverrideStrides const &strides) const;
  int64_t get_workspace_size() const;
  int64_t get_workspace_size(
      FrontendHandle handle, OverrideUids const &uids,
      OverrideShapes const &shapes, OverrideStrides const &strides) const;
  import::Status execute(FrontendHandle handle,
                         VariantPack &uid_to_host_ptr,
                         void *workspace) const;
  import::Status execute(
      FrontendHandle handle, VariantPack &uid_to_host_ptr, void *workspace,
      OverrideUids const &uids, OverrideShapes const &shapes,
      OverrideStrides const &strides) const;
};

} // namespace deepforge::runtime
```

The contract is:

- Argument order and map type match the Frontend UID-map execute overload. The
  map remains a non-const reference for call compatibility, but DeepForge does
  not modify its keys or addresses.
- `handle` exists only for call-shape compatibility and may be null. The CPU
  runtime never dereferences, queries, or forwards it. The CPU-only form uses
  opaque `void *` and requires no CUDA Toolkit or cuDNN backend.
- The variant pack contains every serialized non-virtual argument UID. Extra
  UIDs are ignored.
- Pointers are CPU-addressable host pointers, not CUDA device pointers.
- Every argument meets its recorded alignment. Generated code must not
  unconditionally claim 64-byte alignment.
- The caller supplies workspace aligned to 64 bytes and allocates the byte
  count returned by `get_workspace_size()`. Workspace may be null when that
  count is zero.
- Valid argument and workspace byte ranges do not overlap when either range is
  writable. The runtime can
  validate UIDs, null pointers, alignment, integer overflow, and computable
  range overlap. The C++ interface carries no allocation lengths and therefore
  cannot portably prove that a pointer refers to a sufficiently large object;
  the caller remains responsible for capacity. Only after these checks may an
  internal kernel rely on `noalias`.
- The caller allocates Y; DeepForge does not replace its address in the variant
  pack.
- Executable compile metadata is immutable. Concurrent calls are safe with
  distinct workspaces; one workspace cannot be shared by concurrent
  executions.
- `DeepForge/Runtime/Executable.h` can be included without MLIR, CUDA, or cuDNN
  include paths. MLIR ExecutionEngine, ORC, and compiler metadata appear only
  in internal factory headers.

CPU-only `deepforge::import::Status` provides `is_good/is_bad`, stable error
codes, and a message. It preserves the Frontend-shaped error-checking style but
is not binary-compatible with `cudnn_frontend::error_t`. Runtime failures map
to stable semantics such as `DFE_INVALID_VARIANT_PACK`,
`DFE_UNSUPPORTED_CPU_FEATURE`, and `DFE_GRAPH_EXECUTION_FAILED`. Private
generated-kernel arguments and memref descriptors exist only inside the
runtime. Both raw object functions and `_mlir_ciface_<symbol>` loader wrappers
are ELF `GLOBAL HIDDEN`.

This guarantees the same C++ call shape. It does not promise binary
interchangeability between a DeepForge `Executable` and an NVIDIA `Graph`;
`std::unordered_map`, `std::string`, and compiler ABI remain subject to the
pinned host toolchain in the README.

## 5. Workspace and Buffer Ownership

Every intermediate buffer comes from caller-provided workspace. MVP kernels
must not call `malloc` or `free`:

1. Run One-Shot Bufferize exactly once after Tensor/Linalg transforms.
2. Materialize `memref.alloc` and any remaining
   `bufferization.alloc_tensor` as static memref allocations. This is
   post-bufferization lowering, not a second bufferization.
3. `deepforge-workspace-plan` assigns aligned offsets to static temporaries.
4. Rewrite temporary `memref.alloc` operations as workspace
   `memref.view`/subview operations.
5. No owned allocation remains in final IR, so the kernel needs no deallocation.

The explicitly padded tensor is the primary MVP workspace consumer. A future
boundary-condition implementation or pad/conv fusion may remove it without
changing the public interface.

## 6. Numeric Semantics

The MVP uses f32 inputs, f32 weights, and f32 accumulation. Within the
reduction for one output element, it permits:

- FMA;
- addition reassociation across SIMD lanes;
- R/S/C loop unrolling or tiling.

It does not permit unrelated fast-math assumptions such as `nnan`, `ninf`,
`nsz`, or approximate reciprocal. The default validation rule is:

```text
abs(actual - reference) <= 1e-4 + 1e-3 * abs(reference)
```

The reference uses the same f32 inputs and accumulates in f64. Tests may
override tolerance through explicit parameters, but every relaxation must be
visible in the test report. Dedicated NaN and Inf tests compare classification
instead of applying the finite-value tolerance above.

## 7. CPU Feature Dispatch and Tails

Compiled output contains three internal variants:

| Variant | Vector width | Preconditions |
|---|---:|---|
| scalar | 1 | x86-64 baseline |
| AVX2 | 8 x f32 | AVX2 + FMA |
| AVX-512 | 16 x f32 | AVX-512F + FMA |

The runtime uses CPUID to select the highest supported variant. Each SIMD
variant uses a scalar cleanup loop for `C % VF`, so neither C nor K must be
divisible by a vector width. A function containing a higher ISA must never run
on an unsupported CPU.

This dispatch is a capability and safety decision, not a schedule performance
cost model.

## 8. Failure Strategy

The following are compile-time errors with no silent fallback: schema or
version mismatch, dynamic behavior outside the declared override subset,
layouts or data types outside a per-operation subset, unknown nodes,
inconsistent output shapes, and dimension or byte-count overflow.

Insufficient CPU features are not an error; the runtime falls back to a lower
variant. AVX selection must check both CPUID and operating-system support for
XMM/YMM/ZMM state, for example through `OSXSAVE` and `XGETBV`. It cannot rely
only on hardware feature bits. Numeric tolerance failures, missing UIDs, null
pointers, and overlapping buffers are execution or test errors with actionable
error codes. Workspace pointer and size follow the
`get_workspace_size()` query contract. The Frontend-shaped MVP execute call has
no additional size parameter.

### 8.1 Stable Importer Diagnostics

P1 importer `Status::code()` and message prefixes use these stable identifiers:

| Diagnostic | Meaning |
|---|---|
| `DFE_INVALID_ARGUMENT`, `DFE_IO_ERROR` | Empty input, invalid arguments, or file-read failure |
| `DFE_PARSE_ERROR` | JSON/UBJSON syntax, truncation, trailing bytes, or canonical UBJSON failure |
| `DFE_SCHEMA_VERSION_MISMATCH`, `DFE_FRONTEND_VERSION_MISMATCH` | Graph schema or Frontend producer version mismatch |
| `DFE_MISSING_FIELD`, `DFE_INVALID_FIELD_TYPE`, `DFE_INVALID_VALUE` | Missing fixed-schema field, wrong type, or invalid value |
| `DFE_UNSUPPORTED_NODE`, `DFE_UNSUPPORTED_DATA_TYPE`, `DFE_UNSUPPORTED_EXECUTION_METADATA` | Valid input protocol with semantics outside the MVP |
| `DFE_DUPLICATE_UID`, `DFE_MISSING_UID` | Non-unique or missing UID, name fallback, or unresolved reference |
| `DFE_INVALID_LAYOUT`, `DFE_INVALID_SHAPE` | Invalid packed layout, rank, static dimension, or Conv output shape |
| `DFE_DIMENSION_OVERFLOW` | Integer, dimension product, packed stride, or byte count is out of range |

P2 reuses these stable codes. Physical layout, padding, and shape failures in
the canonical model return `DFE_INVALID_LAYOUT`, `DFE_INVALID_SHAPE`, or
`DFE_DIMENSION_OVERFLOW`. A standard MLIR verifier failure or invalid generated
module returns `DFE_INVALID_VALUE` with a message beginning
`DFE_INVALID_VALUE: mlir:`. P2 introduces no new dialect or custom operation;
UID, shape, and padding data travels in separate `Conv2DCompileMetadata`.

Runtime and artifact loading add fixed numeric assignments that must not be
reordered:

| Value | Diagnostic | Meaning |
|---:|---|---|
| 17 | `DFE_INVALID_VARIANT_PACK` | UID, null, alignment, computable alias, or workspace precondition failed |
| 18 | `DFE_UNSUPPORTED_CPU_FEATURE` | A forced object variant is unsupported by current CPUID or OS XSTATE |
| 19 | `DFE_GRAPH_EXECUTION_FAILED` | JIT or internal entry invocation failed |

See [DFO Artifact Format](../artifact-format.en.md) for `.dfo` layout, version
validation, ORC loading, and the native-code trust boundary. The FNV checksum
detects accidental corruption only; it is neither a signature nor an execution
sandbox. The runtime loads artifacts only from trusted sources.

## 9. Compatibility Decisions

- The producer is pinned to cuDNN Frontend `v1.24.0`; the MVP rejects other
  producer versions.
- Both Graph JSON and single-document UBJSON are accepted. GPU plan metadata is
  neither retained nor re-exported.
- Public execute is fixed to the handle, UID map, and workspace call shape. No
  handle-free overload is published, and the CPU runtime always ignores the
  handle.
- The public ABI is a Frontend-shaped CPU-only form: opaque `void *` handle,
  UID variant pack, workspace, and DeepForge status. CUDA/cuDNN headers are not
  required.
- Exact official `cudnnHandle_t` and `cudnn_frontend::error_t` types, GPU
  execution, and Frontend samples are outside the MVP. Introducing them later
  requires a separate ABI and dependency review.
