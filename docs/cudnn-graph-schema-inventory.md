# cuDNN Frontend v1.24.0 序列化 Graph Schema 清单

[English](cudnn-graph-schema-inventory.en.md)

本文记录 DeepForge 已识别的协议范围。清单已直接对照开源 cuDNN Frontend
v1.24.0 的 `graph_properties.h`、`utils/serialize.h`、`graph_interface.h` 和各
node 的 `serialize()` 实现。可执行范围有意小于协议识别范围。

## 1. Graph 外层格式

Importer 要求以下根字段：

| 字段 | 契约 |
|---|---|
| `json_version` | 字符串，必须为 `1.0` |
| `cudnn_backend_version` | 字符串，作为 producer 元数据保留 |
| `cudnn_frontend_version` | 整数，必须为 `12400` |
| `graph_uid` | 无符号 64 位整数 |
| `context` | 下述 context 对象 |
| `nodes` | 非空且有序的 node 数组 |
| `tensors` | 以十进制 UID 或 tensor 名称为键的对象 |

Context serializer 的七个字段都必须存在：`name`、`compute_data_type`、
`intermediate_data_type`、`io_data_type`、`sm_count`、
`is_dynamic_shape_enabled` 和 `is_override_shape_enabled`。可空字段在 canonical
model 中继续保留为空，`sm_count` 必须落在 `int32` 范围内。

`variant_pack_uids`、`pass_by_values`、`workspace_modifications`、
`variant_pack_replacements`、`fe_workspace_size`、`behavior_notes` 和
`tensors_to_dump` 等 plan 字段不属于 node 语义。已验证的 Conv2D adapter 继续执行
原有的严格 execution metadata 检查；通用 node 不会把这些字段误当成 node 属性。

## 2. Tensor 和 Graph 规则

每个 tensor 都包含 `Tensor_attributes` 输出的十个字段：`name`、`data_type`、
`dim`、`stride`、`is_virtual`、`pass_by_value`、`is_pass_by_value`、
`reordering_type`、`uid` 和 `uid_assigned`。`ragged_offset_uid` 与
`ragged_offset_name` 可选，但必须同时出现。Pass-by-value payload 只能为空或数值。
可识别的 reorder 值为 `NONE`、`INT8x32`、`F16x16` 和 `F8_128x4`。

Importer 识别 v1.24.0 的全部 20 个非 sentinel data type；这只是 enum 识别，
不代表已经可执行。Tensor reference 可以是有符号 64 位 UID 或名称，与 Frontend
压缩后的 graph serializer 一致。所有 reference 必须可解析；一个 tensor 最多有
一个 producer；virtual input 必须由更早的 node 产生；每个 virtual tensor 都必须有
producer。

`BATCHNORM` 和 `DBN` 是 serializer 唯一的特殊输入形式：`peer_stats` 是完整 tensor
对象组成的 vector，而不是压缩后 `inputs` 对象中的普通条目。Importer 会保留该属性，
同时将其中的 tensor 归一化到 graph tensor 表和 canonical input list。根节点和内嵌
位置出现完全相同的描述时合并为一个 tensor；同 UID 或同名但描述冲突时拒绝。

## 3. Operation 清单

每行共有的 `name`、`inputs` 和 `outputs` 不在属性列重复。`?` 表示唯一按条件输出
的属性。端口列列出完整 serializer 端口集合，具体 Frontend 操作只会使用其选项
启用的端口。

| 序列化 tag | 序列化属性 | 输入端口 | 输出端口 |
|---|---|---|---|
| `ADA_LAYER_NORM` | `compute_data_type`, `forward_phase` | `X`, `SCALE`, `BIAS`, `EPSILON` | `Y`, `MEAN`, `INV_VARIANCE` |
| `ADA_LAYER_NORM_BPROP` | `compute_data_type` | `DY`, `X`, `SCALE`, `MEAN`, `INV_VARIANCE`, `EPSILON` | `DX`, `DSCALE`, `DBIAS` |
| `BATCHNORM` | `compute_data_type`, `peer_stats` | `X`, `SCALE`, `BIAS`, `EPSILON`, `PREV_RUNNING_MEAN`, `PREV_RUNNING_VAR`, `MOMENTUM` | `Y`, `MEAN`, `INV_VARIANCE`, `NEXT_RUNNING_MEAN`, `NEXT_RUNNING_VAR` |
| `BATCHNORM_INFERENCE` | `compute_data_type` | `X`, `SCALE`, `BIAS`, `MEAN`, `INV_VARIANCE` | `Y` |
| `BLOCK_SCALE_DEQUANTIZE` | `compute_data_type`, `block_size`, `is_negative_scale` | `X`, `scale` | `Y` |
| `BLOCK_SCALE_QUANTIZE` | `compute_data_type`, `block_size`, `axis` | `X` | `Y`, `scale` |
| `BN_FINALIZE` | `compute_data_type` | `SUM`, `SQ_SUM`, `SCALE`, `BIAS`, `EPSILON`, `ACCUM_COUNT`, `PREV_RUNNING_MEAN`, `PREV_RUNNING_VAR`, `MOMENTUM` | `EQ_SCALE`, `EQ_BIAS`, `MEAN`, `INV_VARIANCE`, `NEXT_RUNNING_MEAN`, `NEXT_RUNNING_VAR` |
| `CONCATENATE` | `axis`, `in_place_index` | 连续数字端口 `0..N-1` | `Y` |
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
| `MOE_GROUPED_MATMUL_BWD` | 无 | `DOutput`, `Token`, `FirstTokenOffset` | `DWeight` |
| `POINTWISE` | `compute_data_type`, `mode`, `axis`, `relu_lower_clip`, `relu_upper_clip`, `relu_lower_clip_slope`, `swish_beta`, `elu_alpha`, `softplus_beta` | `IN_0`, `IN_1`, `IN_2`，由 mode 元数约束 | `OUT_0` |
| `REDUCTION` | `compute_data_type`, `mode`, `is_deterministic` | `X` | `Y` |
| `RESAMPLE` | `generate_index`, `resample_mode`, `padding_mode`, `pre_padding`, `post_padding`, `stride`, `window` | `X` | `Y`, `Index` |
| `RESHAPE` | `compute_data_type`, `dim`, `stride`, `reshape_mode` | `X` | `Y` |
| `RMS_NORM` | `compute_data_type`, `forward_phase` | `X`, `SCALE`, `BIAS`, `EPSILON` | `Y`, `INV_VARIANCE` |
| `RMS_NORM_BPROP` | `compute_data_type` | `DY`, `X`, `SCALE`, `INV_VARIANCE` | `DX`, `DSCALE`, `DBIAS` |
| `RNG` | `distribution`, `dim`, `stride`, `seed`, `bernoulli_probability` | `Seed`, `Offset` | `Y` |
| `ROPE` | `compute_data_type`, `output_scale`, `rope_dim` | `INPUT`, `FREQS` | `OUTPUT` |
| `ROPE_BWD` | `compute_data_type`, `output_scale`, `rope_dim` | `DY`, `FREQS` | `DX` |
| `SDPA` | 下述 forward SDPA 属性 | 下述 forward SDPA 端口 | 下述 forward SDPA 输出 |
| `SDPA_BWD` | 下述 backward SDPA 属性 | 下述 backward SDPA 端口 | 下述 backward SDPA 输出 |
| `SDPA_FP8_BWD` | 下述 FP8 backward 属性 | 下述 FP8 backward 端口 | 下述 FP8 backward 输出 |
| `SDPA_FP8_FWD` | 下述 forward SDPA 属性 | 下述 forward SDPA 端口 | 下述 forward SDPA 输出 |
| `SDPA_MXFP8_BWD` | 下述 FP8 backward 属性 | 下述 FP8 backward 端口 | 下述 FP8 backward 输出 |
| `SDPA_MXFP8_FWD` | 下述 forward SDPA 属性 | 下述 forward SDPA 端口 | 下述 forward SDPA 输出 |
| `SLICE` | `compute_data_type`, `slices`, `slice_strides` | `X` | `Y` |
| `TRANSPOSE` | `compute_data_type`, `permutation` | `X` | `Y` |

Forward SDPA 共用属性为 `generate_stats`、`alibi_mask`、`padding_mask`、
`dropout_probability`、`attn_scale_value`、`max_seq_len_kv`、`mma_core_mode`、
`left_bound`、`right_bound`、`diagonal_alignment`、`implementation`、
`is_mxfp8`、`unfuse_fma` 和 `rescale_threshold?`。输入为 `Q`、`K`、`V`、
`Attn_scale`、`Bias`、`SEQ_LEN_Q`、`SEQ_LEN_KV`、`Seed`、`Offset`、
`Dropout_mask`、`Dropout_scale`、`Page_table_K`、`Page_table_V`、`Block_mask`、
`Descale_Q`、`Descale_K`、`Descale_V`、`Descale_S`、`Scale_S`、`Scale_O` 和
`SINK_TOKEN`；输出为 `O`、`Stats`、`Max`、`Sum_exp`、`RNG_DUMP`、`Amax_S`
和 `Amax_O`。

Backward SDPA 属性为 `alibi_mask`、`padding_mask`、`dropout_probability`、
`attn_scale_value`、`left_bound`、`right_bound`、`diagonal_alignment`、
`max_total_seq_len_q`、`max_total_seq_len_kv` 和 `is_deterministic_algorithm`。
输入为 `Q`、`K`、`V`、`O`、`dO`、`Stats`、`Attn_scale`、`Bias`、
`SEQ_LEN_Q`、`SEQ_LEN_KV`、`Seed`、`Offset`、`Dropout_mask`、
`Dropout_scale`、`Dropout_scale_inv` 和 `SINK_TOKEN`；输出为 `dQ`、`dK`、
`dV`、`dBias`、`RNG_DUMP` 和 `DSINK_TOKEN`。

FP8/MXFP8 backward 属性为 `compute_data_type`、`padding_mask`、
`dropout_probability`、`left_bound`、`right_bound`、`diagonal_alignment`、
`attn_scale_value`、`is_deterministic_algorithm`、`is_mxfp8` 和
`rescale_threshold?`。其 35 个输入和 8 个输出的完整列表以
`lib/Import/Schema.cpp` 为规范，并由矩阵测试逐项覆盖。

## 4. Mode 清单

Pointwise mode 的输入元数被严格验证：

- 单输入：`SQRT`、`RELU_FWD`、`TANH_FWD`、`SIGMOID_FWD`、`ELU_FWD`、
  `GELU_FWD`、`SOFTPLUS_FWD`、`SWISH_FWD`、`ERF`、`IDENTITY`、
  `GELU_APPROX_TANH_FWD`、`GEN_INDEX`、`EXP`、`LOG`、`NEG`、`ABS`、`CEIL`、
  `COS`、`FLOOR`、`RSQRT`、`SIN`、`LOGICAL_NOT`、`TAN`、`RECIPROCAL`。
- 双输入：`ADD`、`MUL`、`DIV`、`ADD_SQUARE`、`SUB`、`CMP_EQ`、`CMP_NEQ`、
  `CMP_GT`、`CMP_GE`、`CMP_LT`、`CMP_LE`、`LOGICAL_AND`、`LOGICAL_OR`、
  `MIN`、`MAX`、`MOD`、`RELU_BWD`、`TANH_BWD`、`SIGMOID_BWD`、`ELU_BWD`、
  `GELU_BWD`、`SOFTPLUS_BWD`、`SWISH_BWD`、`GELU_APPROX_TANH_BWD`、`POW`。
- 三输入：`BINARY_SELECT`。

九个 reduction mode 为 `ADD`、`MUL`、`MIN`、`MAX`、`AMAX`、`AVG`、
`NORM1`、`NORM2` 和 `MUL_NO_ZEROS`。

## 5. Capability 含义

C1 完成时，39 行都达到 `parsed`：JSON 与 canonical UBJSON 都可被接受和归一化，
已知错误字段会被拒绝，reference 构成有序 DAG。只有原有静态、连续、rank-4 f32、
unit-stride `CONV_FPROP` 子集达到 `validated` 并可执行。Schema 识别通过不代表已经
lowering 或可在 CPU 执行。已识别但未 lowering 的 node 返回
`kUnsupportedOperation`，而不是 `kUnsupportedNode`。
