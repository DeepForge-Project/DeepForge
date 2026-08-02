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

39 行都达到 `parsed`：JSON 与 canonical UBJSON 都可被接受和归一化，已知错误
字段会被拒绝，reference 构成有序 DAG。C5 完成时，全部 39 行都具有明确声明的
`validated` 执行子集：

| Tag | 已验证 CPU 子集 |
|---|---|
| `CONV_FPROP` | 静态 f32 rank 3-5、group channel、正 stride/dilation、非负非对称 padding，以及 `CROSS_CORRELATION` 或 `CONVOLUTION` |
| `CONV_DGRAD` | 同一 convolution 子集的 X 梯度 |
| `CONV_WGRAD` | 同一 convolution 子集的 W 梯度 |
| `RESHAPE` | `LOGICAL` mode、元素数相同；`VIEW_ONLY` alias 延后 |
| `TRANSPOSE` | 静态 permutation，每个轴恰好出现一次 |
| `SLICE` | rank 保持、半开区间不越界、正整数 stride |
| `CONCATENATE` | 静态编号输入和非负 axis；不支持 `in_place_index` |
| `POINTWISE` | 全部 50 个 mode、f32 输出、尾维对齐的 NumPy broadcast |
| `REDUCTION` | 全部 9 个 mode，输出保持 rank，被归约维度为 1 |
| `MATMUL` | 相同且至少为 2 的 rank、batch broadcast、无维度 override、padding value 为 0 |
| `RESAMPLE` | 3 个 pooling mode 加整数 `NEAREST`，支持 3 个 padding mode 且无 index 输出；`BILINEAR` 因 v1.24.0 序列化遗漏 fraction denominator 而被拒绝 |
| `ADA_LAYER_NORM` | training/inference、同 rank broadcast 参数、保留 batch 的统计量 |
| `ADA_LAYER_NORM_BPROP` | 使用保存的 mean/inverse standard deviation、显式全 1 shape epsilon tensor、可选 bias gradient |
| `BATCHNORM` | training、per-channel 参数/统计量、可选且全有或全无的 running-stat 更新、空 `peer_stats` |
| `BATCHNORM_INFERENCE` | 使用外部提供的 per-channel mean 和 inverse standard deviation |
| `BN_FINALIZE` | per-channel sum/square-sum finalize、等价 affine、保存统计量和 sample running variance |
| `DBN` | batch normalization data/parameter gradient，要求空 `peer_stats` |
| `DBN_WEIGHT` | batch normalization parameter gradient 和等价 data-gradient coefficient |
| `GENSTATS` | per-channel sum 和 square sum |
| `INSTANCE_NORM` | training/inference、per-channel 参数、per-instance/channel 统计量 |
| `INSTANCE_NORM_BPROP` | 使用保存统计量计算 data 和 per-channel parameter gradient |
| `LAYER_NORM` | training/inference；归一化轴由同 rank broadcast scale shape 推导 |
| `LAYER_NORM_BPROP` | 使用保存统计量和全 1 shape epsilon tensor 计算 data/parameter gradient |
| `RMS_NORM` | training/inference、可选 bias、由 scale shape 推导 RMS 统计量 |
| `RMS_NORM_BPROP` | data/scale gradient，可选 bias gradient |
| `RNG` | 静态 f32 Bernoulli 输出；使用序列化 fixed seed，或显式 scalar INT64 `Seed`/`Offset` |
| `ROPE` | 静态 f32 BHSD split-half rotation、可选 scaled pass-through 前缀和 `[S,1,1,R]` frequency |
| `ROPE_BWD` | 已验证 full/partial `ROPE` 变换的 adjoint，包含 `output_scale` |
| `SDPA` | 静态 f32 BHSD attention；在下述约束内支持 GQA、bias、scale、ALiBi、padding、top-left/bottom-right window、两类 dropout 和序列化 row/RNG 输出 |
| `SDPA_BWD` | 同一 attention 子集的 data/可选 bias gradient，读取序列化 `O` 和 log-sum-exp `Stats` |
| `BLOCK_SCALE_DEQUANTIZE` | FLOAT compute；X 为 FP4 E2M1、FP8 E4M3/E5M2 或 INT4；scale 为 f32/f16/bf16/FP8 E4M3/E8M0；尾部静态 block dimension；Y 为 f32/f16/bf16；E4M3/E8M0 scale 可使用 `F8_128x4` |
| `BLOCK_SCALE_QUANTIZE` | FLOAT compute；X 为 f32/f16/bf16；单个可整除静态 block axis；Y 为 FP4 E2M1 或 FP8 E4M3/E5M2，并输出对应 E4M3/E8M0 scale；FP8 scale 输出可使用 `F8_128x4` |
| `MATMUL_FP8` | A/B 为 FP8 E4M3/E5M2、FLOAT scalar control/amax、静态 rank >= 2 batch broadcast；C 为 FP8/f32/f16/bf16；无 M/N/K override |
| `MOE_GROUPED_MATMUL` | `mode=NONE`、`top_k` 为 0/1、token `[1,T,K]`、weight `[E,K,N]`、INT32 first-token offset `[E,1,1]`，共享 f32/f16/bf16 data type |
| `MOE_GROUPED_MATMUL_BWD` | 同一静态 `mode=NONE` tensor/offset layout 的 per-expert weight gradient |
| `SDPA_FP8_FWD` | 静态 FP8 E4M3/E5M2 BHSD/GQA、scalar FLOAT scale control、两种 diagonal alignment/window、可选 row output 和 amax；无 padding、dropout、ALiBi |
| `SDPA_FP8_BWD` | 同一子集的 Q/K/V gradient 和 amax，读取 serialized `O`、`Stats`、FP8 dO 和 scalar FLOAT scale control |
| `SDPA_MXFP8_FWD` | 静态 BHSD/GQA、32 元素 E8M0 descale block、f16/bf16/f32 O、必需 Stats/amax；descale 使用 `NONE` 或 `F8_128x4` ordering |
| `SDPA_MXFP8_BWD` | 静态 transpose-oriented Q/K/dO 输入和 E8M0 block descale、f16/bf16/f32 gradient/amax；使用 f32 dS reference approximation；descale 使用 `NONE` 或 `F8_128x4` ordering |

全部已验证通用行都要求静态正维度、显式 UID、f32 graph context、正且不重叠的
stride、`NONE` reorder，并且不是 pass-by-value 或 ragged tensor。data tensor 为
f32；C4 额外允许在文档指定端口使用 INT32 sequence length 和 scalar INT64 RNG
seed/offset。virtual 中间值使用规划的 workspace。Convolution group 数由
`X.C / W.C` 推导，输出 channel 必须可被 group 数整除。

9 个 C5 特殊行使用 FLOAT accumulation 和 CPU 软件 conversion。FP8 E4M3/E5M2
使用 saturation + round-to-nearest-even；E8M0 是无符号 2 的幂 scale storage，并
有 canonical NaN；FP4 E2M1 和 INT4 每 byte 存两个逻辑值。只启用上表声明的
operation/port 组合；识别 f64、integer、boolean 或其他 serializer enum，不表示
它可用于通用操作。

两个已验证 MoE 行都把运行时 `FirstTokenOffset` 内容作为调用者前置条件：第 0 个
元素必须为 0，所有值单调非降且位于 `[0,T]`。序列化图只固定 offset tensor 的
shape 和 type，不携带这些运行时数值，编译阶段无法验证它们。

MXFP8 block size 固定为 32。C6 已解码正常 Frontend MXFP8 producer 生成的
`F8_128x4` scale ordering。Padded descriptor 的最后两个 axis 是可互换顺序的 M/K；
M 可被 128 整除、K 可被 4 整除、K stride 为 1、M stride 为 K，leading axis
按 packed 排列。该布局只开放给 block-scale conversion 的 E4M3/E8M0 scale
输入/输出和 E8M0 MXFP8 `Descale_*` 输入；其他端口以及 `INT8x32`/`F16x16`
reorder format 会被拒绝。
FP8/MXFP8 attention 拒绝 padding、dropout、ALiBi、block mask、sink token
和未列出的可选端口；可能产生全 mask row 的 window 组合也会被拒绝。Backward
读取外部提供的 log-sum-exp `Stats`。

epsilon、momentum、accumulation count 等 normalization scalar 输入必须是所有维度
均为 1 的显式 f32 tensor；scalar pass-by-value metadata 仍延后。非空分布式
`peer_stats` 不可执行。running-stat 端口必须全有或全无，运行时值必须提供正的
accumulation count 和有效 epsilon。

`ROPE` 旋转最后一个偶数宽度的 `rope_dim`；`rope_dim == 0` 时旋转整个末维，
前缀为 scaled pass-through。`FREQS` shape 为 `[S,1,1,rope_dim]`，split-half
变换读取其前半。`ROPE_BWD` 使用相同 scale，是该线性变换的 adjoint。

Validated SDPA 使用 rank-4 BHSD Q/K/V/O 和 unit embedding stride，K/V head 可各自
以整数比例做 GQA。Bias/dropout mask 按尾轴 broadcast；padding length 为 INT32
`[B,1,1,1]`。两种 diagonal alignment 和 left/right bound 均可执行；v1.24.0 的
bottom-right causal 路径不与 bias、ALiBi 或 dropout 组合。ALiBi 要求
`right_bound == 0`。Probability dropout 接收 scalar INT64 seed/offset，并可输出
`RNG_DUMP`；custom dropout 接收 mask 和显式 scale，backward 还接收 scale inverse。
静态 f32 forward 还接受独立 external ragged Q/K/V/O storage（INT32/INT64
`[B+1,1,1,1]` element prefix）和独立 paged K/V container
`[num_blocks,H,block_size,D]`（external INT32 `[B,1,page_slots,1]` table）。两种
形式都要求 padding 和两个 sequence length，page ID 必须是合法 container block
index；ragged Q/O 可与 paged K/V 组合。Backward ragged/paged、packed table/row
output、block mask、sink token 和上述 C5 特殊可选能力仍延后。

相同 seed/offset 的 CPU Bernoulli stream 在 DeepForge 各 CPU variant 间稳定且
bit-identical；它属于 CPU 实现定义，不承诺复现 cuDNN GPU Philox 的 bit pattern。

Comparison、logical 和 `GEN_INDEX` 的结果仍以 f32 `0`/`1` 或 f32 index 表示。
连接端口均接受 tensor type 时，C2-C6 操作可在同一个有序 DAG 中混合。C6 接受
两个 dynamic context flag。Runtime shape override 仅可执行一个无 broadcast 的
`POINTWISE` node，其 argument 必须是编译 dimension 相同的 external plain f32
tensor。编译 dimension 是上界；runtime dimension 必须为正且不超过该上界，
runtime stride 必须满足当前支持的正且不重叠条件，每个 storage span 必须位于编译
bound 内。dynamic flag 单独出现时只持久化 metadata，不改变静态 descriptor 语义。
显式 alias、其他动态 operation、SDPA forward 子集外的 ragged tensor 和文档所列
`F8_128x4` scale 子集以外的物理 reorder 处理延后。新 artifact 使用 format v4
记录 ragged storage reference；v1-v3 继续按普通 strided storage 读取。

Schema 识别通过不表示该 tag 的每一种配置都可 lowering 或在 CPU 执行。声明子集
以外的属性组合返回 `kUnsupportedOperation`，而不是 `kUnsupportedNode`。
`validated` 仅指上表声明的子集，不代表该 tag 的所有合法 cuDNN backend 配置。
