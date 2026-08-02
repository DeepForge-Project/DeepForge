# DFO Artifact 格式

[English](artifact-format.en.md)

## 1. 目标

`.dfo` 是 DeepForge CPU MVP 的可重复编译产物。它保存运行时恢复一个
`Executable` 所需的编译 metadata、workspace plan 和三个 x86-64 object，而不暴露
memref descriptor 或生成 kernel 的裸 ABI。当前 writer 输出格式版本 `5`；reader
同时接受版本 `4` 的 ragged-storage artifact、版本 `3` 的 shape-override artifact、
版本 `2` 的静态 metadata artifact 和版本 `1` 的 Conv2D artifact。版本 `1` 会重建
argument table；版本 `1`、`2` 的 dynamic/override metadata 均默认为关闭，版本 `1`
至 `3` 的 tensor storage 默认使用普通 strided addressing，版本 `4` 的全部 ragged
sequence divisor 默认为 1。

写入和读取入口位于 `DeepForge/Compiler/Artifact.h`：

```cpp
serialize_artifact(compilation, bytes);
write_artifact(path, compilation);
parse_artifact(bytes, info);
read_artifact(path, info);
load_artifact_executable(path_or_bytes, executable, optional_info);
```

文件写入先生成带 process ID 和进程内 sequence 的相邻唯一临时文件，再以 rename
发布。读取端先完成边界、版本和 checksum 校验，成功后才修改调用者的输出对象。
并发写同一目标不会共享临时文件；最终目标采用最后一次成功发布的完整内容。

## 2. 编码规则

所有整数使用 little-endian。字符串编码为 `u32 byte_count` 加原始 UTF-8 字节；
object blob 编码为 `u64 byte_count` 加原始字节。当前 writer 按固定顺序输出字段，
相同输入独立编译和重复序列化必须逐字节一致；测试同时比较 JSON 与其 canonical
UBJSON 编译所得的完整 artifact。

```text
magic[8] = "DFOBJ\r\n\x1a"
u32 format_version = 5
u32 endian_marker = 0x01020304

string deepforge_version
string llvm_version
string cudnn_frontend_version
string target_triple
string public_function_name

u32 dynamic_shape_enabled    # boolean
u32 override_shape_enabled   # boolean
u32 shape_override_policy    # 0 none, 1 exact pointwise

u32 tensor_argument_count
repeated tensor_argument {
  i64 uid
  string name
  u32 data_type
  u32 access                  # 0 read, 1 write, 2 read-write
  u32 storage_policy          # 0 strided, 1 ragged batch prefix
  i64 ragged_offset_uid       # storage_policy != 1 时为 0
  i64 ragged_sequence_uid     # storage_policy != 1 时为 0
  i64 ragged_sequence_divisor # 普通 ragged data 为 1
  u64 alignment
  u64 size_bytes
  u32 rank
  i64 dimensions[rank]
  i64 strides[rank]
}

u32 adapter_kind
u64 adapter_metadata_size
u8 adapter_metadata[adapter_metadata_size]

# adapter_kind 0：过渡期 ranked-memref Conv2D adapter payload
i64 x_uid, w_uid, y_uid
i64 x_shape[4], w_shape[4], y_shape[4], padded_x_shape[4]
i64 pre_padding[2], post_padding[2], stride[2], dilation[2]

# adapter_kind 1：通用 ranked-memref pointer-table adapter
# adapter_metadata_size 为 0；tensor argument table 已包含所需信息

u64 workspace_size
u64 workspace_alignment
u32 workspace_allocation_count
repeated allocation {
  string name
  u64 offset, size, alignment, live_start, live_end
}

f64-bits absolute_tolerance
f64-bits relative_tolerance
u32 variant_count = 3
repeated variant {
  u32 variant_id             # 0 scalar, 1 AVX2, 2 AVX-512
  string symbol
  string required_features
  u64 object_size
  u8 object[object_size]
}

u64 fnv1a_64_checksum        # covers every preceding byte
```

argument table 与带长度的 adapter section 有意分离：前者与 operation 无关。
adapter kind `0` 保留 ranked-memref Conv2D 调用；基础多节点图使用 adapter kind
`1`，不携带 operation 专属 payload。runtime 按有序 argument table 构造每个
ranked-memref descriptor 和 workspace descriptor，再调用隐藏的 pointer-table
wrapper。这个内部 adapter 不改变公开的 handle + UID variant-pack + workspace
调用形状。reader 会拒绝未知 adapter kind，也会拒绝非空的 kind-1 metadata。

Runtime pass-by-value scalar 不增加 argument-table flag。编译后它与 external 单元素
只读 argument 具有相同 CPU 调用语义：调用者按 UID 提供地址，因此现有 format 可
精确 reload。内嵌/fused scalar 也不增加 argument-table flag，因为其 UID 会被完全
移除。Private constant global 作为数据保存在每个现有 target object 中，因此 format
v5 无需新增 metadata section 即可 reload graph-owned value。Reader 不得从其他普通
单元素 argument 推断任一 pass-by-value 形式。

版本 `3` 将执行 shape 意图与 argument 的编译最大 shape/byte span 分开记录。policy
`1` 表示 object 为 exact-shape pointwise-DAG override 子集生成了 dynamic memref
descriptor 和 runtime loop bound。Virtual 中间值不进入公开 argument table；object
code 从公共 external runtime shape 派生其动态 packed workspace view。Runtime 在调用
前仍会验证每个 external override。过渡期 Conv adapter 拒绝所有 dynamic/override
metadata。

版本 `4` 增加 argument storage policy。Ragged argument 记录
`[B+1,1,1,1]` INT32/INT64 element-prefix tensor UID 和 `[B,1,1,1]` INT32
sequence-length tensor UID；序列化 `size_bytes` 是编译最大 span。执行前校验 prefix
内容和 segment capacity，再以最后一个 prefix endpoint 作为 alias 检查的实际 byte
span。引用无法解析或 metadata 不一致时，不会调用 native code。

版本 `5` 增加 `ragged_sequence_divisor`。普通 ragged tensor 的值为 1；紧凑
paged-attention page table 记录对应 K/V cache block size，因此 runtime 逻辑 sequence
length `S` 在该 batch segment 中需要 `ceil(S/divisor)` 个 table entry。Divisor 必须
为正，且不能令编译逻辑 sequence bound 溢出。版本 `4` artifact 按 divisor 1 解释。

数值契约固定为 `abs <= 1e-4 + 1e-3 * abs(reference)`。三个符号分别为
`<base>_scalar`、`<base>_avx2` 和 `<base>_avx512`；object 内的 C-interface wrapper
为 `_mlir_ciface_<symbol>`。原始函数和 wrapper 都是 ELF `GLOBAL HIDDEN`，不会成为
DeepForge shared-library 的公开 ABI。

## 3. 装载

loader 要求 artifact 的 DeepForge、LLVM、Frontend 和格式版本受当前 runtime 支持，
并要求 target triple 与当前主机精确一致。每个 object 被加入独立 LLVM ORC `LLJIT`，
避免三个变体的同类内部符号相互冲突。loader 解析 C-interface wrapper 后，将入口、
metadata 和 workspace plan 交给对应 runtime adapter。包含 MLIR math lowering 的
object 从当前进程解析标准 `libm` 符号。

执行前会验证有序 argument table、UID、空指针、每个 tensor 的 alignment 和 byte
range、runtime override policy/bound、workspace 64-byte 对齐、ragged
prefix/reference/divisor 一致性及整数溢出。read/read alias 合法；任何涉及 write
argument 或 workspace 的重叠都会被拒绝。CPUID、FMA、OSXSAVE 和 XGETBV 决定
AVX-512、AVX2 或 scalar 的安全选择；装载 object 本身不执行高 ISA 代码。

## 4. 信任边界

FNV-1a 只检测意外损坏，不是密码学签名。能够重新计算 checksum 的攻击者也能替换
object section；ORC 装载后该 object 以当前进程权限执行。因此：

- 只装载由可信 DeepForge 构建或可信发布渠道产生的 `.dfo`；
- 不把 parser、checksum、UID 校验视为 native-code sandbox；
- 若未来接受不可信 artifact，必须增加签名验证和隔离执行，不能只升级 checksum。

## 5. 兼容策略

reader/loader 拒绝未知 format version、producer 版本、端序、不匹配的 target triple、
数值契约、重复 UID、不一致 shape/padding、非法 workspace
alignment/range/lifetime、错误的 argument table 或 adapter payload、variant
symbol/feature、重复 variant、空 object、截断、尾随 payload 和 checksum 错误。
顶层编码发生变化时必须增加格式版本。版本 `1`、`2`、`3`、`4` 的字段顺序和语义
保持冻结，仅用于读取；新编译结果写版本 `5`。
