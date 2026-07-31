# DFO Artifact 格式

[English](artifact-format.en.md)

## 1. 目标

`.dfo` 是 DeepForge CPU MVP 的可重复编译产物。它保存运行时恢复一个
`Executable` 所需的编译 metadata、workspace plan 和三个 x86-64 object，而不暴露
memref descriptor 或生成 kernel 的裸 ABI。当前 writer 输出格式版本 `2`；reader
同时接受版本 `1` 的 Conv2D artifact，并为其重建 argument table。

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
u32 format_version = 2
u32 endian_marker = 0x01020304

string deepforge_version
string llvm_version
string cudnn_frontend_version
string target_triple
string public_function_name

u32 tensor_argument_count
repeated tensor_argument {
  i64 uid
  string name
  u32 data_type
  u32 access                  # 0 read, 1 write, 2 read-write
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
range、workspace 64-byte 对齐及整数溢出。read/read alias 合法；任何涉及 write
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
顶层编码发生变化时必须增加格式版本。版本 `1` 的字段顺序和语义保持冻结，仅用于
读取；新编译结果写版本 `2`。
