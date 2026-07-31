# DFO Artifact Format

[中文](artifact-format.md)

## 1. Purpose

`.dfo` is the reproducible compilation artifact of the DeepForge CPU MVP. It
stores the compile metadata, workspace plan, and three x86-64 objects needed to
restore an `Executable` without exposing memref descriptors or the raw ABI of a
generated kernel. The current writer emits format version `2`; the reader also
accepts version `1` Conv2D artifacts and reconstructs their argument table.

Read and write entry points are declared in
`DeepForge/Compiler/Artifact.h`:

```cpp
serialize_artifact(compilation, bytes);
write_artifact(path, compilation);
parse_artifact(bytes, info);
read_artifact(path, info);
load_artifact_executable(path_or_bytes, executable, optional_info);
```

Writing first creates a unique adjacent temporary file whose name includes the
process ID and an in-process sequence, then publishes it with rename. Reading
validates bounds, version, and checksum before modifying caller output. Two
concurrent writes to the same destination do not share a temporary file; the
destination contains the complete result of the last successful publication.

## 2. Encoding

All integers are little-endian. A string is encoded as `u32 byte_count`
followed by raw UTF-8 bytes. An object blob is `u64 byte_count` followed by raw
bytes. The current writer emits fields in a fixed order. Independent builds and
repeated serialization of identical input must be byte-for-byte identical.
Tests also compare complete artifacts compiled from JSON and its canonical
UBJSON encoding.

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

# adapter_kind 0: transitional ranked-memref Conv2D adapter payload
i64 x_uid, w_uid, y_uid
i64 x_shape[4], w_shape[4], y_shape[4], padded_x_shape[4]
i64 pre_padding[2], post_padding[2], stride[2], dilation[2]

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

The argument table and the length-delimited adapter section are separate on
purpose. The table is operation-independent. Adapter kind `0` preserves the
current ranked-memref Conv2D invocation while later generic wrappers can add a
new adapter kind without placing operation-specific fields in the top-level
format. Readers reject unknown adapter kinds.

The numeric contract is fixed to
`abs <= 1e-4 + 1e-3 * abs(reference)`. Symbols are `<base>_scalar`,
`<base>_avx2`, and `<base>_avx512`. The C-interface wrapper in each object is
`_mlir_ciface_<symbol>`. Raw functions and wrappers are ELF `GLOBAL HIDDEN` and
do not become part of a DeepForge shared-library public ABI.

## 3. Loading

The loader requires artifact format, DeepForge, LLVM, and Frontend versions
supported by the current runtime, and an exact target-triple match with the
host. Each object is added to a separate LLVM ORC `LLJIT` to prevent similar
internal symbols in three variants from colliding. After resolving the
C-interface wrapper, the loader passes entries, metadata, and workspace plan to
the same runtime adapter used by the development-time MLIR ExecutionEngine.

Execution validates the ordered argument table, UIDs, null pointers, per-tensor
alignment and byte ranges, 64-byte workspace alignment, and integer overflow.
Read/read aliasing is legal; overlap involving a write argument or workspace is
rejected. CPUID, FMA, OSXSAVE, and XGETBV determine safe AVX-512, AVX2, or
scalar selection. Loading an object does not itself execute high-ISA code.

## 4. Trust Boundary

FNV-1a detects accidental corruption and is not a cryptographic signature. An
attacker who can recompute the checksum can replace an object section, and ORC
runs a loaded object with the current process's privileges. Therefore:

- load `.dfo` files only from a trusted DeepForge build or release channel;
- do not treat parser, checksum, or UID validation as a native-code sandbox;
- accepting untrusted artifacts later requires signature verification and
  isolated execution, not merely a stronger checksum.

## 5. Compatibility Policy

The reader and loader reject an unknown format or producer version, wrong
endianness, mismatched target triple or numeric contract, duplicate UIDs,
inconsistent shape or padding, invalid workspace alignment, ranges or
lifetimes, invalid argument tables or adapter payloads, invalid variant symbols
or features, duplicate variants, empty objects, truncation, trailing payload,
and checksum mismatch. Changes to the top-level encoding must increment the
format version. Version `1` field order and semantics remain frozen; it is
read-only. New compilations write version `2`.
