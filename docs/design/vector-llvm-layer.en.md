# Vector and LLVM Layer Design

[中文](vector-llvm-layer.md)

## 1. Responsibility

This layer completely converts standard Vector, Arith, MemRef, Func, and
control-flow dialects to the LLVM Dialect, then translates the result to LLVM
IR. The MVP does not pass through a DeepForge Machine Dialect and does not
hard-code AVX-512 intrinsic names.

The Vector Dialect describes vector semantics. The LLVM backend selects actual
instructions from the module triple and target features. `vector.fma` is not
required to print as a particular x86 intrinsic; the contract is to produce
legal LLVM vector IR and obtain FMA instructions when target features permit.

## 2. MVP Vector Form

A complete vector block is:

```mlir
%xv = vector.load %x[%n, %h, %w, %c]
    : memref<?x?x?x?xf32>, vector<16xf32>
%wv = vector.load %weight[%k, %r, %s, %c]
    : memref<?x?x?x?xf32>, vector<16xf32>
%acc = vector.fma %xv, %wv, %acc0 : vector<16xf32>
%sum = vector.reduction <add>, %acc : vector<16xf32> into f32
```

Real MVP shapes are static; `?` above only omits concrete document dimensions.
The C tail is scalar and performs no out-of-bounds vector load, so masks are
not a correctness requirement.

Vector memory access is contiguous along C. Unless the runtime contract proves
a stronger alignment, loads and stores use 4-byte alignment or omit stronger
alignment metadata. A 64-byte-aligned workspace does not imply that input,
weight, or output pointers are aligned to 64 bytes.

## 3. Complete Conversion Order

The MVP order from loop/vector IR to the LLVM Dialect is:

```text
canonicalize
  -> convert-vector-to-llvm
  -> lower-affine                 # if affine remains
  -> convert-scf-to-cf            # if scf remains
  -> expand-strided-metadata      # when dynamic memref metadata remains
  -> convert-index-to-llvm
  -> convert-arith-to-llvm
  -> finalize-memref-to-llvm
  -> convert-func-to-llvm
  -> convert-cf-to-llvm
  -> reconcile-unrealized-casts
  -> final module/function legality check
```

The implementation follows the MLIR 22.1.8 pass registry. A pipeline builder
must not run only `vector-to-llvm` and claim conversion is complete. Each
conversion declares its legal target and illegal dialects. The pipeline fails
and prints IR if any `affine`, `scf`, `memref`, `arith`, `index`, `func`, `cf`,
or DeepForge temporary operation remains.

`mlir-translate --mlir-to-llvmir` translates the LLVM Dialect to LLVM IR; it
does not replace the dialect conversions above.

## 4. LLVM IR and Object Code

Each target-specific LLVM module contains a hidden function with the same
layout and numeric contract. The packaged artifact contains three variants:

```text
deepforge_conv2d_scalar(...)
deepforge_conv2d_avx2(...)
deepforge_conv2d_avx512(...)
```

Only an internal adapter supplies their arguments; none is exported as a
public ABI. The modules use these target features:

```text
scalar:  x86-64 baseline
avx2:    +avx2,+fma
avx512:  +avx512f,+fma
```

The LLVM backend may select matching machine instructions for `<8 x float>`,
`<16 x float>`, and FMA. Tests inspect assembly and target attributes instead
of depending on the printed name of a current LLVM intrinsic. Before choosing
an AVX variant, the runtime must also verify that the operating system saves
the relevant SIMD state. Checking CPUID without `OSXSAVE` and `XGETBV` can
still cause an illegal instruction on otherwise capable hardware.

Both the raw function and MLIR-generated `_mlir_ciface_<symbol>` wrapper are
marked ELF `GLOBAL HIDDEN`. The wrapper is resolved internally by the `.dfo`
loader or ExecutionEngine; it is not a shared-library public symbol. ISA
inspection tests verify both visibility and actual register classes.

## 5. Numeric Semantics and Fast Math

Allowed optimizations are:

- f32 FMA;
- reassociation across reduction lanes;
- R/S/C loop unrolling.

The MVP does not enable `nnan`, `ninf`, `nsz`, `afn`, or approximate math. A
future `--allow-reassociation` option may affect only the Conv2D reduction, must
be recorded in executable metadata, and remains subject to the tolerance in
`contracts.en.md`.

## 6. Memory Attributes

After the runtime validates pointer metadata, computable range overlap,
alignment, and non-overlap, the internal adapter may pass readonly/noalias
semantics to a kernel. The compiler cannot infer 64-byte external address
alignment from serialized strides and cannot propagate workspace alignment to
X/W/Y.

All memref descriptor construction, stride materialization, and workspace
views remain inside the adapter and `WorkspacePlan`. The public interface still
contains only the ignored opaque handle, UID map, and workspace pointer.

## 7. Scalar Path

The scalar path is a dedicated, semantically equivalent lowering, not an
undefined fallback after vectorization failure:

```text
same direct-conv loop
  -> scalar multiply/add
  -> standard LLVM conversion
```

It serves CPUs without AVX, debugging, numeric comparison, and scalar tail
cleanup. The compiler does not convert vector operations back to SCF and then
attempt to recover their semantics.

## 8. AMX Handling

AMX is outside this layer's MVP path. Before adding bf16/AMX:

1. define filter packing, tile-configuration lifetime, OS XSTATE access, and
   thread state;
2. evaluate upstream X86/AMX conversions in LLVM/MLIR 22.1.8;
3. define separate workspace and numeric modes for AMX kernels;
4. provide feature dispatch with a safe fallback on CPUs without AMX.

Undefined `machine.amx_*` operations or a manually written
`llvm.x86.tileloadd64.internal` intrinsic are not acceptable as the current
implementation strategy.

## 9. Exit Conditions

Before translation, the IR must satisfy all of the following:

- it contains only the LLVM Dialect, LLVM-compatible intrinsic dialects, and
  required module attributes;
- no Tensor, Linalg, Affine, SCF, MemRef, Arith, Index, Func, or CF operation
  remains;
- every `unrealized_conversion_cast` is reconciled;
- `llc` can compile the LLVM IR for the corresponding target features;
- scalar, AVX2, and AVX-512 modules have identical ABI metadata.
