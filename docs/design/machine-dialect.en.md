# Machine Dialect: Deferred Design

[中文](machine-dialect.md)

## 1. Current Status

The Machine Dialect is **not part of the MVP**. The current implementation and
main pipeline must not register, generate, or depend on `machine.*` operations,
and no corresponding TableGen files are created.

The MVP uses:

```text
Linalg -> Affine/SCF -> Vector -> LLVM
```

This path is sufficient to represent the scalar, AVX2, and AVX-512 Conv2D
schedules. Adding an intermediate dialect without a defined lowering,
register semantics, and verifier would only hide unresolved design issues.

## 2. Why It Is Deferred

The original design mixed concerns from several different levels:

- SIMD operation semantics;
- ISA instruction selection;
- cache placement;
- AMX tile register and configuration lifetime;
- threading and parallel resources.

Vector and LLVM already represent the MVP SIMD semantics, while the LLVM
backend performs final instruction selection. L1, L2, and L3 caches are not
directly addressable compiler memory spaces. None of these concerns currently
requires a new IR.

## 3. Re-evaluation Criteria

Machine Dialect design starts only when all of the following are true:

1. At least two backends, such as CPU SIMD and GPU or AMX, require the same
   resource-allocation or scheduling semantics.
2. Upstream dialects cannot express that shared semantics.
3. The proposal defines a verifier, canonical form, buffer and alias rules,
   and complete lowering for every target.
4. Independent round-trip, legality, numeric, and performance tests exist.
5. The abstraction does not model cache as an address space or duplicate LLVM
   register allocation inside DeepForge.

## 4. Future AMX Boundary

Introducing AMX is not a matter of replacing `vector<16xf32>` with a tile. The
following must be specified first:

- bf16 inputs, f32 accumulation, and accepted numeric error;
- filter and input packing, especially conversion from cuDNN KRSC to a
  tile-friendly layout;
- tilecfg initialization and teardown, thread migration, and OS XSTATE access;
- live ranges and allocation of the eight tile registers;
- row strides and workspace for tile loads and stores;
- AVX-512 and scalar fallback plus CPUID dispatch.

LLVM/MLIR 22.1.8 provides upstream X86 capabilities that can be evaluated.
Future work should prefer upstream X86/LLVM dialects and conversions over
defining `machine.amx_tile_load` and manually lowering it to an unverified
intrinsic.

## 5. Target Profile and Cache

The MVP target information is compiler configuration, not a Machine operation:

```text
Target configuration:
  triple = x86_64-unknown-linux-gnu
  vector widths = {1, 8, 16}
  cache line = 64 bytes
  cache capacities = future cost-model hints
```

There is no standalone `TargetProfile` IR abstraction in the current
implementation. Cache capacity may only influence future tiling scores.
Software prefetch, NUMA placement, L1/L2/L3 allocation, and DRAM migration are
not represented by the current IR.

## 6. Migration Conditions

If a Machine Dialect is introduced later, the claims that the current standard
dialect path is the only custom layer must first be revised through a separate
design review. The standard Vector/LLVM MVP path must remain available, and a
Machine lowering failure must not break scalar fallback.
