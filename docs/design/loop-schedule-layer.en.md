# Loop and Schedule Layer Design

[中文](loop-schedule-layer.md)

## 1. Responsibility

This layer converts the bufferized Linalg Conv2D into explicit SCF loops and
applies C-reduction vectorization to SIMD variants. The schedule is a
controlled C++ lowering stage; it does not introduce a second Schedule IR.
The current MVP does not perform outer tiling or explicit R/S unrolling.

The MVP first implements a single-threaded direct convolution. Outer-loop
parallelization and OpenMP remain deferred until synchronization, thread-pool,
and workspace designs are defined.

## 2. Loop Semantics

For `linalg.conv_2d_nhwc_fhwc`, the logical loops are:

```text
for n  in [0,N)       parallel
  for oh in [0,OH)    parallel
    for ow in [0,OW)  parallel
      for k in [0,K)  parallel
        acc = 0
        for r in [0,R) reduction
          for s in [0,S) reduction
            for c in [0,C) reduction
              acc += X[n,oh+r,ow+s,c] * W[k,r,s,c]
        Y[n,oh,ow,k] = acc
```

Iterations of N/OH/OW/K have no write conflicts. R/S/C contribute to the same
output-element reduction. Every reorder, tile, or vectorization must preserve
these iterator semantics.

## 3. Pass Order

The MVP schedule order is fixed:

```text
bufferized linalg.conv_2d_nhwc_fhwc
  -> deepforge-workspace-plan
  -> clone scalar / AVX2 / AVX-512 modules
  -> direct Conv lowering (consume the named Conv op)
       scalar: scalar SCF reduction
       AVX2:   vector<8xf32> + scalar C tail
       AVX512: vector<16xf32> + scalar C tail
  -> convert-linalg-to-loops (remaining fill/copy/pad ops)
  -> canonicalize
```

Direct Conv lowering constructs scalar or vector SCF loops from verified
indexing maps. Standard `convert-linalg-to-loops` handles only the remaining
fill, copy, and padding materialization. The implementation must not first
lower Conv through a generic conversion and then depend on fragile loop-pattern
recognition. It also does not treat the heuristic behavior of
`affine-super-vectorize` as a Conv2D semantic guarantee.

## 4. Correct C-Reduction Vectorization

C is contiguous in the cuDNN packed filter, so the MVP vectorizes along C.
Each SIMD computation still produces only one output K; vector lanes must not
be interpreted as K:

```text
for n, oh, ow, k:
  vacc = vector.splat(0.0)       // vector<VFxf32>
  tail_acc = 0.0                 // f32

  for r, s:
    for c_vec = 0 to C - VF step VF:
      x_vec = load X[n, oh+r, ow+s, c_vec : VF contiguous f32]
      w_vec = load W[k, r, s, c_vec : VF contiguous f32]
      vacc = vector.fma(x_vec, w_vec, vacc)

    for c = floor(C/VF)*VF to C:
      tail_acc += X[n, oh+r, ow+s, c] * W[k, r, s, c]

  acc = vector.reduction(add, vacc) + tail_acc
  store Y[n, oh, ow, k] = acc
```

The invariants are:

1. Lane `i` of `x_vec` and `w_vec` represents the same C index.
2. `vacc` remains live and accumulates across all R/S/C blocks.
3. The reduction result is one output scalar.
4. The C tail uses scalar cleanup, does not read out of bounds, and requires no
   masked load.
5. K need not be divisible by VF because K remains a scalar loop.

A future K-vectorized implementation must first pack the filter explicitly
from `[K,R,S,C]` to `[R,S,C,K]` and define ownership, caching, and execution
costs for that buffer in a new contract. Merely changing an indexing map would
produce incorrect results.

## 5. MLIR Form

Each real variant uses four outer `scf.for` loops, R/S `scf.for iter_args`, a
full C-block loop, a scalar C tail, `vector.fma`, and
`vector.reduction <add>`. `deepforge-compile --dump-ir=llvm:<path>` writes the
fully converted module, while compiler end-to-end tests inspect the loop form
before conversion. `vector.load` is not annotated with a 64-byte alignment that
the runtime cannot prove; the public X/W/Y requirement is only
`alignof(float)`.

## 6. Tiling and Unrolling (Optimize Phase)

The current MVP uses step-one N/OH/OW/K loops, has no active tile parameters,
and performs no explicit R/S unrolling. SIMD lowering generates the C tail.
N/OH/OW/K use exact static upper bounds, so the current implementation has no
tile tails.

Future tiling or unrolling candidates must pass the correctness matrix and
pinned-core benchmarks independently. A cost model may estimate working sets
from cache size, cache-line size, and vector-register pressure, but it must not
create L1/L2/L3 address spaces or claim to place data in a specific cache.
Software prefetch is currently disabled.

### 6.1 Cost Model Ownership and Status

The cost model belongs to this compile-time Loop/Schedule layer because its
output is a schedule choice: tile sizes, loop order, and optional unroll
factors. Target configuration may supply hardware facts such as cache capacity,
cache-line size, vector width, and register budget; benchmark data validates or
calibrates candidate scores.

There is **no active cost model in the MVP**. The current schedule is fixed and
untiled. CPUID/XGETBV runtime dispatch only filters and selects ISA variants
that are safe to execute; it is a capability check, not a performance cost
model. The Machine Dialect is also not the owner of this decision.

When implemented, the model must produce an explicit, inspectable schedule and
fall back to the current untiled schedule when no candidate is valid. It must
not alter graph semantics, public ABI, workspace ownership, or numeric rules.

## 7. Dependence and Legality

The current direct lowering constructs loops from verified named-Conv
semantics and does not cache DependenceAnalysis. Every future pass that changes
loop structure must query current dependence and alias information before the
transform and allow the PassManager to invalidate and recompute analyses.

Writes to `Y[n,oh,ow,k]` do not alias one another, so outer loops may be
parallelized later. Reduction accumulators may only be reordered within C/R/S.
Floating-point reassociation is limited by the error bounds in
`contracts.en.md`; more aggressive fast-math must not be enabled without an
explicit numeric option.

## 8. Parallelization (Future)

The MVP emits neither `affine.parallel` nor OpenMP runtime calls. Future
parallelization must define all of the following together:

- thread tiling and scheduling policy;
- per-thread workspace and temporary buffers;
- output writes and error propagation;
- CPU feature dispatch and thread-pool lifetime.

Changing `affine.for` to `affine.parallel` alone is not a complete design.

## 9. Exit Conditions

When this layer completes:

- Linalg has become SCF, MemRef, Arith, and optionally Vector;
- every vector reduction has an explicit reduction target;
- no temporary non-upstream operations such as `affine.vector_load` remain;
- C cleanup and exact N/OH/OW/K loop bounds are visible in the IR;
- every workspace temporary has a static offset;
- complete LLVM conversion leaves no SCF, MemRef, Arith, Vector, or other
  source dialects.
