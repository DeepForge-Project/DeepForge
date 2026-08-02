# Loop and Schedule Layer Design

[中文](loop-schedule-layer.md)

## 1. Responsibility

This layer converts the bufferized Linalg Conv2D into explicit SCF loops,
applies C-reduction vectorization to SIMD variants, and may unroll independent
K outputs so they reuse each X load. The schedule is a controlled C++ lowering
stage; it does not introduce a second Schedule IR. The current implementation
does not perform outer tiling or explicit R/S unrolling.

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
  -> select one direct-Conv schedule per target variant
  -> direct Conv lowering (consume the named Conv op)
       scalar: scalar SCF reduction
       AVX2:   vector<8xf32> + scalar C tail + optional K-output unroll
       AVX512: vector<16xf32> + scalar C tail + optional K-output unroll
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

C is contiguous in the cuDNN packed filter, so SIMD lanes represent C. A
schedule may maintain `KU` independent accumulators for adjacent K outputs and
reuse one X load across them; this is output-loop unrolling, not
K-vectorization:

```text
for n, oh, ow:
  for k_base = 0 to floor(K/KU)*KU step KU:
    vacc[0:KU] = vector.splat(0.0)  // KU vectors of vector<VFxf32>
    tail_acc[0:KU] = 0.0            // KU independent f32 values

    for r, s:
      for c_vec = 0 to C - VF step VF:
        x_vec = load X[n, oh+r, ow+s, c_vec : VF contiguous f32]
        for u = 0 to KU:
          w_vec = load W[k_base+u, r, s, c_vec : VF contiguous f32]
          vacc[u] = vector.fma(x_vec, w_vec, vacc[u])

      for c = floor(C/VF)*VF to C:
        x = X[n, oh+r, ow+s, c]
        for u = 0 to KU:
          tail_acc[u] += x * W[k_base+u, r, s, c]

    for u = 0 to KU:
      Y[n, oh, ow, k_base+u] = vector.reduce_add(vacc[u]) + tail_acc[u]

  process K mod KU outputs with the same body and KU=1
```

The invariants are:

1. Lane `i` of `x_vec` and `w_vec` represents the same C index.
2. Each `vacc[u]` remains independent and accumulates across all R/S/C blocks.
3. Every reduction result is one output scalar; SIMD lanes never represent K.
4. The C tail uses scalar cleanup, does not read out of bounds, and requires no
   masked load.
5. K need not be divisible by `KU`; a step-one cleanup loop handles its tail.

An actual K-lane-vectorized implementation must first pack the filter explicitly
from `[K,R,S,C]` to `[R,S,C,K]` and define ownership, caching, and execution
costs for that buffer in a new contract. Merely changing an indexing map would
produce incorrect results.

## 5. MLIR Form

The scalar variant uses four step-one outer `scf.for` loops. SIMD variants use
three N/OH/OW loops, a K main loop at step `KU`, and a step-one K cleanup loop.
R/S carry `KU` vector and scalar `iter_args`; the C loops contain
`vector.fma`, scalar cleanup, and one `vector.reduction <add>` per output.
`deepforge-compile --dump-ir=llvm:<path>` writes the fully converted module.
`vector.load` is not annotated with a 64-byte alignment that the runtime cannot
prove; the public X/W/Y requirement is only `alignof(float)`.

## 6. Tiling and Unrolling (Optimize Phase)

N/OH/OW remain step-one loops, and there are no active tile parameters or
explicit R/S unrolling. SIMD lowering generates exact C and K cleanup loops.
All upper bounds are static and no masked or out-of-range access is required.

Future tiling or unrolling candidates must pass the correctness matrix and
pinned-core benchmarks independently. A cost model may estimate working sets
from cache size, cache-line size, and vector-register pressure, but it must not
create L1/L2/L3 address spaces or claim to place data in a specific cache.
Software prefetch is currently disabled.

### 6.1 Active Cost Model

The first active cost model belongs to this compile-time Loop/Schedule layer.
It is intentionally limited to the original optimized static, contiguous f32
single-`CONV_FPROP` path. Generic C2-C6 graphs retain `generic-reference`; the
importer, runtime, and Machine Dialect do not select performance schedules.

The model fixes `VF` from the target (`1`, `8`, or `16`) and considers
`KU in {1,2,4,8}`. Scalar and `baseline` policy permit only `KU=1`. For SIMD
auto policy, a candidate is legal when `KU <= K` and `2*KU+4` does not exceed
the target vector-register budget (`16` for AVX2 and `32` for AVX-512). The
deterministic estimate is:

```text
channel_steps = R*S*(floor(C/VF) + C mod VF)
input_loads   = (floor(K/KU) + K mod KU) * channel_steps
weight_loads  = K * channel_steps
score         = 2*input_loads + weight_loads + 16*KU
```

The common N/OH/OW factor is omitted because it does not change the candidate
ordering. The score favors X-load reuse while charging register/IR pressure.
The baseline is always a candidate, ties retain the smaller schedule, and an
empty legal set therefore cannot prevent compilation. Selected schedules are
exposed as `direct-c-vf<VF>-ku<KU>` in `CompilationResult::variants` and in the
benchmark CSV. `CompileOptions::schedule_policy = kBaseline` provides a stable
A/B and diagnostic fallback.

CPUID/XGETBV runtime dispatch remains a capability check, not a performance
model. The model changes neither graph semantics, public ABI, workspace
ownership, artifact format, nor numeric tolerances. Cache tiling, padding
fusion, and threading remain future benchmark-driven decisions.

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
