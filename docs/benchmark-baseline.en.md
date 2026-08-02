# CPU Performance Baseline

[中文](benchmark-baseline.md)

## 1. Reproduction

`deepforge-benchmark` creates three canonical Conv2D graphs that satisfy the
MVP contract, compiles scalar, AVX2, and AVX-512 variants, performs warmup and
element-wise numeric validation, and reports average execution time. This is a
single-threaded direct-convolution baseline, not a comparison against a vendor
library.

```bash
build/tools/deepforge-benchmark \
  --profile=all --iterations=5 --schedule=both
```

`--schedule` accepts `auto` (default), `baseline`, or `both`. Each output row
contains profile, policy, variant, selected schedule, shape, workspace size,
iteration count, compile time, average execute time, GFLOP/s, and maximum
absolute and relative differences from scalar. CTest runs only a one-iteration
small smoke test. Performance numbers are not unstable pass/fail thresholds.

## 2. Profiles

| Profile | N/H/W/C/K/R/S | Padding | Workspace |
|---|---|---|---:|
| small | `1/8/8/8/8/3/3` | `1x1` | 3,200 bytes |
| medium | `1/32/32/32/32/3/3` | `1x1` | 147,968 bytes |
| large | `1/56/56/64/64/3/3` | `1x1` | 861,184 bytes |

## 3. Local Baseline, 2026-07-30

The host was an Intel Xeon 6986P-C with one socket, 120 cores, and 240 threads.
The benchmark was not pinned to a core, turbo was not disabled, and system load
was not isolated. DeepForge used `RelWithDebInfo`, LLVM/MLIR 22.1.8, GCC 13.4.0,
and five iterations.

| Profile | Variant | Execute ms | GFLOP/s | Max abs | Max rel |
|---|---|---:|---:|---:|---:|
| small | scalar | 0.034 | 2.150 | 0 | 0 |
| small | AVX2 | 0.011 | 6.698 | 5.722e-6 | 3.850e-5 |
| small | AVX-512 | 0.026 | 2.878 | 0 | 0 |
| medium | scalar | 4.710 | 4.007 | 0 | 0 |
| medium | AVX2 | 0.864 | 21.855 | 1.335e-5 | 2.198e-4 |
| medium | AVX-512 | 0.714 | 26.441 | 1.335e-5 | 1.882e-4 |
| large | scalar | 65.255 | 3.543 | 0 | 0 |
| large | AVX2 | 10.893 | 21.227 | 3.433e-5 | 6.454e-4 |
| large | AVX-512 | 7.729 | 29.914 | 3.433e-5 | 6.389e-4 |

SIMD lowering provides a real speedup, but AVX-512 startup and horizontal
reduction cost exceeds AVX2 on the small profile. Future tiling, padding
fusion, and parallelization decisions must use repeated pinned-core benchmark
changes rather than inference from ISA width alone.

## 4. C6.5 Schedule A/B, 2026-08-02

This run used a Release build on the same Intel Xeon 6986P-C host, pinned the
single-threaded process to logical CPU 0 with `taskset -c 0`, and used ten
iterations. `baseline` fixes `KU=1`; `auto` selected `KU=4` for AVX2 and `KU=8`
for AVX-512 on these profiles. Scalar remains `VF=1,KU=1` under both policies.

| Profile | Variant | Baseline schedule | Auto schedule | Baseline GFLOP/s | Auto GFLOP/s | Ratio |
|---|---|---|---|---:|---:|---:|
| small | AVX2 | `vf8-ku1` | `vf8-ku4` | 8.690 | 13.033 | 1.500x |
| small | AVX-512 | `vf16-ku1` | `vf16-ku8` | 3.226 | 5.882 | 1.824x |
| medium | AVX2 | `vf8-ku1` | `vf8-ku4` | 22.171 | 38.428 | 1.733x |
| medium | AVX-512 | `vf16-ku1` | `vf16-ku8` | 26.684 | 41.127 | 1.541x |
| large | AVX2 | `vf8-ku1` | `vf8-ku4` | 21.300 | 34.179 | 1.605x |
| large | AVX-512 | `vf16-ku1` | `vf16-ku8` | 30.009 | 33.789 | 1.126x |

Maximum absolute and relative differences were unchanged from the baseline
schedule. This is one local validation of the candidate model, not a portable
performance guarantee or a reason to select an ISA without the runtime feature
check.
