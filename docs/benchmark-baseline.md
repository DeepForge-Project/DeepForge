# CPU 性能基线

[English](benchmark-baseline.en.md)

## 1. 复测方法

`deepforge-benchmark` 生成三个满足 MVP 契约的 canonical Conv2D，编译 scalar、AVX2
和 AVX-512 变体，先做 warmup 和逐元素数值检查，再报告平均执行时间。它是单线程
direct-convolution 基线，不是供应商库对比。

```bash
build/tools/deepforge-benchmark \
  --profile=all --iterations=5 --schedule=both
```

`--schedule` 可选 `auto`（默认）、`baseline` 或 `both`。每行输出 profile、policy、
variant、实际 schedule、shape、workspace、迭代数、编译时间、平均执行时间、GFLOP/s
以及相对 scalar 的最大绝对/相对误差。CTest 只运行 small/1 iteration smoke；性能
数值不作为不稳定的 pass/fail 阈值。

## 2. Profile

| Profile | N/H/W/C/K/R/S | Padding | Workspace |
|---|---|---|---:|
| small | `1/8/8/8/8/3/3` | `1x1` | 3,200 bytes |
| medium | `1/32/32/32/32/3/3` | `1x1` | 147,968 bytes |
| large | `1/56/56/64/64/3/3` | `1x1` | 861,184 bytes |

## 3. 2026-07-30 本机基线

主机为 Intel Xeon 6986P-C，单 socket、120 cores/240 threads；测试未绑核、未关闭
turbo，也未隔离系统负载。DeepForge 为 `RelWithDebInfo`，工具链为 LLVM/MLIR
22.1.8、GCC 13.4.0，5 次迭代：

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

结果说明 SIMD lowering 已产生实际收益，但 small profile 的 AVX-512 启动和水平归约
成本高于 AVX2。后续 tiling、padding fusion 和并行化应以重复、绑核后的 benchmark
变化为依据，不应仅凭 ISA 宽度推断收益。

## 4. 2026-08-02 C6.5 Schedule A/B

本次使用同一 Intel Xeon 6986P-C 主机的 Release build，通过 `taskset -c 0` 将单线程
进程绑定到 logical CPU 0，每项执行 10 次。`baseline` 固定 `KU=1`；`auto` 在这些
profile 上为 AVX2 选择 `KU=4`、为 AVX-512 选择 `KU=8`。scalar 在两种 policy 下均为
`VF=1,KU=1`。

| Profile | Variant | Baseline schedule | Auto schedule | Baseline GFLOP/s | Auto GFLOP/s | 比值 |
|---|---|---|---|---:|---:|---:|
| small | AVX2 | `vf8-ku1` | `vf8-ku4` | 8.690 | 13.033 | 1.500x |
| small | AVX-512 | `vf16-ku1` | `vf16-ku8` | 3.226 | 5.882 | 1.824x |
| medium | AVX2 | `vf8-ku1` | `vf8-ku4` | 22.171 | 38.428 | 1.733x |
| medium | AVX-512 | `vf16-ku1` | `vf16-ku8` | 26.684 | 41.127 | 1.541x |
| large | AVX2 | `vf8-ku1` | `vf8-ku4` | 21.300 | 34.179 | 1.605x |
| large | AVX-512 | `vf16-ku1` | `vf16-ku8` | 30.009 | 33.789 | 1.126x |

最大绝对/相对误差与 baseline schedule 相同。这只是在本机对候选模型的一次验证，
不是跨主机性能保证，也不能替代 runtime ISA feature check。
