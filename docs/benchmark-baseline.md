# CPU 性能基线

[English](benchmark-baseline.en.md)

## 1. 复测方法

`deepforge-benchmark` 生成三个满足 MVP 契约的 canonical Conv2D，编译 scalar、AVX2
和 AVX-512 变体，先做 warmup 和逐元素数值检查，再报告平均执行时间。它是单线程
direct-convolution 基线，不是供应商库对比。

```bash
build/tools/deepforge-benchmark --profile=all --iterations=5
```

每行输出 profile、变体、shape、workspace、迭代数、编译时间、平均执行时间、
GFLOP/s、相对 scalar 的最大绝对/相对误差。CTest 只运行 small/1 iteration smoke；
性能数值不作为不稳定的 pass/fail 阈值。

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
