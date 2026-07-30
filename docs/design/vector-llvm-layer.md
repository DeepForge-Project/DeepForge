# Vector 与 LLVM 层设计

## 1. 职责

本层把标准 Vector/Arith/MemRef/Func/ControlFlow 等方言完整转换为 LLVM Dialect，
再翻译为 LLVM IR。MVP 不经过 DeepForge Machine Dialect，也不手写 AVX-512
intrinsic 名称。

Vector Dialect 描述向量语义；LLVM backend 根据 module triple 和 target features
选择实际指令。`vector.fma` 不保证一定打印成某一个 x86 intrinsic 名称，设计
契约是生成合法 LLVM vector IR 并在目标特性允许时得到 FMA 指令。

## 2. MVP Vector 形态

完整向量 block：

```mlir
%xv = vector.load %x[%n, %h, %w, %c]
    : memref<?x?x?x?xf32>, vector<16xf32>
%wv = vector.load %weight[%k, %r, %s, %c]
    : memref<?x?x?x?xf32>, vector<16xf32>
%acc = vector.fma %xv, %wv, %acc0 : vector<16xf32>
%sum = vector.reduction <add>, %acc : vector<16xf32> into f32
```

真实 MVP shape 是静态的；上面的 `?` 只表示文档中省略具体尺寸。C tail 是
标量循环，不使用越界 vector load，因此不需要 mask 作为正确性前提。

向量内存访问在 C 维连续。除非 runtime contract 证明更强的对齐，否则 load/store
的 alignment 使用 4 字节或不指定更强 alignment。64-byte workspace 对齐不等于
输入/权重/输出指针对齐 64。

## 3. 完整 conversion 顺序

从 loop/vector IR 到 LLVM Dialect 的 MVP 顺序：

```text
canonicalize
  -> convert-vector-to-llvm
  -> lower-affine                 # if affine remains
  -> convert-scf-to-cf             # if scf remains
  -> expand-strided-metadata       # when dynamic memref metadata remains
  -> convert-index-to-llvm
  -> convert-arith-to-llvm
  -> finalize-memref-to-llvm
  -> convert-func-to-llvm
  -> convert-cf-to-llvm
  -> reconcile-unrealized-casts
  -> final module/function legality check
```

实际实现以 MLIR 22.1.8 的 pass registry 为准；pipeline builder 不允许只调用
`vector-to-llvm` 后宣称完成。每个 conversion pass 都声明 target/illegal
dialects，若任何 `affine`、`scf`、`memref`、`arith`、`index`、`func`、`cf` 或
DeepForge 临时 op 残留，pipeline 失败并打印 IR。

`mlir-translate --mlir-to-llvmir` 只负责 LLVM Dialect 到 LLVM IR 的翻译，不会
替代上述 dialect conversion。

## 4. LLVM IR 目标代码

每个 target-specific LLVM module 包含一个同布局、同数值契约的隐藏函数；打包
产物最终包含三个变体：

```text
deepforge_conv2d_scalar(...)
deepforge_conv2d_avx2(...)
deepforge_conv2d_avx512(...)
```

每个函数的参数只由内部 adapter 传入；函数不作为公共 ABI 导出。三个 module
分别设置 target features：

```text
scalar:  x86-64 baseline
avx2:    +avx2,+fma
avx512:  +avx512f,+fma
```

LLVM backend 可以将 `<8 x float>`/`<16 x float>` 和 FMA 选择为对应机器指令；
测试应检查汇编/目标属性，而不是依赖某个当前 LLVM intrinsic 的打印名称。
runtime 选择 AVX 变体时还要确认操作系统保存了相应的 SIMD 状态；只检查 CPUID
而不检查 `OSXSAVE`/`XGETBV` 可能在合法硬件上触发非法指令。

LLVM object 中的原始函数和 MLIR 生成的 `_mlir_ciface_<symbol>` wrapper 均标记为
ELF `GLOBAL HIDDEN`。wrapper 只供 `.dfo` loader/ExecutionEngine 在内部解析，不是
共享库公开符号；ISA inspection 测试同时检查可见性和实际寄存器类别。

## 5. 数值和 fast-math

允许的优化范围：

- f32 FMA；
- reduction lane 重排；
- R/S/C 循环展开。

不启用 `nnan`、`ninf`、`nsz`、`afn` 或近似数学函数。若后续增加一个
`--allow-reassociation` 选项，它只能影响 Conv2D reduction，并必须在 executable
metadata 中记录，测试使用 contracts.md 的容差校验。

## 6. 内存属性

在 runtime 检查指针元数据、可计算的区间重叠、alignment 和 non-overlap 后，内部
adapter 可以向 kernel 传递 readonly/noalias 语义。编译器不能从 serialized stride 自动推断
外部地址的 64-byte alignment，也不能把 workspace 的对齐假设传播给 X/W/Y。

所有 memref descriptor 的创建、stride materialization 和 workspace view 都在
内部 adapter/WorkspacePlan 中完成；对外仍然只有被忽略的 opaque handle、UID map
和 workspace pointer。

## 7. 标量路径

标量路径不是“向量化失败后的未定义状态”，而是语义等价的专用标量 lowering：

```text
same direct-conv loop
  -> scalar multiply/add
  -> standard LLVM conversion
```

它用于无 AVX CPU、debug、数值对照和所有 tail cleanup。这样不需要把 vector
operations 临时转换成 SCF 再猜测语义。

## 8. AMX 处理

AMX 不在本层 MVP 路径。后续若引入 bf16/AMX：

1. 先定义 filter packing、tile config 生命周期、OS XSTATE 权限和线程状态；
2. 评估 LLVM/MLIR 22.1.8 上游 X86/AMX conversion；
3. 为 AMX kernel 设计独立 workspace 与数值模式；
4. 通过 feature dispatch 在不支持 AMX 的 CPU 上安全回退。

不得继续使用文档中未定义的 `machine.amx_*` 或手写
`llvm.x86.tileloadd64.internal` 作为现行实现方案。

## 9. 退出条件

翻译前 IR 必须满足：

- 仅包含 LLVM Dialect、LLVM-compatible intrinsic dialect 和必要的 module attrs；
- 无 Tensor/Linalg/Affine/SCF/MemRef/Arith/Index/Func/CF 未转换 op；
- 所有 `unrealized_conversion_cast` 已 reconcile；
- LLVM IR 可被 `llc` 在对应 target features 下编译；
- scalar/AVX2/AVX-512 三个 module 的 ABI metadata 一致。
