## 由 Program ID 计算每组元素的索引
PyTorch代码描述高层Tensor操作，通常由框架分发kernel。Triton让你直接描述每个program处理哪些逻辑元素、如何加载/计算/写回，由编译器再生成设备代码。它介于高层算子图和手写线程级CUDA之间。

下面是教学向量加法。完整可运行版本位于 `labs/06_triton_add.py`，需要CUDA GPU、兼容PyTorch与Triton，不在浏览器执行。

```python
@triton.jit
def add_kernel(A, B, C, N: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = idx < N
    a = tl.load(A + idx, mask=valid, other=0.)
    b = tl.load(B + idx, mask=valid, other=0.)
    tl.store(C + idx, a + b, mask=valid)
```

`program_id`选择逻辑分块，`arange`生成分块内索引，mask处理最后一块不足BLOCK的边界。Host wrapper负责检查dtype/device/shape、分配输出与launch grid。没有这些前置检查，kernel很容易在假设被破坏时错误寻址。

## 编译期常量与specialization
`tl.constexpr`表示编译时可用参数。D、BLOCK等确定后，编译器能展开某些索引或分支，但参数取值变化也可能产生新编译变体。更多specialization不一定更好，可能增加编译延迟和cache数量。

`num_warps`影响程序映射资源；`num_stages`常用于软件流水线配置。它们不是“越大越快”的通用旋钮。真正的最佳选择与shape、寄存器、shared memory和硬件相关。

## reduction与数学语义
`tl.sum(x, axis=0)`是程序内逻辑张量的归约，编译器选择具体线程/warp通信方式。输入低精度提升到FP32，仍需考虑最后的cast、FMA融合和近似数学指令。不能因为代码短就忽略执行语义。

## 为自定义 Kernel 声明求导与编译契约
一个手写forward kernel不会自动获得正确backward。需要使用合适的autograd包装或注册。若要接入编译图，还要说明输出shape/stride、输入是否被修改、alias与fake/meta行为。静态KV cache写入是mutation，错误声明为纯函数可能被编译器误优化。

## 先验证语义再调优
先测试小shape、边界shape、非连续入口契约、不同dtype和空输入政策，再比较reference。随后测kernel计时，最后测真实模型。遇到超出shared-memory资源或错误mask的候选，应保留失败记录，不能只挑跑得最快的一行。
