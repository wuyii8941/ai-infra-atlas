## 将 FX 运算转为可调度的计算
高层图的 `relu` 或 `sum` 描述数学操作，GPU代码还需要每个输出对应什么地址、在哪个维度归约、哪些值存为buffer，以及操作间依赖。Inductor lowering把ATen计算转换为便于后端处理的内部表示，包括pointwise/reduction计算、views/layout与extern kernel调用等。

可以把ReLU理解为 `out[i]=max(load(x,index(i)),0)`；sum还要给出输出域与归约域。真实IR不仅是打印出来的for循环，也有延迟计算和buffer等组织。

## 外部 Kernel 是一种后端选择
大GEMM或特定操作可能调用成熟库或模板实现。一个捕获图可以包含自生成Triton和extern调用。走extern是合法后端决策，不自动意味着Dynamo graph break，也不等于eager fallback。

## 融合的合法性与成本
Producer-consumer fusion可将生产计算内联到消费，减少中间存储；horizontal fusion尝试合并兼容的独立计算。需要检查依赖、mutation/alias、设备、索引/迭代域、reduction兼容性、模板接口和资源约束。

“单一消费者”是常见有利条件，不是所有融合必须满足的绝对规则；多消费者可能选择物化、重算或其他组合。transpose view也不是必然的fusion边界，关键是地址与计算映射能否兼容。

## 调度的不同层次
Inductor scheduler管理依赖、节点分组、顺序与融合等；后续codegen和Triton后端还会决定具体tile、warp布局及资源使用。把所有GPU映射细节都归给同一个scheduler函数是不准确的。

同一个reduce在不同尺寸下可能选择不同kernel family。persistent reduction、分块/分段归约具有不同资源权衡。阈值受版本和硬件影响，不应背某个固定数当作全部GPU通用规则。

## 缓冲区生命周期约束内存复用
两个buffer如果生命周期不重叠，可能复用storage；但若alias或mutation依赖判断错，就可能提前覆盖仍需要的数据。因此compiler correctness不仅是计算公式，还包含布局和生命周期语义。

## 与DeepOpFuzz的连接
operation target可以是某种lowering、fusion、schedule或extern选择。真正命中应通过artifact与trace确认，而非只检查输入图含某个operator。这使shape/stride/context成为编译决策的输入，而不仅是执行时的数据。
