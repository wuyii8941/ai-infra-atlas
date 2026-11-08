## 用均方根控制输入尺度
隐藏向量的数值尺度影响下游计算。LayerNorm减去均值再按标准差归一化；RMSNorm只用均方根尺度，不做中心化。对一行 $x\in\mathbb R^D$：

$$r=(\frac1D\sum_i x_i^2+\epsilon)^{-1/2},\qquad y_i=w_ix_ir.$$

w是可学习scale，epsilon避免零附近的分母问题。忽略epsilon且a>0时，RMSNorm(ax)与RMSNorm(x)相同；有epsilon时不是严格尺度不变，a<0还涉及符号。

## 数值小例子
令x=[3,4]，w=[1,1]，epsilon=0。均方为(9+16)/2=12.5，RMS≈3.5355，输出≈[0.8485,1.1314]。它没有把均值变成0，因此不能解释为“把每行平移到零中心”。

Qwen block级RMSNorm沿H维；QK-Norm沿每个head的D维。每行需要独立的r，所以一个program负责一行是自然起点，但并非所有shape上的最佳schedule。

## FP32 归约扩大中间量的表示范围
平方和可能跨越大量元素，低精度直接相加会增加误差。常见方式是低精度输入转FP32计算r，再按契约cast。项目HF兼容kernel保留 normalize→cast→scale→cast 的某些边界，不是“全程FP32最后一次舍入”。

## 由链式法则推导反向传播
令上游梯度为g，定义 $a_i=g_iw_i$。因

$$\frac{\partial r}{\partial x_j}=-\frac{x_j}{D}r^3,$$

有

$$\frac{\partial L}{\partial x_j}=r a_j-\frac{x_jr^3}{D}\sum_i a_ix_i.$$

第一项是直接缩放路径，第二项来自所有元素共同影响r。w的梯度是 $g_jx_jr$，在多行输入下还需沿行累加。只对x乘r而忽略r的导数会产生错误梯度。

这个公式按实数语义推导；真实低精度cast和自定义backward的约定需单独测试。有限差分也需要远离数值不稳定点并选择合适步长。

## GPU成本与项目连接
主要算术和数据量都为O(D)，不像GEMM那样存在大量跨输出复用。优化候选包括减少中间张量、适当的row/block映射与融合消费者。不能仅凭公式就宣称带宽已饱和。

你的QK融合把RMSNorm后紧接的RoPE放入同一program。DFuzz用Liger RMSNorm的训练轨迹说明单步微小差异可能随训练传播。这是性能与数值可靠性的同一个交汇点。
