## 用逐 Token 的门控网络变换特征
Attention在token之间交换信息；FFN主要在每个token的特征维度上做非线性变换。普通两层MLP为 $Y=\phi(XW_1)W_2$。Qwen采用门控结构：

$$G=XW_g,\quad U=XW_u,$$
$$Z=SiLU(G)\odot U,\quad Y=ZW_d,$$
$$SiLU(x)=x\sigma(x),\quad\sigma(x)=\frac1{1+e^{-x}}.$$

gate和up不是同一结果的复制，而是两套学习到的投影。gate的非线性值调节up的每个特征，因此有gate/up/down三块矩阵。

## 从形状推导计算量
X为M×H，gate/up权重为H×I，down为I×H。三次GEMM合计约 $6MHI$ FLOPs。Qwen3-1.7B取H=2048、I=6144，单token约75.5MFLOPs，不包含activation和其他层。

SiLU和乘法都逐元素处理M×I数据。把它们融合，可以避免SiLU输出的物化；但并没有消除三次大矩阵乘法。把gate/up projection也合入另一个kernel，则是不同层次的GEMM/epilogue设计。

## 门控两条支路的反向传播
令上游对Z的梯度为 $g_Z$，则

$$g_U=g_Z\odot SiLU(G),$$
$$g_G=g_Z\odot U\odot[\sigma(G)+G\sigma(G)(1-\sigma(G))].$$

再通过矩阵反向公式求dX和dW。因为X同时参与gate与up，两个分支的dX需要相加，不能覆盖。训练融合还要解决保存或重算G/U的问题，推理kernel不自动拥有正确backward。

## 项目中的独立优化与组合优化
基础 `_activation` 负责SiLU与Mul融合。Decode Gate/Up GEMV针对小M进行特定访存组织。调优版prefill GEMM涉及BM/BN/BK、warp和stage配置。这些实验的比较范围不同，有些只计gate/up，不包含down projection，不可称为完整MLP速度。

项目保留了部分dtype/shape慢于cuBLAS或编译基线的负结果。一个BF16随机矩阵上更快，不足以证明FP16或完整Qwen也更快。

## 端到端融合的成本与条件
数学依赖存在不代表能高效放入一个CTA。跨tile重用、全局归约、寄存器容量和不同计算域会形成边界。合理方案常是只融合GEMM的适当prologue/epilogue，保留成熟库或模板的高吞吐核心。
