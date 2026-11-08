## 线性层从点积组成矩阵乘法
设 $X\in\mathbb R^{M\times K}$、$W\in\mathbb R^{K\times N}$，输出 $Y=XW$。M 是输入行数；LLM 中常把 batch 和 token 两维展开为 M。K 是输入特征数，N 是输出特征数。数学右乘记法下 W 为 K×N；`nn.Linear` 存储的是 N×K。

一个输出元素为

$$Y_{ij}=\sum_{k=1}^{K}X_{ik}W_{kj}.$$

它包含 K 次乘法、K−1 次加法，精确的标量工作计数为 $MN(2K-1)$。工程中常将 multiply-accumulate 计为两个 FLOPs，于是使用

$$F\approx2MNK.$$

FMA 可以是一条硬件指令，但仍按一次乘法加一次加法记为 2 FLOPs。**指令数不等于 FLOPs**。计数惯例不同必须注明。

## 从小矩阵到模型投影
2×3 乘 3×2 产生 4 个输出，各用 3 乘 2 加，精确为20次；近似式给24。大型矩阵中少掉的 MN 次加法相对于 2MNK 很小。

Qwen 的示例 projection 若输入为 `[1,128,2048]`，展开后 M=128。若输出宽度为2048，则

$$F\approx2\cdot128\cdot2048^2=1{,}073{,}741{,}824.$$

这约为1.074 GFLOPs，不能写成1.074 GFLOP/s。前者是工作量，后者需要再除以秒数。

## 用全微分推导两次反向矩阵乘法
令上游梯度 $G=\partial L/\partial Y$，形状为 $M\times N$。损失是标量，因此把每个输出的微小变化按对应梯度加权：

$$dL=\sum_{i,j}G_{ij}\,dY_{ij}
    =\sum_{i,j,k}G_{ij}\big(W_{kj}\,dX_{ik}+X_{ik}\,dW_{kj}\big).$$

固定 $(i,k)$，收集 $dX_{ik}$ 的系数，得到 $\sum_j G_{ij}W_{kj}$。固定 $(k,j)$，收集 $dW_{kj}$ 的系数，得到 $\sum_i X_{ik}G_{ij}$。把这两类系数重新排列成矩阵：

$$\frac{\partial L}{\partial X}=GW^T,\qquad \frac{\partial L}{\partial W}=X^TG.$$

计算 $GW^T$ 时，$(M\times N)(N\times K)$ 得到 $M\times K$；计算 $X^TG$ 时，$(K\times M)(M\times N)$ 得到 $K\times N$。两次乘法各约 $2MNK$ FLOPs，再加上前向的 $2MNK$，总计约 $6MNK$。冻结权重、跳过不需要的梯度或添加重算都会改变成本。

## 2P 与 6P 的适用条件
对主要由 dense Linear 构成的模型，若一个 token 经过大部分矩阵权重，每个参数约参与一次乘加，则 forward/token 粗估为2P，训练约6P。这里不是从参数总数自动推导的定律：embedding lookup 不执行完整矩阵乘法，长序列 attention 的 score/PV 计算也不能忽略；MoE 应区分总参数和活跃参数；activation checkpointing 还增加重算。

## 把计算量与项目开销对齐
QK Fusion 主要减少 launch 和中间张量，并不显著降低 QKV Linear 的2MNK。FSDP 主要降低持久状态复制，也不会把同一训练样本所需的数学运算自动除以卡数。看见 latency 下降，应先问哪个成本变化，而不是直接说 FLOPs 降低。

使用计算实验室改变 M，观察工作量与理想数据量；这是理论估算，不是GPU实测。
