## Attention为什么还需要位置
仅靠内容向量的点积，不能完整表达“同一个token出现在不同位置”。RoPE将位置m映射成旋转矩阵 $R_m$，再作用于query和key，使相对位置影响它们的点积。

对一个二维子空间：

$$R(\theta)=\begin{pmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{pmatrix}.$$

将位置m和n分别编码：

$$(R_mq)^T(R_nk)=q^TR_m^TR_nk=q^TR_{n-m}k.$$

这里利用旋转矩阵的可组合性。不同维度对使用不同频率 $\theta_{m,j}=m\omega_j$，构成多频率相对位置表示。不是给每个token简单加一个独立向量。

## Qwen为什么是split-half
设D=8，split-half把维度配为(0,4)、(1,5)、(2,6)、(3,7)。若 $x=[a,b]$，其中a/b各为D/2维，则

$$rotate\_half(x)=[-b,a],$$
$$y=x\odot cos+rotate\_half(x)\odot sin.$$

相邻配对interleaved方式则是(0,1)、(2,3)等。两种约定在适当重排权重时可建立关系，但不能在已训练checkpoint上不加处理直接互换。

项目索引 `p = where(d < D/2, d+D/2, d-D/2)` 正是寻找另一半搭档。每个program处理一个head，所以不需要跨head通信。

## 为什么作用在Q/K而不是V
相对位置应影响attention评分，即QK点积。原始RoPE形式对Q/K旋转，V作为被加权内容保留。模型变体可以不同，但本项目必须匹配Qwen真实实现。你直接写V cache的kernel只是复制V，不对V做Norm或RoPE。

## 融合的合法性与限制
Norm结果是RoPE输入，可以连续算，省中间写回；不代表可以随意交换两个操作。理想不带逐维scale的正交旋转保持范数，但RMSNorm带learned scale、epsilon和低精度舍入，通常不能用“旋转不改范数”推断任意重排逐位等价。

## 位置与cache
prefill的position是多个连续位置，decode的新token必须使用真实cache position，而不是每次重置为0。向cache写K时应保存与模型契约一致的旋转后K，不能在下次attention中无意二次旋转。

长上下文扩展如RoPE scaling/YaRN会改变频率或其他约定。项目8B报告保留了配置warning并只验证短上下文，不能据此宣称验证了131K语义。
