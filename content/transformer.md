## 从输入到两次残差更新
token通过Embedding变成 `[B,S,H]` 后，要经过多层Decoder Block。每个block包含跨token的信息混合Attention，以及逐token的特征变换FFN。残差连接把输入加回，使每层可以在已有表示上学习增量。

以简化Pre-Norm block表示：

$$X_1=X+Attention(Norm_1(X)),$$
$$X_2=X_1+MLP(Norm_2(X_1)).$$

这只是结构关系，attention内部仍有QKV投影、QK-Norm、RoPE、mask和输出投影。

<div class="flow"><span>输入 X</span><span>RMSNorm</span><span>QKV → QK-Norm → RoPE → Attention → O</span><span>Residual Add</span><span>RMSNorm → SwiGLU MLP</span><span>Residual Add</span></div>

## 用真实形状跟踪一次
Qwen3-1.7B项目配置：28层、H=2048、Q heads=16、KV heads=8、D=128、FFN中间宽度6144。B=1、S=128时，输入为 `[1,128,2048]`。Q投影输出2048维，K/V各输出1024维，再view为 `[1,128,16,128]` 和 `[1,128,8,128]`。

Attention计算后Q head的输出拼回2048维，经O projection后与输入同shape，才能逐元素残差相加。MLP则把2048维映射至更宽的6144维，经过门控后再投影回2048。

## 区分 Hidden 归一化与 Head 归一化
Block输入或post-attention norm沿hidden dimension H计算。Attention内部Q/K norm分别沿每个head的D计算。这两种归约域不同，权重shape也不同。项目的QK-Norm+RoPE融合针对后者，不是把全层所有norm并成一项。

## 残差提供恒等梯度路径，归一化控制尺度
残差提供不必经过整个非线性变换的直接路径。若 $y=x+f(x)$，反向包含identity贡献 $I+J_f$；这有助于理解深层网络为何使用该结构，但不意味着梯度永远不会爆炸或消失。Norm控制表示的尺度，其可学习scale允许模型调整各维贡献。

## 计算账本
每token的Q/K/V/O线性计算约25.2MFLOPs；三矩阵SwiGLU MLP约75.5MFLOPs。此处只计主要投影，不包含score/PV、norm和LM Head。数值由shape推导，不是profiling读数。

RMSNorm与RoPE并非FLOPs大头，但反复出现28次，小kernel和中间张量仍可能影响latency。这就是后续从架构走向fusion的合理切入口，不证明它是所有workload中的首要瓶颈。
