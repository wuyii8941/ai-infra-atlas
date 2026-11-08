## 将文本编码成 Token 并查找向量
Tokenizer将文本映射为离散token ID序列。Token不一定是一个汉字或单词，可能是子词或字节片段。设词表大小V，Embedding矩阵为 $E\in\mathbb R^{V\times H}$。token ID=t的输入向量为E的第t行。

B个样本、每个S个token，经lookup后得到 `[B,S,H]`。lookup只是按索引取行，并不是用全部V×H矩阵做dense GEMM。词表实际token数量和为矩阵对齐而设置的embedding行数也可能不同，计算显存时应检查模型config与权重shape。

## LM Head 将隐藏状态投影为词表分数
经过Transformer层后，模型拥有每个位置的hidden states，但需要预测下一个token属于词表中的哪个项。LM Head做线性投影：

$$Z=HW_{out},\quad Z\in\mathbb R^{B\times S\times V}.$$

Z称为logits，尚未归一化。softmax给出概率 $p_j=e^{z_j}/\sum_i e^{z_i}$。正确标签为y时，cross entropy为

$$L=-\log p_y=\log\sum_i e^{z_i}-z_y.$$

推导导数可得 $\partial L/\partial z_j=p_j-\mathbf1[j=y]$。这说明loss backward不只是输出一个标量，它必须向所有logit分量传递梯度。

## 用下一位置的 Token 构造训练标签
输入 `t0,t1,t2` 时，位置0预测t1，位置1预测t2。训练使用真实前缀，这是teacher forcing；生成时后续输入来自前一步选出的token。一次生成分歧会改变未来输入，不能用单步小误差保证整段文本相同。

Padding或不参与训练的位置通常由ignore index排除。若把不同长度序列的平均loss直接再平均，会改变token权重。跨microbatch和跨rank都需要保留正确的有效token计数。

## Weight tying 与巨大输出矩阵
如果输入Embedding与输出Head共享E，则 $Z=HE^T$。它减少独立参数数量，但两条路径的梯度需要相加。Qwen3-1.7B的项目采用共享权重；8B结构不能默认同样tied。

V约15万、H=2048时，一个BF16 V×H矩阵已经约594MiB。完整 `[B,S,V]` logits也会很大。Fused Linear Cross Entropy尝试避免完整物化logits，但它是独立的kernel与数值问题，不能仅凭公式相同默认适配无风险。

## 练习
给定B=2、S=128、H=2048、V=151936，分别计算hidden states与BF16 logits的元素数和字节数。然后解释为什么推理可只取last-token logits，而训练通常需要所有有效预测位置。
