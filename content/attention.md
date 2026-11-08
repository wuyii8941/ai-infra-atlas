## 从输入投影到位置间的信息混合
输入矩阵的每一行代表一个位置。Self-Attention 让一行输出依赖多个输入位置。先用 Query 和 Key 确定各位置的权重，再沿序列维度对 Value 加权求和。

设 $X:[B,S,H]$。线性投影产生Q/K/V，reshape成heads后，单个head可以记为 $Q:[S_q,D]$、$K:[S_k,D]$、$V:[S_k,D_v]$。于是

$$A=QK^T:[S_q,S_k],\qquad O=PV:[S_q,D_v].$$

Q/K负责生成权重，V负责被加权的内容。它们是学习到的投影，不是固定的“问题、答案、数据库”结构。

## 按点积方差确定缩放因子
以教学假设推导：q和k各维独立，均值0、方差1，且q与k独立。每项 $q_i k_i$ 方差为1，D项相加的点积方差约D，标准差约 $\sqrt D$。除以 $\sqrt D$ 后，score尺度保持在数量级1。

若不缩放，维度增大时softmax容易极端集中，使某些梯度很小。注意这是初始化尺度的解释，不代表真实训练中的Q/K永远独立同分布。QK-Norm改变尺度分布，也不取消必须遵守模型定义的缩放规则。

## Softmax 将打分变为归一化权重
对某个query的一行score $s_j$：

$$p_j=\frac{e^{s_j}}{\sum_k e^{s_k}},\quad \sum_jp_j=1,\quad O_i=\sum_jp_jV_j.$$

这是一个依赖输入的加权组合。为避免overflow，用 $m=\max_j s_j$，同时从分子分母指数中减去m，结果不变。整行都被mask时分母可能为0，这需要额外约定，不能凭公式自动得到安全实现。

例如某个 Query 在掩码和缩放后的两个有效分数是 $(0,\ln 3)$，对应的标量 Value 是 $(2,10)$。指数是 $(1,3)$，分母为 $4$，概率是 $(1/4,3/4)$，输出为 $(1/4)\cdot2+(3/4)\cdot10=8$。实际的 Value 是向量，同一组概率用于其所有坐标。

## 因果掩码限定每个 Query 的可见位置
自回归训练中，位置i只能利用不晚于自己的输入。将未来位置的score设为负无穷，softmax后对应概率为0。decode只有一个新query，但它的位置是缓存末尾，不能机械套用“左上角1×S causal三角mask”；应使用正确的query/key位置语义。

```python
# 教学reference：MHA，Q/K/V均为[B,H,S,D]，非生产实现
scores = q @ k.transpose(-1, -2) / (q.shape[-1] ** 0.5)
scores = scores.masked_fill(future_mask, float('-inf'))
probs = scores.float().softmax(dim=-1).to(v.dtype)
out = probs @ v
```

## 投影之外的注意力计算量
QKᵀ约 $2BH_qS_qS_kD$ FLOPs，PV同量级，合计约 $4BH_qS_qS_kD$。训练prefill有 $S_q=S_k=S$，dense计算随S²增长。因果实现可跳过部分三角区域，但依然是二次量级。decode的query长度1，score/PV随已有cache长度线性增长。

FlashAttention接下来优化的是这些计算的IO组织，不是将所有dense attention FLOPs变成线性。
