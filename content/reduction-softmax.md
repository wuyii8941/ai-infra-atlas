## reduction为什么不是普通逐元素计算
逐元素add的每个输出只依赖对应输入；sum/max需要把多个元素合并。串行求和的依赖深度为N；树形归约把深度降为log₂N，但总加法仍约N−1。改变归约树可能改变浮点舍入顺序。

若一行长度不是block大小的整数倍，需要mask。sum可用0作为无效位置的幺元，max应考虑负无穷。softmax中只给max正确幺元仍不够：无效位置的指数贡献也必须为0，全mask行需要定义。

## stable softmax从两次统计开始
对分数序列s，令 $m=\max s_j$，$l=\sum_j e^{s_j-m}$，则 $p_j=e^{s_j-m}/l$。直接实现通常先求m，再求l，最后归一化。

长序列不能总把一整行放片上，能否逐块合并？设块A的统计为(mA,lA)，块B为(mB,lB)，统一最大值 $m=\max(m_A,m_B)$ 后：

$$l=e^{m_A-m}l_A+e^{m_B-m}l_B.$$

这是因为两块的指数原点不同，必须重新缩放到同一个m。不是把两个分母直接相加。

## Attention输出也能同时累加
令块的未归一化输出 $u_A=\sum_{j\in A}e^{s_j-m_A}V_j$，B同理，则

$$u=e^{m_A-m}u_A+e^{m_B-m}u_B,\qquad O=u/l.$$

这样只需保留m、l和一个输出向量，不需要保存整行probabilities。它是FlashAttention及split-KV合并的关键数学基础。

如果各块保存的是已经归一化的输出 $O_j$ 和 $LSE_j=\log\sum e^{s}$，则总输出必须按 $e^{LSE_j-\max LSE}$ 加权后再除以权重和，不能简单平均各块结果。

## 边界与数值
公式在实数上等价，执行顺序不同可产生舍入差异。最大值都是负无穷时，`-inf - -inf`会产生NaN，因此必须给全mask行明确处理。Softmax整体平移不变是一条有用的metamorphic relation，但浮点溢出、mask和特定contract仍需考虑。

## 实践
`labs/02_online_softmax.py` 将随机score/value切块，逐块更新m/l/u，再与完整softmax输出对比。它在CPU float64上验证代数关系，不代表已经实现高性能GPU attention。
