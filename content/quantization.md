## 从连续数映射到整数
常见affine量化为

$$q=clip(round(x/s)+z,q_{min},q_{max}),\quad \hat x=s(q-z).$$

s是scale，z是zero point。对称量化常取z=0。多个元素共享scale可减少元数据，但不同元素范围差异较大时误差也可能增大。per-channel/per-group增加局部适应性，同时增加scale存储与访存。

## W8A16 的权重存储与激活计算
权重以8位表示，activation仍为16位。执行时可能把权重tile反量化为计算dtype，再做矩阵运算。不是所有硬件都有“INT8权重×FP16激活”的同一种原生高速指令；实际kernel组织与小M权重带宽决定能否加速。

若每G个参数保存一个FP32 scale，理想权重字节约 $P(1+4/G)$，并不恰好P字节。还要考虑zero point、padding、workspace。保留原权重用于回退/对照时，总驻留显存可能反而增加。

## 权重读取节省与反量化开销
小batch decode经常反复读大权重，压缩权重可能降低数据量；但反量化、索引和重复读取scale有成本。大batch可能已经compute-bound，省权重带宽未必弥补转换成本。分组策略和跨batch重用需要独立消融。

## 用对照实验设定误差门槛
单层相对L2和最大误差是局部指标；整模型logits、梯度、validation loss和任务质量回答不同问题。生成中的一次argmax变化会影响未来输入。量化组合产生自然文本分歧时，不能写“无损”，也不能把未通过质量门槛的最快路径视为有效收益。

项目的MLP/LM Head W8A16保留了额外快照和生成分歧，简历主结果优先使用未量化路径。BF16 master训练与INT8权重量化也不是同一种混合精度概念。

## FP8、FP4与KV量化
FP8包含浮点指数与尾数，不等同整数INT8。硬件支持、scale格式和累加dtype会影响可用性。KV量化针对每请求cache，不是模型权重；其误差进入attention score/PV。更低比特既可能提高并发，也可能改变质量，需要相应backend和评测。

本站不把现代低比特硬件宣传峰值当作A6000实测，也不把推理量化自动推广到训练optimizer。
