## 两种Megatron关联必须区分
你的DFuzz确实测试Megatron训练算子优化；但这不代表实现过Megatron整套TP调度。本章是分布式背景推导，DFuzz具体贡献在可靠性章节。

## Column Parallel为什么暂时不用通信
对MLP $Y=\phi(XW_1)W_2$，沿W1输出特征切列：$W_1=[W_{1,0},\ldots,W_{1,p-1}]$。各卡有相同X，算 $H_r=\phi(XW_{1,r})$，H_r是intermediate shard。逐元素activation在本地即可完成，无需先拼成完整H。

随后W2沿对应输入维切行：

$$Z_r=H_rW_{2,r},\qquad Y=\sum_rZ_r.$$

每张卡的Z_r具有完整输出shape，但只是partial sum，不是输出feature shard。因此需要sum collective。Qwen式SwiGLU要将gate/up的对应通道对齐切分，再消费本地门控结果。

## backward通信为什么在另一头
Column-parallel中，各卡计算 $dX_r=dH_rW_{1,r}^T$，输入X被所有分片共同使用，因此完整dX需要求和。局部dW只负责自己的权重分片。

Row-parallel的forward把各partial output相加；backward若上游dY已在各卡正确表示，每卡可算本地dH_r。经典设计把若干collective成对安排，使forward和backward分别承担必要通信；不同SP/overlap实现会改变具体collective形式。

## Attention按head切分
Q/K/V投影列切，局部heads完成attention，O projection行切后求和。GQA的KV heads更少，当TP degree超过KV heads或不能整除时需要额外分组/复制策略，不能盲目宣称每卡总能独立分到整数个KV heads。

## Sequence Parallel为何出现
TP之外的norm、dropout、residual activation若每卡完整保存，会产生重复。SP将某些activation沿sequence维切开，借助all-gather/reduce-scatter与TP GEMM衔接，降低这些activation的冗余。

它不是把所有attention token互相依赖取消；也不应和Context Parallel的长序列attention切分混淆。经典TP all-reduce可分解为reduce-scatter+all-gather，在合适位置延后聚合能保留分片activation。

## 性能取舍
TP每层都有通信，故拓扑与延迟敏感。增加TP可能让单卡GEMM变小并增大通信占比。跨机TP并非禁止，只是应在带宽、模型宽度、batch与其他并行策略间权衡。
