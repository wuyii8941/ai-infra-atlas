## Pipeline Parallel拆的是层
把不同层放到不同stage，activation依次传递，backward传相反方向的梯度。若只有一个microbatch，前后stage会等待。把global batch切为多个microbatch，能让不同stage同时处理不同数据。

在理想等长stage、简单填充/排空模型中，气泡比例常用 $(p-1)/(m+p-1)$ 理解，p为stage数、m为microbatch数。真实1F1B、interleaving、stage不平衡和通信会改变时间，不能把该式当生产实测。

1F1B交替执行forward和backward，可减少等待反向期间要保留的activation。参数更新仍要保证一致，不能让同一个优化step的不同microbatch使用混乱版本的权重。

## Context Parallel拆的是长序列计算
每卡保存不同token片段，但attention query需要相关历史K/V。于是需要在rank间传递K/V或其他等价信息，再做局部计算和合并。它缓解长序列activation与attention资源问题，不是简单删除跨片段注意力。

CP与某些Sequence Parallel概念有交集，但在Megatron语境下SP常指TP周边特定activation的sequence切分，CP还涉及attention的上下文依赖。面试时先声明术语所处框架。

## MoE为什么引入All-to-All
普通FFN每token经过同一组dense权重。MoE使用router为token选择少数专家，每个expert通常是一个FFN。专家分散在不同GPU时，需要把token dispatch到其专家，再把结果combine回来。

All-to-All与All-Reduce不同：前者把不同数据发送给不同目的rank；后者做同位置数值规约。路由不均匀会使热点专家成为瓶颈，因此还需负载均衡、capacity与丢弃/不丢弃策略。

## 总参数不等于活跃参数
MoE可以有很多总权重，但每个token只激活一部分。计算量估算应关注活跃专家与共享组件，而显存和通信仍受总权重、路由和驻留策略影响。不能把dense的6P公式直接套在总MoE参数上。

DeepSeek-V3使用MLA与MoE等结构，Qwen也有dense和MoE不同型号；你的1.7B及所验证8B为dense Qwen架构。这些并行机制作为扩展知识，不是当前仓库已经实现的功能。
