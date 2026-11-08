## 为什么这份实现值得逐行读
它在真实8B级Qwen架构checkpoint上验证全参数短程训练，但采用同步collective、无通信重叠、确定性block。其价值是一个完整的shard生命周期，不是新的NCCL算法。

## 初始化：结构与存储分开
模型从CPU加载。每个unit记录参数名、shape、numel，把参数展平并补齐到world size整除。rank只将自己的区间迁移到GPU，保持FP32 master shard。模块模板转到meta，保留结构而不保留完整GPU参数storage。

shared字典按参数身份复用master shard，tied embedding/head可拥有同一份Parameter；它们分别计算时聚合权重，反向贡献汇总到同一shard，optimizer只更新一次。

## forward：临时提供完整参数
`gather(shard)`先用Triton本地转换为BF16，再调用NCCL `all_gather_into_tensor` 得到完整unit参数。`call`把flat参数按spec切成对应shape，再用 `torch.func.functional_call` 执行原模板。

自定义 `_Call.forward` 保存shard和输入边界供backward使用，不保留完整普通autograd中间图。输入边界仍占显存，不能说“所有activations都不保存”。

## backward：重算到分片梯度
在启用grad的上下文中重新gather，临时full参数设置requires_grad，输入detach后按需求启用梯度。再次执行block，用 `torch.autograd.grad(output,(full,*inputs),grad)` 得到参数和输入梯度。

随后把参数gradient转换为FP32并reduce-scatter SUM，Triton本地除world size。输入梯度保持本rank数据的语义，不像参数梯度那样跨rank平均。梯度若由BF16compute产生，后续cast不会恢复已丢失精度。

<div class="flow"><span>FP32 shard</span><span>BF16 all-gather</span><span>Forward</span><span>重聚合＋重算</span><span>梯度</span><span>FP32 reduce-scatter</span><span>本地AdamW</span></div>

## 显存数字怎样对齐
真实P=8,190,735,360。BF16全权重约15.26GiB；master+m+v约91.54GiB，4卡理想持久分摊约22.88GiB；加FP32 gradient shard约30.51GiB。还需完整当前unit、activation和workspace，因此35.7568GiB测量峰值不等于简单12P/4。

S512、每卡B1、4卡、真实WikiText、3次更新通过；初始及导出重载对应logits最大误差0，仅是有限测试观测。Q projection约18.63%的BF16元素发生变化，说明实际更新，不证明全部参数更新幅度相同。

## 工程限制
仅同有效token数与相同collective顺序；拒绝随机dropout重算、高阶梯度等未支持情况。无跨节点、弹性world size或自动overlap。完整8B optimizer checkpoint未因磁盘限制而验证，不能把HF BF16权重导出称为精确恢复。真实checkpoint是DeepSeek-R1-0528-Qwen3-8B，不是原版Qwen3-8B下载完成。

## 如何验证一个改动
先小模型tied/untied多rank对照所有grad与连续更新，再故意破坏参数/optimizer做恢复测试，最后做真实模型功能验证。性能对照需要剥离诊断成本，并与相同精度、重算、unit策略的官方baseline比较；当前项目没有这个性能胜出结论。
