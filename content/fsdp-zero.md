## 8B权重为什么能放下，训练却放不下
P个参数的BF16权重约2P字节。训练可能还需要FP32 master、gradient与两个Adam moments，加上activation和workspace。以master+FP32 gradient+m+v计为16P字节，不能仅用2P估训练容量。不同precision policy下账本必须重算。

## DDP复制了什么
每卡一份模型与optimizer，各自处理不同数据，再同步梯度。这样提高数据并行吞吐，却不降低每卡完整参数与optimizer状态的基本存储。如果完整训练状态在单卡放不下，简单增加DDP rank并不解决。

## ZeRO的分阶段思想
ZeRO-1切optimizer state；ZeRO-2还切gradient；ZeRO-3进一步切parameter。FSDP full-sharding采用相近的全状态分片思路，但具体API、调度、flatten表示与实现不等于ZeRO库本身。

公式 $M\approx M_{state}/N$ 只能估理想持久分片部分。真实峰值还包括临时unshard、完整当前单元gradient、activation、prefetch、allocator与通信buffer。N倍显存总和不是一个无需调度的统一地址空间。

## 为什么按layer/block临时聚合
计算一个block需要它的权重。按需All-Gather完整单元，算完reshard/释放，降低同时存活的完整参数规模。若单元太大，瞬时峰值高；太小，collective启动次数多。粒度决定内存和通信之间的trade-off。

## FSDP1、FSDP2与自己的原型
官方不同代实现的参数表示和API会变化。FSDP1常见flat-parameter机制，FSDP2采用不同的per-parameter/DTensor组织。你的BlockFSDP是自研同步flat shard原型，不等于完整复刻任何官方版本。

当前源码使用重算配合分片；官方FSDP本身不等于“必须重算所有forward”。是否activation checkpoint、是否reshard以及prefetch/overlap是不同控制维度。

## 下一章的具体问题
你的代码怎样只从CPU搬本地shard？怎样用functional_call提供临时权重？为什么共享embedding只更新一次？为什么梯度通信转FP32却不代表梯度计算全FP32？这些问题只有沿真实forward/backward读一遍才能回答。
