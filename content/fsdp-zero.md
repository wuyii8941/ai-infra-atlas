## 从权重容量推导训练显存
设模型有 $P$ 个参数。BF16 每个元素 2 字节，因此仅存权重时约占 $2P$ 字节。若训练维护 FP32 主权重、FP32 梯度与两个 FP32 Adam 矩状态，这四项各占 $4P$，合计 $16P$ 字节。单独存在的 BF16 计算副本、激活与工作区应分别计入，峰值只累加在同一时刻存活的张量。

例如 $P=8\times10^9$，BF16 权重为 $16$ GB，约 $14.90$ GiB。上面的四个 FP32 状态共 $128$ GB，约 $119.21$ GiB。若四卡均匀分片，每卡的对应状态账本约为 $119.21/4=29.80$ GiB，还要为聚合参数、激活和临时缓冲留空间。实际精度和分片策略不同，按各项的实际元素数与字节数计算。

## DDP 的每卡状态账本
每卡一份模型与optimizer，各自处理不同数据，再同步梯度。这样提高数据并行吞吐，却不降低每卡完整参数与optimizer状态的基本存储。如果完整训练状态在单卡放不下，简单增加DDP rank并不解决。

## ZeRO 的分阶段分片
ZeRO-1切optimizer state；ZeRO-2还切gradient；ZeRO-3进一步切parameter。FSDP full-sharding采用相近的全状态分片思路，但具体API、调度、flatten表示与实现不等于ZeRO库本身。

公式 $M\approx M_{state}/N$ 只能估理想持久分片部分。真实峰值还包括临时unshard、完整当前单元gradient、activation、prefetch、allocator与通信buffer。N倍显存总和不是一个无需调度的统一地址空间。

## 按计算单元临时聚合参数
计算一个block需要它的权重。按需All-Gather完整单元，算完reshard/释放，降低同时存活的完整参数规模。若单元太大，瞬时峰值高；太小，collective启动次数多。粒度决定内存和通信之间的trade-off。

## FSDP1、FSDP2 与项目原型
官方不同代实现的参数表示和API会变化。FSDP1常见flat-parameter机制，FSDP2采用不同的per-parameter/DTensor组织。你的BlockFSDP是自研同步flat shard原型，不等于完整复刻任何官方版本。

当前源码使用重算配合分片；官方FSDP本身不等于“必须重算所有forward”。是否activation checkpoint、是否reshard以及prefetch/overlap是不同控制维度。

## 把分片策略映射到源码
在 [BlockFSDP 源码主线](#article/block-fsdp) 中，先追踪 CPU 权重如何切出本地分片，再看临时聚合的参数如何交给模块执行。反向得到当前单元的梯度后，Reduce-Scatter 把结果交回本地分片。共享参数汇总所有路径的梯度后执行一次更新。

[PyTorch FSDP2 教程](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
