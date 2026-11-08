## 精确续训所需的状态
要执行“下一步相同更新”，除了参数，还需要Adam m/v、step、学习率调度、FP32 master、随机状态、数据游标/顺序、梯度累积阶段与精度设置。仅能from_pretrained加载的模型文件，通常不包含这些训练状态。

你的8B HF导出用于部署并验证logits重载；它不是已验证的8B完整AdamW恢复。小模型的checkpoint恢复通过，也不能自动覆盖大模型存储和I/O风险。

## 分片清单记录布局与版本
每个rank保存自己的参数/optimizer分片。需要记录world size、参数结构、padding、dtype和训练协议；恢复时应拒绝不兼容状态，或明确实现resharding。不能在改变world size后直接按旧rank文件加载并假装完成弹性恢复。

项目采用写临时文件、flush/fsync后rename，并通过完成标记说明各rank保存结束。该方式提高单次写入的可靠性，但分布式失败恢复、网络文件系统一致性和任意故障原子性仍需更完整协议。

## 容量账本
FP32 master与两个FP32 Adam moments约12P字节，P=8.19B时约91.54GiB，未计元数据/安全空间。BF16模型导出约2P字节，即15.26GiB。两者差异来自保存内容，而不是某种神奇压缩。

## 数据顺序也属于状态
相同模型/optimizer若恢复后换了batch顺序，训练轨迹自然不同。记录token文件hash、block索引、shuffle种子与sampler状态，才能区分数据变化和数值错误。验证集只用于既定协议，test不能用于事后调节门槛。

## 用中断前后的连续训练对照验证恢复
连续运行k步得到reference；另一路运行j步保存，主动破坏内存参数并清空optimizer，重新加载再运行k−j步。对比参数、moment和下一步loss，检查错误world size/precision时是否拒绝。真实大模型还要考虑磁盘容量与中途崩溃，不应只验证文件存在。

## 完整作业计时
checkpoint I/O可能占很大比例。你的AdamW报告中写盘约占总主函数时间的三分之一；局部step加速会被I/O稀释。把setup或共享预处理排除时应另报，不能称从原始数据下载到训练完的总加速。
