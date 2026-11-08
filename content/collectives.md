## 进程组中的 Rank 与设备映射
Rank是通信组中的进程编号。通常每进程绑定一个GPU，但进程组可不同，local rank与global rank也不同。Collective要求各参与rank遵守一致的调用顺序、shape/count与dtype契约；一张卡跳过调用可能使其他卡永久等待。

## 用数组推导三种集合通信
两卡分别持有 `[1,2]`、`[3,4]`，All-Gather后两卡都得到 `[1,2,3,4]`。这里是拼接，不是相加。

Reduce-Scatter的输入若各为完整4元素：rank0 `[1,2,3,4]`，rank1 `[10,20,30,40]`，SUM归约为 `[11,22,33,44]`，rank0接收前半 `[11,22]`，rank1接收后半 `[33,44]`。

All-Reduce 让每张卡都拿到完整的规约结果。设每卡输入为 $x_r$，完整结果为 $y=\sum_{r=0}^{N-1}x_r$。Reduce-Scatter 让 Rank $k$ 只接收 $y$ 的第 $k$ 块，再由 All-Gather 按顺序拼出完整的 $y$，于是每卡得到 [11,22,33,44]。公式说明的是操作语义，浮点实现改变求和顺序时可能出现舍入差异。

[NVIDIA NCCL 官方集合通信图](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)

## 从局部平均推导全局梯度
每卡loss是相同有效token数上的mean时，全局平均梯度为 $g=\frac1N\sum_r g_r$。SUM只做求和，需除以N。有效token数不同应使用加权平均，不是简单除rank数。

## Ring 的通信字节数和时间模型
设每个rank参与规约的完整张量为S字节，切成N块，每块S/N。Reduce-Scatter需要N−1轮，All-Gather再N−1轮。每轮每卡发送一块，因此单卡总发送量

$$V_{send}=2\frac{N-1}{N}S.$$

接收量同样大。若统计send+recv字节则应再翻倍；全网发送总量则约 $2(N-1)S$。所以“总通信与卡数无关”是错误表述，渐近接近2S的是每卡发送量。

带每轮延迟alpha的简化模型为

$$T\approx2(N-1)\alpha+2\frac{N-1}{N}\frac S{BW}.$$

小消息和大N时延迟项不可忽略。真实NCCL可能选择tree、分层或多通道，并非任何collective都固定用单ring。

## NVLink、PCIe与RDMA
拓扑决定有效路径。NVLink/NVSwitch、PCIe与跨机IB/RoCE的成本不同；跨机TP不是数学上不可能，而是每层频繁通信可能很贵。RDMA允许网络设备按配置直接访问内存，但不会取消拥塞、同步和拓扑限制。

`labs/03_sharding.py`在CPU模拟上述数组语义与梯度等价性，不测NCCL性能。
