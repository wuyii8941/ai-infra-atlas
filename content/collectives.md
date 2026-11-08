## Rank并不是GPU全局物理编号
Rank是通信组中的进程编号。通常每进程绑定一个GPU，但进程组可不同，local rank与global rank也不同。Collective要求各参与rank遵守一致的调用顺序、shape/count与dtype契约；一张卡跳过调用可能使其他卡永久等待。

## 三个操作用数组解释
两卡分别持有 `[1,2]`、`[3,4]`，All-Gather后两卡都得到 `[1,2,3,4]`。这里是拼接，不是相加。

Reduce-Scatter的输入若各为完整4元素：rank0 `[1,2,3,4]`，rank1 `[10,20,30,40]`，SUM归约为 `[11,22,33,44]`，rank0接收前半 `[11,22]`，rank1接收后半 `[33,44]`。

All-Reduce则让每张卡都拿到完整 `[11,22,33,44]`。概念上，SUM Reduce-Scatter后All-Gather可组成相同语义。

## 平均梯度为什么还要除以world size
每卡loss是相同有效token数上的mean时，全局平均梯度为 $g=\frac1N\sum_r g_r$。SUM只做求和，需除以N。有效token数不同应使用加权平均，不是简单除rank数。

## Ring公式从哪里来
设每个rank参与规约的完整张量为S字节，切成N块，每块S/N。Reduce-Scatter需要N−1轮，All-Gather再N−1轮。每轮每卡发送一块，因此单卡总发送量

$$V_{send}=2\frac{N-1}{N}S.$$

接收量同样大。若统计send+recv字节则应再翻倍；全网发送总量则约 $2(N-1)S$。所以“总通信与卡数无关”是错误表述，渐近接近2S的是每卡发送量。

带每轮延迟alpha的简化模型为

$$T\approx2(N-1)\alpha+2\frac{N-1}{N}\frac S{BW}.$$

小消息和大N时延迟项不可忽略。真实NCCL可能选择tree、分层或多通道，并非任何collective都固定用单ring。

## NVLink、PCIe与RDMA
拓扑决定有效路径。NVLink/NVSwitch、PCIe与跨机IB/RoCE的成本不同；跨机TP不是数学上不可能，而是每层频繁通信可能很贵。RDMA允许网络设备按配置直接访问内存，但不会取消拥塞、同步和拓扑限制。

`labs/03_sharding.py`在CPU模拟上述数组语义与梯度等价性，不测NCCL性能。
