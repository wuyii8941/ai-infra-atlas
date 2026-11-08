## 存储层级与数据复用
寄存器通常是线程私有的高速状态；shared memory在CTA内显式协作；L1/L2是硬件cache；global memory来自设备DRAM。A6000使用GDDR6，不能将所有设备显存都称HBM。线程私有的local memory名字容易误导：register spilling可能落到设备内存路径，不是免费的寄存器。

## 按 Warp 地址集合判断合并访存
一个warp的32个线程读取连续FP32元素，共128字节。若访问落在少量对齐sector，事务效率较高。若每个线程跨很大stride读取，可能触及很多sector，实际传输字节大于有用字节。

不是说线程顺序必须严格单调；即使地址被线程间置换，只要覆盖相同少量sector，也可能合并良好。反之，Tensor整体contiguous不保证某个kernel的线程索引模式coalesced。

## 以转置观察读写地址变化
朴素转置可以做到连续读，但输出地址对相邻线程相距一整行，写入不连续。常见方法把tile连续读入shared memory，在线程间交换访问方向，再连续写出。

在简化的32个bank、4字节宽度模型下，bank索引近似为 `(word_address % 32)`。若shared数组是 `[32][32]`，某些按列读取让线程访问同一bank的不同地址，会产生bank conflict。加一列padding为 `[32][33]` 可改变stride，分散bank。实际行为受元素宽度、广播和硬件架构影响，不能只凭数组维度替代测量。

## 分块改善数据复用与访存
GEMM的A tile会被同一输出tile内多个列复用，B tile会被多个行复用。把tile保存在片上，减少重复从global memory取数。代价是更多片上资源和同步，tile不可能无限大。

## 区分地址推导与硬件观测
Q布局55.4→29.1μs和peak delta下降说明指定路径改善；源码显示去掉了部分copy。它们不能直接证明精确的DRAM事务减少百分比，更不能证明所有数据只读一次。项目QK代码为RoPE配对还会显式加载x/xp与w/wp，实际cache行为需要进一步分析。

更好的实验是保持公式与输入不变，仅改变layout或tile，比较设备时间与端到端时间，并查看真实memory counters。权限不够时，就将计数器结论留空。
