## 两个相同 FLOPs 的程序为什么速度不同
GPU 必须先拿到数据才能计算。若每次读取只做少量运算，就可能先耗尽带宽；若同一份数据被反复复用，才更可能接近计算吞吐上限。于是定义算术强度

$$AI=\frac{F}{Q},$$

F 是 FLOPs，Q 是穿过指定存储层的数据字节数。计算 HBM 级强度时，不能把寄存器访问也混进分母。

## 对 GEMM 做一个理想估算
忽略bias和额外workspace，假设输入与输出每元素b字节、X和W各读一次、Y写一次：

$$Q_{ideal}=b(MK+KN+MN).$$

于是 $AI_{ideal}=2MNK/Q_{ideal}$。FP16、K=N=2048 时，M=1 的强度约1 FLOP/Byte，M=128 约113.8。原因是同一份 W 被更多输入行复用。实际读写可能高于这个理想值，也可能受缓存驻留影响，所以它不是硬件 counter。

## 两条上界怎样组合
设硬件在当前精度下峰值计算能力为 C FLOP/s、带宽为 B Byte/s。搬运 Q 字节至少要 Q/B 秒，计算 F 至少要 F/C 秒。理想重叠情况下

$$T\ge\max(F/C,Q/B),\quad P\le\min(C,AI\cdot B).$$

两条上界在 $AI=C/B$ 相交，这叫 ridge point。低于它时可能 memory-bound，高于它时可能 compute-bound。一个程序也可能因并行度不足、同步或调度而远离两条上界。

## 不要遗漏 launch-bound
许多几微秒的小kernel串联时，Python dispatch 和提交间隙可能变成关键路径。CUDA Graph减少重复提交，而不是提高数学吞吐；fusion可能同时减少中间数据与提交次数。不能把“显存访问很多”和“带宽已经饱和”等同。

RTX A6000 使用 GDDR6，并非 HBM。通用文献常写 HBM traffic；讨论本项目时更准确的说法是 global-memory / DRAM traffic。存储层名称与实际硬件要对齐。

## Amdahl：局部改进为什么被稀释
若原时间中某部分占比p，该部分获得s倍加速，其余不变：

$$S_{total}=\frac{1}{(1-p)+p/s}.$$

例如optimizer占10%，加速2倍，整体仅为1/0.95≈1.053倍。反过来，若完整流程快很多而GPU kernel时间和变化不大，应检查调度、同步、缓存路径，不能声称算术吞吐突然翻倍。

## 实验动作
固定shape与dtype，分别测原版、只改变布局、只改变融合、组合版本。采集timeline定位空闲间隙，再在权限允许时采集kernel counter。你的仓库报告过性能计数器权限不足，因此不能写“已经证明 DRAM 利用率达到某百分比”。
