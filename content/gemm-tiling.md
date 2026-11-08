## 最朴素实现为什么重复取数
计算C[i,j]需要遍历k。若每个线程只负责一个输出，并各自加载A行和B列，邻近输出会反复访问同一输入。Cache可能缓解，却不能替代设计良好的复用。

将输出分成BM×BN tile，K按BK分块。每轮加载A的BM×BK块、B的BK×BN块，更新BM×BN个累加器。

$$C_{tile}\mathrel{+}=A_{tile}B_{tile}.$$

这轮完成约 $2B_MB_NB_K$ FLOPs，加载约 $bB_K(B_M+B_N)$ 字节。较大的BM/BN提高复用，但增加累加器和片上资源占用。

## Tensor Core不是自动等于高性能
Tensor Core执行特定shape/dtype的矩阵乘加指令。库或编译器必须把大矩阵分解成支持的指令形态，并处理对齐、边界、数据布局和累加精度。小M可能使tile利用率差；K/N非整齐会产生mask或padding成本。

TF32、FP16、BF16、FP8的支持与吞吐随架构变化；不能把某型号峰值用于另一dtype。即便计算很快，搬数据或启动不足也会使实际吞吐远离峰值。

## 软件流水线怎样隐藏等待
当前K tile计算时预取下一块，可以在依赖允许时重叠加载与计算。`num_stages`更多通常意味着更多缓冲与资源占用，并不单调提速。必须在BM/BN/BK/warps/stages组合中做受控扫描。

```text
阶段 0: load K-tile 0
阶段 1: compute tile 0 + prepare tile 1
阶段 2: compute tile 1 + prepare tile 2
结束: epilogue + store C
```

## epilogue与fusion
GEMM后紧接bias/activation时，可以在累加器写回之前做适当变换，避免完整中间矩阵。若消费者要求跨tile归约或不兼容布局，则不一定适合融合。一个producer有多个消费者，也不代表绝对不能融合，而是需要权衡物化、重算与数据复用。

## 对应项目实验
仓库的Gate/Up GEMM调优扫了选定BM/BN/BK/warps/stages组合，不是全笛卡尔积。在相同舍入路径上比较tile，才便于隔离性能因素。它保留FP16某些shape慢于cuBLAS的结果，而且benchmark不包含down projection，也不是完整训练。

实验应报告最佳值的搜索范围、调优成本和独立复测。只公布某shape上一次最优值，不能推出通用kernel的优势。
