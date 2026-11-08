## 显式分数矩阵的存储与 IO 成本
朴素attention先生成scores、softmax概率，再乘V。单head的scores为S×S，长序列时中间存储与读写急剧增长。FlashAttention用Q/K/V tile与online统计量，将score/probability保留在局部计算中，避免完整矩阵写回设备显存。

数学目标仍是dense softmax attention，不是自动稀疏化。核心收益是IO-aware执行，而不是把所有dense attention FLOPs从S²变成S。不同版本还改进工作划分、warp并行与硬件流水线。

## 固定一块 Q 并遍历 K/V 块
取Q的若干行，一次读一块K/V，得到局部score，更新每行的running max m、denominator l、输出累加器u。遍历完后写出u/l。前章已推导三个状态的缩放合并，所以不需要保存全局probabilities。

Backward可保存较少统计量并重新计算部分attention概率。这会增加某些算术工作，但减少读写和存储，有机会整体更快。不能仅以FLOPs多少比较两个算法的实际延迟。

## Split-KV 提升长缓存 Decode 的并行度
B很小、query长度1时，若每个query head只有一个program，可用并行工作数量有限。长cache又使单program需要遍历很多K/V。Split-KV把cache分给多个program，各自算partial output和LSE，再用额外kernel合并。

代价是多一次合并与partial workspace。短cache可能不值得拆。不同Query共享KV时，同program处理多个grouped heads可以复用KV读取，却增加程序内状态，不能保证总是更快。

## 项目中的独立 Kernel 实验
A6000/BF16/B1/Hq16/Hkv8/D128，KV=8192时，旧自写Triton约593.818μs，最佳新版约120.934μs，约4.91倍；同表Flash SDPA约60.928μs。因此结论是“改进自写长KV实现”，不是“超越FlashAttention”。真实HF路径仍保留native Flash。

KV=128时，新版比旧版略慢，展示了拆分成本。报告也明确partial/LSE workspace不是0。上述数据是固定张量的Graph GPU计时，不能代替完整生成pipeline或服务吞吐。

## 正确性与性能验收
验证不整齐长度、不同GQA比例、mask政策、累积误差与merge是否计入时间。再在真实模型中看收益是否还存在。Flash、Paged KV、GQA和CUDA Graph解决不同层次的问题，不应混为“attention优化”。
