## 项目出发点
Qwen模型定义给出数学计算，现有框架选择一种具体执行方式。项目研究不同执行方式能否减少延迟和内存开销，并维持规定的数值契约。按资料，应分成推理性能、单卡训练性能和多卡分片三条工作流，而不是虚构“先失败再顿悟”的研发时间线。

## 工作流一：Attention前处理与Decode runtime
**问题来自哪里：** Q/K projection后重复进行Norm/RoPE，并有布局转换与特定HF GQA adapter的复制。它们FLOPs少，却可能有中间物化和启动成本。

**选择为什么合理：** per-head norm与RoPE具有连续依赖，head内计算可放入一个program；stride-aware读写避免为了接口整理数据；compact GQA保持原生Flash；固定decode拓扑适合Graph重放。

**实际实现：** `_qk_norm_rope`、`make_attention_forward`、`StaticDecodeCache/Runner`。直接写K/V cache与packed QKV作为独立开关；Graph只捕获decode forward和LM Head。

| 实验层次 | 报告结果 | baseline与范围 |
|---|---|---|
| Q布局消融 |55.4→29.1μs；12→4MiB|特定BF16 Q分支，GPU时间及新增峰值分配|
| QK fusion+compact GQA |decode约25–32%下降|FP16固定配置，相对此前优化路径，非Graph|
| Static Graph pipeline |约2.4–2.8×|旧优化eager vs Graph组合；prefill+16步|
| 强compile对照 |B1/S128近似持平|同static cache/Flash/Graph条件；不能复用eager倍率|

**结论边界：** 不替换成熟Flash attention；不是生产服务吞吐；BF16有旧反例；不同阶段和dtype的数据不可相加。

## 工作流二：Optimizer与Autograd数据流
**问题来自哪里：** 完整训练step含optimizer与共享参数反向流，单kernel最快不等于全作业最快。

**实际实现：** TritonAdamW以native fused AdamW为对照，处理舍入边界；TiedEmbedding合并lookup/head梯度；consume_proxy缩短临时引用生命周期。

| 证据 | 观察 | 不能推导 |
|---|---|---|
| 单对完整训练作业 |同验证目标7208→5912s，约18%减少|不是跨种子收敛速度保证|
| 完整进程 |8320→7682s，约7.7%减少|终点步数不同，不是同工作量kernel倍率|
| 新强基线稳态step |5.5–6.6%减少，allocated少1.29GiB|不是新的完整收敛实验|
| 16GiB预算实验 |吞吐12.7–13.4%提升，microbatch2→4|仅README报告短程budget结果，原始artifacts未公开|

**负结果同样重要：** S2048内存回退、训练compact GQA梯度gate失败、训练Graph在更强基线下收益不足、量化生成分歧。它们限定采用范围，而不是说明相关算法普遍无效。

## 工作流三：Block-level full sharding
**问题来自哪里：** 8.19B模型BF16权重约15.26GiB，但FP32 master/gradient/m/v可达约122GiB，还不含activation。DDP复制状态不能解决每卡容量。

**实际实现：** CPU初始化、本地FP32 shard、BF16 all-gather、functional_call执行block、backward重算、FP32 reduce-scatter SUM与本地平均，最后只更新本地shard。NCCL负责跨卡，Triton负责本地cast/scale。

**验证：** 四张A6000，真实DeepSeek-R1-0528-Qwen3-8B，S512、每卡B1、三次全参数更新；每卡allocated峰值35.7568GiB；HF导出重载有限测试logits最大差0。第一层Q projection约18.63%BF16元素变化说明确实更新。

**边界：** 同步、单节点、无overlap；不是比官方FSDP快的benchmark；未验证真实8B完整optimizer恢复；不是原版Qwen3-8B，也不证明收敛或长上下文质量。

## 学习依赖图与最终呈现
推理线：GEMM成本→Norm/RoPE依赖→layout/fusion→GQA→Graph→消融。训练线：梯度→AdamW状态→数值政策→共享梯度→生命周期→完整step与作业。分片线：显存账本→collectives→unit sharding→recompute→恢复/导出。

面试介绍应先选与岗位最相关的一条，再用一个源码点和一组同口径数据展开。不要把全部结果压成“用Triton把Qwen加速2.8倍并训练8B”，那会隐藏真正贡献与比较边界。
