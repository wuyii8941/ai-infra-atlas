## 用实现与测量证据组织项目介绍
先讲研究对象和问题，再说你改变了什么，为什么这种改动合理，测了什么、没测什么。源码能证明实现结构；消融能支持性能归因；论文统计能支持特定评测结果。这些证据不可互相替代。

三条可用的叙事依赖如下，而非未经记录的真实研发时间线。

<div class="flow"><span>Qwen计算结构</span><span>执行开销假设</span><span>Kernel/Runtime方案</span><span>数值gate</span><span>消融与端到端</span></div>
<div class="flow"><span>DFuzz数值差异</span><span>训练动态</span><span>配置搜索</span><span>输入搜索</span><span>阈值与根因</span></div>
<div class="flow"><span>DeepOp编译条件</span><span>目标operation</span><span>artifact观测</span><span>oracle</span><span>隔离复现</span></div>

## 用单个指标串起技术因果
被问“Q分支为什么55.4到29.1μs”，应先界定BF16布局消融的GPU时间；解释view/stride与physical copy；指出源码如何消费stride；说明该数据不等于FP16整模型或Graph收益。

被问“8B为什么能在4卡训练”，先做master/grad/m/v显存账本；再说明按block聚合、重算、reduce-scatter；最后用真实checkpoint三步更新与导出证据收束，而不是只说“用了FSDP”。

被问“Megatron工作是什么”，先说DFuzz测试9项优化与两阶段搜索；再用GeLU近似和CrossEntropy cast展开。TP理论可以作为追问知识，不把它包装成论文中的主要实现贡献。

## 用推导和实验检验掌握程度
拿一个未见shape，算出M/K/N与Q/K/V shape；拿一个Tensor，写出地址；拿一段forward，推导一个输入/参数梯度；拿一张结果表，找出baseline与时间范围；拿一个失败，提出能区分两个根因的控制实验。

若只能复述“减少memory traffic”，但说不出哪个buffer被写读、有没有实际counter、是否只是逻辑流量估计，就还没有到可深入面试的程度。

## 为结论标注适用条件
“这段是我读过的通用机制，仓库未实现”“这个数字是报告结果，我没有在当前环境重跑”“这项只通过短程gate，不证明质量无损”都比补造经历好。不要用under review等未经本轮核实的状态替代技术内容。

## 建议阅读路线
从零：Tensor→FLOPs→Roofline→Autograd→Tokens→Transformer→Attention→Norm/RoPE。推理：GPU→Triton→QK Fusion→Flash→Decode→Graph→Serving。训练：AdamW→混合精度→累积/重算→Collectives→FSDP→源码→Checkpoint。编译/研究：Dynamo→AOT→Inductor→Codegen→Oracle→DFuzz/DeepOp。

这份站点重点覆盖上述项目依赖及常见AI Infra基础。硬件特定的极限PTX优化、完整RLHF平台、生产级多租户调度与故障恢复不是这里已经实现的系统。扩展章节提供原理入口，不把范围外能力冒充成项目成果。
