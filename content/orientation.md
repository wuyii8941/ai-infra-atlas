## 这不是一份名词清单
语言模型的每个 token 都要经过张量计算。框架把计算交给编译器、GPU 和运行时；训练又引入反向传播、优化器和多卡通信。AI Infra 关心这条链能否**跑得动、跑得快、算得对**。三个目标有关联，却需要不同证据。

本教程借鉴 AIInfraGuide 的分层学习形式，正文与例子按你的三个项目重新组织，不转载其文章。每篇先说明问题的来源，再推导数学与数据流，最后连接源码、实验和边界。

<div class="flow"><span>模型数学</span><span>Tensor / Autograd</span><span>Compiler / Kernel</span><span>GPU / Runtime</span><span>数值与性能验证</span></div>

## 三个项目分别回答什么
**Qwen3-Triton** 研究执行效率：相同模型计算如何减少布局复制、启动开销和训练状态驻留。其单卡推理、单卡训练、多卡分片是不同实验分支，不应编造成未经记录的线性研发历史。

**DFuzz** 研究训练轨迹的数值偏差：同一优化开关的 ON/OFF 执行，在哪些配置与输入下产生显著不同的 loss 轨迹。Megatron-LM 是实际测试框架之一，不是仅供背景介绍的分布式知识。

**DeepOpFuzz** 研究编译正确性：怎样构造条件，让编译器选择特定 operation 并观察生成产物，进一步检测输出、梯度或执行语义错误。

## 怎样阅读
零基础路线先读 [Tensor 与 stride](#article/tensor-layout)、[矩阵计算量](#article/gemm-flops) 和 [token 到 loss](#article/tokens-loss)，再进入 Transformer。Kernel 路线从模型算子走到 Triton 和 QK Fusion。训练路线从梯度、AdamW 的状态走到 collective 与 FSDP。Compiler 路线从 graph capture 走到 artifact 与两个真实 bug。

不要跳过前置章节。一个词第一次出现时可能有一句定义，完整解释会在对应链接中展开。例如 GQA 不是“一个更快的 attention 函数”，而是 head 共享设计；CUDA Graph 不是让所有算子变成一个 kernel。

## 网站怎样表达证据
标为**原理与推导**的段落是教学解释，算式中的例子不等于项目实测。标为**源码与实验**的章节来自固定仓库版本或上传论文。标为**扩展知识**的部分不构成你的项目贡献。

Qwen 源码固定在 `10be3d87`；DeepOpFuzz 仓库固定在 `c69ead1b`。历史报告里的 2.5.1 推理环境、2.10 训练环境与 compiler 测试环境不同。本站没有在 GPU 上重新执行这些项目实验，性能数字按报告口径呈现。

## 什么算真正掌握
读完定义只是第一步。合格的理解是：给定一个具体 shape，能写出地址或计算量；给定一个优化，能说明它省了什么、增加了什么；给定一组数据，能判断是否足以支持结论。每篇末尾的折叠题和实践任务就是为此设计。

网站不打包你的论文 PDF、简历、私有日志或模型权重。公开部署前仍应确认论文与研究结果可对外展示。
