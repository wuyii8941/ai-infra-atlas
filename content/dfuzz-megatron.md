## 真实测试范围
论文在Megatron-LM的GPT-2上测试9项fusion/optimization，优化开关ON/OFF形成差分对照。最终3项超过tau=0.01：bias_gelu为0.979，bias_dropout为0.222，cross_entropy为0.0698。它们是稿件版本与条件下的loss deviation，不是所有Megatron版本的普遍缺陷断言。

## bias_gelu：数学函数已经不同
精确形式为 $GeLU(x)=x\Phi(x)$，Phi是标准正态CDF。融合路径采用tanh近似：

$$\frac{x}{2}\left[1+\tanh\left(\sqrt{2/\pi}(x+0.044715x^3)\right)\right].$$

未融合路径使用默认exact GeLU。这个差异不只是有限精度重排，而是algorithmic approximation。论文报告把两边GeLU变体对齐后偏差降为0，支持将此案例归因于函数形式差异；“exact”也不是无限精度的实数实现。

## cross_entropy：额外精度转换
Cross entropy对logits的梯度为 $p_i-1[i=y]$。论文指出对应fused backward对gradient硬编码BF16 cast，而unfused保留native dtype。额外downcast改变后续训练数值，不能只用“fusion减少launch”描述开关的全部行为。

这也解释为什么你的Qwen优化需要记录compute、storage、accumulation与communication dtype，而不只标一个BF16。

## bias_dropout：顺序和随机性要分开
该fusion将bias add、dropout、residual组合。论文归为运算重排引入的舍入差异。教学上，Dropout mask若为M，输出可写为 $M\odot(x+b)/(1-p)+residual$。不同中间cast或加法顺序可能不同。

做因果分析还要控制RNG。相同seed不必然保证不同融合路径消耗同样随机数；需检查实际mask/随机状态。当前论文不是完整RNG审计代码，本站不声称这一步已经独立复核。

## 三类“错误”的措辞
数值重排是同一数学目标的有限精度差异；近似是函数改变；实现缺陷如漏loss项则是公式不匹配。论文的额外LLM-generated baseline错误包括SimPO漏NLL、ORPO归约方式与TVD reduction不一致，这些发生在参考baseline中，不应计为三个训练框架都被发现了这些生产bug。

44项中16项超阈值、其中12项此前未记录，是论文定义下的数值风险发现。它们不能与DeepOpFuzz的54个silent compiler bugs直接相加成一个统一“70多个bug”指标。
