## 搜索输入与判断异常的职责
Fuzzer构造程序、输入或配置；oracle检查某种预期关系是否成立。两者职责不同。若只增加测试数量却没有合适oracle，forward正常但backward错误仍会漏掉。

Differential oracle比较预期等价的实现；gradient oracle比较输入/参数梯度；crash oracle观察异常、超时或设备错误；metamorphic oracle检查某种变换前后的语义关系。没有可信reference时，关系检查很有价值，但关系本身也要成立。

## 按参考尺度和精度选择容差
常见逐元素规则为

$$|y-\hat y|\le atol+rtol|\hat y|.$$

当参考接近0，atol很重要；当数值很大，rtol控制相对偏差。阈值受dtype、归约长度、算法、输入条件数与任务风险影响。不能规定所有FP16都1e−3、所有BF16都1e−2就结束。

全局relative L2为 $\|y-\hat y\|_2/(\|\hat y\|_2+\epsilon)$，可能掩盖少数局部严重错误。应同时记录max abs、异常元素、分参数误差与语义性质。

## 三类正确性问题
Qwen的logits/gradient gate检查模型数值接近；DFuzz用loss trajectory最大偏差与经验阈值识别训练数值风险；DeepOpFuzz进一步复现、最小化并确认compiler failure。不能把DFuzz的0.01当作所有kernel的allclose tolerance。

## 控制非确定性并区分输入敏感性
重复每条路径自身，观察噪声；对seed、dropout、backend、精度与TF32等政策做控制。扰动输入可帮助了解敏感性，但“误差突然跃迁”不是逻辑bug的自动证明，离散分支或量化边界也会跳变。

升精度、解析参考、改变布局、禁用某个优化都可以做控制，必须注意这些改动可能切换kernel。最终应把异常现象、推断根因、源码证据和正式修复状态分开记录。

## 用预算和条件表述负结果
候选更快但质量门槛失败，应报告为未采用；随机baseline几乎饱和而引导搜索没有优势，是方法边界；小shape比成熟库慢说明代价未被摊销。删除这些行只会让结论不可追问。

训练轨迹分歧不自动等于任务质量变差，也不自动等于优化函数有实现缺陷。可靠性研究需要清楚定义“错误”在哪一个层次。
