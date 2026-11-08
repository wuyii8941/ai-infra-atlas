## 研究问题
Source program只是编译行为的输入之一。shape/dtype/layout/context与compiler options共同决定具体operation和artifact。方法需要同时解决到达目标和观察错误，而不只增加operator列表多样性。

## 方法的工程组成
从compiler internals抽象目标；path entries/conditions/composition构造程序；观察heuristic/fusion/extern/backward等artifact；按under-reached状态反馈；对适用表面执行oracle，即使没命中目标；最后复现、最小化与去重。

论文说明目标构建不使用评测失败trace或oracle答案，减少target construction与evaluation泄漏。对其他栈的移植使用对应可观察产物，不要求都生成Triton。

## 证据一：主要发现活动
| 栈 | Silent | Crash | 合计 | 已修复（稿件统计） |
|---|---:|---:|---:|---:|
|PyTorch/Inductor|15|16|31|20|
|TensorFlow/XLA|28|3|31|5|
|IREE|3|4|7|2|
|TVM|8|9|17|16|
|总计|54|32|86|43|

这来自上传稿件p7 Table I，不代表本站重新验证每条issue，也不代表当前上游状态永远不变。

## 证据二：30个根因的四小时回放
论文将PyTorch的31项发现按root cause分成30个目标，Table II中DeepOpFuzz到达13，Random5，四个baseline的并集为6。这个实验回答“对相应可复现版本，谁能触达这些根因及编译阶段”。

## 证据三：独立48个历史bug的效率
另一个benchmark建立在历史PyTorch/Inductor版本，包含48个可复现bug，用于RQ2/RQ3。Table III报告5个seed、4小时每轮均值：DeepOpFuzz bugs reached13.0，Random8.0；kernels stressed317与77。它与前面的30根因实验独立，不能把13/30和13.0/48混为同一比例。

“Kernels reached”指至少命中一次，“stressed”指至少被10个不同程序命中。它们是诊断探索信号，不是形式化代码或语义覆盖率。随机方法可能触达更多不同fingerprints，却更少触达深层缺陷。

## 两个案例的角色
pool stride案例说明shape、batch与schedule决定是否暴露错误；slice_scatter案例说明forward正确不能保证backward正确。前者帮助理解reachability，后者帮助理解observability；两者都有source、generated code与控制实验的定位链。

仓库个别issue draft包含强语气或泛化判断，网站只采用最小可支持结论：指定版本、输入与artifact的失败机制。正式修复位置和最新上游状态不由草稿标题推定。

## 面试表述
最有价值的不是“我会fuzzing”，而是能解释如何把编译条件映射为测试输入、如何确认实际生成的代码、如何通过解析或差分oracle定位错误。讨论AI Compiler时，应沿lowering、metadata、scheduling和codegen展开，而不是堆没有实现证据的关键词。
