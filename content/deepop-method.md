## 为什么随机graph不够
同一个pool→flatten→sum，在不同channel、stride、dtype或autograd context下，可能产生不同reduction artifact。Operator出现过，不代表目标schedule或融合被执行；pass运行过，也不代表最终构造了目标kernel。

论文将问题拆成reachability和observability。前者问“怎样到达特定编译产物”，后者问“它错了，哪个检查能看到”。forward-only检查不一定发现gradient或metadata问题。

## 目标如何表达
Operation target包含进入路径的entry、具体编译条件与operator composition约束。entry可能是操作族或执行上下文；condition包括shape、dtype、layout、device与配置；composition说明依赖关系。它还需要matcher说明怎样从artifact判断命中。

<div class="flow"><span>选低频target</span><span>构造程序与条件</span><span>编译/执行</span><span>观察artifact</span><span>更新hit</span><span>执行并验证oracles</span></div>

## 观察什么才算反馈
论文的PyTorch实现从Triton heuristic selector、split-scan与extern dispatch等内部源头构建目标，用delegating wrapper记录heuristic trace，观察fused patterns、extern variants与kernel-name hashes。它们是诊断和探索信号，不是严格代码覆盖率。

目标构建使用compiler source，不用评测bug答案或oracle结果来反向制作目标，以避免evaluation leakage。测试工具当然可能受手写目标taxonomy的覆盖范围限制。

## 一个容易画错的箭头
**目标没命中也要执行适用oracle。** 论文明确oracle dispatch不依赖target hit，因为偏离目标的程序仍可能暴露其他失败。若把流程写成“命中才检查”，会漏掉论文实际设计。

每次候选失败需要fresh-process复测、dtype-aware数值筛查、nondeterminism控制、minimization与去重。matcher可能只看可见artifact的一部分，因此hit不等于已经覆盖某operation内部全部状态。

## 从PyTorch到其他栈
TensorFlow/XLA观察optimized HLO与fusion信息；IREE/TVM是portability case studies，不能说四个栈都有完全相同深度的instrumentation。MLIR operation与论文operation的测试抽象也不是同一名词定义。

## 项目重要性
方法把测试目标从“生成很多source程序”推进到“在具体条件下构造并核验编译选择”。它与你手写Triton优化互补：前者质疑自动生成执行路径是否正确，后者设计并验证特定执行路径是否更高效。
