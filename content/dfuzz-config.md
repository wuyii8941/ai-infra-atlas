## 从论文真正的观察开始
上传DFuzz稿件以Liger RMSNorm为动机：8层、hidden512、BF16的LLaMA训练500步，loss差由4.2×10⁻⁵到1.8×10⁻²，gradient L2差由7.7×10⁻³到8.2×10⁻¹。换20组训练配置，最终偏差跨近8个数量级。

这些是稿件报告的特定实验，不证明所有舍入误差必然累积，也不证明任意轨迹差异都会损害最终质量。它们说明单步固定配置测试不足以刻画风险。

## 为什么训练动态会放大差异
SGD中 $\theta_{t+1}=\theta_t-\eta g(\theta_t)$。两条路径出现小误差后，下一步梯度已经在不同参数点计算。设参数差e，局部线性化可近似理解为

$$e_{t+1}\approx(I-\eta H_t)e_t-\eta\delta_t.$$

H是局部loss Hessian，delta是该步实现引入的梯度偏差。此式是教学分析，不是论文已证明的稳定性定理。不同曲率和learning rate可放大或收缩误差；Adam还增加moment状态。

## 实际搜索目标是什么
配置c包含架构、batch、学习率与精度等。对每个配置运行同seed、同数据条件下的ON/OFF训练轨迹，评价

$$\Delta(c,x)=\max_{1\le t\le N}|L_{on}^{(t)}-L_{off}^{(t)}|.$$

选择max而非只看最后一步，是因为偏差不必单调。论文的fitness在参考off训练有界时取 $\phi_d(c)+\lambda\phi_k(c)$，否则置为负无穷；phi_d是轨迹偏差，phi_k是kernel执行差异信号。

## 为什么用遗传算法
配置混合离散与连续变量，改变架构还会改变图与kernel，每次训练评价昂贵。GA以种群、选择、交叉、变异在合法空间中复用较好候选。它是启发式优化，不证明全局最优。

Megatron实验N=200步、P=20、5代，约100次配置评价/优化；配置的ON/OFF都需执行。论文实验设置写roulette-wheel selection、uniform crossover概率0.6、mutation0.20、top25% elitism；方法概述另用“top-ranked parents”概括。网站保留这一表述差异，不假装已审计DFuzz完整源码来统一二者。

## Kernel discrepancy有什么用
Megatron通过Nsys观察dispatch差异，引导候选真正改变优化执行路径；Liger/Unsloth设置lambda=0，不做这项profiling辅助。它是搜索信号，不是正确性oracle，更不是代码覆盖率。

最终保存最佳配置及偏差峰值时刻t*的两侧model states，交给输入搜索阶段。给定有限预算找到的最大值是已观察风险，不是对所有配置的上界。
