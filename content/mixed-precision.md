## 四个dtype要分别问
输入/矩阵计算的dtype、归约或矩阵累加dtype、参数与moment存储dtype、通信dtype可以不同。写“BF16训练”不能唯一确定整个训练政策。

低精度能减少数据量并利用相应硬件，但权重很大、更新很小时，低精度存储可能无法保留更新。例如BF16在1附近的间隔约0.0078125，远小于此尺度的更新有机会在写回时被舍去。

## FP32 master为什么有用
维护FP32主权重累积小更新，每次计算再转换为BF16，可保留低精度权重暂时表达不了的状态。代价是更多持久显存。你的旧Triton AdamW完整作业使用BF16参数与moment、无master；新的BlockFSDP使用FP32 master/moments、BF16 compute。它们不能被当作同一数值契约比较性能。

## Loss scaling在做什么
FP16中很小的梯度可能下溢。先把loss乘以scale S，backward得到Sg，再在更新前除以S，可以改善某些梯度表示。动态scaler检测overflow调整S。它不能修复forward本身已经产生的NaN，也不应在未unscale时直接用错误尺度做gradient clipping。

BF16的指数范围更大，因此很多BF16设置不需要同样的scaling，但是否使用仍应按框架与任务明确说明。不能把“范围更大”理解成“数值精度比FP16更高”。

## 通信转FP32的真实作用
本项目BlockFSDP先得到compute graph的梯度，再转换为FP32进行reduce-scatter。若原梯度已经是BF16，转换只是用FP32表示已舍入数值，避免后续求和再在BF16中累积，不会恢复过去丢失的信息。

## 测试设计
记录每条路径的参数、grad、moment、accumulator、通信dtype。用梯度更新量、有限性、全量梯度相对误差与训练轨迹分层验证。单一全局L2门槛可能掩盖某些参数的局部大误差，应辅以分参数检查。

DFuzz在Megatron fused CrossEntropy中报告的强制BF16梯度cast，是这种precision policy差异的真实例子；它不只是kernel数量变化。
