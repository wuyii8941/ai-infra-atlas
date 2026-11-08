## 定义待检验的性能假设
“快了30%”缺少baseline、shape、dtype、硬件和时间边界。要先决定测的是算子GPU时间、完整forward、训练step、生成请求还是到验证目标的作业时间。不同指标不能相加或互换。

若旧时间为T0，新为T1，则speedup为T0/T1，latency reduction为1−T1/T0。两倍speedup等于50%延迟下降，不是100%。

## 为异步执行选择正确的计时边界
GPU提交后CPU可能继续运行。只用Python timer包住launch，可能主要测到提交成本。测同步wall-time时应在合理边界同步；测设备区间可以用CUDA events。单个event区间仍受所包范围影响，不能认为天然等于“纯硬件计算”。

```python
# 示意：稳态forward，同步wall-time，不能混入编译setup
for _ in range(warmup):
    fn()
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(repeats):
    fn()
torch.cuda.synchronize()
latency = (time.perf_counter() - t0) / repeats
```

逐次同步与批量循环的负载形态可能不同，报告必须记录方法。`labs/05_cuda_benchmark.py`同时输出wall与event结果，但它是教学微基准，不是复现你的模型数据。

## 从时间线到 Kernel 计数器
PyTorch profiler观察算子、shape和CPU/GPU事件。Nsight Systems观察timeline、launch间隙、同步与通信。Nsight Compute深入单kernel的访存和计算counter。采集本身可能扰动运行，所以profiling与正式计时分开。

kernel时间和是诊断，不总等于wall-time中的GPU忙碌区间：并发kernel会重叠，CPU/GPU也会重叠。不能简单用wall减kernel总和就宣布差额全是CPU launch。

## 用单变量消融隔离收益
保持同权重、同输入、同backend与cache长度，只切换一个开关；随机或正反顺序交错比较，记录warmup、JIT、graph capture是否排除。bootstrap区间只反映这组样本，不自动包含跨天、跨卡、温度和其它进程影响。

每份结果关联源码commit/hash、依赖版本、原始样本和负结果。性能候选必须先通过数值门槛，否则“快但错”不是有效优化。Counter权限不足就明确记录，不能补造利用率。

## 项目上的三种时钟
Q布局实验是GPU微秒；Graph pipeline是prefill+16步wall-time；AdamW报告的18%是同质量目标累计时间。8B分片测试包含梯度诊断、排除部分数据传输，属于功能证据，不拿它计算严肃MFU排名。
