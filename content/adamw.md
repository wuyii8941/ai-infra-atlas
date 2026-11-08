## 从 SGD 更新到 AdamW 的一阶矩与二阶矩
SGD用 $\theta_{t+1}=\theta_t-\eta g_t$ 更新参数。Adam额外跟踪梯度的一阶、二阶指数移动平均：

$$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,$$
$$v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.$$

若初始为0，第一步m的期望会偏向0；bias correction使用 $\hat m_t=m_t/(1-\beta_1^t)$，v同理。AdamW把weight decay从梯度的自适应缩放里分离：

$$\theta_{t+1}=(1-\eta\lambda)\theta_t-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.$$

因此每个参数不仅有自身值，还需要两个moment。学习率eta、beta、epsilon与step计数也是训练状态，恢复checkpoint时不能遗漏。

## 从公式推导数据量
一次更新至少读参数、grad、m、v，写回参数、m、v。全FP32的理想逻辑流量约(4+3)×4=28 bytes/parameter。算术量也随参数数线性增长，复用有限，访存与调度是合理的分析方向。但实际native fused实现可能已充分融合，不能因“自写融合”就解释全部性能优势。

## Triton 实现的精度与舍入策略
`training/triton_adamw.py` 的默认rounded路径在FP32中进行中间标量运算，存储dtype随参数与moment保留。它把beta的补数从host传入，用`tl.div_rn`处理若干除法，并关闭相应浮点融合；历史legacy路径保留作消融。

```python
m = B1*m + C1*g
v = B2*v + C2*g*g
denom = tl.div_rn(tl.sqrt(v), BC2) + EPS
p = p - tl.div_rn(tl.div_rn(LR, BC1)*m, denom)
```

C1/C2 由补数的计算顺序决定。先把 beta 舍入再求补数，与先求补数再舍入，可能得到不同的 FP32 值。原生实现可能用双精度表达式处理超参数，当前 Kernel 使用 FP32 标量，因此应使用数值对照验证，逐位相同需要额外的前提。

代码还是按参数循环launch，不是一个全模型persistent optimizer；不要把另一个多张量实验的结论归给它。

## 优化器的逐步对照验证
只比较最终参数相对误差可能掩盖极小更新。应分别比较参数更新量、m/v、step、保存恢复后的下一步，以及多步loss轨迹。BF16参数/no-master设置与FP32 master训练不等价，不能跨设置直接宣称无损。

## 实验结论
项目完整作业只替换AdamW，HF forward/backward与Flash SDPA未变。相同验证目标累计耗时7208.42→5912.05秒；完整进程8320.38→7682.28秒。目标首次达到都在13750步，但平台停止分别15750/17500步。只有一对顺序作业，不能说收敛步数减少、全部训练配置更快或优于compiled HF完整训练。
