## 为什么训练需要 backward
模型预测错了，需要知道每个参数怎样影响loss。最简单例子为 $y=wx$、$L=(y-t)^2/2$。链式法则给出

$$\frac{\partial L}{\partial w}=(y-t)x,\quad\frac{\partial L}{\partial x}=(y-t)w.$$

反向模式自动微分将上游梯度逐层乘以局部导数，避免对每个参数分别运行一次完整forward。面对数十亿参数、标量loss，这很重要。

## 为什么共享参数的梯度必须相加
若同一个w影响两条分支 $L=f(w)+g(w)$，则 $dL/dw=f'(w)+g'(w)$。因此多条反向路径不能简单覆盖同一grad。Qwen tied embedding/head需要收集两次使用的贡献；FSDP还要在不同rank之间汇总本地数据的贡献。它们是两种不同的“累加”。

## PyTorch训练循环每一步干什么
```python
optimizer.zero_grad(set_to_none=True)
logits = model(inputs)
loss = criterion(logits, targets)
loss.backward()       # 写入/累加参数的 .grad
optimizer.step()      # 用grad更新参数与optimizer状态
```

`backward()` 不自动清空旧梯度。`step()` 不自动重新运行forward。`model.eval()` 改变dropout等模块行为，不等于关闭autograd；`no_grad()` 控制是否记录求导图。推理通常同时需要适当的eval模式与禁用梯度记录。

## 为什么 backward 需要保存中间值
例如SiLU的导数依赖输入；矩阵乘法的dW依赖X。框架会保存求导所需的Tensor，不能在forward一结束就全部释放。保存得越多，activation显存越大；重新计算则用计算换存储。

`detach()` 返回与原Tensor共享存储但脱离当前求导链的视图语义结果，不是复制。`clone()` 复制数据，但在有梯度上下文时仍可保留求导关系。需要独立数值与独立图时常用 `detach().clone()`，但不能机械套用。

## 自定义 autograd.Function
它允许明确写forward和backward的契约。backward必须返回各Tensor输入的梯度，正确处理None、shape、dtype与多分支贡献。你的TiedEmbedding在backward合并梯度；BlockFSDP在backward聚合参数、重算并reduce-scatter。一个是改变梯度的数据流，一个是管理分片生命周期。

实际验证应同时看forward、输入梯度、全部参数梯度以及连续几步更新；只看loss接近会漏掉backward-only bug。
