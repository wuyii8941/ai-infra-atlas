## microbatch与optimizer step不是同一单位
模型一次forward/backward处理Bμ个样本，可以积累A次梯度后再更新。如果有Dp个数据并行rank且样本等权，global batch为

$$B_{global}=D_pB_\mu A.$$

这不是TP degree，TP的多个rank可能在共同计算同一个样本，不能重复乘进去。

## loss应该如何缩放
若每个microbatch有效token数都相同，每个loss是token mean，累积前通常除以A以得到整体mean梯度。如果有效长度不同，应按token数加权，不能无条件把每个microbatch的mean再等权平均。

```python
optimizer.zero_grad(set_to_none=True)
for micro in microbatches:
    loss = model(micro).loss
    (loss / accumulation_steps).backward()  # 仅适用等权示例
optimizer.step()
```

分布式的梯度同步时机也会影响通信量。不能把少同步误认为梯度数学可以任意改变。

## 为什么更大microbatch可能更快
同一global batch下，增加Bμ减少A，可以减少重复调度并提高GEMM的M维利用率，但activation显存会增加。最佳点是硬件、shape、allocator和算法共同决定的。

项目最新README报告16GiB allocator预算、S512、effective batch8下，baseline最优2×4，候选4×2，短程吞吐提高12.7–13.4%。这是报告中的稳态预算实验，不是已独立获取所有原始日志的复测，也不代表收敛更快。

## activation checkpointing为什么增加FLOPs
反向需要forward中的中间值。只保存block输入，不保存全部内部activation，backward再运行该block forward，即用额外计算换内存。若一段forward成本F，训练矩阵计算原约3F，完全重算该forward会增加到约4F，实际比例视保存策略而变。

## 重算必须重现的不只有输入
Dropout等随机算子需要一致的RNG状态，否则重算得到另一条forward，梯度可能不再对应原loss。mutation、buffer更新和副作用也需要保持语义。你的BlockFSDP明确只支持确定性、无随机dropout重算，并保留每个block的输入；不是“整个模型完全不保存激活”。
