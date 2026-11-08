## 微批次、累积步数与全局 Batch
模型一次forward/backward处理Bμ个样本，可以积累A次梯度后再更新。如果有Dp个数据并行rank且样本等权，global batch为

$$B_{global}=D_pB_\mu A.$$

这不是TP degree，TP的多个rank可能在共同计算同一个样本，不能重复乘进去。

## 按有效 Token 数归一化 Loss
设第 $i$ 个微批次有 $n_i$ 个有效 Token，$L_i$ 是其平均损失，$n_i>0$。全局平均损失应先加总所有 Token 的损失，再除以有效 Token 总数：

$$L_{\mathrm{global}}=\frac{\sum_i n_i L_i}{\sum_i n_i},
\qquad \nabla L_{\mathrm{global}}=\sum_i\frac{n_i}{\sum_j n_j}\nabla L_i.$$

例如两个微批次各有 4、12 个有效 Token，梯度权重分别是 $1/4$ 和 $3/4$。只有每批有效 Token 数相等时，权重才都等于 $1/A$。多 Rank 时同一原则适用于全局 Token 数，并需要把通信操作是否额外除以 Rank 数计入缩放。

```python
optimizer.zero_grad(set_to_none=True)
for micro in microbatches:
    loss = model(micro).loss
    (loss / accumulation_steps).backward()  # 仅适用等权示例
optimizer.step()
```

分布式的梯度同步时机也会影响通信量。不能把少同步误认为梯度数学可以任意改变。

## 微批次大小影响算术强度与激活显存
同一global batch下，增加Bμ减少A，可以减少重复调度并提高GEMM的M维利用率，但activation显存会增加。最佳点是硬件、shape、allocator和算法共同决定的。

项目最新README报告16GiB allocator预算、S512、effective batch8下，baseline最优2×4，候选4×2，短程吞吐提高12.7–13.4%。这是报告中的稳态预算实验，不是已独立获取所有原始日志的复测，也不代表收敛更快。

## Activation Checkpointing 的存储与重算账本
**定义。** Activation Checkpointing 在初次前向只保存选定的段边界或必要值，反向走到该段时再生成用来求导的内部激活。例如 $y=xW$ 的反向要计算 $\partial L/\partial W=x^T G$，其中 $G=\partial L/\partial y$。若 $x$ 是段内上一层产生的激活，可以从段输入重新算出 $x$，再立即用它计算梯度。

**显存账本。** 峰值包括权重、梯度、优化器状态，以及同一时刻活跃的激活、临时工作区和通信缓冲。重算首先改变的是激活的生命周期。混合精度改变每个元素的字节数，ZeRO/FSDP 改变每卡常驻的状态，张量并行沿维度拆分计算与部分张量。它们可以与激活重算组合。

**分段粒度。** 设有 $L$ 个等大小 Block，每段含 $s$ 个 Block，段输入占 $b$ 字节，每个 Block 在反向前需要的内部激活约占 $a$ 字节。为看清量级，近似把所有段边界与当前重算段的内部激活一起计入，忽略工作区、边界释放顺序和非均匀 Block：

$$M_{\mathrm{act}}(s)\approx \left\lceil\frac{L}{s}\right\rceil b+sa.$$

第一项是跨前向与反向保留的边界。段越短，段数越多，边界占用越高。第二项是反向处理当前段时短暂重建的激活。段越短，同时存活的段内激活越少。忽略取整时，对 $Lb/s+sa$ 求导得到 $-Lb/s^2+a$，于是连续近似的平衡点为 $s_\ast=\sqrt{Lb/a}$。实际需在 $1\le s\le L$ 中比较可用的整数切法。

例如 $L=24$，取 $a=b=1$ 个内存单位。每段 12 层约为 $2+12=14$，每段 4 层约为 $6+4=10$，每段 2 层又升到 $12+2=14$。因此细切能否继续省显存，取决于段内激活的降幅是否大于新增边界的占用。

**计算账本。** 设初次前向成本为 $F$、反向成本为 $B$、重算成本为 $R$。训练总成本从 $F+B$ 变为 $F+B+R$。在线性层主导、需要计算输入和权重梯度的近似下，$B\approx2F$；如果所有段各完整重算一次，$R\approx F$，则从 $3F$ 增至 $4F$，即这部分算术工作增加约 $1/3$。在相同重算覆盖下，切成更多段未必增加 FLOPs，但可能增加调度开销。实际实现可只重建反向所需的值。

[PyTorch Activation Checkpointing 文档](https://docs.pytorch.org/docs/stable/checkpoint.html)

## 重算的一致性条件
Dropout等随机算子需要一致的RNG状态，否则重算得到另一条forward，梯度可能不再对应原loss。mutation、buffer更新和副作用也需要保持语义。你的BlockFSDP明确只支持确定性、无随机dropout重算，并保留每个block的输入；不是“整个模型完全不保存激活”。
