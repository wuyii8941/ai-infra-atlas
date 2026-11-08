## 由张量元数据计算元素地址
Qwen 的布局优化和 DeepOpFuzz 的索引 bug，都依赖同一个事实：**shape 不是物理地址**。Tensor 通常由 storage、storage offset、shape、stride 和 dtype 共同描述。stride 的单位是元素，不是字节。

若张量元素坐标为 $(i_0,\ldots,i_{n-1})$，其元素偏移为

$$o=o_0+\sum_{k=0}^{n-1} i_k s_k.$$

字节地址还要乘以元素大小。对连续 $2\times3$ 的 FP32 矩阵，stride 是 $(3,1)$。元素 $(1,2)$ 的偏移是 $1\cdot3+2=5$，距首元素 20 bytes。

## 转置通过交换形状与 Stride 创建视图
原矩阵在内存中是 `[0,1,2,3,4,5]`。转置后 shape 变为 $(3,2)$，stride 变为 $(1,3)$。逻辑上的 $(2,1)$ 仍读 offset 5，没有必要生成新 storage。

```python
import torch
x = torch.arange(6).reshape(2, 3)
y = x.t()
print(x.shape, x.stride())  # (2,3), (3,1)
print(y.shape, y.stride())  # (3,2), (1,3)
print(x.untyped_storage().data_ptr() == y.untyped_storage().data_ptr())
z = y.contiguous()
print(z.stride())           # (2,1)，需要时物理复制
```

`contiguous()` 对已经连续的输入可能直接返回原张量，并不保证一定复制。`reshape` 在可行时产生 view，否则可能复制；不能无条件把它当免费操作。原地修改 view 可能影响 base，这就是 alias 关系的重要性。

## 广播视图的共享读与独立写入
`u.shape=[B,1,H]` 扩展到 `[B,S,H]` 时，可以设置中间维 stride 为 0。于是所有序列位置读同一个物理向量。作为只读 broadcast view，这很节省内存；作为需要每个位置不同写入结果的输出，就可能无法表达语义。

DeepOpFuzz 的 `slice_scatter` 案例正是相关输出 metadata 与实际需求不一致。不能由“functional op 返回新张量”进一步断言其输出一定 contiguous；关键是输出 layout 必须容纳正确的逐元素结果，而不是强制某一种连续顺序。

## Qwen 的 BSHD 与 BHSD
设 token 数为 $T$、head 数为 $h$、head dimension 为 $D$。连续 BSHD 的偏移是

$$((bT+t)h+j)D+d.$$

连续 BHSD 的偏移是

$$((bh+j)T+t)D+d.$$

两者改变的是谁在物理上相邻。一个 stride-aware kernel 可以按输入实际 stride 读，直接按目标布局写，避免先调用 transpose-copy。这需要明确支持哪些 stride，不能由一个实例推广到任意非连续张量。

## 实践
运行 `labs/01_layout.py`，观察 NumPy view 与复制的差异。然后对手里的 PyTorch Tensor 同时打印 shape、stride、storage_offset、data_ptr。不要只打印 `is_contiguous()`，因为它不足以描述所有地址关系。
