## 这段优化到底位于哪里
Qwen3 attention先执行Q/K/V projection，再做per-head Q/K norm与RoPE，之后才进入Flash SDPA。项目融合的是attention前处理，不是替换attention score/PV算法。其数学次序仍是Norm→RoPE。

在1.7B示例中，Q为 `[B,T,16,128]`，K为 `[B,T,8,128]`。每个head自己的D维完成归约与旋转，不需要与其他head交换统计量，因此一个program处理一行head是自然起点。

## Grid怎么从shape得到
真实实现把Q与K放在同一个launch中：

```python
row = tl.program_id(0)
token = row // (HQ + HK)
head = row % (HQ + HK)
b = token // T
t = token % T
```

B1/T128时，grid为 $1\times128\times(16+8)=3072$ 个program。head<16用Q和WQ，否则用K和WK。这是program内一致的分支，不是lane随机分支。

## 一个program内部发生什么
逻辑索引 `d=arange(0,BLOCK)` 加载D个有效元素。`p=where(d<D/2,d+D/2,d-D/2)`定位RoPE搭档。代码分别加载x/xp和w/wp，用x的FP32平方和求inv，再计算归一化、scale、cos/sin旋转。

```python
inv = tl.rsqrt(tl.sum(x*x, 0) / D + EPS)
z = (x*inv).to(Y.dtype.element_ty).to(tl.float32)
z = (z*w).to(Y.dtype.element_ty).to(tl.float32)
```

这些cast保留参考的部分低精度边界。代码显式读x和xp，不能把融合夸大为“全链路严格一次读一次写”；合理说法是省去完整norm中间张量的物化与部分launch。

## Stride与输出布局
内部 `_launch` 传入q/k的batch、token stride，消费packed QKV切片时不会先强制复制三份连续输入。kernel假设head内D维连续，用 `b*QB+t*QT+h*D` 寻址，因此不是支持任意strided布局。

普通输出直接写BHSD：

$$offset=((bH+h)T+t)D+d.$$

直接写K cache时，cache是 `[B,capacity,Hkv,D]`，地址改成

$$offset=((bC+position)H_{kv}+h)D+d.$$

后续版本可在K program中顺便写相应V。V只是复制，不能做QK norm/RoPE。公共校验入口要求contiguous，内部模型适配入口的stride契约更灵活，这两条入口不能混说。

## 省掉多少逻辑数据
B1/T128、Q16/K8、D128、FP16下，Q norm中间结果0.5MiB，K为0.25MiB。若完整中间结果各写一次再读一次，可避免约1.5MiB逻辑流量/层。这是理论估计，不是DRAM counter，实际cache和其它读取不在此式。

28层每层由四次Q norm/K norm/Q RoPE/K RoPE变成一次，可减少84次launch。compact GQA再减少特定adapter每层两次复制，共56次。报告kernel事件629→489，恰好少140，提供结构与profiling对应证据。

## 实验如何解释
报告FP16 B1/S128：旧优化路径prefill 22.460ms、decode 22.633ms；只加fusion为18.909/19.119ms；只加compact GQA为19.761/19.780ms；组合为16.642/16.934ms。五路径对照没有使用CUDA Graph。不能与后续16步Graph pipeline结果直接相加。

单独Q布局55.4→29.1μs来自另一个BF16实验，不是本表FP16融合的分项时间。BF16整模型仍有超过预设门槛的输入，不宣称全面无损。
