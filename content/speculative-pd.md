## 为什么生成有串行依赖
下一个token的输入依赖前一个token，因此普通autoregressive decoding需要逐步执行大模型。投机解码使用便宜的draft提出多个候选，再由target批量验证，把部分串行target执行换成并行verification。

它增加draft和验证工作，只有接受率、验证形状效率与额外成本合适时才快。并非减少所有FLOPs。

## 精确采样规则为什么不是简单“相同就接受”
设draft分布q、target分布p，draft抽到x。经典校正规则以 $\min(1,p(x)/q(x))$ 接受；拒绝时从归一化的正差分布 $(p-q)_+$ 采样。证明直觉是：通过draft接受进入x的概率为min(p(x),q(x))，剩余差额由拒绝后的残差分布补足，合计回到p。

实际多token算法在相应前缀条件下逐项验证，遇到拒绝处理分支。温度、top-k/top-p等sampling policy也必须一致纳入分布。greedy验证是另一种特例，不要将采样保证表述为“所有随机种子都逐位生成相同文本”。

## 期望收益来自哪里
若一次提出k个token，接受率高就可用一次较大target forward推进多个输出。接受率低时draft成本浪费；验证输入较大也可能带来额外activation与KV管理。应统计每轮接受长度、target成本、draft成本和总体TPOT，而非只报接受率。

## P/D disaggregation解决另一件事
Prefill与Decode资源形态不同，可让不同worker承担两阶段，再传递KV/state。这样可独立扩缩容与调度，但引入KV传输、队列、网络和容量管理成本。

若要搬运M字节KV，理论传输下界约M/BW；这只是传输时间，不含序列化、拓扑与同步。短prompt或单机小规模时，解耦成本可能高于收益。故不是所有部署都应拆成两套服务。

## 与本项目关系
当前Qwen原型没有实现draft模型、多请求PD服务或传输层。这章帮助把单卡Graph结果放到服务系统背景中，不应将这些扩展写成自己的贡献。
