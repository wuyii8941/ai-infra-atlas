## 为什么K/V成为容量问题
生成新token时需要读取历史位置的K/V。若每个Q head都有独立K/V，这些缓存会随层数、batch和上下文长度增长。MHA取 $H_q=H_{kv}$；MQA取 $H_{kv}=1$；GQA介于两者之间，让一组Q共享一个KV head。

若Hq=16、Hkv=8，group size为2。通常q head 0/1对应kv head0，2/3对应1。映射为

$$h_{kv}=\lfloor h_q/(H_q/H_{kv})\rfloor.$$

每个Q仍执行自己的score/PV，不是减少到只计算8个query。收益主要来自KV投影、存储及共享数据访问。

## KV容量逐项推导
一个token、一层，K与V各有Hkv×D个元素。B个序列、S个历史位置、L层、每元素b字节：

$$M_{KV}=2BSLH_{kv}Db.$$

Qwen3-1.7B的B1/S512、L28/Hkv8/D128、FP16下为56MiB。若Hkv变为16，则112MiB。缓存预分配的capacity大于当前有效长度时，实际分配按capacity算，attention有效读长度仍应正确限制。

## repeat为何抵消部分优势
为了兼容普通MHA接口，可以复制K/V到Hq，但这样会增加临时存储和读取。`expand`本身可能是view；后续reshape是否还能成为view取决于stride，不能只看到expand就断言无复制。

项目中compact GQA避免特定HF版本adapter的复制，保留原生Flash attention。框架版本与具体backend决定是否真的不物化，不应把一次实验推广到所有SDPA路径。

## MLA不是更少几个heads的GQA
DeepSeek-V3报告中的MLA把KV相关信息压缩为低维latent，再结合解耦位置部分与特定投影组织使用。其缓存对象与GQA的显式K/V heads不同，因此不能直接用上面的GQA容量式描述全部MLA缓存。MLA还涉及矩阵吸收和RoPE解耦等设计。

本项目使用的是Qwen架构GQA。8B checkpoint虽然名称含DeepSeek，但报告明确其model_type为qwen3，不能因此说实现了DeepSeek-V3的MLA或MoE。

## 实验室
用KV计算器改变Hkv和S，确认容量线性变化。然后思考：S翻倍时，单步decode attention读缓存可能变多，但模型权重没有翻倍；因此完整step latency不一定恰好翻倍。
