## 静态批处理中的完成时间差异
假设同一batch有请求A、B，A先完成而B还要生成很多token。静态batch可能让A留下的计算/缓存槽不能及时被新请求利用。Continuous Batching按生成iteration调度，允许已完成请求退出，新请求加入，提高利用率。

这要求scheduler、cache manager与model runner协作：scheduler决定本轮执行哪些token；cache manager分配与回收存储；runner把逻辑请求映射为可执行张量与attention metadata。

## 按页分配 KV Cache
若每个请求预分配最长上下文的连续区域，短请求会浪费空间；若不断重新分配增长区域，又会产生搬运与碎片。Paged KV把序列拆成固定token数的逻辑块，通过block table映射到物理页。

逻辑token位置t可分解为 `page=t//block_size`、`slot=t%block_size`。先查物理page，再计算页内偏移。最后一页仍可能有内部碎片，并非数学上完全零浪费。attention backend必须理解分页索引或由系统转换，不能只建立字典就让普通dense kernel自动支持。

## Prefix Cache 复用公共前缀
多个请求有相同前缀，已计算的KV可共享，节省重复prefill。缓存有效性依赖模型权重、token IDs、位置与适配器等上下文，而不是只比较可见文本。共享页被写入时需要合适的copy-on-write或其它所有权策略，不能让不同请求修改彼此历史。

## 分块 Prefill 缩短对 Decode 的阻塞
很长prefill可能阻塞正在decode的请求，恶化TPOT。将prefill拆成若干token chunk，并与decode按token预算调度，可改善公平性与峰值负载；代价可能是prefill完成更晚、更多调度，以及attention上下文管理。Chunking不改变已发生的token依赖。

## 批调度与 CUDA Graph 的协作条件
可对离散batch/token bucket捕获多张graph，使用稳定buffer与page pool，更新block table、长度和slot映射。必要的padding会浪费部分计算，更多图也会增加内存和维护成本。它是一组权衡，不是“动态batch不能用Graph”的绝对冲突。

## 项目边界
你的StaticDecodeRunner没有实现这些生产调度机制。这里是从原型走向serving的扩展知识，不应写成已实现vLLM级引擎。服务评价需要并发请求、输入/输出长度分布、排队、P50/P99和SLO，而非只计一个固定生成请求。
