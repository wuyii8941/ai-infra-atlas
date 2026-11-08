## Prefill 建立缓存，Decode 逐步追加
Prefill一次处理已有prompt；每层产生prompt位置的K/V，并形成首次生成所需输出。Decode之后逐次处理新token，复用历史K/V。小batch下Linear的M从B×S变为B×1，但权重大小不变。

以前章GEMM强度为例：小M使一份权重服务的输入行少，权重读取更难摊销。大M更容易用数据复用接近计算上限。不是每个prefill算子都compute-bound，也不是每种decode配置都memory-bound。

## 从 Query 和 KV 长度推导 Attention 成本
prefill的QK/PV约随S²增长。decode的新query长度1，Attention每步读取的有效K/V长度从prompt开始逐步增大，单步score/PV随cache长度线性增长。Linear权重成本近似不随cache长度改变，因此整模型latency不一定跟S成正比。

KV cache只避免历史K/V重算，不能消除新token在所有层的Linear、norm、MLP和LM Head计算。

## TTFT、TPOT与吞吐
TTFT是请求到第一个输出token的时间，可能包含排队、tokenization和prefill，具体定义要声明。TPOT可用首token之后的总耗时除以输出间隔数估计，但逐token延迟分布仍可能不均匀。Throughput是单位时间处理的tokens或requests，不等于单请求latency的倒数，尤其在batching服务中。

一个服务提高batch后可能tokens/s更高，却使单请求TPOT更差。评价应结合延迟SLO，即预定服务目标，而不是只看平均吞吐。

## 项目对应的测量范围
你的早期HF benchmark比较prefill与单步decode；StaticDecode报告测完整prefill+16次连续decode，包含LM Head、argmax、cache/state维护，但排除tokenizer和服务排队。不能将16次固定KV状态的重复执行当作16次真实生成，因为cache长度应增长。

同样，“优化eager路径快2.8倍”不代表快于compiled HF、vLLM或SGLang。强基线可能已经减少相同开销，所以应继续比较带Graph和相同Flash后端的compile执行路径。

## 推导一个工作量账本
对每层记Linear的2MNK、Attention的4B Hq Sq Sk D、norm/activation元素量和KV读写。再与profiling对应，判断哪部分成本真正主导。这个账本比泛泛说“推理就是带宽瓶颈”更能指导优化。
