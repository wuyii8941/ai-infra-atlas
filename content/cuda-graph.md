## 为什么先做static cache
动态cache若每步cat并分配新storage，地址和shape会变化。一个已经捕获的CUDA Graph记录GPU操作、依赖和kernel参数，包括地址。项目选用预分配cache、固定token buffer及GPU position/length状态，让每步只改变内容而不改变关键存储地址。

这是一种实现方式，不是说所有CUDA Graph永远不能更新参数，也不是说static cache单独必定加速。CUDA还有图更新等机制；本项目没有依靠那些机制处理任意动态服务。

## decode状态怎么变化
`self.token.copy_(new_token)`把新输入写入固定buffer。`cache_position`与有效长度用GPU Tensor更新。QK kernel按位置直接写cache。Attention只读有效前缀，不应该读未初始化的capacity尾部。

项目采用 `[B,capacity,Hkv,D]` 存储，在固定PyTorch 2.5.1下使用私有Flash varlen接口与GPU `seqused_k`。仅给dense接口随意添加length参数，不保证尾部被屏蔽。这个私有API依赖是工程边界，不包装成跨版本公共接口。

## capture了什么，没capture什么
`StaticDecodeRunner`捕获一个token的完整forward及LM Head。prefill不捕获，greedy argmax、输入复制、position/length更新在图外，必须计入完整pipeline。Graph输出是复用缓冲区，跨replay保存结果需clone，否则后续调用可能覆盖旧结果。

Graph减少重复主机调度，不会把所有kernel融合成一个kernel，也不会消除全部CPU开销。捕获过程可以通过框架allocator支持某些分配，不能笼统说“capture中任何张量分配都非法”；稳定的重放存储与生命周期才是关键。

## 消融证据
A6000/FP16/B1/S128，prefill+16步pipeline：旧优化eager263.872ms；static/no-graph270.590ms；graph113.542ms；graph+packed/direct-K110.098ms。static单独无加速，Graph是主要贡献，packing/direct write是增量。

QKV packing保留原权重时另占448MiB快照。更少launch不等于更少总显存，也不等于降低GEMM数学FLOPs。

## 走向服务时缺什么
固定batch/capacity、等长无padding是当前契约。服务需要处理请求进入/结束、不同序列长度、多尺寸图、静态页池与动态block table、取消请求与错误恢复。逻辑状态可动态变化，物理地址保持稳定，但需要另一个调度层。
