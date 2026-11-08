## 共享权重汇总两条路径的梯度
设E为V×H。输入lookup取E[token]，输出head做 $Z=HE^T$。共享参数参与两条计算路径，因此链式法则要求

$$dE=dE_{embedding}+dE_{head}.$$

若一个token多次出现，lookup分支对应同一行还要累计多次贡献；head分支通常产生dense梯度。不能只保留最后到达的一份，也不能给同一参数创建两个互不一致的optimizer状态。

## 一个小例子
词表4行，输入ids=[1,1,3]。lookup梯度为u0/u1/u2，则E第1行收到u0+u1，第3行收到u2，第0/2行无lookup贡献。最后每行再加head梯度。这是数学需求，与具体kernel实现无关。

## 通过自定义 Function 明确反向依赖
项目 `TiedEmbedding.forward` 返回embedding结果与`weight.view_as(weight)` proxy，让两条路径汇合到同一个自定义backward。backward先stable sort token IDs，找到相同token的组，以FP32累计lookup贡献，再与head dense gradient合并。

真实代码还会clone dense gradient输出；它不是完全避免所有V×H缓冲区，也没有把输入grad原地覆盖。算法的舍入顺序与原始HF可能不同，所以必须有完整模型gradient gate。

## 引用与保存张量决定显存生命周期
模型适配把proxy暂存在head属性。`consume_proxy=True`在head取出proxy后删除该属性，减少一个长期引用。Tensor最后一次算术使用完成，不等于所有引用和autograd需求都结束；不能被释放的storage会与后面的临时分配重叠，抬高峰值。

删除属性不是“立刻cudaFree”的保证，也不能仅从显存减少量反推出allocator内部一一对应。合理结论是生命周期改动与峰值改善在受控实验中相关，数值公式未更改，但执行与舍入仍须验证。

## 配对实验和边界
BF16/B1/S512、compiled HF+FlashOptim基线：104.163→98.408ms与反向顺序104.469→97.595ms，下降5.53%/6.58%；peak allocated11.140→9.849GiB。所用FlashOptim是外部基线optimizer，并非本项目新实现。

S2048的backward临时峰值仍可能比baseline更高，及时释放仅部分缓解。256步变动输入、独立validation最大loss差0.00151是短轨迹回归，不是新收敛实验。不能和旧AdamW或Graph实验相加。

## 实践与验收
写一个小词表实现上述手工累加，对照autograd，再测试重复/热点token、非连续grad、FP16/BF16/FP32、梯度累积和重算。训练CLI的恢复模式也应验证，避免一半模块用旧路径、一半用新路径。
