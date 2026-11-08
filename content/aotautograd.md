## 训练编译不能只优化forward
若只把forward交给编译器，backward中的大量算子仍无法统一优化。AOTAutograd利用PyTorch求导机制构造可被后端编译的前反向表示，通常涉及joint graph，再partition为forward与backward图。这里“AOT”相对于后续执行，不意味着在Python程序首次运行前已经完成全部静态编译。

## 为什么要joint graph
以 $y=\sin(x)\cdot x$ 为例，backward需要x和cos(x)。可以在forward保存中间值，也可以backward重算。把两部分依赖一起观察，有机会在保存张量、重算成本和融合之间权衡。partition得到的forward可能返回供backward使用的额外saved tensors，不只是用户可见输出。

## Functionalization不是删除所有副作用
原地操作与view使alias关系复杂。编译前的函数式转换将部分mutation表示为显式输入输出关系，同时确保对用户可观察的修改仍被正确反映。将 `x.add_` 改写为新tensor后，还必须考虑谁引用旧storage、最后怎样把更新反馈出去。

Fake/Meta执行用不持有真实数据的Tensor元信息帮助推导shape、dtype和stride。错误元信息可能让真实算子在eager里正确，进入编译lowering时却出错。它不是“随便填一个正确shape就行”。

## backward输出是什么
对上游向量g，自动微分实际常算vector-Jacobian product，而非显式构造巨大Jacobian。梯度的shape、dtype、alias和requires_grad语义都可能重要。higher-order gradient还涉及对反向计算继续求导，不是所有编译/自定义kernel路径都支持。

## Compiled Autograd不要混用
普通torch.compile经AOTAutograd处理捕获区域的backward；PyTorch另有Compiled Autograd机制，在运行时autograd engine层捕获更大反向图，涉及hooks等不同边界。不能把“backward被编译了”一概说成使用了这个特定功能。

## 项目中的定位方法
`backend='eager'`、`aot_eager`、`inductor`可帮助逐步缩小差异出现的层次。aot_eager正确但Inductor错误，说明后端使用图/metadata的路径值得检查，却不能据此断言所有上游metadata都完美无误。你的slice_scatter案例正是这一细节。
