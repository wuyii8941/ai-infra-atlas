## 先独立推导预期梯度
令x形状 `[4,13,33]`，y为 `[4,6,33]`，把x的dim1前6个位置替换成y，再沿dim1求和。

$$out_{b,h}=\sum_{i<6}y_{b,i,h}+\sum_{i\ge6}x_{b,i,h}.$$

接收上游梯度g[b,h]后：x的前6个位置梯度为0，后7个位置为g；y的每个位置为g。这个解析关系不依赖输入随机值，是比单纯eager对照更强的参考。

## 报告中的错误
对应Inductor路径forward正常、y.grad正常，却把x.grad变为全0。FP64也有相同结构性错误，说明不能用低精度误差解释。只检查forward的fuzzer会漏掉。

## 广播产生 stride-0 上游梯度
sum的backward把 `[B,H]` 上游梯度unsqueeze并expand为 `[B,13,H]`，中间维stride为0。作为broadcast输入，这是合法的；但scatter输出需要不同位置有0或g的数值，不能继续让整维都指向同一物理位置。

报告的AOT metadata仍给slice_scatter输出stride-0。Inductor据此把iteration space从4×13×33折成4×1×33=132，并把dim1索引当成0。判断0是否落在[0,6)为真，最后全写0。

## 用单变量对照定位触发条件
把slice换成[5,12)，折叠坐标0不在其中，判断为假，于是所有20个位置都保留g，而预期只有13个。报告对应维度20的案例梯度总和比例为20/13，符合错误机制的预测。

`aot_eager`执行正常，因为真实ATen执行可以物化满足语义的输出，错误注释不一定影响eager结果。不能据此简单说“AOT图完全正确”，更准确是元信息与后端使用方式的交互需要定位。

## 输出布局需要容纳独立写入
不必。functional输出并非一律要求contiguous；只要layout能忠实表达语义即可。这里问题是有不同值的位置被错误地表示为同一地址。前稿若把修复说成“所有functional op必须contiguous”，会过度泛化。

## 将定位过程复用于布局错误
先写解析梯度，后用backend与metadata缩小范围，再检查generated code；在输入/shape变化下看根因是否预测新症状。梯度值、shape、stride、alias与可继续求导性都属于训练正确性的可能表面，不只是loss一个标量。
