## 程序为什么本来很简单
```python
def program(x):
    y = torch.nn.functional.adaptive_avg_pool2d(x, 7)
    return y.flatten(1).sum(dim=-1)
```

以pool输出 `[B,65,7,7]` 为例，一个batch有65×7×7=3185个元素。连续buffer的batch stride应该是3185，flatten只是把后三维解释为一维。

## 错误地址如何产生症状
报告中的生成reduction kernel使用 `r + 3200*b`，而不是 `r + 3185*b`。3200等于3185向上对齐到32的倍数。**逻辑归约的tile对齐长度不可以冒充真实buffer stride。**

b=0时，错误stride乘以0，地址恰好相同。b=1时，实际有效数据从3185开始，而kernel从3200开始，偏15个元素。若B=2，最后一些读取还会超过逻辑buffer有效范围；实际越界是否崩溃受分配与相邻内存影响，silent结果也可能波动。

## 为什么mask没救
`r < 3185`只验证行内逻辑r的范围；若base本来错误，base+r仍可能越界。局部tail mask不能修复错误的外层寻址。

## 证据链
仓库给出eager/compiled差异、不同batch与shape的规律、FP64复现、生成代码中的buffer stride和索引表达式。再把相同Triton kernel抽出来，在相同输入buffer和launch条件下只改3200→3185，误差回到正常浮点量级。

这有力支持错误机制位于生成索引表达式，而不是GPU随机算错。它不等于已经审计所有上游layout来源，也不代表当前最新版本仍然受影响；状态要按报告与上游issue日期。

## 不要沿用过度简化的描述
仓库case展示pool输出buffer以及消费它的fused view/reduction kernel，不能自动说pool+flatten+sum三个阶段全部在同一个kernel内。论文概念示例与具体artifact分解需分别描述。

较小C=65是否选择故障reduction路径依赖GPU与版本。大的复现shape有助于触发相应路径，但不能仅凭经验宣称在所有架构都必然选择相同schedule。

## 和fuzzing设计的关系
需要B>1才看出outer stride偏移；需要对齐边界与特定schedule才命中。它说明编译条件与artifact观测是重要搜索维度，但不证明所有graph fuzzing永远找不到这个bug。
