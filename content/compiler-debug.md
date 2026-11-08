## 建立可复现的差异基线
固定输入、seed、权重、版本与编译配置。分别记录forward、grad、dtype/shape/stride、NaN/Inf位置。重复eager与compiled自身，估计非确定性。差异度量要与语义契约匹配，不能只看一个最大相对误差。

## 按编译层逐步缩小
比较eager、Dynamo的eager backend、aot_eager、Inductor。若只有Inductor失效，重点看lowering/layout/codegen，但别把这当形式证明“所有上游层都正确”。记录同一图在不同backend的元信息和执行方式。

在目标版本使用适用日志，例如：

```bash
TORCH_LOGS="graph_breaks,recompiles,aot_graphs,output_code" \
TORCH_COMPILE_DEBUG=1 python minimal_repro.py
```

日志键和内部调试API可能随版本变化；先查该版本支持项。调试环境不一定适合性能计时。

## 最小化需要保留触发条件
减少operator、维度、输入元素和上下文，但每次确认仍命中相同失败。若缩小shape导致schedule从non-persistent切到persistent，bug可能消失；这不证明原bug是假，也不代表更小图已经等价地覆盖原artifact。

## 用单变量对照检验根因
改变一个变量：真实stride与错误stride、精确GeLU与tanh近似、启用/禁用特定融合、相同buffer上运行原kernel与patched kernel。控制变量越少，归因越强。

升到FP64仍错是结构错误的有用信号，但不是单独的充分证明。相反，FP64不再错也可能因为走了完全不同kernel，不能立刻判定原差异正常。必须检查artifact是否保持相同。

## 去重与报告
相同operator出现十个输入失败可能是同一个root cause。Fingerprint帮助聚类，但最终需要依据失败机制、版本和修复范围分析，不应把程序数当bug数。

一份好报告包括最小代码、环境、预期语义、实际现象、重复情况、编译产物和控制实验。修复后的回归需覆盖触发边界两侧。发现一次差异、归因一个机制、维护者确认、修复合并是不同证据阶段。
