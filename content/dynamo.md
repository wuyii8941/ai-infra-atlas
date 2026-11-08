## 为什么不直接执行Python
Eager按运行顺序调用Tensor操作，编译器希望看到更大计算范围，以消除中间张量并选择高效实现。`torch.compile`不是把任何Python程序一次性变成一个巨大GPU kernel，而是捕获可编译区域，再交给后端。

Dynamo在Python frame/bytecode执行层跟踪Tensor计算，生成FX graph，并建立图可以被安全复用的条件。这里的graph是操作与依赖，不是CUDA Graph的设备提交拓扑；两个“graph”属于不同层。

## Guards在保护什么
一次调用可能依赖dtype、device、shape/stride约束、Python对象状态或函数身份。Guard检查这些假设是否仍成立。输入变化可能命中已有缓存版本，也可能触发新的specialization/recompile。

Guard失败不等于graph break。Graph break是在捕获一个执行过程时，遇到当前无法表达或安全处理的行为，将执行拆为编译区域和Python部分。过多断图可能减少融合机会，增加调度。

## 动态shape也不是没有任何条件
Symbolic dimension允许某些尺寸在约束范围内变化，编译器仍需保证合法的布局、broadcast和控制条件。把尺寸全做symbolic可能影响优化，也不能自动处理任意数据依赖的Python分支。

```python
# 观察而非保证生成特定kernel；需要兼容PyTorch环境
f = torch.compile(lambda x: (x + 1).relu(), backend='inductor')
y1 = f(torch.randn(32, 128, device='cuda'))
y2 = f(torch.randn(64, 128, device='cuda'))
# 用 TORCH_LOGS='recompiles,graph_breaks' 在目标版本检查日志
```

首次运行可能包含编译；后续shape变化可能再编译。不能将含fallback或编译失败的对照当作有效baseline。

## fullgraph的意义
`fullgraph=True`要求被捕获调用形成一个完整捕获图，遇到不支持的graph break会报错；它不意味着最终只有一个GPU kernel，不保证所有操作都由Triton生成，也不保证性能最好。

## 项目连接
Qwen整模型compile强基线需要对齐static cache、Flash后端与Graph政策，并检查重编译次数。DeepOpFuzz则把shape/dtype/context等作为影响编译选择的输入。相同operator列表并不决定相同artifact。
