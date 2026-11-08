# 练习环境

01–04 只依赖 Python 3 与 NumPy；05–06 需要兼容的 CUDA / PyTorch / Triton 环境。
在独立环境安装所需依赖，不更改项目实验环境。示例是教学练习，不是用户仓库性能复现。

```bash
python labs/01_layout.py
python labs/02_online_softmax.py
python labs/03_sharding.py
python labs/04_precision.py
# 有GPU且已经配置独立环境时：
python labs/05_cuda_benchmark.py
python labs/06_triton_add.py
```

网站计算器采用明确写出的理论模型，不采集硬件信息，也不运行CUDA代码。
