## 为什么“代码一样”仍可能结果不同
模型权重、数据顺序、tokenizer、PyTorch/Triton/CUDA/driver、GPU型号、compiler options与随机状态共同决定运行条件。一个可复现实验至少需要环境清单、源码commit、输入/权重hash、运行命令和原始结果。只有截图不够。

使用独立venv/Conda或容器，避免修改宿主系统Python。容器固定用户态依赖，不会自动固定宿主driver、GPU时钟、网络与文件系统。历史benchmark引用本机路径时先检查参数，不要盲目复制破坏已有checkpoint。

## CPU数据管道如何拖慢GPU
DataLoader可能等待磁盘、tokenization或CPU处理；pageable memory到GPU传输还涉及额外路径。Pinned memory和非阻塞传输可以为重叠创造条件，但前提是缓冲生命周期与stream依赖正确。更多worker不一定更快，可能增加内存、随机性和I/O竞争。

看到GPU空闲，先区分数据未准备、CPU dispatch、同步等待、通信阻塞与真正kernel执行。不要仅凭utilization=0就重启一个正在CPU初始化大模型的任务。

## Linux与进程基本功
知道PID、退出码、stdout/stderr、工作目录、环境变量和文件权限。日志停止更新不自动说明进程死锁；可能在编译、加载或checkpoint写盘。监控应只读，不要因为“没新日志”擅自启动第二份作业写同一路径。

```bash
# 只读诊断示例，不杀进程、不改系统配置
ps -ef | grep python
nvidia-smi
nvidia-smi topo -m
df -h
```

## 多卡hang从collective契约排查
先检查所有rank是否启动、设备绑定是否正确、输入shape/count与dtype是否匹配、collective顺序是否一致。有一张卡先抛OOM，其余卡可能表现为通信等待；表面NCCL超时不一定是网络根因。

复现时保存各rank日志，不仅rank0。缩到小模型/小数据，保留通信模式。跨机还需检查路由、网卡、RDMA权限与容器环境，不在未确认前随意改系统驱动或关闭安全限制。

## 部署与研究的边界
一个多卡训练脚本不是集群调度平台。Kubernetes/Slurm负责资源和作业管理的一部分，GPU任务还要明确设备需求、持久存储、checkpoint与重试。Serving需要健康检查、超时、限流、模型版本与指标。本站只讲原理，不假装你的仓库实现了这些生产平台。

实验目录应不可覆盖，失败结果保留；大模型state通常用pickle或类似序列化，不能加载不可信文件。公开网站不应包含私有语料、模型权重、凭据或未授权论文全文。
