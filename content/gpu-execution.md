## 为什么需要很多线程
CPU擅长复杂控制与低延迟，GPU通过大量并行线程获得吞吐。CUDA把一次kernel launch组织成grid，grid包含多个thread blocks（CTA），block包含threads。一个block在一个SM上执行；多个block可先后或并发调度，程序不能依赖任意两个block的执行顺序。

```cpp
__global__ void add(const float* a, const float* b,
                    float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
// host: add<<<(n+255)/256, 256>>>(a,b,c,n);
```

每个线程处理一个i，最后一个block可能有越界线程，所以需要mask/条件。这里不是“一核对应一个线程”；同一SM会调度许多warp，线程是逻辑执行上下文。

## Warp与分支
NVIDIA warp通常为32个线程。相邻线程执行同一指令路径时效率较好；同一warp分支不同可能需要分别执行各路径。不同program/CTA走不同分支，不等于同一warp内部发生divergence。

QK融合中head类型对一个program一致，因此Q/K选择是program-uniform分支。不能把它误说成每个lane随机处理Q或K。

## 同步范围要说清
`__syncthreads()`是block范围barrier，要求相应参与线程都能正确到达。不能在只被部分线程进入的条件分支中随意调用。Warp-level交换依赖有效线程mask；跨warp用shared memory传值时需要匹配的内存/同步机制。Warp shuffle本身不替代任意跨warp同步。

跨block规约通常采用额外kernel、atomics或有约束的协作机制。把两个需要全局同步的阶段放进普通kernel并不会自动得到全局barrier。

## Occupancy不是分数越高越好
活跃warp数量受到register、shared memory、线程/CTA资源上限约束。更多驻留warp可能隐藏访存延迟，但每线程保留更多中间结果也可能减少读写。为追求occupancy而牺牲数据复用，可能更慢。

不同GPU的warp/SM资源、Tensor Core指令与shared-memory容量需要查对应架构。不要将Hopper的WGMMA/TMA优化直接套到A6000实验。

## 项目连接
Triton `num_warps=4`是编译资源配置，不代表D=128时恰好一个元素一个线程且永不变化。要证明寄存器和occupancy，应查看编译产物与硬件报告，不能仅靠Python源码猜。
