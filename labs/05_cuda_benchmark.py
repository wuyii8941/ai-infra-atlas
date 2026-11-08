"""可选GPU教学实验：同步wall-time与CUDA Events。无CUDA时明确SKIP。"""
import time
import statistics


def main() -> None:
    try:
        import torch
    except ImportError:
        print('SKIP: 请在独立CUDA PyTorch环境中运行。'); return
    if not torch.cuda.is_available():
        print('SKIP: CUDA unavailable.'); return
    print('torch=',torch.__version__, 'GPU=',torch.cuda.get_device_name())
    x = torch.randn(1024, 1024, device='cuda')
    y = torch.randn_like(x)
    with torch.inference_mode():
        for _ in range(10): out = x @ y
        torch.cuda.synchronize()
        event_ms,wall_ms = [],[]
        for _ in range(30):
            start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            t = time.perf_counter(); start.record()
            out = x @ y
            end.record(); end.synchronize()
            wall_ms.append((time.perf_counter()-t)*1000)
            event_ms.append(start.elapsed_time(end))
    print('median event ms=',statistics.median(event_ms))
    print('median sync wall ms=',statistics.median(wall_ms))
    print('warm-cache固定shape教学测量；不锁频、不测服务吞吐；并非仓库性能结果复现。')


if __name__ == '__main__':
    main()
