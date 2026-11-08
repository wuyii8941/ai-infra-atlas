"""用NumPy模拟flat parameter + all-gather + reduce-scatter。不是NCCL benchmark。"""
import numpy as np


def main() -> None:
    rng = np.random.default_rng(42)
    n, world, local_batch = 7, 4, 3
    width = (n+world-1)//world
    w = rng.normal(size=n)
    padded = np.pad(w, (0, width*world-n))
    shards = list(np.split(padded, world))
    gathered = np.concatenate(shards)[:n]
    np.testing.assert_array_equal(gathered, w)
    xs = [rng.normal(size=(local_batch, n)) for _ in range(world)]
    ys = [rng.normal(size=local_batch) for _ in range(world)]
    # loss = mean((Xw-y)^2)/2；梯度是X.T@(Xw-y)/B。
    local_grads = [x.T @ (x@w-y)/len(y) for x,y in zip(xs,ys)]
    global_x, global_y = np.concatenate(xs), np.concatenate(ys)
    reference = global_x.T @ (global_x@w-global_y)/len(global_y)
    summed = np.sum([np.pad(g, (0,width*world-n)) for g in local_grads], axis=0)
    reduced = [piece/world for piece in np.split(summed, world)]
    reconstructed = np.concatenate(reduced)[:n]
    np.testing.assert_allclose(reconstructed, reference, atol=1e-12)
    lr = .05
    updated = np.concatenate([p-lr*g for p,g in zip(shards,reduced)])[:n]
    np.testing.assert_allclose(updated, w-lr*reference, atol=1e-12)
    print('7个参数/4rank：每rank2个元素，padding1个零。')
    print('PASS: gather、平均梯度和一次SGD更新与global batch参考一致。')
    print('条件：各rank样本数相同、loss归约相同；不同有效token数要做加权。')


if __name__ == '__main__':
    main()
