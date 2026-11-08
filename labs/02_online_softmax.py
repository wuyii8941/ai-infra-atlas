"""CPU float64 教学实验：分块 online softmax 的 (m,l,u) 合并。"""
import numpy as np


def streaming_attention(scores: np.ndarray, values: np.ndarray, chunk: int) -> np.ndarray:
    if scores.ndim != 1 or values.ndim != 2 or len(scores) != len(values):
        raise ValueError('Expected scores:[S], values:[S,D]')
    if len(scores) == 0 or chunk <= 0 or not np.all(np.isfinite(scores)):
        raise ValueError('Require nonempty finite scores and positive chunk')
    m, l = -np.inf, 0.0
    u = np.zeros(values.shape[1], dtype=np.float64)
    for start in range(0, len(scores), chunk):
        block = scores[start:start+chunk]
        val = values[start:start+chunk]
        new_m = max(m, float(block.max()))
        alpha = np.exp(m - new_m) if np.isfinite(m) else 0.0
        p = np.exp(block - new_m)
        u = alpha*u + p @ val
        l = alpha*l + p.sum()
        m = new_m
    return u/l


def main() -> None:
    rng = np.random.default_rng(7)
    for length in (1, 17, 129, 513):
        s = rng.normal(size=length)*40.0
        v = rng.normal(size=(length, 8))
        p = np.exp(s-s.max()); p /= p.sum()
        ref = p @ v
        for chunk in (1, 7, 32, 128):
            got = streaming_attention(s, v, chunk)
            np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)
    s = np.array([0., 0., 10.]); v = np.array([[0.], [0.], [1.]])
    wrong = (0.0+1.0)/2  # 两块的已归一化输出不能无权平均。
    right = float(streaming_attention(s, v, 2)[0])
    print(f'无权平均={wrong:.6f}; 正确合并={right:.6f}')
    print('PASS: 16组长度/分块对照通过；这是CPU数值实验，不是FlashAttention性能测试。')


if __name__ == '__main__':
    main()
