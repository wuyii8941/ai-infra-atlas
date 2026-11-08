"""CPU 实验：shape、stride、view 与复制。运行：python labs/01_layout.py"""
import numpy as np


def main() -> None:
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    t = x.T
    c = np.ascontiguousarray(t)
    print('x =\n', x, '\ntranspose =\n', t)
    for name, array in [('x', x), ('transpose', t), ('copy', c)]:
        element_strides = tuple(s // array.itemsize for s in array.strides)
        print(name, 'shape=', array.shape, 'element strides=', element_strides,
              'C-contiguous=', array.flags.c_contiguous)
    assert np.shares_memory(x, t)
    assert not np.shares_memory(x, c)
    # [row=2,col=1] 在转置view中的地址：2*1 + 1*3 = 5。
    assert t[2, 1] == x.reshape(-1)[5]
    b = np.broadcast_to(np.array([7.], dtype=np.float32), (4,))
    assert b.strides == (0,)
    print('broadcast stride=', b.strides, 'writeable=', b.flags.writeable)
    # NumPy广播view通常只读；不要把这个flag外推成PyTorch行为。
    print('PASS: view共享storage；contiguous复制；stride按元素解释。')


if __name__ == '__main__':
    main()
