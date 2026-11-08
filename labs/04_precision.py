"""CPU实验：FP32非结合律与有限值BF16 RNE舍入的位级教学模拟。"""
import numpy as np


def bf16_round_finite(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float32)
    if not np.all(np.isfinite(a)):
        raise ValueError('Only finite values are supported in this educational simulator')
    bits = a.view(np.uint32)
    rounded = bits + np.uint32(0x7fff) + ((bits >> 16) & np.uint32(1))
    return (rounded & np.uint32(0xffff0000)).view(np.float32)


def main() -> None:
    a,b,c = np.float32(1e20),np.float32(-1e20),np.float32(3.)
    print('(a+b)+c =', np.float32(np.float32(a+b)+c))
    print('a+(b+c) =', np.float32(a+np.float32(b+c)))
    values = np.array([1., 1.001, 1.003, 1.004, 1.008, -1.004],dtype=np.float32)
    quant = bf16_round_finite(values)
    print('FP32:', values)
    print('BF16 represented as FP32:', quant)
    assert quant[0] == 1.0 and quant[1] == 1.0
    np.testing.assert_array_equal(bf16_round_finite(quant), quant)
    print('PASS: 有限值rounding幂等；此模拟不覆盖GPU归约、FMA、异常值或硬件吞吐。')


if __name__ == '__main__':
    main()
