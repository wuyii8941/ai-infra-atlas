"""可选GPU：向量相加，覆盖非BLOCK整数倍的mask。CPU环境只报告SKIP。"""

def main() -> None:
    try:
        import torch
        import triton
        import triton.language as tl
    except ImportError:
        print('SKIP: requires CUDA PyTorch and Triton in a separate environment.'); return
    if not torch.cuda.is_available():
        print('SKIP: CUDA unavailable.'); return

    @triton.jit
    def add_kernel(X, Y, O, N: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X+i, i<N, 0)
        y = tl.load(Y+i, i<N, 0)
        tl.store(O+i, x+y, i<N)

    for n in (1, 127, 128, 129, 10003):
        x = torch.randn(n, device='cuda', dtype=torch.float32)
        y = torch.randn_like(x)
        out = torch.empty_like(x)
        add_kernel[(triton.cdiv(n,128),)](x,y,out,n,128)
        torch.testing.assert_close(out, x+y)
    print('PASS: five shapes; only correctness, no speedup claim.')


if __name__ == '__main__':
    main()
