import torch
from parallel_mult import ParallelMatMul
def test_2x1_bug():
    T = 256
    TB = 32
    A = torch.Tensor([[1.0, 1.0], [1.0, 1.0]])
    B = torch.ones(2)
    C = ParallelMatMul(A.cuda(), B.cuda(), T, TB)
    assert torch.allclose(C, torch.matmul(A, B).cuda())

