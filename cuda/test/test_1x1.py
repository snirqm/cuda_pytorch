import torch
from parallel_mult import ParallelMatMul
def test_1x1_bug():
    T = 32
    TB = 1
    A = torch.ones(2)
    B = torch.ones(2)
    C = ParallelMatMul(A.cuda(), B.cuda(), T, TB)
    assert torch.allclose(C, torch.matmul(A, B).cuda())

