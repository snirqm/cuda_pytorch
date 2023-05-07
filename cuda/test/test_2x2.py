import torch
from parallel_mult import ParallelMatMul
def test_2x2_bug():
    T = 256
    TB = 256
    A =  torch.rand(2, 2, dtype=torch.float32, device='cuda') 
    B =  torch.rand(2, 2, dtype=torch.float32, device='cuda')
    C = ParallelMatMul(A, B, T, TB)
    C2 = torch.matmul(A, B).cuda()
    assert torch.allclose(C, C2, atol=1e-4)

    
