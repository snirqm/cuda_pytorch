import torch
import pytest
from parallel_mult import ParallelMatMul
from check_tensors import assert_tensors
# add timeout to all tests
T = 256
TB = 1


@pytest.mark.parametrize(
    "tensors",
    [
        (abs(torch.rand(1, 10)), abs(torch.rand(10, 10))),
        (abs(torch.rand(1, 10)), abs(torch.rand((10, 10)))),
        (abs(torch.rand(1, 54)), abs(torch.rand(54, 10))),
        (abs(torch.rand(1, 54)), abs(torch.rand((54, 10)))),
        (abs(torch.rand(1, 100)), abs(torch.rand(100, 10))),
        (abs(torch.rand(1, 74)), abs(torch.rand(74, 10))),
    ],
)
def test_many_1x2(tensors):
    A, B = tensors
    C = ParallelMatMul(A.cuda(), B.cuda(), T, TB)
    C2 = torch.matmul(A, B).cuda()
    assert_tensors(C, C2)

@pytest.mark.parametrize("TB", [8, 16, 32, 64, 128, 256])
@pytest.mark.parametrize("n", [2, 4, 8, 16, 32, 64])
def test_many_TB_1x2(TB, n):
    B = abs(torch.rand(n, n))
    A = abs(torch.rand(n))
    C2 = torch.matmul(A, B).cuda()
    C1 = ParallelMatMul(A.cuda(), B.cuda(), T, TB)
    assert_tensors(C1, C2)
    
    

@pytest.mark.parametrize("n", range(2, 100))
def test_1x2_rising_n(n):
    A, B = abs(torch.rand(1, n)), abs(torch.rand(n, 10))
    C = ParallelMatMul(A.cuda(), B.cuda(), T, TB)
    C2 = torch.matmul(A, B).cuda()
    assert_tensors(C, C2)
