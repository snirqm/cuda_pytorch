import torch
import pytest
from parallel_mult import ParallelMatMul
from check_tensors import assert_tensors

# add timeout to all tests
T = 256
TB = 1


def test_2x1_bug():
    T = 256
    TB = 32
    A = torch.Tensor([[1.0, 1.0], [1.0, 1.0]])
    B = torch.ones(2)
    C = ParallelMatMul(A.cuda(), B.cuda(), T, TB)
    C2 = torch.matmul(A, B).cuda()
    assert_tensors(C, C2)

@pytest.mark.parametrize("n", range(2, 100))
def test_2x1_rising_n(n):
    A, B = abs(torch.rand(10, n)), abs(torch.rand(n, 1))
    C = ParallelMatMul(A.cuda(), B.cuda(), T, TB)
    C2 = torch.matmul(A, B).cuda()
    assert_tensors(C, C2)

@pytest.mark.parametrize(
    "tensors",
    [
        (torch.Tensor([[1.0, 0.0], [0.0, 1.0]]), torch.Tensor([[1.0], [0.0]])),
        (abs(torch.rand(10, 10)), abs(torch.rand(10, 1))),
        (abs(torch.rand(10, 10)), abs(torch.rand(10))),
        (abs(torch.rand(10, 54)), abs(torch.rand(54, 1))),
        (abs(torch.rand(10, 54)), abs(torch.rand(54))),
        (abs(torch.rand(10, 100)), abs(torch.rand(100, 1))),
        (abs(torch.rand(10, 74)), abs(torch.rand(74, 1))),
    ],
)
def test_many_2x1(tensors):
    A, B = tensors
    C = ParallelMatMul(A.cuda(), B.cuda(), T, TB)
    C2 = torch.matmul(A, B).cuda()
    assert_tensors(C, C2)

@pytest.mark.parametrize("TB", [8, 16, 32, 64, 128, 256])
@pytest.mark.parametrize("n", [2, 4, 8, 16, 32, 64])
def test_many_TB_2x1(TB, n):
    TB = TB
    A = abs(torch.rand(n, n))
    B = abs(torch.rand(n))
    C1 = ParallelMatMul(A.cuda(), B.cuda(), T, TB)
    C2 = torch.matmul(A, B).cuda()
    assert_tensors(C1, C2)
    
def test_single_TB_2x1():
    TB = 8
    n = 2
    A = abs(torch.rand(n, n))
    B = abs(torch.rand(n))
    C1 = ParallelMatMul(A.cuda(), B.cuda(), T, TB)
    C2 = torch.matmul(A, B).cuda()
    assert_tensors(C1, C2)


def test_2x1_bug():
    n =12
    A, B = abs(torch.rand(10, n)), abs(torch.rand(n, 1))
    C = ParallelMatMul(A.cuda(), B.cuda(), T, TB)
    C2 = torch.matmul(A, B).cuda()
    assert_tensors(C, C2)
