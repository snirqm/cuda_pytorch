import torch
import pytest
from parallel_mult import ParallelMatMul

# add timeout to all tests

T = 256
TB = 1


def test_1x1_bug():
    T = 1
    TB = 1
    A = torch.ones(2)
    B = torch.ones(2) + 1
    C = ParallelMatMul(A.cuda(), B.cuda(), T, TB)
    assert torch.allclose(C, torch.matmul(A, B).cuda())


@pytest.mark.parametrize(
    "tensors",
    [
        (torch.Tensor([[1.0, 0.0]]), torch.Tensor([[1.0], [0.0]])),
        (abs(torch.rand(1, 100)), abs(torch.rand(100, 1))),
        (abs(torch.rand(1, 10)), abs(torch.rand(10, 1))),
        (abs(torch.rand(10)), abs(torch.rand(10))),
        (abs(torch.rand(1, 74)), abs(torch.rand(74, 1))),
    ],
)
def test_basic_1x1(tensors):
    A, B = tensors
    C = ParallelMatMul(A.cuda(), B.cuda(), T, TB)
    assert torch.allclose(C, torch.matmul(A, B).cuda())


@pytest.mark.parametrize("TB", [1, 2, 3, 4, 5 , 8, 16, 32, 64, 128, 256])
def test_many_TB_n_1x1(TB):
    n = 1
    A = abs(torch.rand(n))
    B = abs(torch.rand(n))
    C1 = ParallelMatMul(A.cuda(), B.cuda(), T, TB)
    C2 = torch.matmul(A, B)
    assert torch.allclose(C1, C2.cuda())
