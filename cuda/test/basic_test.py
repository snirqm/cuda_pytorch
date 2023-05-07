import torch
import pytest
from parallel_mult import ParallelMatMul
T = 1024
TB = 500 
def test_big_tensor():
    size = 1000
    A = torch.Tensor(size, size).fill_(1.0)
    B = torch.Tensor(size, size).fill_(-1.0)
    C = ParallelMatMul(A.cuda(), B.cuda(), T, TB)
    assert torch.allclose(C, torch.Tensor(size, size).fill_(-size).cuda())


def test_2x1_conversion():
    A = torch.rand(10, 10).cuda()
    B = torch.rand(10).cuda()
    C = ParallelMatMul(A, B, T, TB)
    C_res = torch.matmul(A, B).cuda()
    print(C_res.shape)
    assert torch.allclose(C, C_res)

@pytest.mark.parametrize("tensors", [
    (torch.Tensor([[1.0, 0.0], [0.0, 1.0]]), torch.Tensor([[1.0], [0.0]])),
    (torch.rand(10, 10), torch.rand(10, 1)),
    (torch.rand(10, 10), torch.rand(10)),
    (torch.rand(10, 54), torch.rand(54, 1)),
    (torch.rand(10, 54), torch.rand(54)),
    (torch.rand(10, 100), torch.rand(100, 1)),
    (torch.rand(10, 74), torch.rand(74, 1))
    ])
def test_basic_2x1(tensors):
    A, B = tensors
    C = ParallelMatMul(A.cuda(), B.cuda(), T, TB)
    C_res = torch.matmul(A, B).cuda()
    print(C_res.shape)
    assert torch.allclose(C, C_res)


@pytest.mark.parametrize("tensors", [
    (torch.Tensor([[1.0, 0.0]]), torch.Tensor([[1.0, 0.0], [1.0, 0.0]])),
    (torch.rand(1, 10), torch.rand(10, 10)),
    (torch.rand(1, 100), torch.rand(100, 100)),
    (torch.rand(1, 74), torch.rand(74, 74)),
    (torch.rand(74), torch.rand(74, 74))
    ])
def test_basic_1x2(tensors):
    A, B = tensors
    C = ParallelMatMul(A.cuda(), B.cuda(), T, TB)
    assert torch.allclose(C, torch.matmul(A, B).cuda())

@pytest.mark.parametrize("tensors", [
    (torch.Tensor([[1.0, 0.0]]), torch.Tensor([[1.0], [0.0]])),
    (torch.rand(1, 100), torch.rand(100, 1)),
    (torch.rand(1, 10), torch.rand(10, 1)),
    (torch.rand(10), torch.rand(10)),
    (torch.rand(1, 74), torch.rand(74, 1))
    ])
def test_basic_1x1(tensors):
    A, B = tensors
    C = ParallelMatMul(A.cuda(), B.cuda(), T, TB)
    assert torch.allclose(C, torch.matmul(A, B).cuda())


@pytest.mark.parametrize("tensors", [
    (torch.Tensor([[1.0, 0.0], [0.0, 1.0]]), torch.Tensor([[1.0, 0.0], [0.0, 1.0]])),
    (torch.rand(10, 10), torch.rand(10, 10)),
    (torch.rand(5, 10), torch.rand(10, 7)),
     ])
def test_basic_2x2(tensors):
    A, B = tensors
    C = ParallelMatMul(A.cuda(), B.cuda(), T, TB)
    assert torch.allclose(C, torch.matmul(A, B).cuda())


def test_cpu_vs_cuda():
    A = torch.rand(10, 10)
    B = torch.rand(10, 10)
    C1 = ParallelMatMul(A.cuda(), B.cuda(), T, TB)
    C2 = ParallelMatMul(A, B, T, TB)
    assert torch.allclose(C1, C2.cuda())
