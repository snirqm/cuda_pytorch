import torch
import pytest
from parallel_mult import ParallelMatMul

# add timeout to all tests
T = 256
TB = 1


def test_big_tensor():
    size = 256
    A = torch.Tensor(size, size).fill_(1.0)
    B = torch.Tensor(size, size).fill_(-1.0)
    C = ParallelMatMul(A.cuda(), B.cuda(), T, TB)
    assert torch.allclose(
        C, torch.Tensor(size, size).fill_(-size).cuda()
    ), f"Result is not correct"


def test_2x1_conversion():
    A = abs(torch.rand(10, 10)).cuda()
    B = torch.rand(10).cuda()
    C = ParallelMatMul(A, B, T, TB)
    C_res = torch.matmul(A, B).cuda()
    assert (
        C_res.shape == C.shape
    ), f"Pytorch and CUDA results shapes differ: matmul {C_res.shape} vs cuda {C.shape}"
    assert torch.allclose(
        C, C_res
    ), f"Pytorch and CUDA results differ in indexes {torch.where(C != C_res)}"


def test_1x2_conversion():
    B = abs(torch.rand(10, 10)).cuda()
    A = torch.rand(10).cuda()
    C_res = torch.matmul(A, B).cuda()
    C = ParallelMatMul(A, B, T, TB)
    assert (
        C_res.shape == C.shape
    ), f"Pytorch and CUDA results shapes differ: {C_res.shape} vs {C.shape}"
    assert torch.allclose(
        C, C_res
    ), f"Pytorch and CUDA results differ in indexes {torch.where(C != C_res)}"


def test_basic_2x1():
    A, B = torch.Tensor([[1.0, 0.0], [0.0, 1.0]]), torch.Tensor([[1.0], [0.0]])
    C = ParallelMatMul(A.cuda(), B.cuda(), T, TB)
    C_res = torch.matmul(A, B).cuda()
    assert (
        C_res.shape == C.shape
    ), f"Pytorch and CUDA results shapes differ: {C_res.shape} vs {C.shape}"
    assert torch.allclose(
        C, C_res
    ), f"Pytorch and CUDA results differ in indexes {torch.where(C != C_res)}"


def test_basic_1x2():
    B, A = torch.Tensor([[1.0, 0.0], [0.0, 1.0]]), torch.Tensor([[1.0], [0.0]])
    with pytest.raises(Exception) as e:
        C = ParallelMatMul(A.cuda(), B.cuda(), T, TB)


def test_basic_2x2():
    A, B = torch.Tensor([[1.0, 0.0], [0.0, 1.0]]), torch.Tensor(
        [[1.0, 0.0], [0.0, 1.0]]
    )
    C = ParallelMatMul(A.cuda(), B.cuda(), T, TB)
    C_res = torch.matmul(A, B).cuda()
    assert (
        C_res.shape == C.shape
    ), f"Pytorch and CUDA results shapes differ: {C_res.shape} vs {C.shape}"
    assert torch.allclose(
        C, C_res
    ), f"Pytorch and CUDA results differ in indexes {torch.where(C != C_res)}"


def test_cpu_vs_cuda():
    A = abs(torch.rand(10, 10))
    B = abs(torch.rand(10, 10))
    C1 = ParallelMatMul(A.cuda(), B.cuda(), T, TB)
    C2 = torch.matmul(A, B)
    assert torch.allclose(C1, C2.cuda()), "CPU and CUDA results differ"
