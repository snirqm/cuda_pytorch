import torch
import pytest
from parallel_mult import ParallelMatMul
from check_tensors import assert_tensors

# add timeout to all tests
pytestmark = pytest.mark.timeout(10)
T = 256
TB = 1


def test_2x2_bug():
    T = 256
    TB = 256
    A = torch.rand(2, 2, dtype=torch.float32, device="cuda")
    B = torch.rand(2, 2, dtype=torch.float32, device="cuda")
    C = ParallelMatMul(A, B, T, TB)
    C2 = torch.matmul(A, B).cuda()
    assert torch.allclose(C, C2, atol=1e-4)


@pytest.mark.parametrize(
    "tensors",
    [
        (
            torch.Tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.Tensor([[1.0, 0.0], [0.0, 1.0]]),
        ),
        (abs(torch.rand(10, 10)), abs(torch.rand(10, 10))),
        (abs(torch.rand(5, 10)), abs(torch.rand(10, 7))),
    ],
)
def test_basic_2x2(tensors):
    A, B = tensors
    C = ParallelMatMul(A.cuda(), B.cuda(), T, TB)
    assert torch.allclose(C, torch.matmul(A, B).cuda())


@pytest.mark.parametrize("TB", [1, 2, 4, 8, 16, 32, 64, 128, 256])
def test_many_TB_2x2(TB):
    n = 10
    A = abs(torch.rand(n, n))
    B = abs(torch.rand(n, n))
    C1 = ParallelMatMul(A.cuda(), B.cuda(), T, TB)
    C2 = torch.matmul(A, B)
    assert torch.allclose(
        C1, C2.cuda()
    ), f"Pytorch and CUDA results differ in indexes {torch.where(C1 != C2.cuda())}"


def test_TB_1_2x2_BIG():
    A = abs(torch.rand(256, 256))
    B = abs(torch.rand(256, 256))
    T = 256
    TB = 1
    C1 = ParallelMatMul(A.cuda(), B.cuda(), T, TB)
    C2 = torch.matmul(A, B)
    assert torch.allclose(
        C1, C2.cuda()
    ), f"Pytorch and CUDA results differ in indexes {torch.where(C1 != C2.cuda())}"


@pytest.mark.parametrize("TB", [8, 16, 32, 64, 128, 256])
@pytest.mark.parametrize("n", [2, 4, 8, 16, 32, 64])
def test_many_TB_n_2x2(n, TB):
    A = abs(torch.rand(n, n))
    B = abs(torch.rand(n, n))
    C1 = ParallelMatMul(A.cuda(), B.cuda(), T, TB)
    C2 = torch.matmul(A, B)
    assert torch.allclose(
        C1, C2.cuda()
    ), f"Pytorch and CUDA results differ in indexes {torch.where(C1 != C2.cuda())}"


@pytest.mark.parametrize("TB", [16, 32, 64, 128, 256])
def test_TB_2x2_BIG(TB):
    A = abs(torch.rand(256, 256))
    B = abs(torch.rand(256, 256))
    T = 256
    C1 = ParallelMatMul(A.cuda(), B.cuda(), T, TB)
    C2 = torch.matmul(A, B)
    assert torch.allclose(
        C1, C2.cuda()
    ), f"Pytorch and CUDA results differ in indexes {torch.where(C1 != C2.cuda())}"


@pytest.mark.parametrize("TB", [1, 2, 4, 8, 16, 32, 64, 128, 256])
def test_TB_2x2_ID(TB):
    A = torch.Tensor([[1.0, 0.0], [0.0, 1.0]])
    B = torch.Tensor([[1.0, 0.0], [0.0, 1.0]])
    C1 = ParallelMatMul(A.cuda(), B.cuda(), T, TB)
    C2 = torch.matmul(A, B)
    assert torch.allclose(
        C1, C2.cuda()
    ), f"Pytorch and CUDA results differ in indexes {torch.where(C1 != C2.cuda())}"

@pytest.mark.parametrize("n", range(2, 100))
def test_rising_n(n):
    A, B = abs(torch.rand(10, n)), abs(torch.rand(n, 10))
    C = ParallelMatMul(A.cuda(), B.cuda(), T, TB)
    C2 = torch.matmul(A, B).cuda()
    assert_tensors(C, C2)
