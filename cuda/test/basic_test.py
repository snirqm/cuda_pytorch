import torch
import pytest
from parallel_mult import ParallelMatMul
#add timeout to all tests
pytestmark = pytest.mark.timeout(1)
T = 256
TB = 1

def test_big_tensor():
    size = 256
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
    C2 = torch.matmul(A, B)
    assert torch.allclose(C1, C2.cuda())



@pytest.mark.timeout(0.75)
@pytest.mark.parametrize("TB", [1, 2, 4, 8, 16, 32, 64, 128, 256])
def test_many_TB_2x2(TB):
    n = 10
    A = torch.rand(n, n)
    B = torch.rand(n, n)
    C1 = ParallelMatMul(A.cuda(), B.cuda(), T, TB)
    C2 = torch.matmul(A, B)
    assert torch.allclose(C1, C2.cuda())

@pytest.mark.timeout(10)
@pytest.mark.parametrize("TB", [8, 16, 32, 64, 128, 256])
@pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 32, 64]) 
def test_many_TB_n_2x2(n, TB):
    A = torch.rand(n, n)
    B = torch.rand(n, n)
    C1 = ParallelMatMul(A.cuda(), B.cuda(), T, TB)
    C2 = torch.matmul(A, B)
    assert torch.allclose(C1, C2.cuda())

@pytest.mark.timeout(0.75)
@pytest.mark.parametrize("TB", [1, 2, 4, 8, 16, 32, 64, 128, 256])
def test_TB_2x2_ID(TB):
     A = torch.Tensor([[1.0, 0.0], [0.0, 1.0]])
     B = torch.Tensor([[1.0, 0.0], [0.0, 1.0]])
     C1 = ParallelMatMul(A.cuda(), B.cuda(), T, TB)
     C2 = torch.matmul(A, B)
     assert torch.allclose(C1, C2.cuda())

@pytest.mark.timeout(0.75)
def test_TB_1_2x2_BIG():
    A = torch.rand(256, 256)
    B = torch.rand(256, 256)
    T = 256
    TB = 1
    C1 = ParallelMatMul(A.cuda(), B.cuda(), T, TB)
    C2 = torch.matmul(A, B)
    assert torch.allclose(C1, C2.cuda())

pytest.mark.timeout(0.75)
@pytest.mark.parametrize("TB", [16, 32, 64, 128, 256])
def test_TB_2x2_BIG(TB):
    A = torch.rand(256, 256)
    B = torch.rand(256, 256)
    T = 256
    C1 = ParallelMatMul(A.cuda(), B.cuda(), T, TB)
    C2 = torch.matmul(A, B)
    assert torch.allclose(C1, C2.cuda())


@pytest.mark.timeout(10)
@pytest.mark.parametrize("TB", [8, 16, 32, 64, 128, 256])
@pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 32, 64])
def test_many_TB_2x1(TB, n):
    TB = TB
    A = torch.rand(n, n)
    B = torch.rand(n)
    C1 = ParallelMatMul(A.cuda(), B.cuda(), T, TB)
    C2 = torch.matmul(A, B).cuda()
    # print where C1 != C2
    for i in range(n):
        if torch.allclose(C1[i], C2[i]) == False:
            print(i, C1[i], C2[i])
    assert torch.allclose(C1, C2)



