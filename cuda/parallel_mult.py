import torch
from numba import cuda
import parallel_mult_cuda
def ParallelMatMul(_A: torch.Tensor, _B: torch.Tensor, T: int, TB: int):
    A, B = unsqueeze(_A, _B)

    if A.is_cuda and B.is_cuda:
        C = allocate_matrix(A.shape, B.shape, A.dtype).cuda()
        parallel_mult_cuda.cuda_matmult(A, B, C, T, TB)
        return squeeze(C, _A, _B)
    elif not A.is_cuda and not B.is_cuda:
        C = allocate_matrix(A.shape, B.shape, A.dtype)
        parallel_mult_cuda.cpu_matmult(A, B, C)
        return squeeze(C, _A, _B)
    else:
        raise ValueError("A and B must be on the same device")

def squeeze(C, _A, _B):
    if len(C.shape) == 2:
        if len(_A.shape) == 1 and len(_B.shape) == 1:
            return C[0, 0]
        if len(_A.shape) == 1:
            return C.squeeze(0)
        if len(_B.shape) == 1:
            return C.squeeze(1)
    return C



def unsqueeze(A, B):
    if len(A.shape) == 1:
        A = A.unsqueeze(0)
    if len(B.shape) == 1:
        B = B.unsqueeze(1)
    return A, B


def allocate_matrix(shapeA, shapeB, dtype):
    if len(shapeA) == 1:
        shapeA = (1, shapeA[0])
    if len(shapeB) == 1:
        shapeB = (shapeB[0], 1)
    return torch.zeros((shapeA[0], shapeB[1]), dtype=dtype)
