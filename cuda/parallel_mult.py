import torch
from numba import cuda
import parallel_mult_cuda


class ParallelMatMulCls(object):
    def __init__(self, A: torch.Tensor, B: torch.Tensor):
        self.A, self.B = A, B
        self.X, self.Y = None, None
        self.should_transpose = False

    def validate_device(self):
        assert self.A.is_cuda == self.B.is_cuda, "A and B must be on the same device"
        self.is_cuda = self.A.is_cuda
        return self

    def validate_dimensions(self):
        if len(self.A.shape) == 1 and len(self.B.shape) == 1:
            assert (
                self.A.shape[0] == self.B.shape[0]
            ), "A and B must be compatible for matrix multiplication"
        elif len(self.A.shape) == 1:
            assert (
                self.A.shape[0] == self.B.shape[1]
            ), "A and B must be compatible for matrix multiplication"
        elif len(self.B.shape) == 1:
            assert (
                self.A.shape[1] == self.B.shape[0]
            ), "A and B must be compatible for matrix multiplication"
        elif len(self.A.shape) == 2 and len(self.B.shape) == 2:
            assert (
                self.A.shape[1] == self.B.shape[0]
            ), "A and B must be compatible for matrix multiplication"
        else:
            raise ValueError("A and B must be vectors or matrixes")
        return self

    def load_matrixes(self):
        self.should_transpose = False
        if (len(self.A.shape) == 2 and self.A.shape[0] == 1) and (
            len(self.B.shape) == 2 and self.B.shape[1] == 1
        ):
            self.X = self.A.squeeze(0)
            self.Y = self.B.squeeze(1)
        elif len(self.A.shape) == 2 and self.A.shape[0] == 1:
            self.X = self.B.transpose(0, 1)
            self.Y = self.A.squeeze(0)
            self.should_transpose = True
        elif len(self.B.shape) == 2 and self.B.shape[1] == 1:
            self.X = self.A
            self.Y = self.B.squeeze(1)
        elif len(self.A.shape) == 1 and len(self.B.shape) != 1:
            self.X = self.B.transpose(0, 1)
            self.Y = self.A
        else:
            self.X = self.A
            self.Y = self.B
            self.should_transpose = False
        return self

    @staticmethod
    def allocate_matrix(n: int, m: int, dtype: torch.dtype, cuda: bool):
        return torch.zeros((n, m), dtype=dtype, device="cuda" if cuda else "cpu")

    def matmul(self, T: int, TB: int):
        self.C = self.allocate_matrix(
            self.X.shape[0], self.Y.shape[1] if len(self.Y.shape) != 1 else 1, self.X.dtype, self.is_cuda
        )
        if self.is_cuda:
            parallel_mult_cuda.cuda_matmult(self.X, self.Y, self.C, T, TB)
        else:
            parallel_mult_cuda.cpu_matmult(self.X, self.Y, self.C)
        return self

    def make_contiguous(self):
        if not self.X.is_contiguous():
            self.X = self.X.contiguous()
        if not self.Y.is_contiguous():
            self.Y = self.Y.contiguous()
        return self
    def transpose(self):
        if self.should_transpose:
            self.C = self.C.transpose(0, 1)
        return self

    def squeeze(self):
        if (len(self.A.shape) == 1 or len(self.B.shape) == 1) and len(self.C.shape) == 2:
            self.C =  self.C.squeeze(1)
        return self
    def get_result(self):
        return self.C


def ParallelMatMul(A: torch.Tensor, B: torch.Tensor, T: int, TB: int):
    return (
        ParallelMatMulCls(A, B)
        .validate_device()
        .validate_dimensions()
        .load_matrixes()
        .make_contiguous()
        .matmul(T, TB)
        .transpose()
        .squeeze()
        .get_result()
    )
