import time
import pytest

import torch

from parallel_mult import ParallelMatMul



@pytest.mark.timeout(100)
def test_assign():
    n_list = [2 ** i for i in range(1, 8)]
    TB_list = [2 ** i for i in range(1, 8)]
    for TB in TB_list:
        for n in n_list:
            A = torch.rand(n, n).cuda()
            B1 = torch.rand(n, n).cuda()
            B2 = torch.rand(n).cuda()
            for i in range(100):
                ParallelMatMul(A, B1, 256, TB)
            for i in range(100):
                ParallelMatMul(A, B2, 256, TB)