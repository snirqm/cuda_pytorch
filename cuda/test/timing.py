import torch
from datetime import datetime
from parallel_mult import ParallelMatMul
def test_increase_TB_reduce_runtime():
    T = 256
    TBs = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    times = []
    n = 256
    A =  torch.rand(n, n, dtype=torch.float32, device='cuda')
    B =  torch.rand(n, n, dtype=torch.float32, device='cuda')
    for TB in TBs:
        start = datetime.now()
        for _ in range(100):
            ParallelMatMul(A, B, T, TB)
        end = datetime.now()
        times.append((end - start).total_seconds())

    for i in range(len(times[:-1])):
        print(f"TB={TBs[i]}, time={times[i]}, speedup={times[i]/times[-1]}")
    
