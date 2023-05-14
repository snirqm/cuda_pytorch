import torch
import time
import matplotlib.pyplot as plt
import os, shutil
from parallel_mult import ParallelMatMul


def main():
    n_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    TB_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    T = 256
    TB = 1
    num_iters = 100
    if os.path.exists("report"):
        shutil.rmtree("report")
    os.mkdir("report")

    timings = {
        "matmul": {"ParallelMatMul": []},
        "vector_matmul": {"ParallelMatMul": []},
    }
    for n in n_list:
        A = torch.rand(n, n).cuda()
        B = torch.rand(n).cuda()
        start_time = time.time()
        for i in range(num_iters):
            ParallelMatMul(A, B, T, TB)
        end_time = time.time()
        vector_matmul_time = end_time - start_time
        timings["vector_matmul"]["ParallelMatMul"].append(vector_matmul_time)

    plt.figure()
    plt.plot(
        n_list, timings["vector_matmul"]["ParallelMatMul"], "-.", label="ParallelMatMul"
    )
    plt.xlabel("n")
    plt.ylabel("Time (seconds)")
    plt.title("Matrix x Vector Multiplication Method (T = 256, TB = 1)")
    plt.legend()
    plt.savefig("report/matvec.png", dpi=300)
    print("created report/matvec.png")

    for n in n_list:
        A = torch.rand(n, n).cuda()
        B = torch.rand(n, n).cuda()
        start_time = time.time()
        for i in range(num_iters):
            ParallelMatMul(A, B, T, TB)
        end_time = time.time()
        matmul_time = end_time - start_time
        timings["matmul"]["ParallelMatMul"].append(matmul_time)

    plt.figure()
    plt.plot(n_list, timings["matmul"]["ParallelMatMul"], "-.", label="ParallelMatMul")
    plt.xlabel("n")
    plt.ylabel("Time (seconds)")
    plt.title("Matrix x Matrix Multiplication Method (T = 256, TB = 1)")
    plt.legend()
    plt.savefig("report/matmat.png", dpi=300)
    print("created report/matmat.png")
    
    timings = {
        TB: {"matmul": {"ParallelMatMul": []}, "vector_mat_mul": {"ParallelMatMul": []}}
        for TB in TB_list
    }
    
    for TB in TB_list:
        for n in n_list:
            A = torch.rand(n, n).cuda()
            Bnn = torch.rand(n, n).cuda()
            start_time = time.time()
            for i in range(num_iters):
                ParallelMatMul(A, Bnn, T, TB)
            end_time = time.time()
            matmul_time = end_time - start_time
            timings[TB]["matmul"]["ParallelMatMul"].append(matmul_time)
            
    plt.figure()
    for TB in TB_list:
        plt.plot(
            n_list,
            timings[TB]["matmul"]["ParallelMatMul"],
            "-.",
            label="TB = {} ParallelMatMul".format(TB),
        )
    plt.xlabel("n")
    plt.ylabel("Time (seconds)")
    plt.title("Matrix x Matrix Multiplication Method (T = 256)")
    plt.legend()
    plt.savefig("report/matmat_TB.png", dpi=300)
    print("created report/matmat_TB.png")
    
    
    
    for TB in TB_list:
        for n in n_list:
            A = torch.rand(n, n).cuda()
            Bn = torch.rand(n).cuda()
            start_time = time.time()
            for i in range(num_iters):
                ParallelMatMul(A, Bn, T, TB)
            end_time = time.time()
            vector_matmul_time = end_time - start_time
            timings[TB]["vector_mat_mul"]["ParallelMatMul"].append(vector_matmul_time)


    plt.figure()
    for TB in TB_list:
        plt.plot(
            n_list,
            timings[TB]["vector_mat_mul"]["ParallelMatMul"],
            "-.",
            label="TB = {} ParallelMatMul".format(TB),
        )
    plt.xlabel("n")
    plt.ylabel("Time (seconds)")
    plt.title("Matrix x Vector Multiplication Method (T = 256)")
    plt.legend()
    plt.savefig("report/matvec_TB.png", dpi=300)
    print("created report/matvec_TB.png")


if __name__ == "__main__":
    exit(main())
