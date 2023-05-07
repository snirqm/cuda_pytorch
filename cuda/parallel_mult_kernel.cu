#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdio.h>

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#define BLOCK_SIZE 32

template <typename scalar_type>
__device__ void fast_matmul_kernel(
    const torch::PackedTensorAccessor<scalar_type, 2, torch::RestrictPtrTraits>
        A,
    const torch::PackedTensorAccessor<scalar_type, 2, torch::RestrictPtrTraits>
        B,
    torch::PackedTensorAccessor<scalar_type, 2, torch::RestrictPtrTraits> C,
    const int M, const int N, const int K, const int row, const int col) {

  __shared__ scalar_type As[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ scalar_type Bs[BLOCK_SIZE][BLOCK_SIZE];

  scalar_type acc = 0;

  const int num_tiles = (K + blockDim.x - 1) / blockDim.x;

  for (int t = 0; t < num_tiles; t++) {
    if (row < N && t * blockDim.x + threadIdx.x < K) {
      As[threadIdx.y][threadIdx.x] = A[row][t * blockDim.x + threadIdx.x];
    } else {
      As[threadIdx.y][threadIdx.x] = 0;
    }

    if (t * blockDim.x + threadIdx.y < K && col < M) {
      Bs[threadIdx.y][threadIdx.x] = B[t * blockDim.x + threadIdx.y][col];
    } else {
      Bs[threadIdx.y][threadIdx.x] = 0;
    }

    __syncthreads();

    for (int k = 0; k < blockDim.x && k < K; k++) {
      acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < N && col < M) {
    C[row][col] = acc;
  }
}
template <typename scalar_type>
__global__ void batch_fast_matmul_kernel(
    const torch::PackedTensorAccessor<scalar_type, 2, torch::RestrictPtrTraits>
        A,
    const torch::PackedTensorAccessor<scalar_type, 2, torch::RestrictPtrTraits>
        B,
    torch::PackedTensorAccessor<scalar_type, 2, torch::RestrictPtrTraits> C,
    const int M, const int N, const int K) {
  for (int i = 0; i < M; i += blockDim.x * gridDim.x) {
    for (int j = 0; j < N; j += blockDim.y * gridDim.y) {
      __syncthreads();
      int col = i + threadIdx.x + blockDim.x * blockIdx.x;
      int row = j + threadIdx.y + blockDim.y * blockIdx.y;
      fast_matmul_kernel<scalar_type>(A, B, C, M, N, K, row, col);
    }
  }
  return;
}

void invoke_matmul_kernel(const torch::Tensor A, const torch::Tensor B,
                          torch::Tensor C, const size_t rows,
                          const size_t inner, const size_t cols,
                          const int threadsPerBlock, const int threadBlocks) {

  const int blockRows = std::floor(std::sqrt(threadsPerBlock));
  const dim3 dimBlock(blockRows, blockRows);
  const int gridRows = std::floor(std::sqrt(threadBlocks));
  const dim3 dimGrid(gridRows, gridRows);
  AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "ParallelMatMul", [&] {
    batch_fast_matmul_kernel<scalar_t><<<dimGrid, dimBlock>>>(
        A.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
        B.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
        C.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
        cols, rows, inner);
  });
  cudaDeviceSynchronize();
}
// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)
#define CHECK_INPUTS(x, y, z)                                                  \
  CHECK_INPUT(x);                                                              \
  CHECK_INPUT(y);                                                              \
  CHECK_INPUT(z)
torch::Tensor cuda_matmult(torch::Tensor A, torch::Tensor B, torch::Tensor C,
                           int T, int TB) {
  CHECK_INPUTS(A, B, C);
  TORCH_CHECK(A.dim() <= 2 && B.dim() <= 2,
              "Input tensors must be 1 or 2 dimensional");
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2,
              "Input tensors must be 2 dimensional");
  TORCH_CHECK(A.size(1) == B.size(0) && C.size(0) == A.size(0) &&
                  C.size(1) == B.size(1),
              "Input tensors must be compatible for matrix multiplication");
  invoke_matmul_kernel(A, B, C, A.size(0), A.size(1), B.size(1), T, TB);
  return C;
}

__host__ void calc_cpu_matmult(const torch::Tensor A, const torch::Tensor B,
                               torch::Tensor C, const size_t rows,
                               const size_t inner, const size_t cols) {
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      for (size_t k = 0; k < inner; k++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
  return;
}

torch::Tensor cpu_matmult(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
  // Check input tensor dimensions
  CHECK_CONTIGUOUS(A);
  CHECK_CONTIGUOUS(B);
  CHECK_CONTIGUOUS(C);
  size_t rows, inner, cols;
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2,
              "Input tensors must be 2 dimensional");
  TORCH_CHECK(A.size(1) == B.size(0),
              "Input tensors must be compatible for matrix multiplication");
  TORCH_CHECK(C.size(0) == A.size(0),
              "Input tensors must be compatible for matrix multiplication");
  TORCH_CHECK(C.size(1) == B.size(1),
              "Input tensors must be compatible for matrix multiplication");
  rows = A.size(0);
  inner = A.size(1);
  cols = B.size(1);
  calc_cpu_matmult(A, B, C, rows, inner, cols);
  return C;
}
