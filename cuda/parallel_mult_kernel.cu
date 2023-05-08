#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdio.h>

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#define BLOCK_SIZE 32
using namespace torch;
template <typename scalar_type>
__device__ void fast_matmul_kernel(
    const PackedTensorAccessor<scalar_type, 2, RestrictPtrTraits> A,
    const PackedTensorAccessor<scalar_type, 2, RestrictPtrTraits> B,
    PackedTensorAccessor<scalar_type, 2, RestrictPtrTraits> C, const int row,
    const int col) {
  const size_t N = A.size(0);
  const size_t K = A.size(1);
  const size_t M = B.size(1);

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
    const PackedTensorAccessor<scalar_type, 2, RestrictPtrTraits> A,
    const PackedTensorAccessor<scalar_type, 2, RestrictPtrTraits> B,
    PackedTensorAccessor<scalar_type, 2, RestrictPtrTraits> C) {
  const size_t N = A.size(0);
  const size_t K = A.size(1);
  const size_t M = B.size(1);

  for (int i = 0; i < M; i += blockDim.x * gridDim.x) {
    for (int j = 0; j < N; j += blockDim.y * gridDim.y) {
      __syncthreads();
      int col = i + threadIdx.x + blockDim.x * blockIdx.x;
      int row = j + threadIdx.y + blockDim.y * blockIdx.y;
      fast_matmul_kernel<scalar_type>(A, B, C, row, col);
    }
  }
  return;
}

void invoke_matmul_kernel(const Tensor A, const Tensor B, Tensor C,
                          const unsigned int threadsPerBlock, const unsigned int threadBlocks) {
  const unsigned int blockSize = std::floor(std::sqrt(threadsPerBlock));
  const dim3 dimBlock(blockSize, blockSize);
  dim3 dimGrid;
  if (A.size(0) != 1 && B.size(1) != 1) {
    const unsigned int gridSize = std::floor(std::sqrt(threadBlocks));
    dimGrid = {gridSize, gridSize};
  } else if (A.size(0) == 1 && B.size(1) != 1) {
    dimGrid = {threadBlocks, 1};
  } else if (A.size(0) != 1 && B.size(1) == 1) {
    dimGrid = {1, threadBlocks};
  } else {
    dimGrid = {1, 1};
  }
  AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "ParallelMatMul", [&] {
    batch_fast_matmul_kernel<scalar_t><<<dimGrid, dimBlock>>>(
        A.packed_accessor64<scalar_t, 2, RestrictPtrTraits>(),
        B.packed_accessor64<scalar_t, 2, RestrictPtrTraits>(),
        C.packed_accessor64<scalar_t, 2, RestrictPtrTraits>());
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
Tensor cuda_matmult(Tensor A, Tensor B, Tensor C, const unsigned int T, const unsigned int TB) {
  CHECK_INPUTS(A, B, C);
  TORCH_CHECK(A.dim() <= 2 && B.dim() <= 2,
              "Input tensors must be 1 or 2 dimensional");
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2,
              "Input tensors must be 2 dimensional");
  TORCH_CHECK(A.size(1) == B.size(0) && C.size(0) == A.size(0) &&
                  C.size(1) == B.size(1),
              "Input tensors must be compatible for matrix multiplication");
  invoke_matmul_kernel(A, B, C, T, TB);
  return C;
}

__host__ void calc_cpu_matmult(const Tensor A, const Tensor B, Tensor C) {
  const size_t N = A.size(0);
  const size_t K = A.size(1);
  const size_t M = B.size(1);
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < M; j++) {
      for (size_t k = 0; k < K; k++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
  return;
}

Tensor cpu_matmult(Tensor A, Tensor B, Tensor C) {
  // Check input tensor dimensions
  CHECK_CONTIGUOUS(A);
  CHECK_CONTIGUOUS(B);
  CHECK_CONTIGUOUS(C);
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2,
              "Input tensors must be 2 dimensional");
  TORCH_CHECK(A.size(1) == B.size(0),
              "Input tensors must be compatible for matrix multiplication");
  TORCH_CHECK(C.size(0) == A.size(0),
              "Input tensors must be compatible for matrix multiplication");
  TORCH_CHECK(C.size(1) == B.size(1),
              "Input tensors must be compatible for matrix multiplication");
  calc_cpu_matmult(A, B, C);
  return C;
}
