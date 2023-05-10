#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>

#define BLOCK_SIZE 32

template <typename scalar_type>
__global__ void batch_fast_matmul_kernel_2x1(
    const torch::PackedTensorAccessor<scalar_type, 2, torch::RestrictPtrTraits>
        A,
    const torch::PackedTensorAccessor<scalar_type, 2, torch::RestrictPtrTraits>
        B,
    torch::PackedTensorAccessor<scalar_type, 2, torch::RestrictPtrTraits> C) {
  const size_t N = A.size(0);
  const size_t M = B.size(1);
  const size_t K = A.size(1);
  printf("N = %d, M = %d, K = %d\n", N, M, K);
  __shared__ scalar_type As[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ scalar_type Bs[BLOCK_SIZE];
  for (unsigned int i = 0; i < N; i += blockDim.y * gridDim.y) {
    unsigned int idx = i + threadIdx.y + blockDim.y * blockIdx.y;
    scalar_type acc = 0;
    const unsigned int num_tiles = std::min((K + blockDim.y - 1) / blockDim.y,
                                            (K + BLOCK_SIZE - 1) / BLOCK_SIZE);
    // The loop accumulates the result for each tile in `acc`
    for (unsigned int t = 0; t < num_tiles; t++) {
      if (idx < N && t * blockDim.x + threadIdx.x < K) {
        // Fetch data from A to shared memory
        As[threadIdx.y][threadIdx.x] = A[idx][t * blockDim.x + threadIdx.x];
      } else {
        As[threadIdx.y][threadIdx.x] = 0;
      }
      // Fetch data from B to shared memory
      if (t * blockDim.y + threadIdx.y < K) {
        Bs[threadIdx.y] = B[t * blockDim.y + threadIdx.y][0];
      } else {
        Bs[threadIdx.y] = 0;
      }
      __syncthreads();
      // Accumulate the result
      for (unsigned int k = 0; k < blockDim.x && k < K; k++) {
        acc += As[threadIdx.y][k] * Bs[k];
      }
      __syncthreads();
    }
    if (idx < N) {
      C[idx][0] = acc;
    }
  }
  return;
}

template <typename scalar_type>
__global__ void batch_fast_matmul_kernel_2x2(
    const torch::PackedTensorAccessor<scalar_type, 2, torch::RestrictPtrTraits>
        A,
    const torch::PackedTensorAccessor<scalar_type, 2, torch::RestrictPtrTraits>
        B,
    torch::PackedTensorAccessor<scalar_type, 2, torch::RestrictPtrTraits> C) {
  const size_t N = A.size(0);
  const size_t M = B.size(1);
  const size_t K = A.size(1);
  __shared__ scalar_type As[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ scalar_type Bs[BLOCK_SIZE][BLOCK_SIZE];
  for (unsigned int i = 0; i < M; i += blockDim.x * gridDim.x) {
    for (unsigned int j = 0; j < N; j += blockDim.y * gridDim.y) {
      __syncthreads();
      unsigned int col = i + threadIdx.x + blockDim.x * blockIdx.x;
      unsigned int row = j + threadIdx.y + blockDim.y * blockIdx.y;
      scalar_type acc = 0;
      const unsigned int num_tiles = std::min(
          (K + blockDim.x - 1) / blockDim.x, (K + blockDim.y - 1) / blockDim.y);
      for (unsigned int t = 0; t < num_tiles; t++) {
        if (row < N && t * blockDim.x + threadIdx.x < K) {
          As[threadIdx.y][threadIdx.x] = A[row][t * blockDim.x + threadIdx.x];
        } else {
          As[threadIdx.y][threadIdx.x] = 0;
        }
        if (t * blockDim.y + threadIdx.y < K && col < M) {
          Bs[threadIdx.y][threadIdx.x] = B[t * blockDim.y + threadIdx.y][col];
        } else {
          Bs[threadIdx.y][threadIdx.x] = 0;
        }
        __syncthreads();
        for (unsigned int k = 0; k < blockDim.x && k < K; k++) {
          acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
      }

      if (row < N && col < M) {
        C[row][col] = acc;
      }
    }
  }
  return;
}

void invoke_matmul_kernel(const torch::Tensor A, const torch::Tensor B,
                          torch::Tensor C, const unsigned int threadsPerBlock,
                          unsigned int threadBlocks) {
  const unsigned int blockSize = std::floor(std::sqrt(threadsPerBlock));
  const dim3 dimBlock(blockSize, blockSize);
  dim3 dimGrid;
  if (C.size(0) * C.size(1) <= threadsPerBlock) {
    threadBlocks = 1;
  } else if (C.size(0) * C.size(1) <= threadsPerBlock * threadBlocks) {
    threadBlocks = std::ceil(C.size(0) * C.size(1) / threadsPerBlock);
  }
const unsigned int gridSize = std::floor(sqrt(threadBlocks));
  dimGrid = {gridSize, gridSize};
  if (A.dim() == 2 && B.dim() == 2) {
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "ParallelMatMul", [&] {
      batch_fast_matmul_kernel_2x2<scalar_t><<<dimGrid, dimBlock>>>(
          A.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
          B.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
          C.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>());
    });
  }
  if (B.size(0) == 1 && A.size(1) != 1) {
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "ParallelMatMul", [&] {
      batch_fast_matmul_kernel_2x1<scalar_t><<<dimGrid, dimBlock>>>(
          B.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
          A.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
          C.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>());
    });
  } else if (B.size(0) != 1 && A.size(1) == 1) {
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "ParallelMatMul", [&] {
      batch_fast_matmul_kernel_2x1<scalar_t><<<dimGrid, dimBlock>>>(
          A.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
          B.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
          C.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>());
    });
  } else {
    dimGrid = {1, 1};
  }
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
  CHECK_INPUT(z);

torch::Tensor cuda_matmult(torch::Tensor A, torch::Tensor B, torch::Tensor C,
                           const unsigned int T, const unsigned int TB) {
  CHECK_INPUTS(A, B, C);
  TORCH_CHECK(A.dim() <= 2 && B.dim() <= 2,
              "Input tensors must be 1 or 2 dimensional");
  TORCH_CHECK(A.size(1) == B.size(0) && C.size(0) == A.size(0) &&
                  C.size(1) == B.size(1),
              "Input tensors must be compatible for matrix multiplication");
  invoke_matmul_kernel(A, B, C, T, TB);
  return C;
}

__host__ void calc_cpu_matmult(const torch::Tensor A, const torch::Tensor B,
                               torch::Tensor C) {
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

torch::Tensor cpu_matmult(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
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
