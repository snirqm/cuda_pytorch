#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>

#define BLOCK_SIZE 32
template <typename scalar_type>
__global__ void batch_fast_matmul_kernel_2x1_new(
    const torch::PackedTensorAccessor<scalar_type, 2, torch::RestrictPtrTraits>
        A,
    const torch::PackedTensorAccessor<scalar_type, 1, torch::RestrictPtrTraits>
        B,
    torch::PackedTensorAccessor<scalar_type, 2, torch::RestrictPtrTraits> C) {
  const unsigned int N = A.size(0);
  const unsigned int K = A.size(1);
  unsigned int row_offset = 0;
  for (; row_offset < N; row_offset += gridDim.y * blockDim.y) {
    scalar_type acc = 0;
    for (unsigned int i = 0; i < K; i++) {
      unsigned int j = row_offset + threadIdx.y + blockDim.y * blockIdx.y;
      if (j < N) {
        acc += A[j][i] * B[i];
      }
    }
    C[row_offset + threadIdx.y + blockDim.y * blockIdx.y][0] = acc;
  }
}

template <typename scalar_type>
__global__ void batch_fast_matmul_kernel_2x1(
     const torch::PackedTensorAccessor<scalar_type, 2,
     torch::RestrictPtrTraits>
        A,
    const torch::PackedTensorAccessor<scalar_type, 1,
    torch::RestrictPtrTraits>
        B,
    torch::PackedTensorAccessor<scalar_type, 2, torch::RestrictPtrTraits> C)
    {
  const unsigned int N = A.size(0);
  const unsigned int K = A.size(1);
  __shared__ scalar_type As[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ scalar_type Bs[BLOCK_SIZE];
  unsigned int col = threadIdx.x + blockDim.x * blockIdx.x;
  for (unsigned int i = 0; i < N; i += blockDim.y * gridDim.y) {
    unsigned int row = i + threadIdx.y + blockDim.y * blockIdx.y;
    scalar_type acc = 0;
    const size_t tile_size = std::min((int) blockDim.x, BLOCK_SIZE);
    const unsigned int num_tiles = (K + tile_size - 1) / tile_size;
    // The loop accumulates the result for each tile in `acc`
    for (unsigned int t = 0; t < num_tiles; t++) {
        // Fetch data from A to shared memory
      unsigned int Ax = t * tile_size + col;
      if (Ax < K && row < N) {
        As[threadIdx.y][threadIdx.x] = A[row][Ax];
      } else {
        As[threadIdx.y][threadIdx.x] = 0;
      }
      __syncthreads();
      // Accumulate the result
      for (unsigned int k = 0; k < blockDim.x && t * tile_size + k < K; k++) {
        acc += As[threadIdx.y][k] * B[k + t * tile_size];
      }
      __syncthreads();
    }
    if (row < N && col == 0) {
      C[row][col] = acc;
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
        for (unsigned int k = 0; k < blockDim.x; k++) {
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

#define CASE_2x2(A, B) A.dim() == 2 && B.dim() == 2
#define CASE_2x1(A, B) A.dim() == 2 && B.dim() == 1
#define CASE_1x1(A, B) A.dim() == 1 && B.dim() == 1

#define CUDA_TENSOR(X, n_dim)                                                  \
  X.packed_accessor64<scalar_t, n_dim, torch::RestrictPtrTraits>()

void invoke_matmul_kernel(const torch::Tensor A, const torch::Tensor B,
                          torch::Tensor C, const unsigned int threadsPerBlock,
                          unsigned int threadBlocks) {
  const unsigned int blockSize = std::floor(std::sqrt(threadsPerBlock));
  const unsigned int gridSize = std::floor(sqrt(threadBlocks));
    const dim3 dimBlock(blockSize, blockSize);
  if (CASE_2x2(A, B)) {
    if (C.size(0) * C.size(1) <= threadsPerBlock) {
      threadBlocks = 1;
    } else if (C.size(0) * C.size(1) <= threadsPerBlock * threadBlocks) {
      threadBlocks = std::min(threadBlocks, std::ceil(C.size(0) * C.size(1) / threadsPerBlock));
    }
    const dim3 dimGrid = {gridSize, gridSize};
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "ParallelMatMul_2x2", [&] {
      batch_fast_matmul_kernel_2x2<scalar_t><<<dimGrid, dimBlock>>>(
          CUDA_TENSOR(A, 2), CUDA_TENSOR(B, 2), CUDA_TENSOR(C, 2));
    });
  } else if (CASE_2x1(A, B)) {
    const dim3 dimGrid = {1, threadBlocks};
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "ParallelMatMul_2x1", [&] {
      batch_fast_matmul_kernel_2x1<scalar_t><<<dimGrid, dimBlock>>>(
          CUDA_TENSOR(A, 2), CUDA_TENSOR(B, 1), CUDA_TENSOR(C, 2));
    });
  } else {
    TORCH_CHECK(false, "ParallelMatMul: Unsupported tensor dimensions");
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
  calc_cpu_matmult(A, B, C);
  return C;
}
