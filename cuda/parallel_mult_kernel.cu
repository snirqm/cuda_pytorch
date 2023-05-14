#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>

#define IS_FIRST(threadIdx) (threadIdx.x == 0 && threadIdx.y == 0)

#define BLOCK_SIZE 32
template <typename scalar_type>
__global__ void batch_fast_matmul_kernel_2x1(
    const torch::PackedTensorAccessor<scalar_type, 2, torch::RestrictPtrTraits>
        A,
    const torch::PackedTensorAccessor<scalar_type, 1, torch::RestrictPtrTraits>
        B,
    torch::PackedTensorAccessor<scalar_type, 1, torch::RestrictPtrTraits> C) {
  unsigned int N = A.size(0);
  unsigned int K = A.size(1);
  __shared__ scalar_type Acc[BLOCK_SIZE * BLOCK_SIZE];

  for (unsigned int i = 0; i <= N; i += gridDim.x) {
    unsigned int row = i + blockIdx.x - 1;
    scalar_type acc = 0;
    const size_t tile_size = std::min((int)(blockDim.x * blockDim.y), BLOCK_SIZE * BLOCK_SIZE);
    const unsigned int num_tiles = (K + tile_size - 1) / tile_size;
    for (unsigned int t = 0; t < num_tiles; t++) {
      unsigned int Ax = t * tile_size + threadIdx.x;
      if (Ax < K && row < N) {
        Acc[threadIdx.x] = A[row][Ax] * B[Ax];
      } else {
        Acc[threadIdx.x] = 0;
      }
      __syncthreads();
      if (IS_FIRST(threadIdx) && row < N) {
        for (unsigned int k = 0; k < tile_size && t * tile_size + k < K; k++) {
          acc += Acc[k];
        }
      }
      __syncthreads();
    }

    if (IS_FIRST(threadIdx) && row < N) {
      C[row] = acc;
    }
    __syncthreads();
  }
}


template <typename scalar_type>
__global__ void batch_fast_matmul_kernel_1x1(
    const torch::PackedTensorAccessor<scalar_type, 1, torch::RestrictPtrTraits>
        A,
    const torch::PackedTensorAccessor<scalar_type, 1, torch::RestrictPtrTraits>
        B,
    torch::PackedTensorAccessor<scalar_type, 1, torch::RestrictPtrTraits> C) {
  __shared__ scalar_type Acc[BLOCK_SIZE * BLOCK_SIZE];
  unsigned int N = A.size(0);
  scalar_type acc = 0;
  int idx = blockIdx.x - 1;
  const size_t tile_size =
      std::min((int)(blockDim.x * blockDim.y), BLOCK_SIZE * BLOCK_SIZE);
  const unsigned int num_tiles = (N + tile_size - 1) / tile_size;
  for (unsigned int t = 0; t < num_tiles; t++) {
    unsigned int Ax = t * tile_size + threadIdx.x;
    if (Ax < N) {
      Acc[threadIdx.x] = A[Ax] * B[Ax];
    } else {
      Acc[threadIdx.x] = 0;
    }
    __syncthreads();
    if (IS_FIRST(threadIdx) && (idx <= 0)) {
      for (unsigned int k = 0; k < tile_size && t * tile_size + k < N; k++) {
        acc += Acc[k];
      }
    }
    __syncthreads();
  }

  if (IS_FIRST(threadIdx) && (idx <= 0)) {
    C[0] = acc;
  }
  __syncthreads();
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
  if (CASE_2x2(A, B)) {
    if (C.size(0) * C.size(1) <= threadsPerBlock) {
      threadBlocks = 1;
    } else if (C.size(0) * C.size(1) <= threadsPerBlock * threadBlocks) {
      threadBlocks = std::ceil(C.size(0) * C.size(1) / threadsPerBlock);
    }
    const dim3 dimBlock(blockSize, blockSize);
    const dim3 dimGrid = {gridSize, gridSize};
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "ParallelMatMul_2x2", [&] {
      batch_fast_matmul_kernel_2x2<scalar_t><<<dimGrid, dimBlock>>>(
          CUDA_TENSOR(A, 2), CUDA_TENSOR(B, 2), CUDA_TENSOR(C, 2));
    });
  } else if (CASE_2x1(A, B)) {
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "ParallelMatMul_2x1", [&] {
      batch_fast_matmul_kernel_2x1<scalar_t><<<threadBlocks, threadsPerBlock>>>(
          CUDA_TENSOR(A, 2), CUDA_TENSOR(B, 1), CUDA_TENSOR(C, 1));
    });
  } else if (CASE_1x1(A, B)) {
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "ParallelMatMul_1x1", [&] {
      batch_fast_matmul_kernel_1x1<scalar_t><<<threadBlocks, threadsPerBlock>>>(
          CUDA_TENSOR(A, 1), CUDA_TENSOR(B, 1), CUDA_TENSOR(C, 1));
    });
  } else {
    TORCH_CHECK(false, "ParallelMatMul: Unsupported tensor dimensions");
  }
  cudaDeviceSynchronize();
}

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
  CHECK_CONTIGUOUS(A);
  CHECK_CONTIGUOUS(B);
  CHECK_CONTIGUOUS(C);
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2,
              "Input tensors must be 2 dimensional");
  calc_cpu_matmult(A, B, C);
  return C;
}
