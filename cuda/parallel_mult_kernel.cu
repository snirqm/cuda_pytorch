#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdio.h>

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#define BLOCK_SIZE 32

template <typename scalar_type>
__global__ void fast_matmul_kernel(
    const torch::PackedTensorAccessor<scalar_type, 2, torch::RestrictPtrTraits>
        A,
    const torch::PackedTensorAccessor<scalar_type, 2, torch::RestrictPtrTraits>
        B,
    torch::PackedTensorAccessor<scalar_type, 2, torch::RestrictPtrTraits> C,
    const int M, const int N, const int K) {

  __shared__ scalar_type As[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ scalar_type Bs[BLOCK_SIZE][BLOCK_SIZE];

  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  scalar_type acc = 0;

  const int num_tiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

  for (int t = 0; t < num_tiles; t++) {
    if (row < M && t * BLOCK_SIZE + threadIdx.x < K) {
      As[threadIdx.y][threadIdx.x] = A[row][t * BLOCK_SIZE + threadIdx.x];
    } else {
      As[threadIdx.y][threadIdx.x] = 0;
    }

    if (t * BLOCK_SIZE + threadIdx.y < K && col < N) {
      Bs[threadIdx.y][threadIdx.x] = B[t * BLOCK_SIZE + threadIdx.y][col];
    } else {
      Bs[threadIdx.y][threadIdx.x] = 0;
    }

    __syncthreads();

    for (int k = 0; k < BLOCK_SIZE; k++) {
      acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    C[row][col] = acc;
  }
}
template <typename scalar_type>
__global__ void fast_matmul_kernel(
    const torch::PackedTensorAccessor<scalar_type, 1, torch::RestrictPtrTraits>
        A,
    const torch::PackedTensorAccessor<scalar_type, 2, torch::RestrictPtrTraits>
        B,
    torch::PackedTensorAccessor<scalar_type, 1, torch::RestrictPtrTraits> C,
    const int M, const int N, const int K) {
  __shared__ scalar_type As[BLOCK_SIZE];
  __shared__ scalar_type Bs[BLOCK_SIZE][BLOCK_SIZE];

  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  scalar_type acc = 0;

  const int num_tiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

  for (int t = 0; t < num_tiles; t++) {
    if (t * BLOCK_SIZE + threadIdx.x < K) {
      As[threadIdx.x] = A[t * BLOCK_SIZE + threadIdx.x];
    } else {
      As[threadIdx.x] = 0;
    }

    if (t * BLOCK_SIZE + threadIdx.y < K && col < N) {
      Bs[threadIdx.y][threadIdx.x] = B[t * BLOCK_SIZE + threadIdx.y][col];
    } else {
      Bs[threadIdx.y][threadIdx.x] = 0;
    }

    __syncthreads();

    for (int k = 0; k < BLOCK_SIZE; k++) {
      acc += As[k] * Bs[k][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    C[row] = acc;
  }
}

template <typename scalar_type>
__global__ void fast_matmul_kernel(
    const torch::PackedTensorAccessor<scalar_type, 2, torch::RestrictPtrTraits>
        A,
    const torch::PackedTensorAccessor<scalar_type, 1, torch::RestrictPtrTraits>
        B,
    torch::PackedTensorAccessor<scalar_type, 1, torch::RestrictPtrTraits> C,
    const int M, const int N, const int K) {
  __shared__ scalar_type As[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ scalar_type Bs[BLOCK_SIZE];

  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  scalar_type acc = 0;

  const int num_tiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

  for (int t = 0; t < num_tiles; t++) {
    if (row < M && t * BLOCK_SIZE + threadIdx.x < K) {
      As[threadIdx.y][threadIdx.x] = A[row][t * BLOCK_SIZE + threadIdx.x];
    } else {
      As[threadIdx.y][threadIdx.x] = 0;
    }

    if (t * BLOCK_SIZE + threadIdx.y < K) {
      Bs[threadIdx.y] = B[t * BLOCK_SIZE + threadIdx.y];
    } else {
      Bs[threadIdx.y] = 0;
    }

    __syncthreads();

    for (int k = 0; k < BLOCK_SIZE; k++) {
      acc += As[threadIdx.y][k] * Bs[k];
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    C[row] = acc;
  }
}

template <typename scalar_type>
__global__ void fast_matmul_kernel(
    const torch::PackedTensorAccessor<scalar_type, 1, torch::RestrictPtrTraits>
        A,
    const torch::PackedTensorAccessor<scalar_type, 1, torch::RestrictPtrTraits>
        B,
    torch::PackedTensorAccessor<scalar_type, 1, torch::RestrictPtrTraits> C,
    const int M, const int N, const int K) {
  __shared__ scalar_type As[BLOCK_SIZE];
  __shared__ scalar_type Bs[BLOCK_SIZE];

  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  scalar_type acc = 0;

  const int num_tiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

  for (int t = 0; t < num_tiles; t++) {
    if (t * BLOCK_SIZE + threadIdx.x < K) {
      As[threadIdx.x] = A[t * BLOCK_SIZE + threadIdx.x];
    } else {
      As[threadIdx.x] = 0;
    }

    if (t * BLOCK_SIZE + threadIdx.y < K) {
      Bs[threadIdx.y] = B[t * BLOCK_SIZE + threadIdx.y];
    } else {
      Bs[threadIdx.y] = 0;
    }

    __syncthreads();

    for (int k = 0; k < BLOCK_SIZE; k++) {
      acc += As[k] * Bs[k];
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    C[row] = acc;
  }
}
#define INVOKE_MATMUL_KERNEL(dimA, dimB, dimC)                                 \
  fast_matmul_kernel<scalar_t><<<dimGrid, dimBlock>>>(                         \
      A.packed_accessor64<scalar_t, dimA, torch::RestrictPtrTraits>(),         \
      B.packed_accessor64<scalar_t, dimB, torch::RestrictPtrTraits>(),         \
      C.packed_accessor64<scalar_t, dimC, torch::RestrictPtrTraits>(), rows,   \
      cols, inner);

template <size_t dimA, size_t dimB>
void invoke_matmul_kernel(const torch::Tensor A, const torch::Tensor B,
                          torch::Tensor C, const size_t rows,
                          const size_t inner, const size_t cols, const size_t T,
                          const int TB) {
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x,
               (rows + dimBlock.y - 1) / dimBlock.y);

  AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "ParallelMatMul", [&] {
    if (dimA == 1 && dimB == 1) {
      INVOKE_MATMUL_KERNEL(1, 1, 1);
    } else if (dimA == 1 && dimB == 2) {
      INVOKE_MATMUL_KERNEL(1, 2, 1);
    } else if (dimA == 2 && dimB == 1) {
      INVOKE_MATMUL_KERNEL(2, 1, 1);
    } else if (dimA == 2 && dimB == 2) {
      INVOKE_MATMUL_KERNEL(2, 2, 2);
    }
  });
}
// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

torch::Tensor cuda_matmult(torch::Tensor A, torch::Tensor B, torch::Tensor C,
                           int T, int TB) {
  // Check input tensor dimensions
  TORCH_CHECK(A.dim() <= 2 && B.dim() <= 2,
              "Input tensors must be 1 or 2 dimensional");
  CHECK_INPUT(A);
  CHECK_INPUT(B);
  CHECK_INPUT(C);
  size_t rows, inner, cols;
  if (A.dim() == 1 && B.dim() == 1) {
    TORCH_CHECK(A.size(0) == B.size(0),
                "Input tensors must be compatible for matrix multiplication");
    TORCH_CHECK(C.size(0) == 1,
                "Input tensors must be compatible for matrix multiplication");
    rows = 1;
    inner = A.size(0);
    cols = 1;
    invoke_matmul_kernel<1, 1>(A, B, C, rows, inner, cols, T, TB);
  } else if (A.dim() == 2 && B.dim() == 1) {
    TORCH_CHECK(A.size(1) == B.size(0),
                "Input tensors must be compatible for matrix multiplication");
    TORCH_CHECK(C.size(0) == A.size(0),
                "Input tensors must be compatible for matrix multiplication");
    TORCH_CHECK(C.size(1) == 1,
                "Input tensors must be compatible for matrix multiplication");
    rows = A.size(0);
    inner = A.size(1);
    cols = B.size(0);
    invoke_matmul_kernel<2, 1>(A, B, C, rows, inner, cols, T, TB);
  } else if (A.dim() == 1 && B.dim() == 2) {
    TORCH_CHECK(A.size(0) == B.size(0),
                "Input tensors must be compatible for matrix multiplication");
    TORCH_CHECK(C.size(0) == 1,
                "Input tensors must be compatible for matrix multiplication");
    TORCH_CHECK(C.size(1) == B.size(1),
                "Input tensors must be compatible for matrix multiplication");
    rows = B.size(0);
    inner = B.size(1);
    cols = A.size(0);
    invoke_matmul_kernel<1, 2>(A, B, C, rows, inner, cols, T, TB);
  } else if (A.dim() == 2 && B.dim() == 2) {
    TORCH_CHECK(A.size(1) == B.size(0),
                "Input tensors must be compatible for matrix multiplication");
    TORCH_CHECK(C.size(0) == A.size(0),
                "Input tensors must be compatible for matrix multiplication");
    TORCH_CHECK(C.size(1) == B.size(1),
                "Input tensors must be compatible for matrix multiplication");
    rows = A.size(0);
    inner = A.size(1);
    cols = B.size(1);
    invoke_matmul_kernel<2, 2>(A, B, C, rows, inner, cols, T, TB);
  }
  return C;
}

template <size_t dimA, size_t dimB>
__host__ void calc_cpu_matmult(const torch::Tensor A, const torch::Tensor B,
                               torch::Tensor C, const size_t rows,
                               const size_t inner, const size_t cols) {
  if (dimA == 1 && dimB == 1) {
    for (size_t i = 0; i < rows; i++) {
      C[0] += A[i] * B[i];
    }
  } else if (dimA == 1 && dimB == 2) {
    for (size_t i = 0; i < rows; i++) {
      for (size_t j = 0; j < cols; j++) {
        C[j] += A[i] * B[j][i];
      }
    }
  } else if (dimA == 2 && dimB == 1) {
    for (size_t i = 0; i < rows; i++) {
      for (size_t j = 0; j < cols; j++) {
        C[i] += A[i][j] * B[j];
      }
    }
  } else if (dimA == 2 && dimB == 2) {
    for (size_t i = 0; i < rows; i++) {
      for (size_t j = 0; j < cols; j++) {
        for (size_t k = 0; k < inner; k++) {
          C[i][j] += A[i][k] * B[k][j];
        }
      }
    }
  }
  return;
}

torch::Tensor cpu_matmult(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
  // Check input tensor dimensions
  TORCH_CHECK(A.dim() <= 2 && B.dim() <= 2,
              "Input tensors must be 1 or 2 dimensional");
  CHECK_CONTIGUOUS(A);
  CHECK_CONTIGUOUS(B);
  CHECK_CONTIGUOUS(C);
  size_t rows, inner, cols;
  if (A.dim() == 1 && B.dim() == 1) {
    TORCH_CHECK(A.size(0) == B.size(0),
                "Input tensors must be compatible for matrix multiplication");
    TORCH_CHECK(C.size(0) == 1,
                "Input tensors must be compatible for matrix multiplication");
    rows = 1;
    inner = A.size(0);
    cols = 1;
    calc_cpu_matmult<1, 1>(A, B, C, rows, inner, cols);
  } else if (A.dim() == 2 && B.dim() == 1) {
    TORCH_CHECK(A.size(1) == B.size(0),
                "Input tensors must be compatible for matrix multiplication");
    TORCH_CHECK(C.size(0) == A.size(0),
                "Input tensors must be compatible for matrix multiplication");
    TORCH_CHECK(C.size(1) == 1,
                "Input tensors must be compatible for matrix multiplication");
    rows = A.size(0);
    inner = A.size(1);
    cols = B.size(0);
    calc_cpu_matmult<2, 1>(A, B, C, rows, inner, cols);
  } else if (A.dim() == 1 && B.dim() == 2) {
    TORCH_CHECK(A.size(0) == B.size(0),
                "Input tensors must be compatible for matrix multiplication");
    TORCH_CHECK(C.size(0) == 1,
                "Input tensors must be compatible for matrix multiplication");
    TORCH_CHECK(C.size(1) == B.size(1),
                "Input tensors must be compatible for matrix multiplication");
    rows = B.size(0);
    inner = B.size(1);
    cols = A.size(0);
    calc_cpu_matmult<1, 2>(A, B, C, rows, inner, cols);
  } else if (A.dim() == 2 && B.dim() == 2) {
    TORCH_CHECK(A.size(1) == B.size(0),
                "Input tensors must be compatible for matrix multiplication");
    TORCH_CHECK(C.size(0) == A.size(0),
                "Input tensors must be compatible for matrix multiplication");
    TORCH_CHECK(C.size(1) == B.size(1),
                "Input tensors must be compatible for matrix multiplication");
    rows = A.size(0);
    inner = A.size(1);
    cols = B.size(1);
    calc_cpu_matmult<2, 2>(A, B, C, rows, inner, cols);
  }
  return C;
}
