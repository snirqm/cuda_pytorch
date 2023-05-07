#include <torch/extension.h>
torch::Tensor cuda_matmult(torch::Tensor A, torch::Tensor B, torch::Tensor C, int T, int TB);
torch::Tensor cpu_matmult(torch::Tensor A, torch::Tensor B, torch::Tensor C);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cuda_matmult", &cuda_matmult, "cuda_matmult (CUDA)");
  m.def("cpu_matmult", &cpu_matmult, "cpu_matmult (CPU)");
}

