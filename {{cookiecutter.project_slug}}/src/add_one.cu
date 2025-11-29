// CUDA kernel implementation for add_one operator
// See: https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html

#include "include/add_one.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace {{ cookiecutter.project_slug }} {

namespace {

template <typename scalar_t>
__global__ void add_one_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t numel) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) {
    output[idx] = input[idx] + static_cast<scalar_t>(1.0);
  }
}

} // namespace

at::Tensor add_one(const at::Tensor& input) {
  TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
  TORCH_CHECK(
      input.scalar_type() == at::kFloat || input.scalar_type() == at::kBFloat16,
      "Input must be float32 or bfloat16");

  auto output = at::empty_like(input);
  const int64_t numel = input.numel();

  if (numel == 0) {
    return output;
  }

  const int threads = 256;
  const int blocks = (numel + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES_AND(
      at::kBFloat16, input.scalar_type(), "add_one_cuda", [&] {
        add_one_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            numel);
      });

  return output;
}

} // namespace {{ cookiecutter.project_slug }}

