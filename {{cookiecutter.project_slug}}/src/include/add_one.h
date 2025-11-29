#pragma once

#include <torch/torch.h>

namespace {{ cookiecutter.project_slug }} {

// Add 1.0 to every element of the input tensor
// Supports float32 and bfloat16 CUDA tensors
at::Tensor add_one(const at::Tensor& input);

} // namespace {{ cookiecutter.project_slug }}

