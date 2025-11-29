# C++ Custom Ops Implementation Guide

This directory contains the C++ and CUDA source code for your custom operators.

## Directory Structure

```
src/
├── include/
│   └── add_one.h       # Header files declaring your ops
├── add_one.cu          # CUDA kernel implementations
├── register_ops.cpp    # PyTorch operator registration
└── README.md           # This file
```

## How Custom Ops Work

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  Python Call    │────▶│  torch.ops.*     │────▶│  C++ Implementation │
│  add_one(x)     │     │  dispatcher      │     │  in .cu file        │
└─────────────────┘     └──────────────────┘     └─────────────────────┘
                               │
                               ▼
                        ┌──────────────────┐
                        │  @register_fake  │ (for torch.compile)
                        │  in Python       │
                        └──────────────────┘
```

## Step-by-Step: Adding a New Op

### Step 1: Declare the Function (Header)

Create `src/include/my_op.h`:

```cpp
#pragma once
#include <torch/torch.h>

namespace {{ cookiecutter.project_slug }} {

// Declare your function signature
// Use at::Tensor for tensor arguments and returns
at::Tensor my_op(const at::Tensor& input, double scalar_arg);

// For ops that return multiple tensors:
std::tuple<at::Tensor, at::Tensor> my_op_with_multiple_outputs(
    const at::Tensor& input);

// For in-place ops (mutate input):
void my_op_inplace(at::Tensor& input);  // Note: non-const reference

}  // namespace {{ cookiecutter.project_slug }}
```

### Step 2: Implement the CUDA Kernel

Create `src/my_op.cu`:

```cpp
#include "include/my_op.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace {{ cookiecutter.project_slug }} {

// Define your CUDA kernel
template <typename scalar_t>
__global__ void my_op_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    double scalar_arg,
    int64_t numel) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) {
    // Your kernel logic here
    output[idx] = input[idx] * static_cast<scalar_t>(scalar_arg);
  }
}

// Host function that launches the kernel
at::Tensor my_op(const at::Tensor& input, double scalar_arg) {
  // Input validation
  TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
  TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

  // Allocate output tensor
  auto output = at::empty_like(input);

  // Handle empty tensors
  if (input.numel() == 0) {
    return output;
  }

  // Configure kernel launch
  const int threads = 256;
  const int blocks = (input.numel() + threads - 1) / threads;

  // Dispatch based on dtype
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "my_op_cuda", [&] {
        my_op_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            scalar_arg,
            input.numel());
      });

  // Check for kernel launch errors
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return output;
}

}  // namespace {{ cookiecutter.project_slug }}
```

### Step 3: Register the Operator

Add to `src/register_ops.cpp`:

```cpp
#include "include/my_op.h"  // Add this include

// Inside the TORCH_LIBRARY block, add:

// Define the schema (signature visible to Python)
m.def("my_op(Tensor input, float scalar_arg) -> Tensor");

// Register the CUDA implementation
m.impl("my_op", c10::DispatchKey::CUDA,
       TORCH_FN({{ cookiecutter.project_slug }}::my_op));
```

**Schema syntax reference:**
| Python Type | Schema Type |
|-------------|-------------|
| `torch.Tensor` | `Tensor` |
| `float` | `float` |
| `int` | `int` |
| `bool` | `bool` |
| `Optional[Tensor]` | `Tensor?` |
| `List[int]` | `int[]` |
| `Tuple[Tensor, Tensor]` | `(Tensor, Tensor)` |

**For mutating ops** (in-place), use `Tensor(a!)`:
```cpp
m.def("my_op_inplace(Tensor(a!) input) -> ()");
```

### Step 4: Add Meta Function (Required for torch.compile)

Add to `{{ cookiecutter.project_slug }}/abstract_impls.py`:

```python
@register_fake("{{ cookiecutter.project_slug }}::my_op")
def my_op_meta(input: torch.Tensor, scalar_arg: float) -> torch.Tensor:
    """Meta implementation - describes output shape/dtype without running kernel.

    This is called by torch.compile to trace through your op.
    It must return a tensor with the correct:
    - shape
    - dtype
    - device
    - memory layout (strides)
    """
    # For ops that preserve shape/dtype:
    return torch.empty_like(input)

    # For ops that change shape:
    # return input.new_empty(new_shape)

    # For ops that change dtype:
    # return torch.empty_like(input, dtype=torch.float32)
```

### Step 5: Add Python Wrapper (Optional but Recommended)

Add to `{{ cookiecutter.project_slug }}/__init__.py`:

```python
def my_op(x: torch.Tensor, scalar_arg: float) -> torch.Tensor:
    """Apply my_op to the input tensor.

    Args:
        x: Input CUDA tensor
        scalar_arg: Scalar multiplier

    Returns:
        Result tensor with same shape as input
    """
    return ops.my_op(x, scalar_arg)
```

### Step 6: Add Tests

Create `test/test_my_op.py`:

```python
import torch
import pytest
from {{ cookiecutter.project_slug }} import my_op


def test_my_op_correctness():
    x = torch.randn(32, 32, device="cuda")
    result = my_op(x, 2.0)
    expected = x * 2.0
    torch.testing.assert_close(result, expected)


def test_my_op_opcheck():
    """Verify operator was registered correctly."""
    x = torch.randn(32, 32, device="cuda")
    torch.library.opcheck(
        torch.ops.{{ cookiecutter.project_slug }}.my_op,
        (x, 2.0),
    )


def test_my_op_compile():
    """Verify op works with torch.compile."""
    x = torch.randn(32, 32, device="cuda")

    @torch.compile(fullgraph=True)
    def fn(x):
        return my_op(x, 2.0)

    result = fn(x)
    expected = x * 2.0
    torch.testing.assert_close(result, expected)
```

### Step 7: Rebuild

```bash
pip install -e .

# Or for faster iteration (after initial build):
cd build && ninja && cd ..
```

## Common Patterns

### Multiple Dispatch Keys (CPU + CUDA)

```cpp
// In register_ops.cpp
m.def("my_op(Tensor input) -> Tensor");
m.impl("my_op", c10::DispatchKey::CUDA, TORCH_FN(my_op_cuda));
m.impl("my_op", c10::DispatchKey::CPU, TORCH_FN(my_op_cpu));
```

### Returning Multiple Tensors

```cpp
// Header
std::tuple<at::Tensor, at::Tensor> my_op(const at::Tensor& input);

// Registration
m.def("my_op(Tensor input) -> (Tensor, Tensor)");

// Meta function
@register_fake("{{ cookiecutter.project_slug }}::my_op")
def my_op_meta(input):
    return torch.empty_like(input), torch.empty_like(input)
```

### Optional Tensor Arguments

```cpp
// Schema
m.def("my_op(Tensor input, Tensor? optional_arg) -> Tensor");

// C++ signature
at::Tensor my_op(const at::Tensor& input,
                 const std::optional<at::Tensor>& optional_arg);
```

## Debugging Tips

1. **Kernel not launching?** Add `C10_CUDA_KERNEL_LAUNCH_CHECK()` after kernel call
2. **Wrong results?** Print intermediate values with `std::cout << tensor << std::endl;`
3. **Compile errors?** Check schema matches C++ signature exactly
4. **torch.compile fails?** Ensure meta function returns correct shape/dtype

## References

- [PyTorch Custom C++ Operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)
- [TORCH_LIBRARY API](https://pytorch.org/docs/stable/library.html)
- [LibTorch Stable ABI](https://docs.pytorch.org/docs/main/notes/libtorch_stable_abi.html)

