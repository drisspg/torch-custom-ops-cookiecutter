# {{ cookiecutter.project_name }}

{{ cookiecutter.project_description }}

{% if cookiecutter.variant == 'cpp' %}
This project provides custom CUDA operators implemented in C++ with Python bindings, compatible with `torch.compile`.
{% else %}
This project provides custom operators implemented in Python using Triton, compatible with `torch.compile`.
{% endif %}

## Getting Started After Generation

You just generated this project with cookiecutter. Here's what to do next:

### 1. Replace the Example Op with Your Implementation

{% if cookiecutter.variant == 'cpp' %}
The template includes an example `add_one` op. To add your own ops:

| File | What to Edit |
|------|--------------|
| `src/include/*.h` | Declare your C++ function signatures |
| `src/*.cu` | Implement your CUDA kernels |
| `src/register_ops.cpp` | Register ops with `TORCH_LIBRARY` |
| `{{ cookiecutter.project_slug }}/abstract_impls.py` | Add `@register_fake` meta functions |
| `{{ cookiecutter.project_slug }}/__init__.py` | Add Python wrappers with docstrings |
{% else %}
The template includes an example `add_one` op. To add your own ops:

| File | What to Edit |
|------|--------------|
| `{{ cookiecutter.project_slug }}/ops.py` | Add `@triton.jit` kernels + `@torch.library.custom_op` |
| `{{ cookiecutter.project_slug }}/__init__.py` | Export your new ops |
{% endif %}

### 2. Build & Install

```bash
cd {{ cookiecutter.project_slug }}

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
{% if cookiecutter.variant == 'cpp' %}

# For faster iteration on C++ changes (after initial build):
cd build && ninja && cd ..
{% endif %}
```

### 3. Run Tests Locally

```bash
# Run all tests
pytest test/ -v

# Run a specific test
pytest test/test_add_one.py -v

# Run with print output
pytest test/ -v -s
```

### 4. Set Up Pre-commit Hooks

```bash
# Install hooks (runs automatically on git commit)
pre-commit install

# Run manually on all files
pre-commit run --all-files
```

**Included hooks:**
- **ruff**: Fast Python linter and formatter
- **pyupgrade**: Automatically upgrade Python syntax to {{ cookiecutter.min_python_version }}+
- **pyrefly**: Facebook's type checker
- **pre-commit-hooks**: Trailing whitespace, YAML/TOML validation, merge conflict detection

### 5. CI/CD (GitHub Actions)

Two workflows are included in `.github/workflows/`:

| Workflow | Trigger | What it Does |
|----------|---------|--------------|
| `ruff.yml` | Push/PR to `main` | Runs `ruff check` and `ruff format --check` |
| `publish-to-pypi.yml` | Git tag push | Builds wheel and publishes to PyPI (trusted publishing) |

**To publish a release:**
```bash
git tag v0.1.0
git push origin v0.1.0
```

**Note:** Configure PyPI trusted publishing in your PyPI project settings for the GitHub Actions workflow to work.

---

## Installation

### Development Installation

```bash
# Clone the repository
git clone https://github.com/{{ cookiecutter.github_username }}/{{ cookiecutter.project_slug }}.git
cd {{ cookiecutter.project_slug }}

# Install in editable mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

{% if cookiecutter.variant == 'cpp' %}
### Build Requirements

- Python >= {{ cookiecutter.min_python_version }}
- PyTorch >= {{ cookiecutter.min_torch_version }}
- CUDA Toolkit (matching your PyTorch CUDA version)
- CMake >= 3.26
- A C++17 compatible compiler
{% else %}
### Requirements

- Python >= {{ cookiecutter.min_python_version }}
- PyTorch >= {{ cookiecutter.min_torch_version }}
- Triton
{% endif %}

## Usage

```python
import torch
from {{ cookiecutter.project_slug }} import add_one

# Create a CUDA tensor
x = torch.randn(32, 32, device="cuda")

# Use the custom op
y = add_one(x)

# Works with torch.compile!
@torch.compile
def my_fn(x):
    return add_one(x)

result = my_fn(x)
```

## Running Tests

```bash
pytest test/
```

## Project Structure

```
{{ cookiecutter.project_slug }}/
{% if cookiecutter.variant == 'cpp' %}
├── CMakeLists.txt          # CMake build configuration
├── src/
│   ├── include/
│   │   └── add_one.h       # C++ header
│   ├── add_one.cu          # CUDA kernel implementation
│   └── register_ops.cpp    # PyTorch operator registration
├── {{ cookiecutter.project_slug }}/
│   ├── __init__.py         # Python package init
│   └── abstract_impls.py   # Meta functions for torch.compile
{% else %}
├── {{ cookiecutter.project_slug }}/
│   ├── __init__.py         # Python package init
│   └── ops.py              # Triton kernel implementations
{% endif %}
├── test/
│   └── test_add_one.py     # Tests with opcheck
├── pyproject.toml          # Build configuration
└── README.md
```

## What's Available After Building

After installing the package, users can access your custom ops:

```python
import torch
from {{ cookiecutter.project_slug }} import add_one

# Direct function call
x = torch.randn(32, 32, device="cuda")
y = add_one(x)

# Access via torch.ops namespace
y = torch.ops.{{ cookiecutter.project_slug }}.add_one(x)

# Works seamlessly with torch.compile
@torch.compile
def my_model(x):
    return add_one(x) * 2
```

## Adding New Operators

{% if cookiecutter.variant == 'cpp' %}
### C++ Custom Op Workflow

Here's how to add a new operator called `my_scale` that multiplies a tensor by a scalar:

**Step 1: Create the header** (`src/include/my_scale.h`)
```cpp
#pragma once
#include <torch/torch.h>

namespace {{ cookiecutter.project_slug }} {
at::Tensor my_scale(const at::Tensor& input, double scale);
}
```

**Step 2: Implement the CUDA kernel** (`src/my_scale.cu`)
```cpp
#include "include/my_scale.h"
#include <ATen/cuda/CUDAContext.h>

namespace {{ cookiecutter.project_slug }} {

template <typename scalar_t>
__global__ void my_scale_kernel(
    const scalar_t* input, scalar_t* output, scalar_t scale, int64_t numel) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) {
    output[idx] = input[idx] * scale;
  }
}

at::Tensor my_scale(const at::Tensor& input, double scale) {
  TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
  auto output = at::empty_like(input);
  const int threads = 256;
  const int blocks = (input.numel() + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "my_scale", [&] {
    my_scale_kernel<scalar_t><<<blocks, threads>>>(
        input.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        static_cast<scalar_t>(scale),
        input.numel());
  });
  return output;
}

}
```

**Step 3: Register the op** (add to `src/register_ops.cpp`)
```cpp
#include "include/my_scale.h"

// Inside TORCH_LIBRARY block:
m.def("my_scale(Tensor input, float scale) -> Tensor");
m.impl("my_scale", c10::DispatchKey::CUDA, TORCH_FN({{ cookiecutter.project_slug }}::my_scale));
```

**Step 4: Add meta function** (add to `{{ cookiecutter.project_slug }}/abstract_impls.py`)
```python
@register_fake("{{ cookiecutter.project_slug }}::my_scale")
def my_scale_meta(input: torch.Tensor, scale: float) -> torch.Tensor:
    return torch.empty_like(input)
```

**Step 5: Add Python wrapper** (add to `{{ cookiecutter.project_slug }}/__init__.py`)
```python
def my_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Scale tensor by a scalar value."""
    return ops.my_scale(x, scale)
```

**Step 6: Rebuild**
```bash
pip install -e .
```
{% else %}
### Python/Triton Custom Op Workflow

Here's how to add a new operator called `my_scale` that multiplies a tensor by a scalar:

**Step 1: Add the Triton kernel and custom_op** (add to `{{ cookiecutter.project_slug }}/ops.py`)
```python
@triton.jit
def _my_scale_kernel(
    input_ptr, output_ptr, scale, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x * scale, mask=mask)


@torch.library.custom_op("{{ cookiecutter.project_slug }}::my_scale", mutates_args=())
def my_scale(input: torch.Tensor, scale: float) -> torch.Tensor:
    """Scale tensor by a scalar value."""
    assert input.is_cuda and input.is_contiguous()
    output = torch.empty_like(input)
    n_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _my_scale_kernel[grid](input, output, scale, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output


@my_scale.register_fake
def _(input: torch.Tensor, scale: float) -> torch.Tensor:
    return torch.empty_like(input)
```

**Step 2: Export from `__init__.py`**
```python
from {{ cookiecutter.project_slug }}.ops import add_one, my_scale

__all__ = ["add_one", "my_scale"]
```

**Step 3: Add tests** (add to `test/test_my_scale.py`)
```python
import torch
from {{ cookiecutter.project_slug }} import my_scale

def test_my_scale():
    x = torch.randn(32, 32, device="cuda")
    result = my_scale(x, 2.0)
    torch.testing.assert_close(result, x * 2.0)

def test_my_scale_opcheck():
    x = torch.randn(32, 32, device="cuda")
    torch.library.opcheck(torch.ops.{{ cookiecutter.project_slug }}.my_scale, (x, 2.0))
```

No rebuild needed for Python changes - just re-import!
{% endif %}

{% if cookiecutter.variant == 'cpp' %}
## LibTorch Stable ABI (Future Migration)

For cross-version ABI stability (PyTorch 2.10+), you can migrate to the [LibTorch Stable ABI](https://docs.pytorch.org/docs/main/notes/libtorch_stable_abi.html). Key changes:

1. Replace `TORCH_LIBRARY` with `STABLE_TORCH_LIBRARY`
2. Replace `at::Tensor` with `torch::stable::Tensor`
3. Replace `TORCH_CHECK` with `STD_TORCH_CHECK`
4. Use boxed kernel signatures: `(StableIValue* stack, uint64_t num_args, uint64_t num_outputs) -> void`

**Note**: The stable ABI is still under active development. Check the PyTorch docs for the latest API availability.
{% endif %}

## License

MIT License - see LICENSE file for details.

