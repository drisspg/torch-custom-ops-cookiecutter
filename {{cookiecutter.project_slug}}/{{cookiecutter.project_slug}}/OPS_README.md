# Custom Ops Implementation Guide

This directory contains the Python implementation of your custom operators.

{% if cookiecutter.variant == 'cpp' %}
## For C++ Variant

Your ops are implemented in C++/CUDA in the `src/` directory. This directory contains:

- `__init__.py` - Loads the compiled library and exposes Python wrappers
- `abstract_impls.py` - Meta functions for `torch.compile` compatibility

See `src/README.md` for the full C++ implementation guide.

### Updating `abstract_impls.py`

Every op registered in C++ needs a corresponding `@register_fake` function here:

```python
from torch.library import register_fake

@register_fake("{{ cookiecutter.project_slug }}::my_new_op")
def my_new_op_meta(input: torch.Tensor, arg: float) -> torch.Tensor:
    # Return a tensor with the correct shape/dtype/device
    # This is called during torch.compile tracing
    return torch.empty_like(input)
```

{% else %}
## Directory Structure

```
{{ cookiecutter.project_slug }}/
├── __init__.py      # Exports your ops
├── ops.py           # Triton kernels + @custom_op registration
└── OPS_README.md    # This file
```

## How Python Custom Ops Work

```
┌─────────────────┐     ┌──────────────────────────┐
│  Python Call    │────▶│  @torch.library.custom_op │
│  my_op(x)       │     │  decorated function       │
└─────────────────┘     └──────────────────────────┘
                                    │
         ┌──────────────────────────┼───────────────────────────┐
         ▼                          ▼                           ▼
┌─────────────────┐     ┌──────────────────────┐     ┌──────────────────┐
│  Eager Mode     │     │  torch.compile       │     │  torch.ops.*     │
│  Runs Triton    │     │  Uses .register_fake │     │  namespace       │
│  kernel         │     │  for tracing         │     │  access          │
└─────────────────┘     └──────────────────────┘     └──────────────────┘
```

## Step-by-Step: Adding a New Op

### Step 1: Write the Triton Kernel

Add to `ops.py`:

```python
import triton
import triton.language as tl


@triton.jit
def _my_op_kernel(
    input_ptr,
    output_ptr,
    scalar_arg,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel for my_op.

    Each program instance processes BLOCK_SIZE elements.
    """
    # Get program ID (which block we're processing)
    pid = tl.program_id(axis=0)

    # Calculate offsets for this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Create mask for bounds checking
    mask = offsets < n_elements

    # Load input data
    x = tl.load(input_ptr + offsets, mask=mask)

    # Perform computation
    result = x * scalar_arg

    # Store output
    tl.store(output_ptr + offsets, result, mask=mask)
```

### Step 2: Create the Custom Op Wrapper

Add to `ops.py`:

```python
@torch.library.custom_op("{{ cookiecutter.project_slug }}::my_op", mutates_args=())
def my_op(input: torch.Tensor, scalar_arg: float) -> torch.Tensor:
    """Apply my_op to the input tensor.

    Args:
        input: Input CUDA tensor (must be contiguous)
        scalar_arg: Scalar multiplier

    Returns:
        Result tensor with same shape as input
    """
    # Input validation
    assert input.is_cuda, "Input must be a CUDA tensor"
    assert input.is_contiguous(), "Input must be contiguous"

    # Allocate output
    output = torch.empty_like(input)
    n_elements = input.numel()

    # Handle empty tensors
    if n_elements == 0:
        return output

    # Configure kernel launch
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Launch kernel
    _my_op_kernel[grid](
        input,
        output,
        scalar_arg,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output
```

### Step 3: Register the Fake (Meta) Implementation

Add immediately after the `@custom_op` function:

```python
@my_op.register_fake
def _(input: torch.Tensor, scalar_arg: float) -> torch.Tensor:
    """Meta implementation for torch.compile.

    This function is called during tracing to determine output
    shape/dtype without actually running the kernel.
    """
    return torch.empty_like(input)
```

**Important:** The fake function must:
- Have the same signature as the real function
- Return tensor(s) with correct shape, dtype, device, and strides
- NOT perform any actual computation

### Step 4: Export from `__init__.py`

```python
from {{ cookiecutter.project_slug }}.ops import add_one, my_op

__all__ = ["add_one", "my_op"]
```

### Step 5: Add Tests

Create `test/test_my_op.py`:

```python
import torch
import pytest
from {{ cookiecutter.project_slug }} import my_op


def test_my_op_correctness():
    """Test basic correctness."""
    x = torch.randn(32, 32, device="cuda")
    result = my_op(x, 2.0)
    expected = x * 2.0
    torch.testing.assert_close(result, expected)


def test_my_op_opcheck():
    """Verify operator was registered correctly.

    opcheck tests:
    - Fake kernel produces correct metadata
    - Operator works with torch.compile
    - Autograd works (if registered)
    """
    examples = [
        (torch.randn(32, device="cuda"), 2.0),
        (torch.randn(32, 32, device="cuda"), -1.0),
        (torch.randn(8, 16, 32, device="cuda", dtype=torch.float16), 0.5),
    ]
    for args in examples:
        torch.library.opcheck(
            torch.ops.{{ cookiecutter.project_slug }}.my_op,
            args,
        )


def test_my_op_compile_no_graph_break():
    """Verify op works with torch.compile without graph breaks."""
    x = torch.randn(32, 32, device="cuda")

    @torch.compile(fullgraph=True)  # fullgraph=True errors on graph breaks
    def fn(x):
        return my_op(x, 2.0)

    result = fn(x)
    expected = x * 2.0
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("shape", [(10,), (32, 32), (8, 16, 32)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_my_op_shapes_and_dtypes(shape, dtype):
    """Test various shapes and dtypes."""
    x = torch.randn(shape, device="cuda", dtype=dtype)
    result = my_op(x, 2.0)
    expected = x * 2.0
    torch.testing.assert_close(result, expected)
```

## Common Patterns

### Ops with Multiple Outputs

```python
@torch.library.custom_op("{{ cookiecutter.project_slug }}::my_multi_out", mutates_args=())
def my_multi_out(input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out1 = torch.empty_like(input)
    out2 = torch.empty_like(input)
    # ... kernel launch ...
    return out1, out2


@my_multi_out.register_fake
def _(input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(input), torch.empty_like(input)
```

### In-Place Ops (Mutating)

```python
@torch.library.custom_op("{{ cookiecutter.project_slug }}::my_inplace", mutates_args=("input",))
def my_inplace(input: torch.Tensor) -> None:
    """Modifies input in-place."""
    # ... kernel that writes to input ...
    pass


@my_inplace.register_fake
def _(input: torch.Tensor) -> None:
    return None
```

### Ops with Optional Arguments

```python
from typing import Optional

@torch.library.custom_op("{{ cookiecutter.project_slug }}::my_optional", mutates_args=())
def my_optional(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if scale is None:
        scale = torch.ones(1, device=input.device)
    # ... kernel launch ...
    return output
```

### Autotuned Kernels

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def _my_autotuned_kernel(
    input_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # ... kernel code ...
    pass
```

## Debugging Tips

1. **Kernel crashes?**
   - Check mask logic for out-of-bounds access
   - Verify tensor is contiguous
   - Print shapes: `print(f"{input.shape=}, {input.stride()=}")`

2. **Wrong results?**
   - Compare against PyTorch reference: `torch.testing.assert_close(result, expected)`
   - Test with small tensors you can manually verify

3. **torch.compile fails?**
   - Ensure `.register_fake` returns correct shape/dtype
   - Use `@torch.compile(fullgraph=True)` to catch graph breaks
   - Check with `torch.library.opcheck(...)`

4. **Performance issues?**
   - Use `@triton.autotune` to find optimal block sizes
   - Profile with `torch.profiler` or Nsight

## References

- [PyTorch Custom Python Operators](https://docs.pytorch.org/tutorials/advanced/python_custom_ops.html)
- [Triton Documentation](https://triton-lang.org/main/index.html)
- [torch.library.custom_op](https://pytorch.org/docs/stable/library.html#torch.library.custom_op)
- [torch.library.opcheck](https://pytorch.org/docs/stable/library.html#torch.library.opcheck)
{% endif %}

