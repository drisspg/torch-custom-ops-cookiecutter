{% if cookiecutter.variant == 'python' %}
"""Custom ops implemented in pure Python using Triton.

These ops are torch.compile compatible via @torch.library.custom_op decorator.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _add_one_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel that adds 1.0 to each element."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask)
    output = x + 1.0
    tl.store(output_ptr + offsets, output, mask=mask)


@torch.library.custom_op("{{ cookiecutter.project_slug }}::add_one", mutates_args=())
def add_one(input: torch.Tensor) -> torch.Tensor:
    """Add 1.0 to every element of the input tensor.

    Args:
        input: A CUDA tensor of float32 or bfloat16.

    Returns:
        A new tensor with the same shape and dtype as input, with 1.0 added.
    """
    assert input.is_cuda, "Input must be a CUDA tensor"
    assert input.is_contiguous(), "Input must be contiguous"

    output = torch.empty_like(input)
    n_elements = input.numel()

    if n_elements == 0:
        return output

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    _add_one_kernel[grid](
        input,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


@add_one.register_fake
def _(input: torch.Tensor) -> torch.Tensor:
    """Meta/fake implementation for torch.compile compatibility."""
    return torch.empty_like(input)
{% endif %}

