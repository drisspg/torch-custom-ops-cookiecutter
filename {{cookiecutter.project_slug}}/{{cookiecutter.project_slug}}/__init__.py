"""{{ cookiecutter.project_name }} - {{ cookiecutter.project_description }}"""

{% if cookiecutter.variant == 'cpp' %}
from pathlib import Path

import torch

# Load the compiled C++ library
lib_path = Path(__file__).parent / "lib" / "lib{{ cookiecutter.project_slug }}.so"
torch.ops.load_library(str(lib_path.resolve()))

# Import abstract implementations for torch.compile support
import {{ cookiecutter.project_slug }}.abstract_impls  # noqa: F401

# Create convenient access to ops
ops = torch.ops.{{ cookiecutter.project_slug }}


def add_one(x: torch.Tensor) -> torch.Tensor:
    """Add 1.0 to every element of the input tensor.

    Args:
        x: A CUDA tensor of float32 or bfloat16.

    Returns:
        A new tensor with the same shape and dtype as input, with 1.0 added.
    """
    return ops.add_one(x)
{% else %}
from {{ cookiecutter.project_slug }}.ops import add_one

__all__ = ["add_one"]
{% endif %}

