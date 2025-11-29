{% if cookiecutter.variant == 'cpp' %}
"""Abstract (meta/fake) implementations for {{ cookiecutter.project_slug }} ops.

These functions define the output shapes and dtypes for torch.compile compatibility.
They are referenced by the C++ TORCH_LIBRARY via impl_abstract_pystub().
"""

import torch
from torch.library import register_fake


@register_fake("{{ cookiecutter.project_slug }}::add_one")
def add_one_meta(input: torch.Tensor) -> torch.Tensor:
    """Meta implementation for add_one - returns tensor with same shape/dtype."""
    return torch.empty_like(input)
{% endif %}

