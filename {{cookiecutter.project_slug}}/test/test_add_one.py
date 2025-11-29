"""Tests for the add_one custom op.

Uses torch.library.opcheck to verify correct operator registration.
See: https://docs.pytorch.org/tutorials/advanced/python_custom_ops.html#testing-python-custom-operators
"""

import pytest
import torch

from {{ cookiecutter.project_slug }} import add_one


def reference_add_one(x: torch.Tensor) -> torch.Tensor:
    """Reference implementation for testing."""
    return x + 1.0


@pytest.mark.parametrize("shape", [(10,), (32, 32), (8, 16, 32), (1,), (1024,)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_add_one_correctness(shape: tuple, dtype: torch.dtype):
    """Test that add_one produces correct results."""
    x = torch.randn(shape, device="cuda", dtype=dtype)

    result = add_one(x)
    expected = reference_add_one(x)

    torch.testing.assert_close(result, expected)


def test_add_one_opcheck():
    """Test operator registration using torch.library.opcheck.

    opcheck verifies the operator was registered correctly with proper
    FakeTensor (meta) kernels for torch.compile compatibility.

    From PyTorch docs: "Use torch.library.opcheck to test that the custom
    operator was registered correctly."
    """
    # Create a variety of example inputs to test against
    # Include different shapes, dtypes, and requires_grad settings
    examples = [
        (torch.randn(32, device="cuda"),),
        (torch.randn(32, 32, device="cuda"),),
        (torch.randn(8, 16, 32, device="cuda", dtype=torch.bfloat16),),
        (torch.randn(64, 64, device="cuda", dtype=torch.float32, requires_grad=True),),
    ]

    for example in examples:
        torch.library.opcheck(
            torch.ops.{{ cookiecutter.project_slug }}.add_one,
            example,
        )


def test_add_one_compile():
    """Test that add_one works with torch.compile."""
    x = torch.randn(32, 32, device="cuda", dtype=torch.float32)

    @torch.compile(fullgraph=True)
    def fn(x):
        return add_one(x)

    result = fn(x)
    expected = reference_add_one(x)

    torch.testing.assert_close(result, expected)


def test_add_one_compile_no_graph_break():
    """Verify torch.compile doesn't induce graph breaks.

    fullgraph=True will raise an error if a graph break occurs.
    """
    x = torch.randn(64, 64, device="cuda")

    @torch.compile(fullgraph=True)
    def fn(x):
        y = add_one(x)
        z = add_one(y)
        return z

    result = fn(x)
    expected = x + 2.0

    torch.testing.assert_close(result, expected)


def test_add_one_empty_tensor():
    """Test add_one with an empty tensor."""
    x = torch.empty(0, device="cuda", dtype=torch.float32)
    result = add_one(x)
    assert result.shape == x.shape
    assert result.numel() == 0

