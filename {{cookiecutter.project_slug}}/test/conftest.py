"""Pytest configuration for {{ cookiecutter.project_slug }} tests."""

import pytest
import torch


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "cuda: mark test as requiring CUDA")


@pytest.fixture(autouse=True)
def skip_without_cuda():
    """Skip tests that require CUDA if not available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

