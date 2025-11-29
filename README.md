# Torch Custom Ops Cookiecutter

A [Cookiecutter](https://github.com/cookiecutter/cookiecutter) template for creating `torch.compile`-compatible custom operators in PyTorch.

## Features

- **Two Variants**:
  - **C++/CUDA**: Custom ops with `TORCH_LIBRARY` bindings and scikit-build-core
  - **Python/Triton**: Pure Python ops with `@torch.library.custom_op`

- **torch.compile Compatible**: Both variants include proper meta/fake function registration for full `torch.compile` support with no graph breaks.

- **Modern Tooling**:
  - Pre-commit hooks (ruff, pyupgrade, pyrefly)
  - GitHub Actions (CI linting, PyPI publishing)
  - pytest with `torch.library.opcheck` tests

## Usage

```bash
# Install cookiecutter
pip install cookiecutter

# Generate a new project
cookiecutter gh:drisspg/torch-custom-ops-cookiecutter

# Or from local clone
cookiecutter /path/to/torch-custom-ops-cookiecutter
```

## After Generation

Once you've generated your project:

```bash
cd my_custom_ops

# 1. Build and install
pip install -e ".[dev]"

# 2. Set up pre-commit hooks
pre-commit install

# 3. Run tests to verify everything works
pytest test/ -v

# 4. Replace the example `add_one` op with your implementation
#    See the generated README.md for detailed instructions
```

### What's Included

| Feature | Description |
|---------|-------------|
| **Example op** | Working `add_one` op to use as a template |
| **Tests** | pytest tests with `torch.library.opcheck` validation |
| **Pre-commit** | ruff, pyupgrade, pyrefly hooks |
| **CI** | GitHub Actions for linting + PyPI publishing |
| **torch.compile** | Meta functions for full compile support |

## Template Options

| Option | Description | Default |
|--------|-------------|---------|
| `project_name` | Human-readable project name | "My Custom Ops" |
| `project_slug` | Python package name (auto-generated) | Based on project_name |
| `variant` | `cpp` (C++/CUDA) or `python` (Triton) | `cpp` |
| `cuda_arch` | CUDA compute capability | `9.0` (Hopper) |
| `min_torch_version` | Minimum PyTorch version | `2.7` |

## Generated Project Structure

### C++ Variant

```
my_custom_ops/
├── CMakeLists.txt          # CMake build config
├── pyproject.toml          # scikit-build-core config
├── src/
│   ├── include/
│   │   └── add_one.h       # C++ header declaring the op
│   ├── add_one.cu          # CUDA kernel implementation
│   └── register_ops.cpp    # TORCH_LIBRARY registration
├── my_custom_ops/
│   ├── __init__.py         # Loads .so, exposes Python API
│   └── abstract_impls.py   # @register_fake for torch.compile
└── test/
    └── test_add_one.py     # Tests with opcheck
```

**Workflow:** Write CUDA kernel → Register with `TORCH_LIBRARY` → Add meta function in Python → Rebuild with `pip install -e .`

### Python Variant

```
my_custom_ops/
├── pyproject.toml          # hatchling config
├── my_custom_ops/
│   ├── __init__.py         # Exports ops
│   └── ops.py              # @custom_op + Triton kernels
└── test/
    └── test_add_one.py     # Tests with opcheck
```

**Workflow:** Write Triton kernel → Decorate with `@torch.library.custom_op` → Add `.register_fake` → No rebuild needed!

## What You Get

After generating and installing your project:

```python
import torch
from my_custom_ops import add_one

# Use your custom op
x = torch.randn(32, 32, device="cuda")
y = add_one(x)

# Access via torch.ops namespace
y = torch.ops.my_custom_ops.add_one(x)

# Works with torch.compile (no graph breaks!)
@torch.compile(fullgraph=True)
def fn(x):
    return add_one(x)
```

## References

- [Custom C++ and CUDA Operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)
- [Custom Python Operators](https://docs.pytorch.org/tutorials/advanced/python_custom_ops.html)
- [torch.library documentation](https://pytorch.org/docs/stable/library.html)
- [LibTorch Stable ABI](https://docs.pytorch.org/docs/main/notes/libtorch_stable_abi.html) - For cross-version ABI compatibility (PyTorch 2.10+)

## License

MIT

