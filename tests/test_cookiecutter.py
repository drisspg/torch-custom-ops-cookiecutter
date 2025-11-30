"""Tests for the torch-custom-ops-cookiecutter template."""

import subprocess
import sys


def test_python_variant_generation(cookies):
    """Test that the Python/Triton variant generates correctly."""
    result = cookies.bake(extra_context={"variant": "python", "project_name": "Test Ops"})

    assert result.exit_code == 0, f"Bake failed: {result.exception}"
    assert result.project_path.is_dir()
    assert result.project_path.name == "test_ops"

    # Check expected files exist
    assert (result.project_path / "pyproject.toml").is_file()
    assert (result.project_path / "README.md").is_file()
    assert (result.project_path / "test_ops" / "__init__.py").is_file()
    assert (result.project_path / "test_ops" / "ops.py").is_file()
    assert (result.project_path / "test" / "test_add_one.py").is_file()

    # C++ files should NOT exist for Python variant
    assert not (result.project_path / "CMakeLists.txt").exists()
    assert not (result.project_path / "src").exists()
    assert not (result.project_path / "test_ops" / "abstract_impls.py").exists()


def test_cpp_variant_generation(cookies):
    """Test that the C++/CUDA variant generates correctly."""
    result = cookies.bake(extra_context={"variant": "cpp", "project_name": "Test Ops"})

    assert result.exit_code == 0, f"Bake failed: {result.exception}"
    assert result.project_path.is_dir()
    assert result.project_path.name == "test_ops"

    # Check expected files exist
    assert (result.project_path / "pyproject.toml").is_file()
    assert (result.project_path / "README.md").is_file()
    assert (result.project_path / "CMakeLists.txt").is_file()
    assert (result.project_path / "test_ops" / "__init__.py").is_file()
    assert (result.project_path / "test_ops" / "abstract_impls.py").is_file()
    assert (result.project_path / "src" / "add_one.cu").is_file()
    assert (result.project_path / "src" / "register_ops.cpp").is_file()
    assert (result.project_path / "test" / "test_add_one.py").is_file()

    # Python-only files should NOT exist for C++ variant
    assert not (result.project_path / "test_ops" / "ops.py").exists()


def test_python_variant_syntax(cookies):
    """Test that generated Python code has valid syntax."""
    result = cookies.bake(extra_context={"variant": "python", "project_name": "Syntax Test"})

    assert result.exit_code == 0

    # Check Python files for syntax errors
    python_files = list(result.project_path.rglob("*.py"))
    assert len(python_files) > 0, "No Python files generated"

    for py_file in python_files:
        proc = subprocess.run(
            [sys.executable, "-m", "py_compile", str(py_file)],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, f"Syntax error in {py_file}: {proc.stderr}"


def test_cpp_variant_syntax(cookies):
    """Test that generated Python code in C++ variant has valid syntax."""
    result = cookies.bake(extra_context={"variant": "cpp", "project_name": "Syntax Test"})

    assert result.exit_code == 0

    # Check Python files for syntax errors
    python_files = list(result.project_path.rglob("*.py"))
    assert len(python_files) > 0, "No Python files generated"

    for py_file in python_files:
        proc = subprocess.run(
            [sys.executable, "-m", "py_compile", str(py_file)],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, f"Syntax error in {py_file}: {proc.stderr}"


def test_pyproject_toml_valid(cookies):
    """Test that pyproject.toml is valid TOML for both variants."""
    import tomllib

    for variant in ["python", "cpp"]:
        result = cookies.bake(extra_context={"variant": variant, "project_name": "TOML Test"})

        assert result.exit_code == 0

        pyproject = result.project_path / "pyproject.toml"
        assert pyproject.is_file()

        with open(pyproject, "rb") as f:
            try:
                data = tomllib.load(f)
            except tomllib.TOMLDecodeError as e:
                raise AssertionError(f"Invalid TOML in {variant} variant: {e}")

        # Check required keys
        assert "project" in data
        assert "name" in data["project"]
        assert "build-system" in data


