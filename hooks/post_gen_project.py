#!/usr/bin/env python
"""Post-generation hook to cleanup files based on variant selection."""

import os
import shutil

# Get the variant from cookiecutter
variant = "{{ cookiecutter.variant }}"
project_slug = "{{ cookiecutter.project_slug }}"


def remove_file(filepath: str) -> None:
    """Remove a file if it exists."""
    if os.path.isfile(filepath):
        os.remove(filepath)
        print(f"Removed: {filepath}")


def remove_dir(dirpath: str) -> None:
    """Remove a directory if it exists."""
    if os.path.isdir(dirpath):
        shutil.rmtree(dirpath)
        print(f"Removed: {dirpath}")


def main():
    """Remove files not needed for the selected variant."""
    if variant == "python":
        # Remove C++ specific files for Python variant
        remove_file("CMakeLists.txt")
        remove_dir("src")
        remove_file(os.path.join(project_slug, "abstract_impls.py"))
        print("\nGenerated Python/Triton variant project.")
        print("Your Triton kernels are in: {}/ops.py".format(project_slug))

    elif variant == "cpp":
        # Remove Python-only files for C++ variant
        remove_file(os.path.join(project_slug, "ops.py"))
        print("\nGenerated C++/CUDA variant project.")
        print("Your CUDA kernels are in: src/")
        print("Register ops in: src/register_ops.cpp")
        print("Meta functions in: {}/abstract_impls.py".format(project_slug))

    print("\nNext steps:")
    print("  1. cd {{ cookiecutter.project_slug }}")
    print("  2. pip install -e '.[dev]'")
    print("  3. pre-commit install")
    print("  4. pytest test/")


if __name__ == "__main__":
    main()

