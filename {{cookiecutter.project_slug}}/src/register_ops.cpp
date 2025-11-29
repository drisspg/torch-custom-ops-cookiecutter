// Custom operator registration using TORCH_LIBRARY
// See: https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html
//
// For cross-version ABI stability (PyTorch 2.10+), consider migrating to
// STABLE_TORCH_LIBRARY. See: https://docs.pytorch.org/docs/main/notes/libtorch_stable_abi.html

#include <c10/core/DispatchKey.h>
#include <torch/library.h>

#include "include/add_one.h"

TORCH_LIBRARY({{ cookiecutter.project_slug }}, m) {
  // Point to Python module for abstract (meta) implementations
  // This enables torch.compile compatibility
  m.impl_abstract_pystub("{{ cookiecutter.project_slug }}.abstract_impls");

  // Define and implement add_one op
  m.def("add_one(Tensor input) -> Tensor");
  m.impl("add_one", c10::DispatchKey::CUDA, TORCH_FN({{ cookiecutter.project_slug }}::add_one));
}

