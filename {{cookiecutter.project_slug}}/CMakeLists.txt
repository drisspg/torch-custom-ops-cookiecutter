{% if cookiecutter.variant == 'cpp' %}
cmake_minimum_required(VERSION 3.26 FATAL_ERROR)
project(${SKBUILD_PROJECT_NAME}
        VERSION ${SKBUILD_PROJECT_VERSION}
        LANGUAGES CXX CUDA)

# Set the C++ standard for all targets
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable better clangd support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add line info for debugging
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

find_package(Python REQUIRED COMPONENTS Interpreter Development)
execute_process(
    COMMAND "${Python3_EXECUTABLE}" "-c" "import torch;print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE PT_CMAKE_PREFIX
    COMMAND_ECHO STDOUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)

# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES {{ cookiecutter.cuda_arch.replace('.', '') }})
set(TORCH_CUDA_ARCH_LIST "{{ cookiecutter.cuda_arch }}")

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH};${PT_CMAKE_PREFIX})
find_package(Torch REQUIRED CONFIG)

# Source files
file(GLOB_RECURSE CU_SOURCES "src/*.cu")
file(GLOB_RECURSE CPP_SOURCES "src/*.cpp")

add_library({{ cookiecutter.project_slug }} SHARED
    ${CU_SOURCES}
    ${CPP_SOURCES}
)

# Set the library output directory
set_target_properties({{ cookiecutter.project_slug }} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/{{ cookiecutter.project_slug }}/lib"
)

# Include directories
target_include_directories({{ cookiecutter.project_slug }} PUBLIC
    src/include
)

# Link the library to the Torch library
target_link_libraries({{ cookiecutter.project_slug }} PRIVATE ${TORCH_LIBRARIES})

# Install the library to the wheel distribution
install(TARGETS {{ cookiecutter.project_slug }}
    LIBRARY DESTINATION {{ cookiecutter.project_slug }}/lib
)
{% endif %}

