# CMake and Project Setup
cmake_minimum_required(VERSION 3.15...3.31)
project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX
  DESCRIPTION "Scalar Automatic Differentiation Engine"
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Setup for Pybind11
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

# Adding code
add_subdirectory(src)
install(TARGETS _core DESTINATION ${SKBUILD_PROJECT_NAME})

# For external tools
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")
