# CMake and Project Setup
cmake_minimum_required(VERSION 3.15...3.31)
project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX
  DESCRIPTION "Scalar Automatic Differentiation Engine"
)

# Setup for Pybind11
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

# Adding code
pybind11_add_module(_core MODULE src/main.cpp)
install(TARGETS _core DESTINATION ${SKBUILD_PROJECT_NAME})
