# CMake and Project Setup
cmake_minimum_required(VERSION 3.15...3.31)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(DEFINED SKBUILD)
  project(
    ${SKBUILD_PROJECT_NAME}
    VERSION ${SKBUILD_PROJECT_VERSION}
    LANGUAGES CXX
    DESCRIPTION "Scalar Automatic Differentiation Engine"
  )
  # Setup for Pybind11
  set(PYBIND11_FINDPYTHON ON)
  find_package(pybind11 CONFIG REQUIRED)
  # Define some useful variables
  add_compile_definitions(VERSION_INFO="${SKBUILD_PROJECT_VERSION}")
  add_subdirectory(src)
  install(TARGETS _core DESTINATION ${SKBUILD_PROJECT_NAME})
else()
  project(
    "Nanograd"
    VERSION 0.1.0
    LANGUAGES CXX
    DESCRIPTION "Scalar Automatic Differentiation Engine"
  )
  add_subdirectory(extern/pybind11)
  add_subdirectory(src)
  if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
    include(CTest)
    add_subdirectory(tests)
  endif()
endif()

# For external tools
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")
