add_subdirectory(nanograd_core)

pybind11_add_module(_core MODULE nanograd_bind.cpp)
target_link_libraries(_core PUBLIC nanograd_core)
