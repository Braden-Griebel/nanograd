# Adding code
pybind11_add_module(_core MODULE nanograd_bg_bind.cpp)
add_subdirectory(nanograd_core)
target_link_libraries(_core PUBLIC nanograd_core)
