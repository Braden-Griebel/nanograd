add_library(nanograd_core engine.h engine.cpp nn.h nn.cpp)
set_target_properties(nanograd_core PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(nanograd_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})