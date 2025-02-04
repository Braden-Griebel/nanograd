add_library(nanograd_core STATIC engine.h engine.cpp nn.h nn.cpp
        nanograd_random.h)
target_include_directories(nanograd_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
