cmake_minimum_required (VERSION 3.1)
project(mini_anns LANGUAGES CXX)

add_library(pyanns INTERFACE)
target_include_directories(pyanns INTERFACE .) 

# set(CXX_STANDARD 17)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

SET(CMAKE_CXX_FLAGS  "-Ofast -lrt -std=c++20 -march=native -fpic -fopenmp -ftree-vectorize -fno-exceptions -fno-rtti" )
# SET(CMAKE_CXX_FLAGS  "-Wall -Wextra -Ofast -lrt -std=c++20 -march=native -fpic -fopenmp -ftree-vectorize -fno-exceptions -fno-rtti" )

add_subdirectory(third_party/helpa)


# add_executable(test1 test.cc)
# target_link_libraries(test1 pyanns)
# target_link_libraries(test1 helpa)