cmake_minimum_required (VERSION 2.6)
project(gengt
    LANGUAGES CXX)

# SET( CMAKE_CXX_FLAGS  "-Ofast -lrt -DNDEBUG -std=c++11 -DHAVE_CXX0X -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0" )

SET(CMAKE_CXX_FLAGS  "-Ofast -lrt -DNDEBUG  -std=c++20 -march=native -fpic -fopenmp -ftree-vectorize -fno-exceptions -fno-rtti" )

add_subdirectory(mini_anns)

add_executable(test 
gen_gt.cpp
mini_rnn/RNNDescent.cpp
)

target_link_libraries(test pyanns)
target_link_libraries(test helpa)