cmake_minimum_required(VERSION 3.16)
project(transformer-from-scratch)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")

add_executable(transformer
    src/main.cpp
    src/tensor.cpp
    src/attention.cpp
    src/multihead_attention.cpp
    src/layer_norm.cpp
)

target_include_directories(transformer PRIVATE src)