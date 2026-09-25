# Minimal downstream project: checks that an installed grad.cpp package is
# usable through find_package alone. CI configures it against the prefix
# from `cmake --install`:
#   cmake -S tests/package -B build-package -DCMAKE_PREFIX_PATH=<prefix>
cmake_minimum_required(VERSION 3.16)
project(grad_package_consumer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(grad CONFIG REQUIRED)

add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE grad::core)
