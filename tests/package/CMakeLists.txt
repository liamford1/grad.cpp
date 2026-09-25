# Minimal downstream project: checks that an installed grad.cpp package is
# usable through find_package alone, including the C++20 requirement it
# exports. CI configures it against the prefix
# from `cmake --install`:
#   cmake -S tests/package -B build-package -DCMAKE_PREFIX_PATH=<prefix>
cmake_minimum_required(VERSION 3.16)
project(grad_package_consumer LANGUAGES CXX)

find_package(grad CONFIG REQUIRED)

add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE grad::core)
