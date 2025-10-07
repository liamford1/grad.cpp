cmake_minimum_required(VERSION 3.16)
project(transformer-from-scratch)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(transformer
    src/main.cpp
    src/transformer/tensor.cpp
    src/transformer/attention.cpp
    src/transformer/multihead_attention.cpp
    src/transformer/layer_norm.cpp
    src/transformer/linear.cpp
    src/transformer/feedforward.cpp
    src/transformer/transformer_block.cpp
    src/transformer/token_embedding.cpp
    src/transformer/positional_encoding.cpp
    src/transformer/gpt_model.cpp
    src/transformer/activations.cpp
    src/transformer/text_gen.cpp
    src/transformer/variable.cpp
    src/transformer/optimizer.cpp
    src/tokenizer/bpe_tokenizer.cpp
    src/data/dataset.cpp
    src/data/dataloader.cpp
)

target_include_directories(transformer PRIVATE include)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -DDEBUG")

# Release flags - MAXIMUM PERFORMANCE
if(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang")
    # Apple Clang / Clang (for Mac M1/M2/Intel)
    set(CMAKE_CXX_FLAGS_RELEASE 
        "-O3 -DNDEBUG -march=native -ffast-math -funroll-loops -fno-math-errno"
    )
    message(STATUS "Using Clang optimizations for Mac")
    
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # GCC (for Linux)
    set(CMAKE_CXX_FLAGS_RELEASE 
        "-O3 -DNDEBUG -march=native -ffast-math -funroll-loops -fno-math-errno -flto"
    )
    message(STATUS "Using GCC optimizations")
    
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # MSVC (for Windows)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Oi /Ot /GL /DNDEBUG")
    message(STATUS "Using MSVC optimizations")
endif()

# ============================================================================
# OPTIONAL: Link-Time Optimization (LTO) for extra 5-10% speedup
# ============================================================================
# Uncomment these lines if you want LTO (increases compile time)
# include(CheckIPOSupported)
# check_ipo_supported(RESULT ipo_supported OUTPUT ipo_output)
# if(ipo_supported)
#     set_property(TARGET transformer PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
#     message(STATUS "Link-Time Optimization (LTO) enabled")
# else()
#     message(STATUS "LTO not supported: ${ipo_output}")
# endif()

# ============================================================================
# PLATFORM-SPECIFIC OPTIMIZATIONS
# ============================================================================

# Detect Mac architecture (M1/M2 vs Intel)
if(APPLE)
    execute_process(
        COMMAND uname -m
        OUTPUT_VARIABLE MACOS_ARCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    if(MACOS_ARCH STREQUAL "arm64")
        message(STATUS "Detected Apple Silicon (M1/M2) - using ARM NEON")
        # ARM NEON is enabled automatically with -march=native
    else()
        message(STATUS "Detected Intel Mac - using AVX/SSE")
        # AVX/SSE enabled automatically with -march=native
    endif()
endif()

# ============================================================================
# OPTIONAL: Build different configurations
# ============================================================================

# Fast build (for development - quick compile)
add_custom_target(fast
    COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release -S ${CMAKE_SOURCE_DIR} -B ${CMAKE_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} -j
    COMMENT "Building with optimizations enabled"
)

# Debug build (for debugging)
add_custom_target(debug
    COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug -S ${CMAKE_SOURCE_DIR} -B ${CMAKE_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} -j
    COMMENT "Building with debug symbols"
)

# ============================================================================
# Print configuration info
# ============================================================================
message(STATUS "")
message(STATUS "=== Build Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Release flags: ${CMAKE_CXX_FLAGS_RELEASE}")
else()
    message(STATUS "Debug flags: ${CMAKE_CXX_FLAGS_DEBUG}")
endif()
message(STATUS "===========================")
message(STATUS "")
