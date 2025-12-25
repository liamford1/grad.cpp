cmake_minimum_required(VERSION 3.18)
project(transformer-from-scratch LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA Architecture - adjust for your GPU
# Common values:
#   75 = RTX 20xx (Turing)
#   80 = A100 (Ampere)
#   86 = RTX 30xx (Ampere)
#   89 = RTX 40xx (Ada Lovelace)
#   90 = H100 (Hopper)
set(CMAKE_CUDA_ARCHITECTURES 75 86 89)  # Multi-arch build

# Enable separable compilation for CUDA (needed for device code in multiple files)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

add_executable(transformer
    src/main.cpp
    src/transformer/Tensor.cu
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
    src/utils/training_utils.cpp
    src/utils/metrics.cpp
    src/training/trainer.cpp
)

add_executable(sanity_tests
    tests/integration/sanity_tests.cpp
    src/transformer/Tensor.cu
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
    src/utils/training_utils.cpp
    src/utils/metrics.cpp
)

# CUDA infrastructure test
add_executable(test_cuda_basic
    test_cuda_basic.cpp
    src/transformer/Tensor.cu
)

# cuBLAS matmul test
add_executable(test_matmul
    test_matmul.cpp
    src/transformer/Tensor.cu
)

target_include_directories(transformer PRIVATE include)
target_include_directories(sanity_tests PRIVATE include)
target_include_directories(test_cuda_basic PRIVATE include)
target_include_directories(test_matmul PRIVATE include)

# ============================================================================
# CUDA LIBRARIES
# ============================================================================
find_package(CUDAToolkit REQUIRED)
target_link_libraries(transformer CUDA::cudart CUDA::cublas)
target_link_libraries(sanity_tests CUDA::cudart CUDA::cublas)
target_link_libraries(test_cuda_basic CUDA::cudart CUDA::cublas)
target_link_libraries(test_matmul CUDA::cudart CUDA::cublas)
message(STATUS "CUDA Version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Include: ${CUDAToolkit_INCLUDE_DIRS}")

# ============================================================================
# BLAS INTEGRATION (Use Apple Accelerate on macOS, fallback for CPU ops)
# ============================================================================
if(APPLE)
    find_library(ACCELERATE_LIBRARY Accelerate)
    target_link_libraries(transformer ${ACCELERATE_LIBRARY})
    target_link_libraries(sanity_tests ${ACCELERATE_LIBRARY})
    add_compile_definitions(ACCELERATE_NEW_LAPACK ACCELERATE_LAPACK_ILP64)
    message(STATUS "Using Apple Accelerate framework for CPU BLAS")
else()
    find_package(BLAS REQUIRED)
    target_link_libraries(transformer ${BLAS_LIBRARIES})
    target_link_libraries(sanity_tests ${BLAS_LIBRARIES})
    message(STATUS "Using OpenBLAS for CPU BLAS")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -DDEBUG")

# Release flags - MAXIMUM PERFORMANCE (CPU)
if(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang")
    # Apple Clang / Clang (for Mac M1/M2/Intel)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -ffast-math -funroll-loops -fno-math-errno")
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
# CUDA COMPILATION FLAGS
# ============================================================================
# Debug
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0")  # -G enables device debugging

# Release - MAXIMUM PERFORMANCE
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math -lineinfo")

# Always enable these for better performance
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")  # For lambdas in kernels

message(STATUS "CUDA Flags: ${CMAKE_CUDA_FLAGS}")

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

# ============================================================================
# TESTING (Optional - only built when explicitly requested)
# ============================================================================

# Enable testing framework
enable_testing()

# Option to build tests (OFF by default for faster builds)
option(BUILD_TESTS "Build test suite" OFF)

if(BUILD_TESTS)
    message(STATUS "Building tests...")
    
    # Test executable for gradient checking
    add_executable(test_gradients
        tests/unit/test_gradients.cpp
        # Reuse all the implementation files (no main.cpp!)
        src/transformer/Tensor.cu
        src/transformer/variable.cpp
        src/transformer/activations.cpp
        src/transformer/linear.cpp
        src/transformer/layer_norm.cpp
        src/transformer/attention.cpp
        src/transformer/multihead_attention.cpp
        src/transformer/feedforward.cpp
        src/transformer/transformer_block.cpp
        src/transformer/token_embedding.cpp
        src/transformer/positional_encoding.cpp
        src/transformer/gpt_model.cpp
        src/transformer/text_gen.cpp
        src/transformer/optimizer.cpp
        src/tokenizer/bpe_tokenizer.cpp
        src/data/dataset.cpp
        src/data/dataloader.cpp
    )

    add_executable(test_dropout
        tests/unit/test_dropout.cpp
        src/transformer/Tensor.cu
        src/transformer/variable.cpp
        src/transformer/activations.cpp
    )

    add_executable(test_attention_bias
        tests/unit/test_attention_bias.cpp
        src/transformer/Tensor.cu
        src/transformer/variable.cpp
        src/transformer/activations.cpp
        src/transformer/multihead_attention.cpp
    )

    add_executable(test_weight_tying
        tests/unit/test_weight_tying.cpp
        src/transformer/Tensor.cu
        src/transformer/variable.cpp
        src/transformer/activations.cpp
        src/transformer/token_embedding.cpp
        src/transformer/positional_encoding.cpp
        src/transformer/layer_norm.cpp
        src/transformer/linear.cpp
        src/transformer/multihead_attention.cpp
        src/transformer/feedforward.cpp
        src/transformer/transformer_block.cpp
        src/transformer/gpt_model.cpp
    )

    add_executable(test_attention_gradients
        tests/unit/test_attention_gradients.cpp
        src/transformer/Tensor.cu
        src/transformer/variable.cpp
        src/transformer/activations.cpp
        src/transformer/linear.cpp
        src/transformer/attention.cpp
        src/transformer/multihead_attention.cpp
    )

    target_include_directories(test_gradients PRIVATE include)
    target_link_libraries(test_gradients CUDA::cudart CUDA::cublas)
    if(APPLE)
        target_link_libraries(test_gradients ${ACCELERATE_LIBRARY})
        target_compile_definitions(test_gradients PRIVATE ACCELERATE_NEW_LAPACK ACCELERATE_LAPACK_ILP64)
    else()
        target_link_libraries(test_gradients ${BLAS_LIBRARIES})
    endif()
    add_test(NAME GradientChecking COMMAND test_gradients)

    target_include_directories(test_dropout PRIVATE include)
    target_link_libraries(test_dropout CUDA::cudart CUDA::cublas)
    if(APPLE)
        target_link_libraries(test_dropout ${ACCELERATE_LIBRARY})
        target_compile_definitions(test_dropout PRIVATE ACCELERATE_NEW_LAPACK ACCELERATE_LAPACK_ILP64)
    else()
        target_link_libraries(test_dropout ${BLAS_LIBRARIES})
    endif()
    add_test(NAME DropoutVerification COMMAND test_dropout)

    target_include_directories(test_attention_bias PRIVATE include)
    target_link_libraries(test_attention_bias CUDA::cudart CUDA::cublas)
    if(APPLE)
        target_link_libraries(test_attention_bias ${ACCELERATE_LIBRARY})
        target_compile_definitions(test_attention_bias PRIVATE ACCELERATE_NEW_LAPACK ACCELERATE_LAPACK_ILP64)
    else()
        target_link_libraries(test_attention_bias ${BLAS_LIBRARIES})
    endif()
    add_test(NAME AttentionBiasVerification COMMAND test_attention_bias)

    target_include_directories(test_weight_tying PRIVATE include)
    target_link_libraries(test_weight_tying CUDA::cudart CUDA::cublas)
    if(APPLE)
        target_link_libraries(test_weight_tying ${ACCELERATE_LIBRARY})
        target_compile_definitions(test_weight_tying PRIVATE ACCELERATE_NEW_LAPACK ACCELERATE_LAPACK_ILP64)
    else()
        target_link_libraries(test_weight_tying ${BLAS_LIBRARIES})
    endif()
    add_test(NAME WeightTyingVerification COMMAND test_weight_tying)

    target_include_directories(test_attention_gradients PRIVATE include)
    target_link_libraries(test_attention_gradients CUDA::cudart CUDA::cublas)
    if(APPLE)
        target_link_libraries(test_attention_gradients ${ACCELERATE_LIBRARY})
        target_compile_definitions(test_attention_gradients PRIVATE ACCELERATE_NEW_LAPACK ACCELERATE_LAPACK_ILP64)
    else()
        target_link_libraries(test_attention_gradients ${BLAS_LIBRARIES})
    endif()
    add_test(NAME AttentionGradientChecking COMMAND test_attention_gradients)
    
    message(STATUS "Test targets added. Run with: make test_gradients && ./test_gradients")
endif()