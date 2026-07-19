cmake_minimum_required(VERSION 3.16)
project(grad-cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Metal backend (Apple only): Objective-C++ source compiled into every
# target, since blas_wrapper.h routes large matmuls through it.
if(APPLE)
    enable_language(OBJCXX)
    set(CMAKE_OBJCXX_STANDARD 17)
    set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
    set(METAL_BACKEND_SRC src/transformer/metal_backend.mm)
    # ARC is required for correctness, not style: sgemm alloc/new-creates
    # seven Objective-C objects per call (buffers, matrices, MPS kernel),
    # and without ARC those +1 references leak on every large matmul -
    # including no-copy MTLBuffers that keep GPU mappings alive on tensor
    # pages the CPU side has already freed.
    set_source_files_properties(${METAL_BACKEND_SRC} PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
else()
    set(METAL_BACKEND_SRC "")
endif()

add_executable(grad
    src/main.cpp
    src/transformer/tensor.cpp
    ${METAL_BACKEND_SRC}
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
    src/transformer/inference.cpp
    src/transformer/variable.cpp
    src/transformer/optimizer.cpp
    src/tokenizer/bpe_tokenizer.cpp
    src/data/dataset.cpp
    src/data/token_file.cpp
    src/data/dataloader.cpp
    src/utils/training_utils.cpp
    src/utils/metrics.cpp
    src/utils/dashboard.cpp
    src/training/trainer.cpp
)

add_executable(sanity_tests
    tests/integration/sanity_tests.cpp
    src/transformer/tensor.cpp
    ${METAL_BACKEND_SRC}
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
    src/transformer/inference.cpp
    src/transformer/variable.cpp
    src/transformer/optimizer.cpp
    src/tokenizer/bpe_tokenizer.cpp
    src/data/dataset.cpp
    src/data/token_file.cpp
    src/data/dataloader.cpp
    src/utils/training_utils.cpp
    src/utils/metrics.cpp
)

target_include_directories(grad PRIVATE include)
target_include_directories(sanity_tests PRIVATE include)

# ============================================================================
# BLAS INTEGRATION (Use Apple Accelerate on macOS)
# ============================================================================
if(APPLE)
    find_library(ACCELERATE_LIBRARY Accelerate)
    find_library(METAL_LIBRARY Metal)
    find_library(MPS_LIBRARY MetalPerformanceShaders)
    find_library(FOUNDATION_LIBRARY Foundation)
    set(APPLE_GPU_LIBS ${METAL_LIBRARY} ${MPS_LIBRARY} ${FOUNDATION_LIBRARY})
    target_link_libraries(grad ${ACCELERATE_LIBRARY} ${APPLE_GPU_LIBS})
    target_link_libraries(sanity_tests ${ACCELERATE_LIBRARY} ${APPLE_GPU_LIBS})
    add_compile_definitions(ACCELERATE_NEW_LAPACK ACCELERATE_LAPACK_ILP64)
    message(STATUS "Using Apple Accelerate framework for BLAS")
else()
    find_package(BLAS REQUIRED)
    target_link_libraries(grad ${BLAS_LIBRARIES})
    target_link_libraries(sanity_tests ${BLAS_LIBRARIES})
    message(STATUS "Using OpenBLAS for BLAS")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -DDEBUG")

# Release flags - MAXIMUM PERFORMANCE
if(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang")
    # Apple Clang / Clang (for Mac M1/M2/Intel)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -funroll-loops -fno-math-errno")
    message(STATUS "Using Clang optimizations for Mac")
    
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # GCC (for Linux)
    set(CMAKE_CXX_FLAGS_RELEASE 
        "-O3 -DNDEBUG -march=native -funroll-loops -fno-math-errno -flto"
    )
    message(STATUS "Using GCC optimizations")
    
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # MSVC (for Windows)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Oi /Ot /GL /DNDEBUG")
    message(STATUS "Using MSVC optimizations")
endif()

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
        src/transformer/tensor.cpp
        ${METAL_BACKEND_SRC}
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
        src/transformer/inference.cpp
        src/transformer/optimizer.cpp
        src/tokenizer/bpe_tokenizer.cpp
        src/data/dataset.cpp
        src/data/token_file.cpp
        src/data/dataloader.cpp
    )

    add_executable(test_model_gradients
        tests/unit/test_model_gradients.cpp
        src/transformer/tensor.cpp
        ${METAL_BACKEND_SRC}
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
        src/transformer/inference.cpp
        src/transformer/optimizer.cpp
        src/tokenizer/bpe_tokenizer.cpp
        src/data/dataset.cpp
        src/data/token_file.cpp
        src/data/dataloader.cpp
    )

    add_executable(test_metal_matmul
        tests/unit/test_metal_matmul.cpp
        src/transformer/tensor.cpp
        ${METAL_BACKEND_SRC}
    )

    add_executable(test_dropout
        tests/unit/test_dropout.cpp
        src/transformer/tensor.cpp
        ${METAL_BACKEND_SRC}
        src/transformer/variable.cpp
        src/transformer/activations.cpp
    )

    add_executable(test_attention_bias
        tests/unit/test_attention_bias.cpp
        src/transformer/tensor.cpp
        ${METAL_BACKEND_SRC}
        src/transformer/variable.cpp
        src/transformer/activations.cpp
        src/transformer/multihead_attention.cpp
    )

    add_executable(test_weight_tying
        tests/unit/test_weight_tying.cpp
        src/transformer/tensor.cpp
        ${METAL_BACKEND_SRC}
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
        src/transformer/tensor.cpp
        ${METAL_BACKEND_SRC}
        src/transformer/variable.cpp
        src/transformer/activations.cpp
        src/transformer/linear.cpp
        src/transformer/attention.cpp
        src/transformer/multihead_attention.cpp
    )

    target_include_directories(test_gradients PRIVATE include)
    if(APPLE)
        target_link_libraries(test_gradients ${ACCELERATE_LIBRARY} ${APPLE_GPU_LIBS})
        target_compile_definitions(test_gradients PRIVATE ACCELERATE_NEW_LAPACK ACCELERATE_LAPACK_ILP64)
    else()
        target_link_libraries(test_gradients ${BLAS_LIBRARIES})
    endif()
    add_test(NAME GradientChecking COMMAND test_gradients)

    target_include_directories(test_model_gradients PRIVATE include)
    if(APPLE)
        target_link_libraries(test_model_gradients ${ACCELERATE_LIBRARY} ${APPLE_GPU_LIBS})
        target_compile_definitions(test_model_gradients PRIVATE ACCELERATE_NEW_LAPACK ACCELERATE_LAPACK_ILP64)
    else()
        target_link_libraries(test_model_gradients ${BLAS_LIBRARIES})
    endif()
    add_test(NAME ModelGradientChecking COMMAND test_model_gradients)

    target_include_directories(test_metal_matmul PRIVATE include)
    if(APPLE)
        target_link_libraries(test_metal_matmul ${ACCELERATE_LIBRARY} ${APPLE_GPU_LIBS})
        target_compile_definitions(test_metal_matmul PRIVATE ACCELERATE_NEW_LAPACK ACCELERATE_LAPACK_ILP64)
    else()
        target_link_libraries(test_metal_matmul ${BLAS_LIBRARIES})
    endif()
    add_test(NAME MetalMatmulVsCPU COMMAND test_metal_matmul)

    target_include_directories(test_dropout PRIVATE include)
    if(APPLE)
        target_link_libraries(test_dropout ${ACCELERATE_LIBRARY} ${APPLE_GPU_LIBS})
        target_compile_definitions(test_dropout PRIVATE ACCELERATE_NEW_LAPACK ACCELERATE_LAPACK_ILP64)
    else()
        target_link_libraries(test_dropout ${BLAS_LIBRARIES})
    endif()
    add_test(NAME DropoutVerification COMMAND test_dropout)

    target_include_directories(test_attention_bias PRIVATE include)
    if(APPLE)
        target_link_libraries(test_attention_bias ${ACCELERATE_LIBRARY} ${APPLE_GPU_LIBS})
        target_compile_definitions(test_attention_bias PRIVATE ACCELERATE_NEW_LAPACK ACCELERATE_LAPACK_ILP64)
    else()
        target_link_libraries(test_attention_bias ${BLAS_LIBRARIES})
    endif()
    add_test(NAME AttentionBiasVerification COMMAND test_attention_bias)

    target_include_directories(test_weight_tying PRIVATE include)
    if(APPLE)
        target_link_libraries(test_weight_tying ${ACCELERATE_LIBRARY} ${APPLE_GPU_LIBS})
        target_compile_definitions(test_weight_tying PRIVATE ACCELERATE_NEW_LAPACK ACCELERATE_LAPACK_ILP64)
    else()
        target_link_libraries(test_weight_tying ${BLAS_LIBRARIES})
    endif()
    add_test(NAME WeightTyingVerification COMMAND test_weight_tying)

    target_include_directories(test_attention_gradients PRIVATE include)
    if(APPLE)
        target_link_libraries(test_attention_gradients ${ACCELERATE_LIBRARY} ${APPLE_GPU_LIBS})
        target_compile_definitions(test_attention_gradients PRIVATE ACCELERATE_NEW_LAPACK ACCELERATE_LAPACK_ILP64)
    else()
        target_link_libraries(test_attention_gradients ${BLAS_LIBRARIES})
    endif()
    add_test(NAME AttentionGradientChecking COMMAND test_attention_gradients)
    
    message(STATUS "Test targets added. Run with: make test_gradients && ./test_gradients")
endif()
