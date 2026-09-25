cmake_minimum_required(VERSION 3.16)
project(grad-cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# -march=native bakes in the build machine's ISA, so a binary or an installed
# grad::core built with it will SIGILL on an older CPU. It is on by default
# because every number in BENCHMARKS.md was measured with it; turn it off
# (-DGRAD_NATIVE_ARCH=OFF) for anything you intend to ship or distribute.
option(GRAD_NATIVE_ARCH "Tune the build for this machine's CPU (-march=native)" ON)
set(GRAD_ARCH_FLAGS "")
if(GRAD_NATIVE_ARCH AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(GRAD_ARCH_FLAGS "-march=native")
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -DDEBUG")
if(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -Wall -Wextra ${GRAD_ARCH_FLAGS} -funroll-loops -fno-math-errno")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -Wall -Wextra ${GRAD_ARCH_FLAGS} -funroll-loops -fno-math-errno -flto")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Oi /Ot /GL /DNDEBUG /W4")
endif()

if(APPLE)
    enable_language(OBJCXX)
    set(CMAKE_OBJCXX_STANDARD 17)
    set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
    set(METAL_BACKEND_SRC src/transformer/metal_backend.mm)
    # ARC owns the Objective-C buffers and kernels allocated by every GPU GEMM.
    set_source_files_properties(${METAL_BACKEND_SRC} PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
else()
    set(METAL_BACKEND_SRC "")
endif()

set(GRAD_CORE_SOURCES
    src/transformer/tensor.cpp
    ${METAL_BACKEND_SRC}
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

add_library(grad_core STATIC ${GRAD_CORE_SOURCES})
add_library(grad::core ALIAS grad_core)
set_target_properties(grad_core PROPERTIES OUTPUT_NAME grad EXPORT_NAME core)
target_include_directories(grad_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
find_package(Threads REQUIRED)
target_link_libraries(grad_core PUBLIC Threads::Threads)

if(APPLE)
    find_library(ACCELERATE_LIBRARY Accelerate)
    find_library(METAL_LIBRARY Metal)
    find_library(MPS_LIBRARY MetalPerformanceShaders)
    find_library(FOUNDATION_LIBRARY Foundation)
    if(NOT ACCELERATE_LIBRARY OR NOT METAL_LIBRARY OR NOT MPS_LIBRARY OR NOT FOUNDATION_LIBRARY)
        message(FATAL_ERROR "Required Apple Accelerate/Metal frameworks were not found")
    endif()
    target_link_libraries(grad_core PUBLIC
        ${ACCELERATE_LIBRARY}
        ${METAL_LIBRARY}
        ${MPS_LIBRARY}
        ${FOUNDATION_LIBRARY}
    )
    target_compile_definitions(grad_core PUBLIC
        ACCELERATE_NEW_LAPACK
        ACCELERATE_LAPACK_ILP64
    )
    message(STATUS "Using Apple Accelerate and Metal")
else()
    find_package(BLAS REQUIRED)
    target_link_libraries(grad_core PUBLIC ${BLAS_LIBRARIES})
    message(STATUS "Using system BLAS")
endif()

add_executable(grad src/main.cpp)
target_link_libraries(grad PRIVATE grad::core)

find_package(Git QUIET)
set(GRAD_GIT_SHA "unknown")
if(Git_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE GRAD_GIT_SHA
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    execute_process(
        COMMAND ${GIT_EXECUTABLE} status --porcelain
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE GRAD_GIT_STATUS
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(GRAD_GIT_STATUS)
        string(APPEND GRAD_GIT_SHA "-dirty")
    endif()
endif()
target_compile_definitions(grad PRIVATE
    GRAD_VERSION="${PROJECT_VERSION}"
    GRAD_GIT_SHA="${GRAD_GIT_SHA}"
    GRAD_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
    GRAD_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
    GRAD_SYSTEM="${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}"
)

add_custom_target(fast
    COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release -S ${CMAKE_SOURCE_DIR} -B ${CMAKE_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} -j
    COMMENT "Building with optimizations enabled"
)
add_custom_target(debug
    COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug -S ${CMAKE_SOURCE_DIR} -B ${CMAKE_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} -j
    COMMENT "Building with debug symbols"
)

option(BUILD_TESTS "Build test suite" OFF)
if(BUILD_TESTS)
    enable_testing()

    function(add_grad_test target source test_name)
        add_executable(${target} ${source})
        target_link_libraries(${target} PRIVATE grad::core)
        add_test(NAME ${test_name} COMMAND ${target})
        set_tests_properties(${test_name} PROPERTIES LABELS "unit" TIMEOUT 120)
    endfunction()

    add_grad_test(test_gradients tests/unit/test_gradients.cpp GradientChecking)
    add_grad_test(test_model_gradients tests/unit/test_model_gradients.cpp ModelGradientChecking)
    add_grad_test(test_metal_matmul tests/unit/test_metal_matmul.cpp MetalMatmulVsCPU)
    add_grad_test(test_dropout tests/unit/test_dropout.cpp DropoutVerification)
    add_grad_test(test_attention_bias tests/unit/test_attention_bias.cpp AttentionBiasVerification)
    add_grad_test(test_weight_tying tests/unit/test_weight_tying.cpp WeightTyingVerification)
    add_grad_test(test_attention_gradients tests/unit/test_attention_gradients.cpp AttentionGradientChecking)
    add_grad_test(test_subtract tests/unit/test_subtract.cpp TensorSubtraction)
    add_grad_test(test_core_regressions tests/unit/test_core_regressions.cpp CoreRegressions)

    if(MSVC)
        target_compile_options(test_subtract PRIVATE /UNDEBUG)
    else()
        target_compile_options(test_subtract PRIVATE -UNDEBUG)
    endif()
    set_tests_properties(MetalMatmulVsCPU PROPERTIES SKIP_RETURN_CODE 77 LABELS "unit;hardware")

    add_executable(sanity_tests tests/integration/sanity_tests.cpp)
    target_link_libraries(sanity_tests PRIVATE grad::core)
    add_test(NAME SanityOverfit COMMAND sanity_tests overfit)
    add_test(NAME InferenceParityGPT2 COMMAND sanity_tests parity-gpt2)
    add_test(NAME InferenceParityModern COMMAND sanity_tests parity-modern)
    set_tests_properties(
        SanityOverfit InferenceParityGPT2 InferenceParityModern
        PROPERTIES LABELS "integration" TIMEOUT 120
    )

    message(STATUS "Tests enabled: run ctest --test-dir ${CMAKE_BINARY_DIR} --output-on-failure")
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
configure_package_config_file(
    cmake/gradConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/gradConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/grad
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/gradConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(TARGETS grad grad_core
    EXPORT gradTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT gradTargets
    FILE gradTargets.cmake
    NAMESPACE grad::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/grad
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/gradConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/gradConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/grad
)

message(STATUS "grad.cpp ${PROJECT_VERSION}: ${CMAKE_BUILD_TYPE}, ${CMAKE_CXX_COMPILER_ID}")
