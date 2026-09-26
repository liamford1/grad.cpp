cmake_minimum_required(VERSION 3.16)
project(grad-cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
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

# Optimization flags only; warnings are per target (grad_enable_warnings
# below) so they apply in every build type, including the sanitizer build,
# and never leak into a consumer of the installed package.
if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG ${GRAD_ARCH_FLAGS} -funroll-loops -fno-math-errno")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # -flto=auto rather than -flto: same code, but LTRANS runs in parallel
    # instead of lto-wrapper warning that it fell back to serial.
    # -ffat-lto-objects keeps real object code next to the LTO IR, so the
    # installed libgrad.a also links into consumers that do not use LTO or
    # use a different GCC.
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG ${GRAD_ARCH_FLAGS} -funroll-loops -fno-math-errno -flto=auto -ffat-lto-objects")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Oi /Ot /GL /DNDEBUG")
endif()

option(GRAD_WARNINGS_AS_ERRORS "Treat compiler warnings as errors in grad.cpp's own targets" OFF)

function(grad_enable_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4)
        if(GRAD_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE /WX)
        endif()
    else()
        # COMPILE_LANGUAGE covers the Objective-C++ Metal backend as well.
        #
        # -Wconversion and -Wsign-conversion hold the size convention in
        # place: extents are size_t, hyperparameters int, and the only
        # narrowing to a BLAS integer is in blas_wrapper.h. GCC's -Wshadow
        # also flags parameters and locals that shadow a data member;
        # Clang's -Wshadow does not, so Clang gets -Wshadow-all, which
        # adds that check for constructors and lambdas and keeps a clean
        # AppleClang build clean under GCC as well. -Wold-style-cast is
        # nearly free here (the code had one C-style cast); the one system
        # macro that expands to one, MAP_FAILED, is fenced in token_file.cpp.
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            set(grad_shadow_flag -Wshadow-all)
        else()
            set(grad_shadow_flag -Wshadow)
        endif()
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX,OBJCXX>:-Wall -Wextra -Wpedantic
                -Wconversion -Wsign-conversion ${grad_shadow_flag} -Wold-style-cast>)
        if(GRAD_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX,OBJCXX>:-Werror>)
        endif()
    endif()
endfunction()

# The Metal backend: the threshold GEMM offload in CPU mode and the
# device-resident execution mode (docs/design/metal-resident.md). Off, the
# library links stubs and metal_mode() can never become true, which is how
# every non-Apple build works.
if(APPLE)
    option(GRAD_METAL_BACKEND "Build the Metal GPU backend" ON)
else()
    set(GRAD_METAL_BACKEND OFF)
endif()

if(GRAD_METAL_BACKEND)
    enable_language(OBJCXX)
    set(CMAKE_OBJCXX_STANDARD 20)
    set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
    set(METAL_BACKEND_SRC src/transformer/metal_backend.mm src/transformer/metal_ops.mm)
    # ARC owns the Objective-C buffers, encoders and kernels the backend creates.
    set_source_files_properties(${METAL_BACKEND_SRC} PROPERTIES COMPILE_OPTIONS "-fobjc-arc")

    # The kernels are compiled at runtime (newLibraryWithSource), so the
    # build needs no offline Metal toolchain. The source is embedded as a
    # byte array rather than a string literal, which -Wpedantic would cap
    # at 65,536 characters. Editing the .metal file reruns this step.
    set(GRAD_METAL_KERNELS ${CMAKE_CURRENT_SOURCE_DIR}/src/transformer/metal_kernels.metal)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${GRAD_METAL_KERNELS})
    file(READ ${GRAD_METAL_KERNELS} grad_metal_hex HEX)
    string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," grad_metal_bytes "${grad_metal_hex}")
    set(GRAD_GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
    file(WRITE ${GRAD_GENERATED_DIR}/metal_kernels.inc.tmp
        "// Generated from src/transformer/metal_kernels.metal; do not edit.\n"
        "static const unsigned char kMetalKernelSource[] = {${grad_metal_bytes} 0x00};\n")
    configure_file(${GRAD_GENERATED_DIR}/metal_kernels.inc.tmp
                   ${GRAD_GENERATED_DIR}/metal_kernels.inc COPYONLY)
else()
    set(METAL_BACKEND_SRC src/transformer/metal_stubs.cpp)
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
    src/transformer/device.cpp
    src/transformer/metal_graph.cpp
    src/tokenizer/bpe_tokenizer.cpp
    src/tokenizer/pretokenize.cpp
    src/tokenizer/byte_bpe.cpp
    src/tokenizer/tokenizer.cpp
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
grad_enable_warnings(grad_core)
# Public, so the requirement travels with the exported target to consumers.
target_compile_features(grad_core PUBLIC cxx_std_20)
target_include_directories(grad_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
find_package(Threads REQUIRED)
target_link_libraries(grad_core PUBLIC Threads::Threads)
if(GRAD_METAL_BACKEND)
    # Public: metal_backend.h declares the real API or inline stubs on it,
    # so a consumer of the installed package must see the same choice.
    target_compile_definitions(grad_core PUBLIC GRAD_HAS_METAL=1)
    target_include_directories(grad_core PRIVATE ${GRAD_GENERATED_DIR})
endif()

if(APPLE)
    find_library(ACCELERATE_LIBRARY Accelerate)
    find_library(METAL_LIBRARY Metal)
    find_library(MPS_LIBRARY MetalPerformanceShaders)
    find_library(FOUNDATION_LIBRARY Foundation)
    if(NOT ACCELERATE_LIBRARY OR NOT METAL_LIBRARY OR NOT MPS_LIBRARY OR NOT FOUNDATION_LIBRARY)
        message(FATAL_ERROR "Required Apple Accelerate/Metal frameworks were not found")
    endif()
    # Linked by name, not by the found paths: the paths point into this
    # machine's SDK and would be baked into the installed gradTargets.cmake.
    target_link_libraries(grad_core PUBLIC
        "-framework Accelerate"
        "-framework Metal"
        "-framework MetalPerformanceShaders"
        "-framework Foundation"
    )
    target_compile_definitions(grad_core PUBLIC
        ACCELERATE_NEW_LAPACK
        ACCELERATE_LAPACK_ILP64
    )
    message(STATUS "Using Apple Accelerate and Metal")
else()
    find_package(BLAS REQUIRED)
    # BLAS::BLAS (CMake 3.18+) exports as a target that gradConfig.cmake
    # re-finds on the consumer's machine; BLAS_LIBRARIES would export this
    # machine's absolute library paths.
    if(TARGET BLAS::BLAS)
        target_link_libraries(grad_core PUBLIC BLAS::BLAS)
    else()
        target_link_libraries(grad_core PUBLIC ${BLAS_LIBRARIES})
    endif()
    message(STATUS "Using system BLAS")
endif()

# One source per subcommand; main.cpp is the dispatch table.
add_executable(grad
    src/main.cpp
    src/cli/args.cpp
    src/cli/common.cpp
    src/cli/presets.cpp
    src/cli/presets_command.cpp
    src/cli/prepare.cpp
    src/cli/train.cpp
    src/cli/generate.cpp
    src/cli/chat.cpp
    src/cli/eval.cpp
    src/cli/bench.cpp
    src/cli/watch.cpp
)
# src/ for the executable's private headers (cli/commands.h and friends).
target_include_directories(grad PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(grad PRIVATE grad::core)
grad_enable_warnings(grad)

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
        grad_enable_warnings(${target})
        add_test(NAME ${test_name} COMMAND ${target})
        set_tests_properties(${test_name} PROPERTIES LABELS "unit" TIMEOUT 120)
    endfunction()

    add_grad_test(test_gradients tests/unit/test_gradients.cpp GradientChecking)
    add_grad_test(test_model_gradients tests/unit/test_model_gradients.cpp ModelGradientChecking)
    add_grad_test(test_metal_matmul tests/unit/test_metal_matmul.cpp MetalMatmulVsCPU)
    add_grad_test(test_metal_stream tests/unit/test_metal_stream.cpp MetalStream)
    add_grad_test(test_metal_kernels tests/unit/test_metal_kernels.cpp MetalKernelsVsCPU)
    add_grad_test(test_dropout tests/unit/test_dropout.cpp DropoutVerification)
    add_grad_test(test_attention_bias tests/unit/test_attention_bias.cpp AttentionBiasVerification)
    add_grad_test(test_weight_tying tests/unit/test_weight_tying.cpp WeightTyingVerification)
    add_grad_test(test_attention_gradients tests/unit/test_attention_gradients.cpp AttentionGradientChecking)
    add_grad_test(test_subtract tests/unit/test_subtract.cpp TensorSubtraction)
    add_grad_test(test_core_regressions tests/unit/test_core_regressions.cpp CoreRegressions)
    add_grad_test(test_tokenizer tests/unit/test_tokenizer.cpp Tokenizer)
    # v1 against ids recorded from the pre-v2 binary, using the TinyStories
    # 16000 cache that the 70M checkpoint was trained with; skipped (77)
    # when that cache is not there.
    set(GRAD_V1_TOKENIZER_CACHE
        "${CMAKE_CURRENT_SOURCE_DIR}/data/tinystories.txt.tokenizer_16000.cache"
        CACHE FILEPATH "v1 tokenizer cache for the TokenizerV1Cache test")
    add_test(NAME TokenizerV1Cache COMMAND test_tokenizer v1-cache ${GRAD_V1_TOKENIZER_CACHE})
    set_tests_properties(TokenizerV1Cache PROPERTIES SKIP_RETURN_CODE 77 LABELS "unit" TIMEOUT 120)

    if(MSVC)
        target_compile_options(test_subtract PRIVATE /UNDEBUG)
    else()
        target_compile_options(test_subtract PRIVATE -UNDEBUG)
    endif()
    set_tests_properties(MetalMatmulVsCPU MetalStream MetalKernelsVsCPU
                         PROPERTIES SKIP_RETURN_CODE 77 LABELS "unit;hardware")

    add_executable(sanity_tests tests/integration/sanity_tests.cpp)
    target_link_libraries(sanity_tests PRIVATE grad::core)
    grad_enable_warnings(sanity_tests)
    add_test(NAME SanityOverfit COMMAND sanity_tests overfit)
    add_test(NAME InferenceParityGPT2 COMMAND sanity_tests parity-gpt2)
    add_test(NAME InferenceParityModern COMMAND sanity_tests parity-modern)
    add_test(NAME SanityDataLoader COMMAND sanity_tests dataloader)
    add_test(NAME FileFormatValidation COMMAND sanity_tests file-formats)
    set_tests_properties(
        SanityOverfit InferenceParityGPT2 InferenceParityModern
        SanityDataLoader FileFormatValidation
        PROPERTIES LABELS "integration" TIMEOUT 120
    )
    # `sanity_tests benchmark` is a timing run on data/shakespeare.txt, not
    # a check (it has no pass/fail criterion), so it is not registered.

    message(STATUS "Tests enabled: run ctest --test-dir ${CMAKE_BINARY_DIR} --output-on-failure")

    # Thread-count determinism: the 1-thread run writes the reference hash
    # of a short seeded training run, the 4-thread run must reproduce it bit
    # for bit. The fixture orders them and skips the comparison if the
    # reference run failed.
    add_executable(determinism_probe tests/integration/determinism_probe.cpp)
    target_link_libraries(determinism_probe PRIVATE grad::core)
    grad_enable_warnings(determinism_probe)
    set(GRAD_DETERMINISM_REF ${CMAKE_CURRENT_BINARY_DIR}/determinism_threads1.txt)
    add_test(NAME DeterminismThreads1
             COMMAND determinism_probe ${GRAD_DETERMINISM_REF})
    add_test(NAME DeterminismThreads4
             COMMAND determinism_probe ${CMAKE_CURRENT_BINARY_DIR}/determinism_threads4.txt
                     ${GRAD_DETERMINISM_REF})
    set_tests_properties(DeterminismThreads1 PROPERTIES
        ENVIRONMENT GRAD_THREADS=1
        FIXTURES_SETUP determinism_reference
        LABELS "integration" TIMEOUT 120)
    set_tests_properties(DeterminismThreads4 PROPERTIES
        ENVIRONMENT GRAD_THREADS=4
        FIXTURES_REQUIRED determinism_reference
        LABELS "integration" TIMEOUT 120)

    add_grad_test(test_grad_mode tests/unit/test_grad_mode.cpp GradMode)

    # The argument parser belongs to the grad executable, not grad::core,
    # so its test compiles the one source it covers.
    add_grad_test(test_args tests/unit/test_args.cpp ArgumentParsing)
    target_sources(test_args PRIVATE src/cli/args.cpp)
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
# grad::core's public headers install to <prefix>/include/grad/, so consumers
# write #include "grad/transformer/tensor.h". include/cli is the grad
# executable's parser, not part of the library's API, and is not installed.
install(DIRECTORY include/grad DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
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
