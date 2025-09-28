cmake_minimum_required(VERSION 3.16)
project(transformer-from-scratch)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")

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
    src/tokenizer/bpe_tokenizer.cpp
)

target_include_directories(transformer PRIVATE include)