#pragma once

#include "grad/transformer/tensor.h"
#include "grad/transformer/gpt_model.h"
#include "grad/tokenizer/bpe_tokenizer.h"
#include <functional>
#include <string>
#include <vector>

class TextGen {
    private:
        const GPTModel& model;
        const BPETokenizer* tokenizer;

        int sample_from_logits(const Tensor& logits, float temperature = 1.0f, int top_k = 0, float top_p = 1.0f);
        std::string tokens_to_string(const std::vector<int>& tokens);
    public:
        TextGen(const GPTModel& model, const BPETokenizer* tok = nullptr);

        std::string generate_greedy(const std::vector<int>& prompt_tokens, int max_tokens = 50, float repetition_penalty = 1.2f);
        std::string generate_sample(const std::vector<int>& prompt_tokens, float temperature = 1.0f, int max_tokens = 50, float repetition_penalty = 1.2f, int top_k = 0, float top_p = 1.0f);

        // Like generate_sample, but invokes on_text with each token's text
        // as soon as it is sampled (for interactive streaming output).
        // Returns the number of tokens generated.
        int generate_stream(const std::vector<int>& prompt_tokens,
                            const std::function<void(const std::string&)>& on_text,
                            float temperature = 1.0f, int max_tokens = 50,
                            float repetition_penalty = 1.2f, int top_k = 0, float top_p = 1.0f);
};
