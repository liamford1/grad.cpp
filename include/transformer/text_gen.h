#ifndef TEXT_GEN_H
#define TEXT_GEN_H

#include "tensor.h"
#include "gpt_model.h"
#include <string>
#include <vector>

class TextGen {
    private:
        const GPTModel& model;

        int sample_from_logits(const Tensor& logits, float temperature = 1.0f);
        std::string tokens_to_string(const std::vector<int>& tokens);
    public:
        TextGen(const GPTModel& model);

        std::string generate_greedy(const std::vector<int>& prompt_tokens, int max_tokens = 50);
        std::string generate_sample(const std::vector<int>& prompt_tokens, float temperature = 1.0f, int max_tokens = 50);
};


#endif