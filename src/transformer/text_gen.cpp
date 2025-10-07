#include "transformer/text_gen.h"
#include <random>
#include <algorithm>
#include <cmath>
#include <iostream>

TextGen::TextGen(const GPTModel& model) : model(model) {}

std::string TextGen::generate_greedy(const std::vector<int>& prompt_tokens, int max_tokens) {
    std::vector<int> tokens = prompt_tokens;

    for (int i = 0; i < max_tokens; i++) {
        Tensor input_tensor(1, tokens.size());
        for (size_t i = 0; i < tokens.size(); i++) {
            input_tensor.setValue(0, i, tokens[i]);
        }
        auto input = Variable::create(input_tensor, false);
        auto logits_var = model.forward(input, false);
        Tensor logits = logits_var->getData();

        int last_token = tokens.size() - 1;
        Tensor last_token_logits(1, logits.getCols());
        for (int j = 0; j < logits.getCols(); j++) {
            last_token_logits.setValue(0, j, logits.getValue(0, last_token, j));
        }

        int next_token = 0;
        float max_score = last_token_logits.getValue(0, 0);
        for (int j = 1; j < last_token_logits.getCols(); j++) {
            if (last_token_logits.getValue(0, j) > max_score) {
                max_score = last_token_logits.getValue(0, j);
                next_token = j;
            }
        }
        tokens.push_back(next_token);
    }
    return tokens_to_string(tokens);
}

std::string TextGen::generate_sample(const std::vector<int>& prompt_tokens, float temperature, int max_tokens) {
    std::vector<int> tokens = prompt_tokens;

    for (int i = 0; i < max_tokens; i++) {
        Tensor input_tensor(1, tokens.size());
        for (size_t i = 0; i < tokens.size(); i++) {
            input_tensor.setValue(0, i, tokens[i]);
        }
        auto input = Variable::create(input_tensor, false);
        auto logits_var = model.forward(input, false);
        Tensor logits = logits_var->getData();

        int last_token = tokens.size() - 1;
        Tensor last_token_logits(1, logits.getCols());
        for (int j = 0; j < logits.getCols(); j++) {
            last_token_logits.setValue(0, j, logits.getValue(0, last_token, j));
        }

        int next_token = sample_from_logits(last_token_logits, temperature);

        tokens.push_back(next_token);
    }
    return tokens_to_string(tokens);
}

std::string TextGen::tokens_to_string(const std::vector<int>& tokens) {
    std::string result = "";
    result.reserve(tokens.size());
    
    for (size_t i = 0; i < tokens.size(); i++) {
        result += static_cast<char>(tokens[i]);
        if (i < tokens.size() - 1) {
            result += " ";
        }
    }
    return result;
}

int TextGen::sample_from_logits(const Tensor& logits, float temperature) {
    Tensor scaled_logits = logits.scale(1.0f / temperature);
    Tensor probabilities = scaled_logits.softmax();
    
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    float random_val = dis(gen);
    float cumulative = 0.0f;
    
    for (int i = 0; i < probabilities.getCols(); i++) {
        cumulative += probabilities.getValue(0, i);
        if (random_val <= cumulative) {
            return i;
        }
    }
    return probabilities.getCols() - 1;
}