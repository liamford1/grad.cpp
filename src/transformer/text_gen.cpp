#include "transformer/text_gen.h"
#include <random>
#include <algorithm>
#include <cmath>
#include <iostream>

TextGen::TextGen(const GPTModel& model, const BPETokenizer* tok) : model(model), tokenizer(tok) {}

std::string TextGen::generate_greedy(const std::vector<int>& prompt_tokens, int max_tokens, float repetition_penalty) {
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

        bool is_3d = logits.getIs3D();
        for (size_t j = 0; j < logits.getCols(); j++) {
            float logit_value = is_3d ? logits.getValue(0, last_token, j) : logits.getValue(last_token, j);
            last_token_logits.setValue(0, j, logit_value);
        }

        if (repetition_penalty != 1.0f) {
            int window_size = std::min(50, (int)tokens.size());
            for (int k = tokens.size() - window_size; k < (int)tokens.size(); k++) {
                int token_id = tokens[k];
                float current_logit = last_token_logits.getValue(0, token_id);
                if (current_logit > 0) {
                    last_token_logits.setValue(0, token_id, current_logit / repetition_penalty);
                } else {
                    last_token_logits.setValue(0, token_id, current_logit * repetition_penalty);
                }
            }
        }

        int next_token = 0;
        float max_score = last_token_logits.getValue(0, 0);
        for (size_t j = 1; j < last_token_logits.getCols(); j++) {
            if (last_token_logits.getValue(0, j) > max_score) {
                max_score = last_token_logits.getValue(0, j);
                next_token = j;
            }
        }
        tokens.push_back(next_token);
    }
    return tokens_to_string(tokens);
}

std::string TextGen::generate_sample(const std::vector<int>& prompt_tokens, float temperature, int max_tokens, float repetition_penalty, int top_k, float top_p) {
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

        bool is_3d = logits.getIs3D();
        for (size_t j = 0; j < logits.getCols(); j++) {
            float logit_value = is_3d ? logits.getValue(0, last_token, j) : logits.getValue(last_token, j);
            last_token_logits.setValue(0, j, logit_value);
        }

        if (repetition_penalty != 1.0f) {
            int window_size = std::min(50, (int)tokens.size());
            for (int k = tokens.size() - window_size; k < (int)tokens.size(); k++) {
                int token_id = tokens[k];
                float current_logit = last_token_logits.getValue(0, token_id);
                if (current_logit > 0) {
                    last_token_logits.setValue(0, token_id, current_logit / repetition_penalty);
                } else {
                    last_token_logits.setValue(0, token_id, current_logit * repetition_penalty);
                }
            }
        }

        int next_token = sample_from_logits(last_token_logits, temperature, top_k, top_p);

        tokens.push_back(next_token);
    }
    return tokens_to_string(tokens);
}

std::string TextGen::tokens_to_string(const std::vector<int>& tokens) {
    if (tokenizer != nullptr) {
        return tokenizer->decode(tokens);
    }
    
    std::string result = "";
    result.reserve(tokens.size());
    for (size_t i = 0; i < tokens.size(); i++) {
        result += static_cast<char>(tokens[i]);
    }
    return result;
}

int TextGen::sample_from_logits(const Tensor& logits, float temperature, int top_k, float top_p) {
    Tensor scaled_logits = logits.scale(1.0f / temperature);

    if (top_k > 0 && top_k < (int)scaled_logits.getCols()) {
        std::vector<std::pair<float, int>> logit_pairs;
        for (size_t i = 0; i < scaled_logits.getCols(); i++) {
            logit_pairs.push_back({scaled_logits.getValue(0, i), i});
        }

        std::sort(logit_pairs.begin(), logit_pairs.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });

        float min_logit = logit_pairs[top_k - 1].first;
        for (size_t i = 0; i < scaled_logits.getCols(); i++) {
            if (scaled_logits.getValue(0, i) < min_logit) {
                scaled_logits.setValue(0, i, -1e10f);
            }
        }
    }

    Tensor probabilities = scaled_logits.softmax();

    if (top_p < 1.0f) {
        std::vector<std::pair<float, int>> prob_pairs;
        for (size_t i = 0; i < probabilities.getCols(); i++) {
            prob_pairs.push_back({probabilities.getValue(0, i), i});
        }
        std::sort(prob_pairs.begin(), prob_pairs.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });

        float cumulative = 0.0f;
        size_t cutoff_idx = 0;
        for (size_t i = 0; i < prob_pairs.size(); i++) {
            cumulative += prob_pairs[i].first;
            if (cumulative >= top_p) {
                cutoff_idx = i;
                break;
            }
        }

        float cutoff_prob = prob_pairs[cutoff_idx].first;
        float total_kept = 0.0f;
        for (size_t i = 0; i < probabilities.getCols(); i++) {
            float p = probabilities.getValue(0, i);
            if (p < cutoff_prob) {
                probabilities.setValue(0, i, 0.0f);
            } else {
                total_kept += p;
            }
        }

        if (total_kept > 0.0f) {
            for (size_t i = 0; i < probabilities.getCols(); i++) {
                probabilities.setValue(0, i, probabilities.getValue(0, i) / total_kept);
            }
        }
    }

    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    float random_val = dis(gen);
    float cumulative = 0.0f;

    for (size_t i = 0; i < probabilities.getCols(); i++) {
        cumulative += probabilities.getValue(0, i);
        if (random_val <= cumulative) {
            return i;
        }
    }
    return probabilities.getCols() - 1;
}