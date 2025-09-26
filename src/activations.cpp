#include "transformer/activations.h"
#include <cmath>
#include <random>

Tensor gelu(const Tensor& input) {
    Tensor result(input.getRows(), input.getCols());
    for (int i = 0; i < input.getRows(); i++) {
        for (int j = 0; j < input.getCols(); j++) {
            float x = input.getValue(i, j);
            float gelu_val = 0.5f * x * (1.0f + tanh(0.79788456f * (x + 0.044715f * x * x * x)));
            result.setValue(i, j, gelu_val);
        }
    }
    return result;
}

Tensor dropout(const Tensor& input, float dropout_rate, bool training) {
    if (dropout_rate == 0.0f || !training) { return input; }

    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    Tensor result(input.getRows(), input.getCols());
    float scale = 1.0f / (1.0f - dropout_rate);

    for (int i = 0; i < input.getRows(); i++) {
        for (int j = 0; j < input.getCols(); j++) {
            if (dis(gen) > dropout_rate) {
                result.setValue(i, j, input.getValue(i, j) * scale);
            } else {
                result.setValue(i, j, 0.0f);
            }
        }
    }
    return result;
}