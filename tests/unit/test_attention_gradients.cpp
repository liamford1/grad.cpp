#include "transformer/multihead_attention.h"
#include "transformer/variable.h"
#include "transformer/tensor.h"
#include <iostream>
#include <cmath>
#include <memory>

// Numerical gradient computation for 2D tensors
float numerical_gradient_2d(
    MultiHeadAttention& attention,
    std::shared_ptr<Variable> input,
    int input_idx_i, int input_idx_j,
    float epsilon = 1e-3f
) {
    // Forward with input + epsilon
    float original = input->getData().getValue(input_idx_i, input_idx_j);

    input->getData().setValue(input_idx_i, input_idx_j, original + epsilon);
    auto output_plus = attention.forward(input, false);

    // Compute scalar loss (sum of all outputs)
    float loss_plus = 0.0f;
    int seq_len = output_plus->getData().getRows();
    int d_model = output_plus->getData().getCols();
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < d_model; j++) {
            loss_plus += output_plus->getData().getValue(i, j);
        }
    }

    // Forward with input - epsilon
    input->getData().setValue(input_idx_i, input_idx_j, original - epsilon);
    auto output_minus = attention.forward(input, false);

    // Compute scalar loss
    float loss_minus = 0.0f;
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < d_model; j++) {
            loss_minus += output_minus->getData().getValue(i, j);
        }
    }

    // Restore original value
    input->getData().setValue(input_idx_i, input_idx_j, original);

    // Numerical gradient
    return (loss_plus - loss_minus) / (2.0f * epsilon);
}

// Numerical gradient computation for 3D tensors
float numerical_gradient_3d(
    MultiHeadAttention& attention,
    std::shared_ptr<Variable> input,
    int input_idx_i, int input_idx_j, int input_idx_k,
    float epsilon = 1e-3f
) {
    // Forward with input + epsilon
    float original = input->getData().getValue(input_idx_i, input_idx_j, input_idx_k);

    try {
        input->getData().setValue(input_idx_i, input_idx_j, input_idx_k, original + epsilon);
        auto output_plus = attention.forward(input, false);

        // Compute scalar loss (sum of all outputs)
        float loss_plus = 0.0f;
        int batch_size = output_plus->getData().getBatchSize();
        int seq_len = output_plus->getData().getRows();
        int d_model = output_plus->getData().getCols();
        for (int b = 0; b < batch_size; b++) {
            for (int i = 0; i < seq_len; i++) {
                for (int j = 0; j < d_model; j++) {
                    loss_plus += output_plus->getData().getValue(b, i, j);
                }
            }
        }

        // Forward with input - epsilon
        input->getData().setValue(input_idx_i, input_idx_j, input_idx_k, original - epsilon);
        auto output_minus = attention.forward(input, false);

        // Compute scalar loss
        float loss_minus = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            for (int i = 0; i < seq_len; i++) {
                for (int j = 0; j < d_model; j++) {
                    loss_minus += output_minus->getData().getValue(b, i, j);
                }
            }
        }

        // Restore original value
        input->getData().setValue(input_idx_i, input_idx_j, input_idx_k, original);

        // Numerical gradient
        return (loss_plus - loss_minus) / (2.0f * epsilon);

    } catch (const std::exception& e) {
        std::cerr << "\n!!! ERROR in numerical gradient computation !!!" << std::endl;
        std::cerr << "Exception: " << e.what() << std::endl;
        std::cerr << "  param index: [" << input_idx_i << "," << input_idx_j << "," << input_idx_k << "]" << std::endl;
        std::cerr << "  input shape: (" << input->getData().getBatchSize() << ","
                  << input->getData().getRows() << "," << input->getData().getCols() << ")" << std::endl;

        // Restore original value before rethrowing
        input->getData().setValue(input_idx_i, input_idx_j, input_idx_k, original);
        throw;
    }
}

int main() {
    std::cout << "=== MULTIHEAD ATTENTION GRADIENT CHECK ===" << std::endl;

    // Test configuration
    int d_model = 64;
    int num_heads = 4;
    int seq_len = 8;
    int batch_size = 2;
    float dropout_rate = 0.0f;  // Disable dropout for gradient checking

    int tests_passed = 0;
    int tests_total = 0;

    // Test 1: 2D (non-batched) - multiple points
    std::cout << "\n--- Test 1: 2D (non-batched) ---" << std::endl;
    {
        // Create attention module
        MultiHeadAttention attention(d_model, num_heads, dropout_rate);

        Tensor input_data(seq_len, d_model);
        input_data.xavier(seq_len, d_model);
        auto input = Variable::create(input_data, true);

        // Forward pass
        auto output = attention.forward(input, false);

        // Create a scalar loss by manually summing
        float sum = 0.0f;
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < d_model; j++) {
                sum += output->getData().getValue(i, j);
            }
        }

        // Create scalar variable for loss
        Tensor loss_tensor(1, 1);
        loss_tensor.setValue(0, 0, sum);
        auto loss = Variable::create(loss_tensor, true);

        // Set up backward connection manually
        loss->addChild(output);
        loss->setBackwardFn([output, seq_len, d_model]() {
            // Gradient of sum: all elements get gradient 1.0
            for (int i = 0; i < seq_len; i++) {
                for (int j = 0; j < d_model; j++) {
                    output->getGrad().setValue(i, j, 1.0f);
                }
            }
        });

        // Backward pass
        loss->backward();

        // Check gradients at multiple points
        std::vector<std::pair<int, int>> test_points = {
            {0, 0}, {0, d_model-1}, {seq_len-1, 0}, {seq_len-1, d_model-1},
            {seq_len/2, d_model/2}
        };

        for (const auto& point : test_points) {
            int i = point.first;
            int j = point.second;

            float analytical_grad = input->getGrad().getValue(i, j);
            float numerical_grad = numerical_gradient_2d(attention, input, i, j);
            float rel_error = std::abs(analytical_grad - numerical_grad) /
                             (std::abs(analytical_grad) + std::abs(numerical_grad) + 1e-8f);

            std::cout << "Input gradient [" << i << "," << j << "]:" << std::endl;
            std::cout << "  Analytical: " << analytical_grad << std::endl;
            std::cout << "  Numerical:  " << numerical_grad << std::endl;
            std::cout << "  Rel Error:  " << rel_error << std::endl;

            tests_total++;
            if (rel_error < 3e-2f) {  // 3% tolerance for complex attention gradients
                std::cout << "  ✓ PASS" << std::endl;
                tests_passed++;
            } else {
                std::cout << "  ✗ FAIL" << std::endl;
            }
        }
    }

    // Test 2: 3D (batched) - multiple points
    std::cout << "\n--- Test 2: 3D (batched) ---" << std::endl;
    {
        // Create attention module
        MultiHeadAttention attention(d_model, num_heads, dropout_rate);

        Tensor input_data(batch_size, seq_len, d_model);
        input_data.xavier(seq_len, d_model);
        auto input = Variable::create(input_data, true);

        // Forward pass
        auto output = attention.forward(input, false);

        // Create a scalar loss by manually summing
        float sum = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            for (int i = 0; i < seq_len; i++) {
                for (int j = 0; j < d_model; j++) {
                    sum += output->getData().getValue(b, i, j);
                }
            }
        }

        // Create scalar variable for loss
        Tensor loss_tensor(1, 1);
        loss_tensor.setValue(0, 0, sum);
        auto loss = Variable::create(loss_tensor, true);

        // Set up backward connection manually
        loss->addChild(output);
        loss->setBackwardFn([output, batch_size, seq_len, d_model]() {
            // Gradient of sum: all elements get gradient 1.0
            for (int b = 0; b < batch_size; b++) {
                for (int i = 0; i < seq_len; i++) {
                    for (int j = 0; j < d_model; j++) {
                        output->getGrad().setValue(b, i, j, 1.0f);
                    }
                }
            }
        });

        try {
            // Backward pass
            loss->backward();
        } catch (const std::exception& e) {
            std::cerr << "\n!!! ERROR during backward pass !!!" << std::endl;
            std::cerr << "Exception: " << e.what() << std::endl;
            throw;
        }

        // Check gradients at multiple points
        std::vector<std::tuple<int, int, int>> test_points = {
            {0, 0, 0}, {0, 0, d_model-1}, {0, seq_len-1, 0},
            {batch_size-1, seq_len-1, d_model-1}, {batch_size-1, seq_len/2, d_model/2}
        };

        for (size_t pt_idx = 0; pt_idx < test_points.size(); pt_idx++) {
            const auto& point = test_points[pt_idx];
            int b = std::get<0>(point);
            int i = std::get<1>(point);
            int j = std::get<2>(point);

            std::cout << "\n[Test Point " << (pt_idx + 1) << "/" << test_points.size() << "]" << std::endl;
            std::cout << "Checking gradient at input[" << b << "," << i << "," << j << "]" << std::endl;

            try {
                float analytical_grad = input->getGrad().getValue(b, i, j);
                std::cout << "  Analytical gradient: " << analytical_grad << std::endl;

                std::cout << "  Computing numerical gradient..." << std::endl;
                float numerical_grad = numerical_gradient_3d(attention, input, b, i, j);
                std::cout << "  Numerical gradient:  " << numerical_grad << std::endl;

                float rel_error = std::abs(analytical_grad - numerical_grad) /
                                 (std::abs(analytical_grad) + std::abs(numerical_grad) + 1e-8f);

                std::cout << "  Relative Error:  " << rel_error << std::endl;

                tests_total++;
                if (rel_error < 3e-2f) {  // 3% tolerance for complex attention gradients
                    std::cout << "  ✓ PASS" << std::endl;
                    tests_passed++;
                } else {
                    std::cout << "  ✗ FAIL" << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << "\n!!! EXCEPTION CAUGHT IN MAIN TEST !!!" << std::endl;
                std::cerr << "Test point " << (pt_idx + 1) << "/" << test_points.size() << std::endl;
                std::cerr << "Position: [" << b << "," << i << "," << j << "]" << std::endl;
                std::cerr << "Exception: " << e.what() << std::endl;
                tests_total++;
                // Continue with other tests
            }
        }
    }

    std::cout << "\n=== GRADIENT CHECK COMPLETE ===" << std::endl;
    std::cout << "Tests passed: " << tests_passed << "/" << tests_total << std::endl;

    return (tests_passed == tests_total) ? 0 : 1;
}
