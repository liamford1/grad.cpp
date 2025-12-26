#include "transformer/tensor.h"
#include "transformer/linear.h"
#include "transformer/activations.h"
#include "transformer/feedforward.h"
#include <iostream>

FeedForward::FeedForward(int d_model, int hidden_dim, float dropout_rate) :
    layer1(d_model, (hidden_dim == -1) ? 4 * d_model : hidden_dim),
    layer2((hidden_dim == -1) ? 4 * d_model : hidden_dim, d_model),
    dropout_rate(dropout_rate) {}

std::shared_ptr<Variable> FeedForward::forward(std::shared_ptr<Variable> input, bool training) const {
    const Tensor& input_tensor = input->getData();
    Device original_device = input_tensor.getDevice();

    std::cerr << "[DEBUG] FFN: Input device=" << (original_device == Device::CUDA ? "CUDA" : "CPU") << std::endl;

    if (original_device == Device::CUDA) {
        std::cerr << "[DEBUG] FFN: Moving input to CPU" << std::endl;
        Tensor input_cpu = input_tensor.to(Device::CPU);
        auto input_cpu_var = Variable::create(input_cpu, input->requiresGrad());

        std::cerr << "[DEBUG] FFN: Computing layer1 on CPU" << std::endl;
        auto output = layer1.forward(input_cpu_var);
        output = output->gelu();
        output = output->dropout(dropout_rate, training);

        std::cerr << "[DEBUG] FFN: Computing layer2 on CPU" << std::endl;
        output = layer2.forward(output);
        output = output->dropout(dropout_rate, training);

        std::cerr << "[DEBUG] FFN: Moving output back to CUDA" << std::endl;
        Tensor output_cuda = output->getData().to(Device::CUDA);
        return Variable::create(output_cuda, input->requiresGrad());
    } else {
        auto output = layer1.forward(input);
        output = output->gelu();
        output = output->dropout(dropout_rate, training);
        output = layer2.forward(output);
        return output->dropout(dropout_rate, training);
    }
}