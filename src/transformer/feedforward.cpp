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

        std::cerr << "[DEBUG] FFN: Moving layer1 weights to CPU" << std::endl;
        auto layer1_weights = layer1.getWeights();
        auto layer1_bias = layer1.getBias();
        Tensor w1_cpu = (layer1_weights->getData().getDevice() == Device::CUDA) ?
                        layer1_weights->getData().to(Device::CPU) : layer1_weights->getData();
        Tensor b1_cpu = (layer1_bias->getData().getDevice() == Device::CUDA) ?
                        layer1_bias->getData().to(Device::CPU) : layer1_bias->getData();
        auto w1_cpu_var = Variable::create(w1_cpu, false);
        auto b1_cpu_var = Variable::create(b1_cpu, false);

        std::cerr << "[DEBUG] FFN: Computing layer1 on CPU" << std::endl;
        auto output = input_cpu_var->matmul(w1_cpu_var)->add(b1_cpu_var);
        std::cerr << "[DEBUG] FFN: Layer 1 computed" << std::endl;
        output = output->gelu();
        output = output->dropout(dropout_rate, training);

        std::cerr << "[DEBUG] FFN: Moving layer2 weights to CPU" << std::endl;
        auto layer2_weights = layer2.getWeights();
        auto layer2_bias = layer2.getBias();
        Tensor w2_cpu = (layer2_weights->getData().getDevice() == Device::CUDA) ?
                        layer2_weights->getData().to(Device::CPU) : layer2_weights->getData();
        Tensor b2_cpu = (layer2_bias->getData().getDevice() == Device::CUDA) ?
                        layer2_bias->getData().to(Device::CPU) : layer2_bias->getData();
        auto w2_cpu_var = Variable::create(w2_cpu, false);
        auto b2_cpu_var = Variable::create(b2_cpu, false);

        std::cerr << "[DEBUG] FFN: Computing layer2 on CPU" << std::endl;
        output = output->matmul(w2_cpu_var)->add(b2_cpu_var);
        std::cerr << "[DEBUG] FFN: Layer 2 computed" << std::endl;
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