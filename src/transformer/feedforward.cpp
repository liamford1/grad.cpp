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

    if (original_device == Device::CUDA) {
        Tensor input_cpu = input_tensor.to(Device::CPU);
        auto input_cpu_var = Variable::create(input_cpu, input->requiresGrad());
        auto layer1_weights = layer1.getWeights();
        auto layer1_bias = layer1.getBias();
        Tensor w1_cpu = (layer1_weights->getData().getDevice() == Device::CUDA) ?
                        layer1_weights->getData().to(Device::CPU) : layer1_weights->getData();
        Tensor b1_cpu = (layer1_bias->getData().getDevice() == Device::CUDA) ?
                        layer1_bias->getData().to(Device::CPU) : layer1_bias->getData();
        auto w1_cpu_var = Variable::create(w1_cpu, false);
        auto b1_cpu_var = Variable::create(b1_cpu, false);

        auto output = input_cpu_var->matmul(w1_cpu_var)->add(b1_cpu_var);
        output = output->gelu();
        output = output->dropout(dropout_rate, training);
        auto layer2_weights = layer2.getWeights();
        auto layer2_bias = layer2.getBias();
        Tensor w2_cpu = (layer2_weights->getData().getDevice() == Device::CUDA) ?
                        layer2_weights->getData().to(Device::CPU) : layer2_weights->getData();
        Tensor b2_cpu = (layer2_bias->getData().getDevice() == Device::CUDA) ?
                        layer2_bias->getData().to(Device::CPU) : layer2_bias->getData();
        auto w2_cpu_var = Variable::create(w2_cpu, false);
        auto b2_cpu_var = Variable::create(b2_cpu, false);

        output = output->matmul(w2_cpu_var)->add(b2_cpu_var);
        output = output->dropout(dropout_rate, training);

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