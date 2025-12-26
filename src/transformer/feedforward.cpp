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
        auto w1_cpu_var = Variable::create(w1_cpu, true);
        auto b1_cpu_var = Variable::create(b1_cpu, true);

        auto hidden = input_cpu_var->matmul(w1_cpu_var)->add(b1_cpu_var);
        hidden = hidden->gelu();
        hidden = hidden->dropout(dropout_rate, training);

        auto layer2_weights = layer2.getWeights();
        auto layer2_bias = layer2.getBias();
        Tensor w2_cpu = (layer2_weights->getData().getDevice() == Device::CUDA) ?
                        layer2_weights->getData().to(Device::CPU) : layer2_weights->getData();
        Tensor b2_cpu = (layer2_bias->getData().getDevice() == Device::CUDA) ?
                        layer2_bias->getData().to(Device::CPU) : layer2_bias->getData();
        auto w2_cpu_var = Variable::create(w2_cpu, true);
        auto b2_cpu_var = Variable::create(b2_cpu, true);

        auto output_cpu = hidden->matmul(w2_cpu_var)->add(b2_cpu_var);
        output_cpu = output_cpu->dropout(dropout_rate, training);

        Tensor output_cuda = output_cpu->getData().to(Device::CUDA);
        auto result = Variable::create(output_cuda, input->requiresGrad());

        if (input->requiresGrad()) {
            result->addChild(input);
            result->addChild(layer1_weights);
            result->addChild(layer1_bias);
            result->addChild(layer2_weights);
            result->addChild(layer2_bias);

            result->setBackwardFn([input, layer1_weights, layer1_bias, layer2_weights, layer2_bias,
                                   w1_cpu_var, b1_cpu_var, w2_cpu_var, b2_cpu_var,
                                   input_cpu_var, output_cpu, result]() {
                output_cpu->backward(result->getGrad().to(Device::CPU));

                Tensor dW1 = w1_cpu_var->getGrad();
                Tensor db1 = b1_cpu_var->getGrad();
                Tensor dW2 = w2_cpu_var->getGrad();
                Tensor db2 = b2_cpu_var->getGrad();

                layer1_weights->getGrad().add_inplace(dW1.to(Device::CUDA));
                layer1_bias->getGrad().add_inplace(db1.to(Device::CUDA));
                layer2_weights->getGrad().add_inplace(dW2.to(Device::CUDA));
                layer2_bias->getGrad().add_inplace(db2.to(Device::CUDA));

                Tensor dInput_cpu = input_cpu_var->getGrad();
                input->getGrad().add_inplace(dInput_cpu.to(Device::CUDA));
            });
        }

        return result;
    } else {
        auto output = layer1.forward(input);
        output = output->gelu();
        output = output->dropout(dropout_rate, training);
        output = layer2.forward(output);
        return output->dropout(dropout_rate, training);
    }
}