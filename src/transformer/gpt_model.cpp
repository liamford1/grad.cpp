#include "transformer/tensor.h"
#include "transformer/token_embedding.h"
#include "transformer/positional_encoding.h"
#include "transformer/transformer_block.h"
#include "transformer/linear.h"
#include "transformer/layer_norm.h"
#include "transformer/gpt_model.h"
#include <iostream>
#include <memory>
#include <vector>

GPTModel::GPTModel(int vocab_size, int d_model, int num_layers, int num_heads, int max_len, float dropout_rate) : 
    vocab_size(vocab_size),
    d_model(d_model),
    num_layers(num_layers),
    num_heads(num_heads),
    max_len(max_len),
    dropout_rate(dropout_rate),
    token_embedding(vocab_size, d_model),
    pos_encoding(max_len, d_model),
    final_norm(d_model),
    output_projection(d_model, vocab_size)
{
    for (int i = 0; i < num_layers; i++) {
        transformer_blocks.push_back(std::make_unique<TransformerBlock>(d_model, num_heads, -1, dropout_rate));
    }
}

std::shared_ptr<Variable> GPTModel::forward(std::shared_ptr<Variable> token_ids, bool training) const {
    auto embed_tokens = token_embedding.forward(token_ids);
    auto encode_positions = pos_encoding.forward(embed_tokens);
    auto transformer_output = encode_positions;

    for (int i = 0; i < num_layers; i++) {
        transformer_output = transformer_blocks[i]->forward(transformer_output, training);
    }

    auto normalized_output = final_norm.forward(transformer_output);
    auto logits = output_projection.forward(normalized_output);
    return logits;
}

std::vector<std::shared_ptr<Variable>> GPTModel::getAllParameters() const {
    std::vector<std::shared_ptr<Variable>> params;
    
    params.push_back(token_embedding.getEmbeddingTable());
    
    for (int i = 0; i < num_layers; i++) {
        const TransformerBlock* block = transformer_blocks[i].get();
        
        const MultiHeadAttention& attention = block->getAttention();
        auto attn_params = attention.parameters();
        params.insert(params.end(), attn_params.begin(), attn_params.end());
        
        const FeedForward& ffn = block->getFFN();
        params.push_back(ffn.getLayer1Weights());
        params.push_back(ffn.getLayer1Bias());
        params.push_back(ffn.getLayer2Weights());
        params.push_back(ffn.getLayer2Bias());
        
        const LayerNorm& norm1 = block->getNorm1();
        const LayerNorm& norm2 = block->getNorm2();
        params.push_back(norm1.getGamma());
        params.push_back(norm1.getBeta());
        params.push_back(norm2.getGamma());
        params.push_back(norm2.getBeta());
    }
    
    params.push_back(final_norm.getGamma());
    params.push_back(final_norm.getBeta());
    
    params.push_back(output_projection.getWeights());
    params.push_back(output_projection.getBias());
    
    return params;
}

void writeTensorToBinary(std::ofstream& file, const Tensor& tensor) {
    int rows = tensor.getRows();
    int cols = tensor.getCols();

    file.write(reinterpret_cast<const char*>(&rows), sizeof(int));
    file.write(reinterpret_cast<const char*>(&cols), sizeof(int));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float value = tensor.getValue(i, j);
            file.write(reinterpret_cast<const char*>(&value), sizeof(float));
        }
    }
}

Tensor readTensorFromBinary(std::ifstream& file) {
    int rows;
    int cols;

    file.read(reinterpret_cast<char*>(&rows), sizeof(int));
    file.read(reinterpret_cast<char*>(&cols), sizeof(int));

    Tensor tensor(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float value;
            file.read(reinterpret_cast<char*>(&value), sizeof(float));
            tensor.setValue(i, j, value);
        }
    }
    return tensor;
}

bool GPTModel::save(const std::string& filepath) const {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << filepath << std::endl;
        return false;
    }

    try {
        uint32_t magic = 0x4750544D;
        uint32_t version = 1;
        file.write(reinterpret_cast<const char*>(&magic), sizeof(uint32_t));
        file.write(reinterpret_cast<const char*>(&version), sizeof(uint32_t));

        file.write(reinterpret_cast<const char*>(&vocab_size), sizeof(int));
        file.write(reinterpret_cast<const char*>(&d_model), sizeof(int));
        file.write(reinterpret_cast<const char*>(&num_layers), sizeof(int));
        file.write(reinterpret_cast<const char*>(&num_heads), sizeof(int));
        file.write(reinterpret_cast<const char*>(&max_len), sizeof(int));
        file.write(reinterpret_cast<const char*>(&dropout_rate), sizeof(float));

        writeTensorToBinary(file, token_embedding.getEmbeddingTable()->getData());

        for (int i = 0; i < num_layers; i++) {
            const TransformerBlock* block = transformer_blocks[i].get();
            
            const MultiHeadAttention& attention = block->getAttention();
            writeTensorToBinary(file, attention.getW_q()->getData());
            writeTensorToBinary(file, attention.getW_k()->getData());
            writeTensorToBinary(file, attention.getW_v()->getData());
            writeTensorToBinary(file, attention.getW_o()->getData());
            
            const FeedForward& ff = block->getFFN();
            writeTensorToBinary(file, ff.getLayer1Weights()->getData());
            writeTensorToBinary(file, ff.getLayer1Bias()->getData());
            writeTensorToBinary(file, ff.getLayer2Weights()->getData());
            writeTensorToBinary(file, ff.getLayer2Bias()->getData());
            
            const LayerNorm& norm1 = block->getNorm1();
            const LayerNorm& norm2 = block->getNorm2();
            writeTensorToBinary(file, norm1.getGamma()->getData());
            writeTensorToBinary(file, norm1.getBeta()->getData());
            writeTensorToBinary(file, norm2.getGamma()->getData());
            writeTensorToBinary(file, norm2.getBeta()->getData());
        }

        writeTensorToBinary(file, final_norm.getGamma()->getData());
        writeTensorToBinary(file, final_norm.getBeta()->getData());
        
        writeTensorToBinary(file, output_projection.getWeights()->getData());
        writeTensorToBinary(file, output_projection.getBias()->getData());
        
        file.close();
        std::cout << "Model saved successfully to: " << filepath << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error saving model: " << e.what() << std::endl;
        file.close();
        return false;
    }
}

GPTModel GPTModel::load(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Error: Could not open file for reading: " + filepath);
    }

    try {
        uint32_t magic;
        uint32_t version;
        file.read(reinterpret_cast<char*>(&magic), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));

        if (magic != 0x4750544D) {
            throw std::runtime_error("Invalid file format: wrong magic number");
        }
        if (version != 1) {
            throw std::runtime_error("Unsupported file version: " + std::to_string(version)); 
        }

        int vocab_size, d_model, num_layers, num_heads, max_len;
        float dropout_rate;
        file.read(reinterpret_cast<char*>(&vocab_size), sizeof(int));
        file.read(reinterpret_cast<char*>(&d_model), sizeof(int));
        file.read(reinterpret_cast<char*>(&num_layers), sizeof(int));
        file.read(reinterpret_cast<char*>(&num_heads), sizeof(int));
        file.read(reinterpret_cast<char*>(&max_len), sizeof(int));
        file.read(reinterpret_cast<char*>(&dropout_rate), sizeof(float));

        GPTModel model(vocab_size, d_model, num_layers, num_heads, max_len, dropout_rate);
        Tensor embedding_table = readTensorFromBinary(file);
        model.token_embedding.setEmbeddingTable(embedding_table);

        for (int i = 0; i < num_layers; i++) {
            TransformerBlock* block = model.transformer_blocks[i].get();
            
            MultiHeadAttention& attention = block->getAttentionRef();
            Tensor wq = readTensorFromBinary(file);
            Tensor wk = readTensorFromBinary(file);
            Tensor wv = readTensorFromBinary(file);
            Tensor wo = readTensorFromBinary(file);
            attention.setWeights(wq, wk, wv, wo);
            
            FeedForward& ff = block->getFeedForwardRef();
            Tensor layer1_weights = readTensorFromBinary(file);
            Tensor layer1_bias = readTensorFromBinary(file);
            Tensor layer2_weights = readTensorFromBinary(file);
            Tensor layer2_bias = readTensorFromBinary(file);
            auto layer1_weights_var = Variable::create(layer1_weights, true);
            auto layer1_bias_var = Variable::create(layer1_bias, true);
            auto layer2_weights_var = Variable::create(layer2_weights, true);
            auto layer2_bias_var = Variable::create(layer2_bias, true);
            ff.setWeights(layer1_weights_var, layer1_bias_var, layer2_weights_var, layer2_bias_var);
            
            LayerNorm& norm1 = block->getNorm1Ref();
            LayerNorm& norm2 = block->getNorm2Ref();
            Tensor gamma1 = readTensorFromBinary(file);
            Tensor beta1 = readTensorFromBinary(file);
            Tensor gamma2 = readTensorFromBinary(file);
            Tensor beta2 = readTensorFromBinary(file);
            norm1.setParams(gamma1, beta1);
            norm2.setParams(gamma2, beta2);
        }

        Tensor final_gamma = readTensorFromBinary(file);
        Tensor final_beta = readTensorFromBinary(file);
        model.final_norm.setParams(final_gamma, final_beta);
        
        Tensor output_weights = readTensorFromBinary(file);
        Tensor output_bias = readTensorFromBinary(file);
        auto output_weights_var = Variable::create(output_weights, true);
        auto output_bias_var = Variable::create(output_bias, true);
        model.output_projection.setWeights(output_weights_var, output_bias_var);
        
        file.close();
        std::cout << "Model loaded successfully from: " << filepath << std::endl;
        return model;

    } catch (const std::exception& e) {
        file.close();
        throw std::runtime_error("Error loading model: " + std::string(e.what()));
    }
}