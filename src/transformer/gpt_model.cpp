#include "transformer/tensor.h"
#include "transformer/token_embedding.h"
#include "transformer/positional_encoding.h"
#include "transformer/transformer_block.h"
#include "transformer/linear.h"
#include "transformer/layer_norm.h"
#include "transformer/gpt_model.h"
#include <iostream>
#include <iomanip>
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
    final_norm(d_model)
{
    std::cout << "  Initializing " << num_layers << " transformer layers..." << std::endl;
    for (int i = 0; i < num_layers; i++) {
        float progress = 100.0f * i / num_layers;
        std::cout << "\r    Layer [" << i << "/" << num_layers << "] "
                  << std::fixed << std::setprecision(1) << progress << "%     " << std::flush;
        transformer_blocks.push_back(std::make_unique<TransformerBlock>(d_model, num_heads, -1, dropout_rate));
    }
    std::cout << "\r    Layer [" << num_layers << "/" << num_layers << "] 100.0%     " << std::endl;
}

std::shared_ptr<Variable> GPTModel::forward(std::shared_ptr<Variable> token_ids, bool training) const {
    auto embed_tokens = token_embedding.forward(token_ids);
    auto encode_positions = pos_encoding.forward(embed_tokens);

    auto transformer_input = encode_positions;

    if (training && dropout_rate > 0.0f) {
        transformer_input = encode_positions->dropout(dropout_rate, training);
    }
    auto transformer_output = transformer_input;

    for (int i = 0; i < num_layers; i++) {
        transformer_output = transformer_blocks[i]->forward(transformer_output, training);
    }

    auto normalized_output = final_norm.forward(transformer_output);
    auto embedding_table = token_embedding.getEmbeddingTable();
    const Tensor& emb_data = embedding_table->getData();
    const Tensor& norm_data = normalized_output->getData();

    bool is_3d = norm_data.getIs3D();
    int batch_size = is_3d ? norm_data.getBatchSize() : 1;
    int seq_len = norm_data.getRows();
    int d_model_dim = norm_data.getCols();
    int vocab = emb_data.getRows();

    Tensor emb_transposed = emb_data.transpose();
    Tensor logits_tensor = norm_data.matmul(emb_transposed);

    auto logits = Variable::create(logits_tensor,
                                     normalized_output->requiresGrad() || embedding_table->requiresGrad());

    if (logits->requiresGrad()) {
        logits->addChild(normalized_output);
        logits->addChild(embedding_table);

        logits->setBackwardFn([normalized_output, embedding_table, logits, is_3d]() {
            const Tensor& grad_logits = logits->getGrad();
            const Tensor& norm_data = normalized_output->getData();
            const Tensor& emb_data = embedding_table->getData();

            if (normalized_output->requiresGrad()) {
                Tensor grad_norm = grad_logits.matmul(emb_data);
                normalized_output->getGrad().add_inplace(grad_norm);
            }

            if (embedding_table->requiresGrad()) {
                Tensor grad_logits_transposed = grad_logits.transpose();
                Tensor grad_emb;

                if (is_3d) {
                    int batch = grad_logits.getBatchSize();
                    int seq = grad_logits.getRows();
                    int v = grad_logits.getCols();
                    int d = norm_data.getCols();

                    grad_emb = Tensor(v, d);
                    grad_emb.fill(0.0f);

                    for (int b = 0; b < batch; b++) {
                        for (int s = 0; s < seq; s++) {
                            for (int vi = 0; vi < v; vi++) {
                                float g = grad_logits.getValue(b, s, vi);
                                for (int di = 0; di < d; di++) {
                                    float n = norm_data.getValue(b, s, di);
                                    grad_emb.setValue(vi, di, grad_emb.getValue(vi, di) + g * n);
                                }
                            }
                        }
                    }
                } else {
                    grad_emb = grad_logits_transposed.matmul(norm_data);
                }

                embedding_table->getGrad().add_inplace(grad_emb);
            }
        });
    }

    return logits;
}

std::vector<std::shared_ptr<Variable>> GPTModel::getAllParameters() const {
    std::vector<std::shared_ptr<Variable>> params;
    
    params.push_back(token_embedding.getEmbeddingTable());
    params.push_back(pos_encoding.getPositionEmbeddings());
    
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
        writeTensorToBinary(file, pos_encoding.getPositionEmbeddings()->getData());

        for (int i = 0; i < num_layers; i++) {
            const TransformerBlock* block = transformer_blocks[i].get();

            const MultiHeadAttention& attention = block->getAttention();
            writeTensorToBinary(file, attention.getW_q()->getData());
            writeTensorToBinary(file, attention.getW_k()->getData());
            writeTensorToBinary(file, attention.getW_v()->getData());
            writeTensorToBinary(file, attention.getW_o()->getData());
            writeTensorToBinary(file, attention.getB_q()->getData());
            writeTensorToBinary(file, attention.getB_k()->getData());
            writeTensorToBinary(file, attention.getB_v()->getData());
            writeTensorToBinary(file, attention.getB_o()->getData());

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

        Tensor pos_embeddings = readTensorFromBinary(file);
        model.pos_encoding.setPositionEmbeddings(pos_embeddings);

        for (int i = 0; i < num_layers; i++) {
            TransformerBlock* block = model.transformer_blocks[i].get();
            
            MultiHeadAttention& attention = block->getAttentionRef();
            Tensor wq = readTensorFromBinary(file);
            Tensor wk = readTensorFromBinary(file);
            Tensor wv = readTensorFromBinary(file);
            Tensor wo = readTensorFromBinary(file);
            Tensor bq = readTensorFromBinary(file);
            Tensor bk = readTensorFromBinary(file);
            Tensor bv = readTensorFromBinary(file);
            Tensor bo = readTensorFromBinary(file);

            auto params = attention.parameters();
            params[0]->getData() = wq;  
            params[1]->getData() = wk;  
            params[2]->getData() = wv; 
            params[3]->getData() = wo; 
            params[4]->getData() = bq;
            params[5]->getData() = bk;
            params[6]->getData() = bv; 
            params[7]->getData() = bo;
            
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

        file.close();
        std::cout << "Model loaded successfully from: " << filepath << std::endl;
        return model;

    } catch (const std::exception& e) {
        file.close();
        throw std::runtime_error("Error loading model: " + std::string(e.what()));
    }
}