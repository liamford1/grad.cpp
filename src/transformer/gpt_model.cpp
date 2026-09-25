#include "transformer/tensor.h"
#include "transformer/token_embedding.h"
#include "transformer/positional_encoding.h"
#include "transformer/transformer_block.h"
#include "transformer/linear.h"
#include "transformer/layer_norm.h"
#include "transformer/gpt_model.h"
#include "transformer/blas_wrapper.h"
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

GPTModel::GPTModel(int vocab_size, int d_model, int num_layers, int num_heads, int max_len,
                   float dropout_rate, GPTArch arch) :
    vocab_size(vocab_size),
    d_model(d_model),
    num_layers(num_layers),
    num_heads(num_heads),
    max_len(max_len),
    dropout_rate(dropout_rate),
    arch(arch),
    token_embedding(vocab_size, d_model),
    pos_encoding(max_len, d_model),
    final_norm(d_model, /*rms=*/arch == GPTArch::Modern)
{
    std::cout << "  Initializing " << num_layers << " transformer layers..." << std::endl;
    // Save/restore stream format state: leaking fixed(1) here made every
    // later float print misleading ("Learning rate: 0.0" for 3e-4).
    std::ios old_state(nullptr);
    old_state.copyfmt(std::cout);
    for (int i = 0; i < num_layers; i++) {
        float progress = 100.0f * i / num_layers;
        std::cout << "\r    Layer [" << i << "/" << num_layers << "] "
                  << std::fixed << std::setprecision(1) << progress << "%     " << std::flush;
        transformer_blocks.push_back(std::make_unique<TransformerBlock>(
            d_model, num_heads, -1, dropout_rate, arch == GPTArch::Modern));
    }
    std::cout << "\r    Layer [" << num_layers << "/" << num_layers << "] 100.0%     " << std::endl;
    std::cout.copyfmt(old_state);
}

std::shared_ptr<Variable> GPTModel::forward(std::shared_ptr<Variable> token_ids, bool training) const {
    auto embed_tokens = token_embedding.forward(token_ids);
    // Modern arch: position comes from RoPE inside attention, not from
    // learned embeddings added to the residual stream.
    auto transformer_input = arch == GPTArch::Modern
        ? embed_tokens
        : pos_encoding.forward(embed_tokens);

    if (training && dropout_rate > 0.0f) {
        transformer_input = transformer_input->dropout(dropout_rate, training);
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

    // Weight tying: logits = norm @ E^T. A 3D (batch, seq, d) tensor is
    // contiguous, so it multiplies as one flat (batch*seq, d) matrix, and
    // the transpose happens inside the sgemm instead of materializing E^T.
    int flat_rows = batch_size * seq_len;
    Tensor logits_tensor = is_3d
        ? Tensor::uninitialized(batch_size, seq_len, vocab)
        : Tensor::uninitialized(seq_len, vocab);
    blas_sgemm_ex(norm_data.raw(), emb_data.raw(), logits_tensor.raw(),
                  flat_rows, vocab, d_model_dim,
                  false, true, 1.0f, 0.0f);

    auto logits = Variable::create(std::move(logits_tensor),
                                     normalized_output->requiresGrad() || embedding_table->requiresGrad());

    if (logits->requiresGrad()) {
        logits->addChild(normalized_output);
        logits->addChild(embedding_table);

        logits->setBackwardFn([normalized_output, embedding_table,
                               logits_weak = std::weak_ptr<Variable>(logits),
                               flat_rows, vocab, d_model_dim]() {
            auto logits = logits_weak.lock();
            if (!logits || !logits->hasGrad()) return;
            const Tensor& grad_logits = logits->getGrad();
            const Tensor& norm_data = normalized_output->getData();
            const Tensor& emb_data = embedding_table->getData();

            if (normalized_output->requiresGrad()) {
                normalized_output->ensureGrad();
                // dNorm += dLogits @ E, accumulated in place (beta = 1)
                blas_sgemm_ex(grad_logits.raw(), emb_data.raw(),
                              normalized_output->getGrad().raw(),
                              flat_rows, d_model_dim, vocab,
                              false, false, 1.0f, 1.0f);
            }

            if (embedding_table->requiresGrad()) {
                embedding_table->ensureGrad();
                // dE += dLogits^T @ Norm, summing over batch*seq via the sgemm
                blas_sgemm_ex(grad_logits.raw(), norm_data.raw(),
                              embedding_table->getGrad().raw(),
                              vocab, d_model_dim, flat_rows,
                              true, false, 1.0f, 1.0f);
            }
        });
    }

    return logits;
}

std::vector<std::shared_ptr<Variable>> GPTModel::getAllParameters() const {
    std::vector<std::shared_ptr<Variable>> params;
    
    const bool modern = arch == GPTArch::Modern;

    params.push_back(token_embedding.getEmbeddingTable());
    if (!modern) {
        params.push_back(pos_encoding.getPositionEmbeddings());
    }

    for (int i = 0; i < num_layers; i++) {
        const TransformerBlock* block = transformer_blocks[i].get();

        const MultiHeadAttention& attention = block->getAttention();
        auto attn_params = attention.parameters();
        params.insert(params.end(), attn_params.begin(), attn_params.end());

        const FeedForward& ffn = block->getFFN();
        if (modern) {
            params.push_back(ffn.getGateWeights());
            params.push_back(ffn.getLayer1Weights());
            params.push_back(ffn.getLayer2Weights());
        } else {
            params.push_back(ffn.getLayer1Weights());
            params.push_back(ffn.getLayer1Bias());
            params.push_back(ffn.getLayer2Weights());
            params.push_back(ffn.getLayer2Bias());
        }

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

namespace {

constexpr uint32_t kCheckpointMagic = 0x4750544D;  // "GPTM"

// Upper bounds for header fields. Far above any model this trainer can
// fit in memory, low enough that a corrupt header fails here with a clear
// message instead of as a multi-gigabyte allocation or an int overflow.
constexpr int kMaxVocab = 1 << 24;
constexpr int kMaxDModel = 1 << 16;
constexpr int kMaxLayers = 1 << 12;
constexpr int kMaxLen = 1 << 20;

// On-disk tensor: int32 rows, int32 cols, then rows*cols float32 in
// row-major order (the in-memory layout, so one write/read moves it).
void write_tensor(std::ofstream& file, const Tensor& tensor) {
    if (tensor.getIs3D()) {
        throw std::logic_error("checkpoint tensors are 2D");
    }
    const int rows = static_cast<int>(tensor.getRows());
    const int cols = static_cast<int>(tensor.getCols());
    file.write(reinterpret_cast<const char*>(&rows), sizeof(int));
    file.write(reinterpret_cast<const char*>(&cols), sizeof(int));
    file.write(reinterpret_cast<const char*>(tensor.raw()),
               static_cast<std::streamsize>(tensor.numel() * sizeof(float)));
}

// Reads a checkpoint with every read checked: a truncated or corrupt file
// fails naming the field it was reading, never with values left
// uninitialized by a short read. load() prefixes the path.
class CheckpointReader {
public:
    explicit CheckpointReader(std::ifstream& file) : file_(file) {}

    template <typename T>
    T scalar(const char* what) {
        T value{};
        file_.read(reinterpret_cast<char*>(&value), sizeof(T));
        if (!file_) fail(std::string("file ends while reading ") + what);
        return value;
    }

    // Reads the next tensor, which must have exactly the shape of the
    // parameter it will replace.
    Tensor tensor_like(const Tensor& target, const std::string& what) {
        const int rows = scalar<int>(what.c_str());
        const int cols = scalar<int>(what.c_str());
        if (rows <= 0 || cols <= 0 ||
            static_cast<size_t>(rows) != target.getRows() ||
            static_cast<size_t>(cols) != target.getCols()) {
            fail(what + " is " + std::to_string(rows) + "x" + std::to_string(cols) +
                 ", model expects " + std::to_string(target.getRows()) + "x" +
                 std::to_string(target.getCols()));
        }
        Tensor t = Tensor::uninitialized(rows, cols);
        file_.read(reinterpret_cast<char*>(t.raw()),
                   static_cast<std::streamsize>(t.numel() * sizeof(float)));
        if (!file_) fail("file ends inside " + what);
        return t;
    }

    [[noreturn]] void fail(const std::string& msg) const {
        throw std::runtime_error(msg);
    }

private:
    std::ifstream& file_;
};

void check_range(const CheckpointReader& r, const char* name, int value, int lo, int hi) {
    if (value < lo || value > hi) {
        r.fail(std::string("header field ") + name + " = " + std::to_string(value) +
               " is outside [" + std::to_string(lo) + ", " + std::to_string(hi) + "]");
    }
}

}  // namespace

bool GPTModel::save(const std::string& filepath, bool quiet) const {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << filepath << std::endl;
        return false;
    }

    try {
        // Version 2 adds the arch tag; everything else is unchanged, so
        // v1 files (all GPT-2-style checkpoints) stay loadable.
        uint32_t magic = kCheckpointMagic;
        uint32_t version = 2;
        uint32_t arch_tag = static_cast<uint32_t>(arch);
        file.write(reinterpret_cast<const char*>(&magic), sizeof(uint32_t));
        file.write(reinterpret_cast<const char*>(&version), sizeof(uint32_t));
        file.write(reinterpret_cast<const char*>(&arch_tag), sizeof(uint32_t));

        file.write(reinterpret_cast<const char*>(&vocab_size), sizeof(int));
        file.write(reinterpret_cast<const char*>(&d_model), sizeof(int));
        file.write(reinterpret_cast<const char*>(&num_layers), sizeof(int));
        file.write(reinterpret_cast<const char*>(&num_heads), sizeof(int));
        file.write(reinterpret_cast<const char*>(&max_len), sizeof(int));
        file.write(reinterpret_cast<const char*>(&dropout_rate), sizeof(float));

        const bool modern = arch == GPTArch::Modern;

        write_tensor(file, token_embedding.getEmbeddingTable()->getData());
        if (!modern) {
            write_tensor(file, pos_encoding.getPositionEmbeddings()->getData());
        }

        for (int i = 0; i < num_layers; i++) {
            const TransformerBlock* block = transformer_blocks[i].get();

            const MultiHeadAttention& attention = block->getAttention();
            write_tensor(file, attention.getW_q()->getData());
            write_tensor(file, attention.getW_k()->getData());
            write_tensor(file, attention.getW_v()->getData());
            write_tensor(file, attention.getW_o()->getData());
            write_tensor(file, attention.getB_q()->getData());
            write_tensor(file, attention.getB_k()->getData());
            write_tensor(file, attention.getB_v()->getData());
            write_tensor(file, attention.getB_o()->getData());

            const FeedForward& ff = block->getFFN();
            if (modern) {
                write_tensor(file, ff.getGateWeights()->getData());
                write_tensor(file, ff.getLayer1Weights()->getData());
                write_tensor(file, ff.getLayer2Weights()->getData());
            } else {
                write_tensor(file, ff.getLayer1Weights()->getData());
                write_tensor(file, ff.getLayer1Bias()->getData());
                write_tensor(file, ff.getLayer2Weights()->getData());
                write_tensor(file, ff.getLayer2Bias()->getData());
            }

            const LayerNorm& norm1 = block->getNorm1();
            const LayerNorm& norm2 = block->getNorm2();
            write_tensor(file, norm1.getGamma()->getData());
            write_tensor(file, norm1.getBeta()->getData());
            write_tensor(file, norm2.getGamma()->getData());
            write_tensor(file, norm2.getBeta()->getData());
        }

        write_tensor(file, final_norm.getGamma()->getData());
        write_tensor(file, final_norm.getBeta()->getData());

        // ofstream reports errors (disk full, I/O error) only through its
        // state, and close() is where buffered bytes actually hit the file.
        file.close();
        if (!file) {
            std::cerr << "Error: writing checkpoint failed: " << filepath << std::endl;
            return false;
        }
        if (!quiet) {
            std::cout << "Model saved successfully to: " << filepath << std::endl;
        }
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
        CheckpointReader in(file);
        const uint32_t magic = in.scalar<uint32_t>("magic number");
        if (magic != kCheckpointMagic) {
            in.fail("not a grad.cpp checkpoint (wrong magic number)");
        }
        const uint32_t version = in.scalar<uint32_t>("format version");
        if (version != 1 && version != 2) {
            in.fail("unsupported format version " + std::to_string(version));
        }

        // v1 predates the arch tag: every v1 checkpoint is GPT-2-style.
        GPTArch arch = GPTArch::GPT2;
        if (version == 2) {
            const uint32_t arch_tag = in.scalar<uint32_t>("architecture tag");
            if (arch_tag != static_cast<uint32_t>(GPTArch::GPT2) &&
                arch_tag != static_cast<uint32_t>(GPTArch::Modern)) {
                in.fail("unknown architecture tag " + std::to_string(arch_tag));
            }
            arch = static_cast<GPTArch>(arch_tag);
        }
        const bool modern = arch == GPTArch::Modern;

        const int vocab_size = in.scalar<int>("vocab_size");
        const int d_model = in.scalar<int>("d_model");
        const int num_layers = in.scalar<int>("num_layers");
        const int num_heads = in.scalar<int>("num_heads");
        const int max_len = in.scalar<int>("max_len");
        const float dropout_rate = in.scalar<float>("dropout_rate");

        check_range(in, "vocab_size", vocab_size, 1, kMaxVocab);
        check_range(in, "d_model", d_model, 1, kMaxDModel);
        check_range(in, "num_layers", num_layers, 1, kMaxLayers);
        check_range(in, "num_heads", num_heads, 1, d_model);
        check_range(in, "max_len", max_len, 1, kMaxLen);
        if (d_model % num_heads != 0) {
            in.fail("d_model " + std::to_string(d_model) +
                    " is not divisible by num_heads " + std::to_string(num_heads));
        }
        if (!(dropout_rate >= 0.0f && dropout_rate < 1.0f)) {
            in.fail("dropout_rate " + std::to_string(dropout_rate) + " is outside [0, 1)");
        }

        GPTModel model(vocab_size, d_model, num_layers, num_heads, max_len, dropout_rate, arch);

        model.token_embedding.setEmbeddingTable(in.tensor_like(
            model.token_embedding.getEmbeddingTable()->getData(), "token embedding"));

        if (!modern) {
            model.pos_encoding.setPositionEmbeddings(in.tensor_like(
                model.pos_encoding.getPositionEmbeddings()->getData(), "position embedding"));
        }

        for (int i = 0; i < num_layers; i++) {
            TransformerBlock* block = model.transformer_blocks[i].get();
            const std::string layer = "layer " + std::to_string(i) + " ";

            // Attention parameters load in place, in parameters() order.
            static const char* const kAttnNames[] = {"W_q", "W_k", "W_v", "W_o",
                                                     "b_q", "b_k", "b_v", "b_o"};
            auto attn_params = block->getAttentionRef().parameters();
            for (size_t p = 0; p < attn_params.size(); p++) {
                Tensor& target = attn_params[p]->getData();
                target = in.tensor_like(target, layer + kAttnNames[p]);
            }

            FeedForward& ff = block->getFeedForwardRef();
            if (modern) {
                Tensor gate = in.tensor_like(ff.getGateWeights()->getData(), layer + "FFN gate");
                Tensor up = in.tensor_like(ff.getLayer1Weights()->getData(), layer + "FFN up");
                Tensor down = in.tensor_like(ff.getLayer2Weights()->getData(), layer + "FFN down");
                ff.setGatedWeights(Variable::create(std::move(gate), true),
                                   Variable::create(std::move(up), true),
                                   Variable::create(std::move(down), true));
            } else {
                Tensor w1 = in.tensor_like(ff.getLayer1Weights()->getData(), layer + "FFN W1");
                Tensor b1 = in.tensor_like(ff.getLayer1Bias()->getData(), layer + "FFN b1");
                Tensor w2 = in.tensor_like(ff.getLayer2Weights()->getData(), layer + "FFN W2");
                Tensor b2 = in.tensor_like(ff.getLayer2Bias()->getData(), layer + "FFN b2");
                ff.setWeights(Variable::create(std::move(w1), true),
                              Variable::create(std::move(b1), true),
                              Variable::create(std::move(w2), true),
                              Variable::create(std::move(b2), true));
            }

            LayerNorm& norm1 = block->getNorm1Ref();
            LayerNorm& norm2 = block->getNorm2Ref();
            Tensor gamma1 = in.tensor_like(norm1.getGamma()->getData(), layer + "norm1 gamma");
            Tensor beta1 = in.tensor_like(norm1.getBeta()->getData(), layer + "norm1 beta");
            Tensor gamma2 = in.tensor_like(norm2.getGamma()->getData(), layer + "norm2 gamma");
            Tensor beta2 = in.tensor_like(norm2.getBeta()->getData(), layer + "norm2 beta");
            norm1.setParams(gamma1, beta1);
            norm2.setParams(gamma2, beta2);
        }

        Tensor final_gamma = in.tensor_like(model.final_norm.getGamma()->getData(), "final norm gamma");
        Tensor final_beta = in.tensor_like(model.final_norm.getBeta()->getData(), "final norm beta");
        model.final_norm.setParams(final_gamma, final_beta);

        file.close();
        std::cout << "Model loaded successfully from: " << filepath << std::endl;
        return model;

    } catch (const std::exception& e) {
        file.close();
        throw std::runtime_error("Error loading model from " + filepath + ": " + e.what());
    }
}