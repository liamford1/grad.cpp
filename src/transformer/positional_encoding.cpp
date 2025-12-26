#include "transformer/positional_encoding.h"
#include <stdexcept>

PositionalEncoding::PositionalEncoding(int max_len, int d_model) :
    max_len(max_len),
    d_model(d_model)
{
    Tensor pos_emb(max_len, d_model);
    pos_emb.xavier(max_len, d_model);
    position_embeddings = Variable::create(pos_emb, true);
}

PositionalEncoding::~PositionalEncoding() {}

std::shared_ptr<Variable> PositionalEncoding::forward(std::shared_ptr<Variable> embeddings) const {
    const Tensor& emb_tensor = embeddings->getData();

    if (!emb_tensor.getIs3D()) {
        int seq_len = emb_tensor.getRows();
        
        if (seq_len > max_len) {
            throw std::out_of_range("Sequence length exceeds max_len");
        }
        
        Tensor pos_slice = position_embeddings->getData().slice(0, seq_len, 0, d_model);
        auto pos_var = Variable::create(pos_slice, position_embeddings->requiresGrad());
        
        auto output = embeddings->add(pos_var);
        
        if (embeddings->requiresGrad() || position_embeddings->requiresGrad()) {
            output->addChild(embeddings);
            output->addChild(position_embeddings);
            
            auto self_pos_emb = position_embeddings;
            int self_d_model = d_model;
            output->setBackwardFn([embeddings, self_pos_emb, output_weak = std::weak_ptr<Variable>(output), seq_len, self_d_model]() {
                auto output = output_weak.lock();
                if (!output) return;
                if (embeddings->requiresGrad()) {
                    embeddings->getGrad() = embeddings->getGrad().add(output->getGrad());
                }
                
                if (self_pos_emb->requiresGrad()) {
                    for (int i = 0; i < seq_len; i++) {
                        for (int j = 0; j < self_d_model; j++) {
                            float grad_val = output->getGrad().getValue(i, j);
                            float curr_grad = self_pos_emb->getGrad().getValue(i, j);
                            self_pos_emb->getGrad().setValue(i, j, curr_grad + grad_val);
                        }
                    }
                }
            });
        }
        
        return output;
        
    } else {
        int batch_size = emb_tensor.getBatchSize();
        int seq_len = emb_tensor.getRows();

        if (seq_len > max_len) {
            throw std::out_of_range("Sequence length exceeds max_len");
        }

        if (emb_tensor.getDevice() == Device::CUDA) {
            const Tensor& pos_emb = position_embeddings->getData();
            Tensor pos_emb_gpu = (pos_emb.getDevice() == Device::CUDA) ? pos_emb : pos_emb.to(Device::CUDA);

            Tensor pos_broadcast = pos_encoding_broadcast_gpu(pos_emb_gpu, batch_size, seq_len, d_model);
            auto pos_var = Variable::create(pos_broadcast, position_embeddings->requiresGrad());
            auto output = embeddings->add(pos_var);

            if (embeddings->requiresGrad() || position_embeddings->requiresGrad()) {
                output->addChild(embeddings);
                output->addChild(position_embeddings);
            }
            return output;
        }

        Tensor pos_broadcast(batch_size, seq_len, d_model);
        for (int b = 0; b < batch_size; b++) {
            for (int i = 0; i < seq_len; i++) {
                for (int j = 0; j < d_model; j++) {
                    pos_broadcast.setValue(b, i, j, position_embeddings->getData().getValue(i, j));
                }
            }
        }
        
        auto pos_var = Variable::create(pos_broadcast, position_embeddings->requiresGrad());
        auto output = embeddings->add(pos_var);
        
        if (embeddings->requiresGrad() || position_embeddings->requiresGrad()) {
            output->addChild(embeddings);
            output->addChild(position_embeddings);
            
            auto self_pos_emb = position_embeddings;
            int self_d_model = d_model;
            output->setBackwardFn([embeddings, self_pos_emb, output_weak = std::weak_ptr<Variable>(output), batch_size, seq_len, self_d_model]() {
                auto output = output_weak.lock();
                if (!output) return;
                if (embeddings->requiresGrad()) {
                    embeddings->getGrad() = embeddings->getGrad().add(output->getGrad());
                }
                
                if (self_pos_emb->requiresGrad()) {
                    for (int b = 0; b < batch_size; b++) {
                        for (int i = 0; i < seq_len; i++) {
                            for (int j = 0; j < self_d_model; j++) {
                                float grad_val = output->getGrad().getValue(b, i, j);
                                float curr_grad = self_pos_emb->getGrad().getValue(i, j);
                                self_pos_emb->getGrad().setValue(i, j, curr_grad + grad_val);
                            }
                        }
                    }
                }
            });
        }
        return output;
    }
}