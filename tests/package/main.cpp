// Links and runs one forward pass through the installed library: enough to
// catch missing headers, unexported link dependencies, and absolute paths
// baked into the package config.
#include "grad/tokenizer/bpe_tokenizer.h"
#include "grad/transformer/gpt_model.h"
#include "grad/transformer/tensor.h"
#include "grad/transformer/variable.h"

#include <iostream>

int main() {
    grad::BPETokenizer tokenizer(40);
    tokenizer.train("the cat sat on the mat");
    const std::vector<int> ids = tokenizer.encode("the cat");

    grad::GPTModel model(tokenizer.getCurrentVocabSize(), 16, 1, 2, 8, 0.0f);
    grad::Tensor input(static_cast<int>(ids.size()), 1);
    for (size_t i = 0; i < ids.size(); i++) {
        input.setValue(static_cast<int>(i), 0, static_cast<float>(ids[i]));
    }
    auto logits = model.forward(grad::Variable::create(input, false), false);
    const bool ok = logits->getData().getCols() == static_cast<size_t>(model.getVocabSize());
    logits->release_graph();
    std::cout << (ok ? "grad::core package OK" : "unexpected logits shape") << std::endl;
    return ok ? 0 : 1;
}
