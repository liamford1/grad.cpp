// grad generate: greedy and sampled continuations of one prompt from a
// saved checkpoint.

#include "cli/args.h"
#include "commands.h"
#include "common.h"

#include "grad/data/dataset.h"
#include "grad/data/token_file.h"
#include "grad/transformer/text_gen.h"

#include <iostream>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <vector>

namespace grad::cli {

namespace {

// Held-out prompts for a corpus other than the default: the val token file
// when the corpus was prepared, else the in-memory 95/5 split the trainer
// uses.
std::vector<Prompt> held_out_prompts(const std::string& corpus_path, int vocab_size,
                                     const BPETokenizer& tokenizer) {
    constexpr int kWindow = 6;
    const std::string val_bin = token_bin_path(corpus_path, vocab_size, "val");
    if (tokenfile::exists(val_bin)) {
        return sample_prompts(corpus_path, tokenizer,
                              MappedTokenDataset(val_bin, kWindow, kWindow));
    }
    const std::vector<int> tokens = tokenizer.encode(read_text_file(corpus_path));
    const auto val = std::span<const int>(tokens).subspan(tokens.size() * 95 / 100);
    return sample_prompts(corpus_path, tokenizer,
                          TextDataset(std::vector<int>(val.begin(), val.end()), kWindow, kWindow));
}

// An empty prompt means "pick one": the first speaker tag for the default
// corpus, a held-out opening otherwise.
Prompt choose_prompt(const std::string& prompt, const std::string& corpus_path, int vocab_size,
                     const BPETokenizer& tokenizer) {
    if (!prompt.empty()) return prompt_from_text(tokenizer, prompt);
    if (is_default_corpus(corpus_path)) return prompt_from_text(tokenizer, "ROMEO:\n");
    return held_out_prompts(corpus_path, vocab_size, tokenizer).at(0);
}

// "temp=0.8", plus any cut that is on.
std::string describe_sampling(const SamplingOptions& options) {
    std::ostringstream out;
    out << "temp=" << options.temperature;
    if (options.top_k > 0) out << ", top-k=" << options.top_k;
    if (options.top_p < 1.0f) out << ", top-p=" << options.top_p;
    return out.str();
}

}  // namespace

int run_generate(const Invocation& invocation) {
    std::string checkpoint = "shakespeare_final.bin";
    std::string prompt;
    std::string corpus = kDefaultCorpus;
    std::optional<int> vocab;
    SamplingOptions sampling{.max_tokens = 150};

    Command cmd(invocation.usage_name(), std::string(invocation.summary));
    cmd.describe("Prints a greedy continuation of the prompt, then a sampled one.");
    cmd.optional("ckpt", checkpoint, "checkpoint to sample from");
    cmd.optional("prompt", prompt,
                 "text to continue; empty (\"\") picks one from the corpus's held-out split");
    cmd.optional("corpus", corpus, "the corpus the checkpoint was trained on, for its tokenizer");
    cmd.optional("vocab", vocab, "vocab size; must match the checkpoint's")
        .at_least(1)
        .default_text("the checkpoint's");
    add_sampling_options(cmd, sampling, "print only the greedy continuation");
    if (cmd.parse(invocation.args) == ParseResult::HelpShown) return 0;

    std::cout << "\ngrad.cpp Generation\n" << std::endl;
    const auto [model, tokenizer] = load_for_inference(checkpoint, corpus, vocab);
    const Prompt chosen = choose_prompt(prompt, corpus, model.getVocabSize(), tokenizer);

    TextGen generator(model, &tokenizer);

    std::cout << "\n--- Greedy Decoding ---\n" << std::endl;
    std::cout << "Prompt: \"" << chosen.text << "\"" << std::endl;
    std::cout << generator.generate_greedy(chosen.tokens, sampling.max_tokens,
                                           sampling.repetition_penalty)
              << std::endl;
    if (sampling.greedy) return 0;

    std::cout << "\n--- Sampling (" << describe_sampling(sampling) << ") ---\n" << std::endl;
    std::cout << "Prompt: \"" << chosen.text << "\"" << std::endl;
    std::cout << generator.generate_sample(chosen.tokens, sampling.temperature, sampling.max_tokens,
                                           sampling.repetition_penalty, sampling.top_k,
                                           sampling.top_p)
              << std::endl;
    return 0;
}

}  // namespace grad::cli
