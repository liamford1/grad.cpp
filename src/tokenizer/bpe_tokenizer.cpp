#include "tokenizer/bpe_tokenizer.h"
#include <iostream>
#include <set>

BPETokenizer::BPETokenizer(int vocab_size) : vocab_size(vocab_size) {
    vocab[pad_token] = pad_token_id;
    vocab[eos_token] = eos_token_id;
    vocab[unk_token] = unk_token_id;

    id_to_token[pad_token_id] = pad_token;
    id_to_token[eos_token_id] = eos_token;
    id_to_token[unk_token_id] = unk_token;
}

void BPETokenizer::train(const std::string& training_text) {
    std::set<char> unique_chars;
    for (char c : training_text) {
        unique_chars.insert(c);
    }

    int curr_id = 3;
    for(char c : unique_chars) {
        std::string char_str(1, c);
        vocab[char_str] = curr_id;
        id_to_token[curr_id] = char_str;
        curr_id++;
    }
    std::cout << "Vocab size after character collection: " << vocab.size() << std::endl;
}

int BPETokenizer::getVocabSize() const {
    return vocab_size;
}