#include "tokenizer/bpe_tokenizer.h"
#include <iostream>
#include <set>
#include <sstream>

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
    
    std::vector<std::string> words;
    std::istringstream iss(training_text);
    std::string word;
    while (iss >> word) {
        words.push_back(word);
    }

    std::vector<std::vector<std::string>> word_tokens;
    for (const std::string& word : words) {
        std::vector<std::string> chars;
        for (char c : word) {
            chars.push_back(std::string(1, c));
        }
        word_tokens.push_back(chars);
    }

    while(vocab.size() < vocab_size) {
        auto pair_counts = countPairs(word_tokens);

        if (pair_counts.empty()) { break; }

        std::pair<std::string, std::string> most_freq_pair;
        int max_count = 0;

        for (const auto& entry : pair_counts) {
            if (entry.second > max_count) {
                max_count = entry.second;
                most_freq_pair = entry.first;
            }
        }

        std::string merged_token = most_freq_pair.first + most_freq_pair.second;
        vocab[merged_token] = curr_id;
        id_to_token[curr_id] = merged_token;
        merges.push_back(most_freq_pair);
        curr_id++;

        for (auto& word : word_tokens) {
            std::vector<std::string> new_word;
            for (size_t i = 0; i < word.size(); i++) {
                if (i < word.size() - 1 && word[i] == most_freq_pair.first && word[i + 1] == most_freq_pair.second) {
                    new_word.push_back(merged_token);
                    i++;
                } else {
                    new_word.push_back(word[i]);
                }
            }
            word = new_word;
        }
    }
}

std::vector<int> BPETokenizer::encode(const std::string& text) {
    std::vector<std::string> words;
    std::istringstream iss(text);
    std::string word;
    while (iss >> word) {
        words.push_back(word);
    }

    std::vector<int> token_ids;

    for (const std::string& word : words) {
        std::vector<std::string> word_tokens;
        for (char c : word) {
            word_tokens.push_back(std::string(1, c));
        }

        for (const auto& merge_pair : merges) {
            std::vector<std::string> new_word_tokens;

            for(size_t i = 0; i < word_tokens.size(); i++) {
                if (i < word_tokens.size() - 1 && word_tokens[i] == merge_pair.first && word_tokens[i+1] == merge_pair.second) {
                    new_word_tokens.push_back(merge_pair.first + merge_pair.second);
                    i++;
                } else {
                    new_word_tokens.push_back(word_tokens[i]);
                }
            }
            word_tokens = new_word_tokens;
        }

        for (const std::string& token : word_tokens) {
            auto it = vocab.find(token);
            if (it != vocab.end()) {
                token_ids.push_back(it->second);
            } else {
                token_ids.push_back(unk_token_id);
            }
        }
    }
    return token_ids;
}

std::unordered_map<std::pair<std::string, std::string>, int, PairHash> BPETokenizer::countPairs(const std::vector<std::vector<std::string>>& word_tokens) {
    std::unordered_map<std::pair<std::string, std::string>, int, PairHash> pair_counts;
    for (const auto& word : word_tokens) {
        for (size_t i = 0; i < word.size() - 1; i++) {
            std::pair<std::string, std::string> pair = {word[i], word[i + 1]};
            pair_counts[pair]++;
        }
    }
    return pair_counts;
}

int BPETokenizer::getCurrentVocabSize() const {
    return vocab.size();
}

int BPETokenizer::getVocabSize() const {
    return vocab_size;
}