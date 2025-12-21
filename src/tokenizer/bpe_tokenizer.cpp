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
    unique_chars.insert('_');
    for (char c : training_text) {
        if (c != ' ') {
            unique_chars.insert(c);
        }
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
    bool first = true;
    while (iss >> word) {
        if (!first) {
            words.push_back("_" + word);
        } else {
            words.push_back(word);
            first = false;
        }
    }

    std::vector<std::vector<std::string>> word_tokens;
    for (const std::string& word : words) {
        std::vector<std::string> chars;
        for (char c : word) {
            chars.push_back(std::string(1, c));
        }
        word_tokens.push_back(chars);
    }

    auto pair_counts = countPairs(word_tokens);

    while(vocab.size() < static_cast<size_t>(vocab_size)) {
        if (pair_counts.empty()) { break; }

        std::pair<std::string, std::string> most_freq_pair;
        int max_count = 0;

        for (const auto& entry : pair_counts) {
            if (entry.second > max_count) {
                max_count = entry.second;
                most_freq_pair = entry.first;
            }
        }

        if (max_count == 0) { break; }

        std::string merged_token = most_freq_pair.first + most_freq_pair.second;
        vocab[merged_token] = curr_id;
        id_to_token[curr_id] = merged_token;
        merges.push_back(most_freq_pair);
        curr_id++;

        pair_counts.erase(most_freq_pair);

        for (auto& word : word_tokens) {
            std::vector<std::string> new_word;
            for (size_t i = 0; i < word.size(); i++) {
                if (i < word.size() - 1 && word[i] == most_freq_pair.first && word[i + 1] == most_freq_pair.second) {
                    if (i > 0) {
                        std::pair<std::string, std::string> left_pair = {word[i-1], word[i]};
                        if (pair_counts[left_pair] > 0) {
                            pair_counts[left_pair]--;
                            if (pair_counts[left_pair] == 0) {
                                pair_counts.erase(left_pair);
                            }
                        }
                    }

                    if (i + 2 < word.size()) {
                        std::pair<std::string, std::string> right_pair = {word[i+1], word[i+2]};
                        if (pair_counts[right_pair] > 0) {
                            pair_counts[right_pair]--;
                            if (pair_counts[right_pair] == 0) {
                                pair_counts.erase(right_pair);
                            }
                        }
                    }

                    new_word.push_back(merged_token);

                    if (i > 0) {
                        std::pair<std::string, std::string> new_left_pair = {word[i-1], merged_token};
                        pair_counts[new_left_pair]++;
                    }

                    if (i + 2 < word.size()) {
                        std::pair<std::string, std::string> new_right_pair = {merged_token, word[i+2]};
                        pair_counts[new_right_pair]++;
                    }

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
    bool first = true;
    while (iss >> word) {
        if (!first) {
            words.push_back("_" + word);
        } else {
            words.push_back(word);
            first = false;
        }
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

std::string BPETokenizer::decode(const std::vector<int>& token_ids) const {
    std::string result;

    for (size_t i = 0; i < token_ids.size(); i++) {
        auto it = id_to_token.find(token_ids[i]);
        if (it != id_to_token.end()) {
            result += it->second;
        }
    }

    for (size_t i = 0; i < result.length(); i++) {
        if (result[i] == '_') {
            result[i] = ' ';
        }
    }

    return result;
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