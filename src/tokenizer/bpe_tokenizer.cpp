#include "tokenizer/bpe_tokenizer.h"
#include <iostream>
#include <iomanip>
#include <set>
#include <sstream>
#include <fstream>
#include <chrono>

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

    std::unordered_map<std::string, int> word_freqs;
    std::istringstream iss(training_text);
    std::string word;
    bool first = true;
    while (iss >> word) {
        if (!first) {
            word_freqs["_" + word]++;
        } else {
            word_freqs[word]++;
            first = false;
        }
    }

    std::unordered_map<std::string, std::vector<std::string>> word_tokens;
    word_tokens.reserve(word_freqs.size());
    for (const auto& entry : word_freqs) {
        std::vector<std::string> chars;
        chars.reserve(entry.first.size());
        for (char c : entry.first) {
            chars.push_back(std::string(1, c));
        }
        word_tokens[entry.first] = chars;
    }

    std::unordered_map<std::pair<std::string, std::string>, int, PairHash> pair_counts;
    for (const auto& entry : word_tokens) {
        const auto& word = entry.second;
        int freq = word_freqs[entry.first];
        for (size_t i = 0; i < word.size() - 1; i++) {
            std::pair<std::string, std::string> pair = {word[i], word[i + 1]};
            pair_counts[pair] += freq;
        }
    }

    int merge_count = 0;
    int total_merges = vocab_size - vocab.size();
    auto start_time = std::chrono::high_resolution_clock::now();

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
        merge_count++;

        if (merge_count % 100 == 0 || merge_count == total_merges) {
            auto now = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
            float progress = 100.0f * merge_count / total_merges;
            float merges_per_sec = merge_count / (float)(elapsed + 1);
            int eta_sec = (int)((total_merges - merge_count) / (merges_per_sec + 0.001f));

            std::cout << "\r  Training tokenizer: [" << merge_count << "/" << total_merges << "] "
                      << std::fixed << std::setprecision(1) << progress << "% "
                      << "elapsed=" << elapsed << "s "
                      << "eta=" << (eta_sec / 60) << "m" << (eta_sec % 60) << "s     " << std::flush;
        }

        pair_counts.erase(most_freq_pair);

        for (auto& entry : word_tokens) {
            const std::string& word_key = entry.first;
            auto& word = entry.second;
            int freq = word_freqs[word_key];

            std::vector<std::string> new_word;
            new_word.reserve(word.size());

            for (size_t i = 0; i < word.size(); i++) {
                if (i < word.size() - 1 && word[i] == most_freq_pair.first && word[i + 1] == most_freq_pair.second) {
                    if (i > 0) {
                        std::pair<std::string, std::string> left_pair = {word[i-1], word[i]};
                        pair_counts[left_pair] -= freq;
                        if (pair_counts[left_pair] <= 0) {
                            pair_counts.erase(left_pair);
                        }
                    }

                    if (i + 2 < word.size()) {
                        std::pair<std::string, std::string> right_pair = {word[i+1], word[i+2]};
                        pair_counts[right_pair] -= freq;
                        if (pair_counts[right_pair] <= 0) {
                            pair_counts.erase(right_pair);
                        }
                    }

                    new_word.push_back(merged_token);

                    if (i > 0) {
                        std::pair<std::string, std::string> new_left_pair = {word[i-1], merged_token};
                        pair_counts[new_left_pair] += freq;
                    }

                    if (i + 2 < word.size()) {
                        std::pair<std::string, std::string> new_right_pair = {merged_token, word[i+2]};
                        pair_counts[new_right_pair] += freq;
                    }

                    i++;
                } else {
                    new_word.push_back(word[i]);
                }
            }
            word = new_word;
        }
    }
    std::cout << std::endl;
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

    std::unordered_map<std::string, std::vector<int>> word_cache;
    std::vector<int> token_ids;
    size_t total_words = words.size();
    size_t cache_hits = 0;

    for (size_t word_idx = 0; word_idx < words.size(); word_idx++) {
        const std::string& word = words[word_idx];

        if (word_idx % 5000 == 0) {
            float progress = 100.0f * word_idx / total_words;
            std::cout << "\r  Encoding: [" << word_idx << "/" << total_words << "] "
                      << std::fixed << std::setprecision(1) << progress << "% "
                      << "cache_hits=" << cache_hits << "     " << std::flush;
        }

        auto cache_it = word_cache.find(word);
        if (cache_it != word_cache.end()) {
            cache_hits++;
            token_ids.insert(token_ids.end(), cache_it->second.begin(), cache_it->second.end());
            continue;
        }

        std::vector<std::string> word_tokens;
        word_tokens.reserve(word.size());
        for (char c : word) {
            word_tokens.push_back(std::string(1, c));
        }

        for (const auto& merge_pair : merges) {
            if (word_tokens.size() <= 1) break;

            bool has_pair = false;
            for(size_t i = 0; i < word_tokens.size() - 1; i++) {
                if (word_tokens[i] == merge_pair.first && word_tokens[i+1] == merge_pair.second) {
                    has_pair = true;
                    break;
                }
            }
            if (!has_pair) continue;

            std::vector<std::string> new_word_tokens;
            new_word_tokens.reserve(word_tokens.size());

            for(size_t i = 0; i < word_tokens.size(); i++) {
                if (i < word_tokens.size() - 1 && word_tokens[i] == merge_pair.first && word_tokens[i+1] == merge_pair.second) {
                    new_word_tokens.push_back(merge_pair.first + merge_pair.second);
                    i++;
                } else {
                    new_word_tokens.push_back(word_tokens[i]);
                }
            }

            word_tokens = std::move(new_word_tokens);
        }

        std::vector<int> word_token_ids;
        for (const std::string& token : word_tokens) {
            auto it = vocab.find(token);
            if (it != vocab.end()) {
                word_token_ids.push_back(it->second);
            } else {
                word_token_ids.push_back(unk_token_id);
            }
        }

        word_cache[word] = word_token_ids;
        token_ids.insert(token_ids.end(), word_token_ids.begin(), word_token_ids.end());
    }

    if (total_words > 0) {
        std::cout << "\r  Encoding: [" << total_words << "/" << total_words << "] 100.0% "
                  << "cache_hits=" << cache_hits << "     " << std::flush;
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

void BPETokenizer::save(const std::string& filepath) const {
    std::ofstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file for saving: " + filepath);
    }

    size_t vocab_map_size = vocab.size();
    file.write(reinterpret_cast<const char*>(&vocab_map_size), sizeof(vocab_map_size));
    for (const auto& pair : vocab) {
        size_t key_len = pair.first.size();
        file.write(reinterpret_cast<const char*>(&key_len), sizeof(key_len));
        file.write(pair.first.data(), key_len);
        file.write(reinterpret_cast<const char*>(&pair.second), sizeof(pair.second));
    }

    size_t merges_size = merges.size();
    file.write(reinterpret_cast<const char*>(&merges_size), sizeof(merges_size));
    for (const auto& merge_pair : merges) {
        size_t first_len = merge_pair.first.size();
        size_t second_len = merge_pair.second.size();
        file.write(reinterpret_cast<const char*>(&first_len), sizeof(first_len));
        file.write(merge_pair.first.data(), first_len);
        file.write(reinterpret_cast<const char*>(&second_len), sizeof(second_len));
        file.write(merge_pair.second.data(), second_len);
    }

    file.close();
}

void BPETokenizer::load(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file for loading: " + filepath);
    }

    vocab.clear();
    id_to_token.clear();
    merges.clear();

    size_t vocab_map_size;
    file.read(reinterpret_cast<char*>(&vocab_map_size), sizeof(vocab_map_size));
    for (size_t i = 0; i < vocab_map_size; i++) {
        size_t key_len;
        file.read(reinterpret_cast<char*>(&key_len), sizeof(key_len));
        std::string key(key_len, '\0');
        file.read(&key[0], key_len);
        int value;
        file.read(reinterpret_cast<char*>(&value), sizeof(value));
        vocab[key] = value;
        id_to_token[value] = key;
    }

    size_t merges_size;
    file.read(reinterpret_cast<char*>(&merges_size), sizeof(merges_size));
    for (size_t i = 0; i < merges_size; i++) {
        size_t first_len, second_len;
        file.read(reinterpret_cast<char*>(&first_len), sizeof(first_len));
        std::string first(first_len, '\0');
        file.read(&first[0], first_len);
        file.read(reinterpret_cast<char*>(&second_len), sizeof(second_len));
        std::string second(second_len, '\0');
        file.read(&second[0], second_len);
        merges.push_back({first, second});
    }

    file.close();
}