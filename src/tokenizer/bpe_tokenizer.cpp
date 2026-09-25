#include "grad/tokenizer/bpe_tokenizer.h"
#include <iostream>
#include <iomanip>
#include <set>
#include <sstream>
#include <fstream>
#include <chrono>
#include <cstdint>
#include <stdexcept>

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
    auto start_time = std::chrono::steady_clock::now();

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
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
            float progress = 100.0f * merge_count / total_merges;
            float merges_per_sec = merge_count / static_cast<float>(elapsed + 1);
            int eta_sec = static_cast<int>((total_merges - merge_count) / (merges_per_sec + 0.001f));

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

std::vector<int> BPETokenizer::encode(const std::string& text) const {
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

    // Progress output only makes sense for corpus-scale encodes; stay
    // silent for interactive prompts.
    const bool show_progress = total_words > 20000;

    for (size_t word_idx = 0; word_idx < words.size(); word_idx++) {
        const std::string& word = words[word_idx];

        if (show_progress && word_idx % 5000 == 0) {
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

    if (show_progress) {
        std::cout << "\r  Encoding: [" << total_words << "/" << total_words << "] 100.0% "
                  << "cache_hits=" << cache_hits << "     " << std::endl;
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
    if (!file) {
        throw std::runtime_error("Failed while writing tokenizer cache: " + filepath);
    }
}

namespace {

// Upper bounds for the cache's length fields. They are far above anything
// train() produces (the 16000-entry TinyStories cache has 15893 merges and
// a longest token of 16 bytes), and exist only so a corrupt field fails
// with a message instead of attempting an allocation of that size.
constexpr uint64_t kMaxCacheEntries = 1u << 24;
constexpr uint64_t kMaxTokenBytes = 1u << 16;

template <typename T>
T read_pod(std::istream& in, const std::string& path, const char* what) {
    T value;
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!in) {
        throw std::runtime_error("Tokenizer cache truncated reading " + std::string(what)
                                 + ": " + path);
    }
    return value;
}

std::string read_token(std::istream& in, const std::string& path) {
    const size_t len = read_pod<size_t>(in, path, "token length");
    if (len > kMaxTokenBytes) {
        throw std::runtime_error("Tokenizer cache has an implausible token length ("
                                 + std::to_string(len) + " bytes): " + path);
    }
    std::string token(len, '\0');
    in.read(token.data(), static_cast<std::streamsize>(len));
    if (!in) {
        throw std::runtime_error("Tokenizer cache truncated reading a token: " + path);
    }
    return token;
}

}  // namespace

void BPETokenizer::load(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file for loading: " + filepath);
    }

    // Parse into locals and swap in only once the whole file has read
    // cleanly, so a bad cache cannot leave a half-loaded tokenizer.
    std::unordered_map<std::string, int> new_vocab;
    std::unordered_map<int, std::string> new_id_to_token;
    std::vector<std::pair<std::string, std::string>> new_merges;

    const size_t vocab_map_size = read_pod<size_t>(file, filepath, "vocab size");
    if (vocab_map_size > kMaxCacheEntries) {
        throw std::runtime_error("Tokenizer cache has an implausible vocab size ("
                                 + std::to_string(vocab_map_size) + "): " + filepath);
    }
    new_vocab.reserve(vocab_map_size);
    new_id_to_token.reserve(vocab_map_size);
    for (size_t i = 0; i < vocab_map_size; i++) {
        std::string key = read_token(file, filepath);
        const int value = read_pod<int>(file, filepath, "token id");
        if (value < 0) {
            throw std::runtime_error("Tokenizer cache has a negative token id: " + filepath);
        }
        new_id_to_token[value] = key;
        new_vocab[std::move(key)] = value;
    }

    const size_t merges_size = read_pod<size_t>(file, filepath, "merge count");
    if (merges_size > kMaxCacheEntries) {
        throw std::runtime_error("Tokenizer cache has an implausible merge count ("
                                 + std::to_string(merges_size) + "): " + filepath);
    }
    new_merges.reserve(merges_size);
    for (size_t i = 0; i < merges_size; i++) {
        std::string first = read_token(file, filepath);
        std::string second = read_token(file, filepath);
        new_merges.emplace_back(std::move(first), std::move(second));
    }

    vocab.swap(new_vocab);
    id_to_token.swap(new_id_to_token);
    merges.swap(new_merges);
}
