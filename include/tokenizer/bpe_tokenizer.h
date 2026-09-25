#pragma once

#include <vector>
#include <string>
#include <unordered_map>

struct PairHash {
    size_t operator()(const std::pair<std::string, std::string>& p) const {
        // hash_combine (Boost / N3876): mixes both halves so (a,b) and
        // (b,a) no longer collide and equal halves no longer cancel.
        const size_t h1 = std::hash<std::string>{}(p.first);
        const size_t h2 = std::hash<std::string>{}(p.second);
        return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
    }
};

class BPETokenizer {
    private:
        std::unordered_map<std::string, int> vocab;
        std::unordered_map<int, std::string> id_to_token;
        std::vector<std::pair<std::string, std::string>> merges;
        int vocab_size;

        int pad_token_id = 0;
        int eos_token_id = 1;
        int unk_token_id = 2;
        
        std::string pad_token = "<pad>";
        std::string eos_token = "<eos>";
        std::string unk_token = "<unk>";
    public:
        explicit BPETokenizer(int vocab_size);
        void train(const std::string& training_text);
        std::vector<int> encode(const std::string& text) const;
        std::string decode(const std::vector<int>& tokens) const;

        void save(const std::string& filepath) const;
        // Throws on a missing, truncated, or implausible cache (a corrupt
        // length field would otherwise become a multi-gigabyte allocation).
        // On failure the tokenizer is left unchanged.
        void load(const std::string& filepath);

        int getCurrentVocabSize() const;
        int getVocabSize() const;
};
