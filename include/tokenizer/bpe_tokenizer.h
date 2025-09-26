#ifndef BPE_TOKENIZER_H
#define BPE_TOKENIZER_H

#include <vector>
#include <string>
#include <unordered_map>

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
        BPETokenizer(int vocab_size);
        void train(const std::string& training_text);
        std::vector<int> encode(const std::string& text);
        std::string decode(const std::vector<int>& tokens);
        
        int getVocabSize() const;
};

#endif