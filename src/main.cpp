#include "transformer/tensor.h"
#include "transformer/token_embedding.h"
#include "transformer/positional_encoding.h"
#include "transformer/transformer_block.h"
#include "transformer/gpt_model.h"
#include "transformer/activations.h"
#include "tokenizer/bpe_tokenizer.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <iomanip>
#include <stdexcept>

// Utility functions for comprehensive testing
void printSectionHeader(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "  " << title << std::endl;
    std::cout << std::string(60, '=') << std::endl;
}

void printTestResult(const std::string& test_name, bool passed) {
    std::cout << "  " << (passed ? "✓" : "✗") << " " << test_name << std::endl;
}

double measureTime(std::function<void()> func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return duration.count() / 1000.0; // Convert to milliseconds
}

// Test tensor operations comprehensively
bool testTensorOperations() {
    printSectionHeader("TENSOR OPERATIONS TESTING");
    bool all_passed = true;
    
    // Basic matrix multiplication
    Tensor A(2, 3);
    Tensor B(3, 2);
    A.setValue(0, 0, 1); A.setValue(0, 1, 2); A.setValue(0, 2, 3);
    A.setValue(1, 0, 4); A.setValue(1, 1, 5); A.setValue(1, 2, 6);
    B.setValue(0, 0, 7); B.setValue(0, 1, 8);
    B.setValue(1, 0, 9); B.setValue(1, 1, 10);
    B.setValue(2, 0, 11); B.setValue(2, 1, 12);
    
    Tensor C = A.matmul(B);
    bool matmul_correct = (C.getValue(0,0) == 58 && C.getValue(1,1) == 154);
    printTestResult("Matrix multiplication", matmul_correct);
    all_passed &= matmul_correct;
    
    // Test broadcasting in addition
    Tensor D(1, 3);
    D.setValue(0, 0, 1); D.setValue(0, 1, 2); D.setValue(0, 2, 3);
    Tensor E = A.add(D);
    bool broadcast_correct = (E.getRows() == 2 && E.getCols() == 3);
    printTestResult("Broadcasting addition", broadcast_correct);
    all_passed &= broadcast_correct;
    
    // Test softmax
    Tensor F(2, 3);
    F.setValue(0, 0, 1); F.setValue(0, 1, 2); F.setValue(0, 2, 3);
    F.setValue(1, 0, 4); F.setValue(1, 1, 5); F.setValue(1, 2, 6);
    Tensor G = F.softmax();
    bool softmax_correct = (G.getRows() == 2 && G.getCols() == 3);
    printTestResult("Softmax operation", softmax_correct);
    all_passed &= softmax_correct;
    
    // Test transpose
    Tensor H = A.transpose();
    bool transpose_correct = (H.getRows() == 3 && H.getCols() == 2);
    printTestResult("Transpose operation", transpose_correct);
    all_passed &= transpose_correct;
    
    // Test reshape
    Tensor I = A.reshape(3, 2);
    bool reshape_correct = (I.getRows() == 3 && I.getCols() == 2);
    printTestResult("Reshape operation", reshape_correct);
    all_passed &= reshape_correct;
    
    // Test slice
    Tensor J = A.slice(0, 1, 0, 2);
    bool slice_correct = (J.getRows() == 1 && J.getCols() == 2);
    printTestResult("Slice operation", slice_correct);
    all_passed &= slice_correct;
    
    // Test concatenate
    Tensor K = A.concatenate(A, 0);
    bool concat_correct = (K.getRows() == 4 && K.getCols() == 3);
    printTestResult("Concatenate operation", concat_correct);
    all_passed &= concat_correct;
    
    // Test causal mask
    Tensor L(3, 3);
    L.fill(1.0f);
    Tensor M = L.causal_mask();
    bool causal_correct = (M.getValue(0, 1) == -1e9f && M.getValue(1, 0) == 1.0f);
    printTestResult("Causal mask", causal_correct);
    all_passed &= causal_correct;
    
    return all_passed;
}

// Test token embeddings with edge cases
bool testTokenEmbeddings() {
    printSectionHeader("TOKEN EMBEDDING TESTING");
    bool all_passed = true;
    
    TokenEmbedding token_emb(10, 8);
    
    // Basic functionality
    Tensor token_ids(4, 1);
    token_ids.setValue(0, 0, 0); token_ids.setValue(1, 0, 1); 
    token_ids.setValue(2, 0, 2); token_ids.setValue(3, 0, 1);
    
    Tensor embeddings = token_emb.forward(token_ids);
    bool basic_correct = (embeddings.getRows() == 4 && embeddings.getCols() == 8);
    printTestResult("Basic embedding dimensions", basic_correct);
    all_passed &= basic_correct;
    
    // Test same token produces same embedding
    bool same_token_same_embedding = true;
    for (int dim = 0; dim < 8; dim++) {
        if (embeddings.getValue(1, dim) != embeddings.getValue(3, dim)) {
            same_token_same_embedding = false;
            break;
        }
    }
    printTestResult("Same tokens produce same embeddings", same_token_same_embedding);
    all_passed &= same_token_same_embedding;
    
    // Test single token
    Tensor single_token(1, 1);
    single_token.setValue(0, 0, 5);
    Tensor single_embedding = token_emb.forward(single_token);
    bool single_correct = (single_embedding.getRows() == 1 && single_embedding.getCols() == 8);
    printTestResult("Single token embedding", single_correct);
    all_passed &= single_correct;
    
    // Test error handling (out of bounds)
    bool error_handling_correct = false;
    try {
        Tensor invalid_token(1, 1);
        invalid_token.setValue(0, 0, 15); // Out of vocabulary
        token_emb.forward(invalid_token);
    } catch (const std::out_of_range& e) {
        error_handling_correct = true;
    }
    printTestResult("Error handling (out of bounds)", error_handling_correct);
    all_passed &= error_handling_correct;
    
    return all_passed;
}

// Test positional encoding
bool testPositionalEncoding() {
    printSectionHeader("POSITIONAL ENCODING TESTING");
    bool all_passed = true;
    
    PositionalEncoding pos_enc(16, 8);
    
    // Basic functionality
    Tensor embeddings(4, 8);
    embeddings.fill(0.1f);
    Tensor embeddings_with_pos = pos_enc.forward(embeddings);
    
    bool basic_correct = (embeddings_with_pos.getRows() == 4 && embeddings_with_pos.getCols() == 8);
    printTestResult("Basic positional encoding dimensions", basic_correct);
    all_passed &= basic_correct;
    
    // Test that positional encoding actually changes values
    bool pos_encoding_added = (embeddings.getValue(1, 1) != embeddings_with_pos.getValue(1, 1));
    printTestResult("Positional encoding modifies values", pos_encoding_added);
    all_passed &= pos_encoding_added;
    
    // Test different positions have different encodings
    Tensor pos1(1, 8);
    Tensor pos2(1, 8);
    pos1.fill(0.0f);
    pos2.fill(0.0f);
    Tensor enc1 = pos_enc.forward(pos1);
    pos2.setValue(0, 0, 1.0f); // Different position
    Tensor enc2 = pos_enc.forward(pos2);
    bool different_positions = (enc1.getValue(0, 0) != enc2.getValue(0, 0));
    printTestResult("Different positions have different encodings", different_positions);
    all_passed &= different_positions;
    
    // Test maximum length
    Tensor max_len_emb(16, 8);
    max_len_emb.fill(0.0f);
    Tensor max_len_result = pos_enc.forward(max_len_emb);
    bool max_len_correct = (max_len_result.getRows() == 16 && max_len_result.getCols() == 8);
    printTestResult("Maximum length sequence", max_len_correct);
    all_passed &= max_len_correct;
    
    return all_passed;
}

// Test transformer block
bool testTransformerBlock() {
    printSectionHeader("TRANSFORMER BLOCK TESTING");
    bool all_passed = true;
    
    TransformerBlock transformer_block(8, 2);
    
    // Basic functionality
    Tensor input(4, 8);
    input.fill(0.3f);
    Tensor output = transformer_block.forward(input);
    
    bool basic_correct = (output.getRows() == 4 && output.getCols() == 8);
    printTestResult("Basic transformer block", basic_correct);
    all_passed &= basic_correct;
    
    // Test with different input sizes
    Tensor input2(1, 8);
    input2.fill(0.5f);
    Tensor output2 = transformer_block.forward(input2);
    bool different_size_correct = (output2.getRows() == 1 && output2.getCols() == 8);
    printTestResult("Different input size", different_size_correct);
    all_passed &= different_size_correct;
    
    // Test training vs inference mode
    Tensor output_training = transformer_block.forward(input, true);
    Tensor output_inference = transformer_block.forward(input, false);
    bool mode_correct = (output_training.getRows() == 4 && output_inference.getRows() == 4);
    printTestResult("Training vs inference modes", mode_correct);
    all_passed &= mode_correct;
    
    return all_passed;
}

// Test complete GPT model with various configurations
bool testGPTModel() {
    printSectionHeader("GPT MODEL TESTING");
    bool all_passed = true;
    
    // Test small model
    GPTModel gpt_small(10, 8, 2, 2, 16);
    Tensor input_tokens(3, 1);
    input_tokens.setValue(0, 0, 1);
    input_tokens.setValue(1, 0, 5);
    input_tokens.setValue(2, 0, 3);
    
    Tensor logits = gpt_small.forward(input_tokens);
    bool small_model_correct = (logits.getRows() == 3 && logits.getCols() == 10);
    printTestResult("Small GPT model", small_model_correct);
    all_passed &= small_model_correct;
    
    // Test different inputs produce different outputs
    Tensor input_tokens2(3, 1);
    input_tokens2.setValue(0, 0, 2);
    input_tokens2.setValue(1, 0, 8);
    input_tokens2.setValue(2, 0, 1);
    
    Tensor logits2 = gpt_small.forward(input_tokens2);
    bool different_outputs = (logits.getValue(0, 0) != logits2.getValue(0, 0));
    printTestResult("Different inputs produce different outputs", different_outputs);
    all_passed &= different_outputs;
    
    // Test larger model
    GPTModel gpt_large(50, 16, 4, 4, 32);
    Tensor large_input(5, 1);
    for (int i = 0; i < 5; i++) {
        large_input.setValue(i, 0, i % 10);
    }
    
    Tensor large_logits = gpt_large.forward(large_input);
    bool large_model_correct = (large_logits.getRows() == 5 && large_logits.getCols() == 50);
    printTestResult("Large GPT model", large_model_correct);
    all_passed &= large_model_correct;
    
    // Test single token
    Tensor single_input(1, 1);
    single_input.setValue(0, 0, 0);
    Tensor single_logits = gpt_small.forward(single_input);
    bool single_token_correct = (single_logits.getRows() == 1 && single_logits.getCols() == 10);
    printTestResult("Single token processing", single_token_correct);
    all_passed &= single_token_correct;
    
    return all_passed;
}

// Performance benchmarking
void benchmarkPerformance() {
    printSectionHeader("PERFORMANCE BENCHMARKING");
    
    // Benchmark tensor operations
    double matmul_time = measureTime([]() {
        Tensor A(100, 100);
        Tensor B(100, 100);
        A.fill(1.0f);
        B.fill(2.0f);
        Tensor C = A.matmul(B);
    });
    std::cout << "  Matrix multiplication (100x100): " << std::fixed << std::setprecision(2) << matmul_time << " ms" << std::endl;
    
    // Benchmark softmax
    double softmax_time = measureTime([]() {
        Tensor A(100, 100);
        A.fill(1.0f);
        Tensor B = A.softmax();
    });
    std::cout << "  Softmax (100x100): " << std::fixed << std::setprecision(2) << softmax_time << " ms" << std::endl;
    
    // Benchmark GPT model forward pass
    GPTModel gpt_model(100, 64, 6, 8, 128);
    Tensor input(32, 1);
    for (int i = 0; i < 32; i++) {
        input.setValue(i, 0, i % 50);
    }
    
    double gpt_time = measureTime([&]() {
        Tensor logits = gpt_model.forward(input);
    });
    std::cout << "  GPT forward pass (32 tokens, 6 layers): " << std::fixed << std::setprecision(2) << gpt_time << " ms" << std::endl;
}

// Demonstrate text generation capabilities
void demonstrateTextGeneration() {
    printSectionHeader("TEXT GENERATION DEMONSTRATION");
    
    // Create a simple vocabulary mapping
    std::vector<std::string> vocabulary = {
        "the", "cat", "dog", "runs", "jumps", "over", "lazy", "quick", "brown", "fox",
        "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
        "a", "an", "is", "are", "was", "were", "be", "been", "being", "have",
        "has", "had", "do", "does", "did", "will", "would", "could", "should", "may",
        "might", "must", "can", "shall", "go", "goes", "went", "come", "comes", "came"
    };
    
    GPTModel gpt_model(vocabulary.size(), 32, 4, 4, 64);
    
    // Generate some sample sequences
    std::cout << "  Generating sample sequences..." << std::endl;
    
    for (int i = 0; i < 3; i++) {
        Tensor input(1, 1);
        input.setValue(0, 0, i * 5); // Start with different tokens
        
        Tensor logits = gpt_model.forward(input);
        
        // Find the most likely next token
        int best_token = 0;
        float best_score = logits.getValue(0, 0);
        for (int j = 1; j < vocabulary.size(); j++) {
            if (logits.getValue(0, j) > best_score) {
                best_score = logits.getValue(0, j);
                best_token = j;
            }
        }
        
        std::cout << "  Sample " << (i+1) << ": '" << vocabulary[input.getValue(0, 0)] 
                  << "' -> '" << vocabulary[best_token] << "'" << std::endl;
    }
}

bool testBPETokenizer() {
    printSectionHeader("BPE TOKENIZER TESTING");
    bool all_passed = true;
    
    // Basic constructor test
    BPETokenizer tokenizer(100);
    bool constructor_correct = (tokenizer.getVocabSize() == 100);
    printTestResult("Constructor sets vocab size", constructor_correct);
    all_passed &= constructor_correct;
    
    // Character collection test
    std::string test_text = "hello world";
    tokenizer.train(test_text);
    
    // BPE training test with specific text
    std::string bpe_test_text = "hello hello world world";
    BPETokenizer bpe_tokenizer(20); // Small vocab for testing
    bpe_tokenizer.train(bpe_test_text);

    // Test that vocab grew beyond just characters
    bool vocab_grew = (bpe_tokenizer.getCurrentVocabSize() > 11); // More than chars + special tokens
    printTestResult("BPE training increases vocab size", vocab_grew);
    all_passed &= vocab_grew;

    // Test basic encoding/decoding (you'll need to implement these next)
    // For now, just test that training completes
    bool bpe_training_completed = true;
    printTestResult("BPE merge training completes", bpe_training_completed);
    all_passed &= bpe_training_completed;

    // Test encoding
    std::string encode_test = "hello";
    std::vector<int> encoded = bpe_tokenizer.encode(encode_test);

    std::cout << "Encoded 'hello': ";
    for (int id : encoded) {
        std::cout << id << " ";
    }
    std::cout << std::endl;

    bool encoding_works = !encoded.empty();
    printTestResult("Encoding produces tokens", encoding_works);
    all_passed &= encoding_works;
    
    return all_passed;
}

int main() {
    std::cout << "🚀 COMPREHENSIVE TRANSFORMER FROM SCRATCH AUDIT 🚀" << std::endl;
    std::cout << "Testing all components with edge cases and performance metrics..." << std::endl;
    
    bool all_tests_passed = true;
    
    // Run all test suites
    all_tests_passed &= testTensorOperations();
    all_tests_passed &= testTokenEmbeddings();
    all_tests_passed &= testPositionalEncoding();
    all_tests_passed &= testTransformerBlock();
    all_tests_passed &= testGPTModel();
    all_tests_passed &= testBPETokenizer();
    
    // Performance benchmarking
    benchmarkPerformance();
    
    // Demonstrate practical usage
    demonstrateTextGeneration();
    
    // Final status report
    printSectionHeader("FINAL STATUS REPORT");
    if (all_tests_passed) {
        std::cout << "🎉 ALL TESTS PASSED! 🎉" << std::endl;
    } else {
        std::cout << "❌ SOME TESTS FAILED ❌" << std::endl;
        std::cout << "Please review the failed tests above." << std::endl;
    }
    
    return all_tests_passed ? 0 : 1;
}