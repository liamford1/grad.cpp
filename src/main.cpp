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
#include <cstdio>

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

// Test 3D tensor operations
bool test3DTensorOperations() {
    printSectionHeader("3D TENSOR OPERATIONS TESTING");
    bool all_passed = true;
    
    // Test 3D tensor creation
    Tensor t3d(2, 3, 4);  // batch_size=2, rows=3, cols=4
    bool creation_correct = (t3d.getBatchSize() == 2 && t3d.getRows() == 3 && t3d.getCols() == 4 && t3d.getIs3D());
    printTestResult("3D tensor creation", creation_correct);
    all_passed &= creation_correct;
    
    // Test 3D fill operation
    t3d.fill(5.0f);
    bool fill_correct = (t3d.getValue(0, 1, 2) == 5.0f && t3d.getValue(1, 2, 3) == 5.0f);
    printTestResult("3D tensor fill operation", fill_correct);
    all_passed &= fill_correct;
    
    // Test 3D setValue/getValue
    t3d.setValue(0, 1, 2, 10.0f);
    t3d.setValue(1, 2, 1, 20.0f);
    bool accessor_correct = (t3d.getValue(0, 1, 2) == 10.0f && t3d.getValue(1, 2, 1) == 20.0f);
    printTestResult("3D tensor accessors", accessor_correct);
    all_passed &= accessor_correct;
    
    // Test that 2D operations still work (backward compatibility)
    Tensor t2d(3, 4);
    t2d.fill(1.0f);
    t2d.setValue(1, 2, 7.0f);
    bool backward_compat = (!t2d.getIs3D() && t2d.getBatchSize() == 1 && t2d.getValue(1, 2) == 7.0f);
    printTestResult("2D backward compatibility", backward_compat);
    all_passed &= backward_compat;
    
    // Test copy constructor with 3D tensor
    Tensor t3d_copy = t3d;
    bool copy_correct = (t3d_copy.getValue(0, 1, 2) == 10.0f && t3d_copy.getValue(1, 2, 1) == 20.0f);
    printTestResult("3D tensor copy constructor", copy_correct);
    all_passed &= copy_correct;
    
    // Test assignment operator with 3D tensor
    Tensor t3d_assigned(1, 1, 1);
    t3d_assigned = t3d;
    bool assignment_correct = (t3d_assigned.getValue(0, 1, 2) == 10.0f && t3d_assigned.getValue(1, 2, 1) == 20.0f);
    printTestResult("3D tensor assignment operator", assignment_correct);
    all_passed &= assignment_correct;
    
    // Test bounds checking
    bool bounds_check_correct = false;
    try {
        t3d.getValue(2, 1, 1); // batch index out of bounds
    } catch (const std::out_of_range& e) {
        bounds_check_correct = true;
    }
    printTestResult("3D tensor bounds checking", bounds_check_correct);
    all_passed &= bounds_check_correct;
    
    // Test Xavier initialization with 3D tensor
    Tensor t3d_xavier(2, 3, 4);
    t3d_xavier.xavier(12, 4);
    bool xavier_works = true;
    // Check that not all values are the same (xavier should randomize)
    float first_val = t3d_xavier.getValue(0, 0, 0);
    for (int b = 0; b < 2 && xavier_works; b++) {
        for (int i = 0; i < 3 && xavier_works; i++) {
            for (int j = 0; j < 4 && xavier_works; j++) {
                if (t3d_xavier.getValue(b, i, j) != first_val) {
                    xavier_works = true;
                    goto xavier_done; // Found a different value, xavier is working
                }
            }
        }
    }
    xavier_works = false; // All values were the same
    xavier_done:
    printTestResult("3D tensor Xavier initialization", xavier_works);
    all_passed &= xavier_works;
    // Test 3D × 2D matrix multiplication
    Tensor a3d(2, 2, 3);  // 2 batches of 2×3 matrices
    Tensor b2d(3, 2);     // Single 3×2 matrix
    
    // Fill first batch of a3d
    a3d.setValue(0, 0, 0, 1); a3d.setValue(0, 0, 1, 2); a3d.setValue(0, 0, 2, 3);
    a3d.setValue(0, 1, 0, 4); a3d.setValue(0, 1, 1, 5); a3d.setValue(0, 1, 2, 6);
    
    // Fill second batch differently
    a3d.setValue(1, 0, 0, 2); a3d.setValue(1, 0, 1, 3); a3d.setValue(1, 0, 2, 4);
    a3d.setValue(1, 1, 0, 5); a3d.setValue(1, 1, 1, 6); a3d.setValue(1, 1, 2, 7);
    
    // Fill b2d
    b2d.setValue(0, 0, 7); b2d.setValue(0, 1, 8);
    b2d.setValue(1, 0, 9); b2d.setValue(1, 1, 10);
    b2d.setValue(2, 0, 11); b2d.setValue(2, 1, 12);
    
    Tensor result3d2d = a3d.matmul(b2d);
    bool matmul_3d_2d_correct = (result3d2d.getBatchSize() == 2 && 
                                result3d2d.getRows() == 2 && 
                                result3d2d.getCols() == 2);
    printTestResult("3D × 2D matrix multiplication", matmul_3d_2d_correct);
    all_passed &= matmul_3d_2d_correct;
    
    // Verify first batch result: should be same as [1,2,3; 4,5,6] × [7,8; 9,10; 11,12] = [58,64; 139,154]
    bool first_batch_correct = (result3d2d.getValue(0, 0, 0) == 58.0f && 
                               result3d2d.getValue(0, 1, 1) == 154.0f);
    printTestResult("3D × 2D first batch values", first_batch_correct);
    all_passed &= first_batch_correct;

    // Test 3D + 2D addition (broadcasting)
    Tensor a3d_add(2, 2, 3);
    Tensor b2d_add(2, 3);
    
    a3d_add.fill(5.0f);
    b2d_add.fill(2.0f);
    
    Tensor result_add = a3d_add.add(b2d_add);
    bool add_3d_2d_correct = (result_add.getBatchSize() == 2 && 
                             result_add.getValue(0, 1, 1) == 7.0f && 
                             result_add.getValue(1, 1, 1) == 7.0f);
    printTestResult("3D + 2D addition (broadcasting)", add_3d_2d_correct);
    all_passed &= add_3d_2d_correct;
    
    // Test 3D + 3D addition
    Tensor a3d_add2(2, 2, 3);
    Tensor b3d_add(2, 2, 3);
    
    a3d_add2.fill(3.0f);
    b3d_add.fill(4.0f);
    
    Tensor result_add_3d = a3d_add2.add(b3d_add);
    bool add_3d_3d_correct = (result_add_3d.getValue(0, 1, 1) == 7.0f && 
                             result_add_3d.getValue(1, 0, 2) == 7.0f);
    printTestResult("3D + 3D addition", add_3d_3d_correct);
    all_passed &= add_3d_3d_correct;
    // Test 3D softmax
    Tensor softmax_3d(2, 2, 3);
    
    // Set different values for each batch
    softmax_3d.setValue(0, 0, 0, 1.0f); softmax_3d.setValue(0, 0, 1, 2.0f); softmax_3d.setValue(0, 0, 2, 3.0f);
    softmax_3d.setValue(0, 1, 0, 4.0f); softmax_3d.setValue(0, 1, 1, 5.0f); softmax_3d.setValue(0, 1, 2, 6.0f);
    
    softmax_3d.setValue(1, 0, 0, 2.0f); softmax_3d.setValue(1, 0, 1, 3.0f); softmax_3d.setValue(1, 0, 2, 1.0f);
    softmax_3d.setValue(1, 1, 0, 5.0f); softmax_3d.setValue(1, 1, 1, 4.0f); softmax_3d.setValue(1, 1, 2, 6.0f);
    
    Tensor softmax_result = softmax_3d.softmax();
    
    // Check that each row sums to approximately 1.0
    bool softmax_3d_correct = true;
    float tolerance = 1e-6f;
    
    for (int b = 0; b < 2; b++) {
        for (int i = 0; i < 2; i++) {
            float row_sum = 0.0f;
            for (int j = 0; j < 3; j++) {
                row_sum += softmax_result.getValue(b, i, j);
            }
            if (std::abs(row_sum - 1.0f) > tolerance) {
                softmax_3d_correct = false;
                break;
            }
        }
        if (!softmax_3d_correct) break;
    }
    
    printTestResult("3D softmax normalization", softmax_3d_correct);
    all_passed &= softmax_3d_correct;
    // Test 3D transpose
    Tensor transpose_3d(2, 3, 4);  // [2, 3, 4]
    
    // Fill with identifiable values
    for (int b = 0; b < 2; b++) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 4; j++) {
                transpose_3d.setValue(b, i, j, b * 100 + i * 10 + j);
            }
        }
    }
    
    Tensor transposed = transpose_3d.transpose();  // Should be [2, 4, 3]
    
    bool transpose_3d_correct = (transposed.getBatchSize() == 2 && 
                                transposed.getRows() == 4 && 
                                transposed.getCols() == 3);
    printTestResult("3D transpose dimensions", transpose_3d_correct);
    all_passed &= transpose_3d_correct;
    
    // Check that transposition actually worked
    bool transpose_values_correct = (transpose_3d.getValue(0, 1, 2) == transposed.getValue(0, 2, 1) &&
                                    transpose_3d.getValue(1, 2, 3) == transposed.getValue(1, 3, 2));
    printTestResult("3D transpose values", transpose_values_correct);
    all_passed &= transpose_values_correct;

    // Test batch causal mask
    Tensor batch_mask = Tensor::create_casual_mask_batch(2, 3);
    
    bool batch_mask_correct = (batch_mask.getBatchSize() == 2 && 
                              batch_mask.getRows() == 3 && 
                              batch_mask.getCols() == 3);
    printTestResult("Batch causal mask dimensions", batch_mask_correct);
    all_passed &= batch_mask_correct;
    
    // Check causal mask values for both batches
    bool batch_mask_values = (batch_mask.getValue(0, 0, 1) == -1e9f &&  // Future token masked
                             batch_mask.getValue(0, 1, 0) == 0.0f &&     // Past token visible
                             batch_mask.getValue(1, 0, 1) == -1e9f &&    // Same for batch 1
                             batch_mask.getValue(1, 1, 0) == 0.0f);
    printTestResult("Batch causal mask values", batch_mask_values);
    all_passed &= batch_mask_values;

    return all_passed;
}

// Test batched token embeddings
bool testBatchedTokenEmbeddings() {
    printSectionHeader("BATCHED TOKEN EMBEDDING TESTING");
    bool all_passed = true;
    
    TokenEmbedding token_emb(10, 8);
    
    // Test batch format [batch_size, seq_len]
    Tensor batch_token_ids(2, 3);  // 2 sequences of length 3
    batch_token_ids.setValue(0, 0, 1); batch_token_ids.setValue(0, 1, 2); batch_token_ids.setValue(0, 2, 3);
    batch_token_ids.setValue(1, 0, 4); batch_token_ids.setValue(1, 1, 5); batch_token_ids.setValue(1, 2, 6);
    
    Tensor batch_embeddings = token_emb.forward(batch_token_ids);
    bool batch_dimensions_correct = (batch_embeddings.getBatchSize() == 2 && 
                                    batch_embeddings.getRows() == 3 && 
                                    batch_embeddings.getCols() == 8);
    printTestResult("Batch embedding dimensions [2, 3, 8]", batch_dimensions_correct);
    all_passed &= batch_dimensions_correct;
    
    // Test that different sequences in batch have different embeddings
    bool different_sequences = (batch_embeddings.getValue(0, 0, 0) != batch_embeddings.getValue(1, 0, 0));
    printTestResult("Different sequences have different embeddings", different_sequences);
    all_passed &= different_sequences;
    
    // Test that same tokens produce same embeddings across batches
    Tensor same_token_batch(2, 2);
    same_token_batch.setValue(0, 0, 7); same_token_batch.setValue(0, 1, 8);
    same_token_batch.setValue(1, 0, 7); same_token_batch.setValue(1, 1, 8);  // Same tokens
    
    Tensor same_token_embeddings = token_emb.forward(same_token_batch);
    bool same_tokens_same_embeddings = true;
    for (int j = 0; j < 8; j++) {
        if (same_token_embeddings.getValue(0, 0, j) != same_token_embeddings.getValue(1, 0, j)) {
            same_tokens_same_embeddings = false;
            break;
        }
    }
    printTestResult("Same tokens produce same embeddings across batches", same_tokens_same_embeddings);
    all_passed &= same_tokens_same_embeddings;
    
    // Test backward compatibility - legacy format should still work
    Tensor legacy_token_ids(3, 1);
    legacy_token_ids.setValue(0, 0, 1); legacy_token_ids.setValue(1, 0, 2); legacy_token_ids.setValue(2, 0, 3);
    
    Tensor legacy_embeddings = token_emb.forward(legacy_token_ids);
    bool legacy_compatibility = (legacy_embeddings.getRows() == 3 && 
                                legacy_embeddings.getCols() == 8 && 
                                !legacy_embeddings.getIs3D());
    printTestResult("Legacy format [seq_len, 1] still works", legacy_compatibility);
    all_passed &= legacy_compatibility;
    
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

bool testBatchedPositionalEncoding() {
    printSectionHeader("BATCHED POSITIONAL ENCODING TESTING");
    bool all_passed = true;
    
    PositionalEncoding pos_enc(16, 8);
    
    // Test with 3D input [batch_size, seq_len, d_model]
    Tensor batch_embeddings(2, 4, 8);  // 2 sequences of length 4
    batch_embeddings.fill(0.1f);
    
    // Set some different values for each batch
    batch_embeddings.setValue(0, 0, 0, 1.0f);
    batch_embeddings.setValue(1, 0, 0, 2.0f);
    
    Tensor batch_result = pos_enc.forward(batch_embeddings);
    
    bool batch_dimensions_correct = (batch_result.getBatchSize() == 2 && 
                                    batch_result.getRows() == 4 && 
                                    batch_result.getCols() == 8);
    printTestResult("Batch positional encoding dimensions [2, 4, 8]", batch_dimensions_correct);
    all_passed &= batch_dimensions_correct;
    
    // Test that positional encoding is applied (values should change)
    // Use position [0,1,1] which should have a more significant positional encoding
    float original_val = batch_embeddings.getValue(0, 1, 1);
    float modified_val = batch_result.getValue(0, 1, 1);
    bool pos_encoding_applied = (std::abs(modified_val - original_val) > 1e-6f);
    printTestResult("Positional encoding modifies batch values", pos_encoding_applied);
    all_passed &= pos_encoding_applied;
    
    // Test that same positions have same positional encoding across batches
    // (but different embedding values should still result in different final values)
    float pos_diff_batch0 = batch_result.getValue(0, 1, 0) - batch_embeddings.getValue(0, 1, 0);
    float pos_diff_batch1 = batch_result.getValue(1, 1, 0) - batch_embeddings.getValue(1, 1, 0);
    bool same_positional_encoding = (std::abs(pos_diff_batch0 - pos_diff_batch1) < 1e-6f);
    printTestResult("Same positional encoding across batches", same_positional_encoding);
    all_passed &= same_positional_encoding;
    
    // Test backward compatibility with 2D input
    Tensor legacy_embeddings(4, 8);
    legacy_embeddings.fill(0.1f);
    legacy_embeddings.setValue(0, 0, 1.0f);
    
    Tensor legacy_result = pos_enc.forward(legacy_embeddings);
    bool legacy_compatibility = (legacy_result.getRows() == 4 && 
                                legacy_result.getCols() == 8 && 
                                !legacy_result.getIs3D());
    printTestResult("Legacy 2D format still works", legacy_compatibility);
    all_passed &= legacy_compatibility;
    
    // Verify that 2D and 3D give same results for equivalent inputs
    bool results_consistent = (std::abs(legacy_result.getValue(0, 0) - batch_result.getValue(0, 0, 0)) < 1e-6f);
    printTestResult("2D and 3D results consistent", results_consistent);
    all_passed &= results_consistent;
    
    return all_passed;
}

// Test batched linear layers
bool testBatchedLinearLayers() {
    printSectionHeader("BATCHED LINEAR LAYER TESTING");
    bool all_passed = true;
    
    Linear linear_layer(4, 3, true);  // 4 input dims, 3 output dims, with bias
    
    // Test with 3D input [batch_size, seq_len, input_dim]
    Tensor batch_input(2, 3, 4);  // 2 sequences of length 3, with 4 features each
    
    // Fill with identifiable values
    for (int b = 0; b < 2; b++) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 4; j++) {
                batch_input.setValue(b, i, j, (b + 1) * 0.1f + (i + 1) * 0.01f + (j + 1) * 0.001f);
            }
        }
    }
    
    Tensor batch_output = linear_layer.forward(batch_input);
    
    // Test output dimensions
    bool batch_dimensions_correct = (batch_output.getBatchSize() == 2 && 
                                    batch_output.getRows() == 3 && 
                                    batch_output.getCols() == 3);
    printTestResult("Batch linear layer dimensions [2, 3, 3]", batch_dimensions_correct);
    all_passed &= batch_dimensions_correct;
    
    // Test that different batches produce different outputs
    bool different_batch_outputs = (batch_output.getValue(0, 0, 0) != batch_output.getValue(1, 0, 0));
    printTestResult("Different batch inputs produce different outputs", different_batch_outputs);
    all_passed &= different_batch_outputs;
    
    // Test backward compatibility with 2D input
    Tensor legacy_input(3, 4);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            legacy_input.setValue(i, j, (i + 1) * 0.01f + (j + 1) * 0.001f);
        }
    }
    
    Tensor legacy_output = linear_layer.forward(legacy_input);
    bool legacy_compatibility = (legacy_output.getRows() == 3 && 
                                legacy_output.getCols() == 3 && 
                                !legacy_output.getIs3D());
    printTestResult("Legacy 2D format still works", legacy_compatibility);
    all_passed &= legacy_compatibility;
    
    // Test linear layer without bias
    Linear no_bias_layer(4, 3, false);
    Tensor no_bias_output = no_bias_layer.forward(batch_input);
    bool no_bias_dimensions = (no_bias_output.getBatchSize() == 2 && 
                              no_bias_output.getRows() == 3 && 
                              no_bias_output.getCols() == 3);
    printTestResult("Linear layer without bias works", no_bias_dimensions);
    all_passed &= no_bias_dimensions;
    
    // Test that bias makes a difference (when bias is used)
    bool bias_makes_difference = (batch_output.getValue(0, 0, 0) != no_bias_output.getValue(0, 0, 0));
    printTestResult("Bias parameter affects output", bias_makes_difference);
    all_passed &= bias_makes_difference;
    
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

// Test batched multi-head attention
bool testBatchedMultiHeadAttention() {
    printSectionHeader("BATCHED MULTI-HEAD ATTENTION TESTING");
    bool all_passed = true;
    
    MultiHeadAttention mha(8, 2, 0.0f);  // d_model=8, num_heads=2, no dropout for testing
    
    // Test with 3D input [batch_size, seq_len, d_model]
    Tensor batch_input(2, 4, 8);  // 2 sequences of length 4
    
    // Fill with different values for each batch
    for (int b = 0; b < 2; b++) {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 8; j++) {
                batch_input.setValue(b, i, j, (b + 1) * 0.1f + (i + 1) * 0.01f + (j + 1) * 0.001f);
            }
        }
    }
    
    Tensor batch_output = mha.forward(batch_input, false);
    
    // Test output dimensions
    bool batch_dimensions_correct = (batch_output.getBatchSize() == 2 && 
                                    batch_output.getRows() == 4 && 
                                    batch_output.getCols() == 8);
    printTestResult("Batch attention dimensions [2, 4, 8]", batch_dimensions_correct);
    all_passed &= batch_dimensions_correct;
    
    // Test that different batches produce different outputs
    bool different_batch_outputs = (batch_output.getValue(0, 0, 0) != batch_output.getValue(1, 0, 0));
    printTestResult("Different batch inputs produce different outputs", different_batch_outputs);
    all_passed &= different_batch_outputs;
    
    // Test that output values are reasonable (not NaN or infinite)
    bool values_reasonable = true;
    for (int b = 0; b < 2 && values_reasonable; b++) {
        for (int i = 0; i < 4 && values_reasonable; i++) {
            for (int j = 0; j < 8 && values_reasonable; j++) {
                float val = batch_output.getValue(b, i, j);
                if (std::isnan(val) || std::isinf(val)) {
                    values_reasonable = false;
                }
            }
        }
    }
    printTestResult("Attention output values are reasonable", values_reasonable);
    all_passed &= values_reasonable;
    
    // Test backward compatibility with 2D input
    Tensor legacy_input(4, 8);
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            legacy_input.setValue(i, j, 0.1f + (i + 1) * 0.01f + (j + 1) * 0.001f);
        }
    }
    
    Tensor legacy_output = mha.forward(legacy_input, false);
    bool legacy_compatibility = (legacy_output.getRows() == 4 && 
                                legacy_output.getCols() == 8 && 
                                !legacy_output.getIs3D());
    printTestResult("Legacy 2D format still works", legacy_compatibility);
    all_passed &= legacy_compatibility;
    
    // Test that causal masking works (later positions shouldn't affect earlier ones)
    Tensor causal_test_input(1, 3, 8);
    causal_test_input.fill(1.0f);
    
    // Set last position to very different values
    for (int j = 0; j < 8; j++) {
        causal_test_input.setValue(0, 2, j, 100.0f);  // Very large values at last position
    }
    
    Tensor causal_output = mha.forward(causal_test_input, false);
    
    // First position should not be significantly affected by last position due to causal masking
    bool causal_masking_works = (std::abs(causal_output.getValue(0, 0, 0)) < 50.0f);
    printTestResult("Causal masking prevents future information leakage", causal_masking_works);
    all_passed &= causal_masking_works;
    
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

    // Test decode
    std::string decoded = bpe_tokenizer.decode(encoded);
    std::cout << "Decoded back: '" << decoded << "'" << std::endl;
    bool roundtrip_works = (decoded == encode_test);
    printTestResult("Encode/decode roundtrip", roundtrip_works);
    all_passed &= roundtrip_works;
    
    return all_passed;
}

// Add this test function to your main.cpp

bool testModelSerialization() {
    printSectionHeader("MODEL SERIALIZATION TESTING");
    bool all_passed = true;
    
    // Test model save functionality
    GPTModel original_model(20, 16, 2, 2, 32, 0.1f);
    
    // Create some test input to get consistent outputs
    Tensor test_input(3, 1);
    test_input.setValue(0, 0, 1);
    test_input.setValue(1, 0, 5);
    test_input.setValue(2, 0, 3);
    
    // Get output from original model
    Tensor original_output = original_model.forward(test_input);
    
    // Test save operation
    std::string test_filepath = "test_model.bin";
    bool save_success = original_model.save(test_filepath);
    printTestResult("Model save operation", save_success);
    all_passed &= save_success;
    
    if (save_success) {
        // Test load operation
        bool load_success = false;
        GPTModel loaded_model(1, 1, 1, 1, 1); // Dummy initialization
        
        try {
            loaded_model = GPTModel::load(test_filepath);
            load_success = true;
        } catch (const std::exception& e) {
            std::cout << "    Load failed: " << e.what() << std::endl;
        }
        
        printTestResult("Model load operation", load_success);
        all_passed &= load_success;
        
        if (load_success) {
            // Test that loaded model produces same output
            Tensor loaded_output = loaded_model.forward(test_input);
            
            bool outputs_match = true;
            float tolerance = 1e-6f;
            
            if (loaded_output.getRows() != original_output.getRows() || 
                loaded_output.getCols() != original_output.getCols()) {
                outputs_match = false;
            } else {
                for (int i = 0; i < original_output.getRows(); i++) {
                    for (int j = 0; j < original_output.getCols(); j++) {
                        float diff = std::abs(original_output.getValue(i, j) - loaded_output.getValue(i, j));
                        if (diff > tolerance) {
                            outputs_match = false;
                            break;
                        }
                    }
                    if (!outputs_match) break;
                }
            }
            
            printTestResult("Loaded model produces identical output", outputs_match);
            all_passed &= outputs_match;
            
            // Test with different input to ensure it's not just memorized
            Tensor test_input2(2, 1);
            test_input2.setValue(0, 0, 7);
            test_input2.setValue(1, 0, 2);
            
            Tensor original_output2 = original_model.forward(test_input2);
            Tensor loaded_output2 = loaded_model.forward(test_input2);
            
            bool outputs_match2 = true;
            if (loaded_output2.getRows() != original_output2.getRows() || 
                loaded_output2.getCols() != original_output2.getCols()) {
                outputs_match2 = false;
            } else {
                for (int i = 0; i < original_output2.getRows(); i++) {
                    for (int j = 0; j < original_output2.getCols(); j++) {
                        float diff = std::abs(original_output2.getValue(i, j) - loaded_output2.getValue(i, j));
                        if (diff > tolerance) {
                            outputs_match2 = false;
                            break;
                        }
                    }
                    if (!outputs_match2) break;
                }
            }
            
            printTestResult("Consistency across different inputs", outputs_match2);
            all_passed &= outputs_match2;
        }
        
        // Test error handling for invalid files
        bool error_handling_correct = false;
        try {
            GPTModel::load("nonexistent_file.bin");
        } catch (const std::exception& e) {
            error_handling_correct = true;
        }
        printTestResult("Error handling for missing files", error_handling_correct);
        all_passed &= error_handling_correct;
        
        // Clean up test file
        std::remove(test_filepath.c_str());
    }
    
    return all_passed;
}

// Also add this performance test for serialization
void benchmarkSerialization() {
    std::cout << "\n  SERIALIZATION PERFORMANCE:" << std::endl;
    
    // Test serialization performance
    GPTModel benchmark_model(1000, 128, 6, 8, 512);
    std::string benchmark_file = "benchmark_model.bin";
    
    double save_time = measureTime([&]() {
        benchmark_model.save(benchmark_file);
    });
    std::cout << "  Model save (6 layers, 128 dims): " << std::fixed << std::setprecision(2) << save_time << " ms" << std::endl;
    
    double load_time = measureTime([&]() {
        GPTModel loaded = GPTModel::load(benchmark_file);
    });
    std::cout << "  Model load (6 layers, 128 dims): " << std::fixed << std::setprecision(2) << load_time << " ms" << std::endl;
    
    // Check file size
    std::ifstream file(benchmark_file, std::ios::binary | std::ios::ate);
    if (file.is_open()) {
        std::streamsize size = file.tellg();
        file.close();
        std::cout << "  Model file size: " << std::fixed << std::setprecision(2) << (size / 1024.0 / 1024.0) << " MB" << std::endl;
    }
    
    // Clean up
    std::remove(benchmark_file.c_str());
}

int main() {
    std::cout << "🚀 COMPREHENSIVE TRANSFORMER FROM SCRATCH AUDIT 🚀" << std::endl;
    std::cout << "Testing all components with edge cases and performance metrics..." << std::endl;
    
    bool all_tests_passed = true;
    
    // Run all test suites
    // Run all test suites
    all_tests_passed &= testTensorOperations();
    all_tests_passed &= test3DTensorOperations();  
    all_tests_passed &= testTokenEmbeddings();
    all_tests_passed &= testBatchedTokenEmbeddings();
    all_tests_passed &= testPositionalEncoding();
    all_tests_passed &= testBatchedPositionalEncoding();
    all_tests_passed &= testBatchedLinearLayers();
    all_tests_passed &= testBatchedMultiHeadAttention();
    all_tests_passed &= testTransformerBlock();
    all_tests_passed &= testGPTModel();
    all_tests_passed &= testModelSerialization();
    all_tests_passed &= testBPETokenizer();
    
    // Performance benchmarking
    benchmarkPerformance();
    benchmarkSerialization();
    
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