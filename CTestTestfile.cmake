# CMake generated Testfile for 
# Source directory: /Users/liamford/Documents/projects/transformer-from-scratch
# Build directory: /Users/liamford/Documents/projects/transformer-from-scratch
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(GradientChecking "/Users/liamford/Documents/projects/transformer-from-scratch/test_gradients")
set_tests_properties(GradientChecking PROPERTIES  _BACKTRACE_TRIPLES "/Users/liamford/Documents/projects/transformer-from-scratch/CMakeLists.txt;207;add_test;/Users/liamford/Documents/projects/transformer-from-scratch/CMakeLists.txt;0;")
add_test(DropoutVerification "/Users/liamford/Documents/projects/transformer-from-scratch/test_dropout")
set_tests_properties(DropoutVerification PROPERTIES  _BACKTRACE_TRIPLES "/Users/liamford/Documents/projects/transformer-from-scratch/CMakeLists.txt;215;add_test;/Users/liamford/Documents/projects/transformer-from-scratch/CMakeLists.txt;0;")
add_test(AttentionBiasVerification "/Users/liamford/Documents/projects/transformer-from-scratch/test_attention_bias")
set_tests_properties(AttentionBiasVerification PROPERTIES  _BACKTRACE_TRIPLES "/Users/liamford/Documents/projects/transformer-from-scratch/CMakeLists.txt;223;add_test;/Users/liamford/Documents/projects/transformer-from-scratch/CMakeLists.txt;0;")
add_test(WeightTyingVerification "/Users/liamford/Documents/projects/transformer-from-scratch/test_weight_tying")
set_tests_properties(WeightTyingVerification PROPERTIES  _BACKTRACE_TRIPLES "/Users/liamford/Documents/projects/transformer-from-scratch/CMakeLists.txt;231;add_test;/Users/liamford/Documents/projects/transformer-from-scratch/CMakeLists.txt;0;")
add_test(AttentionGradientChecking "/Users/liamford/Documents/projects/transformer-from-scratch/test_attention_gradients")
set_tests_properties(AttentionGradientChecking PROPERTIES  _BACKTRACE_TRIPLES "/Users/liamford/Documents/projects/transformer-from-scratch/CMakeLists.txt;239;add_test;/Users/liamford/Documents/projects/transformer-from-scratch/CMakeLists.txt;0;")
