# CMake generated Testfile for 
# Source directory: /Users/liamford/Documents/projects/transformer-from-scratch
# Build directory: /Users/liamford/Documents/projects/transformer-from-scratch/build-test
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(GradientChecking "/Users/liamford/Documents/projects/transformer-from-scratch/build-test/test_gradients")
set_tests_properties(GradientChecking PROPERTIES  _BACKTRACE_TRIPLES "/Users/liamford/Documents/projects/transformer-from-scratch/CMakeLists.txt;179;add_test;/Users/liamford/Documents/projects/transformer-from-scratch/CMakeLists.txt;0;")
add_test(DropoutVerification "/Users/liamford/Documents/projects/transformer-from-scratch/build-test/test_dropout")
set_tests_properties(DropoutVerification PROPERTIES  _BACKTRACE_TRIPLES "/Users/liamford/Documents/projects/transformer-from-scratch/CMakeLists.txt;182;add_test;/Users/liamford/Documents/projects/transformer-from-scratch/CMakeLists.txt;0;")
add_test(AttentionBiasVerification "/Users/liamford/Documents/projects/transformer-from-scratch/build-test/test_attention_bias")
set_tests_properties(AttentionBiasVerification PROPERTIES  _BACKTRACE_TRIPLES "/Users/liamford/Documents/projects/transformer-from-scratch/CMakeLists.txt;185;add_test;/Users/liamford/Documents/projects/transformer-from-scratch/CMakeLists.txt;0;")
add_test(WeightTyingVerification "/Users/liamford/Documents/projects/transformer-from-scratch/build-test/test_weight_tying")
set_tests_properties(WeightTyingVerification PROPERTIES  _BACKTRACE_TRIPLES "/Users/liamford/Documents/projects/transformer-from-scratch/CMakeLists.txt;188;add_test;/Users/liamford/Documents/projects/transformer-from-scratch/CMakeLists.txt;0;")
