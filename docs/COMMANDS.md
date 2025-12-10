# Quick Command Reference

## Build Commands

### Initial Setup
```bash
cmake -S . -B build
```
Build the project for the first time.

### Build Everything
```bash
cmake --build . -j
```
Compile all code. Use this after making changes.

### Clean Build
```bash
make clean && make
```
Delete everything and rebuild from scratch.

### Build with Tests
```bash
cmake . -DBUILD_TESTS=ON
cmake --build . -j
```
Enable and build all test executables.

## Running the Program

### Main Program
```bash
./transformer
```
Runs the full training benchmark on Shakespeare data.

### Run Tests
```bash
./test_gradients
./test_attention_gradients
./test_dropout
./test_attention_bias
./test_weight_tying
```
Run individual test suites.

## Git Commands

### Check Status
```bash
git status
```
See what files changed.

### View Changes
```bash
git diff
```
See what you modified.

### Commit Changes
```bash
git add <file>
git commit -m "Your message here"
```
Save your changes with a description.

### View History
```bash
git log --oneline -10
```
See last 10 commits.

### Create Branch
```bash
git checkout -b new-branch-name
```
Make a new branch for experimental work.

## Debugging Commands

### Check for Warnings
```bash
cmake --build . 2>&1 | grep "warning"
```
Find all compiler warnings.

### Check Specific File
```bash
git diff src/transformer/tensor.cpp
```
See changes in one file.

### Count Lines of Code
```bash
find src include -name "*.cpp" -o -name "*.h" | xargs wc -l
```
How much code you've written.

## Performance Commands

### Run Benchmark
```bash
./transformer
```
Measures training speed (takes ~2 minutes).

### Time a Single Run
```bash
time ./test_gradients
```
See how long tests take.

## File Operations

### Find a File
```bash
find . -name "tensor.cpp"
```
Locate files by name.

### Search in Files
```bash
grep -r "matmul" src/
```
Find text in all source files.

### List Recent Files
```bash
ls -lt src/transformer/ | head
```
See what you edited recently.

## Quick Fixes

### Build Failed?
```bash
make clean
cmake --build . -j
```
Clean and rebuild.

### Tests Not Found?
```bash
cmake . -DBUILD_TESTS=ON
cmake --build . -j
```
Make sure tests are enabled.

### Can't Find Library?
The Accelerate framework should link automatically on macOS.
On Linux, install: `sudo apt-get install libblas-dev`

## Common Workflows

### Make a Change and Test
```bash
# 1. Edit your file
# 2. Build
cmake --build . -j
# 3. Run test
./test_gradients
```

### Commit Your Work
```bash
git status                    # See what changed
git add src/transformer/*.cpp # Stage files
git commit -m "Fix bug"       # Commit with message
```

### Start New Feature
```bash
git checkout -b my-feature    # New branch
# ... make changes ...
cmake --build . -j            # Build
./test_gradients             # Test
git commit -am "Add feature"  # Commit
```

## Useful Shortcuts

### Rebuild Just Main Program
```bash
cmake --build . --target transformer
```
Only compile the main executable.

### Run All Tests at Once
```bash
./test_gradients && ./test_attention_gradients && ./test_dropout
```
Run multiple tests sequentially.

### Quick Status Check
```bash
git status --short
```
Compact view of changes.

## Files You'll Edit Most

```
src/transformer/tensor.cpp          - Core tensor operations
src/transformer/variable.cpp        - Autograd/backprop
src/transformer/multihead_attention.cpp - Attention mechanism
src/transformer/layer_norm.cpp      - Layer normalization
src/main.cpp                        - Training loops
tests/unit/test_*.cpp              - Unit tests
```

## Getting Help

### Compiler Errors
Read from bottom up - the first error is usually the real one.

### Linker Errors
Usually means you forgot to link a library or build a file.

### Segfaults
Check array bounds, null pointers, and tensor dimensions.

### Slow Performance
Make sure you built with: `cmake -DCMAKE_BUILD_TYPE=Release`

## That's It!

Most of the time you'll just use:
1. `cmake --build . -j` - Build
2. `./transformer` - Run
3. `./test_gradients` - Test
4. `git commit -am "message"` - Save

Everything else is extra.
