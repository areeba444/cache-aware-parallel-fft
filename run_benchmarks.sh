#!/bin/bash

# Exit on error
set -e

# --- Build executables using Makefile ---
echo "Building executables..."
make all
echo "Build complete."

# --- Prepare results file ---
BENCHMARK_DIR="benchmarks"
RESULTS_FILE="$BENCHMARK_DIR/benchmark_results.csv"
mkdir -p $BENCHMARK_DIR
echo "Implementation,Size,Time(s)" > $RESULTS_FILE

# --- Set OpenMP threads ---
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} # Use existing value or default to 4
echo "Using OMP_NUM_THREADS=$OMP_NUM_THREADS"

# --- Function to run and time an executable ---
run_benchmark() {
    local impl_name=$1
    local executable=$2
    local test_path=$3
    local size=$(basename "$test_path")

    echo "  Testing $impl_name with size $size..."
    
    # The Makefile places executables in bin/
    # The test files are in tests/
    local full_executable_path="./bin/$executable"
    local input_file="$test_path/fft.in"

    # Run the command and capture time.
    # We need to copy the input file to the root, as the C++ programs expect 'fft.in'
    cp "$input_file" "fft.in"

    # Using /usr/bin/time to get wall clock time. -f "%e" gives real time in seconds.
    local time_taken=$(/usr/bin/time -f "%e" "$full_executable_path" 2>&1)
    
    echo "$impl_name,$size,$time_taken" >> $RESULTS_FILE
}

# --- Run benchmarks for all test directories ---
# Find all subdirectories in the tests folder
TEST_DIRS=$(find tests -mindepth 2 -maxdepth 2 -type d | sort -V)

for test_dir in $TEST_DIRS; do
    echo "Running benchmarks for input size $(basename "$test_dir")..."
    run_benchmark "serial" "serial_fft" "$test_dir"
    run_benchmark "omp" "fft_omp" "$test_dir"
    run_benchmark "omp_tiled" "fft_omp_tiled" "$test_dir"
    run_benchmark "avx" "fft_avx" "$test_dir"
done

# Clean up the last used input file
rm -f fft.in

echo "Benchmarking complete. Results are in $RESULTS_FILE"
