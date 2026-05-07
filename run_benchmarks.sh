#!/bin/bash

# Exit on error
set -euo pipefail

# --- Build executables using Makefile ---
echo "Building executables..."
make all
echo "Build complete."

# --- Prepare results file ---
BENCHMARK_DIR="benchmarks"
RESULTS_FILE="$BENCHMARK_DIR/benchmark_results.csv"
mkdir -p "$BENCHMARK_DIR"
echo "config,size,time_s,gflops,AI" > "$RESULTS_FILE"

ORIGINAL_INPUT="tests/fft.in"
BACKUP_INPUT="$(mktemp)"
cp "$ORIGINAL_INPUT" "$BACKUP_INPUT"

cleanup() {
    cp "$BACKUP_INPUT" "$ORIGINAL_INPUT"
    rm -f "$BACKUP_INPUT"
}

trap cleanup EXIT

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

    local full_executable_path="./bin/$executable"
    local input_file="$test_path/fft.in"

    cp "$input_file" "$ORIGINAL_INPUT"

    "$full_executable_path" | tee -a "$RESULTS_FILE"
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

echo "Benchmarking complete. Results are in $RESULTS_FILE"
