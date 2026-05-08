# Cache-Aware Parallel FFT for Big Integer Multiplication

A progressive optimization study of the Cooley-Tukey FFT algorithm applied to big integer multiplication via polynomial convolution. Starting from a serial baseline, each version layers in one additional optimization — OpenMP thread parallelism, cache-aware tiling, and AVX2 SIMD vectorization — so the contribution of each technique can be measured in isolation.

**Course:** High Performance Computing (University Research Project)  
**Author:** Areeba Javed  
**Language:** C++ with OpenMP and AVX2 intrinsics  

---

## Implementations

| File | Optimization | Gap Addressed |
|------|-------------|---------------|
| `serial_fft.cpp` | Baseline Cooley-Tukey FFT, no parallelism | — |
| `fft_omp.cpp` | OpenMP parallel butterfly stages with runtime scheduling | Thread load imbalance |
| `fft_omp_tiled.cpp` | OpenMP + cached bit-reversal + twiddle caching + k-tiling | Cache-unfriendly stage access |
| `fft_avx.cpp` | OpenMP + cached bit-reversal + twiddle caching + AVX2 SIMD butterfly | Scalar arithmetic throughput |

Each version produces the same output as the serial baseline, verified against a reference answer.

---

## Requirements

- GCC with C++11 support or later
- OpenMP (`libgomp`, included with GCC)
- AVX2-capable CPU (Intel Haswell 2013+ or AMD Ryzen 2017+)
- Linux (tested on Ubuntu 22.04)

Check AVX2 support on your machine:
```bash
grep -m1 avx2 /proc/cpuinfo
```

---

## Building

```bash
# Serial baseline
g++ -O2 serial_fft.cpp -o serial_fft_exec

# OpenMP parallel
g++ -O2 -fopenmp fft_omp.cpp -o fft_omp_exec

# OpenMP + cache-blocked tiled transpose
g++ -O2 -fopenmp fft_omp_tiled.cpp -o fft_omp_tiled_exec

# OpenMP + tiled + AVX2 SIMD  (-mfma enables fused multiply-add)
g++ -O3 -fopenmp -mavx2 -mfma fft_avx.cpp -o fft_avx_exec
```

---

## Input Format

All versions read from `fft.in`:

```
N
<first number as string of N digits>
<second number as string of N digits>
```

Example `fft.in`:
```
8
12345678
87654321
```

The program computes the product of the two N-digit integers using FFT-based convolution and writes the result to a `.out` file.

---

## Running

```bash
# Serial
./serial_fft_exec
cat sfft.out

# OpenMP (set thread count with OMP_NUM_THREADS)
OMP_NUM_THREADS=4 ./fft_omp_exec
cat fft_omp.out

# Tiled
OMP_NUM_THREADS=4 ./fft_omp_tiled_exec
cat fft_omp_tiled.out

# AVX
OMP_NUM_THREADS=4 ./fft_avx_exec
cat fft_avx.out
```

All four outputs should match.

---

## Benchmarking

Run all versions across the test suite and collect timings into a CSV:

```bash
bash run_benchmarks.sh
python3 scripts/analysis.py benchmarks/my_laptop/benchmark_results_fresh.csv
```

The benchmark runner uses the inputs in `tests/` and writes a CSV with columns `config,size,time_s,gflops,AI`.

---

## Results

Benchmarked on: Ubuntu 22.04 VM, full input suite in `tests/`

| Version | Threads | Time (ms) |
|---------|---------|-----------|
| serial  | 1       | ~0.25     |
| omp     | 1       | 0.145     |
| omp     | 2       | 0.209     |
| omp     | 4       | 0.972     |
| tiled   | 1       | 0.141     |
| tiled   | 4       | 0.270     |
| avx     | 1       | 0.173     |
| avx     | 4       | 0.662     |

> Note: At small N the OpenMP thread-launch overhead dominates, which is why higher thread counts are slower. The gains from parallelism become visible only at larger sizes, and even then the implementation remains memory-bound.

---

## Roofline Characterization

The project uses the Empirical Roofline Tool (ERT) under `roofline/ert/` to measure the machine's peak compute and sustained memory bandwidth.

For the current laptop/VM setup, ERT reported:

- Peak compute: 977.96 GFLOP/s
- Sustained memory bandwidth: 7.64 GB/s

The FFT kernel has arithmetic intensity of about 2.5 FLOP/byte, so the roofline bound is dominated by memory bandwidth:

$$
	ext{Max performance} \approx \min(977.96,\ 7.64 \times 2.5) \approx 19\ \text{GFLOP/s}
$$

That explains why the measured implementations stay far below peak compute: the code is limited by memory movement and synchronization overhead, not by raw floating-point throughput.

---

## Literature Context

This project targets three gaps left open by existing parallel FFT literature:

- **Gap 2 (cache-hostile transpose):** heFFTe (Ayala et al., 2020) and Pippig & Potts (SC '21) optimize inter-node distributed transposes but leave the intra-node inter-stage transpose unblocked. Addressed by `fft_omp_tiled.cpp`.

- **Gap 4 (static load distribution):** No reviewed paper isolates the contribution of `schedule(dynamic)` vs `schedule(static)` on the butterfly outer loop. Addressed by `fft_omp.cpp`.

- **Gap C (SIMD butterfly):** Franchetti et al. (Euro-Par 2003) vectorize FFT butterfly arithmetic with 128-bit SSE (2-wide). This project extends to 256-bit AVX2 (4-wide double precision). Addressed by `fft_avx.cpp`.

---

## References

1. Ayala et al. (2020). *heFFTe: Highly Efficient FFT for Exascale*. ICCS 2020.
2. Pippig & Potts (2021). *Accelerating Multi-Process Communication for Parallel 3D FFT*. SC '21.
3. Lu et al. (2023). *GFFT: A Task Graph Based FFT Optimization Framework*. ICPP 2023.
4. Franchetti et al. (2003). *Efficient Utilization of SIMD Extensions*. Euro-Par 2003.
5. Gong et al. (2022). *Interval Arithmetic-Based FFT for Large Integer Multiplication*. IEEE HPEC 2022.

## Folder Structure

```
cache-aware-parallel-fft/
├── Makefile
├── README.md
├── run_benchmarks.sh
├── benchmarks/
│   └── benchmark_results.csv
├── bin/
│   ├── serial_fft
│   ├── fft_omp
│   ├── fft_omp_tiled
│   └── fft_avx
├── docs/
│   └── supplementary notes and write-up material
├── include/
│   └── shared headers or future interfaces
├── output/
│   ├── sfft.out
│   ├── fft_omp.out
│   ├── fft_omp_tiled.out
│   └── fft_avx.out
├── roofline/
│   ├── Config/
│   │   └── my_laptop
│   ├── ert/
│   │   └── Empirical_Roofline_Tool-1.1.0/
│   └── results/
│       ├── roofline_chart.pdf
│       ├── roofline_chart.png
│       └── perf / ERT output files
├── scripts/
│   └── plot_roofline.py
├── src/
│   ├── core/
│   │   └── serial_fft.cpp
│   ├── parallel/
│   │   ├── fft_omp.cpp
│   │   └── fft_omp_tiled.cpp
│   └── simd/
│       └── fft_avx.cpp
└── tests/
  ├── fft.in
  ├── generate_tests.py
  ├── small_tests/
  ├── medium_tests/
  └── large_tests/
```

The repository is organized around four FFT implementations in `src/`, one benchmark runner at the top level, and a separate `roofline/` area for ERT configuration and roofline plots.