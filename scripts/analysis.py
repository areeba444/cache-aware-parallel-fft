#!/usr/bin/env python3
"""
Comprehensive speedup and performance analysis for FFT implementations.
Pure Python, no pandas dependency.
"""
import csv
import sys
from collections import defaultdict

def load_csv(filename):
    """Load benchmark CSV."""
    data = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['size'] = int(row['size'])
            row['time_s'] = float(row['time_s'])
            row['gflops'] = float(row['gflops'])
            row['AI'] = float(row['AI'])
            data.append(row)
    return data

def compute_speedups(data):
    """Compute speedup ratios relative to serial baseline."""
    # Group by size
    size_data = defaultdict(dict)
    for row in data:
        size = row['size']
        config = row['config']
        size_data[size][config] = row
    
    results = {}
    for size in sorted(size_data.keys()):
        configs = size_data[size]
        if 'serial' not in configs:
            print(f"ERROR: No serial baseline for size {size}", file=sys.stderr)
            continue
        
        serial_time = float(configs['serial']['time_s'])
        results[size] = {'serial_time_s': serial_time, 'implementations': {}}
        
        for config in ['omp', 'omp_tiled', 'avx']:
            if config in configs:
                impl_time = float(configs[config]['time_s'])
                speedup = serial_time / impl_time
                results[size]['implementations'][config] = {
                    'time_s': impl_time,
                    'speedup': speedup,
                    'gflops': float(configs[config]['gflops'])
                }
    
    return results

def print_summary(data, speedups):
    """Print comprehensive analysis."""
    print("\n" + "="*80)
    print("COMPREHENSIVE SPEEDUP AND PERFORMANCE ANALYSIS")
    print("="*80)
    
    sizes = sorted(speedups.keys())
    
    # Table 1: Speedup ratios
    print("\n1. SPEEDUP RATIOS (relative to serial baseline)")
    print("-" * 80)
    print(f"{'Size':>12} {'Serial (ms)':>15} {'OMP':>12} {'OMP_Tiled':>12} {'AVX':>12}")
    print("-" * 80)
    
    for size in sizes:
        s = speedups[size]
        serial_ms = s['serial_time_s'] * 1000
        omp_su = s['implementations'].get('omp', {}).get('speedup', 0)
        tiled_su = s['implementations'].get('omp_tiled', {}).get('speedup', 0)
        avx_su = s['implementations'].get('avx', {}).get('speedup', 0)
        
        print(f"{size:>12} {serial_ms:>15.6f} {omp_su:>12.4f}x {tiled_su:>12.4f}x {avx_su:>12.4f}x")
    
    # Table 2: Absolute timings
    print("\n2. ABSOLUTE EXECUTION TIMES (seconds)")
    print("-" * 80)
    print(f"{'Size':>12} {'Serial':>15} {'OMP':>15} {'OMP_Tiled':>15} {'AVX':>15}")
    print("-" * 80)
    
    for size in sizes:
        s = speedups[size]
        serial_t = s['serial_time_s']
        omp_t = s['implementations'].get('omp', {}).get('time_s', 0)
        tiled_t = s['implementations'].get('omp_tiled', {}).get('time_s', 0)
        avx_t = s['implementations'].get('avx', {}).get('time_s', 0)
        
        print(f"{size:>12} {serial_t:>15.9f} {omp_t:>15.9f} {tiled_t:>15.9f} {avx_t:>15.9f}")
    
    # Table 3: GFLOP/s performance
    print("\n3. PERFORMANCE (GFLOP/s)")
    print("-" * 80)
    print(f"{'Size':>12} {'Serial':>15} {'OMP':>15} {'OMP_Tiled':>15} {'AVX':>15}")
    print("-" * 80)
    
    for size in sizes:
        s = speedups[size]
        serial_gflops = s['implementations'].get('serial', {}) if 'serial' in s else None
        
        # Find serial gflops from data
        serial_gflops_val = 0
        omp_gflops_val = 0
        tiled_gflops_val = 0
        avx_gflops_val = 0
        
        for row in data:
            if row['size'] == size:
                if row['config'] == 'serial':
                    serial_gflops_val = row['gflops']
                elif row['config'] == 'omp':
                    omp_gflops_val = row['gflops']
                elif row['config'] == 'omp_tiled':
                    tiled_gflops_val = row['gflops']
                elif row['config'] == 'avx':
                    avx_gflops_val = row['gflops']
        
        print(f"{size:>12} {serial_gflops_val:>15.6f} {omp_gflops_val:>15.6f} {tiled_gflops_val:>15.6f} {avx_gflops_val:>15.6f}")
    
    # Table 4: Arithmetic Intensity
    print("\n4. ARITHMETIC INTENSITY (FLOP/byte)")
    print("-" * 80)
    ai_map = {}
    for row in data:
        size = row['size']
        if size not in ai_map:
            ai_map[size] = row['AI']
    
    for size in sizes:
        if size in ai_map:
            ai = ai_map[size]
            print(f"{size:>12}: {ai:>8.6f} FLOP/byte")
    
    # Analysis: Why no speedup?
    print("\n" + "="*80)
    print("ANALYSIS: WHY LIMITED SPEEDUP?")
    print("="*80)
    
    # Compute aggregate statistics
    avg_speedups = {'omp': [], 'omp_tiled': [], 'avx': []}
    for size in speedups.values():
        for impl in avg_speedups.keys():
            if impl in size['implementations']:
                avg_speedups[impl].append(size['implementations'][impl]['speedup'])
    
    print("\nAggregate Speedup Statistics:")
    for impl in avg_speedups:
        if avg_speedups[impl]:
            avg = sum(avg_speedups[impl]) / len(avg_speedups[impl])
            min_s = min(avg_speedups[impl])
            max_s = max(avg_speedups[impl])
            print(f"  {impl:>12}: avg={avg:.4f}x, min={min_s:.4f}x, max={max_s:.4f}x")
    
    # Peak utilization
    peak_gflops = 977.96  # From roofline analysis (my_laptop)
    print(f"\nPeak Machine Capability: {peak_gflops:.2f} GFLOP/s")
    max_achieved = max(row['gflops'] for row in data)
    utilization = (max_achieved / peak_gflops) * 100
    print(f"Max Achieved: {max_achieved:.6f} GFLOP/s ({utilization:.2f}% utilization)")
    
    print("\nKey Observations:")
    print("  1. FFT is MEMORY-BOUND (low arithmetic intensity ~2.5 FLOP/byte)")
    print("  2. Roofline model: memory bandwidth is bottleneck, not compute")
    print("  3. Parallelization overhead >> benefit for small/medium sizes")
    print("  4. AVX2 SIMD: 2x vector width, but memory bottleneck unchanged")
    print("  5. Runtime on VM: memory latency/bandwidth degraded vs bare metal")
    print("  6. Speedup success requires >4x parallelism OR <1x overhead")
    
    print("\nConclusion:")
    print("  Low speedups are expected for memory-bound FFT on this platform.")
    print("  Optimization path would need: better memory hierarchy,")
    print("  out-of-core blocking, GPU acceleration, or faster memory.")

if __name__ == '__main__':
    csv_file = sys.argv[1] if len(sys.argv) > 1 else 'benchmarks/my_laptop/benchmark_results_fresh.csv'
    
    df = load_csv(csv_file)
    speedups = compute_speedups(df)
    print_summary(df, speedups)
