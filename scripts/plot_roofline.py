#!/usr/bin/env python3
"""
Roofline plotter for parallel FFT benchmarks.
Reads kernel CSV and ERT roof parameters and draws the final roofline chart.
"""

import csv
import math
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ────────────────────────────────────────────────────────────────────────────
# ERT ROOF PARAMETERS — customize these from your ERT run results
# ────────────────────────────────────────────────────────────────────────────

# Example values (replace with your ERT measurements):
PEAK_GFLOPS   = 42.5      # Peak floating-point performance (GFLOP/s)
DRAM_BW_GBs   = 44.0      # DRAM bandwidth (GB/s)
L2_BW_GBs     = 583.0      # L2 cache bandwidth (GB/s)
L1_BW_GBs     = 687.0     # L1 cache bandwidth (GB/s)

print("Roofline parameters:")
print(f"  Peak GFLOP/s: {PEAK_GFLOPS}")
print(f"  DRAM BW:      {DRAM_BW_GBs} GB/s")
print(f"  L2 BW:        {L2_BW_GBs} GB/s")
print(f"  L1 BW:        {L1_BW_GBs} GB/s")
print()


# ────────────────────────────────────────────────────────────────────────────
# Generate roofline curves
# ────────────────────────────────────────────────────────────────────────────

ai_range = np.logspace(-2, 2, 500)  # Arithmetic intensity range (log scale)

def roofline(bw):
    """Return performance limited by bandwidth (min of bw line and peak)."""
    return np.minimum(bw * ai_range, PEAK_GFLOPS)


# ────────────────────────────────────────────────────────────────────────────
# Create figure
# ────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 7))
ax.set_xscale('log')
ax.set_yscale('log')

# Plot bandwidth roofs
ax.plot(ai_range, roofline(DRAM_BW_GBs), 'k-',  lw=1.5, label='DRAM BW roof',  zorder=2)
ax.plot(ai_range, roofline(L2_BW_GBs),   'k--', lw=1.0, label='L2 BW roof',    zorder=2)
ax.plot(ai_range, roofline(L1_BW_GBs),   'k:',  lw=1.0, label='L1 BW roof',    zorder=2)

# Plot peak performance ceiling
ax.axhline(PEAK_GFLOPS, color='k', lw=0.8, linestyle='-.', zorder=2)
ax.text(1e2, PEAK_GFLOPS * 1.08, f'Peak {PEAK_GFLOPS:.0f} GFLOP/s', fontsize=9, zorder=2)


# ────────────────────────────────────────────────────────────────────────────
# Read and plot kernel data
# ────────────────────────────────────────────────────────────────────────────

colors = {
    'serial':       '#808080',  # gray
    'omp':          '#378add',  # blue
    'omp_tiled':    '#1d9e75',  # green
    'avx':          '#d85a30',  # orange
}

markers = {
    'serial':       'o',        # circle
    'omp':          's',        # square
    'omp_tiled':    '^',        # triangle
    'avx':          'D',        # diamond
}

legend_added = set()

try:
    with open('benchmarks/benchmark_results.csv') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            print("ERROR: CSV file is empty or malformed.", file=sys.stderr)
            sys.exit(1)

        for row in reader:
            try:
                cfg = row['config']
                ai = float(row['AI'])
                gflops = float(row['gflops'])

                # Avoid plotting NaN or inf
                if not (math.isfinite(ai) and math.isfinite(gflops)):
                    continue

                color = colors.get(cfg, '#cccccc')
                marker = markers.get(cfg, 'x')

                # Plot point
                label_text = cfg if cfg not in legend_added else None
                ax.scatter(ai, gflops,
                          color=color, marker=marker,
                          s=100, alpha=0.8, edgecolors='black', linewidth=0.5,
                          label=label_text, zorder=5)
                
                if cfg not in legend_added:
                    legend_added.add(cfg)

            except (ValueError, KeyError) as e:
                print(f"WARNING: Skipping row due to error: {e}", file=sys.stderr)
                continue

except FileNotFoundError:
    print("ERROR: benchmarks/benchmark_results.csv not found.", file=sys.stderr)
    print("       Have you run run_benchmarks.sh first?", file=sys.stderr)
    sys.exit(1)

if not legend_added:
    print("WARNING: No data points were plotted. Check your benchmark results.", file=sys.stderr)


# ────────────────────────────────────────────────────────────────────────────
# Formatting
# ────────────────────────────────────────────────────────────────────────────

ax.set_xlabel('Arithmetic Intensity (FLOP/byte)', fontsize=11)
ax.set_ylabel('Performance (GFLOP/s)', fontsize=11)
ax.set_title('Roofline — Cache-Aware Parallel FFT', fontsize=12, fontweight='bold')

ax.legend(fontsize=9, loc='upper left', framealpha=0.95)
ax.grid(True, which='both', alpha=0.3, linestyle='-', linewidth=0.5)

# Set axis limits for clarity
ax.set_xlim(1e-2, 1e2)
ax.set_ylim(1e-1, 1e3)


# ────────────────────────────────────────────────────────────────────────────
# Save figures
# ────────────────────────────────────────────────────────────────────────────

plt.tight_layout()

pdf_path = 'roofline/results/roofline_chart.pdf'
png_path = 'roofline/results/roofline_chart.png'

try:
    plt.savefig(pdf_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved roofline chart to {pdf_path}")
except Exception as e:
    print(f"ERROR: Could not save PDF: {e}", file=sys.stderr)

try:
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved roofline chart to {png_path}")
except Exception as e:
    print(f"ERROR: Could not save PNG: {e}", file=sys.stderr)

print("\nRoofline plot complete. Data shown:")
print(f"  - X-axis: Arithmetic intensity (FLOP/byte)")
print(f"  - Y-axis: Performance (GFLOP/s)")
print(f"  - Black lines: Memory bandwidth ceilings")
print(f"  - Colored points: Kernel performance (4 implementations)")
