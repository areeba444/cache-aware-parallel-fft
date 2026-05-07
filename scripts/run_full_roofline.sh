#!/bin/bash
# run_full_roofline.sh
# One-command roofline setup: ERT + benchmarks + plot
# Usage: bash scripts/run_full_roofline.sh

set -e  # stop on any error

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "============================================"
echo " Cache-Aware Parallel FFT — Roofline Setup"
echo " Project root: $PROJECT_ROOT"
echo "============================================"
echo ""

# ── Step 1: Install Python dependencies ──────────────────────────────────
echo "[1/5] Installing Python dependencies..."
pip3 install numpy matplotlib --break-system-packages -q
echo "      Done."
echo ""

# ── Step 2: Build all binaries ────────────────────────────────────────────
echo "[2/5] Building FFT binaries..."
make all
echo "      Done."
echo ""

# ── Step 3: Run ERT to characterize this machine ─────────────────────────
echo "[3/5] Running ERT hardware characterization..."
echo "      (this takes 5-10 minutes)"
cd "$PROJECT_ROOT/roofline"
rm -rf Results
python3 ert/Empirical_Roofline_Tool-1.1.0/ert Config/my_laptop
cd "$PROJECT_ROOT"
echo "      Done."
echo ""

# ── Step 4: Extract ERT ceilings and patch plot script ───────────────────
echo "[4/5] Extracting ERT results and updating plot_roofline.py..."

SUM_FILE="$PROJECT_ROOT/roofline/Results/Run.001/FLOPS.001/OpenMP.0001/sum"

if [ ! -f "$SUM_FILE" ]; then
    echo "ERROR: ERT results not found at $SUM_FILE"
    echo "       ERT may have failed. Check roofline/Results/ manually."
    exit 1
fi

# Extract peak GFLOP/s from sum file header line
PEAK=$(grep "GFLOPs" "$SUM_FILE" | awk '{print $1}')

# Extract bandwidth values from max file (col 8 = GB/s, pick by working set size)
MAX_FILE="$PROJECT_ROOT/roofline/Results/Run.001/FLOPS.001/OpenMP.0001/max"

# L1: smallest working set (first data line)
L1=$(grep -v "^[^0-9]" "$MAX_FILE" | grep -v "META" | awk 'NR==1{print $8}')

# L2: working set around 8192 bytes (line where col1 ~= 8192)
L2=$(grep -v "^[^0-9]" "$MAX_FILE" | grep -v "META" | awk '$1==8192{print $8}')

# DRAM: largest working set (last data line before META_DATA)
DRAM=$(grep -v "^[^0-9]" "$MAX_FILE" | grep -v "META" | tail -1 | awk '{print $8}')

echo "      Extracted ceilings:"
echo "        Peak GFLOP/s : $PEAK"
echo "        L1  BW (GB/s): $L1"
echo "        L2  BW (GB/s): $L2"
echo "        DRAM BW(GB/s): $DRAM"
echo ""

# Patch the plot script with real values
sed -i "s/PEAK_GFLOPS   = [0-9.]*/PEAK_GFLOPS   = $PEAK/" scripts/plot_roofline.py
sed -i "s/DRAM_BW_GBs   = [0-9.]*/DRAM_BW_GBs   = $DRAM/" scripts/plot_roofline.py
sed -i "s/L2_BW_GBs     = [0-9.]*/L2_BW_GBs     = $L2/" scripts/plot_roofline.py
sed -i "s/L1_BW_GBs     = [0-9.]*/L1_BW_GBs     = $L1/" scripts/plot_roofline.py
echo "      plot_roofline.py updated."
echo ""

# ── Step 5: Run benchmarks and generate chart ─────────────────────────────
echo "[5/5] Running FFT benchmarks and generating roofline chart..."
bash run_benchmarks.sh
mkdir -p roofline/results
python3 scripts/plot_roofline.py
echo ""

echo "============================================"
echo " All done!"
echo " Chart saved to:"
echo "   roofline/results/roofline_chart.png"
echo "   roofline/results/roofline_chart.pdf"
echo "============================================"
