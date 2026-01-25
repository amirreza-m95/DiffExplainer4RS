#!/bin/bash

# Unified script for Yahoo dataset visualizations (both evaluation methods)

echo "=== Yahoo Dataset Visualization Script ==="

# Create output directories
mkdir -p ../results/Yahoo/onebyone
mkdir -p ../results/Yahoo/onebyoneuntil20p

echo "=== Running Yahoo visualizations for both evaluation methods ==="

# Onebyone evaluation method (100% files)
echo "--- Processing onebyone evaluation method (100% files) ---"
if [ -f "../results/Yahoo/slurm-480303 Yahoo Diffusion 100%.out" ] && [ -f "../results/Yahoo/slurm-480205 Yahoo 100% Opt Others.out" ]; then
    echo "Found Yahoo onebyone files, creating visualizations..."
    python ../src/visualization_diffusion.py \
        --lxr_slurm_file "../results/Yahoo/slurm-480205 Yahoo 100% Opt Others.out" \
        --diffusion_slurm_file "../results/Yahoo/slurm-480303 Yahoo Diffusion 100%.out" \
        --output_dir "../results/Yahoo/onebyone"
    echo "Yahoo onebyone visualizations complete!"
else
    echo "Warning: Yahoo onebyone files not found!"
fi

echo ""

# Onebyoneuntil20p evaluation method (20% files)
echo "--- Processing onebyoneuntil20p evaluation method (20% files) ---"
if [ -f "../results/Yahoo/slurm-479210 Yahoo Diffusion 20%.out" ] && [ -f "../results/Yahoo/slurm-480206 Yahoo 20% Opt Others.out" ]; then
    echo "Found Yahoo onebyoneuntil20p files, creating visualizations..."
    python ../src/visualization_diffusion.py \
        --lxr_slurm_file "../results/Yahoo/slurm-480206 Yahoo 20% Opt Others.out" \
        --diffusion_slurm_file "../results/Yahoo/slurm-479210 Yahoo Diffusion 20%.out" \
        --output_dir "../results/Yahoo/onebyoneuntil20p"
    echo "Yahoo onebyoneuntil20p visualizations complete!"
else
    echo "Warning: Yahoo onebyoneuntil20p files not found!"
fi

echo ""
echo "=== Yahoo visualization complete! ==="
echo "Results saved in:"
echo "  - results/Yahoo/onebyone/ (onebyone evaluation method)"
echo "  - results/Yahoo/onebyoneuntil20p/ (onebyoneuntil20p evaluation method)"
echo ""
echo "Each directory contains:"
echo "  - 7 metric trend figures (POS, NEG, OTHER for 0-10 and 10-50 ranges, plus special NDCG+POS@3+POS@5 combination)"
echo "  - Step comparisons with separated metric groups for steps 1,2,3,5,10"
echo "  - Separate radar charts for POS and other metrics"
echo "  - Performance heatmap and summary tables with updated naming"
echo "  - Special NDCG+POS@3+POS@5 figure available in both PNG and PDF formats"