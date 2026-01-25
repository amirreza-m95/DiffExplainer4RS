#!/bin/bash

# Unified script for Pinterest dataset visualizations (both evaluation methods)

echo "=== Pinterest Dataset Visualization Script ==="

# Create output directories
mkdir -p ../results/Pinterest/onebyone
mkdir -p ../results/Pinterest/onebyoneuntil20p

echo "=== Running Pinterest visualizations for both evaluation methods ==="

# Onebyone evaluation method (100% files)
echo "--- Processing onebyone evaluation method (100% files) ---"
if [ -f "../results/Pinterest/slurm-484122 Pinterest Diffusion 100%.out" ] && [ -f "../results/Pinterest/slurm-480203 Pinterest 100% Opt Others.out" ]; then
    echo "Found Pinterest onebyone files, creating visualizations..."
    python ../src/visualization_diffusion.py \
        --lxr_slurm_file "../results/Pinterest/slurm-480203 Pinterest 100% Opt Others.out" \
        --diffusion_slurm_file "../results/Pinterest/slurm-484122 Pinterest Diffusion 100%.out" \
        --output_dir "../results/Pinterest/onebyone"
    echo "Pinterest onebyone visualizations complete!"
else
    echo "Warning: Pinterest onebyone files not found!"
fi

echo ""

# Onebyoneuntil20p evaluation method (20% files)
echo "--- Processing onebyoneuntil20p evaluation method (20% files) ---"
if [ -f "../results/Pinterest/slurm-484126 Pinterest Diffusion 20%.out" ] && [ -f "../results/Pinterest/slurm-480204 Pinterest 20% Opt Others.out" ]; then
    echo "Found Pinterest onebyoneuntil20p files, creating visualizations..."
    python ../src/visualization_diffusion.py \
        --lxr_slurm_file "../results/Pinterest/slurm-480204 Pinterest 20% Opt Others.out" \
        --diffusion_slurm_file "../results/Pinterest/slurm-484126 Pinterest Diffusion 20%.out" \
        --output_dir "../results/Pinterest/onebyoneuntil20p"
    echo "Pinterest onebyoneuntil20p visualizations complete!"
else
    echo "Warning: Pinterest onebyoneuntil20p files not found!"
fi

echo ""
echo "=== Pinterest visualization complete! ==="
echo "Results saved in:"
echo "  - results/Pinterest/onebyone/ (onebyone evaluation method)"
echo "  - results/Pinterest/onebyoneuntil20p/ (onebyoneuntil20p evaluation method)"
echo ""
echo "Each directory contains:"
echo "  - 7 metric trend figures (POS, NEG, OTHER for 0-10 and 10-50 ranges, plus special NDCG+POS@3+POS@5 combination)"
echo "  - Step comparisons with separated metric groups for steps 1,2,3,5,10"
echo "  - Separate radar charts for POS and other metrics"
echo "  - Performance heatmap and summary tables with updated naming"
echo "  - Special NDCG+POS@3+POS@5 figure available in both PNG and PDF formats"