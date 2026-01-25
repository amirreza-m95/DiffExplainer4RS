#!/bin/bash

# Unified script for ML1M dataset visualizations (both evaluation methods)

echo "=== ML1M Dataset Visualization Script ==="

# Create output directories
mkdir -p ../results/ML1M/onebyone
mkdir -p ../results/ML1M/onebyoneuntil20p

echo "=== Running ML1M visualizations for both evaluation methods ==="

# Onebyone evaluation method (100% files)
echo "--- Processing onebyone evaluation method (100% files) ---"
if [ -f "../results/ML1M/slurm-480301 ML1M Diffusion 100%.out" ] && [ -f "../results/ML1M/slurm-480128 ML1M Others 100% Opt LXR.out" ]; then
    echo "Found ML1M onebyone files, creating visualizations..."
    python ../src/visualization_diffusion.py \
        --lxr_slurm_file "../results/ML1M/slurm-480128 ML1M Others 100% Opt LXR.out" \
        --diffusion_slurm_file "../results/ML1M/slurm-480301 ML1M Diffusion 100%.out" \
        --output_dir "../results/ML1M/onebyone"
    echo "ML1M onebyone visualizations complete!"
else
    echo "Warning: ML1M onebyone files not found!"
fi

echo ""

# Onebyoneuntil20p evaluation method (20% files)
echo "--- Processing onebyoneuntil20p evaluation method (20% files) ---"
if [ -f "../results/ML1M/slurm-478440 ML1M Diffusion 20%.out" ] && [ -f "../results/ML1M/slurm-480129 ML1M Others 20% Opt LXR.out" ]; then
    echo "Found ML1M onebyoneuntil20p files, creating visualizations..."
    python ../src/visualization_diffusion.py \
        --lxr_slurm_file "../results/ML1M/slurm-480129 ML1M Others 20% Opt LXR.out" \
        --diffusion_slurm_file "../results/ML1M/slurm-478440 ML1M Diffusion 20%.out" \
        --output_dir "../results/ML1M/onebyoneuntil20p"
    echo "ML1M onebyoneuntil20p visualizations complete!"
else
    echo "Warning: ML1M onebyoneuntil20p files not found!"
fi

echo ""

# Binned evaluation method
echo "--- Processing binned evaluation method ---"
if [ -f "../results/ML1M/binned/slurm-484242 Ml1M binned Diffusion.out" ] && [ -f "../results/ML1M/binned/slurm-479901 ML1M binned Others.out" ]; then
    echo "Found ML1M binned files, creating visualizations..."
    python ../src/visualization_diffusion.py \
        --lxr_slurm_file "../results/ML1M/binned/slurm-479901 ML1M binned Others.out" \
        --diffusion_slurm_file "../results/ML1M/binned/slurm-484242 Ml1M binned Diffusion.out" \
        --output_dir "../results/ML1M/binned"
    echo "ML1M binned visualizations complete!"
else
    echo "Warning: ML1M binned files not found!"
fi

echo ""
echo "=== ML1M visualization complete! ==="
echo "Results saved in:"
echo "  - results/ML1M/onebyone/ (onebyone evaluation method)"
echo "  - results/ML1M/onebyoneuntil20p/ (onebyoneuntil20p evaluation method)"
echo "  - results/ML1M/binned/ (binned evaluation method)"
echo ""
echo "Each directory contains:"
echo "  - 7 metric trend figures (POS, NEG, OTHER for 0-10 and 10-50 ranges, plus special NDCG+POS@3+POS@5 combination)"
echo "  - Step comparisons with separated metric groups for steps 1,2,3,5,10"
echo "  - Separate radar charts for POS and other metrics"
echo "  - Performance heatmap and summary tables with updated naming"
echo "  - Special NDCG+POS@3+POS@5 figure available in both PNG and PDF formats"
