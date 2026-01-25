#!/bin/bash

# Master script to run visualizations for all datasets (ML1M, Pinterest, Yahoo)
# Each dataset will be processed for both evaluation methods (onebyone and onebyoneuntil20p)

echo "=== Master Visualization Script for All Datasets ==="

echo "This script will run visualizations for all three datasets:"
echo "  - ML1M (both onebyone and onebyoneuntil20p)"
echo "  - Pinterest (both onebyone and onebyoneuntil20p)"
echo "  - Yahoo (both onebyone and onebyoneuntil20p)"
echo ""

# Run ML1M visualizations
echo "=== Running ML1M visualizations ==="
if [ -f "run_ml1m_visualizations.sh" ]; then
    chmod +x run_ml1m_visualizations.sh
    ./run_ml1m_visualizations.sh
    echo ""
else
    echo "Error: run_ml1m_visualizations.sh not found!"
fi

# Run Pinterest visualizations
echo "=== Running Pinterest visualizations ==="
if [ -f "run_pinterest_visualizations.sh" ]; then
    chmod +x run_pinterest_visualizations.sh
    ./run_pinterest_visualizations.sh
    echo ""
else
    echo "Error: run_pinterest_visualizations.sh not found!"
fi

# Run Yahoo visualizations
echo "=== Running Yahoo visualizations ==="
if [ -f "run_yahoo_visualizations.sh" ]; then
    chmod +x run_yahoo_visualizations.sh
    ./run_yahoo_visualizations.sh
    echo ""
else
    echo "Error: run_yahoo_visualizations.sh not found!"
fi

echo "=== All visualizations complete! ==="
echo ""
echo "Results saved in the following structure:"
echo "  results/"
echo "  ├── ML1M/"
echo "  │   ├── onebyone/"
echo "  │   └── onebyoneuntil20p/"
echo "  ├── Pinterest/"
echo "  │   ├── onebyone/"
echo "  │   └── onebyoneuntil20p/"
echo "  └── Yahoo/"
echo "      ├── onebyone/"
echo "      └── onebyoneuntil20p/"
echo ""
echo "Each directory contains:"
echo "  - 7 metric trend figures (POS, NEG, OTHER for 0-10 and 10-50 ranges, plus special NDCG-P+POS@3+POS@5 combination)"
echo "  - Step comparisons with separated metric groups for steps 1,2,3,5,10"
echo "  - Separate radar charts for POS and other metrics"
echo "  - Performance heatmap and summary tables with updated naming"
echo "  - Special NDCG-P+POS@3+POS@5 figure available in both PNG and PDF formats"
echo ""
echo "Total: 6 visualization sets (3 datasets × 2 evaluation methods)"