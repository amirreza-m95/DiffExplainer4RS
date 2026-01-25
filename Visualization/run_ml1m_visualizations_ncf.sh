#!/bin/bash

set -e

echo "=== ML1M NCF Visualization (model-agnostic) ==="

PKL="results/ml1m/ncf_eval_results.pkl"
OUT="results/ml1m/visualizations_ncf"

if [ ! -f "$PKL" ]; then
  echo "Error: $PKL not found. Run eval_embedding_diffusion_agnostic.py first."
  echo "Example:"
  echo "  cd src/ContinousDiff"
  echo "  python eval_embedding_diffusion_agnostic.py --dataset ml1m --recommender ncf --num_users 100"
  exit 1
fi

mkdir -p "$OUT"

python src/visualize_agnostic_eval.py \
  --results_pkl "$PKL" \
  --output_dir "$OUT"

echo "Saved figures to $OUT"
