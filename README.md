# DiceRec
### Denoising Diffusion-based Guided Counterfactual Explanation for Recommender Systems

## Usage


## Folders

* **benchmarks**: contains the code and results of computational efficiency analysis.
* **src**: contains several code files:
  - ContinousDiff - Main implementation files for DiceRec.
  - LXR - Implementation of the baselines.
  - Diffusion - implementation of the discrete optimization of the diffusion model.
* **results** 
- contains 3 folders each for the full results and visualizations of the 3 datasets.

## Reproducibility and Commands

Below are the commands to set up the environment and reproduce the main results. Paths assume running from the repository root (`/home/amir.reza/counterfactual/DiffExplainer4RS`).

### Environment setup

```bash
# Create and activate a virtual environment (Python 3.10+ recommended)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements_clean.txt
```

### Training the diffusion model (embeddings)

Train a diffusion model on precomputed user embeddings for a dataset.

```bash
# dataset_name ∈ {ml1m, pinterest, yahoo} (default: ml1m)
python src/ContinousDiff/train_diffusion.py --dataset <dataset_name>
```

Defaults (per dataset) for input embeddings and checkpoint paths are defined inside the script.

### Counterfactual generation with diffusion

Generate counterfactual user embeddings and basic statistics for a subset of users.

```bash
# dataset_name ∈ {ml1m, pinterest, yahoo}; num_users default: 10
python src/ContinousDiff/sample_counterfactual.py --dataset <dataset_name> --num_users 50
```

Outputs are written as JSON (e.g., `counterfactual_results_<dataset_name>.json`).

Prerequisites (paths defined inside the script for each dataset):
- Precomputed embeddings in `checkpoints/embeddings/*.npy`
- Trained diffusion checkpoints in `checkpoints/diffusionModels/*.pth`
- Trained VAE checkpoint in `checkpoints/recommenders/VAE/*.pt`
### Results
![ML1M_fine-grained](https://github.com/amirreza-m95/DiffExplainer4RS/blob/main/results/ML1M/onebyone/comprehensive_metric_trends_NDCGP_POS3_POS5_0_10.png)
![ML1M_Constrained](https://github.com/amirreza-m95/DiffExplainer4RS/blob/main/results/ML1M/onebyoneuntil20p/comprehensive_metric_trends_NDCGP_POS3_POS5_0_10.png)
### Embedding-diffusion evaluation (metric curves over removal steps)

Run the integrated-guidance evaluation that computes DEL/INS/NDCG and POS/NEG metrics over 10% removal steps.

```bash
# dataset_name ∈ {ml1m, pinterest, yahoo}; num_users optional
python src/ContinousDiff/eval_embedding_diffusion.py --dataset <dataset_name> --num_users 100
```

Default result paths are configured per dataset inside the script (note Pinterest uses `.npyl` as written in code).

Note: Ensure the dataset CSVs exist under `datasets/lxr-CE/<Dataset>/test_data_<Dataset>.csv` and model checkpoints exist under `checkpoints/...` as configured in the script.

```

### Notes
- Replace paths/checkpoints with your actual files if names differ.
- GPU is automatically used if available; otherwise CPU.
- For large models, consider enabling mixed precision and gradient checkpointing where relevant. 
