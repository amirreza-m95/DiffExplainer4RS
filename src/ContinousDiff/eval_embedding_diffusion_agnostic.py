"""
Model-agnostic evaluation of counterfactual explanations.

This script evaluates DiceRec with any recommender system, computing:
- DEL (Deletion): Score after removing important items
- INS (Insertion): Score when keeping only removed items
- NDCG: Ranking quality
- POS@k: Precision when removing most important items
- NEG@k: Precision when removing least important items
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os
import pickle
import argparse
from tqdm import tqdm

# Add current directory to path
sys.path.append(os.path.dirname(__file__))
from autoencoder import Autoencoder, VariationalAutoencoder
from diffusion_model import TransformerDiffusionModel, DiffusionMLP
from recommender_wrapper import RecommenderWrapper

# === DATASET CONFIGURATIONS ===
DATASET_CONFIGS = {
    'ml1m': {
        'test_data_path': 'datasets/lxr-CE/ML1M/test_data_ML1M.csv',
        'train_data_path': 'datasets/lxr-CE/ML1M/train_data_ML1M.csv',
        'autoencoder_path': 'checkpoints/autoencoders/autoencoder_ml1m_best.pt',
        'diffusion_path': 'checkpoints/diffusionModels/diffusion_transformer_ml1m_best.pth',
        'num_items': 3381,
        'num_users': 6037,
        'timesteps': 120,
        'guidance_lambda': 3.0,
        'diffusion_config': {
            'model_type': 'transformer',
            'embedding_dim': 256,
            'hidden_dim': 256,
            'num_layers': 6,
            'num_heads': 8,
            'dropout': 0.1
        }
    },
    'pinterest': {
        'test_data_path': 'datasets/lxr-CE/Pinterest/test_data_Pinterest.csv',
        'train_data_path': 'datasets/lxr-CE/Pinterest/train_data_Pinterest.csv',
        'autoencoder_path': 'checkpoints/autoencoders/autoencoder_pinterest_best.pt',
        'diffusion_path': 'checkpoints/diffusionModels/diffusion_transformer_pinterest_best.pth',
        'num_items': 9362,
        'num_users': 19155,
        'timesteps': 30,
        'guidance_lambda': 2.0,
        'diffusion_config': {
            'model_type': 'transformer',
            'embedding_dim': 256,
            'hidden_dim': 256,
            'num_layers': 6,
            'num_heads': 8,
            'dropout': 0.1
        }
    },
    'yahoo': {
        'test_data_path': 'datasets/lxr-CE/Yahoo/test_data_Yahoo.csv',
        'train_data_path': 'datasets/lxr-CE/Yahoo/train_data_Yahoo.csv',
        'autoencoder_path': 'checkpoints/autoencoders/autoencoder_yahoo_best.pt',
        'diffusion_path': 'checkpoints/diffusionModels/diffusion_transformer_yahoo_best.pth',
        'num_items': 4604,
        'num_users': 13797,
        'timesteps': 100,
        'guidance_lambda': 2.5,
        'diffusion_config': {
            'model_type': 'transformer',
            'embedding_dim': 256,
            'hidden_dim': 256,
            'num_layers': 6,
            'num_heads': 8,
            'dropout': 0.1
        }
    }
}

RECOMMENDER_CONFIGS = {
    'vae': {
        'ml1m': 'checkpoints/recommenders/VAE/VAE_ML1M_4_28_128newbest.pt',
        'pinterest': 'checkpoints/recommenders/VAE/VAE_Pinterest_8_12_128newbest.pt',
        'yahoo': 'checkpoints/recommenders/VAE/VAE_Yahoo_0_32_256newbest.pt'
    },
    'mlp': {
        'ml1m': 'checkpoints/recommenders/MLP/MLP_ML1M_best.pt',
        'pinterest': 'checkpoints/recommenders/MLP/MLP_Pinterest_best.pt',
        'yahoo': 'checkpoints/recommenders/MLP/MLP_Yahoo_best.pt'
    },
    'ncf': {
        'ml1m': 'checkpoints/recommenders/NCF2_ML1M_6e-05_128_0_19.pt',
        'pinterest': 'checkpoints/recommenders/NCF/NCF_Pinterest_best.pt',
        'yahoo': 'checkpoints/recommenders/NCF/NCF_Yahoo_best.pt'
    }
}

# MLP-specific hyperparameters (must match training)
MLP_CONFIGS = {
    'ml1m': {
        'hidden_size': 64  # From best checkpoint: hd64
    },
    'pinterest': {
        'hidden_size': 512  # From best checkpoint: hd512
    },
    'yahoo': {
        'hidden_size': 64  # From best checkpoint: hd64
    }
}

# NCF-specific hyperparameters (must match training)
NCF_CONFIGS = {
    'ml1m': {
        'factor_num': 16,  # embed_user_GMF is [16, 3381]
        'num_layers': 2,   # MLP has layers up to index 4 (blocks 0,1) = 2 layers
        'dropout': 0.47,
        'ncf_model_type': 'NeuMF-end'
    },
    'pinterest': {
        'factor_num': 32,
        'num_layers': 3,
        'dropout': 0.0,
        'ncf_model_type': 'NeuMF-end'
    },
    'yahoo': {
        'factor_num': 32,
        'num_layers': 3,
        'dropout': 0.0,
        'ncf_model_type': 'NeuMF-end'
    }
}

# === GLOBAL CONFIGURATION ===
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LATENT_DIM = 256
HIDDEN_DIMS = [256, 256]
DROPOUT = 0.5
USE_VAE_AUTOENCODER = False
NUM_BINS = 10  # Evaluation at 10%, 20%, ..., 100% removal


def load_autoencoder(checkpoint_path, num_items, use_vae=False):
    """Load trained autoencoder."""
    if use_vae:
        model = VariationalAutoencoder(
            num_items=num_items,
            latent_dim=LATENT_DIM,
            hidden_dims=HIDDEN_DIMS,
            dropout=DROPOUT,
            device=DEVICE
        ).to(DEVICE)
    else:
        model = Autoencoder(
            num_items=num_items,
            latent_dim=LATENT_DIM,
            hidden_dims=HIDDEN_DIMS,
            dropout=DROPOUT,
            device=DEVICE
        ).to(DEVICE)

    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def load_diffusion_model(checkpoint_path, config):
    """Load trained diffusion model."""
    model_type = config.get('model_type', 'transformer')

    # Create a copy of config without 'model_type' for passing to model constructor
    model_config = {k: v for k, v in config.items() if k != 'model_type'}

    if model_type == 'transformer':
        model = TransformerDiffusionModel(**model_config).to(DEVICE)
    elif model_type == 'mlp':
        model = DiffusionMLP(model_config['embedding_dim']).to(DEVICE)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def sample_counterfactual_embedding(orig_user_profile, autoencoder, recommender,
                                    diffusion, betas, alphas, alphas_cumprod,
                                    timesteps, guidance_lambda):
    """
    Generate counterfactual embedding using guided diffusion.

    Returns:
        cf_embedding: Counterfactual embedding
    """
    # Get original embedding and top-1
    with torch.no_grad():
        orig_embedding = autoencoder.encode(orig_user_profile)
        orig_scores = recommender.predict(orig_user_profile)
        orig_top1 = torch.argmax(orig_scores).item()

    # Start from random noise
    x = torch.randn_like(orig_embedding).unsqueeze(0)

    # Reverse diffusion
    for t in reversed(range(timesteps)):
        t_tensor = torch.full((1,), t, dtype=torch.long, device=DEVICE)

        with torch.no_grad():
            noise_pred = diffusion(x, t_tensor)

        alpha_t = alphas[t]
        alpha_cumprod_t = alphas_cumprod[t]

        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)

        coef1 = 1.0 / torch.sqrt(alpha_t)
        coef2 = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_cumprod_t)
        x = coef1 * (x - coef2 * noise_pred) + torch.sqrt(betas[t]) * noise

        # Guidance
        x.requires_grad_(True)
        cf_profile_temp = autoencoder.decode(x.squeeze(0))
        scores = recommender.predict(cf_profile_temp)
        top1 = torch.argmax(scores).item()

        if top1 == orig_top1:
            loss = guidance_lambda * scores[orig_top1]
            grad = torch.autograd.grad(loss, x)[0]
            x = x - 0.1 * grad

        x = x.detach()

    return x.squeeze(0)


def get_item_importance_scores(orig_profile, cf_profile, recommender):
    """
    Compute item importance scores based on difference between
    original and counterfactual profiles.

    Returns:
        importance_scores: Item importance scores (higher = more important)
    """
    with torch.no_grad():
        cf_scores = recommender.predict(cf_profile)

    # Importance = how much the counterfactual "wants" each item
    # Items with higher counterfactual scores are more important
    importance = cf_scores.clone()

    # Only consider items the user actually has
    importance[orig_profile == 0] = float('-inf')

    return importance


def evaluate_explanation(orig_profile, cf_profile, target_item, recommender, num_bins=10):
    """
    Evaluate explanation quality using DEL, INS, NDCG, POS@k, NEG@k.

    Args:
        orig_profile: Original user profile
        cf_profile: Counterfactual user profile
        target_item: Target item to explain
        recommender: RecommenderWrapper instance
        num_bins: Number of removal bins

    Returns:
        dict: Evaluation metrics
    """
    # Get item importance scores
    importance_scores = get_item_importance_scores(orig_profile, cf_profile, recommender)

    # Get indices of user's items, sorted by importance (descending)
    user_items = torch.where(orig_profile > 0)[0]
    user_item_importance = importance_scores[user_items]
    sorted_indices = torch.argsort(user_item_importance, descending=True)
    sorted_items = user_items[sorted_indices]

    # Get original target item score
    with torch.no_grad():
        orig_scores = recommender.predict(orig_profile)
        orig_target_score = orig_scores[target_item].item()

    num_items_to_remove = len(sorted_items)
    bin_size = max(1, num_items_to_remove // num_bins)

    results = {
        'del': [],  # Deletion metric
        'ins': [],  # Insertion metric
        'ndcg': [], # NDCG metric
        'pos@k': {k: [] for k in [1, 5, 10, 20, 50, 100]},
        'neg@k': {k: [] for k in [1, 5, 10, 20, 50, 100]},
        'bins': []
    }

    # Evaluate at each bin
    for bin_idx in range(1, num_bins + 1):
        num_remove = min(bin_idx * bin_size, num_items_to_remove)

        # DEL: Remove most important items
        del_profile = orig_profile.clone()
        items_to_remove = sorted_items[:num_remove]
        del_profile[items_to_remove] = 0

        with torch.no_grad():
            del_scores = recommender.predict(del_profile)
            del_target_score = del_scores[target_item].item()

        # INS: Keep only removed items
        ins_profile = torch.zeros_like(orig_profile)
        ins_profile[items_to_remove] = orig_profile[items_to_remove]

        with torch.no_grad():
            ins_scores = recommender.predict(ins_profile)
            ins_target_score = ins_scores[target_item].item()

        # NDCG: Rank of target item after deletion
        with torch.no_grad():
            del_sorted = torch.argsort(del_scores, descending=True)
            try:
                rank = torch.where(del_sorted == target_item)[0].item()
                ndcg = 1.0 / np.log2(rank + 2)
            except:
                ndcg = 0.0

        results['del'].append(del_target_score)
        results['ins'].append(ins_target_score)
        results['ndcg'].append(ndcg)
        results['bins'].append(num_remove)

        # POS@k and NEG@k
        for k in [1, 5, 10, 20, 50, 100]:
            if num_remove >= k:
                # POS@k: Remove top-k most important
                pos_profile = orig_profile.clone()
                pos_profile[sorted_items[:k]] = 0

                with torch.no_grad():
                    pos_scores = recommender.predict(pos_profile)
                    pos_target_score = pos_scores[target_item].item()

                results['pos@k'][k].append(pos_target_score)

                # NEG@k: Remove top-k least important
                neg_profile = orig_profile.clone()
                neg_profile[sorted_items[-k:]] = 0

                with torch.no_grad():
                    neg_scores = recommender.predict(neg_profile)
                    neg_target_score = neg_scores[target_item].item()

                results['neg@k'][k].append(neg_target_score)

    return results


def run_evaluation(dataset_name, recommender_type, num_test_users=100):
    """
    Run full evaluation on dataset with specified recommender.
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Dataset '{dataset_name}' not supported.")

    if recommender_type not in RECOMMENDER_CONFIGS:
        raise ValueError(f"Recommender '{recommender_type}' not supported.")

    config = DATASET_CONFIGS[dataset_name]
    recommender_checkpoint = RECOMMENDER_CONFIGS[recommender_type].get(dataset_name)

    if recommender_checkpoint is None:
        raise ValueError(f"No checkpoint for {recommender_type} on {dataset_name}")

    print("=" * 80)
    print(f"Evaluating DiceRec: {dataset_name.upper()} + {recommender_type.upper()}")
    print("=" * 80)

    # Load models
    print("\nLoading models...")
    autoencoder = load_autoencoder(config['autoencoder_path'], config['num_items'], USE_VAE_AUTOENCODER)
    diffusion = load_diffusion_model(config['diffusion_path'], config['diffusion_config'])

    # Get recommender-specific kwargs
    recommender_kwargs = {}
    if recommender_type == 'mlp' and dataset_name in MLP_CONFIGS:
        recommender_kwargs = MLP_CONFIGS[dataset_name]
    elif recommender_type == 'ncf' and dataset_name in NCF_CONFIGS:
        recommender_kwargs = NCF_CONFIGS[dataset_name]

    recommender = RecommenderWrapper(
        model_type=recommender_type,
        checkpoint_path=recommender_checkpoint,
        num_items=config['num_items'],
        num_users=config['num_users'],
        device=DEVICE,
        **recommender_kwargs
    )
    print("Models loaded")

    # Load test data
    print("\nLoading test data...")
    test_df = pd.read_csv(config['test_data_path'], index_col=0)
    if 'user_id' in test_df.columns:
        test_df = test_df.drop(columns=['user_id'])
    test_data = test_df.values.astype(np.float32)
    print(f"Loaded {test_data.shape[0]} test users")

    # Setup diffusion schedule
    timesteps = config['timesteps']
    guidance_lambda = config['guidance_lambda']
    betas = torch.linspace(1e-4, 0.02, timesteps, device=DEVICE)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # Run evaluation
    print(f"\nEvaluating {min(num_test_users, test_data.shape[0])} users...")
    all_results = []

    for user_idx in tqdm(range(min(num_test_users, test_data.shape[0]))):
        orig_profile = torch.FloatTensor(test_data[user_idx]).to(DEVICE)

        # Get target item (top-1 recommendation)
        with torch.no_grad():
            scores = recommender.predict(orig_profile)
            target_item = torch.argmax(scores).item()

        # Generate counterfactual
        cf_embedding = sample_counterfactual_embedding(
            orig_profile, autoencoder, recommender, diffusion,
            betas, alphas, alphas_cumprod, timesteps, guidance_lambda
        )
        cf_profile = autoencoder.decode(cf_embedding)

        # Evaluate
        user_results = evaluate_explanation(
            orig_profile, cf_profile, target_item, recommender, NUM_BINS
        )
        all_results.append(user_results)

    # Aggregate results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    # Save results
    results_dir = Path(f"results/{dataset_name}")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"{recommender_type}_eval_results.pkl"

    with open(results_file, 'wb') as f:
        pickle.dump(all_results, f)

    print(f"Results saved to: {results_file}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model-agnostic DiceRec')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['ml1m', 'pinterest', 'yahoo'],
                       help='Dataset to evaluate on')
    parser.add_argument('--recommender', type=str, required=True,
                       choices=['vae', 'mlp', 'ncf'],
                       help='Recommender system to use')
    parser.add_argument('--num_users', type=int, default=100,
                       help='Number of test users to evaluate')
    parser.add_argument('--use_vae_autoencoder', action='store_true',
                       help='Set if autoencoder was trained as VAE')

    args = parser.parse_args()

    USE_VAE_AUTOENCODER = args.use_vae_autoencoder

    run_evaluation(args.dataset, args.recommender, args.num_users)
