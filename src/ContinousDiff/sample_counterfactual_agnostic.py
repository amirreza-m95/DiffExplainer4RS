"""
Model-agnostic counterfactual generation using guided diffusion.

This script works with any recommender system as a black box, using:
1. Standalone autoencoder for embedding space
2. Diffusion model for generating counterfactual embeddings
3. Black-box recommender for predictions
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os
import argparse
import json

# Add current directory to path
sys.path.append(os.path.dirname(__file__))
from autoencoder import Autoencoder, VariationalAutoencoder
from diffusion_model import TransformerDiffusionModel, DiffusionMLP
from recommender_wrapper import RecommenderWrapper

# === DATASET CONFIGURATIONS ===
DATASET_CONFIGS = {
    'ml1m': {
        'data_path': 'datasets/lxr-CE/ML1M/train_data_ML1M.csv',
        'embedding_path': 'checkpoints/embeddings/user_embeddings_ml1m.npy',
        'autoencoder_path': 'checkpoints/autoencoders/autoencoder_ml1m_best.pt',
        'diffusion_path': 'checkpoints/diffusionModels/diffusion_transformer_ml1m_best.pth',
        'results_file': 'counterfactual_results_ml1m.json',
        'num_items': 3381,
        'num_users': 6037,
        'diffusion_config': {
            'model_type': 'transformer',
            'embedding_dim': 256,
            'hidden_dim': 256,
            'num_layers': 6,
            'num_heads': 8,
            'dropout': 0.1
        },
        'sampling_config': {
            'timesteps': 120,  # Reduced from 1000 for faster sampling
            'guidance_steps': 120,
            'guidance_lambda': 3.0
        }
    },
    'pinterest': {
        'data_path': 'datasets/lxr-CE/Pinterest/train_data_Pinterest.csv',
        'embedding_path': 'checkpoints/embeddings/user_embeddings_pinterest.npy',
        'autoencoder_path': 'checkpoints/autoencoders/autoencoder_pinterest_best.pt',
        'diffusion_path': 'checkpoints/diffusionModels/diffusion_transformer_pinterest_best.pth',
        'results_file': 'counterfactual_results_pinterest.json',
        'num_items': 9362,
        'num_users': 19155,
        'diffusion_config': {
            'model_type': 'transformer',
            'embedding_dim': 256,
            'hidden_dim': 256,
            'num_layers': 6,
            'num_heads': 8,
            'dropout': 0.1
        },
        'sampling_config': {
            'timesteps': 1000,
            'guidance_steps': 30,
            'guidance_lambda': 2.0
        }
    },
    'yahoo': {
        'data_path': 'datasets/lxr-CE/Yahoo/train_data_Yahoo.csv',
        'embedding_path': 'checkpoints/embeddings/user_embeddings_yahoo.npy',
        'autoencoder_path': 'checkpoints/autoencoders/autoencoder_yahoo_best.pt',
        'diffusion_path': 'checkpoints/diffusionModels/diffusion_transformer_yahoo_best.pth',
        'results_file': 'counterfactual_results_yahoo.json',
        'num_items': 4604,
        'num_users': 13797,
        'diffusion_config': {
            'model_type': 'transformer',
            'embedding_dim': 256,
            'hidden_dim': 256,
            'num_layers': 6,
            'num_heads': 8,
            'dropout': 0.1
        },
        'sampling_config': {
            'timesteps': 1000,
            'guidance_steps': 100,
            'guidance_lambda': 2.5
        }
    }
}

# Recommender configurations
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
        'ml1m':  'checkpoints/recommenders/NCF2_ML1M_6e-05_128_0_19.pt',
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
        raise ValueError(f"Unknown diffusion model type: {model_type}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model


def get_top1_from_recommender(user_profile, recommender):
    """
    Get top-1 recommendation from any recommender.

    Args:
        user_profile: User interaction vector (num_items,)
        recommender: RecommenderWrapper instance
    Returns:
        top1: top-1 item index, scores: all item scores
    """
    scores = recommender.predict(user_profile)
    top1 = torch.argmax(scores).item()
    return top1, scores


def sample_counterfactual(orig_embedding, orig_user_profile, orig_top1,
                          autoencoder, recommender, diffusion,
                          betas, alphas, alphas_cumprod,
                          timesteps, guidance_lambda):
    """
    Generate counterfactual embedding using guided diffusion sampling.

    Args:
        orig_embedding: Original user embedding from autoencoder
        orig_user_profile: Original user interaction vector
        orig_top1: Original top-1 recommendation
        autoencoder: Autoencoder model
        recommender: RecommenderWrapper instance
        diffusion: Diffusion model
        betas, alphas, alphas_cumprod: Diffusion schedule
        timesteps: Number of diffusion timesteps
        guidance_lambda: Strength of guidance
    Returns:
        Counterfactual embedding
    """
    # Start from random noise
    x = torch.randn_like(orig_embedding).unsqueeze(0)

    # Reverse diffusion process
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

        # Apply guidance to change top-1 recommendation
        x.requires_grad_(True)

        # Decode embedding to user profile
        cf_user_profile = autoencoder.decode(x.squeeze(0))

        # Get recommendation from black-box recommender
        top1, scores = get_top1_from_recommender(cf_user_profile, recommender)

        if top1 == orig_top1:
            # Compute gradient to reduce score of orig_top1
            loss = guidance_lambda * scores[orig_top1]
            grad = torch.autograd.grad(loss, x)[0]
            x = x - 0.1 * grad  # Step size can be tuned

        x = x.detach()

    return x.squeeze(0)


def generate_counterfactuals(dataset_name, recommender_type, num_users=10):
    """
    Generate counterfactuals for a specific dataset and recommender.

    Args:
        dataset_name: Name of the dataset
        recommender_type: Type of recommender ('vae', 'mlp', 'ncf')
        num_users: Number of users to generate counterfactuals for
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Dataset '{dataset_name}' not supported.")

    if recommender_type not in RECOMMENDER_CONFIGS:
        raise ValueError(f"Recommender '{recommender_type}' not supported.")

    config = DATASET_CONFIGS[dataset_name]
    recommender_checkpoint = RECOMMENDER_CONFIGS[recommender_type].get(dataset_name)

    if recommender_checkpoint is None:
        raise ValueError(f"No checkpoint found for {recommender_type} on {dataset_name}")

    print("=" * 80)
    print(f"Generating counterfactuals for {dataset_name.upper()} with {recommender_type.upper()}")
    print("=" * 80)
    print(f"Data path: {config['data_path']}")
    print(f"Autoencoder: {config['autoencoder_path']}")
    print(f"Diffusion model: {config['diffusion_path']}")
    print(f"Recommender: {recommender_checkpoint}")
    print(f"Device: {DEVICE}")
    print("=" * 80)

    # Load data
    print("\nLoading user data...")
    df = pd.read_csv(config['data_path'], index_col=0)
    if 'user_id' in df.columns:
        df = df.drop(columns=['user_id'])
    user_data = df.values.astype(np.float32)
    print(f"Loaded {user_data.shape[0]} users")

    # Load embeddings
    print("\nLoading embeddings...")
    embeddings = np.load(config['embedding_path'])
    embeddings = torch.FloatTensor(embeddings).to(DEVICE)
    print(f"Loaded {embeddings.shape[0]} embeddings")

    # Load autoencoder
    print("\nLoading autoencoder...")
    autoencoder = load_autoencoder(
        config['autoencoder_path'],
        config['num_items'],
        use_vae=USE_VAE_AUTOENCODER
    )
    print("Autoencoder loaded")

    # Load diffusion model
    print("\nLoading diffusion model...")
    diffusion = load_diffusion_model(
        config['diffusion_path'],
        config['diffusion_config']
    )
    print("Diffusion model loaded")

    # Load recommender
    print(f"\nLoading {recommender_type.upper()} recommender...")

    # Get recommender-specific kwargs
    recommender_kwargs = {}
    if recommender_type == 'mlp' and dataset_name in MLP_CONFIGS:
        recommender_kwargs = MLP_CONFIGS[dataset_name]
        print(f"Using MLP config: {recommender_kwargs}")
    elif recommender_type == 'ncf' and dataset_name in NCF_CONFIGS:
        recommender_kwargs = NCF_CONFIGS[dataset_name]
        print(f"Using NCF config: {recommender_kwargs}")

    recommender = RecommenderWrapper(
        model_type=recommender_type,
        checkpoint_path=recommender_checkpoint,
        num_items=config['num_items'],
        num_users=config['num_users'],
        device=DEVICE,
        **recommender_kwargs
    )
    print("Recommender loaded")

    # Setup diffusion schedule
    sampling_config = config['sampling_config']
    timesteps = sampling_config['timesteps']
    guidance_lambda = sampling_config['guidance_lambda']

    betas = torch.linspace(1e-4, 0.02, timesteps, device=DEVICE)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # Generate counterfactuals
    print(f"\nGenerating counterfactuals for {num_users} users...")
    print("=" * 80)

    results = []

    for user_idx in range(min(num_users, user_data.shape[0])):
        print(f"\nUser {user_idx + 1}/{num_users}")
        print("-" * 80)

        # Original user profile and embedding
        orig_profile = torch.FloatTensor(user_data[user_idx]).to(DEVICE)
        orig_embedding = embeddings[user_idx]

        # Get original top-1 recommendation
        orig_top1, orig_scores = get_top1_from_recommender(orig_profile, recommender)
        print(f"Original top-1 item: {orig_top1}")

        # Generate counterfactual
        print("Generating counterfactual embedding...")
        cf_embedding = sample_counterfactual(
            orig_embedding, orig_profile, orig_top1,
            autoencoder, recommender, diffusion,
            betas, alphas, alphas_cumprod,
            timesteps, guidance_lambda
        )

        # Decode counterfactual embedding to user profile
        cf_profile = autoencoder.decode(cf_embedding)

        # Get counterfactual top-1 recommendation
        cf_top1, cf_scores = get_top1_from_recommender(cf_profile, recommender)
        print(f"Counterfactual top-1 item: {cf_top1}")
        print(f"Success: {'YES' if cf_top1 != orig_top1 else 'NO'}")

        # Store results
        results.append({
            'user_idx': user_idx,
            'orig_top1': int(orig_top1),
            'cf_top1': int(cf_top1),
            'success': cf_top1 != orig_top1,
            'orig_embedding': orig_embedding.cpu().numpy().tolist(),
            'cf_embedding': cf_embedding.cpu().numpy().tolist()
        })

    # Save results
    results_file = f"results/{dataset_name}/{recommender_type}_counterfactual_results.json"
    results_dir = Path(results_file).parent
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    success_rate = sum(r['success'] for r in results) / len(results) * 100
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total users: {len(results)}")
    print(f"Successful counterfactuals: {sum(r['success'] for r in results)}")
    print(f"Success rate: {success_rate:.2f}%")
    print(f"Results saved to: {results_file}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate counterfactuals with any recommender')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['ml1m', 'pinterest', 'yahoo'],
                       help='Dataset to use')
    parser.add_argument('--recommender', type=str, required=True,
                       choices=['vae', 'mlp', 'ncf'],
                       help='Recommender system to use')
    parser.add_argument('--num_users', type=int, default=10,
                       help='Number of users to generate counterfactuals for')
    parser.add_argument('--use_vae_autoencoder', action='store_true',
                       help='Set if autoencoder was trained as VAE')

    args = parser.parse_args()

    USE_VAE_AUTOENCODER = args.use_vae_autoencoder

    generate_counterfactuals(args.dataset, args.recommender, args.num_users)
