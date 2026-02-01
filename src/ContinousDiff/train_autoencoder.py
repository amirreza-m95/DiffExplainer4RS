"""
Train standalone autoencoder for model-agnostic DiceRec.

This script trains an autoencoder to compress user interaction vectors into
a latent space, independently of any specific recommender system.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os
import argparse

# Add current directory to path
sys.path.append(os.path.dirname(__file__))
from autoencoder import Autoencoder, VariationalAutoencoder

# === DATASET CONFIGURATIONS ===
DATASET_CONFIGS = {
    'ml1m': {
        'data_path': 'datasets/lxr-CE/ML1M/train_data_ML1M.csv',
        'checkpoint_path': 'checkpoints/autoencoders/autoencoder_ml1m_best.pt',
        'num_items': 3381
    },
    'pinterest': {
        'data_path': 'datasets/lxr-CE/Pinterest/train_data_Pinterest.csv',
        'checkpoint_path': 'checkpoints/autoencoders/autoencoder_pinterest_best.pt',
        'num_items': 9362
    },
    'yahoo': {
        'data_path': 'datasets/lxr-CE/Yahoo/train_data_Yahoo.csv',
        'checkpoint_path': 'checkpoints/autoencoders/autoencoder_yahoo_best.pt',
        'num_items': 4604
    }
}

# === GLOBAL CONFIGURATION ===
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LATENT_DIM = 256
HIDDEN_DIMS = [256, 256]
DROPOUT = 0.5
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 1e-3
USE_VAE = False  # Set to True to use VAE instead of standard autoencoder
LOSS_TYPE = 'bce'  # 'bce' for binary cross entropy, 'mse' for mean squared error

# VAE-specific config
VAE_ANNEAL_CAP = 0.2
VAE_TOTAL_ANNEAL_STEPS = 200000


def load_user_histories(data_path):
    """
    Load user histories from CSV file.

    Args:
        data_path: Path to CSV file
    Returns:
        np.ndarray of shape (num_users, num_items)
    """
    df = pd.read_csv(data_path, index_col=0)
    if 'user_id' in df.columns:
        df = df.drop(columns=['user_id'])
    data = df.values.astype(np.float32)
    return data


def train_autoencoder(dataset_name):
    """
    Train autoencoder for a specific dataset.

    Args:
        dataset_name: Name of the dataset ('ml1m', 'pinterest', or 'yahoo')
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Dataset '{dataset_name}' not supported. Available: {list(DATASET_CONFIGS.keys())}")

    config = DATASET_CONFIGS[dataset_name]

    print("=" * 80)
    print(f"Training {'VAE' if USE_VAE else 'Autoencoder'} for {dataset_name.upper()} dataset")
    print("=" * 80)
    print(f"Data path: {config['data_path']}")
    print(f"Checkpoint path: {config['checkpoint_path']}")
    print(f"Number of items: {config['num_items']}")
    print(f"Latent dimension: {LATENT_DIM}")
    print(f"Hidden dimensions: {HIDDEN_DIMS}")
    print(f"Dropout: {DROPOUT}")
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    if not USE_VAE:
        print(f"Loss type: {LOSS_TYPE}")
    print("=" * 80)

    # Load data
    print("\nLoading user interaction data...")
    data = load_user_histories(config['data_path'])
    print(f"Loaded {data.shape[0]} users with {data.shape[1]} items")
    print(f"Sparsity: {(1 - np.count_nonzero(data) / data.size) * 100:.2f}%")

    # Create autoencoder
    print(f"\nCreating {'VAE' if USE_VAE else 'autoencoder'}...")
    if USE_VAE:
        model = VariationalAutoencoder(
            num_items=config['num_items'],
            latent_dim=LATENT_DIM,
            hidden_dims=HIDDEN_DIMS,
            dropout=DROPOUT,
            anneal_cap=VAE_ANNEAL_CAP,
            total_anneal_steps=VAE_TOTAL_ANNEAL_STEPS,
            device=DEVICE
        ).to(DEVICE)
    else:
        model = Autoencoder(
            num_items=config['num_items'],
            latent_dim=LATENT_DIM,
            hidden_dims=HIDDEN_DIMS,
            dropout=DROPOUT,
            device=DEVICE
        ).to(DEVICE)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Create checkpoint directory if needed
    checkpoint_dir = Path(config['checkpoint_path']).parent
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    print(f"\nStarting training for {EPOCHS} epochs...")
    print("=" * 80)

    best_loss = float('inf')
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print("-" * 80)

        if USE_VAE:
            avg_loss = model.train_one_epoch(data, optimizer, BATCH_SIZE)
        else:
            avg_loss = model.train_one_epoch(data, optimizer, BATCH_SIZE, loss_type=LOSS_TYPE)

        print(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.6f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), config['checkpoint_path'])
            print(f"✓ Saved new best model (loss: {best_loss:.6f}) to {config['checkpoint_path']}")

    print("\n" + "=" * 80)
    print(f"Training completed! Best loss: {best_loss:.6f}")
    print(f"Model saved to: {config['checkpoint_path']}")
    print("=" * 80)

    # Test reconstruction
    print("\nTesting reconstruction on a sample user...")
    model.eval()
    with torch.no_grad():
        sample_idx = 0
        sample_user = torch.FloatTensor(data[sample_idx:sample_idx+1]).to(DEVICE)
        embedding = model.encode(sample_user)
        reconstruction = model.decode(embedding)

        original_items = torch.where(sample_user[0] > 0)[0].cpu().numpy()
        recon_scores = reconstruction[0].cpu().numpy()
        top_k_recon = np.argsort(recon_scores)[-len(original_items):][::-1]

        print(f"Sample user {sample_idx}:")
        print(f"  Original items (first 10): {original_items[:10]}")
        print(f"  Top-K reconstructed items (first 10): {top_k_recon[:10]}")
        print(f"  Embedding shape: {embedding.shape}")
        print(f"  Embedding norm: {torch.norm(embedding).item():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train autoencoder for DiceRec')
    parser.add_argument('--dataset', type=str, default='ml1m',
                       choices=['ml1m', 'pinterest', 'yahoo'],
                       help='Dataset to train on (default: ml1m)')
    parser.add_argument('--latent_dim', type=int, default=256,
                       help='Latent dimension (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate (default: 1e-3)')
    parser.add_argument('--use_vae', action='store_true',
                       help='Use VAE instead of standard autoencoder')
    parser.add_argument('--loss_type', type=str, default='bce',
                       choices=['bce', 'mse'],
                       help='Loss type for standard autoencoder (default: bce)')

    args = parser.parse_args()

    # Update global config with command line arguments
    LATENT_DIM = args.latent_dim
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    USE_VAE = args.use_vae
    LOSS_TYPE = args.loss_type

    # Train autoencoder
    train_autoencoder(args.dataset)
