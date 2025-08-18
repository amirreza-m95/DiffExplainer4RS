import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os
import argparse

# Add LXR directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'LXR')))
from recommenders_architecture import VAE

# === DATASET CONFIGURATIONS ===
DATASET_CONFIGS = {
    'ml1m': {
        'data_path': 'datasets/lxr-CE/ML1M/train_data_ML1M.csv',
        'checkpoint_path': 'checkpoints/recommenders/VAE/VAE_ML1M_4_28_128newbest.pt',
        'embedding_save_path': 'checkpoints/embeddings/user_embeddings_ml1m.npy',
        'user_history_dim': 3381
    },
    'pinterest': {
        'data_path': 'datasets/lxr-CE/Pinterest/train_data_Pinterest.csv',
        'checkpoint_path': 'checkpoints/recommenders/VAE/VAE_Pinterest_8_12_128newbest.pt',
        'embedding_save_path': 'checkpoints/embeddings/user_embeddings_pinterest.npy',
        'user_history_dim': 9362
    }
}

# === GLOBAL CONFIGURATION ===
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# VAE config (should match training)
VAE_config = {
    "enc_dims": [256, 256],
    "dropout": 0.5,
    "anneal_cap": 0.2,
    "total_anneal_steps": 200000
}

def load_user_histories(data_path):
    """
    Load user histories from CSV file, drop user_id column if present.
    Args:
        data_path: Path to CSV file
    Returns:
        np.ndarray of shape (num_users, USER_HISTORY_DIM)
    """
    df = pd.read_csv(data_path, index_col=0)
    if 'user_id' in df.columns:
        df = df.drop(columns=['user_id'])
    data = df.values.astype(np.float32)
    return data

def load_vae_recommender(checkpoint_path, user_history_dim, device=DEVICE):
    """
    Load a trained VAE recommender from checkpoint.
    Args:
        checkpoint_path: Path to .pt file
        user_history_dim: Number of items in the dataset
        device: torch.device
    Returns:
        VAE model (eval mode, frozen)
    """
    kw_dict = {'device': device, 'num_items': user_history_dim}
    model = VAE(VAE_config, **kw_dict).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model

def extract_embeddings(dataset_name):
    """
    Extract VAE embeddings for a specific dataset.
    Args:
        dataset_name: Name of the dataset ('ml1m' or 'pinterest')
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Dataset '{dataset_name}' not supported. Available datasets: {list(DATASET_CONFIGS.keys())}")
    
    config = DATASET_CONFIGS[dataset_name]
    
    print(f"Extracting embeddings for {dataset_name} dataset...")
    print(f"Data path: {config['data_path']}")
    print(f"Checkpoint path: {config['checkpoint_path']}")
    print(f"User history dimension: {config['user_history_dim']}")
    
    # Load user histories
    data = load_user_histories(config['data_path'])
    print(f"Loaded {data.shape[0]} users with {data.shape[1]} items each")
    
    # Load trained VAE
    vae = load_vae_recommender(config['checkpoint_path'], config['user_history_dim'], device=DEVICE)
    
    embeddings = []
    batch_size = 256
    
    # Extract mean embeddings in batches
    with torch.no_grad():
        for i in range(0, data.shape[0], batch_size):
            batch = torch.tensor(data[i:i+batch_size], device=DEVICE)
            # Forward through encoder layers only
            h = torch.nn.functional.normalize(batch, dim=-1)
            h = torch.nn.functional.dropout(h, p=vae.dropout, training=False)
            for layer in vae.encoder:
                h = layer(h)
            mu_q = h[:, :vae.enc_dims[-1]].cpu().numpy()
            embeddings.append(mu_q)
    
    embeddings = np.concatenate(embeddings, axis=0)
    
    # Save all user embeddings to .npy file
    np.save(config['embedding_save_path'], embeddings)
    print(f"Saved {embeddings.shape[0]} user embeddings to {config['embedding_save_path']}")
    print(f"Embedding shape: {embeddings.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract VAE embeddings for different datasets')
    parser.add_argument('--dataset', type=str, default='ml1m', 
                       choices=['ml1m', 'pinterest'],
                       help='Dataset to extract embeddings for (default: ml1m)')
    
    args = parser.parse_args()
    
    # Extract embeddings for the specified dataset
    extract_embeddings(args.dataset) 