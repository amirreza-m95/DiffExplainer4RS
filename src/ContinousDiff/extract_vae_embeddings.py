import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os

# Add LXR directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'LXR')))
from recommenders_architecture import VAE

# === CONFIGURATION ===
DATA_PATH = Path('datasets/lxr-CE/ML1M/train_data_ML1M.csv')
CHECKPOINT_PATH = Path('checkpoints/recommenders/VAE_ML1M_0_19_128.pt')
EMBEDDING_SAVE_PATH = Path('user_embeddings.npy')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USER_HISTORY_DIM = 3381  # Number of items

# VAE config (should match training)
VAE_config = {
    "enc_dims": [256, 64],
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

def load_vae_recommender(checkpoint_path, device=DEVICE):
    """
    Load a trained VAE recommender from checkpoint.
    Args:
        checkpoint_path: Path to .pt file
        device: torch.device
    Returns:
        VAE model (eval mode, frozen)
    """
    kw_dict = {'device': device, 'num_items': USER_HISTORY_DIM}
    model = VAE(VAE_config, **kw_dict).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model

if __name__ == "__main__":
    # Load user histories
    data = load_user_histories(DATA_PATH)
    # Load trained VAE
    vae = load_vae_recommender(CHECKPOINT_PATH, device=DEVICE)
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
    np.save(EMBEDDING_SAVE_PATH, embeddings)
    print(f"Saved user embeddings to {EMBEDDING_SAVE_PATH}") 