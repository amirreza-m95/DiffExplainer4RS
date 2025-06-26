import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# === Import VAE and get_user_recommended_item ===
from src.LXR.recommenders_architecture import VAE
from src.LXR.help_functions import get_user_recommended_item

# ========== CONFIGURATION ==========
DATA_PATH = Path('datasets/lxr-CE/ML1M/train_data_ML1M.csv')
CHECKPOINT_PATH = Path('checkpoints/recommenders/VAE_ML1M_0_19_128.pt')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USER_HISTORY_DIM = 3381  # Number of items
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 1e-3
LAMBDA_CF = 0.5  # Weight for counterfactual loss
LAMBDA_L1 = 1.0   # Increased weight for L1 loss (encourage minimal changes)
LAMBDA_PRESERVE = 1.0  # New weight to strongly preserve non-noised positions

# ========== DATA LOADING ==========
def load_user_histories(data_path):
    df = pd.read_csv(data_path, index_col=0)
    # Remove user_id column if present
    if 'user_id' in df.columns:
        df = df.drop(columns=['user_id'])
    data = df.values.astype(np.float32)
    return data

# ========== DIFFUSION PROCESS ==========
def bernoulli_diffusion(x0, t, max_steps=10):
    """
    x0: [batch, dim] binary user histories
    t: [batch] time steps (noise levels, 0 <= t < max_steps)
    Returns: x_t (noisy histories), mask (positions that were noised)
    """
    # Linearly increase noise probability from 0 to 0.2 (less noise)
    p = 0.7 * (t / max_steps)[:, None]  # shape: [batch, 1]
    mask = (torch.rand_like(x0) < p).float()
    x_t = x0 * (1 - mask)  # randomly set some 1s to 0
    return x_t, mask

# ========== DENOISING MODEL ==========
class DenoisingMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # More complex architecture with multiple layers
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, input_dim)
        )
        # Skip connection
        self.skip = nn.Linear(input_dim, input_dim)
        
    def forward(self, x):
        # Main path
        main_out = self.layers(x)
        # Skip connection
        skip_out = self.skip(x)
        # Combine and apply sigmoid
        return torch.sigmoid(main_out + skip_out)

# ========== VAE RECOMMENDER LOADING ==========
def load_vae_recommender(checkpoint_path, device=DEVICE):
    # Minimal config for VAE (should match training)
    VAE_config = {
        "enc_dims": [256, 64],
        "dropout": 0.5,
        "anneal_cap": 0.2,
        "total_anneal_steps": 200000
    }
    # Dummy kw_dict for VAE
    kw_dict = {'device': device, 'num_items': USER_HISTORY_DIM}
    model = VAE(VAE_config, **kw_dict).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model

# ========== TRAINING LOOP ==========
def train_denoiser():
    # Load data
    data = load_user_histories(DATA_PATH)
    num_samples = data.shape[0]
    print(f"Loaded {num_samples} user histories of length {data.shape[1]}")

    # Model
    model = DenoisingMLP(USER_HISTORY_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()

    # Load recommender
    recommender = load_vae_recommender(CHECKPOINT_PATH, device=DEVICE)
    # Prepare dummy kw_dict for get_user_recommended_item
    all_items_tensor = torch.eye(USER_HISTORY_DIM, device=DEVICE)
    kw_dict = {
        'device': DEVICE,
        'num_items': USER_HISTORY_DIM,
        'all_items_tensor': all_items_tensor,
        'output_type': 'multiple',
        'recommender_name': 'VAE'
    }

    for epoch in range(EPOCHS):
        perm = np.random.permutation(num_samples)
        total_loss = 0
        total_cf_loss = 0
        total_recon_loss = 0
        total_l1_loss = 0
        total_preserve_loss = 0
        total_changed = 0
        total_count = 0
        for i in range(0, num_samples, BATCH_SIZE):
            batch_idx = perm[i:i+BATCH_SIZE]
            x0 = torch.tensor(data[batch_idx], device=DEVICE)
            
            # Sample random diffusion step for each sample
            t = torch.randint(1, 11, (x0.size(0),), device=DEVICE)
            x_t, noise_mask = bernoulli_diffusion(x0, t)
            
            # Denoising
            x0_hat = model(x_t)
            
            # Improved binarization strategy:
            # 1. For positions that were NOT noised, preserve original values
            # 2. For positions that were noised, use threshold-based binarization
            x0_hat_bin = x0.clone()  # Start with original values
            # Only change positions that were noised
            noised_positions = (noise_mask > 0)
            # Apply threshold only to noised positions
            x0_hat_bin[noised_positions] = (x0_hat[noised_positions] > 0.5).float()
            
            # Get top-1 recommendations
            top1_orig = []
            top1_denoised = []
            for j in range(x0.size(0)):
                top1_orig.append(int(get_user_recommended_item(x0[j], recommender, **kw_dict).cpu().detach().numpy()))
                top1_denoised.append(int(get_user_recommended_item(x0_hat_bin[j], recommender, **kw_dict).cpu().detach().numpy()))
            top1_orig = torch.tensor(top1_orig, device=DEVICE)
            top1_denoised = torch.tensor(top1_denoised, device=DEVICE)
            
            # Counterfactual loss: penalize if top-1 does NOT change
            cf_loss = (top1_orig == top1_denoised).float().mean()
            
            # Masked reconstruction loss: only penalize errors in noised positions
            # noise_mask is 1 where noise was added, 0 where no noise was added
            masked_recon_loss = (criterion(x0_hat, x0) * noise_mask).sum() / (noise_mask.sum() + 1e-8)
            
            # L1 loss (encourage minimal changes) - also masked
            masked_l1_loss = ((x0_hat - x0).abs() * noise_mask).sum() / (noise_mask.sum() + 1e-8)
            
            # NEW: Strong preservation loss for non-noised positions
            # Create inverse mask (1 where no noise was added)
            preserve_mask = 1 - noise_mask
            # Strongly penalize any changes in non-noised positions
            preserve_loss = ((x0_hat - x0).abs() * preserve_mask).sum() / (preserve_mask.sum() + 1e-8)
            
            # Total loss
            loss = masked_recon_loss + LAMBDA_CF * cf_loss + LAMBDA_L1 * masked_l1_loss + LAMBDA_PRESERVE * preserve_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * x0.size(0)
            total_cf_loss += cf_loss.item() * x0.size(0)
            total_recon_loss += masked_recon_loss.item() * x0.size(0)
            total_l1_loss += masked_l1_loss.item() * x0.size(0)
            total_preserve_loss += preserve_loss.item() * x0.size(0)
            total_changed += (top1_orig != top1_denoised).float().sum().item()
            total_count += x0.size(0)

            # Detailed logging for the first batch of each epoch
            if i == 0:
                print(f"\n=== EPOCH {epoch+1} DETAILED LOG ===")
                print(f"Batch size: {x0.size(0)}")
                print(f"Time steps (noise levels): {t.cpu().numpy()}")
                print(f"Max noise probability: {0.2 * 10 / 10:.6f}")
                
                # Analyze first 3 examples in detail
                for k in range(min(3, x0.size(0))):
                    print(f"\n--- Example {k+1} ---")
                    x0_k = x0[k].cpu().numpy()
                    x_t_k = x_t[k].cpu().numpy()
                    x0_hat_k = x0_hat[k].detach().cpu().numpy()
                    x0_hat_bin_k = x0_hat_bin[k].cpu().numpy()
                    noise_mask_k = noise_mask[k].cpu().numpy()
                    
                    # Count 1s in each version
                    ones_orig = np.sum(x0_k)
                    ones_noisy = np.sum(x_t_k)
                    ones_denoised = np.sum(x0_hat_bin_k)
                    ones_noised = np.sum(noise_mask_k)
                    
                    print(f"Original 1s: {ones_orig}, Noisy 1s: {ones_noisy}, Denoised 1s: {ones_denoised}")
                    print(f"Noise level (t): {t[k].item()}, Noise prob: {0.2 * t[k].item() / 10:.6f}")
                    print(f"Positions noised: {ones_noised}")
                    print(f"Items removed by noise: {ones_orig - ones_noisy}")
                    print(f"Items changed by denoising: {np.sum(x0_k != x0_hat_bin_k)}")
                    print(f"Ratio (changed/noised): {np.sum(x0_k != x0_hat_bin_k) / (ones_noised + 1e-8):.2f}")
                    
                    # Show specific changes
                    changed_indices = np.where(x0_k != x0_hat_bin_k)[0]
                    noised_indices = np.where(noise_mask_k > 0)[0]
                    print(f"Noised indices (first 10): {noised_indices[:10]}")
                    print(f"Changed indices (first 10): {changed_indices[:10]}")
                    
                    # Show recommendation changes
                    print(f"Original top-1: {top1_orig[k].item()}, Denoised top-1: {top1_denoised[k].item()}")
                    print(f"Recommendation changed: {top1_orig[k].item() != top1_denoised[k].item()}")
                    
                    # Show some actual values
                    print(f"Original (first 10): {x0_k[:10]}")
                    print(f"Denoised (first 10, rounded): {np.round(x0_hat_k[:10], 3)}")
                    print(f"Binarized (first 10): {x0_hat_bin_k[:10]}")
                
                print(f"\nBatch losses - Masked Recon: {masked_recon_loss.item():.4f}, CF: {cf_loss.item():.4f}, Masked L1: {masked_l1_loss.item():.4f}, Preserve: {preserve_loss.item():.4f}")
                print(f"Batch recommendation change rate: {(top1_orig != top1_denoised).float().mean().item():.2%}")
                print("=" * 50)

            # Print and inspect a few denoised user histories for the first batch of the first epoch
            if epoch == 0 and i == 0:
                print(f"x0_hat mean: {x0_hat.mean().item():.4f}, std: {x0_hat.std().item():.4f}")
                for k in range(min(3, x0.size(0))):  # Print up to 3 examples
                    print(f"\nExample {k+1}:")
                    print("Original (x0):", x0[k].cpu().numpy().astype(int))
                    print("Denoised (x0_hat, rounded):", np.round(x0_hat[k].detach().cpu().numpy(), 2))
                    print("Binarized (x0_hat_bin):", x0_hat_bin[k].cpu().numpy().astype(int))
                    diff_indices = np.where(x0[k].cpu().numpy().astype(int) != x0_hat_bin[k].cpu().numpy().astype(int))[0]
                    print("Changed indices:", diff_indices)
                    print("Num changed:", len(diff_indices))
        avg_loss = total_loss / num_samples
        avg_cf_loss = total_cf_loss / num_samples
        avg_recon_loss = total_recon_loss / num_samples
        avg_l1_loss = total_l1_loss / num_samples
        avg_preserve_loss = total_preserve_loss / num_samples
        change_rate = total_changed / total_count
        print(f"Epoch {epoch+1}/{EPOCHS} - Total Loss: {avg_loss:.4f} | Recon Loss: {avg_recon_loss:.4f} | CF Loss: {avg_cf_loss:.4f} | L1 Loss: {avg_l1_loss:.4f} | Preserve Loss: {avg_preserve_loss:.4f} | Change Rate: {change_rate:.2%}")

# ========== MAIN ==========
if __name__ == "__main__":
    train_denoiser() 