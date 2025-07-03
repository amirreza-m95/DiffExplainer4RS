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

import torch.nn.functional as F
import matplotlib.pyplot as plt

# ========== CONFIGURATION ==========
DATA_PATH = Path('datasets/lxr-CE/ML1M/train_data_ML1M.csv')
CHECKPOINT_PATH = Path('checkpoints/recommenders/VAE_ML1M_0_19_128.pt')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USER_HISTORY_DIM = 3381  # Number of items
BATCH_SIZE = 128
EPOCHS = 200
LEARNING_RATE = 0.01
LAMBDA_CF = 2.9                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 # Weight for counterfactual loss
LAMBDA_L1 = 0.1   # Increased weight for L1 loss (encourage minimal changes)
LAMBDA_PRESERVE = 0.1  # New weight to strongly preserve non-noised positions
max_noise_probability = 0.95

# ========== DATA LOADING ==========
def load_user_histories(data_path):
    df = pd.read_csv(data_path, index_col=0)
    # Remove user_id column if present
    if 'user_id' in df.columns:
        df = df.drop(columns=['user_id'])
    data = df.values.astype(np.float32)
    return data

# ========== DIFFUSION PROCESS ==========
def bernoulli_diffusion(x0, t, max_steps=10, max_noise_probability=0.4):
    """
    x0: [batch, dim] binary user histories
    t: [batch] time steps (noise levels, 0 <= t < max_steps)
    max_noise_probability: maximum probability of noising a position
    Returns: x_t (noisy histories), mask (positions that were noised)
    """
    p = max_noise_probability * (t / max_steps)[:, None]  # shape: [batch, 1]
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
def loss_visualizer(total_losses, recon_losses, cf_losses, l1_losses, preserve_losses, change_rates):
    """
    Plots the loss values and change rate over epochs and saves the figure.
    """
    epochs = range(1, len(total_losses) + 1)
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, total_losses, label='Total Loss')
    plt.plot(epochs, recon_losses, label='Recon Loss')
    plt.plot(epochs, cf_losses, label='CF Loss')
    plt.plot(epochs, l1_losses, label='L1 Loss')
    plt.plot(epochs, preserve_losses, label='Preserve Loss')
    plt.plot(epochs, change_rates, label='Change Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Loss / Change Rate')
    plt.title('Training Losses and Change Rate Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('loss_curves.png')
    plt.close()

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

    best_loss = float('inf')
    best_checkpoint_path = None
    initial_noise_prob = 0.1  # Start small
    final_noise_prob = max_noise_probability  # Use the configured max as the final value
    total_losses = []
    recon_losses = []
    cf_losses = []
    l1_losses = []
    preserve_losses = []
    change_rates = []
    for epoch in range(EPOCHS):
        progress = epoch / (EPOCHS - 1)
        current_max_noise_prob = initial_noise_prob + (final_noise_prob - initial_noise_prob) * progress
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
            t = torch.randint(2, 11, (x0.size(0),), device=DEVICE)
            x_t, noise_mask = bernoulli_diffusion(x0, t, max_steps=10, max_noise_probability=current_max_noise_prob)
            
            # Denoising
            x0_hat = model(x_t)
            
            # Improved binarization strategy:
            # 1. For positions that were NOT noised, preserve original values
            # 2. For positions that were noised, use threshold-based binarization
            x0_hat_bin = x0.clone()  # Start with original values
            # Only change positions that were noised
            noised_positions = (noise_mask > 0)
            # Apply threshold only to noised positions
            x0_hat_bin[noised_positions] = (x0_hat[noised_positions] > 0.59).float()
            
            # Get top-1 recommendations
            top1_orig = []
            top1_denoised = []
            for j in range(x0.size(0)):
                top1_orig.append(int(get_user_recommended_item(x0[j], recommender, **kw_dict).cpu().detach().numpy()))
                top1_denoised.append(int(get_user_recommended_item(x0_hat_bin[j], recommender, **kw_dict).cpu().detach().numpy()))
            top1_orig = torch.tensor(top1_orig, device=DEVICE)
            top1_denoised = torch.tensor(top1_denoised, device=DEVICE)
            
            # Differentiable counterfactual loss (ReLU/margin version)
            margin = 0.1
            with torch.no_grad():
                scores_orig = recommender(x0)  # [batch, num_items]
            scores_denoised = recommender(x0_hat_bin)  # [batch, num_items]
            top1_indices = scores_orig.argmax(dim=1)  # [batch]
            orig_top1_scores = scores_orig.gather(1, top1_indices.unsqueeze(1)).squeeze(1)  # [batch]
            denoised_top1_scores = scores_denoised.gather(1, top1_indices.unsqueeze(1)).squeeze(1)  # [batch]
            # Encourage denoised score to be at least margin lower than original
            cf_loss = F.relu(denoised_top1_scores - orig_top1_scores + margin).mean()
            
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
            
            # Remove normalization: use raw losses directly
            loss = (0.01) * masked_recon_loss + LAMBDA_CF * cf_loss + LAMBDA_L1 * masked_l1_loss + LAMBDA_PRESERVE * preserve_loss
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

            # Detailed logging for just one example in the middle epoch and middle batch
            middle_epoch = EPOCHS // 2
            middle_batch = (num_samples // BATCH_SIZE) // 2
            if epoch == middle_epoch and i == middle_batch * BATCH_SIZE:
                print(f"\n=== MIDDLE EPOCH {epoch+1} MIDDLE BATCH SINGLE EXAMPLE INSPECTION ===")
                k = 0  # Only the first example
                x0_k = x0[k].cpu().numpy()
                x_t_k = x_t[k].cpu().numpy()
                x0_hat_k = x0_hat[k].detach().cpu().numpy()
                x0_hat_bin_k = x0_hat_bin[k].cpu().numpy()
                noise_mask_k = noise_mask[k].cpu().numpy()
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
                changed_indices = np.where(x0_k != x0_hat_bin_k)[0]
                noised_indices = np.where(noise_mask_k > 0)[0]
                print(f"Noised indices (first 10): {noised_indices[:10]}")
                print(f"Changed indices (first 10): {changed_indices[:10]}")
                print(f"Original top-1: {top1_orig[k].item()}, Denoised top-1: {top1_denoised[k].item()}")
                print(f"Recommendation changed: {top1_orig[k].item() != top1_denoised[k].item()}")
                print(f"Original (first 10): {x0_k[:10]}")
                print(f"Denoised (first 10, rounded): {np.round(x0_hat_k[:10], 3)}")
                print(f"Binarized (first 10): {x0_hat_bin_k[:10]}")
        avg_loss = total_loss / num_samples
        avg_cf_loss = total_cf_loss / num_samples
        avg_recon_loss = total_recon_loss / num_samples
        avg_l1_loss = total_l1_loss / num_samples
        avg_preserve_loss = total_preserve_loss / num_samples
        change_rate = total_changed / total_count
        print(f"Epoch {epoch+1}/{EPOCHS} - Total Loss: {avg_loss:.4f} | Recon Loss: {avg_recon_loss:.4f} | CF Loss: {avg_cf_loss:.4f} | L1 Loss: {avg_l1_loss:.4f} | Preserve Loss: {avg_preserve_loss:.4f} | Change Rate: {change_rate:.2%}")
        # Record losses for visualization
        total_losses.append(avg_loss)
        recon_losses.append(avg_recon_loss)
        cf_losses.append(avg_cf_loss)
        l1_losses.append(avg_l1_loss)
        preserve_losses.append(avg_preserve_loss)
        change_rates.append(change_rate)

        # === Save checkpoint only if this is the best model so far ===
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_dir = Path("checkpoints/diffusionModels")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_filename = f"best_denoiser_ML1M_bs{BATCH_SIZE}_lr{LEARNING_RATE}_lcf{LAMBDA_CF}_l1{LAMBDA_L1}_pres{LAMBDA_PRESERVE}.pt"
            checkpoint_path = checkpoint_dir / checkpoint_filename
            torch.save(model.state_dict(), checkpoint_path)
            best_checkpoint_path = checkpoint_path
            print(f"Saved new best checkpoint: {checkpoint_path} (loss={best_loss:.4f})")

    # After training, visualize losses
    loss_visualizer(total_losses, recon_losses, cf_losses, l1_losses, preserve_losses, change_rates)

# ========== MAIN ==========
if __name__ == "__main__":
    train_denoiser() 