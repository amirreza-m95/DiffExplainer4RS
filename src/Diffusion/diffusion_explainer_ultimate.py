import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# === Import VAE and get_user_recommended_item ===
from src.LXR.recommenders_architecture import VAE
from src.LXR.help_functions import get_user_recommended_item

# ========== LOGGING SETUP ==========
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ========== CONFIGURATION ==========
DATA_PATH = Path('datasets/lxr-CE/ML1M/train_data_ML1M.csv')
CHECKPOINT_PATH = Path('checkpoints/recommenders/VAE_ML1M_0_19_128.pt')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USER_HISTORY_DIM = 3381  # Number of items
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 0.005
LAMBDA_CF = 10.0
LAMBDA_PRESERVE = 0.8
LAMBDA_IMPORTANCE = 0.5
max_noise_probability = 0.8

# ========== DATA LOADING ==========
def load_user_histories(data_path):
    df = pd.read_csv(data_path, index_col=0)
    if df is None:
        raise ValueError("Failed to load data as DataFrame from CSV.")
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    if 'user_id' in df.columns:
        df = df.drop(columns=['user_id'])
    data = df.values.astype(np.float32)
    logging.info(f"Loaded user histories from {data_path}, shape: {data.shape}")
    return data

# ========== DIFFUSION PROCESS ==========
def bernoulli_diffusion(original_user_history, t, max_steps=10, max_noise_probability=0.4):
    """
    Improved Bernoulli diffusion that only applies noise to interaction positions (1s)
    original_user_history: [batch, dim] binary user histories
    t: [batch] time steps (noise levels, 0 <= t < max_steps)
    max_noise_probability: maximum probability of noising a position
    Returns: noised_user_history (noisy histories), mask (positions that were noised)
    """
    p = max_noise_probability * (t / max_steps)[:, None]  # shape: [batch, 1]
    
    # Create mask only for positions that are 1 (interactions)
    interaction_mask = (original_user_history == 1).float()  # 1 where user has interactions, 0 elsewhere
    
    # Apply noise only to interaction positions
    noise_mask = (torch.rand_like(original_user_history) < p).float() * interaction_mask
    
    noised_user_history = original_user_history * (1 - noise_mask)  # Remove interactions based on noise mask
    return noised_user_history, noise_mask

# ========== ATTENTION-BASED DENOISING MODEL ==========
class AttentionDenoisingMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, input_dim)
        )
        self.skip = nn.Linear(input_dim, input_dim)
    def forward(self, x):
        attention_weights = self.attention(x)
        attended_input = x * attention_weights
        main_out = self.layers(attended_input)
        skip_out = self.skip(x)
        output = torch.sigmoid(main_out + skip_out)
        return output, attention_weights

# ========== VAE RECOMMENDER LOADING ==========
def load_vae_recommender(checkpoint_path, device=DEVICE):
    VAE_config = {
        "enc_dims": [256, 64],
        "dropout": 0.5,
        "anneal_cap": 0.2,
        "total_anneal_steps": 200000
    }
    kw_dict = {'device': device, 'num_items': USER_HISTORY_DIM}
    model = VAE(VAE_config, **kw_dict).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    logging.info(f"Loaded VAE recommender from {checkpoint_path}")
    return model

# ========== MINIMAL CHANGE LOSS ==========
def compute_minimal_change_loss(model_predictions, original_user_history, noise_mask):
    """
    Encourage at least 10% changes but penalize excessive changes
    - Reward: At least 10% of interactions should be changed
    - Penalty: If more than 30% are changed (excessive)
    """
    noised_positions = (noise_mask > 0)
    original_ones = original_user_history[noised_positions]
    predicted_values = model_predictions[noised_positions]
    
    # Count how many 1s the model is removing
    removed_ones = torch.sum((original_ones == 1) & (predicted_values < 0.5))
    total_ones = torch.sum(original_ones == 1)
    
    # Calculate removal ratio
    removal_ratio = removed_ones / (total_ones + 1e-8)
    
    # Target: at least 10% changes
    min_target = 0.07  # 10%
    max_target = 0.18  # 30% - beyond this gets penalized
    
    # Loss components:
    # 1. Penalty if less than 10% changed (encourage minimum changes)
    min_change_penalty = F.relu(min_target - removal_ratio)
    
    # 2. Penalty if more than 30% changed (discourage excessive changes)
    excessive_change_penalty = F.relu(removal_ratio - max_target)
    
    # Combine penalties
    minimal_change_loss = min_change_penalty + excessive_change_penalty
    
    return minimal_change_loss

# ========== SPARSITY LOSS ==========
def compute_sparsity_loss(x0_hat, x0, noise_mask):
    noised_positions = (noise_mask > 0)
    original_ones = x0[noised_positions]
    predicted_values = x0_hat[noised_positions]
    sparsity_loss = F.binary_cross_entropy(predicted_values, original_ones, reduction='none')
    weight = torch.where(original_ones > 0, 2.0, 1.0)
    weighted_loss = sparsity_loss * weight
    return weighted_loss.mean()

# ========== GRADIENT-BASED IMPORTANCE LOSS ==========
def compute_gradient_importance_loss(model_predictions, recommender, top1_indices, original_user_history):
    """
    Compute gradient-based importance loss to find the most critical interactions
    """
    batch_size = model_predictions.size(0)
    importance_loss = 0.0
    
    for user_idx in range(batch_size):
        # Get user's denoised output and make it a leaf tensor
        user_hat = model_predictions[user_idx:user_idx+1].detach().clone()  # Create new leaf tensor
        user_hat.requires_grad_(True)
        user_hat.retain_grad()  # Ensure gradients are retained
        
        # Get original user data
        user_orig = original_user_history[user_idx:user_idx+1]
        
        # Get recommendation scores
        scores = recommender(user_hat)
        top1_idx = top1_indices[user_idx]
        top1_score = scores[0, top1_idx]
        
        # Compute gradient w.r.t. input
        top1_score.backward(retain_graph=True)
        
        # Get gradient importance
        gradient_importance = user_hat.grad.abs().squeeze(0)
        
        # Normalize gradient importance
        gradient_importance = gradient_importance / (gradient_importance.sum() + 1e-8)
        
        # Create importance target: we want to find interactions that cause big changes
        # The more important an interaction, the higher its gradient should be
        importance_target = gradient_importance
        
        # Compute importance loss: encourage model to focus on high-gradient interactions
        # We want the model output to correlate with gradient importance
        model_importance = model_predictions[user_idx].detach()  # Use original model output
        importance_loss += F.mse_loss(model_importance, importance_target)
        
        # Reset gradients for next iteration
        user_hat.grad.zero_()
    
    return importance_loss / batch_size

# ========== COUNTERFACTUAL LOSS ==========
def compute_counterfactual_loss(original_user_history, denoised_user_history_soft, recommender, top1_indices, orig_top1_scores):
    batch_size = original_user_history.size(0)
    cf_loss = 0.0
    scores_denoised = recommender(denoised_user_history_soft)
    for user_idx in range(batch_size):
        orig_top1 = top1_indices[user_idx]
        orig_score = orig_top1_scores[user_idx]
        denoised_scores = scores_denoised[user_idx]
        direct_loss = denoised_scores[orig_top1]
        mask = torch.ones_like(denoised_scores, dtype=torch.bool)
        mask[orig_top1] = False
        other_scores = denoised_scores[mask]
        if len(other_scores) > 0:
            best_other_score = other_scores.max()
            margin = 0.1
            ranking_loss = F.relu(denoised_scores[orig_top1] - best_other_score + margin)
            cf_loss += direct_loss + ranking_loss
        else:
            cf_loss += direct_loss
    return cf_loss / batch_size

# ========== LOSS VISUALIZATION ==========
def loss_visualizer(total_losses, cf_losses, minimal_change_losses, importance_losses, preserve_losses, change_rates):
    epochs = range(1, len(total_losses) + 1)
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.plot(epochs, total_losses, label='Total Loss', linewidth=2)
    plt.plot(epochs, cf_losses, label='CF Loss')
    plt.plot(epochs, minimal_change_losses, label='Minimal Change Loss')
    plt.plot(epochs, importance_losses, label='Importance Loss')
    plt.plot(epochs, preserve_losses, label='Preserve Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses Over Epochs (Ultimate)')
    plt.legend()
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(epochs, change_rates, label='Change Rate', color='red', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Change Rate')
    plt.title('Recommendation Change Rate Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('loss_curves_ultimate.png')
    plt.close()

# ========== TRAINING LOOP ==========
def train_denoiser():
    data = load_user_histories(DATA_PATH)
    num_samples = data.shape[0]
    logging.info(f"Training on {num_samples} user histories of length {data.shape[1]}")
    model = AttentionDenoisingMLP(USER_HISTORY_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()
    recommender = load_vae_recommender(CHECKPOINT_PATH, device=DEVICE)
    all_items_tensor = torch.eye(USER_HISTORY_DIM, device=DEVICE)
    kw_dict = {
        'device': DEVICE,
        'num_items': USER_HISTORY_DIM,
        'all_items_tensor': all_items_tensor,
        'output_type': 'multiple',
        'recommender_name': 'VAE'
    }
    initial_noise_prob = 0.1
    final_noise_prob = max_noise_probability
    total_losses, cf_losses = [], []
    minimal_change_losses, importance_losses, preserve_losses, change_rates = [], [], [], []
    save_dir = 'checkpoints/diffusionModels'
    best_model_path = os.path.join(save_dir, "best_model.pt")
    os.makedirs(save_dir, exist_ok=True)
    best_loss = float('inf')
    for epoch in range(EPOCHS):
        progress = epoch / (EPOCHS - 1)
        current_max_noise_prob = initial_noise_prob + (final_noise_prob - initial_noise_prob) * progress
        perm = np.random.permutation(num_samples)
        total_loss = total_cf_loss = 0
        total_minimal_change_loss = total_importance_loss = 0
        total_preserve_loss = 0
        total_changed = total_count = 0
        for i in range(0, num_samples, BATCH_SIZE):
            batch_idx = perm[i:i+BATCH_SIZE]
            original_user_history = torch.tensor(data[batch_idx], device=DEVICE)
            t = torch.randint(2, 11, (original_user_history.size(0),), device=DEVICE)
            noised_user_history, noise_mask = bernoulli_diffusion(original_user_history, t, max_steps=10, max_noise_probability=current_max_noise_prob)
            model_predictions, attention_weights = model(noised_user_history)
            # Soft binarization for differentiable CF loss
            denoised_user_history_soft = original_user_history.clone()
            noised_positions = (noise_mask > 0)
            temp = 0.1
            threshold = 0.51
            soft_bin = torch.sigmoid((model_predictions[noised_positions] - threshold) / temp)
            denoised_user_history_soft[noised_positions] = soft_bin
            # Get original recommendations
            with torch.no_grad():
                scores_orig = recommender(original_user_history)
                top1_indices = scores_orig.argmax(dim=1)
                orig_top1_scores = scores_orig.gather(1, top1_indices.unsqueeze(1)).squeeze(1)
            # Counterfactual loss
            cf_loss = compute_counterfactual_loss(original_user_history, denoised_user_history_soft, recommender, top1_indices, orig_top1_scores)
            # Minimal change loss
            minimal_change_loss = compute_minimal_change_loss(model_predictions, original_user_history, noise_mask)
            # Gradient-based importance loss
            importance_loss = compute_gradient_importance_loss(model_predictions, recommender, top1_indices, original_user_history)
            # Preservation loss
            preserve_mask = 1 - noise_mask
            preserve_loss = ((model_predictions - original_user_history).abs() * preserve_mask).sum() / (preserve_mask.sum() + 1e-8)
            # Hard binarization for evaluation
            denoised_user_history_binary = original_user_history.clone()
            denoised_user_history_binary[noised_positions] = (model_predictions[noised_positions] > threshold).float()
            # Get top-1 recommendations for evaluation
            top1_orig = []
            top1_denoised = []
            for j in range(original_user_history.size(0)):
                top1_orig.append(int(get_user_recommended_item(original_user_history[j], recommender, **kw_dict).cpu().detach().numpy()))
                top1_denoised.append(int(get_user_recommended_item(denoised_user_history_binary[j], recommender, **kw_dict).cpu().detach().numpy()))
            top1_orig = torch.tensor(top1_orig, device=DEVICE)
            top1_denoised = torch.tensor(top1_denoised, device=DEVICE)
            # Total loss
            loss = (
                LAMBDA_CF * cf_loss
                + LAMBDA_PRESERVE * preserve_loss
                + LAMBDA_IMPORTANCE * importance_loss
                # + minimal_change_loss  # No lambda needed, it's already scaled
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * original_user_history.size(0)
            total_cf_loss += cf_loss.item() * original_user_history.size(0) if hasattr(cf_loss, 'item') else cf_loss * original_user_history.size(0)
            total_minimal_change_loss += minimal_change_loss.item() * original_user_history.size(0) if hasattr(minimal_change_loss, 'item') else minimal_change_loss * original_user_history.size(0)
            total_importance_loss += importance_loss.item() * original_user_history.size(0) if hasattr(importance_loss, 'item') else importance_loss * original_user_history.size(0)
            total_preserve_loss += preserve_loss.item() * original_user_history.size(0)
            total_changed += (top1_orig != top1_denoised).float().sum().item()
            total_count += original_user_history.size(0)
            if i == 0 or i == (num_samples // (2 * BATCH_SIZE)) * BATCH_SIZE:
                logging.info(f"Epoch {epoch+1}, Batch {i//BATCH_SIZE+1}: Loss={loss.item():.4f}, CF={cf_loss:.4f}, MinChange={minimal_change_loss:.4f}, Importance={importance_loss:.4f}")
            # === DETAILED SAMPLE INSPECTION FOR BATCH 19 ===
            if (i // BATCH_SIZE + 1) == 19:
                sample_idx = 0  # Inspect the first sample in the batch
                original_user_history_sample = original_user_history[sample_idx].cpu().numpy()
                noised_user_history_sample = noised_user_history[sample_idx].cpu().numpy()
                denoised_user_history_binary_sample = denoised_user_history_binary[sample_idx].cpu().numpy()
                noise_mask_sample = noise_mask[sample_idx].cpu().numpy()
                # Only consider the number of interactions (1s in the original user history)
                num_interactions = int(np.sum(original_user_history_sample == 1))
                # Number of changes from noise (original 1s set to 0)
                noise_changes = int(np.sum((original_user_history_sample == 1) & (noised_user_history_sample == 0)))
                noise_changes_pct = 100.0 * noise_changes / num_interactions if num_interactions > 0 else 0.0
                # Number of changes after denoising (positions where original_user_history_sample != denoised_user_history_binary_sample and original_user_history_sample == 1)
                denoise_changes = int(np.sum((original_user_history_sample == 1) & (original_user_history_sample != denoised_user_history_binary_sample)))
                denoise_changes_pct = 100.0 * denoise_changes / num_interactions if num_interactions > 0 else 0.0
                logging.info(f"[Sample Inspection] Epoch {epoch+1}, Batch 19, Sample 0:")
                logging.info(f"  - Noise changes: {noise_changes} ({noise_changes_pct:.2f}% of interactions)")
                logging.info(f"  - Denoising changes: {denoise_changes} ({denoise_changes_pct:.2f}% of interactions)")
        avg_loss = total_loss / total_count
        total_losses.append(avg_loss)
        cf_losses.append(total_cf_loss / total_count)
        minimal_change_losses.append(total_minimal_change_loss / total_count)
        importance_losses.append(total_importance_loss / total_count)
        preserve_losses.append(total_preserve_loss / total_count)
        change_rates.append(total_changed / total_count)
        logging.info(f"Epoch {epoch+1} summary: Total Loss={total_losses[-1]:.4f}, Change Rate={change_rates[-1]:.4f}")
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"Best model saved at {best_model_path} with loss {best_loss:.4f}")
    loss_visualizer(total_losses, cf_losses, minimal_change_losses, importance_losses, preserve_losses, change_rates)
    logging.info("Training complete. Loss curves saved.")
    if best_model_path:
        logging.info(f"Best model overall saved at {best_model_path} with loss {best_loss:.4f}")

if __name__ == "__main__":
    train_denoiser() 