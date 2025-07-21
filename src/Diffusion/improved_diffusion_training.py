#!/usr/bin/env python3
"""
Improved diffusion training approach that explicitly learns importance scores.
This approach modifies the training objective to include importance learning.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import matplotlib.pyplot as plt
import torch.nn.functional as F

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.LXR.recommenders_architecture import VAE
from src.LXR.help_functions import get_user_recommended_item

# ========== CONFIGURATION ==========
DATA_PATH = Path('datasets/lxr-CE/ML1M/train_data_ML1M.csv')
CHECKPOINT_PATH = Path('checkpoints/recommenders/VAE_ML1M_0_19_128.pt')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USER_HISTORY_DIM = 3381
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 0.001

# Loss weights
LAMBDA_RECONSTRUCTION = 1.0
LAMBDA_COUNTERFACTUAL = 5.0
LAMBDA_IMPORTANCE = 2.0
LAMBDA_SPARSITY = 0.1

class ImprovedDiffusionMLP(nn.Module):
    """
    Improved diffusion model that explicitly learns importance scores
    """
    def __init__(self, input_dim, hidden_dim=512, num_layers=3):
        super().__init__()
        
        # Main denoising network
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.1))
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
        
        layers.append(nn.Linear(hidden_dim, input_dim))
        self.denoising_network = nn.Sequential(*layers)
        
        # Importance prediction head
        self.importance_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, input_dim),
            nn.Sigmoid()
        )
        
        # Skip connection
        self.skip = nn.Linear(input_dim, input_dim)
    
    def forward(self, x):
        # Denoising output
        denoised = self.denoising_network(x)
        skip_out = self.skip(x)
        denoised_output = torch.sigmoid(denoised + skip_out)
        
        # Importance output
        importance_scores = self.importance_head(x)
        
        return denoised_output, importance_scores

def load_data():
    """Load training data"""
    df = pd.read_csv(DATA_PATH, index_col=0)
    if 'user_id' in df.columns:
        df = df.drop(columns=['user_id'])
    data = df.values.astype(np.float32)
    return data

def load_recommender():
    """Load the pre-trained recommender"""
    recommender = VAE(USER_HISTORY_DIM, 128, 64, 0.19, DEVICE)
    recommender.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    recommender.eval()
    return recommender

def bernoulli_diffusion(x0, t, max_steps=10, max_noise_probability=0.4):
    """
    Bernoulli diffusion process
    """
    p = max_noise_probability * (t / max_steps)[:, None]
    mask = (torch.rand_like(x0) < p).float()
    x_t = x0 * (1 - mask)
    return x_t, mask

def compute_gradient_importance(user_tensor, recommender, item_id):
    """
    Compute gradient-based importance scores for supervision
    """
    user_tensor.requires_grad_(True)
    
    scores = recommender(user_tensor.unsqueeze(0))
    target_score = scores[0, item_id]
    
    target_score.backward()
    gradient_importance = user_tensor.grad.abs()
    
    # Normalize
    max_grad = gradient_importance.max()
    if max_grad > 0:
        gradient_importance = gradient_importance / max_grad
    
    return gradient_importance.detach()

def compute_counterfactual_loss(original_history, denoised_history, recommender, top1_indices, orig_top1_scores):
    """
    Compute counterfactual loss
    """
    batch_size = original_history.size(0)
    cf_loss = 0.0
    
    scores_denoised = recommender(denoised_history)
    
    for user_idx in range(batch_size):
        orig_top1 = top1_indices[user_idx]
        denoised_scores = scores_denoised[user_idx]
        
        # Directly minimize original top-1 score
        direct_loss = denoised_scores[orig_top1]
        
        # Encourage ranking change
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

def compute_importance_loss(predicted_importance, true_importance, user_mask):
    """
    Compute importance learning loss
    """
    masked_pred = predicted_importance * user_mask
    masked_true = true_importance * user_mask
    
    # MSE loss for importance prediction
    importance_loss = F.mse_loss(masked_pred, masked_true)
    
    return importance_loss

def compute_sparsity_loss(importance_scores, user_mask):
    """
    Encourage sparsity in importance scores
    """
    masked_scores = importance_scores * user_mask
    sparsity_loss = torch.mean(masked_scores)
    return sparsity_loss

def train_improved_diffusion():
    """
    Train the improved diffusion model
    """
    print("=== Training Improved Diffusion Model ===")
    
    # Load data and recommender
    data = load_data()
    recommender = load_recommender()
    
    # Initialize model
    model = ImprovedDiffusionMLP(USER_HISTORY_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training metrics
    total_losses = []
    recon_losses = []
    cf_losses = []
    importance_losses = []
    sparsity_losses = []
    
    print(f"Training on {data.shape[0]} samples...")
    
    for epoch in range(EPOCHS):
        epoch_recon_loss = 0
        epoch_cf_loss = 0
        epoch_importance_loss = 0
        epoch_sparsity_loss = 0
        epoch_total_loss = 0
        num_batches = 0
        
        # Shuffle data
        perm = np.random.permutation(data.shape[0])
        
        for i in range(0, data.shape[0], BATCH_SIZE):
            batch_indices = perm[i:i+BATCH_SIZE]
            x0 = torch.tensor(data[batch_indices], device=DEVICE)
            
            # Sample random diffusion step
            t = torch.randint(2, 11, (x0.size(0),), device=DEVICE)
            x_t, noise_mask = bernoulli_diffusion(x0, t, max_steps=10, max_noise_probability=0.4)
            
            # Forward pass
            denoised_output, importance_scores = model(x_t)
            
            # Get original recommendations
            with torch.no_grad():
                scores_orig = recommender(x0)
                top1_indices = scores_orig.argmax(dim=1)
                orig_top1_scores = scores_orig.gather(1, top1_indices.unsqueeze(1)).squeeze(1)
            
            # Soft binarization for differentiable CF loss
            denoised_soft = x0.clone()
            noised_positions = (noise_mask > 0)
            temp = 0.1
            threshold = 0.5
            soft_bin = torch.sigmoid((denoised_output[noised_positions] - threshold) / temp)
            denoised_soft[noised_positions] = soft_bin
            
            # Compute losses
            # 1. Reconstruction loss (masked)
            recon_loss = (F.mse_loss(denoised_output, x0) * noise_mask).sum() / (noise_mask.sum() + 1e-8)
            
            # 2. Counterfactual loss
            cf_loss = compute_counterfactual_loss(x0, denoised_soft, recommender, top1_indices, orig_top1_scores)
            
            # 3. Importance loss
            importance_loss = 0
            for user_idx in range(x0.size(0)):
                user_tensor = x0[user_idx]
                item_id = top1_indices[user_idx]
                
                # Compute true importance
                true_importance = compute_gradient_importance(user_tensor, recommender, item_id)
                
                # Compute importance loss for this user
                user_mask = (user_tensor > 0).float()
                user_importance_loss = compute_importance_loss(
                    importance_scores[user_idx], true_importance, user_mask
                )
                importance_loss += user_importance_loss
            
            importance_loss = importance_loss / x0.size(0)
            
            # 4. Sparsity loss
            sparsity_loss = 0
            for user_idx in range(x0.size(0)):
                user_mask = (x0[user_idx] > 0).float()
                user_sparsity_loss = compute_sparsity_loss(importance_scores[user_idx], user_mask)
                sparsity_loss += user_sparsity_loss
            
            sparsity_loss = sparsity_loss / x0.size(0)
            
            # Total loss
            total_loss = (
                LAMBDA_RECONSTRUCTION * recon_loss +
                LAMBDA_COUNTERFACTUAL * cf_loss +
                LAMBDA_IMPORTANCE * importance_loss +
                LAMBDA_SPARSITY * sparsity_loss
            )
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Record losses
            epoch_recon_loss += recon_loss.item()
            epoch_cf_loss += cf_loss.item()
            epoch_importance_loss += importance_loss.item()
            epoch_sparsity_loss += sparsity_loss.item()
            epoch_total_loss += total_loss.item()
            num_batches += 1
        
        # Average losses
        avg_recon_loss = epoch_recon_loss / num_batches
        avg_cf_loss = epoch_cf_loss / num_batches
        avg_importance_loss = epoch_importance_loss / num_batches
        avg_sparsity_loss = epoch_sparsity_loss / num_batches
        avg_total_loss = epoch_total_loss / num_batches
        
        # Record for plotting
        total_losses.append(avg_total_loss)
        recon_losses.append(avg_recon_loss)
        cf_losses.append(avg_cf_loss)
        importance_losses.append(avg_importance_loss)
        sparsity_losses.append(avg_sparsity_loss)
        
        if epoch % 3 == 0:
            print(f"Epoch {epoch}/{EPOCHS}")
            print(f"  Total Loss: {avg_total_loss:.6f}")
            print(f"  Recon Loss: {avg_recon_loss:.6f}")
            print(f"  CF Loss: {avg_cf_loss:.6f}")
            print(f"  Importance Loss: {avg_importance_loss:.6f}")
            print(f"  Sparsity Loss: {avg_sparsity_loss:.6f}")
    
    # Plot training curves
    plot_training_curves(total_losses, recon_losses, cf_losses, importance_losses, sparsity_losses)
    
    return model

def plot_training_curves(total_losses, recon_losses, cf_losses, importance_losses, sparsity_losses):
    """Plot training loss curves"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    epochs = range(len(total_losses))
    
    axes[0, 0].plot(epochs, total_losses)
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    
    axes[0, 1].plot(epochs, recon_losses)
    axes[0, 1].set_title('Reconstruction Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    
    axes[0, 2].plot(epochs, cf_losses)
    axes[0, 2].set_title('Counterfactual Loss')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Loss')
    
    axes[1, 0].plot(epochs, importance_losses)
    axes[1, 0].set_title('Importance Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    
    axes[1, 1].plot(epochs, sparsity_losses)
    axes[1, 1].set_title('Sparsity Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    
    # Combined plot
    axes[1, 2].plot(epochs, total_losses, label='Total', linewidth=2)
    axes[1, 2].plot(epochs, recon_losses, label='Recon', alpha=0.7)
    axes[1, 2].plot(epochs, cf_losses, label='CF', alpha=0.7)
    axes[1, 2].plot(epochs, importance_losses, label='Importance', alpha=0.7)
    axes[1, 2].set_title('All Losses')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Loss')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig('improved_diffusion_training.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_improved_model(model, test_data, recommender):
    """
    Evaluate the improved diffusion model
    """
    print("Evaluating improved diffusion model...")
    
    model.eval()
    correlations = []
    
    with torch.no_grad():
        for i in range(min(100, len(test_data))):
            user_vector = test_data[i]
            user_tensor = torch.FloatTensor(user_vector).to(DEVICE)
            
            # Get recommended item
            item_id = int(get_user_recommended_item(user_tensor, recommender).cpu().detach().numpy())
            
            # Get model outputs
            denoised_output, importance_scores = model(user_tensor)
            
            # Compute true importance
            true_importance = compute_gradient_importance(user_tensor, recommender, item_id)
            
            # Compute correlation
            user_mask = (user_tensor > 0).float()
            masked_pred = (importance_scores * user_mask).cpu().numpy()
            masked_true = (true_importance * user_mask).cpu().numpy()
            
            # Get non-zero elements
            non_zero_mask = masked_true > 0
            if np.sum(non_zero_mask) > 0:
                pred_values = masked_pred[non_zero_mask]
                true_values = masked_true[non_zero_mask]
                
                correlation = np.corrcoef(pred_values, true_values)[0, 1]
                if not np.isnan(correlation):
                    correlations.append(correlation)
    
    avg_correlation = np.mean(correlations)
    print(f"Average correlation with true importance: {avg_correlation:.3f}")
    
    return avg_correlation

def save_improved_model(model, save_path):
    """Save the trained improved diffusion model"""
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def main():
    """
    Main training function
    """
    print("=== Improved Diffusion Training ===")
    
    # Load data and recommender
    data = load_data()
    recommender = load_recommender()
    
    # Train model
    model = train_improved_diffusion()
    
    # Evaluate model
    correlation = evaluate_improved_model(model, data, recommender)
    
    # Save model
    save_path = Path('checkpoints/diffusion/improved_diffusion_model.pt')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_improved_model(model, save_path)
    
    print("=== Training Complete ===")
    print(f"Final correlation: {correlation:.3f}")
    
    return model, correlation

if __name__ == "__main__":
    main() 