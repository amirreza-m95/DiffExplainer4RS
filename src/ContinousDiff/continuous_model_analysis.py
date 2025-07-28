#!/usr/bin/env python3
"""
Analysis of continuous diffusion model outputs to understand what the values actually mean.
This file investigates whether the model outputs represent importance, reconstruction confidence, 
or something else entirely for embedding-based diffusion models.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
import argparse

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'LXR')))

from recommenders_architecture import VAE
from diffusion_model import DiffusionMLP

# ========== CONFIGURATION ==========
DATA_PATH = Path('datasets/lxr-CE/ML1M/train_data_ML1M.csv')
VAE_CHECKPOINT_PATH = Path('checkpoints/recommenders/VAE_ML1M_0_19_128.pt')
DIFFUSION_MODEL_PATH = Path('/home/amir.reza/counterfactual/DiffExplainer4RS/diffusion_mlp_best.pth')
EMBEDDING_PATH = Path('/home/amir.reza/counterfactual/DiffExplainer4RS/user_embeddings.npy')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USER_HISTORY_DIM = 3381
EMBEDDING_DIM = 64
TIMESTEPS = 1000

# VAE config (should match training)
VAE_config = {
    "enc_dims": [256, 64],
    "dropout": 0.5,
    "anneal_cap": 0.2,
    "total_anneal_steps": 200000
}

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze outputs of continuous diffusion models.")
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to the diffusion model checkpoint. If not set, uses default.')
    parser.add_argument('--embedding-path', type=str, default=None,
                        help='Path to user embeddings. If not set, uses default.')
    return parser.parse_args()

def load_data():
    """Load training data and user embeddings"""
    # Load user profiles
    df = pd.read_csv(DATA_PATH, index_col=0)
    if 'user_id' in df.columns:
        df = df.drop(columns=['user_id'])
    user_profiles = df.values.astype(np.float32)
    
    # Load user embeddings
    embedding_path = args.embedding_path or EMBEDDING_PATH
    if embedding_path.exists():
        embeddings = np.load(embedding_path)
        embeddings = torch.tensor(embeddings, dtype=torch.float32, device=DEVICE)
    else:
        print(f"Warning: Embedding file {embedding_path} not found. Will generate embeddings.")
        embeddings = None
    
    return user_profiles, embeddings

def load_models(model_path=None):
    """Load the VAE recommender and diffusion model"""
    # Load VAE
    vae = VAE(VAE_config, device=DEVICE, num_items=USER_HISTORY_DIM).to(DEVICE)
    vae.load_state_dict(torch.load(VAE_CHECKPOINT_PATH, map_location=DEVICE))
    vae.eval()
    
    # Load diffusion model
    diffusion_path = model_path or DIFFUSION_MODEL_PATH
    diffusion = DiffusionMLP(EMBEDDING_DIM).to(DEVICE)
    diffusion.load_state_dict(torch.load(diffusion_path, map_location=DEVICE))
    diffusion.eval()
    
    return vae, diffusion

def get_embedding(user_tensor, vae):
    """Get user embedding from VAE encoder"""
    h = torch.nn.functional.normalize(user_tensor, dim=-1)
    h = torch.nn.functional.dropout(h, p=vae.dropout, training=False)
    for layer in vae.encoder:
        h = layer(h)
    mu_q = h[:, :vae.enc_dims[-1]]
    return mu_q

def decode_embedding(embedding, vae):
    """Get item scores from embedding using VAE decoder"""
    h = embedding.unsqueeze(0)
    for layer in vae.decoder:
        h = layer(h)
    return h.squeeze(0)

def compute_gradient_importance_embedding(embedding, vae, item_id):
    """
    Compute gradient-based importance scores for embedding dimensions
    """
    embedding_copy = embedding.clone().detach().requires_grad_(True)
    
    # Get recommendation scores
    scores = decode_embedding(embedding_copy, vae)
    target_score = scores[item_id]
    
    # Compute gradients
    target_score.backward()
    gradient_importance = embedding_copy.grad.abs()
    
    return gradient_importance.detach()

def compute_removal_importance_embedding(embedding, vae, item_id):
    """
    Compute importance by measuring how much removing each embedding dimension affects the recommendation
    """
    original_score = decode_embedding(embedding, vae)[item_id].item()
    
    removal_effects = []
    
    for dim_idx in range(embedding.shape[0]):
        # Create modified embedding with dimension zeroed
        modified_embedding = embedding.clone().detach()
        modified_embedding[dim_idx] = 0
        
        # Get new score
        new_score = decode_embedding(modified_embedding, vae)[item_id].item()
        
        # Importance = how much score changes
        importance = abs(original_score - new_score)
        removal_effects.append((dim_idx, importance))
    
    return dict(removal_effects)

def compute_diffusion_noise_prediction(embedding, diffusion, timestep):
    """
    Get the noise prediction from the diffusion model
    """
    with torch.no_grad():
        noise_pred = diffusion(embedding.unsqueeze(0), torch.tensor([timestep], device=DEVICE))
    return noise_pred.squeeze(0)

def analyze_continuous_model_outputs(model_path=None, embedding_path=None):
    """
    Main analysis function to understand what the continuous diffusion model outputs represent
    """
    print(f"=== Analyzing Continuous Diffusion Model Outputs ===")
    
    # Load data and models
    user_profiles, embeddings = load_data()
    vae, diffusion = load_models(model_path)
    
    # Generate embeddings if not provided
    if embeddings is None:
        print("Generating embeddings from user profiles...")
        embeddings = []
        for i in range(min(100, user_profiles.shape[0])):
            user_tensor = torch.FloatTensor(user_profiles[i]).unsqueeze(0).to(DEVICE)
            embedding = get_embedding(user_tensor, vae).squeeze(0)
            embeddings.append(embedding)
        embeddings = torch.stack(embeddings)
    
    # Sample users for analysis
    num_users = min(100, embeddings.shape[0])
    user_indices = np.random.choice(embeddings.shape[0], num_users, replace=False)
    
    results = {
        'noise_predictions': [],
        'gradient_importance': [],
        'removal_importance': [],
        'embedding_magnitudes': [],
        'recommendation_scores': [],
        'user_history_size': []
    }
    
    print(f"Analyzing {num_users} users...")
    
    for i, user_idx in enumerate(user_indices):
        if i % 10 == 0:
            print(f"Processing user {i}/{num_users}")
        
        # Get user embedding
        embedding = embeddings[user_idx]
        
        # Get user profile for history size
        user_profile = user_profiles[user_idx]
        history_size = np.sum(user_profile > 0)
        
        # Get recommendation scores
        rec_scores = decode_embedding(embedding, vae).detach().cpu().numpy()
        top_item = np.argmax(rec_scores)
        
        # Get noise predictions at different timesteps
        timesteps_to_test = [0, 100, 500, 999]  # Different noise levels
        for t in timesteps_to_test:
            noise_pred = compute_diffusion_noise_prediction(embedding, diffusion, t)
            
            # Get gradient importance
            gradient_imp = compute_gradient_importance_embedding(embedding, vae, top_item)
            
            # Get removal importance
            removal_imp = compute_removal_importance_embedding(embedding, vae, top_item)
            
            # Store results for each dimension
            for dim_idx in range(embedding.shape[0]):
                results['noise_predictions'].append(noise_pred[dim_idx].item())
                results['gradient_importance'].append(gradient_imp[dim_idx].item())
                results['removal_importance'].append(removal_imp.get(dim_idx, 0.0))
                results['embedding_magnitudes'].append(abs(embedding[dim_idx].item()))
                results['recommendation_scores'].append(rec_scores[top_item])
                results['user_history_size'].append(history_size)
    
    # Convert to numpy arrays
    for key in results:
        results[key] = np.array(results[key])
    
    return results

def compute_correlations_continuous(results):
    """
    Compute correlations between diffusion model outputs and various metrics
    """
    print("\n=== Correlation Analysis for Continuous Diffusion ===")
    
    correlations = {}
    
    # Noise predictions vs gradient importance
    corr, p_value = pearsonr(results['noise_predictions'], results['gradient_importance'])
    correlations['noise_vs_gradient'] = (corr, p_value)
    print(f"Noise predictions vs Gradient importance: r={corr:.3f}, p={p_value:.3f}")
    
    # Noise predictions vs removal importance
    corr, p_value = pearsonr(results['noise_predictions'], results['removal_importance'])
    correlations['noise_vs_removal'] = (corr, p_value)
    print(f"Noise predictions vs Removal importance: r={corr:.3f}, p={p_value:.3f}")
    
    # Noise predictions vs embedding magnitudes
    corr, p_value = pearsonr(results['noise_predictions'], results['embedding_magnitudes'])
    correlations['noise_vs_magnitudes'] = (corr, p_value)
    print(f"Noise predictions vs Embedding magnitudes: r={corr:.3f}, p={p_value:.3f}")
    
    # Noise predictions vs recommendation scores
    corr, p_value = pearsonr(results['noise_predictions'], results['recommendation_scores'])
    correlations['noise_vs_rec_scores'] = (corr, p_value)
    print(f"Noise predictions vs Recommendation scores: r={corr:.3f}, p={p_value:.3f}")
    
    # Gradient vs removal importance
    corr, p_value = pearsonr(results['gradient_importance'], results['removal_importance'])
    correlations['gradient_vs_removal'] = (corr, p_value)
    print(f"Gradient vs Removal importance: r={corr:.3f}, p={p_value:.3f}")
    
    return correlations

def plot_continuous_analysis(results, correlations):
    """
    Create visualizations to understand continuous diffusion model outputs
    """
    print("\n=== Creating Visualizations for Continuous Diffusion ===")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Noise predictions vs gradient importance
    axes[0, 0].scatter(results['noise_predictions'], results['gradient_importance'], alpha=0.6)
    axes[0, 0].set_xlabel('Noise Predictions')
    axes[0, 0].set_ylabel('Gradient Importance')
    axes[0, 0].set_title(f'Noise vs Gradient (r={correlations["noise_vs_gradient"][0]:.3f})')
    
    # 2. Noise predictions vs removal importance
    axes[0, 1].scatter(results['noise_predictions'], results['removal_importance'], alpha=0.6)
    axes[0, 1].set_xlabel('Noise Predictions')
    axes[0, 1].set_ylabel('Removal Importance')
    axes[0, 1].set_title(f'Noise vs Removal (r={correlations["noise_vs_removal"][0]:.3f})')
    
    # 3. Noise predictions vs embedding magnitudes
    axes[0, 2].scatter(results['noise_predictions'], results['embedding_magnitudes'], alpha=0.6)
    axes[0, 2].set_xlabel('Noise Predictions')
    axes[0, 2].set_ylabel('Embedding Magnitudes')
    axes[0, 2].set_title(f'Noise vs Magnitudes (r={correlations["noise_vs_magnitudes"][0]:.3f})')
    
    # 4. Noise predictions vs recommendation scores
    axes[1, 0].scatter(results['noise_predictions'], results['recommendation_scores'], alpha=0.6)
    axes[1, 0].set_xlabel('Noise Predictions')
    axes[1, 0].set_ylabel('Recommendation Scores')
    axes[1, 0].set_title(f'Noise vs Rec Scores (r={correlations["noise_vs_rec_scores"][0]:.3f})')
    
    # 5. Gradient vs removal importance
    axes[1, 1].scatter(results['gradient_importance'], results['removal_importance'], alpha=0.6)
    axes[1, 1].set_xlabel('Gradient Importance')
    axes[1, 1].set_ylabel('Removal Importance')
    axes[1, 1].set_title(f'Gradient vs Removal (r={correlations["gradient_vs_removal"][0]:.3f})')
    
    # 6. Noise predictions distribution
    axes[1, 2].hist(results['noise_predictions'], bins=50, alpha=0.7)
    axes[1, 2].set_xlabel('Noise Predictions')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('Noise Predictions Distribution')
    
    plt.tight_layout()
    plt.savefig('continuous_diffusion_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_continuous_patterns(results):
    """
    Analyze patterns in continuous diffusion model outputs
    """
    print("\n=== Continuous Diffusion Pattern Analysis ===")
    
    # Analyze noise prediction distribution
    print(f"Noise predictions - Mean: {np.mean(results['noise_predictions']):.3f}")
    print(f"Noise predictions - Std: {np.std(results['noise_predictions']):.3f}")
    print(f"Noise predictions - Min: {np.min(results['noise_predictions']):.3f}")
    print(f"Noise predictions - Max: {np.max(results['noise_predictions']):.3f}")
    
    # Check if noise predictions are close to zero (good denoising)
    close_to_zero = np.sum(np.abs(results['noise_predictions']) < 0.1)
    total_predictions = len(results['noise_predictions'])
    
    print(f"Noise predictions close to 0 (<0.1): {close_to_zero}/{total_predictions} ({close_to_zero/total_predictions*100:.1f}%)")
    
    # Analyze correlation with user history size
    corr, p_value = pearsonr(results['noise_predictions'], results['user_history_size'])
    print(f"Noise predictions vs User history size: r={corr:.3f}, p={p_value:.3f}")
    
    # Analyze embedding magnitude patterns
    print(f"\nEmbedding magnitudes - Mean: {np.mean(results['embedding_magnitudes']):.3f}")
    print(f"Embedding magnitudes - Std: {np.std(results['embedding_magnitudes']):.3f}")

def analyze_diffusion_timesteps(vae, diffusion, embeddings, num_users=10):
    """
    Analyze how noise predictions change across different timesteps
    """
    print("\n=== Timestep Analysis ===")
    
    timesteps = [0, 50, 100, 200, 500, 750, 999]
    timestep_results = {}
    
    for t in timesteps:
        noise_magnitudes = []
        for i in range(min(num_users, embeddings.shape[0])):
            embedding = embeddings[i]
            noise_pred = compute_diffusion_noise_prediction(embedding, diffusion, t)
            noise_magnitude = torch.norm(noise_pred).item()
            noise_magnitudes.append(noise_magnitude)
        
        timestep_results[t] = {
            'mean_magnitude': np.mean(noise_magnitudes),
            'std_magnitude': np.std(noise_magnitudes),
            'magnitudes': noise_magnitudes
        }
        print(f"Timestep {t}: Mean noise magnitude = {np.mean(noise_magnitudes):.3f} ± {np.std(noise_magnitudes):.3f}")
    
    return timestep_results

def main():
    global args
    args = parse_args()
    print(f"Starting continuous diffusion model analysis")
    
    # Run analysis
    results = analyze_continuous_model_outputs(args.model_path, args.embedding_path)
    
    # Compute correlations
    correlations = compute_correlations_continuous(results)
    
    # Analyze patterns
    analyze_continuous_patterns(results)
    
    # Load models for timestep analysis
    vae, diffusion = load_models(args.model_path)
    user_profiles, embeddings = load_data()
    if embeddings is None:
        # Generate embeddings for timestep analysis
        embeddings = []
        for i in range(min(50, user_profiles.shape[0])):
            user_tensor = torch.FloatTensor(user_profiles[i]).unsqueeze(0).to(DEVICE)
            embedding = get_embedding(user_tensor, vae).squeeze(0)
            embeddings.append(embedding)
        embeddings = torch.stack(embeddings)
    
    timestep_results = analyze_diffusion_timesteps(vae, diffusion, embeddings)
    
    # Create visualizations
    plot_continuous_analysis(results, correlations)
    
    print("\n=== Analysis Complete ===")
    print("Check 'continuous_diffusion_analysis.png' for visualizations")
    
    return results, correlations, timestep_results

if __name__ == "__main__":
    main() 