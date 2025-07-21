#!/usr/bin/env python3
"""
Analysis of diffusion model outputs to understand what the values actually mean.
This file investigates whether the model outputs represent importance, reconstruction confidence, 
or something else entirely.
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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.LXR.recommenders_architecture import VAE
from src.LXR.help_functions import get_user_recommended_item
from src.Diffusion.diffusion_explainer import load_vae_recommender

# ========== CONFIGURATION ==========
DATA_PATH = Path('datasets/lxr-CE/ML1M/train_data_ML1M.csv')
CHECKPOINT_PATH = Path('checkpoints/recommenders/VAE_ML1M_0_19_128.pt')
DIFFUSION_CHECKPOINT_PATH = Path('checkpoints/diffusionModels/best_denoiser_ML1M_bs128_lr0.005_lcf10.0_l10.9_pres0.8_V2.pt')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USER_HISTORY_DIM = 3381

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze outputs of a specified diffusion or importance model.")
    parser.add_argument('--model-name', '-m', type=str, default='diffusion',
                        help='Name of the model to analyze (e.g., diffusion, importance, improved_diffusion).')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Optional: Path to the model checkpoint. If not set, uses default for the model name.')
    return parser.parse_args()

def load_data():
    """Load training and test data"""
    df = pd.read_csv(DATA_PATH, index_col=0)
    if 'user_id' in df.columns:
        df = df.drop(columns=['user_id'])
    data = df.values.astype(np.float32)
    return data

def load_models(model_name, model_path=None):
    """Load the diffusion model and recommender"""
    from src.Diffusion.diffusion_explainer import DenoisingMLP
    diffusion_model = None
    if model_name == 'diffusion':
        checkpoint = model_path or DIFFUSION_CHECKPOINT_PATH
        diffusion_model = DenoisingMLP(USER_HISTORY_DIM).to(DEVICE)
        diffusion_model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
        diffusion_model.eval()
    elif model_name == 'importance':
        from src.Diffusion.direct_importance_training import ImportanceMLP
        checkpoint = model_path or Path('checkpoints/diffusion/importance_model.pt')
        diffusion_model = ImportanceMLP(USER_HISTORY_DIM).to(DEVICE)
        diffusion_model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
        diffusion_model.eval()
    elif model_name == 'improved_diffusion':
        from src.Diffusion.improved_diffusion_training import ImprovedDiffusionMLP
        checkpoint = model_path or Path('checkpoints/diffusion/improved_diffusion_model.pt')
        diffusion_model = ImprovedDiffusionMLP(USER_HISTORY_DIM).to(DEVICE)
        diffusion_model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
        diffusion_model.eval()
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    recommender = load_vae_recommender(CHECKPOINT_PATH, device=DEVICE)
    return diffusion_model, recommender

def compute_gradient_importance(user_tensor, recommender, item_id):
    """
    Compute gradient-based importance scores for comparison
    """
    user_tensor.requires_grad_(True)
    
    # Get recommendation score for the target item
    scores = recommender(user_tensor.unsqueeze(0))
    target_score = scores[0, item_id]
    
    # Compute gradients
    target_score.backward()
    gradient_importance = user_tensor.grad.abs()
    
    return gradient_importance.detach()

def compute_removal_importance(user_tensor, recommender, item_id):
    """
    Compute importance by measuring how much removing each item affects the recommendation
    """
    original_score = recommender(user_tensor.unsqueeze(0))[0, item_id].item()
    
    removal_effects = []
    user_history = torch.where(user_tensor > 0)[0]
    
    for item_idx in user_history:
        # Create modified user tensor with item removed
        modified_tensor = user_tensor.clone()
        modified_tensor[item_idx] = 0
        
        # Get new score
        new_score = recommender(modified_tensor.unsqueeze(0))[0, item_id].item()
        
        # Importance = how much score changes
        importance = abs(original_score - new_score)
        removal_effects.append((item_idx.item(), importance))
    
    return dict(removal_effects)

def analyze_model_outputs(model_name, model_path=None):
    """
    Main analysis function to understand what the diffusion model outputs represent
    """
    print(f"=== Analyzing Model Outputs for: {model_name} ===")
    
    # Load data and models
    train_data = load_data()
    diffusion_model, recommender = load_models(model_name, model_path)
    
    # Prepare kw_dict for get_user_recommended_item
    all_items_tensor = torch.eye(USER_HISTORY_DIM, device=DEVICE)
    kw_dict = {
        'device': DEVICE,
        'num_items': USER_HISTORY_DIM,
        'all_items_tensor': all_items_tensor,
        'output_type': 'multiple',
        'recommender_name': 'VAE'
    }
    
    # Sample users for analysis
    num_users = min(100, train_data.shape[0])
    user_indices = np.random.choice(train_data.shape[0], num_users, replace=False)
    
    results = {
        'model_scores': [],
        'gradient_importance': [],
        'removal_importance': [],
        'item_frequency': [],
        'user_history_size': [],
        'recommendation_scores': []
    }
    
    print(f"Analyzing {num_users} users...")
    
    for i, user_idx in enumerate(user_indices):
        if i % 10 == 0:
            print(f"Processing user {i}/{num_users}")
        
        # Get user data
        user_vector = train_data[user_idx]
        user_tensor = torch.FloatTensor(user_vector).to(DEVICE)
        
        # Get recommended item
        item_id = int(get_user_recommended_item(user_tensor, recommender, **kw_dict).cpu().detach().numpy())
        
        # Get model outputs
        with torch.no_grad():
            model_scores = diffusion_model(user_tensor)
        
        # Get gradient importance
        gradient_imp = compute_gradient_importance(user_tensor, recommender, item_id)
        
        # Get removal importance
        removal_imp = compute_removal_importance(user_tensor, recommender, item_id)
        
        # Get item frequencies (from training data)
        item_freq = np.mean(train_data, axis=0)
        
        # Get user history size
        history_size = np.sum(user_vector)
        
        # Get recommendation scores for all items
        with torch.no_grad():
            rec_scores = recommender(user_tensor.unsqueeze(0))[0].cpu().numpy()
        
        # Store results
        user_history = torch.where(user_tensor > 0)[0]
        
        for item_idx in user_history:
            item_idx = item_idx.item()
            
            results['model_scores'].append(model_scores[item_idx].item())
            results['gradient_importance'].append(gradient_imp[item_idx].item())
            results['removal_importance'].append(removal_imp.get(item_idx, 0.0))
            results['item_frequency'].append(item_freq[item_idx])
            results['user_history_size'].append(history_size)
            results['recommendation_scores'].append(rec_scores[item_idx])
    
    # Convert to numpy arrays
    for key in results:
        results[key] = np.array(results[key])
    
    return results

def compute_correlations(results):
    """
    Compute correlations between model scores and various metrics
    """
    print("\n=== Correlation Analysis ===")
    
    correlations = {}
    
    # Model scores vs gradient importance
    corr, p_value = pearsonr(results['model_scores'], results['gradient_importance'])
    correlations['model_vs_gradient'] = (corr, p_value)
    print(f"Model scores vs Gradient importance: r={corr:.3f}, p={p_value:.3f}")
    
    # Model scores vs removal importance
    corr, p_value = pearsonr(results['model_scores'], results['removal_importance'])
    correlations['model_vs_removal'] = (corr, p_value)
    print(f"Model scores vs Removal importance: r={corr:.3f}, p={p_value:.3f}")
    
    # Model scores vs item frequency
    corr, p_value = pearsonr(results['model_scores'], results['item_frequency'])
    correlations['model_vs_frequency'] = (corr, p_value)
    print(f"Model scores vs Item frequency: r={corr:.3f}, p={p_value:.3f}")
    
    # Model scores vs recommendation scores
    corr, p_value = pearsonr(results['model_scores'], results['recommendation_scores'])
    correlations['model_vs_rec_scores'] = (corr, p_value)
    print(f"Model scores vs Recommendation scores: r={corr:.3f}, p={p_value:.3f}")
    
    # Gradient vs removal importance
    corr, p_value = pearsonr(results['gradient_importance'], results['removal_importance'])
    correlations['gradient_vs_removal'] = (corr, p_value)
    print(f"Gradient vs Removal importance: r={corr:.3f}, p={p_value:.3f}")
    
    return correlations

def plot_analysis(results, correlations):
    """
    Create visualizations to understand model outputs
    """
    print("\n=== Creating Visualizations ===")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Model scores vs gradient importance
    axes[0, 0].scatter(results['model_scores'], results['gradient_importance'], alpha=0.6)
    axes[0, 0].set_xlabel('Model Scores')
    axes[0, 0].set_ylabel('Gradient Importance')
    axes[0, 0].set_title(f'Model vs Gradient (r={correlations["model_vs_gradient"][0]:.3f})')
    
    # 2. Model scores vs removal importance
    axes[0, 1].scatter(results['model_scores'], results['removal_importance'], alpha=0.6)
    axes[0, 1].set_xlabel('Model Scores')
    axes[0, 1].set_ylabel('Removal Importance')
    axes[0, 1].set_title(f'Model vs Removal (r={correlations["model_vs_removal"][0]:.3f})')
    
    # 3. Model scores vs item frequency
    axes[0, 2].scatter(results['model_scores'], results['item_frequency'], alpha=0.6)
    axes[0, 2].set_xlabel('Model Scores')
    axes[0, 2].set_ylabel('Item Frequency')
    axes[0, 2].set_title(f'Model vs Frequency (r={correlations["model_vs_frequency"][0]:.3f})')
    
    # 4. Model scores vs recommendation scores
    axes[1, 0].scatter(results['model_scores'], results['recommendation_scores'], alpha=0.6)
    axes[1, 0].set_xlabel('Model Scores')
    axes[1, 0].set_ylabel('Recommendation Scores')
    axes[1, 0].set_title(f'Model vs Rec Scores (r={correlations["model_vs_rec_scores"][0]:.3f})')
    
    # 5. Gradient vs removal importance
    axes[1, 1].scatter(results['gradient_importance'], results['removal_importance'], alpha=0.6)
    axes[1, 1].set_xlabel('Gradient Importance')
    axes[1, 1].set_ylabel('Removal Importance')
    axes[1, 1].set_title(f'Gradient vs Removal (r={correlations["gradient_vs_removal"][0]:.3f})')
    
    # 6. Model scores distribution
    axes[1, 2].hist(results['model_scores'], bins=50, alpha=0.7)
    axes[1, 2].set_xlabel('Model Scores')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('Model Scores Distribution')
    
    plt.tight_layout()
    plt.savefig('model_output_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_score_patterns(results):
    """
    Analyze patterns in model scores to understand what they represent
    """
    print("\n=== Score Pattern Analysis ===")
    
    # Analyze score distribution
    print(f"Model scores - Mean: {np.mean(results['model_scores']):.3f}")
    print(f"Model scores - Std: {np.std(results['model_scores']):.3f}")
    print(f"Model scores - Min: {np.min(results['model_scores']):.3f}")
    print(f"Model scores - Max: {np.max(results['model_scores']):.3f}")
    
    # Check if scores are close to 0 or 1 (reconstruction confidence)
    close_to_zero = np.sum(results['model_scores'] < 0.1)
    close_to_one = np.sum(results['model_scores'] > 0.9)
    total_scores = len(results['model_scores'])
    
    print(f"Scores close to 0 (<0.1): {close_to_zero}/{total_scores} ({close_to_zero/total_scores*100:.1f}%)")
    print(f"Scores close to 1 (>0.9): {close_to_one}/{total_scores} ({close_to_one/total_scores*100:.1f}%)")
    
    # Analyze correlation with user history size
    corr, p_value = pearsonr(results['model_scores'], results['user_history_size'])
    print(f"Model scores vs User history size: r={corr:.3f}, p={p_value:.3f}")

def main():
    args = parse_args()
    print(f"Starting diffusion model output analysis for model: {args.model_name}")
    
    # Run analysis
    results = analyze_model_outputs(args.model_name, args.model_path)
    
    # Compute correlations
    correlations = compute_correlations(results)
    
    # Analyze patterns
    analyze_score_patterns(results)
    
    # Create visualizations
    plot_analysis(results, correlations)
    
    print("\n=== Analysis Complete ===")
    print("Check 'model_output_analysis.png' for visualizations")
    
    return results, correlations

if __name__ == "__main__":
    main() 