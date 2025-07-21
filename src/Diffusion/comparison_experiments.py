#!/usr/bin/env python3
"""
Comparison experiments between different approaches:
1. Original diffusion model
2. Direct importance training
3. Improved diffusion training

This file evaluates and compares all three approaches.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.LXR.recommenders_architecture import VAE
from src.LXR.help_functions import get_user_recommended_item

# ========== CONFIGURATION ==========
DATA_PATH = Path('datasets/lxr-CE/ML1M/train_data_ML1M.csv')
CHECKPOINT_PATH = Path('checkpoints/recommenders/VAE_ML1M_0_19_128.pt')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USER_HISTORY_DIM = 3381

# Model paths
ORIGINAL_DIFFUSION_PATH = Path('checkpoints/diffusion/diffusion_model.pt')
DIRECT_IMPORTANCE_PATH = Path('checkpoints/diffusion/importance_model.pt')
IMPROVED_DIFFUSION_PATH = Path('checkpoints/diffusion/improved_diffusion_model.pt')

def load_data():
    """Load test data"""
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

def load_original_diffusion_model():
    """Load the original diffusion model"""
    from src.Diffusion.diffusion_explainer import DenoisingMLP
    model = DenoisingMLP(USER_HISTORY_DIM).to(DEVICE)
    model.load_state_dict(torch.load(ORIGINAL_DIFFUSION_PATH, map_location=DEVICE))
    model.eval()
    return model

def load_direct_importance_model():
    """Load the direct importance model"""
    from src.Diffusion.direct_importance_training import ImportanceMLP
    model = ImportanceMLP(USER_HISTORY_DIM).to(DEVICE)
    model.load_state_dict(torch.load(DIRECT_IMPORTANCE_PATH, map_location=DEVICE))
    model.eval()
    return model

def load_improved_diffusion_model():
    """Load the improved diffusion model"""
    from src.Diffusion.improved_diffusion_training import ImprovedDiffusionMLP
    model = ImprovedDiffusionMLP(USER_HISTORY_DIM).to(DEVICE)
    model.load_state_dict(torch.load(IMPROVED_DIFFUSION_PATH, map_location=DEVICE))
    model.eval()
    return model

def compute_gradient_importance(user_tensor, recommender, item_id):
    """Compute gradient-based importance scores"""
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

def compute_removal_importance(user_tensor, recommender, item_id):
    """Compute importance by measuring removal effects"""
    original_score = recommender(user_tensor.unsqueeze(0))[0, item_id].item()
    
    removal_effects = torch.zeros_like(user_tensor)
    user_history = torch.where(user_tensor > 0)[0]
    
    for item_idx in user_history:
        modified_tensor = user_tensor.clone()
        modified_tensor[item_idx] = 0
        
        new_score = recommender(modified_tensor.unsqueeze(0))[0, item_id].item()
        importance = abs(original_score - new_score)
        removal_effects[item_idx] = importance
    
    # Normalize
    max_effect = removal_effects.max()
    if max_effect > 0:
        removal_effects = removal_effects / max_effect
    
    return removal_effects

def evaluate_model_importance(model, test_data, recommender, model_name):
    """
    Evaluate a model's importance prediction performance
    """
    print(f"Evaluating {model_name}...")
    
    model.eval()
    gradient_correlations = []
    removal_correlations = []
    
    with torch.no_grad():
        for i in range(min(200, len(test_data))):
            user_vector = test_data[i]
            user_tensor = torch.FloatTensor(user_vector).to(DEVICE)
            
            # Get recommended item
            item_id = int(get_user_recommended_item(user_tensor, recommender).cpu().detach().numpy())
            
            # Get model prediction
            if model_name == "Original Diffusion":
                predicted_importance = model(user_tensor)
            elif model_name == "Direct Importance":
                predicted_importance = model(user_tensor)
            elif model_name == "Improved Diffusion":
                denoised_output, predicted_importance = model(user_tensor)
            
            # Compute true importance measures
            true_gradient_importance = compute_gradient_importance(user_tensor, recommender, item_id)
            true_removal_importance = compute_removal_importance(user_tensor, recommender, item_id)
            
            # Compute correlations
            user_mask = (user_tensor > 0).float()
            masked_pred = (predicted_importance * user_mask).cpu().numpy()
            masked_gradient = (true_gradient_importance * user_mask).cpu().numpy()
            masked_removal = (true_removal_importance * user_mask).cpu().numpy()
            
            # Get non-zero elements
            non_zero_mask = masked_gradient > 0
            if np.sum(non_zero_mask) > 0:
                pred_values = masked_pred[non_zero_mask]
                gradient_values = masked_gradient[non_zero_mask]
                removal_values = masked_removal[non_zero_mask]
                
                # Compute correlations
                gradient_corr = np.corrcoef(pred_values, gradient_values)[0, 1]
                removal_corr = np.corrcoef(pred_values, removal_values)[0, 1]
                
                if not np.isnan(gradient_corr):
                    gradient_correlations.append(gradient_corr)
                if not np.isnan(removal_corr):
                    removal_correlations.append(removal_corr)
    
    avg_gradient_corr = np.mean(gradient_correlations)
    avg_removal_corr = np.mean(removal_correlations)
    
    print(f"  Gradient correlation: {avg_gradient_corr:.3f}")
    print(f"  Removal correlation: {avg_removal_corr:.3f}")
    
    return {
        'gradient_correlation': avg_gradient_corr,
        'removal_correlation': avg_removal_corr,
        'gradient_correlations': gradient_correlations,
        'removal_correlations': removal_correlations
    }

def run_ablation_study(model, test_data, recommender, model_name):
    """
    Run ablation study to test if model scores correlate with recommendation changes
    """
    print(f"Running ablation study for {model_name}...")
    
    model.eval()
    ablation_results = []
    
    with torch.no_grad():
        for i in range(min(100, len(test_data))):
            user_vector = test_data[i]
            user_tensor = torch.FloatTensor(user_vector).to(DEVICE)
            
            # Get recommended item
            item_id = int(get_user_recommended_item(user_tensor, recommender).cpu().detach().numpy())
            original_score = recommender(user_tensor.unsqueeze(0))[0, item_id].item()
            
            # Get model importance scores
            if model_name == "Original Diffusion":
                importance_scores = model(user_tensor)
            elif model_name == "Direct Importance":
                importance_scores = model(user_tensor)
            elif model_name == "Improved Diffusion":
                denoised_output, importance_scores = model(user_tensor)
            
            # Get user history
            user_history = torch.where(user_tensor > 0)[0]
            
            # Sort items by model importance
            item_importance_pairs = []
            for item_idx in user_history:
                importance = importance_scores[item_idx].item()
                item_importance_pairs.append((item_idx.item(), importance))
            
            # Sort by importance (descending)
            item_importance_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # Test removal effects
            for k in [1, 3, 5, 10]:  # Remove top-k items
                if k <= len(item_importance_pairs):
                    # Remove top-k items by model importance
                    modified_tensor = user_tensor.clone()
                    for item_idx, _ in item_importance_pairs[:k]:
                        modified_tensor[item_idx] = 0
                    
                    # Get new score
                    new_score = recommender(modified_tensor.unsqueeze(0))[0, item_id].item()
                    score_change = abs(original_score - new_score)
                    
                    ablation_results.append({
                        'k': k,
                        'score_change': score_change,
                        'model_name': model_name
                    })
    
    return ablation_results

def plot_comparison_results(results):
    """Plot comparison results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Gradient correlation comparison
    model_names = list(results.keys())
    gradient_corrs = [results[name]['gradient_correlation'] for name in model_names]
    
    axes[0, 0].bar(model_names, gradient_corrs, color=['red', 'blue', 'green'])
    axes[0, 0].set_title('Gradient Correlation Comparison')
    axes[0, 0].set_ylabel('Correlation')
    axes[0, 0].set_ylim(0, 1)
    
    # 2. Removal correlation comparison
    removal_corrs = [results[name]['removal_correlation'] for name in model_names]
    
    axes[0, 1].bar(model_names, removal_corrs, color=['red', 'blue', 'green'])
    axes[0, 1].set_title('Removal Correlation Comparison')
    axes[0, 1].set_ylabel('Correlation')
    axes[0, 1].set_ylim(0, 1)
    
    # 3. Ablation study results
    ablation_data = {}
    for model_name in model_names:
        ablation_data[model_name] = {}
        for k in [1, 3, 5, 10]:
            ablation_data[model_name][k] = []
    
    # Collect ablation data
    for result in ablation_results:
        model_name = result['model_name']
        k = result['k']
        score_change = result['score_change']
        ablation_data[model_name][k].append(score_change)
    
    # Plot ablation results
    k_values = [1, 3, 5, 10]
    for i, model_name in enumerate(model_names):
        avg_changes = []
        for k in k_values:
            changes = ablation_data[model_name][k]
            avg_changes.append(np.mean(changes) if changes else 0)
        
        axes[1, 0].plot(k_values, avg_changes, marker='o', label=model_name)
    
    axes[1, 0].set_title('Ablation Study: Score Changes')
    axes[1, 0].set_xlabel('Number of Items Removed (k)')
    axes[1, 0].set_ylabel('Average Score Change')
    axes[1, 0].legend()
    
    # 4. Correlation distribution
    for i, model_name in enumerate(model_names):
        gradient_corrs = results[model_name]['gradient_correlations']
        axes[1, 1].hist(gradient_corrs, alpha=0.7, label=model_name, bins=20)
    
    axes[1, 1].set_title('Gradient Correlation Distribution')
    axes[1, 1].set_xlabel('Correlation')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('model_comparison_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_summary_results(results):
    """Print summary of comparison results"""
    print("\n=== COMPARISON SUMMARY ===")
    print(f"{'Model':<20} {'Gradient Corr':<15} {'Removal Corr':<15}")
    print("-" * 50)
    
    for model_name, result in results.items():
        gradient_corr = result['gradient_correlation']
        removal_corr = result['removal_correlation']
        print(f"{model_name:<20} {gradient_corr:<15.3f} {removal_corr:<15.3f}")
    
    print("\n=== RECOMMENDATIONS ===")
    
    # Find best model for each metric
    best_gradient = max(results.items(), key=lambda x: x[1]['gradient_correlation'])
    best_removal = max(results.items(), key=lambda x: x[1]['removal_correlation'])
    
    print(f"Best for gradient correlation: {best_gradient[0]} ({best_gradient[1]['gradient_correlation']:.3f})")
    print(f"Best for removal correlation: {best_removal[0]} ({best_removal[1]['removal_correlation']:.3f})")
    
    # Overall assessment
    avg_gradient = np.mean([r['gradient_correlation'] for r in results.values()])
    avg_removal = np.mean([r['removal_correlation'] for r in results.values()])
    
    print(f"\nAverage gradient correlation: {avg_gradient:.3f}")
    print(f"Average removal correlation: {avg_removal:.3f}")
    
    if avg_gradient < 0.3:
        print("WARNING: All models show poor correlation with gradient importance!")
    if avg_removal < 0.3:
        print("WARNING: All models show poor correlation with removal importance!")

def main():
    """
    Main comparison function
    """
    print("=== Model Comparison Experiments ===")
    
    # Load data and recommender
    test_data = load_data()
    recommender = load_recommender()
    
    # Load models
    models = {}
    try:
        models["Original Diffusion"] = load_original_diffusion_model()
        print("✓ Loaded Original Diffusion model")
    except:
        print("✗ Could not load Original Diffusion model")
    
    try:
        models["Direct Importance"] = load_direct_importance_model()
        print("✓ Loaded Direct Importance model")
    except:
        print("✗ Could not load Direct Importance model")
    
    try:
        models["Improved Diffusion"] = load_improved_diffusion_model()
        print("✓ Loaded Improved Diffusion model")
    except:
        print("✗ Could not load Improved Diffusion model")
    
    if not models:
        print("ERROR: No models could be loaded!")
        return
    
    # Evaluate each model
    results = {}
    ablation_results = []
    
    for model_name, model in models.items():
        print(f"\n--- Evaluating {model_name} ---")
        
        # Evaluate importance prediction
        result = evaluate_model_importance(model, test_data, recommender, model_name)
        results[model_name] = result
        
        # Run ablation study
        ablation_result = run_ablation_study(model, test_data, recommender, model_name)
        ablation_results.extend(ablation_result)
    
    # Plot results
    plot_comparison_results(results)
    
    # Print summary
    print_summary_results(results)
    
    print("\n=== Comparison Complete ===")
    print("Check 'model_comparison_results.png' for visualizations")
    
    return results, ablation_results

if __name__ == "__main__":
    main() 