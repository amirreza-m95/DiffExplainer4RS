import torch
import numpy as np
from pathlib import Path
import sys
import os
import argparse
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'LXR')))
from recommenders_architecture import VAE
from diffusion_model import DiffusionMLP
from diffusion_model import TransformerDiffusionModel

# === DATASET CONFIGURATIONS ===
DATASET_CONFIGS = {
    'ml1m': {
        'embedding_path': 'checkpoints/embeddings/user_embeddings_ml1m.npy',
        'model_path': 'checkpoints/diffusionModels/diffusion_transformer_ml1m_best_aug13th_loss065.pth',
        'vae_checkpoint_path': 'checkpoints/recommenders/VAE/VAE_ML1M_4_28_128newbest.pt',
        'results_file': 'counterfactual_results_ml1m.json',
        'user_history_dim': 3381,
        'model_type': 'transformer',
        'model_config': {
            'embedding_dim': None,  # Will be set dynamically
            'hidden_dim': 256,
            'num_layers': 6,
            'num_heads': 8,
            'dropout': 0.1
        }
    },
    'pinterest': {
        'embedding_path': 'checkpoints/embeddings/user_embeddings_pinterest.npy',
        'model_path': 'checkpoints/diffusionModels/diffusion_transformer_pinterest_best.pth',
        'vae_checkpoint_path': 'checkpoints/recommenders/VAE_Pinterest_12_18_0.0001_256.pt',
        'results_file': 'counterfactual_results_pinterest.json',
        'user_history_dim': 9362,
        'model_type': 'transformer',
        'model_config': {
            'embedding_dim': None,  # Will be set dynamically
            'hidden_dim': 256,
            'num_layers': 6,
            'num_heads': 8,
            'dropout': 0.1
        }
    }
}

# === GLOBAL CONFIGURATION ===
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TIMESTEPS = 1000
GUIDANCE_STEPS = 50
GUIDANCE_LAMBDA = 2.0  # Strength of guidance toward changing top-1

# VAE config (should match training)
VAE_config = {
    "enc_dims": [256, 256],
    "dropout": 0.5,
    "anneal_cap": 0.2,
    "total_anneal_steps": 200000
}

def create_diffusion_model(model_type, embedding_dim, model_config):
    """
    Create diffusion model based on type and configuration.
    Args:
        model_type: 'transformer' or 'mlp'
        embedding_dim: Dimension of embeddings
        model_config: Model-specific configuration
    Returns:
        Diffusion model
    """
    if model_type == 'transformer':
        config = model_config.copy()
        config['embedding_dim'] = embedding_dim
        return TransformerDiffusionModel(**config).to(DEVICE)
    elif model_type == 'mlp':
        return DiffusionMLP(embedding_dim).to(DEVICE)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def get_top1(embedding, vae):
    """
    Get top-1 recommendation from VAE.
    Args:
        embedding: [latent_dim], vae: VAE model
    Returns:
        top1: top-1 item index, scores: all item scores
    """
    # Pass through decoder layers only
    h = embedding.unsqueeze(0)
    for layer in vae.decoder:
        h = layer(h)
    scores = h.squeeze(0)
    top1 = torch.argmax(scores).item()
    return top1, scores

def sample_counterfactual(orig_embedding, orig_top1, vae, diffusion, guidance_lambda=GUIDANCE_LAMBDA):
    """
    Generate counterfactual embedding using guided diffusion sampling.
    Args:
        orig_embedding: Original user embedding
        orig_top1: Original top-1 recommendation
        vae: VAE model
        diffusion: Diffusion model
        guidance_lambda: Strength of guidance
    Returns:
        Counterfactual embedding
    """
    x = orig_embedding.clone().detach().unsqueeze(0)
    for t in reversed(range(TIMESTEPS)):
        t_tensor = torch.full((1,), t, dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            noise_pred = diffusion(x, t_tensor)
        alpha_t = alphas[t]
        alpha_cumprod_t = alphas_cumprod[t]
        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)
        coef1 = 1.0 / torch.sqrt(alpha_t)
        coef2 = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_cumprod_t)
        x = coef1 * (x - coef2 * noise_pred) + torch.sqrt(betas[t]) * noise
        # Guidance: encourage change in top-1
        x.requires_grad_(True)
        top1, scores = get_top1(x.squeeze(0), vae)
        if top1 == orig_top1:
            # Compute gradient to reduce score of orig_top1
            loss = guidance_lambda * scores[orig_top1]
            grad = torch.autograd.grad(loss, x)[0]
            x = x - 0.1 * grad  # Step size can be tuned
        x = x.detach()
    return x.squeeze(0)

def generate_counterfactuals(dataset_name, num_users=10):
    """
    Generate counterfactuals for a specific dataset.
    Args:
        dataset_name: Name of the dataset ('ml1m' or 'pinterest')
        num_users: Number of users to generate counterfactuals for
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Dataset '{dataset_name}' not supported. Available datasets: {list(DATASET_CONFIGS.keys())}")
    
    config = DATASET_CONFIGS[dataset_name]
    
    print(f"Generating counterfactuals for {dataset_name} dataset...")
    print(f"Embedding path: {config['embedding_path']}")
    print(f"Model path: {config['model_path']}")
    print(f"VAE checkpoint path: {config['vae_checkpoint_path']}")
    print(f"User history dimension: {config['user_history_dim']}")
    
    # === Load models and data ===
    embeddings = np.load(config['embedding_path'])
    embeddings = torch.tensor(embeddings, dtype=torch.float32, device=DEVICE)
    embedding_dim = embeddings.shape[1]
    
    print(f"Loaded {embeddings.shape[0]} embeddings with dimension {embedding_dim}")
    
    # Load diffusion model
    diffusion = create_diffusion_model(config['model_type'], embedding_dim, config['model_config'])
    diffusion.load_state_dict(torch.load(config['model_path'], map_location=DEVICE))
    diffusion.eval()
    
    # Load VAE model
    vae = VAE(VAE_config, device=DEVICE, num_items=config['user_history_dim']).to(DEVICE)
    vae.load_state_dict(torch.load(config['vae_checkpoint_path'], map_location=DEVICE))
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    
    # === Diffusion Schedule ===
    global alphas, alphas_cumprod, betas
    betas = torch.linspace(1e-4, 0.02, TIMESTEPS, device=DEVICE)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    print(f"Starting counterfactual generation for {num_users} users...")
    
    # === Main: Generate and Save Counterfactuals ===
    results = []
    for idx in range(num_users):
        orig_emb = embeddings[idx]
        orig_top1, orig_scores = get_top1(orig_emb, vae)
        cf_emb = sample_counterfactual(orig_emb, orig_top1, vae, diffusion)
        cf_top1, cf_scores = get_top1(cf_emb, vae)
        l2_dist = torch.norm(cf_emb - orig_emb).item()
        changed = (orig_top1 != cf_top1)
        results.append({
            'user_idx': idx,
            'orig_top1': orig_top1,
            'cf_top1': cf_top1,
            'l2_dist': l2_dist,
            'changed': changed
        })
        print(f"User {idx}: orig_top1={orig_top1}, cf_top1={cf_top1}, l2={l2_dist:.4f}, changed={changed}")
    
    # Save results
    with open(config['results_file'], 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved counterfactual results to {config['results_file']}")
    
    # Print summary statistics
    changed_count = sum(1 for r in results if r['changed'])
    avg_l2 = np.mean([r['l2_dist'] for r in results])
    print(f"Summary: {changed_count}/{len(results)} users had changed top-1 recommendations")
    print(f"Average L2 distance: {avg_l2:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate counterfactuals for different datasets')
    parser.add_argument('--dataset', type=str, default='ml1m', 
                       choices=['ml1m', 'pinterest'],
                       help='Dataset to generate counterfactuals for (default: ml1m)')
    parser.add_argument('--num_users', type=int, default=10,
                       help='Number of users to generate counterfactuals for (default: 10)')
    
    args = parser.parse_args()
    
    # Generate counterfactuals for the specified dataset
    generate_counterfactuals(args.dataset, args.num_users) 