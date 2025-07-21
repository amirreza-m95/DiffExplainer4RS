import torch
import numpy as np
from pathlib import Path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'LXR')))
from recommenders_architecture import VAE
from diffusion_model import DiffusionMLP

# === CONFIGURATION ===
EMBEDDING_PATH = Path('user_embeddings.npy')
MODEL_PATH = Path('diffusion_mlp_best.pth')
VAE_CHECKPOINT_PATH = Path('checkpoints/recommenders/VAE_ML1M_0_19_128.pt')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TIMESTEPS = 1000
GUIDANCE_STEPS = 50
GUIDANCE_LAMBDA = 2.0  # Strength of guidance toward changing top-1

# VAE config (should match training)
VAE_config = {
    "enc_dims": [256, 64],
    "dropout": 0.5,
    "anneal_cap": 0.2,
    "total_anneal_steps": 200000
}
USER_HISTORY_DIM = 3381

# === Load models and data ===
embeddings = np.load(EMBEDDING_PATH)
embeddings = torch.tensor(embeddings, dtype=torch.float32, device=DEVICE)
embedding_dim = embeddings.shape[1]

diffusion = DiffusionMLP(embedding_dim).to(DEVICE)
diffusion.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
diffusion.eval()

vae = VAE(VAE_config, device=DEVICE, num_items=USER_HISTORY_DIM).to(DEVICE)
vae.load_state_dict(torch.load(VAE_CHECKPOINT_PATH, map_location=DEVICE))
vae.eval()
for param in vae.parameters():
    param.requires_grad = False

# === Diffusion Schedule ===
betas = torch.linspace(1e-4, 0.02, TIMESTEPS, device=DEVICE)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

# === Helper: get top-1 recommendation from VAE ===
def get_top1(embedding, vae):
    # embedding: [latent_dim], vae: VAE model
    # Pass through decoder layers only
    h = embedding.unsqueeze(0)
    for layer in vae.decoder:
        h = layer(h)
    scores = h.squeeze(0)
    top1 = torch.argmax(scores).item()
    return top1, scores

# === Guided Sampling for Counterfactuals ===
def sample_counterfactual(orig_embedding, orig_top1, vae, diffusion, guidance_lambda=GUIDANCE_LAMBDA):
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

# === Main: Generate and Save Counterfactuals ===
results = []
for idx in range(10):  # For demonstration, do 10 users
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

import json
with open('counterfactual_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Saved counterfactual results to counterfactual_results.json") 