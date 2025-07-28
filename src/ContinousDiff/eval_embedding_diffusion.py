import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os
import pickle

# Add LXR directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'LXR')))
from recommenders_architecture import VAE
from diffusion_model import DiffusionMLP

# === CONFIGURATION ===
# Paths to data and model checkpoints
TEST_DATA_PATH = Path('datasets/lxr-CE/ML1M/test_data_ML1M.csv')
VAE_CHECKPOINT_PATH = Path('checkpoints/recommenders/VAE_ML1M_0_19_128.pt')
DIFFUSION_MODEL_PATH = Path('diffusion_mlp_best.pth')
EMBEDDING_DIM = 64  # Latent dimension of VAE
USER_HISTORY_DIM = 3381  # Number of items in ML1M
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TIMESTEPS = 120  # Number of diffusion steps
GUIDANCE_LAMBDA = 2.0  # Strength of guidance during sampling
NUM_BINS = 10  # Number of bins for evaluation (10% increments)

# VAE config (should match training)
VAE_config = {
    "enc_dims": [256, 64],
    "dropout": 0.5,
    "anneal_cap": 0.2,
    "total_anneal_steps": 200000
}

# === Load test data ===
def load_test_users():
    """
    Load test user profiles from CSV. Ensures only item columns are used.
    Returns:
        - user_profiles: np.ndarray of shape (num_users, USER_HISTORY_DIM)
        - user_indices: user IDs or DataFrame indices
    """
    df = pd.read_csv(TEST_DATA_PATH, index_col=0)
    if isinstance(df, pd.DataFrame):
        if 'user_id' in df.columns:
            df = df.drop(columns=['user_id'])
        df = df.iloc[:, :USER_HISTORY_DIM]
        return df.values.astype(np.float32), df.index.values
    else:
        raise ValueError("Test data could not be loaded as a DataFrame.")

# === Load models ===
# Load trained VAE recommender
vae = VAE(VAE_config, device=DEVICE, num_items=USER_HISTORY_DIM).to(DEVICE)
vae.load_state_dict(torch.load(VAE_CHECKPOINT_PATH, map_location=DEVICE))
vae.eval()
for param in vae.parameters():
    param.requires_grad = False

# Load trained diffusion model
# This model learns to denoise user embeddings
# and is used to generate counterfactual embeddings
# via guided sampling

# The diffusion model is a simple MLP
# that takes a noisy embedding and timestep as input
# and predicts the noise

diffusion = DiffusionMLP(EMBEDDING_DIM).to(DEVICE)
diffusion.load_state_dict(torch.load(DIFFUSION_MODEL_PATH, map_location=DEVICE))
diffusion.eval()

# === Diffusion Schedule ===
# These are the standard DDPM schedules for noise addition/removal
betas = torch.linspace(1e-4, 0.02, TIMESTEPS, device=DEVICE)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

# === Helper functions ===
def get_embedding(user_tensor, vae):
    """
    Pass a user profile through the VAE encoder to get the mean embedding.
    Args:
        user_tensor: torch.Tensor of shape (1, USER_HISTORY_DIM)
        vae: trained VAE model
    Returns:
        mu_q: torch.Tensor of shape (1, EMBEDDING_DIM)
    """
    h = torch.nn.functional.normalize(user_tensor, dim=-1)
    h = torch.nn.functional.dropout(h, p=vae.dropout, training=False)
    for layer in vae.encoder:
        h = layer(h)
    mu_q = h[:, :vae.enc_dims[-1]]
    return mu_q

def decode_embedding(embedding, vae):
    """
    Pass an embedding through the VAE decoder to get item preference scores.
    Args:
        embedding: torch.Tensor of shape (EMBEDDING_DIM,)
        vae: trained VAE model
    Returns:
        scores: torch.Tensor of shape (USER_HISTORY_DIM,)
    """
    h = embedding.unsqueeze(0)
    for layer in vae.decoder:
        h = layer(h)
    return h.squeeze(0)

def get_top_k(user_tensor, recommender):
    """
    Get the full sorted list of recommended items for a user profile.
    Args:
        user_tensor: torch.Tensor of shape (USER_HISTORY_DIM,)
        recommender: VAE model
    Returns:
        dict: {item_index: score}, sorted by score descending
    """
    with torch.no_grad():
        scores = recommender(user_tensor.unsqueeze(0)).squeeze(0)
        # Mask out items the user already has
        scores[user_tensor > 0] = float('-inf')
        sorted_indices = torch.argsort(scores, descending=True)
        return {i.item(): scores[i].item() for i in sorted_indices}

def get_index_in_the_list(user_tensor, item_id, recommender):
    """
    Get the rank of a specific item in the recommendation list.
    Args:
        user_tensor: torch.Tensor of shape (USER_HISTORY_DIM,)
        item_id: int
        recommender: VAE model
    Returns:
        int: rank (0-based)
    """
    with torch.no_grad():
        scores = recommender(user_tensor.unsqueeze(0)).squeeze(0)
        scores[user_tensor > 0] = float('-inf')
        sorted_indices = torch.argsort(scores, descending=True)
        try:
            rank = torch.where(sorted_indices == item_id)[0].item()
            return rank
        except:
            return len(scores) - 1

def get_ndcg(ranked_list, item_id, k=10):
    """
    Compute NDCG@k for a ranked list and a target item.
    Args:
        ranked_list: list of item indices
        item_id: int
        k: int
    Returns:
        float: NDCG score
    """
    if item_id in ranked_list[:k]:
        idx = ranked_list[:k].index(item_id)
        return 1.0 / np.log2(idx + 2)
    return 0.0

def recommender_run(user_tensor, recommender, item_tensor, item_id):
    """
    Get the recommender's score for a specific item given a user profile.
    Args:
        user_tensor: torch.Tensor of shape (USER_HISTORY_DIM,)
        recommender: VAE model
        item_tensor: torch.Tensor (not used here, for compatibility)
        item_id: int
    Returns:
        float: score for item_id
    """
    with torch.no_grad():
        scores = recommender(user_tensor.unsqueeze(0)).squeeze(0)
        return scores[item_id]

def sample_counterfactual(orig_embedding, orig_profile, vae, diffusion, guidance_lambda=GUIDANCE_LAMBDA):
    """
    Generate a counterfactual embedding using the diffusion model with guidance.
    Args:
        orig_embedding: torch.Tensor (EMBEDDING_DIM,)
        orig_profile: torch.Tensor (USER_HISTORY_DIM,)
        vae: VAE model
        diffusion: DiffusionMLP model
        guidance_lambda: float
    Returns:
        torch.Tensor: counterfactual embedding (EMBEDDING_DIM,)
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
        # Guidance: encourage removal of important items
        x.requires_grad_(True)
        # cf_preferences = decode_embedding(x.squeeze(0), vae)
        # user_items = torch.where(orig_profile == 1)[0]

        # # Mask out user items for recommendation
        # cf_preferences_masked = cf_preferences.clone()
        # cf_preferences_masked[user_items] = float('-inf')
        # top1_idx = torch.argmax(cf_preferences_masked)

        # # Loss to reduce the score of the top-1 recommended item
        # top1_loss = guidance_lambda * cf_preferences[top1_idx]

        # # (Optional) Existing loss for user items
        # if len(user_items) > 0:
        #     item_scores = cf_preferences[user_items]
        #     top_user_item_idx = torch.argmax(item_scores)
        #     user_item_loss = guidance_lambda * item_scores[top_user_item_idx]
        #     # Combine losses (e.g., sum or weighted sum)
        #     loss = user_item_loss + top1_loss
        # else:
        #     loss = top1_loss

        # grad = torch.autograd.grad(loss, x)[0]
        # x = x - 0.1 * grad
        cf_preferences = decode_embedding(x.squeeze(0), vae)
        user_items = torch.where(orig_profile == 1)[0]
        if len(user_items) > 0:
            item_scores = cf_preferences[user_items]
            # Encourage removal of the item with highest score
            top_item_idx = torch.argmax(item_scores)
            loss = guidance_lambda * item_scores[top_item_idx]
            grad = torch.autograd.grad(loss, x)[0]
            x = x - 0.1 * grad
        x = x.detach()
        # print(f"Top-1 score before: {cf_preferences[top1_idx].item()}")
    return x.squeeze(0)

def sample_counterfactual_top1(orig_embedding, orig_profile, vae, diffusion, guidance_lambda=GUIDANCE_LAMBDA):
    """
    Generate a counterfactual embedding by always penalizing the original top-1 user item.
    Args:
        orig_embedding: torch.Tensor (EMBEDDING_DIM,)
        orig_profile: torch.Tensor (USER_HISTORY_DIM,)
        vae: VAE model
        diffusion: DiffusionMLP model
        guidance_lambda: float
    Returns:
        torch.Tensor: counterfactual embedding (EMBEDDING_DIM,)
    """
    # Identify the original top-1 user item
    cf_preferences = decode_embedding(orig_embedding, vae)
    user_items = torch.where(orig_profile == 1)[0]
    if len(user_items) > 0:
        item_scores = cf_preferences[user_items]
        top_item_idx = torch.argmax(item_scores)
        target_item = user_items[top_item_idx]
    else:
        target_item = None

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
        x.requires_grad_(True)
        if target_item is not None:
            cf_preferences = decode_embedding(x.squeeze(0), vae)
            loss = guidance_lambda * cf_preferences[target_item]
            grad = torch.autograd.grad(loss, x)[0]
            x = x - 0.1 * grad
        x = x.detach()
    return x.squeeze(0)

def sample_counterfactual_topX(orig_embedding, orig_profile, vae, diffusion, guidance_lambda=GUIDANCE_LAMBDA, top_percent=0.1):
    """
    Generate a counterfactual embedding by always penalizing the original top X% user items.
    Args:
        orig_embedding: torch.Tensor (EMBEDDING_DIM,)
        orig_profile: torch.Tensor (USER_HISTORY_DIM,)
        vae: VAE model
        diffusion: DiffusionMLP model
        guidance_lambda: float
        top_percent: float (e.g., 0.1 for top 10%)
    Returns:
        torch.Tensor: counterfactual embedding (EMBEDDING_DIM,)
    """
    # Identify the original top X% user items
    cf_preferences = decode_embedding(orig_embedding, vae)
    user_items = torch.where(orig_profile == 1)[0]
    if len(user_items) > 0:
        item_scores = cf_preferences[user_items]
        num_top = max(1, int(len(user_items) * top_percent))
        sorted_indices = torch.argsort(item_scores, descending=True)
        target_items = user_items[sorted_indices[:num_top]]
    else:
        target_items = []

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
        x.requires_grad_(True)
        if len(target_items) > 0:
            cf_preferences = decode_embedding(x.squeeze(0), vae)
            loss = guidance_lambda * cf_preferences[target_items].sum()
            grad = torch.autograd.grad(loss, x)[0]
            x = x - 0.1 * grad
        x = x.detach()
    return x.squeeze(0)

def single_user_metrics_embedding_diffusion(user_vector, user_tensor, item_id, cf_embedding, vae, recommender):
    """
    For a single user, rank their items by importance (from decoded cf_embedding),
    then remove items in 10% increments and compute metrics at each step.
    Args:
        user_vector: np.ndarray (original binary profile)
        user_tensor: torch.Tensor (original binary profile)
        item_id: int (top recommended item)
        cf_embedding: torch.Tensor (counterfactual embedding)
        vae: VAE model
        recommender: VAE model (for scoring)
    Returns:
        list of np.ndarray: metrics for each bin
    """
    # Decode counterfactual embedding to get importance scores
    cf_preferences = decode_embedding(cf_embedding, vae)
    
    # Get items the user has and their importance scores
    user_items = torch.where(user_tensor == 1)[0]
    user_hist_size = len(user_items) # why not user_vector? because they are the same
    
    if user_hist_size == 0:
        # User has no items, return default metrics
        return [np.zeros(NUM_BINS + 1) for _ in range(15)]
    
    # Sort items by importance (highest first for removal)
    item_scores = cf_preferences[user_items]
    sorted_indices_pos = torch.argsort(item_scores, descending=True)
    sorted_items_pos = user_items[sorted_indices_pos]
    sorted_indices_neg = torch.argsort(item_scores, descending=False)
    sorted_items_neg = user_items[sorted_indices_neg]
    
    # Create bins (10% increments)
    bins = [0] + [len(x) for x in np.array_split(np.arange(user_hist_size), NUM_BINS, axis=0)]
    
    # Initialize metric arrays
    POS_at_1 = [0] * len(bins)
    POS_at_5 = [0] * len(bins)
    POS_at_10 = [0] * len(bins)
    POS_at_20 = [0] * len(bins)
    POS_at_50 = [0] * len(bins)
    POS_at_100 = [0] * len(bins)
    
    NEG_at_1 = [0] * len(bins)
    NEG_at_5 = [0] * len(bins)
    NEG_at_10 = [0] * len(bins)
    NEG_at_20 = [0] * len(bins)
    NEG_at_50 = [0] * len(bins)
    NEG_at_100 = [0] * len(bins)
    
    DEL = [0.0] * len(bins)
    INS = [0.0] * len(bins)
    NDCG = [0.0] * len(bins)
    
    total_items = 0
    for i in range(len(bins)):
        total_items += bins[i]
        
        # POS: remove most important items
        masked_profile_pos = user_tensor.clone()
        if total_items > 0:
            items_to_remove_pos = sorted_items_pos[:total_items]
            masked_profile_pos[items_to_remove_pos] = 0
        
        # NEG: remove least important items
        masked_profile_neg = user_tensor.clone()
        if total_items > 0:
            items_to_remove_neg = sorted_items_neg[:total_items]
            masked_profile_neg[items_to_remove_neg] = 0
        
        # Get ranked lists
        POS_ranked_list = get_top_k(masked_profile_pos, recommender)
        
        if item_id in list(POS_ranked_list.keys()):
            POS_index = list(POS_ranked_list.keys()).index(item_id) + 1
        else:
            POS_index = USER_HISTORY_DIM
        
        NEG_index = get_index_in_the_list(masked_profile_neg, item_id, recommender) + 1
        
        # Calculate metrics
        POS_at_1[i] = 1 if POS_index <= 1 else 0
        POS_at_5[i] = 1 if POS_index <= 5 else 0
        POS_at_10[i] = 1 if POS_index <= 10 else 0
        POS_at_20[i] = 1 if POS_index <= 20 else 0
        POS_at_50[i] = 1 if POS_index <= 50 else 0
        POS_at_100[i] = 1 if POS_index <= 100 else 0
        
        NEG_at_1[i] = 1 if NEG_index <= 1 else 0
        NEG_at_5[i] = 1 if NEG_index <= 5 else 0
        NEG_at_10[i] = 1 if NEG_index <= 10 else 0
        NEG_at_20[i] = 1 if NEG_index <= 20 else 0
        NEG_at_50[i] = 1 if NEG_index <= 50 else 0
        NEG_at_100[i] = 1 if NEG_index <= 100 else 0

        # if i > 0:
        #     # For POS metrics: if previous value was 0, current should also be 0
        #     if POS_at_1[i-1] == 0:
        #         POS_at_1[i] = 0
        #     if POS_at_5[i-1] == 0:
        #         POS_at_5[i] = 0
        #     if POS_at_10[i-1] == 0:
        #         POS_at_10[i] = 0
        #     if POS_at_20[i-1] == 0:
        #         POS_at_20[i] = 0
        #     if POS_at_50[i-1] == 0:
        #         POS_at_50[i] = 0
        #     if POS_at_100[i-1] == 0:
        #         POS_at_100[i] = 0
            
        #     # For NEG metrics: if previous value was 0, current should also be 0
        #     if NEG_at_1[i-1] == 0:
        #         NEG_at_1[i] = 0
        #     if NEG_at_5[i-1] == 0:
        #         NEG_at_5[i] = 0
        #     if NEG_at_10[i-1] == 0:
        #         NEG_at_10[i] = 0
        #     if NEG_at_20[i-1] == 0:
        #         NEG_at_20[i] = 0
        #     if NEG_at_50[i-1] == 0:
        #         NEG_at_50[i] = 0
        #     if NEG_at_100[i-1] == 0:
        #         NEG_at_100[i] = 0

        
        # Calculate DEL and INS
        item_tensor = torch.zeros(USER_HISTORY_DIM, device=DEVICE)
        item_tensor[item_id] = 1.0
        DEL[i] = float(recommender_run(masked_profile_pos, recommender, item_tensor, item_id).detach().cpu().numpy())
        INS[i] = float(recommender_run(user_tensor - masked_profile_pos, recommender, item_tensor, item_id).detach().cpu().numpy())
        NDCG[i] = get_ndcg(list(POS_ranked_list.keys()), item_id)
    
    res = [DEL, INS, NDCG, POS_at_1, POS_at_5, POS_at_10, POS_at_20, POS_at_50, POS_at_100, 
           NEG_at_1, NEG_at_5, NEG_at_10, NEG_at_20, NEG_at_50, NEG_at_100]
    
    for i in range(len(res)):
        res[i] = np.array(res[i])
    
    return res

# === Main Evaluation ===
def main():
    """
    Main evaluation loop. For each user, generate a counterfactual embedding,
    rank their items by importance, and compute metrics as items are removed in bins.
    Aggregates and prints/saves average metrics across all users.
    """
    test_users, user_indices = load_test_users()
    test_users = test_users[:10] # number of users to evaluate
    
    # Initialize metric arrays
    users_DEL = np.zeros(NUM_BINS + 1)
    users_INS = np.zeros(NUM_BINS + 1)
    NDCG = np.zeros(NUM_BINS + 1)
    POS_at_1 = np.zeros(NUM_BINS + 1)
    POS_at_5 = np.zeros(NUM_BINS + 1)
    POS_at_10 = np.zeros(NUM_BINS + 1)
    POS_at_20 = np.zeros(NUM_BINS + 1)
    POS_at_50 = np.zeros(NUM_BINS + 1)
    POS_at_100 = np.zeros(NUM_BINS + 1)
    NEG_at_1 = np.zeros(NUM_BINS + 1)
    NEG_at_5 = np.zeros(NUM_BINS + 1)
    NEG_at_10 = np.zeros(NUM_BINS + 1)
    NEG_at_20 = np.zeros(NUM_BINS + 1)
    NEG_at_50 = np.zeros(NUM_BINS + 1)
    NEG_at_100 = np.zeros(NUM_BINS + 1)
    
    print(f"Starting evaluation on {len(test_users)} test users...")
    
    for idx, user_vec in enumerate(test_users):
        if idx % 100 == 0:
            print(f"Processing user {idx}/{len(test_users)}")
        
        user_tensor = torch.tensor(user_vec, dtype=torch.float32, device=DEVICE)
        orig_emb = get_embedding(user_tensor.unsqueeze(0), vae).squeeze(0)
        
        # Get the recommended item for this user
        with torch.no_grad():
            scores = vae(user_tensor.unsqueeze(0)).squeeze(0)
            scores[user_tensor > 0] = float('-inf')  # Mask out items user already has
            item_id = torch.argmax(scores).item()
        orig_top1 = item_id
        
        # Generate counterfactual embedding
        cf_emb = sample_counterfactual(orig_emb, user_tensor, vae, diffusion)
        
        # Decode counterfactual embedding to get importance scores
        cf_preferences = decode_embedding(cf_emb, vae)
        user_items = torch.where(user_tensor == 1)[0]
        user_hist_size = len(user_items)
        if user_hist_size > 0:
            item_scores = cf_preferences[user_items]
            sorted_indices = torch.argsort(item_scores, descending=True)
            sorted_items = user_items[sorted_indices]
            # Remove top 10% most important items for reporting
            num_remove = max(1, user_hist_size // 50)
            items_removed = sorted_items[:num_remove].cpu().numpy().tolist()
            # Create masked profile by removing these items
            masked_profile = user_tensor.clone()
            masked_profile[items_removed] = 0
            # Get new top-1 after removal
            with torch.no_grad():
                scores_masked = vae(masked_profile.unsqueeze(0)).squeeze(0)
                scores_masked[masked_profile > 0] = float('-inf')
                new_top1 = torch.argmax(scores_masked).item()
        else:
            items_removed = []
            new_top1 = orig_top1
        if new_top1 == orig_top1:
            print(f"User {idx}: orig_top1={orig_top1}, new_top1={new_top1}, items_removed={items_removed}, user_hist_len={user_hist_size}")
        else:
            print(f"successful the user history is {user_hist_size}")
        # Calculate metrics for this user
        res = single_user_metrics_embedding_diffusion(user_vec, user_tensor, item_id, cf_emb, vae, vae)
        
        # Accumulate metrics
        users_DEL += res[0]
        users_INS += res[1]
        NDCG += res[2]
        POS_at_1 += res[3]
        POS_at_5 += res[4]
        POS_at_10 += res[5]
        POS_at_20 += res[6]
        POS_at_50 += res[7]
        POS_at_100 += res[8]
        NEG_at_1 += res[9]
        NEG_at_5 += res[10]
        NEG_at_10 += res[11]
        NEG_at_20 += res[12]
        NEG_at_50 += res[13]
        NEG_at_100 += res[14]
    
    # Calculate averages
    num_users = len(test_users)
    print(f"\n=== Embedding Diffusion Evaluation Results ===")
    print(f"Number of users evaluated: {num_users}")
    print(f"Dataset: ML1M, Recommender: VAE")
    print(f"\nAverage Metrics:")
    print(f"DEL: {np.mean(users_DEL)/num_users:.4f}")
    print(f"INS: {np.mean(users_INS)/num_users:.4f}")
    print(f"NDCG: {np.mean(NDCG)/num_users:.4f}")
    print(f"POS@1: {np.mean(POS_at_1)/num_users:.4f}")
    print(f"POS@5: {np.mean(POS_at_5)/num_users:.4f}")
    print(f"POS@10: {np.mean(POS_at_10)/num_users:.4f}")
    print(f"POS@20: {np.mean(POS_at_20)/num_users:.4f}")
    print(f"POS@50: {np.mean(POS_at_50)/num_users:.4f}")
    print(f"POS@100: {np.mean(POS_at_100)/num_users:.4f}")
    print(f"NEG@1: {np.mean(NEG_at_1)/num_users:.4f}")
    print(f"NEG@5: {np.mean(NEG_at_5)/num_users:.4f}")
    print(f"NEG@10: {np.mean(NEG_at_10)/num_users:.4f}")
    print(f"NEG@20: {np.mean(NEG_at_20)/num_users:.4f}")
    print(f"NEG@50: {np.mean(NEG_at_50)/num_users:.4f}")
    print(f"NEG@100: {np.mean(NEG_at_100)/num_users:.4f}")
    
    # Save results
    results = {
        'DEL': users_DEL / num_users,
        'INS': users_INS / num_users,
        'NDCG': NDCG / num_users,
        'POS_at_1': POS_at_1 / num_users,
        'POS_at_5': POS_at_5 / num_users,
        'POS_at_10': POS_at_10 / num_users,
        'POS_at_20': POS_at_20 / num_users,
        'POS_at_50': POS_at_50 / num_users,
        'POS_at_100': POS_at_100 / num_users,
        'NEG_at_1': NEG_at_1 / num_users,
        'NEG_at_5': NEG_at_5 / num_users,
        'NEG_at_10': NEG_at_10 / num_users,
        'NEG_at_20': NEG_at_20 / num_users,
        'NEG_at_50': NEG_at_50 / num_users,
        'NEG_at_100': NEG_at_100 / num_users
    }
    
    with open('embedding_diffusion_eval_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to: embedding_diffusion_eval_results.pkl")

if __name__ == "__main__":
    main() 