#!/usr/bin/env python3
"""
eval_Multidim.py - Enhanced evaluation script for diffusion models with multi-dimensional checkpoints

This script is an improved version of eval.py that can handle checkpoints with:
- Non-consecutive layer indices (e.g., layers.0, layers.3, layers.6, layers.9)
- Different input/output dimensions
- Various model architectures

Usage:
    python eval_Multidim.py --checkpoint checkpoint_name.pt [other_args]
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import pickle
import time
import argparse

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# === Import VAE and get_user_recommended_item ===
from src.LXR.recommenders_architecture import VAE
from src.LXR.help_functions import get_user_recommended_item, get_top_k, get_index_in_the_list, get_ndcg, recommender_run

# ========== CONFIGURATION ==========
parser = argparse.ArgumentParser(description="Evaluate diffusion model for counterfactual explanations")
parser.add_argument('--checkpoint', type=str, default="best_advanced_diffusion_ML1M_VAE_trial9_epoch19.pt", nargs='?')
parser.add_argument('--data_name', type=str, default="ML1M", nargs='?')
parser.add_argument('--recommender_name', type=str, default="VAE", nargs='?')
parser.add_argument('--batch_size', type=int, default=128, nargs='?')
parser.add_argument('--learning_rate', type=float, default=0.001, nargs='?')
parser.add_argument('--lambda_cf', type=float, default=0.5, nargs='?')
parser.add_argument('--lambda_l1', type=float, default=1.0, nargs='?')
parser.add_argument('--lambda_preserve', type=float, default=1.0, nargs='?')

# Model architecture arguments
parser.add_argument('--hidden_dim', type=int, default=1024, nargs='?')
parser.add_argument('--dropout_rate', type=float, default=0.0579318641325684, nargs='?')
parser.add_argument('--num_layers', type=int, default=4, nargs='?')
parser.add_argument('--layer_ratio', type=float, default=0.5, nargs='?')
parser.add_argument('--activation', type=str, default="LeakyReLU", nargs='?')
parser.add_argument('--use_skip_connection', action='store_true', default=True)

args = parser.parse_args()

DATA_PATH = Path(f'datasets/lxr-CE/{args.data_name}/train_data_{args.data_name}.csv')
TEST_DATA_PATH = Path(f'datasets/lxr-CE/{args.data_name}/test_data_{args.data_name}.csv')
STATIC_TEST_DATA_PATH = Path(f'datasets/lxr-CE/{args.data_name}/static_test_data_{args.data_name}.csv')
CHECKPOINT_PATH = Path('checkpoints/recommenders/VAE_ML1M_0_19_128.pt')
DIFFUSION_CHECKPOINT_PATH = Path(f'checkpoints/diffusionModels/{args.checkpoint}')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset configurations
num_users_dict = {
    "ML1M": 6037,
    "Yahoo": 13797, 
    "Pinterest": 19155
}

num_items_dict = {
    "ML1M": 3381,
    "Yahoo": 4604, 
    "Pinterest": 9362
}

output_type_dict = {
    "VAE": "multiple",
    "MLP": "single"
}

USER_HISTORY_DIM = num_items_dict[args.data_name]
output_type = output_type_dict[args.recommender_name]

# ========== DATA LOADING ==========
def load_data():
    """Load training, test, and static test data"""
    train_data = pd.read_csv(DATA_PATH, index_col=0)
    test_data = pd.read_csv(TEST_DATA_PATH, index_col=0)
    static_test_data = pd.read_csv(STATIC_TEST_DATA_PATH, index_col=0)
    
    # Remove user_id column if present
    if 'user_id' in train_data.columns:
        train_data = train_data.drop(columns=['user_id'])
    if 'user_id' in test_data.columns:
        test_data = test_data.drop(columns=['user_id'])
    
    train_array = train_data.values.astype(np.float32)
    test_array = test_data.values.astype(np.float32)
    static_test_array = static_test_data.iloc[:,:-2].values.astype(np.float32)
    
    return train_array, test_array, static_test_array, test_data

# ========== DENOISING MODEL ==========
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class AdvancedDenoisingMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, dropout_rate=0.1, num_layers=3, 
                 layer_ratio=0.5, activation="ReLU", use_skip_connection=True):
        super().__init__()
        
        # Activation function selection
        if activation == "ReLU":
            self.activation = nn.ReLU()
        elif activation == "LeakyReLU":
            self.activation = nn.LeakyReLU(0.1)
        elif activation == "GELU":
            self.activation = nn.GELU()
        elif activation == "Swish":
            self.activation = Swish()
        else:
            self.activation = nn.ReLU()
        
        # Build layers dynamically
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            if i == 0:
                next_dim = hidden_dim
            elif i == num_layers - 1:
                next_dim = input_dim
            else:
                next_dim = int(hidden_dim * (layer_ratio ** i))
            
            layers.extend([
                nn.Linear(current_dim, next_dim),
                self.activation,
                nn.Dropout(dropout_rate)
            ])
            current_dim = next_dim
        
        self.layers = nn.Sequential(*layers[:-1])  # Remove last dropout
        
        # Skip connection
        self.use_skip_connection = use_skip_connection
        if use_skip_connection:
            self.skip = nn.Linear(input_dim, input_dim)
        
    def forward(self, x):
        main_out = self.layers(x)
        
        if self.use_skip_connection:
            skip_out = self.skip(x)
            return torch.sigmoid(main_out + skip_out)
        else:
            return torch.sigmoid(main_out)

def analyze_checkpoint_structure(checkpoint_path):
    """
    Analyze the checkpoint structure to understand the layer configuration.
    Returns detailed information about the model architecture.
    """
    print(f"Analyzing checkpoint structure: {checkpoint_path}")
    
    # Load the checkpoint state dict
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    
    # Analyze the state dict to determine architecture
    layer_keys = [key for key in state_dict.keys() if 'layers.' in key and '.weight' in key]
    layer_keys.sort()
    
    print(f"Found layer keys: {layer_keys}")
    
    # Extract layer indices and dimensions
    layer_info = []
    for key in layer_keys:
        layer_idx = int(key.split('.')[1])
        weight_shape = state_dict[key].shape
        layer_info.append((layer_idx, weight_shape))
        print(f"Layer {key}: {weight_shape}")
    
    layer_info.sort(key=lambda x: x[0])  # Sort by layer index
    
    # Determine input dimension from the first layer
    input_dim = layer_info[0][1][1]  # Second dimension of first layer
    
    # Check if skip connection exists
    has_skip = any('skip' in key for key in state_dict.keys())
    
    print(f"Layer indices: {[info[0] for info in layer_info]}")
    print(f"Input dimension: {input_dim}")
    print(f"Has skip connection: {has_skip}")
    
    return {
        'layer_info': layer_info,
        'input_dim': input_dim,
        'has_skip': has_skip,
        'state_dict': state_dict
    }

def build_exact_match_model(checkpoint_path):
    """
    Build a model that exactly matches the checkpoint structure.
    This handles cases where layer indices are not consecutive.
    """
    print(f"Building exact match model for: {checkpoint_path}")
    
    # Analyze checkpoint structure
    checkpoint_info = analyze_checkpoint_structure(checkpoint_path)
    layer_info = checkpoint_info['layer_info']
    input_dim = checkpoint_info['input_dim']
    has_skip = checkpoint_info['has_skip']
    state_dict = checkpoint_info['state_dict']
    
    # Create a custom model that matches the checkpoint exactly
    class ExactMatchDenoisingMLP(nn.Module):
        def __init__(self, layer_info, input_dim, use_skip_connection=True, activation="LeakyReLU"):
            super().__init__()
            
            # Activation function selection
            if activation == "ReLU":
                self.activation = nn.ReLU()
            elif activation == "LeakyReLU":
                self.activation = nn.LeakyReLU(0.1)
            elif activation == "GELU":
                self.activation = nn.GELU()
            elif activation == "Swish":
                self.activation = Swish()
            else:
                self.activation = nn.LeakyReLU(0.1)
            
            # Create layers exactly as they appear in the checkpoint
            self.layer_modules = nn.ModuleDict()
            
            for layer_idx, (original_idx, weight_shape) in enumerate(layer_info):
                in_features = weight_shape[1]
                out_features = weight_shape[0]
                
                # Create the linear layer with the exact dimensions from checkpoint
                # Use a valid module name without dots
                layer_name = f"layer_{original_idx}"
                self.layer_modules[layer_name] = nn.Linear(in_features, out_features)
                
                print(f"Created layer {layer_name}: {in_features} -> {out_features}")
            
            # Skip connection
            self.use_skip_connection = use_skip_connection
            if use_skip_connection:
                self.skip = nn.Linear(input_dim, input_dim)
        
        def forward(self, x):
            # Apply layers in the order they appear in the checkpoint
            current_input = x
            
            # Sort layer names by the original index to maintain order
            sorted_layer_names = sorted(self.layer_modules.keys(), 
                                      key=lambda name: int(name.split('_')[1]))
            
            for layer_name in sorted_layer_names:
                layer = self.layer_modules[layer_name]
                current_input = layer(current_input)
                
                # Apply activation except for the last layer
                if layer_name != sorted_layer_names[-1]:
                    current_input = self.activation(current_input)
            
            # Apply skip connection if it exists
            if self.use_skip_connection:
                skip_out = self.skip(x)
                return torch.sigmoid(current_input + skip_out)
            else:
                return torch.sigmoid(current_input)
    
    # Create the model
    model = ExactMatchDenoisingMLP(
        layer_info=layer_info,
        input_dim=input_dim,
        use_skip_connection=has_skip,
        activation=args.activation
    ).to(DEVICE)
    
    # Create a mapping from checkpoint keys to model keys
    state_dict_mapping = {}
    for key in state_dict.keys():
        if 'layers.' in key:
            # Extract the layer index and rest of the key
            parts = key.split('.')
            layer_idx = parts[1]
            rest = '.'.join(parts[2:])  # weight or bias
            new_key = f"layer_modules.layer_{layer_idx}.{rest}"
            state_dict_mapping[new_key] = state_dict[key]
        else:
            # Keep other keys as they are (like skip connection)
            state_dict_mapping[key] = state_dict[key]
    
    # Load the state dict with the mapping
    model.load_state_dict(state_dict_mapping)
    model.eval()
    
    print(f"Successfully created exact match model")
    return model

def load_diffusion_model(checkpoint_path):
    """Load the trained diffusion model with exact architecture matching"""
    try:
        # Build exact match model
        model = build_exact_match_model(checkpoint_path)
        return model
    except Exception as e:
        print(f"Failed to build exact match model: {e}")
        print("This checkpoint may have an incompatible structure")
        raise e

# ========== VAE RECOMMENDER LOADING ==========
def load_vae_recommender(checkpoint_path, device=DEVICE):
    VAE_config = {
        "enc_dims": [256, 64],
        "dropout": 0.5,
        "anneal_cap": 0.2,
        "total_anneal_steps": 200000
    }
    
    all_items_tensor = torch.eye(USER_HISTORY_DIM, device=device)
    kw_dict = {
        'device': device, 
        'num_items': USER_HISTORY_DIM,
        'all_items_tensor': all_items_tensor,
        'output_type': output_type,
        'recommender_name': args.recommender_name
    }
    
    model = VAE(VAE_config, **kw_dict).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, kw_dict

# ========== DIFFUSION EXPLANATION FUNCTION ==========
def find_diffusion_mask(user_tensor, item_tensor, item_id, diffusion_model, recommender, **kw_dict):
    """
    Generate counterfactual explanations using the diffusion model.
    Returns a dictionary mapping item indices to their explanation scores.
    """
    diffusion_model.eval()
    
    with torch.no_grad():
        # Get the original recommendation
        original_score = recommender_run(user_tensor, recommender, item_tensor, item_id, 'single', **kw_dict)
        
        # Create a copy of user tensor for perturbation
        user_tensor_copy = user_tensor.clone()
        
        # Apply diffusion model to get counterfactual
        counterfactual = diffusion_model(user_tensor_copy)
        
        # Binarize the counterfactual (threshold at 0.5)
        counterfactual_bin = (counterfactual > 0.5).float()
        
        # Calculate the difference between original and counterfactual
        # This gives us the "importance" of each item
        importance_scores = torch.abs(user_tensor_copy - counterfactual_bin)
        
        # Create explanation dictionary
        item_sim_dict = {}
        user_hist = user_tensor_copy.cpu().numpy()
        
        for i in range(len(user_hist)):
            if user_hist[i] > 0:  # Only consider items in user's history
                item_sim_dict[i] = importance_scores[i].item()
        
        return item_sim_dict

# ========== EVALUATION FUNCTIONS ==========
def single_user_expl_diffusion(user_vector, user_tensor, item_id, item_tensor, recommender, diffusion_model, **kw_dict):
    """
    Generate explanations for a single user using diffusion model.
    Returns a dictionary of explanations, sorted by their scores.
    """
    user_hist_size = int(np.sum(user_vector))
    
    # Get diffusion-based explanations
    sim_items = find_diffusion_mask(user_tensor, item_tensor, item_id, diffusion_model, recommender, **kw_dict)
    
    # Sort by importance scores (descending)
    POS_sim_items = list(sorted(sim_items.items(), key=lambda item: item[1], reverse=True))[:user_hist_size]
    
    return POS_sim_items

def single_user_metrics_diffusion(user_vector, user_tensor, item_id, item_tensor, num_of_bins, recommender_model, expl_dict, **kw_dict):
    """
    Calculate metrics for a single user based on diffusion explanations.
    Follows the same structure as the original single_user_metrics function.
    """
    POS_masked = user_tensor
    NEG_masked = user_tensor
    POS_masked[item_id] = 0
    NEG_masked[item_id] = 0
    user_hist_size = int(np.sum(user_vector))
    
    bins = [0] + [len(x) for x in np.array_split(np.arange(user_hist_size), num_of_bins, axis=0)]
    
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
    
    POS_sim_items = expl_dict
    NEG_sim_items = list(sorted(dict(POS_sim_items).items(), key=lambda item: item[1], reverse=False))
    
    total_items = 0
    for i in range(len(bins)):
        total_items += bins[i]
        
        # Create positive mask (remove most important items)
        POS_masked = torch.zeros_like(user_tensor, dtype=torch.float32, device=DEVICE)
        for j in POS_sim_items[:total_items]:
            POS_masked[j[0]] = 1
        POS_masked = user_tensor - POS_masked
        
        # Create negative mask (remove least important items)
        NEG_masked = torch.zeros_like(user_tensor, dtype=torch.float32, device=DEVICE)
        for j in NEG_sim_items[:total_items]:
            NEG_masked[j[0]] = 1
        NEG_masked = user_tensor - NEG_masked
        
        # Get ranked lists
        POS_ranked_list = get_top_k(POS_masked, user_tensor, recommender_model, **kw_dict)
        
        if item_id in list(POS_ranked_list.keys()):
            POS_index = list(POS_ranked_list.keys()).index(item_id) + 1
        else:
            POS_index = USER_HISTORY_DIM
            
        NEG_index = get_index_in_the_list(NEG_masked, user_tensor, item_id, recommender_model, **kw_dict) + 1
        
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
        
        DEL[i] = float(recommender_run(POS_masked, recommender_model, item_tensor, item_id, **kw_dict).detach().cpu().numpy())
        INS[i] = float(recommender_run(user_tensor - POS_masked, recommender_model, item_tensor, item_id, **kw_dict).detach().cpu().numpy())
        NDCG[i] = get_ndcg(list(POS_ranked_list.keys()), item_id, **kw_dict)
    
    res = [DEL, INS, NDCG, POS_at_1, POS_at_5, POS_at_10, POS_at_20, POS_at_50, POS_at_100, 
           NEG_at_1, NEG_at_5, NEG_at_10, NEG_at_20, NEG_at_50, NEG_at_100]
    
    for i in range(len(res)):
        res[i] = np.array(res[i])
    
    return res

def eval_diffusion_model():
    """
    Main evaluation function for the diffusion model.
    """
    print(f"=== Evaluating Diffusion Model (Multi-dimensional) ===")
    print(f"Dataset: {args.data_name}")
    print(f"Recommender: {args.recommender_name}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {DEVICE}")
    
    # Load data
    train_array, test_array, static_test_array, test_data = load_data()
    print(f"Loaded data - Train: {train_array.shape}, Test: {test_array.shape}, Static Test: {static_test_array.shape}")
    
    # Load models
    diffusion_model = load_diffusion_model(DIFFUSION_CHECKPOINT_PATH)
    recommender, kw_dict = load_vae_recommender(CHECKPOINT_PATH, DEVICE)
    print(f"Loaded diffusion model and recommender")
    
    # Prepare items array
    items_array = np.eye(USER_HISTORY_DIM)
    all_items_tensor = torch.Tensor(items_array).to(DEVICE)
    kw_dict['all_items_tensor'] = all_items_tensor
    kw_dict['items_array'] = items_array
    
    # Initialize metric arrays
    num_of_bins = 10
    users_DEL = np.zeros(num_of_bins + 1)
    users_INS = np.zeros(num_of_bins + 1)
    NDCG = np.zeros(num_of_bins + 1)
    POS_at_1 = np.zeros(num_of_bins + 1)
    POS_at_5 = np.zeros(num_of_bins + 1)
    POS_at_10 = np.zeros(num_of_bins + 1)
    POS_at_20 = np.zeros(num_of_bins + 1)
    POS_at_50 = np.zeros(num_of_bins + 1)
    POS_at_100 = np.zeros(num_of_bins + 1)
    NEG_at_1 = np.zeros(num_of_bins + 1)
    NEG_at_5 = np.zeros(num_of_bins + 1)
    NEG_at_10 = np.zeros(num_of_bins + 1)
    NEG_at_20 = np.zeros(num_of_bins + 1)
    NEG_at_50 = np.zeros(num_of_bins + 1)
    NEG_at_100 = np.zeros(num_of_bins + 1)
    
    # Evaluate on test set
    print(f"Starting evaluation on {static_test_array.shape[0]} test users...")
    
    with torch.no_grad():
        for i in range(static_test_array.shape[0]):
            if i % 500 == 0:
                print(f"Processing user {i}/{static_test_array.shape[0]}")
            
            start_time = time.time()
            
            # Get user data
            user_vector = static_test_array[i]
            user_tensor = torch.FloatTensor(user_vector).to(DEVICE)
            user_id = int(test_data.index[i]) if hasattr(test_data, 'index') else i
            
            # Get top recommended item
            item_id = int(get_user_recommended_item(user_tensor, recommender, **kw_dict).detach().cpu().numpy())
            item_vector = items_array[item_id]
            item_tensor = torch.FloatTensor(item_vector).to(DEVICE)
            
            # Remove the recommended item from user history for evaluation
            user_vector[item_id] = 0
            user_tensor[item_id] = 0
            
            # Generate explanations using diffusion model
            user_expl = single_user_expl_diffusion(user_vector, user_tensor, item_id, item_tensor, 
                                                  recommender, diffusion_model, **kw_dict)
            
            # Calculate metrics
            res = single_user_metrics_diffusion(user_vector, user_tensor, item_id, item_tensor, 
                                               num_of_bins, recommender, user_expl, **kw_dict)
            
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
    num_users = static_test_array.shape[0]
    print(f"\n=== Diffusion Model Evaluation Results (Multi-dimensional) ===")
    print(f"Number of users evaluated: {num_users}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.data_name}, Recommender: {args.recommender_name}")
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
    
    results_path = Path(f'results/diffusion_multidim_{args.data_name}_{args.recommender_name}_results.pkl')
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to: {results_path}")
    
    return results

# ========== MAIN ==========
if __name__ == "__main__":
    print("=== Diffusion Model Evaluation (Multi-dimensional) ===")
    print("Usage examples:")
    print("  # Use default checkpoint (trial 9)")
    print("  python eval_Multidim.py")
    print("")
    print("  # Use a specific checkpoint")
    print("  python eval_Multidim.py --checkpoint best_advanced_diffusion_ML1M_VAE_trial39.pt")
    print("")
    print("  # Use custom parameters")
    print("  python eval_Multidim.py --checkpoint my_checkpoint.pt --data_name ML1M --batch_size 256")
    print("")
    
    results = eval_diffusion_model() 