import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
from pathlib import Path
import argparse
import optuna
import logging
import wandb
import random

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# === Import VAE and get_user_recommended_item ===
from src.LXR.recommenders_architecture import VAE
from src.LXR.help_functions import get_user_recommended_item

# ========== ARGUMENT PARSING ==========
parser = argparse.ArgumentParser(description="Advanced hyperparameter optimization for diffusion explainer")
parser.add_argument('--data_name', type=str, default="ML1M", nargs='?')
parser.add_argument('--recommender_name', type=str, default="VAE", nargs='?')
parser.add_argument('--n_trials', type=int, default=3, nargs='?')
parser.add_argument('--epochs', type=int, default=10, nargs='?')
parser.add_argument('--manual_trial', action='store_true', help='Use manual hyperparameters instead of Optuna')
parser.add_argument('--learning_rate', type=float, default=0.001, nargs='?')
parser.add_argument('--batch_size', type=int, default=128, nargs='?')
parser.add_argument('--lambda_cf', type=float, default=0.8, nargs='?')
parser.add_argument('--lambda_l1', type=float, default=0.8, nargs='?')
parser.add_argument('--lambda_preserve', type=float, default=1.0, nargs='?')
parser.add_argument('--noise_max_prob', type=float, default=0.7, nargs='?')
parser.add_argument('--max_diffusion_steps', type=int, default=15, nargs='?')
parser.add_argument('--hidden_dim', type=int, default=256, nargs='?')
parser.add_argument('--dropout_rate', type=float, default=0.1, nargs='?')
parser.add_argument('--binarization_threshold', type=float, default=0.5, nargs='?')
parser.add_argument('--optimizer', type=str, default="Adam", nargs='?')
parser.add_argument('--scheduler', type=str, default="StepLR", nargs='?')
parser.add_argument('--weight_decay', type=float, default=0.0, nargs='?')
parser.add_argument('--activation', type=str, default="ReLU", nargs='?')
parser.add_argument('--num_layers', type=int, default=3, nargs='?')
parser.add_argument('--layer_ratio', type=float, default=0.5, nargs='?')
parser.add_argument('--use_skip_connection', action='store_true', default=True)
parser.add_argument('--noise_schedule', type=str, default="linear", nargs='?')
parser.add_argument('--early_stopping_patience', type=int, default=5, nargs='?')
parser.add_argument('--gradient_clip', type=float, default=1.0, nargs='?')
parser.add_argument('--whereSaved', type=str, default='', nargs='?')

args = parser.parse_args()

# ========== CONFIGURATION ==========
DATA_NAME = args.data_name
RECOMMENDER_NAME = args.recommender_name
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data dimensions dictionary
num_items_dict = {
    "ML1M": 3381,
    "Yahoo": 4604, 
    "Pinterest": 9362
}

USER_HISTORY_DIM = num_items_dict[DATA_NAME]

# Paths
DATA_PATH = Path(f'datasets/lxr-CE/{DATA_NAME}/train_data_{DATA_NAME}.csv')
CHECKPOINT_PATH = Path(f'checkpoints/recommenders/{RECOMMENDER_NAME}_{DATA_NAME}_0_19_128.pt')

# ========== DATA LOADING ==========
def load_user_histories(data_path):
    df = pd.read_csv(data_path, index_col=0)
    # Remove user_id column if present
    if 'user_id' in df.columns:
        df = df.drop(columns=['user_id'])
    data = df.values.astype(np.float32)
    return data

# ========== ADVANCED DIFFUSION PROCESS ==========
def bernoulli_diffusion_advanced(x0, t, max_steps=10, max_prob=0.6, schedule="linear"):
    """
    Advanced diffusion with different noise schedules
    """
    if schedule == "linear":
        p = max_prob * (t / max_steps)[:, None]
    elif schedule == "cosine":
        # Cosine schedule for smoother noise increase
        p = max_prob * (1 - torch.cos(torch.pi * t / max_steps))[:, None] / 2
    elif schedule == "exponential":
        # Exponential schedule for more aggressive noise
        p = max_prob * (1 - torch.exp(-3 * t / max_steps))[:, None]
    elif schedule == "quadratic":
        # Quadratic schedule
        p = max_prob * ((t / max_steps) ** 2)[:, None]
    else:
        p = max_prob * (t / max_steps)[:, None]
    
    mask = (torch.rand_like(x0) < p).float()
    x_t = x0 * (1 - mask)
    return x_t, mask

# ========== ADVANCED DENOISING MODEL ==========
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
    return model

# ========== ADVANCED TRAINING FUNCTION ==========
def advanced_diffusion_training(trial):
    # Hyperparameter suggestions
    if args.manual_trial:
        learning_rate = args.learning_rate
        batch_size = args.batch_size
        lambda_cf = args.lambda_cf
        lambda_l1 = args.lambda_l1
        lambda_preserve = args.lambda_preserve
        noise_max_prob = args.noise_max_prob
        max_diffusion_steps = args.max_diffusion_steps
        hidden_dim = args.hidden_dim
        dropout_rate = args.dropout_rate
        binarization_threshold = args.binarization_threshold
        optimizer_name = args.optimizer
        scheduler_name = args.scheduler
        weight_decay = args.weight_decay
        activation = args.activation
        num_layers = args.num_layers
        layer_ratio = args.layer_ratio
        use_skip_connection = args.use_skip_connection
        noise_schedule = args.noise_schedule
        early_stopping_patience = args.early_stopping_patience
        gradient_clip = args.gradient_clip
    else:
        learning_rate = trial.suggest_float('learning_rate', 0.001, 0.02, log=True)
        # batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
        batch_size = 128
        # lambda_cf = trial.suggest_float('lambda_cf', 0.1, 3.0)
        lambda_cf = trial.suggest_categorical('lambda_cf', [1.0])   
        # lambda_l1 = trial.suggest_float('lambda_l1', 0.5, 8.0)
        lambda_l1 = trial.suggest_categorical('lambda_l1', [0.3])
        # lambda_preserve = trial.suggest_float('lambda_preserve', 0.5, 5.0)
        lambda_preserve = trial.suggest_categorical('lambda_preserve', [0.1])
        # noise_max_prob = trial.suggest_float('noise_max_prob', 0.2, 0.9)
        noise_max_prob = trial.suggest_categorical('noise_max_prob', [0.8,1.0])
        # max_diffusion_steps = trial.suggest_int('max_diffusion_steps', 5, 20)
        # hidden_dim = trial.suggest_categorical('hidden_dim', [256, 512, 1024, 2048])
        # dropout_rate = trial.suggest_float('dropout_rate', 0.05, 0.4)
        binarization_threshold = trial.suggest_float('binarization_threshold', 0.55, 0.75)
        # optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW', 'SGD', 'RMSprop'])
        # scheduler_name = trial.suggest_categorical('scheduler', ['StepLR', 'CosineAnnealingLR', 'ReduceLROnPlateau', 'None'])
        # weight_decay = trial.suggest_float('weight_decay', 0.0, 0.01)
        # activation = trial.suggest_categorical('activation', ['ReLU', 'LeakyReLU', 'GELU', 'Swish'])
        # num_layers = trial.suggest_int('num_layers', 2, 5)
        # layer_ratio = trial.suggest_float('layer_ratio', 0.3, 0.8)
        # use_skip_connection = trial.suggest_categorical('use_skip_connection', [True, False])
        # noise_schedule = trial.suggest_categorical('noise_schedule', ['linear', 'cosine', 'exponential', 'quadratic'])
        # early_stopping_patience = trial.suggest_int('early_stopping_patience', 3, 10)
        # gradient_clip = trial.suggest_float('gradient_clip', 0.1, 5.0)


        # learning_rate = args.learning_rate
        batch_size = args.batch_size
        # lambda_cf = args.lambda_cf
        # lambda_l1 = args.lambda_l1
        # lambda_preserve = args.lambda_preserve
        # noise_max_prob = args.noise_max_prob
        max_diffusion_steps = args.max_diffusion_steps
        hidden_dim = args.hidden_dim
        dropout_rate = args.dropout_rate
        binarization_threshold = args.binarization_threshold
        optimizer_name = args.optimizer
        scheduler_name = args.scheduler
        weight_decay = args.weight_decay
        activation = args.activation
        num_layers = args.num_layers
        layer_ratio = args.layer_ratio
        use_skip_connection = args.use_skip_connection
        noise_schedule = args.noise_schedule
        early_stopping_patience = args.early_stopping_patience
        gradient_clip = args.gradient_clip

    epochs = args.epochs

    # Initialize wandb
    wandb.init(
        project=f"{DATA_NAME}_{RECOMMENDER_NAME}_advanced_diffusion_optimization",
        name=f"trial_{trial.number if not args.manual_trial else 'manual'}",
        config={
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'lambda_cf': lambda_cf,
            'lambda_l1': lambda_l1,
            'lambda_preserve': lambda_preserve,
            'noise_max_prob': noise_max_prob,
            'max_diffusion_steps': max_diffusion_steps,
            'hidden_dim': hidden_dim,
            'dropout_rate': dropout_rate,
            'binarization_threshold': binarization_threshold,
            'optimizer': optimizer_name,
            'scheduler': scheduler_name,
            'weight_decay': weight_decay,
            'activation': activation,
            'num_layers': num_layers,
            'layer_ratio': layer_ratio,
            'use_skip_connection': use_skip_connection,
            'noise_schedule': noise_schedule,
            'early_stopping_patience': early_stopping_patience,
            'gradient_clip': gradient_clip,
            'architecture': 'AdvancedDenoisingMLP',
            'epochs': epochs,
            'data_name': DATA_NAME,
            'recommender_name': RECOMMENDER_NAME
        }
    )

    # Load data
    data = load_user_histories(DATA_PATH)
    num_samples = data.shape[0]
    print(f"Loaded {num_samples} user histories of length {data.shape[1]}")

    # Model
    model = AdvancedDenoisingMLP(
        USER_HISTORY_DIM, hidden_dim, dropout_rate, num_layers, 
        layer_ratio, activation, use_skip_connection
    ).to(DEVICE)
    
    # Optimizer selection
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    elif optimizer_name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Scheduler selection
    if scheduler_name == "StepLR":
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    elif scheduler_name == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    else:
        scheduler = None
    
    criterion = nn.BCELoss()

    # Load recommender
    recommender = load_vae_recommender(CHECKPOINT_PATH, device=DEVICE)
    all_items_tensor = torch.eye(USER_HISTORY_DIM, device=DEVICE)
    kw_dict = {
        'device': DEVICE,
        'num_items': USER_HISTORY_DIM,
        'all_items_tensor': all_items_tensor,
        'output_type': 'multiple',
        'recommender_name': RECOMMENDER_NAME
    }

    # Training metrics tracking
    best_loss = float('inf')
    best_change_rate = 0.0
    best_model_state = None
    
    # Early stopping
    patience_counter = 0
    best_epoch_loss = float('inf')

    # Initialize running averages for normalization
    recon_running_avg = None
    l1_running_avg = None
    preserve_running_avg = None
    cf_running_avg = None
    alpha = 0.01  # Smoothing factor for running average

    print(f'======================== ADVANCED RUN ========================')
    print(f'Lambda: CF={lambda_cf}, L1={lambda_l1}, Preserve={lambda_preserve}')
    print(f'Training: LR={learning_rate}, BS={batch_size}, HD={hidden_dim}, Layers={num_layers}')
    print(f'Noise: MaxProb={noise_max_prob}, Steps={max_diffusion_steps}, Schedule={noise_schedule}')
    print(f'Architecture: Act={activation}, Dropout={dropout_rate}, Skip={use_skip_connection}')
    print(f'Optimizer: {optimizer_name}, Scheduler: {scheduler_name}, WD={weight_decay}')
    print(f'Threshold: {binarization_threshold}, GradientClip: {gradient_clip}')
    print('==============================================================')

    for epoch in range(epochs):
        perm = np.random.permutation(num_samples)
        total_loss = 0
        total_cf_loss = 0
        total_recon_loss = 0
        total_l1_loss = 0
        total_preserve_loss = 0
        total_changed = 0
        total_count = 0

        model.train()
        for i in range(0, num_samples, batch_size):
            batch_idx = perm[i:i+batch_size]
            x0 = torch.tensor(data[batch_idx], device=DEVICE)
            
            # Sample random diffusion step for each sample
            t = torch.randint(1, max_diffusion_steps + 1, (x0.size(0),), device=DEVICE)
            x_t, noise_mask = bernoulli_diffusion_advanced(x0, t, max_diffusion_steps, noise_max_prob, noise_schedule)
            
            # Denoising
            x0_hat = model(x_t)
            
            # Binarization with configurable threshold
            x0_hat_bin = x0.clone()
            noised_positions = (noise_mask > 0)
            x0_hat_bin[noised_positions] = (x0_hat[noised_positions] > binarization_threshold).float()
            
            # Get top-1 recommendations
            top1_orig = []
            top1_denoised = []
            for j in range(x0.size(0)):
                top1_orig.append(int(get_user_recommended_item(x0[j], recommender, **kw_dict).cpu().detach().numpy()))
                top1_denoised.append(int(get_user_recommended_item(x0_hat_bin[j], recommender, **kw_dict).cpu().detach().numpy()))
            top1_orig = torch.tensor(top1_orig, device=DEVICE)
            top1_denoised = torch.tensor(top1_denoised, device=DEVICE)
            
            # Loss components
            cf_loss = (top1_orig == top1_denoised).float().mean()  
            masked_recon_loss = (criterion(x0_hat, x0) * noise_mask).sum() / (noise_mask.sum() + 1e-8)
            masked_l1_loss = ((x0_hat - x0).abs() * noise_mask).sum() / (noise_mask.sum() + 1e-8)
            preserve_mask = 1 - noise_mask
            preserve_loss = ((x0_hat - x0).abs() * preserve_mask).sum() / (preserve_mask.sum() + 1e-8)

            # Update running averages for normalization
            if recon_running_avg is None:
                recon_running_avg = masked_recon_loss.item()
                l1_running_avg = masked_l1_loss.item()
                preserve_running_avg = preserve_loss.item()
                cf_running_avg = cf_loss.item()
            else:
                recon_running_avg = (1 - alpha) * recon_running_avg + alpha * masked_recon_loss.item()
                l1_running_avg = (1 - alpha) * l1_running_avg + alpha * masked_l1_loss.item()
                preserve_running_avg = (1 - alpha) * preserve_running_avg + alpha * preserve_loss.item()
                cf_running_avg = (1 - alpha) * cf_running_avg + alpha * cf_loss.item()

            # Normalize only the reconstruction, l1, and preserve losses
            norm_recon_loss = masked_recon_loss / (recon_running_avg + 1e-8)
            norm_l1_loss = masked_l1_loss / (l1_running_avg + 1e-8)
            norm_preserve_loss = preserve_loss / (preserve_running_avg + 1e-8)

            # Use the raw cf_loss (not normalized)
            loss = norm_recon_loss + lambda_cf * cf_loss + lambda_l1 * norm_l1_loss + lambda_preserve * norm_preserve_loss
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item() * x0.size(0)
            total_cf_loss += cf_loss.item() * x0.size(0)
            total_recon_loss += masked_recon_loss.item() * x0.size(0)
            total_l1_loss += masked_l1_loss.item() * x0.size(0)
            total_preserve_loss += preserve_loss.item() * x0.size(0)
            total_changed += (top1_orig != top1_denoised).float().sum().item()
            total_count += x0.size(0)

        # Calculate averages
        avg_loss = total_loss / num_samples
        avg_cf_loss = total_cf_loss / num_samples
        avg_recon_loss = total_recon_loss / num_samples
        avg_l1_loss = total_l1_loss / num_samples
        avg_preserve_loss = total_preserve_loss / num_samples
        change_rate = total_changed / total_count

        # Step scheduler
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(avg_loss)
            else:
                scheduler.step()

        # Log metrics to wandb
        train_metrics = {
            "train/total_loss": avg_loss,
            "train/recon_loss": avg_recon_loss,
            "train/cf_loss": avg_cf_loss,
            "train/l1_loss": avg_l1_loss,
            "train/preserve_loss": avg_preserve_loss,
            "train/change_rate": change_rate,
            "train/epoch": epoch,
            "train/learning_rate": optimizer.param_groups[0]['lr']
        }
        
        wandb.log(train_metrics)
        
        print(f'Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} | Recon: {avg_recon_loss:.4f} | CF: {avg_cf_loss:.4f} | L1: {avg_l1_loss:.4f} | Preserve: {avg_preserve_loss:.4f} | Change: {change_rate:.2%}')

        # Track best model state (but don't save yet)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_change_rate = change_rate
            best_model_state = model.state_dict().copy()
            print(f"New best loss: {best_loss:.4f} at epoch {epoch+1}")

        # Early stopping
        if avg_loss < best_epoch_loss:
            best_epoch_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= early_stopping_patience and epoch >= 10:
            print(f'Early stopping at epoch {epoch+1}')
            break

    # Save only the best model at the end of the trial
    if best_model_state is not None:
        checkpoint_dir = Path("checkpoints/diffusionModels")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_filename = f"best_advanced_diffusion_{DATA_NAME}_{RECOMMENDER_NAME}_trial{trial.number if not args.manual_trial else 'manual'}.pt"
        checkpoint_path = checkpoint_dir / checkpoint_filename
        torch.save(best_model_state, checkpoint_path)
        print(f"Saved best checkpoint for trial: {checkpoint_path} (loss={best_loss:.4f})")

    # Final evaluation metrics
    final_metrics = {
        "final/best_loss": best_loss,
        "final/best_change_rate": best_change_rate,
        "final/best_epoch": epoch
    }
    wandb.log(final_metrics)
    
    print(f'Finished trial with best loss: {best_loss:.4f}, best change rate: {best_change_rate:.2%}')
    
    # Return optimization metric
    optimization_metric = best_loss - 0.1 * best_change_rate
    
    return optimization_metric

# ========== MAIN OPTIMIZATION ==========
if __name__ == "__main__":
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Setup logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(f"{DATA_NAME}_{RECOMMENDER_NAME}_advanced_diffusion_optimization.log", mode="w"))

    optuna.logging.enable_propagation()
    optuna.logging.disable_default_handler()

    if args.manual_trial:
        print("Running manual trial with fixed hyperparameters")
        result = advanced_diffusion_training(None)
        print(f"Manual trial result: {result}")
    else:
        print(f"Starting advanced Optuna optimization with {args.n_trials} trials")
        
        # Create study
        study = optuna.create_study(direction='minimize')
        
        # Run optimization
        study.optimize(advanced_diffusion_training, n_trials=args.n_trials)
        
        # Print results
        print("Best hyperparameters:", study.best_params)
        print("Best metric value:", study.best_value)
        
        # Save study
        study_path = Path(f"studies/advanced_diffusion_{DATA_NAME}_{RECOMMENDER_NAME}_study.pkl")
        study_path.parent.mkdir(parents=True, exist_ok=True)
        import pickle
        with open(study_path, 'wb') as f:
            pickle.dump(study, f)
        print(f"Study saved to {study_path}")

    wandb.finish() 