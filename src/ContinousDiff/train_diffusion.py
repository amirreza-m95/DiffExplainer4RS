import torch
import numpy as np
from pathlib import Path
import argparse
from diffusion_model import DiffusionMLP
from diffusion_model import TransformerDiffusionModel

# === DATASET CONFIGURATIONS ===
DATASET_CONFIGS = {
    'ml1m': {
        'embedding_path': 'checkpoints/embeddings/user_embeddings_ml1m.npy',
        'model_save_path': 'checkpoints/diffusionModels/diffusion_transformer_ml1m_best.pth',
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
        'model_save_path': 'checkpoints/diffusionModels/diffusion_transformer_pinterest_best.pth',
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
TIMESTEPS = 1000  # Number of diffusion steps
BATCH_SIZE = 128
EPOCHS = 500
LEARNING_RATE = 1e-3

def create_model(model_type, embedding_dim, model_config):
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

def train_diffusion(dataset_name):
    """
    Train diffusion model for a specific dataset.
    Args:
        dataset_name: Name of the dataset ('ml1m' or 'pinterest')
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Dataset '{dataset_name}' not supported. Available datasets: {list(DATASET_CONFIGS.keys())}")
    
    config = DATASET_CONFIGS[dataset_name]
    
    print(f"Training diffusion model for {dataset_name} dataset...")
    print(f"Embedding path: {config['embedding_path']}")
    print(f"Model type: {config['model_type']}")
    print(f"Model save path: {config['model_save_path']}")
    
    # === Load Data ===
    # Load user embeddings extracted from VAE
    embeddings = np.load(config['embedding_path'])
    embeddings = torch.tensor(embeddings, dtype=torch.float32, device=DEVICE)
    embedding_dim = embeddings.shape[1]
    
    print(f"Loaded {embeddings.shape[0]} embeddings with dimension {embedding_dim}")
    
    # === Diffusion Schedule ===
    # Standard DDPM linear noise schedule
    betas = torch.linspace(1e-4, 0.02, TIMESTEPS, device=DEVICE)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    
    # === Model ===
    model = create_model(config['model_type'], embedding_dim, config['model_config'])
    print(f"Created {config['model_type']} model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.MSELoss()
    
    best_loss = float('inf')
    
    print(f"Starting training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        perm = torch.randperm(embeddings.shape[0])
        total_loss = 0.0
        for i in range(0, embeddings.shape[0], BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            x0 = embeddings[idx]
            batch_size = x0.shape[0]
            # Sample random timesteps for each sample in the batch
            t = torch.randint(0, TIMESTEPS, (batch_size,), device=DEVICE)
            # Sample noise
            noise = torch.randn_like(x0)
            # Apply forward diffusion process at timestep t
            noised = (
                sqrt_alphas_cumprod[t].unsqueeze(1) * x0 +
                sqrt_one_minus_alphas_cumprod[t].unsqueeze(1) * noise
            )
            # Predict the noise
            pred_noise = model(noised, t)
            # Loss: MSE between predicted and true noise
            loss = loss_fn(pred_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_size
        avg_loss = total_loss / embeddings.shape[0]
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.6f}")
        # Save the best model checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), config['model_save_path'])
            print(f"Saved new best model to {config['model_save_path']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train diffusion model for different datasets')
    parser.add_argument('--dataset', type=str, default='ml1m', 
                       choices=['ml1m', 'pinterest'],
                       help='Dataset to train diffusion model for (default: ml1m)')
    
    args = parser.parse_args()
    
    # Train diffusion model for the specified dataset
    train_diffusion(args.dataset) 