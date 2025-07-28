import torch
import numpy as np
from pathlib import Path
from diffusion_model import DiffusionMLP

# === CONFIGURATION ===
EMBEDDING_PATH = Path('user_embeddings.npy')
MODEL_SAVE_PATH = Path('diffusion_mlp_best.pth')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TIMESTEPS = 1000  # Number of diffusion steps
BATCH_SIZE = 128
EPOCHS = 500
LEARNING_RATE = 1e-3

# === Load Data ===
# Load user embeddings extracted from VAE
embeddings = np.load(EMBEDDING_PATH)
embeddings = torch.tensor(embeddings, dtype=torch.float32, device=DEVICE)
embedding_dim = embeddings.shape[1]

# === Diffusion Schedule ===
# Standard DDPM linear noise schedule
betas = torch.linspace(1e-4, 0.02, TIMESTEPS, device=DEVICE)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

# === Model ===
# MLP-based diffusion model for user embeddings
model = DiffusionMLP(embedding_dim).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.MSELoss()

best_loss = float('inf')

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
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Saved new best model to {MODEL_SAVE_PATH}") 