import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionMLP(nn.Module):
    """
    A simple MLP-based diffusion model for user embeddings.
    This model is trained to predict the noise added to user embeddings at each diffusion step.
    During sampling, it is used to denoise embeddings and generate counterfactuals.
    Args:
        embedding_dim (int): Dimensionality of the user embedding (latent space)
        hidden_dim (int): Number of hidden units in each layer
    """
    def __init__(self, embedding_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, x, t):
        """
        Forward pass for the diffusion model.
        Args:
            x: torch.Tensor of shape (batch, embedding_dim), the (possibly noised) embedding
            t: torch.Tensor of shape (batch,), the diffusion timestep (int or float)
        Returns:
            torch.Tensor of shape (batch, embedding_dim): predicted noise
        """
        t = t.float().unsqueeze(1) / 1000.0  # Normalize timestep
        xt = torch.cat([x, t], dim=1)
        return self.net(xt) 