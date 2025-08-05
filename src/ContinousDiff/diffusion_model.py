import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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


class SinusoidalPositionEmbedding(nn.Module):
    """
    Sinusoidal positional embedding for diffusion timesteps.
    This helps the model understand the temporal structure of the diffusion process.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class TransformerBlock(nn.Module):
    """
    Transformer block with self-attention and feed-forward layers.
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        
        # Feed-forward
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerDiffusionModel(nn.Module):
    """
    A transformer-based diffusion model for user embeddings.
    This model uses attention mechanisms to better model the relationships
    between different dimensions of the embedding and the diffusion process.
    
    Args:
        embedding_dim (int): Dimensionality of the user embedding (latent space)
        hidden_dim (int): Number of hidden units in transformer layers
        num_layers (int): Number of transformer layers
        num_heads (int): Number of attention heads
        dropout (float): Dropout rate
        time_embed_dim (int): Dimension of time embeddings
    """
    def __init__(self, embedding_dim, hidden_dim=256, num_layers=6, num_heads=8, 
                 dropout=0.1, time_embed_dim=128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.time_embed_dim = time_embed_dim
        
        # Time embedding network
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # Input projection
        self.input_proj = nn.Linear(embedding_dim, hidden_dim)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, mlp_ratio=4, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, embedding_dim)
        
        # Time projection
        self.time_proj = nn.Linear(time_embed_dim, hidden_dim)

    def forward(self, x, t):
        """
        Forward pass for the transformer diffusion model.
        
        Args:
            x: torch.Tensor of shape (batch, embedding_dim), the (possibly noised) embedding
            t: torch.Tensor of shape (batch,), the diffusion timestep (int or float)
            
        Returns:
            torch.Tensor of shape (batch, embedding_dim): predicted noise
        """
        batch_size = x.shape[0]
        
        # Time embedding
        t_emb = self.time_embed(t)
        t_emb = self.time_proj(t_emb)
        
        # Input projection
        h = self.input_proj(x)  # (batch, hidden_dim)
        
        # Add time embedding
        h = h + t_emb
        
        # Reshape for transformer (treat embedding as sequence)
        h = h.unsqueeze(1)  # (batch, 1, hidden_dim)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            h = layer(h)
        
        # Final normalization and output projection
        h = self.norm(h)
        h = h.squeeze(1)  # (batch, hidden_dim)
        output = self.output_proj(h)
        
        return output


class CrossAttentionTransformerDiffusion(nn.Module):
    """
    An advanced transformer diffusion model with cross-attention between
    the embedding and time information.
    
    Args:
        embedding_dim (int): Dimensionality of the user embedding (latent space)
        hidden_dim (int): Number of hidden units in transformer layers
        num_layers (int): Number of transformer layers
        num_heads (int): Number of attention heads
        dropout (float): Dropout rate
        time_embed_dim (int): Dimension of time embeddings
    """
    def __init__(self, embedding_dim, hidden_dim=256, num_layers=6, num_heads=8, 
                 dropout=0.1, time_embed_dim=128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.time_embed_dim = time_embed_dim
        
        # Time embedding network
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # Input projections
        self.embedding_proj = nn.Linear(embedding_dim, hidden_dim)
        self.time_proj = nn.Linear(time_embed_dim, hidden_dim)
        
        # Cross-attention transformer layers
        self.cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        
        # Self-attention layers
        self.self_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        
        # Feed-forward networks
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, int(hidden_dim * 4)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(int(hidden_dim * 4), hidden_dim),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        
        # Layer normalizations
        self.norm1_layers = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.norm2_layers = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.norm3_layers = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x, t):
        """
        Forward pass for the cross-attention transformer diffusion model.
        
        Args:
            x: torch.Tensor of shape (batch, embedding_dim), the (possibly noised) embedding
            t: torch.Tensor of shape (batch,), the diffusion timestep (int or float)
            
        Returns:
            torch.Tensor of shape (batch, embedding_dim): predicted noise
        """
        batch_size = x.shape[0]
        
        # Time embedding
        t_emb = self.time_embed(t)
        t_emb = self.time_proj(t_emb)
        
        # Input projections
        h = self.embedding_proj(x)  # (batch, hidden_dim)
        
        # Reshape for transformer
        h = h.unsqueeze(1)  # (batch, 1, hidden_dim)
        t_emb = t_emb.unsqueeze(1)  # (batch, 1, hidden_dim)
        
        # Apply transformer layers with cross-attention
        for i in range(len(self.cross_attention_layers)):
            # Cross-attention between embedding and time
            h_cross, _ = self.cross_attention_layers[i](
                query=self.norm1_layers[i](h),
                key=self.norm1_layers[i](t_emb),
                value=self.norm1_layers[i](t_emb)
            )
            h = h + h_cross
            
            # Self-attention on embedding
            h_self, _ = self.self_attention_layers[i](
                query=self.norm2_layers[i](h),
                key=self.norm2_layers[i](h),
                value=self.norm2_layers[i](h)
            )
            h = h + h_self
            
            # Feed-forward
            h = h + self.ffn_layers[i](self.norm3_layers[i](h))
        
        # Output projection
        h = h.squeeze(1)  # (batch, hidden_dim)
        output = self.output_proj(h)
        
        return output


# Factory function to create different types of diffusion models
def create_diffusion_model(model_type="transformer", **kwargs):
    """
    Factory function to create different types of diffusion models.
    
    Args:
        model_type (str): Type of model to create ("mlp", "transformer", "cross_attention")
        **kwargs: Arguments to pass to the model constructor
        
    Returns:
        nn.Module: The created diffusion model
    """
    if model_type == "mlp":
        return DiffusionMLP(**kwargs)
    elif model_type == "transformer":
        return TransformerDiffusionModel(**kwargs)
    elif model_type == "cross_attention":
        return CrossAttentionTransformerDiffusion(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}") 