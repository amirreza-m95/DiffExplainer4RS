"""
Standalone Autoencoder for Model-Agnostic DiceRec

This autoencoder maps user interaction vectors to a latent space where diffusion operates.
It's independent of the recommender system, making DiceRec model-agnostic.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Autoencoder(nn.Module):
    """
    Standalone autoencoder that learns to compress user interaction vectors.

    Architecture:
    - Encoder: User interaction vector (num_items) -> Latent embedding (latent_dim)
    - Decoder: Latent embedding (latent_dim) -> Reconstructed user vector (num_items)

    This is similar to VAE but without the variational component for simplicity.
    Can be easily extended to VAE if needed.
    """

    def __init__(self, num_items, latent_dim=256, hidden_dims=[256, 256], dropout=0.5, device='cpu'):
        """
        Args:
            num_items: Number of items in the dataset (dimension of user vector)
            latent_dim: Dimension of the latent embedding space
            hidden_dims: List of hidden layer dimensions for encoder/decoder
            dropout: Dropout rate
            device: Device to run on
        """
        super(Autoencoder, self).__init__()
        self.device = device
        self.num_items = num_items
        self.latent_dim = latent_dim
        self.dropout = dropout

        # Build encoder: num_items -> hidden_dims[0] -> hidden_dims[1] -> latent_dim
        self.enc_dims = [num_items] + hidden_dims + [latent_dim]
        self.encoder = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(zip(self.enc_dims[:-1], self.enc_dims[1:])):
            self.encoder.append(nn.Linear(d_in, d_out))
            if i != len(self.enc_dims[:-1]) - 1:  # No activation after last layer
                self.encoder.append(nn.ReLU())

        # Build decoder: latent_dim -> hidden_dims[1] -> hidden_dims[0] -> num_items
        self.dec_dims = [latent_dim] + hidden_dims[::-1] + [num_items]
        self.decoder = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(zip(self.dec_dims[:-1], self.dec_dims[1:])):
            self.decoder.append(nn.Linear(d_in, d_out))
            if i != len(self.dec_dims[:-1]) - 1:  # No activation after last layer
                self.decoder.append(nn.ReLU())

        self.to(self.device)

    def encode(self, x):
        """
        Encode user interaction vector to latent embedding.

        Args:
            x: User interaction vector of shape (batch_size, num_items) or (num_items,)
        Returns:
            Latent embedding of shape (batch_size, latent_dim) or (latent_dim,)
        """
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        # Normalize and apply dropout
        h = F.normalize(x, dim=-1)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Pass through encoder
        for layer in self.encoder:
            h = layer(h)

        return h.squeeze(0) if x.shape[0] == 1 else h

    def decode(self, z):
        """
        Decode latent embedding to user interaction vector.

        Args:
            z: Latent embedding of shape (batch_size, latent_dim) or (latent_dim,)
        Returns:
            Reconstructed user vector of shape (batch_size, num_items) or (num_items,)
        """
        if len(z.shape) == 1:
            z = z.unsqueeze(0)

        # Pass through decoder
        h = z
        for layer in self.decoder:
            h = layer(h)

        return h.squeeze(0) if z.shape[0] == 1 else h

    def forward(self, x):
        """
        Full forward pass: encode then decode.

        Args:
            x: User interaction vector
        Returns:
            Reconstructed user vector
        """
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon

    def train_one_epoch(self, dataset, optimizer, batch_size, loss_type='bce'):
        """
        Train autoencoder for one epoch.

        Args:
            dataset: User interaction matrix (num_users, num_items)
            optimizer: PyTorch optimizer
            batch_size: Batch size for training
            loss_type: 'bce' (binary cross entropy) or 'mse' (mean squared error)
        Returns:
            Average loss for the epoch
        """
        self.train()

        num_users = dataset.shape[0]
        num_batches = int(torch.ceil(torch.tensor(num_users / batch_size)))
        perm = torch.randperm(num_users)

        total_loss = 0.0
        for b in range(num_batches):
            optimizer.zero_grad()

            if (b + 1) * batch_size >= num_users:
                batch_idx = perm[b * batch_size:]
            else:
                batch_idx = perm[b * batch_size: (b + 1) * batch_size]

            batch_data = torch.FloatTensor(dataset[batch_idx]).to(self.device)

            # Forward pass
            recon = self.forward(batch_data)

            # Compute loss
            if loss_type == 'bce':
                # Binary cross entropy (good for implicit feedback)
                loss = F.binary_cross_entropy_with_logits(recon, batch_data, reduction='mean')
            elif loss_type == 'mse':
                # Mean squared error
                loss = F.mse_loss(recon, batch_data, reduction='mean')
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_data.shape[0]

            if b % 50 == 0:
                print(f'  Batch ({b:3d} / {num_batches}) loss = {loss.item():.4f}')

        avg_loss = total_loss / num_users
        return avg_loss

    def predict(self, user_vectors, batch_size=256):
        """
        Reconstruct user vectors (for evaluation).

        Args:
            user_vectors: User interaction matrix (num_users, num_items)
            batch_size: Batch size for prediction
        Returns:
            Reconstructed user vectors
        """
        self.eval()

        with torch.no_grad():
            input_matrix = torch.FloatTensor(user_vectors).to(self.device)
            preds = torch.zeros_like(input_matrix)

            num_data = input_matrix.shape[0]
            num_batches = int(torch.ceil(torch.tensor(num_data / batch_size)))

            for b in range(num_batches):
                if (b + 1) * batch_size >= num_data:
                    batch_idx = list(range(b * batch_size, num_data))
                else:
                    batch_idx = list(range(b * batch_size, (b + 1) * batch_size))

                batch_data = input_matrix[batch_idx]
                batch_recon = self.forward(batch_data)
                preds[batch_idx] = batch_recon

            return preds.cpu().numpy()


class VariationalAutoencoder(Autoencoder):
    """
    Variational Autoencoder (VAE) version.

    Extends the base Autoencoder with variational inference.
    The encoder outputs mean and log-variance, and sampling is performed.
    """

    def __init__(self, num_items, latent_dim=256, hidden_dims=[256, 256], dropout=0.5,
                 anneal_cap=0.2, total_anneal_steps=200000, device='cpu'):
        """
        Args:
            num_items: Number of items in the dataset
            latent_dim: Dimension of the latent embedding space
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
            anneal_cap: KL annealing capacity
            total_anneal_steps: Total steps for KL annealing
            device: Device to run on
        """
        super(VariationalAutoencoder, self).__init__(
            num_items=num_items,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            device=device
        )

        self.anneal_cap = anneal_cap
        self.total_anneal_steps = total_anneal_steps
        self.anneal = 0.0
        self.update_count = 0

        # Modify encoder to output 2x latent_dim (mu and logvar)
        # Replace the last encoder layer
        self.encoder = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(zip(self.enc_dims[:-1], self.enc_dims[1:])):
            if i == len(self.enc_dims[:-1]) - 1:
                # Last layer outputs 2x for mu and logvar
                self.encoder.append(nn.Linear(d_in, d_out * 2))
            else:
                self.encoder.append(nn.Linear(d_in, d_out))
                self.encoder.append(nn.ReLU())

        self.to(self.device)

    def encode(self, x):
        """
        Encode to latent distribution parameters (mu, logvar).

        Args:
            x: User interaction vector
        Returns:
            Sampled latent embedding (during training) or mu (during inference)
        """
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Normalize and dropout
        h = F.normalize(x, dim=-1)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Pass through encoder
        for layer in self.encoder:
            h = layer(h)

        # Split into mu and logvar
        mu = h[:, :self.latent_dim]
        logvar = h[:, self.latent_dim:]

        if self.training:
            # Sample during training
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std) * 0.01
            z = mu + eps * std
            # Store for KL computation
            self.mu = mu
            self.logvar = logvar
        else:
            # Use mean during inference
            z = mu

        return z.squeeze(0) if squeeze_output else z

    def train_one_epoch(self, dataset, optimizer, batch_size):
        """
        Train VAE for one epoch with KL annealing.

        Args:
            dataset: User interaction matrix
            optimizer: PyTorch optimizer
            batch_size: Batch size
        Returns:
            Average loss for the epoch
        """
        self.train()

        num_users = dataset.shape[0]
        num_batches = int(torch.ceil(torch.tensor(num_users / batch_size)))
        perm = torch.randperm(num_users)

        total_loss = 0.0
        for b in range(num_batches):
            optimizer.zero_grad()

            if (b + 1) * batch_size >= num_users:
                batch_idx = perm[b * batch_size:]
            else:
                batch_idx = perm[b * batch_size: (b + 1) * batch_size]

            batch_data = torch.FloatTensor(dataset[batch_idx]).to(self.device)

            # Update KL annealing
            if self.total_anneal_steps > 0:
                self.anneal = min(self.anneal_cap, 1.0 * self.update_count / self.total_anneal_steps)
            else:
                self.anneal = self.anneal_cap

            # Forward pass
            recon = self.forward(batch_data)

            # Reconstruction loss (binary cross entropy)
            recon_loss = F.binary_cross_entropy_with_logits(recon, batch_data, reduction='mean')

            # KL divergence loss
            kl_loss = -0.5 * torch.sum(1 + self.logvar - self.mu.pow(2) - self.logvar.exp(), dim=1).mean()

            # Total loss
            loss = recon_loss + self.anneal * kl_loss

            # Backward pass
            loss.backward()
            optimizer.step()

            self.update_count += 1
            total_loss += loss.item() * batch_data.shape[0]

            if b % 50 == 0:
                print(f'  Batch ({b:3d} / {num_batches}) loss = {loss.item():.4f} (recon: {recon_loss.item():.4f}, kl: {kl_loss.item():.4f}, anneal: {self.anneal:.4f})')

        avg_loss = total_loss / num_users
        return avg_loss
