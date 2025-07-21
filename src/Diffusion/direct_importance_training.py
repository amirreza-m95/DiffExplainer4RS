#!/usr/bin/env python3
"""
Direct importance training approach.
This file trains a model specifically to output importance scores using gradient-based importance as supervision.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.LXR.recommenders_architecture import VAE
from src.LXR.help_functions import get_user_recommended_item

# ========== CONFIGURATION ==========
DATA_PATH = Path('datasets/lxr-CE/ML1M/train_data_ML1M.csv')
CHECKPOINT_PATH = Path('checkpoints/recommenders/VAE_ML1M_0_19_128.pt')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USER_HISTORY_DIM = 3381
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001

class ImportanceMLP(nn.Module):
    """
    Neural network specifically designed to predict importance scores
    """
    def __init__(self, input_dim, hidden_dim=512, num_layers=3):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.2))
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
        
        layers.append(nn.Linear(hidden_dim, input_dim))
        layers.append(nn.Sigmoid())  # Output importance scores between 0 and 1
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def load_data():
    """Load training data"""
    df = pd.read_csv(DATA_PATH, index_col=0)
    if 'user_id' in df.columns:
        df = df.drop(columns=['user_id'])
    data = df.values.astype(np.float32)
    return data

def load_recommender():
    """Load the pre-trained recommender"""
    recommender = VAE(USER_HISTORY_DIM, 128, 64, 0.19, DEVICE)
    recommender.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    recommender.eval()
    return recommender

def compute_gradient_importance(user_tensor, recommender, item_id):
    """
    Compute gradient-based importance scores for supervision
    """
    user_tensor.requires_grad_(True)
    
    # Get recommendation score for the target item
    scores = recommender(user_tensor.unsqueeze(0))
    target_score = scores[0, item_id]
    
    # Compute gradients
    target_score.backward()
    gradient_importance = user_tensor.grad.abs()
    
    # Normalize to [0, 1] range
    max_grad = gradient_importance.max()
    if max_grad > 0:
        gradient_importance = gradient_importance / max_grad
    
    return gradient_importance.detach()

def compute_removal_importance(user_tensor, recommender, item_id):
    """
    Compute importance by measuring how much removing each item affects the recommendation
    """
    original_score = recommender(user_tensor.unsqueeze(0))[0, item_id].item()
    
    removal_effects = torch.zeros_like(user_tensor)
    user_history = torch.where(user_tensor > 0)[0]
    
    for item_idx in user_history:
        # Create modified user tensor with item removed
        modified_tensor = user_tensor.clone()
        modified_tensor[item_idx] = 0
        
        # Get new score
        new_score = recommender(modified_tensor.unsqueeze(0))[0, item_id].item()
        
        # Importance = how much score changes
        importance = abs(original_score - new_score)
        removal_effects[item_idx] = importance
    
    # Normalize to [0, 1] range
    max_effect = removal_effects.max()
    if max_effect > 0:
        removal_effects = removal_effects / max_effect
    
    return removal_effects

def generate_training_data(data, recommender, num_samples=1000):
    """
    Generate training data with importance labels
    """
    print("Generating training data...")
    
    # Sample users
    num_users = min(num_samples, data.shape[0])
    user_indices = np.random.choice(data.shape[0], num_users, replace=False)
    
    training_data = []
    
    for i, user_idx in enumerate(user_indices):
        if i % 100 == 0:
            print(f"Processing user {i}/{num_users}")
        
        # Get user data
        user_vector = data[user_idx]
        user_tensor = torch.FloatTensor(user_vector).to(DEVICE)
        
        # Get recommended item
        item_id = int(get_user_recommended_item(user_tensor, recommender).cpu().detach().numpy())
        
        # Compute importance labels
        gradient_importance = compute_gradient_importance(user_tensor, recommender, item_id)
        removal_importance = compute_removal_importance(user_tensor, recommender, item_id)
        
        # Use gradient importance as primary label, removal importance as secondary
        importance_label = (gradient_importance + removal_importance) / 2
        
        training_data.append({
            'user_history': user_tensor,
            'importance_label': importance_label,
            'item_id': item_id
        })
    
    return training_data

def train_importance_model(training_data):
    """
    Train the importance prediction model
    """
    print("Training importance model...")
    
    # Initialize model
    model = ImportanceMLP(USER_HISTORY_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    # Training loop
    model.train()
    losses = []
    
    for epoch in range(EPOCHS):
        epoch_loss = 0
        num_batches = 0
        
        # Shuffle data
        np.random.shuffle(training_data)
        
        for i in range(0, len(training_data), BATCH_SIZE):
            batch_data = training_data[i:i+BATCH_SIZE]
            
            batch_loss = 0
            for sample in batch_data:
                user_history = sample['user_history']
                importance_label = sample['importance_label']
                
                # Forward pass
                predicted_importance = model(user_history)
                
                # Only compute loss for items in user history
                user_mask = (user_history > 0).float()
                masked_prediction = predicted_importance * user_mask
                masked_label = importance_label * user_mask
                
                # Compute loss
                loss = criterion(masked_prediction, masked_label)
                batch_loss += loss
            
            # Backward pass
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
            epoch_loss += batch_loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}/{EPOCHS}, Loss: {avg_loss:.6f}")
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('importance_training_loss.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return model, losses

def evaluate_importance_model(model, test_data, recommender):
    """
    Evaluate the trained importance model
    """
    print("Evaluating importance model...")
    
    model.eval()
    correlations = []
    
    with torch.no_grad():
        for i, sample in enumerate(test_data[:100]):  # Evaluate on subset
            user_history = sample['user_history']
            true_importance = sample['importance_label']
            
            # Get model prediction
            predicted_importance = model(user_history)
            
            # Compute correlation
            user_mask = (user_history > 0).float()
            masked_pred = (predicted_importance * user_mask).cpu().numpy()
            masked_true = (true_importance * user_mask).cpu().numpy()
            
            # Get non-zero elements
            non_zero_mask = masked_true > 0
            if np.sum(non_zero_mask) > 0:
                pred_values = masked_pred[non_zero_mask]
                true_values = masked_true[non_zero_mask]
                
                correlation = np.corrcoef(pred_values, true_values)[0, 1]
                if not np.isnan(correlation):
                    correlations.append(correlation)
    
    avg_correlation = np.mean(correlations)
    print(f"Average correlation with true importance: {avg_correlation:.3f}")
    
    return avg_correlation

def save_importance_model(model, save_path):
    """Save the trained importance model"""
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def main():
    """
    Main training function
    """
    print("=== Direct Importance Training ===")
    
    # Load data and recommender
    data = load_data()
    recommender = load_recommender()
    
    # Generate training data
    training_data = generate_training_data(data, recommender, num_samples=2000)
    
    # Split into train and test
    split_idx = int(0.8 * len(training_data))
    train_data = training_data[:split_idx]
    test_data = training_data[split_idx:]
    
    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Train model
    model, losses = train_importance_model(train_data)
    
    # Evaluate model
    correlation = evaluate_importance_model(model, test_data, recommender)
    
    # Save model
    save_path = Path('checkpoints/diffusion/importance_model.pt')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_importance_model(model, save_path)
    
    print("=== Training Complete ===")
    print(f"Final correlation: {correlation:.3f}")
    
    return model, correlation

if __name__ == "__main__":
    main() 