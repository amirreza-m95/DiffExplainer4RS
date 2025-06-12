import pandas as pd
import numpy as np
import os
from os import path
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import argparse
import time
import logging
import sys
from datetime import datetime
from typing import Dict, Tuple, List, Any, Optional
from tqdm import tqdm

# Initialize global logger
logger = logging.getLogger(__name__)

# Type aliases
UserID = int
ItemID = int
Rating = float
Timestamp = int
SimilarityDict = Dict[Tuple[ItemID, ItemID], float]
TFIDFDict = Dict[UserID, Dict[ItemID, float]]

def parse_args() -> argparse.Namespace:
    """Parse command line arguments for data preprocessing.
    
    Returns:
        argparse.Namespace: Parsed command line arguments with the following fields:
            - dataset (str): Name of the dataset to process ('ML1M', 'Yahoo', or 'Pinterest')
            - min_items (int): Minimum number of items per user
            - min_users (int): Minimum number of users per item
            - test_size (float): Proportion of data to use for testing
            - random_seed (int): Random seed for reproducibility
    """
    parser = argparse.ArgumentParser(
        description='Preprocess recommendation system datasets',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--dataset', 
        type=str, 
        default='ML1M',
        choices=['ML1M', 'Yahoo', 'Pinterest'],
        help='Dataset to process'
    )
    
    parser.add_argument(
        '--min_items', 
        type=int, 
        default=2,
        help='Minimum number of items per user'
    )
    
    parser.add_argument(
        '--min_users', 
        type=int, 
        default=2,
        help='Minimum number of users per item'
    )
    
    parser.add_argument(
        '--test_size', 
        type=float, 
        default=0.2,
        help='Test set size ratio'
    )
    
    parser.add_argument(
        '--random_seed', 
        type=int, 
        default=42,
        help='Random seed for train-test split'
    )
    
    return parser.parse_args()

def load_dataset(data_name: str, files_path: Path) -> pd.DataFrame:
    """Load and validate a recommendation dataset.
    
    Args:
        data_name: Name of the dataset to load ('ML1M', 'Yahoo', or 'Pinterest')
        files_path: Path to the directory containing the dataset files
    
    Returns:
        DataFrame with columns: user_id_original, item_id_original, rating, [timestamp]
    
    Raises:
        FileNotFoundError: If the dataset file is not found
        ValueError: If the dataset name is not recognized
    """
    logger.info(f"Loading {data_name} dataset from {files_path}")
    
    try:
        if data_name == "ML1M":
            data = pd.read_csv(
                Path(files_path, "ratings.dat"), 
                sep="::", 
                engine="python",
                names=["user_id_original", "item_id_original", "rating", "timestamp"]
            )
        elif data_name == "Yahoo":
            data = pd.read_csv(
                Path(files_path, "Yahoo_ratings.csv"),
                names=["user_id_original", "item_id_original", "rating"]
            )
        elif data_name == "Pinterest":
            data = pd.read_csv(
                Path(files_path, "pinterest_data.csv"),
                names=["user_id_original", "item_id_original", "rating"]
            )
        else:
            raise ValueError(f"Unknown dataset: {data_name}")
        
        logger.info(f"Successfully loaded {len(data):,} ratings")
        return data
        
    except FileNotFoundError as e:
        logger.error(f"Dataset file not found in {files_path}")
        raise

def preprocess_ratings(data: pd.DataFrame, data_name: str) -> pd.DataFrame:
    """Convert ratings to binary values and filter positive interactions.
    
    Args:
        data: DataFrame with raw ratings
        data_name: Name of the dataset being processed
    
    Returns:
        DataFrame containing only positive interactions with binary ratings
    """
    logger.info("Converting ratings to binary values")
    
    if data_name == 'Yahoo':
        # For Yahoo, 255 indicates missing values
        data["rating"] = data["rating"].apply(lambda x: 0 if x == 255 else x)
        # Ratings > 70 are considered positive
        data["rating"] = data["rating"].apply(lambda x: 1 if x > 70 else 0)
    elif data_name in ['ML1M', "ML1M_demographic"]:
        # For MovieLens, ratings > 3.5 are considered positive
        data["rating"] = data["rating"].apply(lambda x: 1 if x > 3.5 else 0)
    
    # Keep only positive interactions
    filtered_data = data[data['rating'] == 1]
    logger.info(f"Kept {len(filtered_data):,} positive interactions out of {len(data):,} total")
    
    return filtered_data

def filter_interactions(
    data: pd.DataFrame, 
    min_items_per_user: int, 
    min_users_per_item: int
) -> pd.DataFrame:
    """Filter users and items based on minimum interaction thresholds.
    
    This function iteratively filters users and items until convergence,
    ensuring each user has at least min_items_per_user items and
    each item has at least min_users_per_item users.
    
    Args:
        data: DataFrame with user-item interactions
        min_items_per_user: Minimum number of items per user
        min_users_per_item: Minimum number of users per item
    
    Returns:
        Filtered DataFrame where all users and items meet minimum thresholds
    """
    logger.info("Filtering users and items based on interaction thresholds")
    initial_users = data['user_id_original'].nunique()
    initial_items = data['item_id_original'].nunique()
    
    num_rows_1, num_rows_2 = 1, 2
    iteration = 0
    
    while num_rows_1 != num_rows_2:
        iteration += 1
        logger.debug(f"Filtering iteration {iteration}")
        
        # Filter users
        user_counts = data.groupby(['user_id_original'])['item_id_original'].nunique() \
                         .reset_index(name='item_count')
        filtered_users = user_counts[user_counts['item_count'] >= min_items_per_user]['user_id_original']
        data = data[data['user_id_original'].isin(filtered_users)].reset_index(drop=True)
        num_rows_1 = data.shape[0]
        
        # Filter items
        item_counts = data.groupby(['item_id_original'])['user_id_original'].nunique() \
                         .reset_index(name='user_count')
        filtered_items = item_counts[item_counts['user_count'] >= min_users_per_item]['item_id_original']
        data = data[data['item_id_original'].isin(filtered_items)].reset_index(drop=True)
        num_rows_2 = data.shape[0]
    
    final_users = data['user_id_original'].nunique()
    final_items = data['item_id_original'].nunique()
    
    logger.info(
        f"Filtering complete after {iteration} iterations\n"
        f"Users: {initial_users:,} -> {final_users:,}\n"
        f"Items: {initial_items:,} -> {final_items:,}"
    )
    
    return data

def encode_ids(data: pd.DataFrame) -> Tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
    """Encode user and item IDs to continuous integer indices.
    
    Args:
        data: DataFrame with user_id_original and item_id_original columns
    
    Returns:
        Tuple containing:
        - DataFrame with additional user_id and item_id columns
        - User ID encoder
        - Item ID encoder
    """
    logger.info("Encoding user and item IDs")
    
    item_encoder = LabelEncoder()
    user_encoder = LabelEncoder()
    
    user_encoder.fit(data.user_id_original)
    item_encoder.fit(data.item_id_original)
    
    data["user_id"] = user_encoder.transform(data.user_id_original)
    data["item_id"] = item_encoder.transform(data.item_id_original)
    
    logger.info(
        f"Encoded {len(user_encoder.classes_):,} users and "
        f"{len(item_encoder.classes_):,} items"
    )
    
    return data, user_encoder, item_encoder

def create_user_item_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """Create a binary user-item interaction matrix.
    
    Args:
        data: DataFrame with user_id and item_id columns
    
    Returns:
        DataFrame where:
        - Each row represents a user
        - Each column represents an item
        - Values are 1 for interactions, 0 otherwise
        - Last column contains the user_id
    """
    logger.info("Creating user-item interaction matrix")
    
    # Group items by user
    user_group = data[["user_id", "item_id"]].groupby(data.user_id)
    users_data = pd.DataFrame(
        data={
            "user_id": list(user_group.groups.keys()),
            "item_ids": list(user_group.item_id.apply(list)),
        }    
    )
    
    # Convert to one-hot encoding
    mlb = MultiLabelBinarizer()
    user_one_hot = pd.DataFrame(
        mlb.fit_transform(users_data["item_ids"]),
        columns=mlb.classes_, 
        index=users_data["item_ids"].index
    )
    user_one_hot["user_id"] = users_data["user_id"]
    
    logger.info(
        f"Created matrix with {user_one_hot.shape[0]:,} users × "
        f"{user_one_hot.shape[1]-1:,} items"
    )
    
    return user_one_hot

def create_train_test_split(
    user_one_hot: pd.DataFrame, 
    test_size: float, 
    random_seed: int, 
    num_users: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split user-item matrix into training and test sets.
    
    Args:
        user_one_hot: User-item interaction matrix
        test_size: Proportion of data to use for testing
        random_seed: Random seed for reproducibility
        num_users: Total number of users
    
    Returns:
        Tuple containing:
        - Training data matrix
        - Test data matrix
    """
    logger.info(f"Creating train-test split with {test_size:.0%} test size")
    
    X_train, X_test, y_train, y_test = train_test_split(
        user_one_hot.iloc[:,:-1], 
        user_one_hot.iloc[:,-1], 
        test_size=test_size, 
        random_state=random_seed
    )
    
    X_train.reset_index(drop=True, inplace=True)
    X_test.index = np.arange(X_train.shape[0], num_users)
    
    logger.info(
        f"Split complete:\n"
        f"Train: {X_train.shape[0]:,} users × {X_train.shape[1]:,} items\n"
        f"Test: {X_test.shape[0]:,} users × {X_test.shape[1]:,} items"
    )
    
    return X_train, X_test

def compute_jaccard_similarity(data_array: np.ndarray, num_features: int) -> SimilarityDict:
    """Compute Jaccard similarity between all pairs of items.
    
    Args:
        data_array: Binary user-item interaction matrix
        num_features: Number of items
    
    Returns:
        Dictionary mapping (item1_id, item2_id) tuples to similarity scores
    """
    logger.info("Computing Jaccard similarities")
    jaccard_dict = {}
    
    # Use tqdm for progress tracking
    total_pairs = (num_features * (num_features - 1)) // 2 + num_features  # Include diagonal
    with tqdm(total=total_pairs, desc="Computing Jaccard") as pbar:
        for i in range(num_features):
            for j in range(i, num_features):
                if j >= data_array.shape[1]:  # Skip if index out of bounds
                    continue
                intersection = (data_array[:,i]*data_array[:,j]).sum()
                union = np.count_nonzero(data_array[:,i]+data_array[:,j])
                if union == 0:
                    jaccard_dict[(i,j)] = 0
                else:
                    jaccard_dict[(i,j)] = (intersection/union).astype('float32')
                pbar.update(1)
    
    logger.info(f"Computed {len(jaccard_dict):,} Jaccard similarities")
    return jaccard_dict

def compute_cosine_similarity(X_train: pd.DataFrame) -> SimilarityDict:
    """Compute cosine similarity between all pairs of items.
    
    Args:
        X_train: Training data matrix where rows are users and columns are items
    
    Returns:
        Dictionary mapping (item1_id, item2_id) tuples to similarity scores
    """
    logger.info("Computing cosine similarities")
    
    # Compute full similarity matrix
    cosine_items = cosine_similarity(X_train.T).astype('float32')
    
    # Convert to dictionary format
    cosine_items_dict = {}
    n_items = cosine_items.shape[0]
    
    # Use tqdm for progress tracking
    total_pairs = (n_items * (n_items + 1)) // 2
    with tqdm(total=total_pairs, desc="Converting cosine") as pbar:
        for i in range(n_items):
            for j in range(i, n_items):
                cosine_items_dict[(i, j)] = cosine_items[i][j]
                pbar.update(1)
    
    logger.info(f"Computed {len(cosine_items_dict):,} cosine similarities")
    return cosine_items_dict

def compute_popularity(X_train: pd.DataFrame, num_items: int) -> Dict[ItemID, float]:
    """Compute normalized popularity scores for all items.
    
    Args:
        X_train: Training data matrix
        num_items: Number of items
    
    Returns:
        Dictionary mapping item IDs to normalized popularity scores
    """
    logger.info("Computing item popularity scores")
    
    # Compute normalized popularity scores
    pop_array = (X_train.sum(axis=0)/X_train.sum(axis=0).max()).astype('float32')
    pop_dict = {i: pop_array[i] for i in range(num_items)}
    
    logger.info(f"Computed popularity scores for {len(pop_dict):,} items")
    return pop_dict

def compute_tf_idf(
    data_array: np.ndarray, 
    user_one_hot: pd.DataFrame, 
    num_users: int, 
    num_items: int
) -> TFIDFDict:
    """Compute TF-IDF scores for user-item pairs.
    
    Args:
        data_array: Binary user-item interaction matrix (training set only)
        user_one_hot: User-item interaction DataFrame
        num_users: Total number of users
        num_items: Number of items
    
    Returns:
        Nested dictionary mapping user IDs to dictionaries of item TF-IDF scores
    """
    logger.info("Computing TF-IDF scores")
    
    # Get actual number of users in training set
    n_train_users = data_array.shape[0]
    
    # Compute item counts per user and item frequencies
    w_count = user_one_hot.iloc[:,:-1].sum(axis=1)
    n_appearance = user_one_hot.iloc[:,:-1].sum(axis=0)
    
    # Initialize TF-IDF dictionary
    tf_idf_dict = defaultdict(dict)
    
    # Use tqdm for progress tracking
    with tqdm(total=n_train_users, desc="Computing TF-IDF") as pbar:
        for u in range(n_train_users):  # Only iterate over training users
            for i in range(num_items):
                if data_array[u,i] == 1:
                    tf = 1/w_count[u]
                    idf = np.log10(num_users/n_appearance[i])
                    tf_idf_dict[u][i] = tf*idf
            pbar.update(1)
    
    logger.info(
        f"Computed TF-IDF scores for {len(tf_idf_dict):,} users × "
        f"{sum(len(items) for items in tf_idf_dict.values()):,} items"
    )
    
    return tf_idf_dict

def save_processed_data(
    data_dict: Dict[str, Any], 
    files_path: Path, 
    data_name: str
) -> None:
    """Save all processed data and computed metrics.
    
    Args:
        data_dict: Dictionary containing data to save with keys:
            - X_train: Training data matrix
            - X_test: Test data matrix
            - jaccard_based_sim: Jaccard similarities
            - cosine_based_sim: Cosine similarities
            - pop_dict: Item popularity scores
            - tf_idf_dict: TF-IDF scores
        files_path: Directory to save files in
        data_name: Name of the dataset
    """
    logger.info(f"Saving processed data to {files_path}")
    
    try:
        for name, data in data_dict.items():
            if name in ['X_train', 'X_test']:
                file_path = Path(files_path, f'{name.lower()}_{data_name}.csv')
                data.to_csv(file_path)
                logger.info(f"Saved {name} to {file_path}")
            else:
                file_path = Path(files_path, f'{name}_{data_name}.pkl')
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f)
                logger.info(f"Saved {name} to {file_path}")
    except Exception as e:
        logger.error(f"Error saving data: {str(e)}")
        raise

def setup_logging(dataset_dir: Path) -> None:
    """Setup logging configuration to save logs in the dataset directory."""
    global logger
    
    # Create dataset directory if it doesn't exist
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a log filename with timestamp
    log_filename = dataset_dir / f"preprocessing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Configure logging
    handlers = [
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
    
    # Set format for all handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    logger.setLevel(logging.INFO)
    logger.info(f"Logging to {log_filename}")

def main():
    """Main preprocessing pipeline."""
    # Parse arguments first (before setting up logging)
    args = parse_args()
    
    # Setup paths
    DP_DIR = Path("datasets", args.dataset)
    export_dir = Path(os.getcwd())
    files_path = Path(export_dir, DP_DIR)
    
    # Setup logging in the dataset directory
    setup_logging(files_path)
    
    start_time = time.time()
    logger.info("Starting preprocessing pipeline")
    
    try:
        # Load and preprocess data
        data = load_dataset(args.dataset, files_path)
        data = preprocess_ratings(data, args.dataset)
        data = filter_interactions(data, args.min_items, args.min_users)
        
        # Encode IDs
        data, user_encoder, item_encoder = encode_ids(data)
        num_users = data.user_id.unique().shape[0]
        num_items = data.item_id.unique().shape[0]
        
        # Create user-item matrix and train-test split
        user_one_hot = create_user_item_matrix(data)
        X_train, X_test = create_train_test_split(
            user_one_hot, args.test_size, args.random_seed, num_users
        )
        
        # Convert to numpy array for faster computation
        data_array = X_train.iloc[:,:-1].values
        n_items = data_array.shape[1]  # Use actual number of columns
        n_train_users = data_array.shape[0]  # Number of users in training set
        
        # Compute similarity metrics
        jaccard_dict = compute_jaccard_similarity(data_array, n_items)
        cosine_dict = compute_cosine_similarity(X_train.iloc[:,:-1])
        pop_dict = compute_popularity(X_train.iloc[:,:-1], n_items)
        tf_idf_dict = compute_tf_idf(data_array, X_train, n_train_users, n_items)
        
        # Save all processed data
        data_dict = {
            'X_train': X_train,
            'X_test': X_test,
            'jaccard_based_sim': jaccard_dict,
            'cosine_based_sim': cosine_dict,
            'pop_dict': pop_dict,
            'tf_idf_dict': tf_idf_dict
        }
        save_processed_data(data_dict, files_path, args.dataset)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Preprocessing completed in {elapsed_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    main()