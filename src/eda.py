#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exploratory Data Analysis for DiffExplainer4RS datasets
This script performs various analyses on the recommendation system datasets
to better understand their characteristics and distributions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from typing import Dict, Tuple, List, Optional
import logging
import sys
from datetime import datetime
import os
from io import StringIO

# Configure logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

class TeeStream:
    """Stream wrapper that writes to multiple streams"""
    def __init__(self, streams):
        self.streams = streams
        
    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
            
    def flush(self):
        for stream in self.streams:
            stream.flush()

class DatasetAnalyzer:
    """Base class for dataset analysis with common functionality"""
    
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.data_dir = Path("datasets") / dataset_name
        
        # Create subdirectories for different types of outputs
        self.results_dir = Path("results") / "eda" / dataset_name
        self.plots_dir = self.results_dir / "plots"
        self.stats_dir = self.results_dir / "stats"
        self.logs_dir = self.results_dir / "logs"
        
        # Create all directories
        for dir_path in [self.results_dir, self.plots_dir, self.stats_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Set up logging to both file and console
        log_file = self.logs_dir / f"eda_{timestamp}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Redirect stdout to both console and file
        log_output = open(self.logs_dir / f"output_{timestamp}.txt", 'w')
        sys.stdout = TeeStream([sys.stdout, log_output])
        
        self.raw_data = None
        self.train_data = None
        self.test_data = None
        self.preprocessed_data = {}
        self.aux_data = {}
        
        # Set style for better visualizations
        plt.style.use('ggplot')
        sns.set_palette("husl")
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        
    def save_figure(self, name: str, subdir: str = "") -> None:
        """Save current figure to results directory with optional subdirectory"""
        save_dir = self.plots_dir
        if subdir:
            save_dir = save_dir / subdir
            save_dir.mkdir(exist_ok=True)
        
        plt.savefig(save_dir / f"{name}_{timestamp}.png", bbox_inches='tight')
        plt.close()
        
    def save_stats(self, stats: Dict, name: str, subdir: str = "") -> None:
        """Save statistics to a file with optional subdirectory"""
        save_dir = self.stats_dir
        if subdir:
            save_dir = save_dir / subdir
            save_dir.mkdir(exist_ok=True)
            
        with open(save_dir / f"{name}_{timestamp}.txt", 'w') as f:
            f.write(f"=== {name} Statistics ===\n\n")
            for key, value in stats.items():
                f.write(f"{key}:\n")
                if isinstance(value, dict):
                    for k, v in value.items():
                        f.write(f"  {k}: {v}\n")
                else:
                    f.write(f"  {value}\n")
                f.write("\n")
        
    def load_raw_data(self) -> pd.DataFrame:
        """Load raw dataset. Should be implemented by child classes."""
        raise NotImplementedError
        
    def load_preprocessed_data(self) -> Dict[str, pd.DataFrame]:
        """Load preprocessed train/test data"""
        try:
            self.train_data = pd.read_csv(self.data_dir / f"x_train_{self.dataset_name}.csv")
            self.test_data = pd.read_csv(self.data_dir / f"x_test_{self.dataset_name}.csv")
            self.logger.info(f"Loaded preprocessed train/test data for {self.dataset_name}")
            return {"train": self.train_data, "test": self.test_data}
        except Exception as e:
            self.logger.error(f"Error loading preprocessed data: {e}")
            return {}
            
    def load_auxiliary_data(self) -> Dict:
        """Load auxiliary data like similarity matrices and dictionaries"""
        aux_data = {}
        try:
            # Load TF-IDF dictionary
            with open(self.data_dir / f"tf_idf_dict_{self.dataset_name}.pkl", 'rb') as f:
                aux_data['tf_idf_dict'] = pickle.load(f)
            
            # Load popularity dictionary
            with open(self.data_dir / f"pop_dict_{self.dataset_name}.pkl", 'rb') as f:
                aux_data['pop_dict'] = pickle.load(f)
            
            # Load similarity matrices
            for sim_type in ['cosine', 'jaccard']:
                with open(self.data_dir / f"{sim_type}_based_sim_{self.dataset_name}.pkl", 'rb') as f:
                    aux_data[f'{sim_type}_sim'] = pickle.load(f)
            
            self.logger.info(f"Loaded auxiliary data for {self.dataset_name}")
            return aux_data
        except Exception as e:
            self.logger.error(f"Error loading auxiliary data: {e}")
            return aux_data

    def analyze_rating_distribution(self, data: pd.DataFrame, title: str, rating_col: str = 'rating') -> Dict:
        """Analyze rating distribution"""
        if rating_col not in data.columns:
            self.logger.warning(f"Rating column '{rating_col}' not found in data")
            return {}
            
        stats = {
            'mean': data[rating_col].mean(),
            'median': data[rating_col].median(),
            'std': data[rating_col].std(),
            'min': data[rating_col].min(),
            'max': data[rating_col].max(),
            'skewness': data[rating_col].skew(),
            'kurtosis': data[rating_col].kurtosis(),
            'value_counts': data[rating_col].value_counts().to_dict()
        }
        
        # Save statistics
        self.save_stats(stats, f"rating_distribution_{title.lower()}", "ratings")
        
        # Plot rating distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(data=data, x=rating_col, bins=20)
        plt.title(f'Rating Distribution - {title}')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        self.save_figure(f"rating_dist_{title.lower()}", "ratings")
        
        # Plot rating boxplot
        plt.figure(figsize=(8, 6))
        sns.boxplot(y=data[rating_col])
        plt.title(f'Rating Boxplot - {title}')
        plt.ylabel('Rating')
        self.save_figure(f"rating_boxplot_{title.lower()}", "ratings")
        
        return stats
    
    def analyze_user_activity(self, data: pd.DataFrame, title: str, user_col: str = 'user_id') -> Dict:
        """Analyze user activity patterns"""
        if user_col not in data.columns:
            self.logger.warning(f"User column '{user_col}' not found in data")
            return {}
            
        ratings_per_user = data.groupby(user_col).size()
        user_stats = {
            'ratings_per_user': ratings_per_user.describe().to_dict(),
            'unique_users': data[user_col].nunique(),
            'total_ratings': len(data),
            'avg_ratings_per_user': len(data) / data[user_col].nunique(),
            'activity_percentiles': {
                f'p{p}': np.percentile(ratings_per_user, p)
                for p in [10, 25, 50, 75, 90, 95, 99]
            }
        }
        
        # Save statistics
        self.save_stats(user_stats, f"user_activity_{title.lower()}", "users")
        
        # Plot ratings per user distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(ratings_per_user, bins=50)
        plt.title(f'Ratings per User Distribution - {title}')
        plt.xlabel('Number of Ratings')
        plt.ylabel('Number of Users')
        self.save_figure(f"user_activity_dist_{title.lower()}", "users")
        
        # Plot user activity over time if timestamp is available
        if 'timestamp' in data.columns:
            data_sorted = data.sort_values('timestamp')
            user_activity_timeline = data_sorted.groupby('timestamp')[user_col].count()
            
            plt.figure(figsize=(12, 6))
            plt.plot(user_activity_timeline.index, user_activity_timeline.values)
            plt.title(f'User Activity Timeline - {title}')
            plt.xlabel('Timestamp')
            plt.ylabel('Number of Ratings')
            plt.xticks(rotation=45)
            self.save_figure(f"user_activity_timeline_{title.lower()}", "users")
        
        return user_stats
    
    def analyze_item_popularity(self, data: pd.DataFrame, title: str, item_col: str = 'item_id') -> Dict:
        """Analyze item popularity patterns"""
        if item_col not in data.columns:
            self.logger.warning(f"Item column '{item_col}' not found in data")
            return {}
            
        ratings_per_item = data.groupby(item_col).size()
        item_stats = {
            'ratings_per_item': ratings_per_item.describe().to_dict(),
            'unique_items': data[item_col].nunique(),
            'total_ratings': len(data),
            'avg_ratings_per_item': len(data) / data[item_col].nunique(),
            'popularity_percentiles': {
                f'p{p}': np.percentile(ratings_per_item, p)
                for p in [10, 25, 50, 75, 90, 95, 99]
            }
        }
        
        # Calculate popularity concentration
        sorted_items = ratings_per_item.sort_values(ascending=False)
        cumsum = sorted_items.cumsum()
        total = cumsum.iloc[-1]
        
        item_stats['popularity_concentration'] = {
            'top_1%_items_ratings_share': cumsum.iloc[int(len(sorted_items) * 0.01)] / total,
            'top_5%_items_ratings_share': cumsum.iloc[int(len(sorted_items) * 0.05)] / total,
            'top_10%_items_ratings_share': cumsum.iloc[int(len(sorted_items) * 0.10)] / total,
            'top_20%_items_ratings_share': cumsum.iloc[int(len(sorted_items) * 0.20)] / total
        }
        
        # Save statistics
        self.save_stats(item_stats, f"item_popularity_{title.lower()}", "items")
        
        # Plot ratings per item distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(ratings_per_item, bins=50)
        plt.title(f'Ratings per Item Distribution - {title}')
        plt.xlabel('Number of Ratings')
        plt.ylabel('Number of Items')
        self.save_figure(f"item_popularity_dist_{title.lower()}", "items")
        
        # Plot popularity concentration (long-tail plot)
        plt.figure(figsize=(12, 6))
        plt.plot(np.arange(1, len(sorted_items) + 1) / len(sorted_items),
                cumsum / total)
        plt.title(f'Item Popularity Concentration - {title}')
        plt.xlabel('Fraction of Items (sorted by popularity)')
        plt.ylabel('Fraction of Total Ratings')
        plt.grid(True)
        self.save_figure(f"item_popularity_concentration_{title.lower()}", "items")
        
        # If rating column exists, analyze rating distribution by popularity
        if 'rating' in data.columns:
            item_avg_ratings = data.groupby(item_col)['rating'].agg(['mean', 'count'])
            
            plt.figure(figsize=(10, 6))
            plt.scatter(item_avg_ratings['count'], item_avg_ratings['mean'], alpha=0.5)
            plt.title(f'Average Rating vs Popularity - {title}')
            plt.xlabel('Number of Ratings (Popularity)')
            plt.ylabel('Average Rating')
            plt.xscale('log')
            self.save_figure(f"rating_vs_popularity_{title.lower()}", "items")
        
        return item_stats

    def analyze_preprocessed_data(self, data: pd.DataFrame, title: str) -> None:
        """Analyze preprocessed data structure"""
        print(f"\n=== {title} Dataset Structure ===")
        
        # Create preprocessed subdirectories if they don't exist
        preprocessed_stats_dir = self.stats_dir / "preprocessed"
        preprocessed_plots_dir = self.plots_dir / "preprocessed"
        preprocessed_stats_dir.mkdir(parents=True, exist_ok=True)
        preprocessed_plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Basic information
        info_dict = {
            'shape': data.shape,
            'memory_usage_mb': data.memory_usage().sum() / 1024**2,
            'columns': data.columns.tolist()
        }
        
        # Capture data info in a string buffer
        buffer = StringIO()
        data.info(buf=buffer)
        info_str = buffer.getvalue()
        
        # Basic statistics
        desc_stats = data.describe()
        
        # Save all information
        with open(preprocessed_stats_dir / f"info_{title.lower()}_{timestamp}.txt", 'w') as f:
            f.write(f"=== {title} Dataset Information ===\n\n")
            for key, value in info_dict.items():
                f.write(f"{key}: {value}\n")
            f.write("\nDetailed Information:\n")
            f.write(info_str)
            f.write("\nBasic Statistics:\n")
            f.write(desc_stats.to_string())
            
        # Analyze sparsity
        total_cells = data.shape[0] * data.shape[1]
        non_zero_cells = (data != 0).sum().sum()
        sparsity = 1 - (non_zero_cells / total_cells)
        
        sparsity_stats = {
            'total_cells': total_cells,
            'non_zero_cells': int(non_zero_cells),
            'sparsity_ratio': sparsity,
            'memory_efficiency': {
                'dense_size_mb': (data.shape[0] * data.shape[1] * 8) / 1024**2,
                'actual_size_mb': data.memory_usage().sum() / 1024**2
            }
        }
        self.save_stats(sparsity_stats, f"sparsity_{title.lower()}", "preprocessed")
        
        # Plot sparsity visualizations
        plt.figure(figsize=(12, 8))
        plt.spy(data.values, markersize=0.1)
        plt.title(f'Sparsity Pattern - {title}')
        plt.xlabel('Feature Index')
        plt.ylabel('Sample Index')
        self.save_figure(f"sparsity_pattern_{title.lower()}", "preprocessed")
        
        # Plot distribution of non-zero values
        non_zero_vals = data.values[data.values != 0]
        plt.figure(figsize=(10, 6))
        sns.histplot(non_zero_vals, bins=50)
        plt.title(f'Distribution of Non-zero Values - {title}')
        plt.xlabel('Value')
        plt.ylabel('Count')
        self.save_figure(f"nonzero_dist_{title.lower()}", "preprocessed")
        
        # Plot feature density
        plt.figure(figsize=(15, 6))
        feature_density = (data != 0).mean() * 100
        plt.plot(feature_density.values)
        plt.title(f'Feature Density - {title}')
        plt.xlabel('Feature Index')
        plt.ylabel('Percentage of Non-zero Values')
        plt.grid(True)
        self.save_figure(f"feature_density_{title.lower()}", "preprocessed")
        
        # Save feature density statistics
        density_stats = {
            'mean_density': feature_density.mean(),
            'std_density': feature_density.std(),
            'min_density': feature_density.min(),
            'max_density': feature_density.max(),
            'density_percentiles': {
                f'p{p}': np.percentile(feature_density, p)
                for p in [10, 25, 50, 75, 90, 95, 99]
            }
        }
        self.save_stats(density_stats, f"feature_density_{title.lower()}", "preprocessed")

    def analyze_auxiliary_data(self) -> None:
        """Analyze auxiliary data like similarity matrices and dictionaries"""
        if not self.aux_data:
            self.aux_data = self.load_auxiliary_data()
            
        if not self.aux_data:
            self.logger.warning("No auxiliary data available for analysis")
            return
            
        # Create auxiliary subdirectories if they don't exist
        aux_stats_dir = self.stats_dir / "auxiliary"
        aux_plots_dir = self.plots_dir / "auxiliary"
        aux_stats_dir.mkdir(parents=True, exist_ok=True)
        aux_plots_dir.mkdir(parents=True, exist_ok=True)
            
        # Analyze TF-IDF dictionary
        if 'tf_idf_dict' in self.aux_data:
            tfidf_stats = {
                'num_items': len(self.aux_data['tf_idf_dict']),
                'avg_features': np.mean([len(v) for v in self.aux_data['tf_idf_dict'].values()]),
                'max_features': max(len(v) for v in self.aux_data['tf_idf_dict'].values()),
                'min_features': min(len(v) for v in self.aux_data['tf_idf_dict'].values())
            }
            self.save_stats(tfidf_stats, "tfidf_analysis", "auxiliary")
            
            # Plot distribution of feature counts
            plt.figure(figsize=(10, 6))
            feature_counts = [len(v) for v in self.aux_data['tf_idf_dict'].values()]
            sns.histplot(feature_counts, bins=50)
            plt.title('Distribution of TF-IDF Features per Item')
            plt.xlabel('Number of Features')
            plt.ylabel('Count')
            self.save_figure("tfidf_feature_dist", "auxiliary")
        
        # Analyze popularity dictionary
        if 'pop_dict' in self.aux_data:
            pop_values = list(self.aux_data['pop_dict'].values())
            pop_stats = {
                'num_items': len(self.aux_data['pop_dict']),
                'avg_popularity': np.mean(pop_values),
                'popularity_stats': {
                    'min': np.min(pop_values),
                    'max': np.max(pop_values),
                    'mean': np.mean(pop_values),
                    'std': np.std(pop_values),
                    'median': np.median(pop_values),
                    'percentiles': {
                        f'p{p}': np.percentile(pop_values, p)
                        for p in [10, 25, 50, 75, 90, 95, 99]
                    }
                }
            }
            self.save_stats(pop_stats, "popularity_analysis", "auxiliary")
            
            # Plot popularity distribution
            plt.figure(figsize=(10, 6))
            sns.histplot(pop_values, bins=50)
            plt.title('Item Popularity Distribution')
            plt.xlabel('Popularity Score')
            plt.ylabel('Count')
            self.save_figure("popularity_dist", "auxiliary")
            
            # Plot popularity rank distribution
            plt.figure(figsize=(10, 6))
            sorted_pop = sorted(pop_values, reverse=True)
            plt.plot(range(1, len(sorted_pop) + 1), sorted_pop)
            plt.title('Item Popularity Rank Distribution')
            plt.xlabel('Rank')
            plt.ylabel('Popularity Score')
            plt.xscale('log')
            plt.yscale('log')
            plt.grid(True)
            self.save_figure("popularity_rank_dist", "auxiliary")
        
        # Analyze similarity matrices
        for sim_type in ['cosine', 'jaccard']:
            key = f'{sim_type}_sim'
            if key in self.aux_data:
                sim_dict = self.aux_data[key]
                
                # Debug information
                print(f"\nAnalyzing {sim_type} similarity matrix:")
                print(f"Type: {type(sim_dict)}")
                if isinstance(sim_dict, dict):
                    print(f"Number of items: {len(sim_dict)}")
                    sample_key = next(iter(sim_dict))
                    print(f"Sample key type: {type(sample_key)}")
                    print(f"Sample value type: {type(sim_dict[sample_key])}")
                    if isinstance(sim_dict[sample_key], dict):
                        sample_inner_key = next(iter(sim_dict[sample_key]))
                        print(f"Sample inner key type: {type(sample_inner_key)}")
                        print(f"Sample inner value type: {type(sim_dict[sample_key][sample_inner_key])}")
                else:
                    print(f"Shape: {sim_dict.shape if hasattr(sim_dict, 'shape') else 'No shape attribute'}")
                
                # Extract similarity values based on the data structure
                if isinstance(sim_dict, dict):
                    sim_values = []
                    for item_sims in sim_dict.values():
                        if isinstance(item_sims, dict):
                            sim_values.extend(item_sims.values())
                        else:
                            sim_values.append(float(item_sims))
                else:
                    sim_values = sim_dict.flatten() if hasattr(sim_dict, 'flatten') else [float(sim_dict)]
                
                sim_stats = {
                    'data_type': str(type(sim_dict)),
                    'num_values': len(sim_values),
                    'similarity_stats': {
                        'min': np.min(sim_values),
                        'max': np.max(sim_values),
                        'mean': np.mean(sim_values),
                        'std': np.std(sim_values),
                        'median': np.median(sim_values),
                        'percentiles': {
                            f'p{p}': np.percentile(sim_values, p)
                            for p in [10, 25, 50, 75, 90, 95, 99]
                        }
                    }
                }
                self.save_stats(sim_stats, f"{sim_type}_similarity_analysis", "auxiliary")
                
                # Plot similarity distribution
                plt.figure(figsize=(10, 6))
                sns.histplot(sim_values, bins=50)
                plt.title(f'{sim_type.capitalize()} Similarity Distribution')
                plt.xlabel('Similarity Score')
                plt.ylabel('Count')
                self.save_figure(f"{sim_type}_similarity_dist", "auxiliary")
                
                # If it's a matrix, plot the sparsity pattern
                if hasattr(sim_dict, 'shape'):
                    plt.figure(figsize=(12, 8))
                    plt.spy(sim_dict, markersize=0.1)
                    plt.title(f'{sim_type.capitalize()} Similarity Matrix Sparsity Pattern')
                    plt.xlabel('Item Index')
                    plt.ylabel('Item Index')
                    self.save_figure(f"{sim_type}_similarity_sparsity", "auxiliary")
                    
                    # Plot similarity heatmap for a sample of items
                    if sim_dict.shape[0] > 100:
                        sample_size = 100
                        sample_indices = np.random.choice(sim_dict.shape[0], sample_size, replace=False)
                        sample_matrix = sim_dict[sample_indices][:, sample_indices]
                    else:
                        sample_matrix = sim_dict
                    
                    plt.figure(figsize=(12, 10))
                    sns.heatmap(sample_matrix, cmap='viridis', xticklabels=False, yticklabels=False)
                    plt.title(f'{sim_type.capitalize()} Similarity Heatmap (Sample)')
                    plt.xlabel('Item Index')
                    plt.ylabel('Item Index')
                    plt.colorbar(label='Similarity Score')
                    self.save_figure(f"{sim_type}_similarity_heatmap", "auxiliary")

class ML1MAnalyzer(DatasetAnalyzer):
    """Specific analyzer for ML1M dataset"""
    
    def __init__(self):
        super().__init__("ML1M")
    
    def load_raw_data(self) -> pd.DataFrame:
        """Load ML1M raw ratings data"""
        try:
            # ML1M ratings.dat format: UserID::MovieID::Rating::Timestamp
            self.raw_data = pd.read_csv(
                self.data_dir / "ratings.dat",
                sep="::",
                engine='python',
                names=['user_id', 'item_id', 'rating', 'timestamp']
            )
            self.logger.info("Loaded ML1M raw data successfully")
            return self.raw_data
        except Exception as e:
            self.logger.error(f"Error loading ML1M raw data: {e}")
            return pd.DataFrame()

def main():
    """Main function to run the analysis"""
    # Initialize ML1M analyzer
    ml1m = ML1MAnalyzer()
    
    print("=== Starting ML1M Dataset Analysis ===\n")
    
    # Load and analyze raw data
    raw_data = ml1m.load_raw_data()
    if raw_data is not None:
        print("\n=== Raw Data Analysis ===")
        
        # Analyze ratings
        rating_stats = ml1m.analyze_rating_distribution(raw_data, "Raw Data")
        print("\nRating Statistics:")
        print(f"Mean rating: {rating_stats['mean']:.2f}")
        print(f"Median rating: {rating_stats['median']:.2f}")
        print(f"Standard deviation: {rating_stats['std']:.2f}")
        print(f"Rating range: {rating_stats['min']} - {rating_stats['max']}")
        
        # Analyze user activity
        user_stats = ml1m.analyze_user_activity(raw_data, "Raw Data")
        print("\nUser Activity Statistics:")
        print(f"Total unique users: {user_stats['unique_users']}")
        print(f"Average ratings per user: {user_stats['avg_ratings_per_user']:.2f}")
        print(f"Median ratings per user: {user_stats['ratings_per_user']['50%']:.2f}")
        
        # Analyze item popularity
        item_stats = ml1m.analyze_item_popularity(raw_data, "Raw Data")
        print("\nItem Popularity Statistics:")
        print(f"Total unique items: {item_stats['unique_items']}")
        print(f"Average ratings per item: {item_stats['avg_ratings_per_item']:.2f}")
        print(f"Median ratings per item: {item_stats['ratings_per_item']['50%']:.2f}")
        print("\nPopularity Concentration:")
        for metric, value in item_stats['popularity_concentration'].items():
            print(f"{metric}: {value*100:.2f}%")
    
    # Load and analyze preprocessed data
    preprocessed = ml1m.load_preprocessed_data()
    if preprocessed:
        print("\n=== Preprocessed Data Analysis ===")
        for data_type, data in preprocessed.items():
            print(f"\nAnalyzing {data_type.upper()} data...")
            ml1m.analyze_preprocessed_data(data, data_type.upper())
    
    # Load and analyze auxiliary data
    print("\n=== Auxiliary Data Analysis ===")
    ml1m.analyze_auxiliary_data()
    
    print("\n=== Analysis Complete ===")
    print(f"Results have been saved to: {ml1m.results_dir}")
    print("\nResults directory structure:")
    print("  - plots/: All visualizations")
    print("    - ratings/: Rating distribution plots")
    print("    - users/: User activity plots")
    print("    - items/: Item popularity plots")
    print("    - preprocessed/: Preprocessed data plots")
    print("    - auxiliary/: Auxiliary data plots")
    print("  - stats/: All statistical analysis")
    print("    - ratings/: Rating statistics")
    print("    - users/: User activity statistics")
    print("    - items/: Item popularity statistics")
    print("    - preprocessed/: Preprocessed data statistics")
    print("    - auxiliary/: Auxiliary data statistics")
    print("  - logs/: Analysis logs and outputs")

if __name__ == "__main__":
    main() 