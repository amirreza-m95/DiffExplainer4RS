import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_dataset(dataset_name):
    """
    Load train and test datasets for a given dataset name.
    
    Args:
        dataset_name (str): Name of the dataset ('ML1M', 'Yahoo', 'Pinterest')
    
    Returns:
        tuple: (train_data, test_data) as pandas DataFrames
    """
    base_path = Path("datasets/lxr-CE") / dataset_name
    
    # Load train data
    train_path = base_path / f"train_data_{dataset_name}.csv"
    train_data = pd.read_csv(train_path, index_col=0)
    
    # Load test data
    test_path = base_path / f"test_data_{dataset_name}.csv"
    test_data = pd.read_csv(test_path, index_col=0)
    
    print(f"Loaded {dataset_name}:")
    print(f"  Train data: {train_data.shape[0]} users × {train_data.shape[1]} items")
    print(f"  Test data: {test_data.shape[0]} users × {test_data.shape[1]} items")
    
    return train_data, test_data

def calculate_user_history_sizes(train_data, test_data):
    """
    Calculate user history interaction sizes for both train and test sets.
    
    Args:
        train_data (pd.DataFrame): Training data
        test_data (pd.DataFrame): Test data
    
    Returns:
        dict: Dictionary containing history size statistics
    """
    # Calculate interaction counts (sum of 1s in each row)
    train_sizes = train_data.sum(axis=1)
    test_sizes = test_data.sum(axis=1)
    
    # Combine train and test for overall statistics
    combined_sizes = pd.concat([train_sizes, test_sizes])
    
    # Calculate percentiles
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    train_percentiles = np.percentile(train_sizes, percentiles)
    test_percentiles = np.percentile(test_sizes, percentiles)
    combined_percentiles = np.percentile(combined_sizes, percentiles)
    
    # Calculate distribution statistics
    train_skew = train_sizes.skew()
    test_skew = test_sizes.skew()
    combined_skew = combined_sizes.skew()
    
    train_kurtosis = train_sizes.kurtosis()
    test_kurtosis = test_sizes.kurtosis()
    combined_kurtosis = combined_sizes.kurtosis()
    
    stats = {
        'train_sizes': train_sizes,
        'test_sizes': test_sizes,
        'combined_sizes': combined_sizes,
        'train_mean': train_sizes.mean(),
        'test_mean': test_sizes.mean(),
        'combined_mean': combined_sizes.mean(),
        'train_median': train_sizes.median(),
        'test_median': test_sizes.median(),
        'combined_median': combined_sizes.median(),
        'train_min': train_sizes.min(),
        'test_min': test_sizes.min(),
        'combined_min': combined_sizes.min(),
        'train_max': train_sizes.max(),
        'test_max': test_sizes.max(),
        'combined_max': combined_sizes.max(),
        'train_std': train_sizes.std(),
        'test_std': test_sizes.std(),
        'combined_std': combined_sizes.std(),
        'train_percentiles': dict(zip(percentiles, train_percentiles)),
        'test_percentiles': dict(zip(percentiles, test_percentiles)),
        'combined_percentiles': dict(zip(percentiles, combined_percentiles)),
        'train_skew': train_skew,
        'test_skew': test_skew,
        'combined_skew': combined_skew,
        'train_kurtosis': train_kurtosis,
        'test_kurtosis': test_kurtosis,
        'combined_kurtosis': combined_kurtosis
    }
    
    return stats

def create_detailed_visualizations(datasets_stats):
    """
    Create comprehensive visualizations including histograms, box plots, and percentile analysis.
    
    Args:
        datasets_stats (dict): Dictionary containing statistics for all datasets
    """
    datasets = list(datasets_stats.keys())
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    # 1. Histogram comparison
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('User History Interaction Size Distributions', fontsize=16, fontweight='bold')
    
    for idx, dataset in enumerate(datasets):
        # Train set histogram
        ax1 = axes[0, idx]
        train_sizes = datasets_stats[dataset]['train_sizes']
        ax1.hist(train_sizes, bins=50, alpha=0.7, color=colors[idx], edgecolor='black', linewidth=0.5)
        ax1.set_title(f'{dataset} - Train Set', fontweight='bold')
        ax1.set_xlabel('Interaction Count')
        ax1.set_ylabel('Number of Users')
        ax1.grid(alpha=0.3)
        
        # Test set histogram
        ax2 = axes[1, idx]
        test_sizes = datasets_stats[dataset]['test_sizes']
        ax2.hist(test_sizes, bins=50, alpha=0.7, color=colors[idx], edgecolor='black', linewidth=0.5)
        ax2.set_title(f'{dataset} - Test Set', fontweight='bold')
        ax2.set_xlabel('Interaction Count')
        ax2.set_ylabel('Number of Users')
        ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('user_history_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Box plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.suptitle('User History Interaction Size: Box Plot Comparison', fontsize=16, fontweight='bold')
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        train_sizes = datasets_stats[dataset]['train_sizes']
        test_sizes = datasets_stats[dataset]['test_sizes']
        
        data_to_plot = [train_sizes, test_sizes]
        bp = ax.boxplot(data_to_plot, labels=['Train', 'Test'], patch_artist=True)
        
        # Color the boxes
        bp['boxes'][0].set_facecolor(colors[idx])
        bp['boxes'][1].set_facecolor(colors[idx])
        bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_alpha(0.7)
        
        ax.set_title(f'{dataset} Dataset', fontweight='bold')
        ax.set_ylabel('Interaction Count')
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('user_history_boxplots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Percentile comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.suptitle('User History Interaction Size: Percentile Analysis', fontsize=16, fontweight='bold')
    
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        train_percentiles = [datasets_stats[dataset]['train_percentiles'][p] for p in percentiles]
        test_percentiles = [datasets_stats[dataset]['test_percentiles'][p] for p in percentiles]
        
        x = np.arange(len(percentiles))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, train_percentiles, width, label='Train', alpha=0.8, color=colors[idx])
        bars2 = ax.bar(x + width/2, test_percentiles, width, label='Test', alpha=0.8, color=colors[idx])
        
        ax.set_xlabel('Percentile')
        ax.set_ylabel('Interaction Count')
        ax.set_title(f'{dataset} Dataset', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{p}%' for p in percentiles])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars (only for key percentiles to avoid clutter)
        key_percentiles = [25, 50, 75]
        for i, p in enumerate(percentiles):
            if p in key_percentiles:
                ax.text(i - width/2, train_percentiles[i] + 1, f'{train_percentiles[i]:.0f}', 
                       ha='center', va='bottom', fontsize=8)
                ax.text(i + width/2, test_percentiles[i] + 1, f'{test_percentiles[i]:.0f}', 
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('user_history_percentiles.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Cumulative distribution
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.suptitle('User History Interaction Size: Cumulative Distribution', fontsize=16, fontweight='bold')
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        train_sizes = datasets_stats[dataset]['train_sizes']
        test_sizes = datasets_stats[dataset]['test_sizes']
        
        # Sort for cumulative plot
        train_sorted = np.sort(train_sizes)
        test_sorted = np.sort(test_sizes)
        
        # Calculate cumulative percentages
        train_cumulative = np.arange(1, len(train_sorted) + 1) / len(train_sorted) * 100
        test_cumulative = np.arange(1, len(test_sorted) + 1) / len(test_sorted) * 100
        
        ax.plot(train_sorted, train_cumulative, label='Train', linewidth=2, color=colors[idx])
        ax.plot(test_sorted, test_cumulative, label='Test', linewidth=2, color=colors[idx], linestyle='--')
        
        ax.set_xlabel('Interaction Count')
        ax.set_ylabel('Cumulative Percentage (%)')
        ax.set_title(f'{dataset} Dataset', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('user_history_cumulative.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_visualization(datasets_stats):
    """
    Create bar chart visualization of user history interaction sizes.
    
    Args:
        datasets_stats (dict): Dictionary containing statistics for all datasets
    """
    # Prepare data for plotting
    datasets = list(datasets_stats.keys())
    metrics = ['Mean', 'Median', 'Min', 'Max']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('User History Interaction Size Analysis Across Datasets', fontsize=16, fontweight='bold')
    
    # Colors for each dataset
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    # Plot each metric
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        # Extract values for this metric
        if metric == 'Mean':
            values = [datasets_stats[dataset]['combined_mean'] for dataset in datasets]
        elif metric == 'Median':
            values = [datasets_stats[dataset]['combined_median'] for dataset in datasets]
        elif metric == 'Min':
            values = [datasets_stats[dataset]['combined_min'] for dataset in datasets]
        elif metric == 'Max':
            values = [datasets_stats[dataset]['combined_max'] for dataset in datasets]
        
        # Create bar plot
        bars = ax.bar(datasets, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title(f'{metric} Interaction Size', fontsize=14, fontweight='bold')
        ax.set_ylabel('Number of Interactions', fontsize=12)
        ax.set_xlabel('Dataset', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        # Rotate x-axis labels for better readability
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('user_history_interaction_sizes.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create additional detailed comparison plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Prepare data for detailed comparison
    x = np.arange(len(datasets))
    width = 0.35
    
    train_means = [datasets_stats[dataset]['train_mean'] for dataset in datasets]
    test_means = [datasets_stats[dataset]['test_mean'] for dataset in datasets]
    
    bars1 = ax.bar(x - width/2, train_means, width, label='Train Set', color='#FF6B6B', alpha=0.8)
    bars2 = ax.bar(x + width/2, test_means, width, label='Test Set', color='#4ECDC4', alpha=0.8)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Dataset', fontsize=14)
    ax.set_ylabel('Mean Interaction Size', fontsize=14)
    ax.set_title('Mean User History Interaction Size: Train vs Test', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('train_vs_test_interaction_sizes.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_detailed_statistics(datasets_stats):
    """
    Print detailed statistics for each dataset including percentiles and distribution metrics.
    
    Args:
        datasets_stats (dict): Dictionary containing statistics for all datasets
    """
    print("\n" + "="*100)
    print("DETAILED STATISTICS WITH PERCENTILES AND DISTRIBUTION ANALYSIS")
    print("="*100)
    
    for dataset, stats in datasets_stats.items():
        print(f"\n{dataset} Dataset:")
        print("=" * 60)
        
        # Basic statistics
        print(f"Train Set Statistics:")
        print(f"  Mean interactions: {stats['train_mean']:.2f}")
        print(f"  Median interactions: {stats['train_median']:.2f}")
        print(f"  Min interactions: {stats['train_min']:.0f}")
        print(f"  Max interactions: {stats['train_max']:.0f}")
        print(f"  Std deviation: {stats['train_std']:.2f}")
        print(f"  Skewness: {stats['train_skew']:.3f}")
        print(f"  Kurtosis: {stats['train_kurtosis']:.3f}")
        
        print(f"\n  Percentiles:")
        for p, value in stats['train_percentiles'].items():
            print(f"    {p}th percentile: {value:.1f}")
        
        print(f"\nTest Set Statistics:")
        print(f"  Mean interactions: {stats['test_mean']:.2f}")
        print(f"  Median interactions: {stats['test_median']:.2f}")
        print(f"  Min interactions: {stats['test_min']:.0f}")
        print(f"  Max interactions: {stats['test_max']:.0f}")
        print(f"  Std deviation: {stats['test_std']:.2f}")
        print(f"  Skewness: {stats['test_skew']:.3f}")
        print(f"  Kurtosis: {stats['test_kurtosis']:.3f}")
        
        print(f"\n  Percentiles:")
        for p, value in stats['test_percentiles'].items():
            print(f"    {p}th percentile: {value:.1f}")
        
        print(f"\nCombined (Train + Test) Statistics:")
        print(f"  Mean interactions: {stats['combined_mean']:.2f}")
        print(f"  Median interactions: {stats['combined_median']:.2f}")
        print(f"  Min interactions: {stats['combined_min']:.0f}")
        print(f"  Max interactions: {stats['combined_max']:.0f}")
        print(f"  Std deviation: {stats['combined_std']:.2f}")
        print(f"  Skewness: {stats['combined_skew']:.3f}")
        print(f"  Kurtosis: {stats['combined_kurtosis']:.3f}")
        
        print(f"\n  Percentiles:")
        for p, value in stats['combined_percentiles'].items():
            print(f"    {p}th percentile: {value:.1f}")
        
        # Distribution interpretation
        print(f"\nDistribution Analysis:")
        if stats['combined_skew'] > 1:
            print(f"  - Highly right-skewed distribution (skewness: {stats['combined_skew']:.3f})")
        elif stats['combined_skew'] > 0.5:
            print(f"  - Moderately right-skewed distribution (skewness: {stats['combined_skew']:.3f})")
        elif stats['combined_skew'] < -0.5:
            print(f"  - Left-skewed distribution (skewness: {stats['combined_skew']:.3f})")
        else:
            print(f"  - Approximately symmetric distribution (skewness: {stats['combined_skew']:.3f})")
        
        if stats['combined_kurtosis'] > 3:
            print(f"  - Heavy-tailed distribution (kurtosis: {stats['combined_kurtosis']:.3f})")
        elif stats['combined_kurtosis'] < 3:
            print(f"  - Light-tailed distribution (kurtosis: {stats['combined_kurtosis']:.3f})")
        else:
            print(f"  - Normal-like tail behavior (kurtosis: {stats['combined_kurtosis']:.3f})")

def main():
    """
    Main function to execute the analysis.
    """
    print("User History Interaction Size Analysis - Enhanced Version")
    print("="*60)
    
    # Define datasets to analyze
    datasets = ['ML1M', 'Yahoo', 'Pinterest']
    datasets_stats = {}
    
    # Process each dataset
    for dataset in datasets:
        print(f"\nProcessing {dataset}...")
        try:
            # Load datasets
            train_data, test_data = load_dataset(dataset)
            
            # Calculate statistics
            stats = calculate_user_history_sizes(train_data, test_data)
            datasets_stats[dataset] = stats
            
        except Exception as e:
            print(f"Error processing {dataset}: {e}")
            continue
    
    # Create visualizations
    if datasets_stats:
        print("\nCreating basic visualizations...")
        create_visualization(datasets_stats)
        
        print("\nCreating detailed visualizations...")
        create_detailed_visualizations(datasets_stats)
        
        # Print detailed statistics
        print_detailed_statistics(datasets_stats)
        
        print(f"\nVisualizations saved as:")
        print("- user_history_interaction_sizes.png")
        print("- train_vs_test_interaction_sizes.png")
        print("- user_history_distributions.png")
        print("- user_history_boxplots.png")
        print("- user_history_percentiles.png")
        print("- user_history_cumulative.png")
    else:
        print("No datasets were successfully processed.")

if __name__ == "__main__":
    main()
