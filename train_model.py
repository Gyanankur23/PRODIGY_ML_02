"""
Main training script for K-Means clustering model.
"""
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import DataPreprocessor
from clustering import KMeansClustering, plot_elbow_method, plot_silhouette_scores
from visualization import ClusterVisualizer


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("Customer Segmentation - K-Means Clustering")
    print("=" * 60)
    
    # Initialize paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data', 'Mall_Customers.csv')
    model_path = os.path.join(base_dir, 'models', 'kmeans_model.pkl')
    plots_dir = os.path.join(base_dir, 'plots')
    
    # Create plots directory
    os.makedirs(plots_dir, exist_ok=True)
    
    # Step 1: Data Preprocessing
    print("\n[1/5] Loading and preprocessing data...")
    preprocessor = DataPreprocessor(data_path)
    original_data, scaled_features = preprocessor.preprocess()
    
    print(f"   - Loaded {len(original_data)} customer records")
    print(f"   - Features: {preprocessor.get_feature_names()}")
    print(f"   - Data shape: {scaled_features.shape}")
    
    # Step 2: Find Optimal Clusters
    print("\n[2/5] Finding optimal number of clusters...")
    clustering = KMeansClustering(n_clusters=5, random_state=42)
    results = clustering.find_optimal_clusters(scaled_features, max_clusters=10)
    
    print(f"   - Tested k = {results['cluster_range']}")
    print(f"   - Best silhouette score: {max(results['silhouette_scores']):.4f}")
    print(f"   - Optimal k: {results['silhouette_scores'].index(max(results['silhouette_scores'])) + 2}")
    
    # Plot Elbow Method
    plot_elbow_method(
        results['cluster_range'],
        results['inertias'],
        save_path=os.path.join(plots_dir, 'elbow_method.png')
    )
    print("   - Saved elbow method plot")
    
    # Plot Silhouette Scores
    plot_silhouette_scores(
        results['cluster_range'],
        results['silhouette_scores'],
        save_path=os.path.join(plots_dir, 'silhouette_scores.png')
    )
    print("   - Saved silhouette scores plot")
    
    # Step 3: Train Model
    print("\n[3/5] Training K-Means model...")
    optimal_k = 5  # Based on analysis
    clustering.n_clusters = optimal_k
    labels = clustering.fit(scaled_features)
    
    print(f"   - Number of clusters: {optimal_k}")
    print(f"   - Inertia: {clustering.inertia:.2f}")
    print(f"   - Silhouette Score: {clustering.silhouette_score:.4f}")
    
    # Step 4: Generate Visualizations
    print("\n[4/5] Generating visualizations...")
    visualizer = ClusterVisualizer()
    
    feature_names = preprocessor.get_feature_names()
    
    # Cluster distribution
    visualizer.plot_cluster_distribution(
        labels,
        save_path=os.path.join(plots_dir, 'cluster_distribution.png')
    )
    print("   - Saved cluster distribution plot")
    
    # 2D clusters
    visualizer.plot_2d_clusters(
        scaled_features,
        labels,
        feature_names,
        save_path=os.path.join(plots_dir, '2d_clusters.png')
    )
    print("   - Saved 2D clusters plot")
    
    # 3D clusters
    visualizer.plot_3d_clusters(
        scaled_features,
        labels,
        feature_names,
        save_path=os.path.join(plots_dir, '3d_clusters.png')
    )
    print("   - Saved 3D clusters plot")
    
    # Boxplots
    visualizer.plot_cluster_boxplots(
        original_data,
        labels,
        save_path=os.path.join(plots_dir, 'cluster_boxplots.png')
    )
    print("   - Saved cluster boxplots")
    
    # Pairwise relationships
    visualizer.plot_pairwise_relationships(
        original_data,
        labels,
        save_path=os.path.join(plots_dir, 'pairwise_relationships.png')
    )
    print("   - Saved pairwise relationships plot")
    
    # Step 5: Save Model and Statistics
    print("\n[5/5] Saving model and statistics...")
    clustering.save_model(model_path)
    print(f"   - Saved model to {model_path}")
    
    # Get cluster statistics
    cluster_stats = clustering.get_cluster_statistics(scaled_features, original_data)
    cluster_stats.to_csv(os.path.join(plots_dir, 'cluster_statistics.csv'))
    print("   - Saved cluster statistics")
    
    # Print cluster summary
    print("\n" + "=" * 60)
    print("CLUSTER SUMMARY")
    print("=" * 60)
    print(cluster_stats)
    
    # Add cluster labels to original data
    original_data['Cluster'] = labels
    original_data.to_csv(os.path.join(base_dir, 'data', 'customers_with_clusters.csv'), index=False)
    print(f"\n- Saved data with cluster labels to data/customers_with_clusters.csv")
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
