import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Optional


class ClusterVisualizer:
    """Visualize clustering results."""
    
    def __init__(self, style: str = 'whitegrid'):
        sns.set_style(style)
        self.colors = sns.color_palette("husl", 10)
    
    def plot_cluster_distribution(self, labels: np.ndarray, save_path: str = None):
        """Plot distribution of customers across clusters."""
        unique, counts = np.unique(labels, return_counts=True)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(unique, counts, color=self.colors[:len(unique)])
        plt.xlabel('Cluster', fontsize=12)
        plt.ylabel('Number of Customers', fontsize=12)
        plt.title('Customer Distribution Across Clusters', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_2d_clusters(self, X: np.ndarray, labels: np.ndarray, 
                        feature_names: list, save_path: str = None):
        """Plot 2D scatter plot of clusters."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        feature_pairs = [
            (0, 1),  # Age vs Annual Income
            (0, 2),  # Age vs Spending Score
            (1, 2)   # Annual Income vs Spending Score
        ]
        
        for idx, (i, j) in enumerate(feature_pairs):
            for cluster in np.unique(labels):
                mask = labels == cluster
                axes[idx].scatter(X[mask, i], X[mask, j], 
                                 c=[self.colors[cluster]], 
                                 label=f'Cluster {cluster}',
                                 alpha=0.6, s=50)
            
            axes[idx].set_xlabel(feature_names[i], fontsize=11)
            axes[idx].set_ylabel(feature_names[j], fontsize=11)
            axes[idx].set_title(f'{feature_names[i]} vs {feature_names[j]}', 
                              fontsize=12, fontweight='bold')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_3d_clusters(self, X: np.ndarray, labels: np.ndarray, 
                        feature_names: list, save_path: str = None):
        """Plot 3D scatter plot of clusters."""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        for cluster in np.unique(labels):
            mask = labels == cluster
            ax.scatter(X[mask, 0], X[mask, 1], X[mask, 2],
                     c=[self.colors[cluster]], 
                     label=f'Cluster {cluster}',
                     alpha=0.6, s=50)
        
        ax.set_xlabel(feature_names[0], fontsize=11)
        ax.set_ylabel(feature_names[1], fontsize=11)
        ax.set_zlabel(feature_names[2], fontsize=11)
        ax.set_title('3D Cluster Visualization', fontsize=14, fontweight='bold')
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_cluster_boxplots(self, data: pd.DataFrame, labels: np.ndarray, 
                            save_path: str = None):
        """Plot boxplots for each feature by cluster."""
        data_with_clusters = data.copy()
        data_with_clusters['Cluster'] = labels
        
        features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, feature in enumerate(features):
            sns.boxplot(data=data_with_clusters, x='Cluster', y=feature, 
                       ax=axes[idx], hue='Cluster', palette=self.colors[:len(np.unique(labels))],
                       legend=False)
            axes[idx].set_xlabel('Cluster', fontsize=11)
            axes[idx].set_ylabel(feature, fontsize=11)
            axes[idx].set_title(f'{feature} by Cluster', fontsize=12, fontweight='bold')
            axes[idx].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_interactive_3d(self, X: np.ndarray, labels: np.ndarray, 
                          feature_names: list) -> go.Figure:
        """Create interactive 3D plot using Plotly."""
        df = pd.DataFrame(X, columns=feature_names)
        df['Cluster'] = labels.astype(str)
        
        fig = px.scatter_3d(df, x=feature_names[0], y=feature_names[1], 
                          z=feature_names[2], color='Cluster',
                          title='Interactive 3D Cluster Visualization',
                          color_discrete_sequence=px.colors.qualitative.Plotly)
        
        fig.update_layout(scene=dict(
            xaxis_title=feature_names[0],
            yaxis_title=feature_names[1],
            zaxis_title=feature_names[2]
        ))
        
        return fig
    
    def plot_pairwise_relationships(self, data: pd.DataFrame, labels: np.ndarray,
                                   save_path: str = None):
        """Plot pairwise relationships colored by cluster."""
        data_with_clusters = data.copy()
        data_with_clusters['Cluster'] = labels.astype(str)
        
        features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
        
        g = sns.pairplot(data_with_clusters[features + ['Cluster']], 
                        hue='Cluster', palette=self.colors[:len(np.unique(labels))],
                        plot_kws={'alpha': 0.6, 's': 50})
        g.fig.suptitle('Pairwise Relationships by Cluster', 
                      y=1.02, fontsize=14, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
