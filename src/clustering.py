import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns


class KMeansClustering:
    """K-Means clustering for customer segmentation."""
    
    def __init__(self, n_clusters: int = 5, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = None
        self.labels = None
        self.inertia = None
        self.silhouette_score = None
    
    def find_optimal_clusters(self, X: np.ndarray, max_clusters: int = 10) -> Dict:
        """Find optimal number of clusters using Elbow Method and Silhouette Score."""
        inertias = []
        silhouette_scores = []
        cluster_range = range(2, max_clusters + 1)
        
        for k in cluster_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
            
            if k > 1:
                sil_score = silhouette_score(X, kmeans.labels_)
                silhouette_scores.append(sil_score)
        
        return {
            'cluster_range': list(cluster_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores
        }
    
    def fit(self, X: np.ndarray) -> np.ndarray:
        """Fit K-Means model to data."""
        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        self.labels = self.model.fit_predict(X)
        self.inertia = self.model.inertia_
        self.silhouette_score = silhouette_score(X, self.labels)
        
        return self.labels
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new data."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(X)
    
    def get_cluster_centers(self) -> np.ndarray:
        """Get cluster centers."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.cluster_centers_
    
    def save_model(self, filepath: str):
        """Save trained model to file."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        joblib.dump(self.model, filepath)
    
    def load_model(self, filepath: str):
        """Load trained model from file."""
        self.model = joblib.load(filepath)
        self.n_clusters = self.model.n_clusters
    
    def get_cluster_statistics(self, X: np.ndarray, original_data: pd.DataFrame) -> pd.DataFrame:
        """Get statistics for each cluster."""
        if self.labels is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        df_with_clusters = original_data.copy()
        df_with_clusters['Cluster'] = self.labels
        
        cluster_stats = df_with_clusters.groupby('Cluster').agg({
            'Age': ['mean', 'std', 'min', 'max'],
            'Annual Income (k$)': ['mean', 'std', 'min', 'max'],
            'Spending Score (1-100)': ['mean', 'std', 'min', 'max'],
            'CustomerID': 'count'
        }).round(2)
        
        cluster_stats.columns = ['_'.join(col).strip() for col in cluster_stats.columns.values]
        cluster_stats = cluster_stats.rename(columns={'CustomerID_count': 'Count'})
        
        return cluster_stats


def plot_elbow_method(cluster_range: list, inertias: list, save_path: str = None):
    """Plot Elbow Method for optimal cluster selection."""
    plt.figure(figsize=(10, 6))
    plt.plot(cluster_range, inertias, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Inertia', fontsize=12)
    plt.title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_silhouette_scores(cluster_range: list, silhouette_scores: list, save_path: str = None):
    """Plot Silhouette Scores for different cluster numbers."""
    plt.figure(figsize=(10, 6))
    plt.plot(cluster_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Silhouette Score', fontsize=12)
    plt.title('Silhouette Score Analysis', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
