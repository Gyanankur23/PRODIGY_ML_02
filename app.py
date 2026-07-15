"""
Streamlit application for Customer Segmentation Dashboard.
"""
import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import DataPreprocessor
from clustering import KMeansClustering
from visualization import ClusterVisualizer


# Page configuration
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_data():
    """Load and preprocess data."""
    # Get the directory where this script is located
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data', 'Mall_Customers.csv')
    preprocessor = DataPreprocessor(data_path)
    original_data, scaled_features = preprocessor.preprocess()
    return original_data, scaled_features, preprocessor


@st.cache_resource
def load_model():
    """Load trained model."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'models', 'kmeans_model.pkl')
    if os.path.exists(model_path):
        clustering = KMeansClustering(n_clusters=5, random_state=42)
        clustering.load_model(model_path)
        return clustering
    return None


def main():
    """Main application."""
    # Title
    st.markdown('<h1 class="main-title">🎯 Customer Segmentation Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading data..."):
        original_data, scaled_features, preprocessor = load_data()
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    # Model training option
    train_model = st.sidebar.button("🔄 Train New Model")
    
    if train_model:
        with st.spinner("Training model..."):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            clustering = KMeansClustering(n_clusters=5, random_state=42)
            labels = clustering.fit(scaled_features)
            clustering.save_model(os.path.join(base_dir, 'models', 'kmeans_model.pkl'))
            st.sidebar.success("Model trained successfully!")
            st.rerun()
    
    # Load model
    clustering = load_model()
    
    if clustering is None:
        st.error("❌ Model not found. Please ensure the trained model is included in the deployment.")
        st.stop()
    
    labels = clustering.predict(scaled_features)
    
    # Add labels to data
    original_data['Cluster'] = labels
    
    # Metrics
    st.markdown('<div class="section-header">📊 Key Metrics</div>', 
                unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", len(original_data))
    
    with col2:
        st.metric("Number of Clusters", len(np.unique(labels)))
    
    with col3:
        sil_score = clustering.silhouette_score if clustering.silhouette_score is not None else "N/A"
        st.metric("Silhouette Score", sil_score if sil_score == "N/A" else f"{sil_score:.4f}")
    
    with col4:
        inertia = clustering.inertia if clustering.inertia is not None else "N/A"
        st.metric("Model Inertia", inertia if inertia == "N/A" else f"{inertia:.2f}")
    
    # Data Overview
    st.markdown('<div class="section-header">📋 Data Overview</div>', 
                unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sample Data")
        st.dataframe(original_data.head(10), use_container_width=True)
    
    with col2:
        st.subheader("Data Statistics")
        st.dataframe(original_data.describe(), use_container_width=True)
    
    # Cluster Analysis
    st.markdown('<div class="section-header">🎯 Cluster Analysis</div>', 
                unsafe_allow_html=True)
    
    # Cluster distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Customer Distribution by Cluster")
        cluster_counts = original_data['Cluster'].value_counts().sort_index()
        fig_dist = px.bar(
            x=cluster_counts.index,
            y=cluster_counts.values,
            labels={'x': 'Cluster', 'y': 'Number of Customers'},
            color=cluster_counts.index,
            color_continuous_scale='viridis'
        )
        fig_dist.update_layout(showlegend=False)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        st.subheader("Cluster Statistics")
        cluster_stats = original_data.groupby('Cluster').agg({
            'Age': 'mean',
            'Annual Income (k$)': 'mean',
            'Spending Score (1-100)': 'mean',
            'CustomerID': 'count'
        }).round(2)
        cluster_stats.columns = ['Avg Age', 'Avg Income (k$)', 
                                'Avg Spending Score', 'Count']
        st.dataframe(cluster_stats, use_container_width=True)
    
    # 2D Visualizations
    st.markdown('<div class="section-header">📈 2D Cluster Visualizations</div>', 
                unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    feature_names = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    feature_pairs = [
        ('Age', 'Annual Income (k$)'),
        ('Age', 'Spending Score (1-100)'),
        ('Annual Income (k$)', 'Spending Score (1-100)')
    ]
    
    for idx, (col, (feat1, feat2)) in enumerate(zip([col1, col2, col3], feature_pairs)):
        with col:
            st.subheader(f"{feat1} vs {feat2}")
            fig = px.scatter(
                original_data,
                x=feat1,
                y=feat2,
                color='Cluster',
                color_continuous_scale='viridis',
                hover_data=['CustomerID', 'Gender']
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    # 3D Visualization
    st.markdown('<div class="section-header">🌐 3D Cluster Visualization</div>', 
                unsafe_allow_html=True)
    
    visualizer = ClusterVisualizer()
    fig_3d = visualizer.plot_interactive_3d(scaled_features, labels, feature_names)
    st.plotly_chart(fig_3d, use_container_width=True)
    
    # Cluster Profiles
    st.markdown('<div class="section-header">👥 Cluster Profiles</div>', 
                unsafe_allow_html=True)
    
    cluster_descriptions = {
        0: "🎯 **Target Customers**: High income, high spending score. Ideal for premium marketing.",
        1: "💰 **Conservative Spenders**: High income, low spending score. Potential for upselling.",
        2: "🎓 **Young Professionals**: Lower income, moderate spending. Growth potential.",
        3: "🛒 **Regular Shoppers**: Average income, average spending. Loyal customers.",
        4: "⚡ **Impulse Buyers**: Moderate income, high spending. Responsive to promotions."
    }
    
    for cluster_id, description in cluster_descriptions.items():
        with st.expander(f"Cluster {cluster_id}"):
            st.markdown(description)
            
            cluster_data = original_data[original_data['Cluster'] == cluster_id]
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Customers", len(cluster_data))
            with col2:
                st.metric("Avg Age", f"{cluster_data['Age'].mean():.1f}")
            with col3:
                st.metric("Avg Income", f"${cluster_data['Annual Income (k$)'].mean():.1f}k")
            
            st.dataframe(cluster_data[['CustomerID', 'Gender', 'Age', 
                                      'Annual Income (k$)', 'Spending Score (1-100)']].head(5),
                        use_container_width=True)
    
    # Download option
    st.markdown('<div class="section-header">💾 Export Data</div>', 
                unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = original_data.to_csv(index=False)
        st.download_button(
            label="Download Data with Clusters",
            data=csv,
            file_name=f'customer_segments_{datetime.now().strftime("%Y%m%d")}.csv',
            mime='text/csv'
        )
    
    with col2:
        cluster_stats_csv = cluster_stats.to_csv()
        st.download_button(
            label="Download Cluster Statistics",
            data=cluster_stats_csv,
            file_name=f'cluster_stats_{datetime.now().strftime("%Y%m%d")}.csv',
            mime='text/csv'
        )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Customer Segmentation Dashboard | PRODIGY ML Project</p>
        <p>Built with Streamlit, Scikit-learn, and Plotly</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
