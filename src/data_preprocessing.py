import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple


class DataPreprocessor:
    """Handle data preprocessing for customer segmentation."""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.data = None
        self.features = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
    
    def load_data(self) -> pd.DataFrame:
        """Load the customer dataset."""
        self.data = pd.read_csv(self.filepath)
        return self.data
    
    def handle_missing_values(self) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        if self.data.isnull().sum().sum() > 0:
            self.data = self.data.dropna()
        return self.data
    
    def encode_categorical(self) -> pd.DataFrame:
        """Encode categorical variables."""
        if 'Gender' in self.data.columns:
            self.data['Gender'] = self.label_encoder.fit_transform(self.data['Gender'])
        return self.data
    
    def select_features(self, feature_columns: list = None) -> pd.DataFrame:
        """Select features for clustering."""
        if feature_columns is None:
            feature_columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
        
        self.features = self.data[feature_columns].values
        return self.data[feature_columns]
    
    def scale_features(self) -> np.ndarray:
        """Scale features using StandardScaler."""
        self.features = self.scaler.fit_transform(self.features)
        return self.features
    
    def preprocess(self, feature_columns: list = None) -> Tuple[pd.DataFrame, np.ndarray]:
        """Complete preprocessing pipeline."""
        self.load_data()
        self.handle_missing_values()
        self.encode_categorical()
        self.select_features(feature_columns)
        scaled_features = self.scale_features()
        
        return self.data, scaled_features
    
    def get_feature_names(self) -> list:
        """Get the names of selected features."""
        if self.features is not None:
            return ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
        return []
    
    def inverse_transform(self, scaled_data: np.ndarray) -> np.ndarray:
        """Inverse transform scaled data back to original scale."""
        return self.scaler.inverse_transform(scaled_data)
