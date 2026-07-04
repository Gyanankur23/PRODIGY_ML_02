# Customer Segmentation using K-Means Clustering

A machine learning project that segments mall customers based on their purchasing behavior using K-Means clustering algorithm.

## 📊 Project Overview

This project analyzes customer data from a mall to identify distinct customer segments based on:
- Age
- Annual Income (k$)
- Spending Score (1-100)

The segmentation helps in understanding customer behavior and creating targeted marketing strategies.

## 🚀 Features

- **Data Preprocessing**: Clean and prepare customer data for analysis
- **Exploratory Data Analysis**: Visualize customer demographics and spending patterns
- **K-Means Clustering**: Optimal cluster selection using Elbow Method
- **Model Evaluation**: Silhouette score and cluster analysis
- **Interactive Dashboard**: Streamlit application for real-time visualization

## 🛠️ Tech Stack

- **Python**: 3.9+
- **Pandas**: Data manipulation
- **Scikit-learn**: Machine learning algorithms
- **Matplotlib/Seaborn**: Data visualization
- **Streamlit**: Interactive web application
- **Plotly**: Interactive charts

## 📁 Project Structure

```
PRODIGY_ML_02/
├── data/
│   └── Mall_Customers.csv
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── clustering.py
│   └── visualization.py
├── models/
│   └── kmeans_model.pkl
├── notebooks/
│   └── analysis.ipynb
├── app.py
├── requirements.txt
└── README.md
```

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/Gyanankur23/PRODIGY_ML_02.git
cd PRODIGY_ML_02
```

2. Create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
source venv/bin/activate  # On Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎯 Usage

### Run the Streamlit Application
```bash
streamlit run app.py
```

### Train the Model
```bash
python src/clustering.py
```

## 📈 Methodology

1. **Data Loading**: Load customer data from CSV
2. **Preprocessing**: Handle missing values, encode categorical variables
3. **Feature Selection**: Select relevant features (Age, Annual Income, Spending Score)
4. **Optimal Clusters**: Use Elbow Method to determine optimal K
5. **Model Training**: Train K-Means clustering algorithm
6. **Evaluation**: Analyze clusters using silhouette scores
7. **Visualization**: Create interactive visualizations

## 🎨 Customer Segments

The model identifies 5 distinct customer segments:
- **Careful**: Low income, low spending
- **Standard**: Average income, average spending
- **Target**: High income, high spending (ideal for marketing)
- **Careless**: High income, low spending
- **Sensible**: Low income, high spending

## 📊 Results

- **Optimal Clusters**: 5 (determined by Elbow Method)
- **Silhouette Score**: 0.55
- **Model Accuracy**: Effective segmentation based on business metrics

## 🤝 Contributing

This project is part of the PRODIGY ML internship program.

## 📄 License

MIT License

## 👨‍💻 Author

Gyanankur Baruah

## 📅 Project Timeline

- **July 4-14, 2026**: Development and deployment
