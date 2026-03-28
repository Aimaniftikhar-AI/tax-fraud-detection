# Tech Tax Fraud Detection System

A Machine Learning-based web application built using **Streamlit** that detects tax fraud using multiple classification models and provides a complete ML pipeline including preprocessing, SMOTE balancing, model training, evaluation, and prediction.

## Features

- Upload CSV dataset
- Dataset preview & statistics
- Data preprocessing:
  - Missing value handling
  - One-hot encoding
  - Outlier removal (IQR method)
  - Feature scaling (StandardScaler)
- Class balancing using SMOTE
- Train 5 ML models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
- Model evaluation:
  - Accuracy, Precision, Recall, F1 Score
  - Confusion matrices
- Overfitting detection
- Auto best model selection
- Real-time prediction module
- Visualizations (heatmaps, bar charts, comparisons)
- Custom modern UI with CSS styling

## Machine Learning Pipeline

1. Data Upload
2. Data Cleaning
3. Missing Value Handling
4. Encoding Categorical Variables
5. Outlier Removal (IQR)
6. Feature Scaling
7. SMOTE Balancing
8. Train/Test Split
9. Model Training
10. Evaluation
11. Prediction Interface
