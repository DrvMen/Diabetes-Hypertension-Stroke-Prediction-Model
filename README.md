# Diabetes, Stroke, and Hypertension Prediction Model

This project develops and evaluates machine learning models to predict the likelihood of diabetes, stroke, and hypertension based on input features. The notebook integrates various preprocessing steps, exploratory data analysis, and multiple classification algorithms to achieve optimal predictive performance.

## Project Description

This notebook explores a dataset containing health-related features and diagnoses. The goal is to build predictive models for three target variables: diabetes, stroke, and hypertension. The models were evaluated using metrics like accuracy and confusion matrices.

## Features

- Data preprocessing, including scaling and dimensionality reduction using PCA.
- Multiple classification algorithms:
  - Random Forest Classifier
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - XGBoost Classifier
  - Linear Regression (adapted for binary classification)
- Model performance comparison using accuracy scores and visualization.
- Generation of confusion matrices for detailed evaluation.

## Requirements

The following libraries are used in this project:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`

Ensure you have these libraries installed in your Python environment.

## Dataset

The notebook uses a dataset named `merged_data.csv`, which should include health-related features for prediction. Replace this dataset path with your local file path if necessary.

## Workflow

1. **Data Loading and Preprocessing**:
   - Read the dataset.
   - Perform data cleaning and preprocessing.
   - Apply scaling and dimensionality reduction.

2. **Model Training and Evaluation**:
   - Train models using `train_test_split` for each target variable.
   - Evaluate using metrics like accuracy and classification reports.
   - Visualize confusion matrices and performance comparisons.

3. **Results Visualization**:
   - Compare model performance using bar charts.
   - Analyze confusion matrices for detailed insights.

## Models

The following models were trained and evaluated:

- Random Forest Classifier
- Logistic Regression
- K-Nearest Neighbors (KNN)
- XGBoost Classifier
- Linear Regression

Each model's hyperparameters were tuned for better accuracy and efficiency.

## Usage

To use this notebook:

1. Clone the repository or download the `.ipynb` file.
2. Place your dataset in the same directory.
3. Open the notebook in Jupyter Notebook or Jupyter Lab.
4. Run the cells sequentially to reproduce the results.

## Results

The notebook compares model accuracy using visualization tools and highlights the best-performing model for each target variable.


