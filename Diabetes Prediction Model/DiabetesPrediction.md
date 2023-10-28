# Diabetes Prediction Machine Learning Model

This repository contains code for a machine learning model that predicts the likelihood of an individual having diabetes based on certain health indicators. The model uses the Support Vector Machine (SVM) algorithm for classification.

## Code Overview

### Importing Dependencies
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
```

### Data Analysis
- Load and explore the dataset.
- Check the shape and statistical summary of the data.
- Display the count of diabetic and non-diabetic cases.
- Calculate and display the mean values for different features based on the outcome (diabetic or non-diabetic).

### Data Standardization
- Standardize the input data to have zero mean and unit variance.

### Splitting Data
- Split the dataset into training and testing sets for model evaluation.

### Training Model
- Train a Support Vector Machine (SVM) classifier with a linear kernel using the training data.

### Model Evaluation
- Calculate and display the accuracy score for the model on both the training and testing datasets.

### Making Predictive Model
- Use the trained model to predict diabetes based on custom input data.
- Standardize the input data.
- Make predictions and print the result as "Diabetic" or "Non-Diabetic."

### Model Persistence
- Save the trained model to a file using the `pickle` library.

## Usage
1. Ensure you have the required dependencies installed, preferably in a virtual environment.
2. Run the code to train the model, evaluate its performance, and make predictions.
3. You can use the saved model for future predictions.

## Dataset
The dataset used for this project is the 'diabetes.csv' file, which contains information about various health indicators and whether or not the individuals are diabetic.

## Dependencies
- numpy
- pandas
- scikit-learn

## Model Persistence
The trained model is saved as 'diabetes_model.sav' using the `pickle` library.

Feel free to clone this repository, explore the code, and use the model for your own diabetes prediction tasks.
