# Breast Cancer Prediction Machine Learning Model

This repository contains a machine learning model for breast cancer prediction using the Breast Cancer Wisconsin (Diagnostic) Data Set. The model is implemented in Python and uses the scikit-learn library for building a Support Vector Machine (SVM) classifier. The goal of this model is to classify breast cancer tumors as either malignant (M) or benign (B) based on various features computed from cell nuclei.

## Dataset Information

The dataset used for this model contains the following attributes:

1. Radius (Mean of Distances from Center to Points on the Perimeter)
2. Texture (Standard Deviation of Gray-Scale Values)
3. Perimeter
4. Area
5. Smoothness (Local Variation in Radius Lengths)
6. Compactness (Perimeter^2 / Area - 1.0)
7. Concavity (Severity of Concave Portions of the Contour)
8. Concave Points (Number of Concave Portions of the Contour)
9. Symmetry
10. Fractal Dimension ("Coastline Approximation" - 1)

The target variable is the "Diagnosis," where M stands for Malignant and B stands for Benign. The dataset contains a total of 569 instances, with 357 benign cases and 212 malignant cases.

## Model Overview

List of dependencies and libraries required for running the code:

```python
# Dependencies
- Python 3.x
- scikit-learn
- pandas
- joblib

# Installation Instructions

# 1. Python 3.x
# You should have Python 3.x installed. If not, you can download it from the official website: https://www.python.org/downloads/

# 2. scikit-learn
# Install scikit-learn using pip:
# Make sure you have pip (Python package manager) installed.
pip install scikit-learn

# 3. pandas
# Install pandas using pip:
pip install pandas

# 4. joblib
# Install joblib using pip:
pip install joblib
```

These instructions assume that you have Python already installed on your system. You can use a virtual environment to manage dependencies if needed.
You can also add installation instructions if needed.

The model follows the following steps:

1. **Data Loading**: The dataset is loaded using pandas.

2. **Data Preprocessing**:
   - The "Outcome" column is mapped to numerical values, with Malignant (M) mapped to 1 and Benign (B) mapped to 0.
   - Features with low correlation to the target variable are dropped.

3. **Data Standardization**: The feature data is standardized using the StandardScaler to ensure all features have a mean of 0 and standard deviation of 1.

4. **Data Splitting**: The dataset is split into training and testing sets. 80% of the data is used for training, and 20% is used for testing.

5. **Model Training**: A Support Vector Machine (SVM) classifier with a linear kernel is used to train the model.

6. **Model Evaluation**: The accuracy of the model is calculated on both the training and testing data.

7. **Model Saving**: The trained model is saved using joblib.

8. **Model Loading**: The saved model can be loaded for making predictions on new data.

## Usage

### Train the Model

To train the model, you can run the code provided in the Jupyter notebook or Python script. Make sure you have the required libraries (scikit-learn, pandas, joblib) installed.

### Make Predictions

You can use the trained model to make predictions on new data. Here's an example of how to use the model for prediction:

```python
from joblib import load

# Load the model
model = load('breastcancer_model.joblib')

# Input data for prediction
input_data = (17.99, 122.80, 1001.0, 0.27760, 0.30010, 0.14710)

# Standardize the input data
inp = scaler.transform([input_data])

# Make a prediction
prediction = model.predict(inp)

if int(prediction[0]) == 1:
    print('Malignant')
else:
    print('Benign')
```

You can replace `input_data` with your own feature values for prediction.

### Model File

The trained model file is included in this repository as "breastcancer_model.joblib."

Feel free to use this model for breast cancer prediction or as a basis for further exploration and improvement.
