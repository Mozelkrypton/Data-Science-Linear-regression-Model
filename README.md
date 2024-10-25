# Data-Science-Linear-regression-Model
Data-Science Linear regression Model for Road Accident Severity.
# Accident Severity Prediction

## Overview
This project focuses on predicting the severity of accidents based on various features such as the driver's age band, experience, weather conditions, and road surface type. Using linear regression, we aim to understand the factors contributing to accident severity.

## Dependencies
To run this project, you will need the following Python packages:

- `pandas`
- `numpy`
- `sklearn`
- `joblib`

You can install these packages using pip:

```bash
pip install pandas numpy scikit-learn joblib
Data
The data used for this analysis is stored in a CSV file named cleaned.csv, which is in this repository
Code Overview
Importing Packages
The script begins by importing necessary packages and loading the dataset.

python
Copy code
import pandas as pd
import numpy as np
import os

os.chdir("C:\\Users\\amos\\Desktop\\year2sem2\\DataScience")
data = pd.read_csv("cleaned.csv")
Data Exploration
We explore the dataset to understand its structure and perform initial checks:

python
Copy code
data.head()
data.describe()
data.isnull().sum()
Feature Selection
Selected features relevant to predicting accident severity:

python
Copy code
selected_columns = [
    'Driving_experience',
    'Weather_conditions',
    'Road_surface_type',
    'Types_of_Junction',
    'Age_band_of_driver',
]
X = data[selected_columns]
y = data['Accident_severity']
Creating Dummy Variables
Converts categorical variables into dummy/indicator variables for model training:

python
Copy code
data_dummies = pd.get_dummies(data[columns_to_convert], drop_first=True)
Splitting the Dataset
The dataset is split into training and testing sets:

python
Copy code
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_dummies, y, test_size=0.2, random_state=42)
Model Training
A linear regression model is created and trained:

python
Copy code
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
Model Evaluation
The model is evaluated using Mean Squared Error (MSE) and R² score:

python
Copy code
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
Saving the Model
The trained model is saved for future use:

python
Copy code
import joblib
joblib.dump(model, 'accident_severity_model.pkl')
Making Predictions
You can make predictions using hypothetical input data:

python
Copy code
predicted_severity = model.predict(hypothetical_input)

Usage
Clone the repository or download the script.
Ensure that the cleaned.csv file is in the specified directory.
Run the script in your Python environment.

Contributing
Feel free to fork the repository and submit pull requests if you have suggestions for improvements or additional features
