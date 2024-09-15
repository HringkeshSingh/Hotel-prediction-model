# Hotel Booking Cancellation Prediction Model

## Overview

This project implements a machine learning model to predict hotel booking cancellations. The model is trained on a dataset of hotel bookings and can be used to estimate the likelihood of a booking being canceled based on various features.

## Features

- Prediction of hotel booking cancellations using Logistic Regression
- Feature selection using Lasso regularization
- Data preprocessing and cleaning

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- seaborn
- matplotlib
- plotly
- sort_dataframeby_monthorweek

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/HringkeshSingh/hotel-prediction-model.git
   cd hotel-booking-cancellation-model
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate
 # On Windows,
    use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install pandas numpy scikit-learn seaborn matplotlib plotly
   ```

4. Install additional required packages:
   ```
   pip install sorted-months-weekdays
   pip install sort_dataframeby_monthorweek
   ```

5. Download the dataset:
   - Place the 'hotel_bookings.csv' file in the project directory.

## Data Cleaning and Preprocessing

The model uses a dataset of hotel bookings. The following preprocessing steps are applied:

1. Removal of bookings with zero guests
2. Handling of missing values
3. Feature engineering (e.g., 'family' and 'total_customers' features)
4. Encoding of categorical variables

## Feature Selection

Important features for the model are selected using Lasso regularization with `SelectFromModel` from scikit-learn.

## Model

The prediction model uses Logistic Regression. The model is trained and evaluated using the following steps:

1. Splitting the data into training and test sets (75% train, 25% test)
2. Training the Logistic Regression model
3. Making predictions on the test set
4. Evaluating the model using confusion matrix and accuracy score
5. Performing cross-validation to assess model stability

## Usage

To use this model:

1. Ensure you have completed the installation steps.
2. Open a Python environment (e.g., Jupyter Notebook or Python script).
3. Load your hotel booking data:
   ```python
   import pandas as pd
   from pathlib import Path

   file_path = Path("hotel_bookings.csv")
   df = pd.read_csv(file_path)
   ```
4. Run the data cleaning and preprocessing steps as outlined in the provided code.
5. Perform feature selection.
6. Split your data into training and test sets.
7. Train the Logistic Regression model.
8. Use the trained model to make predictions on new data.

## Model Performance

The model's performance is evaluated using:

- Confusion matrix
- Accuracy score
- Cross-validation scores

## Future Improvements

- Experiment with other machine learning algorithms (e.g., Random Forest, XGBoost)
- Implement hyperparameter tuning
- Explore more feature engineering opportunities
- Implement model explainability techniques (e.g., SHAP values)

## Contributing

Contributions to improve the model are welcome. Please feel free to submit a Pull Request.

