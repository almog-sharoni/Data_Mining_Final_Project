# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb

# Load the data
train_data = pd.read_csv('hotels_train.csv')
test_data = pd.read_csv('hotels_test.csv')

# Define models
models = {
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'XGBoost': xgb.XGBClassifier()
}


# Phase 1: Data Exploration and Preprocessing

def explore_data(data):
    # Display basic information and statistics
    print(data.info())
    print(data.describe())

    # Check for missing values
    print(data.isnull().sum())

    # Visualize distributions and correlations
    sns.pairplot(data)
    plt.show()
    # sns.heatmap(data.drop().corr(), annot=True)
    # plt.show()


def preprocess_data(data):
    # Encode categorical variables
    data = pd.get_dummies(data, drop_first=True)

    # Handle missing values
    data = data.fillna(data.median())

    # Feature scaling
    scaler = StandardScaler()
    numerical_features = data.select_dtypes(include=['int64', 'float64']).columns
    data[numerical_features] = scaler.fit_transform(data[numerical_features])

    return data


# Exploratory Data Analysis
explore_data(train_data)

# Data Preprocessing
train_data_clean = preprocess_data(train_data)
test_data_clean = preprocess_data(test_data)

# Phase 2: Model Training and Evaluation

# Split data into training and validation sets
X = train_data_clean.drop('is_canceled', axis=1)
y = train_data_clean['is_canceled']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


def train_and_evaluate_models(X_train, y_train, X_val, y_val):
    # Train and evaluate each model
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        print(f'{name} Evaluation:')
        print(f'Accuracy: {accuracy_score(y_val, y_pred)}')
        print(f'Precision: {precision_score(y_val, y_pred)}')
        print(f'Recall: {recall_score(y_val, y_pred)}')
        print(f'F1 Score: {f1_score(y_val, y_pred)}')
        print(confusion_matrix(y_val, y_pred))
        print('-' * 30)


train_and_evaluate_models(X_train, y_train, X_val, y_val)


# Phase 3: Hyperparameter Tuning
def hyperparameter_tuning(X_train, y_train):
    param_grid = {
        'Random Forest': {
            'n_estimators': [100, 200],
            'max_depth': [10, 20]
        },
        'Gradient Boosting': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1]
        },
        'XGBoost': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1]
        }
    }

    best_models = {}

    for name, params in param_grid.items():
        grid_search = GridSearchCV(models[name], params, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        best_models[name] = grid_search.best_estimator_
        print(f'Best parameters for {name}: {grid_search.best_params_}')

    return best_models


best_models = hyperparameter_tuning(X_train, y_train)


# Phase 4: Final Prediction and Report Generation
def final_predictions(best_model, X_test):
    predictions = best_model.predict(X_test)
    submission = pd.DataFrame({'ID': test_data['ID'], 'is_canceled': predictions})
    submission.to_csv('predictions.csv', index=False)


# Choose the best model based on cross-validation performance
best_model = best_models['XGBoost']  # Replace with the chosen model
final_predictions(best_model, test_data_clean)

# Ensure all steps and results are well documented and visualized

