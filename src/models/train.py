from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Generalized function to fine-tune hyperparameters using GridSearchCV
def fine_tune_hyperparameters(model: BaseEstimator, X, y, param_grid):
    """
    Fine-tune hyperparameters using GridSearchCV for any model.
    
    Parameters:
    - model: BaseEstimator, a scikit-learn compatible model instance
    - X: Input features
    - y: Target variable
    - param_grid: Hyperparameter grid for GridSearchCV
    
    Returns:
    - best_params_: dict, the best parameters found
    """
    # Perform Grid Search with 5-fold cross-validation
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=5,
        verbose=0,
        n_jobs=-1
    )

    # Fit the model to the entire dataset
    grid_search.fit(X, y)

    # Print and return the best parameters and best score
    print("Best Parameters:", grid_search.best_params_)
    print("Best CV Score (Negative MSE):", grid_search.best_score_)

    return grid_search.best_params_


# Generalized function to run regression models and collect predictions
def run_regression_model(X, y, model):
    predictions_dict = defaultdict(list)  # Store predictions per actual value
    train_mse = []  # Track train MSE across iterations
    test_mse = []  # Track test MSE across iterations

    for trial in range(100):  # 100 iterations for robust evaluation
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=trial)

        # Fit model
        model.fit(X_train, y_train)

        # Predict on train and test sets
        y_pred = model.predict(X_test)
        train_pred = model.predict(X_train)

        # Collect predictions
        for pred, actual in zip(y_pred, y_test):
            predictions_dict[actual].append(pred)

        # Calculate and store MSE
        train_mse.append(mean_squared_error(y_train, train_pred))
        test_mse.append(mean_squared_error(y_test, y_pred))

    # Convert predictions dictionary to DataFrame
    predictions_df = pd.DataFrame.from_dict(predictions_dict, orient='index').transpose()
    avg_train_mse = np.mean(train_mse)
    avg_test_mse = np.mean(test_mse)

    return avg_train_mse, avg_test_mse, predictions_df