import pytest
import pandas as pd
from src.utils.data_preprocessing import load_and_split_data
from src.utils.visualization import plot_predictions
import matplotlib.pyplot as plt


# Test 1: Data Loading and Splitting with Original Data
def test_load_and_split_data_with_original_data():
    """Test load_and_split_data function using original data."""
    file_path = 'data/processed/structured_data_11_17.csv'  # Path to your actual dataset
    drop_columns = ['copy_number', 'time_to_threshold']
    target_column = 'log_copy_number'

    # Call the function
    X_train, X_test, y_train, y_test = load_and_split_data(
        file_path, drop_columns, target_column
    )

    # Assertions for splitting
    assert X_train.shape[0] > 0, "X_train should not be empty"
    assert X_test.shape[0] > 0, "X_test should not be empty"
    assert y_train.shape[0] > 0, "y_train should not be empty"
    assert y_test.shape[0] > 0, "y_test should not be empty"

    # Assertions for correct feature dropping
    expected_features = pd.read_csv(file_path).shape[1] - len(drop_columns) - 1  # Target column is also dropped
    assert X_train.shape[1] == expected_features, "Unexpected number of features after dropping columns"

    # Assertions for target column
    assert target_column not in X_train.columns, f"Target column '{target_column}' should not be in features"


# Test 2: Plot Predictions with Original Data
def test_plot_predictions_with_original_data():
    """Test the plot_predictions function using predictions from actual data."""
    file_path = 'data/processed/structured_data_11_17.csv'
    drop_columns = ['copy_number', 'time_to_threshold']
    target_column = 'log_copy_number'

    # Load and split the data
    X_train, X_test, y_train, y_test = load_and_split_data(
        file_path, drop_columns, target_column
    )

    # Use a simple model for testing (e.g., Linear Regression)
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Call the plotting function
    plot_predictions(y_test, y_pred, "Linear Regression")

    # Assert that a plot was generated
    assert plt.gcf() is not None, "A figure should have been created by the plot function"
    plt.close()  # Close the plot after the test to clean up