import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from models.train import run_regression_model

# Function to plot predictions vs actual values with error bars
def plot_model_predictions(X, y, model, num_trials=100, output_dir=None):
    """
    Run a model, collect predictions, and plot results. Optionally save the plot.
    
    Parameters:
    - X: Input features
    - y: Target variable
    - model: Trained regression model
    - num_trials: Number of train-test splits to evaluate
    - output_dir: Directory to save the plot (optional)
    """
    avg_train_mse, avg_test_mse, predictions_df = run_regression_model(X, y, model)
    regression_model_name = model.__class__.__name__

    print(f'Average Train MSE: {avg_train_mse}')
    print(f'Average Test MSE: {avg_test_mse}')

    # Plot errors
    plot_model_errors(
        [predictions_df],
        [f'{regression_model_name} ({num_trials} trials)'],
        output_dir=output_dir,
        filename=f"{regression_model_name}_predictions.png"
    )


def plot_model_errors(predictions_dfs, model_names, output_dir=None, filename="plot.png"):
    """
    Plot predictions vs actual values with error bars and save to file if output_dir is specified.
    
    Parameters:
    - predictions_dfs: List of DataFrames with predictions
    - model_names: List of model names for labeling
    - output_dir: Directory to save the plot (optional)
    - filename: Name of the file to save the plot as (default: 'plot.png')
    """
    plt.figure(figsize=(10, 10))

    for predictions_df, model_name in zip(predictions_dfs, model_names):
        means, stds = [], []

        for y in predictions_df.columns:
            vals = [val for val in predictions_df[y] if not pd.isna(val)]
            means.append(np.mean(vals))
            stds.append(np.std(vals))

        plt.errorbar(predictions_df.columns, means, yerr=stds, fmt='o', capsize=5, label=model_name)

    plt.plot([min(predictions_df.columns), max(predictions_df.columns)],
             [min(predictions_df.columns), max(predictions_df.columns)],
             color='red', linestyle='--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Predictions vs Actual Values with Error Bars')
    plt.legend()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, filename)
        plt.savefig(file_path)
        print(f"Plot saved to {file_path}")
    else:
        plt.show()

    plt.close()