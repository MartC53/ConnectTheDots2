import sys
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Add src to PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from utils.visualization import plot_model_predictions, plot_model_errors
from utils.model_io import save_model
from models.linear_regression import fine_tune_linear_model
from models.random_forest import fine_tune_random_forest
from models.xgboost import fine_tune_xgboost
from models.train import run_regression_model

# Step 1: Load Dataset
data_path = "data/processed/structured_data_12_4.csv"
df = pd.read_csv(data_path)

# Step 2: Define features and target
X = df.drop(['copy_number', 'log_copy_number', 'time_to_threshold'], axis=1)
y = df['log_copy_number']

# Step 3: Optimize hyperparameters for each model
print("Optimizing Linear Regression...")
_, linear_model = fine_tune_linear_model(X, y, model_type='linear')

print("Optimizing Random Forest...")
best_rf_params = fine_tune_random_forest(X, y)
rf_model = RandomForestRegressor(random_state=42, **best_rf_params)
rf_model.fit(X, y)

print("Optimizing XGBoost...")
best_xgb_params = fine_tune_xgboost(X, y)
xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42, **best_xgb_params)
xgb_model.fit(X, y)

# Step 4: Evaluate Models
print("Evaluating Linear Regression...")
lr_avg_train_mse, lr_avg_test_mse, lr_predictions_df = run_regression_model(X, y, linear_model)

print("Evaluating Random Forest...")
rf_avg_train_mse, rf_avg_test_mse, rf_predictions_df = run_regression_model(X, y, rf_model)

print("Evaluating XGBoost...")
xgb_avg_train_mse, xgb_avg_test_mse, xgb_predictions_df = run_regression_model(X, y, xgb_model)

# Step 5: Plot Results
models_results = {
    "Linear Regression": (lr_avg_train_mse, lr_avg_test_mse, lr_predictions_df, linear_model),
    "Random Forest": (rf_avg_train_mse, rf_avg_test_mse, rf_predictions_df, rf_model),
    "XGBoost": (xgb_avg_train_mse, xgb_avg_test_mse, xgb_predictions_df, xgb_model),
}

models_dir = "result"
os.makedirs(models_dir, exist_ok=True)

for model_name, (avg_train_mse, avg_test_mse, predictions_df, model) in models_results.items():
    print(f"{model_name} Metrics:")
    print(f"  Average Train MSE: {avg_train_mse}")
    print(f"  Average Test MSE: {avg_test_mse}")

    # Save plots to the result directory
    print(f"Plotting and saving predictions for {model_name}...")
    plot_model_predictions(X, y, model, output_dir=models_dir)

# Step 6: Save the Best Model
best_model_name = min(models_results, key=lambda x: models_results[x][1])  # Based on lowest test MSE
best_model = models_results[best_model_name][3]  # Retrieve the model instance

model_path = os.path.join(models_dir, f"{best_model_name.replace(' ', '_').lower()}.pkl")
save_model(best_model, model_path)

print(f"The best model '{best_model_name}' has been saved to {model_path}.")