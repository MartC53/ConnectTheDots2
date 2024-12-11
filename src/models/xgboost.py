from xgboost import XGBRegressor
from models.train import fine_tune_hyperparameters  # Import the function from train.py

# Function to fine-tune and return the best parameters for XGBoost
def fine_tune_xgboost(X, y):
    """
    Fine-tune XGBoostRegressor hyperparameters.
    
    Parameters:
    - X: Input features
    - y: Target variable
    
    Returns:
    - best_xgb_params: dict, the best hyperparameters found
    """
    # Define hyperparameter grid for fine-tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 5, 6],
        'learning_rate': [0.05, 0.1, 0.15],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9],
        'reg_alpha': [0, 0.1],
        'reg_lambda': [1, 1.5],
    }

    # Initialize XGBoost model
    xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)

    # Fine-tune hyperparameters
    best_xgb_params = fine_tune_hyperparameters(xgb_model, X, y, param_grid)
    return best_xgb_params