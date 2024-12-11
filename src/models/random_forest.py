# models/random_forest.py
from sklearn.ensemble import RandomForestRegressor
from models.train import fine_tune_hyperparameters  # Import from models/train.py

# Function to fine-tune and return the best parameters for RandomForestRegressor
def fine_tune_random_forest(X, y):
    """
    Fine-tune RandomForestRegressor hyperparameters.
    
    Parameters:
    - X: Input features
    - y: Target variable
    
    Returns:
    - best_rf_params: dict, the best hyperparameters found
    """
    rf_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
    }

    rf_model = RandomForestRegressor(random_state=42)
    best_rf_params = fine_tune_hyperparameters(rf_model, X, y, rf_param_grid)
    return best_rf_params