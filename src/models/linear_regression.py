from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from models.train import fine_tune_hyperparameters  # Import the function for consistency

# Function to fine-tune Ridge or Lasso regression (or simply use LinearRegression)
def fine_tune_linear_model(X, y, model_type='linear'):
    """
    Fine-tune hyperparameters for Ridge or Lasso regression, or train LinearRegression.
    
    Parameters:
    - X: Input features
    - y: Target variable
    - model_type: str, one of 'linear', 'ridge', 'lasso'
    
    Returns:
    - best_params: dict, best hyperparameters (for Ridge or Lasso) or None (for LinearRegression)
    - model: Trained model instance
    """
    if model_type == 'linear':
        # Simple Linear Regression (no hyperparameters to tune)
        model = LinearRegression()
        model.fit(X, y)
        return None, model

    elif model_type == 'ridge':
        # Ridge Regression with hyperparameter tuning
        param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}  # Regularization strength
        model = Ridge(random_state=42)
    elif model_type == 'lasso':
        # Lasso Regression with hyperparameter tuning
        param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}  # Regularization strength
        model = Lasso(random_state=42)
    else:
        raise ValueError("Invalid model_type. Choose from 'linear', 'ridge', or 'lasso'.")

    # Perform hyperparameter tuning
    best_params = fine_tune_hyperparameters(model, X, y, param_grid)
    model.set_params(**best_params)
    model.fit(X, y)

    return best_params, model