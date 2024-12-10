from src.models.boosting import train_random_forest, train_xgboost
from src.models.ensemble_models import train_ensemble

def train_all_models(X_train, y_train, X_test, y_test):
    # Train Random Forest
    rf_model, rf_mse, rf_r2 = train_random_forest(X_train, y_train, X_test, y_test)
    print(f"Random Forest - MSE: {rf_mse:.2f}, R²: {rf_r2:.2f}")

    # Train XGBoost
    xgb_model, xgb_mse, xgb_r2 = train_xgboost(X_train, y_train, X_test, y_test)
    print(f"XGBoost - MSE: {xgb_mse:.2f}, R²: {xgb_r2:.2f}")

    # Train Ensemble Model
    ensemble_model, ensemble_mse, ensemble_r2 = train_ensemble(X_train, y_train, X_test, y_test)
    print(f"Ensemble - MSE: {ensemble_mse:.2f}, R²: {ensemble_r2:.2f}")

    return {
        'Random Forest': (rf_model, rf_mse, rf_r2),
        'XGBoost': (xgb_model, xgb_mse, xgb_r2),
        'Ensemble': (ensemble_model, ensemble_mse, ensemble_r2)
    }