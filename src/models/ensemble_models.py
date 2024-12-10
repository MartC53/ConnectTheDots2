from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

def train_ensemble(X_train, y_train, X_test, y_test):
    # Define base models
    base_models = [
        ('rf', RandomForestRegressor(random_state=42, n_estimators=100)),
        ('xgb', XGBRegressor(objective='reg:squarederror', random_state=42, n_estimators=100))
    ]

    # Define meta-model
    meta_model = LinearRegression()

    # Create Stacking Regressor
    model = StackingRegressor(estimators=base_models, final_estimator=meta_model)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model, mse, r2