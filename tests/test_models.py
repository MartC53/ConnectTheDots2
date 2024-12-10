import pytest
from sklearn.datasets import make_regression
from src.models.boosting import train_random_forest, train_xgboost
from src.models.ensemble_models import train_ensemble

@pytest.fixture
def regression_data():
    X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)
    return X[:80], X[80:], y[:80], y[80:]

def test_random_forest(regression_data):
    X_train, X_test, y_train, y_test = regression_data
    model, mse, r2 = train_random_forest(X_train, y_train, X_test, y_test)
    assert mse > 0
    assert 0 <= r2 <= 1

def test_xgboost(regression_data):
    X_train, X_test, y_train, y_test = regression_data
    model, mse, r2 = train_xgboost(X_train, y_train, X_test, y_test)
    assert mse > 0
    assert 0 <= r2 <= 1

def test_ensemble(regression_data):
    X_train, X_test, y_train, y_test = regression_data
    model, mse, r2 = train_ensemble(X_train, y_train, X_test, y_test)
    assert mse > 0
    assert 0 <= r2 <= 1