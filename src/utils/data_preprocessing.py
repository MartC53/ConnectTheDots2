import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data(file_path, drop_columns, target_column, test_size=0.2, random_state=42):
    df = pd.read_csv(file_path)
    X = df.drop(drop_columns, axis=1)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test