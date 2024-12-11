import joblib
import os

def save_model(model, filepath):
    """
    Save a trained model to a file.
    
    Parameters:
    - model: The trained model instance to save.
    - filepath: str, path to the file where the model will be saved.
    """
    # Ensure the directory exists before saving
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    try:
        joblib.dump(model, filepath)
        print(f"Model saved successfully to {filepath}")
    except Exception as e:
        print(f"Error saving model to {filepath}: {e}")

def load_model(filepath):
    """
    Load a trained model from a file.
    
    Parameters:
    - filepath: str, path to the file from which the model will be loaded.
    
    Returns:
    - The loaded model instance.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No such file: {filepath}")
    try:
        model = joblib.load(filepath)
        print(f"Model loaded successfully from {filepath}")
        return model
    except Exception as e:
        print(f"Error loading model from {filepath}: {e}")
        raise