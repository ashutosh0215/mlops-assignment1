from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def load_data() -> pd.DataFrame:

    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    feature_names = [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
        'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
    ]
    df = pd.DataFrame(data, columns=feature_names)
    df['MEDV'] = target
    return df

def split_features_targets(df: pd.DataFrame, target_col: str = 'MEDV') -> Tuple[np.ndarray, np.ndarray]:
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    return X, y

def train_test_split_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def evaluate_mse(model, X_test: np.ndarray, y_test: np.ndarray) -> float:
    preds = model.predict(X_test)
    return mean_squared_error(y_test, preds)
