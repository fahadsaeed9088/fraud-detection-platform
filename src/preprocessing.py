import logging
import os

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ['Time', 'Amount', 'Class'] + [f'V{i}' for i in range(1, 29)]


def load_data(filepath: str) -> pd.DataFrame:
    """Load dataset from a CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    df = pd.read_csv(filepath)
    logger.info("Loaded dataset with shape: %s", df.shape)
    return df


def validate_columns(df: pd.DataFrame) -> dict:
    """Validate that all required columns are present."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return {'valid': True, 'shape': df.shape, 'columns': list(df.columns)}


def generate_eda_summary(df: pd.DataFrame) -> dict:
    """Return a basic EDA summary of the dataset."""
    class_dist = df['Class'].value_counts().to_dict()
    return {
        'shape': list(df.shape),
        'n_rows': int(df.shape[0]),
        'n_cols': int(df.shape[1]),
        'missing_values': int(df.isnull().sum().sum()),
        'class_distribution': {str(k): int(v) for k, v in class_dist.items()},
        'fraud_rate': float(round(df['Class'].mean() * 100, 4)),
        'amount_stats': {
            'min': float(df['Amount'].min()),
            'max': float(df['Amount'].max()),
            'mean': float(round(df['Amount'].mean(), 2)),
            'median': float(round(df['Amount'].median(), 2)),
            'std': float(round(df['Amount'].std(), 2)),
        },
    }


def preprocess_data(df: pd.DataFrame):
    """Scale Amount and Time, then return feature matrix X and target y."""
    df = df.copy()
    scaler = StandardScaler()
    df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
    df['Time_scaled'] = scaler.fit_transform(df[['Time']])
    df.drop(columns=['Time', 'Amount'], inplace=True)
    X = df.drop(columns=['Class'])
    y = df['Class']
    return X, y


def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """Stratified train/test split."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def apply_smote(X_train, y_train, random_state: int = 42):
    """Apply SMOTE only on training data to avoid data leakage."""
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    before = pd.Series(y_train).value_counts().to_dict()
    after = pd.Series(y_res).value_counts().to_dict()
    logger.info("SMOTE — before: %s  after: %s", before, after)
    return X_res, y_res


def run_full_pipeline(filepath: str, test_size: float = 0.2, random_state: int = 42) -> dict:
    """Execute the full preprocessing pipeline and return split data + EDA."""
    df = load_data(filepath)
    validate_columns(df)
    eda = generate_eda_summary(df)
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size, random_state)
    X_train_res, y_train_res = apply_smote(X_train, y_train, random_state)
    return {
        'X_train': X_train_res,
        'X_test': X_test,
        'y_train': y_train_res,
        'y_test': y_test,
        'eda_summary': eda,
        'feature_names': list(X.columns),
    }
