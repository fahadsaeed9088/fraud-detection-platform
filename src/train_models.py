import logging
import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

_BASE = os.path.dirname(os.path.dirname(__file__))
DEFAULT_MODELS_DIR = os.path.join(_BASE, 'models')


def train_logistic_regression(X_train, y_train, random_state: int = 42):
    """Train a Logistic Regression classifier."""
    logger.info("Training Logistic Regression...")
    model = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        class_weight='balanced',
        solver='lbfgs',
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    logger.info("Logistic Regression — done.")
    return model


def train_random_forest(X_train, y_train, random_state: int = 42):
    """Train a Random Forest classifier."""
    logger.info("Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=random_state,
        class_weight='balanced',
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    logger.info("Random Forest — done.")
    return model


def train_xgboost(X_train, y_train, random_state: int = 42):
    """Train an XGBoost classifier."""
    logger.info("Training XGBoost...")
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = neg / pos if pos > 0 else 1.0
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        eval_metric='logloss',
    )
    model.fit(X_train, y_train)
    logger.info("XGBoost — done.")
    return model


def train_neural_network(X_train, y_train, epochs: int = 20, batch_size: int = 256):
    """Train a Keras Sequential neural network."""
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.models import Sequential

    logger.info("Training Neural Network...")
    input_dim = X_train.shape[1]

    model = Sequential([
        Dense(32, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=0,
    )
    logger.info("Neural Network — done.")
    return model


def save_model(model, model_name: str, models_dir: str = None) -> str:
    """Persist a trained model to disk."""
    if models_dir is None:
        models_dir = DEFAULT_MODELS_DIR
    os.makedirs(models_dir, exist_ok=True)

    if model_name == 'neural_network':
        path = os.path.join(models_dir, 'neural_network.h5')
        model.save(path)
    else:
        path = os.path.join(models_dir, f'{model_name}.pkl')
        joblib.dump(model, path)

    logger.info("Saved model to %s", path)
    return path


def train_all_models(X_train, y_train, random_state: int = 42, models_dir: str = None) -> dict:
    """Train all four models, save them, and return a dict of model objects."""
    models = {
        'logistic_regression': train_logistic_regression(X_train, y_train, random_state),
        'random_forest': train_random_forest(X_train, y_train, random_state),
        'xgboost': train_xgboost(X_train, y_train, random_state),
        'neural_network': train_neural_network(X_train, y_train),
    }
    for name, model in models.items():
        save_model(model, name, models_dir)
    return models
