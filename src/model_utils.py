import logging
import os

import joblib

logger = logging.getLogger(__name__)

_BASE = os.path.dirname(os.path.dirname(__file__))
DEFAULT_MODELS_DIR = os.path.join(_BASE, 'models')

_MODEL_FILES = {
    'logistic_regression': 'logistic_regression.pkl',
    'random_forest': 'random_forest.pkl',
    'xgboost': 'xgboost.pkl',
    'neural_network': 'neural_network.h5',
}


def load_model(model_name: str, models_dir: str = None):
    """Load a single trained model from disk."""
    if models_dir is None:
        models_dir = DEFAULT_MODELS_DIR

    filename = _MODEL_FILES.get(model_name)
    if not filename:
        raise ValueError(f"Unknown model '{model_name}'. Valid: {list(_MODEL_FILES)}")

    path = os.path.join(models_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    if model_name == 'neural_network':
        from tensorflow.keras.models import load_model as _keras_load
        return _keras_load(path)

    return joblib.load(path)


def load_all_models(models_dir: str = None) -> dict:
    """Load all trained models. Missing files are skipped with a warning."""
    models = {}
    for name in _MODEL_FILES:
        try:
            models[name] = load_model(name, models_dir)
            logger.info("Loaded model: %s", name)
        except FileNotFoundError:
            logger.warning("Model file missing: %s", name)
    return models


def models_exist(models_dir: str = None) -> bool:
    """Return True only if all four model files are present on disk."""
    if models_dir is None:
        models_dir = DEFAULT_MODELS_DIR
    return all(
        os.path.exists(os.path.join(models_dir, fname))
        for fname in _MODEL_FILES.values()
    )
