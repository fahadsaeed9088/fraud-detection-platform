import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

_MODEL_DISPLAY = {
    'logistic_regression': 'Logistic Regression',
    'random_forest': 'Random Forest',
    'xgboost': 'XGBoost',
    'neural_network': 'Neural Network',
}


def evaluate_model(model, X_test, y_test, model_name: str = 'Model') -> dict:
    """Compute classification metrics for one model."""
    if hasattr(model, 'predict_proba'):
        # sklearn / XGBoost
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
    else:
        # Keras Sequential
        y_prob = model.predict(X_test, verbose=0).flatten()
        y_pred = (y_prob >= 0.5).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    return {
        'model_name': model_name,
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, y_prob)),
        'confusion_matrix': cm.tolist(),
    }


def evaluate_all_models(models: dict, X_test, y_test) -> dict:
    """Evaluate all models and return keyed by display name."""
    results = {}
    for key, model in models.items():
        display = _MODEL_DISPLAY.get(key, key)
        logger.info("Evaluating %s ...", display)
        results[display] = evaluate_model(model, X_test, y_test, display)
    return results


def generate_comparison_dataframe(results: dict) -> pd.DataFrame:
    """Return a tidy DataFrame sorted by ROC-AUC descending."""
    rows = [
        {
            'Model': name,
            'Accuracy': round(m['accuracy'], 4),
            'Precision': round(m['precision'], 4),
            'Recall': round(m['recall'], 4),
            'F1-Score': round(m['f1'], 4),
            'ROC-AUC': round(m['roc_auc'], 4),
        }
        for name, m in results.items()
    ]
    df = pd.DataFrame(rows).sort_values('ROC-AUC', ascending=False).reset_index(drop=True)
    return df


def get_best_model(results: dict) -> str:
    """Return the display name of the model with the highest ROC-AUC."""
    return max(results, key=lambda k: results[k]['roc_auc'])
