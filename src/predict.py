import logging

import numpy as np

logger = logging.getLogger(__name__)


def get_risk_level(probability: float) -> str:
    """Map a fraud probability to a human-readable risk level."""
    if probability < 0.30:
        return 'Low'
    elif probability < 0.70:
        return 'Medium'
    return 'High'


def predict_transaction(model, transaction_data, model_type: str = 'sklearn') -> dict:
    """
    Predict fraud for a single transaction.

    Args:
        model: Trained model (sklearn/XGBoost or Keras)
        transaction_data: numpy array of shape (1, n_features)
        model_type: 'sklearn' or 'keras'

    Returns:
        dict with prediction, probability, risk_level, is_fraud
    """
    if model_type == 'keras':
        prob = float(model.predict(transaction_data, verbose=0).flatten()[0])
        pred = int(prob >= 0.5)
    else:
        prob = float(model.predict_proba(transaction_data)[0][1])
        pred = int(model.predict(transaction_data)[0])

    return {
        'prediction': pred,
        'probability': round(prob, 4),
        'risk_level': get_risk_level(prob),
        'is_fraud': pred == 1,
    }


def predict_batch(model, data, model_type: str = 'sklearn') -> list:
    """
    Predict fraud for a batch of transactions.

    Returns a list of prediction dicts.
    """
    if model_type == 'keras':
        probs = model.predict(data, verbose=0).flatten()
        preds = (probs >= 0.5).astype(int)
    else:
        probs = model.predict_proba(data)[:, 1]
        preds = model.predict(data)

    return [
        {
            'prediction': int(pred),
            'probability': round(float(prob), 4),
            'risk_level': get_risk_level(float(prob)),
            'is_fraud': int(pred) == 1,
        }
        for pred, prob in zip(preds, probs)
    ]
