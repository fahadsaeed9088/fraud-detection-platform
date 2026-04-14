"""Tests for model training and evaluation modules."""
import numpy as np
import pytest

from src.evaluate_models import evaluate_all_models, evaluate_model, get_best_model
from src.train_models import (train_logistic_regression, train_random_forest,
                               train_xgboost)


def _make_data(n_samples: int = 400, n_features: int = 29, seed: int = 42):
    """Generate a balanced synthetic classification dataset."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
    rng.shuffle(y)
    return X, y


# ── Training ─────────────────────────────────────────────────────────

class TestTrainModels:
    def setup_method(self):
        self.X, self.y = _make_data()

    def test_logistic_regression_trains(self):
        model = train_logistic_regression(self.X, self.y, random_state=42)
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')

    def test_random_forest_trains(self):
        model = train_random_forest(self.X, self.y, random_state=42)
        assert hasattr(model, 'predict')
        assert model.n_estimators == 100

    def test_xgboost_trains(self):
        model = train_xgboost(self.X, self.y, random_state=42)
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')

    def test_sklearn_models_predict_correct_shape(self):
        for fn in [train_logistic_regression, train_random_forest, train_xgboost]:
            model = fn(self.X, self.y, random_state=42)
            preds = model.predict(self.X)
            assert preds.shape == (len(self.X),)
            probs = model.predict_proba(self.X)
            assert probs.shape == (len(self.X), 2)


# ── Evaluation ───────────────────────────────────────────────────────

class TestEvaluateModels:
    def setup_method(self):
        X, y = _make_data(n_samples=400)
        # Train a quick model for testing
        mid = len(X) // 2
        self.X_train, self.X_test = X[:mid], X[mid:]
        self.y_train, self.y_test = y[:mid], y[mid:]
        self.model = train_logistic_regression(self.X_train, self.y_train)

    def test_evaluate_model_keys(self):
        result = evaluate_model(self.model, self.X_test, self.y_test)
        for key in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'confusion_matrix']:
            assert key in result

    def test_metrics_in_range(self):
        result = evaluate_model(self.model, self.X_test, self.y_test)
        for key in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            assert 0.0 <= result[key] <= 1.0, f"{key} out of range"

    def test_confusion_matrix_shape(self):
        result = evaluate_model(self.model, self.X_test, self.y_test)
        cm = result['confusion_matrix']
        assert len(cm) == 2 and len(cm[0]) == 2

    def test_evaluate_all_models_returns_all_keys(self):
        models = {
            'logistic_regression': self.model,
            'random_forest': train_random_forest(self.X_train, self.y_train),
        }
        results = evaluate_all_models(models, self.X_test, self.y_test)
        assert 'Logistic Regression' in results
        assert 'Random Forest' in results

    def test_get_best_model(self):
        models = {
            'logistic_regression': self.model,
            'random_forest': train_random_forest(self.X_train, self.y_train),
        }
        results = evaluate_all_models(models, self.X_test, self.y_test)
        best = get_best_model(results)
        assert best in results
