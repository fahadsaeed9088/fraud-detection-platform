"""Tests for src/preprocessing.py"""
import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    REQUIRED_COLUMNS,
    apply_smote,
    generate_eda_summary,
    preprocess_data,
    split_data,
    validate_columns,
)


def _make_df(n_legit: int = 200, n_fraud: int = 20) -> pd.DataFrame:
    """Build a minimal synthetic DataFrame matching creditcard.csv schema."""
    rng = np.random.default_rng(42)
    n = n_legit + n_fraud
    data = {col: rng.standard_normal(n) for col in REQUIRED_COLUMNS}
    data['Time'] = rng.uniform(0, 172800, n)
    data['Amount'] = rng.uniform(0, 5000, n)
    data['Class'] = [0] * n_legit + [1] * n_fraud
    return pd.DataFrame(data)


# ── Column validation ────────────────────────────────────────────────

class TestValidateColumns:
    def test_valid_dataframe_passes(self):
        df = _make_df()
        result = validate_columns(df)
        assert result['valid'] is True

    def test_missing_column_raises(self):
        df = _make_df().drop(columns=['Class'])
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_columns(df)

    def test_partial_v_columns_raises(self):
        df = _make_df().drop(columns=['V1', 'V2'])
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_columns(df)


# ── EDA summary ──────────────────────────────────────────────────────

class TestEDASummary:
    def test_eda_keys_present(self):
        df = _make_df()
        eda = generate_eda_summary(df)
        for key in ['shape', 'n_rows', 'n_cols', 'missing_values',
                    'class_distribution', 'fraud_rate', 'amount_stats']:
            assert key in eda

    def test_correct_fraud_count(self):
        df = _make_df(n_legit=180, n_fraud=20)
        eda = generate_eda_summary(df)
        assert eda['class_distribution'][1] == 20
        assert eda['class_distribution'][0] == 180

    def test_no_missing_values(self):
        df = _make_df()
        eda = generate_eda_summary(df)
        assert eda['missing_values'] == 0


# ── Preprocessing ────────────────────────────────────────────────────

class TestPreprocessData:
    def test_output_shapes(self):
        df = _make_df()
        X, y = preprocess_data(df)
        assert len(X) == len(y) == len(df)

    def test_original_columns_removed(self):
        df = _make_df()
        X, _ = preprocess_data(df)
        assert 'Time' not in X.columns
        assert 'Amount' not in X.columns
        assert 'Class' not in X.columns

    def test_scaled_columns_added(self):
        df = _make_df()
        X, _ = preprocess_data(df)
        assert 'Amount_scaled' in X.columns
        assert 'Time_scaled' in X.columns


# ── Train/test split ─────────────────────────────────────────────────

class TestSplitData:
    def test_split_ratio(self):
        df = _make_df(200, 20)
        X, y = preprocess_data(df)
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
        n = len(df)
        assert len(X_train) + len(X_test) == n
        assert abs(len(X_test) / n - 0.2) < 0.05  # ~20% test

    def test_no_leakage_between_sets(self):
        df = _make_df()
        X, y = preprocess_data(df)
        X_train, X_test, _, _ = split_data(X, y)
        # Index sets must be disjoint
        assert set(X_train.index).isdisjoint(set(X_test.index))


# ── SMOTE ────────────────────────────────────────────────────────────

class TestSMOTE:
    def test_smote_balances_classes(self):
        df = _make_df(n_legit=200, n_fraud=20)
        X, y = preprocess_data(df)
        X_train, _, y_train, _ = split_data(X, y)
        X_res, y_res = apply_smote(X_train, y_train)
        counts = pd.Series(y_res).value_counts()
        assert counts[0] == counts[1], "SMOTE should balance classes"

    def test_smote_applied_only_on_training(self):
        """Verify the test set is not touched by SMOTE."""
        df = _make_df(n_legit=200, n_fraud=20)
        X, y = preprocess_data(df)
        X_train, X_test, y_train, y_test = split_data(X, y)
        X_res, y_res = apply_smote(X_train, y_train)
        # Test set size must remain unchanged
        assert len(X_test) == len(y_test)
        # Resampled training set must be larger than original
        assert len(X_res) >= len(X_train)
