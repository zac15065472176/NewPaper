import numpy as np
from sklearn.datasets import make_classification
from adaprune.core.adaprune_gbdt import AdaPruneGBDT


class TestAdaPruneGBDT:
    def test_fit_predict_binary(self):
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        model = AdaPruneGBDT(n_estimators=20)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape[0] == X.shape[0]

    def test_fit_predict_multiclass(self):
        X, y = make_classification(n_samples=300, n_features=15, n_classes=3, n_informative=5)
        model = AdaPruneGBDT(n_estimators=20)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape[0] == X.shape[0]

    def test_predict_proba_shape(self):
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        model = AdaPruneGBDT(n_estimators=20)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape[0] == X.shape[0]

    def test_different_strategies(self):
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        for s in ["hybrid", "gap_based", "trend_based", "data_aware"]:
            model = AdaPruneGBDT(n_estimators=10, strategy=s)
            model.fit(X, y)

    def test_early_stopping(self):
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        model = AdaPruneGBDT(n_estimators=50, early_stopping_rounds=5)
        model.fit(X, y)
        assert len(model.trees) <= 50

    def test_adaptation_history_recorded(self):
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        model = AdaPruneGBDT(n_estimators=10)
        model.fit(X, y)
        hist = model.get_adaptation_history()
        assert len(hist["param_history"]) > 0

    def test_verbose_output(self, capsys):
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        model = AdaPruneGBDT(n_estimators=5, verbose=1)
        model.fit(X, y)
        captured = capsys.readouterr()
        assert "Iter" in captured.out